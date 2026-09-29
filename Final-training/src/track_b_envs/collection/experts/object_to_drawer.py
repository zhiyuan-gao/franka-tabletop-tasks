"""Physical long-horizon expert for opening a drawer and placing a banana."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.task_scene import BANANA_STABLE_ROTATION_WXYZ
from track_b_envs.envs.common.timing import (
    angle_per_step,
    distance_per_step,
    duration_reached,
    environment_control_dt_s,
)

from .common import (
    absolute_pose_to_relative_action,
    advance_pose_toward_waypoint,
    object_to_drawer_task_state,
)


class ObjectToDrawerExpert:
    REST = 0
    APPROACH_HANDLE_ABOVE = 1
    APPROACH_HANDLE = 2
    GRASP_HANDLE = 3
    PULL_DRAWER = 4
    RELEASE_HANDLE = 5
    RETREAT_HANDLE = 6
    ABOVE_OBJECT = 7
    DESCEND_OBJECT = 8
    GRASP_OBJECT = 9
    LIFT_OBJECT = 10
    ABOVE_DRAWER = 11
    PLACE_IN_DRAWER = 12
    RELEASE_OBJECT = 13
    SETTLE = 14

    STATE_NAMES = (
        "rest",
        "approach_handle_above",
        "approach_handle",
        "grasp_handle",
        "pull_drawer",
        "release_handle",
        "retreat_handle",
        "above_object",
        "descend_object",
        "grasp_object",
        "lift_object",
        "above_drawer",
        "place_in_drawer",
        "release_object",
        "settle",
    )

    def __init__(self, env):
        self.num_envs = env.num_envs
        self.device = env.device
        self.state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wait = torch.zeros_like(self.state)
        self.control_dt_s = environment_control_dt_s(env)
        self.controller_scale = torch.as_tensor(
            env.cfg.actions.arm_action.scale,
            device=self.device,
            dtype=torch.float32,
        )
        self.drawer_direction = torch.zeros((self.num_envs, 2), device=self.device)
        cabinet = env.scene["cabinet"]
        self.drawer_joint_ids, names = cabinet.find_joints("UpperDrawerJoint")
        if names != ["UpperDrawerJoint"]:
            raise RuntimeError(f"expected UpperDrawerJoint, found {names}")
        self.drawer_hold_target = torch.zeros(
            (self.num_envs, 1), device=self.device
        )
        self.drawer_velocity_target = torch.zeros_like(self.drawer_hold_target)
        self.drawer_effort_target = torch.zeros_like(self.drawer_hold_target)
        self.lift_targets = torch.zeros((self.num_envs, 3), device=self.device)
        self.grasp_targets = torch.zeros((self.num_envs, 3), device=self.device)
        self.grasped_object_height = torch.zeros(
            self.num_envs, device=self.device
        )
        self.grasp_quaternions = torch.zeros((self.num_envs, 4), device=self.device)
        self.grasp_quaternions[:, 0] = 1.0
        self.release_object_targets = torch.zeros((self.num_envs, 3), device=self.device)
        self.handle_retry_count = torch.zeros_like(self.state)
        self.object_retry_count = torch.zeros_like(self.state)
        self.grasp_verified = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.command_pose = torch.zeros((self.num_envs, 7), device=self.device)

    def reset(self, env) -> None:
        from isaaclab.utils.math import quat_apply

        self.state.zero_()
        self.wait.zero_()
        self.handle_retry_count.zero_()
        self.object_retry_count.zero_()
        self.grasp_verified.zero_()
        cabinet_quaternion = env.scene["cabinet"].data.root_quat_w
        local_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(
            self.num_envs, -1
        )
        self.drawer_direction.copy_(
            quat_apply(cabinet_quaternion, local_axis)[:, :2]
        )
        self.lift_targets.zero_()
        self.drawer_hold_target.zero_()
        self.drawer_velocity_target.zero_()
        self.drawer_effort_target.zero_()
        env.scene["cabinet"].set_joint_position_target(
            self.drawer_hold_target, joint_ids=self.drawer_joint_ids
        )
        env.scene["cabinet"].set_joint_velocity_target(
            self.drawer_velocity_target, joint_ids=self.drawer_joint_ids
        )
        env.scene["cabinet"].set_joint_effort_target(
            self.drawer_effort_target, joint_ids=self.drawer_joint_ids
        )
        self.grasp_targets.zero_()
        self.grasped_object_height.zero_()
        self.grasp_quaternions.zero_()
        self.grasp_quaternions[:, 0] = 1.0
        self.release_object_targets.zero_()
        self.command_pose.copy_(object_to_drawer_task_state(env)["ee_pose"])

    def _transition(self, mask: torch.Tensor, next_state: int) -> None:
        self.state[mask] = next_state
        self.wait[mask] = 0

    def compute(self, env) -> torch.Tensor:
        from isaaclab.utils.math import (
            quat_apply,
            quat_apply_inverse,
            quat_inv,
            quat_mul,
            subtract_frame_transforms,
        )

        task = object_to_drawer_task_state(env)
        ee_pose = task["ee_pose"]
        handle = task["handle_pose"][:, :3]
        obj = task["object_pose"][:, :3]
        drawer_target = task["target_pose"][:, :3]
        direction = self.drawer_direction

        tool_horizontal = torch.tensor(
            [0.70710678, 0.70710678, 0.0, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        handle_quaternion = quat_mul(
            task["handle_pose"][:, 3:7], tool_horizontal
        )
        # Preserve the validated nominal top-down grasp as an object-local
        # transform, then rotate it with the independently randomized banana.
        # The fingers therefore continue to close across its short axis.
        nominal_object_quaternion = torch.tensor(
            BANANA_STABLE_ROTATION_WXYZ, device=self.device
        ).expand(self.num_envs, -1)
        nominal_object_quaternion_target = torch.tensor(
            [0.0, 0.95371695, 0.30070580, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        object_to_gripper_quaternion = quat_mul(
            quat_inv(nominal_object_quaternion),
            nominal_object_quaternion_target,
        )
        object_quaternion = quat_mul(
            task["object_pose"][:, 3:7], object_to_gripper_quaternion
        )
        # The YCB banana is C-shaped: its root/bounding-box center lies in the
        # curve's open space.  Shift toward the solid arc so both fingers close
        # on geometry instead of through the center of the C shape.
        # Recovered from the *commanded* Cartesian targets in three previously
        # successful physical episodes.  The TCP itself lags this waypoint
        # during closure, so using its measured pose as the offset is wrong.
        nominal_grasp_offset = torch.tensor(
            [0.01433941, -0.02047880, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        object_local_grasp_offset = quat_apply_inverse(
            nominal_object_quaternion, nominal_grasp_offset
        )
        grasp_offset = quat_apply(
            task["object_pose"][:, 3:7], object_local_grasp_offset
        )

        local_front = torch.tensor(
            [0.0, 1.0, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        front_normal = quat_apply(
            task["handle_pose"][:, 3:7], local_front
        )[:, :2]
        handle_front = handle.clone()
        handle_front[:, :2] += 0.100 * front_normal
        # Align in front of the handle at the final grasp height, then insert
        # only along the cabinet face normal to avoid sweeping the panel.
        handle_above = handle_front.clone()
        penetration = torch.clamp(
            0.010 + 0.002 * self.handle_retry_count.to(torch.float32), max=0.014
        )
        handle_grasp = handle.clone()
        handle_grasp[:, :2] -= penetration[:, None] * front_normal
        handle_pull = handle.clone()
        pull_progress = torch.clamp(
            self.wait.to(torch.float32) * self.control_dt_s / 1.60,
            min=0.0,
            max=1.0,
        )
        linear_lead = -0.006 + 0.036 * pull_progress
        handle_pull[:, :2] += linear_lead[:, None] * front_normal

        object_above = obj + grasp_offset
        object_above[:, 2] += 0.140
        object_grasp = obj + grasp_offset
        object_grasp[:, 2] += 0.012
        drawer_above = drawer_target.clone()
        drawer_above[:, 2] += 0.180
        drawer_place = drawer_target.clone()
        drawer_place[:, 2] += 0.020

        desired_position = ee_pose[:, :3].clone()
        desired_quaternion = ee_pose[:, 3:7].clone()

        def orientation_error(quaternion):
            dot = (
                (ee_pose[:, 3:7] * quaternion)
                .sum(dim=-1)
                .abs()
                .clamp(0.0, 1.0)
            )
            return 2.0 * torch.acos(dot)

        def assign(phase, position, quaternion):
            mask = self.state == phase
            desired_position[mask] = position[mask]
            desired_quaternion[mask] = quaternion[mask]

        assign(self.APPROACH_HANDLE_ABOVE, handle_above, handle_quaternion)
        assign(self.APPROACH_HANDLE, handle_grasp, handle_quaternion)
        assign(self.GRASP_HANDLE, handle_grasp, handle_quaternion)
        assign(self.PULL_DRAWER, handle_pull, handle_quaternion)
        release_clear = handle.clone()
        release_clear[:, :2] += 0.060 * direction
        retreat_clear = release_clear.clone()
        retreat_clear[:, 2] += 0.180
        assign(self.RELEASE_HANDLE, release_clear, handle_quaternion)
        assign(self.RETREAT_HANDLE, retreat_clear, handle_quaternion)
        assign(self.ABOVE_OBJECT, object_above, object_quaternion)
        # Freeze the selected target throughout descent and closure.  The
        # successful 50 Hz episodes used this fixed command; following the
        # moving banana instead creates a chasing loop on its curved surface.
        assign(self.DESCEND_OBJECT, self.grasp_targets, self.grasp_quaternions)
        assign(self.GRASP_OBJECT, self.grasp_targets, self.grasp_quaternions)
        assign(self.LIFT_OBJECT, self.lift_targets, self.grasp_quaternions)
        assign(self.ABOVE_DRAWER, drawer_above, self.grasp_quaternions)
        assign(self.PLACE_IN_DRAWER, drawer_place, self.grasp_quaternions)
        assign(
            self.RELEASE_OBJECT,
            self.release_object_targets,
            self.grasp_quaternions,
        )
        assign(self.SETTLE, drawer_above, self.grasp_quaternions)

        self.wait += 1
        self._transition(
            (self.state == self.REST)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.APPROACH_HANDLE_ABOVE,
        )
        self._transition(
            (self.state == self.APPROACH_HANDLE_ABOVE)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - handle_above, dim=-1) < 0.008)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.APPROACH_HANDLE,
        )
        self._transition(
            (self.state == self.APPROACH_HANDLE)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - handle_grasp, dim=-1) < 0.006)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.GRASP_HANDLE,
        )
        handle_in_ee, _ = subtract_frame_transforms(
            ee_pose[:, :3],
            ee_pose[:, 3:7],
            task["handle_pose"][:, :3],
            task["handle_pose"][:, 3:7],
        )
        fingers = task["finger_positions"]
        valid_aperture = (
            (fingers.min(dim=-1).values >= 0.0015)
            & (fingers.max(dim=-1).values <= 0.0120)
            & ((fingers[:, 0] - fingers[:, 1]).abs() <= 0.0040)
        )
        handle_centered = (
            (handle_in_ee[:, 0].abs() <= 0.030)
            & (handle_in_ee[:, 1].abs() <= 0.014)
            & (handle_in_ee[:, 2] >= -0.040)
            & (handle_in_ee[:, 2] <= 0.004)
        )
        verified_handle = (
            (self.state == self.GRASP_HANDLE) & valid_aperture & handle_centered
        )
        grasp_complete = verified_handle & duration_reached(
            self.wait, self.control_dt_s, 0.70
        )
        self.grasp_verified[grasp_complete] = True
        self._transition(grasp_complete, self.PULL_DRAWER)
        lost_handle = (
            (self.state == self.PULL_DRAWER)
            & duration_reached(self.wait, self.control_dt_s, 0.16)
            & ~valid_aperture
        )
        self.grasp_verified[lost_handle] = False
        self.handle_retry_count[lost_handle] += 1
        self._transition(lost_handle, self.APPROACH_HANDLE_ABOVE)
        retry_handle = (
            (self.state == self.GRASP_HANDLE)
            & duration_reached(self.wait, self.control_dt_s, 1.50)
            & ~verified_handle
        )
        self.handle_retry_count[retry_handle] += 1
        self._transition(retry_handle, self.APPROACH_HANDLE_ABOVE)
        drawer_opened = (
            (self.state == self.PULL_DRAWER)
            & (task["joint_position"] >= 0.145)
        )
        if torch.any(drawer_opened):
            self.drawer_hold_target[drawer_opened, 0] = task["joint_position"][
                drawer_opened
            ]
        self._transition(drawer_opened, self.RELEASE_HANDLE)
        self._transition(
            (self.state == self.RELEASE_HANDLE)
            & duration_reached(self.wait, self.control_dt_s, 0.70),
            self.RETREAT_HANDLE,
        )
        self._transition(
            (self.state == self.RETREAT_HANDLE)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - retreat_clear, dim=-1) < 0.030)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.ABOVE_OBJECT,
        )
        above_object_reached = (
            (self.state == self.ABOVE_OBJECT)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - object_above, dim=-1) < 0.025)
            & (orientation_error(object_quaternion) < 0.060)
            & duration_reached(self.wait, self.control_dt_s, 0.20)
        )
        if torch.any(above_object_reached):
            self.grasp_targets[above_object_reached] = object_grasp[above_object_reached]
            self.grasp_quaternions[above_object_reached] = object_quaternion[
                above_object_reached
            ]
        self._transition(above_object_reached, self.DESCEND_OBJECT)
        self._transition(
            (self.state == self.DESCEND_OBJECT)
            & (
                torch.linalg.vector_norm(
                    ee_pose[:, :3] - self.grasp_targets, dim=-1
                )
                < 0.008
            )
            & (orientation_error(self.grasp_quaternions) < 0.060)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.GRASP_OBJECT,
        )
        valid_object_aperture = (
            (fingers.min(dim=-1).values >= 0.0020)
            & (fingers.max(dim=-1).values <= 0.0250)
            & ((fingers[:, 0] - fingers[:, 1]).abs() <= 0.0060)
        )
        grasp_complete = (
            (self.state == self.GRASP_OBJECT)
            & duration_reached(self.wait, self.control_dt_s, 1.60)
            & valid_object_aperture
        )
        if torch.any(grasp_complete):
            self.grasped_object_height[grasp_complete] = obj[grasp_complete, 2]
            self.lift_targets[grasp_complete] = ee_pose[grasp_complete, :3]
            self.lift_targets[grasp_complete, 2] = torch.maximum(
                self.lift_targets[grasp_complete, 2],
                torch.full_like(self.lift_targets[grasp_complete, 2], 0.555),
            )
        self._transition(grasp_complete, self.LIFT_OBJECT)
        retry_object = (
            (self.state == self.GRASP_OBJECT)
            & duration_reached(self.wait, self.control_dt_s, 2.20)
            & ~valid_object_aperture
        )
        self.object_retry_count[retry_object] += 1
        self._transition(retry_object, self.ABOVE_OBJECT)
        lost_object = (
            (self.state == self.LIFT_OBJECT)
            & duration_reached(self.wait, self.control_dt_s, 0.60)
            & (
                (obj[:, 2] < self.grasped_object_height + 0.010)
                | ~valid_object_aperture
            )
        )
        self.object_retry_count[lost_object] += 1
        self._transition(lost_object, self.ABOVE_OBJECT)
        self._transition(
            (self.state == self.LIFT_OBJECT)
            & (obj[:, 2] >= 0.445)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - self.lift_targets, dim=-1) < 0.035),
            self.ABOVE_DRAWER,
        )
        lost_during_transfer = (
            (self.state == self.ABOVE_DRAWER)
            & duration_reached(self.wait, self.control_dt_s, 0.30)
            & (
                (torch.linalg.vector_norm(ee_pose[:, :3] - obj, dim=-1) > 0.080)
                | ~valid_object_aperture
            )
        )
        self.object_retry_count[lost_during_transfer] += 1
        self._transition(lost_during_transfer, self.ABOVE_OBJECT)
        self._transition(
            (self.state == self.ABOVE_DRAWER)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - drawer_above, dim=-1) < 0.030)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.PLACE_IN_DRAWER,
        )
        object_above_interior = (
            torch.linalg.vector_norm(obj[:, :2] - drawer_target[:, :2], dim=-1)
            < 0.060
        ) & ((obj[:, 2] - drawer_target[:, 2]) < 0.075)
        ready_to_release = (
            (self.state == self.PLACE_IN_DRAWER)
            & object_above_interior
            & duration_reached(self.wait, self.control_dt_s, 0.30)
        )
        if torch.any(ready_to_release):
            self.release_object_targets[ready_to_release] = ee_pose[
                ready_to_release, :3
            ]
        self._transition(ready_to_release, self.RELEASE_OBJECT)
        self._transition(
            (self.state == self.RELEASE_OBJECT)
            & duration_reached(self.wait, self.control_dt_s, 0.90),
            self.SETTLE,
        )

        # During the physical pull the target follows measured joint position,
        # so the drive contributes essentially no opening force.  From release
        # onward it holds the captured opening against residual rail drift and
        # incidental contact; it never writes the joint state.
        following = self.state <= self.PULL_DRAWER
        self.drawer_hold_target[following, 0] = task["joint_position"][following]
        self.drawer_velocity_target[following, 0] = task["joint_velocity"][following]
        self.drawer_velocity_target[~following, 0] = 0.0
        self.drawer_effort_target[following, 0] = 0.0
        self.drawer_effort_target[~following, 0] = (
            50.0
            * (
                self.drawer_hold_target[~following, 0]
                - task["joint_position"][~following]
            )
            - 10.0 * task["joint_velocity"][~following]
        ).clamp(-20.0, 20.0)
        env.scene["cabinet"].set_joint_position_target(
            self.drawer_hold_target, joint_ids=self.drawer_joint_ids
        )
        env.scene["cabinet"].set_joint_velocity_target(
            self.drawer_velocity_target, joint_ids=self.drawer_joint_ids
        )
        env.scene["cabinet"].set_joint_effort_target(
            self.drawer_effort_target, joint_ids=self.drawer_joint_ids
        )

        # Closed for both physical grasps and throughout transport; open while
        # approaching, releasing, and waiting for the success settle window.
        handle_closed = (self.state >= self.GRASP_HANDLE) & (
            self.state <= self.PULL_DRAWER
        )
        # Hold the final banana waypoint with the fingers fully open before the
        # single close command.  This removes residual Cartesian motion at first
        # contact without reintroducing an intermediate aperture or changing the
        # spatial arm trajectory.
        object_close_ready = (self.state > self.GRASP_OBJECT) | (
            (self.state == self.GRASP_OBJECT)
            & duration_reached(self.wait, self.control_dt_s, 0.35)
        )
        object_closed = object_close_ready & (self.state <= self.PLACE_IN_DRAWER)
        closed = handle_closed | object_closed
        gripper = torch.where(
            closed,
            -torch.ones(self.num_envs, device=self.device),
            torch.ones(self.num_envs, device=self.device),
        )
        # Both grasps now use the hardware-compatible binary interface: remain
        # fully open until the final handle/object waypoint, then issue one close
        # command.  Finger motion remains continuous through the 100 Hz actuator
        # dynamics, but the high-level command and recorded label are binary.
        translation_step = torch.full(
            (self.num_envs,),
            distance_per_step(0.15, self.control_dt_s),
            device=self.device,
        )
        translation_step[self.state == self.APPROACH_HANDLE] = distance_per_step(
            0.05, self.control_dt_s
        )
        translation_step[self.state == self.GRASP_HANDLE] = distance_per_step(
            0.025, self.control_dt_s
        )
        translation_step[self.state == self.PULL_DRAWER] = distance_per_step(
            0.075, self.control_dt_s
        )
        translation_step[self.state == self.DESCEND_OBJECT] = distance_per_step(
            0.05, self.control_dt_s
        )
        translation_step[self.state == self.GRASP_OBJECT] = distance_per_step(
            0.025, self.control_dt_s
        )
        translation_step[self.state == self.LIFT_OBJECT] = distance_per_step(
            0.05, self.control_dt_s
        )
        translation_step[self.state == self.PLACE_IN_DRAWER] = distance_per_step(
            0.075, self.control_dt_s
        )
        waypoint_pose = torch.cat((desired_position, desired_quaternion), dim=-1)
        self.command_pose.copy_(
            advance_pose_toward_waypoint(
                self.command_pose,
                waypoint_pose,
                translation_step,
                angle_per_step(1.0, self.control_dt_s),
            )
        )
        tracking_limit = torch.full(
            (self.num_envs,),
            distance_per_step(1.0, self.control_dt_s),
            device=self.device,
        )
        return absolute_pose_to_relative_action(
            ee_pose,
            self.command_pose,
            gripper,
            self.controller_scale,
            max_translation_step_m=tracking_limit,
            max_rotation_step_rad=angle_per_step(1.6, self.control_dt_s),
        )
