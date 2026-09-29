"""Physical experts for the calibrated upper/lower drawer subtask family."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.task_scene import BANANA_STABLE_ROTATION_WXYZ
from track_b_envs.envs.common.timing import (
    angle_per_step,
    distance_per_step,
    duration_reached,
    environment_control_dt_s,
)
from track_b_envs.envs.object_to_drawer.mdp.geometry import root_inside_drawer

from .articulated_opening import ArticulatedOpeningExpert
from .common import (
    absolute_pose_to_relative_action,
    advance_pose_toward_waypoint,
    drawer_subtask_state,
)


class _PutObjectDrawerExpert:
    REST = 0
    ABOVE_OBJECT = 1
    DESCEND_OBJECT = 2
    GRASP_OBJECT = 3
    LIFT_OBJECT = 4
    TRANSFER_HIGH = 5
    ABOVE_DRAWER = 6
    PLACE_IN_DRAWER = 7
    RELEASE_OBJECT = 8
    RETREAT_OBJECT = 9
    SETTLE_OBJECT = 10
    APPROACH_HANDLE_FRONT = 11
    APPROACH_HANDLE = 12
    GRASP_HANDLE = 13
    PUSH_DRAWER = 14
    RELEASE_HANDLE = 15
    RETREAT_HANDLE = 16
    DONE = 17
    ABOVE_HANDLE_FRONT = 18

    STATE_NAMES = (
        "rest",
        "above_object",
        "descend_object",
        "grasp_object",
        "lift_object",
        "transfer_high",
        "above_drawer",
        "place_in_drawer",
        "release_object",
        "retreat_object",
        "settle_object",
        "approach_handle_front",
        "approach_handle",
        "grasp_handle",
        "push_drawer",
        "release_handle",
        "retreat_handle",
        "done",
        "above_handle_front",
    )

    def __init__(self, env):
        if env.cfg.task_mode not in {"put", "put_and_close"}:
            raise ValueError(f"unsupported placement mode: {env.cfg.task_mode}")
        self.mode = env.cfg.task_mode
        self.drawer_level = env.cfg.drawer_level
        self.joint_name = env.cfg.active_joint_name
        self.num_envs = env.num_envs
        self.device = env.device
        self.control_dt_s = environment_control_dt_s(env)
        self.controller_scale = torch.as_tensor(
            env.cfg.actions.arm_action.scale,
            device=self.device,
            dtype=torch.float32,
        )
        self.state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wait = torch.zeros_like(self.state)
        self.retry_count = torch.zeros_like(self.state)
        self.grasp_verified = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if self.mode == "put_and_close":
            self.grasp_verified.zero_()
        self.closing_contact_verified = torch.zeros_like(self.grasp_verified)
        self.push_start_joint = torch.zeros(self.num_envs, device=self.device)
        self.command_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.grasp_pose = torch.zeros_like(self.command_pose)
        self.lift_pose = torch.zeros_like(self.command_pose)
        self.release_pose = torch.zeros_like(self.command_pose)
        self.initial_object_height = torch.zeros(self.num_envs, device=self.device)
        cabinet = env.scene["cabinet"]
        self.drawer_joint_ids, names = cabinet.find_joints(self.joint_name)
        if names != [self.joint_name]:
            raise RuntimeError(f"expected {self.joint_name}, found {names}")
        self.drawer_hold_target = torch.zeros((self.num_envs, 1), device=self.device)
        self.drawer_velocity_target = torch.zeros_like(self.drawer_hold_target)
        self.drawer_effort_target = torch.zeros_like(self.drawer_hold_target)

    def reset(self, env) -> None:
        self.state.zero_()
        self.wait.zero_()
        self.retry_count.zero_()
        self.grasp_verified.fill_(self.mode == "put")
        self.closing_contact_verified.zero_()
        state = drawer_subtask_state(env, self.joint_name)
        self.command_pose.copy_(state["ee_pose"])
        self.grasp_pose.zero_()
        self.lift_pose.zero_()
        self.release_pose.zero_()
        self.initial_object_height.copy_(state["object_pose"][:, 2])
        self.drawer_hold_target[:, 0] = state["joint_position"]
        self.drawer_velocity_target.zero_()
        self.drawer_effort_target.zero_()
        cabinet = env.scene["cabinet"]
        cabinet.set_joint_position_target(
            self.drawer_hold_target, joint_ids=self.drawer_joint_ids
        )
        cabinet.set_joint_velocity_target(
            self.drawer_velocity_target, joint_ids=self.drawer_joint_ids
        )
        cabinet.set_joint_effort_target(
            self.drawer_effort_target, joint_ids=self.drawer_joint_ids
        )

    def _transition(self, mask: torch.Tensor, next_state: int) -> None:
        self.state[mask] = next_state
        self.wait[mask] = 0

    def compute(self, env) -> torch.Tensor:
        from isaaclab.utils.math import (
            quat_apply,
            quat_apply_inverse,
            quat_inv,
            quat_mul,
        )

        task = drawer_subtask_state(env, self.joint_name)
        ee_pose = task["ee_pose"]
        obj = task["object_pose"][:, :3]
        drawer_target = task["target_pose"][:, :3].clone()
        # Use the accessible front portion, within the original success volume,
        # so the banana clears the cabinet lip / closed upper drawer overhang.
        drawer_target += quat_apply(
            task["target_pose"][:, 3:7],
            torch.tensor([0.0, 0.035, 0.0], device=self.device).expand(self.num_envs, -1),
        )
        handle = task["handle_pose"][:, :3]
        fingers = task["finger_positions"]

        nominal_object_quaternion = torch.tensor(
            BANANA_STABLE_ROTATION_WXYZ, device=self.device
        ).expand(self.num_envs, -1)
        nominal_gripper_quaternion = torch.tensor(
            [0.0, 0.95371695, 0.30070580, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        object_to_gripper_quaternion = quat_mul(
            quat_inv(nominal_object_quaternion), nominal_gripper_quaternion
        )
        object_quaternion = quat_mul(
            task["object_pose"][:, 3:7], object_to_gripper_quaternion
        )
        # Keep the banana-local contact point recovered from the successful
        # object_to_drawer demonstrations.  Its root is in the open part of
        # the C-shaped mesh, so grasping at the root (or from the near side)
        # tends to squeeze the fruit away instead of closing on geometry.
        nominal_grasp_offset = torch.tensor(
            [0.01433941, -0.02047880, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        local_grasp_offset = quat_apply_inverse(
            nominal_object_quaternion, nominal_grasp_offset
        )
        grasp_offset = quat_apply(
            task["object_pose"][:, 3:7], local_grasp_offset
        )
        object_above = obj + grasp_offset
        # The calibrated cabinet is far from the base and an already-open
        # drawer narrows the wrist workspace.  Nine centimetres clears the
        # banana and cabinet top while remaining inside the measured reachable
        # workspace for the unchanged video-calibrated layout.
        object_above[:, 2] += 0.090
        object_grasp = obj + grasp_offset
        object_grasp[:, 2] += 0.012

        transfer_high = drawer_target.clone()
        transfer_high[:, 2] = 0.480
        drawer_above = drawer_target.clone()
        drawer_above[:, 2] += 0.180
        drawer_place = drawer_target.clone()
        drawer_place[:, 2] += 0.020

        # Closing is a front-panel push with closed fingers. Preserve the
        # transport wrist orientation; a second bilateral grasp is unnecessary.
        handle_quaternion = self.grasp_pose[:, 3:7]
        local_front = torch.tensor(
            [0.0, 1.0, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        front_normal = quat_apply(
            task["handle_pose"][:, 3:7], local_front
        )[:, :2]
        # At the lower drawer, push the lower half of the panel so the palm
        # stays below the closed upper drawer. Descend farther outside first.
        front_clearance = 0.130 if self.drawer_level == "lower" else 0.065
        push_height_offset = -0.055 if self.drawer_level == "lower" else 0.010
        handle_front = handle.clone()
        handle_front[:, :2] += front_clearance * front_normal
        handle_front[:, 2] += push_height_offset
        handle_front_above = handle_front.clone()
        handle_front_above[:, 2] = handle[:, 2] + 0.180
        handle_grasp = handle.clone()
        handle_grasp[:, :2] += 0.012 * front_normal
        handle_grasp[:, 2] += push_height_offset
        handle_push = handle.clone()
        handle_push[:, 2] += push_height_offset
        push_progress = torch.clamp(
            self.wait.to(torch.float32) * self.control_dt_s / 2.0,
            min=0.0,
            max=1.0,
        )
        push_lead = 0.008 + 0.035 * push_progress
        handle_push[:, :2] -= push_lead[:, None] * front_normal
        handle_release = handle.clone()
        handle_release[:, :2] += 0.065 * front_normal
        handle_retreat = handle_release.clone()
        handle_retreat[:, 2] += 0.160

        desired_position = ee_pose[:, :3].clone()
        desired_quaternion = ee_pose[:, 3:7].clone()

        def assign(phase, position, quaternion):
            mask = self.state == phase
            desired_position[mask] = position[mask]
            desired_quaternion[mask] = quaternion[mask]

        assign(self.ABOVE_OBJECT, object_above, object_quaternion)
        assign(self.DESCEND_OBJECT, self.grasp_pose[:, :3], self.grasp_pose[:, 3:7])
        assign(self.GRASP_OBJECT, self.grasp_pose[:, :3], self.grasp_pose[:, 3:7])
        assign(self.LIFT_OBJECT, self.lift_pose[:, :3], self.lift_pose[:, 3:7])
        # Match ObjectToDrawerExpert: once grasped, the world-frame tool
        # orientation is fixed. Recomputing it from the moving fruit creates
        # feedback between contact-induced fruit rotation and wrist targets.
        transport_quaternion = self.grasp_pose[:, 3:7]
        assign(self.TRANSFER_HIGH, transfer_high, transport_quaternion)
        assign(self.ABOVE_DRAWER, drawer_above, transport_quaternion)
        assign(self.PLACE_IN_DRAWER, drawer_place, transport_quaternion)
        assign(self.RELEASE_OBJECT, self.release_pose[:, :3], self.release_pose[:, 3:7])
        assign(self.RETREAT_OBJECT, drawer_above, transport_quaternion)
        assign(self.SETTLE_OBJECT, drawer_above, transport_quaternion)
        assign(self.ABOVE_HANDLE_FRONT, handle_front_above, handle_quaternion)
        assign(self.APPROACH_HANDLE_FRONT, handle_front, handle_quaternion)
        assign(self.APPROACH_HANDLE, handle_grasp, handle_quaternion)
        assign(self.GRASP_HANDLE, handle_grasp, handle_quaternion)
        assign(self.PUSH_DRAWER, handle_push, handle_quaternion)
        assign(self.RELEASE_HANDLE, handle_release, handle_quaternion)
        assign(self.RETREAT_HANDLE, handle_retreat, handle_quaternion)
        assign(self.DONE, handle_retreat, handle_quaternion)

        self.wait += 1
        self._transition(
            (self.state == self.REST)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.ABOVE_OBJECT,
        )
        above_object_reached = (
            (self.state == self.ABOVE_OBJECT)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - object_above, dim=-1) < 0.025)
            & duration_reached(self.wait, self.control_dt_s, 0.20)
        )
        if torch.any(above_object_reached):
            self.grasp_pose[above_object_reached, :3] = object_grasp[
                above_object_reached
            ]
            self.grasp_pose[above_object_reached, 3:7] = object_quaternion[
                above_object_reached
            ]
        self._transition(above_object_reached, self.DESCEND_OBJECT)
        self._transition(
            (self.state == self.DESCEND_OBJECT)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - self.grasp_pose[:, :3], dim=-1) < 0.015)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.GRASP_OBJECT,
        )
        valid_object_aperture = (
            (fingers.min(dim=-1).values >= 0.0020)
            & (fingers.max(dim=-1).values <= 0.0250)
            & ((fingers[:, 0] - fingers[:, 1]).abs() <= 0.0060)
        )
        object_grasped = (
            (self.state == self.GRASP_OBJECT)
            & duration_reached(self.wait, self.control_dt_s, 1.60)
            & valid_object_aperture
        )
        if torch.any(object_grasped):
            self.lift_pose[object_grasped] = ee_pose[object_grasped]
            self.lift_pose[object_grasped, 2] = 0.520
        self._transition(object_grasped, self.LIFT_OBJECT)
        retry_object = (
            (self.state == self.GRASP_OBJECT)
            & duration_reached(self.wait, self.control_dt_s, 2.20)
            & ~valid_object_aperture
        )
        self._transition(retry_object, self.ABOVE_OBJECT)
        self._transition(
            (self.state == self.LIFT_OBJECT)
            & (obj[:, 2] >= self.initial_object_height + 0.060)
            & valid_object_aperture
            & duration_reached(self.wait, self.control_dt_s, 0.40),
            self.TRANSFER_HIGH,
        )
        self._transition(
            (self.state == self.TRANSFER_HIGH)
            & (torch.linalg.vector_norm(obj[:, :2] - drawer_target[:, :2], dim=-1) < 0.030)
            & (obj[:, 2] >= self.initial_object_height + 0.060)
            & valid_object_aperture
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.ABOVE_DRAWER,
        )
        self._transition(
            (self.state == self.ABOVE_DRAWER)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - drawer_above, dim=-1) < 0.025)
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
            self.release_pose[ready_to_release] = ee_pose[ready_to_release]
        self._transition(ready_to_release, self.RELEASE_OBJECT)
        self._transition(
            (self.state == self.RELEASE_OBJECT)
            & duration_reached(self.wait, self.control_dt_s, 0.90),
            self.RETREAT_OBJECT,
        )
        retreat_reached = (
            (self.state == self.RETREAT_OBJECT)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - drawer_above, dim=-1) < 0.035)
            & duration_reached(self.wait, self.control_dt_s, 0.20)
        )
        self._transition(retreat_reached, self.SETTLE_OBJECT)
        if self.mode == "put_and_close":
            object_inside = root_inside_drawer(
                env,
                type("Cfg", (), {"name": "object"})(),
                type("Cfg", (), {"name": "drawer_interior_frame"})(),
            )
            settle_complete = (
                (self.state == self.SETTLE_OBJECT)
                & object_inside
                & duration_reached(self.wait, self.control_dt_s, 0.60)
            )
            self.push_start_joint[settle_complete] = task["joint_position"][settle_complete]
            self._transition(settle_complete, self.ABOVE_HANDLE_FRONT)

        # Exit above the front wall before descending outside it. A direct
        # diagonal from the drawer interior to the low push point hits the
        # inside of the front panel, preventing the hand from reaching outside.
        self._transition(
            (self.state == self.ABOVE_HANDLE_FRONT)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - handle_front_above, dim=-1) < 0.040)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.APPROACH_HANDLE_FRONT,
        )

        # The closed hand can contact the panel before its TCP reaches the
        # nominal front point. Detect that inward motion and enter the free
        # push phase immediately, rather than fighting it with drawer hold.
        near_panel = torch.linalg.vector_norm(ee_pose[:, :3] - handle, dim=-1) < 0.120
        inward_motion = task["joint_position"] < self.push_start_joint - 0.005
        front_ready = (
            (self.state == self.APPROACH_HANDLE_FRONT)
            & (
                (torch.linalg.vector_norm(ee_pose[:, :3] - handle_front, dim=-1) < 0.045)
                | (near_panel & inward_motion)
            )
            & (fingers.max(dim=-1).values < 0.010)
            & duration_reached(self.wait, self.control_dt_s, 0.20)
        )
        self._transition(front_ready, self.PUSH_DRAWER)
        self._transition(
            (self.state == self.APPROACH_HANDLE)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - handle_grasp, dim=-1) < 0.007)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.GRASP_HANDLE,
        )
        push_ready = (
            (self.state == self.GRASP_HANDLE)
            & (fingers.max(dim=-1).values < 0.010)
            & duration_reached(self.wait, self.control_dt_s, 0.40)
        )
        self.push_start_joint[push_ready] = task["joint_position"][push_ready]
        self._transition(push_ready, self.PUSH_DRAWER)
        contact_motion = (
            (self.state == self.PUSH_DRAWER)
            & inward_motion
            & near_panel
        )
        self.closing_contact_verified[contact_motion] = True
        drawer_closed = (
            (self.state == self.PUSH_DRAWER)
            & (task["joint_position"] <= 0.025)
        )
        if torch.any(drawer_closed):
            self.drawer_hold_target[drawer_closed, 0] = 0.0
        self._transition(drawer_closed, self.RELEASE_HANDLE)
        self._transition(
            (self.state == self.RELEASE_HANDLE)
            & duration_reached(self.wait, self.control_dt_s, 0.70),
            self.RETREAT_HANDLE,
        )
        self._transition(
            (self.state == self.RETREAT_HANDLE)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - handle_retreat, dim=-1) < 0.040)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.DONE,
        )

        # Match the validated object_to_drawer grasp timing: first hold the
        # final Cartesian grasp waypoint with the fingers open, then issue one
        # binary close command.  Closing immediately on the state transition
        # lets residual tracking motion push the curved banana out of reach.
        object_close_ready = (self.state > self.GRASP_OBJECT) | (
            (self.state == self.GRASP_OBJECT)
            & duration_reached(self.wait, self.control_dt_s, 0.35)
        )
        closed_object = object_close_ready & (self.state <= self.PLACE_IN_DRAWER)
        closed_handle = (self.state >= self.APPROACH_HANDLE_FRONT) & (
            self.state <= self.PUSH_DRAWER
        )
        closed_handle |= self.state == self.ABOVE_HANDLE_FRONT
        gripper = torch.where(
            closed_object | closed_handle,
            -torch.ones(self.num_envs, device=self.device),
            torch.ones(self.num_envs, device=self.device),
        )

        translation_step = torch.full(
            (self.num_envs,),
            distance_per_step(0.15, self.control_dt_s),
            device=self.device,
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
            0.06, self.control_dt_s
        )
        translation_step[self.state == self.APPROACH_HANDLE] = distance_per_step(
            0.05, self.control_dt_s
        )
        translation_step[self.state == self.GRASP_HANDLE] = distance_per_step(
            0.025, self.control_dt_s
        )
        translation_step[self.state == self.PUSH_DRAWER] = distance_per_step(
            0.06, self.control_dt_s
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

        cabinet = env.scene["cabinet"]
        pushing = self.state == self.PUSH_DRAWER
        self.drawer_hold_target[pushing, 0] = task["joint_position"][pushing]
        self.drawer_velocity_target[pushing, 0] = task["joint_velocity"][pushing]
        self.drawer_effort_target[pushing, 0] = 0.0
        held = ~pushing
        self.drawer_velocity_target[held, 0] = 0.0
        self.drawer_effort_target[held, 0] = (
            50.0
            * (self.drawer_hold_target[held, 0] - task["joint_position"][held])
            - 10.0 * task["joint_velocity"][held]
        ).clamp(-20.0, 20.0)
        cabinet.set_joint_position_target(
            self.drawer_hold_target, joint_ids=self.drawer_joint_ids
        )
        cabinet.set_joint_velocity_target(
            self.drawer_velocity_target, joint_ids=self.drawer_joint_ids
        )
        cabinet.set_joint_effort_target(
            self.drawer_effort_target, joint_ids=self.drawer_joint_ids
        )

        return absolute_pose_to_relative_action(
            ee_pose,
            self.command_pose,
            gripper,
            self.controller_scale,
            max_translation_step_m=distance_per_step(1.0, self.control_dt_s),
            max_rotation_step_rad=angle_per_step(1.6, self.control_dt_s),
        )


class DrawerSubtaskExpert:
    """Dispatch to the physical opener or the placement/closing expert."""

    def __init__(self, env):
        if env.cfg.task_mode == "open":
            self._expert = ArticulatedOpeningExpert(
                env,
                task="drawer_opening",
                joint_name=env.cfg.active_joint_name,
            )
        else:
            self._expert = _PutObjectDrawerExpert(env)

    @property
    def state(self):
        return self._expert.state

    @property
    def grasp_verified(self):
        return self._expert.grasp_verified

    @property
    def closing_contact_verified(self):
        return self._expert.closing_contact_verified

    def reset(self, env) -> None:
        self._expert.reset(env)

    def compute(self, env) -> torch.Tensor:
        return self._expert.compute(env)
