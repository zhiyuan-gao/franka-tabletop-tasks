"""Closed-loop grasp-and-pull experts for the project-local drawer and door."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.timing import (
    angle_per_step,
    distance_per_step,
    duration_reached,
    environment_control_dt_s,
)

from .common import (
    absolute_pose_to_relative_action,
    advance_pose_toward_waypoint,
    opening_task_state,
)


class ArticulatedOpeningExpert:
    REST = 0
    APPROACH_ABOVE = 1
    APPROACH_HANDLE = 2
    GRASP = 3
    PULL = 4

    def __init__(self, env, *, task: str):
        if task not in {"drawer_opening", "cabinet_opening"}:
            raise ValueError(f"unsupported opening expert task: {task}")
        self.task = task
        self.joint_name = (
            "UpperDrawerJoint" if task == "drawer_opening" else "DoorHingeJoint"
        )
        self.num_envs = env.num_envs
        self.device = env.device
        self.state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wait = torch.zeros_like(self.state)
        self.control_dt_s = environment_control_dt_s(env)
        self.controller_scale = torch.as_tensor(
            env.cfg.actions.arm_action.scale, device=self.device, dtype=torch.float32
        )
        self.drawer_direction = torch.zeros((self.num_envs, 2), device=self.device)
        self.retry_count = torch.zeros_like(self.state)
        self.grasp_verified = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.command_pose = torch.zeros((self.num_envs, 7), device=self.device)

    def reset(self, env) -> None:
        self.state.zero_()
        self.wait.zero_()
        self.retry_count.zero_()
        self.grasp_verified.zero_()
        state = opening_task_state(env, self.joint_name)
        self.command_pose.copy_(state["ee_pose"])
        if self.task == "drawer_opening":
            cabinet_quaternion = env.scene["cabinet"].data.root_quat_w
            from isaaclab.utils.math import quat_apply

            local_axis = torch.tensor(
                [0.0, 1.0, 0.0], device=self.device
            ).expand(self.num_envs, -1)
            axis_world = quat_apply(cabinet_quaternion, local_axis)
            self.drawer_direction.copy_(axis_world[:, :2])

    def _opening_direction(self, env, handle_position: torch.Tensor) -> torch.Tensor:
        if self.task == "drawer_opening":
            return self.drawer_direction
        # Door hinge in cabinet-local coordinates. The positive revolute-joint
        # tangent is a 90-degree CCW rotation of hinge-to-handle radius.
        from isaaclab.utils.math import quat_apply

        cabinet = env.scene["cabinet"]
        hinge_local = torch.tensor(
            [-0.1725, 0.1625, 0.217], device=self.device
        ).expand(self.num_envs, -1)
        hinge_world = cabinet.data.root_pos_w + quat_apply(
            cabinet.data.root_quat_w, hinge_local
        )
        robot = env.scene["robot"]
        hinge_base = hinge_world - robot.data.root_pos_w
        radius = handle_position[:, :2] - hinge_base[:, :2]
        tangent = torch.stack((-radius[:, 1], radius[:, 0]), dim=-1)
        return tangent / torch.clamp(
            torch.linalg.vector_norm(tangent, dim=-1, keepdim=True), min=1.0e-6
        )

    def compute(self, env) -> torch.Tensor:
        task_state = opening_task_state(env, self.joint_name)
        ee_pose = task_state["ee_pose"]
        handle = task_state["handle_pose"][:, :3]
        from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

        tool_horizontal = torch.tensor(
            [0.70710678, 0.70710678, 0.0, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        # The handle-local X axis follows the horizontal bar. Rotating 90 deg
        # about that axis makes the gripper approach horizontally from the
        # cabinet front while its fingers close vertically around the bar.
        # For the door this pose follows the moving handle automatically.
        tool_quaternion = quat_mul(
            task_state["handle_pose"][:, 3:7], tool_horizontal
        )
        direction = self._opening_direction(env, handle)
        local_front = torch.tensor(
            [0.0, 1.0, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        front_normal = quat_apply(
            task_state["handle_pose"][:, 3:7], local_front
        )[:, :2]
        front = handle.clone()
        front[:, :2] += 0.100 * front_normal
        # First align at the handle height with 10 cm of face-normal clearance.
        # The subsequent approach is therefore a straight insertion instead of
        # the previous diagonal descent that swept one finger across the panel.
        above = front.clone()
        # The configured EE frame is the fingertip TCP.  Passing it beyond the
        # handle puts the bar inside the finger pads; the previous +5 mm front
        # offset left the bar outside the fingertips and enabled palm hooking.
        base_penetration = 0.008 if self.task == "cabinet_opening" else 0.006
        penetration = torch.clamp(
            base_penetration + 0.002 * self.retry_count.to(torch.float32),
            max=0.012 if self.task == "cabinet_opening" else 0.010,
        )
        grasp = handle.clone()
        grasp[:, :2] -= penetration[:, None] * front_normal
        pull = handle.clone()
        pull_ramp_s = 2.40 if self.task == "cabinet_opening" else 1.60
        pull_progress = torch.clamp(
            self.wait.to(torch.float32) * self.control_dt_s / pull_ramp_s,
            min=0.0,
            max=1.0,
        )
        if self.task == "drawer_opening":
            # Ramp the Cartesian lead from 10 mm behind the handle to 30 mm in
            # front.  This builds enough tensile force to cross the calibrated
            # rail breakaway without an impulsive target jump.
            linear_lead = -0.006 + 0.036 * pull_progress
            pull[:, :2] += linear_lead[:, None] * front_normal
        else:
            # Follow the rotating face normal and ramp tangential tension about
            # the hinge while keeping the bar inside the finger-pad depth.
            tangent_lead = 0.010 + 0.035 * pull_progress
            pull[:, :2] -= 0.006 * front_normal
            pull[:, :2] += tangent_lead[:, None] * direction

        desired_position = ee_pose[:, :3].clone()
        desired_position[self.state == self.APPROACH_ABOVE] = above[
            self.state == self.APPROACH_ABOVE
        ]
        desired_position[self.state == self.APPROACH_HANDLE] = grasp[
            self.state == self.APPROACH_HANDLE
        ]
        desired_position[self.state == self.GRASP] = grasp[self.state == self.GRASP]
        desired_position[self.state == self.PULL] = pull[self.state == self.PULL]

        self.wait += 1
        transition = (self.state == self.REST) & duration_reached(
            self.wait, self.control_dt_s, 0.20
        )
        self.state[transition] = self.APPROACH_ABOVE
        self.wait[transition] = 0
        transition = (
            (self.state == self.APPROACH_ABOVE)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - above, dim=-1) < 0.008)
            & duration_reached(self.wait, self.control_dt_s, 0.20)
        )
        self.state[transition] = self.APPROACH_HANDLE
        self.wait[transition] = 0
        transition = (
            (self.state == self.APPROACH_HANDLE)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - grasp, dim=-1) < 0.006)
            & duration_reached(self.wait, self.control_dt_s, 0.20)
        )
        self.state[transition] = self.GRASP
        self.wait[transition] = 0
        handle_in_ee, _ = subtract_frame_transforms(
            ee_pose[:, :3],
            ee_pose[:, 3:7],
            task_state["handle_pose"][:, :3],
            task_state["handle_pose"][:, 3:7],
        )
        fingers = task_state["finger_positions"]
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
        verified = (self.state == self.GRASP) & valid_aperture & handle_centered
        transition = verified & duration_reached(
            self.wait, self.control_dt_s, 0.70
        )
        self.grasp_verified[transition] = True
        self.state[transition] = self.PULL
        self.wait[transition] = 0
        lost_grasp = (
            (self.state == self.PULL)
            & duration_reached(self.wait, self.control_dt_s, 0.16)
            & ~valid_aperture
        )
        self.grasp_verified[lost_grasp] = False
        self.retry_count[lost_grasp] += 1
        self.state[lost_grasp] = self.APPROACH_ABOVE
        self.wait[lost_grasp] = 0
        retry = (
            (self.state == self.GRASP)
            & duration_reached(self.wait, self.control_dt_s, 1.50)
            & ~verified
        )
        self.retry_count[retry] += 1
        self.state[retry] = self.APPROACH_ABOVE
        self.wait[retry] = 0

        waypoint_pose = torch.cat((desired_position, tool_quaternion), dim=-1)
        gripper = torch.where(
            self.state >= self.GRASP,
            -torch.ones(self.num_envs, device=self.device),
            torch.ones(self.num_envs, device=self.device),
        )
        # Stay fully open through the complete approach.  Once the final handle
        # waypoint is reached, issue one close command and wait for the existing
        # physical grasp verification before beginning the pull.
        # Generate the trajectory in physical speed units so changing policy
        # frequency preserves the calibrated Cartesian motion.
        translation_step = torch.full(
            (self.num_envs,),
            distance_per_step(0.15, self.control_dt_s),
            device=self.device,
        )
        translation_step[self.state == self.APPROACH_HANDLE] = distance_per_step(
            0.05, self.control_dt_s
        )
        translation_step[self.state == self.GRASP] = distance_per_step(
            0.025, self.control_dt_s
        )
        translation_step[self.state == self.PULL] = distance_per_step(
            0.05 if self.task == "cabinet_opening" else 0.075,
            self.control_dt_s,
        )
        self.command_pose.copy_(
            advance_pose_toward_waypoint(
                self.command_pose,
                waypoint_pose,
                translation_step,
                angle_per_step(1.0, self.control_dt_s),
            )
        )
        return absolute_pose_to_relative_action(
            ee_pose,
            self.command_pose,
            gripper,
            self.controller_scale,
            max_translation_step_m=distance_per_step(1.0, self.control_dt_s),
            max_rotation_step_rad=angle_per_step(1.6, self.control_dt_s),
        )


class DrawerOpeningExpert(ArticulatedOpeningExpert):
    def __init__(self, env):
        super().__init__(env, task="drawer_opening")


class CabinetOpeningExpert(ArticulatedOpeningExpert):
    def __init__(self, env):
        super().__init__(env, task="cabinet_opening")
