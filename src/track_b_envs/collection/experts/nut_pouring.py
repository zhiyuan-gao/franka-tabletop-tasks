"""Closed-loop physical expert for pouring one nut from a cup into a bowl."""

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
    nut_pouring_task_state,
)


class NutPouringExpert:
    REST = 0
    ABOVE_CUP = 1
    DESCEND_CUP = 2
    GRASP_CUP = 3
    LIFT_CUP = 4
    ABOVE_BOWL = 5
    LOWER_FOR_POUR = 6
    TILT_START = 7
    TILT_MID = 8
    TILT_FULL = 9
    POUR_HOLD = 10
    RETURN_UPRIGHT = 11
    VERIFY = 12

    STATE_NAMES = (
        "rest",
        "above_cup",
        "descend_cup",
        "grasp_cup",
        "lift_cup",
        "above_bowl",
        "lower_for_pour",
        "tilt_start",
        "tilt_mid",
        "tilt_full",
        "pour_hold",
        "return_upright",
        "verify",
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
        self.grasp_targets = torch.zeros((self.num_envs, 3), device=self.device)
        self.lift_targets = torch.zeros_like(self.grasp_targets)
        self.transfer_targets = torch.zeros_like(self.grasp_targets)
        self.pour_targets = torch.zeros_like(self.grasp_targets)
        self.cup_support_z = torch.zeros(self.num_envs, device=self.device)
        self.bowl_support_z = torch.zeros(self.num_envs, device=self.device)
        self.command_pose = torch.zeros((self.num_envs, 7), device=self.device)

    def reset(self, env) -> None:
        self.state.zero_()
        self.wait.zero_()
        self.grasp_targets.zero_()
        self.lift_targets.zero_()
        self.transfer_targets.zero_()
        self.pour_targets.zero_()
        task = nut_pouring_task_state(env)
        self.command_pose.copy_(task["ee_pose"])
        self.cup_support_z.copy_(task["cup_pose"][:, 2])
        self.bowl_support_z.copy_(task["bowl_pose"][:, 2])

    def _transition(self, mask: torch.Tensor, next_state: int) -> None:
        self.state[mask] = next_state
        self.wait[mask] = 0

    def compute(self, env) -> torch.Tensor:
        from isaaclab.utils.math import quat_mul

        task = nut_pouring_task_state(env)
        ee_pose = task["ee_pose"]
        cup = task["cup_pose"][:, :3]
        nut = task["nut_pose"][:, :3]
        bowl = task["bowl_pose"][:, :3]

        side_grasp = torch.tensor(
            [0.70710678, 0.0, 0.70710678, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        tilt_start = torch.tensor(
            [0.96592583, -0.25881905, 0.0, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        tilt_mid = torch.tensor(
            [0.81915204, -0.57357644, 0.0, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        tilt_full = torch.tensor(
            [0.50000000, -0.86602540, 0.0, 0.0], device=self.device
        ).expand(self.num_envs, -1)
        start_quaternion = quat_mul(tilt_start, side_grasp)
        mid_quaternion = quat_mul(tilt_mid, side_grasp)
        full_quaternion = quat_mul(tilt_full, side_grasp)

        cup_above = cup.clone()
        # Approach the narrower lower body from the robot-facing side.  Both
        # fingers stay outside the cup and close across its lower diameter.
        cup_above[:, 0] -= 0.120
        cup_above[:, 1] += 0.008
        cup_above[:, 2] += 0.040
        cup_grasp = cup.clone()
        # The first contact trial displaced the cup about 11 mm toward -Y.
        # Biasing the TCP toward +Y balances the opposing finger contacts.
        cup_grasp[:, 1] += 0.008
        cup_grasp[:, 2] += 0.040

        desired_position = ee_pose[:, :3].clone()
        desired_quaternion = ee_pose[:, 3:7].clone()

        def orientation_error(quaternion):
            # q and -q encode the same rotation.  Waiting for both Cartesian
            # errors prevents an angled/open finger from clipping the cup as
            # the horizontal insertion begins.
            dot = (ee_pose[:, 3:7] * quaternion).sum(dim=-1).abs().clamp(0.0, 1.0)
            return 2.0 * torch.acos(dot)

        def assign(phase, position, quaternion):
            mask = self.state == phase
            desired_position[mask] = position[mask]
            desired_quaternion[mask] = quaternion[mask]

        assign(self.ABOVE_CUP, cup_above, side_grasp)
        assign(self.DESCEND_CUP, self.grasp_targets, side_grasp)
        assign(self.GRASP_CUP, self.grasp_targets, side_grasp)
        assign(self.LIFT_CUP, self.lift_targets, side_grasp)
        assign(self.ABOVE_BOWL, self.transfer_targets, side_grasp)
        assign(self.LOWER_FOR_POUR, self.pour_targets, side_grasp)
        assign(self.TILT_START, self.pour_targets, start_quaternion)
        assign(self.TILT_MID, self.pour_targets, mid_quaternion)
        assign(self.TILT_FULL, self.pour_targets, full_quaternion)
        pour_sweep = self.pour_targets.clone()
        sweep_fraction = (
            self.wait.float() * self.control_dt_s / 2.0
        ).clamp(0.0, 1.0)
        pour_sweep[:, 1] += env.cfg.expert_pour_y_sweep_m * sweep_fraction
        assign(self.POUR_HOLD, pour_sweep, full_quaternion)
        assign(self.RETURN_UPRIGHT, self.transfer_targets, side_grasp)
        assign(self.VERIFY, self.transfer_targets, side_grasp)

        self.wait += 1
        self._transition(
            (self.state == self.REST)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.ABOVE_CUP,
        )
        above_reached = (
            (self.state == self.ABOVE_CUP)
            & (
                torch.linalg.vector_norm(ee_pose[:, :3] - cup_above, dim=-1)
                < env.cfg.expert_above_cup_position_tolerance_m
            )
            & (orientation_error(side_grasp) < 0.060)
            & duration_reached(self.wait, self.control_dt_s, 0.20)
        )
        if torch.any(above_reached):
            self.grasp_targets[above_reached] = cup_grasp[above_reached]
        self._transition(above_reached, self.DESCEND_CUP)
        self._transition(
            (self.state == self.DESCEND_CUP)
            & (
                torch.linalg.vector_norm(
                    ee_pose[:, :3] - self.grasp_targets, dim=-1
                )
                < 0.008
            )
            & (orientation_error(side_grasp) < 0.060)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.GRASP_CUP,
        )
        grasped = (self.state == self.GRASP_CUP) & duration_reached(
            self.wait, self.control_dt_s, 1.10
        )
        if torch.any(grasped):
            self.lift_targets[grasped] = ee_pose[grasped, :3]
            self.lift_targets[grasped, 2] = (
                self.cup_support_z[grasped]
                + env.cfg.expert_lift_ee_height_above_support_m
            )
        self._transition(grasped, self.LIFT_CUP)

        lifted = (
            (self.state == self.LIFT_CUP)
            & (
                cup[:, 2]
                >= self.cup_support_z
                + env.cfg.expert_cup_lift_threshold_above_support_m
            )
            & (
                torch.linalg.vector_norm(ee_pose[:, :3] - self.lift_targets, dim=-1)
                < 0.030
            )
        )
        if torch.any(lifted):
            self.transfer_targets[lifted] = bowl[lifted]
            self.transfer_targets[lifted, 1] += 0.040
            self.transfer_targets[lifted, 2] = (
                self.bowl_support_z[lifted]
                + env.cfg.expert_transfer_ee_height_above_support_m
            )
            self.pour_targets[lifted] = bowl[lifted]
            # The first physical drop landed 42.95 mm toward +X, just outside
            # the conservative 40 mm success radius.  Preserve the criterion
            # and compensate the release waypoint instead of relaxing it.
            self.pour_targets[lifted, 0] += env.cfg.expert_pour_x_bias_m
            # Start slightly toward +Y; the following 20 mm smooth sweep puts
            # the ballistic landing near the bowl center.
            self.pour_targets[lifted, 1] += env.cfg.expert_pour_y_bias_m
            # The lower-body grasp is 4 cm above the cup base.  At the final
            # tilt this keeps the wrist clear while the contents fall into the
            # bowl, rather than forcing a low, collision-limited wrist pose.
            self.pour_targets[lifted, 2] = (
                self.bowl_support_z[lifted]
                + env.cfg.expert_pour_ee_height_above_support_m
            )
        self._transition(lifted, self.ABOVE_BOWL)
        self._transition(
            (self.state == self.ABOVE_BOWL)
            & (
                torch.linalg.vector_norm(
                    ee_pose[:, :3] - self.transfer_targets, dim=-1
                )
                < 0.030
            )
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.LOWER_FOR_POUR,
        )
        self._transition(
            (self.state == self.LOWER_FOR_POUR)
            & (
                torch.linalg.vector_norm(
                    ee_pose[:, :3] - self.pour_targets, dim=-1
                )
                < 0.025
            )
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.TILT_START,
        )
        self._transition(
            (self.state == self.TILT_START)
            & duration_reached(self.wait, self.control_dt_s, 1.10),
            self.TILT_MID,
        )
        self._transition(
            (self.state == self.TILT_MID)
            & duration_reached(self.wait, self.control_dt_s, 1.40),
            self.TILT_FULL,
        )
        self._transition(
            (self.state == self.TILT_FULL)
            & duration_reached(self.wait, self.control_dt_s, 1.50),
            self.POUR_HOLD,
        )
        nut_left_cup = torch.linalg.vector_norm(nut[:, :2] - cup[:, :2], dim=-1) > 0.055
        self._transition(
            (self.state == self.POUR_HOLD)
            & nut_left_cup,
            self.RETURN_UPRIGHT,
        )
        self._transition(
            (self.state == self.RETURN_UPRIGHT)
            & duration_reached(self.wait, self.control_dt_s, 1.60),
            self.VERIFY,
        )

        closed = (self.state >= self.GRASP_CUP) & (self.state <= self.VERIFY)
        gripper = torch.where(
            closed,
            -torch.ones(self.num_envs, device=self.device),
            torch.ones(self.num_envs, device=self.device),
        )
        waypoint_pose = torch.cat((desired_position, desired_quaternion), dim=-1)
        translation_step = torch.full(
            (self.num_envs,),
            distance_per_step(0.15, self.control_dt_s),
            device=self.device,
        )
        translation_step[self.state == self.DESCEND_CUP] = distance_per_step(
            0.075, self.control_dt_s
        )
        translation_step[self.state == self.GRASP_CUP] = distance_per_step(
            0.025, self.control_dt_s
        )
        translation_step[self.state == self.LOWER_FOR_POUR] = distance_per_step(
            0.075, self.control_dt_s
        )
        tilting = (self.state >= self.TILT_START) & (self.state <= self.POUR_HOLD)
        translation_step[tilting] = distance_per_step(0.05, self.control_dt_s)
        self.command_pose.copy_(
            advance_pose_toward_waypoint(
                self.command_pose,
                waypoint_pose,
                translation_step,
                angle_per_step(0.70, self.control_dt_s),
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
