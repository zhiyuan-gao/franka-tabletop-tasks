"""Physical experts for top- and side-grasp cup/cabinet tasks."""

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
    cup_cabinet_task_state,
)


class _CupCabinetExpert:
    TASK_NAME = ""

    REST = 0
    SAFE_HIGH = 1
    OUTSIDE_LOW = 2
    PREGRASP = 3
    APPROACH = 4
    GRASP = 5
    CLEAR_DRAWER = 6
    LIFT = 7
    ABOVE_TARGET = 8
    LOWER = 9
    RELEASE = 10
    RETREAT = 11
    SETTLE = 12

    STATE_NAMES = (
        "rest",
        "safe_high",
        "outside_low",
        "pregrasp",
        "approach",
        "grasp",
        "clear_drawer",
        "lift",
        "above_target",
        "lower",
        "release",
        "retreat",
        "settle",
    )

    def __init__(self, env):
        if env.cfg.task_name != self.TASK_NAME:
            raise ValueError(
                f"{type(self).__name__} requires {self.TASK_NAME!r}, "
                f"got {env.cfg.task_name!r}"
            )
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
        self.command_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.grasp_pose = torch.zeros_like(self.command_pose)
        self.clear_pose = torch.zeros_like(self.command_pose)
        self.lift_pose = torch.zeros_like(self.command_pose)
        self.above_target_pose = torch.zeros_like(self.command_pose)
        self.place_pose = torch.zeros_like(self.command_pose)
        self.retreat_pose = torch.zeros_like(self.command_pose)
        self.initial_cup_height = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env) -> None:
        self.state.zero_()
        self.wait.zero_()
        task = cup_cabinet_task_state(env)
        self.command_pose.copy_(task["ee_pose"])
        self.grasp_pose.zero_()
        self.clear_pose.zero_()
        self.lift_pose.zero_()
        self.above_target_pose.zero_()
        self.place_pose.zero_()
        self.retreat_pose.zero_()
        self.initial_cup_height.copy_(task["cup_pose"][:, 2])

    def _transition(self, mask: torch.Tensor, next_state: int) -> None:
        self.state[mask] = next_state
        self.wait[mask] = 0

    def compute(self, env) -> torch.Tensor:
        task = cup_cabinet_task_state(env)
        ee_pose = task["ee_pose"]
        cup_pose = task["cup_pose"]
        target_pose = task["target_pose"]

        side_from_front = torch.tensor(
            [0.5, 0.5, 0.5, -0.5], device=self.device
        ).expand(self.num_envs, -1)
        if self.TASK_NAME == "cup_to_drawer":
            # Vertical tool pose with the finger axis aligned to the cabinet.
            # The older 35-degree yaw came from a different cabinet placement
            # and unnecessarily drove the wrist toward its limit here.
            grasp_quaternion = torch.tensor(
                [0.0, 0.99708688, 0.07627424, 0.0], device=self.device
            ).expand(self.num_envs, -1)
        else:
            # Horizontal tool approach from the table/front side (-world Y),
            # while the finger closing axis remains horizontal (world X).
            grasp_quaternion = side_from_front

        live_grasp_position = cup_pose[:, :3].clone()
        live_grasp_position[:, 2] += 0.043
        pregrasp_position = live_grasp_position.clone()
        if self.TASK_NAME == "cup_to_drawer":
            pregrasp_position[:, 2] += 0.140
        else:
            # Keep the Figure-4 cup location and approach it horizontally from
            # the robot/table-front side.  This avoids both the open upper
            # drawer and the low-workspace wrist singularity of a world-X
            # approach at this location.
            live_grasp_position[:, 0] -= 0.010
            live_grasp_position[:, 1] += 0.008
            live_grasp_position[:, 2] -= 0.003
            pregrasp_position = live_grasp_position.clone()
            pregrasp_position[:, 1] += 0.120

        desired_position = ee_pose[:, :3].clone()
        desired_quaternion = ee_pose[:, 3:7].clone()

        def orientation_error(quaternion):
            dot = (ee_pose[:, 3:7] * quaternion).sum(dim=-1).abs().clamp(0.0, 1.0)
            return 2.0 * torch.acos(dot)

        def assign(phase, pose):
            mask = self.state == phase
            desired_position[mask] = pose[mask, :3]
            desired_quaternion[mask] = pose[mask, 3:7]

        live_pregrasp_pose = torch.cat((pregrasp_position, grasp_quaternion), dim=-1)
        outside_low_position = pregrasp_position.clone()
        outside_low_position[:, 1] += 0.200
        outside_low_pose = torch.cat(
            (outside_low_position, grasp_quaternion), dim=-1
        )
        safe_high_position = outside_low_position.clone()
        safe_high_position[:, 2] = 0.450
        safe_high_pose = torch.cat((safe_high_position, grasp_quaternion), dim=-1)
        pregrasp_position_tolerance = (
            0.020 if self.TASK_NAME == "cup_to_drawer" else 0.012
        )
        approach_position_tolerance = (
            0.008 if self.TASK_NAME == "cup_to_drawer" else 0.012
        )
        orientation_tolerance = 0.070
        if self.TASK_NAME == "cup_to_cabinet_top":
            assign(self.SAFE_HIGH, safe_high_pose)
            assign(self.OUTSIDE_LOW, outside_low_pose)
        assign(self.PREGRASP, live_pregrasp_pose)
        assign(self.APPROACH, self.grasp_pose)
        assign(self.GRASP, self.grasp_pose)
        assign(self.CLEAR_DRAWER, self.clear_pose)
        assign(self.LIFT, self.lift_pose)
        assign(self.ABOVE_TARGET, self.above_target_pose)
        assign(self.LOWER, self.place_pose)
        assign(self.RELEASE, self.place_pose)
        assign(self.RETREAT, self.retreat_pose)
        assign(self.SETTLE, self.retreat_pose)

        self.wait += 1
        self._transition(
            (self.state == self.REST)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            (
                self.SAFE_HIGH
                if self.TASK_NAME == "cup_to_cabinet_top"
                else self.PREGRASP
            ),
        )
        self._transition(
            (self.state == self.SAFE_HIGH)
            & (
                torch.linalg.vector_norm(
                    ee_pose[:, :3] - safe_high_position, dim=-1
                )
                < 0.020
            )
            & (orientation_error(grasp_quaternion) < orientation_tolerance)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.OUTSIDE_LOW,
        )
        self._transition(
            (self.state == self.OUTSIDE_LOW)
            & (
                torch.linalg.vector_norm(
                    ee_pose[:, :3] - outside_low_position, dim=-1
                )
                < 0.015
            )
            & (orientation_error(grasp_quaternion) < orientation_tolerance)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.PREGRASP,
        )
        pregrasp_reached = (
            (self.state == self.PREGRASP)
            & (
                torch.linalg.vector_norm(
                    ee_pose[:, :3] - pregrasp_position, dim=-1
                )
                < pregrasp_position_tolerance
            )
            & (orientation_error(grasp_quaternion) < orientation_tolerance)
            & duration_reached(self.wait, self.control_dt_s, 0.20)
        )
        if torch.any(pregrasp_reached):
            self.grasp_pose[pregrasp_reached, :3] = live_grasp_position[pregrasp_reached]
            self.grasp_pose[pregrasp_reached, 3:7] = grasp_quaternion[
                pregrasp_reached
            ]
        self._transition(pregrasp_reached, self.APPROACH)
        self._transition(
            (self.state == self.APPROACH)
            & (
                torch.linalg.vector_norm(
                    ee_pose[:, :3] - self.grasp_pose[:, :3], dim=-1
                )
                < approach_position_tolerance
            )
            & (orientation_error(self.grasp_pose[:, 3:7]) < orientation_tolerance)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.GRASP,
        )

        grasp_complete = (
            (self.state == self.GRASP)
            & duration_reached(self.wait, self.control_dt_s, 1.00)
        )
        if torch.any(grasp_complete):
            self.clear_pose[grasp_complete] = self.grasp_pose[grasp_complete]
            if self.TASK_NAME == "cup_to_cabinet_top":
                # Retract toward the table front while lifting just enough to
                # avoid dragging the cup on the tabletop.  The cup still stays
                # well below the open drawer during this clearance segment.
                # the robot-suitable alternative to the human top grasp shown
                # in KnowDemo-Fig4, which would collide with the open drawer.
                self.clear_pose[grasp_complete, 1] += 0.160
                self.clear_pose[grasp_complete, 2] += 0.040
            self.lift_pose[grasp_complete] = self.clear_pose[grasp_complete]
            self.lift_pose[grasp_complete, 2] = 0.545
        self._transition(grasp_complete, self.CLEAR_DRAWER)

        clear_position_tolerance = (
            0.018 if self.TASK_NAME == "cup_to_drawer" else 0.030
        )
        clear_reached = (
            (self.state == self.CLEAR_DRAWER)
            & (
                torch.linalg.vector_norm(
                    ee_pose[:, :3] - self.clear_pose[:, :3], dim=-1
                )
                < clear_position_tolerance
            )
            & duration_reached(self.wait, self.control_dt_s, 0.20)
        )
        self._transition(clear_reached, self.LIFT)

        lifted = (
            (self.state == self.LIFT)
            & (cup_pose[:, 2] >= self.initial_cup_height + 0.075)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - self.lift_pose[:, :3], dim=-1) < 0.030)
        )
        if torch.any(lifted):
            self.above_target_pose[lifted, :3] = (
                target_pose[lifted, :3]
                - (cup_pose[lifted, :3] - ee_pose[lifted, :3])
            )
            self.above_target_pose[lifted, 2] += 0.200
            self.above_target_pose[lifted, 3:7] = grasp_quaternion[lifted]
            self.place_pose[lifted, :3] = self.above_target_pose[lifted, :3]
            self.place_pose[lifted, 2] -= 0.200
            self.place_pose[lifted, 3:7] = grasp_quaternion[lifted]
            self.retreat_pose[lifted] = self.above_target_pose[lifted]
        self._transition(lifted, self.ABOVE_TARGET)
        self._transition(
            (self.state == self.ABOVE_TARGET)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - self.above_target_pose[:, :3], dim=-1) < 0.030)
            & duration_reached(self.wait, self.control_dt_s, 0.20),
            self.LOWER,
        )
        cup_near_target = (
            torch.linalg.vector_norm(
                cup_pose[:, :2] - target_pose[:, :2], dim=-1
            )
            < 0.035
        ) & (
            (cup_pose[:, 2] - target_pose[:, 2]).abs()
            < (0.035 if self.TASK_NAME == "cup_to_drawer" else 0.022)
        )
        self._transition(
            (self.state == self.LOWER)
            & cup_near_target
            & duration_reached(self.wait, self.control_dt_s, 0.25),
            self.RELEASE,
        )
        self._transition(
            (self.state == self.RELEASE)
            & duration_reached(self.wait, self.control_dt_s, 0.80),
            self.RETREAT,
        )
        self._transition(
            (self.state == self.RETREAT)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - self.retreat_pose[:, :3], dim=-1) < 0.035),
            self.SETTLE,
        )

        closed = (self.state >= self.GRASP) & (self.state <= self.LOWER)
        gripper = torch.where(
            closed,
            -torch.ones(self.num_envs, device=self.device),
            torch.ones(self.num_envs, device=self.device),
        )
        translation_step = torch.full(
            (self.num_envs,),
            distance_per_step(0.14, self.control_dt_s),
            device=self.device,
        )
        translation_step[self.state == self.APPROACH] = distance_per_step(
            0.055, self.control_dt_s
        )
        translation_step[self.state == self.GRASP] = distance_per_step(
            0.020, self.control_dt_s
        )
        translation_step[self.state == self.CLEAR_DRAWER] = distance_per_step(
            0.055, self.control_dt_s
        )
        translation_step[self.state == self.LIFT] = distance_per_step(
            0.060, self.control_dt_s
        )
        translation_step[self.state == self.ABOVE_TARGET] = distance_per_step(
            0.070, self.control_dt_s
        )
        translation_step[self.state == self.LOWER] = distance_per_step(
            0.055, self.control_dt_s
        )
        waypoint_pose = torch.cat((desired_position, desired_quaternion), dim=-1)
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


class CupToDrawerExpert(_CupCabinetExpert):
    """Top-down cup grasp followed by placement into the open upper drawer."""

    TASK_NAME = "cup_to_drawer"


class CupToCabinetTopExpert(_CupCabinetExpert):
    """Side grasp and lateral extraction from below the open upper drawer."""

    TASK_NAME = "cup_to_cabinet_top"
