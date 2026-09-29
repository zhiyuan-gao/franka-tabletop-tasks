"""Closed-loop approach-behind expert for PushObjectToRegion."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.timing import (
    distance_per_step,
    duration_reached,
    environment_control_dt_s,
)

from .common import absolute_pose_to_relative_action, task_space_state


class PushObjectToRegionExpert:
    REST = 0
    APPROACH_ABOVE = 1
    APPROACH_BEHIND = 2
    PUSH = 3

    def __init__(self, env):
        self.num_envs = env.num_envs
        self.device = env.device
        self.state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wait = torch.zeros_like(self.state)
        self.control_dt_s = environment_control_dt_s(env)
        scale = env.cfg.actions.arm_action.scale
        self.controller_scale = torch.as_tensor(scale, device=self.device, dtype=torch.float32)
        self.tool_quaternion = torch.zeros((self.num_envs, 4), device=self.device)

    def reset(self, env) -> None:
        self.state.zero_()
        self.wait.zero_()
        self.tool_quaternion.copy_(task_space_state(env)["ee_pose"][:, 3:7])

    @staticmethod
    def _targets(
        object_position: torch.Tensor, target_position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_xy = target_position[:, :2] - object_position[:, :2]
        direction = delta_xy / torch.clamp(
            torch.linalg.vector_norm(delta_xy, dim=-1, keepdim=True), min=1.0e-6
        )
        behind = object_position.clone()
        behind[:, :2] -= 0.060 * direction
        behind[:, 2] = object_position[:, 2] + 0.006
        above = behind.clone()
        above[:, 2] = 0.18
        # Follow the moving object rather than commanding one distant fixed
        # waypoint. A small lead produces continuous pushing force while the
        # target's object-relative update prevents the fingers from running away.
        push_goal = object_position.clone()
        push_goal[:, :2] += 0.030 * direction
        push_goal[:, 2] = object_position[:, 2] + 0.006
        return above, behind, push_goal, direction

    def compute(self, env) -> torch.Tensor:
        state = task_space_state(env)
        ee_pose = state["ee_pose"]
        object_position = state["object_pose"][:, :3]
        target_position = state["target_pose"][:, :3]
        above, behind, push_goal, direction = self._targets(
            object_position, target_position
        )
        # A rectangular object can slide out from the closed fingers. Once the
        # TCP has overtaken the object along the desired push direction, recover
        # through the same collision-safe above/behind approach and push again.
        forward_offset = torch.sum(
            (ee_pose[:, :2] - object_position[:, :2]) * direction, dim=-1
        )
        lost_contact = (
            (self.state == self.PUSH)
            & (forward_offset > 0.015)
            & duration_reached(self.wait, self.control_dt_s, 0.20)
        )
        self.state[lost_contact] = self.APPROACH_ABOVE
        self.wait[lost_contact] = 0
        desired_position = ee_pose[:, :3].clone()
        desired_position[self.state == self.APPROACH_ABOVE] = above[
            self.state == self.APPROACH_ABOVE
        ]
        desired_position[self.state == self.APPROACH_BEHIND] = behind[
            self.state == self.APPROACH_BEHIND
        ]
        desired_position[self.state == self.PUSH] = push_goal[self.state == self.PUSH]

        self.wait += 1
        rest_done = (self.state == self.REST) & duration_reached(
            self.wait, self.control_dt_s, 0.10
        )
        self.state[rest_done] = self.APPROACH_ABOVE
        self.wait[rest_done] = 0
        above_done = (
            (self.state == self.APPROACH_ABOVE)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - above, dim=-1) < 0.025)
            & duration_reached(self.wait, self.control_dt_s, 0.10)
        )
        self.state[above_done] = self.APPROACH_BEHIND
        self.wait[above_done] = 0
        behind_done = (
            (self.state == self.APPROACH_BEHIND)
            & (torch.linalg.vector_norm(ee_pose[:, :3] - behind, dim=-1) < 0.018)
            & duration_reached(self.wait, self.control_dt_s, 0.10)
        )
        self.state[behind_done] = self.PUSH
        self.wait[behind_done] = 0

        desired_pose = torch.cat((desired_position, self.tool_quaternion), dim=-1)
        gripper = -torch.ones(self.num_envs, device=self.device)
        return absolute_pose_to_relative_action(
            ee_pose,
            desired_pose,
            gripper,
            self.controller_scale,
            max_translation_step_m=distance_per_step(1.0, self.control_dt_s),
        )
