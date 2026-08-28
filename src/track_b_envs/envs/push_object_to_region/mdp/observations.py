"""Privileged simulator state for critic, expert, and recording."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.observations import franka_proprio


def oracle_push_state(env, robot_cfg, ee_frame_cfg, object_cfg, target_cfg) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

    robot = env.scene[robot_cfg.name]
    object_asset = env.scene[object_cfg.name]
    target_asset = env.scene[target_cfg.name]
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        object_asset.data.root_pos_w,
        object_asset.data.root_quat_w,
    )
    target_pos_b, target_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target_asset.data.root_pos_w,
        target_asset.data.root_quat_w,
    )
    object_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, object_asset.data.root_lin_vel_w)
    object_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, object_asset.data.root_ang_vel_w)
    return torch.cat(
        (
            franka_proprio(env, robot_cfg, ee_frame_cfg),
            object_pos_b,
            object_quat_b,
            object_lin_vel_b,
            object_ang_vel_b,
            target_pos_b,
            target_quat_b,
        ),
        dim=-1,
    )
