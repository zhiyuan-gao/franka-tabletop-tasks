"""Privileged articulation state for expert generation and asymmetric critics."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.observations import franka_proprio


def oracle_opening_state(env, robot_cfg, ee_frame_cfg, handle_frame_cfg, cabinet_cfg):
    from isaaclab.utils.math import subtract_frame_transforms

    robot = env.scene[robot_cfg.name]
    handle_frame = env.scene.sensors[handle_frame_cfg.name]
    handle_pos_b, handle_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        handle_frame.data.target_pos_w[..., 0, :],
        handle_frame.data.target_quat_w[..., 0, :],
    )
    cabinet = env.scene[cabinet_cfg.name]
    return torch.cat(
        (
            franka_proprio(env, robot_cfg, ee_frame_cfg),
            handle_pos_b,
            handle_quat_b,
            cabinet.data.joint_pos[:, cabinet_cfg.joint_ids],
            cabinet.data.joint_vel[:, cabinet_cfg.joint_ids],
        ),
        dim=-1,
    )
