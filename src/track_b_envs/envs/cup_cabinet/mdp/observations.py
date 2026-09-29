"""Privileged critic state for cup/cabinet placement."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.observations import franka_proprio


def oracle_cup_cabinet_state(
    env,
    robot_cfg,
    ee_frame_cfg,
    cup_cfg,
    target_frame_cfg,
    cabinet_cfg,
) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

    robot = env.scene[robot_cfg.name]
    cup = env.scene[cup_cfg.name]
    target = env.scene.sensors[target_frame_cfg.name]
    cabinet = env.scene[cabinet_cfg.name]
    root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w

    cup_pos, cup_quat = subtract_frame_transforms(
        root_pos, root_quat, cup.data.root_pos_w, cup.data.root_quat_w
    )
    target_pos, target_quat = subtract_frame_transforms(
        root_pos,
        root_quat,
        target.data.target_pos_w[..., 0, :],
        target.data.target_quat_w[..., 0, :],
    )
    return torch.cat(
        (
            franka_proprio(env, robot_cfg, ee_frame_cfg),
            cup_pos,
            cup_quat,
            quat_apply_inverse(root_quat, cup.data.root_lin_vel_w),
            quat_apply_inverse(root_quat, cup.data.root_ang_vel_w),
            target_pos,
            target_quat,
            cabinet.data.joint_pos[:, cabinet_cfg.joint_ids],
            cabinet.data.joint_vel[:, cabinet_cfg.joint_ids],
        ),
        dim=-1,
    )
