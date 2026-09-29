"""Privileged state for NutPouring."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.observations import franka_proprio


def oracle_nut_pouring_state(
    env, robot_cfg, ee_frame_cfg, cup_cfg, nut_cfg, bowl_cfg
) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

    robot = env.scene[robot_cfg.name]
    cup = env.scene[cup_cfg.name]
    nut = env.scene[nut_cfg.name]
    bowl = env.scene[bowl_cfg.name]
    root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w

    def pose_in_base(asset):
        pos, quat = subtract_frame_transforms(
            root_pos, root_quat, asset.data.root_pos_w, asset.data.root_quat_w
        )
        return torch.cat((pos, quat), dim=-1)

    return torch.cat(
        (
            franka_proprio(env, robot_cfg, ee_frame_cfg),
            pose_in_base(cup),
            quat_apply_inverse(root_quat, cup.data.root_lin_vel_w),
            quat_apply_inverse(root_quat, cup.data.root_ang_vel_w),
            pose_in_base(nut),
            quat_apply_inverse(root_quat, nut.data.root_lin_vel_w),
            quat_apply_inverse(root_quat, nut.data.root_ang_vel_w),
            pose_in_base(bowl),
        ),
        dim=-1,
    )
