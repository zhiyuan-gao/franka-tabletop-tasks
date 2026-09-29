"""Success and failure predicates for physical cup placement."""

from __future__ import annotations

import torch

from .geometry import cup_inside_target, cup_is_upright


def instantaneous_cup_at_target(
    env,
    cup_cfg,
    target_frame_cfg,
    cabinet_cfg,
    finger_cfg,
    drawer_open_threshold: float = 0.205,
    finger_open_threshold: float = 0.025,
    max_linear_speed: float = 0.06,
    max_angular_speed: float = 1.0,
) -> torch.Tensor:
    cup = env.scene[cup_cfg.name]
    cabinet = env.scene[cabinet_cfg.name]
    robot = env.scene[finger_cfg.name]
    drawer_open = (
        cabinet.data.joint_pos[:, cabinet_cfg.joint_ids].squeeze(-1)
        >= drawer_open_threshold
    )
    released = (
        robot.data.joint_pos[:, finger_cfg.joint_ids].mean(dim=-1)
        >= finger_open_threshold
    )
    slow = (
        torch.linalg.vector_norm(cup.data.root_lin_vel_w, dim=-1) <= max_linear_speed
    ) & (
        torch.linalg.vector_norm(cup.data.root_ang_vel_w, dim=-1) <= max_angular_speed
    )
    return (
        drawer_open
        & released
        & slow
        & cup_is_upright(env, cup_cfg)
        & cup_inside_target(env, cup_cfg, target_frame_cfg)
    )


def settled_cup_at_target(
    env,
    cup_cfg,
    target_frame_cfg,
    cabinet_cfg,
    finger_cfg,
    settle_steps: int = 6,
) -> torch.Tensor:
    valid = instantaneous_cup_at_target(
        env, cup_cfg, target_frame_cfg, cabinet_cfg, finger_cfg
    )
    counter = getattr(env, "_cup_cabinet_settle_counter", None)
    if counter is None or counter.shape != valid.shape:
        counter = torch.zeros_like(valid, dtype=torch.long)
        env._cup_cabinet_settle_counter = counter
    counter.copy_(torch.where(valid, counter + 1, torch.zeros_like(counter)))
    return counter >= settle_steps


def cup_dropped_or_out_of_workspace(env, cup_cfg) -> torch.Tensor:
    position = env.scene[cup_cfg.name].data.root_pos_w - env.scene.env_origins
    return (
        (position[:, 0] < 0.10)
        | (position[:, 0] > 1.05)
        | (torch.abs(position[:, 1]) > 0.60)
        | (position[:, 2] < -0.015)
        | (position[:, 2] > 0.95)
    )
