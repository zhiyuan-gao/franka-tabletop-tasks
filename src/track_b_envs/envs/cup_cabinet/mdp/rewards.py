"""Dense rewards for the two physical cup placement tasks."""

from __future__ import annotations

import torch

from .geometry import cup_inside_target, cup_is_upright
from .terminations import instantaneous_cup_at_target


def ee_cup_proximity(env, ee_frame_cfg, cup_cfg, std: float = 0.15):
    ee = env.scene.sensors[ee_frame_cfg.name]
    cup = env.scene[cup_cfg.name]
    grasp_point = cup.data.root_pos_w.clone()
    grasp_point[:, 2] += 0.043
    distance = torch.linalg.vector_norm(
        ee.data.target_pos_w[..., 0, :] - grasp_point, dim=-1
    )
    return 1.0 - torch.tanh(distance / std)


def cup_lift_progress(env, cup_cfg, lift_height: float = 0.12):
    cup = env.scene[cup_cfg.name]
    start_height = 0.371 if env.cfg.task_name == "cup_to_drawer" else 0.001
    return ((cup.data.root_pos_w[:, 2] - start_height) / lift_height).clamp(0.0, 1.0)


def cup_target_proximity(env, cup_cfg, target_frame_cfg, std: float = 0.18):
    cup = env.scene[cup_cfg.name]
    target = env.scene.sensors[target_frame_cfg.name]
    distance = torch.linalg.vector_norm(
        cup.data.root_pos_w - target.data.target_pos_w[..., 0, :], dim=-1
    )
    return 1.0 - torch.tanh(distance / std)


def cup_at_target_bonus(env, cup_cfg, target_frame_cfg):
    return (cup_inside_target(env, cup_cfg, target_frame_cfg) & cup_is_upright(env, cup_cfg)).float()


def settled_success_bonus(env, cup_cfg, target_frame_cfg, cabinet_cfg, finger_cfg):
    return instantaneous_cup_at_target(
        env, cup_cfg, target_frame_cfg, cabinet_cfg, finger_cfg
    ).float()
