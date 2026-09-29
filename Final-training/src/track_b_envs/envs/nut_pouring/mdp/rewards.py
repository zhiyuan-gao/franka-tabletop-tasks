"""Dense rewards for physical cup grasp, transfer, and pour."""

from __future__ import annotations

import torch

from .geometry import nut_inside_bowl
from .terminations import instantaneous_nut_in_bowl


def ee_cup_proximity(env, ee_frame_cfg, cup_cfg, std: float = 0.15):
    ee = env.scene.sensors[ee_frame_cfg.name]
    cup = env.scene[cup_cfg.name]
    grasp_point = cup.data.root_pos_w.clone()
    grasp_point[:, 2] += 0.040
    distance = torch.linalg.vector_norm(
        ee.data.target_pos_w[..., 0, :] - grasp_point, dim=-1
    )
    return 1.0 - torch.tanh(distance / std)


def cup_lift_progress(env, cup_cfg, start_height: float = 0.001, target_height: float = 0.16):
    cup = env.scene[cup_cfg.name]
    return ((cup.data.root_pos_w[:, 2] - start_height) / target_height).clamp(0.0, 1.0)


def cup_above_bowl(env, cup_cfg, bowl_cfg, std: float = 0.18):
    cup = env.scene[cup_cfg.name]
    bowl = env.scene[bowl_cfg.name]
    horizontal = torch.linalg.vector_norm(
        cup.data.root_pos_w[:, :2] - bowl.data.root_pos_w[:, :2], dim=-1
    )
    return 1.0 - torch.tanh(horizontal / std)


def cup_tilt_progress(env, cup_cfg):
    from isaaclab.utils.math import quat_apply

    cup = env.scene[cup_cfg.name]
    local_up = torch.tensor(
        [0.0, 0.0, 1.0], device=cup.device, dtype=cup.data.root_quat_w.dtype
    ).expand(env.num_envs, -1)
    up_world = quat_apply(cup.data.root_quat_w, local_up)
    return (1.0 - up_world[:, 2]).clamp(0.0, 1.0)


def nut_bowl_proximity(env, nut_cfg, bowl_cfg, std: float = 0.20):
    nut = env.scene[nut_cfg.name]
    bowl = env.scene[bowl_cfg.name]
    target = bowl.data.root_pos_w.clone()
    target[:, 2] += 0.025
    distance = torch.linalg.vector_norm(nut.data.root_pos_w - target, dim=-1)
    return 1.0 - torch.tanh(distance / std)


def nut_inside_bowl_bonus(env, nut_cfg, bowl_cfg):
    return nut_inside_bowl(env, nut_cfg, bowl_cfg).float()


def settled_success_bonus(env, nut_cfg, bowl_cfg):
    return instantaneous_nut_in_bowl(env, nut_cfg, bowl_cfg).float()
