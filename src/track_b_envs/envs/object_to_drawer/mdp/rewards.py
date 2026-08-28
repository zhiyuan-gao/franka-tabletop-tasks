"""Stage-compatible dense rewards plus physical placement bonuses."""

from __future__ import annotations

import torch

from .geometry import root_inside_drawer
from .terminations import instantaneous_object_in_drawer


def ee_object_proximity(env, ee_frame_cfg, object_cfg, std: float = 0.12):
    ee = env.scene.sensors[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]
    distance = torch.linalg.vector_norm(
        ee.data.target_pos_w[..., 0, :] - obj.data.root_pos_w, dim=-1
    )
    return 1.0 - torch.tanh(distance / std)


def ee_task_target_proximity(env, ee_frame_cfg, target_frame_cfg, std: float = 0.15):
    ee = env.scene.sensors[ee_frame_cfg.name]
    target = env.scene.sensors[target_frame_cfg.name]
    distance = torch.linalg.vector_norm(
        ee.data.target_pos_w[..., 0, :] - target.data.target_pos_w[..., 0, :], dim=-1
    )
    return 1.0 - torch.tanh(distance / std)


def drawer_opening_progress(env, cabinet_cfg, normalizer: float = 0.16):
    cabinet = env.scene[cabinet_cfg.name]
    return (
        cabinet.data.joint_pos[:, cabinet_cfg.joint_ids].squeeze(-1) / normalizer
    ).clamp(0.0, 1.0)


def object_vertical_progress(env, object_cfg, start_height: float = 0.383, lift_height: float = 0.10):
    obj = env.scene[object_cfg.name]
    return ((obj.data.root_pos_w[:, 2] - start_height) / lift_height).clamp(0.0, 1.0)


def object_target_proximity(env, object_cfg, target_frame_cfg, std: float = 0.20):
    obj = env.scene[object_cfg.name]
    target = env.scene.sensors[target_frame_cfg.name]
    distance = torch.linalg.vector_norm(
        obj.data.root_pos_w - target.data.target_pos_w[..., 0, :], dim=-1
    )
    return 1.0 - torch.tanh(distance / std)


def object_in_drawer_bonus(env, object_cfg, target_frame_cfg):
    return root_inside_drawer(env, object_cfg, target_frame_cfg).float()


def settled_success_bonus(env, object_cfg, target_frame_cfg, cabinet_cfg, finger_cfg):
    # This term deliberately uses the instantaneous predicate and does not
    # mutate the consecutive-state counter owned by the termination term.
    return instantaneous_object_in_drawer(
        env, object_cfg, target_frame_cfg, cabinet_cfg, finger_cfg
    ).float()
