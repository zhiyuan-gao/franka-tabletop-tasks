"""Simple collection-time shaping for articulated opening tasks."""

from __future__ import annotations

import torch

from .terminations import opening_joint_reached


def ee_handle_proximity(env, ee_frame_cfg, handle_frame_cfg, std: float = 0.12):
    ee = env.scene.sensors[ee_frame_cfg.name]
    handle = env.scene.sensors[handle_frame_cfg.name]
    distance = torch.linalg.vector_norm(
        ee.data.target_pos_w[..., 0, :] - handle.data.target_pos_w[..., 0, :], dim=-1
    )
    return 1.0 - torch.tanh(distance / std)


def joint_opening(env, cabinet_cfg):
    cabinet = env.scene[cabinet_cfg.name]
    return cabinet.data.joint_pos[:, cabinet_cfg.joint_ids].squeeze(-1)


def opening_success_bonus(env, cabinet_cfg, threshold: float):
    return opening_joint_reached(env, cabinet_cfg, threshold).float()
