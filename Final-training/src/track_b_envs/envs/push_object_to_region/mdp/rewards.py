"""Dense shaping and sparse success reward."""

from __future__ import annotations

import torch

from .geometry import upright_cosine_wxyz
from .terminations import object_footprint_in_target


def ee_object_proximity(env, object_cfg, ee_frame_cfg, std: float = 0.12) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    ee = env.scene.sensors[ee_frame_cfg.name]
    distance = torch.linalg.vector_norm(ee.data.target_pos_w[..., 0, :] - obj.data.root_pos_w, dim=-1)
    return 1.0 - torch.tanh(distance / std)


def object_target_proximity(env, object_cfg, target_cfg, std: float = 0.20) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]
    distance = torch.linalg.vector_norm(obj.data.root_pos_w[:, :2] - target.data.root_pos_w[:, :2], dim=-1)
    return 1.0 - torch.tanh(distance / std)


def object_tilt_penalty(env, object_cfg) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    return 1.0 - upright_cosine_wxyz(obj.data.root_quat_w).clamp(-1.0, 1.0)


def success_bonus(
    env,
    object_cfg,
    target_cfg,
    object_half_extents_xy: tuple[float, float] | None = None,
    target_half_extents_xy: tuple[float, float] | None = None,
) -> torch.Tensor:
    params = {}
    if object_half_extents_xy is not None:
        params["object_half_extents_xy"] = object_half_extents_xy
    if target_half_extents_xy is not None:
        params["target_half_extents_xy"] = target_half_extents_xy
    return object_footprint_in_target(
        env, object_cfg, target_cfg, **params
    ).float()
