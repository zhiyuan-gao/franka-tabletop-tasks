"""Success and failure predicates."""

from __future__ import annotations

import math
import torch

from .geometry import footprint_inside_rectangle, upright_cosine_wxyz


OBJECT_HALF_EXTENTS_XY_M = (0.0705, 0.0360)
TARGET_HALF_EXTENTS_XY_M = (0.090, 0.060)


def object_footprint_in_target(
    env,
    object_cfg,
    target_cfg,
    max_tilt_deg: float = 25.0,
    object_half_extents_xy: tuple[float, float] = OBJECT_HALF_EXTENTS_XY_M,
    target_half_extents_xy: tuple[float, float] = TARGET_HALF_EXTENTS_XY_M,
) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]
    inside = footprint_inside_rectangle(
        obj.data.root_pos_w[:, :2],
        obj.data.root_quat_w,
        target.data.root_pos_w[:, :2],
        target.data.root_quat_w,
        object_half_extents_xy,
        target_half_extents_xy,
        tolerance=0.001,
    )
    upright = upright_cosine_wxyz(obj.data.root_quat_w) >= math.cos(math.radians(max_tilt_deg))
    return inside & upright


def object_tipped(env, object_cfg, max_tilt_deg: float = 55.0) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    return upright_cosine_wxyz(obj.data.root_quat_w) < math.cos(math.radians(max_tilt_deg))


def object_out_of_workspace(env, object_cfg) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    return (pos[:, 0] < 0.15) | (pos[:, 0] > 0.92) | (torch.abs(pos[:, 1]) > 0.38) | (pos[:, 2] < -0.03)
