"""Drawer-relative geometry shared by rewards, success checks, and diagnostics."""

from __future__ import annotations

import torch


# Bounds are expressed relative to DRAWER_INTERIOR_TARGET_OFFSET_M.  They keep
# the banana root well away from the 30.5 x 27.5 cm inner walls.  The z range
# spans a settled banana on the floor without accepting an object above the rim.
INTERIOR_ROOT_LOWER_M = (-0.105, -0.090, -0.012)
INTERIOR_ROOT_UPPER_M = (0.105, 0.060, 0.055)


def object_relative_to_drawer_target(env, object_cfg, target_frame_cfg) -> torch.Tensor:
    """Return object-root xyz in the moving drawer-target coordinate frame."""

    from isaaclab.utils.math import quat_apply_inverse

    obj = env.scene[object_cfg.name]
    frame = env.scene.sensors[target_frame_cfg.name]
    target_pos = frame.data.target_pos_w[..., 0, :]
    target_quat = frame.data.target_quat_w[..., 0, :]
    return quat_apply_inverse(target_quat, obj.data.root_pos_w - target_pos)


def root_inside_drawer(env, object_cfg, target_frame_cfg, tolerance: float = 0.0) -> torch.Tensor:
    relative = object_relative_to_drawer_target(env, object_cfg, target_frame_cfg)
    lower = torch.tensor(
        INTERIOR_ROOT_LOWER_M, device=relative.device, dtype=relative.dtype
    ) - tolerance
    upper = torch.tensor(
        INTERIOR_ROOT_UPPER_M, device=relative.device, dtype=relative.dtype
    ) + tolerance
    return torch.all((relative >= lower) & (relative <= upper), dim=-1)
