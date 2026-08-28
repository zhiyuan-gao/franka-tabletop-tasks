"""Pure torch planar target geometry used by reward and tests."""

from __future__ import annotations

import torch


def yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def upright_cosine_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Cosine between the object's local +Z axis and world +Z."""

    _, x, y, _ = quaternion.unbind(-1)
    return 1.0 - 2.0 * (x.square() + y.square())


def footprint_inside_rectangle(
    object_position_xy: torch.Tensor,
    object_quaternion_wxyz: torch.Tensor,
    target_position_xy: torch.Tensor,
    target_quaternion_wxyz: torch.Tensor,
    object_half_extents_xy: tuple[float, float],
    target_half_extents_xy: tuple[float, float],
    tolerance: float = 0.0,
) -> torch.Tensor:
    """Check all four oriented object corners against an oriented target."""

    device, dtype = object_position_xy.device, object_position_xy.dtype
    hx, hy = object_half_extents_xy
    local_corners = torch.tensor(
        [[hx, hy], [hx, -hy], [-hx, hy], [-hx, -hy]], device=device, dtype=dtype
    )
    object_yaw = yaw_from_wxyz(object_quaternion_wxyz)
    c, s = object_yaw.cos(), object_yaw.sin()
    rotation = torch.stack((c, -s, s, c), dim=-1).reshape(-1, 2, 2)
    corners_world = object_position_xy[:, None, :] + torch.bmm(
        local_corners.expand(object_position_xy.shape[0], -1, -1), rotation.transpose(1, 2)
    )
    target_yaw = yaw_from_wxyz(target_quaternion_wxyz)
    ct, st = target_yaw.cos(), target_yaw.sin()
    world_to_target = torch.stack((ct, st, -st, ct), dim=-1).reshape(-1, 2, 2)
    corners_target = torch.bmm(
        corners_world - target_position_xy[:, None, :], world_to_target.transpose(1, 2)
    )
    target_half = torch.tensor(target_half_extents_xy, device=device, dtype=dtype) + tolerance
    return torch.all(torch.abs(corners_target) <= target_half, dim=(1, 2))
