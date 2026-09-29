"""Bowl-relative geometry shared by rewards and success checks."""

from __future__ import annotations

import torch


# The authored bowl has a 90 mm inner bottom, 115 mm inner rim and 55 mm
# usable depth.  The root-radius bound remains inside the 45.7 mm inner bottom
# and accepts a slowly settled nut against the sloped interior, but not one on
# the 60 mm outer rim.
BOWL_SAFE_RADIUS_M = 0.043
# A half-scale M16 nut settles at roughly 2.5 mm root height when flat and
# roughly 9--10 mm when resting on an edge.  Accept both physical poses.
BOWL_ROOT_Z_LOWER_M = 0.002
BOWL_ROOT_Z_UPPER_M = 0.052


def nut_relative_to_bowl(env, nut_cfg, bowl_cfg) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply_inverse

    nut = env.scene[nut_cfg.name]
    bowl = env.scene[bowl_cfg.name]
    return quat_apply_inverse(
        bowl.data.root_quat_w, nut.data.root_pos_w - bowl.data.root_pos_w
    )


def nut_inside_bowl(env, nut_cfg, bowl_cfg, tolerance: float = 0.0) -> torch.Tensor:
    relative = nut_relative_to_bowl(env, nut_cfg, bowl_cfg)
    radial = torch.linalg.vector_norm(relative[:, :2], dim=-1)
    return (
        (radial <= BOWL_SAFE_RADIUS_M + tolerance)
        & (relative[:, 2] >= BOWL_ROOT_Z_LOWER_M - tolerance)
        & (relative[:, 2] <= BOWL_ROOT_Z_UPPER_M + tolerance)
    )
