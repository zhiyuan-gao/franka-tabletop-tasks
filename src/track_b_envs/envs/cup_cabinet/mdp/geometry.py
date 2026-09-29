"""Target-relative geometry for both cup/cabinet placement tasks."""

from __future__ import annotations

import torch


TARGET_BOUNDS_M = {
    "cup_to_drawer": ((-0.080, -0.070, -0.006), (0.080, 0.070, 0.018)),
    "cup_to_cabinet_top": ((-0.055, -0.055, -0.006), (0.055, 0.055, 0.018)),
}


def cup_relative_to_target(env, cup_cfg, target_frame_cfg) -> torch.Tensor:
    """Return cup-base xyz in the moving target coordinate frame."""

    from isaaclab.utils.math import quat_apply_inverse

    cup = env.scene[cup_cfg.name]
    target = env.scene.sensors[target_frame_cfg.name]
    target_pos = target.data.target_pos_w[..., 0, :]
    target_quat = target.data.target_quat_w[..., 0, :]
    return quat_apply_inverse(target_quat, cup.data.root_pos_w - target_pos)


def cup_inside_target(env, cup_cfg, target_frame_cfg) -> torch.Tensor:
    relative = cup_relative_to_target(env, cup_cfg, target_frame_cfg)
    try:
        lower_values, upper_values = TARGET_BOUNDS_M[env.cfg.task_name]
    except KeyError as exc:
        raise ValueError(f"unsupported cup/cabinet task {env.cfg.task_name!r}") from exc
    lower = torch.tensor(lower_values, device=relative.device, dtype=relative.dtype)
    upper = torch.tensor(upper_values, device=relative.device, dtype=relative.dtype)
    return torch.all((relative >= lower) & (relative <= upper), dim=-1)


def cup_is_upright(env, cup_cfg, minimum_up_z: float = 0.94) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply

    cup = env.scene[cup_cfg.name]
    local_up = torch.tensor(
        [0.0, 0.0, 1.0], device=cup.device, dtype=cup.data.root_quat_w.dtype
    ).expand(env.num_envs, -1)
    return quat_apply(cup.data.root_quat_w, local_up)[:, 2] >= minimum_up_z
