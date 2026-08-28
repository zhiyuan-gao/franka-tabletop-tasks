"""Physical success and failure predicates for NutPouring."""

from __future__ import annotations

import torch

from .geometry import nut_inside_bowl


def instantaneous_nut_in_bowl(
    env,
    nut_cfg,
    bowl_cfg,
    max_linear_speed: float = 0.08,
    max_angular_speed: float = 2.0,
) -> torch.Tensor:
    nut = env.scene[nut_cfg.name]
    slow = (
        torch.linalg.vector_norm(nut.data.root_lin_vel_w, dim=-1) <= max_linear_speed
    ) & (
        torch.linalg.vector_norm(nut.data.root_ang_vel_w, dim=-1) <= max_angular_speed
    )
    return nut_inside_bowl(env, nut_cfg, bowl_cfg) & slow


def settled_nut_in_bowl(
    env, nut_cfg, bowl_cfg, settle_steps: int = 6
) -> torch.Tensor:
    valid = instantaneous_nut_in_bowl(env, nut_cfg, bowl_cfg)
    counter = getattr(env, "_nut_pouring_settle_counter", None)
    if counter is None or counter.shape != valid.shape:
        counter = torch.zeros_like(valid, dtype=torch.long)
        env._nut_pouring_settle_counter = counter
    counter.copy_(torch.where(valid, counter + 1, torch.zeros_like(counter)))
    return counter >= settle_steps


def nut_or_cup_out_of_workspace(env, nut_cfg, cup_cfg) -> torch.Tensor:
    origin = env.scene.env_origins
    nut_pos = env.scene[nut_cfg.name].data.root_pos_w - origin
    cup_pos = env.scene[cup_cfg.name].data.root_pos_w - origin

    def outside(pos):
        return (
            (pos[:, 0] < 0.10)
            | (pos[:, 0] > 1.05)
            | (torch.abs(pos[:, 1]) > 0.55)
            | (pos[:, 2] < -0.01)
            | (pos[:, 2] > 0.95)
        )

    nut_below_support = nut_pos[:, 2] < -0.002
    return outside(nut_pos) | outside(cup_pos) | nut_below_support
