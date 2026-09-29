"""Physical success and failure predicates for ObjectToDrawer."""

from __future__ import annotations

import torch

from .geometry import root_inside_drawer


def instantaneous_object_in_drawer(
    env,
    object_cfg,
    target_frame_cfg,
    cabinet_cfg,
    finger_cfg,
    drawer_threshold: float = 0.145,
    finger_open_threshold: float = 0.025,
    max_linear_speed: float = 0.06,
    max_angular_speed: float = 1.0,
) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    cabinet = env.scene[cabinet_cfg.name]
    robot = env.scene[finger_cfg.name]
    drawer_open = (
        cabinet.data.joint_pos[:, cabinet_cfg.joint_ids].squeeze(-1)
        >= drawer_threshold
    )
    released = (
        robot.data.joint_pos[:, finger_cfg.joint_ids].mean(dim=-1)
        >= finger_open_threshold
    )
    slow = (
        torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1) <= max_linear_speed
    ) & (
        torch.linalg.vector_norm(obj.data.root_ang_vel_w, dim=-1) <= max_angular_speed
    )
    return (
        drawer_open
        & released
        & slow
        & root_inside_drawer(env, object_cfg, target_frame_cfg)
    )


def settled_object_in_drawer(
    env,
    object_cfg,
    target_frame_cfg,
    cabinet_cfg,
    finger_cfg,
    settle_steps: int = 6,
) -> torch.Tensor:
    """Require consecutive valid simulation states, not a one-frame crossing."""

    valid = instantaneous_object_in_drawer(
        env, object_cfg, target_frame_cfg, cabinet_cfg, finger_cfg
    )
    counter = getattr(env, "_object_to_drawer_settle_counter", None)
    if counter is None or counter.shape != valid.shape:
        counter = torch.zeros_like(valid, dtype=torch.long)
        env._object_to_drawer_settle_counter = counter
    counter.copy_(torch.where(valid, counter + 1, torch.zeros_like(counter)))
    return counter >= settle_steps


def object_dropped_or_out_of_workspace(env, object_cfg) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    return (
        (pos[:, 0] < 0.10)
        | (pos[:, 0] > 1.05)
        | (torch.abs(pos[:, 1]) > 0.55)
        | (pos[:, 2] < 0.02)
        | (pos[:, 2] > 0.90)
    )
