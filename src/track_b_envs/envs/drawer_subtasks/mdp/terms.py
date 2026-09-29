"""MDP terms shared by the six calibrated upper/lower drawer subtasks."""

from __future__ import annotations

import torch

from track_b_envs.envs.object_to_drawer.mdp.geometry import root_inside_drawer


def drawer_task_progress(
    env,
    cabinet_cfg,
    mode: str,
    open_position: float = 0.22,
) -> torch.Tensor:
    """Return normalized joint progress toward the requested drawer state."""

    joint = env.scene[cabinet_cfg.name].data.joint_pos[
        :, cabinet_cfg.joint_ids
    ].squeeze(-1)
    if mode in {"open", "put"}:
        return (joint / open_position).clamp(0.0, 1.0)
    if mode == "put_and_close":
        return (1.0 - joint / open_position).clamp(0.0, 1.0)
    raise ValueError(f"unsupported drawer subtask mode: {mode}")


def instantaneous_drawer_subtask_success(
    env,
    object_cfg,
    target_frame_cfg,
    cabinet_cfg,
    finger_cfg,
    mode: str,
    open_threshold: float = 0.16,
    closed_threshold: float = 0.025,
    finger_open_threshold: float = 0.025,
    max_linear_speed: float = 0.06,
    max_angular_speed: float = 1.0,
    opening_task_threshold: float = 0.21,
) -> torch.Tensor:
    """Evaluate the physical terminal contract for one drawer subtask."""

    cabinet = env.scene[cabinet_cfg.name]
    joint = cabinet.data.joint_pos[:, cabinet_cfg.joint_ids].squeeze(-1)
    if mode == "open":
        # Opening demonstrations should show nearly the full 0.22 m travel.
        # Placement retains its independent 0.16 m open-drawer requirement.
        return joint >= opening_task_threshold

    obj = env.scene[object_cfg.name]
    robot = env.scene[finger_cfg.name]
    released = (
        robot.data.joint_pos[:, finger_cfg.joint_ids].mean(dim=-1)
        >= finger_open_threshold
    )
    slow = (
        torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)
        <= max_linear_speed
    ) & (
        torch.linalg.vector_norm(obj.data.root_ang_vel_w, dim=-1)
        <= max_angular_speed
    )
    inside = root_inside_drawer(env, object_cfg, target_frame_cfg)
    if mode == "put":
        drawer_state_valid = joint >= open_threshold
    elif mode == "put_and_close":
        drawer_state_valid = joint <= closed_threshold
    else:
        raise ValueError(f"unsupported drawer subtask mode: {mode}")
    return drawer_state_valid & released & slow & inside


def settled_drawer_subtask_success(
    env,
    object_cfg,
    target_frame_cfg,
    cabinet_cfg,
    finger_cfg,
    mode: str,
    settle_steps: int = 6,
) -> torch.Tensor:
    """Require the requested final state for consecutive control steps."""

    valid = instantaneous_drawer_subtask_success(
        env,
        object_cfg,
        target_frame_cfg,
        cabinet_cfg,
        finger_cfg,
        mode,
    )
    counter = getattr(env, "_drawer_subtask_settle_counter", None)
    if counter is None or counter.shape != valid.shape:
        counter = torch.zeros_like(valid, dtype=torch.long)
        env._drawer_subtask_settle_counter = counter
    counter.copy_(torch.where(valid, counter + 1, torch.zeros_like(counter)))
    return counter >= settle_steps


def drawer_subtask_success_bonus(
    env,
    object_cfg,
    target_frame_cfg,
    cabinet_cfg,
    finger_cfg,
    mode: str,
) -> torch.Tensor:
    """Non-mutating instantaneous success bonus."""

    return instantaneous_drawer_subtask_success(
        env,
        object_cfg,
        target_frame_cfg,
        cabinet_cfg,
        finger_cfg,
        mode,
    ).float()


def reset_drawer_subtask_counter(env, env_ids: torch.Tensor) -> None:
    if hasattr(env, "_drawer_subtask_settle_counter"):
        env._drawer_subtask_settle_counter[env_ids] = 0
