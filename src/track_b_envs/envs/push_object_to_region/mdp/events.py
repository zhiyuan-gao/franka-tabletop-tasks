"""Randomize the pushed object while keeping the marker-aligned paper target fixed."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.constants import WORLD_MARKER_RECTANGLE_YAW_RAD


def reset_push_layout(
    env,
    env_ids: torch.Tensor,
    # Region centered at the user-annotated policy-image location.
    object_x_range=(0.430, 0.490),
    object_y_range=(-0.335, -0.275),
    object_yaw_range=(
        WORLD_MARKER_RECTANGLE_YAW_RAD - 0.2618,
        WORLD_MARKER_RECTANGLE_YAW_RAD + 0.2618,
    ),
) -> None:
    from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

    object_asset = env.scene["object"]
    count = len(env_ids)
    device = object_asset.device

    def sample(bounds):
        return sample_uniform(bounds[0], bounds[1], (count,), device=device)

    object_pose = object_asset.data.default_root_state[env_ids, :7].clone()
    object_pose[:, 0] = sample(object_x_range)
    object_pose[:, 1] = sample(object_y_range)
    object_pose[:, :3] += env.scene.env_origins[env_ids]
    object_yaw = sample(object_yaw_range)
    zeros = torch.zeros_like(object_yaw)
    object_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, object_yaw)

    object_asset.write_root_pose_to_sim(object_pose, env_ids=env_ids)
    object_asset.write_root_velocity_to_sim(torch.zeros((count, 6), device=device), env_ids=env_ids)
