"""Coupled reset for cup, contained nut, and receiving bowl."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.randomization import (
    BOWL_XY_OFFSET_RANGE_M,
    CUP_BOWL_DISTANCE_RANGE_M,
    CUP_XY_OFFSET_RANGE_M,
    CUP_YAW_OFFSET_RANGE_RAD,
)


def reset_nut_pouring_layout(
    env,
    env_ids: torch.Tensor,
    cup_x_range=CUP_XY_OFFSET_RANGE_M,
    cup_y_range=CUP_XY_OFFSET_RANGE_M,
    cup_yaw_range=CUP_YAW_OFFSET_RANGE_RAD,
    bowl_x_range=BOWL_XY_OFFSET_RANGE_M,
    bowl_y_range=BOWL_XY_OFFSET_RANGE_M,
    cup_bowl_distance_range=CUP_BOWL_DISTANCE_RANGE_M,
) -> None:
    """Randomize the vessels while keeping the nut physically inside the cup."""

    from isaaclab.utils.math import (
        quat_apply,
        quat_apply_inverse,
        quat_from_euler_xyz,
        quat_inv,
        quat_mul,
        sample_uniform,
    )

    cup = env.scene["object"]
    nut = env.scene["factory_nut"]
    bowl = env.scene["sorting_bowl"]
    count, device = len(env_ids), cup.device

    def sample(bounds):
        return sample_uniform(bounds[0], bounds[1], (count,), device=device)

    cup_default = cup.data.default_root_state[env_ids, :7]
    nut_default = nut.data.default_root_state[env_ids, :7]
    bowl_default = bowl.data.default_root_state[env_ids, :7]
    nut_relative_position = quat_apply_inverse(
        cup_default[:, 3:7], nut_default[:, :3] - cup_default[:, :3]
    )
    nut_relative_quaternion = quat_mul(
        quat_inv(cup_default[:, 3:7]), nut_default[:, 3:7]
    )

    cup_x_offset = sample(cup_x_range)
    cup_y_offset = sample(cup_y_range)
    bowl_x_offset = sample(bowl_x_range)
    bowl_y_offset = sample(bowl_y_range)
    min_distance, max_distance = cup_bowl_distance_range
    if not 0.0 < min_distance < max_distance:
        raise ValueError("cup_bowl_distance_range must be positive and ordered")
    invalid = torch.ones(count, dtype=torch.bool, device=device)
    for _ in range(32):
        cup_xy = cup_default[:, :2] + torch.stack(
            (cup_x_offset, cup_y_offset), dim=-1
        )
        bowl_xy = bowl_default[:, :2] + torch.stack(
            (bowl_x_offset, bowl_y_offset), dim=-1
        )
        distance = torch.linalg.vector_norm(bowl_xy - cup_xy, dim=-1)
        invalid = (distance < min_distance) | (distance > max_distance)
        if not torch.any(invalid):
            break
        invalid_count = int(invalid.sum().item())
        cup_x_offset[invalid] = sample_uniform(
            cup_x_range[0], cup_x_range[1], (invalid_count,), device=device
        )
        cup_y_offset[invalid] = sample_uniform(
            cup_y_range[0], cup_y_range[1], (invalid_count,), device=device
        )
        bowl_x_offset[invalid] = sample_uniform(
            bowl_x_range[0], bowl_x_range[1], (invalid_count,), device=device
        )
        bowl_y_offset[invalid] = sample_uniform(
            bowl_y_range[0], bowl_y_range[1], (invalid_count,), device=device
        )
    cup_xy = cup_default[:, :2] + torch.stack((cup_x_offset, cup_y_offset), dim=-1)
    bowl_xy = bowl_default[:, :2] + torch.stack(
        (bowl_x_offset, bowl_y_offset), dim=-1
    )
    distance = torch.linalg.vector_norm(bowl_xy - cup_xy, dim=-1)
    invalid = (distance < min_distance) | (distance > max_distance)
    if torch.any(invalid):
        raise RuntimeError("failed to sample a valid cup-to-bowl reset layout")

    cup_pose = cup_default.clone()
    cup_pose[:, 0] += cup_x_offset
    cup_pose[:, 1] += cup_y_offset
    yaw = sample(cup_yaw_range)
    zeros = torch.zeros_like(yaw)
    cup_pose[:, 3:7] = quat_mul(
        quat_from_euler_xyz(zeros, zeros, yaw), cup_pose[:, 3:7]
    )
    cup_pose[:, :3] += env.scene.env_origins[env_ids]

    nut_pose = nut_default.clone()
    nut_pose[:, :3] = cup_pose[:, :3] + quat_apply(
        cup_pose[:, 3:7], nut_relative_position
    )
    nut_pose[:, 3:7] = quat_mul(cup_pose[:, 3:7], nut_relative_quaternion)

    bowl_pose = bowl_default.clone()
    bowl_pose[:, 0] += bowl_x_offset
    bowl_pose[:, 1] += bowl_y_offset
    bowl_pose[:, :3] += env.scene.env_origins[env_ids]

    zeros6 = torch.zeros((count, 6), device=device)
    cup.write_root_pose_to_sim(cup_pose, env_ids=env_ids)
    cup.write_root_velocity_to_sim(zeros6, env_ids=env_ids)
    nut.write_root_pose_to_sim(nut_pose, env_ids=env_ids)
    nut.write_root_velocity_to_sim(zeros6, env_ids=env_ids)
    bowl.write_root_pose_to_sim(bowl_pose, env_ids=env_ids)
    bowl.write_root_velocity_to_sim(zeros6, env_ids=env_ids)
    if hasattr(env, "_nut_pouring_settle_counter"):
        env._nut_pouring_settle_counter[env_ids] = 0
