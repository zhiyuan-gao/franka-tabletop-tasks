"""Conservative cabinet-pose randomization for demonstration diversity."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.randomization import (
    CABINET_XY_OFFSET_RANGE_M,
    CABINET_YAW_OFFSET_RANGE_RAD,
)


def reset_cabinet_pose(
    env,
    env_ids: torch.Tensor,
    x_offset_range=CABINET_XY_OFFSET_RANGE_M,
    y_offset_range=CABINET_XY_OFFSET_RANGE_M,
    yaw_offset_range=CABINET_YAW_OFFSET_RANGE_RAD,
) -> None:
    """Reset the fixed cabinet near its validated nominal pose."""

    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

    cabinet = env.scene["cabinet"]
    count = len(env_ids)
    device = cabinet.device

    def sample(bounds):
        return sample_uniform(bounds[0], bounds[1], (count,), device=device)

    pose = cabinet.data.default_root_state[env_ids, :7].clone()
    pose[:, 0] += sample(x_offset_range)
    pose[:, 1] += sample(y_offset_range)
    pose[:, :3] += env.scene.env_origins[env_ids]
    yaw = sample(yaw_offset_range)
    zeros = torch.zeros_like(yaw)
    pose[:, 3:7] = quat_mul(
        pose[:, 3:7], quat_from_euler_xyz(zeros, zeros, yaw)
    )
    cabinet.write_root_pose_to_sim(pose, env_ids=env_ids)
    cabinet.write_root_velocity_to_sim(
        torch.zeros((count, 6), device=device), env_ids=env_ids
    )
