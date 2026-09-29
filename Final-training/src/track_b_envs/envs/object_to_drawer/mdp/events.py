"""Coupled cabinet/banana reset for ObjectToDrawer."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.randomization import (
    BANANA_LOCAL_XY_OFFSET_RANGE_M,
    BANANA_LOCAL_YAW_OFFSET_RANGE_RAD,
    CABINET_XY_OFFSET_RANGE_M,
    CABINET_YAW_OFFSET_RANGE_RAD,
)


def reset_object_to_drawer_layout(
    env,
    env_ids: torch.Tensor,
    x_offset_range=CABINET_XY_OFFSET_RANGE_M,
    y_offset_range=CABINET_XY_OFFSET_RANGE_M,
    yaw_offset_range=CABINET_YAW_OFFSET_RANGE_RAD,
    object_local_x_range=BANANA_LOCAL_XY_OFFSET_RANGE_M,
    object_local_y_range=BANANA_LOCAL_XY_OFFSET_RANGE_M,
    object_local_yaw_range=BANANA_LOCAL_YAW_OFFSET_RANGE_RAD,
) -> None:
    """Randomize the cabinet and independently perturb the stable banana pose."""

    from isaaclab.utils.math import (
        quat_apply,
        quat_apply_inverse,
        quat_from_euler_xyz,
        quat_inv,
        quat_mul,
        sample_uniform,
    )

    cabinet = env.scene["cabinet"]
    obj = env.scene["object"]
    count, device = len(env_ids), cabinet.device

    def sample(bounds):
        return sample_uniform(bounds[0], bounds[1], (count,), device=device)

    cabinet_default = cabinet.data.default_root_state[env_ids, :7]
    object_default = obj.data.default_root_state[env_ids, :7]
    relative_position = quat_apply_inverse(
        cabinet_default[:, 3:7], object_default[:, :3] - cabinet_default[:, :3]
    )
    relative_quaternion = quat_mul(
        quat_inv(cabinet_default[:, 3:7]), object_default[:, 3:7]
    )
    relative_position[:, 0] += sample(object_local_x_range)
    relative_position[:, 1] += sample(object_local_y_range)
    object_local_yaw = sample(object_local_yaw_range)
    zeros = torch.zeros_like(object_local_yaw)
    relative_quaternion = quat_mul(
        quat_from_euler_xyz(zeros, zeros, object_local_yaw),
        relative_quaternion,
    )

    cabinet_pose = cabinet_default.clone()
    cabinet_pose[:, 0] += sample(x_offset_range)
    cabinet_pose[:, 1] += sample(y_offset_range)
    yaw = sample(yaw_offset_range)
    zeros = torch.zeros_like(yaw)
    cabinet_pose[:, 3:7] = quat_mul(
        quat_from_euler_xyz(zeros, zeros, yaw), cabinet_pose[:, 3:7]
    )
    cabinet_pose[:, :3] += env.scene.env_origins[env_ids]

    object_pose = object_default.clone()
    object_pose[:, :3] = cabinet_pose[:, :3] + quat_apply(
        cabinet_pose[:, 3:7], relative_position
    )
    object_pose[:, 3:7] = quat_mul(
        cabinet_pose[:, 3:7], relative_quaternion
    )

    cabinet.write_root_pose_to_sim(cabinet_pose, env_ids=env_ids)
    cabinet.write_root_velocity_to_sim(
        torch.zeros((count, 6), device=device), env_ids=env_ids
    )
    obj.write_root_pose_to_sim(object_pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim(
        torch.zeros((count, 6), device=device), env_ids=env_ids
    )
    if hasattr(env, "_object_to_drawer_settle_counter"):
        env._object_to_drawer_settle_counter[env_ids] = 0
