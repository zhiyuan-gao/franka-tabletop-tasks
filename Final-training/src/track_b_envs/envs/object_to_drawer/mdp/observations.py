"""Privileged state for the asymmetric critic and demonstration recorder."""

from __future__ import annotations

import torch

from track_b_envs.envs.common.observations import franka_proprio


def oracle_object_to_drawer_state(
    env,
    robot_cfg,
    ee_frame_cfg,
    object_cfg,
    handle_frame_cfg,
    target_frame_cfg,
    cabinet_cfg,
) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    handle = env.scene.sensors[handle_frame_cfg.name]
    target = env.scene.sensors[target_frame_cfg.name]
    root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w

    def pose_in_base(position, quaternion):
        pos_b, quat_b = subtract_frame_transforms(
            root_pos, root_quat, position, quaternion
        )
        return torch.cat((pos_b, quat_b), dim=-1)

    cabinet = env.scene[cabinet_cfg.name]
    return torch.cat(
        (
            franka_proprio(env, robot_cfg, ee_frame_cfg),
            pose_in_base(obj.data.root_pos_w, obj.data.root_quat_w),
            quat_apply_inverse(root_quat, obj.data.root_lin_vel_w),
            quat_apply_inverse(root_quat, obj.data.root_ang_vel_w),
            pose_in_base(
                handle.data.target_pos_w[..., 0, :],
                handle.data.target_quat_w[..., 0, :],
            ),
            pose_in_base(
                target.data.target_pos_w[..., 0, :],
                target.data.target_quat_w[..., 0, :],
            ),
            cabinet.data.joint_pos[:, cabinet_cfg.joint_ids],
            cabinet.data.joint_vel[:, cabinet_cfg.joint_ids],
        ),
        dim=-1,
    )
