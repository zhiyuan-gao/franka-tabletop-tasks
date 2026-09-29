"""Deployable visual/proprioceptive observation terms."""

from __future__ import annotations

import torch

from .compositing import RealBackgroundCompositor


def composited_policy_rgb(env, sensor_cfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    cache_name = "_track_b_real_background_compositor"
    compositor = getattr(env, cache_name, None)
    if compositor is None:
        compositor = RealBackgroundCompositor(env.device)
        setattr(env, cache_name, compositor)
    composite = compositor.compose(
        sensor.data.output["rgb"],
        sensor.data.output["semantic_segmentation"],
        sensor.data.intrinsic_matrices,
    )
    return compositor.policy_view(composite)


def franka_proprio(env, robot_cfg, ee_frame_cfg) -> torch.Tensor:
    """Real-available joint state and FK-derived TCP pose in the robot base."""

    from isaaclab.utils.math import subtract_frame_transforms

    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene.sensors[ee_frame_cfg.name]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        ee_frame.data.target_pos_w[..., 0, :],
        ee_frame.data.target_quat_w[..., 0, :],
    )
    return torch.cat((robot.data.joint_pos, robot.data.joint_vel, ee_pos_b, ee_quat_b), dim=-1)
