"""Cartesian-action helpers shared by scripted Franka experts."""

from __future__ import annotations

import torch


def canonicalize_relative_action(action: torch.Tensor) -> torch.Tensor:
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(f"expected (N, 7) actions, got {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise ValueError("expert produced non-finite actions")
    result = action.clone()
    result[:, :6].clamp_(-1.0, 1.0)
    # The deployment gripper accepts only open/close commands.  Keep the
    # simulation and recorded expert action contract binary as well: positive
    # values mean fully open and zero/negative values mean fully closed.
    result[:, 6] = torch.where(
        result[:, 6] > 0.0,
        torch.ones_like(result[:, 6]),
        -torch.ones_like(result[:, 6]),
    )
    return result


def absolute_pose_to_relative_action(
    current_pose_wxyz: torch.Tensor,
    target_pose_wxyz: torch.Tensor,
    gripper: torch.Tensor,
    controller_scale: torch.Tensor,
    *,
    max_translation_step_m: float | torch.Tensor | None = None,
    max_rotation_step_rad: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert an absolute waypoint into a bounded IK-relative tracking action.

    The optional limits are distances per environment control step, not gains.
    They turn each semantic waypoint transition into a velocity-limited
    Cartesian segment instead of sending the full waypoint error in one step.
    """

    from isaaclab.utils.math import compute_pose_error

    position_error, rotation_error = compute_pose_error(
        current_pose_wxyz[:, :3],
        current_pose_wxyz[:, 3:7],
        target_pose_wxyz[:, :3],
        target_pose_wxyz[:, 3:7],
        rot_error_type="axis_angle",
    )

    def limit_norm(vector: torch.Tensor, maximum) -> torch.Tensor:
        if maximum is None:
            return vector
        bound = torch.as_tensor(maximum, device=vector.device, dtype=vector.dtype)
        while bound.ndim < vector.ndim:
            bound = bound.unsqueeze(-1)
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        scale = torch.clamp(bound / torch.clamp(norm, min=1.0e-8), max=1.0)
        return vector * scale

    position_error = limit_norm(position_error, max_translation_step_m)
    rotation_error = limit_norm(rotation_error, max_rotation_step_rad)
    pose_command = torch.cat((position_error, rotation_error), dim=-1)
    action = torch.cat((pose_command / controller_scale, gripper.reshape(-1, 1)), dim=-1)
    return canonicalize_relative_action(action)


def advance_pose_toward_waypoint(
    command_pose_wxyz: torch.Tensor,
    waypoint_pose_wxyz: torch.Tensor,
    max_translation_step_m: float | torch.Tensor,
    max_rotation_step_rad: float | torch.Tensor,
) -> torch.Tensor:
    """Advance a persistent Cartesian command by one bounded trajectory step."""

    position_delta = waypoint_pose_wxyz[:, :3] - command_pose_wxyz[:, :3]
    position_norm = torch.linalg.vector_norm(position_delta, dim=-1, keepdim=True)
    translation_bound = torch.as_tensor(
        max_translation_step_m,
        device=command_pose_wxyz.device,
        dtype=command_pose_wxyz.dtype,
    ).reshape(-1, 1)
    if translation_bound.shape[0] == 1:
        translation_bound = translation_bound.expand(command_pose_wxyz.shape[0], -1)
    position_fraction = torch.clamp(
        translation_bound / torch.clamp(position_norm, min=1.0e-8), max=1.0
    )
    next_position = command_pose_wxyz[:, :3] + position_fraction * position_delta

    current_quaternion = command_pose_wxyz[:, 3:7]
    target_quaternion = waypoint_pose_wxyz[:, 3:7]
    # Use the shortest quaternion hemisphere before normalized interpolation.
    target_quaternion = torch.where(
        (current_quaternion * target_quaternion).sum(dim=-1, keepdim=True) < 0.0,
        -target_quaternion,
        target_quaternion,
    )
    dot = (current_quaternion * target_quaternion).sum(dim=-1).abs().clamp(0.0, 1.0)
    angular_distance = 2.0 * torch.acos(dot)
    rotation_bound = torch.as_tensor(
        max_rotation_step_rad,
        device=command_pose_wxyz.device,
        dtype=command_pose_wxyz.dtype,
    ).reshape(-1)
    if rotation_bound.shape[0] == 1:
        rotation_bound = rotation_bound.expand(command_pose_wxyz.shape[0])
    rotation_fraction = torch.clamp(
        rotation_bound / torch.clamp(angular_distance, min=1.0e-8), max=1.0
    ).unsqueeze(-1)
    next_quaternion = torch.nn.functional.normalize(
        (1.0 - rotation_fraction) * current_quaternion
        + rotation_fraction * target_quaternion,
        dim=-1,
    )
    return torch.cat((next_position, next_quaternion), dim=-1)


def task_space_state(env) -> dict[str, torch.Tensor]:
    """Return EE, object, and target poses in the Franka base frame."""

    from isaaclab.utils.math import subtract_frame_transforms

    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    object_asset = env.scene["object"]
    target = env.scene["target_region"]
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    ee_pos, ee_quat = subtract_frame_transforms(
        root_pos,
        root_quat,
        ee_frame.data.target_pos_w[..., 0, :],
        ee_frame.data.target_quat_w[..., 0, :],
    )
    object_pos, object_quat = subtract_frame_transforms(
        root_pos, root_quat, object_asset.data.root_pos_w, object_asset.data.root_quat_w
    )
    target_pos, target_quat = subtract_frame_transforms(
        root_pos, root_quat, target.data.root_pos_w, target.data.root_quat_w
    )
    return {
        "ee_pose": torch.cat((ee_pos, ee_quat), dim=-1),
        "object_pose": torch.cat((object_pos, object_quat), dim=-1),
        "target_pose": torch.cat((target_pos, target_quat), dim=-1),
    }


def opening_task_state(env, joint_name: str) -> dict[str, torch.Tensor]:
    """Return EE/handle poses and one opening joint in the Franka base frame."""

    from isaaclab.utils.math import subtract_frame_transforms

    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    handle_frame = env.scene["cabinet_frame"]
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    ee_pos, ee_quat = subtract_frame_transforms(
        root_pos,
        root_quat,
        ee_frame.data.target_pos_w[..., 0, :],
        ee_frame.data.target_quat_w[..., 0, :],
    )
    handle_pos, handle_quat = subtract_frame_transforms(
        root_pos,
        root_quat,
        handle_frame.data.target_pos_w[..., 0, :],
        handle_frame.data.target_quat_w[..., 0, :],
    )
    cabinet = env.scene["cabinet"]
    joint_ids, names = cabinet.find_joints(joint_name)
    if names != [joint_name]:
        raise RuntimeError(f"expected one {joint_name!r}, found {names}")
    finger_ids, finger_names = robot.find_joints("panda_finger_joint.*")
    if len(finger_names) != 2:
        raise RuntimeError(f"expected two Franka fingers, found {finger_names}")
    return {
        "ee_pose": torch.cat((ee_pos, ee_quat), dim=-1),
        "handle_pose": torch.cat((handle_pos, handle_quat), dim=-1),
        "joint_position": cabinet.data.joint_pos[:, joint_ids].squeeze(-1),
        "joint_velocity": cabinet.data.joint_vel[:, joint_ids].squeeze(-1),
        "finger_positions": robot.data.joint_pos[:, finger_ids],
    }


def object_to_drawer_task_state(env) -> dict[str, torch.Tensor]:
    """Return all physical state used by the long-horizon collection expert."""

    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

    state = opening_task_state(env, "UpperDrawerJoint")
    robot = env.scene["robot"]
    obj = env.scene["object"]
    target_frame = env.scene["drawer_interior_frame"]
    root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w
    object_pos, object_quat = subtract_frame_transforms(
        root_pos, root_quat, obj.data.root_pos_w, obj.data.root_quat_w
    )
    target_pos, target_quat = subtract_frame_transforms(
        root_pos,
        root_quat,
        target_frame.data.target_pos_w[..., 0, :],
        target_frame.data.target_quat_w[..., 0, :],
    )
    finger_ids, names = robot.find_joints("panda_finger_joint.*")
    if len(names) != 2:
        raise RuntimeError(f"expected two Franka fingers, found {names}")
    state.update(
        {
            "object_pose": torch.cat((object_pos, object_quat), dim=-1),
            "object_linear_velocity": quat_apply_inverse(
                root_quat, obj.data.root_lin_vel_w
            ),
            "object_angular_velocity": quat_apply_inverse(
                root_quat, obj.data.root_ang_vel_w
            ),
            "target_pose": torch.cat((target_pos, target_quat), dim=-1),
            "finger_positions": robot.data.joint_pos[:, finger_ids],
        }
    )
    return state


def nut_pouring_task_state(env) -> dict[str, torch.Tensor]:
    """Return EE, vessels, nut, velocities, and fingers in the robot base frame."""

    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    cup = env.scene["object"]
    nut = env.scene["factory_nut"]
    bowl = env.scene["sorting_bowl"]
    root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w

    def pose(asset):
        position, quaternion = subtract_frame_transforms(
            root_pos, root_quat, asset.data.root_pos_w, asset.data.root_quat_w
        )
        return torch.cat((position, quaternion), dim=-1)

    ee_position, ee_quaternion = subtract_frame_transforms(
        root_pos,
        root_quat,
        ee_frame.data.target_pos_w[..., 0, :],
        ee_frame.data.target_quat_w[..., 0, :],
    )
    finger_ids, names = robot.find_joints("panda_finger_joint.*")
    if len(names) != 2:
        raise RuntimeError(f"expected two Franka fingers, found {names}")
    return {
        "ee_pose": torch.cat((ee_position, ee_quaternion), dim=-1),
        "cup_pose": pose(cup),
        "cup_linear_velocity": quat_apply_inverse(
            root_quat, cup.data.root_lin_vel_w
        ),
        "cup_angular_velocity": quat_apply_inverse(
            root_quat, cup.data.root_ang_vel_w
        ),
        "nut_pose": pose(nut),
        "nut_linear_velocity": quat_apply_inverse(
            root_quat, nut.data.root_lin_vel_w
        ),
        "nut_angular_velocity": quat_apply_inverse(
            root_quat, nut.data.root_ang_vel_w
        ),
        "bowl_pose": pose(bowl),
        "finger_positions": robot.data.joint_pos[:, finger_ids],
    }
