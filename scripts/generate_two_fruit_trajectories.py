"""Generate controlled orange-first and banana-first placement trajectories.

The two runs start from byte-identical robot and object state arrays.  They use
the same target plate and differ only in which fruit is picked first.  The
objects are kinematically attached during transport; this script is intended
for layout and trajectory visualization, not as a contact-rich physics demo.
Either existing plate can be selected. A grasp-failure variant places the
first fruit and then misses the second without moving it. Use --videos-only
for an MP4-only deliverable and --wide-sim-view to include both whole plates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("outputs/two_fruit_placement"),
)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--physics-steps-per-frame", type=int, default=4)
parser.add_argument("--target-plate", choices=("purple", "green"), default="purple")
parser.add_argument("--outcome", choices=("success", "grasp_failure"), default="success")
parser.add_argument("--videos-only", action="store_true", help="Write only named MP4s; validation goes to stdout.")
parser.add_argument("--wide-sim-view", action="store_true", help="1920x1080 simulation-only view including both complete plates.")
parser.add_argument(
    "--initial-only",
    action="store_true",
    help="Render only the shared initial state (useful for layout calibration).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


@dataclass(frozen=True)
class FruitSpec:
    name: str
    initial_position: tuple[float, float, float]
    final_position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    hand_clearance_m: float
    closed_finger_position_m: float


@dataclass(frozen=True)
class Segment:
    phase: str
    target_hand_position: tuple[float, float, float]
    frames: int
    finger_position_m: float
    carried_fruit: str | None = None
    placed_fruit: str | None = None


def _smoothstep(alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _interpolate(start, end, alpha: float):
    blend = _smoothstep(alpha)
    return tuple(a + blend * (b - a) for a, b in zip(start, end, strict=True))


def _write_object_pose(torch, device, asset, position, orientation) -> None:
    """Move a visual-only object through Fabric so the camera sees the pose."""

    asset.set_world_poses(
        positions=torch.tensor([position], dtype=torch.float32, device=device),
        orientations=torch.tensor([orientation], dtype=torch.float32, device=device),
        usd=False,
    )


def _read_object_pose(asset):
    position, orientation = asset.get_world_poses(usd=False)
    return position, orientation


def _trajectory_segments(order, fruit_specs, home_position, *, grasp_failure=False):
    open_finger = 0.04
    segments = [Segment("initial_hold", home_position, 6, open_finger)]
    for fruit_index, fruit_name in enumerate(order):
        spec = fruit_specs[fruit_name]
        source = spec.initial_position
        target = spec.final_position
        grasp = (source[0], source[1], source[2] + spec.hand_clearance_m)
        place = (target[0], target[1], target[2] + spec.hand_clearance_m)
        if grasp_failure and fruit_index == 1:
            miss = (source[0] - 0.090, source[1], grasp[2])
            above_miss = (miss[0], miss[1], 0.32)
            segments.extend((
                Segment(f"approach_miss_{fruit_name}", above_miss, 11, open_finger),
                Segment(f"descend_miss_{fruit_name}", miss, 8, open_finger),
                Segment(f"grasp_miss_{fruit_name}", miss, 8, 0.0),
                Segment(f"empty_lift_{fruit_name}", above_miss, 8, 0.0),
                Segment(f"failed_grasp_hold_{fruit_name}", above_miss, 8, 0.0),
            ))
            break
        segments.extend(
            (
                Segment(f"approach_{fruit_name}", (source[0], source[1], 0.34), 11, open_finger),
                Segment(f"descend_{fruit_name}", grasp, 8, open_finger),
                Segment(
                    f"grasp_{fruit_name}",
                    grasp,
                    4,
                    spec.closed_finger_position_m,
                    carried_fruit=fruit_name,
                ),
                Segment(
                    f"lift_{fruit_name}",
                    (source[0], source[1], 0.32),
                    8,
                    spec.closed_finger_position_m,
                    carried_fruit=fruit_name,
                ),
                Segment(
                    f"transfer_{fruit_name}",
                    (target[0], target[1], 0.32),
                    12,
                    spec.closed_finger_position_m,
                    carried_fruit=fruit_name,
                ),
                Segment(
                    f"place_{fruit_name}",
                    place,
                    8,
                    spec.closed_finger_position_m,
                    carried_fruit=fruit_name,
                ),
                Segment(
                    f"release_{fruit_name}",
                    place,
                    4,
                    open_finger,
                    placed_fruit=fruit_name,
                ),
                Segment(f"retreat_{fruit_name}", (target[0], target[1], 0.32), 7, open_finger),
            )
        )
    segments.extend(
        (
            Segment("return_home", home_position, 40, open_finger),
            Segment("final_hold", home_position, 10, open_finger),
        )
    )
    return segments


def _mesh_points_triangles(path, scale, quaternion=(1.0, 0.0, 0.0, 0.0)):
    """Read composed USD mesh geometry, including authored local transforms."""
    import numpy as np
    from pxr import Usd, UsdGeom
    from scipy.spatial.transform import Rotation

    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    point_sets, faces = [], []
    offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        points = np.c_[points, np.ones(len(points))] @ matrix
        point_sets.append(points[:, :3] * scale)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        start = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            polygon = indices[start:start + count] + offset
            faces.extend((polygon[0], polygon[j], polygon[j + 1]) for j in range(1, count - 1))
            start += count
        offset += len(points)
    points = np.concatenate(point_sets)
    w, x, y, z = quaternion
    points = Rotation.from_quat([x, y, z, w]).apply(points)
    return points, np.asarray(faces, dtype=int)


def _supported_root_height(plate_mesh, plate_position, fruit_mesh, fruit_xy):
    """Sample mesh vertices, edge midpoints and centroids against plate triangles."""
    import numpy as np

    plate_points, plate_faces = plate_mesh
    plate_triangles = (plate_points + np.asarray(plate_position))[plate_faces]
    fruit_points, fruit_faces = fruit_mesh
    ft = fruit_points[fruit_faces]
    samples = np.concatenate((fruit_points, ft.mean(axis=1),
        (ft[:, 0] + ft[:, 1]) / 2, (ft[:, 1] + ft[:, 2]) / 2, (ft[:, 2] + ft[:, 0]) / 2))
    samples[:, :2] += np.asarray(fruit_xy)
    a, b, c = plate_triangles[:, 0], plate_triangles[:, 1], plate_triangles[:, 2]
    v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    denominator = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    valid = np.abs(denominator) > 1e-12
    a, b, c, v0, v1, denominator = (v[valid] for v in (a, b, c, v0, v1, denominator))
    required = -float(fruit_points[:, 2].min()) + 0.001
    overlaps = 0
    # Exact broad-phase rejection in 1 cm XY cells avoids comparing every
    # sample against every plate triangle. All overlapping triangles remain.
    minimum_xy = np.minimum(np.minimum(a[:, :2], b[:, :2]), c[:, :2])
    maximum_xy = np.maximum(np.maximum(a[:, :2], b[:, :2]), c[:, :2])
    cells, groups = np.unique(np.floor(samples[:, :2] / 0.01).astype(int), axis=0, return_inverse=True)
    for group, cell in enumerate(cells):
        chunk = samples[groups == group]
        lower, upper = cell * 0.01 - 1e-7, (cell + 1) * 0.01 + 1e-7
        candidate = np.all(maximum_xy >= lower, axis=1) & np.all(minimum_xy <= upper, axis=1)
        if not candidate.any():
            continue
        aa, bb, cc, vv0, vv1, dd = (value[candidate] for value in (a, b, c, v0, v1, denominator))
        delta = chunk[:, None, :2] - aa[None, :, :2]
        u = (delta[..., 0] * vv1[:, 1] - delta[..., 1] * vv1[:, 0]) / dd
        v = (vv0[:, 0] * delta[..., 1] - vv0[:, 1] * delta[..., 0]) / dd
        inside = (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1.0000001)
        height = aa[:, 2] + u * (bb[:, 2] - aa[:, 2]) + v * (cc[:, 2] - aa[:, 2])
        overlaps += int(inside.any(axis=1).sum())
        if inside.any():
            required = max(required, float(np.where(inside,
                height - chunk[:, None, 2] + 0.001, -np.inf).max()))
    if not overlaps:
        raise RuntimeError("fruit has no support overlap with target plate")
    return required


def main() -> None:
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils.math import subtract_frame_transforms
    from PIL import Image

    from track_b_envs.envs.common.compositing import RealBackgroundCompositor
    from track_b_envs.envs.two_fruit_placement import (
        BANANA_FINAL_POSITION_M,
        BANANA_INITIAL_POSITION_M,
        ORANGE_FINAL_POSITION_M,
        ORANGE_INITIAL_POSITION_M,
        TARGET_PLATE_POSITION_M,
        build_two_fruit_scene_cfg,
    )
    from track_b_envs.envs.two_fruit_placement.two_fruit_scene import (
        BANANA_STABLE_ROTATION_WXYZ,
        DISTRACTOR_PLATE_POSITION_M,
        DISTRACTOR_PLATE_SCALE,
        REFERENCE_FRAME,
        REFERENCE_PIXEL_CENTERS,
        REFERENCE_TO_BACKGROUND_HOMOGRAPHY,
        REFERENCE_VIDEO,
        TARGET_PLATE_SCALE,
        YCB_BANANA_SCALE,
        YCB_ORANGE_SCALE,
        YCB_ORANGE_USD,
        YCB_PLATE_USD,
    )
    from track_b_envs.envs.common.task_scene import YCB_BANANA_VISUAL_USD

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim = SimulationContext(SimulationCfg(dt=0.01, device=args.device))
    scene_cfg = build_two_fruit_scene_cfg()
    if args.wide_sim_view:
        import isaaclab.sim as sim_utils

        scene_cfg.policy_camera.width = 1920
        scene_cfg.policy_camera.height = 1080
        scene_cfg.policy_camera.spawn.horizontal_aperture *= 1.5
        # Show the extended tabletop in the wider visualization. The robot,
        # fruits and plates retain exactly the video-calibrated world poses.
        scene_cfg.platform.spawn.visible = True
        scene_cfg.platform.spawn.size = (1.4, 1.3, 0.04)
        scene_cfg.platform.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.48, 0.49, 0.47), roughness=0.95, metallic=0.0,
        )
        scene_cfg.plane.spawn.visible = True
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    camera = scene["policy_camera"]
    compositor = None if args.wide_sim_view else RealBackgroundCompositor(scene.device)
    robot_cfg = SceneEntityCfg(
        "robot",
        joint_names=["panda_joint.*"],
        body_names=["panda_hand"],
    )
    robot_cfg.resolve(scene)
    ee_body_id = robot_cfg.body_ids[0]
    ee_jacobian_id = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
    controller = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        ),
        num_envs=1,
        device=scene.device,
    )
    finger_ids, _ = robot.find_joints("panda_finger_joint.*")

    selected_plate_position = TARGET_PLATE_POSITION_M if args.target_plate == "purple" else DISTRACTOR_PLATE_POSITION_M
    selected_plate_scale = TARGET_PLATE_SCALE if args.target_plate == "purple" else DISTRACTOR_PLATE_SCALE
    target_positions = {}
    plate_mesh = _mesh_points_triangles(YCB_PLATE_USD, selected_plate_scale)
    for name, old_position, asset_path, scale, quaternion in (
        ("banana", BANANA_FINAL_POSITION_M, YCB_BANANA_VISUAL_USD, YCB_BANANA_SCALE, BANANA_STABLE_ROTATION_WXYZ),
        ("orange", ORANGE_FINAL_POSITION_M, YCB_ORANGE_USD, YCB_ORANGE_SCALE, (1., 0., 0., 0.)),
    ):
        xy = np.asarray(selected_plate_position[:2]) + (
            np.asarray(old_position[:2]) - np.asarray(TARGET_PLATE_POSITION_M[:2])
        ) * (selected_plate_scale / TARGET_PLATE_SCALE)
        height = _supported_root_height(plate_mesh, selected_plate_position,
            _mesh_points_triangles(asset_path, scale, quaternion), xy)
        target_positions[name] = (float(xy[0]), float(xy[1]), height)
    print(json.dumps({"target_plate": args.target_plate, "supported_final_positions": target_positions,
        "sampled_plate_clearance_m": 0.001, "object_transport": "kinematic visualization"}), flush=True)

    fruit_specs = {
        "banana": FruitSpec(
            "banana",
            BANANA_INITIAL_POSITION_M,
            target_positions["banana"],
            tuple(BANANA_STABLE_ROTATION_WXYZ),
            hand_clearance_m=0.112,
            closed_finger_position_m=0.020,
        ),
        "orange": FruitSpec(
            "orange",
            ORANGE_INITIAL_POSITION_M,
            target_positions["orange"],
            (1.0, 0.0, 0.0, 0.0),
            hand_clearance_m=0.115,
            closed_finger_position_m=0.028,
        ),
    }

    default_joint_position = robot.data.default_joint_pos.clone()
    default_joint_velocity = robot.data.default_joint_vel.clone()
    initial_pose_arrays = {
        "robot_joint_position": default_joint_position[0].cpu().numpy().copy(),
        "banana_pose_wxyz": np.asarray(
            (*BANANA_INITIAL_POSITION_M, *BANANA_STABLE_ROTATION_WXYZ), dtype=np.float32
        ),
        "orange_pose_wxyz": np.asarray(
            (*ORANGE_INITIAL_POSITION_M, 1.0, 0.0, 0.0, 0.0), dtype=np.float32
        ),
    }
    initial_signature = hashlib.sha256(
        b"".join(value.tobytes() for value in initial_pose_arrays.values())
    ).hexdigest()

    def reset_common_state():
        robot.write_joint_state_to_sim(default_joint_position, default_joint_velocity)
        robot.set_joint_position_target(default_joint_position)
        controller.reset()
        _write_object_pose(
            torch,
            scene.device,
            scene["banana"],
            BANANA_INITIAL_POSITION_M,
            BANANA_STABLE_ROTATION_WXYZ,
        )
        _write_object_pose(
            torch,
            scene.device,
            scene["orange"],
            ORANGE_INITIAL_POSITION_M,
            (1.0, 0.0, 0.0, 0.0),
        )
        # Render during reset so RTX/camera history from the previous variant
        # cannot leave a ghost of a fruit in the shared initial frames.
        for _ in range(60):
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim.get_physics_dt())

    def current_hand_pose_base():
        hand_pose_w = robot.data.body_pose_w[:, ee_body_id]
        root_pose_w = robot.data.root_pose_w
        return subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            hand_pose_w[:, :3],
            hand_pose_w[:, 3:7],
        )

    def render_frame():
        if args.wide_sim_view:
            return camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        composite = compositor.compose(
            camera.data.output["rgb"],
            camera.data.output["semantic_segmentation"],
            camera.data.intrinsic_matrices,
        )[0]
        return composite.cpu().numpy().astype(np.uint8)

    if args.initial_only:
        reset_common_state()
        image_path = args.output_dir / "common_initial_state.png"
        Image.fromarray(render_frame()).save(image_path)
        print(f"saved: {image_path.resolve()}", flush=True)
        del scene
        sim.clear_all_callbacks()
        sim.clear_instance()
        return

    metadata_path = args.output_dir / "metadata.json"

    def write_metadata(reports):
        metadata = {
            "scene_contract": "same initial robot, banana, orange, and plate poses; order only varies",
            "object_transport": "kinematic attachment for visualization",
            "selected_target_plate": args.target_plate,
            "requested_outcome": args.outcome,
            "layout_reference": {
                "canonical_video": REFERENCE_VIDEO,
                "canonical_frame": str(REFERENCE_FRAME),
                "canonical_episode": 41,
                "camera_view": "right",
                "registration": "DICT_5X5_50 board homography to empty_background.png",
                "registration_common_marker_ids": [1, 3, 5, 7, 10, 12, 13, 14, 15],
                "registration_corner_median_error_px": 0.8398119211,
                "reference_to_background_homography": REFERENCE_TO_BACKGROUND_HOMOGRAPHY,
                "registered_object_centers_px": REFERENCE_PIXEL_CENTERS,
            },
            "initial_state": {
                **{key: value.tolist() for key, value in initial_pose_arrays.items()},
                "target_plate_position_m": list(TARGET_PLATE_POSITION_M),
                "distractor_plate_position_m": list(DISTRACTOR_PLATE_POSITION_M),
                "target_plate_scale": TARGET_PLATE_SCALE,
                "distractor_plate_scale": DISTRACTOR_PLATE_SCALE,
                "banana_scale": YCB_BANANA_SCALE,
                "orange_scale": YCB_ORANGE_SCALE,
            },
            "final_state": {
                "description": "both-fruit success goal; a failure leaves the second fruit at its initial pose",
                "banana_position_m": list(target_positions["banana"]),
                "orange_position_m": list(target_positions["orange"]),
                "fruit_center_separation_xy_m": float(np.linalg.norm(
                    np.asarray(target_positions["banana"][:2]) - np.asarray(target_positions["orange"][:2]))),
                "banana_plate_clearance_m": 0.001,
            },
            "trajectories": reports,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    reports = []
    order_variants = (
        ("orange_then_banana", ("orange", "banana")),
        ("banana_then_orange", ("banana", "orange")),
    )
    for variant_name, order in order_variants:
        reset_common_state()
        hand_position, hand_orientation = current_hand_pose_base()
        home_position = tuple(float(value) for value in hand_position[0].cpu())
        fixed_orientation = hand_orientation.clone()
        previous_target = home_position
        segments = _trajectory_segments(order, fruit_specs, home_position,
            grasp_failure=args.outcome == "grasp_failure")
        placed = set()

        video_name = (f"two_fruit_{args.target_plate}_{variant_name}_{args.outcome}"
            if args.videos_only else variant_name)
        video_path = args.output_dir / f"{video_name}.mp4"
        if video_path.exists():
            raise FileExistsError(video_path)
        trajectory_path = args.output_dir / f"{variant_name}.npz"
        writer = imageio.get_writer(
            video_path,
            fps=args.fps,
            codec="libx264",
            quality=7,
            macro_block_size=None,
            ffmpeg_log_level="error",
        )
        recorded = {
            "joint_position": [],
            "banana_pose_wxyz": [],
            "orange_pose_wxyz": [],
            "phase": [],
        }
        first_frame = None
        try:
            for segment in segments:
                if segment.placed_fruit is not None:
                    placed.add(segment.placed_fruit)
                    spec = fruit_specs[segment.placed_fruit]
                    _write_object_pose(
                        torch,
                        scene.device,
                        scene[segment.placed_fruit],
                        spec.final_position,
                        spec.orientation_wxyz,
                    )
                for frame_index in range(segment.frames):
                    alpha = (frame_index + 1) / segment.frames
                    command_position = _interpolate(
                        previous_target,
                        segment.target_hand_position,
                        alpha,
                    )
                    if segment.carried_fruit is not None:
                        spec = fruit_specs[segment.carried_fruit]
                        carried_position = (
                            command_position[0],
                            command_position[1],
                            command_position[2] - spec.hand_clearance_m,
                        )
                        _write_object_pose(
                            torch,
                            scene.device,
                            scene[segment.carried_fruit],
                            carried_position,
                            spec.orientation_wxyz,
                        )
                    for physics_step in range(args.physics_steps_per_frame):
                        if segment.phase in ("return_home", "final_hold"):
                            # A Cartesian-only return leaves the redundant
                            # Franka elbow in an order-dependent posture. Use
                            # the repository's default joint target so both
                            # previews finish in the same canonical home pose.
                            desired_joint_position = default_joint_position[
                                :, robot_cfg.joint_ids
                            ]
                        else:
                            ik_command = torch.cat(
                                (
                                    torch.tensor(
                                        [command_position],
                                        dtype=torch.float32,
                                        device=scene.device,
                                    ),
                                    fixed_orientation,
                                ),
                                dim=-1,
                            )
                            controller.set_command(ik_command)
                            jacobian = robot.root_physx_view.get_jacobians()[
                                :, ee_jacobian_id, :, robot_cfg.joint_ids
                            ]
                            current_position, current_orientation = current_hand_pose_base()
                            arm_joint_position = robot.data.joint_pos[
                                :, robot_cfg.joint_ids
                            ]
                            desired_joint_position = controller.compute(
                                current_position,
                                current_orientation,
                                jacobian,
                                arm_joint_position,
                            )
                        robot.set_joint_position_target(
                            desired_joint_position,
                            joint_ids=robot_cfg.joint_ids,
                        )
                        finger_target = torch.full(
                            (1, len(finger_ids)),
                            segment.finger_position_m,
                            device=scene.device,
                        )
                        robot.set_joint_position_target(finger_target, joint_ids=finger_ids)
                        scene.write_data_to_sim()
                        sim.step(render=physics_step == args.physics_steps_per_frame - 1)
                        scene.update(sim.get_physics_dt())

                    frame = render_frame()
                    if first_frame is None:
                        first_frame = frame.copy()
                    writer.append_data(frame)
                    recorded["joint_position"].append(
                        robot.data.joint_pos[0].detach().cpu().numpy().copy()
                    )
                    for fruit_name in ("banana", "orange"):
                        position, orientation = _read_object_pose(scene[fruit_name])
                        recorded[f"{fruit_name}_pose_wxyz"].append(
                            torch.cat((position[0], orientation[0]))
                            .detach()
                            .cpu()
                            .numpy()
                            .copy()
                        )
                    recorded["phase"].append(segment.phase)
                previous_target = segment.target_hand_position
        finally:
            writer.close()

        expected_placed = set(order) if args.outcome == "success" else {order[0]}
        if placed != expected_placed:
            raise RuntimeError(f"unexpected placed fruit set: {placed}")
        for fruit_name, spec in fruit_specs.items():
            expected = spec.final_position if fruit_name in placed else spec.initial_position
            actual = np.asarray(recorded[f"{fruit_name}_pose_wxyz"])[-1, :3]
            if not np.allclose(actual, expected, atol=1e-5):
                raise RuntimeError(f"unexpected final {fruit_name} pose: {actual}")
            if fruit_name not in placed:
                if not np.allclose(np.asarray(recorded[f"{fruit_name}_pose_wxyz"])[:, :3], expected, atol=1e-5):
                    raise RuntimeError("failed-grasp fruit moved")
        home_error = float(np.max(np.abs(recorded["joint_position"][-1] - default_joint_position[0].cpu().numpy())))
        if home_error > 0.02:
            raise RuntimeError(f"robot did not return home: {home_error}")
        print(json.dumps({"video": video_path.name, "target_plate": args.target_plate,
            "order": order, "outcome": args.outcome, "placed": sorted(placed),
            "frames": len(recorded["phase"]), "home_joint_error_rad": home_error,
            "common_initial_signature": initial_signature, "validated_visual_outcome": True,
            "physical_grasp_validation": False}), flush=True)
        if args.videos_only:
            print(f"saved: {video_path.resolve()}", flush=True)
            continue

        np.savez_compressed(
            trajectory_path,
            timestamp_s=np.arange(len(recorded["phase"]), dtype=np.float32) / args.fps,
            joint_position=np.asarray(recorded["joint_position"], dtype=np.float32),
            banana_pose_wxyz=np.asarray(recorded["banana_pose_wxyz"], dtype=np.float32),
            orange_pose_wxyz=np.asarray(recorded["orange_pose_wxyz"], dtype=np.float32),
            phase=np.asarray(recorded["phase"]),
            order=np.asarray(order),
            common_initial_signature=np.asarray(initial_signature),
        )
        if variant_name == "orange_then_banana":
            Image.fromarray(first_frame).save(args.output_dir / "common_initial_state.png")
        reports.append(
            {
                "variant": variant_name,
                "order": list(order),
                "frames": len(recorded["phase"]),
                "fps": args.fps,
                "duration_s": len(recorded["phase"]) / args.fps,
                "video": video_path.name,
                "trajectory": trajectory_path.name,
                "common_initial_signature": initial_signature,
            }
        )
        print(f"saved: {video_path.resolve()}", flush=True)
        print(f"saved: {trajectory_path.resolve()}", flush=True)
        # Persist after each variant so completed outputs retain their exact
        # provenance even if the renderer is interrupted during later cleanup.
        write_metadata(reports)
        print(f"saved: {metadata_path.resolve()}", flush=True)

    del scene
    sim.clear_all_callbacks()
    sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
