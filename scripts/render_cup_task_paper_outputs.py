"""Replay one cup episode from a paper-friendly, simulator-only camera."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task", required=True, choices=("cup_to_drawer", "cup_to_cabinet_top")
)
parser.add_argument("--episode", type=Path, required=True)
parser.add_argument("--output-video", type=Path, required=True)
parser.add_argument("--output-grasp-frame", type=Path, required=True)
parser.add_argument("--grasp-frame-index", type=int, required=True)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--focal-length-mm", type=float, default=None)
parser.add_argument("--camera-eye", type=float, nargs=3, default=None)
parser.add_argument("--camera-target", type=float, nargs=3, default=None)
parser.add_argument("--warmup-steps", type=int, default=10)
parser.add_argument(
    "--max-frame",
    type=int,
    default=None,
    help="Render only through this zero-based frame; useful for camera previews.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


DEFAULT_VIEWS = {
    # A low front-right view exposes both fingers during the vertical grasp.
    "cup_to_drawer": {
        "eye": (1.12, 0.02, 0.48),
        "target": (0.54, -0.30, 0.27),
        "focal_length_mm": 24.0,
    },
    # Looking almost along the horizontal approach direction exposes the
    # finger gap and the cup below the extended upper drawer.
    "cup_to_cabinet_top": {
        "eye": (0.60, 0.62, 0.58),
        "target": (0.52, -0.15, 0.17),
        "focal_length_mm": 20.0,
    },
}


def _ffmpeg_command(output: Path, width: int, height: int, fps: float) -> list[str]:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise FileNotFoundError("ffmpeg is required to encode the paper video")
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.8f}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def main() -> None:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import TiledCameraCfg
    from PIL import Image

    from track_b_envs.envs.cup_cabinet import (
        CupToCabinetTopEnvCfg,
        CupToDrawerEnvCfg,
    )

    if args.width <= 0 or args.height <= 0:
        raise ValueError("camera width and height must be positive")
    if args.grasp_frame_index < 0:
        raise ValueError("grasp frame index must be non-negative")

    with np.load(args.episode, allow_pickle=False) as episode:
        actions = np.asarray(episode["action"], dtype=np.float32)
        timestamps = np.asarray(episode["timestamp_s"], dtype=np.float64)
        recorded_success = np.asarray(episode["success"], dtype=np.bool_)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"expected (T, 7) actions, got {actions.shape}")
    if len(actions) != len(timestamps) or len(actions) != len(recorded_success):
        raise ValueError("episode arrays are not synchronized")
    if not len(actions) or not bool(recorded_success[-1]):
        raise ValueError("source episode does not end in success")
    if args.grasp_frame_index >= len(actions):
        raise ValueError("grasp frame index lies outside the episode")
    dt = float(np.median(np.diff(timestamps)))
    fps = 1.0 / dt
    final_frame = len(actions) - 1
    if args.max_frame is not None:
        final_frame = min(final_frame, args.max_frame)
    if args.grasp_frame_index > final_frame:
        raise ValueError("grasp frame is after --max-frame")

    cfg_cls = (
        CupToDrawerEnvCfg if args.task == "cup_to_drawer" else CupToCabinetTopEnvCfg
    )
    cfg = cfg_cls()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = 20260827
    cfg.episode_length_s = max(
        cfg.episode_length_s, (final_frame + 3) * dt
    )

    # The policy observation uses a calibrated real-background compositor.
    # Remove that camera and observation entirely; the paper camera below is
    # an independent raw simulator RGB sensor.
    cfg.observations.policy.rgb = None
    cfg.scene.policy_camera = None
    view = DEFAULT_VIEWS[args.task]
    focal_length_mm = (
        args.focal_length_mm
        if args.focal_length_mm is not None
        else view["focal_length_mm"]
    )
    cfg.scene.paper_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/PaperCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length_mm,
            horizontal_aperture=36.0,
            clipping_range=(0.01, 20.0),
            focus_distance=1.0,
            f_stop=0.0,
        ),
        width=args.width,
        height=args.height,
        data_types=["rgb"],
        update_latest_camera_pose=True,
    )

    # Replace the transparent real-table placeholder with a matte simulated
    # tabletop.  A dark neutral surface keeps the white Franka fingers legible
    # in crops and avoids specular highlights.
    cfg.scene.platform.spawn.visible = True
    cfg.scene.platform.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.12, 0.14, 0.16), roughness=0.92, metallic=0.0
    )
    cfg.scene.plane.spawn.visible = False
    cfg.scene.real_background_key_light.spawn.intensity = 1150.0
    cfg.scene.light.spawn.intensity = 350.0

    eye_values = args.camera_eye if args.camera_eye is not None else view["eye"]
    target_values = (
        args.camera_target if args.camera_target is not None else view["target"]
    )

    env = ManagerBasedRLEnv(cfg=cfg)
    process = None
    replay_success = False
    try:
        camera = env.scene["paper_camera"]
        eyes = torch.tensor([eye_values], device=env.device, dtype=torch.float32)
        targets = torch.tensor([target_values], device=env.device, dtype=torch.float32)

        if args.warmup_steps > 0:
            env.reset(seed=cfg.seed)
            camera.set_world_poses_from_view(eyes, targets)
            zero_action = torch.zeros(env.action_space.shape, device=env.device)
            for _ in range(args.warmup_steps):
                env.step(zero_action)

        env.reset(seed=cfg.seed)
        camera.set_world_poses_from_view(eyes, targets)
        # Force one render at the requested pose before the first captured
        # pre-action observation.
        env.sim.render()
        camera.update(0.0, force_recompute=True)

        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        args.output_grasp_frame.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            _ffmpeg_command(args.output_video, args.width, args.height, fps),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None

        for frame_index in range(final_frame + 1):
            frame = (
                camera.data.output["rgb"][0, ..., :3]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8, copy=False)
            )
            process.stdin.write(frame.tobytes())
            if frame_index == args.grasp_frame_index:
                Image.fromarray(frame).save(args.output_grasp_frame)

            action = torch.as_tensor(
                actions[frame_index : frame_index + 1], device=env.device
            )
            _, _, terminated, _, _ = env.step(action)
            if bool(terminated[0]):
                replay_success = bool(
                    env.termination_manager.get_term("success")[0].detach().cpu()
                )

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with status {return_code}: {stderr}")
        process = None

        report = {
            "task": args.task,
            "source_episode": str(args.episode.resolve()),
            "video": str(args.output_video.resolve()),
            "grasp_frame": str(args.output_grasp_frame.resolve()),
            "grasp_frame_index": args.grasp_frame_index,
            "grasp_timestamp_s": float(timestamps[args.grasp_frame_index]),
            "rendered_frames": final_frame + 1,
            "fps": fps,
            "resolution": [args.width, args.height],
            "camera_eye_m": list(eye_values),
            "camera_target_m": list(target_values),
            "focal_length_mm": focal_length_mm,
            "simulator_only": True,
            "source_replay_success_observed": replay_success,
        }
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if process is not None:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            process.terminate()
            process.wait()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
