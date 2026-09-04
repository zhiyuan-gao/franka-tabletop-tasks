#!/usr/bin/env python3
"""Evaluate frozen BC and trained residual TD3+BC for one Track B task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_NAMES = (
    "push_object_to_region",
    "push_cube_to_region",
    "drawer_opening",
    "cabinet_opening",
    "object_to_drawer",
    "nut_pouring",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=TASK_NAMES, required=True)
parser.add_argument("--residual-checkpoint", type=Path, required=True)
parser.add_argument("--bc-checkpoint", type=Path, default=None)
parser.add_argument("--policies", nargs="+", choices=("bc", "residual"), default=("bc", "residual"))
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--video-dir",
    type=Path,
    default=None,
    help="Optional directory for deterministic MP4 rollouts.",
)
parser.add_argument("--video-episodes", type=int, default=1)
parser.add_argument("--video-env", type=int, default=0)
parser.add_argument("--video-fps", type=float, default=20.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output is None:
    args.output = Path("outputs") / f"{args.task}_residual_td3bc_eval.json"
if args.video_dir is None:
    args.video_dir = args.output.parent / "videos"
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def encode_video(frames, output, fps):
    import subprocess

    import numpy as np

    if not frames:
        raise ValueError("cannot encode an empty video")
    array = np.stack(frames).astype(np.uint8, copy=False)
    height, width = array.shape[1:3]
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
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
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, input=array.tobytes(), check=True)
    if not output.is_file():
        raise RuntimeError(f"ffmpeg did not create {output}")


def evaluate(env, bc_model, residual_actor, policy_name, episodes, seed, device):
    import torch

    from task_pipeline_common import (
        compose_action,
        environment_success,
        policy_observation,
    )

    observations, _ = env.reset(seed=seed)
    episode_returns = torch.zeros(args.num_envs, device=device)
    episode_lengths = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    rows = []
    video_frames = []
    video_files = []
    recorded_video_episodes = 0
    while len(rows) < episodes or recorded_video_episodes < args.video_episodes:
        rgb, proprio = policy_observation(observations, device)
        if args.video_dir is not None and recorded_video_episodes < args.video_episodes:
            frame = (
                rgb[args.video_env]
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            video_frames.append(frame)
        with torch.inference_mode():
            base_action, feature = bc_model.deterministic_action_and_feature(rgb, proprio)
            if policy_name == "bc":
                actions = base_action
            else:
                residual = residual_actor(feature, base_action)
                actions = compose_action(base_action, residual)
        observations, rewards, terminated, truncated, _ = env.step(actions)
        success = environment_success(env)
        episode_returns += rewards
        episode_lengths += 1
        ended = terminated | truncated
        for env_id in ended.nonzero(as_tuple=False).flatten().tolist():
            if len(rows) < episodes:
                rows.append(
                    {
                        "episode": len(rows),
                        "env_id": env_id,
                        "success": bool(success[env_id].item()),
                        "terminated": bool(terminated[env_id].item()),
                        "truncated": bool(truncated[env_id].item()),
                        "return": float(episode_returns[env_id].item()),
                        "length": int(episode_lengths[env_id].item()),
                    }
                )
            episode_returns[env_id] = 0.0
            episode_lengths[env_id] = 0
            if (
                args.video_dir is not None
                and env_id == args.video_env
                and recorded_video_episodes < args.video_episodes
            ):
                outcome = "success" if bool(success[env_id].item()) else "failure"
                video_path = (
                    args.video_dir
                    / f"{args.task}_{policy_name}_episode_{recorded_video_episodes:03d}_{outcome}.mp4"
                )
                encode_video(video_frames, video_path, args.video_fps)
                video_files.append(str(video_path.resolve()))
                print(f"saved video: {video_path}", flush=True)
                video_frames = []
                recorded_video_episodes += 1
    successes = sum(int(row["success"]) for row in rows)
    return {
        "policy": policy_name,
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "mean_return": sum(row["return"] for row in rows) / len(rows),
        "mean_length": sum(row["length"] for row in rows) / len(rows),
        "videos": video_files,
        "results": rows,
    }


def main() -> dict:
    import gymnasium as gym
    import torch

    from task_pipeline_common import (
        load_bc_checkpoint,
        load_residual_actor,
    )
    from track_b_envs.rl import configure_normalized_action_space, load_registered_env_cfg

    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("episodes and num-envs must be positive")
    if args.video_episodes < 0 or args.video_fps <= 0.0:
        raise ValueError("video-episodes must be non-negative and video-fps must be positive")
    if not 0 <= args.video_env < args.num_envs:
        raise ValueError("video-env must be in [0, num-envs)")
    if not args.residual_checkpoint.is_file():
        raise FileNotFoundError(args.residual_checkpoint)
    device = torch.device(args.device)
    residual_actor, residual_payload = load_residual_actor(
        args.residual_checkpoint, device, expected_task=args.task
    )
    bc_checkpoint = args.bc_checkpoint
    if bc_checkpoint is None:
        bc_checkpoint = Path(residual_payload["bc_checkpoint"])
    if not bc_checkpoint.is_file():
        raise FileNotFoundError(
            f"BC checkpoint not found: {bc_checkpoint}; pass --bc-checkpoint explicitly"
        )
    bc_model, bc_payload = load_bc_checkpoint(
        bc_checkpoint, device, expected_task=args.task
    )
    if residual_actor.feature_dim != bc_model.feature_dim:
        raise ValueError(
            f"feature mismatch: residual={residual_actor.feature_dim}, BC={bc_model.feature_dim}"
        )

    task_id, cfg = load_registered_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    env = gym.make(task_id, cfg=cfg)
    configure_normalized_action_space(env)
    try:
        evaluations = [
            evaluate(
                env,
                bc_model,
                residual_actor,
                policy_name,
                args.episodes,
                args.seed,
                device,
            )
            for policy_name in args.policies
        ]
    finally:
        env.close()
    result = {
        "task": args.task,
        "seed": args.seed,
        "bc_checkpoint": str(bc_checkpoint.resolve()),
        "bc_epoch": int(bc_payload["epoch"]),
        "residual_checkpoint": str(args.residual_checkpoint.resolve()),
        "evaluations": evaluations,
    }
    print(json.dumps(result, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
