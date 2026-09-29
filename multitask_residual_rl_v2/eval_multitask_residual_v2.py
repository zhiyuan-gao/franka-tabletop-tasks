#!/usr/bin/env python3
"""Evaluate one shared multitask residual actor on a selected task.

Reports BC-only and BC+residual on the same seeds so the delta is honest:

    python scripts/eval_task_residual_v2.py --task drawer_opening \
        --residual-checkpoint outputs/drawer_opening_residualv2_seed42/checkpoints/residual_best.pt \
        --episodes 50 --num-envs 8 --compare-bc
"""

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
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--video-dir", type=Path, default=None)
parser.add_argument(
    "--video-episodes",
    type=int,
    default=-1,
    help="Videos to save per evaluated policy. -1 saves every episode; 0 disables.",
)
parser.add_argument("--video-fps", type=float, default=20.0)
parser.add_argument("--compare-bc", action="store_true")
parser.add_argument("--gripper-authority", type=float, default=None)
parser.add_argument("--flip-margin", type=float, default=None)
parser.add_argument("--ensemble-decay", type=float, default=None)
parser.add_argument("--no-ensemble", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video_episodes < -1:
    raise ValueError("--video-episodes must be -1, 0, or positive")
if args.video_episodes == -1:
    args.video_episodes = args.episodes
if args.video_episodes > 0 and args.num_envs != 1:
    print("video recording enabled: forcing --num-envs 1 so every reported episode has a video", flush=True)
    args.num_envs = 1
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def main() -> dict:
    import gymnasium as gym
    import torch

    from task_pipeline_v2 import (
        ChunkRunner,
        encode_video,
        load_bc_checkpoint_v2,
        load_residual_actor_v2,
        rollout_evaluate,
        set_seed,
    )
    from bc_checkpoint_adapter_v2 import load_bc_checkpoint_for_task
    from track_b_envs.rl import configure_normalized_action_space, load_registered_env_cfg

    if not args.residual_checkpoint.is_file():
        raise FileNotFoundError(args.residual_checkpoint)
    set_seed(args.seed)
    device = torch.device(args.device)

    # A shared residual checkpoint is deliberately reusable across every task.
    actor, payload = load_residual_actor_v2(
        args.residual_checkpoint, device, expected_task=None
    )
    residual_config = payload["config"]
    bc_checkpoint = args.bc_checkpoint or Path(payload["bc_checkpoint"])
    if not bc_checkpoint.is_file():
        raise FileNotFoundError(
            f"BC checkpoint {bc_checkpoint} not found; pass --bc-checkpoint explicitly"
        )
    policy, normalizer, bc_payload = load_bc_checkpoint_for_task(
        bc_checkpoint,
        device,
        args.task,
        prefer_ema=True,
    )
    bc_config = bc_payload["config"]

    authority = (
        args.gripper_authority
        if args.gripper_authority is not None
        else float(payload.get("extra", {}).get("gripper_authority", residual_config["gripper_authority"]))
    )
    flip_margin = (
        args.flip_margin
        if args.flip_margin is not None
        else float(residual_config["flip_margin"])
    )
    ensemble_decay = (
        args.ensemble_decay
        if args.ensemble_decay is not None
        else float(residual_config.get("ensemble_decay", 0.35))
    )
    output = args.output or args.residual_checkpoint.parent.parent / "residual_eval.json"
    video_dir = args.video_dir or output.parent / "videos"

    task_id, cfg = load_registered_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    env = gym.make(task_id, cfg=cfg)
    configure_normalized_action_space(env)

    def make_runner() -> ChunkRunner:
        return ChunkRunner(
            policy,
            normalizer,
            num_envs=args.num_envs,
            device=device,
            frame_stack=int(bc_config["frame_stack"]),
            frame_stride=int(bc_config["frame_stride"]),
            chunk=int(bc_config["chunk"]),
            image_size=int(bc_config["image_size"]),
            ensemble_decay=ensemble_decay,
            use_ensemble=not args.no_ensemble,
        )

    summary: dict = {
        "task": args.task,
        "residual_checkpoint": str(args.residual_checkpoint.resolve()),
        "bc_checkpoint": str(Path(bc_checkpoint).resolve()),
        "gripper_authority": authority,
        "flip_margin": flip_margin,
        "episodes": args.episodes,
    }

    def video_writer(policy_name):
        def save(row, frames):
            outcome = "success" if row["success"] else "failure"
            path = video_dir / policy_name / (
                f"{args.task}_{policy_name}_episode_{row['episode']:03d}_{outcome}.mp4"
            )
            encode_video(frames, path, args.video_fps)
            print(f"saved video: {path}", flush=True)
            return str(path.resolve())

        return save

    try:
        if args.compare_bc:
            set_seed(args.seed)
            bc_result = rollout_evaluate(
                env,
                make_runner(),
                device,
                episodes=args.episodes,
                agent=None,
                capture_episodes=args.video_episodes,
                video_callback=video_writer("bcv2") if args.video_episodes else None,
                reset_seed=args.seed,
            )
            summary["bc_only"] = {
                key: bc_result[key]
                for key in ("episodes", "successes", "success_rate", "mean_return", "mean_length")
            }
            summary["bc_videos"] = bc_result["videos"]
            print(json.dumps({"stage": "bc_only", **summary["bc_only"]}, indent=2), flush=True)

        set_seed(args.seed)
        result = rollout_evaluate(
            env,
            make_runner(),
            device,
            episodes=args.episodes,
            actor=actor,
            gripper_authority=authority,
            flip_margin=flip_margin,
            capture_episodes=args.video_episodes,
            video_callback=video_writer("residualv2") if args.video_episodes else None,
            reset_seed=args.seed,
        )
    finally:
        env.close()

    files = result.pop("videos")
    summary["residual"] = {
        key: result[key]
        for key in ("episodes", "successes", "success_rate", "mean_return", "mean_length")
    }
    summary["videos"] = files
    summary["results"] = result["results"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
