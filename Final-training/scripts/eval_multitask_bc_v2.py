#!/usr/bin/env python3
"""Evaluate/select one task-conditioned BC checkpoint across four tasks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

MULTITASK_TASKS = (
    "push_cube_to_region",
    "drawer_opening",
    "object_to_drawer",
    "nut_pouring",
)


parser = argparse.ArgumentParser(description=__doc__)
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--checkpoint", type=Path)
source.add_argument("--checkpoint-dir", type=Path)
parser.add_argument("--tasks", nargs="+", choices=MULTITASK_TASKS, default=list(MULTITASK_TASKS))
parser.add_argument("--episodes-per-task", type=int, default=20)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=4242)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--select-into", type=Path, default=None)
parser.add_argument("--video-dir", type=Path, default=None)
parser.add_argument(
    "--video-episodes",
    type=int,
    default=0,
    help="Videos per task/checkpoint. -1 records every evaluated episode.",
)
parser.add_argument("--video-fps", type=float, default=20.0)
parser.add_argument("--ensemble-decay", type=float, default=0.35)
parser.add_argument("--raw-weights", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.episodes_per_task <= 0:
    raise ValueError("--episodes-per-task must be positive")
if args.video_episodes < -1:
    raise ValueError("--video-episodes must be -1, 0, or positive")
if args.video_episodes == -1:
    args.video_episodes = args.episodes_per_task
if args.video_episodes > 0 and args.num_envs != 1:
    print("video enabled: forcing --num-envs 1", flush=True)
    args.num_envs = 1
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def main() -> dict:
    import gymnasium as gym
    import torch

    from task_pipeline_multitask_v2 import (
        TASK_TO_ID,
        BoundTaskPolicy,
        load_multitask_bc_checkpoint,
    )
    from task_pipeline_v2 import ChunkRunner, encode_video, rollout_evaluate, set_seed
    from track_b_envs.rl import configure_normalized_action_space, load_registered_env_cfg

    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        checkpoints = [args.checkpoint]
    else:
        checkpoints = sorted(args.checkpoint_dir.glob("bc_multitask_epoch_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"no multitask epoch checkpoints in {args.checkpoint_dir}")

    video_root = args.video_dir or args.output.parent / "multitask_videos"
    device = torch.device(args.device)
    set_seed(args.seed)
    entries = []

    for checkpoint in checkpoints:
        policy, normalizers, payload = load_multitask_bc_checkpoint(
            checkpoint, device, prefer_ema=not args.raw_weights
        )
        config = payload["config"]
        task_results = {}
        for task in args.tasks:
            task_id, cfg = load_registered_env_cfg(
                task,
                device=args.device,
                num_envs=args.num_envs,
                seed=args.seed,
                use_fabric=not getattr(args, "disable_fabric", False),
            )
            env = gym.make(task_id, cfg=cfg)
            configure_normalized_action_space(env)
            try:
                bound_policy = BoundTaskPolicy(policy, TASK_TO_ID[task])
                runner = ChunkRunner(
                    bound_policy,
                    normalizers[task],
                    num_envs=args.num_envs,
                    device=device,
                    frame_stack=int(config["frame_stack"]),
                    frame_stride=int(config["frame_stride"]),
                    chunk=int(config["chunk"]),
                    image_size=int(config["image_size"]),
                    ensemble_decay=args.ensemble_decay,
                    use_ensemble=True,
                )

                def save_video(row, frames, *, task_name=task):
                    outcome = "success" if row["success"] else "failure"
                    prefix = checkpoint.stem if len(checkpoints) > 1 else "selected"
                    path = video_root / prefix / task_name / (
                        f"{task_name}_multitask_bcv2_episode_{row['episode']:03d}_{outcome}.mp4"
                    )
                    encode_video(frames, path, args.video_fps)
                    print(f"saved video: {path}", flush=True)
                    return str(path.resolve())

                result = rollout_evaluate(
                    env,
                    runner,
                    device,
                    episodes=args.episodes_per_task,
                    agent=None,
                    capture_episodes=args.video_episodes,
                    video_callback=save_video if args.video_episodes else None,
                    reset_seed=args.seed,
                )
            finally:
                env.close()
            task_results[task] = result
            print(json.dumps({
                "checkpoint": str(checkpoint),
                "epoch": int(payload["epoch"]),
                "task": task,
                "successes": result["successes"],
                "episodes": result["episodes"],
                "success_rate": result["success_rate"],
                "mean_return": result["mean_return"],
                "mean_length": result["mean_length"],
            }), flush=True)

        mean_success = sum(
            result["success_rate"] for result in task_results.values()
        ) / len(task_results)
        entry = {
            "checkpoint": str(checkpoint.resolve()),
            "epoch": int(payload["epoch"]),
            "mean_task_success_rate": mean_success,
            "tasks": task_results,
        }
        entries.append(entry)
        print(json.dumps({
            "stage": "checkpoint_complete",
            "checkpoint": entry["checkpoint"],
            "epoch": entry["epoch"],
            "mean_task_success_rate": mean_success,
        }), flush=True)

    entries.sort(key=lambda row: (-row["mean_task_success_rate"], row["epoch"]))
    best = entries[0]
    summary = {
        "policy": "multitask_bc_v2",
        "selection_seed": args.seed,
        "episodes_per_task": args.episodes_per_task,
        "evaluated_tasks": list(args.tasks),
        "best_checkpoint": best["checkpoint"],
        "best_epoch": best["epoch"],
        "best_mean_task_success_rate": best["mean_task_success_rate"],
        "best_task_results": best["tasks"],
        "all_checkpoints": [
            {
                "checkpoint": row["checkpoint"],
                "epoch": row["epoch"],
                "mean_task_success_rate": row["mean_task_success_rate"],
                "task_success_rates": {
                    task: result["success_rate"]
                    for task, result in row["tasks"].items()
                },
            }
            for row in entries
        ],
    }
    if args.checkpoint_dir is not None:
        target = args.select_into or args.checkpoint_dir.parent / "bc_multitask_selected.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best["checkpoint"], target)
        summary["selected_checkpoint"] = str(target.resolve())
        print(f"selected: {best['checkpoint']} -> {target}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "best_checkpoint": summary["best_checkpoint"],
        "best_epoch": summary["best_epoch"],
        "best_mean_task_success_rate": summary["best_mean_task_success_rate"],
        "task_success_rates": {
            task: result["success_rate"]
            for task, result in summary["best_task_results"].items()
        },
    }, indent=2))
    return summary


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
