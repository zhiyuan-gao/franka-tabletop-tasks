#!/usr/bin/env python3
"""Closed-loop evaluation and checkpoint selection for the v2 BC policy.

Evaluate one checkpoint:

    python scripts/eval_task_bc_v2.py --task drawer_opening \
        --checkpoint outputs/drawer_opening_bcv2_seed42/bc_last.pt --episodes 40

Sweep every periodic checkpoint and copy the best-by-success one to
``bc_selected.pt`` -- this is the fix for "bc_best.pt is the best validation
loss, not the best policy":

    python scripts/eval_task_bc_v2.py --task drawer_opening \
        --checkpoint-dir outputs/drawer_opening_bcv2_seed42/checkpoints \
        --episodes 30 --num-envs 8
"""

from __future__ import annotations

import argparse
import json
import shutil
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
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--checkpoint-dir", type=Path, default=None)
parser.add_argument("--episodes", type=int, default=40)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--video-dir", type=Path, default=None)
parser.add_argument(
    "--video-episodes",
    type=int,
    default=-1,
    help="Videos to save per checkpoint. -1 saves every episode; 0 disables.",
)
parser.add_argument("--video-fps", type=float, default=20.0)
parser.add_argument("--no-ensemble", action="store_true")
parser.add_argument("--ensemble-decay", type=float, default=0.35)
parser.add_argument("--raw-weights", action="store_true", help="Use non-EMA weights.")
parser.add_argument(
    "--select-into",
    type=Path,
    default=None,
    help="Copy the winning checkpoint here (default: <checkpoint-dir>/../bc_selected.pt).",
)
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
        rollout_evaluate,
        set_seed,
    )
    from track_b_envs.rl import configure_normalized_action_space, load_registered_env_cfg

    if (args.checkpoint is None) == (args.checkpoint_dir is None):
        raise ValueError("pass exactly one of --checkpoint or --checkpoint-dir")
    if args.checkpoint is not None:
        checkpoints = [args.checkpoint]
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        default_output = args.checkpoint.parent / "bc_eval.json"
    else:
        checkpoints = sorted(args.checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"no .pt files in {args.checkpoint_dir}")
        default_output = args.checkpoint_dir.parent / "bc_selection.json"
    output = args.output or default_output
    video_dir = args.video_dir or output.parent / "videos"

    set_seed(args.seed)
    device = torch.device(args.device)
    task_id, cfg = load_registered_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    env = gym.make(task_id, cfg=cfg)
    configure_normalized_action_space(env)

    entries = []
    try:
        for checkpoint in checkpoints:
            policy, normalizer, payload = load_bc_checkpoint_v2(
                checkpoint, device, expected_task=args.task, prefer_ema=not args.raw_weights
            )
            config = payload["config"]
            runner = ChunkRunner(
                policy,
                normalizer,
                num_envs=args.num_envs,
                device=device,
                frame_stack=int(config["frame_stack"]),
                frame_stride=int(config["frame_stride"]),
                chunk=int(config["chunk"]),
                image_size=int(config["image_size"]),
                ensemble_decay=args.ensemble_decay,
                use_ensemble=not args.no_ensemble,
            )
            capture = args.video_episodes

            def save_video(row, frames):
                outcome = "success" if row["success"] else "failure"
                directory = video_dir / checkpoint.stem if len(checkpoints) > 1 else video_dir
                path = directory / (
                    f"{args.task}_bcv2_episode_{row['episode']:03d}_{outcome}.mp4"
                )
                encode_video(frames, path, args.video_fps)
                print(f"saved video: {path}", flush=True)
                return str(path.resolve())

            result = rollout_evaluate(
                env,
                runner,
                device,
                episodes=args.episodes,
                agent=None,
                capture_episodes=capture,
                video_callback=save_video if capture else None,
                reset_seed=args.seed,
            )
            files = result.pop("videos")
            entry = {
                "checkpoint": str(checkpoint.resolve()),
                "epoch": int(payload["epoch"]),
                "success_rate": result["success_rate"],
                "successes": result["successes"],
                "episodes": result["episodes"],
                "mean_return": result["mean_return"],
                "mean_length": result["mean_length"],
                "videos": files,
                "results": result["results"] if len(checkpoints) == 1 else None,
            }
            entries.append(entry)
            print(
                json.dumps(
                    {k: v for k, v in entry.items() if k != "results"}, indent=2
                ),
                flush=True,
            )
    finally:
        env.close()

    entries.sort(key=lambda item: (-item["success_rate"], item["epoch"]))
    best = entries[0]
    summary = {
        "policy": "bc_v2",
        "task": args.task,
        "episodes_per_checkpoint": args.episodes,
        "best": {k: v for k, v in best.items() if k != "results"},
        "all": [{k: v for k, v in entry.items() if k != "results"} for entry in entries],
        "detail": entries[0].get("results"),
    }
    if len(checkpoints) > 1:
        target = args.select_into or (args.checkpoint_dir.parent / "bc_selected.pt")
        shutil.copyfile(best["checkpoint"], target)
        summary["selected_checkpoint"] = str(Path(target).resolve())
        print(f"selected: {best['checkpoint']} -> {target}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["best"], indent=2), flush=True)
    return summary


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
