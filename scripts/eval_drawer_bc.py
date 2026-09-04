#!/usr/bin/env python3
"""Run deterministic closed-loop evaluation of the drawer-opening BC policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def main() -> dict:
    import gymnasium as gym
    import torch

    from drawer_pipeline_common import (
        TASK_NAME,
        load_bc_checkpoint,
        policy_observation,
    )
    from track_b_envs.rl import configure_normalized_action_space, load_registered_env_cfg

    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("episodes and num-envs must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device(args.device)
    task_id, cfg = load_registered_env_cfg(
        TASK_NAME,
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    env = gym.make(task_id, cfg=cfg)
    configure_normalized_action_space(env)
    model, payload = load_bc_checkpoint(args.checkpoint, device)
    observations, _ = env.reset(seed=args.seed)
    episode_returns = torch.zeros(args.num_envs, device=device)
    episode_lengths = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    rows = []
    try:
        while len(rows) < args.episodes:
            rgb, proprio = policy_observation(observations, device)
            with torch.inference_mode():
                actions, _ = model.deterministic_action_and_feature(rgb, proprio)
            observations, rewards, terminated, truncated, _ = env.step(actions)
            episode_returns += rewards
            episode_lengths += 1
            ended = terminated | truncated
            for env_id in ended.nonzero(as_tuple=False).flatten().tolist():
                if len(rows) >= args.episodes:
                    break
                rows.append(
                    {
                        "episode": len(rows),
                        "env_id": env_id,
                        "success": bool(terminated[env_id].item()),
                        "truncated": bool(truncated[env_id].item()),
                        "return": float(episode_returns[env_id].item()),
                        "length": int(episode_lengths[env_id].item()),
                    }
                )
                episode_returns[env_id] = 0.0
                episode_lengths[env_id] = 0
    finally:
        env.close()
    successes = sum(int(row["success"]) for row in rows)
    result = {
        "policy": "bc",
        "task": TASK_NAME,
        "checkpoint": str(args.checkpoint.resolve()),
        "bc_epoch": int(payload["epoch"]),
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "mean_return": sum(row["return"] for row in rows) / len(rows),
        "mean_length": sum(row["length"] for row in rows) / len(rows),
        "results": rows,
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
