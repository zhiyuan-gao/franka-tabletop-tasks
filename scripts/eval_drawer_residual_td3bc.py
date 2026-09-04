#!/usr/bin/env python3
"""Evaluate frozen BC and trained residual TD3+BC on drawer opening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--residual-checkpoint", type=Path, required=True)
parser.add_argument("--bc-checkpoint", type=Path, default=None)
parser.add_argument("--policies", nargs="+", choices=("bc", "residual"), default=("bc", "residual"))
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, default=Path("outputs/drawer_residual_td3bc/eval.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def evaluate(env, bc_model, residual_actor, policy_name, episodes, seed, device):
    import torch

    from drawer_pipeline_common import compose_action, policy_observation

    observations, _ = env.reset(seed=seed)
    episode_returns = torch.zeros(args.num_envs, device=device)
    episode_lengths = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    rows = []
    while len(rows) < episodes:
        rgb, proprio = policy_observation(observations, device)
        with torch.inference_mode():
            base_action, feature = bc_model.deterministic_action_and_feature(rgb, proprio)
            if policy_name == "bc":
                actions = base_action
            else:
                residual = residual_actor(feature, base_action)
                actions = compose_action(base_action, residual)
        observations, rewards, terminated, truncated, _ = env.step(actions)
        episode_returns += rewards
        episode_lengths += 1
        ended = terminated | truncated
        for env_id in ended.nonzero(as_tuple=False).flatten().tolist():
            if len(rows) >= episodes:
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
    successes = sum(int(row["success"]) for row in rows)
    return {
        "policy": policy_name,
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "mean_return": sum(row["return"] for row in rows) / len(rows),
        "mean_length": sum(row["length"] for row in rows) / len(rows),
        "results": rows,
    }


def main() -> dict:
    import gymnasium as gym
    import torch

    from drawer_pipeline_common import (
        TASK_NAME,
        load_bc_checkpoint,
        load_residual_actor,
    )
    from track_b_envs.rl import configure_normalized_action_space, load_registered_env_cfg

    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("episodes and num-envs must be positive")
    if not args.residual_checkpoint.is_file():
        raise FileNotFoundError(args.residual_checkpoint)
    device = torch.device(args.device)
    residual_actor, residual_payload = load_residual_actor(
        args.residual_checkpoint, device
    )
    bc_checkpoint = args.bc_checkpoint
    if bc_checkpoint is None:
        bc_checkpoint = Path(residual_payload["bc_checkpoint"])
    if not bc_checkpoint.is_file():
        raise FileNotFoundError(
            f"BC checkpoint not found: {bc_checkpoint}; pass --bc-checkpoint explicitly"
        )
    bc_model, bc_payload = load_bc_checkpoint(bc_checkpoint, device)
    if residual_actor.feature_dim != bc_model.feature_dim:
        raise ValueError(
            f"feature mismatch: residual={residual_actor.feature_dim}, BC={bc_model.feature_dim}"
        )

    task_id, cfg = load_registered_env_cfg(
        TASK_NAME,
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
        "task": TASK_NAME,
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
