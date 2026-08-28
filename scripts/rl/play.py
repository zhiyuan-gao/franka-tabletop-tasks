"""Evaluate a deterministic SKRL visual policy checkpoint."""

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
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--config", type=Path, default=None)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=20260827)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch
    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    from track_b_envs.agents.skrl import VisualGaussianPolicy, load_visual_ppo_config
    from track_b_envs.rl import (
        configure_normalized_action_space,
        load_registered_env_cfg,
    )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("num-envs and steps must be positive")
    print(json.dumps({"stage": "loading_environment", "task": args.task}), flush=True)
    task_id, env_cfg = load_registered_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    env = gym.make(task_id, cfg=env_cfg)
    print(json.dumps({"stage": "environment_ready", "gym_id": task_id}), flush=True)
    configure_normalized_action_space(env)
    wrapped_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    raw_cfg = load_visual_ppo_config(args.config)
    model_cfg = raw_cfg["model"]
    policy = VisualGaussianPolicy(
        wrapped_env.observation_space,
        wrapped_env.action_space,
        wrapped_env.device,
        image_feature_dim=int(model_cfg["image_feature_dim"]),
        proprio_feature_dim=int(model_cfg["proprio_feature_dim"]),
        hidden_dims=tuple(model_cfg["actor_hidden_dims"]),
        initial_log_std=float(model_cfg["initial_log_std"]),
    )
    policy.to(wrapped_env.device)
    payload = torch.load(
        args.checkpoint, map_location=wrapped_env.device, weights_only=False
    )
    if isinstance(payload, dict) and "policy" in payload:
        payload = payload["policy"]
    policy.load_state_dict(payload)
    policy.eval()
    print(json.dumps({"stage": "checkpoint_loaded", "path": str(args.checkpoint)}), flush=True)

    observations, _ = wrapped_env.reset()
    print(json.dumps({"stage": "evaluating", "steps": args.steps}), flush=True)
    terminated_count = 0
    truncated_count = 0
    try:
        for step in range(args.steps):
            with torch.inference_mode():
                _, outputs = policy.act(
                    {"observations": observations}, role="policy"
                )
                actions = outputs["mean_actions"]
                observations, _, terminated, truncated, _ = wrapped_env.step(actions)
                terminated_count += int(terminated.sum().item())
                truncated_count += int(truncated.sum().item())
            if step == 0:
                print(
                    json.dumps(
                        {
                            "stage": "first_step_complete",
                            "action_min": float(actions.min().item()),
                            "action_max": float(actions.max().item()),
                        }
                    ),
                    flush=True,
                )
        print(
            json.dumps(
                {
                    "task": args.task,
                    "steps": args.steps,
                    "terminated_episodes": terminated_count,
                    "truncated_episodes": truncated_count,
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import traceback

        print(f"[play] failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
