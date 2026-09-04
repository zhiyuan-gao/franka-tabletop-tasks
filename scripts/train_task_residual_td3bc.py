#!/usr/bin/env python3
"""Train residual TD3+BC over a frozen BC policy for one Track B task."""

from __future__ import annotations

import argparse
import json
from collections import deque
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
parser.add_argument("--bc-checkpoint", type=Path, required=True)
parser.add_argument("--dataset-dir", type=Path, action="append", default=None)
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument("--transitions", type=int, default=140_000)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--buffer-size", type=int, default=150_000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--demo-batch-size", type=int, default=256)
parser.add_argument("--demo-loader-batch-size", type=int, default=256)
parser.add_argument("--start-training-after", type=int, default=2_000)
parser.add_argument("--random-warmup-transitions", type=int, default=4_000)
parser.add_argument("--updates-per-env-step", type=int, default=1)
parser.add_argument("--discount", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--policy-noise", type=float, default=0.05)
parser.add_argument("--noise-clip", type=float, default=0.08)
parser.add_argument("--policy-delay", type=int, default=2)
parser.add_argument("--actor-lr", type=float, default=5.0e-5)
parser.add_argument("--critic-lr", type=float, default=5.0e-5)
parser.add_argument("--weight-decay", type=float, default=1.0e-5)
parser.add_argument("--arm-residual-limit", type=float, default=0.10)
parser.add_argument("--gripper-residual-limit", type=float, default=2.0)
parser.add_argument("--exploration-noise", type=float, default=0.04)
parser.add_argument("--bc-weight-start", type=float, default=0.80)
parser.add_argument("--bc-weight-end", type=float, default=0.20)
parser.add_argument("--bc-decay-transitions", type=int, default=140_000)
parser.add_argument("--checkpoint-interval", type=int, default=20_000)
parser.add_argument("--log-interval", type=int, default=2_000)
parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="Keep 0 by default because Isaac Sim is already initialized in this process.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--skip-full-validation", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output_dir is None:
    args.output_dir = Path("outputs") / f"{args.task}_residual_td3bc_seed{args.seed}"
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def main() -> Path:
    import gymnasium as gym
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from task_pipeline_common import (
        ACTION_DIM,
        TaskStepDataset,
        ReplayBuffer,
        ResidualTD3BCAgent,
        build_demo_feature_bank,
        compose_action,
        environment_success,
        inspect_datasets,
        load_bc_checkpoint,
        load_episode_refs,
        oracle_observation,
        policy_observation,
        resolve_dataset_dirs,
        set_seed,
        split_episode_refs,
        to_numpy,
    )
    from track_b_envs.rl import configure_normalized_action_space, load_registered_env_cfg

    for name in (
        "transitions",
        "num_envs",
        "buffer_size",
        "batch_size",
        "demo_batch_size",
        "start_training_after",
        "random_warmup_transitions",
        "updates_per_env_step",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if not args.bc_checkpoint.is_file():
        raise FileNotFoundError(args.bc_checkpoint)
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir = resolve_dataset_dirs(args.task, args.dataset_dir)

    if not args.skip_full_validation:
        inspect_datasets(args.dataset_dir, args.task)
    bc_model, bc_payload = load_bc_checkpoint(
        args.bc_checkpoint, device, expected_task=args.task
    )
    refs = load_episode_refs(args.dataset_dir, args.task)
    bc_config = bc_payload["config"]
    validation_ratio = 0.10
    split_seed = args.seed
    if bc_config is not None:
        validation_ratio = float(bc_config.get("validation_ratio", validation_ratio))
        split_seed = int(bc_config.get("seed", split_seed))
    train_refs, _ = split_episode_refs(refs, validation_ratio, split_seed)
    dataset = TaskStepDataset(train_refs)
    demo_loader = DataLoader(
        dataset,
        batch_size=args.demo_loader_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    print("building BC residual targets from demonstrations", flush=True)
    demo_bank = build_demo_feature_bank(
        bc_model,
        demo_loader,
        device,
        args.arm_residual_limit,
        args.gripper_residual_limit,
        args.seed,
    )
    print(f"demo feature bank rows: {len(demo_bank)}", flush=True)

    task_id, env_cfg = load_registered_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    env = gym.make(task_id, cfg=env_cfg)
    configure_normalized_action_space(env)
    observations, _ = env.reset(seed=args.seed)
    rgb, proprio = policy_observation(observations, device)
    oracle = oracle_observation(observations, device)
    with torch.inference_mode():
        base_action, feature = bc_model.deterministic_action_and_feature(rgb, proprio)

    feature_dim = int(feature.shape[1])
    oracle_dim = int(oracle.shape[1])
    replay = ReplayBuffer(args.buffer_size, feature_dim, oracle_dim, args.seed)
    agent = ResidualTD3BCAgent(
        feature_dim,
        oracle_dim,
        device,
        arm_limit=args.arm_residual_limit,
        gripper_limit=args.gripper_residual_limit,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
    )
    config = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    config.update(
        {
            "task": args.task,
            "bc_checkpoint": str(args.bc_checkpoint.resolve()),
            "dataset_dirs": [str(path.expanduser().resolve()) for path in args.dataset_dir],
            "bc_epoch": int(bc_payload["epoch"]),
            "feature_dim": feature_dim,
            "oracle_dim": oracle_dim,
            "demo_episodes": len(train_refs),
            "demo_rows": len(demo_bank),
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    transition_step = 0
    next_log = args.log_interval
    next_checkpoint = args.checkpoint_interval
    recent_success = deque(maxlen=100)
    recent_return = deque(maxlen=100)
    episode_returns = torch.zeros(args.num_envs, device=device)
    last_logs: dict[str, float] = {}
    limits = agent.actor.limits
    try:
        while transition_step < args.transitions:
            if transition_step < args.random_warmup_transitions:
                residual = (2.0 * torch.rand_like(base_action) - 1.0) * limits
            else:
                residual = agent.act(
                    feature, base_action, exploration_std=args.exploration_noise
                )
            actions = compose_action(base_action, residual)
            next_observations, rewards, terminated, truncated, _ = env.step(actions)
            success = environment_success(env)
            done = terminated | truncated
            next_rgb, next_proprio = policy_observation(next_observations, device)
            next_oracle = oracle_observation(next_observations, device)
            with torch.inference_mode():
                next_base_action, next_feature = bc_model.deterministic_action_and_feature(
                    next_rgb, next_proprio
                )
            replay.add_batch(
                feature=to_numpy(feature),
                oracle=to_numpy(oracle),
                base_action=to_numpy(base_action),
                residual=to_numpy(residual),
                reward=to_numpy(rewards),
                next_feature=to_numpy(next_feature),
                next_oracle=to_numpy(next_oracle),
                next_base_action=to_numpy(next_base_action),
                done=to_numpy(done.float()),
            )
            transition_step += args.num_envs
            episode_returns += rewards
            for env_id in done.nonzero(as_tuple=False).flatten().tolist():
                recent_success.append(float(success[env_id].item()))
                recent_return.append(float(episode_returns[env_id].item()))
                episode_returns[env_id] = 0.0

            if replay.size >= max(args.batch_size, args.start_training_after):
                for _ in range(args.updates_per_env_step):
                    last_logs = agent.train_step(
                        replay,
                        demo_bank,
                        args.batch_size,
                        args.demo_batch_size,
                        transition_step,
                        args.bc_weight_start,
                        args.bc_weight_end,
                        args.bc_decay_transitions,
                    )

            feature, oracle, base_action = next_feature, next_oracle, next_base_action
            if transition_step >= next_log:
                row = {
                    "transitions": transition_step,
                    "replay_size": replay.size,
                    "episodes_window": len(recent_success),
                    "success_rate_window": float(np.mean(recent_success)) if recent_success else None,
                    "return_window": float(np.mean(recent_return)) if recent_return else None,
                    **last_logs,
                }
                print(json.dumps(row), flush=True)
                with (args.output_dir / "train_log.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                next_log += args.log_interval
            if transition_step >= next_checkpoint:
                path = args.output_dir / "checkpoints" / f"residual_{transition_step:09d}.pt"
                agent.save(path, args.bc_checkpoint, config, args.task)
                print(f"saved checkpoint: {path}", flush=True)
                next_checkpoint += args.checkpoint_interval
    finally:
        env.close()

    final_path = args.output_dir / "checkpoints" / "residual_final.pt"
    agent.save(final_path, args.bc_checkpoint, config, args.task)
    summary = {
        "task": args.task,
        "transitions": transition_step,
        "updates": agent.total_updates,
        "episodes_window": len(recent_success),
        "success_rate_window": float(np.mean(recent_success)) if recent_success else None,
        "return_window": float(np.mean(recent_return)) if recent_return else None,
        "checkpoint": str(final_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return final_path


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
