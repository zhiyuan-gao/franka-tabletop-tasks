#!/usr/bin/env python3
"""Residual TD3+BC on top of a frozen v2 BC policy.

Everything that made the previous run peak at 55% and then collapse is
addressed here:

* the optimized reward is success-dominated (dense env reward is scaled down,
  first-time success gets a large bonus), so partial-opening reward hacking no
  longer pays,
* gamma defaults to 0.997 and returns are n-step, so terminal success is not
  discounted into irrelevance over a ~280-step episode,
* time-limit truncation is no longer treated as a terminal state,
* BC anchoring uses the TD3+BC normalized lambda and does not decay to zero,
* successful trajectories go into a protected replay buffer and into a
  self-imitation bank, so good behaviour cannot be diluted by later failures,
* the gripper correction has to clear a margin to flip the BC decision, its
  authority is annealed in after the arm residual has settled, and flips are
  penalized,
* deterministic evaluation runs during training and the best policy is saved.

Example:

    python scripts/train_task_residual_v2.py --task drawer_opening \
        --bc-checkpoint outputs/drawer_opening_bcv2_seed42/bc_selected.pt \
        --num-envs 8 --transitions 300000
"""

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
parser.add_argument("--transitions", type=int, default=300_000)
parser.add_argument("--num-envs", type=int, default=8)

# replay / updates
parser.add_argument("--buffer-size", type=int, default=250_000)
parser.add_argument("--success-buffer-size", type=int, default=100_000)
parser.add_argument("--success-fraction", type=float, default=0.25)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--demo-batch-size", type=int, default=256)
parser.add_argument("--start-training-after", type=int, default=5_000)
parser.add_argument("--random-warmup-transitions", type=int, default=3_000)
parser.add_argument("--warmup-residual-scale", type=float, default=0.5)
parser.add_argument(
    "--updates-per-env-step",
    type=int,
    default=0,
    help="0 selects --num-envs, i.e. one gradient update per collected transition.",
)
parser.add_argument("--n-step", type=int, default=5)

# TD3+BC
parser.add_argument("--discount", type=float, default=0.997)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--policy-noise", type=float, default=0.10)
parser.add_argument("--noise-clip", type=float, default=0.25)
parser.add_argument("--policy-delay", type=int, default=2)
parser.add_argument("--actor-lr", type=float, default=1.0e-4)
parser.add_argument("--critic-lr", type=float, default=3.0e-4)
parser.add_argument("--actor-hidden", type=int, default=512)
parser.add_argument("--weight-decay", type=float, default=1.0e-5)
parser.add_argument("--alpha", type=float, default=1.0, help="TD3+BC RL weight.")
parser.add_argument("--lambda-max", type=float, default=10.0)
parser.add_argument("--bc-weight-start", type=float, default=1.0)
parser.add_argument("--bc-weight-end", type=float, default=0.50)
parser.add_argument("--bc-decay-transitions", type=int, default=200_000)
parser.add_argument("--self-imitation-weight", type=float, default=0.50)
parser.add_argument("--self-imitation-capacity", type=int, default=100_000)

# residual authority
parser.add_argument("--arm-residual-limit", type=float, default=0.15)
parser.add_argument("--gripper-authority", type=float, default=2.0)
parser.add_argument("--gripper-warmup-transitions", type=int, default=40_000)
parser.add_argument("--gripper-ramp-transitions", type=int, default=40_000)
parser.add_argument("--flip-margin", type=float, default=0.25)

# reward shaping
parser.add_argument("--dense-reward-scale", type=float, default=0.10)
parser.add_argument("--success-bonus", type=float, default=10.0)
parser.add_argument("--flip-penalty", type=float, default=0.02)
parser.add_argument("--residual-penalty", type=float, default=0.01)

# exploration
parser.add_argument("--exploration-noise", type=float, default=0.15)
parser.add_argument("--noise-correlation", type=float, default=0.85)
parser.add_argument("--anchor-episode-probability", type=float, default=0.10)

# evaluation / logging
parser.add_argument("--eval-interval", type=int, default=10_000)
parser.add_argument("--eval-episodes", type=int, default=24)
parser.add_argument(
    "--eval-seed",
    type=int,
    default=1234,
    help="Fixed Isaac reset seed used for every deterministic checkpoint evaluation.",
)
parser.add_argument("--eval-bc-baseline", action="store_true")
parser.add_argument("--checkpoint-interval", type=int, default=25_000)
parser.add_argument("--log-interval", type=int, default=2_000)
parser.add_argument("--ensemble-decay", type=float, default=0.35)
parser.add_argument("--no-ensemble", action="store_true")
parser.add_argument("--raw-bc-weights", action="store_true")
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--skip-full-validation", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output_dir is None:
    args.output_dir = Path("outputs") / f"{args.task}_residualv2_seed{args.seed}"
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def gripper_authority_at(transition_step: int) -> float:
    """0 while the arm residual settles, then ramp linearly to full authority."""

    if transition_step <= args.gripper_warmup_transitions:
        return 0.0
    span = max(args.gripper_ramp_transitions, 1)
    progress = (transition_step - args.gripper_warmup_transitions) / span
    return float(min(progress, 1.0) * args.gripper_authority)


def bc_weight_at(transition_step: int) -> float:
    fraction = min(transition_step / max(args.bc_decay_transitions, 1), 1.0)
    return float(
        (1.0 - fraction) * args.bc_weight_start + fraction * args.bc_weight_end
    )


def main() -> Path:
    import gymnasium as gym
    import numpy as np
    import torch

    from task_pipeline_v2 import (
        ACTION_DIM,
        ARM_DIM,
        ChunkRunner,
        EpisodeRecorder,
        MixedReplay,
        ResidualDemoBank,
        ResidualTD3BCAgentV2,
        SequentialEpisodeDataset,
        build_nstep_transitions,
        compose_action_v2,
        demonstration_residual_target,
        environment_success,
        inspect_datasets,
        json_line,
        load_bc_checkpoint_v2,
        load_episode_refs,
        oracle_observation,
        policy_observation,
        resolve_dataset_dirs,
        rollout_evaluate,
        set_seed,
        split_episode_refs,
    )
    from track_b_envs.rl import configure_normalized_action_space, load_registered_env_cfg

    for name in ("transitions", "num_envs", "buffer_size", "batch_size", "n_step"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not args.bc_checkpoint.is_file():
        raise FileNotFoundError(args.bc_checkpoint)
    updates_per_env_step = args.updates_per_env_step or args.num_envs

    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir = resolve_dataset_dirs(args.task, args.dataset_dir)
    if not args.skip_full_validation:
        inspect_datasets(args.dataset_dir, args.task)

    bc_policy, normalizer, bc_payload = load_bc_checkpoint_v2(
        args.bc_checkpoint, device, expected_task=args.task, prefer_ema=not args.raw_bc_weights
    )
    bc_config = bc_payload["config"]

    # ------------------------------------------------------------------ #
    # demonstration residual bank, built by replaying the *online* runner
    # ------------------------------------------------------------------ #
    refs = load_episode_refs(args.dataset_dir, args.task)
    train_refs, _ = split_episode_refs(
        refs,
        float(bc_config.get("validation_ratio", 0.10)),
        int(bc_config.get("seed", args.seed)),
    )
    demo_dataset = SequentialEpisodeDataset(train_refs, normalizer, image_size=224)
    demo_runner = ChunkRunner(
        bc_policy,
        normalizer,
        num_envs=1,
        device=device,
        frame_stack=int(bc_config["frame_stack"]),
        frame_stride=int(bc_config["frame_stride"]),
        chunk=int(bc_config["chunk"]),
        image_size=int(bc_config["image_size"]),
        ensemble_decay=args.ensemble_decay,
        use_ensemble=not args.no_ensemble,
    )
    demo_bank = ResidualDemoBank(args.seed)
    print("rebuilding base actions on the demonstrations", flush=True)
    total_rows = 0
    for episode_id in range(len(demo_dataset)):
        episode = demo_dataset[episode_id]
        rgb = episode["rgb"].to(device).float().div_(255.0)
        proprio = episode["proprio"].to(device)
        demo_action = episode["action"].numpy()
        demo_runner.mark_reset()
        features = []
        base_actions = []
        for step in range(rgb.shape[0]):
            base, feature = demo_runner.step(rgb[step : step + 1], proprio[step : step + 1])
            base_actions.append(base[0].cpu().numpy())
            features.append(feature[0].cpu().numpy())
        base_actions = np.stack(base_actions).astype(np.float32)
        features = np.stack(features).astype(np.float32)
        target = demonstration_residual_target(
            demo_action,
            base_actions,
            args.arm_residual_limit,
            args.gripper_authority,
            args.flip_margin,
        )
        if demo_bank.capacity is None:
            demo_bank.capacity = sum(ref.steps for ref in train_refs)
        demo_bank.add(features, proprio.cpu().numpy(), base_actions, target)
        total_rows += rgb.shape[0]
        if (episode_id + 1) % 10 == 0 or episode_id + 1 == len(demo_dataset):
            print(f"demo episodes replayed: {episode_id + 1}/{len(demo_dataset)}", flush=True)
    feature_dim = int(demo_bank.feature.shape[1])
    print(json.dumps({"stage": "demo_bank", "rows": len(demo_bank), "feature_dim": feature_dim}), flush=True)

    # ------------------------------------------------------------------ #
    # environment
    # ------------------------------------------------------------------ #
    task_id, env_cfg = load_registered_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    env = gym.make(task_id, cfg=env_cfg)
    configure_normalized_action_space(env)

    runner = ChunkRunner(
        bc_policy,
        normalizer,
        num_envs=args.num_envs,
        device=device,
        frame_stack=int(bc_config["frame_stack"]),
        frame_stride=int(bc_config["frame_stride"]),
        chunk=int(bc_config["chunk"]),
        image_size=int(bc_config["image_size"]),
        ensemble_decay=args.ensemble_decay,
        use_ensemble=not args.no_ensemble,
    )

    observations, _ = env.reset(seed=args.seed)
    oracle = oracle_observation(observations, device)
    oracle_dim = int(oracle.shape[1])
    replay = MixedReplay(
        args.buffer_size, args.success_buffer_size, feature_dim, oracle_dim, args.seed
    )
    self_imitation = ResidualDemoBank(args.seed + 7, capacity=args.self_imitation_capacity)
    agent = ResidualTD3BCAgentV2(
        feature_dim,
        oracle_dim,
        device,
        arm_limit=args.arm_residual_limit,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        alpha=args.alpha,
        lambda_max=args.lambda_max,
        hidden=args.actor_hidden,
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
            "dataset_dirs": [str(path) for path in args.dataset_dir],
            "feature_dim": feature_dim,
            "oracle_dim": oracle_dim,
            "demo_rows": len(demo_bank),
            "updates_per_env_step": updates_per_env_step,
            "bc_config": bc_config,
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    if args.eval_bc_baseline:
        baseline = rollout_evaluate(
            env,
            runner,
            device,
            episodes=args.eval_episodes,
            agent=None,
            reset_seed=args.eval_seed,
        )
        baseline.pop("videos", None)
        baseline.pop("results", None)
        baseline["stage"] = "bc_baseline"
        print(json.dumps(baseline, indent=2), flush=True)
        json_line(args.output_dir / "eval_log.jsonl", {"transitions": 0, **baseline})
        observations, _ = env.reset(seed=args.seed)

    # ------------------------------------------------------------------ #
    # rollout state
    # ------------------------------------------------------------------ #
    runner.mark_reset()
    rgb, proprio = policy_observation(observations, device)
    oracle = oracle_observation(observations, device)
    base_action, feature = runner.step(rgb, proprio)

    recorder = EpisodeRecorder(args.num_envs)
    generator = np.random.default_rng(args.seed + 3)
    if args.num_envs > 1:
        noise_scale = args.exploration_noise * (
            0.25 + 1.75 * np.arange(args.num_envs) / (args.num_envs - 1)
        )
    else:
        noise_scale = np.full(args.num_envs, args.exploration_noise)
    noise_scale_t = torch.tensor(noise_scale, dtype=torch.float32, device=device).view(-1, 1)
    correlated_noise = torch.zeros((args.num_envs, ACTION_DIM), device=device)
    anchor = torch.from_numpy(
        generator.random(args.num_envs) < args.anchor_episode_probability
    ).to(device)
    previous_success = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    transition_step = 0
    next_log = args.log_interval
    next_eval = args.eval_interval
    next_checkpoint = args.checkpoint_interval
    recent_success = deque(maxlen=200)
    recent_shaped_return = deque(maxlen=200)
    episode_shaped_return = torch.zeros(args.num_envs, device=device)
    last_logs: dict[str, float] = {}
    best_success = -1.0
    best_checkpoint = args.output_dir / "checkpoints" / "residual_best.pt"

    def store_episode(env_id: int, terminated_flag: bool, truncated_flag: bool, successful: bool) -> None:
        if recorder.length(env_id) < 2:
            recorder.pop(env_id)
            return
        episode = recorder.pop(env_id)
        if truncated_flag and not terminated_flag:
            # the observation after a time limit belongs to the next episode
            episode = {key: value[:-1] for key, value in episode.items()}
            if episode["reward"].shape[0] < 1:
                return
        transitions = build_nstep_transitions(
            episode,
            args.n_step,
            args.discount,
            terminated_at_end=bool(terminated_flag),
        )
        replay.add_batch(successful=successful, **transitions)
        if successful and args.self_imitation_weight > 0.0:
            self_imitation.add(
                episode["feature"],
                episode["proprio"],
                episode["base_action"],
                episode["residual"],
            )

    try:
        while transition_step < args.transitions:
            authority = gripper_authority_at(transition_step)
            if transition_step < args.random_warmup_transitions:
                residual = (
                    (2.0 * torch.rand_like(base_action) - 1.0)
                    * agent.actor.limits
                    * args.warmup_residual_scale
                )
            else:
                correlated_noise = (
                    args.noise_correlation * correlated_noise
                    + (1.0 - args.noise_correlation**2) ** 0.5
                    * torch.randn_like(correlated_noise)
                    * noise_scale_t
                )
                residual = agent.act(feature, base_action, proprio, noise=correlated_noise)
            residual = torch.where(anchor.view(-1, 1), torch.zeros_like(residual), residual)

            action = compose_action_v2(base_action, residual, authority, args.flip_margin)
            flipped = (action[:, ARM_DIM] != base_action[:, ARM_DIM]).float()

            next_observations, rewards, terminated, truncated, _ = env.step(action)
            success = environment_success(env)
            done = terminated | truncated
            success_event = (success & ~previous_success).float()

            shaped = (
                args.dense_reward_scale * rewards
                + args.success_bonus * success_event
                - args.flip_penalty * flipped
                - args.residual_penalty
                * (residual[:, :ARM_DIM] / max(args.arm_residual_limit, 1e-6))
                .pow(2)
                .mean(dim=-1)
            )

            next_rgb, next_proprio = policy_observation(next_observations, device)
            next_oracle = oracle_observation(next_observations, device)
            runner.mark_reset(done)
            next_base_action, next_feature = runner.step(next_rgb, next_proprio)

            feature_np = feature.cpu().numpy()
            proprio_np = proprio.cpu().numpy()
            oracle_np = oracle.cpu().numpy()
            base_np = base_action.cpu().numpy()
            residual_np = residual.cpu().numpy()
            shaped_np = shaped.detach().cpu().numpy()
            next_feature_np = next_feature.cpu().numpy()
            next_proprio_np = next_proprio.cpu().numpy()
            next_oracle_np = next_oracle.cpu().numpy()
            next_base_np = next_base_action.cpu().numpy()
            for env_id in range(args.num_envs):
                recorder.append(
                    env_id,
                    feature=feature_np[env_id],
                    proprio=proprio_np[env_id],
                    oracle=oracle_np[env_id],
                    base_action=base_np[env_id],
                    residual=residual_np[env_id],
                    reward=np.float32(shaped_np[env_id]),
                    next_feature=next_feature_np[env_id],
                    next_proprio=next_proprio_np[env_id],
                    next_oracle=next_oracle_np[env_id],
                    next_base_action=next_base_np[env_id],
                )

            transition_step += args.num_envs
            episode_shaped_return += shaped.detach()

            if bool(done.any()):
                done_ids = done.nonzero(as_tuple=False).flatten().tolist()
                for env_id in done_ids:
                    successful = bool(success[env_id].item())
                    store_episode(
                        env_id,
                        bool(terminated[env_id].item()),
                        bool(truncated[env_id].item()),
                        successful,
                    )
                    recent_success.append(float(successful))
                    recent_shaped_return.append(float(episode_shaped_return[env_id].item()))
                    episode_shaped_return[env_id] = 0.0
                    correlated_noise[env_id] = 0.0
                    anchor[env_id] = bool(
                        generator.random() < args.anchor_episode_probability
                    )
                previous_success[done] = False
            previous_success = torch.where(done, torch.zeros_like(success), success)

            if (
                replay.main.size >= max(args.batch_size, args.start_training_after)
                and len(demo_bank) >= args.demo_batch_size
            ):
                weight = bc_weight_at(transition_step)
                for _ in range(updates_per_env_step):
                    last_logs = agent.train_step(
                        replay,
                        demo_bank,
                        self_imitation,
                        args.batch_size,
                        args.demo_batch_size,
                        args.success_fraction,
                        weight,
                        args.self_imitation_weight,
                    )

            feature, proprio, oracle = next_feature, next_proprio, next_oracle
            base_action = next_base_action
            observations = next_observations

            if transition_step >= next_log:
                row = {
                    "transitions": transition_step,
                    "updates": agent.total_updates,
                    "replay_main": replay.main.size,
                    "replay_success": replay.success.size,
                    "self_imitation": len(self_imitation),
                    "gripper_authority": authority,
                    "episodes_window": len(recent_success),
                    "train_success_rate": (
                        float(np.mean(recent_success)) if recent_success else None
                    ),
                    "shaped_return": (
                        float(np.mean(recent_shaped_return)) if recent_shaped_return else None
                    ),
                    **last_logs,
                }
                print(json.dumps(row), flush=True)
                json_line(args.output_dir / "train_log.jsonl", row)
                next_log += args.log_interval

            if transition_step >= next_eval:
                evaluation = rollout_evaluate(
                    env,
                    runner,
                    device,
                    episodes=args.eval_episodes,
                    agent=agent,
                    gripper_authority=authority,
                    flip_margin=args.flip_margin,
                    reset_seed=args.eval_seed,
                )
                evaluation.pop("videos", None)
                evaluation.pop("results", None)
                row = {"transitions": transition_step, "gripper_authority": authority, **evaluation}
                print(json.dumps({"stage": "eval", **row}), flush=True)
                json_line(args.output_dir / "eval_log.jsonl", row)
                if evaluation["success_rate"] > best_success:
                    best_success = evaluation["success_rate"]
                    agent.save(
                        best_checkpoint,
                        args.bc_checkpoint,
                        config,
                        args.task,
                        {
                            "transitions": transition_step,
                            "eval_success_rate": best_success,
                            "gripper_authority": authority,
                        },
                    )
                    print(
                        f"new best deterministic success {best_success:.3f} -> {best_checkpoint}",
                        flush=True,
                    )
                next_eval += args.eval_interval

                # evaluation restarts every environment: drop partial episodes
                recorder.clear()
                episode_shaped_return.zero_()
                correlated_noise.zero_()
                previous_success.zero_()
                # Evaluation deliberately reuses one fixed seed.  Resume
                # training from a different deterministic seed so checkpoint
                # evaluation does not repeatedly inject the same starts into
                # the training distribution.
                observations, _ = env.reset(seed=args.seed + transition_step)
                runner.mark_reset()
                rgb, proprio = policy_observation(observations, device)
                oracle = oracle_observation(observations, device)
                base_action, feature = runner.step(rgb, proprio)

            if transition_step >= next_checkpoint:
                path = args.output_dir / "checkpoints" / f"residual_{transition_step:09d}.pt"
                agent.save(
                    path,
                    args.bc_checkpoint,
                    config,
                    args.task,
                    {"transitions": transition_step, "gripper_authority": authority},
                )
                print(f"saved checkpoint: {path}", flush=True)
                next_checkpoint += args.checkpoint_interval
    finally:
        env.close()

    final_path = args.output_dir / "checkpoints" / "residual_final.pt"
    agent.save(
        final_path,
        args.bc_checkpoint,
        config,
        args.task,
        {"transitions": transition_step, "gripper_authority": gripper_authority_at(transition_step)},
    )
    summary = {
        "task": args.task,
        "transitions": transition_step,
        "updates": agent.total_updates,
        "best_eval_success_rate": best_success,
        "best_checkpoint": str(best_checkpoint.resolve()) if best_success >= 0 else None,
        "final_checkpoint": str(final_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return final_path


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
