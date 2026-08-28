"""Train a Track B single-view visual policy with SKRL PPO."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
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
parser.add_argument("--config", type=Path, default=None)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--timesteps", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=Path, default=None, help="Resume a complete SKRL agent checkpoint.")
parser.add_argument(
    "--actor-checkpoint",
    type=Path,
    default=None,
    help="Initialize only the visual actor, for example from behavior cloning.",
)
parser.add_argument("--log-dir", type=Path, default=Path("outputs/rl/skrl"))
parser.add_argument(
    "--smoke",
    action="store_true",
    help="Run one tiny PPO update to validate the complete training stack.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _load_actor_checkpoint(policy, path: Path) -> None:
    import torch

    payload = torch.load(path, map_location=policy.device, weights_only=False)
    if isinstance(payload, dict) and "policy" in payload:
        payload = payload["policy"]
    if not isinstance(payload, dict):
        raise TypeError("actor checkpoint must contain a PyTorch state dictionary")
    missing, unexpected = policy.load_state_dict(payload, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected actor checkpoint keys: {unexpected}")
    print(
        json.dumps(
            {
                "stage": "actor_initialized",
                "checkpoint": str(path),
                "missing_keys": list(missing),
            }
        ),
        flush=True,
    )


def main() -> Path:
    import gymnasium as gym
    import yaml
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.agents.torch.ppo import PPO
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed

    from track_b_envs.agents.skrl import (
        OracleValue,
        VisualGaussianPolicy,
        build_ppo_config,
        load_visual_ppo_config,
    )
    from track_b_envs.rl import (
        configure_normalized_action_space,
        load_registered_env_cfg,
    )

    raw_cfg = load_visual_ppo_config(args.config)
    seed = args.seed if args.seed is not None else int(raw_cfg.get("seed", 20260827))
    num_envs = args.num_envs or int(raw_cfg["environment"]["num_envs"])
    timesteps = args.timesteps or int(raw_cfg["trainer"]["timesteps"])
    if args.smoke:
        num_envs = args.num_envs or 2
        timesteps = args.timesteps or 4
        raw_cfg["agent"].update(
            {"rollouts": 2, "learning_epochs": 1, "mini_batches": 1}
        )
        raw_cfg["experiment"].update(
            {"write_interval": 0, "checkpoint_interval": 0}
        )
        raw_cfg["trainer"]["disable_progressbar"] = True
    if num_envs < 2:
        raise ValueError("SKRL SequentialTrainer requires at least two parallel environments")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    for checkpoint in (args.checkpoint, args.actor_checkpoint):
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    set_seed(seed)
    task_id, env_cfg = load_registered_env_cfg(
        args.task,
        device=args.device,
        num_envs=num_envs,
        seed=seed,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    env = gym.make(task_id, cfg=env_cfg)
    configure_normalized_action_space(env)
    wrapped_env = SkrlVecEnvWrapper(env, ml_framework="torch")

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
    value = OracleValue(
        wrapped_env.state_space,
        wrapped_env.device,
        hidden_dims=tuple(model_cfg["critic_hidden_dims"]),
    )
    if args.actor_checkpoint is not None:
        _load_actor_checkpoint(policy, args.actor_checkpoint)

    rollouts = int(raw_cfg["agent"]["rollouts"])
    memory = RandomMemory(
        memory_size=rollouts,
        num_envs=wrapped_env.num_envs,
        device=wrapped_env.device,
    )
    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.task}"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    ppo_cfg = build_ppo_config(
        raw_cfg,
        state_space=wrapped_env.state_space,
        device=wrapped_env.device,
        experiment_dir=args.log_dir.resolve(),
        experiment_name=run_name,
    )
    agent = PPO(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=wrapped_env.observation_space,
        state_space=wrapped_env.state_space,
        action_space=wrapped_env.action_space,
        device=wrapped_env.device,
        cfg=ppo_cfg,
    )
    if args.checkpoint is not None:
        agent.load(str(args.checkpoint))

    trainer_cfg = dict(raw_cfg["trainer"])
    trainer_cfg.update(
        {
            "timesteps": timesteps,
            "headless": bool(args.headless),
            "close_environment_at_exit": False,
        }
    )
    trainer = SequentialTrainer(env=wrapped_env, agents=agent, cfg=trainer_cfg)
    run_dir = Path(agent.experiment_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": args.task,
        "gym_id": task_id,
        "seed": seed,
        "num_envs": num_envs,
        "timesteps": timesteps,
        "actor_inputs": {"rgb": [3, 224, 224], "proprio": [25]},
        "critic_input": list(wrapped_env.state_space.shape),
        "action": [7],
        "config": raw_cfg,
    }
    with (run_dir / "run_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    print(
        json.dumps(
            {
                "stage": "training",
                "task": args.task,
                "num_envs": num_envs,
                "timesteps": timesteps,
                "run_dir": str(run_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        trainer.train()
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_checkpoint = checkpoint_dir / "agent_final.pt"
        agent.save(str(final_checkpoint))
        print(
            json.dumps(
                {"stage": "complete", "checkpoint": str(final_checkpoint)},
                indent=2,
            ),
            flush=True,
        )
        return final_checkpoint
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
