#!/usr/bin/env python3
"""Evaluate Nut Pouring BC with arm-only temporal ensembling.

This is an isolated inference experiment.  It does not modify the general
ChunkRunner or eval_task_bc_v2.py.  Arm actions retain the original ACT
temporal ensemble.  The binary gripper uses only the newest chunk's t=0 logit
and requires a configurable number of consecutive close requests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=9101)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--video-dir", type=Path, required=True)
parser.add_argument(
    "--video-episodes",
    type=int,
    default=-1,
    help="Videos to save. -1 saves every episode; 0 disables video.",
)
parser.add_argument("--video-fps", type=float, default=20.0)
parser.add_argument("--ensemble-decay", type=float, default=0.35)
parser.add_argument("--close-confirm-steps", type=int, default=2)
parser.add_argument(
    "--close-margin",
    type=float,
    default=0.0,
    help="A close request requires the newest t=0 logit to be below -margin.",
)
parser.add_argument(
    "--open-margin",
    type=float,
    default=0.0,
    help="Opening is immediate when the newest t=0 logit is at least +margin.",
)
parser.add_argument("--raw-weights", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.episodes <= 0:
    raise ValueError("--episodes must be positive")
if args.close_confirm_steps <= 0:
    raise ValueError("--close-confirm-steps must be positive")
if args.close_margin < 0.0 or args.open_margin < 0.0:
    raise ValueError("gripper margins must be non-negative")
if args.video_episodes < -1:
    raise ValueError("--video-episodes must be -1, 0, or positive")
if args.video_episodes == -1:
    args.video_episodes = args.episodes
if args.video_episodes > 0 and args.num_envs != 1:
    print(
        "video recording enabled: forcing --num-envs 1 so every episode has a video",
        flush=True,
    )
    args.num_envs = 1

args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def main() -> dict:
    import math

    import gymnasium as gym
    import torch
    import torch.nn.functional as F

    from task_pipeline_v2 import (
        ARM_DIM,
        PROPRIO_DIM,
        ChunkRunner,
        encode_video,
        load_bc_checkpoint_v2,
        rollout_evaluate,
        set_seed,
    )
    from track_b_envs.rl import configure_normalized_action_space, load_registered_env_cfg

    class PouringGripperFixRunner(ChunkRunner):
        """Original arm ensemble plus newest-logit confirmed binary gripper."""

        def __init__(self, *runner_args, close_confirm_steps: int, close_margin: float,
                     open_margin: float, **runner_kwargs) -> None:
            super().__init__(*runner_args, **runner_kwargs)
            self.close_confirm_steps = int(close_confirm_steps)
            self.close_margin = float(close_margin)
            self.open_margin = float(open_margin)
            self.close_counter = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.gripper_state = torch.ones(self.num_envs, device=self.device)

        def mark_reset(self, done: torch.Tensor | None = None) -> None:
            super().mark_reset(done)
            if done is None:
                self.close_counter.zero_()
                self.gripper_state.fill_(1.0)
            else:
                mask = done.to(self.device).bool()
                self.close_counter[mask] = 0
                self.gripper_state[mask] = 1.0

        @torch.no_grad()
        def step(
            self, rgb: torch.Tensor, proprio: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # This observation path is intentionally identical to ChunkRunner.
            source_u8_domain = (rgb * 255.0).round().clamp_(0.0, 255.0)
            image = F.interpolate(
                source_u8_domain,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).round().clamp_(0.0, 255.0).div_(255.0)

            self.pointer = (self.pointer + 1) % self.history_length
            self.frames[:, self.pointer] = image
            self.proprio[:, self.pointer] = proprio
            if bool(self.pending_reset.any()):
                mask = self.pending_reset
                self.frames[mask] = image[mask].unsqueeze(1).expand(
                    -1, self.history_length, -1, -1, -1
                )
                self.proprio[mask] = proprio[mask].unsqueeze(1).expand(
                    -1, self.history_length, -1
                )
                self._clear_ensemble(mask)
                self.close_counter[mask] = 0
                self.gripper_state[mask] = 1.0
                self.pending_reset[:] = False

            offsets = [
                (self.pointer - k * self.frame_stride) % self.history_length
                for k in reversed(range(self.frame_stack))
            ]
            frames = self.frames[:, offsets]
            proprio_history = self.proprio[:, offsets]
            arm_chunk, gripper_chunk, feature = self.policy(frames, proprio_history)

            # Preserve the original temporal ensemble for arm positioning.
            decay = math.exp(-self.ensemble_decay)
            self.arm_accumulator = (
                torch.roll(self.arm_accumulator, shifts=-1, dims=1) * decay
            )
            self.weight_sum = torch.roll(self.weight_sum, shifts=-1, dims=1) * decay
            self.arm_accumulator[:, -1] = 0.0
            self.weight_sum[:, -1] = 0.0
            self.arm_accumulator += arm_chunk
            self.weight_sum += 1.0
            weight = self.weight_sum[:, 0].clamp(min=1.0e-6).unsqueeze(-1)
            arm_norm = self.arm_accumulator[:, 0] / weight

            # Gripper deliberately ignores predictions carried by older chunks.
            newest_logit = gripper_chunk[:, 0]
            close_request = newest_logit < -self.close_margin
            open_request = newest_logit >= self.open_margin
            self.close_counter = torch.where(
                close_request,
                self.close_counter + 1,
                torch.zeros_like(self.close_counter),
            )
            confirmed_close = self.close_counter >= self.close_confirm_steps
            self.gripper_state = torch.where(
                confirmed_close,
                -torch.ones_like(self.gripper_state),
                self.gripper_state,
            )
            # Release stays immediate; this avoids delaying the pour/release phase.
            self.gripper_state = torch.where(
                open_request,
                torch.ones_like(self.gripper_state),
                self.gripper_state,
            )

            arm = self.normalizer.denormalize_torch(arm_norm)
            action = torch.cat((arm, self.gripper_state.unsqueeze(-1)), dim=-1)
            if action.shape != (self.num_envs, ARM_DIM + 1):
                raise RuntimeError(f"unexpected action shape: {tuple(action.shape)}")
            if proprio.shape != (self.num_envs, PROPRIO_DIM):
                raise RuntimeError(f"unexpected proprio shape: {tuple(proprio.shape)}")
            return action, feature

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    set_seed(args.seed)
    device = torch.device(args.device)
    task_id, cfg = load_registered_env_cfg(
        "nut_pouring",
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    env = gym.make(task_id, cfg=cfg)
    configure_normalized_action_space(env)

    try:
        policy, normalizer, payload = load_bc_checkpoint_v2(
            args.checkpoint,
            device,
            expected_task="nut_pouring",
            prefer_ema=not args.raw_weights,
        )
        config = payload["config"]
        runner = PouringGripperFixRunner(
            policy,
            normalizer,
            num_envs=args.num_envs,
            device=device,
            frame_stack=int(config["frame_stack"]),
            frame_stride=int(config["frame_stride"]),
            chunk=int(config["chunk"]),
            image_size=int(config["image_size"]),
            ensemble_decay=args.ensemble_decay,
            use_ensemble=True,
            close_confirm_steps=args.close_confirm_steps,
            close_margin=args.close_margin,
            open_margin=args.open_margin,
        )

        def save_video(row, frames):
            outcome = "success" if row["success"] else "failure"
            path = args.video_dir / (
                f"nut_pouring_bcv2_gripperfix_episode_{row['episode']:03d}_{outcome}.mp4"
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
            capture_episodes=args.video_episodes,
            video_callback=save_video if args.video_episodes else None,
            reset_seed=args.seed,
        )
    finally:
        env.close()

    summary = {
        "policy": "nut_pouring_bc_v2_arm_ensemble_newest_gripper",
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": int(payload["epoch"]),
        "close_confirm_steps": args.close_confirm_steps,
        "close_margin": args.close_margin,
        "open_margin": args.open_margin,
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return summary


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
