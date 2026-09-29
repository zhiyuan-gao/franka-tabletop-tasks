#!/usr/bin/env python3
"""Train the v2 chunked BC policy for one Track B task.

Differences that matter versus the old single-frame BC:

* frame history (default 3 frames, stride 3) so the phase of the task is
  observable,
* action chunks (default 8) with an ACT-style temporal ensemble at rollout,
* per-dimension action normalization computed from the training split,
* random-shift and photometric augmentation,
* gripper loss weighted around the open/close transitions,
* EMA weights and periodic checkpoints so that the *deployment* checkpoint can
  be chosen by closed-loop success instead of validation loss.

No Isaac Sim is needed here. Use ``eval_task_bc_v2.py --checkpoint-dir`` afterwards
to pick the checkpoint by rollout success.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from task_pipeline_v2 import (
    ActionNormalizer,
    ChunkBCPolicy,
    ChunkStepDataset,
    EMA,
    TASK_NAMES,
    chunk_bc_loss,
    inspect_datasets,
    load_episode_refs,
    photometric_jitter,
    random_shift,
    resolve_dataset_dirs,
    save_bc_checkpoint_v2,
    set_seed,
    split_episode_refs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--dataset-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--gripper-transition-weight", type=float, default=4.0)
    parser.add_argument("--gripper-transition-decay", type=float, default=3.0)
    parser.add_argument("--horizon-decay", type=float, default=0.0)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--frame-stack", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--image-feature-dim", type=int, default=384)
    parser.add_argument("--proprio-feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--keypoints", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--shift-pad", type=int, default=6)
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--cache-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--amp", action="store_true", help="Enable bf16 autocast.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--skip-full-validation", action="store_true")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    return parser.parse_args()


def run_epoch(
    model,
    loader,
    device,
    args,
    optimizer=None,
    scheduler=None,
    ema=None,
    max_steps: int = 0,
) -> dict:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    rows = 0
    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=bool(args.amp)
    )
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for step, batch in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            frames = batch["frames"].to(device, non_blocking=True).float().div_(255.0)
            proprio = batch["proprio"].to(device, non_blocking=True)
            arm_target = batch["arm"].to(device, non_blocking=True)
            gripper_target = batch["gripper"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            gripper_weight = batch["gripper_weight"].to(device, non_blocking=True)
            if training:
                frames = random_shift(frames, args.shift_pad)
                frames = photometric_jitter(frames, args.jitter)
            with autocast:
                arm, gripper_logit, _ = model(frames, proprio)
                loss, metrics = chunk_bc_loss(
                    arm.float(),
                    gripper_logit.float(),
                    arm_target,
                    gripper_target,
                    mask,
                    gripper_weight,
                    args.gripper_loss_weight,
                    args.horizon_decay,
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)
            count = int(frames.shape[0])
            rows += count
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value * count
    return {key: value / max(rows, 1) for key, value in sums.items()}


def main() -> None:
    args = parse_args()
    args.dataset_dir = resolve_dataset_dirs(args.task, args.dataset_dir)
    if args.output_dir is None:
        args.output_dir = Path("outputs") / f"{args.task}_bcv2_seed{args.seed}"
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)

    if not args.skip_full_validation:
        inspect_datasets(args.dataset_dir, args.task)
    refs = load_episode_refs(args.dataset_dir, args.task)
    train_refs, val_refs = split_episode_refs(refs, args.validation_ratio, args.seed)
    normalizer = ActionNormalizer.from_refs(train_refs)
    print(
        json.dumps(
            {
                "stage": "normalizer",
                "low": normalizer.low.tolist(),
                "high": (normalizer.low + normalizer.span).tolist(),
            }
        ),
        flush=True,
    )

    dataset_kwargs = dict(
        normalizer=normalizer,
        frame_stack=args.frame_stack,
        frame_stride=args.frame_stride,
        chunk=args.chunk,
        image_size=args.image_size,
        gripper_transition_weight_scale=args.gripper_transition_weight,
        gripper_transition_decay=args.gripper_transition_decay,
        cache_size=args.cache_size,
    )
    train_dataset = ChunkStepDataset(train_refs, **dataset_kwargs)
    val_dataset = ChunkStepDataset(val_refs, **dataset_kwargs)
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )

    model = ChunkBCPolicy(
        frame_stack=args.frame_stack,
        chunk=args.chunk,
        image_feature_dim=args.image_feature_dim,
        proprio_feature_dim=args.proprio_feature_dim,
        hidden_dim=args.hidden_dim,
        keypoints=args.keypoints,
        dropout=args.dropout,
    ).to(device)
    ema = EMA(model, args.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = args.max_steps_per_epoch or max(len(train_loader), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * max(args.warmup_epochs, 0)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    config = {
        "task": args.task,
        "dataset_dirs": [str(path) for path in args.dataset_dir],
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gripper_loss_weight": args.gripper_loss_weight,
        "gripper_transition_weight": args.gripper_transition_weight,
        "gripper_transition_decay": args.gripper_transition_decay,
        "horizon_decay": args.horizon_decay,
        "validation_ratio": args.validation_ratio,
        "frame_stack": args.frame_stack,
        "frame_stride": args.frame_stride,
        "chunk": args.chunk,
        "image_size": args.image_size,
        "image_feature_dim": args.image_feature_dim,
        "proprio_feature_dim": args.proprio_feature_dim,
        "hidden_dim": args.hidden_dim,
        "keypoints": args.keypoints,
        "dropout": args.dropout,
        "shift_pad": args.shift_pad,
        "jitter": args.jitter,
        "ema_decay": args.ema_decay,
        "train_episodes": len(train_refs),
        "validation_episodes": len(val_refs),
        "train_steps": len(train_dataset),
        "validation_steps": len(val_dataset),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    history = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args,
            optimizer,
            scheduler,
            ema,
            args.max_steps_per_epoch,
        )
        val_metrics = run_epoch(model, val_loader, device, args)
        ema_metrics = run_epoch(ema.shadow, val_loader, device, args)
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation": val_metrics,
            "validation_ema": ema_metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

        save_bc_checkpoint_v2(
            args.output_dir / "bc_last.pt",
            model,
            ema.shadow,
            optimizer,
            epoch,
            row,
            config,
            normalizer,
            args.task,
        )
        if ema_metrics["loss"] < best_loss:
            best_loss = ema_metrics["loss"]
            save_bc_checkpoint_v2(
                args.output_dir / "bc_best_loss.pt",
                model,
                ema.shadow,
                optimizer,
                epoch,
                row,
                config,
                normalizer,
                args.task,
            )
        if args.checkpoint_every > 0 and (
            epoch % args.checkpoint_every == 0 or epoch == args.epochs
        ):
            save_bc_checkpoint_v2(
                args.output_dir / "checkpoints" / f"bc_epoch_{epoch:04d}.pt",
                model,
                ema.shadow,
                optimizer,
                epoch,
                row,
                config,
                normalizer,
                args.task,
            )

    print(
        json.dumps(
            {
                "stage": "complete",
                "best_validation_loss_ema": best_loss,
                "checkpoint_dir": str((args.output_dir / "checkpoints").resolve()),
                "note": (
                    "validation loss does not predict rollout success -- run "
                    "eval_task_bc_v2.py --checkpoint-dir over the epoch "
                    "checkpoints to pick the deployment policy"
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
