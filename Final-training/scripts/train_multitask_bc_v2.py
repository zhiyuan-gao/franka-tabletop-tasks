#!/usr/bin/env python3
"""Train one task-conditioned BC policy across the four selected Track B tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from task_pipeline_multitask_v2 import (
    MULTITASK_TASKS,
    TASK_TO_ID,
    MultiTaskChunkBCPolicy,
    TaskTaggedDataset,
    equal_task_sample_weights,
    save_multitask_bc_checkpoint,
)
from task_pipeline_v2 import (
    ActionNormalizer,
    ChunkStepDataset,
    EMA,
    chunk_bc_loss,
    inspect_datasets,
    load_episode_refs,
    photometric_jitter,
    random_shift,
    set_seed,
    split_episode_refs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="TASK=PATH",
        help="Repeat for every dataset directory. Multiple paths per task are allowed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples-per-epoch", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--gripper-transition-weight", type=float, default=6.0)
    parser.add_argument("--gripper-transition-decay", type=float, default=3.0)
    parser.add_argument("--horizon-decay", type=float, default=0.0)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--frame-stack", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--image-feature-dim", type=int, default=512)
    parser.add_argument("--proprio-feature-dim", type=int, default=128)
    parser.add_argument("--task-embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--keypoints", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--shift-pad", type=int, default=6)
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--cache-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--skip-full-validation", action="store_true")
    return parser.parse_args()


def parse_dataset_map(specs: list[str]) -> dict[str, list[Path]]:
    result = {task: [] for task in MULTITASK_TASKS}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid --dataset {spec!r}; expected TASK=PATH")
        task, raw_path = spec.split("=", 1)
        if task not in result:
            raise ValueError(f"unsupported multitask dataset task {task!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        result[task].append(path)
    missing = [task for task, paths in result.items() if not paths]
    if missing:
        raise ValueError(f"missing datasets for tasks: {missing}")
    return result


def run_epoch(model, loader, device, args, optimizer=None, scheduler=None, ema=None):
    training = optimizer is not None
    model.train(training)
    overall: dict[str, float] = {}
    per_task: dict[str, dict[str, float]] = {task: {} for task in MULTITASK_TASKS}
    per_task_rows = {task: 0 for task in MULTITASK_TASKS}
    rows = 0
    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=bool(args.amp)
    )
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True).float().div_(255.0)
            proprio = batch["proprio"].to(device, non_blocking=True)
            task_id = batch["task_id"].to(device, non_blocking=True)
            arm_target = batch["arm"].to(device, non_blocking=True)
            gripper_target = batch["gripper"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            gripper_weight = batch["gripper_weight"].to(device, non_blocking=True)
            if training:
                frames = photometric_jitter(
                    random_shift(frames, args.shift_pad), args.jitter
                )
            with autocast:
                arm, gripper_logit, _ = model(frames, proprio, task_id)
                losses = []
                group_metrics = []
                for index, task in enumerate(MULTITASK_TASKS):
                    select = task_id == index
                    count = int(select.sum().item())
                    if count == 0:
                        continue
                    loss, metrics = chunk_bc_loss(
                        arm[select].float(),
                        gripper_logit[select].float(),
                        arm_target[select],
                        gripper_target[select],
                        mask[select],
                        gripper_weight[select],
                        args.gripper_loss_weight,
                        args.horizon_decay,
                    )
                    losses.append(loss * (count / frames.shape[0]))
                    group_metrics.append((task, count, metrics))
                if not losses:
                    raise RuntimeError("empty multitask batch")
                total_loss = torch.stack(losses).sum()
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)

            batch_rows = int(frames.shape[0])
            rows += batch_rows
            for task, count, metrics in group_metrics:
                per_task_rows[task] += count
                for key, value in metrics.items():
                    per_task[task][key] = per_task[task].get(key, 0.0) + value * count
                    overall[key] = overall.get(key, 0.0) + value * count

    result = {key: value / max(rows, 1) for key, value in overall.items()}
    result["per_task"] = {
        task: {
            key: value / max(per_task_rows[task], 1)
            for key, value in values.items()
        }
        for task, values in per_task.items()
    }
    result["samples"] = rows
    return result


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.samples_per_epoch <= 0:
        raise ValueError("epochs, batch-size and samples-per-epoch must be positive")
    dataset_map = parse_dataset_map(args.dataset)
    set_seed(args.seed)
    device = torch.device(args.device)

    train_sets = []
    val_sets = {}
    normalizers = {}
    dataset_summary = {}
    for task in MULTITASK_TASKS:
        paths = dataset_map[task]
        if not args.skip_full_validation:
            inspect_datasets(paths, task)
        refs = load_episode_refs(paths, task)
        train_refs, val_refs = split_episode_refs(refs, args.validation_ratio, args.seed)
        normalizer = ActionNormalizer.from_refs(train_refs)
        normalizers[task] = normalizer
        kwargs = dict(
            normalizer=normalizer,
            frame_stack=args.frame_stack,
            frame_stride=args.frame_stride,
            chunk=args.chunk,
            image_size=args.image_size,
            gripper_transition_weight_scale=args.gripper_transition_weight,
            gripper_transition_decay=args.gripper_transition_decay,
            cache_size=args.cache_size,
        )
        train = TaskTaggedDataset(
            ChunkStepDataset(train_refs, **kwargs), TASK_TO_ID[task]
        )
        validation = TaskTaggedDataset(
            ChunkStepDataset(val_refs, **kwargs), TASK_TO_ID[task]
        )
        train_sets.append(train)
        val_sets[task] = validation
        dataset_summary[task] = {
            "directories": [str(path) for path in paths],
            "train_episodes": len(train_refs),
            "validation_episodes": len(val_refs),
            "train_steps": len(train),
            "validation_steps": len(validation),
            "normalizer": normalizer.to_dict(),
        }

    combined = ConcatDataset(train_sets)
    weights = equal_task_sample_weights(train_sets)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=generator,
    )
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loaders = {
        task: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for task, dataset in val_sets.items()
    }

    model = MultiTaskChunkBCPolicy(
        frame_stack=args.frame_stack,
        chunk=args.chunk,
        image_feature_dim=args.image_feature_dim,
        proprio_feature_dim=args.proprio_feature_dim,
        task_embedding_dim=args.task_embedding_dim,
        hidden_dim=args.hidden_dim,
        keypoints=args.keypoints,
        dropout=args.dropout,
    ).to(device)
    ema = EMA(model, args.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * max(args.warmup_epochs, 0)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    config = {
        "tasks": list(MULTITASK_TASKS),
        "datasets": dataset_summary,
        "seed": args.seed,
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
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
        "task_embedding_dim": args.task_embedding_dim,
        "hidden_dim": args.hidden_dim,
        "keypoints": args.keypoints,
        "dropout": args.dropout,
        "shift_pad": args.shift_pad,
        "jitter": args.jitter,
        "ema_decay": args.ema_decay,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    history = []
    best_mean_validation = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, args, optimizer, scheduler, ema
        )
        validation = {
            task: run_epoch(model, loader, device, args)
            for task, loader in val_loaders.items()
        }
        validation_ema = {
            task: run_epoch(ema.shadow, loader, device, args)
            for task, loader in val_loaders.items()
        }
        mean_validation_ema = sum(
            metrics["loss"] for metrics in validation_ema.values()
        ) / len(validation_ema)
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation": validation,
            "validation_ema": validation_ema,
            "mean_validation_loss_ema": mean_validation_ema,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

        save_multitask_bc_checkpoint(
            args.output_dir / "bc_multitask_last.pt",
            model, ema.shadow, optimizer, epoch, row, config, normalizers,
        )
        if mean_validation_ema < best_mean_validation:
            best_mean_validation = mean_validation_ema
            save_multitask_bc_checkpoint(
                args.output_dir / "bc_multitask_best_loss.pt",
                model, ema.shadow, optimizer, epoch, row, config, normalizers,
            )
        if args.checkpoint_every > 0 and (
            epoch % args.checkpoint_every == 0 or epoch == args.epochs
        ):
            save_multitask_bc_checkpoint(
                args.output_dir / "checkpoints" / f"bc_multitask_epoch_{epoch:04d}.pt",
                model, ema.shadow, optimizer, epoch, row, config, normalizers,
            )

    print(json.dumps({
        "stage": "complete",
        "best_mean_validation_loss_ema": best_mean_validation,
        "checkpoint_dir": str((args.output_dir / "checkpoints").resolve()),
        "note": "Select the deployment checkpoint by mean closed-loop success across all four tasks.",
    }, indent=2))


if __name__ == "__main__":
    main()
