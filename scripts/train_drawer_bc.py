#!/usr/bin/env python3
"""Train the deployable RGB+proprio drawer-opening BC policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from drawer_pipeline_common import (
    DrawerStepDataset,
    VisualBCPolicy,
    bc_loss,
    inspect_datasets,
    load_episode_refs,
    save_bc_checkpoint,
    set_seed,
    split_episode_refs,
)


def run_epoch(model, loader, device, gripper_weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "arm_loss": 0.0, "gripper_loss": 0.0, "gripper_accuracy": 0.0}
    rows = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            proprio = batch["proprio"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            arm, gripper_logit, _ = model(rgb, proprio)
            loss, metrics = bc_loss(arm, gripper_logit, action, gripper_weight)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            count = int(rgb.shape[0])
            rows += count
            for key in sums:
                sums[key] += metrics[key] * count
    return {key: value / max(rows, 1) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/drawer_bc"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gripper-loss-weight", type=float, default=0.25)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--image-feature-dim", type=int, default=256)
    parser.add_argument("--proprio-feature-dim", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-full-validation", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    if not args.skip_full_validation:
        inspect_datasets(args.dataset_dir)
    refs = load_episode_refs(args.dataset_dir)
    train_refs, val_refs = split_episode_refs(refs, args.validation_ratio, args.seed)
    train_dataset = DrawerStepDataset(train_refs)
    val_dataset = DrawerStepDataset(val_refs)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = VisualBCPolicy(
        image_feature_dim=args.image_feature_dim,
        proprio_feature_dim=args.proprio_feature_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    config = {
        "task": "drawer_opening",
        "dataset_dirs": [str(path.expanduser().resolve()) for path in args.dataset_dir],
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gripper_loss_weight": args.gripper_loss_weight,
        "validation_ratio": args.validation_ratio,
        "image_feature_dim": args.image_feature_dim,
        "proprio_feature_dim": args.proprio_feature_dim,
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
            model, train_loader, device, args.gripper_loss_weight, optimizer
        )
        val_metrics = run_epoch(model, val_loader, device, args.gripper_loss_weight)
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        save_bc_checkpoint(
            args.output_dir / "bc_last.pt",
            model,
            optimizer,
            epoch,
            val_metrics["loss"],
            config,
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_bc_checkpoint(
                args.output_dir / "bc_best.pt",
                model,
                optimizer,
                epoch,
                best_loss,
                config,
            )
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(
        json.dumps(
            {
                "stage": "complete",
                "best_validation_loss": best_loss,
                "checkpoint": str((args.output_dir / "bc_best.pt").resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
