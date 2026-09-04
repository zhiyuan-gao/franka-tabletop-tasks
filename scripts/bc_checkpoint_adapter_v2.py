#!/usr/bin/env python3
"""Load either a single-task or task-bound multitask BC checkpoint."""

from pathlib import Path

import torch


def load_bc_checkpoint_for_task(
    checkpoint: Path,
    device: torch.device | str,
    task: str,
    prefer_ema: bool = True,
):
    from task_pipeline_multitask_v2 import (
        MULTITASK_BC_FORMAT,
        TASK_TO_ID,
        BoundTaskPolicy,
        load_multitask_bc_checkpoint,
    )
    from task_pipeline_v2 import load_bc_checkpoint_v2

    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != MULTITASK_BC_FORMAT:
        return load_bc_checkpoint_v2(
            checkpoint,
            device,
            expected_task=task,
            prefer_ema=prefer_ema,
        )

    if task not in TASK_TO_ID:
        raise ValueError(f"task {task!r} is not represented by the multitask BC")
    policy, normalizers, payload = load_multitask_bc_checkpoint(
        checkpoint,
        device,
        prefer_ema=prefer_ema,
    )
    bound = BoundTaskPolicy(policy, TASK_TO_ID[task]).to(device)
    bound.eval()
    return bound, normalizers[task], payload

