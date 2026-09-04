"""Task-conditioned four-task BC components built on the existing v2 pipeline.

The single-task implementation is not modified.  This module adds a shared
visual/proprioceptive backbone, an explicit learned task embedding, and small
task-specific arm/gripper heads.  Each task keeps its own action normalizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import Dataset

from task_pipeline_v2 import (
    ARM_DIM,
    PROPRIO_DIM,
    ActionNormalizer,
    ChunkStepDataset,
    VisionEncoder,
)


MULTITASK_BC_FORMAT = "track_b_multitask_chunk_bc_v1"
MULTITASK_TASKS = (
    "push_cube_to_region",
    "drawer_opening",
    "object_to_drawer",
    "nut_pouring",
)
TASK_TO_ID = {task: index for index, task in enumerate(MULTITASK_TASKS)}


class TaskTaggedDataset(Dataset):
    """Add a task id to samples produced by the unchanged ChunkStepDataset."""

    def __init__(self, dataset: ChunkStepDataset, task_id: int) -> None:
        self.dataset = dataset
        self.task_id = int(task_id)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[item]
        sample["task_id"] = torch.tensor(self.task_id, dtype=torch.long)
        return sample


class MultiTaskChunkBCPolicy(nn.Module):
    """One shared policy with explicit task conditioning and per-task heads."""

    def __init__(
        self,
        num_tasks: int = len(MULTITASK_TASKS),
        frame_stack: int = 3,
        chunk: int = 8,
        image_feature_dim: int = 512,
        proprio_feature_dim: int = 128,
        task_embedding_dim: int = 64,
        hidden_dim: int = 768,
        keypoints: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.frame_stack = int(frame_stack)
        self.chunk = int(chunk)
        self.hidden_dim = int(hidden_dim)
        self.encoder = VisionEncoder(3 * self.frame_stack, image_feature_dim, keypoints)
        self.proprio_encoder = nn.Sequential(
            nn.LayerNorm(PROPRIO_DIM * self.frame_stack),
            nn.Linear(PROPRIO_DIM * self.frame_stack, proprio_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proprio_feature_dim, proprio_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.task_embedding = nn.Embedding(self.num_tasks, task_embedding_dim)
        self.trunk = nn.Sequential(
            nn.Linear(
                image_feature_dim + proprio_feature_dim + task_embedding_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Separate small output heads prevent conflicting gripper schedules and
        # action ranges from forcing compromise in the shared representation.
        self.arm_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, self.chunk * ARM_DIM) for _ in range(self.num_tasks)]
        )
        self.gripper_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, self.chunk) for _ in range(self.num_tasks)]
        )
        for head in list(self.arm_heads) + list(self.gripper_heads):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        self.feature_dim = self.hidden_dim

    def forward(
        self,
        frames: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = frames.shape[0]
        task_id = task_id.to(device=frames.device, dtype=torch.long).reshape(batch)
        if bool(((task_id < 0) | (task_id >= self.num_tasks)).any()):
            raise ValueError("task id outside configured multitask policy range")
        image = frames.reshape(batch, -1, frames.shape[-2], frames.shape[-1])
        image_features = self.encoder(image)
        proprio_features = self.proprio_encoder(proprio.reshape(batch, -1))
        task_features = self.task_embedding(task_id)
        feature = self.trunk(
            torch.cat((image_features, proprio_features, task_features), dim=-1)
        )

        arm_all = torch.stack(
            [torch.tanh(head(feature)).view(batch, self.chunk, ARM_DIM)
             for head in self.arm_heads],
            dim=1,
        )
        gripper_all = torch.stack(
            [head(feature).view(batch, self.chunk) for head in self.gripper_heads],
            dim=1,
        )
        row = torch.arange(batch, device=frames.device)
        return arm_all[row, task_id], gripper_all[row, task_id], feature


class BoundTaskPolicy(nn.Module):
    """Bind one known task id so the original ChunkRunner can be reused exactly."""

    def __init__(self, policy: MultiTaskChunkBCPolicy, task_id: int) -> None:
        super().__init__()
        self.policy = policy
        self.task_id = int(task_id)

    def forward(
        self, frames: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = torch.full(
            (frames.shape[0],), self.task_id, dtype=torch.long, device=frames.device
        )
        return self.policy(frames, proprio, ids)


def save_multitask_bc_checkpoint(
    path: Path,
    model: MultiTaskChunkBCPolicy,
    ema_model: nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict,
    config: dict,
    normalizers: dict[str, ActionNormalizer],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": MULTITASK_BC_FORMAT,
            "tasks": list(MULTITASK_TASKS),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": None if ema_model is None else ema_model.state_dict(),
            "optimizer_state_dict": (
                None if optimizer is None else optimizer.state_dict()
            ),
            "epoch": int(epoch),
            "metrics": metrics,
            "config": config,
            "normalizers": {
                task: normalizers[task].to_dict() for task in MULTITASK_TASKS
            },
        },
        path,
    )


def load_multitask_bc_checkpoint(
    path: Path,
    device: torch.device | str,
    prefer_ema: bool = True,
) -> tuple[MultiTaskChunkBCPolicy, dict[str, ActionNormalizer], dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != MULTITASK_BC_FORMAT:
        raise ValueError(
            f"{path} is not a {MULTITASK_BC_FORMAT} checkpoint "
            f"(got {payload.get('format')!r})"
        )
    if tuple(payload.get("tasks", ())) != MULTITASK_TASKS:
        raise ValueError("checkpoint task ordering does not match this pipeline")
    config = payload["config"]
    model = MultiTaskChunkBCPolicy(
        num_tasks=len(MULTITASK_TASKS),
        frame_stack=int(config["frame_stack"]),
        chunk=int(config["chunk"]),
        image_feature_dim=int(config["image_feature_dim"]),
        proprio_feature_dim=int(config["proprio_feature_dim"]),
        task_embedding_dim=int(config["task_embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        keypoints=int(config["keypoints"]),
        dropout=float(config["dropout"]),
    ).to(device)
    state = payload["model_state_dict"]
    if prefer_ema and payload.get("ema_state_dict") is not None:
        state = payload["ema_state_dict"]
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    normalizers = {
        task: ActionNormalizer.from_dict(payload["normalizers"][task])
        for task in MULTITASK_TASKS
    }
    return model, normalizers, payload


def equal_task_sample_weights(datasets: Sequence[Dataset]) -> torch.Tensor:
    """Per-sample weights yielding an equal expected fraction for every task."""

    pieces = []
    for dataset in datasets:
        if len(dataset) <= 0:
            raise ValueError("multitask dataset cannot be empty")
        pieces.append(torch.full((len(dataset),), 1.0 / len(dataset), dtype=torch.double))
    return torch.cat(pieces)
