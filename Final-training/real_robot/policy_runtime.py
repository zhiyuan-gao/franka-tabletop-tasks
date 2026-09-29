"""Checkpoint-faithful single-task or multitask BC+residual inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

from observation import policy_rgb_tensor


TASK = "drawer_opening"


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: str | Path, expected_sha256: str | None) -> tuple[Path, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = sha256_file(resolved)
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {resolved}: expected {expected_sha256}, got {actual}"
        )
    return resolved, actual


@dataclass(frozen=True)
class RuntimeDescription:
    profile: str
    task: str
    policy_kind: str
    task_index: int | None
    bc_checkpoint: str
    bc_sha256: str
    residual_checkpoint: str
    residual_sha256: str
    residual_label: str
    gripper_authority: float
    flip_margin: float
    frame_stack: int
    frame_stride: int
    chunk: int
    image_size: int


class ResidualPolicyRuntime:
    """Stateful temporal BC runner plus a frozen residual actor."""

    def __init__(
        self,
        *,
        profile: str,
        scripts_dir: str | Path,
        bc_checkpoint: str | Path,
        residual_checkpoint: str | Path,
        device: str,
        bc_sha256: str | None,
        residual_sha256: str | None,
        residual_label: str,
        gripper_authority: float,
        flip_margin: float,
        ensemble_decay: float = 0.35,
        use_ensemble: bool = True,
    ) -> None:
        scripts = Path(scripts_dir).expanduser().resolve()
        if not (scripts / "task_pipeline_v2.py").is_file():
            raise FileNotFoundError(f"training scripts not found at {scripts}")
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))

        from task_pipeline_v2 import (
            ChunkRunner,
            compose_action_v2,
            load_bc_checkpoint_v2,
            load_residual_actor_v2,
        )
        from task_pipeline_multitask_v2 import (
            MULTITASK_BC_FORMAT,
            BoundTaskPolicy,
            load_multitask_bc_checkpoint,
        )

        self.profile = str(profile)
        self.task = TASK
        self.device = torch.device(device)
        self._compose_action = compose_action_v2
        self.gripper_authority = float(gripper_authority)
        self.flip_margin = float(flip_margin)
        if self.gripper_authority != 0.0:
            raise ValueError(
                "drawer_opening deployment must keep residual gripper authority at 0.0"
            )

        bc_path, bc_hash = verify_file(bc_checkpoint, bc_sha256)
        residual_path, residual_hash = verify_file(residual_checkpoint, residual_sha256)
        bc_header = torch.load(bc_path, map_location="cpu", weights_only=False, mmap=True)
        if bc_header.get("format") == MULTITASK_BC_FORMAT:
            policy_kind = "multitask"
            tasks = list(bc_header.get("tasks", ()))
            if self.task not in tasks:
                raise ValueError(f"{self.task!r} is absent from multitask checkpoint")
            task_index = tasks.index(self.task)
            policy, normalizers, bc_payload = load_multitask_bc_checkpoint(
                bc_path, self.device, prefer_ema=True
            )
            policy = BoundTaskPolicy(policy, task_index).to(self.device).eval()
            normalizer = normalizers[self.task]
            residual_expected_task = None
        else:
            policy_kind = "single_task"
            task_index = None
            policy, normalizer, bc_payload = load_bc_checkpoint_v2(
                bc_path, self.device, expected_task=self.task, prefer_ema=True
            )
            residual_expected_task = self.task

        actor, residual_payload = load_residual_actor_v2(
            residual_path, self.device, expected_task=residual_expected_task
        )
        if int(residual_payload["feature_dim"]) != int(policy.feature_dim):
            raise ValueError(
                "BC feature dimension does not match residual actor: "
                f"{policy.feature_dim} vs {residual_payload['feature_dim']}"
            )
        referenced_bc = Path(residual_payload["bc_checkpoint"]).expanduser()
        if referenced_bc.is_file() and sha256_file(referenced_bc.resolve()) != bc_hash:
            raise ValueError("residual actor was trained against a different BC checkpoint")

        config = bc_payload["config"]
        self.runner = ChunkRunner(
            policy,
            normalizer,
            num_envs=1,
            device=self.device,
            frame_stack=int(config["frame_stack"]),
            frame_stride=int(config["frame_stride"]),
            chunk=int(config["chunk"]),
            image_size=int(config["image_size"]),
            ensemble_decay=float(ensemble_decay),
            use_ensemble=bool(use_ensemble),
        )
        self.actor = actor
        self.description = RuntimeDescription(
            profile=self.profile,
            task=self.task,
            policy_kind=policy_kind,
            task_index=task_index,
            bc_checkpoint=str(bc_path),
            bc_sha256=bc_hash,
            residual_checkpoint=str(residual_path),
            residual_sha256=residual_hash,
            residual_label=str(residual_label),
            gripper_authority=self.gripper_authority,
            flip_margin=self.flip_margin,
            frame_stack=int(config["frame_stack"]),
            frame_stride=int(config["frame_stride"]),
            chunk=int(config["chunk"]),
            image_size=int(config["image_size"]),
        )
        self.reset()

    def reset(self) -> None:
        self.runner.mark_reset()

    @torch.inference_mode()
    def infer(self, rgb: np.ndarray, proprio: np.ndarray) -> dict[str, Any]:
        proprio_array = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if proprio_array.shape != (25,) or not np.isfinite(proprio_array).all():
            raise ValueError("proprio must be a finite 25-vector")
        image = policy_rgb_tensor(rgb, self.device)
        proprio_tensor = torch.from_numpy(proprio_array.copy()).to(self.device).unsqueeze(0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        base_action, feature = self.runner.step(image, proprio_tensor)
        residual = self.actor(feature, base_action, proprio_tensor)
        final_action = self._compose_action(
            base_action,
            residual,
            self.gripper_authority,
            self.flip_margin,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0
        return {
            "base_action": base_action[0].detach().cpu().float().tolist(),
            "residual_action": residual[0].detach().cpu().float().tolist(),
            "final_action": final_action[0].detach().cpu().float().tolist(),
            "inference_ms": inference_ms,
        }

