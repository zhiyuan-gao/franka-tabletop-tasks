"""Portable episode format shared by scripted experts and future teleoperation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


DATASET_FORMAT = "track_b_successful_demonstrations_v1"


def _numpy(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError("trajectory value contains NaN or infinity")
    return array


def policy_rgb_to_uint8_hwc(value: Any) -> np.ndarray:
    """Convert one CHW/HWC float or uint8 policy frame to uint8 HWC."""

    image = _numpy(value)
    if image.ndim != 3:
        raise ValueError(f"expected one 3D RGB frame, got {image.shape}")
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"expected RGB channels, got {image.shape}")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    return np.ascontiguousarray(
        np.rint(np.clip(image.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    )


@dataclass
class EpisodeBuffer:
    """In-memory attempt buffer; only successful attempts are persisted."""

    policy_rgb: list[np.ndarray] = field(default_factory=list)
    policy_proprio: list[np.ndarray] = field(default_factory=list)
    oracle_state: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    success: list[bool] = field(default_factory=list)
    timestamp_s: list[float] = field(default_factory=list)

    def append(
        self,
        *,
        policy_rgb: Any,
        policy_proprio: Any,
        oracle_state: Any,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        success: bool,
        timestamp_s: float,
    ) -> None:
        self.policy_rgb.append(policy_rgb_to_uint8_hwc(policy_rgb))
        self.policy_proprio.append(_numpy(policy_proprio, np.float32).reshape(-1))
        self.oracle_state.append(_numpy(oracle_state, np.float32).reshape(-1))
        action_array = _numpy(action, np.float32).reshape(-1)
        if np.max(np.abs(action_array)) > 1.0 + 1.0e-6:
            raise ValueError("normalized action leaves [-1, 1]")
        self.action.append(action_array)
        self.reward.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.success.append(bool(success))
        self.timestamp_s.append(float(timestamp_s))

    def __len__(self) -> int:
        return len(self.action)

    def arrays(self, *, require_success: bool = True) -> dict[str, np.ndarray]:
        if not self.action:
            raise ValueError("cannot save an empty episode")
        arrays = {
            "observation.rgb": np.stack(self.policy_rgb).astype(np.uint8, copy=False),
            "observation.proprio": np.stack(self.policy_proprio).astype(np.float32, copy=False),
            "observation.oracle": np.stack(self.oracle_state).astype(np.float32, copy=False),
            "action": np.stack(self.action).astype(np.float32, copy=False),
            "reward": np.asarray(self.reward, dtype=np.float32),
            "terminated": np.asarray(self.terminated, dtype=np.bool_),
            "truncated": np.asarray(self.truncated, dtype=np.bool_),
            "success": np.asarray(self.success, dtype=np.bool_),
            "timestamp_s": np.asarray(self.timestamp_s, dtype=np.float64),
        }
        lengths = {len(value) for value in arrays.values()}
        if lengths != {len(self)}:
            raise RuntimeError(f"misaligned episode arrays: {sorted(lengths)}")
        if require_success and not arrays["success"][-1]:
            raise ValueError("successful dataset episode must end with success=True")
        if np.any(np.diff(arrays["timestamp_s"]) <= 0.0):
            raise ValueError("episode timestamps must be strictly increasing")
        return arrays


class SuccessfulDatasetWriter:
    """Write one compressed NPZ per success and an incrementally updated manifest."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        task_name: str,
        task_id: str,
        requested_episodes: int,
        control_dt_s: float,
        dataset_metadata: dict[str, Any] | None = None,
    ) -> None:
        if requested_episodes <= 0 or control_dt_s <= 0.0:
            raise ValueError("requested_episodes and control_dt_s must be positive")
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite dataset: {self.output_dir}")
        self.output_dir.mkdir(parents=True)
        self.episodes_dir = self.output_dir / "episodes"
        self.episodes_dir.mkdir()
        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest: dict[str, Any] = {
            "format": DATASET_FORMAT,
            "task_name": task_name,
            "task_id": task_id,
            "requested_episodes": int(requested_episodes),
            "successful_episodes": 0,
            "attempts": 0,
            "control_dt_s": float(control_dt_s),
            "observation": {
                "policy": ["rgb", "proprio"],
                "privileged": ["oracle"],
            },
            "action": (
                "normalized 6D relative EE delta axis-angle + binary gripper "
                "(+1 open, -1 closed)"
            ),
            "episodes": [],
            "attempt_log": [],
            **(dataset_metadata or {}),
        }
        self._write_manifest()

    @property
    def successful_episodes(self) -> int:
        return int(self.manifest["successful_episodes"])

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2) + "\n")
        temporary.replace(self.manifest_path)

    def record_attempt(
        self,
        *,
        success: bool,
        seed: int,
        source_env_id: int,
        steps: int,
        reason: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.manifest["attempts"] = int(self.manifest["attempts"]) + 1
        self.manifest["attempt_log"].append(
            {
                "attempt": int(self.manifest["attempts"]) - 1,
                "success": bool(success),
                "seed": int(seed),
                "source_env_id": int(source_env_id),
                "steps": int(steps),
                "reason": reason,
                "metrics": metrics or {},
            }
        )
        self._write_manifest()

    def save_success(
        self,
        episode: EpisodeBuffer,
        *,
        seed: int,
        source_env_id: int,
        episode_metadata: dict[str, Any] | None = None,
    ) -> Path:
        arrays = episode.arrays(require_success=True)
        episode_index = self.successful_episodes
        relative_path = Path("episodes") / f"episode_{episode_index:06d}.npz"
        path = self.output_dir / relative_path
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        metadata = dict(episode_metadata or {})
        initial_state = metadata.pop("initial_state", {})
        record = {
            "episode_index": episode_index,
            "file": relative_path.as_posix(),
            "seed": int(seed),
            "source_env_id": int(source_env_id),
            "steps": len(episode),
            "duration_s": float(arrays["timestamp_s"][-1]),
            "initial_state": initial_state,
            **metadata,
        }
        self.manifest["episodes"].append(record)
        self.manifest["successful_episodes"] = episode_index + 1
        self._write_manifest()
        return path

    def finalize(self) -> dict[str, Any]:
        complete = self.successful_episodes >= int(self.manifest["requested_episodes"])
        self.manifest["status"] = "complete" if complete else "insufficient_successes"
        self._write_manifest()
        return self.manifest
