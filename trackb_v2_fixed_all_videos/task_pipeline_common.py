"""Shared dataset, policy, replay, and residual-TD3 code for all Track B tasks."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


TASK_NAMES = (
    "push_object_to_region",
    "push_cube_to_region",
    "drawer_opening",
    "cabinet_opening",
    "object_to_drawer",
    "nut_pouring",
)
DATASET_FORMAT = "track_b_successful_demonstrations_v1"
RGB_SHAPE = (224, 224, 3)
PROPRIO_DIM = 25
ACTION_DIM = 7
ARM_DIM = 6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_print(payload: dict) -> None:
    print(json.dumps(payload, indent=2), flush=True)


@dataclass(frozen=True)
class EpisodeRef:
    dataset_dir: Path
    file: Path
    episode_index: int
    steps: int


def validate_task_name(task_name: str) -> str:
    if task_name not in TASK_NAMES:
        raise ValueError(f"unsupported task {task_name!r}; choose from {TASK_NAMES}")
    return task_name


def resolve_dataset_dirs(
    task_name: str,
    dataset_dirs: Sequence[Path] | None,
    repository_dir: Path | None = None,
) -> list[Path]:
    """Use explicit roots or discover the released roots for the selected task."""

    validate_task_name(task_name)
    if dataset_dirs:
        return [Path(path).expanduser().resolve() for path in dataset_dirs]
    root = (repository_dir or Path.cwd()).expanduser().resolve()
    resolved = [root / "datasets" / "primary_50" / task_name]
    additional = root / "datasets" / "additional_50_seed20260927" / task_name
    if additional.is_dir():
        resolved.append(additional)
    return resolved


def load_episode_refs(dataset_dirs: Sequence[Path], task_name: str) -> list[EpisodeRef]:
    validate_task_name(task_name)
    refs: list[EpisodeRef] = []
    seen: set[Path] = set()
    for raw_dir in dataset_dirs:
        dataset_dir = raw_dir.expanduser().resolve()
        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != DATASET_FORMAT:
            raise ValueError(
                f"unsupported dataset format in {manifest_path}: {manifest.get('format')!r}"
            )
        if manifest.get("task_name") != task_name:
            raise ValueError(
                f"expected task_name={task_name!r}, got {manifest.get('task_name')!r} "
                f"in {manifest_path}"
            )
        episodes = manifest.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"manifest contains no episodes: {manifest_path}")
        for item in episodes:
            path = (dataset_dir / item["file"]).resolve()
            if path in seen:
                raise ValueError(f"duplicate episode path: {path}")
            if not path.is_file():
                raise FileNotFoundError(path)
            seen.add(path)
            refs.append(
                EpisodeRef(
                    dataset_dir=dataset_dir,
                    file=path,
                    episode_index=int(item["episode_index"]),
                    steps=int(item["steps"]),
                )
            )
    if not refs:
        raise ValueError("no episodes found")
    return refs


REQUIRED_KEYS = (
    "observation.rgb",
    "observation.proprio",
    "observation.oracle",
    "action",
    "reward",
    "terminated",
    "truncated",
    "success",
    "timestamp_s",
)


def validate_episode(ref: EpisodeRef) -> dict:
    with np.load(ref.file, allow_pickle=False) as episode:
        missing = [key for key in REQUIRED_KEYS if key not in episode]
        if missing:
            raise ValueError(f"{ref.file} is missing keys: {missing}")
        arrays = {key: episode[key] for key in REQUIRED_KEYS}

    lengths = {key: int(value.shape[0]) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"misaligned arrays in {ref.file}: {lengths}")
    steps = next(iter(lengths.values()))
    if steps != ref.steps:
        raise ValueError(
            f"manifest steps={ref.steps}, file steps={steps} for {ref.file}"
        )
    rgb = arrays["observation.rgb"]
    proprio = arrays["observation.proprio"]
    oracle = arrays["observation.oracle"]
    action = arrays["action"]
    if rgb.shape != (steps, *RGB_SHAPE) or rgb.dtype != np.uint8:
        raise ValueError(f"invalid RGB array in {ref.file}: {rgb.shape}, {rgb.dtype}")
    if proprio.shape != (steps, PROPRIO_DIM) or proprio.dtype != np.float32:
        raise ValueError(
            f"invalid proprio array in {ref.file}: {proprio.shape}, {proprio.dtype}"
        )
    if oracle.ndim != 2 or oracle.shape[0] != steps or oracle.dtype != np.float32:
        raise ValueError(
            f"invalid oracle array in {ref.file}: {oracle.shape}, {oracle.dtype}"
        )
    if action.shape != (steps, ACTION_DIM) or action.dtype != np.float32:
        raise ValueError(
            f"invalid action array in {ref.file}: {action.shape}, {action.dtype}"
        )
    if not np.isfinite(proprio).all() or not np.isfinite(oracle).all():
        raise ValueError(f"non-finite observation in {ref.file}")
    if not np.isfinite(action).all() or np.max(np.abs(action)) > 1.0 + 1.0e-6:
        raise ValueError(f"action outside [-1, 1] or non-finite in {ref.file}")
    gripper = action[:, -1]
    if not np.all(np.isin(gripper, (-1.0, 1.0))):
        raise ValueError(f"gripper is not strictly binary in {ref.file}")
    timestamps = arrays["timestamp_s"]
    if steps > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"non-monotonic timestamps in {ref.file}")
    if not bool(arrays["success"][-1]):
        raise ValueError(f"episode does not end in success: {ref.file}")
    return {
        "steps": steps,
        "oracle_dim": int(oracle.shape[1]),
        "action_min": action.min(axis=0),
        "action_max": action.max(axis=0),
        "duration_s": float(timestamps[-1]),
    }


def inspect_datasets(dataset_dirs: Sequence[Path], task_name: str) -> dict:
    refs = load_episode_refs(dataset_dirs, task_name)
    total_steps = 0
    durations: list[float] = []
    oracle_dims: set[int] = set()
    action_min = np.full((ACTION_DIM,), np.inf, dtype=np.float32)
    action_max = np.full((ACTION_DIM,), -np.inf, dtype=np.float32)
    for index, ref in enumerate(refs, start=1):
        result = validate_episode(ref)
        total_steps += result["steps"]
        durations.append(result["duration_s"])
        oracle_dims.add(result["oracle_dim"])
        action_min = np.minimum(action_min, result["action_min"])
        action_max = np.maximum(action_max, result["action_max"])
        if index % 10 == 0 or index == len(refs):
            print(f"validated {index}/{len(refs)} episodes", flush=True)
    if len(oracle_dims) != 1:
        raise ValueError(f"inconsistent oracle dimensions: {sorted(oracle_dims)}")
    return {
        "task": task_name,
        "dataset_dirs": [str(path.expanduser().resolve()) for path in dataset_dirs],
        "episodes": len(refs),
        "steps": total_steps,
        "oracle_dim": next(iter(oracle_dims)),
        "duration_s_mean": float(np.mean(durations)),
        "duration_s_min": float(np.min(durations)),
        "duration_s_max": float(np.max(durations)),
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "status": "valid",
    }


def split_episode_refs(
    refs: Sequence[EpisodeRef], validation_ratio: float, seed: int
) -> tuple[list[EpisodeRef], list[EpisodeRef]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if len(refs) < 2:
        raise ValueError("at least two episodes are required for a split")
    order = np.arange(len(refs))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    n_val = max(1, int(round(len(refs) * validation_ratio)))
    n_val = min(n_val, len(refs) - 1)
    val_ids = set(order[:n_val].tolist())
    train_refs = [ref for i, ref in enumerate(refs) if i not in val_ids]
    val_refs = [ref for i, ref in enumerate(refs) if i in val_ids]
    return train_refs, val_refs


class _EpisodeCache:
    def __init__(self, max_items: int = 8):
        self.max_items = max_items
        self.cache: OrderedDict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if path in self.cache:
            value = self.cache.pop(path)
            self.cache[path] = value
            return value
        with np.load(path, allow_pickle=False) as episode:
            value = (
                np.asarray(episode["observation.rgb"], dtype=np.uint8),
                np.asarray(episode["observation.proprio"], dtype=np.float32),
                np.asarray(episode["action"], dtype=np.float32),
            )
        self.cache[path] = value
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return value


class TaskStepDataset(Dataset):
    def __init__(self, refs: Sequence[EpisodeRef], cache_size: int = 8):
        self.refs = list(refs)
        self.index: list[tuple[int, int]] = []
        for episode_id, ref in enumerate(self.refs):
            self.index.extend((episode_id, step) for step in range(ref.steps))
        self.cache = _EpisodeCache(cache_size)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_id, step = self.index[item]
        rgb, proprio, action = self.cache.get(self.refs[episode_id].file)
        image = torch.from_numpy(rgb[step].copy()).permute(2, 0, 1).float().div_(255.0)
        return {
            "rgb": image,
            "proprio": torch.from_numpy(proprio[step].copy()),
            "action": torch.from_numpy(action[step].copy()),
        }


class VisualBCPolicy(nn.Module):
    """Single-view RGB/proprio BC actor with a separate binary gripper head."""

    def __init__(self, image_feature_dim: int = 256, proprio_feature_dim: int = 64):
        super().__init__()
        self.image_feature_dim = int(image_feature_dim)
        self.proprio_feature_dim = int(proprio_feature_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, self.image_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.proprio_encoder = nn.Sequential(
            nn.LayerNorm(PROPRIO_DIM),
            nn.Linear(PROPRIO_DIM, self.proprio_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proprio_feature_dim, self.proprio_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = self.image_feature_dim + self.proprio_feature_dim
        self.trunk = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.arm_head = nn.Linear(128, ARM_DIM)
        self.gripper_head = nn.Linear(128, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.arm_head.weight, gain=0.01)
        nn.init.orthogonal_(self.gripper_head.weight, gain=0.01)

    def encode(self, rgb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(rgb.clamp(0.0, 1.0))
        proprio_features = self.proprio_encoder(proprio)
        return torch.cat((image_features, proprio_features), dim=-1)

    def forward(
        self, rgb: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.encode(rgb, proprio)
        hidden = self.trunk(feature)
        arm = torch.tanh(self.arm_head(hidden))
        gripper_logit = self.gripper_head(hidden).squeeze(-1)
        return arm, gripper_logit, feature

    @torch.inference_mode()
    def deterministic_action_and_feature(
        self, rgb: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        arm, gripper_logit, feature = self(rgb, proprio)
        gripper = torch.where(
            gripper_logit >= 0.0,
            torch.ones_like(gripper_logit),
            -torch.ones_like(gripper_logit),
        )
        return torch.cat((arm, gripper.unsqueeze(-1)), dim=-1), feature


def bc_loss(
    arm_pred: torch.Tensor,
    gripper_logit: torch.Tensor,
    target_action: torch.Tensor,
    gripper_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    arm_loss = F.smooth_l1_loss(arm_pred, target_action[:, :ARM_DIM])
    gripper_target = (target_action[:, ARM_DIM] > 0.0).float()
    gripper_loss = F.binary_cross_entropy_with_logits(gripper_logit, gripper_target)
    total = arm_loss + float(gripper_weight) * gripper_loss
    gripper_accuracy = (
        ((gripper_logit >= 0.0) == (gripper_target > 0.5)).float().mean()
    )
    return total, {
        "loss": float(total.detach()),
        "arm_loss": float(arm_loss.detach()),
        "gripper_loss": float(gripper_loss.detach()),
        "gripper_accuracy": float(gripper_accuracy.detach()),
    }


def save_bc_checkpoint(
    path: Path,
    model: VisualBCPolicy,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    validation_loss: float,
    config: dict,
    task_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "track_b_visual_bc_v2",
        "task": validate_task_name(task_name),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "validation_loss": float(validation_loss),
        "config": config,
    }
    torch.save(payload, path)


def load_bc_checkpoint(
    path: Path,
    device: torch.device | str,
    expected_task: str | None = None,
) -> tuple[VisualBCPolicy, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    supported_formats = {"track_b_visual_bc_v2", "track_b_drawer_visual_bc_v1"}
    if payload.get("format") not in supported_formats:
        raise ValueError(f"unsupported BC checkpoint: {path}")
    checkpoint_task = payload.get("task", "drawer_opening")
    if expected_task is not None and checkpoint_task != expected_task:
        raise ValueError(
            f"BC checkpoint task={checkpoint_task!r}, requested task={expected_task!r}"
        )
    cfg = payload["config"]
    model = VisualBCPolicy(
        image_feature_dim=int(cfg["image_feature_dim"]),
        proprio_feature_dim=int(cfg["proprio_feature_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def policy_observation(observations: dict, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    policy = observations["policy"]
    rgb = policy["rgb"].to(device=device, dtype=torch.float32)
    proprio = policy["proprio"].to(device=device, dtype=torch.float32)
    if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 224, 224):
        raise RuntimeError(f"unexpected online RGB shape: {tuple(rgb.shape)}")
    if proprio.ndim != 2 or proprio.shape[1] != PROPRIO_DIM:
        raise RuntimeError(f"unexpected online proprio shape: {tuple(proprio.shape)}")
    return rgb.clamp(0.0, 1.0), proprio


def oracle_observation(observations: dict, device: torch.device | str) -> torch.Tensor:
    oracle = observations["critic"].to(device=device, dtype=torch.float32)
    if oracle.ndim != 2:
        raise RuntimeError(f"unexpected oracle shape: {tuple(oracle.shape)}")
    return oracle


def environment_success(env) -> torch.Tensor:
    """Read the named success term instead of treating every termination as success."""

    base = env.unwrapped if hasattr(env, "unwrapped") else env
    return base.termination_manager.get_term("success").clone().bool()


def compose_action(
    base_action: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    arm = (base_action[:, :ARM_DIM] + residual[:, :ARM_DIM]).clamp(-1.0, 1.0)
    grip_score = base_action[:, ARM_DIM] + residual[:, ARM_DIM]
    gripper = torch.where(
        grip_score >= 0.0,
        torch.ones_like(grip_score),
        -torch.ones_like(grip_score),
    )
    return torch.cat((arm, gripper.unsqueeze(-1)), dim=-1)


class ResidualActor(nn.Module):
    def __init__(self, feature_dim: int, arm_limit: float, gripper_limit: float):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.register_buffer(
            "limits",
            torch.tensor(
                [float(arm_limit)] * ARM_DIM + [float(gripper_limit)],
                dtype=torch.float32,
            ).view(1, ACTION_DIM),
        )
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim + ACTION_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, ACTION_DIM),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.cat((feature, base_action), dim=-1))
        return torch.tanh(raw) * self.limits


class Critic(nn.Module):
    def __init__(self, feature_dim: int, oracle_dim: int):
        super().__init__()
        input_dim = int(feature_dim) + int(oracle_dim) + 2 * ACTION_DIM
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        feature: torch.Tensor,
        oracle: torch.Tensor,
        base_action: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.cat((feature, oracle, base_action, residual), dim=-1)
        return self.net(value)


class ReplayBuffer:
    FIELDS = (
        "feature",
        "oracle",
        "base_action",
        "residual",
        "reward",
        "next_feature",
        "next_oracle",
        "next_base_action",
        "done",
    )

    def __init__(
        self, capacity: int, feature_dim: int, oracle_dim: int, seed: int
    ) -> None:
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)
        self.arrays = {
            "feature": np.empty((capacity, feature_dim), dtype=np.float32),
            "oracle": np.empty((capacity, oracle_dim), dtype=np.float32),
            "base_action": np.empty((capacity, ACTION_DIM), dtype=np.float32),
            "residual": np.empty((capacity, ACTION_DIM), dtype=np.float32),
            "reward": np.empty((capacity, 1), dtype=np.float32),
            "next_feature": np.empty((capacity, feature_dim), dtype=np.float32),
            "next_oracle": np.empty((capacity, oracle_dim), dtype=np.float32),
            "next_base_action": np.empty((capacity, ACTION_DIM), dtype=np.float32),
            "done": np.empty((capacity, 1), dtype=np.float32),
        }

    def add_batch(self, **values: np.ndarray) -> None:
        if set(values) != set(self.FIELDS):
            raise ValueError(f"replay fields mismatch: {sorted(values)}")
        count = int(values["feature"].shape[0])
        if count <= 0:
            return
        indices = (np.arange(count) + self.position) % self.capacity
        for key in self.FIELDS:
            value = np.asarray(values[key], dtype=np.float32)
            if key in {"reward", "done"} and value.ndim == 1:
                value = value[:, None]
            self.arrays[key][indices] = value
        self.position = int((self.position + count) % self.capacity)
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError(f"replay has {self.size} rows, needs {batch_size}")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {
            key: torch.from_numpy(array[indices]).to(device)
            for key, array in self.arrays.items()
        }


class DemoFeatureBank:
    def __init__(
        self,
        feature: np.ndarray,
        base_action: np.ndarray,
        target_residual: np.ndarray,
        seed: int,
    ) -> None:
        self.feature = np.asarray(feature, dtype=np.float32)
        self.base_action = np.asarray(base_action, dtype=np.float32)
        self.target_residual = np.asarray(target_residual, dtype=np.float32)
        if not (
            len(self.feature) == len(self.base_action) == len(self.target_residual)
        ):
            raise ValueError("misaligned demo feature bank")
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.feature)

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        indices = self.rng.integers(0, len(self), size=batch_size)
        return {
            "feature": torch.from_numpy(self.feature[indices]).to(device),
            "base_action": torch.from_numpy(self.base_action[indices]).to(device),
            "target_residual": torch.from_numpy(self.target_residual[indices]).to(device),
        }


@torch.inference_mode()
def build_demo_feature_bank(
    model: VisualBCPolicy,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device | str,
    arm_limit: float,
    gripper_limit: float,
    seed: int,
) -> DemoFeatureBank:
    features: list[np.ndarray] = []
    base_actions: list[np.ndarray] = []
    target_residuals: list[np.ndarray] = []
    limits = torch.tensor(
        [float(arm_limit)] * ARM_DIM + [float(gripper_limit)],
        device=device,
        dtype=torch.float32,
    )
    model.eval()
    for batch_id, batch in enumerate(loader, start=1):
        rgb = batch["rgb"].to(device, non_blocking=True)
        proprio = batch["proprio"].to(device, non_blocking=True)
        target = batch["action"].to(device, non_blocking=True)
        base, feature = model.deterministic_action_and_feature(rgb, proprio)
        target_residual = (target - base).clamp(-limits, limits)
        features.append(feature.cpu().numpy())
        base_actions.append(base.cpu().numpy())
        target_residuals.append(target_residual.cpu().numpy())
        if batch_id % 20 == 0:
            print(f"demo feature batches: {batch_id}", flush=True)
    return DemoFeatureBank(
        np.concatenate(features),
        np.concatenate(base_actions),
        np.concatenate(target_residuals),
        seed,
    )


class ResidualTD3BCAgent:
    def __init__(
        self,
        feature_dim: int,
        oracle_dim: int,
        device: torch.device | str,
        arm_limit: float = 0.10,
        gripper_limit: float = 2.0,
        actor_lr: float = 5.0e-5,
        critic_lr: float = 5.0e-5,
        weight_decay: float = 1.0e-5,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.05,
        noise_clip: float = 0.08,
        policy_delay: int = 2,
    ) -> None:
        import copy

        self.device = torch.device(device)
        self.feature_dim = int(feature_dim)
        self.oracle_dim = int(oracle_dim)
        self.arm_limit = float(arm_limit)
        self.gripper_limit = float(gripper_limit)
        self.discount = float(discount)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay)
        self.total_updates = 0

        self.actor = ResidualActor(feature_dim, arm_limit, gripper_limit).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1 = Critic(feature_dim, oracle_dim).to(self.device)
        self.critic2 = Critic(feature_dim, oracle_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr, weight_decay=weight_decay
        )
        self.critic_optimizer = torch.optim.AdamW(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
            weight_decay=weight_decay,
        )

    @torch.inference_mode()
    def act(
        self,
        feature: torch.Tensor,
        base_action: torch.Tensor,
        exploration_std: float = 0.0,
    ) -> torch.Tensor:
        residual = self.actor(feature, base_action)
        if exploration_std > 0.0:
            residual = residual + torch.randn_like(residual) * exploration_std
            residual = torch.maximum(
                torch.minimum(residual, self.actor.limits), -self.actor.limits
            )
        return residual

    def train_step(
        self,
        replay: ReplayBuffer,
        demo_bank: DemoFeatureBank,
        batch_size: int,
        demo_batch_size: int,
        transition_step: int,
        bc_weight_start: float,
        bc_weight_end: float,
        bc_decay_transitions: int,
    ) -> dict[str, float]:
        batch = replay.sample(batch_size, self.device)
        with torch.no_grad():
            noise = torch.randn_like(batch["residual"]) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_residual = self.actor_target(
                batch["next_feature"], batch["next_base_action"]
            ) + noise
            next_residual = torch.maximum(
                torch.minimum(next_residual, self.actor_target.limits),
                -self.actor_target.limits,
            )
            target_q1 = self.critic1_target(
                batch["next_feature"],
                batch["next_oracle"],
                batch["next_base_action"],
                next_residual,
            )
            target_q2 = self.critic2_target(
                batch["next_feature"],
                batch["next_oracle"],
                batch["next_base_action"],
                next_residual,
            )
            target = batch["reward"] + self.discount * (1.0 - batch["done"]) * torch.minimum(
                target_q1, target_q2
            )

        q1 = self.critic1(
            batch["feature"], batch["oracle"], batch["base_action"], batch["residual"]
        )
        q2 = self.critic2(
            batch["feature"], batch["oracle"], batch["base_action"], batch["residual"]
        )
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 5.0
        )
        self.critic_optimizer.step()

        logs = {
            "critic_loss": float(critic_loss.detach()),
            "q1": float(q1.detach().mean()),
            "q2": float(q2.detach().mean()),
        }
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            actor_residual = self.actor(batch["feature"], batch["base_action"])
            actor_rl_loss = -self.critic1(
                batch["feature"],
                batch["oracle"],
                batch["base_action"],
                actor_residual,
            ).mean()
            demo = demo_bank.sample(demo_batch_size, self.device)
            predicted_demo_residual = self.actor(demo["feature"], demo["base_action"])
            actor_bc_loss = F.smooth_l1_loss(
                predicted_demo_residual, demo["target_residual"]
            )
            fraction = min(float(transition_step) / max(bc_decay_transitions, 1), 1.0)
            bc_weight = (1.0 - fraction) * bc_weight_start + fraction * bc_weight_end
            actor_loss = actor_rl_loss + bc_weight * actor_bc_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            logs.update(
                {
                    "actor_loss": float(actor_loss.detach()),
                    "actor_rl_loss": float(actor_rl_loss.detach()),
                    "actor_bc_loss": float(actor_bc_loss.detach()),
                    "bc_weight": float(bc_weight),
                }
            )
        return logs

    @torch.no_grad()
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            target_parameter.mul_(1.0 - self.tau).add_(source_parameter, alpha=self.tau)

    def save(
        self,
        path: Path,
        bc_checkpoint: Path,
        config: dict,
        task_name: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "track_b_residual_td3bc_v2",
                "task": validate_task_name(task_name),
                "bc_checkpoint": str(bc_checkpoint.resolve()),
                "feature_dim": self.feature_dim,
                "oracle_dim": self.oracle_dim,
                "arm_limit": self.arm_limit,
                "gripper_limit": self.gripper_limit,
                "total_updates": self.total_updates,
                "actor_state_dict": self.actor.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic1_state_dict": self.critic1.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "critic1_target_state_dict": self.critic1_target.state_dict(),
                "critic2_target_state_dict": self.critic2_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "config": config,
            },
            path,
        )


def load_residual_actor(
    checkpoint: Path,
    device: torch.device | str,
    expected_task: str | None = None,
) -> tuple[ResidualActor, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    supported_formats = {
        "track_b_residual_td3bc_v2",
        "track_b_drawer_residual_td3bc_v1",
    }
    if payload.get("format") not in supported_formats:
        raise ValueError(f"unsupported residual checkpoint: {checkpoint}")
    checkpoint_task = payload.get("task", "drawer_opening")
    if expected_task is not None and checkpoint_task != expected_task:
        raise ValueError(
            f"residual checkpoint task={checkpoint_task!r}, requested task={expected_task!r}"
        )
    actor = ResidualActor(
        int(payload["feature_dim"]),
        float(payload["arm_limit"]),
        float(payload["gripper_limit"]),
    ).to(device)
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    return actor, payload


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)
