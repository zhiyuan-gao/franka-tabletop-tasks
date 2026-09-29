"""Track B v2 pipeline.

Everything that both the BC stage and the residual-RL stage need:

* dataset with frame history, action chunks, per-dimension action normalization
  and gripper-transition weighting,
* a ResNet-18/spatial-softmax visuomotor policy that predicts action chunks,
* a runner that reproduces the *exact* training-time observation and the
  ACT-style temporal ensemble at rollout time,
* replay buffers with n-step returns and a protected success buffer,
* residual TD3+BC with normalized BC anchoring, a gated gripper correction and
  correct time-limit bootstrapping.

Import-compatible with the existing repository: dataset discovery, manifest
validation and the Isaac Lab observation helpers are reused from
``task_pipeline_common``.
"""

from __future__ import annotations

import copy
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from task_pipeline_common import (  # noqa: F401  (re-exported on purpose)
    ACTION_DIM,
    ARM_DIM,
    PROPRIO_DIM,
    RGB_SHAPE,
    TASK_NAMES,
    EpisodeRef,
    environment_success,
    inspect_datasets,
    load_episode_refs,
    oracle_observation,
    policy_observation,
    resolve_dataset_dirs,
    set_seed,
    split_episode_refs,
    to_numpy,
    validate_task_name,
)

BC_FORMAT = "track_b_chunk_bc_v3"
RESIDUAL_FORMAT = "track_b_residual_td3bc_v3"


# --------------------------------------------------------------------------- #
# action normalization
# --------------------------------------------------------------------------- #
class ActionNormalizer:
    """Per-dimension affine map from the demonstrated arm range onto [-1, 1]."""

    def __init__(self, low: Sequence[float], high: Sequence[float]) -> None:
        low = np.asarray(low, dtype=np.float32).reshape(ARM_DIM)
        high = np.asarray(high, dtype=np.float32).reshape(ARM_DIM)
        span = np.maximum(high - low, 1.0e-3).astype(np.float32)
        self.low = low
        self.span = span

    @classmethod
    def from_refs(
        cls,
        refs: Sequence[EpisodeRef],
        low_percentile: float = 0.5,
        high_percentile: float = 99.5,
        margin: float = 0.05,
    ) -> "ActionNormalizer":
        chunks: list[np.ndarray] = []
        for ref in refs:
            with np.load(ref.file, allow_pickle=False) as episode:
                chunks.append(np.asarray(episode["action"], dtype=np.float32)[:, :ARM_DIM])
        actions = np.concatenate(chunks, axis=0)
        low = np.percentile(actions, low_percentile, axis=0).astype(np.float32)
        high = np.percentile(actions, high_percentile, axis=0).astype(np.float32)
        pad = margin * np.maximum(high - low, 1.0e-3)
        low = np.maximum(low - pad, -1.0)
        high = np.minimum(high + pad, 1.0)
        degenerate = (high - low) < 1.0e-3
        low[degenerate] = -1.0
        high[degenerate] = 1.0
        return cls(low, high)

    def normalize(self, arm: np.ndarray) -> np.ndarray:
        value = 2.0 * (arm - self.low) / self.span - 1.0
        return np.clip(value, -1.0, 1.0).astype(np.float32)

    def denormalize_torch(self, arm: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(self.low, device=arm.device, dtype=arm.dtype)
        span = torch.as_tensor(self.span, device=arm.device, dtype=arm.dtype)
        return (low + 0.5 * (arm + 1.0) * span).clamp(-1.0, 1.0)

    def normalize_torch(self, arm: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(self.low, device=arm.device, dtype=arm.dtype)
        span = torch.as_tensor(self.span, device=arm.device, dtype=arm.dtype)
        return (2.0 * (arm - low) / span - 1.0).clamp(-1.0, 1.0)

    def to_dict(self) -> dict:
        return {"low": self.low.tolist(), "high": (self.low + self.span).tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "ActionNormalizer":
        return cls(payload["low"], payload["high"])


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def _resize_uint8_episode(rgb: np.ndarray, image_size: int) -> np.ndarray:
    """Resize a whole episode once, at cache time, and keep it as uint8."""

    if rgb.shape[1] == image_size and rgb.shape[2] == image_size:
        return np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))
    out = np.empty((rgb.shape[0], 3, image_size, image_size), dtype=np.uint8)
    step = 64
    for start in range(0, rgb.shape[0], step):
        block = torch.from_numpy(rgb[start : start + step]).permute(0, 3, 1, 2).float()
        block = F.interpolate(
            block, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        out[start : start + step] = block.round().clamp_(0, 255).byte().numpy()
    return out


def gripper_transition_weight(
    gripper: np.ndarray, transition_weight: float, decay_steps: float
) -> np.ndarray:
    """Up-weight the few frames around every open/close switch."""

    steps = gripper.shape[0]
    weight = np.ones(steps, dtype=np.float32)
    if transition_weight <= 0.0 or steps < 2:
        return weight
    switches = np.flatnonzero(np.abs(np.diff(gripper)) > 0.5) + 1
    if switches.size == 0:
        return weight
    index = np.arange(steps, dtype=np.float32)
    for switch in switches:
        weight += transition_weight * np.exp(
            -np.abs(index - float(switch)) / max(decay_steps, 1.0e-3)
        )
    return weight


class _EpisodeCacheV2:
    def __init__(self, image_size: int, normalizer: ActionNormalizer, max_items: int = 6):
        self.image_size = int(image_size)
        self.normalizer = normalizer
        self.max_items = int(max_items)
        self.cache: "OrderedDict[Path, dict]" = OrderedDict()

    def get(self, path: Path, transition_weight: float, decay_steps: float) -> dict:
        if path in self.cache:
            value = self.cache.pop(path)
            self.cache[path] = value
            return value
        with np.load(path, allow_pickle=False) as episode:
            rgb = np.asarray(episode["observation.rgb"], dtype=np.uint8)
            proprio = np.asarray(episode["observation.proprio"], dtype=np.float32)
            action = np.asarray(episode["action"], dtype=np.float32)
        gripper = (action[:, ARM_DIM] > 0.0).astype(np.float32)
        value = {
            "rgb": _resize_uint8_episode(rgb, self.image_size),
            "proprio": proprio,
            "arm": self.normalizer.normalize(action[:, :ARM_DIM]),
            "gripper": gripper,
            "gripper_weight": gripper_transition_weight(
                gripper, transition_weight, decay_steps
            ),
        }
        self.cache[path] = value
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return value


class ChunkStepDataset(Dataset):
    """One sample = frame history + proprio history -> normalized action chunk."""

    def __init__(
        self,
        refs: Sequence[EpisodeRef],
        normalizer: ActionNormalizer,
        frame_stack: int = 3,
        frame_stride: int = 3,
        chunk: int = 8,
        image_size: int = 128,
        gripper_transition_weight_scale: float = 4.0,
        gripper_transition_decay: float = 3.0,
        cache_size: int = 6,
    ) -> None:
        self.refs = list(refs)
        self.frame_stack = int(frame_stack)
        self.frame_stride = int(frame_stride)
        self.chunk = int(chunk)
        self.image_size = int(image_size)
        self.transition_weight = float(gripper_transition_weight_scale)
        self.transition_decay = float(gripper_transition_decay)
        self.cache = _EpisodeCacheV2(image_size, normalizer, cache_size)
        self.index: list[tuple[int, int]] = []
        for episode_id, ref in enumerate(self.refs):
            self.index.extend((episode_id, step) for step in range(ref.steps))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_id, step = self.index[item]
        data = self.cache.get(
            self.refs[episode_id].file, self.transition_weight, self.transition_decay
        )
        steps = data["proprio"].shape[0]

        history = [
            max(0, step - k * self.frame_stride)
            for k in reversed(range(self.frame_stack))
        ]
        frames = data["rgb"][history]  # (K, 3, S, S) uint8
        proprio = data["proprio"][history]  # (K, 25)

        future = np.minimum(np.arange(step, step + self.chunk), steps - 1)
        mask = (np.arange(step, step + self.chunk) < steps).astype(np.float32)
        arm = data["arm"][future]
        gripper = data["gripper"][future]
        gripper_weight = data["gripper_weight"][future]

        return {
            "frames": torch.from_numpy(np.ascontiguousarray(frames)),
            "proprio": torch.from_numpy(np.ascontiguousarray(proprio)),
            "arm": torch.from_numpy(arm),
            "gripper": torch.from_numpy(gripper),
            "mask": torch.from_numpy(mask),
            "gripper_weight": torch.from_numpy(gripper_weight),
        }


class SequentialEpisodeDataset(Dataset):
    """Whole episodes, in order -- used to rebuild base actions for residual RL."""

    def __init__(
        self,
        refs: Sequence[EpisodeRef],
        normalizer: ActionNormalizer,
        image_size: int = 128,
    ) -> None:
        self.refs = list(refs)
        self.normalizer = normalizer
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        ref = self.refs[item]
        with np.load(ref.file, allow_pickle=False) as episode:
            rgb = np.asarray(episode["observation.rgb"], dtype=np.uint8)
            proprio = np.asarray(episode["observation.proprio"], dtype=np.float32)
            action = np.asarray(episode["action"], dtype=np.float32)
        return {
            "rgb": torch.from_numpy(_resize_uint8_episode(rgb, self.image_size)),
            "proprio": torch.from_numpy(proprio),
            "action": torch.from_numpy(action),
        }


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #
def random_shift(frames: torch.Tensor, pad: int = 6) -> torch.Tensor:
    """DrQ-v2 random shift, identical for every frame in a stack."""

    if pad <= 0:
        return frames
    batch, stack, channels, height, width = frames.shape
    flat = frames.view(batch * stack, channels, height, width)
    flat = F.pad(flat, (pad,) * 4, mode="replicate")
    eps_h = torch.randint(0, 2 * pad + 1, (batch,), device=frames.device)
    eps_w = torch.randint(0, 2 * pad + 1, (batch,), device=frames.device)
    eps_h = eps_h.repeat_interleave(stack)
    eps_w = eps_w.repeat_interleave(stack)
    rows = torch.arange(height, device=frames.device).view(1, height, 1)
    cols = torch.arange(width, device=frames.device).view(1, 1, width)
    rows = rows + eps_h.view(-1, 1, 1)
    cols = cols + eps_w.view(-1, 1, 1)
    batch_index = torch.arange(flat.shape[0], device=frames.device).view(-1, 1, 1)
    out = flat[batch_index, :, rows, cols]  # (B*K, H, W, C)
    out = out.permute(0, 3, 1, 2).contiguous()
    return out.view(batch, stack, channels, height, width)


def photometric_jitter(frames: torch.Tensor, strength: float = 0.15) -> torch.Tensor:
    if strength <= 0.0:
        return frames
    batch = frames.shape[0]
    shape = (batch, 1, 1, 1, 1)
    brightness = 1.0 + (torch.rand(shape, device=frames.device) * 2.0 - 1.0) * strength
    contrast = 1.0 + (torch.rand(shape, device=frames.device) * 2.0 - 1.0) * strength
    mean = frames.mean(dim=(2, 3, 4), keepdim=True)
    return ((frames - mean) * contrast + mean * brightness).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# vision encoder
# --------------------------------------------------------------------------- #
def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 1
    for candidate in (32, 16, 8, 4, 2, 1):
        if channels % candidate == 0:
            groups = candidate
            break
    return nn.GroupNorm(groups, channels)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class SpatialSoftmax(nn.Module):
    """Keypoint bottleneck: where things are, not just what is present."""

    def __init__(self, in_channels: int, keypoints: int = 32, temperature: float = 1.0):
        super().__init__()
        self.project = nn.Conv2d(in_channels, keypoints, 1)
        self.keypoints = int(keypoints)
        self.log_temperature = nn.Parameter(torch.tensor(float(math.log(temperature))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heat = self.project(x)
        batch, channels, height, width = heat.shape
        heat = heat.view(batch, channels, height * width) / self.log_temperature.exp().clamp(
            min=1.0e-2
        )
        weight = torch.softmax(heat, dim=-1).view(batch, channels, height, width)
        ys = torch.linspace(-1.0, 1.0, height, device=x.device).view(1, 1, height, 1)
        xs = torch.linspace(-1.0, 1.0, width, device=x.device).view(1, 1, 1, width)
        expected_y = (weight * ys).sum(dim=(2, 3))
        expected_x = (weight * xs).sum(dim=(2, 3))
        return torch.cat((expected_x, expected_y), dim=-1)


class VisionEncoder(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int = 384, keypoints: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            _group_norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.keypoints = SpatialSoftmax(256, keypoints)
        self.project = nn.Sequential(
            nn.Linear(512 + 2 * keypoints, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        mid = self.layer3(x)
        deep = self.layer4(mid)
        pooled = F.adaptive_avg_pool2d(deep, 1).flatten(1)
        points = self.keypoints(mid)
        return self.project(torch.cat((pooled, points), dim=-1))


class ChunkBCPolicy(nn.Module):
    """History-conditioned visuomotor policy that outputs an action chunk."""

    def __init__(
        self,
        frame_stack: int = 3,
        chunk: int = 8,
        image_feature_dim: int = 384,
        proprio_feature_dim: int = 128,
        hidden_dim: int = 512,
        keypoints: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
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
        self.trunk = nn.Sequential(
            nn.Linear(image_feature_dim + proprio_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.arm_head = nn.Linear(hidden_dim, self.chunk * ARM_DIM)
        self.gripper_head = nn.Linear(hidden_dim, self.chunk)
        nn.init.orthogonal_(self.arm_head.weight, gain=0.01)
        nn.init.zeros_(self.arm_head.bias)
        nn.init.orthogonal_(self.gripper_head.weight, gain=0.01)
        nn.init.zeros_(self.gripper_head.bias)
        self.feature_dim = int(hidden_dim)

    def forward(
        self, frames: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = frames.shape[0]
        image = frames.reshape(batch, -1, frames.shape[-2], frames.shape[-1])
        image_features = self.encoder(image)
        proprio_features = self.proprio_encoder(proprio.reshape(batch, -1))
        feature = self.trunk(torch.cat((image_features, proprio_features), dim=-1))
        arm = torch.tanh(self.arm_head(feature)).view(batch, self.chunk, ARM_DIM)
        gripper_logit = self.gripper_head(feature).view(batch, self.chunk)
        return arm, gripper_logit, feature


def chunk_bc_loss(
    arm_pred: torch.Tensor,
    gripper_logit: torch.Tensor,
    arm_target: torch.Tensor,
    gripper_target: torch.Tensor,
    mask: torch.Tensor,
    gripper_weight: torch.Tensor,
    gripper_loss_weight: float,
    horizon_decay: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    chunk = arm_pred.shape[1]
    decay = torch.exp(
        -horizon_decay * torch.arange(chunk, device=arm_pred.device, dtype=arm_pred.dtype)
    ).view(1, chunk)
    weight = mask * decay
    denominator = weight.sum().clamp(min=1.0)

    arm_error = F.smooth_l1_loss(arm_pred, arm_target, reduction="none", beta=0.05)
    arm_loss = (arm_error.mean(dim=-1) * weight).sum() / denominator

    gripper_error = F.binary_cross_entropy_with_logits(
        gripper_logit, gripper_target, reduction="none"
    )
    gripper_weighting = weight * gripper_weight
    gripper_loss = (gripper_error * gripper_weighting).sum() / gripper_weighting.sum().clamp(
        min=1.0
    )

    total = arm_loss + float(gripper_loss_weight) * gripper_loss
    with torch.no_grad():
        correct = ((gripper_logit >= 0.0).float() == gripper_target).float()
        accuracy = (correct * weight).sum() / denominator
        first_step_accuracy = (
            (correct[:, 0] * mask[:, 0]).sum() / mask[:, 0].sum().clamp(min=1.0)
        )
        transition = (gripper_weight > 1.5).float() * weight
        transition_accuracy = (correct * transition).sum() / transition.sum().clamp(min=1.0)
    return total, {
        "loss": float(total.detach()),
        "arm_loss": float(arm_loss.detach()),
        "gripper_loss": float(gripper_loss.detach()),
        "gripper_accuracy": float(accuracy),
        "gripper_accuracy_t0": float(first_step_accuracy),
        "gripper_accuracy_transition": float(transition_accuracy),
    }


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for shadow, live in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if shadow.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(live.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(live)


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def save_bc_checkpoint_v2(
    path: Path,
    model: ChunkBCPolicy,
    ema_model: nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict,
    config: dict,
    normalizer: ActionNormalizer,
    task_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": BC_FORMAT,
            "task": validate_task_name(task_name),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": None if ema_model is None else ema_model.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "config": config,
            "normalizer": normalizer.to_dict(),
        },
        path,
    )


def load_bc_checkpoint_v2(
    path: Path,
    device: torch.device | str,
    expected_task: str | None = None,
    prefer_ema: bool = True,
) -> tuple[ChunkBCPolicy, ActionNormalizer, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != BC_FORMAT:
        raise ValueError(
            f"{path} is not a {BC_FORMAT} checkpoint (got {payload.get('format')!r}). "
            "Train with train_task_bc_v2.py."
        )
    task = payload.get("task")
    if expected_task is not None and task != expected_task:
        raise ValueError(f"checkpoint task={task!r}, requested task={expected_task!r}")
    config = payload["config"]
    model = ChunkBCPolicy(
        frame_stack=int(config["frame_stack"]),
        chunk=int(config["chunk"]),
        image_feature_dim=int(config["image_feature_dim"]),
        proprio_feature_dim=int(config["proprio_feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        keypoints=int(config["keypoints"]),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    state = payload["model_state_dict"]
    if prefer_ema and payload.get("ema_state_dict") is not None:
        state = payload["ema_state_dict"]
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    normalizer = ActionNormalizer.from_dict(payload["normalizer"])
    return model, normalizer, payload


# --------------------------------------------------------------------------- #
# rollout runner (identical observation construction online and offline)
# --------------------------------------------------------------------------- #
class ChunkRunner:
    """Frame-history ring buffer + ACT temporal ensemble, vectorized over envs."""

    def __init__(
        self,
        policy: ChunkBCPolicy,
        normalizer: ActionNormalizer,
        num_envs: int,
        device: torch.device | str,
        frame_stack: int,
        frame_stride: int,
        chunk: int,
        image_size: int,
        ensemble_decay: float = 0.35,
        use_ensemble: bool = True,
    ) -> None:
        self.policy = policy
        self.normalizer = normalizer
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.frame_stack = int(frame_stack)
        self.frame_stride = int(frame_stride)
        self.chunk = int(chunk)
        self.image_size = int(image_size)
        self.ensemble_decay = float(ensemble_decay)
        self.use_ensemble = bool(use_ensemble)

        self.history_length = (self.frame_stack - 1) * self.frame_stride + 1
        self.frames = torch.zeros(
            (self.num_envs, self.history_length, 3, self.image_size, self.image_size),
            device=self.device,
        )
        self.proprio = torch.zeros(
            (self.num_envs, self.history_length, PROPRIO_DIM), device=self.device
        )
        self.pointer = 0
        self.pending_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.arm_accumulator = torch.zeros(
            (self.num_envs, self.chunk, ARM_DIM), device=self.device
        )
        self.gripper_accumulator = torch.zeros(
            (self.num_envs, self.chunk), device=self.device
        )
        self.weight_sum = torch.zeros((self.num_envs, self.chunk), device=self.device)

    def mark_reset(self, done: torch.Tensor | None = None) -> None:
        if done is None:
            self.pending_reset[:] = True
            return
        self.pending_reset |= done.to(self.device).bool()

    def _clear_ensemble(self, mask: torch.Tensor) -> None:
        self.arm_accumulator[mask] = 0.0
        self.gripper_accumulator[mask] = 0.0
        self.weight_sum[mask] = 0.0

    @torch.no_grad()
    def step(
        self, rgb: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """rgb: (N, 3, 224, 224) in [0, 1]; returns (action (N, 7), feature (N, D))."""

        # Match _resize_uint8_episode exactly.  The dataset resizes in the
        # uint8 intensity domain and rounds before dividing by 255.  Keeping
        # the online path as an unquantized float resize creates a subtle
        # train/deployment mismatch.
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
            self.pending_reset[:] = False

        offsets = [
            (self.pointer - k * self.frame_stride) % self.history_length
            for k in reversed(range(self.frame_stack))
        ]
        frames = self.frames[:, offsets]
        proprio_history = self.proprio[:, offsets]

        arm_chunk, gripper_chunk, feature = self.policy(frames, proprio_history)

        if self.use_ensemble:
            decay = math.exp(-self.ensemble_decay)
            self.arm_accumulator = torch.roll(self.arm_accumulator, shifts=-1, dims=1) * decay
            self.gripper_accumulator = (
                torch.roll(self.gripper_accumulator, shifts=-1, dims=1) * decay
            )
            self.weight_sum = torch.roll(self.weight_sum, shifts=-1, dims=1) * decay
            self.arm_accumulator[:, -1] = 0.0
            self.gripper_accumulator[:, -1] = 0.0
            self.weight_sum[:, -1] = 0.0
            self.arm_accumulator += arm_chunk
            self.gripper_accumulator += gripper_chunk
            self.weight_sum += 1.0
            weight = self.weight_sum[:, 0].clamp(min=1.0e-6).unsqueeze(-1)
            arm_norm = self.arm_accumulator[:, 0] / weight
            gripper_logit = self.gripper_accumulator[:, 0] / weight.squeeze(-1)
        else:
            arm_norm = arm_chunk[:, 0]
            gripper_logit = gripper_chunk[:, 0]

        arm = self.normalizer.denormalize_torch(arm_norm)
        gripper = torch.where(
            gripper_logit >= 0.0,
            torch.ones_like(gripper_logit),
            -torch.ones_like(gripper_logit),
        )
        action = torch.cat((arm, gripper.unsqueeze(-1)), dim=-1)
        return action, feature


# --------------------------------------------------------------------------- #
# residual policy
# --------------------------------------------------------------------------- #
def compose_action_v2(
    base_action: torch.Tensor,
    residual: torch.Tensor,
    gripper_authority: float,
    flip_margin: float,
) -> torch.Tensor:
    """Arm correction is additive; the gripper flip needs to clear a margin."""

    arm = (base_action[:, :ARM_DIM] + residual[:, :ARM_DIM]).clamp(-1.0, 1.0)
    base_gripper = base_action[:, ARM_DIM]
    score = base_gripper + float(gripper_authority) * residual[:, ARM_DIM]
    flipped = torch.where(score >= 0.0, torch.ones_like(score), -torch.ones_like(score))
    gripper = torch.where(score.abs() > float(flip_margin), flipped, base_gripper)
    return torch.cat((arm, gripper.unsqueeze(-1)), dim=-1)


class ResidualActorV2(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        arm_limit: float = 0.15,
        hidden: int = 512,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.arm_limit = float(arm_limit)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim + ACTION_DIM + PROPRIO_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, ACTION_DIM),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer(
            "limits",
            torch.tensor([self.arm_limit] * ARM_DIM + [1.0], dtype=torch.float32).view(
                1, ACTION_DIM
            ),
        )

    def forward(
        self, feature: torch.Tensor, base_action: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        raw = self.net(torch.cat((feature, base_action, proprio), dim=-1))
        return torch.tanh(raw) * self.limits


class CriticV2(nn.Module):
    def __init__(self, feature_dim: int, oracle_dim: int, hidden: int = 512):
        super().__init__()
        input_dim = int(feature_dim) + int(oracle_dim) + 2 * ACTION_DIM
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 256),
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
        return self.net(torch.cat((feature, oracle, base_action, residual), dim=-1))


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #
class TransitionBuffer:
    FIELDS = (
        "feature",
        "proprio",
        "oracle",
        "base_action",
        "residual",
        "reward",
        "discount",
        "done",
        "next_feature",
        "next_proprio",
        "next_oracle",
        "next_base_action",
    )

    def __init__(self, capacity: int, feature_dim: int, oracle_dim: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)
        half = np.float16
        self.arrays = {
            "feature": np.zeros((capacity, feature_dim), dtype=half),
            "proprio": np.zeros((capacity, PROPRIO_DIM), dtype=half),
            "oracle": np.zeros((capacity, oracle_dim), dtype=half),
            "base_action": np.zeros((capacity, ACTION_DIM), dtype=np.float32),
            "residual": np.zeros((capacity, ACTION_DIM), dtype=np.float32),
            "reward": np.zeros((capacity, 1), dtype=np.float32),
            "discount": np.zeros((capacity, 1), dtype=np.float32),
            "done": np.zeros((capacity, 1), dtype=np.float32),
            "next_feature": np.zeros((capacity, feature_dim), dtype=half),
            "next_proprio": np.zeros((capacity, PROPRIO_DIM), dtype=half),
            "next_oracle": np.zeros((capacity, oracle_dim), dtype=half),
            "next_base_action": np.zeros((capacity, ACTION_DIM), dtype=np.float32),
        }

    def add_batch(self, **values: np.ndarray) -> None:
        if set(values) != set(self.FIELDS):
            raise ValueError(f"replay field mismatch: {sorted(set(values) ^ set(self.FIELDS))}")
        count = int(values["feature"].shape[0])
        if count <= 0:
            return
        indices = (np.arange(count) + self.position) % self.capacity
        for key in self.FIELDS:
            value = np.asarray(values[key])
            if value.ndim == 1:
                value = value[:, None]
            self.arrays[key][indices] = value.astype(self.arrays[key].dtype, copy=False)
        self.position = int((self.position + count) % self.capacity)
        self.size = min(self.capacity, self.size + count)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        return self.rng.integers(0, self.size, size=batch_size)

    def gather(self, indices: np.ndarray, device: torch.device | str) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(array[indices].astype(np.float32)).to(device)
            for key, array in self.arrays.items()
        }


class MixedReplay:
    """Main buffer plus a protected buffer that only ever holds successes."""

    def __init__(
        self,
        capacity: int,
        success_capacity: int,
        feature_dim: int,
        oracle_dim: int,
        seed: int,
    ) -> None:
        self.main = TransitionBuffer(capacity, feature_dim, oracle_dim, seed)
        self.success = TransitionBuffer(success_capacity, feature_dim, oracle_dim, seed + 1)

    @property
    def size(self) -> int:
        return self.main.size + self.success.size

    def add_batch(self, successful: bool, **values: np.ndarray) -> None:
        self.main.add_batch(**values)
        if successful:
            self.success.add_batch(**values)

    def sample(
        self, batch_size: int, success_fraction: float, device: torch.device | str
    ) -> dict[str, torch.Tensor]:
        n_success = 0
        if self.success.size > 0 and success_fraction > 0.0:
            n_success = min(int(round(batch_size * success_fraction)), self.success.size)
        n_main = batch_size - n_success
        if self.main.size < n_main:
            raise ValueError(f"main replay has {self.main.size} rows, needs {n_main}")
        batch = self.main.gather(self.main.sample_indices(n_main), device)
        if n_success > 0:
            extra = self.success.gather(self.success.sample_indices(n_success), device)
            batch = {key: torch.cat((batch[key], extra[key]), dim=0) for key in batch}
        return batch


class ResidualDemoBank:
    """(feature, proprio, base action) -> residual that reproduces the demonstration."""

    def __init__(self, seed: int, capacity: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.capacity = capacity
        self.feature: np.ndarray | None = None
        self.proprio: np.ndarray | None = None
        self.base_action: np.ndarray | None = None
        self.target: np.ndarray | None = None
        self.size = 0
        self.position = 0

    def _allocate(self, feature_dim: int) -> None:
        capacity = int(self.capacity)
        self.feature = np.zeros((capacity, feature_dim), dtype=np.float16)
        self.proprio = np.zeros((capacity, PROPRIO_DIM), dtype=np.float16)
        self.base_action = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.target = np.zeros((capacity, ACTION_DIM), dtype=np.float32)

    def add(
        self,
        feature: np.ndarray,
        proprio: np.ndarray,
        base_action: np.ndarray,
        target: np.ndarray,
    ) -> None:
        count = int(feature.shape[0])
        if count == 0:
            return
        if self.capacity is None:
            self.capacity = max(count, 1)
        if self.feature is None:
            self._allocate(int(feature.shape[1]))
        if count > self.capacity:
            feature, proprio = feature[-self.capacity :], proprio[-self.capacity :]
            base_action, target = base_action[-self.capacity :], target[-self.capacity :]
            count = self.capacity
        indices = (np.arange(count) + self.position) % self.capacity
        self.feature[indices] = feature.astype(np.float16, copy=False)
        self.proprio[indices] = proprio.astype(np.float16, copy=False)
        self.base_action[indices] = base_action
        self.target[indices] = target
        self.position = int((self.position + count) % self.capacity)
        self.size = min(self.capacity, self.size + count)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {
            "feature": torch.from_numpy(self.feature[indices].astype(np.float32)).to(device),
            "proprio": torch.from_numpy(self.proprio[indices].astype(np.float32)).to(device),
            "base_action": torch.from_numpy(self.base_action[indices]).to(device),
            "target": torch.from_numpy(self.target[indices]).to(device),
        }


def demonstration_residual_target(
    demo_action: np.ndarray,
    base_action: np.ndarray,
    arm_limit: float,
    gripper_authority: float,
    flip_margin: float,
) -> np.ndarray:
    """Residual that, composed with the base action, reproduces the demonstration."""

    arm = np.clip(demo_action[:, :ARM_DIM] - base_action[:, :ARM_DIM], -arm_limit, arm_limit)
    demo_gripper = demo_action[:, ARM_DIM]
    base_gripper = base_action[:, ARM_DIM]
    authority = max(float(gripper_authority), 1.0e-3)
    needed = (flip_margin + 0.2 + np.abs(base_gripper)) / authority
    flip = np.where(demo_gripper == base_gripper, 0.0, demo_gripper * needed)
    flip = np.clip(flip, -1.0, 1.0).astype(np.float32)
    return np.concatenate((arm.astype(np.float32), flip[:, None]), axis=1)


# --------------------------------------------------------------------------- #
# agent
# --------------------------------------------------------------------------- #
class ResidualTD3BCAgentV2:
    def __init__(
        self,
        feature_dim: int,
        oracle_dim: int,
        device: torch.device | str,
        arm_limit: float = 0.15,
        actor_lr: float = 1.0e-4,
        critic_lr: float = 3.0e-4,
        weight_decay: float = 1.0e-5,
        discount: float = 0.997,
        tau: float = 0.005,
        policy_noise: float = 0.1,
        noise_clip: float = 0.25,
        policy_delay: int = 2,
        alpha: float = 1.0,
        lambda_max: float = 10.0,
        hidden: int = 512,
    ) -> None:
        self.device = torch.device(device)
        self.feature_dim = int(feature_dim)
        self.oracle_dim = int(oracle_dim)
        self.arm_limit = float(arm_limit)
        self.discount = float(discount)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay)
        self.alpha = float(alpha)
        self.lambda_max = float(lambda_max)
        self.total_updates = 0

        self.actor = ResidualActorV2(feature_dim, arm_limit, hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1 = CriticV2(feature_dim, oracle_dim, hidden).to(self.device)
        self.critic2 = CriticV2(feature_dim, oracle_dim, hidden).to(self.device)
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

    @torch.no_grad()
    def act(
        self,
        feature: torch.Tensor,
        base_action: torch.Tensor,
        proprio: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.actor(feature, base_action, proprio)
        if noise is not None:
            residual = residual + noise * self.actor.limits
            residual = torch.clamp(residual, -self.actor.limits, self.actor.limits)
        return residual

    def train_step(
        self,
        replay: MixedReplay,
        demo_bank: ResidualDemoBank,
        self_imitation: ResidualDemoBank | None,
        batch_size: int,
        demo_batch_size: int,
        success_fraction: float,
        bc_weight: float,
        self_imitation_weight: float,
    ) -> dict[str, float]:
        batch = replay.sample(batch_size, success_fraction, self.device)
        with torch.no_grad():
            noise = (torch.randn_like(batch["residual"]) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_residual = self.actor_target(
                batch["next_feature"], batch["next_base_action"], batch["next_proprio"]
            )
            next_residual = torch.clamp(
                next_residual + noise * self.actor_target.limits,
                -self.actor_target.limits,
                self.actor_target.limits,
            )
            target_q = torch.minimum(
                self.critic1_target(
                    batch["next_feature"],
                    batch["next_oracle"],
                    batch["next_base_action"],
                    next_residual,
                ),
                self.critic2_target(
                    batch["next_feature"],
                    batch["next_oracle"],
                    batch["next_base_action"],
                    next_residual,
                ),
            )
            target = batch["reward"] + batch["discount"] * (1.0 - batch["done"]) * target_q

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
            "q_mean": float(q1.detach().mean()),
            "target_mean": float(target.mean()),
        }
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            actor_residual = self.actor(
                batch["feature"], batch["base_action"], batch["proprio"]
            )
            q_value = self.critic1(
                batch["feature"], batch["oracle"], batch["base_action"], actor_residual
            )
            # TD3+BC normalized weighting: keeps the RL term scale-free. The
            # upper clamp matters early on, when |Q| is still near zero and the
            # unclamped ratio would drown the BC anchor.
            lambda_rl = (
                self.alpha / q_value.abs().mean().detach().clamp(min=1.0e-3)
            ).clamp(max=self.lambda_max)
            actor_rl_loss = -lambda_rl * q_value.mean()

            demo = demo_bank.sample(demo_batch_size, self.device)
            demo_prediction = self.actor(demo["feature"], demo["base_action"], demo["proprio"])
            demo_loss = F.smooth_l1_loss(demo_prediction, demo["target"], beta=0.02)
            actor_loss = actor_rl_loss + bc_weight * demo_loss
            logs["actor_demo_loss"] = float(demo_loss.detach())

            if (
                self_imitation is not None
                and len(self_imitation) >= demo_batch_size
                and self_imitation_weight > 0.0
            ):
                own = self_imitation.sample(demo_batch_size, self.device)
                own_prediction = self.actor(own["feature"], own["base_action"], own["proprio"])
                own_loss = F.smooth_l1_loss(own_prediction, own["target"], beta=0.02)
                actor_loss = actor_loss + self_imitation_weight * own_loss
                logs["actor_self_imitation_loss"] = float(own_loss.detach())

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
                    "lambda_rl": float(lambda_rl),
                    "bc_weight": float(bc_weight),
                    "residual_arm_abs": float(
                        actor_residual[:, :ARM_DIM].abs().mean().detach()
                    ),
                    "residual_flip_abs": float(actor_residual[:, ARM_DIM].abs().mean().detach()),
                }
            )
        return logs

    @torch.no_grad()
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            target_parameter.mul_(1.0 - self.tau).add_(source_parameter, alpha=self.tau)

    def save(self, path: Path, bc_checkpoint: Path, config: dict, task_name: str, extra: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": RESIDUAL_FORMAT,
                "task": validate_task_name(task_name),
                "bc_checkpoint": str(Path(bc_checkpoint).resolve()),
                "feature_dim": self.feature_dim,
                "oracle_dim": self.oracle_dim,
                "arm_limit": self.arm_limit,
                "total_updates": self.total_updates,
                "actor_state_dict": self.actor.state_dict(),
                "critic1_state_dict": self.critic1.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "config": config,
                "extra": extra,
            },
            path,
        )


def load_residual_actor_v2(
    checkpoint: Path, device: torch.device | str, expected_task: str | None = None
) -> tuple[ResidualActorV2, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("format") != RESIDUAL_FORMAT:
        raise ValueError(f"{checkpoint} is not a {RESIDUAL_FORMAT} checkpoint")
    if expected_task is not None and payload.get("task") != expected_task:
        raise ValueError(
            f"residual checkpoint task={payload.get('task')!r}, requested {expected_task!r}"
        )
    actor = ResidualActorV2(
        int(payload["feature_dim"]),
        float(payload["arm_limit"]),
        int(payload["config"].get("actor_hidden", 512)),
    ).to(device)
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor, payload


# --------------------------------------------------------------------------- #
# n-step assembly
# --------------------------------------------------------------------------- #
def build_nstep_transitions(
    episode: dict[str, np.ndarray],
    n_step: int,
    discount: float,
    terminated_at_end: bool,
) -> dict[str, np.ndarray]:
    """Turn one finished episode into n-step transitions with correct bootstrapping.

    ``terminated_at_end`` must be True only for a real MDP terminal (success or
    failure termination), never for a time-limit truncation.
    """

    steps = int(episode["reward"].shape[0])
    rewards = episode["reward"].astype(np.float32)
    returns = np.zeros(steps, dtype=np.float32)
    discounts = np.zeros(steps, dtype=np.float32)
    dones = np.zeros(steps, dtype=np.float32)
    bootstrap = np.zeros(steps, dtype=np.int64)
    for t in range(steps):
        horizon = min(n_step, steps - t)
        acc = 0.0
        factor = 1.0
        for k in range(horizon):
            acc += factor * float(rewards[t + k])
            factor *= discount
        returns[t] = acc
        discounts[t] = factor
        bootstrap[t] = t + horizon - 1  # index whose next_* we bootstrap from
        if t + horizon >= steps and terminated_at_end:
            dones[t] = 1.0
    return {
        "feature": episode["feature"],
        "proprio": episode["proprio"],
        "oracle": episode["oracle"],
        "base_action": episode["base_action"],
        "residual": episode["residual"],
        "reward": returns[:, None],
        "discount": discounts[:, None],
        "done": dones[:, None],
        "next_feature": episode["next_feature"][bootstrap],
        "next_proprio": episode["next_proprio"][bootstrap],
        "next_oracle": episode["next_oracle"][bootstrap],
        "next_base_action": episode["next_base_action"][bootstrap],
    }


class EpisodeRecorder:
    """Per-environment step accumulation until the episode ends."""

    FIELDS = (
        "feature",
        "proprio",
        "oracle",
        "base_action",
        "residual",
        "reward",
        "next_feature",
        "next_proprio",
        "next_oracle",
        "next_base_action",
    )

    def __init__(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        self.data: list[dict[str, list[np.ndarray]]] = [
            {key: [] for key in self.FIELDS} for _ in range(self.num_envs)
        ]

    def append(self, env_id: int, **values: np.ndarray) -> None:
        target = self.data[env_id]
        for key in self.FIELDS:
            target[key].append(values[key])

    def length(self, env_id: int) -> int:
        return len(self.data[env_id]["reward"])

    def pop(self, env_id: int) -> dict[str, np.ndarray]:
        target = self.data[env_id]
        episode = {
            key: np.asarray(np.stack(value), dtype=np.float32) for key, value in target.items()
        }
        self.data[env_id] = {key: [] for key in self.FIELDS}
        return episode

    def clear(self) -> None:
        self.data = [{key: [] for key in self.FIELDS} for _ in range(self.num_envs)]


# --------------------------------------------------------------------------- #
# evaluation shared by both eval scripts and by training
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout_evaluate(
    env,
    runner: ChunkRunner,
    device: torch.device | str,
    episodes: int,
    agent: ResidualTD3BCAgentV2 | None = None,
    actor: ResidualActorV2 | None = None,
    gripper_authority: float = 0.0,
    flip_margin: float = 0.25,
    capture_env: int | None = None,
    capture_episodes: int = 0,
    video_callback: Callable[[dict, Sequence[np.ndarray]], str | None] | None = None,
    reset_seed: int | None = None,
) -> dict:
    """Deterministic closed-loop evaluation. Returns success rate and per-episode rows."""

    policy_actor = actor if actor is not None else (agent.actor if agent is not None else None)
    if reset_seed is None:
        observations, _ = env.reset()
    else:
        observations, _ = env.reset(seed=int(reset_seed))
    num_envs = runner.num_envs
    runner.mark_reset()
    episode_returns = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    rows: list[dict] = []
    videos: list[object] = []
    capture_all_envs = capture_episodes > 0 and capture_env is None
    capture_ids = (
        set(range(num_envs))
        if capture_all_envs
        else ({int(capture_env)} if capture_episodes > 0 and capture_env is not None else set())
    )
    frames: dict[int, list[np.ndarray]] = {env_id: [] for env_id in capture_ids}
    captured = 0

    while len(rows) < episodes:
        rgb, proprio = policy_observation(observations, device)
        if captured < capture_episodes:
            for env_id in capture_ids:
                frames[env_id].append(
                    rgb[env_id]
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .round()
                    .byte()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
        base_action, feature = runner.step(rgb, proprio)
        if policy_actor is not None:
            residual = policy_actor(feature, base_action, proprio)
            action = compose_action_v2(base_action, residual, gripper_authority, flip_margin)
        else:
            action = base_action
        observations, rewards, terminated, truncated, _ = env.step(action)
        success = environment_success(env)
        episode_returns += rewards
        episode_lengths += 1
        done = terminated | truncated
        if bool(done.any()):
            for env_id in done.nonzero(as_tuple=False).flatten().tolist():
                row = None
                if len(rows) < episodes:
                    row = {
                        "episode": len(rows),
                        "env_id": int(env_id),
                        "success": bool(success[env_id].item()),
                        "return": float(episode_returns[env_id].item()),
                        "length": int(episode_lengths[env_id].item()),
                    }
                    rows.append(row)
                if env_id in frames:
                    if row is not None and captured < capture_episodes and frames[env_id]:
                        if video_callback is None:
                            videos.append(frames[env_id])
                        else:
                            result = video_callback(row, frames[env_id])
                            if result is not None:
                                videos.append(result)
                        captured += 1
                    frames[env_id] = []
                episode_returns[env_id] = 0.0
                episode_lengths[env_id] = 0
            runner.mark_reset(done)
    successes = sum(int(row["success"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / max(len(rows), 1),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_length": float(np.mean([row["length"] for row in rows])),
        "results": rows,
        "videos": videos,
    }


def encode_video(frames: Sequence[np.ndarray], output: Path, fps: float = 20.0) -> None:
    import subprocess

    if not frames:
        raise ValueError("cannot encode an empty video")
    array = np.stack(frames).astype(np.uint8, copy=False)
    height, width = array.shape[1:3]
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", f"{fps:.6f}",
        "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    subprocess.run(command, input=array.tobytes(), check=True)


def json_line(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
