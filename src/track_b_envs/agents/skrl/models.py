"""Multimodal actor and privileged critic for SKRL PPO."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch
from torch import nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


@dataclass(frozen=True)
class _FlatPolicyLayout:
    rgb_slice: slice
    proprio_slice: slice
    rgb_shape: tuple[int, int, int]
    proprio_size: int
    total_size: int

    @classmethod
    def from_space(cls, space: gym.Space) -> "_FlatPolicyLayout":
        if not isinstance(space, gym.spaces.Dict):
            raise TypeError("visual policy observation space must be gym.spaces.Dict")
        if set(space.spaces) != {"rgb", "proprio"}:
            raise ValueError(
                f"expected rgb/proprio policy terms, got {tuple(space.spaces)}"
            )

        offsets: dict[str, slice] = {}
        cursor = 0
        # SKRL flatten_tensorized_space sorts dictionary keys.
        for key in sorted(space.spaces):
            size = gym.spaces.flatdim(space.spaces[key])
            offsets[key] = slice(cursor, cursor + size)
            cursor += size

        rgb_shape = tuple(space.spaces["rgb"].shape)
        proprio_size = gym.spaces.flatdim(space.spaces["proprio"])
        if rgb_shape != (3, 224, 224):
            raise ValueError(f"expected one 3-channel 224x224 view, got {rgb_shape}")
        if proprio_size != 25:
            raise ValueError(f"expected 25D Franka proprioception, got {proprio_size}")
        return cls(
            rgb_slice=offsets["rgb"],
            proprio_slice=offsets["proprio"],
            rgb_shape=rgb_shape,
            proprio_size=proprio_size,
            total_size=cursor,
        )


class VisualGaussianPolicy(GaussianMixin, Model):
    """Single-view RGB + proprio Gaussian actor.

    The SKRL Isaac Lab wrapper supplies a flattened tensor. Slices are derived
    from the registered Gym Dict space, so their order is never assumed by the
    training script.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        *,
        image_feature_dim: int = 256,
        proprio_feature_dim: int = 64,
        hidden_dims: tuple[int, int] = (256, 128),
        initial_log_std: float = -1.5,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_mean_actions=True,
            clip_log_std=True,
            min_log_std=-5.0,
            max_log_std=1.0,
            reduction="sum",
            role="policy",
        )
        self.layout = _FlatPolicyLayout.from_space(observation_space)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, image_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.proprio_encoder = nn.Sequential(
            nn.LayerNorm(self.layout.proprio_size),
            nn.Linear(self.layout.proprio_size, proprio_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proprio_feature_dim, proprio_feature_dim),
            nn.ReLU(inplace=True),
        )
        fusion_dim = image_feature_dim + proprio_feature_dim
        self.action_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[0]),
            nn.ELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ELU(),
            nn.Linear(hidden_dims[1], self.num_actions),
            nn.Tanh(),
        )
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), float(initial_log_std))
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.action_head[-2]
        nn.init.orthogonal_(final.weight, gain=0.01)

    def compute(self, inputs, role=""):
        flat = inputs["observations"]
        if flat.shape[-1] != self.layout.total_size:
            raise RuntimeError(
                f"policy expected {self.layout.total_size} values, got {flat.shape[-1]}"
            )
        rgb = flat[:, self.layout.rgb_slice].reshape(-1, *self.layout.rgb_shape)
        proprio = flat[:, self.layout.proprio_slice]
        image_features = self.image_encoder(rgb.clamp(0.0, 1.0))
        proprio_features = self.proprio_encoder(proprio)
        mean_actions = self.action_head(
            torch.cat((image_features, proprio_features), dim=-1)
        )
        return mean_actions, {"log_std": self.log_std_parameter}


class OracleValue(DeterministicMixin, Model):
    """Value network that consumes only the simulator-provided critic state."""

    def __init__(
        self,
        state_space,
        device,
        *,
        hidden_dims: tuple[int, int, int] = (256, 256, 128),
    ) -> None:
        Model.__init__(self, state_space=state_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False, role="value")
        layers: list[nn.Module] = []
        input_dim = self.num_states
        for output_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, output_dim), nn.ELU()))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 1))
        self.value_net = nn.Sequential(*layers)
        for module in self.value_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def compute(self, inputs, role=""):
        states = inputs.get("states")
        if states is None:
            raise RuntimeError("oracle critic state is required by the value network")
        return self.value_net(states), {}
