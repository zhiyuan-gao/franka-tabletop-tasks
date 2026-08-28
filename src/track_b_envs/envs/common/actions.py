"""Project-local action terms that preserve the compact 7D task interface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class SymmetricFingerPositionAction(ActionTerm):
    """Map one normalized scalar to equal Panda finger-position targets."""

    cfg: "SymmetricFingerPositionActionCfg"

    def __init__(self, cfg: "SymmetricFingerPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        if not isinstance(self._asset, Articulation):
            raise TypeError("symmetric finger action requires an Articulation")
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        if len(self._joint_ids) != 2:
            raise RuntimeError(
                f"expected two finger joints, found {self._joint_names}"
            )
        self._raw_actions = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float32
        )
        self._processed_actions = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float32
        )

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        target = self._raw_actions * self.cfg.scale + self.cfg.offset
        self._processed_actions[:] = target.expand(-1, 2)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self.cfg.offset


@configclass
class SymmetricFingerPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`SymmetricFingerPositionAction`."""

    joint_names: list[str] = MISSING
    scale: float = 0.02
    offset: float = 0.02
    class_type: type[ActionTerm] = SymmetricFingerPositionAction
