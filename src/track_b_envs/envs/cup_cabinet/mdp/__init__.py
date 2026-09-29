"""MDP terms for cup/cabinet placement."""

from .geometry import TARGET_BOUNDS_M, cup_inside_target, cup_is_upright, cup_relative_to_target
from .observations import oracle_cup_cabinet_state
from .rewards import (
    cup_at_target_bonus,
    cup_lift_progress,
    cup_target_proximity,
    ee_cup_proximity,
    settled_success_bonus,
)
from .terminations import (
    cup_dropped_or_out_of_workspace,
    instantaneous_cup_at_target,
    settled_cup_at_target,
)

__all__ = [
    "TARGET_BOUNDS_M",
    "cup_at_target_bonus",
    "cup_dropped_or_out_of_workspace",
    "cup_inside_target",
    "cup_is_upright",
    "cup_lift_progress",
    "cup_relative_to_target",
    "cup_target_proximity",
    "ee_cup_proximity",
    "instantaneous_cup_at_target",
    "oracle_cup_cabinet_state",
    "settled_cup_at_target",
    "settled_success_bonus",
]
