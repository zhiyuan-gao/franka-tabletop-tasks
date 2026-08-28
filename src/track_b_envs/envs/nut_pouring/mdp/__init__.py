"""MDP terms for NutPouring."""

from .events import reset_nut_pouring_layout
from .geometry import (
    BOWL_ROOT_Z_LOWER_M,
    BOWL_ROOT_Z_UPPER_M,
    BOWL_SAFE_RADIUS_M,
    nut_inside_bowl,
    nut_relative_to_bowl,
)
from .observations import oracle_nut_pouring_state
from .rewards import (
    cup_above_bowl,
    cup_lift_progress,
    cup_tilt_progress,
    ee_cup_proximity,
    nut_bowl_proximity,
    nut_inside_bowl_bonus,
    settled_success_bonus,
)
from .terminations import (
    instantaneous_nut_in_bowl,
    nut_or_cup_out_of_workspace,
    settled_nut_in_bowl,
)

__all__ = [
    "BOWL_ROOT_Z_LOWER_M",
    "BOWL_ROOT_Z_UPPER_M",
    "BOWL_SAFE_RADIUS_M",
    "cup_above_bowl",
    "cup_lift_progress",
    "cup_tilt_progress",
    "ee_cup_proximity",
    "instantaneous_nut_in_bowl",
    "nut_bowl_proximity",
    "nut_inside_bowl",
    "nut_inside_bowl_bonus",
    "nut_or_cup_out_of_workspace",
    "nut_relative_to_bowl",
    "oracle_nut_pouring_state",
    "reset_nut_pouring_layout",
    "settled_nut_in_bowl",
    "settled_success_bonus",
]
