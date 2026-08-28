"""Shared reset ranges for expert collection and RL environments."""

from __future__ import annotations

import math


CABINET_XY_OFFSET_RANGE_M = (-0.02, 0.02)
CABINET_YAW_OFFSET_RANGE_RAD = (-math.radians(5.0), math.radians(5.0))

BANANA_LOCAL_XY_OFFSET_RANGE_M = (-0.02, 0.02)
BANANA_LOCAL_YAW_OFFSET_RANGE_RAD = (-math.radians(10.0), math.radians(10.0))

CUP_XY_OFFSET_RANGE_M = (-0.015, 0.015)
CUP_YAW_OFFSET_RANGE_RAD = (-math.radians(8.0), math.radians(8.0))
BOWL_XY_OFFSET_RANGE_M = (-0.015, 0.015)
CUP_BOWL_DISTANCE_RANGE_M = (0.13, 0.21)
