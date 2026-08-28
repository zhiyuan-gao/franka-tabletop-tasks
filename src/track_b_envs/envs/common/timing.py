"""Shared physical-time contract for environments and scripted experts."""

from __future__ import annotations

import math


PHYSICS_RATE_HZ = 100
POLICY_CONTROL_RATE_HZ = 20
PHYSICS_DT_S = 1.0 / PHYSICS_RATE_HZ
CONTROL_DT_S = 1.0 / POLICY_CONTROL_RATE_HZ
CONTROL_DECIMATION = PHYSICS_RATE_HZ // POLICY_CONTROL_RATE_HZ

SETTLE_DURATION_S = 0.30
SETTLE_STEPS = math.ceil(SETTLE_DURATION_S / CONTROL_DT_S - 1.0e-9)


def environment_control_dt_s(env) -> float:
    """Return and validate the high-level action period configured by Isaac Lab."""

    control_dt_s = float(env.cfg.sim.dt * env.cfg.decimation)
    if control_dt_s <= 0.0:
        raise ValueError("environment control period must be positive")
    return control_dt_s


def duration_reached(wait_steps, control_dt_s: float, duration_s: float):
    """Compare a per-state integer counter against a duration in seconds."""

    if duration_s < 0.0:
        raise ValueError("duration must be non-negative")
    return wait_steps.float() * control_dt_s >= duration_s - 1.0e-9


def distance_per_step(speed_m_s: float, control_dt_s: float) -> float:
    """Convert a Cartesian speed to a control-step displacement bound."""

    if speed_m_s < 0.0 or control_dt_s <= 0.0:
        raise ValueError("speed and control period must be non-negative/positive")
    return speed_m_s * control_dt_s


def angle_per_step(speed_rad_s: float, control_dt_s: float) -> float:
    """Convert an angular speed to a control-step rotation bound."""

    if speed_rad_s < 0.0 or control_dt_s <= 0.0:
        raise ValueError("speed and control period must be non-negative/positive")
    return speed_rad_s * control_dt_s
