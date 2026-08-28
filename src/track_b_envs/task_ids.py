"""Stable task identifiers and configuration entry points.

This module deliberately has no Isaac Sim imports.  It can therefore be used
by command-line tools to validate task names before launching the simulator.
"""

TASK_IDS = {
    "push_object_to_region": "TrackB-PushObjectToRegion-Franka-IK-Rel-Visual-v0",
    "push_cube_to_region": "TrackB-PushCubeToRegion-Franka-IK-Rel-Visual-v0",
    "drawer_opening": "TrackB-DrawerOpening-Franka-IK-Rel-Visual-v0",
    "cabinet_opening": "TrackB-CabinetOpening-Franka-IK-Rel-Visual-v0",
    "object_to_drawer": "TrackB-ObjectToDrawer-Franka-IK-Rel-Visual-v0",
    "nut_pouring": "TrackB-NutPouring-Franka-IK-Rel-Visual-v0",
}

ENV_CFG_ENTRY_POINTS = {
    "push_object_to_region": (
        "track_b_envs.envs.push_object_to_region:PushObjectToRegionEnvCfg"
    ),
    "push_cube_to_region": (
        "track_b_envs.envs.push_object_to_region:PushCubeToRegionEnvCfg"
    ),
    "drawer_opening": (
        "track_b_envs.envs.articulated_opening:DrawerOpeningEnvCfg"
    ),
    "cabinet_opening": (
        "track_b_envs.envs.articulated_opening:CabinetOpeningEnvCfg"
    ),
    "object_to_drawer": (
        "track_b_envs.envs.object_to_drawer:ObjectToDrawerEnvCfg"
    ),
    "nut_pouring": "track_b_envs.envs.nut_pouring:NutPouringEnvCfg",
}

TASK_NAMES = tuple(TASK_IDS)

if TASK_IDS.keys() != ENV_CFG_ENTRY_POINTS.keys():
    raise RuntimeError("task IDs and environment config entry points are inconsistent")

__all__ = ["ENV_CFG_ENTRY_POINTS", "TASK_IDS", "TASK_NAMES"]
