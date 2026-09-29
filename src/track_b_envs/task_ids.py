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
    "cup_to_drawer": "TrackB-CupToDrawer-Franka-IK-Rel-Visual-v0",
    "cup_to_cabinet_top": "TrackB-CupToCabinetTop-Franka-IK-Rel-Visual-v0",
    "open_upper_drawer": "TrackB-OpenUpperDrawer-Franka-IK-Rel-Visual-v0",
    "open_lower_drawer": "TrackB-OpenLowerDrawer-Franka-IK-Rel-Visual-v0",
    "put_object_in_upper_drawer": "TrackB-PutObjectInUpperDrawer-Franka-IK-Rel-Visual-v0",
    "put_object_in_lower_drawer": "TrackB-PutObjectInLowerDrawer-Franka-IK-Rel-Visual-v0",
    "put_object_in_upper_drawer_and_close": "TrackB-PutObjectInUpperDrawerAndClose-Franka-IK-Rel-Visual-v0",
    "put_object_in_lower_drawer_and_close": "TrackB-PutObjectInLowerDrawerAndClose-Franka-IK-Rel-Visual-v0",
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
    "cup_to_drawer": (
        "track_b_envs.envs.cup_cabinet:CupToDrawerEnvCfg"
    ),
    "cup_to_cabinet_top": (
        "track_b_envs.envs.cup_cabinet:CupToCabinetTopEnvCfg"
    ),
    "open_upper_drawer": (
        "track_b_envs.envs.drawer_subtasks:OpenUpperDrawerEnvCfg"
    ),
    "open_lower_drawer": (
        "track_b_envs.envs.drawer_subtasks:OpenLowerDrawerEnvCfg"
    ),
    "put_object_in_upper_drawer": (
        "track_b_envs.envs.drawer_subtasks:PutObjectInUpperDrawerEnvCfg"
    ),
    "put_object_in_lower_drawer": (
        "track_b_envs.envs.drawer_subtasks:PutObjectInLowerDrawerEnvCfg"
    ),
    "put_object_in_upper_drawer_and_close": (
        "track_b_envs.envs.drawer_subtasks:PutObjectInUpperDrawerAndCloseEnvCfg"
    ),
    "put_object_in_lower_drawer_and_close": (
        "track_b_envs.envs.drawer_subtasks:PutObjectInLowerDrawerAndCloseEnvCfg"
    ),
}

TASK_NAMES = tuple(TASK_IDS)

if TASK_IDS.keys() != ENV_CFG_ENTRY_POINTS.keys():
    raise RuntimeError("task IDs and environment config entry points are inconsistent")

__all__ = ["ENV_CFG_ENTRY_POINTS", "TASK_IDS", "TASK_NAMES"]
