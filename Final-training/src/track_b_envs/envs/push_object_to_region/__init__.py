"""Canonical public entry point for PushObjectToRegion."""

__all__ = ["PushCubeToRegionEnvCfg", "PushObjectToRegionEnvCfg"]


def __getattr__(name: str):
    if name == "PushObjectToRegionEnvCfg":
        from .push_object_to_region_env_cfg import PushObjectToRegionEnvCfg

        return PushObjectToRegionEnvCfg
    if name == "PushCubeToRegionEnvCfg":
        from .push_cube_to_region_env_cfg import PushCubeToRegionEnvCfg

        return PushCubeToRegionEnvCfg
    raise AttributeError(name)
