"""Calibrated upper/lower drawer task family."""

_CONFIG_NAMES = {
    "DrawerSubtaskEnvCfg",
    "OpenUpperDrawerEnvCfg",
    "OpenLowerDrawerEnvCfg",
    "PutObjectInUpperDrawerEnvCfg",
    "PutObjectInLowerDrawerEnvCfg",
    "PutObjectInUpperDrawerAndCloseEnvCfg",
    "PutObjectInLowerDrawerAndCloseEnvCfg",
}

__all__ = sorted(_CONFIG_NAMES)


def __getattr__(name):
    if name in _CONFIG_NAMES:
        from . import drawer_subtasks_env_cfg

        return getattr(drawer_subtasks_env_cfg, name)
    raise AttributeError(name)

