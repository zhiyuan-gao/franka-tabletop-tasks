"""Cup placement tasks sharing the calibrated open-drawer cabinet."""

__all__ = ["CupToCabinetTopEnvCfg", "CupToDrawerEnvCfg"]


def __getattr__(name):
    if name in __all__:
        from .cup_cabinet_env_cfg import (
            CupToCabinetTopEnvCfg,
            CupToDrawerEnvCfg,
        )

        return {
            "CupToCabinetTopEnvCfg": CupToCabinetTopEnvCfg,
            "CupToDrawerEnvCfg": CupToDrawerEnvCfg,
        }[name]
    raise AttributeError(name)
