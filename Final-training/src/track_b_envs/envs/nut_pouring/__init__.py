"""Physical cup-to-bowl nut-pouring environment."""

__all__ = ["NutPouringEnvCfg"]


def __getattr__(name):
    if name == "NutPouringEnvCfg":
        from .nut_pouring_env_cfg import NutPouringEnvCfg

        return NutPouringEnvCfg
    raise AttributeError(name)
