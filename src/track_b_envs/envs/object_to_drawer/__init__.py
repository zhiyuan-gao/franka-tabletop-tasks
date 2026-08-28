"""Long-horizon banana-to-drawer environment."""

__all__ = ["ObjectToDrawerEnvCfg"]


def __getattr__(name):
    # Keep pure geometry/test modules importable outside a running Isaac app.
    if name == "ObjectToDrawerEnvCfg":
        from .object_to_drawer_env_cfg import ObjectToDrawerEnvCfg

        return ObjectToDrawerEnvCfg
    raise AttributeError(name)
