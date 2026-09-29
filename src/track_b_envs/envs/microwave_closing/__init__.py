"""Passive physical microwave door closing task."""

def __getattr__(name):
    if name == 'MicrowaveClosingEnvCfg':
        from .microwave_env_cfg import MicrowaveClosingEnvCfg
        return MicrowaveClosingEnvCfg
    raise AttributeError(name)
