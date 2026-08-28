"""SKRL visual PPO models and configuration."""

from importlib.resources import files

from .config import build_ppo_config, load_visual_ppo_config
from .models import OracleValue, VisualGaussianPolicy


def default_config_path():
    return files(__package__).joinpath("ppo_visual.yaml")


__all__ = [
    "OracleValue",
    "VisualGaussianPolicy",
    "build_ppo_config",
    "default_config_path",
    "load_visual_ppo_config",
]
