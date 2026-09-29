"""Load and materialize the project-local SKRL PPO configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def load_visual_ppo_config(path=None) -> dict:
    import yaml

    if path is None:
        from . import default_config_path

        resource = default_config_path()
        with resource.open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
    else:
        with Path(path).open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict) or not {"model", "agent", "trainer"} <= set(cfg):
        raise ValueError("SKRL config must contain model, agent, and trainer mappings")
    return cfg


def build_ppo_config(raw_cfg: dict, *, state_space, device, experiment_dir, experiment_name) -> dict:
    """Convert YAML values into the SKRL 2.1 PPO configuration mapping."""

    from skrl.resources.preprocessors.torch import RunningStandardScaler

    cfg = deepcopy(raw_cfg["agent"])
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": state_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["experiment"] = deepcopy(raw_cfg.get("experiment", {}))
    cfg["experiment"]["directory"] = str(experiment_dir)
    cfg["experiment"]["experiment_name"] = experiment_name
    return cfg
