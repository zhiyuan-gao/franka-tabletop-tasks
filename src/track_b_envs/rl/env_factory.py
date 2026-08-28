"""Environment construction helpers shared by trainer adapters."""

from __future__ import annotations

import importlib


def load_registered_env_cfg(
    task_name: str,
    *,
    device: str,
    num_envs: int,
    seed: int,
    use_fabric: bool = True,
):
    """Instantiate a Track B config through its Gym registration.

    This avoids importing the complete ``isaaclab_tasks`` registry, which is
    unnecessary for this standalone package and can load unrelated optional
    extensions.
    """

    import gymnasium as gym

    from track_b_envs import TASK_IDS

    if task_name not in TASK_IDS:
        raise KeyError(f"unknown Track B task: {task_name!r}")
    task_id = TASK_IDS[task_name]
    entry_point = gym.spec(task_id).kwargs["env_cfg_entry_point"]
    module_name, class_name = entry_point.split(":", maxsplit=1)
    cfg_cls = getattr(importlib.import_module(module_name), class_name)
    cfg = cfg_cls()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.sim.use_fabric = use_fabric
    cfg.seed = seed
    return task_id, cfg


def configure_normalized_action_space(env) -> None:
    """Advertise and enforce the normalized seven-dimensional action contract.

    Isaac Lab manager environments expose an unbounded Gym Box even when each
    action term expects normalized commands. SKRL uses the Gym limits when
    clipping Gaussian samples, so the metadata must match the real contract.
    """

    import gymnasium as gym
    import numpy as np

    base = env.unwrapped
    action_dim = base.action_manager.total_action_dim
    if action_dim != 7:
        raise RuntimeError(f"expected the Track B 7D action contract, got {action_dim}")
    base.single_action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(action_dim,),
        dtype=np.float32,
    )
    base.action_space = gym.vector.utils.batch_space(
        base.single_action_space, base.num_envs
    )
