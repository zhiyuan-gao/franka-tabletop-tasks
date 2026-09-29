"""Standalone Track B Isaac Lab environments and Gym registrations."""

from .task_ids import ENV_CFG_ENTRY_POINTS, TASK_IDS, TASK_NAMES


def register_environments() -> None:
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return
    for task_name, cfg_entry_point in ENV_CFG_ENTRY_POINTS.items():
        task_id = TASK_IDS[task_name]
        if task_id not in gym.registry:
            gym.register(
                id=task_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": cfg_entry_point,
                    "skrl_cfg_entry_point": (
                        "track_b_envs.agents.skrl:ppo_visual.yaml"
                    ),
                },
            )


register_environments()

__all__ = [
    "ENV_CFG_ENTRY_POINTS",
    "TASK_IDS",
    "TASK_NAMES",
    "register_environments",
]
