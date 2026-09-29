"""PushCubeToRegion variant with a low-profile procedural red cube."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from track_b_envs.envs.common.constants import (
    FOREGROUND_SEMANTIC_TAG,
    PUSH_CUBE_COLOR_RGB,
    PUSH_CUBE_HALF_EXTENTS_XY_M,
    PUSH_CUBE_MASS_KG,
    PUSH_CUBE_SIZE_M,
)

from .push_object_to_region_env_cfg import PushObjectToRegionEnvCfg


@configclass
class PushCubeToRegionEnvCfg(PushObjectToRegionEnvCfg):
    """Push a 7 x 7 x 5 cm red cube into the existing printed region."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/RedPushCube",
            spawn=sim_utils.CuboidCfg(
                size=PUSH_CUBE_SIZE_M,
                semantic_tags=[FOREGROUND_SEMANTIC_TAG],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=2,
                    max_depenetration_velocity=3.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=PUSH_CUBE_MASS_KG),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.003,
                    rest_offset=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=PUSH_CUBE_COLOR_RGB,
                    roughness=0.75,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.46, -0.305, PUSH_CUBE_SIZE_M[2] / 2.0 + 0.0005)
            ),
        )
        self.events.object_mass.params["mass_distribution_params"] = (
            PUSH_CUBE_MASS_KG,
            PUSH_CUBE_MASS_KG,
        )
        for term in (self.rewards.task_success, self.terminations.success):
            term.params["object_half_extents_xy"] = PUSH_CUBE_HALF_EXTENTS_XY_M
