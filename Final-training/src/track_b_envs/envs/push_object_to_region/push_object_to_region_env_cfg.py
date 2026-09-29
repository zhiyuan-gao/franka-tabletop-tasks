"""ManagerBasedRLEnv config for pushing a stable YCB object to a region."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import (
    FrankaCubeLiftEnvCfg,
)

from track_b_envs.envs.common.constants import (
    CALIBRATION,
    FOREGROUND_SEMANTIC_TAG,
    WORLD_MARKER_CENTER_M,
    WORLD_MARKER_NORMAL_M,
    WORLD_MARKER_ROTATION_WXYZ,
    YCB_SUGAR_BOX_USD,
)
from track_b_envs.envs.common.observations import composited_policy_rgb, franka_proprio
from track_b_envs.envs.common.scene import apply_calibrated_scene
from track_b_envs.envs.common.timing import CONTROL_DECIMATION, PHYSICS_DT_S

from . import mdp


ROBOT = SceneEntityCfg("robot")
EE_FRAME = SceneEntityCfg("ee_frame")
OBJECT = SceneEntityCfg("object")
TARGET = SceneEntityCfg("target_region")
CAMERA = SceneEntityCfg("policy_camera")


@configclass
class PushObjectToRegionObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        rgb = ObsTerm(func=composited_policy_rgb, params={"sensor_cfg": CAMERA})
        proprio = ObsTerm(
            func=franka_proprio,
            params={"robot_cfg": ROBOT, "ee_frame_cfg": EE_FRAME},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        full_state = ObsTerm(
            func=mdp.oracle_push_state,
            params={
                "robot_cfg": ROBOT,
                "ee_frame_cfg": EE_FRAME,
                "object_cfg": OBJECT,
                "target_cfg": TARGET,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class PushObjectToRegionEventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_layout = EventTerm(func=mdp.reset_push_layout, mode="reset")
    object_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": OBJECT,
            "mass_distribution_params": (0.08, 0.08),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    object_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": OBJECT,
            "static_friction_range": (0.50, 0.50),
            "dynamic_friction_range": (0.40, 0.40),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )


@configclass
class PushObjectToRegionRewardsCfg:
    reach_object = RewTerm(
        func=mdp.ee_object_proximity,
        weight=0.25,
        params={"object_cfg": OBJECT, "ee_frame_cfg": EE_FRAME, "std": 0.12},
    )
    move_to_target = RewTerm(
        func=mdp.object_target_proximity,
        weight=2.0,
        params={"object_cfg": OBJECT, "target_cfg": TARGET, "std": 0.20},
    )
    keep_upright = RewTerm(func=mdp.object_tilt_penalty, weight=-0.5, params={"object_cfg": OBJECT})
    task_success = RewTerm(
        func=mdp.success_bonus,
        weight=15.0,
        params={"object_cfg": OBJECT, "target_cfg": TARGET},
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-4)
    joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-1.0e-4, params={"asset_cfg": ROBOT})


@configclass
class PushObjectToRegionTerminationsCfg:
    success = DoneTerm(
        func=mdp.object_footprint_in_target,
        params={"object_cfg": OBJECT, "target_cfg": TARGET},
    )
    tipped = DoneTerm(func=mdp.object_tipped, params={"object_cfg": OBJECT})
    out_of_workspace = DoneTerm(func=mdp.object_out_of_workspace, params={"object_cfg": OBJECT})
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class PushObjectToRegionEnvCfg(FrankaCubeLiftEnvCfg):
    observations: PushObjectToRegionObservationsCfg = PushObjectToRegionObservationsCfg()
    rewards: PushObjectToRegionRewardsCfg = PushObjectToRegionRewardsCfg()
    terminations: PushObjectToRegionTerminationsCfg = PushObjectToRegionTerminationsCfg()
    events: PushObjectToRegionEventsCfg = PushObjectToRegionEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = None
        self.curriculum = None
        CALIBRATION.validate()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        apply_calibrated_scene(self.scene, include_camera=True)

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=YCB_SUGAR_BOX_USD,
                scale=(0.80, 0.80, 0.80),
                semantic_tags=[FOREGROUND_SEMANTIC_TAG],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=5.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.46, -0.305, 0.018)),
        )
        sheet_center = tuple(
            WORLD_MARKER_CENTER_M[index] + 0.0005 * WORLD_MARKER_NORMAL_M[index]
            for index in range(3)
        )
        target_center = tuple(
            WORLD_MARKER_CENTER_M[index] + 0.0015 * WORLD_MARKER_NORMAL_M[index]
            for index in range(3)
        )
        self.scene.target_sheet = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetSheet",
            spawn=sim_utils.CuboidCfg(
                # ISO A4 in landscape orientation, aligned to the calibrated marker.
                size=(0.297, 0.210, 0.001),
                semantic_tags=[FOREGROUND_SEMANTIC_TAG],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.94, 0.94, 0.94),
                    roughness=0.95,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=sheet_center,
                rot=WORLD_MARKER_ROTATION_WXYZ,
            ),
        )
        self.scene.target_region = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/PrintedTargetRectangle",
            spawn=sim_utils.CuboidCfg(
                size=(0.180, 0.120, 0.001),
                semantic_tags=[FOREGROUND_SEMANTIC_TAG],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 0.25, 0.85),
                    emissive_color=(0.005, 0.01, 0.03),
                    roughness=0.9,
                ),
            ),
            # Printed rectangle centered on the A4 sheet that covers the marker.
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=target_center,
                rot=WORLD_MARKER_ROTATION_WXYZ,
            ),
        )

        self.actions.arm_action.scale = (0.05, 0.05, 0.05, 0.08, 0.08, 0.08)
        self.decimation = CONTROL_DECIMATION
        self.episode_length_s = 45.0
        self.sim.dt = PHYSICS_DT_S
        self.sim.render_interval = self.decimation
        self.sim.physx.friction_correlation_distance = 0.00625
