"""ManagerBasedRLEnv config for the full banana-to-drawer task."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
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

from track_b_envs.envs.common.constants import CALIBRATION
from track_b_envs.envs.common.observations import composited_policy_rgb, franka_proprio
from track_b_envs.envs.common.scene import apply_calibrated_scene
from track_b_envs.envs.common.task_scene import apply_task_assets_to_scene
from track_b_envs.envs.common.timing import (
    CONTROL_DECIMATION,
    PHYSICS_DT_S,
    SETTLE_STEPS,
)

from . import mdp


ROBOT = SceneEntityCfg("robot")
FINGERS = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
EE_FRAME = SceneEntityCfg("ee_frame")
OBJECT = SceneEntityCfg("object")
HANDLE_FRAME = SceneEntityCfg("cabinet_frame")
TARGET_FRAME = SceneEntityCfg("drawer_interior_frame")
DRAWER = SceneEntityCfg("cabinet", joint_names=["UpperDrawerJoint"])
CAMERA = SceneEntityCfg("policy_camera")


@configclass
class ObjectToDrawerObservationsCfg:
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
            func=mdp.oracle_object_to_drawer_state,
            params={
                "robot_cfg": ROBOT,
                "ee_frame_cfg": EE_FRAME,
                "object_cfg": OBJECT,
                "handle_frame_cfg": HANDLE_FRAME,
                "target_frame_cfg": TARGET_FRAME,
                "cabinet_cfg": DRAWER,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObjectToDrawerEventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_layout = EventTerm(func=mdp.reset_object_to_drawer_layout, mode="reset")


@configclass
class ObjectToDrawerRewardsCfg:
    reach_handle = RewTerm(
        func=mdp.ee_task_target_proximity,
        weight=0.15,
        params={"ee_frame_cfg": EE_FRAME, "target_frame_cfg": HANDLE_FRAME},
    )
    drawer_open = RewTerm(
        func=mdp.drawer_opening_progress,
        weight=1.0,
        params={"cabinet_cfg": DRAWER},
    )
    reach_object = RewTerm(
        func=mdp.ee_object_proximity,
        weight=0.25,
        params={"ee_frame_cfg": EE_FRAME, "object_cfg": OBJECT},
    )
    lift_object = RewTerm(
        func=mdp.object_vertical_progress,
        weight=1.0,
        params={"object_cfg": OBJECT},
    )
    move_to_drawer = RewTerm(
        func=mdp.object_target_proximity,
        weight=2.0,
        params={"object_cfg": OBJECT, "target_frame_cfg": TARGET_FRAME},
    )
    inside_drawer = RewTerm(
        func=mdp.object_in_drawer_bonus,
        weight=5.0,
        params={"object_cfg": OBJECT, "target_frame_cfg": TARGET_FRAME},
    )
    task_success = RewTerm(
        func=mdp.settled_success_bonus,
        weight=15.0,
        params={
            "object_cfg": OBJECT,
            "target_frame_cfg": TARGET_FRAME,
            "cabinet_cfg": DRAWER,
            "finger_cfg": FINGERS,
        },
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-4)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2, weight=-1.0e-4, params={"asset_cfg": ROBOT}
    )


@configclass
class ObjectToDrawerTerminationsCfg:
    success = DoneTerm(
        func=mdp.settled_object_in_drawer,
        params={
            "object_cfg": OBJECT,
            "target_frame_cfg": TARGET_FRAME,
            "cabinet_cfg": DRAWER,
            "finger_cfg": FINGERS,
            "settle_steps": SETTLE_STEPS,
        },
    )
    dropped_or_out = DoneTerm(
        func=mdp.object_dropped_or_out_of_workspace,
        params={"object_cfg": OBJECT},
    )
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class ObjectToDrawerEnvCfg(FrankaCubeLiftEnvCfg):
    observations: ObjectToDrawerObservationsCfg = ObjectToDrawerObservationsCfg()
    events: ObjectToDrawerEventsCfg = ObjectToDrawerEventsCfg()
    rewards: ObjectToDrawerRewardsCfg = ObjectToDrawerRewardsCfg()
    terminations: ObjectToDrawerTerminationsCfg = ObjectToDrawerTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        CALIBRATION.validate()
        self.commands = None
        self.curriculum = None
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        apply_calibrated_scene(self.scene, include_camera=True)
        apply_task_assets_to_scene(self.scene, "object_to_drawer")
        # Match the articulated-opening environment's approximately 70 N
        # total Panda grasp-force ceiling for the handle and banana grasps.
        self.scene.robot.actuators["panda_hand"].stiffness = 6000.0
        self.scene.robot.actuators["panda_hand"].effort_limit_sim = 35.0
        # Retain the inherited binary open/close finger action.  Grasp timing is
        # handled by the expert state machine rather than intermediate apertures.
        # The collection expert contributes no drive effort while pulling and
        # only holds the measured opening after handle release.  Keep the same
        # calibrated physical rail damping used by DrawerOpening.
        self.scene.cabinet.actuators["drawers"].stiffness = 0.0
        self.actions.arm_action.scale = (0.05, 0.05, 0.05, 0.08, 0.08, 0.08)
        self.decimation = CONTROL_DECIMATION
        self.episode_length_s = 40.0
        self.sim.dt = PHYSICS_DT_S
        self.sim.render_interval = self.decimation
        self.sim.physx.friction_correlation_distance = 0.00625
