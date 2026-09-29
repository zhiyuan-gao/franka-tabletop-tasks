"""Manager-based environments for the two open-drawer cup tasks."""

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
from track_b_envs.envs.common.timing import CONTROL_DECIMATION, PHYSICS_DT_S, SETTLE_STEPS

from . import mdp


ROBOT = SceneEntityCfg("robot")
FINGERS = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
EE_FRAME = SceneEntityCfg("ee_frame")
CUP = SceneEntityCfg("object")
TARGET_FRAME = SceneEntityCfg("cup_target_frame")
UPPER_DRAWER = SceneEntityCfg("cabinet", joint_names=["UpperDrawerJoint"])
CAMERA = SceneEntityCfg("policy_camera")


@configclass
class CupCabinetObservationsCfg:
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
            func=mdp.oracle_cup_cabinet_state,
            params={
                "robot_cfg": ROBOT,
                "ee_frame_cfg": EE_FRAME,
                "cup_cfg": CUP,
                "target_frame_cfg": TARGET_FRAME,
                "cabinet_cfg": UPPER_DRAWER,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CupCabinetEventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")


@configclass
class CupCabinetRewardsCfg:
    reach_cup = RewTerm(
        func=mdp.ee_cup_proximity,
        weight=0.25,
        params={"ee_frame_cfg": EE_FRAME, "cup_cfg": CUP},
    )
    lift_cup = RewTerm(func=mdp.cup_lift_progress, weight=1.0, params={"cup_cfg": CUP})
    move_to_target = RewTerm(
        func=mdp.cup_target_proximity,
        weight=2.0,
        params={"cup_cfg": CUP, "target_frame_cfg": TARGET_FRAME},
    )
    at_target = RewTerm(
        func=mdp.cup_at_target_bonus,
        weight=5.0,
        params={"cup_cfg": CUP, "target_frame_cfg": TARGET_FRAME},
    )
    task_success = RewTerm(
        func=mdp.settled_success_bonus,
        weight=15.0,
        params={
            "cup_cfg": CUP,
            "target_frame_cfg": TARGET_FRAME,
            "cabinet_cfg": UPPER_DRAWER,
            "finger_cfg": FINGERS,
        },
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-4)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2, weight=-1.0e-4, params={"asset_cfg": ROBOT}
    )


@configclass
class CupCabinetTerminationsCfg:
    success = DoneTerm(
        func=mdp.settled_cup_at_target,
        params={
            "cup_cfg": CUP,
            "target_frame_cfg": TARGET_FRAME,
            "cabinet_cfg": UPPER_DRAWER,
            "finger_cfg": FINGERS,
            "settle_steps": SETTLE_STEPS,
        },
    )
    dropped_or_out = DoneTerm(
        func=mdp.cup_dropped_or_out_of_workspace,
        params={"cup_cfg": CUP},
    )
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class _CupCabinetEnvCfg(FrankaCubeLiftEnvCfg):
    observations: CupCabinetObservationsCfg = CupCabinetObservationsCfg()
    events: CupCabinetEventsCfg = CupCabinetEventsCfg()
    rewards: CupCabinetRewardsCfg = CupCabinetRewardsCfg()
    terminations: CupCabinetTerminationsCfg = CupCabinetTerminationsCfg()
    task_name: str = "cup_to_drawer"

    def __post_init__(self):
        super().__post_init__()
        CALIBRATION.validate()
        self.commands = None
        self.curriculum = None
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        apply_calibrated_scene(self.scene, include_camera=True)
        apply_task_assets_to_scene(self.scene, self.task_name)
        self.scene.robot.actuators["panda_hand"].stiffness = 6000.0
        self.scene.robot.actuators["panda_hand"].effort_limit_sim = 35.0
        self.actions.arm_action.scale = (0.05, 0.05, 0.05, 0.08, 0.08, 0.08)
        self.decimation = CONTROL_DECIMATION
        self.episode_length_s = 35.0
        self.sim.dt = PHYSICS_DT_S
        self.sim.render_interval = self.decimation
        self.sim.physx.friction_correlation_distance = 0.00625
        # The cup is held externally by two narrow finger-pad contacts.  Use a
        # rubber-like default contact material so it does not migrate to one
        # finger during the long cabinet-to-drawer transfer.
        self.sim.physics_material.static_friction = 2.0
        self.sim.physics_material.dynamic_friction = 1.5
        self.sim.physics_material.friction_combine_mode = "max"


@configclass
class CupToDrawerEnvCfg(_CupCabinetEnvCfg):
    """Cup starts on the cabinet and is top-grasped into the upper drawer."""

    task_name: str = "cup_to_drawer"


@configclass
class CupToCabinetTopEnvCfg(_CupCabinetEnvCfg):
    """Cup starts below the open drawer and is side-grasped onto the cabinet."""

    task_name: str = "cup_to_cabinet_top"
