"""Shared ManagerBasedRLEnv execution contract for drawer and door collection."""

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
from track_b_envs.envs.common.timing import CONTROL_DECIMATION, PHYSICS_DT_S

from . import mdp


ROBOT = SceneEntityCfg("robot")
EE_FRAME = SceneEntityCfg("ee_frame")
HANDLE_FRAME = SceneEntityCfg("cabinet_frame")
CAMERA = SceneEntityCfg("policy_camera")


@configclass
class OpeningObservationsCfg:
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
            func=mdp.oracle_opening_state,
            params={
                "robot_cfg": ROBOT,
                "ee_frame_cfg": EE_FRAME,
                "handle_frame_cfg": HANDLE_FRAME,
                "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["UpperDrawerJoint"]),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class OpeningEventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    randomize_cabinet_pose = EventTerm(func=mdp.reset_cabinet_pose, mode="reset")


@configclass
class OpeningRewardsCfg:
    reach_handle = RewTerm(
        func=mdp.ee_handle_proximity,
        weight=0.25,
        params={"ee_frame_cfg": EE_FRAME, "handle_frame_cfg": HANDLE_FRAME},
    )
    opening = RewTerm(
        func=mdp.joint_opening,
        weight=2.0,
        params={"cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["UpperDrawerJoint"])},
    )
    success = RewTerm(
        func=mdp.opening_success_bonus,
        weight=10.0,
        params={
            "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["UpperDrawerJoint"]),
            "threshold": 0.16,
        },
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-4)


@configclass
class OpeningTerminationsCfg:
    success = DoneTerm(
        func=mdp.opening_joint_reached,
        params={
            "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["UpperDrawerJoint"]),
            "threshold": 0.16,
        },
    )
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class ArticulatedOpeningEnvCfg(FrankaCubeLiftEnvCfg):
    observations: OpeningObservationsCfg = OpeningObservationsCfg()
    events: OpeningEventsCfg = OpeningEventsCfg()
    rewards: OpeningRewardsCfg = OpeningRewardsCfg()
    terminations: OpeningTerminationsCfg = OpeningTerminationsCfg()
    task_name: str = "drawer_opening"
    active_joint_name: str = "UpperDrawerJoint"
    success_threshold: float = 0.16

    def __post_init__(self):
        super().__post_init__()
        CALIBRATION.validate()
        self.commands = None
        self.curriculum = None
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        apply_calibrated_scene(self.scene, include_camera=True)
        apply_task_assets_to_scene(self.scene, self.task_name)
        # Approximate the real Panda gripper's 70 N total grasp force.  The
        # upstream lift preset (2 kN/m with a 200 N per-finger limit) produces
        # only about 22 N on this 11 mm handle before the position error closes,
        # which cannot overcome the calibrated rail friction without slipping.
        self.scene.robot.actuators["panda_hand"].stiffness = 6000.0
        self.scene.robot.actuators["panda_hand"].effort_limit_sim = 35.0
        # Retain the inherited BinaryJointPositionAction: the high-level command
        # selects only fully open (0.04 m/finger) or fully closed (0.0 m/finger).
        # The physical finger drives still move continuously between those targets.
        joint_cfgs = (
            self.observations.critic.full_state.params["cabinet_cfg"],
            self.rewards.opening.params["cabinet_cfg"],
            self.rewards.success.params["cabinet_cfg"],
            self.terminations.success.params["cabinet_cfg"],
        )
        for cfg in joint_cfgs:
            cfg.joint_names = [self.active_joint_name]
        self.rewards.success.params["threshold"] = self.success_threshold
        self.terminations.success.params["threshold"] = self.success_threshold
        self.actions.arm_action.scale = (0.05, 0.05, 0.05, 0.08, 0.08, 0.08)
        self.decimation = CONTROL_DECIMATION
        # The velocity-limited door path needs about 12.3 s at the calibrated
        # seed, so retain margin instead of timing out immediately before the
        # physical success threshold.
        self.episode_length_s = 18.0 if self.task_name == "cabinet_opening" else 14.0
        self.sim.dt = PHYSICS_DT_S
        self.sim.render_interval = self.decimation


@configclass
class DrawerOpeningEnvCfg(ArticulatedOpeningEnvCfg):
    task_name: str = "drawer_opening"
    active_joint_name: str = "UpperDrawerJoint"
    success_threshold: float = 0.16


@configclass
class CabinetOpeningEnvCfg(ArticulatedOpeningEnvCfg):
    task_name: str = "cabinet_opening"
    active_joint_name: str = "DoorHingeJoint"
    success_threshold: float = 1.20
