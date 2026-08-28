"""ManagerBasedRLEnv config for physical nut pouring."""

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
EE_FRAME = SceneEntityCfg("ee_frame")
CUP = SceneEntityCfg("object")
NUT = SceneEntityCfg("factory_nut")
BOWL = SceneEntityCfg("sorting_bowl")
CAMERA = SceneEntityCfg("policy_camera")


@configclass
class NutPouringObservationsCfg:
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
            func=mdp.oracle_nut_pouring_state,
            params={
                "robot_cfg": ROBOT,
                "ee_frame_cfg": EE_FRAME,
                "cup_cfg": CUP,
                "nut_cfg": NUT,
                "bowl_cfg": BOWL,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class NutPouringEventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_layout = EventTerm(func=mdp.reset_nut_pouring_layout, mode="reset")


@configclass
class NutPouringRewardsCfg:
    reach_cup = RewTerm(
        func=mdp.ee_cup_proximity,
        weight=0.2,
        params={"ee_frame_cfg": EE_FRAME, "cup_cfg": CUP},
    )
    lift_cup = RewTerm(func=mdp.cup_lift_progress, weight=1.0, params={"cup_cfg": CUP})
    transfer = RewTerm(
        func=mdp.cup_above_bowl,
        weight=1.5,
        params={"cup_cfg": CUP, "bowl_cfg": BOWL},
    )
    tilt = RewTerm(func=mdp.cup_tilt_progress, weight=0.5, params={"cup_cfg": CUP})
    nut_to_bowl = RewTerm(
        func=mdp.nut_bowl_proximity,
        weight=2.0,
        params={"nut_cfg": NUT, "bowl_cfg": BOWL},
    )
    nut_inside = RewTerm(
        func=mdp.nut_inside_bowl_bonus,
        weight=5.0,
        params={"nut_cfg": NUT, "bowl_cfg": BOWL},
    )
    task_success = RewTerm(
        func=mdp.settled_success_bonus,
        weight=15.0,
        params={"nut_cfg": NUT, "bowl_cfg": BOWL},
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-4)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2, weight=-1.0e-4, params={"asset_cfg": ROBOT}
    )


@configclass
class NutPouringTerminationsCfg:
    success = DoneTerm(
        func=mdp.settled_nut_in_bowl,
        params={"nut_cfg": NUT, "bowl_cfg": BOWL, "settle_steps": SETTLE_STEPS},
    )
    dropped_or_out = DoneTerm(
        func=mdp.nut_or_cup_out_of_workspace,
        params={"nut_cfg": NUT, "cup_cfg": CUP},
    )
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class NutPouringEnvCfg(FrankaCubeLiftEnvCfg):
    observations: NutPouringObservationsCfg = NutPouringObservationsCfg()
    events: NutPouringEventsCfg = NutPouringEventsCfg()
    rewards: NutPouringRewardsCfg = NutPouringRewardsCfg()
    terminations: NutPouringTerminationsCfg = NutPouringTerminationsCfg()
    # Expert heights are expressed relative to the support surface encoded by
    # the reset cup/bowl root Z. This preserves the table task while allowing a
    # raised-support variant without trajectories descending through geometry.
    expert_lift_ee_height_above_support_m: float = 0.279
    expert_cup_lift_threshold_above_support_m: float = 0.119
    expert_transfer_ee_height_above_support_m: float = 0.299
    expert_pour_ee_height_above_support_m: float = 0.204
    expert_pour_x_bias_m: float = -0.010
    expert_pour_y_bias_m: float = 0.015
    expert_pour_y_sweep_m: float = -0.020
    expert_above_cup_position_tolerance_m: float = 0.006

    def __post_init__(self):
        super().__post_init__()
        CALIBRATION.validate()
        self.commands = None
        self.curriculum = None
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        apply_calibrated_scene(self.scene, include_camera=True)
        apply_task_assets_to_scene(self.scene, "nut_pouring")
        # The lower-body external grasp needs enough normal force to transmit
        # the pouring moment through the two rubber-pad/cup contacts.
        self.scene.robot.actuators["panda_hand"].stiffness = 6000.0
        self.scene.robot.actuators["panda_hand"].effort_limit_sim = 35.0
        self.actions.arm_action.scale = (0.05, 0.05, 0.05, 0.08, 0.08, 0.08)
        self.decimation = CONTROL_DECIMATION
        self.episode_length_s = 35.0
        self.sim.dt = PHYSICS_DT_S
        self.sim.render_interval = self.decimation
        self.sim.physx.friction_correlation_distance = 0.00625
