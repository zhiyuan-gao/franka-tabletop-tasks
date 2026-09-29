"""Registered upper/lower drawer subtasks in the calibrated banana scene."""

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
from track_b_envs.envs.common.task_scene import (
    OBJECT_TO_DRAWER_CABINET_POSITION_M,
    OBJECT_TO_DRAWER_CABINET_ROTATION_WXYZ,
    apply_task_assets_to_scene,
    build_cabinet_handle_frame_cfg,
    build_drawer_interior_frame_cfg,
)
from track_b_envs.envs.common.timing import (
    CONTROL_DECIMATION,
    PHYSICS_DT_S,
    SETTLE_STEPS,
)

from . import mdp


DRAWER_SUBTASK_OPEN_POSITION_M = 0.22


ROBOT = SceneEntityCfg("robot")
FINGERS = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
EE_FRAME = SceneEntityCfg("ee_frame")
OBJECT = SceneEntityCfg("object")
HANDLE_FRAME = SceneEntityCfg("cabinet_frame")
TARGET_FRAME = SceneEntityCfg("drawer_interior_frame")
DRAWER = SceneEntityCfg("cabinet", joint_names=["UpperDrawerJoint"])
CAMERA = SceneEntityCfg("policy_camera")


@configclass
class DrawerSubtaskObservationsCfg:
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
class DrawerSubtaskEventsCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_layout = EventTerm(func=mdp.reset_object_to_drawer_layout, mode="reset")
    reset_success_counter = EventTerm(
        func=mdp.reset_drawer_subtask_counter, mode="reset"
    )


@configclass
class DrawerSubtaskRewardsCfg:
    reach_handle = RewTerm(
        func=mdp.ee_task_target_proximity,
        weight=0.15,
        params={"ee_frame_cfg": EE_FRAME, "target_frame_cfg": HANDLE_FRAME},
    )
    drawer_progress = RewTerm(
        func=mdp.drawer_task_progress,
        weight=1.0,
        params={"cabinet_cfg": DRAWER, "mode": "open"},
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
        func=mdp.drawer_subtask_success_bonus,
        weight=15.0,
        params={
            "object_cfg": OBJECT,
            "target_frame_cfg": TARGET_FRAME,
            "cabinet_cfg": DRAWER,
            "finger_cfg": FINGERS,
            "mode": "open",
        },
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-4)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": ROBOT},
    )


@configclass
class DrawerSubtaskTerminationsCfg:
    success = DoneTerm(
        func=mdp.settled_drawer_subtask_success,
        params={
            "object_cfg": OBJECT,
            "target_frame_cfg": TARGET_FRAME,
            "cabinet_cfg": DRAWER,
            "finger_cfg": FINGERS,
            "mode": "open",
            "settle_steps": SETTLE_STEPS,
        },
    )
    dropped_or_out = DoneTerm(
        func=mdp.object_dropped_or_out_of_workspace,
        params={"object_cfg": OBJECT},
    )
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class DrawerSubtaskEnvCfg(FrankaCubeLiftEnvCfg):
    """Base configuration parameterized by drawer level and task mode."""

    observations: DrawerSubtaskObservationsCfg = DrawerSubtaskObservationsCfg()
    events: DrawerSubtaskEventsCfg = DrawerSubtaskEventsCfg()
    rewards: DrawerSubtaskRewardsCfg = DrawerSubtaskRewardsCfg()
    terminations: DrawerSubtaskTerminationsCfg = DrawerSubtaskTerminationsCfg()
    task_name: str = "open_upper_drawer"
    task_mode: str = "open"
    drawer_level: str = "upper"
    active_joint_name: str = "UpperDrawerJoint"
    drawer_body_name: str = "UpperDrawer"

    def __post_init__(self):
        super().__post_init__()
        if self.task_mode not in {"open", "put", "put_and_close"}:
            raise ValueError(f"unsupported drawer task mode: {self.task_mode}")
        if self.drawer_level not in {"upper", "lower"}:
            raise ValueError(f"unsupported drawer level: {self.drawer_level}")
        CALIBRATION.validate()
        self.commands = None
        self.curriculum = None
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        apply_calibrated_scene(self.scene, include_camera=True)
        # Reuse the exact cabinet, banana, camera, and tabletop configuration
        # calibrated for object_to_drawer, then select the requested drawer.
        apply_task_assets_to_scene(self.scene, "object_to_drawer")
        self.scene.cabinet.init_state.pos = OBJECT_TO_DRAWER_CABINET_POSITION_M
        self.scene.cabinet.init_state.rot = OBJECT_TO_DRAWER_CABINET_ROTATION_WXYZ
        target_open = self.task_mode in {"put", "put_and_close"}
        self.scene.cabinet.init_state.joint_pos = {
            "UpperDrawerJoint": (
                DRAWER_SUBTASK_OPEN_POSITION_M
                if target_open and self.drawer_level == "upper"
                else 0.0
            ),
            "LowerDrawerJoint": (
                DRAWER_SUBTASK_OPEN_POSITION_M
                if target_open and self.drawer_level == "lower"
                else 0.0
            ),
        }
        self.scene.cabinet_frame = build_cabinet_handle_frame_cfg(
            "drawer", drawer_body=self.drawer_body_name
        )
        self.scene.drawer_interior_frame = build_drawer_interior_frame_cfg(
            drawer_body=self.drawer_body_name
        )

        joint_cfgs = (
            self.observations.critic.full_state.params["cabinet_cfg"],
            self.rewards.drawer_progress.params["cabinet_cfg"],
            self.rewards.task_success.params["cabinet_cfg"],
            self.terminations.success.params["cabinet_cfg"],
        )
        for cfg in joint_cfgs:
            cfg.joint_names = [self.active_joint_name]
        self.rewards.drawer_progress.params["mode"] = self.task_mode
        self.rewards.task_success.params["mode"] = self.task_mode
        self.terminations.success.params["mode"] = self.task_mode

        self.scene.robot.actuators["panda_hand"].stiffness = 6000.0
        self.scene.robot.actuators["panda_hand"].effort_limit_sim = 35.0
        self.scene.cabinet.actuators["drawers"].stiffness = 0.0
        self.actions.arm_action.scale = (0.05, 0.05, 0.05, 0.08, 0.08, 0.08)
        self.decimation = CONTROL_DECIMATION
        self.episode_length_s = {
            "open": 20.0,
            "put": 38.0,
            "put_and_close": 58.0,
        }[self.task_mode]
        self.sim.dt = PHYSICS_DT_S
        self.sim.render_interval = self.decimation
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class OpenUpperDrawerEnvCfg(DrawerSubtaskEnvCfg):
    task_name: str = "open_upper_drawer"
    task_mode: str = "open"
    drawer_level: str = "upper"
    active_joint_name: str = "UpperDrawerJoint"
    drawer_body_name: str = "UpperDrawer"


@configclass
class OpenLowerDrawerEnvCfg(DrawerSubtaskEnvCfg):
    task_name: str = "open_lower_drawer"
    task_mode: str = "open"
    drawer_level: str = "lower"
    active_joint_name: str = "LowerDrawerJoint"
    drawer_body_name: str = "LowerDrawer"


@configclass
class PutObjectInUpperDrawerEnvCfg(DrawerSubtaskEnvCfg):
    task_name: str = "put_object_in_upper_drawer"
    task_mode: str = "put"
    drawer_level: str = "upper"
    active_joint_name: str = "UpperDrawerJoint"
    drawer_body_name: str = "UpperDrawer"


@configclass
class PutObjectInLowerDrawerEnvCfg(DrawerSubtaskEnvCfg):
    task_name: str = "put_object_in_lower_drawer"
    task_mode: str = "put"
    drawer_level: str = "lower"
    active_joint_name: str = "LowerDrawerJoint"
    drawer_body_name: str = "LowerDrawer"


@configclass
class PutObjectInUpperDrawerAndCloseEnvCfg(DrawerSubtaskEnvCfg):
    task_name: str = "put_object_in_upper_drawer_and_close"
    task_mode: str = "put_and_close"
    drawer_level: str = "upper"
    active_joint_name: str = "UpperDrawerJoint"
    drawer_body_name: str = "UpperDrawer"


@configclass
class PutObjectInLowerDrawerAndCloseEnvCfg(DrawerSubtaskEnvCfg):
    task_name: str = "put_object_in_lower_drawer_and_close"
    task_mode: str = "put_and_close"
    drawer_level: str = "lower"
    active_joint_name: str = "LowerDrawerJoint"
    drawer_body_name: str = "LowerDrawer"
