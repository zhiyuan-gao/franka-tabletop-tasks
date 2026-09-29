"""Calibrated microwave, with a passive hinge and physical contact sensing."""

import math
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import FrankaCubeLiftEnvCfg
from track_b_envs.envs.common.constants import PROJECT_ROOT, FOREGROUND_SEMANTIC_TAG
from track_b_envs.envs.common.scene import apply_calibrated_scene
from track_b_envs.envs.common.observations import composited_policy_rgb, franka_proprio

ASSET = PROJECT_ROOT/'assets/task_assets/microwave/BlackMicrowave.usda'
ROOT_POS = (.62,-.45,0.)
ROOT_YAW = math.radians(35)
HINGE_LOCAL = (.170,.1585,0.)
INITIAL_ANGLE = -math.radians(86)


class ContactPointHistorySensor(ContactSensor):
    """Keep substep contact positions alongside Isaac Lab's force history.

The 20 Hz video can otherwise miss 100 Hz collision impulses between frames.
"""
    def _initialize_impl(self):
        super()._initialize_impl()
        points = self._data.contact_pos_w
        self.contact_pos_w_history = torch.full(
            (points.shape[0], self.cfg.history_length, *points.shape[1:]),
            float('nan'), device=self.device)

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        self.contact_pos_w_history[env_ids] = torch.roll(
            self.contact_pos_w_history[env_ids], 1, dims=1)
        self.contact_pos_w_history[env_ids,0] = self._data.contact_pos_w[env_ids]

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'contact_pos_w_history'):
            self.contact_pos_w_history[slice(None) if env_ids is None else env_ids] = float('nan')


def oracle_state(env):
    state = franka_proprio(env, SceneEntityCfg('robot'), SceneEntityCfg('ee_frame'))
    door = env.scene['microwave']
    return torch.cat((state, door.data.joint_pos, door.data.joint_vel), dim=-1)


@configclass
class Observations:
    @configclass
    class Policy(ObservationGroupCfg):
        rgb = ObservationTermCfg(func=composited_policy_rgb, params={'sensor_cfg':SceneEntityCfg('policy_camera')})
        proprio = ObservationTermCfg(func=franka_proprio, params={'robot_cfg':SceneEntityCfg('robot'),'ee_frame_cfg':SceneEntityCfg('ee_frame')})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    @configclass
    class Critic(ObservationGroupCfg):
        state = ObservationTermCfg(func=oracle_state)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: Policy = Policy()
    critic: Critic = Critic()


@configclass
class Events:
    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode='reset')


@configclass
class Terminations:
    # Demo completion is checked after the push and withdrawal, so an automatic
    # success reset cannot conceal rebound or truncate the withdrawal frames.
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class MicrowaveClosingEnvCfg(FrankaCubeLiftEnvCfg):
    observations: Observations = Observations()
    events: Events = Events()
    terminations: Terminations = Terminations()
    def __post_init__(self):
        super().__post_init__()
        self.commands = None
        self.curriculum = None
        self.rewards = None
        self.scene.object = None
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        apply_calibrated_scene(self.scene)
        self.scene.real_background_key_light.spawn.intensity = 750.
        self.scene.light.spawn.intensity = 250.
        self.scene.microwave = ArticulationCfg(
            prim_path='{ENV_REGEX_NS}/Microwave',
            spawn=sim_utils.UsdFileCfg(usd_path=str(ASSET), activate_contact_sensors=True,
                semantic_tags=[FOREGROUND_SEMANTIC_TAG],
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True,enabled_self_collisions=False)),
            init_state=ArticulationCfg.InitialStateCfg(pos=ROOT_POS,
                rot=(math.cos(ROOT_YAW/2),0,0,math.sin(ROOT_YAW/2)), joint_pos={'DoorHingeJoint':INITIAL_ANGLE}),
            actuators={'hinge':ImplicitActuatorCfg(joint_names_expr=['DoorHingeJoint'],
                stiffness=0., damping=.10, friction=.025, effort_limit_sim=5., velocity_limit_sim=3.)},
        )
        self.scene.door_contact = ContactSensorCfg(
            class_type=ContactPointHistorySensor,
            prim_path='{ENV_REGEX_NS}/Microwave/Door', track_contact_points=True,
            max_contact_data_count_per_prim=256,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/panda_leftfinger',
                '{ENV_REGEX_NS}/Robot/panda_rightfinger','{ENV_REGEX_NS}/Robot/panda_hand'],
            update_period=0.0, history_length=5,
        )
        self.scene.robot.actuators['panda_hand'].stiffness = 6000.
        self.scene.robot.actuators['panda_hand'].effort_limit_sim = 35.
        self.actions.arm_action.scale = (.05,.05,.05,.08,.08,.08)
        self.decimation = 5
        self.sim.dt = .01
        self.sim.render_interval = 5
        self.episode_length_s = 70.
