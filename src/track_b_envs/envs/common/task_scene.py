"""Shared initial-state geometry for the calibrated Track B tasks."""

from __future__ import annotations

from pathlib import Path

from .constants import FOREGROUND_SEMANTIC_TAG, PROJECT_ROOT
from .scene import apply_calibrated_scene


SUPPORTED_TASKS = (
    "drawer_opening",
    "cabinet_opening",
    "object_to_drawer",
    "nut_pouring",
    "cup_to_drawer",
    "cup_to_cabinet_top",
)

TASK_ASSET_ROOT = PROJECT_ROOT / "assets" / "task_assets"
DRAWER_CABINET_USD = TASK_ASSET_ROOT / "cabinets" / "TwoDrawerWhiteCabinet.usda"
DOOR_CABINET_USD = TASK_ASSET_ROOT / "cabinets" / "SingleDoorWhiteCabinet.usda"
POURING_CUP_USD = TASK_ASSET_ROOT / "pouring" / "BluePouringCup.usda"
POURING_BOWL_USD = TASK_ASSET_ROOT / "pouring" / "PurplePouringBowl.usda"
YCB_BANANA_VISUAL_USD = TASK_ASSET_ROOT / "ycb" / "011_banana.usd"
YCB_BANANA_USD = TASK_ASSET_ROOT / "ycb" / "011_banana_physics.usda"
YCB_BANANA_TEXTURE = TASK_ASSET_ROOT / "ycb" / "Materials" / "Textures" / "011_banana_COLOR.png"

# A diagonal working face is visible from the fixed camera while remaining
# approachable from the Franka side of the table.
CABINET_POSITION_M = (0.72, -0.18, 0.0)
DOOR_CABINET_POSITION_M = (0.75, -0.18, 0.0)
CABINET_ROTATION_WXYZ = (0.9537169507482269, 0.0, 0.0, 0.3007057995042731)
# Pose recovered from the four top-panel corners in the first unobstructed
# frame of ``banana_in_drawer_then_close_episode_000048.mp4``.  Fifteen board
# markers were used to recover the per-video camera pose before intersecting
# the corner rays with the z=0.37 m cabinet-top plane.  The reconstructed
# 0.373/0.387 m width and 0.340/0.341 m depth agree with this asset's
# 0.38 x 0.34 m footprint; the rigid rectangle-fit corner residual is 3-4 mm.
OBJECT_TO_DRAWER_CABINET_POSITION_M = (0.60386895, -0.52161895, 0.0)
OBJECT_TO_DRAWER_CABINET_YAW_DEG = 8.74888170
OBJECT_TO_DRAWER_CABINET_ROTATION_WXYZ = (
    0.9970868767,
    0.0,
    0.0,
    0.0762742440,
)
# The Figure-4 cup tasks preserve the same cabinet yaw and all cabinet-local
# cup coordinates, but translate the complete setup 8 cm toward the Franka so
# the requested centered top grasp stays inside the physical workspace.
CUP_CABINET_POSITION_M = (0.5382724247, -0.4758248907, 0.0)
DRAWER_HANDLE_OFFSET_M = (0.0, 0.19, -0.004)
# Desired banana-root pose near the center of the usable upper-drawer volume.
# The drawer floor top is z=-0.048 m in UpperDrawer coordinates and the
# banana's broad-side support offset is about 13.6 mm at the configured scale.
# Bias the placement toward the drawer front.  At 18 cm opening this point is
# clear of the cabinet top, so a top-down Franka placement is collision-free.
DRAWER_INTERIOR_TARGET_OFFSET_M = (0.0, 0.070, -0.034)
DOOR_HANDLE_OFFSET_M = (0.27, 0.0275, 0.045)

# Cabinet-local (0.0, +0.10) projected into world XY. The YCB mesh origin is
# near its center, not on its underside. These measured mesh values keep the
# broad support face 0.5 mm above the cabinet top instead of embedding it.
BANANA_SCALE = 0.85
CABINET_TOP_HEIGHT_M = 0.370
BANANA_STABLE_LOCAL_MIN_Z_M = -0.01596104628431209
BANANA_SUPPORT_CLEARANCE_M = 0.0005
BANANA_ROOT_HEIGHT_M = (
    CABINET_TOP_HEIGHT_M
    - BANANA_SCALE * BANANA_STABLE_LOCAL_MIN_Z_M
    + BANANA_SUPPORT_CLEARANCE_M
)
BANANA_POSITION_M = (0.588658540, -0.422782502, BANANA_ROOT_HEIGHT_M)
# Convex-hull stable pose (second broad side), followed by the calibrated
# object-to-drawer cabinet yaw. This genuinely lays the curved banana on its
# broad side; the previous quaternion was almost only a Z yaw and left it
# edge-on.
BANANA_STABLE_ROTATION_WXYZ = (
    0.6522058600,
    -0.7537526300,
    -0.0706074200,
    0.0387178200,
)
# Place the cup almost directly below the side-grasp pre-approach generated
# from the Franka home pose.  Compared with (0.38, -0.28), this reduces the
# initial horizontal EE sweep while retaining clearance from the bowl.
POURING_CUP_POSITION_M = (0.43, -0.18, 0.001)
POURING_NUT_HEIGHT_M = 0.050
POURING_BOWL_POSITION_M = (0.52, -0.04, 0.001)

# Cup/cabinet tasks reuse the cabinet pose calibrated from the banana drawer
# video.  Coordinates below are expressed in the cabinet frame first, then
# projected into world XY.  The cup USD origin is at the base of the cup.
# The local XY measurements reproduce the object placement in KnowDemo-Fig4:
# the first cup is close to the drawer-side edge of the cabinet top, while the
# second is centered laterally below the extended upper-drawer front.
CUP_HEIGHT_M = 0.087
CUP_TO_DRAWER_LOCAL_POSITION_M = (0.035, 0.080, CABINET_TOP_HEIGHT_M + 0.001)
CUP_TO_DRAWER_POSITION_M = (0.5606968538, -0.3914320891, 0.371)
# The cup is centered laterally below the open upper-drawer front, as in the
# Figure-4 demonstration.  A staged high-to-low approach keeps this literal
# image placement reachable without crossing the drawer.
CUP_TO_CABINET_TOP_LOCAL_POSITION_M = (0.035, 0.364, 0.001)
CUP_TO_CABINET_TOP_POSITION_M = (0.5174992907, -0.1107365769, 0.001)
UPPER_DRAWER_OPEN_POSITION_M = 0.220
# UpperDrawer origin is at z=0.284 m.  Its floor top is at local z=-0.048 m;
# the extra millimetre keeps the cup base out of the collision surface.
UPPER_DRAWER_CUP_TARGET_OFFSET_M = (0.035, 0.085, -0.047)
CABINET_TOP_CUP_TARGET_OFFSET_M = CUP_TO_DRAWER_LOCAL_POSITION_M


def validate_task_assets() -> None:
    for path in (
        DRAWER_CABINET_USD,
        DOOR_CABINET_USD,
        POURING_CUP_USD,
        POURING_BOWL_USD,
        YCB_BANANA_USD,
        YCB_BANANA_VISUAL_USD,
        YCB_BANANA_TEXTURE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def _cabinet_cfg(kind: str, *, upper_drawer_open: bool = False):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    if kind == "drawer":
        usd_path = DRAWER_CABINET_USD
        joint_names = [".*DrawerJoint"]
        joint_pos = (
            {
                "LowerDrawerJoint": 0.0,
                "UpperDrawerJoint": UPPER_DRAWER_OPEN_POSITION_M,
            }
            if upper_drawer_open
            else {".*DrawerJoint": 0.0}
        )
        actuator_name = "drawers"
        # Empirical nominal rail resistance.  The values were selected from an
        # effort sweep so that a light incidental collision cannot open the
        # drawer, while a centered Franka handle grasp still can.
        joint_damping = 15.0
        joint_static_friction = 10.0
        joint_dynamic_friction = 8.0
    elif kind == "door":
        usd_path = DOOR_CABINET_USD
        joint_names = ["DoorHingeJoint"]
        joint_pos = {"DoorHingeJoint": 0.0}
        actuator_name = "door"
        # Lower hinge resistance than the drawer rail, likewise calibrated by
        # a controlled joint-effort sweep rather than left frictionless.
        joint_damping = 1.0
        joint_static_friction = 1.8
        joint_dynamic_friction = 1.0
    else:
        raise ValueError(f"unsupported cabinet kind: {kind}")

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            semantic_tags=[FOREGROUND_SEMANTIC_TAG],
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(
                DOOR_CABINET_POSITION_M
                if kind == "door"
                else CABINET_POSITION_M
            ),
            rot=CABINET_ROTATION_WXYZ,
            joint_pos=joint_pos,
        ),
        actuators={
            actuator_name: ImplicitActuatorCfg(
                joint_names_expr=joint_names,
                effort_limit_sim=80.0,
                stiffness=0.0,
                damping=joint_damping,
                friction=joint_static_friction,
                dynamic_friction=joint_dynamic_friction,
                viscous_friction=0.0,
            )
        },
    )


def build_drawer_cabinet_cfg():
    """Return the calibrated two-drawer cabinet for composite task scenes."""

    validate_task_assets()
    return _cabinet_cfg("drawer")


def build_cabinet_handle_frame_cfg(
    kind: str, *, drawer_body: str = "UpperDrawer"
):
    from isaaclab.sensors import FrameTransformerCfg
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

    if kind == "drawer":
        if drawer_body not in {"UpperDrawer", "LowerDrawer"}:
            raise ValueError(f"unsupported drawer body: {drawer_body}")
        body_prim = drawer_body
        offset = DRAWER_HANDLE_OFFSET_M
    elif kind == "door":
        body_prim = "Door"
        offset = DOOR_HANDLE_OFFSET_M
    else:
        raise ValueError(f"unsupported cabinet kind: {kind}")
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cabinet/{body_prim}",
                name="task_handle",
                offset=OffsetCfg(pos=offset),
            )
        ],
    )


def build_drawer_interior_frame_cfg(*, drawer_body: str = "UpperDrawer"):
    """Track a conservative placement target in the moving upper drawer."""

    from isaaclab.sensors import FrameTransformerCfg
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

    if drawer_body not in {"UpperDrawer", "LowerDrawer"}:
        raise ValueError(f"unsupported drawer body: {drawer_body}")
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cabinet/{drawer_body}",
                name="drawer_interior",
                offset=OffsetCfg(pos=DRAWER_INTERIOR_TARGET_OFFSET_M),
            )
        ],
    )


def _cup_target_frame_cfg(task: str):
    """Track the cup-base target for either open-drawer placement task."""

    from isaaclab.sensors import FrameTransformerCfg
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

    if task == "cup_to_drawer":
        body_prim = "UpperDrawer"
        offset = UPPER_DRAWER_CUP_TARGET_OFFSET_M
    elif task == "cup_to_cabinet_top":
        body_prim = "CabinetBody"
        offset = CABINET_TOP_CUP_TARGET_OFFSET_M
    else:
        raise ValueError(f"unsupported cup target task: {task}")
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cabinet/{body_prim}",
                name="cup_target",
                offset=OffsetCfg(pos=offset),
            )
        ],
    )


def _banana_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Banana",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(YCB_BANANA_USD),
            scale=(BANANA_SCALE, BANANA_SCALE, BANANA_SCALE),
            semantic_tags=[FOREGROUND_SEMANTIC_TAG],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=3.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.066),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.003,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BANANA_POSITION_M,
            rot=BANANA_STABLE_ROTATION_WXYZ,
        ),
    )


def _pouring_cfgs():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BluePouringCup",
        spawn=UsdFileCfg(
            usd_path=str(POURING_CUP_USD),
            semantic_tags=[FOREGROUND_SEMANTIC_TAG],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=POURING_CUP_POSITION_M),
    )
    nut = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FactoryNut",
        spawn=UsdFileCfg(
            usd_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/"
                "nut_pour_assets/factory_m16_nut_green.usd"
            ),
            scale=(0.5, 0.5, 0.5),
            semantic_tags=[FOREGROUND_SEMANTIC_TAG],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                linear_damping=0.2,
                angular_damping=0.2,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            # The nut is half-scale; the previous 5 mm offset was comparable
            # to its own radius and produced persistent resting-contact jitter.
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                POURING_CUP_POSITION_M[0],
                POURING_CUP_POSITION_M[1],
                POURING_NUT_HEIGHT_M,
            )
        ),
    )
    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PurplePouringBowl",
        spawn=UsdFileCfg(
            usd_path=str(POURING_BOWL_USD),
            semantic_tags=[FOREGROUND_SEMANTIC_TAG],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=POURING_BOWL_POSITION_M),
    )
    return cup, nut, bowl


def _cup_cfg(position):
    """Return the physical blue cup without the nut used by NutPouring."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BluePouringCup",
        spawn=UsdFileCfg(
            usd_path=str(POURING_CUP_USD),
            semantic_tags=[FOREGROUND_SEMANTIC_TAG],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=3.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def apply_task_assets_to_scene(scene, task: str) -> None:
    """Add one task's project-local assets to an existing Franka scene config."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task {task!r}; choose one of {SUPPORTED_TASKS}")
    validate_task_assets()
    if task in {
        "drawer_opening",
        "object_to_drawer",
        "cup_to_drawer",
        "cup_to_cabinet_top",
    }:
        scene.cabinet = _cabinet_cfg(
            "drawer",
            upper_drawer_open=task in {"cup_to_drawer", "cup_to_cabinet_top"},
        )
        if task == "object_to_drawer":
            scene.cabinet.init_state.pos = OBJECT_TO_DRAWER_CABINET_POSITION_M
            scene.cabinet.init_state.rot = OBJECT_TO_DRAWER_CABINET_ROTATION_WXYZ
        elif task in {"cup_to_drawer", "cup_to_cabinet_top"}:
            scene.cabinet.init_state.pos = CUP_CABINET_POSITION_M
            scene.cabinet.init_state.rot = OBJECT_TO_DRAWER_CABINET_ROTATION_WXYZ
        scene.cabinet_frame = build_cabinet_handle_frame_cfg("drawer")
        scene.object = None if task == "drawer_opening" else _banana_cfg()
        if task == "object_to_drawer":
            scene.drawer_interior_frame = build_drawer_interior_frame_cfg()
        elif task in {"cup_to_drawer", "cup_to_cabinet_top"}:
            # The task begins with the drawer already open.  A position drive
            # keeps it at the demonstrated state without per-step state writes.
            scene.cabinet.actuators["drawers"].stiffness = 2000.0
            scene.cabinet.actuators["drawers"].damping = 80.0
            scene.object = _cup_cfg(
                CUP_TO_DRAWER_POSITION_M
                if task == "cup_to_drawer"
                else CUP_TO_CABINET_TOP_POSITION_M
            )
            scene.cup_target_frame = _cup_target_frame_cfg(task)
    elif task == "cabinet_opening":
        scene.cabinet = _cabinet_cfg("door")
        scene.cabinet_frame = build_cabinet_handle_frame_cfg("door")
        scene.object = None
    else:
        scene.object, scene.factory_nut, scene.sorting_bowl = _pouring_cfgs()


def build_initial_state_scene_cfg(task: str, *, include_camera: bool = True):
    """Build a preview scene that is the geometry source for later RL configs."""

    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import (
        FrankaCubeLiftEnvCfg,
    )

    base_cfg = FrankaCubeLiftEnvCfg()
    scene = base_cfg.scene
    scene.num_envs = 1
    scene.env_spacing = 2.5
    apply_calibrated_scene(scene, include_camera=include_camera)
    apply_task_assets_to_scene(scene, task)

    return scene


__all__ = [
    "BANANA_POSITION_M",
    "BANANA_ROOT_HEIGHT_M",
    "BANANA_SCALE",
    "BANANA_STABLE_LOCAL_MIN_Z_M",
    "BANANA_SUPPORT_CLEARANCE_M",
    "CABINET_TOP_HEIGHT_M",
    "CABINET_POSITION_M",
    "DOOR_CABINET_POSITION_M",
    "CABINET_ROTATION_WXYZ",
    "CABINET_TOP_CUP_TARGET_OFFSET_M",
    "CUP_CABINET_POSITION_M",
    "CUP_HEIGHT_M",
    "CUP_TO_CABINET_TOP_LOCAL_POSITION_M",
    "CUP_TO_CABINET_TOP_POSITION_M",
    "CUP_TO_DRAWER_LOCAL_POSITION_M",
    "CUP_TO_DRAWER_POSITION_M",
    "OBJECT_TO_DRAWER_CABINET_POSITION_M",
    "OBJECT_TO_DRAWER_CABINET_ROTATION_WXYZ",
    "OBJECT_TO_DRAWER_CABINET_YAW_DEG",
    "DRAWER_INTERIOR_TARGET_OFFSET_M",
    "UPPER_DRAWER_CUP_TARGET_OFFSET_M",
    "UPPER_DRAWER_OPEN_POSITION_M",
    "POURING_BOWL_POSITION_M",
    "POURING_CUP_POSITION_M",
    "POURING_NUT_HEIGHT_M",
    "SUPPORTED_TASKS",
    "apply_task_assets_to_scene",
    "build_cabinet_handle_frame_cfg",
    "build_drawer_cabinet_cfg",
    "build_drawer_interior_frame_cfg",
    "build_initial_state_scene_cfg",
    "validate_task_assets",
]
