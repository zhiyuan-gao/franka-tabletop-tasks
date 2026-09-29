"""Shared calibrated Franka tabletop scene helpers."""

from __future__ import annotations

from .constants import (
    BASE_BOARD_CENTER_M,
    BASE_BOARD_ROTATION_WXYZ,
    BASE_BOARD_SIZE_M,
    CAMERA_POSITION_M,
    CAMERA_ROTATION_WXYZ,
    FOREGROUND_SEMANTIC_TAG,
    FRANKA_HOME_JOINTS_RAD,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RENDER_FOCAL_PIXELS,
    TABLE_CENTER_M,
    TABLE_SIZE_M,
)


def _add_foreground_tag(spawn_cfg) -> None:
    tags = list(spawn_cfg.semantic_tags or [])
    if FOREGROUND_SEMANTIC_TAG not in tags:
        tags.append(FOREGROUND_SEMANTIC_TAG)
    spawn_cfg.semantic_tags = tags


def apply_calibrated_scene(scene, *, include_camera: bool = True) -> None:
    """Replace the stock lab table with the calibrated physical contract."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import TiledCameraCfg

    scene.table = None
    scene.platform = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE_M,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_CENTER_M),
    )
    scene.franka_base_board = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FrankaBaseBoard",
        spawn=sim_utils.CuboidCfg(
            size=BASE_BOARD_SIZE_M,
            semantic_tags=[FOREGROUND_SEMANTIC_TAG],
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.018, 0.018, 0.018), roughness=1.0
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=BASE_BOARD_CENTER_M, rot=BASE_BOARD_ROTATION_WXYZ
        ),
    )
    scene.real_background_key_light = AssetBaseCfg(
        prim_path="/World/RealBackgroundKeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=1800.0, angle=2.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9238795, 0.3826834, 0.0, 0.0)),
    )
    if include_camera:
        focal_length = 24.0
        scene.policy_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/PolicyCamera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=CAMERA_POSITION_M, rot=CAMERA_ROTATION_WXYZ, convention="opengl"
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                horizontal_aperture=IMAGE_WIDTH * focal_length / RENDER_FOCAL_PIXELS,
                clipping_range=(0.01, 20.0),
                focus_distance=2.0,
            ),
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            data_types=["rgb", "semantic_segmentation"],
            semantic_filter="class:track_b_foreground",
            colorize_semantic_segmentation=False,
            update_latest_camera_pose=True,
        )
    scene.plane.init_state.pos = (0.0, 0.0, -0.80)
    scene.plane.spawn.visible = False
    scene.light.spawn.intensity = 500.0

    scene.robot.init_state.pos = (0.0, 0.0, 0.0)
    scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    scene.robot.init_state.joint_pos = {
        **{f"panda_joint{i}": q for i, q in enumerate(FRANKA_HOME_JOINTS_RAD, start=1)},
        "panda_finger_joint.*": 0.04,
    }
    _add_foreground_tag(scene.robot.spawn)
