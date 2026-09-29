"""Shared geometry for controlled banana/orange placement trajectories."""

from __future__ import annotations

from track_b_envs.envs.common.constants import FOREGROUND_SEMANTIC_TAG, PROJECT_ROOT
from track_b_envs.envs.common.task_scene import (
    BANANA_STABLE_LOCAL_MIN_Z_M,
    YCB_BANANA_VISUAL_USD,
)


# Episode 41's first unobstructed right-camera frame is the canonical layout
# for *both* generated trajectories.  Its calibration-board detections were
# registered to ``empty_background.png`` with a homography (median corner
# residual: 0.84 px), then each object centre was back-projected onto the
# calibrated table plane.  A final silhouette pass compensates for the USD
# assets' non-centred origins and above-table geometry.  Keeping this one set
# of poses makes pick order the only experimental variable.
REFERENCE_FRAME = (
    PROJECT_ROOT
    / "four_task_datasets/two_fruit_placement_selected/orange_then_banana/initial_000000.jpg"
)
REFERENCE_VIDEO = (
    "/raid/home/than/four_demo_videos/"
    "two_fruit_orange_first_episode_000041.mp4"
)
REFERENCE_TO_BACKGROUND_HOMOGRAPHY = (
    (1.685747268430, 0.011491421688, 112.226559503275),
    (0.018032754779, 1.719821338946, -47.903804285239),
    (0.000007124265, 0.000038242841, 1.0),
)
REFERENCE_PIXEL_CENTERS = {
    "target_plate": (416.978, 399.638),
    "orange": (456.078, 483.531),
    "banana": (407.980, 554.580),
    "distractor_plate": (553.337, 701.068),
}
TARGET_PLATE_POSITION_M = (0.69545284, -0.37345040, 0.0005)
DISTRACTOR_PLATE_POSITION_M = (0.74049220, 0.14566956, 0.0005)

YCB_BANANA_SCALE = 0.93
YCB_ORANGE_SCALE = 0.87
BANANA_INITIAL_POSITION_M = (
    0.78916854,
    -0.11451637,
    -YCB_BANANA_SCALE * BANANA_STABLE_LOCAL_MIN_Z_M + 0.0005,
)
ORANGE_INITIAL_POSITION_M = (0.71014740, -0.18584285, 0.0325)

# Split the final poses across opposite halves of the plate.  Their centres
# are about 11.5 cm apart, avoiding the former orange/banana overlap.  At this
# banana XY pose its curved end rests on the raised plate rim; a mesh-on-mesh
# height query requires a 21.51 mm lift over the table support pose to retain
# 1 mm clearance from the YCB plate instead of intersecting it.
BANANA_FINAL_POSITION_M = (0.72851501, -0.31824435, 0.03736)
ORANGE_FINAL_POSITION_M = (0.64423530, -0.39707469, 0.0365)

YCB_ASSET_ROOT = PROJECT_ROOT / "assets" / "task_assets" / "ycb"
YCB_ORANGE_USD = YCB_ASSET_ROOT / "017_orange.usd"
YCB_PLATE_USD = YCB_ASSET_ROOT / "029_plate.usd"
TARGET_PLATE_SCALE = 0.92
DISTRACTOR_PLATE_SCALE = 0.91

# Same broad-side stable pose as the shared YCB banana, with only a world-Z
# yaw change so its long axis follows the nearly horizontal pose in the video.
BANANA_STABLE_ROTATION_WXYZ = (
    0.6221574120,
    -0.7283670556,
    0.2064215942,
    -0.1994786017,
)


def _matte_material(color: tuple[float, float, float]):
    import isaaclab.sim as sim_utils

    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        roughness=0.92,
        metallic=0.0,
    )


def _plate_cfg(
    name: str,
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    scale: float,
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(YCB_PLATE_USD),
            scale=(scale, scale, scale),
            semantic_tags=[FOREGROUND_SEMANTIC_TAG],
            visual_material=_matte_material(color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


def _banana_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Banana",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(YCB_BANANA_VISUAL_USD),
            scale=(YCB_BANANA_SCALE, YCB_BANANA_SCALE, YCB_BANANA_SCALE),
            semantic_tags=[FOREGROUND_SEMANTIC_TAG],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=BANANA_INITIAL_POSITION_M,
            rot=BANANA_STABLE_ROTATION_WXYZ,
        ),
    )


def _orange_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Orange",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(YCB_ORANGE_USD),
            scale=(YCB_ORANGE_SCALE, YCB_ORANGE_SCALE, YCB_ORANGE_SCALE),
            semantic_tags=[FOREGROUND_SEMANTIC_TAG],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=ORANGE_INITIAL_POSITION_M),
    )


def build_two_fruit_scene_cfg():
    """Build the common initial scene used by both order variants."""

    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import (
        FrankaCubeLiftEnvCfg,
    )

    from track_b_envs.envs.common.scene import apply_calibrated_scene

    scene = FrankaCubeLiftEnvCfg().scene
    scene.num_envs = 1
    scene.env_spacing = 2.5
    scene.object = None
    apply_calibrated_scene(scene, include_camera=True)
    scene.real_background_key_light.spawn.intensity = 900.0
    scene.light.spawn.intensity = 350.0

    scene.target_plate = _plate_cfg(
        "TargetPlate",
        TARGET_PLATE_POSITION_M,
        (0.43, 0.25, 0.40),
        scale=TARGET_PLATE_SCALE,
    )
    scene.distractor_plate = _plate_cfg(
        "DistractorPlate",
        DISTRACTOR_PLATE_POSITION_M,
        (0.005, 0.12, 0.035),
        scale=DISTRACTOR_PLATE_SCALE,
    )
    scene.banana = _banana_cfg()
    scene.orange = _orange_cfg()
    return scene


__all__ = [
    "BANANA_FINAL_POSITION_M",
    "BANANA_INITIAL_POSITION_M",
    "BANANA_STABLE_ROTATION_WXYZ",
    "DISTRACTOR_PLATE_POSITION_M",
    "ORANGE_FINAL_POSITION_M",
    "ORANGE_INITIAL_POSITION_M",
    "REFERENCE_FRAME",
    "REFERENCE_PIXEL_CENTERS",
    "REFERENCE_TO_BACKGROUND_HOMOGRAPHY",
    "REFERENCE_VIDEO",
    "TARGET_PLATE_POSITION_M",
    "build_two_fruit_scene_cfg",
]
