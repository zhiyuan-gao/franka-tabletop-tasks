"""Portable calibrated-scene constants shared by every task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CALIBRATION_ROOT = PROJECT_ROOT / "assets" / "calibration"
BACKGROUND_IMAGE = CALIBRATION_ROOT / "background" / "empty_background.png"
CAMERA_CONFIG = CALIBRATION_ROOT / "configs" / "simulation_camera.yaml"
CAMERA_TRANSFORM = CALIBRATION_ROOT / "transforms" / "T_base_from_camera_usd.npy"
WORLD_MARKER_TRANSFORM = CALIBRATION_ROOT / "transforms" / "T_base_from_marker.npy"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
POLICY_IMAGE_SIZE = (224, 224)

CAMERA_INTRINSIC_MATRIX = (
    956.573303223,
    0.0,
    642.299030749,
    0.0,
    971.536132812,
    349.211118028,
    0.0,
    0.0,
    1.0,
)
CAMERA_POSITION_M = (0.9016981804871146, 0.6241598574006839, 0.8147045783970387)
CAMERA_ROTATION_WXYZ = (
    0.22343577490842612,
    0.10859155980241988,
    0.4367801844572838,
    0.8645851017084494,
)
RENDER_FOCAL_PIXELS = (CAMERA_INTRINSIC_MATRIX[0] + CAMERA_INTRINSIC_MATRIX[4]) / 2.0

TABLE_SIZE_M = (1.20, 0.80, 0.04)
TABLE_CENTER_M = (0.50, 0.0, -0.02)
BASE_BOARD_SIZE_M = (0.60, 0.42, 0.018)
BASE_BOARD_CENTER_M = (-0.040289092, 0.017341600, -0.009)
BASE_BOARD_ROTATION_WXYZ = (0.9977159097961057, 0.0, 0.0, -0.0675497101380097)
FRANKA_HOME_JOINTS_RAD = (0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398)

# Measured 7 x 5 ChArUco board used as the fixed real-world reference.
WORLD_MARKER_SIZE_M = (0.268, 0.190)
WORLD_MARKER_CENTER_M = (0.5004318804096815, -0.028285689550494125, 0.006235404646318975)
WORLD_MARKER_ROTATION_WXYZ = (
    0.06754536470168693,
    -0.002192447269989635,
    -0.0018326598933670708,
    0.9977121119040301,
)
WORLD_MARKER_NORMAL_M = (-0.004622437754, -0.003360754644, 0.999983669065)
# Rectangle-equivalent board yaw in [-pi/2, pi/2], used to center object-yaw randomization.
WORLD_MARKER_RECTANGLE_YAW_RAD = -0.13520237408892344

FOREGROUND_SEMANTIC_TAG = ("class", "track_b_foreground")
YCB_ASSET_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics"
)
YCB_SUGAR_BOX_USD = f"{YCB_ASSET_ROOT}/004_sugar_box.usd"

# Procedural low-profile cube used by PushCubeToRegion. The dimensions are
# intentionally easy to reproduce in the real setup while remaining large
# enough for the single policy camera and closed Franka gripper to see/push.
PUSH_CUBE_SIZE_M = (0.070, 0.070, 0.050)
PUSH_CUBE_HALF_EXTENTS_XY_M = (
    PUSH_CUBE_SIZE_M[0] / 2.0,
    PUSH_CUBE_SIZE_M[1] / 2.0,
)
PUSH_CUBE_MASS_KG = 0.180
PUSH_CUBE_COLOR_RGB = (0.80, 0.03, 0.03)


@dataclass(frozen=True)
class CalibrationContract:
    background_image: Path = BACKGROUND_IMAGE
    camera_config: Path = CAMERA_CONFIG
    camera_transform: Path = CAMERA_TRANSFORM
    world_marker_transform: Path = WORLD_MARKER_TRANSFORM
    resolution: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT)
    world_frame: str = "panda_link0"

    def validate(self) -> None:
        for path in (
            self.background_image,
            self.camera_config,
            self.camera_transform,
            self.world_marker_transform,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)


CALIBRATION = CalibrationContract()
