"""Real-observation conversion with the same actor contract as simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
POLICY_JOINT_NAMES = ARM_JOINT_NAMES + FINGER_JOINT_NAMES
PROPRIO_DIM = 25
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
POLICY_VIEW_SIZE = 224


def center_crop_policy_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return the simulator-equivalent square RGB crop without resizing it.

    The calibrated simulator takes the center 720 columns from a 1280x720 RGB
    frame.  Sending this lossless crop over the network costs about 30 MB/s at
    20 Hz and leaves the numerically sensitive resize on the inference server.
    """

    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        raise TypeError(f"RGB image must be uint8, got {image.dtype}")
    if image.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3):
        left = (CAMERA_WIDTH - CAMERA_HEIGHT) // 2
        image = image[:, left : left + CAMERA_HEIGHT]
    elif image.shape != (CAMERA_HEIGHT, CAMERA_HEIGHT, 3):
        raise ValueError(
            "expected RGB shape (720, 1280, 3) or pre-cropped (720, 720, 3), "
            f"got {image.shape}"
        )
    return np.ascontiguousarray(image)


def policy_rgb_tensor(rgb: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """Convert a raw/cropped RGB8 frame to the online 224x224 policy view.

    No uint8 rounding is done after this resize.  ``ChunkRunner.step`` performs
    the same rounding and checkpoint-specific second resize used during data
    collection/training.
    """

    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB8 image, got {image.shape} {image.dtype}")
    if image.shape[:2] in ((CAMERA_HEIGHT, CAMERA_WIDTH), (CAMERA_HEIGHT, CAMERA_HEIGHT)):
        image = center_crop_policy_rgb(image)
    elif image.shape[:2] != (POLICY_VIEW_SIZE, POLICY_VIEW_SIZE):
        raise ValueError(
            "expected 1280x720 camera RGB, 720x720 calibrated crop, or "
            f"224x224 recorded policy RGB; got {image.shape}"
        )
    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(
        device=device, dtype=torch.float32
    )
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).div_(255.0)
    if image.shape[:2] != (POLICY_VIEW_SIZE, POLICY_VIEW_SIZE):
        tensor = F.interpolate(
            tensor,
            size=(POLICY_VIEW_SIZE, POLICY_VIEW_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    return tensor


@dataclass
class QuaternionContinuity:
    """Keep ROS quaternion signs in the simulator/checkpoint hemisphere."""

    reference_wxyz: np.ndarray

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference_wxyz, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(reference))
        if not np.isfinite(norm) or norm < 1.0e-8:
            raise ValueError("quaternion reference must be finite and non-zero")
        self.reference_wxyz = reference / norm
        self._previous = self.reference_wxyz.copy()

    def reset(self) -> None:
        self._previous = self.reference_wxyz.copy()

    def from_xyzw(self, quaternion_xyzw: Sequence[float]) -> np.ndarray:
        xyzw = np.asarray(quaternion_xyzw, dtype=np.float32).reshape(4)
        wxyz = xyzw[[3, 0, 1, 2]]
        norm = float(np.linalg.norm(wxyz))
        if not np.isfinite(norm) or norm < 1.0e-8:
            raise ValueError("TCP quaternion must be finite and non-zero")
        wxyz /= norm
        if float(np.dot(wxyz, self._previous)) < 0.0:
            wxyz = -wxyz
        self._previous = wxyz.copy()
        return wxyz


def build_proprio(
    joint_names: Sequence[str],
    joint_positions: Sequence[float],
    joint_velocities: Sequence[float],
    tcp_position_base_m: Sequence[float],
    tcp_quaternion_wxyz: Sequence[float],
) -> np.ndarray:
    """Build the exact 25-vector, independent of ROS JointState ordering."""

    names = list(joint_names)
    positions = np.asarray(joint_positions, dtype=np.float32).reshape(-1)
    velocities = np.asarray(joint_velocities, dtype=np.float32).reshape(-1)
    if len(names) != positions.size or len(names) != velocities.size:
        raise ValueError("joint name/position/velocity lengths do not match")
    if len(set(names)) != len(names):
        raise ValueError("JointState contains duplicate joint names")
    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in POLICY_JOINT_NAMES if name not in index]
    if missing:
        raise ValueError(f"JointState is missing required joints: {missing}")
    q = np.asarray([positions[index[name]] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    dq = np.asarray([velocities[index[name]] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    tcp_position = np.asarray(tcp_position_base_m, dtype=np.float32).reshape(3)
    tcp_quaternion = np.asarray(tcp_quaternion_wxyz, dtype=np.float32).reshape(4)
    quaternion_norm = float(np.linalg.norm(tcp_quaternion))
    if quaternion_norm < 1.0e-8:
        raise ValueError("TCP quaternion must be non-zero")
    tcp_quaternion = tcp_quaternion / quaternion_norm
    proprio = np.concatenate((q, dq, tcp_position, tcp_quaternion)).astype(
        np.float32, copy=False
    )
    if proprio.shape != (PROPRIO_DIM,) or not np.isfinite(proprio).all():
        raise ValueError("constructed proprio is not a finite 25-vector")
    return proprio

