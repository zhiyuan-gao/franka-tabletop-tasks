#!/usr/bin/env python3
"""HTTP inference server for the selected Franka drawer-opening policy.

The server is deliberately policy-specific.  It accepts one newest real-robot
observation per 20 Hz request and keeps the checkpoint's seven-control-step
history and eight-action temporal ensemble per client session.  It can serve
either the frozen BC policy alone or the deployable BC + residual TD3+BC actor.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException

from task_pipeline_v2 import (
    ChunkRunner,
    compose_action_v2,
    load_bc_checkpoint_v2,
    load_residual_actor_v2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs/drawer_opening_bcv2_seed42/bc_selected.pt"
)
TASK_NAME = "drawer_opening"
TRANSPORT_CAMERA_VIEW = "right"
MODEL_CAMERA_KEY = "rgb"
POLICY_RATE_HZ = 20.0
SOURCE_POLICY_IMAGE_SIZE = 224
SIM_TRANSLATION_SCALE_M = (0.05, 0.05, 0.05)
SIM_ROTATION_SCALE_RAD = (0.08, 0.08, 0.08)
ACTION_NAMES = (
    "dx_base_normalized",
    "dy_base_normalized",
    "dz_base_normalized",
    "d_axis_angle_x_base_normalized",
    "d_axis_angle_y_base_normalized",
    "d_axis_angle_z_base_normalized",
    "gripper_open_plus1_close_minus1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (length,):
        raise ValueError(f"{name} shape is {array.shape}, expected ({length},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def decode_transport_image(image_payload: Dict[str, Any]) -> np.ndarray:
    if not isinstance(image_payload, dict):
        raise ValueError("right_image must be an object")
    image_format = image_payload.get("format")
    encoded_key = {
        "png_base64": "png_base64",
        "jpeg_base64": "jpeg_base64",
    }.get(image_format)
    if encoded_key is None:
        raise ValueError(
            f"right_image.format={image_format!r}; expected png_base64 or jpeg_base64"
        )
    encoded = image_payload.get(encoded_key)
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"right_image.{encoded_key} is missing")
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError(f"right_image.{encoded_key} is invalid base64: {exc}") from exc
    bgr = cv2.imdecode(np.frombuffer(compressed, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("right_image could not be decoded as a three-channel image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def real_image_to_policy_input(
    rgb: np.ndarray,
    device: torch.device,
    policy_view_preprocessed: bool,
) -> torch.Tensor:
    """Center-crop to square, resize/round to training's saved 224 RGB path."""

    if policy_view_preprocessed:
        if rgb.shape != (SOURCE_POLICY_IMAGE_SIZE, SOURCE_POLICY_IMAGE_SIZE, 3):
            raise ValueError(
                f"preprocessed RGB shape is {rgb.shape}, expected "
                f"({SOURCE_POLICY_IMAGE_SIZE}, {SOURCE_POLICY_IMAGE_SIZE}, 3)"
            )
        return (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
            .div_(255.0)
        )

    height, width = rgb.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    square = np.ascontiguousarray(rgb[top : top + side, left : left + side])
    tensor = torch.from_numpy(square).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)
    tensor = F.interpolate(
        tensor,
        size=(SOURCE_POLICY_IMAGE_SIZE, SOURCE_POLICY_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return tensor.round().clamp_(0.0, 255.0).div_(255.0)


@dataclass
class SessionState:
    runner: ChunkRunner
    last_timestamp_ns: int = -1
    request_count: int = 0


class DrawerPolicyService:
    def __init__(
        self,
        checkpoint: Path,
        device_name: str,
        residual_checkpoint: Path | None = None,
    ) -> None:
        self.checkpoint = checkpoint.resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.policy, self.normalizer, self.payload = load_bc_checkpoint_v2(
            self.checkpoint,
            self.device,
            expected_task=TASK_NAME,
            prefer_ema=True,
        )
        self.config = dict(self.payload["config"])
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.residual_checkpoint = (
            residual_checkpoint.resolve() if residual_checkpoint is not None else None
        )
        self.residual_actor = None
        self.residual_payload: Dict[str, Any] | None = None
        self.residual_checkpoint_sha256: str | None = None
        self.gripper_authority = 0.0
        self.flip_margin = 0.0
        if self.residual_checkpoint is not None:
            if not self.residual_checkpoint.is_file():
                raise FileNotFoundError(self.residual_checkpoint)
            self.residual_actor, self.residual_payload = load_residual_actor_v2(
                self.residual_checkpoint,
                self.device,
                expected_task=TASK_NAME,
            )
            recorded_bc = Path(self.residual_payload["bc_checkpoint"]).resolve()
            if recorded_bc != self.checkpoint:
                raise ValueError(
                    "residual checkpoint was trained with a different BC checkpoint: "
                    f"{recorded_bc} != {self.checkpoint}"
                )
            residual_config = dict(self.residual_payload["config"])
            residual_extra = dict(self.residual_payload.get("extra", {}))
            self.gripper_authority = float(
                residual_extra.get(
                    "gripper_authority",
                    residual_config.get("gripper_authority", 0.0),
                )
            )
            self.flip_margin = float(residual_config.get("flip_margin", 0.25))
            self.residual_checkpoint_sha256 = sha256_file(self.residual_checkpoint)
        self.history_length = (
            (int(self.config["frame_stack"]) - 1) * int(self.config["frame_stride"]) + 1
        )
        self.sessions: Dict[str, SessionState] = {}
        self.lock = threading.RLock()
        self.started_wall_time = time.time()
        self._warm_up()

    def new_runner(self) -> ChunkRunner:
        return ChunkRunner(
            self.policy,
            self.normalizer,
            num_envs=1,
            device=self.device,
            frame_stack=int(self.config["frame_stack"]),
            frame_stride=int(self.config["frame_stride"]),
            chunk=int(self.config["chunk"]),
            image_size=int(self.config["image_size"]),
            ensemble_decay=0.35,
            use_ensemble=True,
        )

    def _warm_up(self) -> None:
        runner = self.new_runner()
        rgb = torch.zeros(
            (1, 3, SOURCE_POLICY_IMAGE_SIZE, SOURCE_POLICY_IMAGE_SIZE),
            device=self.device,
        )
        proprio = torch.zeros((1, 25), device=self.device)
        with torch.inference_mode():
            self._run_policy(runner, rgb, proprio)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _run_policy(
        self,
        runner: ChunkRunner,
        rgb: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_action, feature = runner.step(rgb, proprio)
        if self.residual_actor is None:
            residual = torch.zeros_like(base_action)
            return base_action, base_action, residual
        residual = self.residual_actor(feature, base_action, proprio)
        action = compose_action_v2(
            base_action,
            residual,
            self.gripper_authority,
            self.flip_margin,
        )
        return action, base_action, residual

    def schema(self) -> Dict[str, Any]:
        schema = {
            "schema_version": 2,
            "task": TASK_NAME,
            "ready": True,
            "policy_type": "bc_residual" if self.residual_actor is not None else "bc",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_format": self.payload.get("format"),
            "checkpoint_epoch": int(self.payload.get("epoch", -1)),
            "weights": "ema_state_dict",
            "device": str(self.device),
            "policy_rate_hz": POLICY_RATE_HZ,
            "transport": {
                "camera_views": [TRANSPORT_CAMERA_VIEW],
                "camera_field": "right_image",
                "accepted_encodings": ["png_base64", "jpeg_base64"],
                "observations_per_request": 1,
                "session_stateful": True,
                "first_request_must_set_reset": True,
                "preferred_input": "client_preprocessed lossless 224x224 RGB policy view",
            },
            "image_preprocess": {
                "decoded_color": "RGB",
                "crop": "center square",
                "policy_view_resize": [224, 224],
                "policy_view_interpolation": "torch bilinear align_corners=False",
                "policy_view_quantization": "round/clamp uint8 domain before float32 / 255",
                "client_preprocessed_contract": (
                    "center square then torch-bilinear-align_corners_false-equivalent "
                    "resize/round to uint8 224x224 before lossless PNG"
                ),
                "model_resize": [int(self.config["image_size"])] * 2,
                "model_normalization": "float32 [0,1]; no ImageNet mean/std",
                "model_camera_key": MODEL_CAMERA_KEY,
            },
            "proprio": {
                "shape": [25],
                "ordering": [
                    "arm_joint_position_rad[7]",
                    "finger_joint_position_m[2]",
                    "arm_joint_velocity_rad_s[7]",
                    "finger_joint_velocity_m_s[2]",
                    "tcp_position_base_m[3]",
                    "tcp_quaternion_base_wxyz[4]",
                ],
                "tcp_frame_sim": "panda_hand + local z offset 0.1034 m",
                "base_frame": "panda_link0",
            },
            "recorded_reset_state": {
                "arm_joint_position_rad": [
                    0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398
                ],
                "finger_joint_position_m": [0.04, 0.04],
                "tcp_position_base_m": [0.3068907, -1.19209e-7, 0.4868819],
                "tcp_quaternion_base_wxyz": [3.527010e-7, 1.0, 2.215497e-7, 3.200540e-8],
                "source": "all 50 primary drawer_opening episodes have this first state",
            },
            "temporal": {
                "frame_stack": int(self.config["frame_stack"]),
                "frame_stride_control_steps": int(self.config["frame_stride"]),
                "history_buffer_control_steps": self.history_length,
                "history_samples": ["t-6", "t-3", "t"],
                "model_action_chunk": int(self.config["chunk"]),
                "temporal_ensemble": True,
                "ensemble_decay": 0.35,
                "response_action_horizon": 1,
                "execute_horizon": 1,
            },
            "action": {
                "response_shape": [1, 7],
                "names": list(ACTION_NAMES),
                "network_output_unnormalized_by_checkpoint_normalizer": True,
                "arm_domain": "simulator normalized command, normally clipped to [-1,1]",
                "translation_frame": "panda_link0/base",
                "action_control_point": "panda_hand + local z 0.107 m",
                "control_point_from_proprio_tcp_local_m": [0.0, 0.0, 0.0036],
                "translation_sim_scale_m_per_step": list(SIM_TRANSLATION_SCALE_M),
                "rotation": "base-frame axis-angle vector, radians after scale",
                "rotation_sim_scale_rad_per_step": list(SIM_ROTATION_SCALE_RAD),
                "rotation_composition": "q_target_wxyz = q_delta_wxyz * q_current_wxyz",
                "gripper": "+1 open, -1 close",
                "real_robot_note": "apply approved reduced scale and safety limits locally",
            },
            "normalizer": self.payload["normalizer"],
            "active_sessions": len(self.sessions),
            "uptime_s": time.time() - self.started_wall_time,
        }
        if self.residual_actor is not None:
            assert self.residual_checkpoint is not None
            assert self.residual_payload is not None
            schema["residual"] = {
                "checkpoint": str(self.residual_checkpoint),
                "checkpoint_sha256": self.residual_checkpoint_sha256,
                "checkpoint_format": self.residual_payload.get("format"),
                "base_checkpoint": str(self.checkpoint),
                "composition": "clip(base_arm + residual_arm, -1, 1); gripper from BC when authority=0",
                "arm_residual_limit_normalized_per_dimension": float(
                    self.residual_payload["arm_limit"]
                ),
                "gripper_authority": self.gripper_authority,
                "flip_margin": self.flip_margin,
                "training_transitions": int(
                    self.residual_payload.get("extra", {}).get("transitions", -1)
                ),
                "selected_eval_success_rate": self.residual_payload.get("extra", {}).get(
                    "eval_success_rate"
                ),
                "weights": "actor_state_dict",
            }
        return schema

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if request.get("schema_version") != 2:
            raise ValueError("schema_version must be 2")
        if request.get("task_name") != TASK_NAME:
            raise ValueError(f"task_name must be {TASK_NAME!r}")
        if request.get("camera_views") != [TRANSPORT_CAMERA_VIEW]:
            raise ValueError(f"camera_views must be [{TRANSPORT_CAMERA_VIEW!r}]")
        session_id = request.get("session_id")
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("session_id must be a non-empty string of at most 128 characters")
        reset = request.get("reset")
        if not isinstance(reset, bool):
            raise ValueError("reset must be boolean")
        observations = request.get("observations")
        if not isinstance(observations, list) or len(observations) != 1:
            raise ValueError("observations must contain exactly one newest observation")
        observation = observations[0]
        if not isinstance(observation, dict):
            raise ValueError("observation must be an object")
        timestamp_ns = observation.get("timestamp_ns")
        if not isinstance(timestamp_ns, int) or timestamp_ns <= 0:
            raise ValueError("observation.timestamp_ns must be a positive integer")

        decode_start = time.monotonic()
        right_image = observation.get("right_image")
        rgb_array = decode_transport_image(right_image)
        if not isinstance(right_image, dict) or not right_image.get("policy_view_preprocessed"):
            raise ValueError("right_image.policy_view_preprocessed must be true")
        if rgb_array.shape != (224, 224, 3):
            raise ValueError(
                f"preprocessed right image shape is {rgb_array.shape}, expected (224, 224, 3)"
            )
        proprio_array = finite_vector(observation.get("proprio_25d"), 25, "proprio_25d")
        rgb = real_image_to_policy_input(rgb_array, self.device, True)
        proprio = torch.from_numpy(proprio_array).unsqueeze(0).to(self.device)
        preprocess_ms = (time.monotonic() - decode_start) * 1000.0

        with self.lock:
            state = self.sessions.get(session_id)
            if state is None:
                if not reset:
                    raise ValueError("new session must set reset=true")
                state = SessionState(self.new_runner())
                self.sessions[session_id] = state
            elif reset:
                state = SessionState(self.new_runner())
                self.sessions[session_id] = state
            if not reset and timestamp_ns <= state.last_timestamp_ns:
                raise ValueError(
                    "observation.timestamp_ns is stale or out of order: "
                    f"{timestamp_ns} <= {state.last_timestamp_ns}"
                )

            inference_start = time.monotonic()
            with torch.inference_mode():
                action, base_action, residual = self._run_policy(
                    state.runner,
                    rgb,
                    proprio,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_ms = (time.monotonic() - inference_start) * 1000.0
            action_array = action.detach().cpu().numpy().astype(np.float32, copy=False)
            if action_array.shape != (1, 7):
                raise RuntimeError(f"model returned shape {action_array.shape}, expected (1, 7)")
            if not np.all(np.isfinite(action_array)):
                raise RuntimeError("model returned NaN or Inf")
            state.last_timestamp_ns = timestamp_ns
            state.request_count += 1
            request_count = state.request_count

        normalized = action_array[0]
        base_action_array = base_action.detach().cpu().numpy().astype(np.float32, copy=False)
        residual_array = residual.detach().cpu().numpy().astype(np.float32, copy=False)
        sim_delta = np.zeros(7, dtype=np.float32)
        sim_delta[:3] = normalized[:3] * np.asarray(SIM_TRANSLATION_SCALE_M, dtype=np.float32)
        sim_delta[3:6] = normalized[3:6] * np.asarray(SIM_ROTATION_SCALE_RAD, dtype=np.float32)
        sim_delta[6] = normalized[6]
        return {
            "schema_version": 2,
            "task_name": TASK_NAME,
            "session_id": session_id,
            "actions": action_array.tolist(),
            "diagnostics": {
                "reset_applied": reset,
                "session_request_count": request_count,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "input_image_shape_hwc": list(rgb_array.shape),
                "input_proprio_shape": list(proprio_array.shape),
                "action_shape": list(action_array.shape),
                "action_all_finite": True,
                "policy_type": "bc_residual" if self.residual_actor is not None else "bc",
                "base_action": base_action_array.tolist(),
                "residual_action": residual_array.tolist(),
                "residual_arm_abs_max": float(np.max(np.abs(residual_array[:, :6]))),
                "gripper_authority": self.gripper_authority,
                "simulator_scaled_delta_for_diagnostics_only": sim_delta.tolist(),
                "simulator_scaled_position_norm_m": float(np.linalg.norm(sim_delta[:3])),
                "simulator_scaled_rotation_norm_rad": float(np.linalg.norm(sim_delta[3:6])),
            },
        }


def create_app(service: DrawerPolicyService) -> FastAPI:
    app = FastAPI(title="Franka Drawer Opening Policy", version="2")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return service.schema()

    @app.get("/schema")
    def schema() -> Dict[str, Any]:
        return service.schema()

    @app.post("/predict")
    def predict(request: Dict[str, Any]) -> Dict[str, Any]:
        start = time.monotonic()
        try:
            response = service.predict(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        response["diagnostics"]["server_total_ms"] = (time.monotonic() - start) * 1000.0
        return response

    return app


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--residual-checkpoint",
        type=Path,
        default=None,
        help="optional residual TD3+BC actor; its recorded BC checkpoint must match --checkpoint",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    service = DrawerPolicyService(
        args.checkpoint,
        args.device,
        residual_checkpoint=args.residual_checkpoint,
    )
    uvicorn.run(create_app(service), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
