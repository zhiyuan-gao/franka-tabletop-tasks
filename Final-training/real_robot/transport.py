"""Versioned ZeroMQ wire protocol for robot-to-server policy inference."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import msgpack
import numpy as np


PROTOCOL_VERSION = 1
ACTION_DIM = 7
PROPRIO_DIM = 25
ALLOWED_RGB_SHAPES = ((720, 720, 3), (224, 224, 3))


class ProtocolError(ValueError):
    pass


def pack_header(header: dict[str, Any]) -> bytes:
    return msgpack.packb(header, use_bin_type=True)


def unpack_header(payload: bytes) -> dict[str, Any]:
    value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    if not isinstance(value, dict):
        raise ProtocolError("wire header is not a map")
    return value


@dataclass(frozen=True)
class ObservationPacket:
    session_id: str
    sequence: int
    sensor_time_ns: int
    profile: str
    reset: bool
    rgb: np.ndarray
    proprio: np.ndarray

    def to_frames(self) -> list[bytes]:
        rgb = np.ascontiguousarray(self.rgb)
        proprio = np.ascontiguousarray(self.proprio, dtype=np.float32)
        if rgb.dtype != np.uint8 or tuple(rgb.shape) not in ALLOWED_RGB_SHAPES:
            raise ProtocolError(f"unsupported RGB payload: {rgb.shape} {rgb.dtype}")
        if proprio.shape != (PROPRIO_DIM,) or not np.isfinite(proprio).all():
            raise ProtocolError("proprio must be a finite float32[25]")
        header = {
            "version": PROTOCOL_VERSION,
            "kind": "observation",
            "session_id": str(self.session_id),
            "sequence": int(self.sequence),
            "sensor_time_ns": int(self.sensor_time_ns),
            "client_send_time_ns": time.time_ns(),
            "profile": str(self.profile),
            "reset": bool(self.reset),
            "rgb_shape": list(rgb.shape),
            "rgb_encoding": "rgb8",
            "proprio_shape": [PROPRIO_DIM],
            "proprio_dtype": "float32-le",
        }
        return [pack_header(header), rgb.tobytes(), proprio.astype("<f4", copy=False).tobytes()]

    @classmethod
    def from_frames(cls, frames: list[bytes]) -> "ObservationPacket":
        if len(frames) != 3:
            raise ProtocolError(f"observation must contain 3 frames, got {len(frames)}")
        header = unpack_header(frames[0])
        if header.get("version") != PROTOCOL_VERSION or header.get("kind") != "observation":
            raise ProtocolError("unsupported observation protocol/version")
        shape = tuple(int(v) for v in header.get("rgb_shape", ()))
        if shape not in ALLOWED_RGB_SHAPES or header.get("rgb_encoding") != "rgb8":
            raise ProtocolError(f"unsupported RGB metadata: {shape}")
        expected_rgb_bytes = int(np.prod(shape))
        if len(frames[1]) != expected_rgb_bytes:
            raise ProtocolError("RGB byte count does not match its declared shape")
        if len(frames[2]) != PROPRIO_DIM * np.dtype("<f4").itemsize:
            raise ProtocolError("proprio byte count is invalid")
        rgb = np.frombuffer(frames[1], dtype=np.uint8).reshape(shape).copy()
        proprio = np.frombuffer(frames[2], dtype="<f4").astype(np.float32, copy=True)
        if not np.isfinite(proprio).all():
            raise ProtocolError("proprio contains NaN or infinity")
        return cls(
            session_id=str(header["session_id"]),
            sequence=int(header["sequence"]),
            sensor_time_ns=int(header["sensor_time_ns"]),
            profile=str(header["profile"]),
            reset=bool(header.get("reset", False)),
            rgb=rgb,
            proprio=proprio,
        )


def encode_result(result: dict[str, Any]) -> bytes:
    payload = {"version": PROTOCOL_VERSION, "kind": "result", **result}
    return pack_header(payload)


def decode_result(payload: bytes) -> dict[str, Any]:
    result = unpack_header(payload)
    if result.get("version") != PROTOCOL_VERSION or result.get("kind") != "result":
        raise ProtocolError("unsupported result protocol/version")
    if bool(result.get("ok", False)):
        for key in ("base_action", "residual_action", "final_action"):
            value = np.asarray(result.get(key), dtype=np.float32)
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise ProtocolError(f"invalid {key} in result")
    return result

