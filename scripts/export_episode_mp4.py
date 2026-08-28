"""Encode one recorded successful episode's policy RGB stream as H.264 MP4."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()

    with np.load(args.episode) as episode:
        frames = episode["observation.rgb"]
        timestamps = episode["timestamp_s"]
        success = episode["success"]
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"unexpected RGB shape: {frames.shape}")
        if frames.dtype != np.uint8:
            raise ValueError(f"expected uint8 RGB, got {frames.dtype}")
        if len(frames) != len(timestamps) or len(frames) != len(success):
            raise ValueError("episode arrays are not synchronized")
        if not bool(success[-1]):
            raise ValueError(f"episode does not end in success: {args.episode}")
        if len(timestamps) < 2:
            raise ValueError("episode is too short to infer frame rate")
        dt = float(np.median(np.diff(timestamps.astype(np.float64))))
        fps = 1.0 / dt
        height, width = frames.shape[1:3]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.8f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ]
        process = subprocess.run(command, input=frames.tobytes(), check=True)
        if process.returncode != 0 or not args.output.is_file():
            raise RuntimeError(f"ffmpeg did not create {args.output}")
        print(
            f"{args.output}: {len(frames)} frames, {fps:.2f} fps, "
            f"{len(frames) / fps:.2f} s, {width}x{height}"
        )


if __name__ == "__main__":
    main()
