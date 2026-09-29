"""Extract the two completed-placement states from each fruit trajectory.

For each order variant, the selected frames are the final frames of the first
and second ``retreat_*`` phases.  They show each placement immediately after
the gripper has moved far enough away for the newly placed fruit to be clear.
The full episode continues after the second key frame until the Franka returns
to its default joint configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("outputs/two_fruit_placement"),
)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--hold-seconds", type=float, default=2.0)
args = parser.parse_args()


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame_bgr


def final_phase_frame(phases: np.ndarray, phase: str) -> int:
    indices = np.flatnonzero(phases == phase)
    if not len(indices):
        raise ValueError(f"Trajectory does not contain phase {phase!r}")
    return int(indices[-1])


def main() -> None:
    hold_frames = max(1, round(args.hold_seconds * args.fps))
    variants = ("orange_then_banana", "banana_then_orange")
    reports = {}

    for variant in variants:
        video_path = args.output_dir / f"{variant}.mp4"
        trajectory_path = args.output_dir / f"{variant}.npz"
        with np.load(trajectory_path) as trajectory:
            order = [str(value) for value in trajectory["order"]]
            phases = trajectory["phase"]
            frame_indices = np.asarray(
                [
                    final_phase_frame(phases, f"retreat_{order[0]}"),
                    final_phase_frame(phases, f"retreat_{order[1]}"),
                ],
                dtype=np.int64,
            )
            state_labels = np.asarray(
                [f"{order[0]}_placed", "both_fruits_placed"]
            )

            variant_dir = args.output_dir / "keyframes" / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            first_image_path = variant_dir / f"01_{order[0]}_placed.png"
            both_image_path = variant_dir / "02_both_fruits_placed.png"
            image_paths = (first_image_path, both_image_path)
            frames_bgr = [
                read_video_frame(video_path, int(frame_index))
                for frame_index in frame_indices
            ]
            for image_path, frame_bgr in zip(image_paths, frames_bgr, strict=True):
                if not cv2.imwrite(str(image_path), frame_bgr):
                    raise RuntimeError(f"Could not write {image_path}")

            key_video_path = variant_dir / f"{variant}_key_states.mp4"
            writer = imageio.get_writer(
                key_video_path,
                fps=args.fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
                ffmpeg_log_level="error",
            )
            try:
                for frame_bgr in frames_bgr:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    for _ in range(hold_frames):
                        writer.append_data(frame_rgb)
            finally:
                writer.close()

            key_trajectory_path = variant_dir / f"{variant}_key_states.npz"
            np.savez_compressed(
                key_trajectory_path,
                source_frame_index=frame_indices,
                timestamp_s=trajectory["timestamp_s"][frame_indices],
                joint_position=trajectory["joint_position"][frame_indices],
                banana_pose_wxyz=trajectory["banana_pose_wxyz"][frame_indices],
                orange_pose_wxyz=trajectory["orange_pose_wxyz"][frame_indices],
                phase=phases[frame_indices],
                state_label=state_labels,
                order=np.asarray(order),
                common_initial_signature=trajectory["common_initial_signature"],
            )

        reports[variant] = {
            "order": order,
            "selection_rule": "last frame of each retreat phase",
            "source_frame_indices": frame_indices.tolist(),
            "state_labels": state_labels.tolist(),
            "images": [str(path.relative_to(args.output_dir)) for path in image_paths],
            "key_state_video": str(key_video_path.relative_to(args.output_dir)),
            "key_state_trajectory": str(
                key_trajectory_path.relative_to(args.output_dir)
            ),
            "video_hold_seconds_per_state": args.hold_seconds,
        }

    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["key_states"] = reports
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    for variant, report in reports.items():
        print(f"{variant}: frames {report['source_frame_indices']}")
        print(f"  {args.output_dir / report['images'][0]}")
        print(f"  {args.output_dir / report['images'][1]}")
        print(f"  {args.output_dir / report['key_state_video']}")


if __name__ == "__main__":
    main()
