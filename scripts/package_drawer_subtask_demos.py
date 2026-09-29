"""Package drawer demos and derive placement subtasks from successful close runs.

Oracle rows are pre-action states. The state at row k is the result of action
k-1, so a prefix ending at state k contains exactly k action/observation rows.
Only task-dependent rewards and terminal flags are relabelled; observations
and actions are copied exactly from the source trajectory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from track_b_envs.task_ids import TASK_IDS
from track_b_envs.envs.common.constants import FRANKA_HOME_JOINTS_RAD
from track_b_envs.envs.object_to_drawer.mdp.geometry import (
    INTERIOR_ROOT_LOWER_M,
    INTERIOR_ROOT_UPPER_M,
)


def placement_predicates(oracle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror instantaneous_drawer_subtask_success on recorded base-frame states."""
    if oracle.shape[1] != 54:
        raise ValueError(f"expected 54-D drawer oracle, got {oracle.shape}")
    obj = oracle[:, 25:28]
    target = oracle[:, 45:48]
    target_wxyz = oracle[:, 48:52]
    relative = Rotation.from_quat(target_wxyz[:, [1, 2, 3, 0]]).inv().apply(obj - target)
    inside = np.all(
        (relative >= np.asarray(INTERIOR_ROOT_LOWER_M))
        & (relative <= np.asarray(INTERIOR_ROOT_UPPER_M)), axis=-1
    )
    released = oracle[:, 7:9].mean(axis=-1) >= 0.025
    slow = (np.linalg.norm(oracle[:, 32:35], axis=-1) <= 0.06) & (
        np.linalg.norm(oracle[:, 35:38], axis=-1) <= 1.0
    )
    common = inside & released & slow
    joint = oracle[:, 52]
    return common & (joint >= 0.16), common & (joint <= 0.025)


def load_source(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest["successful_episodes"] != 1:
        raise ValueError(f"expected exactly one completed success in {directory}")
    with np.load(directory / manifest["episodes"][0]["file"], allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if not arrays["success"][-1] or not arrays["terminated"][-1]:
        raise ValueError(f"source is not a successful episode: {directory}")
    if not np.isfinite(arrays["action"]).all():
        raise ValueError("nonfinite source actions")
    steps = len(arrays["action"])
    if any(len(value) != steps for value in arrays.values()):
        raise ValueError("source array lengths differ")
    if steps != manifest["episodes"][0]["steps"]:
        raise ValueError("source length does not match manifest")
    if arrays["action"].shape != (steps, 7) or np.max(np.abs(arrays["action"])) > 1.000001:
        raise ValueError("source actions do not follow the normalized 7-D contract")
    if not np.allclose(np.abs(arrays["action"][:, -1]), 1.0):
        raise ValueError("nonbinary source gripper actions")
    oracle = arrays["observation.oracle"]
    if not np.isfinite(oracle).all():
        raise ValueError("nonfinite oracle state")
    if not np.allclose(oracle[0, :7], FRANKA_HOME_JOINTS_RAD, atol=1e-5, rtol=0):
        raise ValueError("source robot does not start at standard home joints")
    dt = float(manifest["control_dt_s"])
    if not np.allclose(arrays["timestamp_s"], np.arange(1, steps + 1) * dt, atol=1e-9):
        raise ValueError("source timestamps do not match the control rate")
    required_preterminal = round(0.30 / dt) - 1
    if manifest["environment"]["task_mode"] == "open":
        # Do not package old 16 cm demos as successes under the current task.
        valid = oracle[:, 52] >= 0.21
    else:
        _, valid = placement_predicates(oracle)
    # The final post-action state is represented by the live success flag;
    # the preceding settled states remain independently visible in the oracle.
    if not np.all(valid[-required_preterminal:]):
        raise ValueError("source oracle does not support its terminal success")
    return manifest, arrays


def derive_put(manifest: dict, arrays: dict) -> tuple[dict, dict, int]:
    dt = float(manifest["control_dt_s"])
    valid_put, valid_close = placement_predicates(arrays["observation.oracle"])
    required = round(0.30 / dt)
    count = 0
    end = None
    # Skip reset state: the environment evaluates success after each action.
    for k in range(1, len(valid_put)):
        count = count + 1 if valid_put[k] else 0
        if count >= required:
            end = k
            break
    if end is None:
        raise ValueError("source has no settled open-drawer placement segment")
    result = {key: value[:end].copy() for key, value in arrays.items()}
    result["success"][:] = False
    result["terminated"][:] = False
    result["truncated"][:] = False
    result["success"][-1] = result["terminated"][-1] = True
    # The two task configurations differ only in drawer_progress and
    # task_success rewards. Isaac Lab multiplies each weighted term by dt.
    post_joint = arrays["observation.oracle"][1:end + 1, 52]
    open_progress = np.clip(post_joint / 0.22, 0.0, 1.0)
    closed_progress = np.clip(1.0 - post_joint / 0.22, 0.0, 1.0)
    result["reward"] += dt * (
        open_progress - closed_progress
        + 15.0 * (valid_put[1:end + 1].astype(float) - valid_close[1:end + 1])
    )
    out = copy.deepcopy(manifest)
    task = out["task_name"].removesuffix("_and_close")
    out.update(task_name=task, task_id=TASK_IDS[task], attempts=1)
    out["expert"] = "exact successful placement prefix of the paired close-task trajectory"
    out["environment"]["task_mode"] = "put"
    episode = out["episodes"][0]
    episode.update(steps=end, duration_s=end * dt, success_term="released banana settled inside the open drawer for 0.30 s")
    out["attempt_log"] = [{
        "attempt": 0, "success": True, "seed": episode["seed"],
        "source_env_id": episode["source_env_id"], "steps": end,
        "reason": "verified_placement_prefix", "metrics": {
            "consecutive_valid_post_action_states": required,
            "terminal_state_source_row": end,
        },
    }]
    out["derivation"] = {
        "source_task": manifest["task_name"], "source_rows": [0, end],
        "terminal_state_source_row": end,
        "unchanged": ["observations", "actions", "timestamps"],
        "relabelled": ["task-dependent rewards", "success", "terminated", "truncated"],
        "predicate": "same drawer interior bounds, release, velocity and 0.30 s settle checks as environment",
    }
    return out, result, end


def write_demo(directory: Path, manifest: dict, arrays: dict, final_rgb: np.ndarray) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "episodes").mkdir()
    episode = directory / "episodes/episode_000000.npz"
    np.savez_compressed(episode, **arrays)
    manifest["episodes"][0]["file"] = "episodes/episode_000000.npz"
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    Image.fromarray(arrays["observation.rgb"][0]).save(directory / "initial.png")
    Image.fromarray(final_rgb).save(directory / "final.png")
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("export_episode_mp4.py")),
        str(episode), str(directory / "demo.mp4"),
    ], check=True)
    return {
        "task": manifest["task_name"], "steps": len(arrays["action"]),
        "duration_s": float(arrays["timestamp_s"][-1]),
        "success": True, "video": f"{directory.name}/demo.mp4",
        "validation": "synchronized finite state/action arrays, standard home joints, recorded physical success",
        "dataset": f"{directory.name}/manifest.json",
        "episode_sha256": hashlib.sha256(episode.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-root", type=Path, required=True)
    parser.add_argument("--close-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    # Validate all four physical runs before creating the bundle.
    sources = {}
    for level in ("upper", "lower"):
        for task, root in (
            (f"open_{level}_drawer", args.open_root),
            (f"put_object_in_{level}_drawer_and_close", args.close_root),
        ):
            sources[task] = load_source(root / task)
    rows = []
    for level in ("upper", "lower"):
        open_task = f"open_{level}_drawer"
        manifest, arrays = sources[open_task]
        rows.append(write_demo(args.output_dir / open_task, manifest, arrays, arrays["observation.rgb"][-1]))
        close_task = f"put_object_in_{level}_drawer_and_close"
        manifest, arrays = sources[close_task]
        put_manifest, prefix, end = derive_put(manifest, arrays)
        if not np.array_equal(prefix["action"], arrays["action"][:end]):
            raise AssertionError("placement prefix altered source actions")
        rows.append(write_demo(args.output_dir / put_manifest["task_name"], put_manifest, prefix, arrays["observation.rgb"][end]))
        rows.append(write_demo(args.output_dir / close_task, manifest, arrays, arrays["observation.rgb"][-1]))
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    shutil.copy2(Path(__file__).parents[1] / "docs/drawer_subtasks.md", args.output_dir / "README.md")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
