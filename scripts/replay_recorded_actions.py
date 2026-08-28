"""Replay a recorded demonstration dataset through PhysX without an expert.

The collector records vectorized reset seeds and source environment indices.
This tool recreates those reset batches, applies only the stored normalized 7D
actions, and independently evaluates the environment termination conditions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-dir", type=Path, required=True)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="JSON report path (default: <dataset-dir>/action_replay_report.json)",
)
parser.add_argument(
    "--max-episodes",
    type=int,
    default=None,
    help="Validate only the first N episodes; intended for smoke tests.",
)
parser.add_argument("--warmup-steps", type=int, default=30)
parser.add_argument("--log-interval", type=int, default=100)
parser.add_argument(
    "--physics-only",
    action="store_true",
    help="Disable live camera rendering after statically validating stored RGB arrays.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.physics_only
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


REQUIRED_ARRAYS = (
    "observation.rgb",
    "observation.proprio",
    "observation.oracle",
    "action",
    "reward",
    "terminated",
    "truncated",
    "success",
    "timestamp_s",
)


def _load_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "track_b_successful_demonstrations_v1":
        raise ValueError(f"unsupported dataset format: {manifest.get('format')}")
    return manifest


def _load_and_check_episode(dataset_dir: Path, metadata: dict, control_dt_s: float) -> dict:
    episode_path = dataset_dir / metadata["file"]
    if not episode_path.is_file():
        raise FileNotFoundError(f"missing episode: {episode_path}")
    errors: list[str] = []
    with np.load(episode_path, allow_pickle=False) as episode:
        missing = [name for name in REQUIRED_ARRAYS if name not in episode]
        if missing:
            raise ValueError(f"{episode_path}: missing arrays {missing}")
        arrays = {name: episode[name] for name in REQUIRED_ARRAYS}

    lengths = {name: len(value) for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        errors.append(f"misaligned arrays: {lengths}")
    steps = lengths["action"]
    if steps != int(metadata["steps"]):
        errors.append(f"manifest steps={metadata['steps']}, file steps={steps}")
    if arrays["action"].shape != (steps, 7):
        errors.append(f"action shape is {arrays['action'].shape}, expected ({steps}, 7)")
    if arrays["observation.rgb"].shape[0] == steps:
        if arrays["observation.rgb"].ndim != 4 or arrays["observation.rgb"].shape[-1] != 3:
            errors.append(f"invalid RGB shape: {arrays['observation.rgb'].shape}")
        if arrays["observation.rgb"].dtype != np.uint8:
            errors.append(f"RGB dtype is {arrays['observation.rgb'].dtype}, expected uint8")
    if not np.all(np.isfinite(arrays["action"])):
        errors.append("action contains NaN or infinity")
    if np.max(np.abs(arrays["action"]), initial=0.0) > 1.0 + 1.0e-6:
        errors.append("action leaves normalized range [-1, 1]")
    gripper = arrays["action"][:, -1]
    if not np.all(np.isclose(np.abs(gripper), 1.0, atol=1.0e-6)):
        errors.append("gripper action is not strictly binary")
    timestamps = arrays["timestamp_s"]
    if len(timestamps):
        expected = np.arange(1, len(timestamps) + 1, dtype=np.float64) * control_dt_s
        if not np.allclose(timestamps, expected, atol=1.0e-9, rtol=0.0):
            errors.append("timestamps are not a uniform control-rate sequence")
    if not len(arrays["success"]) or not bool(arrays["success"][-1]):
        errors.append("recorded episode does not end with success=True")
    if not len(arrays["terminated"]) or not bool(arrays["terminated"][-1]):
        errors.append("recorded episode does not end with terminated=True")

    return {
        "metadata": metadata,
        "path": str(episode_path),
        "actions": np.asarray(arrays["action"], dtype=np.float32),
        "static_errors": errors,
    }


def _pose_errors(actual, expected):
    import torch

    expected_tensor = torch.as_tensor(expected, device=actual.device, dtype=actual.dtype)
    position_error = torch.linalg.vector_norm(actual[:3] - expected_tensor[:3])
    actual_quaternion = actual[3:7] / torch.clamp(
        torch.linalg.vector_norm(actual[3:7]), min=1.0e-12
    )
    expected_quaternion = expected_tensor[3:7] / torch.clamp(
        torch.linalg.vector_norm(expected_tensor[3:7]), min=1.0e-12
    )
    quaternion_dot = (
        torch.sum(actual_quaternion * expected_quaternion).abs().clamp(0.0, 1.0)
    )
    angle_error = 2.0 * torch.acos(quaternion_dot)
    return float(position_error.cpu()), float(angle_error.cpu())


def _initial_state_errors(task: str, actual: dict, env_id: int, expected: dict) -> dict:
    pose_key_map = {
        "push_object_to_region": {
            "object_pose": "object_pose_base_wxyz",
            "target_pose": "target_pose_base_wxyz",
        },
        "push_cube_to_region": {
            "object_pose": "object_pose_base_wxyz",
            "target_pose": "target_pose_base_wxyz",
        },
        "drawer_opening": {"handle_pose": "handle_pose_base_wxyz"},
        "cabinet_opening": {"handle_pose": "handle_pose_base_wxyz"},
        "object_to_drawer": {
            "object_pose": "object_pose_base_wxyz",
            "target_pose": "drawer_target_pose_base_wxyz",
            "handle_pose": "handle_pose_base_wxyz",
        },
        "nut_pouring": {
            "cup_pose": "cup_pose_base_wxyz",
            "nut_pose": "nut_pose_base_wxyz",
            "bowl_pose": "bowl_pose_base_wxyz",
        },
    }
    result = {}
    for actual_key, expected_key in pose_key_map[task].items():
        position_error, angle_error = _pose_errors(actual[actual_key][env_id], expected[expected_key])
        result[actual_key] = {
            "position_error_m": position_error,
            "orientation_error_rad": angle_error,
        }
    joint_keys = {
        "drawer_opening": ("joint_position", "joint_position"),
        "cabinet_opening": ("joint_position", "joint_position"),
        "object_to_drawer": ("joint_position", "drawer_joint_position"),
    }
    if task in joint_keys:
        actual_key, expected_key = joint_keys[task]
        result[actual_key] = {
            "absolute_error": abs(float(actual[actual_key][env_id].cpu()) - float(expected[expected_key]))
        }
    return result


def _initial_state_matches(errors: dict) -> bool:
    for values in errors.values():
        if values.get("position_error_m", 0.0) > 1.0e-4:
            return False
        if values.get("orientation_error_rad", 0.0) > 1.0e-3:
            return False
        if values.get("absolute_error", 0.0) > 1.0e-5:
            return False
    return True


def _task_diagnostics(task: str, state: dict, env_id: int) -> dict:
    keys = {
        "push_object_to_region": ("object_pose", "target_pose"),
        "push_cube_to_region": ("object_pose", "target_pose"),
        "drawer_opening": ("ee_pose", "handle_pose", "finger_positions", "joint_position"),
        "cabinet_opening": ("ee_pose", "handle_pose", "finger_positions", "joint_position"),
        "object_to_drawer": (
            "ee_pose",
            "object_pose",
            "target_pose",
            "finger_positions",
            "joint_position",
            "joint_velocity",
        ),
        "nut_pouring": (
            "ee_pose",
            "cup_pose",
            "nut_pose",
            "bowl_pose",
            "finger_positions",
            "nut_linear_velocity",
        ),
    }[task]
    return {
        key: state[key][env_id].detach().cpu().tolist()
        for key in keys
        if key in state
    }


def main() -> dict:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from track_b_envs.collection.experts.common import (
        nut_pouring_task_state,
        object_to_drawer_task_state,
        opening_task_state,
        task_space_state,
    )
    from track_b_envs.envs.articulated_opening import CabinetOpeningEnvCfg, DrawerOpeningEnvCfg
    from track_b_envs.envs.nut_pouring import NutPouringEnvCfg
    from track_b_envs.envs.object_to_drawer import ObjectToDrawerEnvCfg
    from track_b_envs.envs.push_object_to_region import (
        PushCubeToRegionEnvCfg,
        PushObjectToRegionEnvCfg,
    )

    dataset_dir = args.dataset_dir.resolve()
    manifest = _load_manifest(dataset_dir)
    task = manifest["task_name"]
    task_map = {
        "push_object_to_region": (
            PushObjectToRegionEnvCfg,
            task_space_state,
        ),
        "push_cube_to_region": (
            PushCubeToRegionEnvCfg,
            task_space_state,
        ),
        "drawer_opening": (
            DrawerOpeningEnvCfg,
            lambda env: opening_task_state(env, "UpperDrawerJoint"),
        ),
        "cabinet_opening": (
            CabinetOpeningEnvCfg,
            lambda env: opening_task_state(env, "DoorHingeJoint"),
        ),
        "object_to_drawer": (ObjectToDrawerEnvCfg, object_to_drawer_task_state),
        "nut_pouring": (NutPouringEnvCfg, nut_pouring_task_state),
    }
    if task not in task_map:
        raise ValueError(f"unsupported task in manifest: {task}")
    metadata = list(manifest["episodes"])
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes must be positive")
        metadata = metadata[: args.max_episodes]
    control_dt_s = float(manifest["control_dt_s"])
    episodes = [
        _load_and_check_episode(dataset_dir, item, control_dt_s) for item in metadata
    ]
    static_failures = sum(bool(item["static_errors"]) for item in episodes)
    # Preserve the collector's vectorization width even for --max-episodes
    # smoke tests. Reset samplers consume one stream element per environment,
    # so shrinking the batch can otherwise change later sampled quantities.
    num_envs = max(int(item["source_env_id"]) for item in manifest["episodes"]) + 1
    groups = defaultdict(dict)
    for episode in episodes:
        seed = int(episode["metadata"]["seed"])
        env_id = int(episode["metadata"]["source_env_id"])
        if env_id in groups[seed]:
            raise ValueError(f"duplicate episode for reset stream seed={seed}, env={env_id}")
        groups[seed][env_id] = episode

    cfg_cls, state_fn = task_map[task]
    cfg = cfg_cls()
    if args.physics_only:
        # RGB has already been decompressed and checked above. Camera rendering
        # cannot affect these task dynamics and needlessly slows a 250-episode
        # action replay, so remove the sensor and observation term together.
        cfg.observations.policy.rgb = None
        cfg.scene.policy_camera = None
    cfg.scene.num_envs = num_envs
    cfg.sim.device = args.device
    cfg.seed = int(manifest.get("dataset_metadata", {}).get("seed", metadata[0]["seed"]))
    replay_control_dt_s = float(cfg.sim.dt * cfg.decimation)
    if not math.isclose(replay_control_dt_s, control_dt_s, abs_tol=1.0e-9):
        raise RuntimeError(
            f"dataset control dt={control_dt_s}, environment control dt={replay_control_dt_s}"
        )
    cfg.episode_length_s = max(
        cfg.episode_length_s,
        max(len(item["actions"]) for item in episodes) * control_dt_s + control_dt_s,
    )

    env = ManagerBasedRLEnv(cfg=cfg)
    results = []
    try:
        if args.warmup_steps > 0:
            env.reset(seed=cfg.seed)
            zero_action = torch.zeros(env.action_space.shape, device=env.device)
            for _ in range(args.warmup_steps):
                env.step(zero_action)

        for group_index, (seed, group) in enumerate(sorted(groups.items())):
            env.reset(seed=seed)
            initial = state_fn(env)
            active = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
            group_results = {}
            for env_id, episode in group.items():
                initial_errors = _initial_state_errors(
                    task,
                    initial,
                    env_id,
                    episode["metadata"]["initial_state"],
                )
                result = {
                    "episode_index": int(episode["metadata"]["episode_index"]),
                    "file": episode["metadata"]["file"],
                    "seed": seed,
                    "source_env_id": env_id,
                    "recorded_steps": len(episode["actions"]),
                    "static_errors": episode["static_errors"],
                    "initial_state_matches": _initial_state_matches(initial_errors),
                    "initial_state_errors": initial_errors,
                    "replay_success": False,
                    "replay_step": None,
                    "reason": "recorded_action_horizon_exhausted",
                }
                group_results[env_id] = result
                active[env_id] = True

            max_steps = max(len(episode["actions"]) for episode in group.values())
            for step in range(max_steps):
                actions = torch.zeros(env.action_space.shape, device=env.device)
                for env_id, episode in group.items():
                    if bool(active[env_id]) and step < len(episode["actions"]):
                        actions[env_id] = torch.as_tensor(
                            episode["actions"][step], device=env.device
                        )
                _, _, terminated, truncated, _ = env.step(actions)
                success = env.termination_manager.get_term("success").clone()
                ended = active & (terminated | truncated)
                for env_id in ended.nonzero(as_tuple=False).squeeze(-1).tolist():
                    result = group_results[env_id]
                    result["replay_success"] = bool(success[env_id].cpu())
                    result["replay_step"] = step + 1
                    if bool(success[env_id]):
                        result["reason"] = "success"
                    elif bool(truncated[env_id]):
                        result["reason"] = "time_out"
                    else:
                        active_terms = {
                            name: bool(value[0])
                            for name, value in env.termination_manager.get_active_iterable_terms(env_id)
                        }
                        result["reason"] = next(
                            (name for name, value in active_terms.items() if value),
                            "failure_termination",
                        )
                    result["final_task_state"] = _task_diagnostics(
                        task, state_fn(env), env_id
                    )
                active[ended] = False
                exhausted = []
                for env_id, episode in group.items():
                    if bool(active[env_id]) and step + 1 >= len(episode["actions"]):
                        exhausted.append(env_id)
                if exhausted:
                    final_state = state_fn(env)
                    for env_id in exhausted:
                        group_results[env_id]["final_task_state"] = _task_diagnostics(
                            task, final_state, env_id
                        )
                    active[torch.as_tensor(exhausted, device=env.device)] = False
                if args.log_interval > 0 and step % args.log_interval == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "replay_progress",
                                "task": task,
                                "reset_group": group_index + 1,
                                "reset_groups": len(groups),
                                "seed": seed,
                                "step": step,
                                "active": int(active.sum().cpu()),
                            }
                        ),
                        flush=True,
                    )
                if not torch.any(active):
                    break
            results.extend(group_results.values())
            passed_so_far = sum(item["replay_success"] for item in results)
            print(
                json.dumps(
                    {
                        "stage": "reset_group_complete",
                        "task": task,
                        "reset_group": group_index + 1,
                        "reset_groups": len(groups),
                        "validated": len(results),
                        "successful": passed_so_far,
                    }
                ),
                flush=True,
            )
    finally:
        env.close()

    replay_successes = sum(item["replay_success"] for item in results)
    initial_matches = sum(item["initial_state_matches"] for item in results)
    report = {
        "status": (
            "passed"
            if replay_successes == len(results) and static_failures == 0 and initial_matches == len(results)
            else "failed"
        ),
        "validation_contract": "stored normalized 7D actions only; no expert execution or state overwrite",
        "dataset_dir": str(dataset_dir),
        "task": task,
        "control_hz": round(1.0 / control_dt_s),
        "live_camera_rendering": not args.physics_only,
        "episodes": len(results),
        "static_passes": len(results) - static_failures,
        "initial_state_matches": initial_matches,
        "replay_successes": replay_successes,
        "replay_success_rate": replay_successes / len(results),
        "results": sorted(results, key=lambda item: item["episode_index"]),
    }
    output = args.output or dataset_dir / "action_replay_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(json.dumps({"stage": "report_written", "path": str(output)}), flush=True)
    return report


if __name__ == "__main__":
    try:
        result = main()
        if result["status"] != "passed":
            raise RuntimeError(
                f"{result['task']}: replayed {result['replay_successes']}/{result['episodes']} successes"
            )
    finally:
        simulation_app.close()
