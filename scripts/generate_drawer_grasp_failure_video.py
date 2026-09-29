"""Render one controlled, single-attempt grasp failure using live physics.

The scene, reset seed, and success predicate are unchanged. Only the attempted
grasp waypoint is perturbed. No robot/object state is overwritten after reset.
The only output artifact is an MP4; validation measurements go to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, choices=(
    "open_upper_drawer", "open_lower_drawer",
    "put_object_in_upper_drawer", "put_object_in_lower_drawer",
    "put_object_in_upper_drawer_and_close", "put_object_in_lower_drawer_and_close",
))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=20260827)
parser.add_argument("--max-steps", type=int, default=400)
parser.add_argument("--full-resolution", action="store_true",
    help="Capture the native calibrated 1280x720 composite instead of policy RGB.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main():
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply, quat_mul
    from isaaclab_tasks.utils import parse_env_cfg

    from track_b_envs.task_ids import TASK_IDS
    from track_b_envs.collection.experts.drawer_subtasks import DrawerSubtaskExpert
    from track_b_envs.collection.experts.common import (
        absolute_pose_to_relative_action, advance_pose_toward_waypoint,
        drawer_subtask_state,
    )
    from track_b_envs.collection.recorder import policy_rgb_to_uint8_hwc
    from track_b_envs.envs.common.constants import FRANKA_HOME_JOINTS_RAD
    from track_b_envs.envs.common.compositing import RealBackgroundCompositor

    cfg = parse_env_cfg(TASK_IDS[args.task], device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.episode_length_s = max(cfg.episode_length_s, (args.max_steps + 2) * 0.05)
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        warmup = torch.zeros(env.action_space.shape, device=env.device)
        for _ in range(10):
            env.step(warmup)
        observations, _ = env.reset(seed=args.seed)
        camera = env.scene["policy_camera"]
        compositor = RealBackgroundCompositor(env.device) if args.full_resolution else None
        if compositor is not None:
            env.sim.render()
            camera.update(0.0, force_recompute=True)
        baseline = DrawerSubtaskExpert(env)
        baseline.reset(env)
        expert = baseline._expert
        initial = drawer_subtask_state(env, cfg.active_joint_name)
        initial_object = initial["object_pose"][0, :3].clone()
        initial_joint = initial["joint_position"].clone()
        assert torch.allclose(
            env.scene["robot"].data.joint_pos[0, :7],
            torch.tensor(FRANKA_HOME_JOINTS_RAD, device=env.device), atol=1e-5, rtol=0,
        )
        opening = cfg.task_mode == "open"
        dt = float(cfg.sim.dt * cfg.decimation)
        scale = torch.tensor(cfg.actions.arm_action.scale, device=env.device)
        phase = "baseline_approach"
        phase_steps = 0
        frames, grippers = [], []
        maximum_lift = 0.0
        maximum_joint_change = 0.0
        grasp_pose = retreat_pose = command = None
        finished = False
        for step in range(args.max_steps):
            state = drawer_subtask_state(env, cfg.active_joint_name)
            ee = state["ee_pose"]
            if phase == "baseline_approach":
                action = baseline.compute(env)
                takeover = (int(expert.state[0]) == (
                    expert.APPROACH_HANDLE if opening else expert.DESCEND_OBJECT
                ))
                if takeover:
                    if opening:
                        front = quat_apply(state["handle_pose"][:, 3:7],
                            torch.tensor([[0., 1., 0.]], device=env.device))
                        grasp_pose = state["handle_pose"].clone()
                        grasp_pose[:, :3] += 0.040 * front
                        grasp_pose[:, 3:7] = quat_mul(
                            state["handle_pose"][:, 3:7],
                            torch.tensor([[0.70710678, 0.70710678, 0., 0.]], device=env.device),
                        )
                        retreat_pose = grasp_pose.clone()
                        retreat_pose[:, :3] += 0.160 * front
                        retreat_pose[:, 2] += 0.060
                    else:
                        grasp_pose = expert.grasp_pose.clone()
                        # Miss beside the fruit at its height, keeping the
                        # failed finger closure visible in the policy camera.
                        grasp_pose[:, 0] -= 0.085
                        retreat_pose = grasp_pose.clone()
                        # Withdraw vertically clear of the cabinet after the
                        # empty closure; do not sweep the palm across its roof.
                        retreat_pose[:, 2] += 0.075
                    command = expert.command_pose.clone()
                    phase, phase_steps = "miss_approach", 0
            if phase != "baseline_approach":
                target = retreat_pose if phase in {"empty_withdraw", "final_hold"} else grasp_pose
                closed = phase in {"close_empty", "empty_withdraw", "final_hold"}
                speed = 0.10 if phase == "empty_withdraw" else 0.05
                command = advance_pose_toward_waypoint(command, target, speed * dt, 1.0 * dt)
                action = absolute_pose_to_relative_action(
                    ee, command, torch.tensor([-1. if closed else 1.], device=env.device),
                    scale, max_translation_step_m=1.0 * dt,
                    max_rotation_step_rad=1.6 * dt,
                )
                phase_steps += 1
                error = float(torch.linalg.vector_norm(ee[:, :3] - target[:, :3]))
                if phase == "miss_approach" and error < 0.015:
                    phase, phase_steps = "settle_open", 0
                elif phase == "settle_open" and phase_steps * dt >= 0.40:
                    phase, phase_steps = "close_empty", 0
                elif phase == "close_empty" and phase_steps * dt >= 1.00:
                    if float(state["finger_positions"].max()) >= 0.0015:
                        raise RuntimeError("gripper is not empty: refusing to label a grasp failure")
                    phase, phase_steps = "empty_withdraw", 0
                elif phase == "empty_withdraw" and error < 0.030:
                    phase, phase_steps = "final_hold", 0
                elif phase == "final_hold" and phase_steps * dt >= 1.00:
                    finished = True
            if compositor is not None:
                frames.append(compositor.compose(
                    camera.data.output["rgb"],
                    camera.data.output["semantic_segmentation"],
                    camera.data.intrinsic_matrices,
                )[0].cpu().numpy())
            else:
                frames.append(policy_rgb_to_uint8_hwc(observations["policy"]["rgb"][0]))
            grippers.append(float(action[0, -1]))
            observations, _, terminated, truncated, _ = env.step(action)
            if bool(env.termination_manager.get_term("success")[0]):
                raise RuntimeError("task unexpectedly succeeded; refusing failure export")
            if bool(terminated[0] | truncated[0]):
                raise RuntimeError("unexpected environment termination before failure validation")
            after = drawer_subtask_state(env, cfg.active_joint_name)
            maximum_lift = max(maximum_lift, float(after["object_pose"][0, 2] - initial_object[2]))
            maximum_joint_change = max(maximum_joint_change, float((after["joint_position"] - initial_joint).abs().max()))
            if step % 40 == 0 or finished:
                print(json.dumps({"step": step, "phase": phase,
                    "ee": after["ee_pose"][0, :3].tolist(),
                    "fingers": after["finger_positions"][0].tolist(),
                    "max_object_lift_m": maximum_lift,
                    "drawer_change_m": maximum_joint_change}), flush=True)
            if finished:
                break
        if not finished:
            raise RuntimeError(f"failure attempt did not complete: {phase}")
        changes = np.flatnonzero(np.diff(grippers)) + 1
        assert len(changes) == 1 and grippers[0] == 1 and grippers[-1] == -1
        assert maximum_lift < 0.025, "object was lifted; not the requested grasp failure"
        assert maximum_joint_change < 0.020, "drawer moved despite intended handle miss"
        assert float(torch.linalg.vector_norm(after["object_pose"][0, :3] - initial_object)) < 0.035
        assert float(after["finger_positions"].max()) < 0.0015
        print(json.dumps({"task": args.task, "validated_failure": True,
            "failure_type": "handle_grasp_miss" if opening else "object_grasp_miss",
            "controlled_perturbation": "grasp 4 cm in front of handle" if opening else "grasp 8.5 cm beside successful grasp waypoint in base X",
            "single_grasp": True, "task_success": False,
            "steps": len(frames), "duration_s": len(frames) * dt,
            "initial_object_position": initial_object.tolist(),
            "initial_drawer_position": initial_joint.tolist(),
            "maximum_object_lift_m": maximum_lift,
            "maximum_drawer_change_m": maximum_joint_change}), flush=True)
        frames = np.stack(frames)
        height, width = frames.shape[1:3]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(1.0 / dt), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(args.output)], input=frames.tobytes(), check=True)
        print(f"Saved {args.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
