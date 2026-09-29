"""Render a single verified physical drawer success at native calibrated 720p.

Runs the same closed-loop expert as demonstration collection, including its
drawer-retention effort. No robot, object or drawer states are overwritten.
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
parser.add_argument("--max-steps", type=int, default=800)
parser.add_argument("--reference-episode", type=Path,
    help="Require exact robot/object trajectory agreement over the shared prefix.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def main():
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils import parse_env_cfg

    from track_b_envs.task_ids import TASK_IDS
    from track_b_envs.collection.experts.drawer_subtasks import DrawerSubtaskExpert
    from track_b_envs.collection.experts.common import drawer_subtask_state
    from track_b_envs.envs.common.constants import FRANKA_HOME_JOINTS_RAD
    from track_b_envs.envs.common.compositing import RealBackgroundCompositor

    reference = None
    if args.reference_episode is not None:
        with np.load(args.reference_episode, allow_pickle=False) as data:
            reference = data["observation.oracle"].copy()
    cfg = parse_env_cfg(TASK_IDS[args.task], device=args.device, num_envs=1)
    cfg.seed = args.seed
    dt = float(cfg.sim.dt * cfg.decimation)
    cfg.episode_length_s = max(cfg.episode_length_s, (args.max_steps + 2) * dt)
    env = ManagerBasedRLEnv(cfg=cfg)
    process = None
    try:
        env.reset(seed=args.seed)
        for _ in range(10):
            env.step(torch.zeros(env.action_space.shape, device=env.device))
        observations, _ = env.reset(seed=args.seed)
        expert = DrawerSubtaskExpert(env)
        expert.reset(env)
        opening = cfg.task_mode == "open"
        assert torch.allclose(env.scene["robot"].data.joint_pos[0, :7],
            torch.tensor(FRANKA_HOME_JOINTS_RAD, device=env.device), atol=1e-5, rtol=0)
        camera = env.scene["policy_camera"]
        compositor = RealBackgroundCompositor(env.device)
        env.sim.render()
        camera.update(0.0, force_recompute=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", "1280x720",
            "-framerate", str(1 / dt), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "slow", "-crf", "16",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        grippers, openings, states = [], [], []
        max_reference_error = 0.0
        success = False
        for step in range(args.max_steps):
            state = drawer_subtask_state(env, cfg.active_joint_name)
            if reference is not None and step < len(reference):
                error = float(np.max(np.abs(observations["critic"][0].cpu().numpy() - reference[step])))
                max_reference_error = max(max_reference_error, error)
                if error > 1e-4:
                    raise RuntimeError(f"Source trajectory diverged at frame {step}: {error}")
            action = expert.compute(env)
            states.append(int(expert.state[0]))
            grippers.append(float(action[0, -1]))
            openings.append(float(state["joint_position"][0]))
            frame = compositor.compose(camera.data.output["rgb"],
                camera.data.output["semantic_segmentation"], camera.data.intrinsic_matrices)[0].cpu().numpy()
            process.stdin.write(frame.tobytes())
            observations, _, terminated, truncated, _ = env.step(action)
            if step % 80 == 0:
                print(json.dumps(dict(step=step, expert_state=states[-1], drawer_position_m=openings[-1])), flush=True)
            if bool(terminated[0] | truncated[0]):
                success = bool(env.termination_manager.get_term("success")[0])
                break
        if not success:
            raise RuntimeError("Task did not finish successfully")
        retries = int(expert._expert.retry_count[0])
        print(json.dumps(dict(stage="terminal_check", success=success,
            frames=len(grippers), retries=retries, last_openings_m=openings[-8:],
            handle_grasp_verified=bool(expert.grasp_verified[0]))), flush=True)
        if retries:
            raise RuntimeError(f"Refusing an episode with {retries} retries")
        closes = np.flatnonzero((np.array(grippers[:-1]) > 0) & (np.array(grippers[1:]) < 0)) + 1
        if opening:
            assert len(closes) == 1 and bool(expert.grasp_verified[0])
            assert min(openings[-5:]) >= .21, "missing near-full-open hold"
        else:
            grasp_state = expert._expert.GRASP_OBJECT
            assert sum(s == grasp_state and (i == 0 or states[i-1] != grasp_state)
                for i, s in enumerate(states)) == 1, "object grasp was retried"
            if cfg.task_mode == "put_and_close":
                assert bool(expert.closing_contact_verified[0])
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace")
        return_code = process.wait()
        process = None
        if return_code:
            raise RuntimeError(stderr)
        print(json.dumps(dict(task=args.task, validated_success=True,
            resolution=[1280,720], fps=1/dt, frames=len(grippers), duration_s=len(grippers)*dt,
            retries=retries, gripper_close_frames=closes.tolist(),
            last_pre_action_drawer_position_m=openings[-1], maximum_drawer_position_m=max(openings),
            opening_success_threshold_m=.21 if opening else None,
            max_reference_oracle_error=max_reference_error if reference is not None else None,
            output=str(args.output)), indent=2), flush=True)
    finally:
        if process is not None:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
            process.wait()
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
