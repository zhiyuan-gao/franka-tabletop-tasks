"""Collect successful Track B demonstrations through the policy action interface."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    choices=(
        "push_object_to_region",
        "push_cube_to_region",
        "drawer_opening",
        "cabinet_opening",
        "object_to_drawer",
        "nut_pouring",
    ),
    default="push_object_to_region",
)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-episodes", type=int, default=1)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--max-attempts", type=int, default=10)
# The staged physical NutPouring expert needs about 1,300 control steps at the
# current conservative waypoint speeds.  Keep the default large enough that a
# valid episode is not rejected solely by the collector horizon.
parser.add_argument("--max-steps", type=int, default=1600)
parser.add_argument("--warmup-steps", type=int, default=30)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    help="Print closed-loop expert diagnostics every N steps; use 0 to disable.",
)
parser.add_argument(
    "--save-debug-frames",
    action="store_true",
    help="Save policy RGB frames at each diagnostic interval.",
)
parser.add_argument("--seed", type=int, default=20260827)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _validate_args() -> None:
    for name in ("num_episodes", "num_envs", "max_attempts", "max_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_attempts < args.num_episodes:
        raise ValueError("--max-attempts cannot be smaller than --num-episodes")


def main() -> dict:
    import json

    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from track_b_envs import TASK_IDS
    from track_b_envs.collection import EpisodeBuffer, SuccessfulDatasetWriter
    from track_b_envs.collection.experts import (
        CabinetOpeningExpert,
        DrawerOpeningExpert,
        ObjectToDrawerExpert,
        NutPouringExpert,
        PushObjectToRegionExpert,
    )
    from track_b_envs.collection.experts.common import (
        object_to_drawer_task_state,
        nut_pouring_task_state,
        opening_task_state,
        task_space_state,
    )
    from track_b_envs.envs.articulated_opening import (
        CabinetOpeningEnvCfg,
        DrawerOpeningEnvCfg,
    )
    from track_b_envs.envs.push_object_to_region import (
        PushCubeToRegionEnvCfg,
        PushObjectToRegionEnvCfg,
    )
    from track_b_envs.envs.object_to_drawer import ObjectToDrawerEnvCfg
    from track_b_envs.envs.nut_pouring import NutPouringEnvCfg
    from track_b_envs.envs.common.timing import SETTLE_DURATION_S, SETTLE_STEPS

    _validate_args()
    if args.task in {"push_object_to_region", "push_cube_to_region"}:
        cfg = (
            PushCubeToRegionEnvCfg()
            if args.task == "push_cube_to_region"
            else PushObjectToRegionEnvCfg()
        )
        expert_cls = PushObjectToRegionExpert
        state_fn = task_space_state
        success_description = "full oriented object footprint inside target"
        expert_description = "closed-loop approach-behind planar push state machine"
        task_metadata = {
            "object_asset": (
                "procedural red cuboid (0.070 x 0.070 x 0.050 m)"
                if args.task == "push_cube_to_region"
                else str(cfg.scene.object.spawn.usd_path)
            ),
            "object_reset_x_m": [0.430, 0.490],
            "object_reset_y_m": [-0.335, -0.275],
            "target": "18 x 12 cm rectangle on calibrated A4 sheet",
            "position_adjustment": None,
        }
    elif args.task == "drawer_opening":
        cfg = DrawerOpeningEnvCfg()
        expert_cls = DrawerOpeningExpert
        state_fn = lambda env: opening_task_state(env, "UpperDrawerJoint")
        success_description = "upper drawer displacement >= 0.16 m"
        expert_description = "closed-loop horizontal handle grasp and linear pull"
        task_metadata = {
            "cabinet_root_xy_m": [0.72, -0.18],
            "cabinet_yaw_deg": 35.0,
            "active_joint": "UpperDrawerJoint",
            "success_threshold_m": 0.16,
            "reset_randomization": {"x_m": [-0.02, 0.02], "y_m": [-0.02, 0.02], "yaw_deg": [-5.0, 5.0]},
            "position_adjustment": None,
            "physics_contract": (
                "PhysX joint motion plus verified bilateral handle grasp; "
                "no articulation-state overwrite"
            ),
        }
    elif args.task == "cabinet_opening":
        cfg = CabinetOpeningEnvCfg()
        expert_cls = CabinetOpeningExpert
        state_fn = lambda env: opening_task_state(env, "DoorHingeJoint")
        success_description = "door hinge angle >= 1.20 rad"
        expert_description = "closed-loop horizontal handle grasp and hinge-following pull"
        task_metadata = {
            "cabinet_root_xy_m": [0.75, -0.18],
            "cabinet_yaw_deg": 35.0,
            "active_joint": "DoorHingeJoint",
            "success_threshold_rad": 1.20,
            "reset_randomization": {"x_m": [-0.02, 0.02], "y_m": [-0.02, 0.02], "yaw_deg": [-5.0, 5.0]},
            "position_adjustment": None,
            "physics_contract": (
                "PhysX joint motion plus verified bilateral handle grasp; "
                "no articulation-state overwrite"
            ),
        }
    elif args.task == "object_to_drawer":
        cfg = ObjectToDrawerEnvCfg()
        expert_cls = ObjectToDrawerExpert
        state_fn = object_to_drawer_task_state
        success_description = (
            "drawer open, released banana inside moving drawer volume, "
            f"and physically settled for {SETTLE_DURATION_S:.2f} s"
        )
        expert_description = (
            "closed-loop handle pull, physical banana grasp, transport, "
            "gravity release, and settle verification"
        )
        task_metadata = {
            "cabinet_root_xy_m": [0.72, -0.18],
            "cabinet_yaw_deg": 35.0,
            "object_asset": str(cfg.scene.object.spawn.usd_path),
            "object_mass_kg": 0.066,
            "active_joint": "UpperDrawerJoint",
            "drawer_open_threshold_m": 0.145,
            "settle_duration_s": SETTLE_DURATION_S,
            "settle_steps": SETTLE_STEPS,
            "reset_randomization": {
                "cabinet_x_m": [-0.02, 0.02],
                "cabinet_y_m": [-0.02, 0.02],
                "cabinet_yaw_deg": [-5.0, 5.0],
                "banana_local_x_m": [-0.02, 0.02],
                "banana_local_y_m": [-0.02, 0.02],
                "banana_local_yaw_deg": [-10.0, 10.0],
            },
            "drawer_retention": (
                "zero assist while pulling; explicit physical PD hold force "
                "after handle release; no state overwrite"
            ),
            "handle_grasp_contract": (
                "drawer pull accepted only after centered bilateral handle "
                "grasp with nonzero finger aperture"
            ),
            "position_adjustment": None,
        }
    else:
        cfg = NutPouringEnvCfg()
        expert_cls = NutPouringExpert
        state_fn = nut_pouring_task_state
        success_description = (
            "nut physically inside the bowl at low velocity for "
            f"{SETTLE_DURATION_S:.2f} s"
        )
        expert_description = (
            "closed-loop lower-body cup grasp, lift, transfer, staged tilt, "
            "gravity pour, and bowl settle verification"
        )
        task_metadata = {
            "cup_asset": str(cfg.scene.object.spawn.usd_path),
            "nut_asset": str(cfg.scene.factory_nut.spawn.usd_path),
            "bowl_asset": str(cfg.scene.sorting_bowl.spawn.usd_path),
            "cup_nominal_position_m": list(cfg.scene.object.init_state.pos),
            "bowl_nominal_position_m": list(cfg.scene.sorting_bowl.init_state.pos),
            "settle_duration_s": SETTLE_DURATION_S,
            "settle_steps": SETTLE_STEPS,
            "reset_randomization": {
                "cup_x_m": [-0.015, 0.015],
                "cup_y_m": [-0.015, 0.015],
                "cup_yaw_deg": [-8.0, 8.0],
                "bowl_x_m": [-0.015, 0.015],
                "bowl_y_m": [-0.015, 0.015],
                "cup_bowl_distance_m": [0.13, 0.21],
            },
            "position_adjustment": (
                "cup moved from [0.38, -0.28, 0.001] m to "
                "[0.43, -0.18, 0.001] m to shorten the home-to-grasp sweep"
            ),
            "physics_contract": (
                "no root-pose writes after reset; cup grasped by contacts and nut "
                "transferred by gravity/collision"
            ),
        }
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    control_dt_s = float(cfg.sim.dt * cfg.decimation)
    cfg.episode_length_s = max(cfg.episode_length_s, (args.max_steps + 1) * control_dt_s)
    print(
        json.dumps(
            {
                "stage": "creating_environment",
                "task": args.task,
                "num_envs": args.num_envs,
            }
        ),
        flush=True,
    )
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        if env.action_space.shape != (args.num_envs, 7):
            raise RuntimeError(f"unexpected action space: {env.action_space}")
        if args.warmup_steps > 0:
            env.reset(seed=args.seed)
            zero_action = torch.zeros(env.action_space.shape, device=env.device)
            for _ in range(args.warmup_steps):
                env.step(zero_action)

        writer = SuccessfulDatasetWriter(
            args.output_dir,
            task_name=args.task,
            task_id=TASK_IDS[args.task],
            requested_episodes=args.num_episodes,
            control_dt_s=control_dt_s,
            dataset_metadata={
                "robot": "Franka Panda",
                "expert": expert_description,
                "environment": task_metadata,
                "seed": args.seed,
            },
        )
        expert = expert_cls(env)
        max_batches = math.ceil(args.max_attempts / args.num_envs)
        stop = False
        for batch_index in range(max_batches):
            if writer.successful_episodes >= args.num_episodes or stop:
                break
            batch_seed = args.seed + batch_index
            observations, _ = env.reset(seed=batch_seed)
            expert.reset(env)
            buffers = [EpisodeBuffer() for _ in range(args.num_envs)]
            active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
            initial = state_fn(env)
            if args.task in {"push_object_to_region", "push_cube_to_region"}:
                initial_states = [
                    {
                        "object_pose_base_wxyz": initial["object_pose"][env_id]
                        .detach().cpu().tolist(),
                        "target_pose_base_wxyz": initial["target_pose"][env_id]
                        .detach().cpu().tolist(),
                    }
                    for env_id in range(args.num_envs)
                ]
            elif args.task in {"drawer_opening", "cabinet_opening"}:
                initial_states = [
                    {
                        "handle_pose_base_wxyz": initial["handle_pose"][env_id]
                        .detach().cpu().tolist(),
                        "joint_position": float(
                            initial["joint_position"][env_id].detach().cpu()
                        ),
                    }
                    for env_id in range(args.num_envs)
                ]
            elif args.task == "object_to_drawer":
                initial_states = [
                    {
                        "object_pose_base_wxyz": initial["object_pose"][env_id]
                        .detach()
                        .cpu()
                        .tolist(),
                        "drawer_target_pose_base_wxyz": initial["target_pose"][env_id]
                        .detach()
                        .cpu()
                        .tolist(),
                        "handle_pose_base_wxyz": initial["handle_pose"][env_id]
                        .detach()
                        .cpu()
                        .tolist(),
                        "drawer_joint_position": float(
                            initial["joint_position"][env_id].detach().cpu()
                        ),
                    }
                    for env_id in range(args.num_envs)
                ]
            else:
                initial_states = [
                    {
                        "cup_pose_base_wxyz": initial["cup_pose"][env_id]
                        .detach().cpu().tolist(),
                        "nut_pose_base_wxyz": initial["nut_pose"][env_id]
                        .detach().cpu().tolist(),
                        "bowl_pose_base_wxyz": initial["bowl_pose"][env_id]
                        .detach().cpu().tolist(),
                    }
                    for env_id in range(args.num_envs)
                ]
            minimum_pre_step_distance = torch.full(
                (args.num_envs,), float("inf"), device=env.device
            )

            for step in range(args.max_steps):
                state_before = state_fn(env)
                if args.task in {"push_object_to_region", "push_cube_to_region"}:
                    distance_before = torch.linalg.vector_norm(
                        state_before["object_pose"][:, :2]
                        - state_before["target_pose"][:, :2], dim=-1
                    )
                    diagnostic_state = {
                        "object_position_m": state_before["object_pose"][:, :3],
                        "target_position_m": state_before["target_pose"][:, :3],
                        "center_distance_m": distance_before,
                    }
                elif args.task in {"drawer_opening", "cabinet_opening"}:
                    distance_before = torch.linalg.vector_norm(
                        state_before["ee_pose"][:, :3]
                        - state_before["handle_pose"][:, :3], dim=-1
                    )
                    diagnostic_state = {
                        "handle_position_m": state_before["handle_pose"][:, :3],
                        "joint_position": state_before["joint_position"],
                        "ee_handle_distance_m": distance_before,
                    }
                elif args.task == "object_to_drawer":
                    distance_before = torch.linalg.vector_norm(
                        state_before["object_pose"][:, :3]
                        - state_before["target_pose"][:, :3],
                        dim=-1,
                    )
                    diagnostic_state = {
                        "object_position_m": state_before["object_pose"][:, :3],
                        "drawer_target_position_m": state_before["target_pose"][:, :3],
                        "object_target_distance_m": distance_before,
                        "drawer_joint_position": state_before["joint_position"],
                        "finger_positions": state_before["finger_positions"],
                    }
                else:
                    distance_before = torch.linalg.vector_norm(
                        state_before["nut_pose"][:, :3]
                        - state_before["bowl_pose"][:, :3],
                        dim=-1,
                    )
                    diagnostic_state = {
                        "cup_position_m": state_before["cup_pose"][:, :3],
                        "nut_position_m": state_before["nut_pose"][:, :3],
                        "bowl_position_m": state_before["bowl_pose"][:, :3],
                        "nut_bowl_distance_m": distance_before,
                        "finger_positions": state_before["finger_positions"],
                    }
                minimum_pre_step_distance = torch.minimum(
                    minimum_pre_step_distance, distance_before
                )
                actions = expert.compute(env)
                if args.log_interval > 0 and step % args.log_interval == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "expert_diagnostic",
                                "batch": batch_index,
                                "step": step,
                                "expert_state": expert.state.detach().cpu().tolist(),
                                "ee_position_m": state_before["ee_pose"][:, :3]
                                .detach()
                                .cpu()
                                .tolist(),
                                **{
                                    name: value.detach().cpu().tolist()
                                    for name, value in diagnostic_state.items()
                                },
                            }
                        ),
                        flush=True,
                    )
                    if args.save_debug_frames:
                        from PIL import Image

                        from track_b_envs.collection.recorder import (
                            policy_rgb_to_uint8_hwc,
                        )

                        debug_dir = writer.output_dir / "debug_frames"
                        debug_dir.mkdir(exist_ok=True)
                        for env_id in active.nonzero(as_tuple=False).squeeze(-1).tolist():
                            frame = policy_rgb_to_uint8_hwc(
                                observations["policy"]["rgb"][env_id]
                            )
                            Image.fromarray(frame).save(
                                debug_dir
                                / f"batch_{batch_index:03d}_env_{env_id:02d}_step_{step:04d}.png"
                            )
                next_observations, rewards, terminated, truncated, _ = env.step(actions)
                success = env.termination_manager.get_term("success").clone()

                for env_id in active.nonzero(as_tuple=False).squeeze(-1).tolist():
                    buffers[env_id].append(
                        policy_rgb=observations["policy"]["rgb"][env_id],
                        policy_proprio=observations["policy"]["proprio"][env_id],
                        oracle_state=observations["critic"][env_id],
                        action=actions[env_id],
                        reward=float(rewards[env_id].detach().cpu()),
                        terminated=bool(terminated[env_id]),
                        truncated=bool(truncated[env_id]),
                        success=bool(success[env_id]),
                        timestamp_s=(step + 1) * control_dt_s,
                    )

                ended = active & (terminated | truncated)
                for env_id in ended.nonzero(as_tuple=False).squeeze(-1).tolist():
                    handle_grasp_required = args.task in {
                        "drawer_opening",
                        "cabinet_opening",
                        "object_to_drawer",
                    }
                    handle_grasp_verified = (
                        bool(expert.grasp_verified[env_id].detach().cpu())
                        if handle_grasp_required
                        else True
                    )
                    succeeded = bool(success[env_id]) and handle_grasp_verified
                    active_terms = {
                        name: bool(value[0])
                        for name, value in env.termination_manager.get_active_iterable_terms(
                            env_id
                        )
                    }
                    if succeeded:
                        reason = "success"
                    elif bool(success[env_id]) and not handle_grasp_verified:
                        reason = "unverified_handle_grasp"
                    else:
                        reason = next(
                            (name for name, value in active_terms.items() if value),
                            "terminated",
                        )
                    writer.record_attempt(
                        success=succeeded,
                        seed=batch_seed,
                        source_env_id=env_id,
                        steps=len(buffers[env_id]),
                        reason=reason,
                        metrics={
                            "minimum_pre_step_task_distance_m": float(
                                minimum_pre_step_distance[env_id].detach().cpu()
                            ),
                            "termination_terms": active_terms,
                            "handle_grasp_verified": handle_grasp_verified,
                        },
                    )
                    if succeeded and writer.successful_episodes < args.num_episodes:
                        episode_path = writer.save_success(
                            buffers[env_id],
                            seed=batch_seed,
                            source_env_id=env_id,
                            episode_metadata={
                                "batch_index": batch_index,
                                "initial_state": initial_states[env_id],
                                "success_term": success_description,
                            },
                        )
                        print(
                            json.dumps(
                                {
                                    "stage": "episode_saved",
                                    "file": str(episode_path),
                                    "successful_total": writer.successful_episodes,
                                    "attempts": writer.manifest["attempts"],
                                }
                            ),
                            flush=True,
                        )
                    active[env_id] = False
                    if int(writer.manifest["attempts"]) >= args.max_attempts:
                        stop = True
                observations = next_observations
                if not torch.any(active) or stop:
                    break

            for env_id in active.nonzero(as_tuple=False).squeeze(-1).tolist():
                writer.record_attempt(
                    success=False,
                    seed=batch_seed,
                    source_env_id=env_id,
                    steps=len(buffers[env_id]),
                    reason="collector_max_steps",
                    metrics={
                        "minimum_pre_step_task_distance_m": float(
                            minimum_pre_step_distance[env_id].detach().cpu()
                        ),
                        "final_expert_state": int(expert.state[env_id].detach().cpu()),
                        "final_ee_position_m": state_before["ee_pose"][env_id, :3]
                        .detach()
                        .cpu()
                        .tolist(),
                        "final_task_state": {
                            name: value[env_id].detach().cpu().tolist()
                            for name, value in state_before.items()
                            if name != "ee_pose"
                        },
                    },
                )
                if int(writer.manifest["attempts"]) >= args.max_attempts:
                    stop = True

            print(
                json.dumps(
                    {
                        "stage": "batch_complete",
                        "batch": batch_index,
                        "successful_total": writer.successful_episodes,
                        "attempts": writer.manifest["attempts"],
                    }
                ),
                flush=True,
            )

        manifest = writer.finalize()
        print(json.dumps({"stage": "complete", **manifest}), flush=True)
        if manifest["status"] != "complete":
            raise RuntimeError(
                f"collected {writer.successful_episodes}/{args.num_episodes} successes "
                f"in {writer.manifest['attempts']} attempts"
            )
        return manifest
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
