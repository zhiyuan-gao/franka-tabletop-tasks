#!/usr/bin/env python3
import argparse
import base64
import json
import urllib.request
from pathlib import Path

from isaaclab.app import AppLauncher

TASKS = (
    "push_object_to_region",
    "push_cube_to_region",
    "drawer_opening",
    "cabinet_opening",
    "object_to_drawer",
    "nut_pouring",
)

INSTRUCTIONS = {
    "push_object_to_region": "push the object to the target region",
    "push_cube_to_region": "push the cube to the target region",
    "drawer_opening": "open the drawer",
    "cabinet_opening": "open the cabinet",
    "object_to_drawer": "put the object in the drawer",
    "nut_pouring": "pour the nuts",
}

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=TASKS, required=True)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--server", default="http://127.0.0.1:8123")
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def query_openvla(rgb):
    arr = (
        rgb[0]
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    payload = json.dumps({
        "image": base64.b64encode(arr.tobytes()).decode(),
        "shape": list(arr.shape),
        "instruction": INSTRUCTIONS[args.task],
    }).encode()

    req = urllib.request.Request(
        args.server,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())["action"]


def main():
    import gymnasium as gym
    import numpy as np
    import torch

    from task_pipeline_v2 import policy_observation, environment_success, encode_video
    from track_b_envs.rl import (
        configure_normalized_action_space,
        load_registered_env_cfg,
    )

    device = torch.device(args.device)

    task_id, cfg = load_registered_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        seed=args.seed,
        use_fabric=not getattr(args, "disable_fabric", False),
    )

    env = gym.make(task_id, cfg=cfg)
    configure_normalized_action_space(env)

    obs, _ = env.reset(seed=args.seed)

    results = []
    ep_return = 0.0
    ep_length = 0
    frames = []
    video_dir = Path("outputs/openvla_4000_videos") / args.task
    video_dir.mkdir(parents=True, exist_ok=True)

    try:
        while len(results) < args.episodes:
            rgb, _ = policy_observation(obs, device)

            frames.append(
                rgb[0]
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            action = query_openvla(rgb)
            action = torch.tensor(
                action,
                dtype=torch.float32,
                device=device,
            ).view(1, 7)

            obs, reward, terminated, truncated, _ = env.step(action)

            success = environment_success(env)
            ep_return += float(reward[0])
            ep_length += 1

            done = bool((terminated | truncated)[0])

            if done:
                row = {
                    "episode": len(results),
                    "success": bool(success[0]),
                    "return": ep_return,
                    "length": ep_length,
                }
                outcome = "success" if row["success"] else "failure"
                video_path = video_dir / f"{args.task}_episode_{len(results):03d}_{outcome}.mp4"
                encode_video(frames, video_path, 20.0)
                row["video"] = str(video_path.resolve())

                results.append(row)
                frames = []

                print(
                    f"{args.task}: episode {len(results)}/{args.episodes} "
                    f"success={row['success']}",
                    flush=True,
                )

                ep_return = 0.0
                ep_length = 0

    finally:
        env.close()

    successes = sum(x["success"] for x in results)

    summary = {
        "task": args.task,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "instruction": INSTRUCTIONS[args.task],
        "results": results,
    }

    output = args.output or Path(
        f"outputs/openvla_4000_{args.task}_eval.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
