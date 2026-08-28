"""Render a calibrated policy-frame preview for one remaining task."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    required=True,
    choices=("drawer_opening", "cabinet_opening", "object_to_drawer", "nut_pouring"),
)
parser.add_argument("--save-frame", required=True)
parser.add_argument("--warmup-steps", type=int, default=60)
parser.add_argument(
    "--fully-open",
    action="store_true",
    help="Preview the active drawer or door at its full useful joint limit.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    import numpy as np
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, SimulationContext
    from PIL import Image

    from track_b_envs.envs.common.compositing import RealBackgroundCompositor
    from track_b_envs.envs.common.task_scene import build_initial_state_scene_cfg

    sim = SimulationContext(SimulationCfg(dt=0.01, device=args.device))
    scene = InteractiveScene(build_initial_state_scene_cfg(args.task, include_camera=True))
    sim.reset()
    scene.reset()
    if args.fully_open and "cabinet" in scene.articulations:
        cabinet = scene["cabinet"]
        if args.task in {"drawer_opening", "object_to_drawer"}:
            joint_ids, _ = cabinet.find_joints("UpperDrawerJoint")
            open_position = 0.22
        else:
            joint_ids, _ = cabinet.find_joints("DoorHingeJoint")
            open_position = 1.745329252
        joint_position = cabinet.data.default_joint_pos.clone()
        joint_velocity = cabinet.data.default_joint_vel.clone()
        joint_position[:, joint_ids] = open_position
        joint_velocity[:, joint_ids] = 0.0
        cabinet.write_joint_state_to_sim(joint_position, joint_velocity)
        cabinet.set_joint_position_target(joint_position)
    for _ in range(args.warmup_steps):
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())

    camera = scene["policy_camera"]
    composite = RealBackgroundCompositor(scene.device).compose(
        camera.data.output["rgb"],
        camera.data.output["semantic_segmentation"],
        camera.data.intrinsic_matrices,
    )
    policy_rgb = RealBackgroundCompositor.policy_view(composite)[0].permute(1, 2, 0)
    image = (policy_rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(args.save_frame)
    print(f"task: {args.task}", flush=True)
    print(f"saved: {args.save_frame}", flush=True)
    if "cabinet" in scene.articulations:
        cabinet = scene["cabinet"]
        handle = scene["cabinet_frame"].data.target_pos_w[0, 0].detach().cpu().tolist()
        root = cabinet.data.root_pos_w[0].detach().cpu().tolist()
        joint_positions = {
            name: float(cabinet.data.joint_pos[0, index].detach().cpu())
            for index, name in enumerate(cabinet.joint_names)
        }
        print(f"cabinet root: {root}", flush=True)
        print(f"handle: {handle}", flush=True)
        print(f"joint positions: {joint_positions}", flush=True)
    for name, asset in scene.rigid_objects.items():
        print(f"{name}: {asset.data.root_pos_w[0].detach().cpu().tolist()}", flush=True)

    del scene
    sim.clear_all_callbacks()
    sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
