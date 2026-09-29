"""Render a calibrated policy-frame preview for one remaining task."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    required=True,
    choices=(
        "drawer_opening",
        "cabinet_opening",
        "object_to_drawer",
        "nut_pouring",
        "cup_to_drawer",
        "cup_to_cabinet_top",
    ),
)
parser.add_argument("--save-frame", required=True)
parser.add_argument("--warmup-steps", type=int, default=60)
parser.add_argument(
    "--full-resolution",
    action="store_true",
    help="Save the calibrated 1280x720 composite instead of the 224x224 policy crop.",
)
parser.add_argument(
    "--sim-only",
    action="store_true",
    help="Save the raw simulator RGB frame without compositing the real background.",
)
parser.add_argument(
    "--fully-open",
    action="store_true",
    help="Preview the active drawer or door at its full useful joint limit.",
)
parser.add_argument(
    "--drawer-joint",
    choices=("upper", "lower"),
    default="upper",
    help="Drawer to open when --fully-open is used for a drawer task.",
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
    scene_cfg = build_initial_state_scene_cfg(args.task, include_camera=True)
    if hasattr(scene_cfg, "cabinet"):
        print(
            f"configured cabinet joints: {scene_cfg.cabinet.init_state.joint_pos}",
            flush=True,
        )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    # InteractiveScene.reset() alone does not guarantee that an unconstrained
    # preview keeps the canonical Franka posture during the render warm-up.
    # Match the trajectory generators by explicitly writing and commanding the
    # configured default joint state.
    robot = scene["robot"]
    home_joint_position = robot.data.default_joint_pos.clone()
    home_joint_velocity = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(home_joint_position, home_joint_velocity)
    robot.set_joint_position_target(home_joint_position)
    if "cabinet" in scene.articulations:
        cabinet = scene["cabinet"]
        cabinet_joint_position = cabinet.data.default_joint_pos.clone()
        cabinet_joint_velocity = cabinet.data.default_joint_vel.clone()
        cabinet.write_joint_state_to_sim(
            cabinet_joint_position, cabinet_joint_velocity
        )
        cabinet.set_joint_position_target(cabinet_joint_position)
    if args.fully_open and "cabinet" in scene.articulations:
        cabinet = scene["cabinet"]
        if args.task in {
            "drawer_opening",
            "object_to_drawer",
            "cup_to_drawer",
            "cup_to_cabinet_top",
        }:
            drawer_joint_name = (
                "UpperDrawerJoint"
                if args.drawer_joint == "upper"
                else "LowerDrawerJoint"
            )
            joint_ids, _ = cabinet.find_joints(drawer_joint_name)
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
        robot.set_joint_position_target(home_joint_position)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())

    camera = scene["policy_camera"]
    compositor = RealBackgroundCompositor(scene.device)
    if args.sim_only:
        rendered_image = compositor.warp_rgb(
            camera.data.output["rgb"], camera.data.intrinsic_matrices
        )
    else:
        rendered_image = compositor.compose(
            camera.data.output["rgb"],
            camera.data.output["semantic_segmentation"],
            camera.data.intrinsic_matrices,
        )
    if args.full_resolution:
        image = rendered_image[0].cpu().numpy()
    else:
        policy_rgb = RealBackgroundCompositor.policy_view(rendered_image)[0].permute(1, 2, 0)
        image = (
            policy_rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0
        ).round().astype(np.uint8)
    Image.fromarray(image).save(args.save_frame)
    print(f"task: {args.task}", flush=True)
    print(f"saved: {args.save_frame}", flush=True)
    print(
        "robot joints: "
        f"{robot.data.joint_pos[0].detach().cpu().tolist()}",
        flush=True,
    )
    print(
        "robot default max error rad: "
        f"{(robot.data.joint_pos - home_joint_position).abs().max().item():.9f}",
        flush=True,
    )
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
        if "cup_target_frame" in scene.sensors:
            cup_target = (
                scene["cup_target_frame"].data.target_pos_w[0, 0]
                .detach()
                .cpu()
                .tolist()
            )
            print(f"cup target: {cup_target}", flush=True)
    for name, asset in scene.rigid_objects.items():
        print(f"{name}: {asset.data.root_pos_w[0].detach().cpu().tolist()}", flush=True)

    del scene
    sim.clear_all_callbacks()
    sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
