"""Render the close-microwave demonstration's initial tabletop layout.

This is deliberately a visual-only scene.  The microwave is assembled from
static primitives and its door is fixed in the open pose seen at the start of
``close_microwave_episode_000015.mp4``.  It reuses the project's calibrated
Franka, camera, table, lighting, semantic mask, and real-background compositor.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--save-frame",
    type=Path,
    default=Path("outputs/close_microwave_initial_state.png"),
    help="Path for the full-resolution calibrated composite.",
)
parser.add_argument("--warmup-steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)


# Visual measurements from the demonstration.  The root placement follows the
# same Franka-base coordinate convention used by the existing task scenes.
MICROWAVE_ROOT_POSITION_M = (0.62, -0.45, 0.0)
MICROWAVE_ROOT_YAW_DEG = 35.0
MICROWAVE_SIZE_M = (0.38, 0.285, 0.220)
MICROWAVE_DOOR_OPEN_DEG = 86.0
MICROWAVE_KEY_LIGHT_INTENSITY = 750.0
MICROWAVE_AMBIENT_LIGHT_INTENSITY = 250.0


def _yaw_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad))


def _world_pose(
    local_position: tuple[float, float, float], local_yaw_rad: float = 0.0
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Transform a microwave-local pose into the Franka base frame."""

    root_yaw = math.radians(MICROWAVE_ROOT_YAW_DEG)
    cosine = math.cos(root_yaw)
    sine = math.sin(root_yaw)
    local_x, local_y, local_z = local_position
    root_x, root_y, root_z = MICROWAVE_ROOT_POSITION_M
    position = (
        root_x + cosine * local_x - sine * local_y,
        root_y + sine * local_x + cosine * local_y,
        root_z + local_z,
    )
    return position, _yaw_quaternion(root_yaw + local_yaw_rad)


def _add_box(
    scene,
    name: str,
    *,
    size: tuple[float, float, float],
    local_position: tuple[float, float, float],
    color: tuple[float, float, float],
    local_yaw_rad: float = 0.0,
    roughness: float = 0.92,
) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from track_b_envs.envs.common.constants import FOREGROUND_SEMANTIC_TAG

    position, rotation = _world_pose(local_position, local_yaw_rad)
    setattr(
        scene,
        name,
        AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                semantic_tags=[FOREGROUND_SEMANTIC_TAG],
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    roughness=roughness,
                    metallic=0.0,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=rotation),
        ),
    )


def _add_microwave(scene) -> None:
    """Add a black countertop microwave with its door held open."""

    width, depth, height = MICROWAVE_SIZE_M
    shell = (0.025, 0.025, 0.027)
    black = (0.006, 0.007, 0.007)
    edge_black = (0.002, 0.002, 0.002)
    inner_black = (0.004, 0.004, 0.004)
    glass = (0.025, 0.033, 0.032)
    display = (0.10, 0.16, 0.145)
    button = (0.26, 0.28, 0.27)

    # Hollow body: top/bottom/side/rear panels preserve the visible dark cavity.
    _add_box(
        scene,
        "microwave_top",
        size=(width, depth, shell[2]),
        local_position=(0.0, 0.0, height - shell[2] / 2.0),
        color=(0.001, 0.001, 0.001),
        roughness=1.0,
    )
    _add_box(
        scene,
        "microwave_bottom",
        size=(width, depth, shell[2]),
        local_position=(0.0, 0.0, shell[2] / 2.0),
        color=edge_black,
    )
    _add_box(
        scene,
        "microwave_far_side",
        size=(shell[0], depth, height - 2.0 * shell[2]),
        local_position=(width / 2.0 - shell[0] / 2.0, 0.0, height / 2.0),
        color=black,
    )
    _add_box(
        scene,
        "microwave_side_label",
        size=(0.004, 0.135, 0.070),
        local_position=(width / 2.0 + 0.0025, -0.025, height / 2.0),
        color=(0.72, 0.72, 0.68),
        roughness=0.90,
    )
    _add_box(
        scene,
        "microwave_rear",
        size=(width - 2.0 * shell[0], shell[1], height - 2.0 * shell[2]),
        local_position=(0.0, -depth / 2.0 + shell[1] / 2.0, height / 2.0),
        color=inner_black,
    )

    # The right-front column and a few raised elements suggest the control pad
    # visible in the source video without depending on an external CAD asset.
    control_width = 0.085
    control_x = -width / 2.0 + control_width / 2.0
    front_y = depth / 2.0
    _add_box(
        scene,
        "microwave_control_body",
        size=(control_width, depth - 2.0 * shell[1], height - 2.0 * shell[2]),
        local_position=(control_x, 0.0, height / 2.0),
        color=black,
        roughness=0.96,
    )
    _add_box(
        scene,
        "microwave_control_column",
        size=(control_width, 0.030, height - 0.030),
        local_position=(control_x, front_y - 0.009, height / 2.0),
        color=(0.035, 0.040, 0.039),
        roughness=0.90,
    )
    _add_box(
        scene,
        "microwave_display",
        size=(0.056, 0.006, 0.036),
        local_position=(control_x, front_y + 0.009, height - 0.052),
        color=display,
        roughness=0.82,
    )
    for index, z_position in enumerate((0.155, 0.120, 0.085, 0.050)):
        _add_box(
            scene,
            f"microwave_button_{index}",
            size=(0.048, 0.006, 0.018),
            local_position=(control_x, front_y + 0.010, z_position),
            color=button,
            roughness=0.90,
        )

    # Bezel surrounding the cavity while leaving the front visibly open.
    cavity_left = -width / 2.0 + control_width
    cavity_right = width / 2.0 - shell[0]
    cavity_width = cavity_right - cavity_left
    cavity_center_x = (cavity_left + cavity_right) / 2.0
    _add_box(
        scene,
        "microwave_front_lower_bezel",
        size=(cavity_width, 0.022, 0.025),
        local_position=(cavity_center_x, front_y - 0.002, 0.031),
        color=edge_black,
    )
    _add_box(
        scene,
        "microwave_front_upper_bezel",
        size=(cavity_width, 0.022, 0.025),
        local_position=(cavity_center_x, front_y - 0.002, height - 0.031),
        color=edge_black,
    )
    _add_box(
        scene,
        "microwave_cavity_floor",
        size=(cavity_width - 0.020, depth - 0.055, 0.012),
        local_position=(cavity_center_x, -0.004, 0.041),
        color=(0.075, 0.078, 0.070),
        roughness=0.95,
    )

    # Door hinged at the cavity's left edge.  Each part shares the same door
    # yaw and is positioned from the hinge, so changing the angle remains easy.
    door_yaw = math.radians(180.0 - MICROWAVE_DOOR_OPEN_DEG)
    door_width = cavity_width + 0.012
    door_height = height - 0.022
    hinge_x = cavity_right + 0.005
    hinge_y = front_y + 0.016
    along_x = math.cos(door_yaw)
    along_y = math.sin(door_yaw)
    normal_x = -along_y
    normal_y = along_x

    def door_point(along: float, outward: float, z_position: float):
        return (
            hinge_x + along_x * along + normal_x * outward,
            hinge_y + along_y * along + normal_y * outward,
            z_position,
        )

    rail = 0.033
    door_thickness = 0.028
    door_center = door_point(door_width / 2.0, 0.0, door_height / 2.0 + 0.011)
    _add_box(
        scene,
        "microwave_door_glass",
        size=(door_width - 2.0 * rail, door_thickness, door_height - 2.0 * rail),
        local_position=door_center,
        local_yaw_rad=door_yaw,
        color=glass,
        roughness=0.90,
    )
    for name, along, z_position, size in (
        ("microwave_door_hinge_rail", rail / 2.0, door_height / 2.0 + 0.011, (rail, door_thickness + 0.006, door_height)),
        ("microwave_door_outer_rail", door_width - rail / 2.0, door_height / 2.0 + 0.011, (rail, door_thickness + 0.006, door_height)),
        ("microwave_door_lower_rail", door_width / 2.0, rail / 2.0 + 0.011, (door_width - 2.0 * rail, door_thickness + 0.006, rail)),
        ("microwave_door_upper_rail", door_width / 2.0, door_height - rail / 2.0 + 0.011, (door_width - 2.0 * rail, door_thickness + 0.006, rail)),
    ):
        _add_box(
            scene,
            name,
            size=size,
            local_position=door_point(along, 0.0, z_position),
            local_yaw_rad=door_yaw,
            color=edge_black,
            roughness=0.95,
        )
    _add_box(
        scene,
        "microwave_door_handle",
        size=(0.024, 0.032, door_height - 0.065),
        local_position=door_point(door_width - 0.045, -0.028, door_height / 2.0 + 0.011),
        local_yaw_rad=door_yaw,
        color=(0.025, 0.027, 0.027),
        roughness=0.95,
    )


def _build_scene_cfg():
    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import (
        FrankaCubeLiftEnvCfg,
    )

    from track_b_envs.envs.common.scene import apply_calibrated_scene

    scene = FrankaCubeLiftEnvCfg().scene
    scene.num_envs = 1
    scene.env_spacing = 2.5
    scene.object = None
    apply_calibrated_scene(scene, include_camera=True)
    # The shared scene is deliberately bright for varied task assets.  This
    # black appliance needs softer, diffuse illumination to avoid white
    # specular patches on its top and open door.
    scene.real_background_key_light.spawn.intensity = MICROWAVE_KEY_LIGHT_INTENSITY
    scene.light.spawn.intensity = MICROWAVE_AMBIENT_LIGHT_INTENSITY
    _add_microwave(scene)
    return scene


def main() -> None:
    import numpy as np
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, SimulationContext
    from PIL import Image

    from track_b_envs.envs.common.compositing import RealBackgroundCompositor

    sim = SimulationContext(SimulationCfg(dt=0.01, device=args.device))
    scene = InteractiveScene(_build_scene_cfg())
    sim.reset()
    scene.reset()

    # ``InteractiveScene.reset()`` clears controller buffers but does not
    # guarantee that a standalone preview's initial PD target equals the
    # configured joint state.  Pin both state and target to the same calibrated
    # home pose before the render warm-up, just as an environment reset does.
    robot = scene["robot"]
    home_joint_position = robot.data.default_joint_pos.clone()
    home_joint_velocity = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(home_joint_position, home_joint_velocity)
    robot.set_joint_position_target(home_joint_position)
    for _ in range(args.warmup_steps):
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim.get_physics_dt())

    camera = scene["policy_camera"]
    composite = RealBackgroundCompositor(scene.device).compose(
        camera.data.output["rgb"],
        camera.data.output["semantic_segmentation"],
        camera.data.intrinsic_matrices,
    )[0]
    image = composite.cpu().numpy().astype(np.uint8)
    args.save_frame.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.save_frame)
    print(f"saved: {args.save_frame.resolve()}", flush=True)
    print(f"microwave root (m): {MICROWAVE_ROOT_POSITION_M}", flush=True)
    print(f"microwave yaw (deg): {MICROWAVE_ROOT_YAW_DEG}", flush=True)
    print(f"door open (deg): {MICROWAVE_DOOR_OPEN_DEG}", flush=True)
    print(f"robot joints (rad): {robot.data.joint_pos[0].cpu().tolist()}", flush=True)

    del scene
    sim.clear_all_callbacks()
    sim.clear_instance()


if __name__ == "__main__":
    args = parser.parse_args()
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        main()
    finally:
        simulation_app.close()
