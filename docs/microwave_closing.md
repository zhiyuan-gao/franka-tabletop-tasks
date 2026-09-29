# Microwave-closing demonstration pair

The two demonstrations use the same calibrated microwave layout as
`scripts/render_close_microwave_initial_state.py`: root `(0.62, -0.45, 0)` m,
35-degree root yaw, 86-degree initially open door, and the standard Franka home
joint configuration. They differ in the vertical contact region on the outside
of the door, near its free edge / handle.

- `upper`: commanded tool height 0.155 m above the table.
- `lower`: commanded tool height 0.110 m above the table, leaving clearance
  between the sideways gripper body and the table.

Actual surface contact heights differ slightly from tool heights because the
gripper has finite geometry. Each episode's `report.json` records the measured
median contact height; `episode.npz` retains contact forces and door-local contact
positions at the five physics substeps per video frame.

## Physics and control

`assets/task_assets/microwave/BlackMicrowave.usda` preserves the preview's box
geometry and matte materials, adding a fixed body, colliders and a passive door
hinge. Hinge position stiffness is zero; damping and friction are passive.
The robot approaches above the door, lowers outside it, pushes along the hinge
arc using velocity-limited Cartesian waypoints and differential IK, and then
withdraws. This is a contact-feedback controller, not a global motion planner.
The script never overwrites the door or robot state after the initial reset.

The completion check requires a door angle within 2 degrees of closed and speed
below 0.1 rad/s for the last 0.5 seconds after withdrawal, plus negligible final
robot-door contact force. Publication also requires at least five recorded
physical contact samples and a median contact height within 3.5 cm of the
commanded region. Initial opening and default robot joints are checked too.

## Reproduction

Use the `lw_benchhub` environment, with Isaac Lab / this project installed. Each
output directory must be new. Launch from the repository root:

```bash
export LD_LIBRARY_PATH=/raid/home/than/miniforge3/envs/lw_benchhub/lib/python3.11/site-packages/cmeel.prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/generate_microwave_closing_demo.py --contact upper --output-dir outputs/microwave_closing_demos/upper_new --headless
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/generate_microwave_closing_demo.py --contact lower --output-dir outputs/microwave_closing_demos/lower_new --headless
```

The native camera output is 1280 × 720, 20 fps, with the existing calibrated real
background. Outputs include `demo.mp4`, `report.json`, `episode.npz`, and periodic
inspection images. Always inspect the report and validation messages: Isaac Sim
shutdown can suppress an exception's nonzero process exit status.

## Delivered pair

The verified source episodes are `outputs/microwave_closing_demos/upper_v5`
(20.8 s) and `outputs/microwave_closing_demos/lower_v3` (20.0 s). Measured median
contact heights are 0.14639 m and 0.10137 m, respectively: 4.50 cm apart. Both
start from identical recorded robot/door states and end with the door closed
after the robot withdraws.

`scripts/package_microwave_closing_demos.py` checks recorded states, contact
locations, final settling, initial-state equality and full video decoding before
copying the videos. Its validation report is
`outputs/microwave_closing_demos/published_pair.json`. Only these two MP4s are
added to `/raid/home/than/paper_sim_video720p/`:

- `microwave_close_upper_contact_success.mp4`
- `microwave_close_lower_contact_success.mp4`

## Palm-side push matching the real-robot example (2026-09-21)

Reference: `/raid/home/than/icra_video_materials/05_microwave_close_ep022.mp4`.
New verified source: `outputs/microwave_closing_demos/palm_v6`.
This is a newly controlled physical simulation of the contact strategy, not a
frame-exact retargeting of the real robot's joints.

The gripper points downward. Its palm (`panda_hand`) side pushes the door near
the free edge, at a measured median contact height of 0.17385 m. TCP height is
0.09 m; the TCP is not the contacting point. The layout, initial 86-degree door
opening, and standard robot initial joint configuration are unchanged.

The original preview-derived appliance USD did not author an angular DriveAPI,
so an actuator damping setting alone did not create the intended hinge brake.
For the new palm mode only, the spawn wrapper authors that schema before physics
initialization, with zero stiffness, zero target velocity, and damping 0.10.
This is a passive dissipative brake, not an active closing motor. The shared USD
and older finger demonstrations remain unchanged. Exploratory palm_v2–v5 clips
mostly coasted after initial palm impulses and are not the published demo.

Validation of the published 500-frame, 25-second, native 1280×720 / 20 fps clip:

- 99.565% of pushing control frames contain a palm contact above 0.05 N.
- Measured palm contacts span door angles −85.74 to −1.03 degrees.
- All recorded robot-door contact impulse is from `panda_hand`; neither fingers
  nor any other monitored arm link report a door contact.
- After withdrawal, the door is closed and settled, with zero robot-door force.
- The full MP4 decodes successfully. A contact-confirmed frame at 16.45 seconds
  is also exported; the source NPZ retains all substep contacts.

Reproduce with a new output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/generate_microwave_closing_demo.py --contact upper --effector palm --palm-offset .05 --touch-speed .01 --hinge-damping .10 --output-dir outputs/microwave_closing_demos/palm_new --headless --max-steps 800
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/package_microwave_palm_demo.py --source outputs/microwave_closing_demos/palm_v6
```

Use the same `LD_LIBRARY_PATH` setup described above. Published files are under
`/raid/home/than/icra_video_materials`: `microwave_close_palm_push_success.mp4`,
`images/microwave_close_palm_push_success.png`, and corresponding metadata and
trajectory files. The real reference video and all older files are preserved.
The v6 source report's legacy `planned_contact_door_xyz` field describes the
commanded TCP, not the palm surface; the current generator names this field
`planned_tcp_door_xyz` explicitly.

### Open-gripper revision

The requested open-gripper variant is `outputs/microwave_closing_demos/palm_open_v1`.
Add `--keep-gripper-open` to the palm command above. Every recorded gripper action
is +1, and the minimum measured finger-joint aperture is 0.0799991 m. It preserves
the 500-frame duration, 99.565% pushing-frame palm contact coverage, zero finger
contact, default initial joints, and stable closure after withdrawal.

The packager automatically selects the distinct filename
`microwave_close_palm_push_open_gripper_success.mp4` for this variant, preserving
the older closed-gripper video. It also verifies all gripper commands and actual
apertures from the saved trajectory before publishing.

### Second open-gripper palm contact region

New source: `outputs/microwave_closing_demos/palm_open_panel_v1`.
Delivery: `microwave_close_palm_push_open_gripper_door_panel_success.mp4`, 414
frames / 20.7 seconds, native 1280×720 at 20 fps. The actual contact is on the
**upper door rail near the hinge**, not on the handle. The desired TCP location
alone is not used to establish this distinction: the recorded physical palm
contacts are compared in the rotating door coordinate system.

| Property | Original open-gripper palm push | Second palm push |
| --- | --- | --- |
| Median contact XYZ in door frame (m) | −0.23670, 0.04628, 0.17385 | −0.05582, 0.01895, 0.20873 |
| Region | Handle near free edge | Upper door rail near hinge |
| Push frames with palm contact >0.05 N | 99.565% | 100% |
| Finger contact | None recorded | None recorded |

The median contact separation is 0.18624 m. Initial oracle states are exactly
equal; the same passive damping, open fingers, layout, and default robot joints
are preserved. The new episode has 100% of measured robot-door impulse on
`panda_hand`, zero on all other monitored robot bodies, and zero robot-door force
after withdrawal. The door remains closed and settled.

Reproduction (same Python / library-path setup as above, new output directory):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/generate_microwave_closing_demo.py --contact upper --effector palm --along .10 --contact-height .105 --palm-offset .020 --touch-speed .01 --keep-gripper-open --output-dir outputs/microwave_closing_demos/palm_open_panel_new --headless --max-steps 800
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/package_microwave_palm_demo.py --source outputs/microwave_closing_demos/palm_open_panel_v1 --variant-label door_panel --reference-source outputs/microwave_closing_demos/palm_open_v1
```

The publisher checks actual 3D contact separation (>6 cm), matching initial
states and gripper configuration, contact body identity, closure and settling,
and full video decoding before copying. Existing media remain untouched. A
contact-confirmed frame near a −45-degree door angle is also exported.
