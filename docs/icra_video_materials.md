# ICRA video-material preparation

## Second palm closing contact and requested migration folder

`outputs/icra_materials_build/drawer_palm_alternate_v1` is published as
`drawer_close_palm_push_open_gripper_alternate_contact_success.mp4`
(304 frames, 15.2 s, native 720p at 20 fps). The palm contacts the opposite
upper panel edge (image left), rather than the first video's image-right edge.
Measured median drawer-local XYZ `[.144346,.172120,.064580]` differs from the
original `[-.152959,.172022,.065031]` by 0.297305 m. Full initial oracle states
are identical. Open aperture stays approximately 80 mm; every one of 85 closing
frames has palm contact, other monitored robot bodies record zero force against
the upper drawer. After contact-free withdrawal, opening is stable at 6.514 mm.
No fingertip compliance, auxiliary force, or state writes are used.

The publisher now supports a measured-reference contact comparison and an
additional video-only destination. Both this new video and the previous-camera
cup lateral-entry failure were copied to `/raid/home/than/drawer_closing_two_videos`.
Its earlier two videos are preserved; that folder contains only four MP4s.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/generate_drawer_closing_pair.py --method palm_push --output-dir outputs/icra_materials_build/drawer_palm_alternate_new --keep-gripper-open --panel-lateral-offset .14 --panel-height-offset -.015 --palm-offset -.005 --headless
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/package_drawer_palm_push.py --source outputs/icra_materials_build/drawer_palm_alternate_new --variant-label alternate_contact --reference-source outputs/icra_materials_build/drawer_palm_open_v4 --video-only-destination /raid/home/than/drawer_closing_two_videos
```

## Corrected human approach: lateral entry, retain the failure

Camera-only revision: `cup_human_side_entry_top_grasp_blocked_previous_camera.mp4`
replays the exact v4 recorded actions using the older camera eye
`[.95,.25,.32]`, target `[.52,-.15,.22]`. Source build is
`outputs/icra_materials_build/cup_human_side_entry_previous_camera_v1`.
All 174 oracle states and contact-force arrays match v4 exactly (maximum
absolute error zero), with the same frame-136 collision. No pose/state writes
or new planning were used. The source NPZ is copied unchanged; replay audit is
included in the published report. Earlier camera/video remain available.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/test_cup_human_motion_transfer.py --output-dir outputs/icra_materials_build/cup_side_camera_new --replay-source outputs/icra_materials_build/cup_human_side_entry_v4 --camera-eye .95 .25 .32 --camera-target .52 -.15 .22 --headless
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/package_cup_human_motion_transfer.py --source outputs/icra_materials_build/cup_side_camera_new --variant-label previous_camera
```

The user corrected the earlier front-diagonal interpretation: the human hand
enters from the SIDE of the drawer toward above the cup. The user explicitly
wants the failed motion retained, not repaired into a successful task.

Source `outputs/icra_materials_build/cup_human_side_entry_v4`, delivered as
`cup_human_side_entry_top_grasp_blocked.mp4` (174 frames, 8.7 s, native 720p,
20 fps, simulator background). The exact initial oracle still matches the
successful side-grasp reference. After staging at the near side, the planned
entry is along cabinet-local X only, at TCP z=0.10 m and constant local Y.
Tool Z remains downward; finger opening axis is cabinet-local Y, perpendicular
to the lateral entry. This is lateral approach with a top grasp, NOT side grasp.

Entry starts at frame 128, and link6 first collides at frame 136 (6.8 s).
Eight consecutive frames above 1 N trigger a hold at 143 (7.15 s); peak force
is 76.50 N. No grasp or subsequent transfer is executed. The cup settles by
1 mm, with no meaningful horizontal displacement. Actual median tilt during
collision is 0.506 degrees. Collision-free entry motion in cabinet axes is
[+9.37, -1.82, -4.47] mm; Y/Z errors reflect staging convergence. Contact then
deflects the arm, so direction validation uses pre-contact samples and exact
commanded waypoints instead of demanding a minimum travel through the obstacle.
The lateral entry is blocked almost immediately, approximately 1 cm in; the
robot does not reach above the cup. All these limits are recorded in metadata.

Explorations v1–v3 did not give the requested segment: far-side staging was
unreachable or collided before entry, and the near-side initial orientation
failed to reach staging. None were published. The prior front-diagonal video
is preserved but superseded for correspondence to this human demonstration.
No scene geometry, object placement, or physics parameter was changed.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/test_cup_human_motion_transfer.py --output-dir outputs/icra_materials_build/cup_human_side_entry_new --approach-mode side_lateral --side-clearance -.26 --side-entry-height .10 --camera-eye .2 .6 .45 --camera-target .5 -.15 .2 --headless
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/package_cup_human_motion_transfer.py --source outputs/icra_materials_build/cup_human_side_entry_new
```

## New uploaded human cup video: qualitative motion-transfer failure

Source human video: project-root `cup_cabinet_human_pause.mp4`, 203 frames,
30 fps, 854×480. Visually inspected approach, paused upper-wall/rim grasp,
withdrawal/lift, and placement. It is not a vertical descent through the drawer.

`scripts/test_cup_human_motion_transfer.py` manually maps that grasp strategy and
waypoint order into the existing below-drawer cup scene: front/side diagonal
approach with downward-pointing gripper, rim grasp, outward extraction, lift,
cabinet-top placement. These are authored scene-relative waypoints, not measured
3-D human poses. The robot stops before grasping, so later waypoints are never
executed. No obstacle, cabinet, cup, target, or physics parameter was changed to
induce failure. Only camera/lighting and collision instrumentation were added.

Source `outputs/icra_materials_build/cup_human_motion_transfer_v1` is published
as `cup_human_top_grasp_transfer_blocked.mp4` in the materials bundle: 229 frames,
11.45 s, 1280×720, 20 fps, simulator-only background. Full initial oracle equality
to successful side-grasp v11 is checked. `panda_link7` first contacts the upper
drawer at frame 191 (9.55 s), peak 12.786 N; 8 consecutive frames above 1 N
trigger a safety hold at frame 198, followed by 30 hold steps. No physical grasp
occurs, all gripper commands remain open, and the cup only settles by 1 mm.
The TCP remains 135.45 mm from the requested grasp waypoint. Measured median
tool-axis tilt during collision is 0.780° from downward vertical. Thus this is
an **approach failure of a manually retargeted human-inspired top grasp**, not
a post-grasp transfer collision, not an exact replay of human joint motion,
and not proof that all possible top grasps fail.

The publisher `scripts/package_cup_human_motion_transfer.py` validates initial
state, collision timing/forces, near-vertical tool direction, unexecuted grasp,
cup stability, and full video decode. It records the human source hash and
mapping, publishes initial/collision screenshots and an NPZ trajectory, and
preserves older materials. Reproduce with the usual Isaac library setup:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/test_cup_human_motion_transfer.py --output-dir outputs/icra_materials_build/cup_human_motion_transfer_new --headless
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/package_cup_human_motion_transfer.py --source outputs/icra_materials_build/cup_human_motion_transfer_new
```

## Open-gripper panel push away from the handle

New source: `outputs/icra_materials_build/drawer_panel_open_far_v4`.
Published as `drawer_close_panel_push_open_gripper_far_handle_success.mp4`
in `/raid/home/than/icra_video_materials` (336 frames, 16.8 s, 1280×720,
20 fps). The old panel-push and handle-grasp videos are preserved.

The complete initial oracle state equals `drawer_panel_v2` exactly. The gripper
is commanded open throughout (minimum measured aperture 79.987 mm). Contact
center is shifted -115 mm along the cabinet's local X, towards image right;
the two tips contact the panel above and below that center. Minimum distance
to the handle bar/mount surfaces is 72.94 / 69.45 mm. Both fingers exceed
0.1 N in all 83 active closing-contact frames (99.52% of their physics
substeps); no palm contact is recorded. Final opening after contact-free
withdrawal is 4.293 mm and remains below 1 cm.

This opt-in variant binds a compliant contact material to the two fingers,
stiffness 10000 N/m and damping 100 N·s/m, to tolerate small orientation
tracking errors. Shared robot assets and the handle-grasp method are unchanged.
The drawer is passive; there are no state overrides or auxiliary forces after
reset. Earlier rigid-tip trials v1–v3 closed but failed the bilateral-contact
quality check and were not published. The new metadata records the material
parameters, measured contact points, clearance, and validation results.

Reproduce using the Isaac environment/library setup below and a new output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/generate_drawer_closing_pair.py --method panel_push --output-dir outputs/icra_materials_build/drawer_panel_open_far_new --keep-gripper-open --panel-lateral-offset -.115 --panel-height-offset .004 --fingertip-stiffness 10000 --headless
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/package_open_finger_panel_push.py --source outputs/icra_materials_build/drawer_panel_open_far_new
```

## Earlier material preparation

### Palm-only drawer closing

Source `outputs/icra_materials_build/drawer_palm_open_v4`, published as
`drawer_close_palm_push_open_gripper_success.mp4` (246 frames, 12.3 s,
1280×720 at 20 fps) with a contact screenshot, trajectory, bindings, and report.
Exact initial oracle equality to `drawer_panel_v2` is verified. Fingers remain
open (minimum aperture 79.999 mm), tool points down, and the palm side pushes
the image-right upper edge of the panel, not its center. Median drawer-local
contact XYZ is `[-0.15296, 0.17202, 0.06503]` m. All 75 close-phase frames
record palm contact; fingers and all eight monitored arm links record zero
contact force with the upper drawer. Palm peak is 13.28 N. The passive drawer
remains at 6.93 mm opening after contact-free retreat and a 1 s hold. No
compliant material, attachments, joint-state overrides, or auxiliary forces
are used. Earlier trials v1–v3 were not published: v1 pushed the handle with
the palm, v2 also contacted with link7, and v3 contacted with a finger.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/generate_drawer_closing_pair.py --method palm_push --output-dir outputs/icra_materials_build/drawer_palm_open_new --keep-gripper-open --panel-lateral-offset -.16 --panel-height-offset -.015 --palm-offset -.005 --headless
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/package_drawer_palm_push.py --source outputs/icra_materials_build/drawer_palm_open_new
```

The new physical fruit generator is separate from the old kinematic
`generate_two_fruit_trajectories.py`; existing videos and scripts are preserved.

## Physical fruit pair

`scripts/generate_physical_fruit_demo.py` uses the original calibrated layout,
YCB banana convex-decomposition collision, an orange convex-hull collision and
the original plate mesh as a static triangle-mesh collider. Both fruits are
dynamic rigid bodies. The plate is kinematic and stationary. The controller
commands only the robot: no fruit attachments, object-pose writes or transport
forces are used after reset.

The two orders share a common initial scene and purple target plate. The
controller uses Cartesian waypoint interpolation and differential IK for
manipulation, and bounded joint-position PD commands for safe return to the
default joint configuration. Homing commands are recorded separately in
`joint_position_command`; the 7D Cartesian actions alone do not encode this
mode. This is not a global collision-aware motion planner. Completion checks include bilateral
finger contact at closure and after physical lift, gravity release, contact
with the target plate, low final fruit velocities and robot withdrawal. A
banana grasp near the curved solid arc needs a lower fingertip waypoint than
the old cabinet-top grasp, because the tabletop fruit is larger and lower.

Each output has a video, state/action/contact arrays and a verification report.
The scene's physically settling fruit poses may differ slightly from the
initial authored mesh poses; no subsequent pose correction is applied.

## Drawer strategy pair

`scripts/generate_drawer_closing_pair.py` starts the same upper drawer at 0.22 m
open with the robot at its standard home configuration. The methods are:

- `panel_push`: horizontally approach above the handle with closed fingers,
  then push the front panel inward.
- `handle_grasp`: insert open fingers around the handle, verify two-sided
  contact and a nonzero finger aperture, then close the drawer while holding
  the bar. Open the gripper and withdraw afterward.

Drawer stiffness is zero and no auxiliary drawer-retention force is applied.
The success check requires opening below 0.01 m after withdrawal. Packaging
additionally checks maintained finger aperture and bilateral contact during
the handle-closing segment. Merely contacting the handle while the fingers
are closed is not accepted as a handle grasp.

`runtime_bindings.json` is exported from the instantiated environment, including
the actual seeded cabinet pose, upper joint and handle frame. The accompanying
task contract is manually authored from the requested semantics. It is not
presented as an automatic human-video extraction result, and is not passed
off as output of the older bridge schema (which does not support CLOSE).

## Cup failure candidate

`scripts/render_cup_blocked_top_grasp.py` tests one vertical approach to the cup
under the open upper drawer. The recorded drawer/robot contact force, actual
TCP height and desired cup-grasp height document the obstruction. This is a
new controlled simulation, not an archived planner rejection and not a proof
that every possible top-grasp trajectory is infeasible. The view uses only
the simulated background. Existing successful top/side-grasp cup videos are
included for comparison.

## Export

Use the `lw_benchhub` environment and the project's usual cmeel library path.
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` prevents excessive
CPU thread contention when independent rendering processes run together.
Use new output directories; generators refuse to overwrite previous trials.

`scripts/package_icra_video_materials.py` validates both common initial states,
physical success reports and full video decoding before publishing new MP4s.
It copies rather than modifies the existing cup and microwave videos. The
destination contains videos at the root, with `images/`, `metadata/` and
`trajectories/` below. Unverified trials and the old kinematic fruit videos are
excluded.

## Verified delivery (2026-09-21)

`/raid/home/than/icra_video_materials/` contains nine 720p/20 fps videos, images,
JSON metadata and the new state/control/contact trajectories. The valid build
directories under `outputs/icra_materials_build/` are:

- `fruit_orange_verified`: 993 frames, 49.65 seconds.
- `fruit_banana_verified`: 1122 frames, 56.10 seconds.
- `drawer_panel_v2`: 362 frames, 18.10 seconds; final opening 6.02 mm.
- `drawer_handle_v2`: 359 frames, 17.95 seconds; final opening 4.17 mm.
- `cup_blocked_v1`: 225 frames, 11.25 seconds; peak drawer contact 22.18 N.

Both fruit runs were independently replayed using their saved Cartesian actions
and joint-position targets, with zero recorded-state error over the whole
episode. Final plate support and default robot joints were checked again in
the replay. Both fruit orders maintained bilateral finger contact for every
recorded transport frame, and the final fruit centers are about 11 cm apart.

The source `fruit_*_v5` simulations completed but their original JSON export
failed on a NumPy boolean. The boolean serialization was fixed, and verification
reports were regenerated by the independent physics replay, not by relabeling
the failed exporter. Earlier development trials remain outside the delivery.

## Additional candidate: successful top grasp, colliding transfer

Source: `outputs/icra_materials_build/cup_direct_transfer_v2`.
Generator: `scripts/test_cup_top_grasp_direct_placement.py`.
Publisher: `scripts/package_cup_direct_transfer_candidate.py`.

The complete 47-dimensional initial state is exactly equal to the original
successful `cup_on_cabinet_to_upper_drawer_top_grasp_v8` episode. No cabinet,
cup, drawer opening, or target location was changed. The successful expert's
high transfer waypoint above the drawer is omitted after a verified top grasp;
the candidate instead follows a direct descending Cartesian transfer.

The cup is lifted 7.55 cm and remains in physical bilateral finger contact.
At frame 254, the cup contacts the cabinet top near its front edge. After five
consecutive collision frames (>0.5 N), the controller deliberately stops and
holds for 20 frames. Peak cup-cabinet force is 21.17 N. No cup drop occurs; the
final target error is 18.36 cm. The 278-frame / 13.9-second clip and two screenshots
are published under `cup_top_grasp_direct_transfer_collision_stop*`.

This is an unsafe transfer-path candidate with an explicit safety stop, **not**
evidence that the grasp is incompatible with every placement motion, and not an
exhaustive failed planning search. The previously published high-clearance path
succeeds with a top grasp. Use this distinction in paper captions.

Run with the `lw_benchhub` Python, the same `LD_LIBRARY_PATH` as the other Isaac
scripts, and a new output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/test_cup_top_grasp_direct_placement.py --output-dir outputs/icra_materials_build/cup_direct_transfer_new --headless
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/package_cup_direct_transfer_candidate.py --source outputs/icra_materials_build/cup_direct_transfer_v2
```

The newer microwave video with the gripper always open is documented separately
in `docs/microwave_closing.md`. Both additions preserve prior media and the user's
real-robot video.

## Corrected task: below-drawer cup, grasp succeeds before upward collision

The user clarified that the requested counterexample belongs to **the cup below
the open upper drawer moving to the cabinet top**, not the cup-on-cabinet task.
The earlier 13.9-second direct-transfer clip is a different task and must not
be cited for the clarified request.

New source: `outputs/icra_materials_build/below_top_lift_tilt45_v4`.
Delivery: `cup_below_drawer_oblique_top_grasp_then_lift_collision.mp4` (398
frames, 19.9 s, 1280×720, 20 fps, pure simulated background). The scene's full
initial oracle state is exactly identical to the existing successful side-grasp
episode `below_upper_drawer_to_cabinet_top_side_grasp_v11`.

The robot approaches below the extended drawer from the front, with a tool
orientation 45 degrees from vertical. It grasps physically, then attempts a
direct vertical lift toward the cabinet-top transfer height. Publication checks:

- No earlier robot/cup contact with the upper drawer.
- Persistent bilateral cup-finger forces after grasp (minimum 31.03 / 30.38 N).
- A conservative geometric cup-bottom clearance over 5 mm **before** the first
  drawer contact, accounting for cup tilt; at frame 367 it is 13.16 mm.
- At frame 368, wrist link `panda_link6` begins contacting the upper drawer;
  peak force 5.38 N. After 10 consecutive contact frames, an explicit safety
  hold lasts 20 frames. No drop occurs, and the cabinet-top goal is not reached.
- The measured tool angle is 44.93 degrees, so this is labeled **oblique top
  grasp**, never strictly vertical top grasp. It demonstrates one upward path
  collision, not universal infeasibility of alternative transfer paths.

The source's earlier root-height verification frame 360 is weaker than a true
bottom-clearance check for a tilted cup. The publisher independently uses actual
cup orientation and a conservative tapered-cup bound, selects frame 367 for the
grasp screenshot, and verifies ordering against the first collision at 368.

Exploratory strict vertical (`below_top_lift_v1`) and 30-degree
(`below_top_lift_tilt30_v1`) candidates were blocked before grasping. The higher
45-degree variant v2 lifted the cup but had earlier wrist contact, and was
excluded. Other lower-height variants timed out before verified grasp. None of
these exploratory clips were published as successful-grasp/failed-transfer cases.

Reproduce with a new output directory, the same Isaac environment and library
path setup used above:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/test_below_drawer_top_grasp_lift.py --output-dir outputs/icra_materials_build/below_top_lift_new --tilt-deg -45 --pregrasp-height .044 --grasp-height .040 --camera-eye .95 .25 .26 --camera-target .52 -.14 .14 --headless --max-steps 600
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python scripts/package_below_drawer_top_lift_candidate.py --source outputs/icra_materials_build/below_top_lift_tilt45_v4
```
