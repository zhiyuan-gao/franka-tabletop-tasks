# Calibrated drawer subtasks

These six task IDs reuse the current `object_to_drawer` cabinet, YCB banana,
Franka default joint configuration, camera and reset randomization. The nominal
cabinet position is `(0.60386895, -0.52161895, 0)` m, yaw `8.74888170` degrees.

| Task | Upper drawer | Lower drawer | Initial target drawer | Success |
|---|---|---|---|---|
| Open drawer | `open_upper_drawer` | `open_lower_drawer` | Closed | Opening >= 0.21 m |
| Put the object in the drawer | `put_object_in_upper_drawer` | `put_object_in_lower_drawer` | 0.22 m | Banana released and settled inside the open drawer |
| Put the object in the drawer and close drawer | `put_object_in_upper_drawer_and_close` | `put_object_in_lower_drawer_and_close` | 0.22 m | Banana released and settled inside, drawer opening <= 0.025 m |

The other drawer starts closed. The banana starts on the cabinet top; it is not
already grasped. Every episode starts at the standard Franka joint angles.
Success must persist for 0.30 seconds. Closed-loop collection additionally
requires a verified bilateral handle grasp for opening. Closing uses a
closed-finger front-panel push and verifies inward drawer motion near the tool.
The lower-drawer push point is 5.5 cm below the handle, with 13 cm front
clearance before descent, to avoid the closed upper drawer. The upper drawer
uses a push point 1 cm above the handle and 6.5 cm front clearance.

Opening tasks require 0.21 m of the cabinet's 0.22 m maximum travel. This
near-full-opening criterion does not change the placement-only requirement
(drawer opening >= 0.16 m) or the placement-and-close criterion. Older opening
demos collected with the 0.16 m criterion are not relabeled as 0.21 m successes.
The opening expert brakes its Cartesian target once the drawer reaches 0.212 m
and holds the gripper pose through the success check, avoiding continued pulling
against the mechanical stop and an unnecessary handle-grasp retry.

The experts use the same velocity-limited Cartesian waypoint interpolation and
differential IK as the existing tasks, with physical contact, gravity and
actuators executing the manipulation. This is not a global collision-aware path
planner. The banana grasp orientation is frozen before descent and preserved
during transport and retreat, as in `ObjectToDrawerExpert`.
For both drawers, the placement waypoint is 3.5 cm forward of the interior
reference frame, to clear the closed upper drawer overhang while staying inside
the drawer's original success volume.

During placement, the expert applies a drawer-retention effort to keep the
pre-opened drawer still. This effort is disabled while the gripper pushes it
closed. Like the existing `object_to_drawer` collection, that auxiliary effort
is not included in the 7-D robot action: these are successful closed-loop
demonstrations, not a guarantee of action-only open-loop replay.

`scripts/package_drawer_subtask_demos.py` derives each placement-only demo from
the exact prefix of its paired successful placement-and-close episode. It
checks the same physical placement predicate for six consecutive post-action
states and adjusts only task-dependent rewards and terminal flags. Robot,
object and drawer trajectories are unchanged. Prefix extraction was checked
against an independently collected placement episode: actions, proprioception,
oracle state and terminal flags matched exactly; reward error was below 2e-7.

Collect a demo (use any task name from the table):

```bash
python scripts/collect_successful_demos.py \
  --task put_object_in_lower_drawer_and_close \
  --output-dir outputs/my_lower_drawer_close_demo \
  --num-episodes 1 --max-attempts 5 --headless
```

The output directory must not already exist. Each successful episode contains
20 Hz RGB, robot proprioception, privileged state, normalized 7D actions and
success flags. `manifest.json` records the exact reset seed and success result.

Export its recorded RGB to MP4:

```bash
python scripts/export_episode_mp4.py \
  outputs/my_lower_drawer_close_demo/episodes/episode_000000.npz \
  outputs/my_lower_drawer_close_demo/demo.mp4
```
