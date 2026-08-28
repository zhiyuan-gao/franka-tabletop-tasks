# Successful demonstration collection

The collection interface executes automatic experts through the same normalized
7D action consumed by the policy. It never teleports the robot or manipulated
object during an episode. Failed attempts are logged but only successful
episodes are committed.

Each compressed episode contains synchronized arrays:

| Key | Description |
|---|---|
| `observation.rgb` | uint8 composite policy RGB, `[T, H, W, 3]` |
| `observation.proprio` | deployable Franka state, `[T, D]` |
| `observation.oracle` | simulator-only task state, `[T, D]` |
| `action` | normalized IK-relative action, `[T, 7]` |
| `reward` | environment reward |
| `terminated`, `truncated`, `success` | episode flags |
| `timestamp_s` | control timestamp |

The dataset `manifest.json` records every attempt, reset seed, initial object
and target poses, success metric, expert type, and any intentional scene-position
adjustment.

The final action component is strictly binary: `+1` for fully open and `-1`
for fully closed. DrawerOpening, CabinetOpening, and ObjectToDrawer keep the
original Cartesian waypoint trajectories while remaining fully open through
the approach and issuing one close command at the final grasp waypoint.

Current executable collectors use one command and differ only by `--task`:

```bash
cd track_b_six_task_project_20260828
python scripts/collect_successful_demos.py \
  --task push_object_to_region \
  --output-dir outputs/demos/push_smoke \
  --num-episodes 1 \
  --max-attempts 10 \
  --headless
```

Supported values are:

- `push_object_to_region`: closed-loop planar pushing;
- `push_cube_to_region`: the same closed-loop pushing contract with a
  `7 x 7 x 5 cm` red cube;
- `drawer_opening`: verified bilateral handle grasp followed by a linear pull;
- `cabinet_opening`: verified bilateral handle grasp followed by a hinge-following pull;
- `object_to_drawer`: open the drawer, physically grasp and transport the YCB
  banana, release it under gravity, and verify that it settles inside.
- `nut_pouring`: physically grasp the hollow blue cup, lift and move it over
  the receiving bowl, tilt it through two controlled stages, and verify that
  the freely falling nut settles inside the bowl.

For example, replace the task and output directory with:

```bash
--task drawer_opening --output-dir outputs/demos/drawer_run
--task push_cube_to_region --output-dir outputs/demos/push_cube_run
--task cabinet_opening --output-dir outputs/demos/cabinet_run
--task object_to_drawer --output-dir outputs/demos/object_to_drawer_run
--task nut_pouring --output-dir outputs/demos/nut_pouring_run
```

The drawer tasks use the cabinet pose `(0.72, -0.18)` m. The door cabinet is
3 cm farther along X at `(0.75, -0.18)` m so the gripper can preserve a
bilateral handle grasp throughout the hinge sweep. Both use yaw `35 deg`, with
reset offsets of +/-2 cm in XY and +/-5 degrees in yaw. The
collector only accepts an opening if the handle is centered inside the finger
pads, both finger joints retain a nonzero aperture, and that grasp remains
valid while pulling. The drawer moving mass is 1.7 kg with nominal rail
friction/damping `(10.0, 8.0, 15.0)`; the 1.2 kg door uses
`(1.8, 1.0, 1.0)`. These are empirical PhysX parameters selected from
controlled effort sweeps, not measured hardware parameters.

ObjectToDrawer applies the same cabinet randomization, plus an independent
cabinet-local banana offset of +/-2 cm in XY and +/-10 degrees in yaw. The
expert resolves its grasp frame from the sampled banana pose and freezes that
frame at descent so contact-induced object motion cannot destabilize the
wrist. It uses no drawer assist while the robot is pulling.
After handle release, an explicit PD force holds the measured release position,
modeling rail static friction without overwriting joint state. The expert first
clears the handle and retreats vertically before moving over the cabinet, so
the robot does not push the open drawer closed. A successful
episode additionally requires the released banana to remain inside the moving
drawer volume at low velocity for 0.30 s (6 steps at 20 Hz). The requested
randomized contract was physically revalidated with the binary gripper at 20 Hz
with 10/10 successful trials at seed `20260827`, completing in 26.00--27.25 s.
For the banana grasp, the arm holds the unchanged final waypoint fully open for
0.35 s, issues one close command, and waits for physical contact to settle
before lifting. No intermediate aperture command is generated.

The NutPouring expert performs all manipulation through the same normalized 7D
policy action. Root poses are written only by the reset event; during each
episode the cup is held by gripper contact and the nut moves only through
gravity and collision. Success requires the nut root to remain within a
conservative bowl interior and below the rim at low linear/angular velocity for
0.30 s (6 steps at 20 Hz). Its first-stage reset randomizes cup and bowl XY
by +/-1.5 cm, cup yaw by +/-8 degrees, and rejection-samples a cup-to-bowl
center distance of 13--21 cm. The bowl-relative success volume accepts the
half-scale nut both flat and edge-resting, while remaining inside the physical
inner bowl. The 20 Hz randomized validation passed 10/10 trials at seed
`20260827` in 25.80--27.25 s. The collector's default horizon is 1,600 steps
(80 s), leaving margin for recovery.

## 20 Hz pre-collection validation

The six primary experts were run through the full camera-enabled environment with 10
parallel randomized trials. RGB was observed but not written during this
validation.

| Task | Result | Successful duration range |
|---|---:|---:|
| PushObjectToRegion | 10/10 | 7.30--40.80 s |
| PushCubeToRegion | 10/10 | 4.20--4.55 s |
| DrawerOpening | 10/10 | 10.55--10.60 s |
| CabinetOpening | 10/10 | 14.45--14.55 s |
| ObjectToDrawer | 10/10 | 26.00--27.25 s |
| NutPouring | 10/10 | 25.80--27.25 s |

To collect 50 successful episodes per task, run one command per task with a
separate output directory:

```bash
python scripts/collect_successful_demos.py \
  --task <task_name> \
  --output-dir outputs/demos/20hz_50/<task_name> \
  --num-episodes 50 \
  --max-attempts 250 \
  --max-steps 1600 \
  --headless
```

## Stored-action replay validation

After collection, `scripts/replay_recorded_actions.py` can reconstruct the
original vectorized reset stream and replay only the saved normalized 7D
actions through PhysX. The full 2026-08-28 audit, per-task results, and the
important distinction between collection-time success and open-loop action
repeatability are documented in
[`recorded_action_replay_validation.md`](recorded_action_replay_validation.md).

## Released data layout

Demonstrations are published separately through Hugging Face rather than
committed to the GitHub code repository. After downloading, use this layout:

```text
datasets/
├── primary_50/                         # six tasks, 50 successes each
└── additional_50_seed20260927/         # +50 ObjectToDrawer and NutPouring
```

The original collection directories are not required. Episode paths inside
each manifest are relative to that manifest's dataset directory. The local
`datasets/` directory is covered by `.gitignore`.
