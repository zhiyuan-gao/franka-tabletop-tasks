# Recorded-action PhysX replay validation

Validation date: 2026-08-28

This audit replays the stored normalized 7D actions through PhysX. It does not
run the collection expert and does not overwrite robot, object, or articulation
state during an episode. Stored RGB is decompressed and statically checked;
live cameras are disabled during the physics pass because rendering does not
affect the task dynamics.

The original four-environment reset width, per-batch seed, and source
environment index are preserved. This is necessary to reproduce the reset
random-number streams.

## Results

| Task | Static data | Initial task state | PhysX action replay |
|---|---:|---:|---:|
| PushObjectToRegion | 50/50 | 50/50 | 14/50 (28%) |
| PushCubeToRegion | 50/50 | 50/50 | 50/50 (100%) |
| DrawerOpening | 50/50 | 50/50 | 50/50 (100%) |
| CabinetOpening | 50/50 | 50/50 | 50/50 (100%) |
| ObjectToDrawer | 50/50 | 50/50 | 34/50 (68%) |
| NutPouring | 50/50 | 50/50 | 25/50 (50%) |
| **Total** | **300/300** | **300/300** | **223/300 (74.3%)** |

The static check verifies synchronized array lengths, 20 Hz timestamps, RGB
shape and dtype, finite normalized actions, binary gripper commands, and the
recorded terminal-success flag. Initial-state validation compares the task
poses and articulation position reconstructed from the recorded seed with the
values stored in the manifest.

## Interpretation

All stored episodes were successful physical executions when collected: the
collector only commits an episode after the live PhysX success predicate is
true. The replay percentage answers a stricter, different question: can the
recorded relative-action sequence reproduce success open loop, without the
expert's per-step state feedback?

DrawerOpening and CabinetOpening are fully repeatable under this test. The
pushing policy is closed-loop during collection and updates its contact path
from the measured box pose; after a small contact divergence, stored actions no
longer correct the box. Thirty-six PushObjectToRegion replays therefore reached
the recorded action horizon without satisfying the full-footprint target test.
In contrast, the low-profile red cube has substantially more target margin and
its 50 newly collected episodes passed 50/50 strict action replays.

ObjectToDrawer has two sources of replay sensitivity. It contains a long chain
of handle, banana, and placement contacts, and the collection expert also
applies a drawer-retention effort outside the normalized 7D robot action after
releasing the handle. That retention command is not currently part of the
episode. The 16 failed strict replays ended with drawer positions between
0.1438 and 0.1486 m, close to the 0.145 m requirement, but did not maintain all
release, containment, velocity, and six-step settling conditions together.
Before treating ObjectToDrawer as a self-contained policy dataset, the drawer
retention behavior should be moved into the environment contract or removed
and the affected episodes recollected.

NutPouring is sensitive to small cup-contact and ballistic-release differences.
Twenty replays exhausted the action horizon without a settled nut and five
terminated because the nut or cup left the workspace. This does not invalidate
the recorded state-action pairs for closed-loop imitation learning, but it does
show that the action sequence alone is not a robust open-loop controller.

## Reproduction

Run one dataset with:

```bash
cd <repository-directory>
python \
  scripts/replay_recorded_actions.py \
  --dataset-dir datasets/primary_50/push_object_to_region \
  --headless --physics-only --log-interval 0
```

Use the corresponding binary-gripper dataset directory for DrawerOpening,
CabinetOpening, and ObjectToDrawer. Each dataset directory contains
`action_replay_report.json` with per-episode reset errors, replay outcome,
failure reason, replay step, and final task state, plus `action_replay.log`.

## Additional ObjectToDrawer and NutPouring batch

An additional non-overwriting batch was collected with base seed `20260927`
under `datasets/additional_50_seed20260927`. Both datasets use the
same 20 Hz observation/action contract and task randomization as the primary
batch.

| Task | Collection | Transitions | Static data | Initial task state | PhysX action replay |
|---|---:|---:|---:|---:|---:|
| ObjectToDrawer | 50 successes / 64 attempts | 27,111 | 50/50 | 50/50 | 31/50 (62%) |
| NutPouring | 50 successes / 60 attempts | 26,522 | 50/50 | 50/50 | 23/50 (46%) |

All 19 ObjectToDrawer replay failures exhausted the recorded action horizon
without jointly satisfying the drawer, containment, release, velocity, and
settling conditions. For NutPouring, 21 failures exhausted the action horizon
and 6 terminated because the cup or nut left the workspace. These results are
consistent with the first batch: the recorded episodes are valid closed-loop
expert executions, while their contact-sensitive action sequences are not
fully repeatable open loop.
