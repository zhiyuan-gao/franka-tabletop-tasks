# Reinforcement-learning environments

## Current boundary

All six tasks are registered Isaac Lab `ManagerBasedRLEnv` environments. The
successful-demonstration experts and the RL policy use the same normalized 7D
control interface:

```text
[dx, dy, dz, dRx, dRy, dRz, gripper]
```

The six Cartesian values are relative end-effector commands. The gripper term
is strictly binary: `+1` means fully open and `-1` means fully closed. A real
robot adapter maps these two values to the hardware's `1/0` open/close API; no
intermediate aperture is exposed to the policy.

The policy observation is a dictionary rather than a privileged state vector:

```text
policy.rgb       float32 [N, 3, 224, 224], range [0, 1]
policy.proprio   float32 [N, 25]
critic           float32 [N, task-dependent state dimension]
```

`policy.rgb` is the Isaac-rendered foreground composited over the calibrated
real background. `policy.proprio` contains joint positions, joint velocities,
and the FK-derived end-effector pose. The actor must only consume these two
terms. The `critic` group contains simulator-only poses, velocities, and joint
state and is intended for an asymmetric critic, reward evaluation, expert
generation, and debugging.

Each environment already defines task randomization, dense reward terms,
physical success conditions, failure/timeout termination, and automatic reset.
The expert trajectory code is not called by `env.step()` and does not leak
waypoints or oracle actions into the learned policy.

## Registered tasks

| Task name | Gym ID |
| --- | --- |
| `push_object_to_region` | `TrackB-PushObjectToRegion-Franka-IK-Rel-Visual-v0` |
| `push_cube_to_region` | `TrackB-PushCubeToRegion-Franka-IK-Rel-Visual-v0` |
| `drawer_opening` | `TrackB-DrawerOpening-Franka-IK-Rel-Visual-v0` |
| `cabinet_opening` | `TrackB-CabinetOpening-Franka-IK-Rel-Visual-v0` |
| `object_to_drawer` | `TrackB-ObjectToDrawer-Franka-IK-Rel-Visual-v0` |
| `nut_pouring` | `TrackB-NutPouring-Franka-IK-Rel-Visual-v0` |

## Training order

Environment correctness and policy learnability are separate milestones. The
recommended order is:

1. Train `push_cube_to_region` first to validate the multimodal actor and
   asymmetric critic implementation, then compare it with the more
   contact-sensitive `push_object_to_region` sugar-box task.
2. Train `drawer_opening` and `cabinet_opening`, reusing the shared handle
   approach and articulation rewards.
3. Use behavior cloning from successful trajectories to initialize the
   long-horizon `object_to_drawer` and `nut_pouring` policies, then fine-tune
   with RL and task-stage curricula.

The first trainer should preserve the RGB/proprio structure and use a CNN image
encoder plus an MLP proprio encoder. Flattening the 224x224 image directly into
a generic MLP would technically run but is not the intended visual policy.

## SKRL training

SKRL 2.1 is the single supported trainer. The project-local actor decodes the
SKRL-flattened Gym dictionary using slices derived from the observation space:

```text
one main RGB view [3, 224, 224] -> CNN ----+
                                                +-> Gaussian 7D actor
Franka proprio [25]               -> MLP ----+

simulator-only critic state       -> MLP ------> scalar value
```

The `3` in the RGB shape means red, green, and blue channels, not three camera
views. The actor checkpoint therefore remains a single-camera deployable
policy. The critic state is never passed into the actor.

Run a small end-to-end PPO update first:

```bash
python scripts/rl/train.py \
  --task push_object_to_region \
  --smoke \
  --headless
```

Start a normal run using the packaged YAML configuration:

```bash
python scripts/rl/train.py \
  --task push_object_to_region \
  --num-envs 4 \
  --headless
```

Runs are written under `outputs/rl/skrl`. Every run contains its resolved
configuration and checkpoints. Resume a complete run with `--checkpoint`, or
initialize only the actor from a behavior-cloning weight file with
`--actor-checkpoint`.

Evaluate deterministic mean actions with:

```bash
python scripts/rl/play.py \
  --task push_object_to_region \
  --checkpoint outputs/rl/skrl/<run>/checkpoints/agent_final.pt
```
