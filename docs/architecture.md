# Architecture contract

## Environment API

All tasks will use the Isaac Lab manager-based environment pattern and the
standard Gymnasium return signatures:

```python
observations, info = env.reset(seed=seed)
observations, reward, terminated, truncated, info = env.step(action)
```

The common action is a seven-dimensional relative Cartesian command:

```text
[dx, dy, dz, d_axis_angle_x, d_axis_angle_y, d_axis_angle_z, gripper]
```

The six Cartesian components are normalized controller commands. The gripper
is binary, using `+1` for fully open and `-1` for fully closed. Deployment maps
those values to the real gripper's `1/0` commands.

## Observation groups

`policy` is deployable on the real robot:

- calibrated composite RGB;
- joint position and velocity;
- gripper state;
- end-effector pose computed from robot kinematics.

`critic` is privileged and simulation-only:

- object and target poses;
- rigid-body velocities;
- cabinet joint state;
- task progress variables.

Raw oracle values may also be written to datasets and `info`, but they must not
silently enter the visual actor input.

## Module boundaries

- `envs/common`: shared Franka, calibrated table, camera, compositing, actions.
- `envs/<task>/mdp`: observations, rewards, events, and terminations.
- `data`: synchronized demonstration recording and conversion.
- `wrappers`: policy-library adapters; environment definitions do not depend on
  DPPO, SKRL, RSL-RL, or a particular imitation-learning library.

## Migration rule

The custom calibrated scene is the authoritative geometry. Existing Track B
reference environments are sources for action, observation, expert, and reward
logic only. Thresholds and joint names must be recalibrated against the custom
assets; for example, the custom drawer has a 0.22 m limit and cannot use the old
0.30 m success threshold.
