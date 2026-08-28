# Articulated-task physics validation

This note records the empirical PhysX calibration and the acceptance contract
used for DrawerOpening, CabinetOpening, and ObjectToDrawer. These values are
simulation estimates selected from controlled effort sweeps; they are not
measurements of the real cabinet.

## Nominal parameters

| Component | Moving mass | Static friction | Dynamic friction | Joint damping |
|---|---:|---:|---:|---:|
| drawer rail | 1.7 kg | 10.0 | 8.0 | 15.0 |
| door hinge | 1.2 kg | 1.8 | 1.0 | 1.0 |

The cabinet contact material remains at 0.45 static and 0.38 dynamic friction.
The door handle has a separate effective rubber-pad/aluminum contact material
at 0.80 static and 0.60 dynamic friction.
The articulated-task gripper uses a 35 N per-finger effort ceiling, representing
an approximately 70 N total Panda grasp-force limit.

PhysX joint friction fields are solver coefficients, not direct SI force or
torque measurements. The values above should therefore be treated as nominal
simulation parameters and later randomized or replaced after hardware system
identification.

## Handle-grasp acceptance

The old expert stopped the fingertip TCP outside the handle. Both finger joints
then closed to approximately 0.0001 m and the palm geometry hooked the handle.
That behavior is no longer accepted.

Before pulling, the current expert requires:

- the handle center to lie inside the finger-pad region;
- both finger joints to remain between 0.0015 and 0.0120 m;
- left/right finger asymmetry to remain below 0.0040 m;
- the grasp to remain valid during the pull.

The high-level gripper command is binary. It remains fully open during handle
approach and switches once to fully closed at the final grasp waypoint. The
nonzero finger aperture reported below is caused by physical handle contact,
not by commanding a partial opening.

The collector additionally requires the expert's persistent
`handle_grasp_verified` flag at task termination. Reaching the drawer or door
joint threshold without this flag is logged as `unverified_handle_grasp` and is
not saved.

## Randomized validation

| Task | Result | Duration range | Terminal condition |
|---|---:|---:|---|
| DrawerOpening | 10/10 | 10.55--10.60 s | drawer reaches 0.16 m |
| CabinetOpening | 10/10 | 14.45--14.55 s | door reaches 1.20 rad |
| ObjectToDrawer | 10/10 | 26.00--27.25 s | drawer open and released banana settled inside for 0.30 s |

These results use 100 Hz PhysX with 20 Hz actions/observations and cabinet XY
randomization of +/-2 cm plus yaw randomization of +/-5 degrees. The door
cabinet nominal X position is 0.75 m; the drawer cabinet remains at 0.72 m.
Across the accepted DrawerOpening and CabinetOpening pull phases, both finger
joints remain nonzero and satisfy the bilateral-grasp check. No articulation
joint state is overwritten during an episode. ObjectToDrawer uses an additional
vertical retreat after releasing the handle, preventing the arm from striking
and closing the open drawer on its way to the banana. The banana grasp holds the
unchanged final waypoint open for 0.35 s, sends one close command, then waits
for the physical grasp to stabilize before lifting.

The accepted datasets are stored in:

- `outputs/demos/drawer_physical_validation_v1`
- `outputs/demos/cabinet_physical_validation_v1`
- `outputs/demos/object_to_drawer_physical_validation_v1`

Episode zero from each set is rendered in `five_task_success_gallery`.
