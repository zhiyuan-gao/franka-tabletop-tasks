# Task definitions

## PushObjectToRegion

Language instruction: **Push the sugar box into the blue rectangle.**

The manipulated object is the official YCB `004_sugar_box`; the stock Isaac
Lab cube is not part of this task. This object was selected because its compact,
flat footprint is easier to push and less prone to rolling or tipping than a
can or bottle. The box must start in a stable tabletop
orientation. Reset randomization places the box near `(0.46, -0.305)` m with a
roughly 6 x 6 cm planar range, plus randomized yaw. The
18 x 12 cm blue rectangle is printed at the center of a landscape A4 white sheet. The
sheet is aligned over and visually covers the existing ChArUco world marker.
Its pose is fixed by the marker calibration without adding another fiducial.

The deployable actor observes only composite RGB and Franka proprioception.
Object pose, velocity, target-rectangle pose, footprint containment, tilt, and workspace
validity are privileged values for the critic, expert, reward, termination, and
dataset annotations.

Success requires the complete oriented box footprint to lie inside the target
rectangle while the box stays upright. Falling from the table or exceeding the
allowed tilt is failure.

## Banana to Drawer

Language instruction: **Pick up the banana and place it inside the drawer.**

The manipulated object is the official YCB `011_banana`; the procedurally
generated cube in the earlier reference environment is not part of this task.
The cabinet is randomized by +/-2 cm in XY and +/-5 degrees in yaw. The banana
receives a separate cabinet-local XY offset of +/-2 cm and yaw offset of +/-10
degrees while preserving its physically stable broad-side pose. The drawer
begins closed unless an explicit easier curriculum stage states otherwise.

Banana grasp proposals must account for its curved, asymmetric geometry. The
old cube-specific magnetic attachment must not silently define the final task.
An optional assisted-grasp curriculum may exist, but it must be named and
reported separately from the contact-physics environment.

Success requires all of the following:

- the drawer is open far enough to admit the banana;
- the banana lies inside the drawer's interior volume with suitable clearance;
- the gripper has released the banana;
- the banana remains inside for a short settling window.

The actor receives composite RGB and Franka proprioception. Banana pose, drawer
joint state, handle pose, drawer interior volume, velocities, and containment
tests remain privileged critic/oracle information.
