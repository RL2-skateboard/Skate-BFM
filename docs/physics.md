# Stage 03 model definitions

X forward, Y left, Z up; angles radians; quaternions wxyz. Nominal board
parameters are engineering assumptions. No real-world accuracy is claimed.

Source reviewed: HUSKY skateboard.xml at commit
`d93833e80deff7f927c0b80ef9c435d8b5c488fe`:
https://github.com/TeleHuman/humanoid_skateboarding/blob/d93833e80deff7f927c0b80ef9c435d8b5c488fe/src/mjlab_husky/asset_zoo/robots/skateboard/xmls/skateboard.xml

The source uses a floating chassis, a loaded deck child rotating around -X,
front steering -Z and rear steering +Z. Two equalities say
q_tilt = -0.577 q_steer (not the inverse). Our reduced comparison reverses all
three axes, retaining that equation. The actual deck meshes and foot anchors
belong to the roll child. A physical chassis spine carries 10% of the original
deck mass; the deck carries 90%, preserving total mass. Trucks attach to the
chassis. No empty tilt body or commanded world yaw is used. This reimplements
the published topology/relationship with our procedural geometry and parameters.

Truck model: deck -> inclined front/rear hinges -> wheels. Native linear spring
and damping plus substep cubic spring -k3*q^3. Bearing resistance is
-b*tanh(w/0.1), wheel viscous resistance is -c*w. There is no additional rolling
resistance term; native sliding friction is distinct from bearing loss.

Coast initializes root velocity and wheel angular velocities consistently.
Release starts with truck displacements (reduced also initializes the coupled
deck roll); root is lifted only enough to avoid initial wheel penetration.
Drop starts 0.08 m above nominal. Lean is a roll-torque fixture, turn combines
that fixture with initial forward speed. Both apply a measured external roll
torque, not yaw pose control. E releases the fixture. Freeboard coast/release
have no fixture. External work integrates world force/torque power at body COM
with trapezoidal endpoint velocities. Native energy includes all bodies and
linear springs; cubic spring energy is added separately. Contact dissipation
and integration error must be distinguished from external work.

## Stage 09 parameter experiments

Stage 09 keeps nominal model fields as the baseline and applies one bounded
episode parameter set after reset. Supported parameters and ranges live in
`src/skate_sim/physics/randomize.py`; sampled values are logged with the seed
and source. MuJoCo applies mass/inertia, friction, truck and wheel-loss fields
to the native model; COM offsets modify the deck inertial frame rather than
moving the deck geometry. mjlab keeps its CUDA model as the physics owner and
adds per-environment randomized mass-load, truck/wheel loss, COM gravity
torque, PD and observation terms through explicit runtime fields.

The CEM command generates separated synthetic complete trials by running the
native board wrapper, fits bounded parameters appropriate to the selected
release/coast experiment, and reports nominal baseline, training and held-out
forward-rollout loss plus synthetic-truth parameter RMSE. Synthetic results are
method validation only; without measured trials the project makes no claim to
have reduced a real sim-to-real gap.
