# Procedural assets

Board and gallery geometry is generated locally by skate-sim, revision 1.
Board/gallery assets embed no third-party meshes. Units: metres, kilograms, seconds, radians.
The original rounded deck uses convex mesh strips, including raised nose and tail.
Bridge/axle collision shapes are boxes, wheels are cylinders, and the ramp is a
convex triangular prism.
Masses are design assumptions. build/play use the same BoardStyle objects
and SHA-256 configuration hash. Generated XML is output, not a parameter source.

The unified scene keeps these project assets. It borrows only the deck-tilt /
opposed-truck steering coupling principle from
[HUSKY](https://github.com/TeleHuman/humanoid_skateboarding/tree/d93833e80deff7f927c0b80ef9c435d8b5c488fe).
Its flat deck, marker, mass properties and actuator settings are not runtime assets.
The project vertical bridges become the fixed chassis, while the original axles
and wheels steer and the original deck tilts. No extra ballast is added. At neutral
pose, each original geom and the total mass, centre of mass and inertia are retained.
The articulation ownership of the bridge/axle mass changes as part of the reduced
kinematic assumption. Standard total mass is 3.00 kg; the other styles retain their
original 2.18 / 4.00 / 2.60 / 5.04 kg values.

The G1 XML and meshes under `robot/g1/` are the robot assets used by the
BFM-Zero deployment adapter. Sources and the fixed deploy revision are recorded
in [the robot contract](../docs/robot.md); preserve embedded third-party notices.
The upstream BFM-Zero repository's CC BY-NC 4.0 license is retained alongside
the small official validation configuration fixtures under
`src/skate_sim/validation/reference05/LICENSE`. This notice does not replace
any asset-specific third-party terms. Checkpoints are external, ignored files
and are not included in the source project.

Joint inspection creates a temporary scene with an explicitly labelled pelvis
weld. It is removed after model compilation; free-base pose/policy and rider
scenes do not use that inspection fixture.

## Compiled inventory (2026-11-02)

Five board styles are generated from `models/board.py`: street, standard,
cruiser, longboard and downhill. Each has a free deck, two passive truck tilt
joints, four independent wheel-spin joints and 19 named sites. Their geometry,
wheel dimensions, mass and truck parameters differ; they are not five color
variants. `assets/board/manifest.json` is generated parameter evidence.

The active G1 has 29 controlled hinges (including six wrist hinges) and a
floating base, not articulated fingers. There are 62 STL files on disk; the
active official model references 36. Separate/ground and three rider preset
XMLs are scene compositions of that robot. The legacy standalone
`assets/robot/rider_standard_free.xml` fails its include resolution and is not
a runnable standalone asset; use the scene factory through play.py.

Gallery obstacles are procedural ramp, low box platform and horizontal
cylindrical rail; there is no separate terrain asset package. Default board
and rider scenes omit those obstacles. The contact-test foot is a 0.8 kg free
capsule (endpoints x=±0.07 m, radius0.045 m); actual G1 soles instead use four
sphere collision proxies per foot.

The exact compiled counts, all parameters, anchors, active/unused mesh names
and external checkpoint context names are in
`runs/inventory_20261102/assets.json`. Disk presence alone is not a claim that
an old XML or every supported-looking option is runnable.
