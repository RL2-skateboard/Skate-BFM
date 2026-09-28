# skate-sim

Manual tilt/steering from rest (MuJoCo):

```bash
python play.py --backend mujoco --scene board --model reduced --experiment manual --record runs/manual_points
```

Initial linear/angular/wheel velocities are zero; no roll fixture is enabled.
Double-click the exact deck surface point, then Ctrl+right-drag for translation
force. Double-click again at a different point to move the grab point. Shift
changes the drag plane. The pink marker and overlay report the selected point;
E releases inputs, R restores rest. Native mouse manipulation supports one
active grab at a time, not simultaneous independent mouse forces. The world
point argument to MuJoCo apply_wrench now adds the correct moment about COM.

Stage 01 has been accepted by the user. Stage 02 assets are in progress.
Inspect the new assets with:

```bash
conda activate skate-sim-mjlab
python -m skate_sim build --asset board --style all
python play.py --backend mujoco --scene board --style standard --inspect
python play.py --backend mujoco --scene board --style cruiser --inspect
python play.py --backend mujoco --scene board --style longboard --inspect
python play.py --backend mujoco --scene gallery --inspect
python -m skate_sim check --stage 02 --backend mujoco --output runs/02_check
```

The same board asset can be inspected through the active CUDA mjlab backend and
its white Viser web viewer:

```bash
python play.py --backend mjlab --scene board --style standard --inspect
```

The terminal prints the actual Viser URL because each server uses a free port.
Open that URL. The Viser panel provides pause, step, reset, board-asset
selection, scene visibility and X/Y/Z force controls. Set the sliders to apply
a world-frame force to the deck, then click Release Force to clear it. The
terminal should show only one Viser HTTP URL for the active process. Changing
`Board asset` keeps that URL and browser connection while rebuilding the active
mjlab/Warp model in place. The new asset starts from its nominal initial state;
no `play.py` restart is needed.

Inspect: Space resumes physics, 1–4 edit individual wheel angles while paused,
V toggles transparency, J toggles joint axes, R resets. See docs/stages/02.md
for measured checks and remaining requirements.

In board/gallery windows, left-drag a selected deck, truck or wheel to apply a
finite world perturbation. Test both drag directions, watch the board move or
tilt, then release and confirm the mouse force clears. Records distinguish
`mouse_force_N` from keyboard force. This drag is an external inspection input;
natural truck turning is evaluated in Stage 03.

`skate-sim` is an independent simulation project for a skateboard, robot and
their contact experiments. The active delivery scope is MuJoCo plus mjlab /
MuJoCo Warp. Isaac Sim is explicitly deferred and is not a required dependency,
backend, or acceptance target for the current project. The project is being
implemented in the acceptance order described in [`plan/`](plan/00_overview.md).
Stage 01 currently exposes a real MuJoCo workbench with a ground plane, axes
supplied by the viewer, a massive collision cube, reset, stepping, finite
wrench input and JSONL records.

## Environments

The target machine was checked on 2026-09-26:

| item | observed value |
| --- | --- |
| OS / architecture | Ubuntu 24.04, x86_64, GLIBC 2.39 |
| GPU / driver | RTX 4060 Ti 16 GB, NVIDIA 575.57.08, CUDA 12.9 |
| active conda environment | `skate-sim-mjlab`, Python 3.11.16 |
| MuJoCo | 3.11.0 |
| mjlab / Warp | 1.6.0 / 1.17.0, MuJoCo Warp 3.11.0 |
| Isaac Sim | deferred, not part of active delivery |

The active runtime uses one environment. The source tree and the `play.py`
entry point remain shared:

```bash
conda env create -f environment-mjlab.yml
conda activate skate-sim-mjlab
python -m pip install -r requirements-mjlab.txt
```

## Stage 01 commands

```bash
conda activate skate-sim-mjlab
python play.py --help
python -m skate_sim doctor --backend mujoco --display
python play.py --backend mujoco --scene bench --seed 0 --record runs/01_mujoco
python -m skate_sim check --stage 01 --backend mujoco --output runs/01_check
```

The mjlab package probe runs in its own environment:

```bash
conda activate skate-sim-mjlab
python -m skate_sim doctor --backend mjlab
```

Isaac Sim is deferred from the active scope. The `skate-sim-isaac` environment
is retained only as an installation experiment and is not used by the active
checks or delivery commands.

The MuJoCo window supports Space (pause), N (one physics step), M (one
control step), R (reset), F (apply a 20 N world-X pulse for 0.2 simulation seconds), and E (release the
force). The record contains metadata, states, and event timestamps.

`mjlab` runs MuJoCo Warp on CUDA. Do not treat the deferred Isaac installation
as a working backend. Active checks run separately in the MuJoCo/mjlab
environment; no active command uses `--backend all`.

The previous Isaac Sim experiment was blocked by host-level file-watcher
limits. It is outside the active project scope and is not a prerequisite for
the next stage.
