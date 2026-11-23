# skate-sim acceptance record

Correction: 2026-11-04. The user identified an unintended flat-deck asset
substitution in the unified scene. The prior asset-consistency signoff is
withdrawn. The five original BoardStyle assets are now restored, including
kicked tips, bridge/axle geometry, wheels and original dimensions, masses and
material parameters. Only the tilt-steer kinematic principle is borrowed.
Obstacle previews now use actual geometry and new additions are staggered.
The current 62 unit tests pass. Restored-asset CPU integration passed 500 ticks;
CUDA no-controller and real BFM-action records each passed 500 continuous ticks
in independent processes. CUDA partial-reset replay passed two 500-tick segments.
All qpos/qvel/time errors are zero at the unchanged 1e-7 threshold, with model
and numerical identities verified. Both backends' visual state/force readback
passed exactly. Evidence is under `runs/asset_restore_20261104/`.
Real CUDA registered-callback checks cover obstacle shapes, staggered placement,
apply, drag/revert, save/load and recording locks; they are not browser pixel
acceptance. No user window signoff is recorded. Current Warp logs warn that
cylinder–cylinder/box/mesh pairs support only a single contact for these cases;
this engine limitation is distinct from display failures and capacity overflow.
Current-asset stock-vs-hook probes retain the same contact physical payloads
and constraint counts; contact row addresses reorder and numerical forces are
not bitwise identical between the two implementations. Strict replay compares
each pinned implementation to its own recorded trajectory. N64 subset-reset
and short-run finite/capacity checks pass; N64 is not certified for a 500-tick
independent replay. Paired GPU timing (3 warmup steps, then three 10-step windows,
no concurrent stepping) gives N2 median 0.01469/0.52961 s and N64 median
0.01678/2.66574 s for stock/hook, about 36/159 times slower. This is a material
throughput limitation, not a claim of efficient training performance.

The following numerical evidence used the unintended flat-deck asset and is
retained as history, not as acceptance of the restored assets. Pinned CUDA execution fixes
the constraint reduction and contact ordering; two-world no-controller and
real BFM-action recordings each passed independent-process integration over
500 continuous ticks at the unchanged 1e-7 threshold. A partial-reset record
passed two 500-tick segments. Maximum qpos/qvel/time errors were all zero;
compiled model and numerical implementation identities were verified.
Latest local evidence: `runs/gpu_strict_20261104/formal/`. The 57 unit tests,
physical contact-preservation checks, five board styles and N64 subset reset
checks passed. N64 short-test median step cost increased about 215 times;
this material performance limitation remains. User signoff is NOT recorded.
Stage 04 remains frozen and the overall research project remains incomplete.

The dated evidence below describes the prior numerical implementation. Its
GPU failures are retained, not rewritten as successes of the new strategy.

Updated: 2026-11-02. Scope: complete Stage 05 engineering and later noncontact
infrastructure. Stage 04 contact research is frozen pending discussion with the
user. No contact model, learning-data preparation or fitting scheme was changed.

**Overall project: INCOMPLETE.** Automatic submatrix passes do not imply user
window signoff or full-stage completion. Stages 01–03 retain their recorded
historical user acceptance. Stage 05 is now ACCEPTED: on 2026-11-02 the user
reopened its review window, completed verification and stated “好了，验证完成”.
That confirmation applies only to Stage 05, not the other stages.

## Verified in this iteration

- Stage 05: independent 721-field numerical reference, 128 native entry states,
  real CUDA 128-state reference (maximum absolute error 4.7684e-7), previous-command
  scaling, history/reset, dynamic substep PD and official tracking windows.
- Native joint inspection: all 29 joint directions/limits, explicit fixed-base
  inspection fixture, free pose/policy switching, state readouts, robot pelvis
  perturbation, pause/physical-step/control-step/reset semantics.
- Real CUDA: robot N=1 and N=64 and rider N=64 reward each executed 15,000 physical
  steps and 1,500 policy updates per environment over the measured 30 simulated
  seconds. Setup probes are counted separately. Physics is CUDA; ONNX inference
  uses CPUExecutionProvider. Finite-state/interface checks do not establish
  autonomous riding or generalization.
- Native noncontact Stage 06: three presets, ten reset cycles each, atomic
  state roundtrip, named-body world/body-frame point force, exact duration and
  clearing, explicit terminated/time-limit-truncated guards, actual policy with
  force/reset replay. The guard defaults are engineering diagnostics, not
  calibrated success definitions.
- GPU interfaces: non-contiguous subset state/reset/force isolation, correct
  point torque, independent fall/timeout, physical-step offboard latch, pelvis
  and deck bounds, genuine nonconstant controller diagnostics.
- Version-2 recording: actual initial integration/control state, commands and
  events per physical tick including paused steps; checkpoint/latent/config/model
  hashes. Dynamics replay integrates from initial state; snapshots only compare.
  Legacy logs remain visual-only when required inputs are missing. Visual replay
  has no checkpoint dependency and rejects dynamical controls.
- CLI: each backend runs in its own process. Scope, automatic checks, stage
  acceptance and overall project status are separate. Missing checkpoint,
  failed/blocked/not-run/deferred items cannot silently become full acceptance.
  A successfully written report does not mean the reported checks passed.
- Existing Stage 09 noncontact parameter/nominal-reset/seed/PD/noise/delay checks
  and synthetic board-release CEM were rerun. No new contact method was activated.

## Results and limits

Native bench/board/robot none/pose/policy recordings and the rider policy record
had zero error under the registered 1e-7 short-horizon replay comparison. Each
100-tick record covers 0.20 s and resets at tick 50; its longest continuous
interval is 0.10 s. This must not be described as a complete continuous 1 s test.

**GPU robot strict dynamics replay: FAIL at 1e-7.** The latest 100-tick replay
reached about 1.055e-5 maximum mixed state error. A separate three-way 200-tick
comparison included original integration, a fresh normally stepped instance and
a restored replay instance. Both fresh and restored instances showed divergence:
peak qvel differences about 1.204e-5 and 1.089e-5 respectively. Initial public
and integration fields, raw actions, targets, external forces and clocks matched.
PD feedback force differences developed after state differences; actual applied
joint forces were not claimed to remain identical. This rules out attributing
all drift to restore alone; no particular GPU kernel cause is established.
The original failure is preserved and no tolerance was relaxed.

Five-style, 2 s coast/release cross-backend comparisons check finite assets and
report numerical differences. They do not replace Stage 03's full 60 s,
step-convergence and energy acceptance matrix.

The working tree and an isolated source copy passed the formal environment's
unittest suite. The copy was installed editable in a temporary virtual environment
with verified existing runtime dependencies exposed via system-site-packages.
It imported the isolated source, passed pip check and reran Stage 05 with the
real external checkpoint. This is source isolation and package reconstruction;
it is not a claim of a fresh network dependency solve or clean conda install.

Actual native window and Viser service startup succeeded. Real-service callback
checks verified one continuous physics owner, synchronized pause, one physical
N-step without policy advancement, visual replay locks and recording/style locks.
Current Work settings disable local browser use and native computer control;
the agent did not obtain screenshots or perform clicked UI acceptance. The
user subsequently completed the native Stage-05 review and confirmed it.
Stage-06/07 user window confirmation is still outstanding.

## Evidence and reproduction

Consolidated metrics, exact raw paths, versions, source hashes and frozen-contact
hashes: `runs/noncontact_completion/verification.json`.

Selected raw records:

- `runs/05_observation_fix/verification.json`: earlier complete native observation
  repair evidence, including reward/tracking/goal 30 s records.
- `runs/05_completion/native_joint_guards.json`: native joint/guard tests.
- `runs/05_completion/ui_review/observations.md`: actual UI access limitation.
- `runs/noncontact_completion/unittest_final.txt` and `clean_unittest.txt`.
- `runs/noncontact_completion/clean_stage05_signed_scope/result.json`.
- `runs/noncontact_completion/cli_native06/result.json` and `cli_native09/result.json`.
- `runs/noncontact_completion/gpu_oracle_n64.json`, `gpu_robot_n1.json`,
  `gpu_robot_n64.json`, `gpu_rider_n64.json`, `gpu_interfaces_n64.json`.
- `runs/noncontact_completion/native_wrench_replay.json`, `native06_records/`,
  `gpu_replay_final.json`, `gpu_repro_diagnosis.json`, `gpu_web_callbacks.json`.

From the project root, using the existing skate-sim-mjlab environment:

```bash
python -m unittest discover -s tests -v
python -m skate_sim check --stage 05 --backend mujoco --scope noncontact \
  --checkpoint checkpoints/bfm-zero --output runs/05_check
python -m skate_sim check --stage 06 --backend mujoco --scope noncontact \
  --checkpoint checkpoints/bfm-zero --output runs/06_noncontact
python -m skate_sim check --stage 07 --backend mjlab --scope noncontact \
  --num-envs 64 --checkpoint checkpoints/bfm-zero --output runs/07_noncontact
```

The last command checks the batch/control subset; its output explicitly lists
unexecuted strict-replay/manual gates. Its scope PASS cannot override the saved
strict replay FAIL or incomplete user acceptance.

The sole visualization entry remains `play.py`. For Stage 05 review:

```bash
python play.py --backend mujoco --scene robot --controller joint --inspect \
  --checkpoint checkpoints/bfm-zero --record runs/05_joint_review
```

Select joint with [ / ], adjust with - / +, P selects free pose, O selects policy,
E releases external force/inspection weld, Space pauses, N advances one physical
step, M one control step and R resets. The window labels inspection support.
The user's confirmation required by plan/05_robot.md was provided after the
reopened Stage-05 review, independently of the agent's programmatic tests.
New recordings reject overwriting existing logs.

## Current capability inventory (2026-11-02)

After user acceptance of Stage05, a fresh read-only run sampled the full
feature surface. All six scenes on both engines loaded and advanced; five
board styles, the native truck/reduced experiment variants, native robot
controllers and three BFM context tasks, rider presets, recorded playback,
existing offline model evaluation and synthetic calibration were exercised.
The current unittest suite passed 30 tests. This does not turn every runnable
contact scene into an accepted physical experiment.

Two extra GPU limitations were actually reproduced: a named goal fails to load
because the name is not passed through, and a custom reward selection silently
resolves to the default reward. GPU tracking runs. Joint inspection is a
native-only feature, not a CUDA feature. Explicit experiment flags are rejected
for mjlab; native model selection is routed through its experiment configuration.

The contact fixed-board selector and clamp-release defects were compiled and
probed, without modifying the frozen implementation. Current force/power
diagnostics also do not meet the planned signed point-contact definitions.
Stage04 remains unaccepted. A legacy standalone rider XML has unresolved
includes; the normal scene factory works.

A new no-controller GPU board record passed continuous500-tick/1s strict
dynamics replay with zero error. This does not supersede the failed GPU
robot-policy/force/reset case. Raw current smoke tests, expected input guards,
unsupported combinations, historical failures and reused long-run evidence are
separate rows in `runs/inventory_20261102/validation.json`; compiled asset
counts and compatibility boundaries are in `assets.json`.

## Readiness for contact research: latest user target

The user clarified that the current deliverable is an environment-building
chain: independently select and place/remove the robot, board and other assets
in both MuJoCo and mjlab, with usable basic board dynamics, before starting
foot-contact and later skateboard-motion-model research.

Against that target the preparation is not yet complete. Existing presets and
Stage05 robot acceptance are useful components, not a general scene composer.
The minimum interpretation is configuration before launch and rebuilding the
scene; a live drag/drop editor or arbitrary multi-robot system is not assumed.

Two essential work packages remain:

1. A shared scene specification/compiler used by both backends. Existing robot,
   board style and each gallery obstacle need independent enable/disable and
   pose settings. Current gallery hardcodes all three obstacles, while rider
   constructs robot+board without those obstacles. Loading/removing assets should
   remove their bodies/collisions as well as appearance and preserve reproducible
   reset/recorded configuration.
2. A shared basic board-dynamics interface and matched tests. Native has the
   existing accepted truck/reduced board-only baseline. The public experiment
   path rejects mjlab; model selection without an experiment is ignored. Native
   and CUDA steer_start also disagree on initial wheel speeds. Board-only and
   robot+board should retain the same declared passive mechanisms. A common
   truck baseline is enough to start; a reduced+rider option must either work
   or be explicitly unavailable, not silently route or fail inside composition.

New read-only probes in `runs/readiness_20261102/probes.json` confirmed:

- All five truck styles compile with G1, with the intended board masses and
  wheel radii. This is a composition/geometry check, not a new long behavior test.
- reduced+rider fails in both backends' shared builder: `board XML has no deck`.
- standard steer_start has native deck/pelvis speed1m/s and four wheel speeds
  28.5714rad/s; real RTX4060Ti CUDA has the same linear speeds but four zero
  wheel speeds before any integration.

Minimum completion examples are robot-only, board-only, robot+board, each with
selected obstacles where requested, and removal of individual assets, under the
same configuration on both engines. Check all board styles for loading/short
steps, then use a representative style for matched coast, release and left/right
turn response. Keep robot and board initially separate when checking passive
board behavior; stable riding, a new foot-contact law and a full motion-learning
model are not preparation prerequisites.

Stage04 defects, GPU named-context options and strict robot-policy replay remain
recorded separately. They should not displace the scene-composition and basic
dynamics work as the immediate priority for this clarified target. No source
implementation was changed during this readiness check.
