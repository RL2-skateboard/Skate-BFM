# H1 Experiment Specification

## Goal

Measure the short-horizon motion coverage of a **frozen official BFM0**
inside the HUSKY skateboard simulator. H1 asks whether the tested BFM0
latent space contains push/steer-like robot motion under the current
23DoF HUSKY control interface. It does not train Skate-BFM and does not
claim complete skateboarding.

The two matched sub-experiments are:

- `H1-without-prior`: goal retrieval without expert latent proposals.
- `H1-with-prior`: retrieval initialized from expert backward-map latents.

Implementation:
[`src/skate_bfm/exp/h1_bfm_coverage/core.py`](../src/skate_bfm/exp/h1_bfm_coverage/core.py)  
Configuration:
[`configs/h1_bfm_coverage.yaml`](../configs/h1_bfm_coverage.yaml)

## Common Protocol

- Frozen checkpoint: `model/bfm-zero-official`.
- Latent dimension: `z in R^256`, projected to `||z||_2 = 16`.
- Control rate: `50 Hz`; horizon: `0.5 s`.
- Seeds: `0, 1, 2`; robustness trials: `20`.
- CEM population: `64`; CEM iterations: `6`; elite fraction: `0.15`.
- Geodesic support angles: `5, 10, 20, 40, 80` degrees, `16` samples per
  angle.
- Action path: BFM29 normalized action -> name-based HUSKY23 action mapping ->
  HUSKY MuJoCo PD control.
- Expert data defines target motion, initial state, and evaluation score.
  It is not automatically a latent proposal in `without_prior`.

The latent-space distance used in the reports is:

```text
d_geo(z1, z2) =
  acos( clamp( <z1, z2> / (||z1||_2 ||z2||_2), -1, 1 ) )
```

Dynamic human-push files do not contain synchronized skateboard state. Their
initial root trajectory is therefore rigidly aligned to the static push
reference: the right support foot is placed on the deck, while the left foot
keeps its source-relative position and height.

## H1-without-prior

**Method**

1. Sample `256` constant latent directions globally on the normalized latent
   sphere.
2. Score each direction against the target rollout.
3. Start broad CEM from the global best with maximum search angle `180 deg`,
   `initial_std=1.0`, `min_std=0.05`, and temporal correlation `0.0`.
4. Run robustness trials with the configured reset noise.

The global scan searches the latent space directly. CEM refines the selected
global candidate; it does not use an expert backward latent.

**Verification**

- [x] Global scan, CEM, robustness evaluation, plots, and videos completed.
- [x] The same target/initial-state protocol was used for all targets.
- [x] Six of eight targets reached the recorded local-coverage criterion.
- [ ] A positive result proves complete board contact and task success.
- [ ] A finite failed search proves that no matching latent exists.

## H1-with-prior

**Method**

For a dynamic expert window, reconstruct the official backward observation and
compute the time-aligned latent sequence:

```text
z_t = project_z( B( normalize(o_{t+1}) ) )
```

The dynamic window uses `24` latent steps. Static poses use one encoded goal
latent. CEM starts from the complete encoded trajectory with
`max_angle=40 deg`, `initial_std=0.25`, `min_std=0.02`, and temporal noise
correlation `0.9`.

**Verification**

- [x] Backward-map latent reconstruction and trajectory-local CEM completed.
- [x] All eight targets produced finite results and visualizations.
- [x] Encoded scores improved after local CEM.
- [x] No target met the robust coverage criterion.
- [ ] Expert latent encoding guarantees physical executability.
- [ ] This experiment validates a trained Skate-BFM policy.

## Findings

- `without_prior` retrieved locally stable short-horizon joint-motion
  candidates for `6/8` targets under the recorded thresholds.
- `with_prior` reached `0/8` robustly covered targets despite improving
  encoded scores.
- Some successful-looking without-prior videos lose board contact; current
  scoring does not include synchronized board state or foot contact.
- H1 therefore measures tested frozen-BFM motion coverage, not the complete
  capability of BFM0 as a skate motion library.

## Shared Limitations

- Static target velocities are set to zero.
- Dynamic velocities are reconstructed by finite differences at `50 Hz`.
- Human-push source files lack synchronized skateboard state.
- Dynamic scoring uses the confirmed common 23 joint positions.
- Foot contact and full robot-board contact are not part of the H1 success
  criterion.

## Record Index

- Results, tables, plots, and videos:
  [`exp_res.md`](exp_res.md)
- Dated task log:
  [`exp_logs.md`](exp_logs.md)
