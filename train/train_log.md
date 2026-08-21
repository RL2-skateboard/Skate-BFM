# Training Log

The outer level follows the same three experiment stages as `train.md` and
`train_res.md`; dates are short records inside their owning stage.

## Experiment 1: Training Workspace and BFM-HUSKY Integration

### 2026-08-03

Created the training workspace and initial HUSKY integration tools.

### 2026-08-06

Added synchronized project-progress documentation for the main and training branches.

### 2026-08-09

Completed the BFM-HUSKY runtime, MotionLib, viewer, and Base/Skate expert-sampling integration.

## Experiment 2: HUSKY Expert Dataset Collection and Construction

### 2026-08-03

Added HUSKY recording, phase output, failure cleanup, and pose/video export.

### 2026-08-04

Configured the 75-cell command grid, parallel rounds, duration target, and replacement collection.

### 2026-08-13

Completed the 150-minute formal collection and Phase MotionLib conversion/QC.

### 2026-08-14

Built and published the Continuous dataset and organized raw, Phase, and Continuous Hugging Face artifacts.

## Experiment 3: BFM + Skate Expert Training and Semantics Alignment

### 2026-08-10

Connected HUSKY online replay to native BFM updates and validated the initial Base/Skate training boundary.

### 2026-08-11

Validated target encoding, auxiliary rewards, termination, and native full-update dependencies.

### 2026-08-12

Completed fixed-replay stability checks, closed-loop bring-up, and the first 20k baseline.

### 2026-08-14

Parameterized the formal Phase/Continuous trainer and validated expert-conditioned parallel online readiness.

### 2026-08-15

Completed the formal Phase 100k training and matched frozen-checkpoint evaluation, revealing behavioral failure.

### 2026-08-16

Audited and corrected source physics, action subspace, reset/latent alignment, observation scale, and temporal action semantics.

### 2026-08-17

Finalized exact/projected expert-action translation and completed the post-alignment frozen closed-loop preflight.

### 2026-08-18

Unified data and runtime paths under `train/dataset/` and published the named 100k checkpoint with reproducible download/evaluation commands.

### 2026-08-19

Restored and validated the official BFM hard-waist control contract in the HUSKY runtime.

### 2026-08-21

Compared the 20k, 50k, and 100k formal Phase checkpoints on one fixed 80-case held-out Test benchmark and reconstructed their tracking-latent directions.
