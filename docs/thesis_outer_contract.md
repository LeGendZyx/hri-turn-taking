# Deadline MSc outer-evaluation contract

## Purpose and status

This route is the shortest honest path from one audited, train-ready real-data
source to dissertation-ready **A / H / A+H** results. It is deliberately a
separate deadline runner, not a production-backend promotion.

- Data are real only. Synthetic or smoke sources fail closed.
- The matrix is `primary × seed 13 × {A, H, A+H} × 4 outer folds` (12 units).
- All artifacts say `production_eligible=false` and `counted_eligible=false`.
- The existing outer orchestrator therefore seals each unit under
  `result_role=backend_contract_smoke`; it must never be described as a
  repository-counted production experiment.
- If all gates pass, the pooled, group-disjoint real-data result is suitable as
  the main empirical result of this MSc dissertation, provided the thesis calls
  it a **deadline non-production nested-LOGO evaluation** and reports the
  limitations below. "Non-counted" is a repository governance status, not a
  claim that the saved real-data measurements are synthetic.

This contract does not add ASD, Gaze, facial features, D3 sensitivity, a second
seed, or a production registry entry.

## Deadline corpus scope (owner decision 2026-08-16)

The immutable ten-scene scientific assets remain in place, but this deadline
run uses an exact eight-scene, four-group view containing **64,299 rows**. The
runner structurally opens the full source and fails unless both excluded scenes
exist, then removes them before target readiness, feature masking, statistics,
training, or evaluation:

- `05_15__4`: `no A speech in Transcript; pending owner manual review`;
- `05_15__5`: `A/R extremely sparse; pending owner manual review`.

The included scenes are exactly `03_20__4`, `03_26__3`, `05_14__2`,
`05_14__3`, `05_14__4`, `05_15__2`, `05_15__3`, and `05_15__6`. No other
exclusion is accepted. Split, run, per-unit result, benchmark, matrix-index and
aggregate artifacts bind the two exclusions and their reasons. This is a
target-QC decision made before model-outcome inspection, not an outcome-based
scene deletion. The original ten-scene files are neither moved nor rewritten.

## Frozen scientific design

The feature conditions are exact and paired:

| Condition | Feature set | Meaning |
|---|---|---|
| `A` | `audio_only` | causal energy, F0 and global voiced evidence |
| `H` | `head_pose_only` | causal held head-pose evidence and its validity/quality |
| `A+H` | `audio_plus_head_pose` | the union of the same A and H inputs |

The same seed, group folds, label configuration, trainer configuration and model
configuration are used for all three conditions. With group ids sorted by the
frozen protocol, the held-out groups are:

| Fold | Outer-test group |
|---|---|
| `outer_01` | `03_20` |
| `outer_02` | `03_26` |
| `outer_03` | `05_14` |
| `outer_04` | `05_15` |

For each outer fold, the other three groups form outer-train. Three inner LOGO
fits each hold out one of those groups. Inner-validation strict macro per-bin
mAP chooses an epoch in each inner fold; the median of the three epochs is then
used for one fresh final fit over all outer-train sessions. No inner checkpoint
warms the final model.

The outer runner, not the backend, fits and supplies each training partition's
normalizer and per-bin positive weights. The backend copies and validates those
statistics and fails if they or the supplied tensors change; it never calls
`FeatureNormalizer.fit` itself. `aux_vad_weight` is exactly `0.0`.

The optional `d3_le80` membership is recomputed within the eight-scene scope:
because `05_15__5` is excluded, it contains five scenes (still all four groups),
not the former six-scene set.

## Leakage boundary and saved predictions

Training and selection never include the outer group. The final checkpoint
contains only outer-train state. Outer probability generation consumes only
robot-observable `features` and the static `present` mask; replacing outer
labels, gold VAD or the evaluation-valid mask cannot change probabilities.

Each unit performs one full-session outer prediction pass and saves it before
metrics are aggregated. Its prediction artifact contains:

- exact session/group identity and contiguous real `tick_25hz` values;
- participant-wise probabilities with shape `time × participant × 4 bins`;
- labels, VAD and validity masks explicitly as **evaluation-only gold**, needed
  to recompute pooled frame and event metrics from the saved pass;
- checkpoint, run-manifest, source-report and real-session-authority hashes;
- the exact 25 Hz label configuration and non-production/non-counted markers.

Aggregation strictly reloads the 12 result manifests, predictions and metrics,
checks their hashes and condition/fold/seed identities, and then pools all four
folds' saved out-of-fold examples. It does not instantiate a model. The main
frame mAP is therefore computed on pooled OOF predictions, not by averaging four
fold mAP values. Event pairs are likewise reconstructed from saved probabilities
and evaluation-only VAD and pooled across folds. Four group-level summaries stay
available for showing dispersion.

The fixed threshold `0.5` is a diagnostic classification/event threshold only.
It is not tuned on the outer data and is not a calibrated robot policy claim.

## No-overwrite and single-writer rule

Freeze the protocol before any matrix run. Every protocol, benchmark, unit,
matrix-index and aggregate path is create-once. The runner preflights all 12
deterministic unit paths before the first fit; it never deletes, resumes,
auto-suffixes or silently reruns a partial matrix.

Use a trusted local **single-user, single-writer** boundary. There is no atomic
cross-process run ledger. A concurrent replacement race therefore remains a
documented non-production blocker. If a run stops after creating part of its
output, retain it for diagnosis and use a newly named output root only after
review; do not treat a partial tree as resumable. Apply the owner's maximum of
three repair-and-retest rounds to the same reproducible blocker.

## Same-process prepared-condition optimisation

The deadline `run-matrix` command performs the expensive audited source replay
exactly once for each of A, H and A+H, after the whole-matrix no-overwrite
preflight and before any fit. Each replay produces a process-local,
factory-sealed `PreparedOuterCondition`; its audited frame/report build the
backend real-session authority, and its condition-masked sessions are reused by
that condition's four outer folds. This reduces the former 39 full replays
(three authority loads plus three loads per unit) to three without creating a
persisted cache.

The prepared object is accepted only in the process that issued it and is
strictly rebound on every use to the resolved protocol directory, split/run
manifest hashes, frozen source path, condition, readiness payload, D3 records
and complete session-tensor hash. A copied or hand-built object fails closed.
Every unit still re-hashes all frozen source bindings at its existing gates,
and both the staging and final result validators still execute. Their immutable
source/artifact snapshots and the held final snapshot are unchanged. Calling
`run_outer_unit(..., prepared=None)` retains the original fresh-replay behavior;
the optimisation is limited to the deadline matrix process and is not a general
production-cache claim.

## Commands

Run from the repository root in PowerShell. Replace angle-bracketed paths with
the already audited, train-ready real source and new output locations.

```powershell
# 1. Optional but recommended: time one predeclared inner fit.
# It excludes outer_01's test group and prints duration/memory only.
.\.venv\Scripts\python.exe tools\run_thesis_outer.py benchmark-inner `
  --data <train-ready-real-source> `
  --output <new-benchmark-directory> `
  --condition A+H `
  --seed 13

# 2. Freeze exact eight-scene source hashes, exclusions, splits, A/H/A+H,
# model/trainer config and budget. Exclusion is code-owned; there is no arbitrary flag.
.\.venv\Scripts\python.exe tools\run_thesis_outer.py freeze `
  --data <train-ready-real-source> `
  --output <new-frozen-protocol-directory> `
  --max-epochs <chosen-positive-integer>

# 3. Execute exactly 12 paired primary units. This command prints no scores.
.\.venv\Scripts\python.exe tools\run_thesis_outer.py run-matrix `
  --protocol <frozen-protocol-directory> `
  --data <the-same-train-ready-real-source> `
  --output <new-matrix-directory> `
  --seed 13

# 4. Only after all 12 units exist, pool the saved outer predictions.
.\.venv\Scripts\python.exe tools\run_thesis_outer.py aggregate `
  --run-dir <matrix-directory> `
  --output <new-aggregate-json-path>
```

Choose `max_epochs` once, using the timing benchmark and the remaining deadline;
changing it requires a new frozen protocol. The benchmark does not call outer
evaluation and its console JSON contains only elapsed time and memory fields.

## What may be claimed in the dissertation

Acceptable wording after a clean 12-unit run:

> We evaluated audio, head-pose, and fused audio-plus-head-pose conditions on
> four participant-group-disjoint nested leave-one-group-out folds using the
> same seed and split assignments. Reported pooled metrics were recomputed from
> one saved out-of-fold prediction pass per condition and fold.

Also state that the implementation is a deadline, non-production evaluation;
the local single-writer assumption, one-seed scope, fixed 0.5 diagnostic
threshold, lack of long-term annotation retest, and deferred D3/ASD/Gaze routes
are limitations. Do not claim production readiness, registry-counted status,
causal ASD evidence, statistical significance from one seed, or a physical
Furhat result.

Future publication-grade work would need, at minimum, a reviewed production
backend/evidence path, an atomic cross-process run ledger, and the separately
predeclared robustness/replication studies. Those gates are intentionally not
part of this four-day MSc route.
