# Previous-VAD oracle contract (post-hoc, non-deployable)

Status: isolated implementation and tests only. A real O0/O1 matrix is not a
result until all eight units finish and the saved-probability aggregate passes
the independent audit described below.

## Question and claim boundary

This diagnostic asks whether the model architecture can benefit from perfect
knowledge of who was speaking exactly one 25 Hz tick (40 ms) earlier. It does
not test a robot-observable VAD/ASD estimator. O1 consumes a gold-tier signal,
so every protocol, checkpoint, prediction, result, matrix and aggregate fixes:

- `posthoc_exploratory = true`;
- `oracle_gold_input = true`;
- `robot_observable = false`;
- `deployable = false`;
- `production_eligible = false`;
- `counted_eligible = false`; and
- `formal_result_eligible = false`.

The output is an architectural upper-bound diagnostic. It cannot establish H1,
H2, deployed ASD/VAD performance, live robot latency, or a production feature.

## Frozen matched design

The source is the exact eight owner-released scenes: 64,299 rows across four
groups, excluding `05_15__4` and `05_15__5`. The source must independently pass
the real-data `head_pose_only` readiness audit. The Oracle protocol is also
bound to the existing exact-eight A/H/A+H protocol with `max_epochs=15`; this
prevents a look-alike corpus or training configuration from becoming authority.

Both conditions use the same isolated 37-dimensional schema, model, optimizer,
nested leave-one-group-out folds, seed 13, maximum 15 epochs, zero auxiliary-VAD
loss, epoch-selection rule, and per-fold initial state:

- **O0:** H-only inputs plus four all-zero previous-VAD channels.
- **O1:** H-only inputs plus participant-wise gold VAD from exactly the previous
  tick in the same session.

At session tick 0, every previous-VAD channel is zero. No value may cross a
session boundary. Absent participant slots are zero. The previous-VAD channel
uses an identity transform (`mean=0`, `std=1`) in both conditions, while all H
and global statistics are fitted only on the current training groups. Thus O0
and O1 differ only in the value of the added channel, not in schema, learned
normalisation statistics, or initial weights.

## Causality and leakage gates

For source VAD `v`, the only permitted Oracle input is:

```text
oracle[session, 0, participant] = 0
oracle[session, t, participant] = v[session, t - 1, participant], t >= 1
```

Changing VAD at tick `t` may first change an O1 model input or causal prediction
at tick `t+1`; it must not change any input or prediction at or before `t`.
Changing a later suffix must not alter a frozen prefix. Current/future gold VAD,
future-bin labels, event labels and held-out outcomes are never passed to the
model. The saved sidecar binds inference inputs, the underlying Oracle-source
VAD, and evaluation targets with separate SHA-256 values.

This is causal only with respect to the *timing transformation of an oracle
signal*. The gold signal itself is not available to a deployed robot, which is
why O1 remains non-deployable regardless of its score.

## Nested-LOGO execution and publication

There are exactly eight units: O0/O1 × four outer groups. Within each outer
unit, each of the other three groups is used once for inner validation; the
median selected epoch is used for a fresh final fit on all three outer-training
groups. Every inner and final fit reconstructs the model from the same frozen
seed-13 state. Outer probabilities are saved once and aggregation reads those
arrays only—it does not load a checkpoint or run inference.

Protocols, unit bundles, matrices and aggregates are create-once. Publication
uses staging, full pre-publication audit, no-clobber publication, and a second
audit at the final path. Directory artifacts use rename; the single aggregate
uses a same-volume hard link so an existing file is never replaced. A failed
final audit removes the new destination before best-effort stage cleanup.
File-set, canonical JSON,
size, SHA-256, schema, split, normalizer, initial-state, history, median-epoch,
probability and recomputed-metric checks fail closed. Test-fixture artifacts use
different kinds and roles and are rejected by the real audit API.

## Commands

Use new output paths for every invocation. Do not start `run` until code review
and the focused test suite have passed.

```powershell
& .\.venv\Scripts\python.exe tools\run_previous_vad_oracle.py freeze `
  data\processed\formal_a_h_ah_v1 `
  outputs\thesis_posthoc_15epoch_20260817\protocol_seed13_v1 `
  outputs\previous_vad_oracle\protocol_v1

& .\.venv\Scripts\python.exe tools\run_previous_vad_oracle.py run `
  outputs\previous_vad_oracle\protocol_v1 `
  data\processed\formal_a_h_ah_v1 `
  outputs\previous_vad_oracle\matrix_v1

& .\.venv\Scripts\python.exe tools\run_previous_vad_oracle.py audit `
  outputs\previous_vad_oracle\matrix_v1 `
  data\processed\formal_a_h_ah_v1

& .\.venv\Scripts\python.exe tools\run_previous_vad_oracle.py aggregate `
  outputs\previous_vad_oracle\matrix_v1 `
  data\processed\formal_a_h_ah_v1 `
  outputs\previous_vad_oracle\aggregate_v1.json

& .\.venv\Scripts\python.exe tools\run_previous_vad_oracle.py audit-aggregate `
  outputs\previous_vad_oracle\aggregate_v1.json `
  outputs\previous_vad_oracle\matrix_v1 `
  data\processed\formal_a_h_ah_v1
```

Run `tests/test_oracle_prev_vad.py` and scoped Ruff first. Based on the measured
15-epoch H-only four-unit runtime (4,432.90 s), eight matched O0/O1 units are
expected to take about 8,866 s (2 h 28 min) on the same CPU, with normal run-to-
run variation. Allow roughly 3 hours for real training plus final audit, and
roughly 570 MiB for the matrix.
