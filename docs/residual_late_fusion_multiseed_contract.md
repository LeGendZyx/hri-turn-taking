# Residual late-fusion multi-seed stability contract

## Boundary

This is a version-2, post-outcome stability replay of the already audited
seed-13 saved-probability diagnostic. It adds only seeds 42 and 73 from the
completed exact-eight, 15-epoch A/H/A+H matrices. It does not train or load a
model, create a checkpoint, implement a trainable gated fusion model, test H2,
or replace the seed-13/five-epoch primary analysis. Every artifact remains
`production_eligible=false`, `counted_eligible=false`, and ineligible for a
confirmatory hypothesis verdict.

## Frozen implementation and sources

The runner raw-hash binds the immutable version-1 implementation and the
audited seed-13 aggregate. It admits exactly the repository-owned seed-42 and
seed-73 15-epoch matrix indices and their published matrix aggregates. For each
new seed it also binds and replays the common canonical protocol manifest, its
raw bytes, exact eight sessions, four groups, two corpus exclusions, all 12
A/H/A+H result manifests, and their declared prediction artifacts.

The full historical source validator is run for all 12 units of each new seed.
A+H is integrity-checked only as matrix context. Lambda selection and the
primary comparison load only paired A and H probabilities.

## Unchanged diagnostic

For each outer fold, exactly three inner OOF groups select lambda from
`{0, 0.125, 0.25, 0.5, 1}` using the equal mean of participant-macro strict
mAP. Ties retain the smaller lambda. The frozen formula is:

`sigmoid(logit(H_probability) + lambda * logit(A_probability))`

The historical integrity validator may pre-read outer artifact bytes while it
binds and cold-audits each complete source unit. That integrity pass is not
lambda computation: no decoded outer probability payload, outer label, or
outer metric is passed to or used by the lambda selector. The diagnostic loads
the paired outer payload for formula application and evaluation only after the
three inner OOF payloads have selected lambda. `lambda=0` must remain an exact
float32 copy of H with identical canonical probability bytes. Paired ticks,
labels, VAD, validity, presence, sessions, groups, label configuration,
threshold policy, run identity and source seed must match exactly.

The seed-13 v1 implementation and its original `1e-7` diagnostic gate remain
immutable and hash-bound. The first v2 real-source gate failed closed before
any v2 result was published because one seed-73 A vector differed by exactly
four float32 ULPs (`1.1920928955078125e-7`). Repair round 1 then scanned all
32 seed-42/73 A inner/outer probability artifacts without computing gold
metrics: the maxima were three ULPs (`8.940696716308594e-8`) for seed 42 and
four ULPs (`1.1920928955078125e-7`) for seed 73. The v2 adapter therefore
freezes a dual numerical-invariance gate: every present-participant
frame/bin vector must satisfy both maximum absolute spread `<=2e-7` and
maximum positive-float32 bit/ULP distance `<=4`. Four ULPs pass; five ULPs or
an absolute spread above `2e-7` fail. This repair changes only source
acceptance for machine-precision-equivalent A probabilities; it does not
rewrite probabilities or change lambda, the fusion formula, metrics, success
criteria, or v1.

The seed-level descriptive criteria are unchanged from version 1: at least
`+0.02` participant-macro mAP, positive differences in at least three of four
outer folds, and shift-prediction and overlap-resolution AP differences no
worse than `-0.02`. They were specified after historical outer outcomes were
available, so they are descriptive rather than preregistered or confirmatory.

## Three-seed report

The final aggregate binds the audited seed-13 v1 aggregate and the newly
replayed seed-42/73 aggregates. It reports every seed plus arithmetic mean and
sample standard deviation across seeds 13/42/73 for selected fusion, matched H
control, and their differences. Metrics include frame mAP, participant-macro
mAP, shift-prediction AP, overlap-resolution AP, robot-entry AP, Next Speaker
participant-macro one-vs-rest AP and Next Speaker top-1 accuracy. No confidence
interval, p-value, statistical-significance test, H1 revision or H2 inference
is produced.

## Publication and audit

`run` is create-once. It writes a sibling stage, replay-audits the complete
file set, atomically renames it within one parent, and replay-audits again.
Post-rename rollback deletes only the directory identity owned by that run; a
missing or replaced path is never deleted. Existing output is never
overwritten.

`audit` independently revalidates both 12-unit source matrices, recomputes
inner-only selection and all fused float32 arrays, replays fold/pooled metrics
from saved probabilities, reconstructs the three-seed mean/sample-SD report,
and requires canonical equality of all protocol, fold and aggregate manifests.
Unexpected files or any source/output byte change fail closed.

## Commands

```powershell
python tools/run_residual_late_fusion_multiseed.py run
python tools/run_residual_late_fusion_multiseed.py audit
```

The real replay must not start until focused tests, scoped Ruff and an
independent read-only implementation review pass.
