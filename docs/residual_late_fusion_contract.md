# Residual late-fusion diagnostic contract

## Status and scientific boundary

This is an isolated **post-outcome exploratory diagnostic** over already saved
outer probabilities. It does not train or load a model, create a checkpoint, or
implement a deployable/trainable gated-fusion architecture. Every artifact is
`production_eligible=false`, `counted_eligible=false`, and ineligible for a
confirmatory hypothesis verdict. Its threshold check is descriptive because the
historical outer outcomes existed before this diagnostic was specified.

The diagnostic must not replace the seed-13 five-epoch primary matrix, change H1
or H2, or be described as a new model. H2 is not tested: no policy replay is run.

## Immutable source

The only admitted source is the exact-eight, four-group, seed-13 **15-epoch**
primary A/H/A+H matrix. Intake validates:

- the exact session, group, exclusion, condition and four-fold identities;
- all 12 matrix result manifests and their source hashes;
- the canonical absolute `protocol_directory/run_manifest.json`;
- the frozen canonical-JSON run-manifest SHA-256, a separate raw-file binding,
  and the exact 15-epoch protocol contract;
- paired A/H ticks, labels, VAD, validity, presence, sessions, groups, label
  configuration, threshold policy and run-manifest identity; and
- participant symmetry of every A probability array at maximum absolute
  difference `1e-7`.

Five-epoch matrices, relabelled indices, incomplete grids, substituted protocol
directories, changed manifests and unpaired saved probabilities fail closed.
Historical A+H is validated only as matrix context and is not loaded for lambda
selection or the primary comparison.

## Frozen diagnostic

For paired probabilities strictly inside `(0, 1)`, the diagnostic computes

`sigmoid(logit(H) + lambda * logit(A))`

for `lambda in {0, 0.125, 0.25, 0.5, 1}`. Lambda is selected independently in
each outer fold from exactly three inner OOF groups by the equal mean of their
participant-macro strict mAP values. Ties select the smaller lambda. The source
validator may read all historical artifacts for integrity, but outer labels and
outer probability payloads are never passed to or used by lambda computation.

`lambda=0` is the matched control and must be an exact float32 array copy of H;
its canonical probability bytes must have the same SHA-256. Outer labels are
used only after lambda selection for saved-probability metric replay.

The descriptive threshold check asks for participant-macro mAP improvement of
at least 0.02, a positive fold difference in at least three of four folds, and
shift-prediction and overlap-resolution AP changes no worse than -0.02. Passing
or failing these thresholds is not a confirmatory or hypothesis-level verdict.

## Publication and audit

`run` is a create-once transaction. It publishes to a sibling staging directory,
replay-audits the complete file set before rename, atomically renames within the
same parent, and replay-audits again. Artifact contents and relative bindings are
independent of the staging directory name. If the post-rename audit fails, only
the output directory renamed by that invocation is rolled back only if its
rename-stable directory identity still matches the staging directory. A missing
or concurrently replaced path is never deleted and is reported as an ownership
change. Existing output is never overwritten.

The initial `run` source gate still fully validates all 12 historical result
units. The expensive real-data replay is prepared once per condition and reused
for that condition's four folds, using the existing process-local sealed
`prepare_outer_condition` authority. The resulting fully validated `SourceMatrix`
retains deduplicated source, matrix-index, protocol, result-manifest and declared
role-artifact snapshots. Both staging and post-rename audits freshly re-hash those
snapshots at entry and exit, so a concurrent source change fails closed without
repeating real-data assembly. The `run` CLI prints the already completed
post-rename audit receipt; it does not start a fourth full source validation.

`audit` rebinds the immutable source, redoes inner-only lambda selection,
recomputes every fused float32 array, replays fold and pooled metrics from the
saved probabilities, and requires canonical equality of protocol, fold and
aggregate manifests. It is deliberately independent: each separate `audit`
command performs a fresh full 12-unit source validation before replay. Unexpected
files, source mutations or byte changes fail closed.

## Commands

```powershell
python tools/run_residual_late_fusion.py run `
  --source outputs/thesis_posthoc_15epoch_20260817/matrix_seed13_v1 `
  --output outputs/residual_late_fusion_posthoc_20260818

python tools/run_residual_late_fusion.py audit `
  --output outputs/residual_late_fusion_posthoc_20260818
```

Running the real diagnostic requires an explicit owner decision after code and
fixture review. The implementation/test task itself must not run it.
