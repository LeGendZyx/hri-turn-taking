# H2 matched-replay contract (post-hoc, non-counted)

Status: implemented as an isolated contract; no real replay result has been run or
reported by this change.

## Purpose and claim boundary

This experiment asks a narrower, matched offline question than live H2: on the
same held-out robot-entry opportunities, does the saved A+H VAP policy trigger
earlier or miss fewer demonstrator entries than a causal acoustic-silence policy?
It cannot establish live robot latency because it does not simulate action
feedback, Furhat/TTS startup, far-field robot-ear audio, or a deployed perception
stack. Every output is therefore fixed to:

- `result_role = posthoc_exploratory_h2_matched_replay`;
- `production_eligible = false`;
- `counted_eligible = false`;
- `h2_established = false`.

The primary model source is the saved **A+H, seed 13, max_epochs=5** outer
matrix. The matching **A+H, seed 13, max_epochs=15** matrix is a post-outcome
sensitivity analysis. No model is retrained and neither historical runner nor
historical artifact is edited.

## Four create-once phases

1. `freeze` strictly validates the two existing matrix indexes, their exact-eight
   nested-LOGO split, four A+H result manifests, twelve inner-OOF artifacts per
   budget, their twelve paired inner checkpoints, four final checkpoints, and
   four outer artifacts per budget. Each inner checkpoint must train on exactly
   the other two inner groups, exclude its validation and outer-test sessions,
   and match the OOF artifact's `checkpoint_sha256`. Each final checkpoint must
   train on exactly the outer-training sessions and match the outer artifact.
   Freeze then projects each outer artifact to a new view containing only
   session/group identity, 25 Hz ticks, probabilities, and the static `present`
   mask. The views are bound by a self-contained `projection_seal.json` inside
   the freeze directory. Although the legacy source JSON embeds
   labels/VAD/validity, none of those fields crosses into the trace phase.
2. `calibrate-and-seal-traces` may read inner OOF gold, but never outer gold. It
   fits calibration and writes all outer policy traces before evaluation.
3. `evaluate` opens the hash-bound outer artifacts only after the trace seal
   exists. It derives opportunity labels and evaluates both policies.
4. `audit` does more than re-hash. For the trace phase it regenerates every CSV
   byte-for-byte from the sealed calibration, the freeze-owned gold-free view,
   and the E-only acoustic descendants. For evaluation it reopens the frozen
   outer gold, regenerates the event CSV and metrics from the sealed traces, and
   requires canonical bytes and fields to match. It rejects promotion or
   mutation. Every output directory is create-once; reruns require a new path.

After freeze, the original outer prediction path and hash are retained only as
opaque provenance. Trace generation and trace audit neither open nor hash that
source; they validate the freeze-owned projection bytes instead. The evaluator
is the first later phase permitted to dereference the original outer artifact.

The intended process split is important: do not combine phase 2 and phase 3 in a
new convenience function.

## E-only sealed acoustic loader

The replay loader reads only the already published descendants under
`data/processed/acoustic_features`: feature/source CSV, evidence, pending
manifest, independent review receipt, and reviewed assembly-v3 manifest. It
cross-checks those bytes against each frozen run-readiness source artifact and
the assembled provenance acoustic descendant.

It intentionally does **not** call `load_reviewed_acoustic_scene`, decode a WAV,
recompute features, or dereference a path on `F:`. Historical upstream raw paths
inside evidence remain opaque provenance strings. Each trace manifest records
`raw_source_replay = skipped` and `F_drive_paths_dereferenced = false`.

The acoustic feature order is frozen as:

1. `array_energy`;
2. `voiced`;
3. `f0_proxy`.

`voiced` remains a normalized-autocorrelation voicing flag, not complete speech
VAD. The replacement detector below is therefore calibrated explicitly instead
of silently treating `voiced` as speech.

## Inner-only calibration

For each outer fold and epoch budget, the three existing inner validation groups
form group-OOF calibration data. The held-out outer group cannot enter any fit or
selection step.

### Current-frame speech detector

- target: current-frame `any(participant VAD)` from the human-reviewed-
  Transcript-derived VAD, inner OOF only;
- inputs: `array_energy`, `voiced`, `f0_proxy`;
- standardisation: `sklearn.preprocessing.StandardScaler`;
- classifier: L2 logistic regression, `C=1.0`, `class_weight="balanced"`,
  `solver="lbfgs"`, `l1_ratio=0.0`, `max_iter=1000`, `tol=1e-4`, seed 13;
- cross-fitting: fit on two inner groups and predict the third, repeated three
  times;
- probability threshold grid: 0.05, 0.10, ..., 0.95;
- selection: maximum frame balanced accuracy, then speech recall, then distance
  to 0.5, then the higher threshold;
- final detector: refit on all three outer-train groups and serialize scaler,
  coefficient, and intercept for outer trace generation.

### Reactive policy

Candidate silent runs are 5, 10, 15, 20, 25, 30, and 38 frames. A detected
speech frame resets the run; exactly one rising trigger occurs when the selected
run length is first reached. Selection maximises inner robot-entry balanced
accuracy, then minimises false-entry rate, maximises recall, minimises mean
true-positive latency, and finally prefers the shorter run.

### Predictive policy

The predictive route is the P1 `DemonstrationPolicy`, not the floor-holder
policy. No gold or inferred floor-holder is an input. `p_now` and `p_future` are
the existing duration-weighted VAP readouts. Both threshold grids are
0.05, 0.10, ..., 1.00, with `min_consecutive_frames=3` and
`refractory_frames=25`.

Only pairs whose inner false-entry rate is no greater than the selected reactive
policy are feasible. Selection then maximises recall, minimises mean
true-positive latency, minimises false-entry rate, and prefers higher thresholds.

## Matched evaluation

The evaluator reuses the existing shift/hold extractor with 25 Hz data,
`min_gap_s=0.2`, `max_gap_s=2.0`, and `context_s=0.4`. It excludes opportunities
whose previous speaker is P1 and requires full gap validity. Each comparison
must contain the identical **303 unique event IDs: 111 positive robot entries and
192 negatives**.

- Primary decision window: from 0.4 s before gap start to gap end (half-open).
- Sensitivity window: gap frames only.

The event CSV retains policy, budget, fold, window, event ID, decision, and first
trigger latency relative to gap start. Negative latency means a predictive lead.
Metrics include TP/FP/FN/TN, balanced accuracy, positive recall, negative
false-entry rate, and mean latency among detected positive opportunities.

The prior voiced-only diagnostic event CSV is checked by its known SHA-256 and
event-ID set when present, but it is explicitly non-authoritative and is not used
as a policy, label source, or calibration source.

## Commands

Use new output directories for every invocation:

```powershell
python tools/run_h2_matched_replay.py freeze `
  --output outputs/h2_matched_replay/freeze_v1

python tools/run_h2_matched_replay.py calibrate-and-seal-traces `
  --freeze outputs/h2_matched_replay/freeze_v1 `
  --output outputs/h2_matched_replay/traces_v1

python tools/run_h2_matched_replay.py evaluate `
  --freeze outputs/h2_matched_replay/freeze_v1 `
  --traces outputs/h2_matched_replay/traces_v1 `
  --output outputs/h2_matched_replay/evaluation_v1

python tools/run_h2_matched_replay.py audit `
  --freeze outputs/h2_matched_replay/freeze_v1 `
  --traces outputs/h2_matched_replay/traces_v1 `
  --evaluation outputs/h2_matched_replay/evaluation_v1
```

Run `tests/test_h2_matched_replay.py` before any real phase. The fixtures verify
gold replacement invariance of the trace input, group-disjoint detector fits,
paired checkpoint scopes, prefix causality and chunk equivalence, the
303/111/192 support gate, absence of a floor-holder input, E-only/F-drive
boundaries, original-outer zero-dereference during trace generation, the full
freeze→trace→evaluate→audit chain, canonical replay mutation detection, and
no-clobber publication.
