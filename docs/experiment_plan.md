# Experiment Plan: Real-data Participant-wise VAP First

> Status: counted outer-test protocol frozen before outcome inspection, 2026-08-15. This plan follows
> [`PROJECT_SCOPE.md`](PROJECT_SCOPE.md) and [`methodology.md`](methodology.md).
> The existing CLI/trainer remains development scaffolding because it selects and reports the
> same inner validation split. The separate outer-experiment orchestrator now freezes and
> contract-tests the protocol below, but deliberately has no registered production training
> backend. Environment capture and the formal run index are also not implemented. Therefore no
> real fit or counted result has yet been run, and its test backend can issue smoke artifacts only.

## 1. Required outcome and stopping rule

The minimum dissertation result is a reproducible, group-disjoint real-data comparison of a
single participant-wise four-bin VAP tensor. Under the 2026-08-16 four-day route, the first
counted matrix is paired `A / H / A+H`; ASD, Gaze, Facial/MAR, GVAP and physical Furhat work
are deferred rather than made prerequisites.

The core study is complete only when:

- all counted sessions have confirmed identity mappings and trainable provenance;
- selected inputs are real, causal, and robot-observable; gold VAD is target/evaluation only;
- every compared condition uses paired group folds and paired random seeds;
- frame, event, and policy metrics are saved together with class/event supports;
- failed folds, missing classes, and excluded sessions are reported rather than hidden;
- the complete run can be reconstructed from local manifests and checkpoints.

Synthetic training may validate software and hyperparameters, but cannot satisfy this outcome.

## 2. Present blocker snapshot

As of the 2026-07-28 upstream snapshot (verified locally 2026-08-03), HPE
head-pose D1 and participant-mapping D2 are complete for 21 formal scenes
across five recording groups, all human-confirmed, and **14 scenes covering
all five groups publish a usable `6cam` D3 offset** (`03_20__{4,5}`,
`03_24_1__{2,3}`, `03_26__3`, `05_14__{2..5}`, `05_15__{2..6}`). 9 of the 14
are `degraded` and 8 exceed the ±40 ms target (max ±104.4 ms), so the §3.1
synchronization exclusion/sensitivity decision is now live. The 7 scenes
without a physical bridge (`03_20__{2,3,6}`, `03_26__{4,6,7}`, `05_14__6`)
remain on local six-camera time. The roster and per-scene offsets are
recorded in `configs/hpe_delivery_inventory.json`.

The repository's only older processed real artifact is still
`05_15__2` / `05_15`. Its provenance is `trainable: false`, lists 23
placeholders, and contains only real head-yaw, pose-validity, and pose-quality
channels. Gold VAD, ASD, MAR, gaze, gazed-at, energy, F0 proxy,
and DoA are placeholders. Its existing sidecar has no offset-quality summary
and records `6cam = -6.3225 s`, superseded by the republished `-3.7621 s`
(degraded, ±36.2 ms), so the table cannot be reproduced from the current
delivery and must be regenerated via current `vap-intake`. Its legacy sidecar
also lacks the new explicit `data_kind` and complete D3 provenance contract,
so the current loader rejects it at the structural gate before target/feature
readiness is considered.

Therefore:

- the wider upstream synchronization inventory spans five groups, while the accepted
  complete-case corpus now contains ten scenes in four groups; no counted run exists until
  human-reviewed-Transcript-derived VAD, formally built and reviewed acoustics, and head-pose inputs are merged
  and audited;
- the training gate must continue to reject the stale table;
- `--allow-untrainable` may be used only for a labelled engineering smoke test and never for
  a result table; real-data partitions are always group-disjoint;
- the optional GVAP/Furhat gate remains **CLOSED**.

## 3. Data acceptance and corpus manifest

### 3.1 Session acceptance

A session may enter the primary experiment only if all of the following are true:

1. `mapping.json.review_status == confirmed` and the slot map is P1=R, P2=A, P3=B, P4=C;
2. the processed table and provenance bind to the same `<group>__<scenario>` delivery;
3. `trainable == true` and every feature used by that condition is marked real rather than a
   zero-filled placeholder;
4. per-participant human-reviewed-Transcript-derived VAD is finite, binary, aligned at 25 fps,
   and zero on absent slots;
5. the full two-second future horizon and per-frame masks validate;
6. source timestamps and extraction windows satisfy the strict causal contract;
7. the D3 offset-quality record is structurally complete, self-consistent, and has a finite
   error bound; the prespecified primary/sensitivity handling in §4 must be applied.

The ±40 ms value remains the engineering target, not the primary exclusion rule. The
primary, paired ≤80 ms sensitivity, and descriptive ≤40 ms analyses were frozen before
outcome inspection in §4. A session with no published finite error bound fails the structural
D3 gate; all ten currently accepted scenes have one.

### 3.2 Dataset audit before training

For each accepted session, calculate and save:

- duration, participant count, frame count, and valid future-horizon frame count;
- fraction valid and mean quality for every participant feature;
- speaking prevalence per participant and per future bin;
- number/duration of silence gaps, holds, shifts, overlaps, backchannels, competitive overlaps,
  and robot-entry opportunities;
- audio source identity (the sample-replayed equal-weight `master_aligned.wav` mix, explicitly
  a close-talk surrogate rather than robot-ear audio), sample rate, clipping, missing intervals,
  and extractor version;
- synchronization offset, quality status, error bound, and the action taken when degraded;
- exact real and placeholder column lists.

The corpus manifest records SHA-256 hashes of processed tables and provenance files. Raw media
remain local and ignored by git.

## 4. Required split roles and frozen protocol

The accepted complete-case corpus has exactly four eligible recording groups. Section 5
records the nested leave-one-group-out protocol frozen before outcome inspection. A change to
group count, session membership, or resampling scheme requires a new versioned protocol and
cannot inherit this protocol's counted status.

Every formal protocol must distinguish three roles:

- **train:** fits model weights, normalisation, and class weights;
- **inner validation:** selects configuration, epoch/checkpoint, calibration, and thresholds;
- **outer test:** group-disjoint data that participate in none of those choices and are scored
  only after selection is complete.

If the available groups cannot support those roles without leakage, executions are explicitly
development/exploratory or smoke-only and are not formal counted results.

No scenario video from a recording group may cross an inner or outer boundary. Normalisation,
class weights, acoustic calibration, early stopping, and decision thresholds are fitted inside
the relevant training/inner-validation partition.

The outer folds are generated once and reused by every feature condition. The split manifest
stores exact group and session identifiers for every inner/outer partition; no group is split
or reassigned using feature prevalence, event support, or model outcomes.

If feature delivery differs by modality, the primary paired comparison uses the intersection
of eligible groups/sessions. A larger modality-specific run may be reported only as an
explicitly unpaired supplemental analysis.

### Frozen D3 uncertainty policy (2026-08-15)

The primary analysis retains all ten scenes whose delivered `6cam` bridge is
self-checked, using the delivered central offset. It never silently equates a
`degraded` bridge with exact synchronization. Before any model outcome was
observed, two sensitivity analyses were fixed:

1. repeat the paired protocol on the four-group subset whose D3 error bound is
   at most 80 ms (two 25 Hz frames): `03_20__4`, `03_26__3`, `05_14__2`,
   `05_15__2`, `05_15__3`, and `05_15__5`;
2. for every outer-test scene in a condition containing head pose, evaluate the
   already selected checkpoint with the pose stream shifted by both
   `-error_bound_ms` and `+error_bound_ms`, without recalibration or choosing a
   favourable direction. Report the central score and full perturbation range.

Both the ten-scene primary analysis and this six-scene `d3_le80` analysis are executable
four-group split/run variants in the same versioned protocol. They reuse the same condition
matrix and seeds; the sensitivity subset is not a result-dependent exclusion. The strict
target-quality subset (error bound at most 40 ms) contains only
`03_20__4`, `03_26__3`, and `05_15__2`; it is descriptive, not a separate
formal model comparison. No scene is excluded after model performance is seen.
The per-scene bounds used for the prespecified pose shifts are frozen in the
split manifest (33.5, 33.5, 69.6, 101.3, 104.4, 36.2, 56.4, 87.2, 43.7,
and 101.3 ms in the primary-session order). Past-only central/±bound pose
reassembly is implemented and exercised inside the read-only production-evidence
foundation. That foundation is permanently non-counted and has no publisher; a
reviewed production integration must replay the same policy before any
head-pose condition is counted.

## 5. Frozen four-group outer protocol and model defaults

The eligible corpus has four recording groups. Before any real model outcome
was inspected, the counted protocol was fixed as nested group leave-one-out:
each group is the outer test set once; the remaining three groups form three
inner leave-one-group-out folds. For each condition and seed, the median of the
three selected inner epochs is fixed, the model is reinitialised and trained on
all three outer-training groups for exactly that many epochs, and only then is
the untouched outer group scored. Normalisation and class weights are refit on
the three outer-training groups. Calibration and thresholds use inner
out-of-fold predictions only. The paired seeds are `13`, `42`, and `73`.

Every feature condition uses identical outer/inner memberships and seeds.
Outer labels, features, prevalence, event support, or metrics may not influence
epoch, threshold, hyperparameter, or exclusion decisions.

`src/hri_turn_taking/vap/outer_experiment.py` owns the versioned split/run
manifests and the fail-closed execution state machine. A protocol bundle is not
a result. The currently registered backend is code-owned
`backend_contract_smoke`; it always records `counted_experiment_complete=false`
and cannot self-promote by claiming another ID or role. A future production
backend must be added to the source-controlled registry and independently
reviewed. Even then, `result_role=counted_outer_test` remains gated on all three
inner fits, pooled inner-OOF calibration, same-seed from-scratch fixed-epoch fit,
one untouched outer evaluation, environment capture, and a formal run index.
The current orchestrator revalidates source/readiness/D3 bindings, recomputes
training-only normalisation and class-weight records, rechecks source/checkpoint
hashes, and publishes no-clobber artifacts transactionally, but it cannot yet
issue a counted result.

The current lightweight implementation defaults are listed for reproducibility, not frozen as
the counted-run protocol:

- frame rate: 25 fps;
- history/training window: 250 frames (10 s), stride 125 frames for training;
- prediction bins: 0.2, 0.6, 1.2, and 2.0 s edges; active ratio 0.5;
- model: participant-structured Transformer, `d_model=64`, two blocks, four heads,
  dropout `0.1`;
- optimiser: AdamW, learning rate `3e-4`, weight decay `1e-4` only on matrix weights;
- scheduler: 10% linear warm-up then cosine decay; gradient clip `1.0`;
- batch size: 16 on CPU unless memory forces a documented change;
- development selection metric: inner-validation frame mAP;
- auxiliary setting: `aux_vad_weight=0.0` for the first formal run (owner-approved
  2026-08-03; the implementation default is still `0.3` and must be overridden);
- paired seeds: `13`, `42`, and `73` (frozen above).

The same frozen seed controls initialisation and window order across feature conditions. A changed
batch size, epoch budget, or architecture creates a separately named experiment; it cannot be
quietly applied only to the preferred model.

The maximum inner epoch budget and optimizer settings must still be frozen in the run manifest
before a formal run. The auxiliary weight, seeds, nested split, epoch-selection rule, and
outer-test access rule are already frozen above. The current single train/validation runner
cannot produce a counted result, and neither can the outer orchestrator until a reviewed
production backend, reproducible environment capture, and formal run index are implemented.

## 6. First counted real-data matrix

The owner-frozen first matrix is deliberately narrow:

```text
A   = causal combined-mix energy + F0 + voiced flag
H   = per-participant head yaw + pose validity/quality
A+H = their union
All three -> the same participant-wise 4-bin VAP tensor
```

Requirements are human-reviewed-Transcript-derived VAD, formally built and reviewed causal acoustics, causal head pose
reassembled under the frozen D3 policy,
and the exact four-group frozen split. Other unavailable cues
are excluded by an explicit feature list;
they are not zero placeholders presented as observations. The current implementation must
record the actual selected columns and `aux_vad_weight` in the checkpoint/run manifest. Until
the reviewed production backend seals and executes the untouched outer-test stage, any run
remains a development result.

R0 establishes that the single participant-wise VAP objective trains, predicts, saves, reloads,
and evaluates on real data. It is not the final multimodal claim. The acoustic representation
was owner-selected on 2026-08-03 (energy + F0 + voiced); its extractor parameters must still
be versioned in the R0 manifest. No GVAP, addressee head, or robot action module is
introduced before R0 and the required comparisons below; addressee work is out of scope
entirely (owner decision 2026-08-03).

## 7. Baselines

### B0. Last-speaker persistence

At each frame, assign probability 1 to the most recently observed speaker for every future bin
and 0 to the other participants. The primary deployable version must identify that speaker
from a genuinely causal ASD thresholded on inner-validation data. The official stock
Light-ASD/LR-ASD implementations have verified future dependence (symmetric temporal
convolutions plus backward/reversed-sequence GRUs) and therefore cannot supply this baseline;
their scores are diagnostic `offline_noncausal_oracle` evidence only. A variant using causal
past gold VAD may be reported only as `oracle last-speaker`, never as a deployable baseline.

The read-only ASD source preflight is implemented but remains **BLOCKED at R3/3** because
fixture provenance can masquerade as formal provenance and ancestors above the configured
authority root are not reparse-checked. The three-round repair limit is exhausted pending owner
review. No ASD model or weights were downloaded, no model was executed, and no probability or
feature was generated; insufficient Cam6 evidence for R remains fail-closed.

### B1. Reactive silence

Trigger after 1.5 s of continuous silence. The primary real-data baseline obtains `any speech`
from a causal combined-mix detector calibrated inside training/validation groups. The existing
offline implementation replays gold `any_speech`; until replaced or supplemented it must be
named `oracle reactive replay`. Reactive silence is a policy baseline, not a replacement VAP
output.

### B2. Structured GRU VAP

Under the eventual approved protocol, use matched participant tokens, features, target, loss,
folds, and seeds, replacing only the temporal Transformer mixer with the one-layer GRU. This
isolates temporal architecture cost.

### B3. Audio-only Transformer VAP

Use only the owner-selected causal combined-mix representation (energy + F0 + voiced,
frozen 2026-08-03). DoA remains inert. With identical global inputs broadcast to shared
participant tokens, present-participant predictions are expected to be symmetric. This is an
intentional lower bound on participant attribution, not a bug and not a per-speaker-audio
comparison.

## 8. Feature-set matrix

The proposed matrix uses the same approved Transformer configuration, folds, and seeds:

| ID | Included input | Correct interpretation |
|---|---|---|
| `A` | owner-selected causal combined-mix representation; inert DoA | audio-only, participant-symmetric lower bound |
| `H` | head yaw + pose validity/quality | head-pose-only diagnostic |
| `V` | MAR, head yaw, gaze-to-robot, gazed-at, validity/quality | **strict visual-only**, excludes ASD |
| `V+ASD` | `V` + causality-audited pretrained/adapted ASD probability | audio-visual detector condition, not visual-only; stock Light/LR scores are ineligible |
| `A+V` | `A` + strict visual features | multimodal without ASD |
| `A+V+ASD` | `A` + `V` + ASD probability | full schema-compatible core condition |

The first scientific comparison is paired `A` versus `H` versus `A+H`.
`V`, `V+ASD`, `A+V`, and `A+V+ASD` are retained only as deferred schema-compatible routes.

For every condition, fit a new training-only normaliser and enumerate the actual columns in
the manifest. Excluded features are removed or neutralised after normalisation by an explicit
mask. A placeholder column cannot count as a modality.

## 9. Targeted ablations

The first matrix directly tests the incremental head-pose contribution through paired
`A` versus `A+H`; `H` provides the head-only diagnostic. The following removals are deferred
until their source modalities have sufficient real coverage on the same paired sessions:

- `-head`: remove head yaw and its head-pose quality channels;
- `-gaze`: remove gaze-to-robot and discrete/continuous gaze-target evidence;
- `-ASD`: remove pretrained ASD probability while retaining strict visual cues;
- `-MAR`: interpretable lip-motion ablation, secondary to the required removals;
- candidate auxiliary-loss comparison: `aux_vad_weight 0.0` versus `0.3`, only if approved
  after the clean core is stable;
- temporal mixer: Transformer versus GRU using the same feature set.

Owner decision 2026-08-03: speaking_towards data is not consumed by the dissertation, so no
speaking-towards ablation exists. The current Gaze export supplies only nullable 2D
`robot` / `participant` / `elsewhere` evidence; it is not an existing 3D-derived
`gazed_at` input. Any later approved gaze-target feature is covered by the `-gaze`
removal.

CPC is a gated optional acoustic ablation (the core representation was frozen as
energy + F0 + voiced on 2026-08-03). If attempted, use a causal, versioned encoder and a
controlled paired comparison whose folds, seeds, encoder state, and fusion capacity are
approved and recorded in advance.

## 10. Metrics

### 10.1 Frame/bin metrics

Report on complete-horizon frames and present slots:

- AP for each future bin and overall macro mAP;
- participant-macro and bin-macro AP where support permits;
- macro F1 and balanced accuracy using thresholds selected on inner validation under the
  owner-approved protocol;
- positive/negative support and prevalence for every bin and participant role.

Do not report raw accuracy alone. Test predictions are generated once from the selected
checkpoint and saved before aggregate metrics are calculated.

### 10.2 Event metrics

All event scores are deterministic readouts from the same VAP tensor. Pool out-of-fold event
predictions within each seed, while retaining group identifiers for uncertainty estimates.

- **Shift/Hold:** AP, balanced accuracy, and F1; include counts for shift and hold.
- **Shift Prediction:** the same metrics over the previous speaker's contiguous
  sole-speaker tail before offset. A request for 0.5 s is capped at the 0.4 s context
  verified by event extraction; earlier or overlapping speakers never enter the readout.
- **Next Speaker:** at the prespecified transition point, rank present participants by
  `p_future`, using that same bounded sole-speaker tail; primary metrics are
  participant-macro one-versus-rest AP, tie-aware balanced
  recall, and mean reciprocal rank. Report per-participant positive/negative support and NA
  reasons. Ordinary top-1 is secondary and must state its tie-credit rule and tie rate. Holds
  may retain the current speaker. This is a readout, not a head.
- **Overlap Resolution:** AP, balanced accuracy, and F1, with separate support for
  backchannel/failed challenge and competitive holder-yield events.
- **Robot Entry:** AP and F1 for whether P1 actually enters after a human turn, plus recall on
  demonstrator-taken opportunities and false-entry rate on demonstrator-declined opportunities.

The shared evaluator derives this pooled standalone Next Speaker summary from the same VAP
tensor. A tie receives uniform fractional top-1 credit and average rank for MRR; no metric may
resolve ties with slot-index-biased `argmax`.

### 10.3 Policy metrics

Compare the owner-approved dual-threshold configuration, demonstration-driven robot-slot
policy, and 1.5 s reactive baseline on identical opportunities:

- number of eligible gaps and policy entries;
- recall on demonstrator-taken gaps;
- false entries and false-entry rate on declined gaps;
- interruption rate and successful floor acquisition where labels support them;
- entry latency from gap onset: mean, median, and 95th percentile;
- lead/lag relative to the human demonstrator onset;
- policy trigger latency, feature/inference latency, and any TTS start-up latency as separate
  quantities.

Policy thresholds, debounce, and refractory settings require owner approval and are then
locked before outer-test scoring.

## 11. Missing classes, failures, and exclusions

Every metrics file includes `n_positive`, `n_negative`, `n_events`, and `n_groups` where
applicable. JSON uses standard `null` for unavailable values; bare `NaN`/`Infinity` is invalid.

- If a test fold contains only one class, AP/balanced metrics are recorded as `NA` with the
  support counts; they are not converted to zero and the fold is not silently dropped.
- If an event extractor finds no eligible events, record zero events and `NA` metrics.
- If a training bin has zero positives, record the condition and do not interpret its
  thresholded metric even if the implementation supplies a numerically safe weight.
- A failed training seed remains in the run index with the exception, epoch, and last finite
  loss; it is not rerun under a different unpaired seed only for one condition.
- Session exclusions list the exact gate, affected feature set, and whether all paired
  conditions were correspondingly restricted.
- NaN/Inf predictions, causality violations, participant-asymmetry in the pure global-audio
  baseline, or test-group leakage are hard run failures. Expected identical present-slot
  predictions in `A` are checked and documented.

## 12. Paired comparison and uncertainty

The unit of independence is the recording group, not a frame. For each feature comparison:

1. match the same outer fold and random seed;
2. compute the metric difference condition-by-condition;
3. report every group/fold value, the paired mean and median difference, and seed variability;
4. obtain a descriptive 95% interval by resampling recording groups, not frames or windows.

With very few groups, confidence intervals and significance tests are unstable. Report effect
sizes and all group-level points rather than claiming significance from millions of correlated
frames. A modality that does not improve performance is a valid controlled finding.

## 13. Artifacts and manifests

The following is the required production artifact set, not a description of what the current
smoke orchestrator already emits. Each eventual local ignored directory
`outputs/experiments/<run_id>/` contains:

```text
dataset_manifest.json   processed-file hashes, provenance, feature sources, sync quality
split_manifest.json     outer/inner group and session membership
run_config.json         feature list, model/trainer/label/policy config, all seeds
environment.txt         Python/PyTorch/package versions, CPU/device information
source_state.txt        git commit and dirty-file list
checkpoint_<fold>_<seed>.pt
frame_predictions_<fold>_<seed>.csv
frame_predictions_<fold>_<seed>.metadata.json
event_predictions_<fold>_<seed>.csv
policy_trace_<fold>_<seed>.csv
metrics.json            frame/event/policy metrics plus supports and failures
run_index.json          command, timestamps, completion status, exclusions
```

The manifest must explicitly include `aux_vad_weight`, chosen feature columns, acoustic
window definitions, ASD model/checkpoint identity, normaliser fit groups, checkpoint format,
and whether a result is primary, oracle, sensitivity, or smoke-only. Raw audio/video and
participant-identifying material remain local and are never committed.

The current outer orchestrator covers versioned split/run/result manifests and per-unit
artifacts only under its smoke backend. Capturing `environment.txt`/source state and maintaining
an atomic formal `run_index.json` across all condition × seed × fold × analysis-variant units
remain production blockers; missing either forbids counted status.

The permanently non-counted production-evidence foundation holds read-only handles for the
result dependency set and replays protected-column central/±D3 pose variants. Those checks reduce
mixed-snapshot risk but are not a publication boundary. A same-directory concurrent-replacement
cleanup race remains reproducible; three repair/retest rounds are exhausted. The owner accepted
a trusted local single-user/single-writer workspace on 2026-08-16. Hash binding, replay and
no-overwrite checks remain mandatory, and the residual risk is disclosed; the atomic formal run
index and controlled production environment remain pending.

## 14. Execution order

### Phase 0: close data blockers

1. retain the already confirmed four-group, ten-scene primary corpus and its six-scene
   four-group `d3_le80` paired sensitivity subset;
2. obtain only the four targeted impossible-interval decisions, then merge the exact
   human-reviewed-Transcript-derived VAD; lavaliers remain automatic consistency evidence;
3. integrate and independently review the already implemented read-only D3 central/±error-bound
   reassembly inside the formal production backend; the current evidence foundation is
   permanently non-counted and cannot authorize a head-pose condition;
4. use the owner-frozen `0.45` voicing threshold. The current in-memory acoustic scaffold can
   never publish or authorize assembly; complete a new
   formal builder and artifact format, strict validation, and independent review;
5. integrate each available visual feature with real/placeholder provenance;
6. generate the corpus and split manifests, then run schema/causality checks;
7. implement and independently review the production PyTorch backend, environment capture, and
   formal no-clobber run index. Until then, only `backend_contract_smoke` outputs are permitted.

### Phase 1: prove the single real-data output

1. run paired `A`, `H`, and `A+H` under the owner-approved auxiliary weight and split protocol;
2. reload each checkpoint and reproduce predictions;
3. verify output shape, horizon masking, absent-slot isolation, group separation, and causal
   replay equivalence;
4. save frame and available event supports even when performance is poor.

### Phase 2: required comparisons

1. B0 last-speaker and B1 reactive baselines;
2. `A`, `H`, and `A+H` on paired folds/seeds;
3. GRU versus Transformer;
4. defer `-gaze`, `-ASD`, `-MAR`, and extended visual matrices until after the four-day core;
5. frame, event, and policy reports with failures/supports.

### Phase 3: apply the optional-work gate

The gate is evaluated only after Phases 1--2 and reproducibility review.

- If a physical Furhat and a same-feature causal live perception pipeline are ready, take the
  Furhat route first using the current WebSocket Realtime API.
- If either hardware or live perception is not ready, retain the unchanged participant-wise
  VAP baseline and consider the offline GVAP auxiliary extension.
- Recorded causal-feature replay to physical/virtual Furhat may run in parallel as secondary
  control-path evidence, but it is not a live-perception result.
- If both routes are genuinely ready, complete Furhat first and GVAP second without reducing
  the required ablations.

Furhat-live readiness means hardware access, stable participant identity tracking, causal
runtime versions of every checkpoint input, measured feature/inference latency, a Realtime API
control test, an intent gate, safe interruption/stop handling, and a deployment checkpoint
whose feature subset matches runtime. GVAP, if selected, is a separately named auxiliary-head
ablation (for example the 136-state role-relative projection); it is not implemented or
assumed by this plan.

No route supports a "first multimodal multi-party robotic VAP" claim. Replay validates the
control path; only a full live loop validates live perception and actuation.
