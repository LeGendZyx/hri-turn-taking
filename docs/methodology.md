# Methodology: Core Participant-wise Multimodal VAP

> Status: draft methodology handoff, 2026-07-14. This document operationalises the
> supervisor-directed scope in [`PROJECT_SCOPE.md`](PROJECT_SCOPE.md). If the two
> documents conflict, `PROJECT_SCOPE.md` takes precedence. The outer scientific
> state machine and four-group protocol are implemented, but only a non-counted
> smoke backend is registered. A separately reviewed PyTorch backend boundary
> exists for deterministic synthetic fixtures only and is deliberately
> unregistered/non-counted; there is no real production registration, formal
> run index, or counted result. A read-only production-evidence foundation
> replays live source/environment state, canonical real ticks and central/±D3
> pose variants, but it is permanently non-counted and has no publisher. After
> three repair/retest rounds, the owner accepted a trusted local single-user/single-writer
> publication boundary; residual concurrent/malicious replacement risk is disclosed.
> Unless explicitly attributed to an owner-approved decision, numeric extractor,
> optimisation, seed, and protocol choices below are proposals or implementation defaults.

## 1. Owner-approved core and proposed details

The required dissertation result is an offline, causal comparison on the project's own
three-/four-person receptionist-style recordings. The question is whether participant-level
robot-observable social cues improve future voice-activity prediction over global mono audio
and simple turn-taking baselines.

The core has exactly **one learned primary output**:

```text
q[t, i, k] = P(participant i is voice-active in future bin k | inputs at or before t)
shape       = time x participant x 4
```

The bins are `[0.0, 0.2)`, `[0.2, 0.6)`, `[0.6, 1.2)`, and `[1.2, 2.0)` seconds.
For four present participants this is one head containing 16 Bernoulli probabilities per
frame. It is a lightweight factorised multi-party approximation, not a joint 65,536-class
codebook.

Everything below is derived from that same tensor and is **not** another learned output:

- duration-weighted `p_now` and `p_future`;
- Shift/Hold and Shift Prediction scores;
- likely Next Speaker;
- overlap-resolution and robot-entry scores;
- the downstream dual-threshold floor-entry decision.

The existing `vad_head` predicts current VAD only as an optional training regulariser. It is
not a delivered output and must not be used to make the primary task look like a two-module
system. **Owner decision 2026-08-03: the first formal run uses
`aux_vad_weight = 0.0`;** the current implementation default `0.3` must be overridden and the
effective value recorded in every checkpoint and run manifest. The 0.0-versus-0.3 paired
comparison remains an unapproved candidate.

GVAP, semantic context, and Furhat deployment are outside this first output and remain gated
options in Section 10. Addressee work is out of scope entirely (owner decision 2026-08-03:
speaking_towards is not consumed; see `final_model_plan.md` §1).

## 2. Data, slots, and supervision

Each session is resampled to a 25 fps shared timeline whose zero point is the mic-A audio
axis. Real session identifiers are `<group>__<scenario>`. Slots are read from the confirmed
mapping rather than inferred from position:

```text
P1 = R (human demonstrator in the future robot position)
P2 = A
P3 = B
P4 = C, or statically absent in a three-person session
```

`P{i}_present` means that a participant exists in the session and is static. A tracker
dropout is represented by the relevant per-frame validity/quality values, never by changing
the presence mask.

For the first counted run, owner-confirmed human-reviewed Transcript intervals are mapped to
the mic-A axis and rasterized at 25 Hz after only four targeted impossible-interval decisions.
This **human-reviewed-Transcript-derived VAD** creates the target for participant `i`, bin `k`,
at frame `t`:

$$
y_{t,i,k} = \mathbb{1}\left[
\frac{1}{|W_k|}\sum_{\tau \in W_k}\mathrm{VAD}_i(t+\tau) \ge 0.5
\right].
$$

Absent participants and the last two seconds of each session, where the complete prediction
horizon is unavailable, are excluded from the loss. Gold VAD also defines evaluation events;
it is never concatenated to the model input.

The robot-role participant is ordinary supervised slot `P1`. Its output can be interpreted as
"would a human standing in the robot's position speak in this future interval?" This supports
offline demonstration-agreement analysis, but it does not erase the human-to-robot transfer
gap.

## 3. Strict feature tiers

### Gold tier: target and offline evaluation only

- human-reviewed-Transcript-derived VAD for the first run;
- aligned per-participant lavaliers as automatic consistency/oracle evidence, not a second
  manual reannotation or sample-perfect target claim;
- manually checked overlap type (`backchannel` versus `competitive`) where available;
- any oracle per-person audio or multi-view variables used only for diagnostic upper bounds.

Gold-tier use must be named `oracle` in tables and manifests. It cannot be presented as a
deployable input.

### Robot-observable tier: model input

The current schema provides seven values per participant:

1. `asd_prob`;
2. `mar`;
3. `gaze_robot`;
4. `head_yaw_deg`;
5. `gazed_at` (reserved schema field; the speaking_towards delivery is not consumed by the
   dissertation, and the current Gaze export supplies only nullable 2D `robot` /
   `participant` / `elsewhere` evidence, not an existing 3D-derived `gazed_at` input);
6. `pose_valid`;
7. `pose_quality`.

It also provides global `doa_angle_deg`, `array_energy`, `f0_proxy`, and `voiced`. DoA is permanently
inert on real data because no microphone array was recorded; it must not be geometry-
simulated. The implemented schema v2 dataset transform expands DoA to sine/cosine, yielding
32 raw input columns and 33 model dimensions. Each participant token therefore receives its
seven own features plus five broadcast global features, i.e. a 12-dimensional token before
the shared `12 -> 64` embedding.

The optional post-hoc Gaze route does not mutate that primary schema. Its parallel schema v3
adds one `P{i}_gaze_robot_valid` bit per slot: 36 raw columns become 37 model dimensions
after DoA sine/cosine, and each token has eight own plus five broadcast global features
before a shared `13 -> 64` embedding. C0 disables both Gaze value and validity, CM enables
validity only, and CG enables validity plus the observed A/B/C Gaze value. Consequently,
`CG-CM` is the content contrast, while `CM-C0` measures observability/missingness.

Owner decision 2026-08-16: the four-day first counted matrix is `A` (causal combined-mix
energy/F0/voiced), `H` (head yaw plus pose validity/quality), and `A+H`. ASD, Gaze,
Facial/MAR, GVAP, and physical Furhat work are deferred from this first matrix.

Training-time head yaw is currently projected from multi-view 3D. Head pose is a feature kind
that a robot-view tracker can produce, so this is recorded as a train/deploy domain shift, not
used as permission to feed non-observable geometry to the model. No equivalent 3D
`gazed_at`/`gaze_robot` derivation currently exists. The available Gaze evidence is nullable
and 2D; any later model-input promotion requires an approved derivation, explicit
validity/coverage semantics, and a recorded source extractor.

The completed post-hoc visual route is deliberately narrower. It uses only the
A/B/C binary `gaze_robot` evidence together with a separate
`gaze_robot_valid` bit; P1/R remains present but visually unobserved. The
schema-matched A+H anchor (C0), validity-only condition (CM), and
value-plus-validity condition (CG) separate Gaze content from missingness.
Across the frozen exact eight scenes (64,299 rows, four groups), seed 13 and
`max_epochs=15`, all 12 condition/fold units completed. Pooled frame mAP was
0.2717/0.2461/0.2843 for C0/CM/CG; `CG-CM=+0.0381` and was positive in all
four outer folds, whereas `CM-C0=-0.0256`. The saved-probability aggregate
SHA-256 is `0b4f2c5e3deafafa0d23cdb4822f328d95af5c0788ba662cd9c04d7414955cea`.
This is single-seed post-hoc exploratory evidence, not production or counted
evidence. The actual upstream extractor variant/weights/configuration are not
fully bound, exact upstream regeneration is unavailable, and implementation-
level counterfactual causality has not been verified; the result therefore
does not establish H1 or causal deployment equivalence.
Facial is not used in the dissertation experiments: the delivered export has no
reproducible mouth landmarks, its `mouth_motion` is not MAR, its hard emotion
class is unvalidated, and the anonymous AU order/extractor binding is
insufficient for a reproducible causal input. The original Facial bytes remain
preserved as evidence rather than being relabelled or zero-filled.

### ASD naming rule

Light-ASD/LR-ASD is an **audio-visual** active-speaker detector. Consequently:

- `strict visual` excludes `asd_prob` and contains only camera-derived features;
- `visual + ASD` includes `asd_prob` and must be named audio-visual, not visual-only;
- `audio-only` contains global combined-mix acoustics and excludes ASD.

This distinction prevents hidden audio from inflating a claimed visual-only result.

The official pretrained implementations are also **not causal as shipped**. A source-level
audit on 15 August 2026 bound Light-ASD commit
`ed38c232de5efe0261dbd68627c0ade7cdfe14eb` and LR-ASD commit
`1b6dcd2d8fc2895683de6508ec6294ec47d388ca`: both apply symmetric temporal
convolutions and then use a reversed/backward GRU over the evaluated face track. Official
validation feeds whole tracks. Consequently a score at time $t$ can depend on frames after
$t$ up to the track/chunk boundary; cutting arbitrary chunks or shifting the outputs cannot
turn that unbounded-within-chunk dependence into a declared fixed latency. Stock scores are
permitted only as `offline_noncausal_oracle` diagnostics or human-review proposals. A formal
robot-observable `asd_prob` requires a separately versioned causal model/head that passes
future-pixel, future-audio, prefix-extension and chunk-boundary counterfactuals. No counted
ASD feature currently exists.

A genuinely causal real-time ASD architecture has been reported by Gurvich et al.
([arXiv:2309.08295](https://arxiv.org/abs/2309.08295)), but its spatial-query
front end assumes a microphone array and a 360-degree camera. This corpus has neither a
recorded array nor that camera geometry, and the present audit found no author-released
drop-in code/weights. It is therefore evidence that causal ASD is technically possible, not a
source that can be substituted into this experiment without a new, separately validated model.

The ten-scene read-only ASD source preflight is implemented but formal acceptance remains
**BLOCKED at R3/3**: fixture provenance can masquerade as formal provenance, and ancestors
above the configured authority root are not reparse-checked. The three-round repair/retest
limit is exhausted pending owner review. No ASD model or weights were downloaded, no model
was executed, no probability or feature was generated, and nothing was published;
insufficient Cam6 evidence for R fails closed.

An eventual ASD source contract must also distinguish a valid low speaking probability from
a missing/invalid face observation. The proposed `P{i}_asd_valid` binary mask is therefore an
owner-pending schema change, not something that may be silently overloaded onto
`pose_valid` or represented by an unqualified zero.

## 4. Core acoustic representation (owner-frozen 2026-08-03)

The current table can hold causal log energy plus a causal F0 proxy from the single combined-
mix track. **Owner decision 2026-08-03: this causal energy + F0 block, extended with a
separate global `voiced` flag, is the core acoustic representation** (CPC is deferred to a
gated optional ablation, see below). Sampling rate, window lengths, F0 range, voicing rule,
and normalization below remain the candidate implementation and must be versioned in the
manifest of the first counted run.

The implementation and real-source wiring completed a read-only in-memory
preflight on all ten canonical analysis bases (78,582 ticks): every selected row
had complete finite causal history after full-source replay. This scaffold has
no publisher and can never publish or authorize assembly. The owner froze the
normalized-autocorrelation threshold at `0.45` on 2026-08-16; the strict
read-only, non-publishing scaffold is **ACCEPTED at R3/3** only for this
feasibility role. After that decision, a new formal builder and
artifact format must pass their own strict validation and independent review.

One candidate implementation at each 25 fps timestamp `t` is:

- take the deterministic equal-weight mono average of the sample-replayed
  `master_aligned.wav` channels, then causally resample it to 16 kHz;
- set `array_energy` to `log(max(RMS, 1e-6))` over the completed trailing 40 ms
  interval `(t-0.040, t]`;
- estimate periodicity from the completed trailing 80 ms interval `(t-0.080, t]`, after
  mean removal, using normalized autocorrelation over 50--500 Hz lags;
- call the frame voiced when the owner-frozen normalized-peak threshold `0.45` is met and set `f0_proxy`
  to `log2(F0 / 100 Hz)`. The earlier encoding collision (a voiced 100 Hz frame and an
  unvoiced frame both mapping to `0`) is resolved by owner decision 2026-08-03: the schema
  adds a separate global `voiced` 0/1 column (schema v2, implemented 2026-08-15;
  32 raw columns → 33 model dims); `f0_proxy` is meaningful only where `voiced = 1` and stays `0`
  where `voiced = 0`;
- set real-data `doa_angle_deg` to the documented constant `0`; after train-only
  normalisation its transformed constant channels carry no information.

Any accepted causal extractor must use the frame timestamp as the **right edge** of every
acoustic window. No centred framing,
bidirectional pitch smoothing, full-utterance normalisation, or future-dependent voicing
decoder is permitted. If the exact extractor implementation differs, it must still produce
this representation and pass the causality check; its version and parameters belong in the
dataset manifest.

The 2026-08-15 source audit established that the available master is assembled
from the four close-talk lavaliers, while the shorter participant-video WAV is
a mux/extraction derivative rather than an independent room microphone. The
input is therefore reported as an equal-weight close-talk surrogate. The
difference between this training source and Furhat's runtime far-field
microphone is a required limitation and, if deployment is reached, a measured
domain shift.

CPC is no longer a core candidate: the owner ordered the candidates on 2026-08-03
(energy + F0 + voiced = core). CPC survives only as a gated optional ablation after the core
real-data result; if attempted, its causal availability, encoder state, comparison folds,
randomization, and fusion capacity must be approved and recorded before use.

## 5. Causality contract

For a prediction timestamp `t`, every input must be computable from information received at
or before `t`:

- intake uses latest-sample-at-or-before-frame zero-order hold, with no interpolation and no
  future sample selecting a past fragment;
- camera and audio windows are trailing/right-aligned;
- smoothing, detector context, and confidence calibration are causal or their algorithmic
  delay is explicitly added to the feature timestamp;
- normaliser and class weights are fitted on training groups only;
- thresholds and early stopping use inner-validation groups only;
- a held-out group's labels, statistics, event counts, or missingness patterns cannot affect
  training or model selection.

The primary causality test perturbs all source samples after `t` and requires every feature and
VAP probability up to `t` to remain unchanged. Replay timing must reproduce each extractor's
buffering delay rather than pretending that an offline feature was available earlier.

## 6. Participant-structured model

`MultipartyVAP` is the required starting point:

1. construct one token per participant and broadcast the same global acoustic block to it;
2. apply a shared participant embedding;
3. apply two causal temporal blocks, using banded Transformer attention with ALiBi for the
   main model or a GRU for the low-cost comparison;
4. apply cross-participant attention at each frame, masking absent slots as keys;
5. apply a shared four-logit head to every participant token.

The current scaffolding defaults are `d_model=64`, two blocks, four attention heads, dropout
`0.1`, and a 250-frame (10 s) causal attention/training window. These are implementation
defaults, not an approved counted-run configuration. Weight sharing makes the network
slot-permutation-equivariant and supports both three- and four-person sessions. Full-session
Transformer inference retains `n_layers x attention_window` left context when chunked and
must exactly match a full causal pass. GRU inference currently uses a full-session forward
pass unless recurrent hidden state is explicitly carried between chunks; resetting hidden
state at chunk boundaries is invalid.

Because the real audio block is global, contains no DoA, and is broadcast identically, an
`audio-only` input makes all present participant tokens identical under the shared,
permutation-equivariant model. Their participant-wise predictions should therefore collapse
to the same values. This is an intentional weak but deployable baseline: it measures what
global audio alone can say about **whether someone** will speak, while exposing that it cannot
identify **which participant** without participant-specific observable cues. It must not be
quietly strengthened with per-lavalier audio or oracle geometry.

## 7. Objective and optimisation

The primary loss is masked, per-bin class-weighted binary cross-entropy:

$$
\mathcal{L}_{VAP} =
\frac{
\sum_{(t,i)\in\mathcal{M}}\sum_k
\left[(1-y_{t,i,k})\,\operatorname{softplus}(\ell_{t,i,k})
+ p_k y_{t,i,k}\,\operatorname{softplus}(-\ell_{t,i,k})\right]
}{|\mathcal{M}|K}.
$$

`p_k` is PyTorch `pos_weight`: it multiplies only the positive-class term and is computed from
training groups only (with the implementation's numerical clipping). The denominator is the
actual count of future-horizon-valid, present-slot `(time, participant, bin)` entries. The
auxiliary current-VAD BCE is normalized over present-slot `(time, participant)` entries and
may use trailing frames because its target is current rather than future activity. The complete training loss is
`L_VAP + lambda_aux L_current-VAD`; owner decision 2026-08-03: `lambda_aux = 0.0` for the
first formal run (the 0.0-versus-0.3 comparison remains an unapproved candidate).

The scaffolding currently uses AdamW, no decay on biases/norm parameters, linear warm-up,
cosine decay, gradient clipping, and early stopping on an inner-development split. These are
candidate optimisation choices. Its reported post-selection metrics on that same split are
development results only. A formal result requires the pre-outcome protocol frozen on
2026-08-15: four outer leave-one-group-out folds, three inner leave-one-group-out folds within
each outer-training partition, and paired seeds 13/42/73. The untouched outer group is never
used for configuration, threshold, epoch, normalisation, class weighting, or checkpoint
selection.

## 8. Readouts and evaluation semantics

With bin widths `(0.2, 0.4, 0.6, 0.8)` seconds:

$$
p_{now} = \frac{0.2q_0+0.4q_1}{0.6}, \qquad
p_{future} = \frac{0.6q_2+0.8q_3}{1.4}.
$$

These are deterministic summaries of the sole VAP tensor. Next Speaker is likewise obtained
by comparing participant scores at a prespecified transition point; it is not a classifier
head. The dual-threshold policy is downstream evaluation. It must report whether it uses the
current floor-holder readout or the robot-slot demonstration readout.

The primary metrics are support-aware mAP plus participant-macro/balanced/ranking metrics and
event tests. A bin or participant with only one class is `NA`, with its positive/negative
counts and reason retained; it cannot be silently removed from an apparently higher mAP. Raw
accuracy alone is never sufficient. Exact metrics and paired comparisons are specified in
`experiment_plan.md`.

## 9. Current real-data readiness audit

The wider HPE handoff contains 21 mapped scenes and 14 scenes with a usable
`6cam` D3 offset. The dissertation subset is deliberately narrower: on
2026-08-15, ten scenes across four groups obtained sealed Transcript-semantic
identity evidence and were materialised as version-2 pose bases under
`data/processed/intake_v2/`. Full deterministic source replay accepts all ten.
The older `05_15__2` artifact remains stale and is not part of this corpus.

| Check | Exact observed state | Consequence |
|---|---|---|
| HPE D1/D2 deliveries | 21 scenes across 5 groups; all mappings confirmed | head pose is consumable on each scene's local six-camera axis |
| shared-timeline HPE deliveries | 14 across all 5 groups (`03_20__{4,5}`, `03_24_1__{2,3}`, `03_26__3`, `05_14__{2..5}`, `05_15__{2..6}`) | the synchronization side of a group-disjoint split is now satisfiable; the 7 scenes without a physical bridge stay held |
| selected v2 pose bases | 10 scenes / 4 groups / 79,159 rows at 25 Hz | all pass sealed identity, D3, half-open-window, hash, and deterministic pose replay |
| corrected lavalier alignment | 10 formal time-map manifests; 13 non-zero constant channel corrections | original WAVs remain unchanged; `05_14__2` R is identity-mapped with `coarse_warning=true` and an independent 5 ms resolution pass |
| formal common-window bases | 10 scenes / 4 groups / 78,582 rows at 25 Hz | exact intersection of pose and corrected-lavalier/reference coverage; 577 unavailable pose rows are excluded rather than zero-filled; sealed audit reports `structural_ready=true` |
| current structural provenance | parent `vap_real_pose_intake` v2 plus canonical child `vap_real_analysis_base` v2; sealed crosswalk, audio-gold and exact parent-slice bindings | both directory audits report `structural_ready=true`; child targets/features remain placeholders |
| provenance `trainable` | `false` on all ten | correct until gold VAD and selected causal inputs are assembled |
| real feature columns | 12: head yaw, pose validity, and pose quality for P1--P4 | only head-pose intake is real |
| placeholders | 24 | P1--P4 ASD/MAR/gaze/gazed-at/VAD plus energy/F0/voiced and inert real-data DoA are zero-filled and explicitly non-real |
| synchronization | every v2 sidecar binds its current offset and quality; `05_15__2` uses the republished `-3.7621 s` (degraded, ±36.2 ms) | the stale `-6.3225 s` value cannot re-enter through the replay gate |
| Transcript review | exact ten-scene Transcripts are owner-confirmed existing human annotations; 1,094 in-window events | only four impossible intervals receive targeted new decisions; the other 1,090 are inherited, with no all-event relisten, 50-item-per-scene blind audit, or seven-day retest |
| first-run gold-tier VAD | ten human-reviewed-Transcript-derived VAD v1 target bundles published and replayed | exact hash-bound intervals are rasterized at 25 Hz; lavaliers are consistency/oracle evidence, not sample-perfect labels |
| deadline experiment runner | exact eight-scene / four-group A, H and A+H nested-LOGO matrix completed at seed 13 and `max_epochs=5` | all 12 units and the saved-OOF aggregate passed independent replay; artifacts remain explicitly `production_eligible=false` and `counted_eligible=false`; D3 sensitivity and additional seeds remain deferred |
| post-hoc Gaze isolation | exact eight scenes / 64,299 rows / four groups; schema v3 C0/CM/CG × four outer folds, seed 13, `max_epochs=15` | 12/12 units and the saved-probability aggregate passed independent audit; `CG-CM=+0.0381` pooled frame mAP and is positive in all folds, while `CM-C0=-0.0256`; P1 has no direct Gaze, exact upstream regeneration is unavailable and implementation counterfactual causality is unverified |
| post-hoc matched policy replay | inner-calibrated P1 predictive and causal acoustic-silence policies evaluated on the same 303 held-out robot-entry opportunities | complete and independently replay-audited; neither the five- nor 15-epoch comparison supports H2, and no robot/TTS/live latency was tested |

The core environment itself is healthy on CPU (`torch 2.12.0+cpu`, pandas 3.0.3,
NumPy 2.4.6, scikit-learn 1.9.0; `pip check` passes). The local environment does not currently
contain the optional extraction/runtime packages `torchaudio`, `librosa`, `sounddevice`,
OpenCV, MediaPipe, PyArrow, pyannote, or Silero VAD, and no local Light-ASD or CPC weights/code
were found. These are extraction dependencies to provision locally when their corresponding
condition is activated; they are not a reason to weaken the data gates or upload raw media.

The provenance-bound common analysis windows remain immutable structural parents
under `data/processed/analysis_base/`; their placeholder columns correctly keep
those bases `trainable=false`. Separate versioned target, acoustic and assembled
artifacts were published without rewriting the bases. Ten assembled scene pairs
exist, while the owner-frozen eight-scene view is the exact target- and
selected-feature-ready A/H/A+H training corpus. `05_15__4` and `05_15__5` remain
preserved outside that view pending owner review.

The completed deadline matrix is a reproducible real-data result but not a
production/counting-registry promotion. Publication-grade replication still
requires the deferred D3 sensitivity analyses, additional seeds and the
production backend/registry gates; those absences are reported rather than
silently inferred from the deadline run.
The owner accepted the trusted local single-user/single-writer publication boundary on
2026-08-16; hash/replay/no-overwrite checks remain mandatory and residual concurrent or
malicious path-replacement risk is disclosed. One-group or
`--allow-untrainable` executions are engineering smoke tests, not dissertation results;
real-data partitions are always group-disjoint.

## 10. Optional-work decision gate

**Current state: CLOSED.** Neither GVAP nor Furhat work may displace the real participant-wise
VAP run and its required modality ablations.

The gate opens only after:

1. a trainable real-data artifact passes the tier and causality checks;
2. group-disjoint core, baseline, and ablation runs complete on paired folds/seeds;
3. frame, event, and policy metrics include class supports and failure reports;
4. checkpoints, predictions, split definitions, and manifests are reproducible.

After it opens:

- if a physical Furhat and a causal live perception pipeline are both ready, prioritise the
  Furhat route, because it has explicit supervisor value;
- otherwise, retain the unchanged single-output model as the baseline and consider a GVAP
  auxiliary extension as the lower-risk offline comparison;
- a recorded-feature Furhat replay/control integration may proceed in parallel as secondary
  engineering evidence, but it does not validate live perception and cannot open the gate;
- if both full routes are ready and core evidence is secure, do Furhat first and GVAP second.

No GVAP or Furhat implementation is claimed here. A later GVAP study must be labelled an
ablation against the unchanged participant-wise output, not a replacement. A later Furhat
study must distinguish recorded replay/control from a complete live loop. The contribution
must not be described as the first multimodal multi-party robotic VAP.

## 11. Required limitations

- global combined-mix audio cannot localise a participant, and the real corpus has no DoA;
- Light-ASD is audio-visual, not a pure camera cue;
- multi-view-derived training head pose and robot-view runtime head pose differ in domain;
- a combined receiver mix and a Furhat far-field microphone differ in domain;
- a human playing the receptionist and a physical Furhat differ in prosody, response time,
  embodiment, and the behaviour they elicit;
- the policy is trained in receptionist-style scenarios and is not evidence for general
  open-domain conversation;
- synthetic results validate software properties, not the dissertation hypothesis.
