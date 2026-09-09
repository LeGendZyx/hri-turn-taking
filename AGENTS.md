# AGENTS.md — Agent Guide for the Predictive Turn-Taking Project

> Canonical instructions for AI coding agents (Codex, Claude Code, and any
> `agents.md`-aware tool) working in this repository. `CLAUDE.md` imports this
> file, so **this is the single source of truth — edit shared guidance here.**
> Last verified against the codebase: 2026-08-18.

## Communication

- **Language:** respond to the user in **Simplified Chinese (简体中文)** by default;
  keep code, commands, and error messages in English.
- **Owner retry limit (2026-08-15):** for the same reproducible blocking
  condition, perform at most **three** independent repair-and-retest rounds.
  If the condition is still reproducible after the third round, stop adding
  patches, mark the item pending owner review, and present the smallest safe
  alternatives. Do not silently start a fourth round.

## 1. Project Overview

**MSc Robotics dissertation (King's College London, supervised by Dr. Oya
Celiktutan).**
Title: *Learning from Humans: A Hybrid Multimodal Framework for Predictive
Turn-Taking in Multi-Party HRI.*

Goal: build and evaluate a **predictive multimodal model** that estimates upcoming
speaker transitions (VAP-style future voice-activity prediction) in 3–4 person
group conversations. Physical Furhat deployment is an optional, decision-gated
stretch route rather than a condition for completing the core dissertation result.

**Current supervisor-directed scope handoff (2026-07-14):** before proposing
methodology, experiment, or Furhat changes, read **`docs/PROJECT_SCOPE.md`**.
It records the latest decision to make one participant-wise VAP tensor the
mandatory first output, with GVAP and Furhat treated as gated later work.

**Current identity/data-availability handoff (2026-08-15):** before selecting a
real scene, resuming participant-identity work, or claiming a modality is
available, read **`docs/IDENTITY_DATA_STATUS.md`**. It is the live Phase 1
availability ledger; dated examples in the annotation manual and command files
are not completion status.

**Current training-data handoff (2026-08-15):** before building a real feature,
assembling a training table, or starting an experiment, read
**`docs/TRAINING_DATA_STATUS.md`**. It is the live post-identity readiness
ledger; a structurally accepted pose table is not the same as a train-ready
multimodal table.

**Core novelty framing** (vs. 2025 SOTA): participant-wise multimodal future
voice-activity prediction for 3–4 person, receptionist-style HRI recordings,
with causal robot-observable inputs and supervised behaviour for the human
demonstrator occupying the robot position. Do not claim "first multi-party VAP"
(Elmers et al., Interspeech 2025, audio-only triadic) or "first multimodal VAP"
(Onishi et al., IEICE 2025, dyadic). If a full physical Furhat study is actually
completed, it may support an additional deployment contribution; the core novelty
must not depend on that optional outcome.

## 2. Research Framing (from the proposal, Feb 2026)

- **Problem**: reactive silence heuristics cause unnatural latency and are
  "blind to overlap"; audio-only speaker ID degrades badly in overlapping speech.
- **Hypotheses**: **H1** — fusing visual voice-activity cues with audio improves
  active-speaker detection during overlap; **H2** — a predictive model lets the
  robot identify turn-transition points *during* speech (prediction) instead of
  *after* silence (reaction), cutting turn-entry latency and missed turns.
- **Research questions**: RQ1 multimodal perception (fuse gaze + lip/ASD with
  audio DoA for robust speaker ID); RQ2 hybrid architecture (drop the predictive
  model into the modular Furhat stack, replacing heuristics); RQ3 evaluation
  (floor acquisition in overlap + latency vs. the reactive baseline).
- **Evaluation targets**: offline balanced metrics + VAP event tests; online
  KPIs (floor-entry latency, successful floor acquisition) + Godspeed (Animacy)
  + perceived-latency Likert. End-to-end latency budget **< 500 ms** (camera
  ~33 ms, feature extraction, inference, bridge, Furhat TTS startup 200–400 ms).

> **The implemented method has evolved past the proposal — follow the current
> design in §4, not the proposal's older framing.** The proposal described a
> 3-class LSTM (Speaking/Listening/TRP), MAR-based V-VAD as the *core* innovation,
> and a single `P(TRP) > δ` threshold. The project has since moved to VAP
> future-voice-activity prediction, **pretrained ASD as the core V-VAD** (MAR
> demoted to an ablation), and a **dual-threshold** policy. Do not re-introduce
> the older framing.

## 3. Supervisor-Agreed Direction (email thread, 2026)

- Supervisor (Oya) and the PhD (Michelle) **agreed to focus on VAP**. The
  reference Oya shared, **arXiv:2205.09812** (Ekstedt & Skantze, *Voice Activity
  Projection*, Interspeech 2022), is the original VAP paper — it reinforces the
  existing key reference, it does not change scope.
- Liaise with **Michelle** on the precise problem formulation (she owns audio
  sync and VAP framing on the team).
- **Audio-channel decision (user, 2026-07-03):** the *deployed* VAP model takes
  **single-channel** (robot's-ear) audio as its input, with GCC-PHAT DoA from
  the 4-channel ReSpeaker array kept as an auxiliary spatial feature — i.e.
  exactly what Furhat can consume at runtime. Multi-channel audio is reserved
  for offline ablation / oracle experiments only. This answers Michelle's
  single-vs-multi-channel question; the code already follows this split.
  **Amendment (2026-07-05):** the recorded corpus contains **no mic array**
  (see §8 audio inventory), so the DoA auxiliary feature has no real-data
  source — resolution pending (drop vs. geometry-simulate). The single-channel
  core of the decision stands. **Source closure (2026-08-15):** the available
  `master_aligned.wav` is a four-channel container assembled from the four
  lavaliers, not an independent robot-ear recording. Real model inputs use its
  sample-replayed equal-weight mono average as an explicitly reported
  close-talk surrogate.
  **Resolved (user, 2026-07-10): DoA is dropped for real data** (not
  geometry-simulated — that risks oracle leakage). The `doa_angle_deg` column
  stays in the schema as an inert constant on real tables (constant features
  z-score to zero) so the model layout is unchanged; the synthetic pipeline and
  a mic-array ablation may keep using it.
- **Head-pose feature source (user, 2026-07-10):** training-time
  `P{i}_head_yaw_deg` is projected from the multi-view `world_persons.csv`
  facing vectors even though multi-view 3D is gold-tier infrastructure — the
  feature *kind* (head pose) is robot-observable at runtime, so this creates a
  train/deploy **domain shift, not a tier violation**; treat it exactly like
  the close-talk→far-field audio shift and report it honestly in the thesis.
- **Owner decisions (2026-08-03), consolidated in `docs/final_model_plan.md`:**
  ① **speaking_towards data is out of the dissertation scope** — not a model
  input, no `-speaking-towards` ablation, addressee-accuracy evaluation
  deferred out of scope. The current Gaze export supplies only nullable 2D
  `robot` / `participant` / `elsewhere` evidence; no 3D-derived `gazed_at`
  feature exists yet, and promotion remains gated. The team-level
  speaking-towards deliverable itself is unaffected. ② **Core acoustics
  frozen** — causal energy + F0 +
  global `voiced` flag from the combined mix (§4.4). ③ **`aux_vad_weight =
  0.0` for the first formal run** (the 0.0-vs-0.3 comparison stays an
  unapproved candidate).
- **Owner decisions (2026-08-16, four-day completion route):** the exact current
  ten-scene Transcripts are owner-confirmed existing human annotations that
  already account for speaker identity, overlap, quiet speech and pauses.  The
  first counted label route therefore rasterizes the exact, hash-bound,
  mechanically normalised Transcript intervals at 25 Hz after resolving only
  the four impossible in-window intervals; it must be named
  **human-reviewed-Transcript-derived VAD**, not sample-perfect lavalier VAD.
  The aligned lavaliers provide full-corpus consistency diagnostics rather than
  a second manual reannotation.  No 1,094-event repeat review, 50-item-per-scene
  blind review, or seven-day retest is required; the absence of long-term
  intra-rater reliability is reported as a limitation.  The owner also froze
  the normalized-autocorrelation voicing threshold at **0.45**, selected the
  first counted input matrix **A / H / A+H**, deferred ASD/Gaze and excluded the
  current Facial export from that matrix, and accepted a trusted local
  single-user/single-writer publication
  boundary with residual concurrent-replacement risk documented.

## 4. Key Design Decisions — DO NOT silently change

Ask the user before altering any of these.

1. **Training objective = VAP future voice-activity prediction**, not a
   hand-labeled 3-class Speaking/Listening/TRP classifier.  For the first real
   run, labels are rasterized from the owner-confirmed, human-reviewed,
   speaker-attributed Transcript intervals after exact-hash binding and the four
   targeted impossible-interval decisions; aligned per-participant lavaliers
   are consistency evidence.  Turn events (shift/hold, shift-prediction,
   backchannel, overlap) are *derived* at evaluation time.
2. **Two feature tiers — never mix them in the deployed model:**
   - *Gold tier* (training labels + offline eval ONLY): per-speaker lavalier VAD,
     per-participant transcripts, raw per-person lavalier audio, multi-view 3D
     reconstruction.
   - *Robot-observable tier* (model INPUT): robot-view camera features (gaze,
     head pose, lip/ASD, per-frame observation validity/quality) and
     single-channel audio (on the real corpus: the combined mix — no mic array
     was recorded, §3/§8; mic-array DoA survives only in the synthetic/ablation
     pipeline as an inert-on-real-data column). The deployed model must only consume
     features Furhat can obtain at runtime. A "full-feature oracle" may be trained
     as an upper bound for comparison.
3. **V-VAD = a lightweight pretrained-ASD route**, with Light-ASD / LR-ASD
   retained as the reference architecture/weight source and MAR (Mouth Aspect
   Ratio) only as an interpretable ablation baseline. **Causality gate
   (verified 2026-08-15):** the official stock Light-ASD commit
   `ed38c232de5efe0261dbd68627c0ade7cdfe14eb` and LR-ASD commit
   `1b6dcd2d8fc2895683de6508ec6294ec47d388ca` both use symmetric temporal
   convolutions and a backward/reversed-sequence GRU over the evaluated track.
   Their stock per-frame scores therefore depend on future frames up to the
   track/chunk end and are **offline non-causal diagnostics only**. They must not
   be declared zero-lookahead or promoted to a counted robot-observable input.
   A counted ASD condition requires a separately versioned causal pretrained
   model or an approved/retrained causal temporal head plus future-perturbation,
   prefix-extension and chunk-boundary tests. This readiness finding does not
   silently replace the chosen V-VAD feature kind. The read-only ten-scene
   source preflight is implemented but remains **BLOCKED at R3/3**: fixture
   provenance can masquerade as formal provenance, and ancestors above the
   configured authority root are not reparse-checked. The three-round repair
   limit is exhausted pending owner review. No ASD model or weights were
   downloaded, no model was executed, no probability or feature was generated,
   and nothing was published;
   insufficient Cam6 evidence for R fails closed.  Owner decision 2026-08-16:
   ASD is deferred from the first counted A / H / A+H matrix.
4. **Audio encoding = causal energy + F0 from the combined mix (owner decision
   2026-08-03; voicing threshold **0.45** frozen 2026-08-16)**, plus a separate
   global `voiced` 0/1 column that resolves the
   F0-encoding collision. Schema v2 is implemented: 32 raw columns become 33
   model dimensions after DoA sin/cos; each token has 7 own + 5 broadcast
   global dimensions and uses `Linear(12→64)`. The pretrained CPC encoder
   (Ekstedt's VAP repo) is
   demoted to a gated optional ablation after the core result. GCC-PHAT DoA
   is an auxiliary spatial feature —
   **synthetic/mic-array-ablation only**: the real corpus has no array (§3,
   resolved 2026-07-10), so `doa_angle_deg` is an inert constant on real data.
   openSMILE/Praat may back the energy/F0 extractor only if it reproduces the
   right-aligned causal windows frozen in docs/methodology.md §4 and passes
   the extractor causality check; the exact extractor version and parameters
   go in the run manifest.
5. **Training strategy**: pretrain/freeze encoders, fine-tune only a light fusion
   + prediction head — our dataset (~a few hours) is too small to train a
   transformer from scratch. A lightweight LSTM/GRU stays as a low-cost comparison;
   AMI corpus considered for pretraining experiments.
6. **Robot decision policy = dual-threshold p_now / p_future** (Skantze & Irfan,
   HRI 2025), not a single `P(TRP) > δ`.
7. **Evaluation reports balanced metrics + event-level tests, never raw accuracy
   alone** (heavy class imbalance): balanced accuracy / F1 / mAP, plus shift/hold,
   shift-prediction, and overlap tests. Main offline comparison = our own
   audio-only ablation (NOT the ≈18% figure from Abbo et al., a different
   measurement).

## 5. Repository Map

**Core method — `src/hri_turn_taking/vap/`:**
- `labels.py` — VAP bin labels from gold VAD (bins 0.2 / 0.6 / 1.2 / 2.0 s,
  active-ratio 0.5); duration-weighted p_now / p_future readout.
- `model.py` — `MultipartyVAP`, participant-structured (~135k params): one
  token per participant, shared embedding (global acoustic features
  broadcast — combined-mix-derived on real data), then per layer [causal temporal attention with ALiBi bias +
  banded window (GRU as low-cost comparison) → cross-participant attention
  with absent slots masked → FFN], shared per-token heads (future bins +
  auxiliary current VAD). Slot-permutation-equivariant (unit-tested);
  independent per-bin BCE — a multi-party simplification of the 256-class
  joint VAP state space.
- `dataset.py` — wide table → tensors; DoA angle → sin/cos (wrap-safe);
  slot-shared feature normalisation (stats pooled over present slots — needed
  for permutation equivariance); sliding windows; session-level splits,
  group-aware from gated provenance `group_id` on real data (`group_delimiter`
  is a synthetic-only convenience).
- `trainer.py` — masked, per-bin class-weighted BCE (absent slots + truncated
  horizon excluded), AdamW (no decay on norms/biases), warmup+cosine schedule,
  early stopping on inner-validation macro per-bin mAP; the current runner labels
  its report `development_inner_validation` and has no formal outer-test stage.
  Strict mAP becomes NA if any bin is single-class and retains per-bin/participant
  supports and reasons. Checkpoint format v4 stores model + trainer + label config +
  normaliser + feature metadata + source manifest + smoke/override status, and rejects any
  other format version or feature schema with a "retrain" error (stale
  checkpoints must not fail late in the normaliser); chunked full-session
  Transformer inference equals a full pass (left context = n_layers ×
  attention_window); GRU inference uses a complete forward pass unless hidden
  state is explicitly carried.
- `events.py` — shift/hold, shift-prediction, pooled Next Speaker,
  overlap-resolution
  (backchannel vs. competitive barge-in, RQ3) and robot-entry (does the
  demonstrator take a gap following a human turn) event tests; per-event
  (label, score) lists are pooled across sessions before scoring.
- `policy.py` — two dual-threshold entry modes, both with debouncing +
  refractory period: `DualThresholdPolicy` gated on the **floor holder**
  (original Skantze & Irfan semantics) and `DemonstrationPolicy` driven by the
  **robot slot's own** p_now/p_future ("a human in my position would speak
  now"); which mode drives the live robot is still an open choice.
- `offline_demo.py` — predictive vs. reactive floor-entry latency comparison,
  plus the demonstration-agreement comparison (per-policy recall / false
  entries / lead time against the demonstrator's own entries).
- `reactive_baseline.py` — `SilenceBaseline`, the reactive fixed-silence baseline
  the predictive policy is compared against (relocated out of `legacy/` — it is
  core evaluation infrastructure, not the frozen TRP ablation).
- `feature_sets.py` — auditable input masks for candidate R0-Core, audio-only,
  strict visual, multimodal, and targeted ablations; disabled inputs stay in
  the canonical schema but are zeroed before fitting the normaliser.
- `baselines.py` — robot-observable ASD persistence and causal last-speaker
  probability baselines; neither reads gold VAD as an input.
- `predictions.py` — exports the sole delivered participant-wise VAP tensor as
  a two-phase, fail-closed CSV/metadata pair (`time × participant × 4 bins`), including
  separate per-bin temporal-horizon and gold-target availability, full label config,
  checkpoint/data/provenance
  hashes, and inherited smoke/override status; the auxiliary current-VAD head is
  never exported.
- `gaze_outer_experiment.py` + `gaze_posthoc_backend.py` — isolated
  post-hoc nested-LOGO runner/backend for the exact-eight schema-v3 C0/CM/CG
  matrix. It consumes only audited source views, saves out-of-group
  probabilities, and aggregates without new inference; every artifact is
  explicitly post-hoc, non-production and non-counted.
- `h2_matched_replay.py` — post-hoc exact-eight matched replay of
  inner-calibrated P1 predictive and causal acoustic-silence policies on the
  same held-out robot-entry opportunities; it is not a live robot/TTS test.

**Real-data assembly/readiness — `src/hri_turn_taking/data/assembly.py`** (CLIs
`vap-assemble`, `vap-audit`): exact-key, no-interpolation feature/VAD merge;
explicit `data_kind`; mandatory real-data provenance; non-bypassable confirmed
mapping and D3 gates; complete exclusive real/placeholder column declarations;
versioned intake/assembled artifact kinds whose sidecars bind the final CSV bytes,
ordered columns, parsed dtypes, and row count; per-feature source manifests
(column/participant/slot/source-column/source-hash/causality binding with exact
keyed-value replay); separate target and selected-feature readiness reports.
Synthetic inputs use a different versioned sidecar and are accepted only after
deterministic replay of the repository-owned generator contract; this is a
trusted-code reproducibility/integrity gate, not a cryptographic signature.

**Real-data intake — `src/hri_turn_taking/data/intake.py`** (CLI `vap-intake`,
added 2026-07-10): converts one HPE delivery directory
(`outputs/delivery/<group>__<scenario>/`) into a validated 25 fps wide table +
`.provenance.json`. Its artifact kind is pose-intake only: the sole real columns
are the present slots' head-yaw, `pose_valid`, and `pose_quality`; promotion of
any model input or gold target requires unique coverage by an assembly feature
manifest. Fail-closed gates (hardened 2026-07-11 per Codex review R1
in COORDINATION.md, R1.2 round 2026-07-12): reject `review_status != confirmed`,
D2 label-set violations ({R,A,B}/{R,A,B,C} only), missing offset stream;
`mapping.json` must bind to THIS delivery
(`participant_map` ≡ CSV — and a JSON object, `delivery`/`group`/`scenario` ≡
dir name); persons
listed in `mapping.json.unmapped_persons` (a JSON array of scalars — a bare
string is rejected, R1.2-3) are dropped by name — 议题 5② —
while undeclared unmapped persons still fail; `offset_quality.json` is
REQUIRED and fully asserted **with exact JSON types** (convention fullmatch,
`timebase_keys` an object with the stream as a KEY, final-offset
consistency, `self_check_ok` a real boolean, `error_bound_ms` a non-bool finite
number, status ∈ {ok, degraded} — R1.2-2); NaN/Inf and degenerate
facing vectors rejected. Several person fragments may map to one participant
(real HPE mappings do); simultaneous conflicting fragments (position >0.35 m
or facing spread >0.5) are rejected by default (`fragment_overlap_policy=
dominant` keeps the per-instant best-quality row, recorded in provenance).
Resampling is **strictly past-only** (Codex R1.4-2): each 25 fps frame holds
the LATEST source sample at or before it (zero-order hold, no interpolation),
valid only while that sample is ≤ max_gap_s old — zero lookahead, so a future
sample can never change an earlier frame (regression-tested with Codex's
adjacent-sample and beyond-gap counterexamples); staleness is bounded by
max_gap_s (0.2 s), which is perception buffer semantics the deployed robot
reproduces, not added pipeline latency. `max_gap_s` itself is gated (finite,
in (0, 1]) so a NaN cannot disable the gap logic. Fragment overlap winners are
chosen by **per-instant local quality** (lowest `facing_src_abs_yaw`, ties by
person id — never whole-track row counts, so a fragment's future cannot decide
a past frame), and simultaneous fragments must agree on position AND facing
direction (facing component spread > 0.5 conflicts, R1.4-3). Frames with no
sample within max_gap_s before them get
`pose_valid=0` + zeroed features (`present` stays static; validity propagates
through the yaw geometry — non-R needs R observed; R needs **≥ 2 valid centroid
members** computed from the valid subset only, per the Codex R1.2-5 ruling, with
the continuous valid-member fraction exposed as `P{i}_pose_quality`; a
zero-length bearing is invalid, not 90°). Applies the D3 offset (`shared = timestamp_sec + offset`, stream
key `6cam` per HPE); slot mapping P1=R / P2=A / P3=B / P4=C; derives
`P{i}_head_yaw_deg` (horizontal deviation of facing from the bearing to R; for
R itself, from the bearing to the others' centroid). All other columns are
zero-filled placeholders listed in the provenance report (`trainable: false`
until gold VAD merges in). Structural provenance is always enforced;
`vap-train` separately refuses target or selected-feature readiness failures
unless `--allow-untrainable` is used for a recorded smoke-only run. Real group
splits use only the gated provenance `group_id`; punctuation in `session_id`
cannot select or bypass the real-data path.

**Data schema — `src/hri_turn_taking/data/multiparty.py`:** wide frame table at
25 fps, columns `P{i}_<feature>`, up to 4 slots with `P{i}_present` masks
(3-party sessions pad slot 4). Input tier = `P{i}_asd_prob`, `P{i}_mar`,
`P{i}_gaze_robot`, `P{i}_head_yaw_deg`, `P{i}_gazed_at`, `P{i}_pose_valid`
(per-frame 0/1 observation-validity flag, added 2026-07-11 per Codex R1.1:
`present` stays STATIC "slot exists"; tracker dropouts set `pose_valid=0` with
zeroed features so missing ≠ a genuine 0° yaw), `P{i}_pose_quality`
(per-frame continuous [0,1] observation quality, added 2026-07-12 per the
Codex R1.2-5 ruling: R = valid-centroid-member fraction, non-R = 1 iff R
observed; deliberately NOT overloaded into the binary flag) + global
`doa_angle_deg`, `array_energy`, `f0_proxy`, `voiced` (**32 raw columns → 33
model dims** after the DoA sin/cos transform in `vap/dataset.py`; schema v2
implemented 2026-08-15). `voiced` is binary and `f0_proxy` must be zero where
`voiced=0`, preventing an unvoiced frame from colliding with voiced 100 Hz.
The optional Gaze schema v3 is a parallel path, not a replacement: it adds
`P{i}_gaze_robot_valid` for all four slots (**36 raw → 37 transformed
dimensions**), so each token has 8 own + 5 global dimensions and uses
`Linear(13→64)`. C0 masks Gaze value and validity, CM enables validity only,
and CG enables validity plus value; only `CG-CM` isolates Gaze content.
`validate_multiparty_dataframe` enforces the mask semantics at value level:
`present` binary and static per session, `pose_valid` binary, `pose_quality`
in [0,1], no validity on absent slots. `P{i}_vad` is gold tier — labels and
event eval ONLY, never model input.

**Frozen legacy ablation — do NOT extend.** The scikit-learn frame-level TRP path
lives under `src/hri_turn_taking/legacy/`:
`legacy/data/{schema,synthetic,dataset}.py`,
`legacy/models/{baseline,trainer,inference}.py`, `legacy/audio/gcc_phat.py`,
`legacy/vision/mouth.py`, `legacy/features/fusion.py`,
`legacy/realtime/pipeline.py`. Kept solely as a
low-cost comparison baseline. (`SilenceBaseline` was moved out to
`vap/reactive_baseline.py` — it is core eval, not legacy.)

**Shared / other:** `cli.py`, `config.py`, `configs/default.yaml`, `tests/`,
`docs/`. Architecture diagram: **`ARCHITECTURE.md`**
(Mermaid) — keep it in sync with `vap/model.py`, the policy thresholds, and the
feature schema.

## 6. Build / Test / Run

Environment: Windows, PowerShell primary. Python ≥ 3.10. PyTorch (CPU build) is a
core dependency, already installed in `.venv`. Do not assume a GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"      # core + pytest/ruff
```

Core VAP pipeline (the method that matters):

```powershell
hri-turn-taking vap-generate           # synthetic multiparty dataset (P{i}_ schema)
hri-turn-taking vap-train --data-kind synthetic              # train MultipartyVAP; prints frame + event metrics
hri-turn-taking vap-predict <data> --data-kind synthetic     # export sole (time, participant, 4-bin) output
hri-turn-taking vap-baselines <data> --data-kind synthetic   # ASD persistence + last-speaker baselines
hri-turn-taking vap-demo --data-kind synthetic               # two predictive policies vs. reactive silence
                                       # + demonstration agreement (--robot-slot)
hri-turn-taking vap-intake <delivery>  # real data: HPE delivery dir -> gated,
                                       # aligned 25 fps wide table + provenance
hri-turn-taking vap-assemble <base> <output> --feature <csv>  # exact-key merge
hri-turn-taking vap-audit --data <file-or-dir> --data-kind real --require-ready # readiness gate
```

Frozen legacy ablation (do not extend):
`hri-turn-taking generate-synthetic | train | predict <csv>`.

Run **both** before declaring work done:

```powershell
pytest                                 # full suite: causality, permutation equivariance,
                                       # absent-slot isolation, chunked-inference
                                       # equivalence, group-level splits, a training
                                       # smoke, ...
ruff check .                           # line-length 100; rules E, F, I, UP, B
```

Reference numbers on synthetic data (12×180 s, 20 epochs, CPU,
participant-structured transformer; events pooled over the 3 validation
sessions): frame mAP ≈ 0.94 (per-bin AP ≈ 0.98/0.95/0.91/0.82), shift/hold
AP ≈ 0.99, shift-prediction AP ≈ 0.91, overlap-resolution AP ≈ 0.99; the
predictive policy enters ~40–46% of held-out gaps at 0.08–0.09 s while the
1.5 s silence baseline never enters. The GRU low-cost variant scores on par
(frame mAP ≈ 0.94) — on synthetic data the gain comes from the participant
structure, not the temporal mixer. (The pre-restructure flat encoder scored
frame mAP ≈ 0.88, shift/hold AP ≈ 0.68, shift-prediction AP ≈ 0.72 under the
same budget.)

Demonstration-agreement demo (pseudo-demonstrator = slot P1, same checkpoint,
pooled over the 3 validation sessions): the demonstration policy enters 15/15
demonstrator-taken gaps with 0/57 false entries at ≈0.4 s lead over the
demonstrator's own onset; the floor-holder policy scores 13/15 recall but
20/57 false entries (it answers "is the floor free", not "is it my turn");
the reactive baseline enters nothing. Pooled robot-entry AP = 1.00 — a
synthetic ceiling, not a real-data claim.
These synthetic results validate software properties and pipeline wiring only;
they do not test or support H1/H2.

## 7. Conventions

- All timestamps in seconds (float) on the shared session timeline; zero point
  = the **mic A audio axis** (cross-project decision D3, `../COORDINATION.md`,
  2026-07-08 — supersedes the earlier "session sync beacon" wording).
  Per-stream offsets onto this axis are measured by Head_pose_extraction and
  delivered per scenario as offset.json (`streams` format) with
  alignment-error estimates (target ±40 ms; exceedances don't block intake but
  must be reported in the thesis).
- Feature files: one parquet/npz per session per modality, aligned to a 25 fps
  grid; column naming `P{i}_<feature>` (e.g. `P2_gaze_target`).
- **Real-data drop-in:** produce the same wide table from extracted features and
  feed `vap-train` directly; only `input_dim` changes if features are added.
- **Real-data slot convention (2026-07-10):** participant labels map to slots as
  `P1 = R` (robot-player / demonstrator), `P2 = A`, `P3 = B`, `P4 = C`;
  3-person groups set `P4_present = 0`. Intake must read the delivered
  `participant_map.csv` per scenario, apply this mapping, and record it in a
  session manifest (`participant_to_slot`) for verification — never assume a
  positional order; reject deliveries whose `mapping.json.review_status` is not
  `confirmed`. `vap-demo --robot-slot` is therefore slot 1 (P1) on real data.
- **Real-data session naming:** `<group>__<scenario>` remains the human-readable
  convention (e.g. `g03__s2`), but code must not infer data kind or group from
  that string. Real splits use the structurally gated provenance `group_id`, so
  all videos of a recording group stay on one side. `--group-delimiter` is only
  for explicitly declared synthetic data.
- New modelling work goes in `src/hri_turn_taking/vap/`; the legacy sklearn TRP
  path is frozen.
- Synthetic features must be *probabilistic noisy correlates* of the speech state
  with genuine anticipatory cues (turn-final prosodic lowering, speaker gaze
  redirection, pre-onset articulation), gated by `cue_reliability`. They must
  **never** be deterministic functions of the labels (an earlier scaffold had that
  leakage — do not reintroduce it).

## 8. Project Status, Data & Team

- **Status (2026-08-18):** the immutable dissertation source remains exactly
  ten scenes across four groups and 78,582 common-window rows. Sealed identity,
  audio/time-map, human-reviewed-Transcript-derived VAD, causal acoustic and
  unified A/H/A+H assembled artifacts are complete for all ten scenes under
  strict replay/review; the original files were not rewritten. Before outcome
  inspection, the owner placed `05_15__4` (no A speech in the Transcript) and
  `05_15__5` (A/R extremely sparse) on manual-review hold. The deadline
  evaluation therefore froze the other exact eight scenes (64,299 rows, four
  groups), seed 13, `max_epochs=5`, and A/H/A+H × four outer folds. All 12
  nested-LOGO units completed and the saved-prediction aggregate passed
  independent audit. A separate post-outcome `max_epochs=15` sensitivity run
  also completed 12/12 paired units; its only protocol change is the epoch
  budget, and it does not reverse the scientific conclusions. A separately
  audited post-hoc exploratory matched replay compared inner-calibrated P1
  predictive and causal acoustic-silence policies on the same 303 robot-entry
  opportunities (111 positive/192 negative). At five epochs, predictive
  recall/FPR/balanced accuracy were 0.4324/0.5156/0.4584 versus reactive
  0.5766/0.3594/0.6086; at 15 epochs predictive values were
  0.6306/0.6458/0.4924. Both-hit positives showed descriptive predictive leads
  of mean 374 ms (`n=29`) and 355 ms (`n=44`), both with 400 ms medians, but
  this noncanonical event-CSV derivation and the gap-only sensitivity do not
  show a stable overall advantage. H2 is not supported and remains
  unestablished; no robot/TTS/live latency was evaluated. A causal voiced-only
  reactive diagnostic remains a separate fixed-rule check. Every result is
  explicitly `production_eligible=false` and `counted_eligible=false`. H1 is
  not supported under this limited head-pose operationalisation. A separate
  exact-eight schema-v3 Gaze post-hoc run has now completed seed 13,
  `max_epochs=15`, C0/CM/CG × four outer folds (12/12) over 64,299 rows and
  four groups. Its independently audited saved-probability aggregate has
  SHA-256
  `0b4f2c5e3deafafa0d23cdb4822f328d95af5c0788ba662cd9c04d7414955cea`.
  Pooled frame mAP is 0.2717/0.2461/0.2843; the primary content contrast
  `CG-CM=+0.0381` is positive in all four folds, while
  `CM-C0=-0.0256` records the observability/missingness effect
  (`CG-C0=+0.0126` total package). This is suggestive single-seed post-hoc
  evidence and does not reclassify H1 as supported. P1/R has no direct Gaze
  observation; the actual extractor variant/weights/configuration are not
  fully bound, exact upstream regeneration is unavailable, and
  implementation-level counterfactual causality is unverified. ASD remains
  deferred. The current Facial export is dropped
  from the dissertation experiment route and retained as source evidence only.
  D3 sensitivity, additional seeds, GVAP and physical Furhat remain deferred. Read
  `docs/TRAINING_DATA_STATUS.md` for the live ledger and
  `docs/thesis_outer_contract.md` for the claim boundary.
- **HPE head-pose handoff (D1/D2 completed 2026-07-27; D3 closed out
  2026-07-28):** head-pose D1 and participant-mapping D2 are complete and
  consumable for **21 formal scenes across 5 recording groups**, with all 21
  mappings human-confirmed (every scene maps four participants A/B/C/R;
  `world_persons.csv` uses the 23-column header with orientation-gate audit
  columns). The authoritative cross-project data contract is
  `../Head_pose_extraction/outputs/delivery/README.md` (+ `ROLLOUT_STATUS.md`);
  the membership list comes from `HEAD_POSE_DELIVERY_INDEX.md` and is mirrored
  with per-scene offsets and timing readiness in
  `configs/hpe_delivery_inventory.json`. **14 scenes publish a usable,
  self-checked `6cam` D3 offset**, covering all 5 groups: `03_20__{4,5}`,
  `03_24_1__{2,3}`, `03_26__3`, `05_14__{2..5}` (mic_A+6cam only — sufficient
  for head-pose↔audio alignment), `05_15__{2..6}`. 9 of the 14 are `degraded`
  and 8 exceed the ±40 ms target (max ±104.4 ms). For the selected ten-scene
  corpus, the original publication-grade plan keeps all ten centrally, adds a
  paired six-scene/four-group ≤80 ms sensitivity run, reports the three-scene
  ≤40 ms subset descriptively, and applies frozen ±error-bound shifts to
  head-pose conditions. This study remains unexecuted. The completed deadline
  matrix used the exact eight-scene view and central offsets only; in that view
  the ≤80 ms membership is five scenes because `05_15__5` is excluded. The 7
  scenes without a physical sync bridge (`03_20__{2,3,6}`, `03_26__{4,6,7}`,
  `05_14__6`) stay on local six-camera time and must not borrow offsets.
  `03_24_2__2` through `03_24_2__6` are timing-only and are not part of the
  21 head-pose deliveries.
- **Corpus design (recorded; user-confirmed 2026-07-05/07):** 3–4 person
  scenario-based sessions (receptionist-style service-robot scenarios — e.g.
  hotel front desk, station information desk). **One participant role-plays the
  robot receptionist**, standing where Furhat will be deployed; the robot role
  is an *identity only* — the player was instructed to respond naturally as a
  human, with no attempt to imitate robotic speech. Conversation flows both
  human↔human and human↔robot-player within each scenario. The other 2–3
  participants stand face-to-face with the robot-player. Camera split: ≈3 of
  the 6 calibrated cameras face the robot-player (the demonstrator's
  behaviour), ≈3 face the others (≈ the robot's-eye view). Everyone wore a lavalier, **including the robot-player**,
  so the robot slot has gold VAD like any other slot. Each scenario video runs
  ≈3–7 min (total video/group count still to confirm). **Audio inventory
  (resolved 2026-07-05; source semantics closed 2026-08-15): no mic array.**
  Recorded audio = per-participant DJI Mic Mini wireless lavaliers (per-person
  tracks — exactly the gold-tier VAD source). `master_aligned.wav` is a
  four-channel container replayed exactly as a permutation of those tracks;
  the participant-video reference WAV is a shorter mux/extraction derivative,
  not an independent physical-position microphone. `array_energy`, `f0_proxy`
  and `voiced` use the deterministic equal-weight mono master mix as a declared
  close-talk surrogate; `doa_angle_deg` has **no
  real-data source** — **resolved (user, 2026-07-10): dropped on real data**
  (no geometry simulation — oracle-leak risk; per-face ASD already carries
  who-is-speaking visually); the column stays as an inert constant so the
  synthetic/ablation layout is unchanged (see §3). Document the close-talk→far-field shift between
  training audio and the robot's runtime microphone (precedent: VAP models
  transfer across acoustic domains — trained on telephone speech, deployed on
  robot mics). If one group recorded several scenario videos, train/val splits
  must keep a group's videos together (person-disjoint), not merely split by
  video.
- **Robot slot = supervised demonstration:** the robot-player occupies a regular
  `P{i}` slot, so the model's predictions for that slot are supervised "what a
  human in the robot's position would do" — the literal *Learning from Humans*
  reading; precedent: Skantze & Irfan trained on human-human dialogue and fed
  TTS into the robot channel at deployment. Deployment substitutes Furhat
  self-state features for that slot: `asd_prob`↔TTS-active, gaze↔commanded gaze
  target, `head_yaw_deg`↔commanded head pose, `mar`↔lip-sync parameter,
  `gazed_at`↔aggregate of the others' gaze-at-robot. Honest limitations to keep
  in the thesis: human-plays-robot ≠ robot (human prosody/timing; people behave
  differently toward real robots; Furhat TTS startup 200–400 ms sits on top of
  any policy trigger — report the two separately, with the demonstrator's own
  entry latency as the human ceiling); the learned policy is
  receptionist-domain, matched to the deployment scenario.
- **Sensors (as actually recorded; corrected 2026-07-07):** 6 calibrated
  monochrome (grayscale) cameras surrounding the space (+ calibration data);
  2 **RealSense D435** RGB-D cameras recording RGB video plus depth `.npz`
  files — **not calibrated against the 6 mono cameras** (no shared extrinsics;
  fusing their depth with the multi-view reconstruction would need a separate
  registration step); per-participant
  **DJI Mic Mini wireless lavaliers** (gold audio) — four per-person tracks plus
  their derived four-channel aligned master. **No microphone array or
  independent robot-ear/room mix was recorded** (the planned
  4-channel ReSpeaker was not used), so `doa_angle_deg` has no real-data
  source; the sample-replayed equal-weight master mix is the approved
  close-talk proxy for global acoustic features. Sync (per COORDINATION.md D3): shared time axis = mic A
  audio; per-stream offsets measured by Head_pose_extraction (RGB-D↔audio via
  the RealSenses' audio-twin `*_rgb_audio.mp4` cross-correlation; 6cam↔RGB-D
  via head-motion correlation — the 6cam AVIs carry no audio), delivered per
  scenario as offset.json with error estimates; the recorded audiovisual
  beacon (clap/beep) stays as a manual cross-check. Target alignment within
  ±1 frame (40 ms).
- **Team feature-extraction split (updated 2026-07-27):** **Yixuan (this
  repo's owner)** — head pose (**D1/D2 complete for 21 scenes**),
  speaking-towards / addressee (a team-level deliverable only — not consumed
  by this dissertation since 2026-08-03). Others (their
  outputs are available for Yixuan to use): synchronization, transcript, and
  speaker detection (Michelle's block); Zobaid — whole-body 3D; Tomas — facial
  keypoints, gaze direction / interaction (MCGaze). Speaker detection is **no
  longer Yixuan's task** (moved off his plate since the 2026-07-05 split).
- **Annotation requirements Yixuan must still secure (even where the tier is
  owned by someone else):** the speaker-detection / VAD tier needs an
  **overlap-type tier** distinguishing `backchannel` vs `competitive`
  (barge-in) — required for the RQ3 overlap analysis; coordinate this with
  whoever finalises speaker detection. The **speaking-towards / addressee
  tier** stays Yixuan's team-level task, but since 2026-08-03 the dissertation
  does not consume it (no model input, no ablation, no addressee-accuracy
  metric); if produced for the team it should still use a target set that
  includes the robot-player — {each participant, robot, group}. For the
  completed deadline route, VAP labels are the exact
  human-reviewed-Transcript-derived VAD artifacts; aligned lavaliers are
  full-corpus consistency evidence rather than a second annotation. A future
  separate per-lavalier speaker-detection tier may still use semi-automatic
  Silero/pyannote → ELAN → manual correction.
- **Feature status and remaining gaps:** causal energy + F0 + global `voiced`
  from the combined mix is complete for all ten scenes under
  `data/processed/acoustic_features/`, with the owner-frozen `0.45` threshold,
  strict causal replay and independent review. The formal ten-scene A/H/A+H
  assembly is complete under `data/processed/formal_a_h_ah_v1/`; the mix is
  still a declared close-talk surrogate rather than robot-ear audio. CPC stays
  a gated optional ablation. A causal
  pretrained-ASD speaking probability per face
  remains a feature gap; the
  official stock Light-ASD/LR-ASD implementations are future-dependent and may
  only be used as offline diagnostics or annotation proposals, never as a
  zero-lookahead counted input. Its read-only source preflight remains R3/3
  blocked and produced no model, weights, probabilities or features. Current
  Gaze evidence remains nullable 2D `robot` / `participant` / `elsewhere`,
  not an existing 3D `gazed_at` derivation. The exact-eight schema-v3
  C0/CM/CG post-hoc experiment is complete and audited, but P1 has no direct
  observation, exact upstream regeneration is unavailable, and
  implementation counterfactual causality is unverified. Silence/pause/overlap
  statistics are auto-derived from VAD for evaluation.

## 9. Guardrails / Do-Not

- **Ethics**: raw audio/video must never leave the local machine or be committed
  to git. Per `.gitignore`, local data, outputs, models, logs, report sources,
  recovery snapshots, and agent runtime state are excluded. Only the documented
  data-directory scaffold is tracked. Keep it that way.
- Do not extend the frozen legacy sklearn TRP path; treat it as a fixed baseline.
- Do not change any item in §4 (Key Design Decisions) without asking the user.
- Report balanced metrics + event tests, never raw accuracy alone.
- torch stays CPU-only here.

## 10. Verify Your Work

- Run `pytest` (must stay green, incl. the causality test that perturbs future
  frames) and `ruff check .` before reporting completion.
- If you touch `vap/model.py`, the policy thresholds, or the feature schema,
  update `ARCHITECTURE.md` to match.
- State explicitly what you ran and its result; if a step was skipped, say so.

## 11. Environment Notes

- Use the repository-configured `origin` for GitHub operations. Do not record
  workstation-specific executable paths, credentials, or account state here.
- Furhat SDK (Kotlin / remote API) ↔ ROS2 bridge is the eventual deployment path;
  add it only once the local hardware/runtime path is confirmed.
- Perception toolchain (team-wide): MediaPipe Face Mesh (landmarks), MCGaze
  (gaze), CMU `ssp` toolkit (body/gesture). ML pipeline is Python + PyTorch.

## 12. Scope / Priority (if time runs short)

1. **Core deliverable**: the offline multimodal prediction model (H1 + part of H2).
2. Second: the Furhat integration demo.
3. Stretch: the formal user study (can shrink to a 2–3 group pilot).

## Key References

- Ekstedt & Skantze 2022 — VAP (arXiv:2205.09812; code:
  github.com/ErikEkstedt/VoiceActivityProjection)
- Inoue et al. 2024 — Real-time continuous VAP (arXiv:2401.04868)
- Skantze & Irfan, HRI 2025 — VAP on Furhat, dual-threshold policy (arXiv:2501.08946)
- Elmers et al., Interspeech 2025 — Triadic audio-only VAP (arXiv:2507.07518)
- Onishi et al., IEICE 2025 — Multimodal dyadic VAP (gaze/AU/head pose/gesture)
- Heo et al., Interspeech 2025 — Gaze-enhanced triadic turn prediction (arXiv:2505.13688)
- Liao et al. — Light-ASD (CVPR 2023) / LR-ASD (IJCV 2025),
  github.com/Junhua-Liao/Light-ASD
- Abbo et al. 2025 — Reactive baseline system (arXiv:2503.15496)
