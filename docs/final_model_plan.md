# Final Model Plan — Participant-wise Multimodal VAP

> Owner-approved consolidation, 2026-08-03. This file records the frozen
> architecture and data plan for the counted real-data experiments, on top of
> [`PROJECT_SCOPE.md`](PROJECT_SCOPE.md) (scope authority),
> [`methodology.md`](methodology.md) (protocol detail), and
> [`experiment_plan.md`](experiment_plan.md) (study design). Those documents
> have been amended in place with the 2026-08-03 decisions; where an older
> passage still reads as an open option, the `PROJECT_SCOPE.md` §8 decision
> log and this file take precedence. Nothing here weakens the tier, causality,
> or group-disjoint split contracts.
>
> Execution status, 2026-08-18: the ten-scene formal target, acoustic and
> A/H/A+H assembly route is complete. The separate deadline evaluation used an
> owner-frozen exact eight-scene view and completed all 12 seed-13 nested-LOGO
> units plus an independently audited aggregate. Its artifacts remain
> explicitly non-production and non-counted under
> [`thesis_outer_contract.md`](thesis_outer_contract.md). A separate
> exact-eight schema-v3 C0/CM/CG Gaze experiment has also completed all 12
> seed-13, 15-epoch units and an independently audited saved-probability
> aggregate; it remains post-hoc exploratory evidence and does not alter the
> frozen deadline result.

## 1. Decisions frozen on 2026-08-03 and 2026-08-16

1. **speaking_towards is out of the dissertation scope.** It is not a model
   input, there is no `-speaking-towards` ablation, and the
   addressee-accuracy evaluation is deferred out of scope. The current Gaze
   export supplies only nullable 2D `robot` / `participant` / `elsewhere`
   evidence, not an existing 3D-derived `gazed_at` feature. Any later
   gaze-target input remains gated and is covered by the `-gaze` removal. The
   team-level speaking-towards deliverable is unaffected — only its consumption
   here is removed.
2. **Core acoustic representation = causal log-energy + F0 + `voiced` from
   the single-channel combined mix.** The versioned extractor recorded in the
   formal acoustic and deadline run manifests uses: 16 kHz resample;
   `array_energy` =
   `log(max(RMS, 1e-6))` over the trailing 40 ms `(t-0.040, t]`; F0 by
   normalized autocorrelation over 50–500 Hz on the trailing 80 ms
   `(t-0.080, t]` after mean removal; `voiced = 1` when the normalized
   autocorrelation peak meets the owner-frozen voicing threshold **0.45**
   (decision 2026-08-16, before formal feature publication and before outcome
   inspection); `f0_proxy =
   log2(F0 / 100 Hz)` meaningful only
   where `voiced = 1`, `0` where `voiced = 0`. All windows right-aligned at
   the frame timestamp — no centred framing, no bidirectional smoothing, no
   full-utterance normalisation — and any backend (e.g. openSMILE/Praat)
   must reproduce exactly this causal windowing and pass the extractor
   causality check. CPC is a gated optional ablation after the core result,
   never the core.

   The read-only real-source wrapper first replayed all ten canonical audio
   contracts and 78,582 exact analysis ticks to close feasibility. The separate
   formal builder has since published versioned artifacts, evidence, manifests
   and independent reviews for all ten scenes under
   `data/processed/acoustic_features/`. Strict replay binds the `0.45`
   threshold, causal windows and equal-weight master mix. The mix remains an
   explicitly reported close-talk surrogate, not robot-ear audio.
3. **Schema v2 (implemented 2026-08-15):** the global `voiced` 0/1 column is
   present. The current schema has 32 raw columns and 33 model dimensions after
   DoA sin/cos; each participant token has 7 own + 5 broadcast global
   dimensions and uses `Linear(12→64)`. The
   checkpoint feature-schema guard must reject v1 checkpoints with the
   existing "retrain" error; `ARCHITECTURE.md` is updated together with the
   code change, not before.
4. **`aux_vad_weight = 0.0` for the first formal run.** The auxiliary
   current-VAD head stays in the architecture at zero weight; the
   0.0-versus-0.3 paired comparison remains an unapproved candidate ablation.
5. **First-run supervision = human-reviewed-Transcript-derived VAD.** Rasterize
   the exact hash-bound existing human annotations after only the four targeted
   impossible-interval decisions. Lavaliers are consistency/oracle evidence;
   no 1,094-event relisten, 50-item-per-scene blind audit, or seven-day retest is required.
6. **Four-day deadline matrix = `A / H / A+H`.** ASD, Gaze, Facial/MAR,
   GVAP and physical Furhat work are deferred. The trusted local
   single-user/single-writer publication boundary is accepted, with residual
   concurrent or malicious path-replacement risk disclosed. The completed
   deadline artifacts retain `production_eligible=false` and
   `counted_eligible=false`; this scientific matrix did not promote the
   repository production backend.

## 2. Goal restated

One learned primary output — `q[t, i, k] = P(participant i voice-active in
future bin k | inputs ≤ t)`, shape `time × participant × 4` (bins 0.2 / 0.6 /
1.2 / 2.0 s) — evaluated as a group-disjoint real-data comparison of the
deadline `A / H / A+H` conditions. The named reactive/policy and model
baselines remain separate experiments; the post-hoc matched H2 replay has now
run, while unexecuted baselines remain deferred. The robot
slot (P1 = R) is ordinary supervised demonstration: its predictions read as
"would a human in the robot's position speak now". Everything else (p_now /
p_future, Shift/Hold, Shift Prediction, Next Speaker, overlap resolution,
robot entry, dual-threshold policy) is a deterministic readout of that tensor.

## 3. Available data → schema mapping (owner inventory, 2026-08-03)

Locations were confirmed on 2026-08-03 by a read-only scan of the authorised
team share and this repository. The local corpus root is not version controlled.

| Source (in hand) | Schema target | Derivation status | Time axis |
|---|---|---|---|
| HPE head pose (`world_persons.csv`, 21 upstream scenes) | `P{i}_head_yaw_deg`, `P{i}_pose_valid`, `P{i}_pose_quality` | ten dissertation scenes have a sealed Transcript-semantic crosswalk and independently replayed `vap-intake` v2 pose bases (79,159 rows total) | 6cam local → scene-specific `6cam` D3 offset; the ten selected scenes span four recording groups and retain their `ok`/`degraded` quality |
| Per-participant transcript (`<date>/transcript/<n>/timeline_video.json`) | first-run **human-reviewed-Transcript-derived VAD** | complete: the four impossible in-window intervals received explicit owner decisions; the other 1,090 events are inherited rather than claimed as newly re-audited; ten exact 25 Hz target artifacts/manifests pass strict replay and aligned-lavalier consistency checks | owner-frozen 30/1 non-drop `HH:MM:SS:FF` convention + scene `offset.json.offset_sec` → mic-A axis; the missing declared edited MP4 means 30 fps is not claimed as a measured video-header fact |
| Gaze export (`gaze_features_export/`) | post-hoc `P{i}_gaze_robot` + `P{i}_gaze_robot_valid` only | the repaired read-only preflight passes 10/10 scenes and confirms high-coverage nullable A/B/C evidence. The fixed exact-eight schema-v3 view has 64,299 rows/four groups. Seed 13, `max_epochs=15`, C0/CM/CG × four folds completed 12/12 and the aggregate SHA-256 is `0b4f2c5e3deafafa0d23cdb4822f328d95af5c0788ba662cd9c04d7414955cea`. Pooled frame mAP is 0.2717/0.2461/0.2843; `CG-CM=+0.0381` is positive in all four folds, while `CM-C0=-0.0256` exposes the missingness effect. The export cannot identify a specific `gazed_at` participant and has no R/P1 direct source row. Actual upstream variant/weights/configuration are incompletely bound, exact regeneration is unavailable, and implementation counterfactual causality is unverified | bound six-camera frame axis → the same scene D3 chain, then cropped to the common analysis window |
| Facial export (`facial_features_export/`) | no dissertation model input | retained as source evidence only. No mouth landmarks are exported, `mouth_motion.csv` is not MAR and remains forbidden, `emotion_argmax` is unvalidated, and the eight anonymous AU indices lack a verified semantic/order/extractor binding. Facial is therefore dropped from the remaining experiment route rather than zero-filled or over-interpreted | same accepted Gaze window; upstream bytes remain preserved |
| Per-person lavalier audio (`audio/aligned_audio/{A,B,C,R}_aligned.wav`, **48 kHz PCM16**, strong bleed) | Transcript-label consistency diagnostics + oracle-only ablations | ten formal source/sync manifests and ten formal gold-alignment time maps pass replay; 13 channels use non-zero constant corrections, while `05_14__2` R remains identity-mapped with an explicit coarse warning and successful independent 5 ms check | mic-A-family sample axis; current bytes/header/hashes are bound per scene and original WAVs are never rewritten |
| `master_aligned.wav` | four-channel aligned-lavalier container, not a measured robot-ear channel | all ten masters replay exactly as permutations of A/B/C/R; declared channel order conflicts in eight scenes; the 2026-08-15 source decision selects the sample-replayed equal-weight mono mix and reports it as a close-talk surrogate | aligned-lavalier/mic-A axis; gold-only per-channel corrections are never applied to this model input |
| — not yet in hand — | `P{i}_asd_prob` | stock Light-ASD/LR-ASD were source-audited as future-dependent and are ineligible for counted zero-lookahead input; requires a separately validated causal model/head, combined-mix audio, identity-stable face tracks and an owner decision on `P{i}_asd_valid` | follows its chosen video stream (+ that stream's D3 offset) |
| — no real source — | `doa_angle_deg` | inert constant on real data (decided 2026-07-10) | n/a |

**Sufficiency verdict (updated 2026-08-18):** no new recording is required for
the completed deadline A/H/A+H evaluation. All ten selected scenes now have
formal targets, causal acoustic artifacts and audited unified tables under
`data/processed/formal_a_h_ah_v1/`, totalling 78,582 rows across four groups;
the 577 pose rows outside joint pose/audio coverage remain excluded rather than
labelled as silence. The owner placed `05_15__4` and `05_15__5` on manual
review hold before outcome inspection, so the frozen deadline view contains
the other exact eight scenes and 64,299 rows. This closes the MSc deadline data
route only. The single-seed post-hoc Gaze isolation is also complete, but a
reviewed production backend, D3 sensitivity, multi-seed replication, causal
ASD, and exactly regenerable/counterfactually verified Gaze remain open.

## 4. Human-reviewed-Transcript-derived VAD (first-run recipe)

The exact ten Transcripts are inherited as existing human annotations for
speaker identity, overlap, quiet speech and pauses. Apply only deterministic,
exact-preimage syntax/timecode normalisation and the four explicit targeted
decisions, map intervals to the mic-A axis, and rasterize half-open intervals
onto the exact 25 fps grid. The resulting target must be named
**human-reviewed-Transcript-derived VAD**, not sample-perfect lavalier VAD.

Aligned per-participant lavaliers provide full-corpus automatic consistency
diagnostics and clearly named oracle evidence, never a second manual
reannotation or a model input. No repeat listening of all 1,094 events,
50-item-per-scene blind audit, or seven-day retest is required. Report 30 fps
quantization, within-utterance pauses and the absence of long-term intra-rater
reliability as limitations.

## 5. Model architecture (final; core unchanged)

Dropping speaking_towards and freezing the acoustics changes no structural
element. The counted-run model is the implemented `MultipartyVAP`, with the
scaffold defaults now proposed as the formal configuration (any change
creates a separately named experiment):

- one token per participant: 7 own features + 5 broadcast global dims
  (energy, f0_proxy, voiced, DoA sin/cos) = 12 dims → shared
  `Linear(12→64) + LN + GELU`, multiplied by the static presence mask;
- 2 pre-norm residual blocks, each: causal temporal MHA (4 heads, ALiBi
  relative bias, 250-frame banded window; weights shared across
  participants) → cross-participant MHA (absent slots masked as keys) →
  FFN 64→256→64; dropout 0.1;
- final LayerNorm → shared per-token heads: `bin_head` `Linear(64→4)`
  (the sole delivered output) and `vad_head` `Linear(64→1)` (auxiliary,
  weight 0.0 in the first formal run);
- ≈135k parameters (134,853 in the current v1 schema; 134,917 after the
  voiced column); slot permutation equivariance, strict causality, and
  chunked-inference equivalence (left context = n_layers × window) stay
  unit-tested invariants;
- the one-layer GRU temporal mixer is retained as baseline B2 under matched
  tokens, features, folds, and seeds.

## 6. Objective, optimisation, protocol

Masked, per-bin `pos_weight` BCE over present slots and complete 2 s
horizons; `pos_weight` from training groups only (clipped [1, 10]);
`aux_vad_weight = 0.0` — note the implementation default is `0.3`
(`trainer.py` and the CLI), so the first formal run **must explicitly
override it** and the manifest must record the effective value. AdamW 3e-4
(no decay on norms/biases), 10% warmup + cosine, gradient clip 1.0, early
stopping on inner-validation macro mAP — these are the code defaults and
become the formal configuration when recorded in the run manifest. Splits are
group-disjoint. With four eligible groups now known, the protocol was frozen on
2026-08-15 before model outcome inspection: four outer leave-one-group-out
folds; within each, three inner leave-one-group-out folds choose the epoch;
then reinitialise and fit all three outer-training groups for the median
selected epoch before the untouched outer group is evaluated. The broader
publication-grade plan retains seeds 13/42/73 paired across conditions. The
outer-test orchestrator now freezes
and contract-tests both the ten-scene primary and six-scene/four-group
`d3_le80` variants, with fresh D3 provenance checks and training-only statistic
replay. It currently registers only a code-owned smoke backend, which is never
counted. An isolated PyTorch backend boundary has passed R2 review on synthetic
fixtures: it replays exact session ticks/content, consumes runner-supplied
statistics without refitting, covers full-session OOF and writes strict
non-counted artifacts. It remains deliberately unregistered and rejects real
data. A separate read-only evidence foundation captures live
source/environment bytes, replays canonical real ticks, and performs
protected-column, past-only central/±D3 pose reassembly. At that pre-publication
stage, a strict in-memory build covered all ten scenes, 78,582 ticks and four
groups and correctly kept target, audio-only and audio-plus-head-pose readiness
false because formal labels/acoustics were then absent. Downstream formal
artifacts and assembly were completed later; the read-only foundation itself
still has no publisher and is permanently non-counted. The owner accepted the
trusted local
single-user/single-writer publication boundary after three repair/retest rounds;
residual concurrent/malicious replacement risk remains disclosed. Formal run-manifest/spec
binding, batch/warmup-cosine execution, the atomic run index, backend registration and a
reviewed result publisher
remain hard gates for production promotion; the existing trainer remains
development-only.

The separate deadline runner in `thesis_outer_contract.md` did not weaken or
promote those gates. It froze the exact eight-scene, 64,299-row view, seed 13
and `max_epochs=5`, then completed A/H/A+H × four outer folds (12/12 units).
Its aggregate was independently recomputed from the saved out-of-fold
predictions and audited. Every artifact remains
`production_eligible=false` and `counted_eligible=false`.

## 7. Experiment matrix (final)

Baselines: B0 last-speaker persistence (the genuinely causal ASD version is deferred — gated on
a separately validated causal model/head; stock Light-ASD/LR-ASD cannot supply
it; the gold-VAD variant is named `oracle`), B1 reactive silence (the
calibrated causal combined-mix speech detector is implemented by the post-hoc
matched replay in §8 item 6 but remains explicitly non-counted;
an exact-eight `voiced`-only post-hoc diagnostic may be reported separately but
is not a complete speech-VAD or H2 baseline; a replay driven by gold VAD is
named `oracle reactive replay`), B2 structured GRU, B3 audio-only Transformer (expected
participant-symmetric — an intentional attribution lower bound).

Shared prerequisites for **every** condition: the primary schema v2 (§8.1) for
A/H/A+H or the parallel Gaze schema v3 for C0/CM/CG, the audited gold labels
(§8.3), D3 offsets for every stream involved, and ≥ 3 eligible groups. On top
of those:

| ID | Inputs | Additional prerequisites |
|---|---|---|
| `A` | energy + F0 + voiced (DoA inert) | acoustic extractor (§8.2) |
| `H` | head yaw + pose validity/quality | — (head-pose intake exists) |
| `A+H` | `A` + `H` | minimum multimodal condition; deadline result is non-counted |
| `C0` (completed post-hoc) | A+H under the Gaze-capable schema, with Gaze value and validity both disabled | schema-matched anchor; pooled frame mAP 0.2717 |
| `CM` (completed post-hoc) | A+H + real Gaze-validity mask, Gaze value disabled | missingness/observability control; pooled frame mAP 0.2461; `CM-C0=-0.0256` |
| `CG` (completed post-hoc) | A+H + A/B/C `gaze_robot` + Gaze-validity mask | content condition; pooled frame mAP 0.2843; primary `CG-CM=+0.0381`, positive in all four outer folds |

The minimum paired scientific matrix is the `A / H / A+H` comparison; its
deadline version is complete under the non-production/non-counted contract.
The former `V`/MAR/`gazed_at` matrix is retired for the dissertation because
the delivered Facial/Gaze bytes cannot support those claims. The separate
`C0/CM/CG` Gaze-only post-hoc route is now complete and audited, but remains
single-seed, non-production, non-counted and outside the frozen deadline
matrix. Causal ASD is still a
separately owner-gated future condition. Gated optional
extras (each needs separate approval): aux 0.0-vs-0.3, CPC-vs-energy/F0,
mic-array DoA on synthetic only. No speaking-towards condition exists
anywhere in the matrix.

## 8. Work queue implied by this plan

1. **Schema v2 — complete 2026-08-15:** global `voiced` is present in the
   multiparty schema, validation, dataset transform, feature-set masks and
   intake placeholders; checkpoint loading asserts the exact current feature
   schema, and regression tests cover the voiced/F0 contract and leakage.
2. **Acoustic extractor — complete for the ten-scene source corpus:** the owner
   froze `0.45` on 2026-08-16 before outcome inspection. The formal builder,
   exact causal artifacts, manifests and independent reviews now cover all ten
   scenes under `data/processed/acoustic_features/`.
3. **Transcript-derived target builder — complete for all ten scenes:** the four
   targeted impossible intervals have owner decisions, and exact 25 Hz
   human-reviewed-Transcript-derived VAD artifacts pass replay. The aligned
   lavaliers are automatic consistency evidence; the old blind-audit candidate
   is not the first-run label source.
4. **Gaze/Facial evidence and post-hoc run — complete within its declared
   exploratory boundary:** identity binding and the 10-scene read-only Gaze
   replay are complete. The parallel schema-v3 publisher created and
   re-audited the exact-eight 64,299-row/four-group source; C0/CM/CG masking and
   the validity-aware train-only normaliser are tested. Seed 13,
   `max_epochs=15`, C0/CM/CG × four outer folds completed 12/12, and the
   independently audited aggregate SHA-256 is
   `0b4f2c5e3deafafa0d23cdb4822f328d95af5c0788ba662cd9c04d7414955cea`.
   The primary `CG-CM` frame-mAP delta is +0.0381 and positive in every fold;
   `CM-C0=-0.0256` records the observability/missingness effect rather than
   Gaze content. Preserve nullable A/B/C
   `gaze_robot` evidence; never
   misdescribe it as 3D gaze, participant-specific `gazed_at`, addressee or
   speaking-towards, and never fill missing R observations without an explicit
   validity bit. P1/R remains directly unobserved. The actual extractor
   variant/weights/configuration are incompletely bound, exact upstream
   regeneration is unavailable, and implementation-level counterfactual
   causality is unverified; stronger claims require a new preregistered,
   multi-seed protocol after closing those gaps. Facial remains archived but is
   dropped from dissertation modelling: no MAR can be reproduced, anonymous AUs
   lack the required semantic/extractor binding, and `mouth_motion.csv` is not a
   substitute.
5. **Causal-ASD route** for `asd_prob` — a read-only ten-scene source
   preflight is implemented but formal acceptance remains **BLOCKED at R3/3**.
   Fixture provenance can masquerade as formal provenance, and ancestors above
   the configured authority root are not reparse-checked. The three-round
   repair/retest limit is exhausted pending owner review. No ASD model or
   weights were downloaded, no model was executed, no probability or feature
   was generated or published, and insufficient Cam6 evidence for R fails
   closed. The official stock Light-ASD
   (`ed38c232...`) and LR-ASD (`1b6dcd2...`) implementations were audited on
   2026-08-15 and both use symmetric temporal convolutions plus a
   backward/reversed-sequence GRU over the evaluated track. Their scores may be
   retained only as `offline_noncausal_oracle` diagnostics; they cannot be
   described as fixed-delay or zero-lookahead input. A counted route requires
   either a genuinely causal pretrained detector or an owner-approved causal
   temporal-head adaptation/retraining, followed by future-input and
   prefix/chunk counterfactual tests. The remaining source plan is: audio =
   the **combined mix** (tier-honest robot-observable analog; per-person
   lavalier audio is gold tier and must not feed a model input); name the
   video stream and its D3 offset + audio-mux requirement (6cam AVIs carry
   no audio and are monochrome — a domain shift versus AVA RGB pretraining;
   RealSense RGB sits on its own time axis); provision identity-stable face
   tracks bound to confirmed P{i} slots. A separate `P{i}_asd_valid` mask is
   recommended so missing face observations cannot masquerade as probability
   zero; adding it is a checkpoint/schema change and remains owner-pending.
   Only a separately validated causal implementation unblocks the primary B0
   baseline; the blocked source preflight does not.
6. **Causal combined-mix silence detector and matched H2 replay — completed as
   post-hoc exploratory evidence.** For each outer fold, the current-frame
   speech detector and reactive/predictive policy parameters are fitted or
   selected only from the three inner group-OOF partitions. The create-once
   output under `outputs/h2_matched_replay_posthoc_v1/` seals gold-free outer
   views and traces before opening outer gold, then evaluates the same 303
   robot-entry opportunities (111 positive/192 negative). It is independently
   replay-audited, but remains `production_eligible=false`,
   `counted_eligible=false`, and does not support H2. The separately labelled
   `posthoc_exact_eight_voiced_reactive` 1.5 s diagnostic remains distinct: it
   is not the calibrated comparator. A gold-VAD-driven replay is still named
   `oracle reactive replay`.
7. **Audio-side alignment (chain now explicit, source closed 2026-08-15):** transcript
   time = `timecode @ 30 fps + offset.json.offset_sec` on the
   aligned-lavalier axis; the model-input mix is the sample-replayed
   equal-weight mono average of `master_aligned.wav` (aligned close-talk
   lavaliers). Its channel identity, complete payload and mic-A axis are now
   bound by the ten audio contracts; the shorter Transcript reference is a
   mux/extraction derivative and is not an independent robot-ear recording.
   The first-run chain is now complete: input feature manifests are published,
   the frozen 30/1 non-drop timecode rule (including versioned `FF=30` carry
   normalization) replays, and full-corpus aligned-lavalier consistency
   diagnostics are retained as audit evidence.
8. **Outer-test nested runner — production path deferred; deadline route complete:** the
   four-group nested LOGO memberships, seeds, median-inner-epoch selection,
   primary/`d3_le80` variants, and fresh source/D3/statistics gates were frozen
   before outcome inspection. The only registered backend is
   `backend_contract_smoke`, so it cannot issue a counted result. The separately
   reviewed fixture-only PyTorch boundary is deliberately unregistered and
   permanently non-counted. The independently reviewed read-only evidence
   foundation now implements environment/source capture, canonical 25 Hz real
   tick authority and the ±D3-error-bound pose reassembly used by every feature
   set whose active columns include `head_yaw_deg`; it deliberately has no
   publisher and cannot authorize readiness. Still required: bind the exact
   outer run manifest and formal batch/scheduler, implement the atomic formal
   run index, register a production backend, and add the reviewed result
   publisher. An adversarial
   same-directory concurrent-replacement claim reached the owner-defined
   three-round limit. The owner accepted the trusted local single-user/single-writer
   boundary; preserve hash/replay/no-overwrite checks and disclose the residual
   risk. Separately, the deadline runner completed seed 13,
   `max_epochs=5`, A/H/A+H × four folds (12/12 units) and an independently
   audited saved-prediction aggregate. A separately derived 15-epoch post-hoc
   sensitivity protocol changed only the epoch budget, also completed 12/12,
   and passed full saved-probability/artifact/source audit. Neither result is
   production or repository-counted evidence.
9. **Upstream identity and pose-intake dependency — resolved for the selected
   corpus (2026-08-15):** ten scenes across four groups have sealed Gaze/Facial
   identity windows, an independent HPE-person→Transcript crosswalk, and
   materialised `data/processed/intake_v2/` pose bases. Full sealed-source replay
   reports 79,159 rows, `structural_ready=true`, and the expected
   `target_ready=false`. The provenance-bound crop is now materialised as ten
   canonical `data/processed/analysis_base/` pairs (78,582 rows total) and
   passes sealed directory audit. Those staging bases retain placeholders and
   remain `trainable=false`, but downstream formal target, acoustic and
   unified A/H/A+H assembly is complete for the same ten scenes. The wider
   14-scene HPE D3 roster is not automatically part of this dissertation
   dataset.

## 9. Deadline result interpretation

The immutable source retains all ten formal scenes. Before any outcome was
inspected, the owner excluded `05_15__4` (no A speech in the Transcript) and
`05_15__5` (A/R extremely sparse) from the deadline view pending manual
review. The exact eight included scenes span four recording groups and 64,299
rows.

H1 is **not supported under the limited head-pose operationalisation used in
the frozen deadline matrix**. ASD, Gaze and lip/facial visual voice-activity
cues were not part of that primary matrix, and fused A+H did not improve
overlap AP over A. The 15-epoch budget check does not reverse that conclusion:
H improves on pooled frame mAP, but A+H still loses to H and its overlap AP
declines. The separate single-seed post-hoc Gaze isolation is suggestive:
`CG-CM=+0.0381` pooled frame mAP and is positive in all four outer folds.
However, `CM-C0=-0.0256` demonstrates a material missingness effect, P1/R has
no direct Gaze observation, and exact upstream regeneration plus
implementation-level counterfactual causality are unverified. It therefore
does not reclassify H1 as supported or justify a claim about all possible visual
cues. H2 is **not supported and remains
unestablished**. In the separately audited post-hoc matched replay, the primary
five-epoch predictive policy obtained recall/FPR/balanced accuracy
0.4324/0.5156/0.4584 versus reactive 0.5766/0.3594/0.6086. The 15-epoch
predictive sensitivity reached 0.6306/0.6458/0.4924 and did not reverse the
balanced verdict. On positive events hit by both policies, prediction was
descriptively earlier by mean 374 ms/median 400 ms at five epochs (`n=29`) and
mean 355 ms/median 400 ms at 15 epochs (`n=44`), but this noncanonical
post-derived summary comes from the audit-bound event CSV and the gap-only
advantage was unstable. The matrix and replay contain only seed 13 and central
D3 offsets; D3 sensitivity and multi-seed replication remain pending. No
statistical-significance, production-readiness, physical-Furhat, TTS, live
latency or deployment claim follows from this result. The earlier voiced-only
reactive diagnostic remains a separate fixed-rule check.

## 10. Out of scope / gated

speaking_towards consumption and any addressee metric (decision 1); GVAP;
Furhat replay or live deployment (gate CLOSED — `methodology.md` §10,
`PROJECT_SCOPE.md` §5/§8); CPC ablation; per-speaker-audio inputs beyond
named oracle diagnostics.
