# Training-data readiness status

> **Operational source of truth after identity Phase 1.**
> Last updated: **2026-08-18**.
> Read this file before building a feature, assembling a real table, claiming
> train readiness, or starting an experiment. Identity details remain in
> `IDENTITY_DATA_STATUS.md`; scientific scope remains in `PROJECT_SCOPE.md` and
> `final_model_plan.md`.

## Current corpus

The selected dissertation corpus contains exactly ten scenes across four
recording groups:

`03_20__4`, `03_26__3`, `05_14__2`, `05_14__3`, `05_14__4`,
`05_15__2`, `05_15__3`, `05_15__4`, `05_15__5`, `05_15__6`.

The wider HPE delivery roster is not automatically part of this dataset.
`05_14__3` and `05_14__4` are first-segment-only throughout the pipeline.

## Stage ledger

| Stage | Status | Formal evidence / blocker |
|---|---|---|
| Gaze/Facial identity | complete | ten accepted artifacts under `data/annotations/identity/`; zero unresolved manual tracklets and zero identity conflicts |
| HPE→Transcript semantic crosswalk | complete | ten finalized artifacts plus sealed inventory `configs/hpe_transcript_crosswalk_inventory.json` |
| Pose intake v2 | complete | ten CSV/sidecar pairs under `data/processed/intake_v2/`; 79,159 rows, four groups, full deterministic replay, `structural_ready=true` |
| Pose-base target/input readiness | staging bases intentionally incomplete; downstream formal assembly complete | pose is real, while the canonical pose-only bases retain VAD/acoustic/ASD/Gaze/MAR placeholders and therefore remain `trainable=false`. Formal targets and A/H/A+H inputs were later merged into separate audited tables under `data/processed/formal_a_h_ah_v1/` |
| Audio source identity and reference sync | complete | ten independently replayed `data/annotations/audio_contract/*.audio_contract.json`; complete PCM payloads, exact master permutations, reference↔master sync pass in all ten |
| Inter-lavalier timing | complete for the selected ten-scene scope | ten formal manifests under `data/annotations/audio_gold_alignment/`; 13 non-zero constant channel corrections are applied only through versioned time maps, original WAVs are unchanged, and `05_14__2` R retains its explicit coarse-resolution warning plus successful independent 5 ms check |
| Transcript activity labels | complete for the ten-scene source corpus | the owner confirmed that the exact current Transcripts were already human-reviewed for speaker identity, overlap, quiet speech and pauses. The four impossible in-window intervals received explicit owner decisions, and only exact-preimage deterministic normalisation was applied. The other 1,090 in-window events remain inherited rather than falsely claimed as newly re-audited. The label role is named human-reviewed-Transcript-derived VAD and retains 30 fps quantization/within-utterance-pause limitations |
| Common analysis window | complete | ten canonical `vap_real_analysis_base` CSV/sidecar pairs under `data/processed/analysis_base/`; 78,582 rows on exact 25 Hz half-open tick windows, full strict build replay plus sealed directory audit, `structural_ready=true`. Exactly 577 pose rows outside joint pose/audio coverage were excluded rather than labelled as silence. Publication was per-scene no-clobber, not atomic across ten scenes, and wrote no corpus completion inventory; exact file-count/per-pair replay closes the present matrix. All bases remain `trainable=false` because gold VAD and causal acoustics are placeholders |
| Formal VAD targets | complete for all ten scenes | exact 25 Hz target artifacts and manifests are under `data/processed/transcript_derived_vad_targets/`; strict replay and aligned-lavalier consistency diagnostics passed. The lavaliers are consistency evidence, not a second manual reannotation, and the old candidate/blind-audit scaffold remains non-promotable |
| Causal energy/F0/voiced | complete for all ten scenes | versioned artifacts, evidence, manifests and independent reviews are under `data/processed/acoustic_features/`. They bind the owner-frozen normalized-autocorrelation threshold `0.45`, exact causal windows and the sample-replayed equal-weight master mix. This input remains an explicitly reported close-talk surrogate rather than robot-ear audio |
| Gaze post-hoc / Facial-MAR rejection | exact-eight C0/CM/CG training complete and independently audited; exploratory/non-production/non-counted. Facial dropped from modelling | the repaired read-only Gaze preflight passes 10/10 scenes and the create-once `data/processed/gaze_posthoc_exact8_v1/` schema-v3 matrix contains the frozen exact eight scenes, 64,299 rows and four groups. Seed 13, `max_epochs=15`, C0/CM/CG × four outer folds completed 12/12. The saved-probability aggregate `outputs/gaze_posthoc_15epoch_20260817/aggregate_seed13_v1.json` has SHA-256 `0b4f2c5e3deafafa0d23cdb4822f328d95af5c0788ba662cd9c04d7414955cea`. Pooled frame mAP is 0.2717/0.2461/0.2843 for C0/CM/CG; the primary content contrast `CG-CM` is +0.0381 and is positive in all four outer folds, while `CM-C0=-0.0256` records a material observability/missingness effect (`CG-C0=+0.0126` total package). R/P1 has no direct Gaze row. The actual upstream extractor variant, weights and configuration are not fully bound, exact upstream regeneration is unavailable, and implementation-level counterfactual causality is unverified. Facial still has no reproducible mouth landmarks, `mouth_motion` is not MAR, its hard emotion class is unvalidated, and its anonymous AU vector lacks an exact semantic/extractor binding; missing values are never promoted as zero |
| ASD | read-only source preflight implemented; **R3/3 BLOCK; deferred from first run** | official Light-ASD/LR-ASD commits are future-dependent and may be offline diagnostics only. The preflight retains two exhausted authority defects and produced no model, weights, probability or feature. Owner decision 2026-08-16 defers ASD from the first counted A / H / A+H matrix; no fourth repair round is started |
| Unified A/H/A+H tables | ten-scene assembly complete; exact eight-scene training view ready | ten audited CSV/sidecar pairs under `data/processed/formal_a_h_ah_v1/` contain 78,582 rows. `05_15__4` remains non-trainable because `P2_vad` is degenerate, and `05_15__5` remains on owner hold for sparse A/R speech. The frozen eight-scene, 64,299-row view alone passes target plus selected-feature readiness for A, H and A+H; all ten source assets remain intact |
| Deadline outer evaluation | complete; explicitly non-production and non-counted | the exact eight-scene, 64,299-row, four-group view completed seed 13, `max_epochs=5`, A/H/A+H × four outer folds (12/12 nested-LOGO units). The aggregate was independently rebuilt/audited from saved out-of-fold predictions. Every result retains `production_eligible=false` and `counted_eligible=false`; this is the dissertation's deadline real-data evaluation, not a production-backend promotion |
| Post-hoc H2 matched replay | complete and independently replay-audited; exploratory/non-production/non-counted | `outputs/h2_matched_replay_posthoc_v1/` binds self-contained gold-free outer views, inner-only calibration, E-only causal acoustic descendants, sealed traces, and evaluation on the same 303 robot-entry opportunities (111 positive/192 negative). Trace audit and evaluation audit reconstruct canonical bytes. Five epochs remains primary and 15 epochs sensitivity; no physical robot, TTS, action feedback, far-field robot-ear audio, or live latency was evaluated |

## Pose-base audit snapshot

The accepted v2 pose bases contain:

| Scene | Rows | 25 Hz tick interval `[start,end)` |
|---|---:|---:|
| `03_20__4` | 5,000 | `[332,5332)` |
| `03_26__3` | 4,350 | `[286,4636)` |
| `05_14__2` | 6,750 | `[601,7351)` |
| `05_14__3` | 11,714 | `[624,12338)` |
| `05_14__4` | 11,754 | `[629,12383)` |
| `05_15__2` | 6,500 | `[656,7156)` |
| `05_15__3` | 5,966 | `[586,6552)` |
| `05_15__4` | 6,875 | `[583,7458)` |
| `05_15__5` | 7,500 | `[481,7981)` |
| `05_15__6` | 12,750 | `[629,13379)` |

## Formal common-window snapshot

The canonical analysis bases crop the pose tables to the intersection with the
fully replayed per-lavalier/audio-reference authority. The output interval is
strictly half-open and no unavailable row is zero-filled as VAD:

| Scene | Analysis rows | 25 Hz tick interval `[start,end)` | Pose rows excluded |
|---|---:|---:|---:|
| `03_20__4` | 4,956 | `[376,5332)` | 44 |
| `03_26__3` | 4,321 | `[315,4636)` | 29 |
| `05_14__2` | 6,750 | `[601,7351)` | 0 |
| `05_14__3` | 11,714 | `[624,12338)` | 0 |
| `05_14__4` | 11,696 | `[687,12383)` | 58 |
| `05_15__2` | 6,317 | `[656,6973)` | 183 |
| `05_15__3` | 5,812 | `[586,6398)` | 154 |
| `05_15__4` | 6,840 | `[618,7458)` | 35 |
| `05_15__5` | 7,443 | `[481,7924)` | 57 |
| `05_15__6` | 12,733 | `[629,13362)` | 17 |
| **Total** | **78,582** | — | **577** |

## Visual-evidence verdict

The current exports do **not** support a complete strict-visual condition:

- Gaze supplies only `robot / participant / elsewhere`; a 2D gaze-peak→unique
  confirmed bbox rule can produce evidence for most participant targets, but
  it is not the previously described 3D gaze-vector intersection.
- The robot-player R has no participant Gaze/Facial source row. Consequently
  `P1_gaze_robot` and complete incoming-gaze denominators for P2–P4 are not
  observed.
- Facial `openface_features.csv` has no landmark coordinates from which MAR can
  be reproduced. `mouth_motion.csv` is neither MAR nor an accepted source.

`audio_plus_head_pose` therefore remains the first viable multimodal route.
Full visual conditions stay gated; no missing visual value may be relabelled as
a genuine zero or hidden behind `pose_valid`.

The completed post-hoc Gaze experiment tests only this narrow nullable binary
cue. Its C0/CM/CG design keeps the schema-matched A+H anchor, separates the
validity/missingness channel from the Gaze value, and leaves P1/R directly
unobserved. It therefore does not create a complete strict-visual condition or
establish causal deployment equivalence.

## Deadline evaluation snapshot (2026-08-17)

The immutable scientific source still contains all ten formal scenes. Before
any model outcome was inspected, the owner placed two scenes on manual-review
hold and excluded them only from the deadline evaluation view:

- `05_15__4`: no A speech in the Transcript; pending owner manual review;
- `05_15__5`: A/R speech is extremely sparse; pending owner manual review.

The included view is therefore exactly `03_20__4`, `03_26__3`, `05_14__2`,
`05_14__3`, `05_14__4`, `05_15__2`, `05_15__3`, and `05_15__6`: 64,299 rows
across four recording groups. The frozen deadline protocol used seed 13,
`max_epochs=5`, and paired A/H/A+H nested leave-one-group-out evaluation. All
12 condition/fold units completed, and the saved-prediction aggregate passed
independent audit.

The scientific interpretation is deliberately narrow. In the frozen deadline
matrix, H1 is **not supported under this limited head-pose operationalisation**:
ASD, Gaze and lip/facial voice-activity cues were not part of that primary
matrix, and A+H did not improve the overlap result over A. The later post-hoc
Gaze isolation found `CG-CM=+0.0381` pooled frame mAP with a positive delta in
all four folds, but `CM-C0=-0.0256`, P1 has no direct Gaze observation, only one
seed was run, and upstream exact regeneration plus implementation-level
counterfactual causality remain unverified. This is suggestive incremental
content evidence, not enough to reclassify H1 as supported. A separately
audited post-hoc exploratory H2 matched replay has now
compared the P1 predictive policy with an inner-calibrated causal
speech-detector/silence policy on the same 303 held-out opportunities. At the
primary five-epoch budget, predictive recall/FPR/balanced accuracy were
0.4324/0.5156/0.4584 versus reactive 0.5766/0.3594/0.6086. The 15-epoch
predictive sensitivity reached 0.6306/0.6458/0.4924, so it did not reverse the
balanced verdict. H2 is therefore **not supported and remains unestablished**.
D3 sensitivity, additional seeds, causal ASD and a preregistered,
exactly-regenerable/causality-verified Gaze condition, and physical Furhat work
remain deferred; the current Facial export is dropped from
modelling rather than treated as a future-ready input. These limitations do not
turn the saved real-data measurements into synthetic results, but they do forbid
production, statistical-significance or deployment claims.

The post-hoc analyses completed without rewriting the primary result. A
15-epoch budget sensitivity run changed only `max_epochs` from 5 to 15 and
completed all 12 paired units. H pooled frame mAP increased from 0.309 to
0.400, while A+H remained 0.264 and still did not beat H; A+H overlap AP fell
from 0.578 to 0.525. Nine of 36 inner fits and three of 12 selected final fits
still touched the 15-epoch ceiling, so convergence is not claimed. In the
matched H2 replay, positive opportunities hit by both policies showed a
descriptive predictive lead of mean 374 ms/median 400 ms at five epochs
(`n=29`) and mean 355 ms/median 400 ms at 15 epochs (`n=44`). These summaries
were derived from the audit-bound event CSV rather than declared as canonical
prespecified metrics, and the gap-only sensitivity did not show a stable
advantage across budgets. Separately, the exact-eight `voiced`-only reactive
diagnostic replayed the same 303
robot-entry opportunities and produced TP/FP/FN/TN = 5/6/106/186 (recall
0.0450, negative false-entry rate 0.0313, balanced accuracy 0.5069). It is not a
complete speech detector or a substitute for the calibrated matched comparison.
The matched replay still contains no real robot/TTS/live-latency evidence.

## Non-negotiable gates

1. Never edit or overwrite upstream WAV, HPE, Gaze, Facial or Transcript files.
2. Never reuse `data/processed/intake/`; it is stale.
3. Never widen either 05_14 first-segment window.
4. Use only the exact hash-bound **human-reviewed-Transcript-derived VAD**
   approved on 2026-08-16; never relabel it sample-perfect lavalier VAD or
   claim that all Transcript events were newly re-audited.
5. Never use per-person lavalier audio as a model input; it is gold/oracle tier.
6. Never start formal training from a table with placeholders, failed source
   replay, fewer than three groups, or no pre-declared outer-test protocol.
7. Preserve D3 status/error bounds. The deadline result must explicitly state
   that sensitivity was not run; any later D3-robustness or production claim
   requires the frozen uncertainty analysis rather than silently treating
   degraded timing as exact.
8. For the same reproducible blocker, stop after three independent
   repair-and-retest rounds and request an owner decision; do not start a
   fourth patch round implicitly.

## Frozen D3 sensitivity

This predeclared sensitivity study remains frozen but **was not executed** in
the deadline matrix. The original ten-scene plan keeps central offsets, adds a
six-scene ≤80 ms paired run, deterministic `±error_bound_ms` head-pose shifts,
and a descriptive three-scene ≤40 ms subset. Within the exact eight-scene
deadline view, exclusion of `05_15__5` reduces the ≤80 ms membership to five
scenes while retaining all four groups. The completed seed-13 matrix used only
the central offsets; neither the ≤80 ms variant nor the ±bound shifts may be
claimed as completed.

## Owner decisions resolved on 2026-08-16

The owner accepted the trusted local single-user/single-writer workspace.  All
source/hash/replay/no-overwrite gates remain, and the residual concurrent or
malicious path-replacement risk is disclosed rather than represented as closed.
The owner froze acoustic voicing threshold `0.45` and selected A / H / A+H as
the first scientific matrix, with ASD/Gaze/Facial deferred. The completed
deadline matrix uses that comparison but remains explicitly non-production and
non-counted; any repository-counted promotion is separate future work.

The owner also confirmed that the existing exact Transcripts are already human
annotations. They are inherited as such; only the four impossible in-window
intervals received new owner decisions. No seven-day repeat or large blind
reannotation is required, and no long-term intra-rater reliability is claimed.

The ASD source preflight also reached the owner-defined three-round limit with two formal-
authority defects: fixture provenance can masquerade as formal provenance, and ancestors
above the configured authority root are not reparse-checked. Do not start a fourth repair
round. Keep the preflight diagnostic-only and omit ASD, or obtain an owner decision for a new
versioned authority design and independent review.

## Remaining deferred sequence

1. Keep `05_15__4` and `05_15__5` out of the current deadline result until the
   owner completes manual review. Any later inclusion requires a newly frozen
   dataset/protocol and must not rewrite the completed run.
2. If publication-grade replication is pursued, complete the predeclared D3
   sensitivity and additional paired seeds, then review the production backend
   and atomic cross-process run ledger separately.
3. The exact-eight calibrated predictive-versus-reactive matched replay is
   complete as post-hoc exploratory evidence on the same 303 opportunities;
   its result does not support H2. Keep it distinct from the earlier
   `voiced`-only fixed-rule diagnostic. Any stronger latency/missed-turn claim
   now requires preregistered replication and, separately, real robot/TTS/live
   timing rather than another reinterpretation of the same event CSV.
4. The exact-eight C0/CM/CG Gaze run is complete as single-seed post-hoc
   exploratory evidence. Preserve `CG-CM` as the content contrast and `CM-C0`
   as the missingness/observability contrast. A stronger Gaze claim requires
   exact upstream regeneration, implementation-level counterfactual causality,
   and preregistered multi-seed replication; do not reinterpret this aggregate
   as such evidence.
5. Keep causal ASD, GVAP and physical Furhat work gated as documented. Facial
   is archived source evidence and dropped from the current modelling route;
   none of these is required to preserve the completed MSc deadline result.
