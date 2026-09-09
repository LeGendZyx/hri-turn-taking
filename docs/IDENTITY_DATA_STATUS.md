# Participant identity and modality availability status

> **Operational source of truth for Phase 1 identity work.**
> Last updated: **2026-08-15**.
> Read this file before opening an identity artifact or deciding that a scene is
> available. `docs/manual_identity_annotation.md` explains the tool, but its
> historical examples and dated command snapshots are not live status.

This document records the currently usable intersection of Transcript, confirmed
Head Pose (HPE), Gaze, and Facial data. It covers participant-identity recovery
only. It does **not** claim that the formal shared timeline, acoustic features,
gold VAD, or model-training readiness is complete.

## Fixed identity contract

- Transcript labels are the canonical identities: `A`, `B`, `C`, `R`.
- Model slots are fixed: `R=P1`, `A=P2`, `B=P3`, `C=P4`.
- Raw Gaze/Facial `person_idx` is a within-frame detection key, never a
  cross-frame identity.
- Facial inherits identity only through the exact same-frame key
  `(frame_index, person_idx)` against Gaze bridge
  `(frame_index, source_person_idx)`.
- Only `annotation_status == confirmed` is usable. `uncertain`, `ignored`, and
  unreviewed rows remain missing.
- Every accepted artifact must have zero rows in `identity_conflicts.csv` and
  pass source/hash/resume replay with the interpreter recorded by the workflow.
- `mouth_motion.csv` is not a formal feature.
- HPE files are never overwritten when Transcript and HPE labels differ. The
  separate, sealed final-project crosswalk described below now proves the
  Transcript-to-HPE final-person semantics for exactly these ten accepted
  scenes; it does not alter the upstream HPE mappings.

## Accepted identity windows

All ten rows below have a confirmed HPE mapping, a published/self-checked 6cam
offset, Gaze, Facial, Transcript, a completed identity artifact, zero unreviewed
manual tracklets, and zero identity conflicts **within the stated window**.

Definitions:

- **Bridge confirmed** = confirmed identity observations divided by all Gaze
  bridge observations. Extra duplicate/false detections remain in the
  denominator, so this can be lower than participant coverage.
- **Canonical coverage** = confirmed `R/A/B/C` observations divided by the
  four-participant frame grid in the accepted window.
- **Facial confirmed** = Facial rows in the accepted window that inherit a
  confirmed identity by the exact key. It is not an interpolated match.

| Scene | Camera | Accepted local frame window | HPE-core coverage | Bridge confirmed | Canonical coverage | Facial confirmed | Conflicts | Identity status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `03_20__4` | Cam_2 | `[600,6600)` | 100% | 22,549/23,498 (95.96%) | 93.95% | 2,478/2,579 (96.08%) | 0 | accepted |
| `03_26__3` | Cam_2 | `[390,5610)` | 100% | 20,868/20,901 (99.84%) | 99.94% | 2,461/2,461 (100%) | 0 | accepted |
| `05_14__2` | Cam_2 | `[1800,18000)` | 100% | 63,306/79,482 (79.65%) | 97.69% | 3,678/3,804 (96.69%) | 0 | accepted |
| `05_14__3` | Cam_2 | `[1800,29913)` | 82.20% | 103,790/121,417 (85.48%) | 92.30% | 5,174/5,716 (90.52%) | 0 | accepted, **first segment only** |
| `05_14__4` | Cam_2 | `[1800,30008)` | 74.62% | 112,832/131,837 (85.58%) | 100% | 6,918/6,918 (100%) | 0 | accepted, **first segment only** |
| `05_15__2` | Cam_1 | `[900,8700)` | 100% | 31,200/31,204 (99.99%) | 100% | 3,679/3,679 (100%) | 0 | accepted |
| `05_15__3` | Cam_1 | `[900,8060)` | 100% | 27,028/27,899 (96.88%) | 94.37% | 2,949/3,049 (96.72%) | 0 | accepted; `p000698` manually rechecked as A |
| `05_15__4` | Cam_1 | `[900,9150)` | 100% | 32,020/32,520 (98.46%) | 97.03% | 3,850/3,903 (98.64%) | 0 | accepted; ambiguous frame 2540 remains missing after split |
| `05_15__5` | Cam_1 | `[900,9900)` | 100% | 35,472/35,865 (98.90%) | 98.53% | 4,198/4,239 (99.03%) | 0 | accepted |
| `05_15__6` | Cam_1 | `[900,16200)` | 100% | 61,191/61,263 (99.88%) | 99.99% | 7,216/7,216 (100%) | 0 | accepted |

### Observation-weighted coverage by canonical participant

The denominator for each participant is the number of frames in that scene's
accepted local window.

| Scene | R | A | B | C | Overall four-participant grid |
|---|---:|---:|---:|---:|---:|
| `03_20__4` | 100% | 94.85% | 94.85% | 86.12% | 93.95% |
| `03_26__3` | 100% | 99.77% | 100% | 100% | 99.94% |
| `05_14__2` | 100% | 96.93% | 96.93% | 96.93% | 97.69% |
| `05_14__3` | 100% | 90.80% | 86.82% | 91.57% | 92.30% |
| `05_14__4` | 100% | 100% | 100% | 100% | 100% |
| `05_15__2` | 100% | 100% | 100% | 100% | 100% |
| `05_15__3` | 99.97% | 95.64% | 85.67% | 96.20% | 94.37% |
| `05_15__4` | 100% | 98.12% | 91.87% | 98.13% | 97.03% |
| `05_15__5` | 99.82% | 98.43% | 97.44% | 98.43% | 98.53% |
| `05_15__6` | 100% | 99.98% | 99.98% | 99.98% | 99.99% |

## Owner decision: 05_14 first-segment-only scenes

`05_14__3` and `05_14__4` each contain two physical Cam_2 AVI files. HPE and
Facial processed the concatenated two-file chain, but the delivered Gaze CSVs
contain only the first AVI. On 2026-08-15 the owner decided that Gaze cannot be
re-exported and these scenes will use the existing first segment only.

| Scene | Full two-segment video | Delivered Gaze | HPE logical core | Accepted identity window | Excluded Facial rows inside the uncovered core |
|---|---:|---:|---:|---:|---:|
| `05_14__3` | 37,277 frames (`29,913 + 7,364`) | `[0,29913)` | `[1800,36000)` | `[1800,29913)` | 1,518 |
| `05_14__4` | 41,038 frames (`30,008 + 11,030`) | `[0,30008)` | `[1800,39600)` | `[1800,30008)` | 2,385 |

Consequences:

- Do not extend either identity bridge into segment 2 using Facial
  `person_idx`; those rows have no exact current-Gaze key and remain unusable.
- The formal timeline crop now restricts Transcript/HPE descendants to the
  accepted common window through the canonical `vap_real_analysis_base`
  artifacts. The omitted portion remains unavailable data, not a negative
  observation or a zero-filled feature interval.
- The accepted first-segment artifacts remain valid and must not be overwritten
  or migrated merely because the physical session has a second AVI.

## D3 timing quality attached to the accepted scenes

All entries below publish a self-checked 6cam offset and therefore pass the
identity-stage timing-presence gate. `degraded` and values above the ±40 ms
target do not invalidate identity. Their counted-run treatment is now frozen:
the primary analysis keeps all ten central offsets, the paired `d3_le80`
variant keeps six scenes/four groups, and every head-pose condition reports
outer-test `±error_bound_ms` shifts without recalibration.

| Scene | Offset quality | Error bound |
|---|---|---:|
| `03_20__4` | ok | 33.5 ms |
| `03_26__3` | ok | 33.5 ms |
| `05_14__2` | degraded | 69.6 ms |
| `05_14__3` | degraded | 101.3 ms |
| `05_14__4` | degraded | 104.4 ms |
| `05_15__2` | degraded | 36.2 ms |
| `05_15__3` | degraded | 56.4 ms |
| `05_15__4` | degraded | 87.2 ms |
| `05_15__5` | degraded | 43.7 ms |
| `05_15__6` | degraded | 101.3 ms |

## Known scenes that are not currently usable

These scenes must not be presented as members of the current formal
Transcript+HPE+Gaze+Facial identity intersection.

| Scene(s) | Available pieces | Blocking condition / decision |
|---|---|---|
| `03_20__1` | Gaze, Facial | No matching confirmed HPE delivery |
| `03_20__2`, `03_20__3` | confirmed HPE, Gaze, Facial | No published 6cam D3 offset; expected Transcript path also unavailable |
| `03_20__5` | confirmed HPE, D3, Transcript | No usable Gaze or Facial export |
| `03_20__6` | confirmed HPE | No D3, Gaze, or Facial export |
| `03_24_1__2`, `03_24_1__3` | confirmed HPE, D3, Transcript | No Gaze or Facial export |
| `03_26__1`, `03_26__2` | Gaze, Facial | No matching confirmed HPE delivery |
| `03_26__4` | confirmed stitched HPE scene | No D3 or Transcript; physical Gaze sessions 4 and 5 must not be misrepresented as one supported session |
| `03_26__5` | Gaze, Facial | No independent HPE scene; it is part of stitched logical `03_26__4` |
| `03_26__6` | confirmed HPE, Gaze, Facial | No D3; expected Transcript path unavailable |
| `03_26__7` | confirmed HPE | No D3, Gaze, or Facial export |
| `05_14__1` | Gaze, Facial | No matching confirmed HPE delivery |
| `05_14__5` | confirmed HPE, D3, Gaze, Facial | Transcript missing; explicitly deferred |
| `05_14__6` | confirmed HPE, Gaze, Facial | No D3; expected Transcript path unavailable |
| `05_15__1` | Gaze, Facial | No matching confirmed HPE delivery |

`03_24_2__2` through `03_24_2__6` are timing-only records, not confirmed HPE
deliveries and not identity candidates.

## Facial exact-key caveats

- Formal Facial rows without an exact Gaze key remain missing; notably
  `05_14__3` has 26 such first-segment rows.
- Current Facial duplicate-key counts are: `03_20__4` = 6 keys,
  `05_14__3` = 3, `05_15__3` = 1, and `05_15__4` = 35. The identity bridge is
  unique at those keys, so duplicate Facial rows inherit the same identity, but
  the later feature assembler still needs a versioned row-selection or
  aggregation rule. Original Facial files must not be deduplicated in place.
- Facial rows for the uncovered second segments of `05_14__3/__4` are excluded,
  regardless of whether their raw `person_idx` resembles first-segment values.

## Artifact locations and downstream acceptance

Each accepted scene lives at:

```text
data/annotations/identity/<scene>/
```

The authoritative files are:

- `identity_annotations.json` — human decisions and source/scope bindings;
- `tracklet_observations.csv` — immutable/revision-bound observation rows;
- `identity_bridge.csv` — exact observation-to-canonical-identity mapping;
- `identity_conflicts.csv` — must contain only the header;
- `annotation_audit.jsonl` — append-only review history.

Before downstream use, reload the manifest, replay source/hash validation,
rebuild the bridge/conflicts in memory, require zero conflicts, and select only
confirmed rows. A file existing on disk is not by itself an acceptance gate.

## HPE-to-Transcript crosswalk status

The separate review workflow is implemented and documented in
`docs/hpe_transcript_crosswalk.md`. **All ten accepted scenes have now been
manually reviewed, finalized, independently replayed, and sealed.** Their
outputs live under `data/annotations/hpe_transcript_crosswalk/<scene>/`; they
never overwrite HPE deliveries or the accepted Gaze identity artifacts.

The immutable downstream authority is:

- `configs/hpe_transcript_crosswalk_inventory.json`, SHA-256
  `d0799fa2d6882a23db5cd7e88fc3e5272501af6be4528b1b933a41976681e42b`;
- `configs/hpe_transcript_crosswalk_ledger_snapshot.md`, SHA-256
  `304db1a26eda2b423d29a49e1ebc71d30e936483737aaf4cdf9b83e9bb886ecf`.

The sealed loader replays the exact artifact bytes, HPE/Gaze authority chains,
accepted half-open window, HPE logical-axis frame rate, and complete
`person -> A/B/C/R -> P1/P2/P3/P4` mapping. All ten scenes load under the
project `.venv`; the raw-evidence re-audit was also replayed successfully on
the six-fragment `05_15__3` case. This closes the Phase 1 HPE-to-Transcript
semantic crosswalk.

All old `data/processed/intake/` pose tables predate this authority and are
stale. Ten replacement pose bases now exist under `data/processed/intake_v2/`,
bind this sealed inventory, and pass deterministic source replay: 79,159 rows
across ten sessions/four groups with `structural_ready=true`. They remain
`trainable=false`, as intended. The common-window crop is now complete as ten
canonical `data/processed/analysis_base/` CSV/sidecar pairs (78,582 rows;
sealed structural audit passed); 577 pose rows outside joint audio coverage
were excluded, not zero-filled. Gold VAD and causal acoustics remain
placeholders, so neither the intake nor analysis-base tables are train-ready.
An old CSV must never be treated as upgraded merely because the crosswalk or
the v2 directory now exists.

## Maintenance rule

Update this document whenever any of the following changes:

1. a scene gains or loses a required modality;
2. an identity artifact is newly completed, split, reset, or re-reviewed;
3. an accepted frame window or first-segment decision changes;
4. an exact-key Facial mismatch/duplicate count changes;
5. Transcript availability or D3 quality policy changes.

Record the new date, recompute the tables from the formal artifacts, and keep
historical command files clearly marked as non-authoritative. Never infer a
new usable scene merely from a filename or raw `person_idx`.
