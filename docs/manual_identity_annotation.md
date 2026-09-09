# Manual participant identity annotation

> **Live availability/status:** read `docs/IDENTITY_DATA_STATUS.md` before
> choosing or resuming a scene. This document explains the annotation tool;
> dated examples below are not the current completion ledger.

This document covers the completed Gaze/Facial identity stage. The later,
separate review that maps final HPE numeric people onto the same Transcript
identities is documented in `docs/hpe_transcript_crosswalk.md`; it does not
rerun or overwrite the work described here.

The gaze export documents `person_idx` as a **within-frame detection index**.
It can change when detector ordering changes, so it must not be treated as a
participant identity. The facial export inherits the same-frame index from the
gaze pipeline.

`tools/annotate_person_ids.py` links nearby bounding boxes into conservative
short tracklets without using `person_idx`, overlays them on the original
camera video, and lets a reviewer assign `A`, `B`, `C`, or `R`. It never edits
the colleague's gaze or facial exports.

The tracker records `tracker_algorithm=crowding_quarantine_v2` in new annotation
manifests. A strict overlap/proximity frame is cut into one-frame tracklets;
those tracks, plus isolated one-frame participant fragments, are automatically
marked `uncertain`, so they stay missing and are skipped by `N`.
This preserves the no-identity-mixing guarantee without asking the reviewer to
label thousands of ambiguous fragments. Clear frames still use bbox-only
linking; raw `person_idx` is never treated as a cross-frame identity. A scene may
use a stricter centre-distance gate in a new revision, recorded in
`tracker_config`; changing that gate cannot silently resume an older revision.

Fresh revisions may also set `--short-tracklet-max-duration-s`. This rule applies
only to `participant_detection` tracklets and measures
`max(timestamp_sec)-min(timestamp_sec)`, not frame count or `person_idx`; the
recorded boundary uses a `1e-12 s` numerical tolerance. Matching short tracklets
remain `uncertain`/missing. The value and policy version are stored in the
manifest, and changing the value requires a new revision. The default is `0.0`
for backward-compatible resume. For fresh `05_14` artifacts, use the audited
`0.10 s` setting to leave the very short detection tail missing.

## Formal-dialogue scope (default)

The annotator defaults to the owner-confirmed HPE `core_window_sec` for the
same scene. This removes setup, staff entry, and post-dialogue footage **before**
tracklets are built. The half-open interval is `[start, end)`: the start frame
is included and the end frame is excluded. Original gaze/facial CSVs and source
videos are never shortened or overwritten; only this annotation revision is
scoped.

The scope is fail-closed and provenance-bound. The tool follows HPE's confirmed
delivery mapping to `validation_complete.json`, verifies the geometry-profile
and camera-marker SHA-256 values, and records the exact logical and local frame
ranges in `identity_annotations.json`. It preserves the original local
`frame_index` and `timestamp_sec`; it never rebases them. `--allow-full-session`
exists for smoke/debug work only and must not be used for counted real data.

The colleague exports cover four dates (`03_20`, `03_26`, `05_14`, `05_15`),
22 gaze/facial sessions in total. Do **not** annotate all 22 first. The current
formal-training intersection should prioritise these 10 sessions:

- `03_20__4`, `03_26__3`;
- `05_14__2`, `05_14__3`, `05_14__4`;
- `05_15__2` through `05_15__6`.

`05_14__5` also has gaze and a usable six-camera offset, but its transcript is
currently missing, so defer it unless that blocker is resolved. `03_26__4` is
a stitched HPE scene spanning physical gaze `session_4` and `session_5`; it has
no D3 shared-time offset and is deliberately unsupported/fail-closed in this
workflow rather than being misrepresented as a complete cropped session.

Some available gaze exports cover only one physical AVI segment of an HPE
logical scene. The tool reports `coverage` explicitly (notably partial coverage
for `05_14__3/__4/__5`) and maps the logical core to the correct local frames.
For `05_15`, gaze uses excluded pose camera `Cam_1`; the confirmed scene/profile
remains authoritative, while the exact Cam_1 video axis is separately hash-bound.

## One-session quick start

Install the annotation-only dependency if the active environment has no
OpenCV:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[annotation]"
```

Before opening a source video stored on OneDrive, select it in Explorer and
choose **Always keep on this device**. The annotator refuses an Offline
placeholder by default so it cannot silently download a multi-gigabyte file.

The `03_26/session_3` migration is retained as a historical resume example; it
is no longer the current in-progress scene. Its confirmed HPE window is
`[13, 187)` seconds, or local frames `[390, 5610)` at 30 fps. During migration
on 2026-08-04, 37 tracklets inherited their exact review state, 7
boundary-affected tracks were reset for review, and 80 fully out-of-core tracks
were excluded. Its hash-verified original is preserved at
`data/annotations/identity/03_26__3_full_v1_backup_20260804T225622Z`.

Use `docs/IDENTITY_DATA_STATUS.md` for its current completion and coverage
figures.

Do **not** rerun `--migrate-hpe-core` on the current v2 artifact and do not use
`--overwrite`. Resume the UI normally:

```powershell
.\.venv\Scripts\python.exe tools\annotate_person_ids.py `
  --gaze-csv "gaze_features_export\03_26\session_3\frame_features.csv" `
  --video "<local path to the exact Cam_2 video declared by session_metadata.json>" `
  --output-dir "data\annotations\identity\03_26__3"
```

An alternate interpreter may be used if it contains OpenCV, but record the
interpreter in the resulting artifact and preserve compatible replay semantics.

`03_20__4` uses `--crowding-center-distance-ratio 0.25` in its fresh
`crowding_quarantine_v2` revision. The default `0.80` conservatively fragmented
long periods in which participants merely stood close together. The stricter
gate is recorded in the manifest; do not omit or change it when resuming that
scene.

The preparation stage can be checked without downloading or opening video:

```powershell
.\.venv\Scripts\python.exe tools\annotate_person_ids.py `
  --gaze-csv "gaze_features_export\03_26\session_3\frame_features.csv" `
  --output-dir "data\annotations\identity\03_26__3" `
  --prepare-only
```

## Controls

| Input | Action |
|---|---|
| Mouse click | Select the smallest bounding box under the cursor |
| `V` | Verify the selected tracklet at the current required anchor frame |
| `A`, `B`, `C`, `R` | Confirm identity after all required anchors are verified |
| `U` | Mark uncertain; it is not accepted as a confirmed identity |
| `I` | Ignore a false detection or unusable tracklet |
| `N`, `P` | Jump to the next or previous unreviewed tracklet |
| `[`, `]` | Jump to the selected tracklet's first or last observation |
| `X` | Split a manually reviewable tracklet at the first wrong frame after an ID switch |
| Arrow keys | Move one frame or one second; comma/period also move one frame |
| Space | Play or pause |
| `Z` | Undo the latest label change from this run |
| `H` | Toggle the on-video help |
| `Q` or Esc | Save and exit |

The separately exported `robot_*` box is drawn as `ROBOT?`. It remains only a
candidate until the reviewer explicitly labels its tracklet `R`.

`N`/`P` prioritise tracklets by observation count, so long, high-coverage
tracks are reviewed first. You do **not** need to label every one-frame false
positive: label real people, mark obvious false detections `I`, and leave the
remaining tiny/unresolved tracks missing. In the most fragmented current
export (`05_14/session_3`), singleton false detections form a very long tail;
reviewing longest-first captures useful coverage without assigning identities
to those uncertain fragments. The overlay reports observation-weighted resolved
and confirmed coverage while you work.

Tracklets are proposals, not ground truth. Confirmation is blocked until the
track's first, middle, last, every 5-second periodic checkpoint, both sides of
an observation gap, and any risk-flagged frames have been explicitly verified
with `V`. After each `V`, the UI advances to the next required checkpoint. Use
`N` to select a high-coverage track and `[`/`]` for its endpoints. Play through
any close crossing. If identity changes inside a track, move to the first wrong
frame and press `X`; both resulting tracks must be reviewed separately unless a
side meets the recorded automatic-missing policy. An automatically quarantined
tracklet cannot be split or labelled; it remains missing.

This is a **human-reviewed conservative mapping**, not a claim that a
geometry-only tracker has zero ID switches. An instantaneous swap between two
sampled frames cannot always be inferred from bounding boxes alone; periodic
human checkpoints and optional playback are the final identity safeguard.

## Outputs and acceptance rule

The output directory contains:

- `tracklet_observations.csv`: each box and conservative link; its SHA-256,
  ordered schema, parsed dtypes, and row count are bound by the manifest and
  updated only by an audited manual split;
- `identity_annotations.json`: resumable labels, required/reviewed anchors,
  structural split events, source CSV/metadata hashes, tracker settings, and
  the confirmed HPE scope/authority chain;
- `annotation_audit.jsonl`: append-only label, undo, and track-split history;
- `identity_bridge.csv`: per-observation mapping to participant and canonical
  slot (`R=P1`, `A=P2`, `B=P3`, `C=P4`);
- `identity_conflicts.csv`: any frame where two confirmed tracklets claim the
  same participant.

Downstream code must use only rows with `annotation_status == "confirmed"`
and must require `identity_conflicts.csv` to have zero data rows. `uncertain`,
`ignored`, and unreviewed rows remain missing rather than being guessed.

To attach identity to gaze or facial rows, join participant-detection bridge
rows on exact `(frame_index, source_person_idx)` against the export's
`(frame_index, person_idx)`. Facial rows that have no exact current-gaze match
must remain missing. This is especially important for `05_14/session_3` to
`session_6`, where the current facial and gaze exports have different frame
coverage. Do not use `mouth_motion.csv` as a formal feature: its temporal
difference was computed before stable identity recovery and cannot be safely
reconstructed because the original 98 landmarks were not exported.
