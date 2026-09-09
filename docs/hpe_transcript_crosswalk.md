# HPE person to Transcript identity review

This tool creates the missing semantic crosswalk between the final numeric
`person` values in an HPE delivery and the Transcript canonical identities
`A/B/C/R`.

It is deliberately separate from both upstream HPE mapping files and the
accepted Gaze identity artifacts. It never edits either source.

## What the reviewer is deciding

The HPE delivery has already linked camera detections and multi-view fragments
into final numeric people (`person = 1, 2, ...`). That tracking work is trusted
here and is not repeated. The reviewer answers only:

> Which Transcript person (`A`, `B`, `C`, or `R`) is this HPE numeric person?

Do not copy the letters in HPE `participant_map.csv`. Those letters describe
the HPE scene/seat convention and have not yet been proven to have the same
semantics as the Transcript identities.

The review window shows:

- left: the already-confirmed Gaze bridge labelled with Transcript canonical
  `A/B/C/R`;
- right: the same logical instant with neutral labels such as `HPE person 2`;
- one early, one middle, and one late sample for every final HPE person in the
  accepted window.

For the first five accepted scenes both panels use Cam_2. For `05_15`, the
canonical Gaze reference is Cam_1 and HPE evidence comes from an automatically
validated active camera, normally Cam_2. The two panels are synchronized by
their six-camera logical frame index; D3 audio offsets are not involved in this
identity-only step.

## Scope

The tool reads the accepted-scene table in `docs/IDENTITY_DATA_STATUS.md` at
runtime. It refuses scenes or frame windows not listed there. In particular,
`05_14__3` and `05_14__4` are restricted to their accepted first segments;
their second segments can never enter the review plan.

The set of identities to review is taken only from the final `person` column of
the delivered `world_persons.csv` inside that accepted half-open frame window.
It is not expanded from camera-local detections. Consequently:

- `05_14__3` does not acquire a second-segment-only person;
- `05_15__3` reviews all six final HPE person fragments even though they map to
  four canonical people.

Several non-overlapping HPE person fragments may be confirmed as the same
canonical person. Two simultaneously present HPE people may not be assigned
the same canonical identity.

## Running the tool

Open PowerShell in the repository root. Use the interpreter recorded by the
accepted artifact, with the annotation dependencies installed. Do not migrate
an accepted artifact merely to change interpreters.

Prepare and validate the deterministic review plan without opening a window:

```powershell
python tools\review_hpe_transcript_crosswalk.py `
  --scene 03_20__4 `
  --prepare-only
```

Then start or resume the review:

```powershell
python tools\review_hpe_transcript_crosswalk.py `
  --scene 03_20__4
```

Startup may spend one or two minutes hashing the large HPE/Gaze sources. The
tool prints a validation message during that work; this is expected and does
not download an Offline video.

The default paths are:

- HPE repository: `..\Head_pose_extraction`;
- accepted Gaze identity artifact:
  `data\annotations\identity\<scene>`;
- new output:
  `data\annotations\hpe_transcript_crosswalk\<scene>`.

If a required source frame is not already available as a hash-bound HPE review
still, the tool reports the exact AVI it needs. It refuses a OneDrive
online-only placeholder instead of silently downloading a multi-gigabyte
video. Make only that AVI local, finish the scene, and release it through
OneDrive again afterwards if desired. `--gaze-video` and repeatable
`--hpe-video Cam_N=<path>` can select an explicitly local copy; every selected
copy must match the recorded camera, dimensions, frame count, and video
fingerprint.

## Keys

- `A`, `B`, `C`, `R`: record the identity decision for the displayed sample;
- `U`: mark the sample unresolved;
- `N` / `P`: next / previous sample;
- `Z`: undo the most recent decision;
- `F`: finalize when every required sample is reviewed;
- `H`: show or hide help;
- `Q` or `Esc`: save current progress and exit.

Each early/middle/late sample receives its own decision. A final HPE person is
confirmed only when all of its required samples agree. If the resulting
crosswalk differs from HPE's seat-based proposal, the UI shows the difference
only at finalization and requires an explicit second acknowledgement.

## Outputs and completion meaning

Each scene is written under:

```text
data/annotations/hpe_transcript_crosswalk/<scene>/
```

The artifact contains a versioned manifest, deterministic review samples, the
final mapping CSV, a conflict report, and an append-only audit log. It binds the
accepted-window ledger, rebuilt Gaze bridge, HPE delivery and identity markers,
the relabelled camera evidence, and every source image/video version used for
review. A changed source causes resume/finalization to fail closed.

Finishing one scene means only that its Transcript-to-HPE semantic identity
crosswalk is confirmed. The full crosswalk stage is complete only after all ten
accepted scenes are confirmed and the live status ledger is updated. It still
does not mean that the mic-A shared timeline, multimodal feature table, or model
training readiness is complete.

Finally, `vap-intake` currently consumes the upstream HPE participant map. Once
the reviews are complete, its next revision must consume and bind these
crosswalk artifacts, and any older intake tables made with semantically
different HPE letters must be treated as stale and rebuilt.
