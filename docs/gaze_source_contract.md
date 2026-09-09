# Nullable 2D Gaze evidence preflight contract

> Status: implemented as a strictly read-only, repository-local feasibility
> preflight. The canonical `--all` replay passes 10/10 scenes and 78,582 shared
> ticks after the crosswalk-API name was corrected. It remains permanently
> non-publishing and cannot authorize assembly, schema promotion, a counted
> input, or a visual experiment condition.

## Purpose and scientific boundary

The current `gaze_features_export/` files contain a per-detection 2D category:

```text
gaze_target in {robot, participant, elsewhere}
```

That category is the strongest supported claim. `participant` does not identify
which participant was viewed. The export is not a calibrated 3D gaze vector and
this preflight does not intersect a ray with 3D head positions. It therefore:

- never derives participant-wise `gazed_at`;
- never claims a 3D gaze intersection;
- never supplies complete incoming-gaze denominators;
- never treats `speaking_towards` as a substitute; and
- never promotes any value into an assembly feature manifest.

For a human participant with a confirmed identity observation, the categorical
evidence may support a nullable binary diagnostic:

```text
gaze_robot = 1 iff gaze_target == robot
gaze_robot = 0 iff gaze_target in {participant, elsewhere}
```

The binary is valid only when the same source frame also contains the uniquely
confirmed robot bounding-box reference. Missing identity, a missing target, or
a missing confirmed robot reference is unavailable evidence (`valid=false`,
value `NA`), not a negative observation.

## Frozen corpus and authority chain

The preflight accepts only the ten scenes and 78,582 shared ticks frozen by
`vap_real_analysis_base`:

`03_20__4`, `03_26__3`, `05_14__2`, `05_14__3`, `05_14__4`,
`05_15__2`, `05_15__3`, `05_15__4`, `05_15__5`, `05_15__6`.

`05_14__3` and `05_14__4` remain first-segment-only. A caller cannot supply a
different scene root, accepted window, identity file, or source CSV. The loader
follows this exact repository-local chain instead of inferring authority from a
basename:

```text
canonical analysis-base CSV + provenance
  -> sealed crosswalk inventory and its five-file scene record
  -> finalized crosswalk manifest and conflict-free final mapping
  -> bound Gaze identity manifest and tracklet observations
  -> bound raw frame_features.csv and sibling session_metadata.json
```

Every consumed file is path-, size-, and/or SHA-256-bound by its owning sealed
document. Ordered columns, parsed dtypes where authoritative, row counts,
session identity, accepted window, fixed `R=P1, A=P2, B=P3, C=P4` slots, static
presence, and the analysis-base integer ticks are replayed. The raw Gaze export
must have the documented ordered 16-column schema. A non-null target outside
the three allowed categories fails the scene.

The preflight reads repository-local CSV/JSON/Markdown evidence only. It does
not open, hash, decode, download, or hydrate a video, and it does not inspect
the `F:` corpus. This is a sealed-small-file integrity/reproducibility check,
not a new raw-media re-audit and not a cryptographic signature.

## Exact past-only time mapping

For shared 25 Hz tick `t`, D3 offset `o`, and source frame rate `f`, the selected
Gaze frame is:

```text
floor((t / 25 - o) * f)
```

The code uses exact rational arithmetic from the recorded decimal scalar and
checks:

```text
frame / f + o <= t / 25 < (frame + 1) / f + o
```

The selected frame must remain inside the sealed half-open Gaze window.
Nearest-frame rounding, interpolation, tolerance joins, forward fill,
extrapolation, and future lookahead are forbidden.

## Nullable row semantics

The in-memory diagnostic grid has four rows per analysis tick, one per fixed
slot. Its two independent validity fields prevent missingness from becoming a
false zero:

- `gaze_target_valid`: a unique confirmed non-R detection carries one allowed
  2D target category;
- `gaze_robot_valid`: the category is valid and the same frame also has a
  unique confirmed R bounding-box reference.

The robot player R appears in the source only as the robot bounding-box
candidate. There is no R gaze-estimation row. Every P1/R row therefore remains
`gaze_target_valid=false`, `gaze_robot_valid=false`, and both values are `NA`.
This means *unobserved*, never “R is not looking at the robot.”

An absent slot, if a three-person table is introduced, is independently marked
`slot_present=false`; both validity fields are false and both values remain
`NA`. It is not conflated with R's missing modality and not conflated with an
observed binary zero.

Unconfirmed/uncertain identity rows remain unavailable. A missing observation
closes only the affected tick/slot to invalid + `NA`, so coverage can be
reported honestly. In contrast, any of the following invalidates the whole
scene:

- duplicate raw `(frame_index, person_idx)` rows;
- duplicate/conflicting confirmed `(frame_index, canonical participant)` rows;
- a non-empty identity or crosswalk conflict report;
- an unsupported target category;
- a path/hash/schema/row/tick/window/slot mismatch; or
- a source file changing during the preflight snapshot.

No duplicate is averaged, selected by row order, or silently dropped.

## Publication and counting boundary

`src/hri_turn_taking/data/gaze_source_readiness.py` returns only an in-memory
dataframe and report. Every report hard-codes:

```text
assembly_eligible = false
counted_input_ready = false
writer_available = false
artifact_or_feature_published = false
```

There is no output-path argument, writer, publisher, assembly manifest, feature
promotion function, or connection to `vap-assemble`, `vap-train`, or the formal
outer runner. Passing this preflight proves only source/timeline/identity
feasibility and nullable coverage. Counted Gaze still requires an owner-approved
derivation, modality-specific validity schema, formal artifact contract,
publisher, and independent review.

Run the read-only report with:

```powershell
& .\.venv\Scripts\python.exe tools\audit_gaze_source_readiness.py --scene 03_20__4
& .\.venv\Scripts\python.exe tools\audit_gaze_source_readiness.py --all
```

The tool writes no file. Progress goes to stderr and the JSON coverage summary
goes to stdout.
