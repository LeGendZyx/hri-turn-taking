# Production evidence foundation contract (v1)

Status: implemented as a read-only, in-memory foundation. It is deliberately
**not** a counted experiment system.

The implementation is `hri_turn_taking.vap.production_evidence`. It supplies
four fail-closed evidence objects for a future production runner:

1. a byte-level source-state inventory;
2. a fresh runtime/environment snapshot;
3. a real-data 25 Hz tick-authority snapshot; and
4. a pure D3 central/minus/plus pose-timing reassembly contract.

Every object permanently carries:

```text
counted_eligible = false
publisher_available = false
evidence_role = in_memory_foundation_not_a_formal_result
```

There is no writer, run-index mutation, backend registration, training entry
point, or result publisher in this module. Passing these gates therefore cannot
turn an artifact or a run into dissertation evidence.

## 1. Source-state inventory

`build_source_state_snapshot(repository_root)` reads the live bytes directly;
it does not call Git and does not depend on the index or HEAD. Consequently,
modified and untracked source files under the frozen scope are sealed exactly
as they exist at capture time.

The code-owned scope is exact:

- root files: `AGENTS.md` and `pyproject.toml`;
- every `.py`/`.pyi` regular file below `src/`;
- every `.json`/`.md`/`.toml`/`.yaml`/`.yml` regular file below `configs/`;
- only the protocol documents named by `EXPLICIT_PROTOCOL_DOCS`.

Each file record contains a canonical POSIX repository-relative path, byte
size, and SHA-256. The sorted file list and its complete sorted directory
ancestry are hashed together with the scope declaration. Symlinks, path
escapes, duplicate paths, unsafe entries, and an in-flight file change fail the
capture. The repository root, both tree roots, every traversed file/directory,
and even an ignored `__pycache__` entry are inspected with `lstat` before the
code enters or ignores them. POSIX symlinks, Windows junctions, and every entry
carrying `FILE_ATTRIBUTE_REPARSE_POINT` are rejected; resolving a path first
must never erase this evidence. For every explicitly named root file and
protocol document, each path segment from the repository root is checked before
the next segment is opened. The resolved repository-relative path must also be
byte-for-byte identical to the code-declared canonical path. Thus a linked
`docs/` parent cannot redirect an apparently valid protocol filename to another
tree, even when the target remains inside the repository.

Top-level `data`, `model(s)`, `log(s)`, `report(s)`, and `output(s)` trees are
forbidden. They may exist in the working repository, but they are never
admitted to this source inventory. A supplied inventory that names one of them,
or any file outside the exact scope, is rejected. Code packages such as
`src/hri_turn_taking/data/` remain source because their accepted contents are
restricted to `.py`/`.pyi`. Raw media and generated
experiment/report bytes therefore cannot be mistaken for source state.

`validate_source_state_snapshot(..., fresh_replay=True)` re-enumerates and
rehashes the live tree. An added, removed, renamed, or changed scoped file makes
the replay fail.

## 2. Environment snapshot

`build_environment_snapshot()` accepts no caller-supplied environment fields.
It observes the current process and records:

- resolved Python executable, implementation, version, and version tuple;
- OS, release, version, machine, and architecture;
- exact versions of PyTorch, NumPy, pandas, and scikit-learn;
- logical CPU count and physical count when `psutil` can provide it;
- PyTorch intra-op/inter-op thread counts, deterministic-algorithm and
  warn-only flags, MKLDNN state, CUDA availability, and default dtype; and
- a canonically sorted `importlib.metadata` distribution inventory plus its
  hash (the pip-freeze-equivalent inventory, obtained without a pip subprocess).

All JSON types are checked exactly: booleans cannot stand in for integers,
non-finite numbers are forbidden, and nested schemas reject extra/missing
fields. `validate_environment_snapshot(..., fresh_replay=True)` rebuilds the
snapshot from the live process; a caller cannot make a self-reported value
authoritative by merely recomputing its JSON hash.

The snapshot is process-local. A future formal runner must capture it in the
same fresh process that executes the run; this module does not implement that
runner.

## 3. Real 25 Hz tick authority

`build_real_tick_authority_snapshot(csv_paths, feature_sets=..., authority=...)`
accepts only canonical real `vap_real_analysis_base` or authorised assembled
CSV/`.provenance.json` pairs. It delegates structural/provenance replay to the
existing assembly gate and additionally binds, per scene:

- scene and provenance-owned group identity;
- absolute canonical CSV and sidecar paths, sizes, and SHA-256 values;
- the complete contiguous integer `tick_25hz` sequence and bounds;
- the shared `mic_A_shared_seconds` axis at exactly 25 Hz;
- real and placeholder column declarations;
- target-readiness blockers and readiness for each explicitly frozen feature
  set; and
- the exact self-checked `6cam` D3 offset-quality record, including the
  recorded timing error bound.

The snapshot also binds the complete dataframe produced by the freshly replayed
canonical loader. Its digest domain includes the ordered column names, exact
pandas dtypes, row count, and every cell value. Numeric columns are required to
be finite and are hashed in canonical little-endian binary form; object columns
must contain exact UTF-8 strings. This is distinct from the CSV byte hash: the
CSV hash proves which source bytes were loaded, while the dataframe digest
proves which parsed table is later presented to D3 reassembly.

Dataframe timestamps must equal `tick_25hz / 25` exactly. When a `tick_25hz`
column is present it must have an integer dtype and equal `timestamp_s * 25`
row-for-row. Tick start/end/count fields use exact-integer checks, so JSON
booleans cannot pass as `0` or `1`. The sequence must match the analysis
sidecar's authorised tick range. Files are hashed before and after
loading, and loader-reported hashes must equal the live bytes. The snapshot is
only returned in memory and remains non-counted even when every data-readiness
field is true.

`validate_real_tick_authority_snapshot(..., fresh_replay=True)` reloads every
bound CSV/sidecar through the same canonical authority. Structurally plausible
but shifted ticks, changed bytes, a substituted scene/group, or stale readiness
cannot pass fresh replay.

## 4. D3 central/minus/plus contract

D3 is a sensitivity reassembly, not a label or audio perturbation.
`build_d3_reassembly_contract(...)` freshly replays the real tick authority by
default and binds the scene, group, feature set, full input ticks, source CSV
and provenance hashes, the complete canonical-loader dataframe digest/schema,
D3 quality, and tick-authority inventory hash. `reassemble_d3_variants` first
recomputes that full digest. A caller cannot alter an acoustic value, gold VAD,
column order, dtype, or any other cell while merely inheriting the original
source hashes.

The transform has two modes:

- If the selected feature set activates `P*_head_yaw_deg`, each active yaw is
  shifted together with its `pose_valid` and `pose_quality` channels.
- If no head-yaw input is active (notably `audio_only`), central, minus, and
  plus are byte-semantically identical dataframe copies with no crop or shift.

For an active pose condition, let `e` be the non-negative recorded
`error_bound_ms`. The three aligned source-time deltas are exactly `0`, `-e`,
and `+e` milliseconds for central, minus, and plus. For each target tick the
algorithm chooses the latest shifted pose sample whose aligned timestamp is at
or before that target. It never interpolates and never extrapolates. All three
variants are cropped to their common non-extrapolated tick intersection.

Only the active yaw/validity/quality triplets can change. Gold VAD, causal
acoustics (`array_energy`, `f0_proxy`, `voiced`), session/timestamp fields, and
every other column are copied at the same target rows and asserted identical
across variants. The result contains an in-memory audit with the exact source
tick selected for each variant.

## 5. Hard blockers before any counted use

Every v1 evidence object records all of these blockers:

- owner filesystem/source-state trust decision is pending;
- an atomic formal run index is not implemented;
- formal data readiness is not authorised by this foundation;
- a publisher is intentionally absent; and
- no production backend is registered for counted use.

Closing one blocker must not silently remove the others. A future publisher
must be a separate reviewed component and must freshly validate all evidence,
the formal split/run manifest, and owner decisions. These v1 objects must never
be grandfathered into counted artifacts by editing a flag.

## Verification

The bounded fixture tests do not access the corpus or the F: drive:

```powershell
pytest -q tests/test_production_evidence.py
ruff check src/hri_turn_taking/vap/production_evidence.py `
  tests/test_production_evidence.py
```

They cover changed/untracked source bytes, forbidden/out-of-scope paths,
fresh environment replay, canonical real-sidecar/tick/hash binding, shifted
tick forgery, D3 no-lookahead/no-extrapolation behaviour, protected-column
identity, audio-only identity, and permanent non-counted blockers.

After the R3/3 adversarial review, one read-only real build replayed the exact
ten canonical analysis-base pairs: 78,582 contiguous 25 Hz ticks across groups
`03_20`, `03_26`, `05_14`, and `05_15`. `target_ready`, `audio_only` readiness,
and `audio_plus_head_pose` readiness all remained false, as required while gold
VAD and formal acoustic manifests are absent. No snapshot was written and this
smoke does not alter the permanent non-counted status.
