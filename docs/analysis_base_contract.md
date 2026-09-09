# Common analysis-base contract

Status: the fail-closed builder/loader contract is implemented and ten formal
CSV/sidecar pairs are published under `data/processed/analysis_base/`. The
matrix has 78,582 rows and passes sealed directory audit. It remains
`trainable=false`; publication created neither gold VAD nor acoustic inputs.

## Purpose

Each accepted pose-intake table covers the full human-reviewed HPE interval.
Corrected A/B/C/R lavalier channels and the Transcript reference can have a
shorter real coverage. The analysis base is the exact common 25 Hz key grid on
which later acoustic features and gold VAD are built. It remains
`trainable=false`; cropping does not create labels or promote placeholders.

The only supported transform is an integer row-offset slice:

```text
shared tick k = 25 * timestamp_s
rows = parent[(k_start - parent_start):(k_end - parent_start)]
interval = [k_start, k_end)
```

There is no float-tolerance join, interpolation, endpoint padding,
extrapolation, or caller-supplied bound. `05_14__3` and `05_14__4` retain their
sealed first-segment-only scope.

## Provenance and replay

Artifact kind `vap_real_analysis_base` uses real provenance format v2 and adds
a versioned `vap_real_analysis_window_contract`. Two validation modes are
explicit and must never be conflated:

- **Strict build/raw re-audit.** Every in-memory derivation, publication, and
  explicit re-audit calls the complete `load_audio_gold_alignment` path. This
  replays the audio contract, source WAV evidence, pose intake, alignment
  hierarchy, and exact time map. A publisher performs a strict replay before
  staging and again after publication.
- **Sealed downstream replay.** Routine assembly, audit, split, and fold loads
  strictly parse and hash the already-reviewed formal gold-alignment manifest,
  re-hash its bound audio-contract and exact pose CSV/sidecar, replay the full
  pose parent and integer crop, and compare the complete correction/warning
  snapshot. They deliberately do not decode several GiB of WAVs on every fold.
  This is not a silent downgrade: the sidecar records both modes, and the
  explicit raw re-audit remains available.

Both modes prove:

1. the canonical `data/processed/intake_v2/<scene>.csv` and sidecar, including
   the sealed HPE→Transcript identity and D3 source chain;
2. the canonical
   `data/annotations/audio_gold_alignment/<scene>.audio_gold_alignment.json`
   as exact formal bytes (with full raw replay in strict mode);
3. the fact that the gold artifact binds the exact same parent CSV and sidecar;
4. the authorised integer ticks and exact parent row slice.

The sidecar's `validation_modes` records `strict_build_replay` and
`downstream_sealed_replay` explicitly. It binds the bytes of all three upstream files, the parent and output
tick ranges, row offsets, A/B/C/R mapping corrections, every channel's status
and `coarse_warning`, the complete D3 status/error summary, and the sealed pose
window. It also binds the original published analysis-base path, so copying or
hard-linking a CSV/sidecar pair to manufacture a second parent is rejected.
The path is not trusted merely because the sidecar repeats it: live
`AnalysisBaseAuthority` fixes the formal base to
`data/processed/analysis_base/<scene>.csv`. Test authorities must be injected
explicitly; the production CLI exposes no authority/path override.
Thus the `05_14__2` R `resolution_confirmed` warning cannot disappear
through cropping. An assembled descendant must retain and replay the identical
analysis-window contract and immediate analysis lineage.

Generated intake, analysis-base, or assembled tables—and byte copies of the
current base—cannot serve as independent feature ownership in a feature
manifest. Only independently bound upstream feature evidence may promote a
placeholder.

Publication uses unique staging files and no-clobber hard-link publication for
the CSV/sidecar pair. It is an explicitly two-phase, fail-closed pair and does
not falsely claim that two filesystem links are one OS-atomic operation. A
`BaseException` atomically moves each final name to a unique quarantine name
before inspecting ownership. A proven writer inode is deleted; a foreign inode
is restored without replacing a newer owner, or retained at an explicitly
reported recovery path if restoration is no longer possible. Rollback never
uses a path-check-then-unlink sequence. A crash-time partial pair is rejected on
load and must not be re-signed. Sources are replayed before and after
publication, and the published pair is loaded through the full gate.
After the potentially long final strict raw-source replay, the writer checks
the destination through held read handles. Each handle is checked by
`fstat -> hash/read -> fstat`, the live path must still name that same inode,
and the exact pair is replayed while both handles remain open. On Windows the
handles allow other readers but deny WRITE and DELETE sharing during this
validation; on POSIX they use `O_NOFOLLOW` and repeat the live-path identity
check. In-place mutation and same-byte new-inode replacement therefore fail;
rollback deletes only writer-owned inodes.

This is a trusted-code, single-writer integrity contract, not a claim of
security against arbitrary hostile kernel/filesystem scheduling. In
particular, the POSIX live-path checks detect the supported race schedules but
do not purport to defeat an attacker that can repeatedly replace and restore a
directory entry between every syscall. Such an attacker is outside the stated
research-artifact trust boundary and must not be invoked to weaken the normal
fail-closed gates.

Analysis descendants have a second live code-owned path gate. The production
authority authorises exactly the ten scene-matched CSV paths under
`data/processed/formal_a_h_ah_v1/`, one full-schema table per scene shared by
the downstream A, H, and A+H feature masks. It does not authorise the stale
`data/processed/real/` directory, alternative filenames, or arbitrary caller
destinations. Fixtures enumerate their exact descendant paths explicitly.
`assembly.output_csv` must also equal the artifact currently being loaded, so a
copied/re-anchored descendant fails before use. Freezing these destinations
authorises publication only; it does not assert that any assembled artifact
already exists or is train-ready.

## Frozen ten-scene matrix

| Scene | Authorised 25 Hz ticks | Rows |
|---|---:|---:|
| `03_20__4` | `[376,5332)` | 4,956 |
| `03_26__3` | `[315,4636)` | 4,321 |
| `05_14__2` | `[601,7351)` | 6,750 |
| `05_14__3` | `[624,12338)` | 11,714 |
| `05_14__4` | `[687,12383)` | 11,696 |
| `05_15__2` | `[656,6973)` | 6,317 |
| `05_15__3` | `[586,6398)` | 5,812 |
| `05_15__4` | `[618,7458)` | 6,840 |
| `05_15__5` | `[481,7924)` | 7,443 |
| `05_15__6` | `[629,13362)` | 12,733 |

Total: **78,582 rows**. Any changed range fails closed and requires a newly
reviewed corpus contract rather than silently changing the counted dataset.

On 2026-08-15, the strict builder replayed all ten real scenes from the raw
gold-alignment evidence and reproduced every row count above (78,582 total),
including `05_14__2` R as `resolution_confirmed` with
`coarse_warning=true`. After an independent transaction review, all ten
canonical pairs were published with no-clobber semantics and a subsequent
sealed directory audit returned `structural_ready=true`,
`target_ready=false`, and selected-acoustic readiness false. Exactly ten CSVs
and ten sidecars exist; no extra file or corpus completion inventory was
created.

## Commands

Read-only replay (does not create `data/processed/analysis_base/`):

```powershell
python tools/build_analysis_base.py --scene 05_14__2
python tools/build_analysis_base.py --all
```

Explicit full raw-source re-audit of an already published base or descendant:

```powershell
python tools/build_analysis_base.py `
  --reaudit data/processed/analysis_base/05_14__2.csv
```

The publication command used for the current canonical matrix was:

```powershell
python tools/build_analysis_base.py --all --publish
```

The tool has no arbitrary output option and refuses existing CSVs or sidecars.
Ten-scene publication consists of ten independently fail-closed no-clobber
pairs; it is **not** an atomic corpus transaction and this version did not
write a completion inventory. The completed matrix was therefore verified by
exact file count, per-pair canonical path/hash/row/tick replay, and the sealed
directory audit. Any future batch interruption still follows the conservative
rule: do not train, re-sign, silently skip, bulk-delete, or rerun over existing
paths. Record the completed paths and independently validate ownership/hashes
before any recovery. A corpus-level immutable completion seal remains a future
hardening item, not evidence that the current bases are train-ready.

Formal training remains blocked
until separately audited gold VAD and the selected input feature manifests are
assembled and pass `vap-audit --require-ready`.
