# ASD source preflight contract

> Status: the read-only preflight is implemented, but formal acceptance is
> **BLOCKED at owner retry round R3/3**. Fixture provenance can masquerade as
> formal provenance, and ancestors above the configured authority root are not
> checked for reparse points. The three-round repair-and-retest limit is
> exhausted: do not start a fourth repair round without owner direction. No ASD
> model or weights have been downloaded, no model has been executed, no
> probability or feature has been generated, and nothing has been published or
> admitted to assembly.

## Frozen ten-scene camera plan

The participant-facing view is the sole input for `A/B/C`; the opposite view is
the sole input for the human demonstrator `R`.

| Scenes | A/B/C | R | Source FPS |
|---|---|---|---:|
| `03_20__4`, `03_26__3` | `Cam_2` | `Cam_6` | 30 |
| `05_14__2..4` | `Cam_2` | `Cam_6` | 60 |
| `05_15__2..6` | `Cam_1` | `Cam_6` | 30 |

`05_14__3` and `05_14__4` are strictly first-segment-only. Segment 2 is
excluded unavailable data, never a zero-filled negative observation.

## Authority replay

`audit_asd_source_preflight.py` loads the canonical `vap_real_analysis_base`,
replays its sealed HPE-to-Transcript crosswalk, verifies the bound HPE validation
and identity manifests, and invokes the production audio-contract loader for a
complete fresh source replay. The audio manifest must be type-exactly identical
to that replay, including SHA-256, WAV header, complete PCM payload, and the
four-lavalier exact-permutation proof. Copies at caller-selected paths are
rejected even if their content is identical. Consumed authority documents use
before/read/after file-identity checks and are rehashed after a second sealed
crosswalk replay. These checks reduce mixed-snapshot risk only within the paths
they inspect; they do not establish formal provenance while fixture provenance
can masquerade as formal, and they do not inspect ancestors above the
configured authority root for reparse points. Formal authority acceptance
therefore remains fail-closed pending owner review.

Shared 25 Hz tick `t` maps to the camera frame

```text
floor((t / 25 - d3_offset_s) * source_fps)
```

using exact rational arithmetic. The selected frame must be the latest frame at
or before the shared tick and must remain inside the reviewed half-open source
window. Nearest-frame matching, interpolation, tolerance joins, extrapolation,
and future lookahead are forbidden.

ASD audio may use only `audio_contract.sources.master`, deterministically mixed
by exact PCM16-to-int64 summation followed by float64 division by
`4 * 32768`. A single lavalier, Transcript field, gold VAD, or gold-channel time
correction is forbidden as model input.

Cam6 is not accepted merely because an AVI path exists. The sealed crosswalk
must bind both canonical `relabeled_Cam_6.csv` and Cam6 `run_metadata.json`.
The run metadata, not string replacement alone, owns the selected AVI segment.
For the confirmed HPE person mapped to `R`, every analysis tick is classified as
valid, missing, or ambiguous from a unique `face_matched` row with a finite,
positive face box and non-empty association reason. Missing and ambiguous rows
remain unavailable and are never filled with zero. The current 05_15 Cam6
relabeled schema has no face-box/association columns, so those R sources are
explicitly blocked; earlier scenes without sealed Cam6 bindings are also
blocked. This preflight does not claim a usable R ASD stream.

## Raw-media and publication boundary

The preflight only calls `lstat`-style filesystem metadata operations for AVI
files. It never opens an AVI byte stream. Directly checked directories,
symlinks, junctions, and reparse points (including OneDrive placeholders) are
rejected without being followed or hydrated; ordinary local files remain
`bytes_unverified_no_hash_read`. This is a read-only path/type preflight, not
proof of raw-media bytes, content, participant identity, camera suitability, or
formal provenance. In particular, the available Cam6-R evidence is insufficient
and remains fail-closed; it must not be promoted to an accepted ASD source.
Ancestors above the configured authority root are not yet covered by the
reparse-point check.
Every status is `ready_for_extraction=false`.
This module has no output path, writer, publisher, assembly manifest, or schema
promotion API. Every report hard-codes `assembly_eligible=false` and
`counted_input_ready=false`.

## Stock-model causality finding

- Light-ASD is frozen for review at commit
  `ed38c232de5efe0261dbd68627c0ade7cdfe14eb`. Its symmetric temporal padding,
  backward/reversed GRU pass, and whole-track validation make the stock model
  noncausal.
- LR-ASD is frozen for review at commit
  `1b6dcd2d8fc2895683de6508ec6294ec47d388ca`. Its symmetric temporal
  convolution, forward/backward GRU fusion, and whole-track validation also make
  the stock model noncausal.

Therefore neither stock implementation can be called a causal robot-observable
feature extractor. A future adapted implementation must pass all three tests on
real model outputs before publication: future-perturbation invariance, prefix
equivalence, and chunk-boundary equivalence. Any failure blocks publication;
the preflight records these as required but does not fabricate model outputs.

Run the read-only check with:

```powershell
& .\.venv\Scripts\python.exe tools\audit_asd_source_preflight.py --all
```
