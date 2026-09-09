# Real-audio source and synchronization contract

> Version 1, implemented and read-only-smoke-tested on 2026-08-15. The code
> authority is `src/hri_turn_taking/data/audio_contract.py`. This contract is
> source/synchronization evidence only; it is **not** a gold-VAD artifact and
> can never set `counted_gold_ready=true`.

## Purpose and boundary

The filename `*_aligned.wav` is not evidence that four lavalier tracks are
sample-aligned. Likewise, a channel number in `master_aligned.wav` is not a
participant identity. Before gold-VAD construction or combined-mix acoustic
features, this contract independently establishes:

1. which immutable bytes and WAV headers belong to a scene;
2. whether the four-channel master is exactly the four named mono tracks in
   some permutation;
3. whether the transcript reference and master agree at the declared offset
   throughout the recording, rather than only at one global correlation peak;
4. whether B/C/R retain a stable residual lag relative to the mic-A anchor;
5. which half-open shared-time interval has real source coverage.

Passing these checks is only an alignment prerequisite. Counted gold still
requires transcript/lavalier refinement and the manual audit in
`final_model_plan.md` §4. No source is rewritten, shifted, or padded here.

## Code-owned source authority

Production callers cannot supply source paths through a manifest. For one
sealed scene `<group>__<scenario>`, the implementation derives and binds the
following exact canonical paths under the fixed corpus and HPE roots:

```text
<corpus>/<group>/Processed/<scenario>/audio/aligned_audio/A_aligned.wav
<corpus>/<group>/Processed/<scenario>/audio/aligned_audio/B_aligned.wav
<corpus>/<group>/Processed/<scenario>/audio/aligned_audio/C_aligned.wav
<corpus>/<group>/Processed/<scenario>/audio/aligned_audio/R_aligned.wav
<corpus>/<group>/Processed/<scenario>/audio/aligned_audio/master_aligned.wav
<corpus>/<group>/Processed/<scenario>/video/participants_rgb_audio_edited.wav
<corpus>/<group>/Processed/<scenario>/audio/aligned_audio/offset_report.json
<corpus>/<group>/transcript/<scenario>/offset.json
<hpe-delivery>/<group>__<scenario>/offset.json
<hpe-delivery>/<group>__<scenario>/offset_quality.json
```

Every source binding records absolute path, byte length, SHA-256, and, for WAV,
channel count, sample width, sample rate, frame count, and compression type.
It additionally reads the entire PCM payload in bounded blocks and records the
actual decoded-frame count, frame size, frame alignment, and completion flag.
The actual count must equal the header's `n_frames`; a physically truncated
payload is rejected even when its untouched RIFF header still advertises the
old duration.
JSON is decoded as strict UTF-8; duplicate keys, `NaN`/`Infinity`, bool-as-number,
and float/bool values in integer fields are rejected. The only upstream legacy
exception is Transcript `fps: 30.0`, which must be exactly the finite JSON float
`30.0`; it is not a general numeric coercion.

Expected audio layout is fixed:

- A/B/C/R: 48 kHz, mono, PCM16, equal frame counts;
- master: 48 kHz, four-channel, PCM16, the same frame count;
- Transcript reference: 16 kHz, mono, PCM16.

The HPE evidence must define `mic_A=0`, use
`shared_time = stream_local_time + offset_s`, name `A_aligned.wav` as the
anchor, and give the anchor zero error with status `ok`. Where HPE includes an
`rgb_d3_source_bundle`, its A path/hash/header must match exactly. Older scenes
without that bundle are explicitly reported as `semantic_anchor_only`, not
silently upgraded to historical exact provenance.

## Master-channel replay

The master and four mono files are read together in bounded 65,536-frame
chunks. Each master channel is compared sample-for-sample against every named
mono track. Acceptance requires one unique bijection covering A/B/C/R. A
duplicate channel, missing participant, altered sample, unequal frame count,
or ambiguous identical mono pair fails closed.

`offset_report.json.channels`, where present, is retained only as an upstream
declaration. A disagreement is reported as `conflicts_with_sample_replay`;
identity still comes exclusively from exact sample replay. Equal-weight
mixdown is order-invariant and is the only permitted master mixdown here.

## Segmented synchronization audit

The v1 audit produces a log-RMS envelope in non-overlapping 32 ms hops (512
samples at 16 kHz; 1,536 at 48 kHz). It then checks 30 s segments independently.
For each segment it searches ±256 ms around the declared lag, computes Pearson
correlation, refines the maximum parabolically to sub-hop resolution, and
compares it with the best peak outside the adjacent ±1-hop neighbourhood.
Degenerate/silent segments are not treated as evidence.

The same audit is used for:

- reference versus equal-weight master, centred on the Transcript-declared
  `lag_frames_rms × 32 ms`; and
- B/C/R versus A, centred on zero residual.

Version-1 quality thresholds are explicit and serialized into every report:

| Gate | v1 threshold |
|---|---:|
| Usable segments | at least 3 |
| Median peak correlation | at least 0.20 |
| Unique-peak fraction | at least 0.75 |
| Peak margin | at least 0.025 |
| Absolute median residual | at most 24 ms |
| 95th percentile absolute residual | at most 40 ms |
| Robust end-to-end drift span | at most 40 ms |

Drift uses a Theil–Sen median pairwise slope over segment centres. A periodic
signal with several equally good peaks therefore fails uniqueness even when a
global argmax exists. An offset with the wrong sign or one extra 32 ms hop
fails the residual gate. Positive B/C/R residual means that target channel is
later than A; `proposed_target_correction_ms = -median_residual_ms`. A proposed
correction is diagnostic only and is never applied to a WAV.

## Analysis-window rule

Transcript reference time maps to the mic-A shared axis as:

```text
[transcript_offset_sec,
 transcript_offset_sec + reference_duration_sec)
```

The usable non-padding interval is the intersection of that half-open range
with `[0, aligned_lavalier_duration)`. `validate_analysis_window()` rejects any
requested interval outside it, non-finite values, booleans, and zero-width
windows. Before returning an interval it performs a fresh full scene audit and
requires a type-exact match with the supplied report. Thus neither a stale
source binding nor a forged `nonpadding_intersection` can authorize a window
extending into a truncated or invented tail. Missing head/tail coverage must be
cropped; it must never be encoded as VAD=0 or acoustic silence.

## Real ten-scene contracts (2026-08-15)

Ten no-clobber manifests are published under
`data/annotations/audio_contract/` and have been independently reloaded with
full source/hash/header/PCM/fresh-report replay. All ten pass WAV structure,
strict JSON, exact master permutation, and segmented reference↔master sync.
They all retain `counted_gold_ready=false`. Inter-lavalier results correctly
keep six scenes behind the alignment-prerequisite gate:

| Scene | Exact master order (ch 0→3) | B/C/R median residual (ms) | Alignment prerequisite |
|---|---|---:|---|
| `03_20__4` | A/C/B/R | -0.4 / +10.4 / **-25.9** | fail |
| `03_26__3` | R/C/A/B | **-27.1** / -7.2 / -7.2 | fail |
| `05_14__2` | R/B/A/C | +0.6 / +2.5 / -4.9 | pass |
| `05_14__3` | R/B/A/C | **-30.0** / +13.3 / -19.5 | fail (C also drift/p95) |
| `05_14__4` | R/B/A/C | +1.6 / +22.0 / -10.0 | pass |
| `05_15__2` | R/C/A/B | -5.5 / -9.8 / -14.4 | pass |
| `05_15__3` | R/C/A/B | -3.6 / +4.2 / -22.9 | pass |
| `05_15__4` | R/C/A/B | **-28.6 / -67.4 / -28.8** | fail |
| `05_15__5` | R/C/A/B | +4.0 / **+24.6** / -0.7 | fail |
| `05_15__6` | R/C/A/B | **-36.0 / -38.4 / -28.1** | fail |

The eight 05_14/05_15 alignment reports that declare A/B/C/R conflict
with sample replay; the older 03_20/03_26 reports contain no channel-order
field. Exact historical HPE A-anchor bundles are present for `03_20__4`,
`03_26__3`, and `05_15__3`; the other seven retain semantic-anchor evidence
plus this contract's current exact A-WAV binding.

These measurements are quality evidence, not permission to shift sources.
Before counted use, failed scenes require a versioned correction proposal and
owner decision, followed by a fresh replay; unusual conversational behaviour
is not itself an exclusion reason.

## Commands and publication semantics

Read-only one-scene or all-scene audit:

```powershell
python tools/audit_audio_contract.py --scene 05_15__4
python tools/audit_audio_contract.py --all
```

Publication is optional and explicit (`--output` or `--output-dir`). The
builder writes a unique staged JSON, re-hashes every source, publishes through
an atomic no-clobber hard link, fully reloads/replays the published contract,
and removes its own output on any `BaseException`. It never overwrites an
existing destination. This is trusted-code integrity/reproducibility, not a
cryptographic signature or protection against a malicious local code runner.
