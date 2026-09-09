# Gold-audio channel alignment contract

Status: implemented and published for the selected ten-scene corpus. Ten
versioned, fail-closed time-map manifests have passed deterministic replay and
independent review; the original WAV files remain unchanged.

## Purpose and boundary

The per-person A/B/C/R lavalier tracks are gold-label sources for VAD. They are
not model inputs. This contract maps each lavalier's local time into the shared
mic-A axis without modifying or remixing any WAV:

```text
shared_time_s = channel_local_time_s + correction_s
channel_local_time_s = shared_time_s - correction_s
```

The combined master/reference audio used for robot-observable energy, F0 and
`voiced` is not shifted by this contract. A successful time map still records
`counted_gold_ready=false`: transcript-conditioned VAD validation remains a
separate required gate.

## Frozen selection hierarchy (v1)

For every scene and every B/C/R channel, the same hierarchy applies:

1. Replay the published audio contract and the sealed pose-intake CSV/sidecar.
2. Test an identity map over the formal common window with the frozen 32-ms
   segmented audit.
3. If all coarse gates pass, retain identity.
4. If the only coarse failure is drift, `abs(drift_span_ms) <= 72`, and an
   independent 5-ms replay passes all original 40-ms gates, retain identity as
   `resolution_confirmed` and preserve the coarse warning. The 72-ms ceiling is
   the 40-ms target plus a conservative half-hop error at both ends of a 32-ms
   span.
5. Otherwise make exactly one second attempt: add the negative of the published
   full-file v1 median residual. This number is independent of the formal
   outcome window; no value is fitted to formal-window results.
6. Apply the same coarse/fine hierarchy to that constant map. Any other failure
   is fail-closed. Linear warp, manual shift and scene-specific tuning are not
   supported.

The 5-ms audit uses non-overlapping log-RMS, 30-s segments, and at least 12 s
for a tail segment. The requested ±256-ms search and 32-ms peak-exclusion
neighbourhood are conservatively rounded up to the next 5-ms tick (±260 ms and
35 ms). It uses the same median, p95, drift, uniqueness and correlation
thresholds as v1.

## Current ten-scene read-only replay

The table records seconds added to channel-local time. A blank correction means
identity. Tick intervals are exact half-open 25-Hz mic-A ranges.

| Scene | B correction | C correction | R correction | Authorised ticks |
|---|---:|---:|---:|---:|
| `03_20__4` | — | — | `+0.025896740` | `[376, 5332)` |
| `03_26__3` | `+0.027105291` | — | — | `[315, 4636)` |
| `05_14__2` | — | — | —* | `[601, 7351)` |
| `05_14__3` | `+0.030030763` | `-0.013279631` | — | `[624, 12338)` |
| `05_14__4` | — | `-0.022002523` | — | `[687, 12383)` |
| `05_15__2` | — | — | — | `[656, 6973)` |
| `05_15__3` | — | — | `+0.022913371` | `[586, 6398)` |
| `05_15__4` | `+0.028646922` | `+0.067382168` | `+0.028784864` | `[618, 7458)` |
| `05_15__5` | — | `-0.024632953` | — | `[481, 7924)` |
| `05_15__6` | `+0.035984352` | `+0.038395943` | `+0.028116300` | `[629, 13362)` |

`05_14__2` R is the sole current resolution-confirmed identity: the 32-ms drift
span is -41.130 ms while median, p95, uniqueness and correlation pass; the
independent 5-ms replay passes every gate with drift -32.737277 ms. The contract
does not call the coarse result a pass and does not apply an unnecessary
+4.938-ms shift.

Across the ten windows this exposes 78,582 usable 25-Hz ticks. No extrapolation
or silence padding is permitted. For a channel of duration `D` and correction
`c`, real shared coverage is `[c, D+c)`. The common window is the intersection
of all four such intervals, transcript-reference coverage, and the sealed pose
range. Its exact ticks are:

```text
k_start = max(pose_start, ceil(25 * common_start))
k_end   = min(pose_end,   ceil(25 * common_end))
ticks   = [k_start, k_end)
```

All arithmetic that selects these bounds uses `Decimal`; no unavailable sample
may be treated as zero-valued audio.

## Scientific caveat

The residual is an acoustic-envelope alignment diagnostic, not automatically a
hardware-clock estimate. In `05_14__3`, residual changes are strongly associated
with which wearer is speaking, consistent with lavalier dominance, bleed and
propagation effects. This is why v1 rejects linear drift correction even when a
line could improve aggregate correlation. Final counted labels still require a
speaker-conditioned transcript/VAD audit and a documented exclusion/sensitivity
policy.

## Usage

Read-only replay (the default):

```powershell
python tools/audit_audio_gold_alignment.py --scene 05_14__2
python tools/audit_audio_gold_alignment.py --all
```

Publication is deliberately explicit and no-clobber via `--output` or
`--output-dir`. Do not publish until independent review approves this contract.
