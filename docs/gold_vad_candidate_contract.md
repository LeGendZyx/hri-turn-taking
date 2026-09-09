# Gold-VAD candidate evidence and blind-audit contract

> Status: **implemented candidate scaffold; methodology parameters are not
> owner-approved**. Code authority:
> `src/hri_turn_taking/data/gold_vad_candidate.py`. This stage cannot publish
> counted gold, create `P1_vad`–`P4_vad`, or write an assembly feature manifest.

## Scientific boundary

The Transcript is a coarse, speaker-attributed anchor. It is not counted gold.
The four corrected lavalier channels are gold-tier evidence and are never model
inputs. This scaffold proposes boundary evidence and a deterministic blind
listening queue; it does not decide labels.

Every bundle permanently contains:

```json
{
  "candidate_only": true,
  "assembly_eligible": false,
  "counted_gold_ready": false,
  "source_selection": "owner_approval_pending"
}
```

The loader rejects any changed value. There is no CLI argument or code path that
can set one of these fields to a more permissive value. Renaming or copying a
bundle cannot make it formal gold.

## Required formal inputs

For scene `<scene>`, production code derives exactly these code-owned paths:

```text
data/processed/analysis_base/<scene>.csv
data/processed/analysis_base/<scene>.provenance.json
data/annotations/transcript_bridge/<scene>/
data/annotations/audio_gold_alignment/<scene>.audio_gold_alignment.json
```

An arbitrary bridge directory is refused even if the content-level bridge
loader accepts it. Alternate roots exist only in a private test API and are not
available from the production tool.

Before computation, the builder fully replays:

1. the `vap_real_analysis_base` parent and its exact integer 25-Hz crop;
2. the confirmed Transcript review-v3/final-v3 → acceptance-v3 → coarse-raster bridge;
3. the formal channel time map and its complete audio/pose/source replay;
4. all four 48-kHz mono PCM16 payloads and their hashes.

The analysis, Transcript and audio-authority half-open tick ranges must be
identical. Channel-local time is always
`shared_time_s - correction_s`. The bundle records each channel's inclusive
start sample, exclusive end sample and both exact fractional ceil residuals, so
correction-to-PCM quantisation can be independently replayed. The builder opens
no sample outside that formal interval and never pads or treats unavailable
coverage as silence.
Symlinks, junctions/reparse points, OneDrive Offline/recall placeholders,
changed hashes, short PCM payloads, non-finite values and mixed snapshots fail
closed. Source handles are held throughout calculation; sources are replayed
again before return/publication.

## Unapproved v1 candidate algorithm

These values reflect the present ten-scene diagnostic proposal. They are
versioned evidence for owner review, not a frozen counted method.

1. Compute per-channel RMS in a 20-ms window on a 10-ms grid, wholly inside the
   formal window. A parallel 40-ms view is diagnostic only.
2. Express energy in dB and subtract each speaker/session's own 10th percentile
   floor. For speaker `i`, define
   `relative_i = own_calibrated_i - median(other_calibrated)`.
3. Enter speech at `relative >= 3 dB` and own energy `>= floor + 8 dB`.
   Hold at `relative >= 0 dB` and own energy `>= floor + 5 dB`.
4. Fill internal gaps no longer than 40 ms; remove speech runs shorter than
   60 ms. Strong-core evidence requires `relative >= 6 dB` and own energy
   `>= floor + 10 dB`.
5. For every accepted Transcript event, inspect at most ±0.4 s and make the two
   endpoint choices jointly. Every suggestion must satisfy
   `formal_start <= start < end <= formal_end`; an inversion retains both coarse
   boundaries and records `ambiguity` plus `order_conflict`. Adjacent-event
   order/reuse and same-speaker coarse/refined overlaps receive the same
   conservative rollback. Intervals remain strictly half-open.
6. A global missed-token probe additionally requires `relative >= 6 dB`, own
   `>= floor + 10 dB`, own minus the strongest other channel `>= -3 dB`, and
   at least 80 ms. On the 10-ms decision grid, the same-speaker Transcript
   half-open mask is subtracted *before* connected components and the 80-ms
   minimum are evaluated. Cuts and immediate Transcript adjacency are recorded.
   It creates a **review flag only** and can never add speech.

The report includes per-speaker Transcript/candidate/strong prevalence,
support, Transcript-to-strong lag (positive means acoustic evidence is later),
missed-probe counts, refined/transcript overlap flags, and 20-ms versus 40-ms
agreement. Unusual prevalence is evidence for review, not an automatic scene exclusion.

The default edge rule does not use the 20 ms before the formal window. A
boundary whose requested context crosses an edge is flagged. Whether an
independently real, provenance-bound 20-ms outside context may later be used is
an explicit owner decision; this candidate does not assume permission.

## Deterministic stratified blind audit

The queue contains at least 50 items per formal scene:

| Stratum | Initial quota |
|---|---:|
| normal | 10 |
| low margin / large shift / ambiguity | 10 |
| overlap | 10 |
| short | 10 |
| candidate-only missed probe | 5 |
| Transcript-active but acoustic-low | 5 |

If a stratum, especially `short`, has fewer items, its shortfall is
deterministically reassigned by a SHA-256 rank bound to all source hashes and
the parameter hash. When evidence permits, every A/B/C/R participant receives
at least eight audit items. **Every flagged candidate is added to the audit
regardless of quotas**, so the queue can exceed 50.

The reviewer-facing queue contains only a neutral sequential audit ID, speaker
channel, original coarse event (or null for an unanchored clip), listening clip,
and a hidden-suggestion commitment. It never exposes candidate type, stratum,
flags, low-margin/large-shift/ambiguity diagnoses, boundary, 40-ms suggestion or
evidence score. In particular, an unanchored missed-probe clip is a fixed 4-s
window chosen from a 1-s formal-scene grid (plus a predeclared scene-end anchor)
with a source-hash-seeded tie-break. It is never `suggested boundary ± padding`
and is never clamped to a candidate boundary; multiple distinct suggestions can
therefore map to the same neutral clip. The source-hash-bound selection inventory
remains custodian-only during listening and is displayed alongside the
suggestion only after reveal.
`init-decisions` creates a separate file with null/false human fields; it never
autofills a decision. `reveal` refuses to run until every item has an explicit
listened/speech/speaker/boundary decision and a non-empty note. Only then does it
produce a separate comparison artifact. Reveal still has
`counted_gold_ready=false` and performs no automatic acceptance.

## Publication and replay

Candidate, decision and reveal artifacts are single strict-JSON files. Each is
staged on the destination filesystem, fsynced, and published with a no-clobber
hard link. Existing destinations and concurrent creators are preserved. A
`BaseException`, including one raised immediately after link creation, removes
only a destination still proven by staged identity, `samefile` and exact bytes
to be the writer's own hard link; a foreign replacement is preserved. Loading a
candidate re-hashes every bound source and recomputes the
entire evidence/audit plan from formal PCM before exact type-aware comparison.

Commands (do not run until the upstream formal bridge and analysis base exist):

```powershell
python tools/build_gold_vad_candidate.py build `
  --scene 05_15__3 `
  --output data/annotations/gold_vad_candidate/05_15__3.candidate.json

python tools/build_gold_vad_candidate.py audit `
  data/annotations/gold_vad_candidate/05_15__3.candidate.json

python tools/build_gold_vad_candidate.py init-decisions `
  data/annotations/gold_vad_candidate/05_15__3.candidate.json `
  --output <new-path>/05_15__3.blind_decisions.json

# Only after every blind field is completed by a human:
python tools/build_gold_vad_candidate.py reveal `
  data/annotations/gold_vad_candidate/05_15__3.candidate.json `
  <new-path>/05_15__3.blind_decisions.json `
  --output <new-path>/05_15__3.audit_reveal.json
```

No command above produces a train-ready table. A future, separately reviewed
gold-label contract is required after owner approval and completed blind-audit
acceptance criteria.

## Owner decisions still required

1. Approve or replace the q10 calibration, 20/10-ms window/hop, energy-ratio,
   hysteresis, morphology, snap and missed-probe thresholds.
2. Decide whether a real 20-ms context immediately outside the common formal
   window may be read, or whether the current in-window-only edge flag remains.
3. Freeze the human acceptance rule (the plan proposes ≥50 items/session,
   median absolute onset/offset error ≤40 ms and frame agreement ≥90%) and the
   action for failed speakers/scenes.
4. Decide how completed manual decisions become versioned `P{i}_vad` labels.
   This scaffold intentionally has no such promotion path.
