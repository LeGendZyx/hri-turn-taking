# Transcript manual-review workflow

This workflow prepares human review of the ten sealed Transcript timelines. It
does not edit any file under the corpus root, does not silently apply a proposed
semantic correction, and cannot produce counted gold VAD.

The Transcript remains a **coarse, speaker-attributed activity reference**.
Per-participant lavalier VAD must be prepared and audited separately before a
counted experiment.

## Safety contract

- Draft and final artifacts bind the exact timeline, Transcript offset, the
  fully replayed audio contract, the formal audio-gold-alignment artifact,
  reference WAV, and aligned A/B/C/R lavalier WAV bytes.  A draft cannot be
  created until that scene's formal gold-alignment artifact exists and passes
  a fresh audio/pose/source replay.
- Production review artifacts are anchored to
  `data/annotations/transcript_review/<scene>/`; the initial draft is
  `draft.rev0000.json`, later drafts are `draft.revNNNN.json`, and the single
  final is `final.v3.json`. The formal bridge is anchored separately to
  `data/annotations/transcript_bridge/<scene>/`. Production loaders reject a
  copied/renamed artifact or bundle even when all embedded hashes are otherwise
  self-consistent. Explicit paths remain available only to injected test
  authority helpers.
- Source JSON types and schemas are replayed. In particular, the known legacy
  `offset.json` token `"fps": 30.0` is accepted only through its exact
  byte-offset-bound type normalization to integer `30`.
- Formal review is restricted to the gold-alignment artifact's authorised
  half-open 25 Hz interval. Every source event appears exactly once either in
  the editable in-scope queue or in the immutable excluded inventory. Counts
  of source, in-scope, excluded and boundary-crossing events are replayed and
  conserved; missing, duplicate or reordered snapshots fail validation.
- An event crossing a window boundary contributes only its half-open
  intersection. A wholly outside event is never promoted by a manual repair.
  A non-positive interval that is provably wholly outside is inventory, not a
  blocker; a non-positive interval touching the window still requires review.
- Mechanical repairs are exact-preimage operations. The five audio/owner
  boundary proposals start as `unresolved` and affect only the suggested
  listening interval until a reviewer explicitly accepts them.
- Publication is no-clobber. Sources and input review bytes are replayed
  immediately before and after publication. Bundle rollback requires the
  invocation token, original physical directory identity and exact payload
  bytes; a concurrently replaced target or reused staging name is preserved.
- A finalized review always retains:

  ```json
  {
    "output_role": "coarse_transcript_activity_reference",
    "review_status": "confirmed",
    "counted_gold_ready": false
  }
  ```

## Commands

Create the canonical initial draft:

```powershell
python tools/review_transcript_events.py init `
  --scene 05_15__3
```

Audit a manually edited draft and print progress:

```powershell
python tools/review_transcript_events.py audit `
  data\annotations\transcript_review\05_15__3\draft.rev0000.json
```

The recommended authoring path is the standard-library review GUI, so the
reviewer does not need to edit roughly one thousand JSON event records by hand:

```powershell
python tools/review_transcript_events.py gui `
  data\annotations\transcript_review\05_15__3\draft.rev0000.json
```

It shows scene/event progress, source text and flags, prioritises unresolved or
flagged events, and plays the four correction-mapped A/B/C/R lavaliers either
separately or as an equal-weight listening mix. Clips are generated on demand
under the operating-system temporary directory and deleted when the UI closes;
the corpus WAVs are opened read-only. Missing files, links/reparse points and
OneDrive Offline/recall placeholders fail closed before full replay, so the UI
does not initiate an implicit multi-GB download. Each playback hashes the full
draft-bound WAV and reads the requested samples through that same no-follow
handle; on Windows the handle denies WRITE and DELETE for the whole operation.
Playback and navigation never set `reviewed=true`: only **Apply reviewed
decision** does so. Semantic proposals also need their own explicit
accept/reject listening note.

**Save new revision** publishes a new, fully replayed, no-clobber draft JSON;
it never overwrites the opened draft. Resume by starting `gui` with that latest
revision, or by using **Open / resume revision**. Because saving re-hashes the
bound sources, a checkpoint may take a little time. Draft revisions and bridge
bundles are explicitly refused inside the corpus and HPE source roots.

Finalize only after the audit reports every **in-scope** event and in-scope
semantic proposal resolved, `decision_consistency_ok=true`, and
`ready_to_finalize=true`:

```powershell
python tools/review_transcript_events.py finalize `
  data\annotations\transcript_review\05_15__3\draft.rev0001.json `
  --reviewer "<reviewer name>" `
  --reviewed-at "2026-08-15T12:00:00+00:00"
```

No command overwrites an existing artifact. Make a new revision path when a
decision changes.

Re-audit a finalized artifact, including its bound draft and live sources:

```powershell
python tools/review_transcript_events.py audit-final `
  data\annotations\transcript_review\05_15__3\final.v3.json
```

After a review-v3 final passes that replay, bridge it deterministically into
the independent acceptance-v3 contract and its strictly windowed 25-Hz raster:

```powershell
python tools/review_transcript_events.py bridge `
  data\annotations\transcript_review\05_15__3\final.v3.json

python tools/review_transcript_events.py audit-bridge `
  data\annotations\transcript_bridge\05_15__3
```

The output directory is a no-clobber bridge-receipt-v2 bundle containing
`transcript_acceptance.v3.json`, `coarse_activity_25hz.csv`, and
`bridge_receipt.json`. The receipt binds the exact final and draft, the compact
acceptance and raster, accepted repair set, candidate decisions, per-event
reviews, formal window and source/in-scope/excluded/crossing count conservation.
Accepted events can only be replayed from the confirmed final's post-review
repairs. Both outputs retain `output_role=coarse_transcript_activity_reference`
and `counted_gold_ready=false`; they are not training labels.

## Review fields

Each event contains the immutable source snapshot, flags, an A/B/C/R listening
clip plan, and this editable decision block:

```json
{
  "reviewed": false,
  "speaker": {"action": "unreviewed", "replacement": null},
  "start_timecode": {"action": "unreviewed", "replacement": null},
  "end_timecode": {"action": "unreviewed", "replacement": null},
  "note": ""
}
```

For a reviewed event, set `reviewed` to the JSON boolean `true`; set every
field action to `keep` or `change`. A `change` requires a replacement and a
non-empty note. A `keep` requires a JSON `null` replacement. Speaker
replacements are restricted to A/B/C/R; timecodes remain strict 30/1 non-drop
`HH:MM:SS:FF`.

Each semantic candidate has an independent decision:

```json
"decision": {"status": "unresolved", "note": ""}
```

The reviewer must change `status` to `accepted` or `rejected` and add a note.
Acceptance must also be mirrored by the matching event field as a `change` to
the exact proposed value. Rejection cannot select that value indirectly.

The clip plan is planning metadata only. It first intersects the event with the
authorised shared-time window, clamps 1.5 s padding to that same window, and
then gives a separate local interval for each read-only lavalier using
`channel_local_time_s = shared_time_s - correction_s`. A positive correction
therefore moves the local clip earlier; a negative correction moves it later.
It never expands authority, cuts, or writes audio.

Prevalence is never reused across stages. Artifacts report three independently
recomputed views: `source_pre_review`, `in_scope_pre_review`, and (in a final
review) `accepted_post_review`. Scene flags that describe the raw source are
explicitly prefixed `source_pre_review_` and are retained even if manual review
later repairs the imbalance. In review-v3 the field remains named
`scene_flags`, but former live review requirements are preserved only as
historical `review_trigger_*` flags. They must not be read as unresolved after a
confirmed final; measured `source_pre_review_*`, `in_scope_pre_review_*`, and
`accepted_post_review_*` facts retain their explicit stage prefixes.

The review-v3 draft/final schema is intentionally incompatible with review v1
and v2. Those old artifacts are rejected with an instruction to recreate the
draft and repeat human review; they are never relabelled or migrated in place.
The bridge accepts only a fully replayed, confirmed current review-v3 final,
uses bridge receipt v2, and emits only the independent current acceptance-v3;
old/unconfirmed review, receipt, or acceptance versions fail closed. A final review
asserts `all_in_scope_events_reviewed=true`; it never claims that excluded
source events were listened to or accepted.

## Built-in ten-scene review priorities

All events must still be listened to; flags only determine priority.

| Scene | Priority review |
|---|---|
| `03_20__4` | suspected e36 speaker mismatch; long e18 |
| `03_26__3` | C e11/e12 overlap |
| `05_14__2` | e27 end candidate; four trailing-timecode spaces; long e79/e113 |
| `05_14__3` | e146 is wholly outside the authorised window and remains immutable inventory; e151 FF=30 is a global mechanical carry normalization; review in-window overlaps/long event |
| `05_14__4` | long e130 boundary |
| `05_15__2` | e76 end candidate and targeted speaker-assignment mismatches |
| `05_15__3` | e17/e88 end candidates, ambiguous segmentation, targeted speaker mismatches and overlaps |
| `05_15__4` | A has zero labelled events; repeated-text cluster e7–e20; full speaker/text reannotation required |
| `05_15__5` | B has 98/119 labelled events; full speaker reannotation required |
| `05_15__6` | exact missing-quote and speaker-space repairs; source-pre-review low C prevalence; in-window speaker/overlap/order review |

## Replayed ten-scene scope counts

These counts use the formal gold-alignment windows and are regression evidence,
not human-review completion status. Columns are `in scope / excluded / crossing`:

| Scene | Counts |
|---|---:|
| `03_20__4` | 57 / 1 / 1 |
| `03_26__3` | 97 / 1 / 2 |
| `05_14__2` | 125 / 51 / 2 |
| `05_14__3` | 122 / 31 / 2 |
| `05_14__4` | 130 / 48 / 0 |
| `05_15__2` | 80 / 7 / 0 |
| `05_15__3` | 128 / 1 / 1 |
| `05_15__4` | 75 / 1 / 0 |
| `05_15__5` | 117 / 2 / 1 |
| `05_15__6` | 163 / 9 / 1 |

The implementation contains the source-indexed repair catalogue and evidence in
`src/hri_turn_taking/data/transcript_review.py`. As of implementation, all ten
plans have been replayed in memory against their current sources, but **no
formal draft or final artifact has been published and no scene has been claimed
as human-confirmed**.
