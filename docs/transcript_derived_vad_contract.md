# Human-reviewed Transcript-derived VAD contract

Status: **target v1 + semantic-decision v2 implemented and published for all
ten scenes**. The owner supplied the four targeted event decisions; all ten
target bundles pass deterministic replay. Code authority:
`src/hri_turn_taking/data/transcript_derived_vad.py`.

## Why this path exists

On 2026-08-16 the project owner clarified that the selected Transcript
annotations had already been human-reviewed and reliably encode speaker,
overlap, quiet speech and pauses. The earlier proposal to listen again to all
1,094 in-window events, audit 50 clips per scene, or repeat part of the audit
after seven days is therefore retired. It would duplicate completed annotation
work and cannot fit the remaining dissertation schedule.

This is a new owner-attested contract. It does not edit or reinterpret the old
review-v3 artifacts, and it does not falsely claim that review-v3 was completed.
Every owner attestation says existing_human_annotation=true,
targeted_reaudit_only=true, and all_events_reaudited=false.

The research label is **human-reviewed-transcript-derived VAD**. It is not
described as sample-perfect lavalier VAD. A 30 fps annotation boundary is
quantized, and a labelled utterance may contain a short within-utterance pause;
both limitations are retained in every target contract and in the dissertation.

## The only new human decisions

Only four impossible or ambiguous in-window events need an explicit owner
decision. The immutable shared Transcript JSON remains untouched; each decision
is a local exact-preimage event overlay:

| Candidate | Exact owner-confirmed event overlay |
|---|---|
| 05-14-2-e27-event | end 00:00:29:21 -> 00:00:39:21 |
| 05-15-2-e76-event | end 00:03:27:02 -> 00:03:57:02; speaker B -> A |
| 05-15-3-e17-event | start 00:00:34:25 -> 00:00:33:25; text `Okay, xxxx.` -> `Okay, let's stop there.`; end remains 00:00:34:25 |
| 05-15-3-e88-event | end 00:02:28:15 -> 00:02:38:15 |

Each decision is accepted, rejected, or replaced. Acceptance applies the whole
event overlay atomically; an alternate replacement supplies one complete
`start_timecode/end_timecode/speaker/text` event object. Rejection is recorded
honestly; if it leaves a non-positive source interval, target construction
remains blocked until a positive replacement is supplied. Semantic decision
files are versioned as `semantic_decisions.v2.json`; owner attestations and VAD
target bundles remain v1. The old 05_14__3 event 146 proposal is wholly outside
the canonical analysis window and is not a fifth decision.

Mechanical changes (trailing ASCII spaces, the unique FF=30 carry, and the
unique missing JSON quote) remain code-owned exact-preimage operations.

## Target derivation

For each scene:

1. replay the canonical timeline, offset, analysis-base CSV/sidecar, audio
   contract, gold-alignment contract and A/B/C/R WAV hashes;
2. validate the owner attestation against those exact live paths and bytes;
3. apply only exact-preimage mechanical repairs and explicit targeted decisions;
4. union same-speaker half-open intervals;
5. transform t_shared = timecode_frame / 30 + transcript_offset_sec;
6. set a canonical analysis tick active iff
   start_shared <= tick/25 < end_shared.

The output covers exactly the analysis-base interval. It neither creates rows
outside that interval nor fills unavailable coverage as silence. Slots are
fixed as P1=R, P2=A, P3=B, P4=C. A three-person scene keeps P4 explicitly
absent and P4_vad=0; the separate static presence column prevents that zero
from being mistaken for an observed silent participant.

Every bundle contains:

- vad_source.v1.csv: independent evidence with keys, presence and VAD;
- vad_target.v1.csv: assembly input with keys and P1–P4 VAD;
- vad_contract.v1.json: owner/source/decision/tick/QC contract;
- vad_target.v1.manifest.json: assembly feature-manifest v3.

The manifest uses purpose gold_target, offline alignment and exact identity
replay from the independent source CSV. It is directly consumable by
vap-assemble. Automatic QC reports event/speaker counts, source ordering,
same- and cross-speaker overlap, union counts, active ticks, exact window,
binary/absence rules and all hashes. No new signal threshold is a hard gate.

## Trust and publication

The owner accepted a trusted, single-writer local workspace. Production paths
are code-owned; copies and re-anchored artifacts fail replay. Writes are
canonical and no-overwrite. The four-file target bundle is an explicitly
multi-file transaction and does not claim OS-level atomicity or resistance to
a hostile concurrent replacer.

Under this deadline-oriented boundary, the writer performs exactly one fresh,
full live-source replay before constructing and publishing the bundle. After
publication it verifies only the four files it just wrote: guarded canonical
paths, exact staged bytes and SHA-256 values, strict JSON equality/schema, CSV
parsing and dtype/value round trips. The writer deliberately does **not** claim
a post-publication fresh source replay. Consequently, a concurrent source
replacement after the pre-publication replay is a documented residual risk
accepted by the owner under the single-user/single-writer assumption.

The standalone loader/audit remains the consumption gate: every invocation
performs one fresh full source replay, deterministically reconstructs the
expected bundle and runs the assembly feature-manifest replay. A bundle must
pass that loader/audit against then-current sources before later assembly use.
Injected fixture authority always produces authority_scope=fixture,
assembly_eligible=false, and counted_gold_ready=false; it cannot masquerade as
a production artifact.

## Commands

Record the already-completed annotation fact once per scene:

    python tools/build_transcript_derived_vad.py attest --scene 03_20__4 ^
      --owner "<owner>" --attested-at "2026-08-16T10:00:00+01:00"

List/export the four correction-aware short candidates:

    python tools/build_transcript_derived_vad.py catalog
    python tools/build_transcript_derived_vad.py clips ^
      --candidate-id 05-15-3-e88-event --mode separate

Publish decisions:

    python tools/build_transcript_derived_vad.py decide --scene 05_15__3 ^
      --reviewer "<owner>" --decided-at "2026-08-16T10:30:00+01:00" ^
      --decision "05-15-3-e17-event=accepted=owner supplied start and wording" ^
      --decision "05-15-3-e88-event=accepted=owner confirmed end boundary"

Then build/audit:

    python tools/build_transcript_derived_vad.py build --scene 05_15__3
    python tools/build_transcript_derived_vad.py audit --scene 05_15__3

No command overwrites an existing attestation, decision, clip directory or
target-bundle member.
