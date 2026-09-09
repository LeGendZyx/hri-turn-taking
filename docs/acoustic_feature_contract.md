# Causal acoustic feature contract

> Extractor version 1, implemented 2026-08-15. Code authority:
> `src/hri_turn_taking/data/acoustic_features.py`. This contract implements the
> frozen algorithm. The real-data source decision was closed on 2026-08-15:
> use the sample-replayed equal-weight mono mix of the four channels in
> `master_aligned.wav`. This is an explicitly reported close-talk-lavalier
> surrogate, not a robot-ear or far-field recording.

## Boundary and output status

The extractor accepts one explicitly named mono PCM16 WAV candidate. It emits
only a versioned `.npz` evidence artifact with source-local timestamps,
nullable feature values, validity masks, the exact source hash/header, and the
complete algorithm parameters plus NumPy/SciPy runtime versions. Its artifact kind is
`vap_causal_acoustic_candidate_evidence`; both `assembly_eligible` and
`counted_input_ready` are permanently `false`.

Consequently, renaming this artifact cannot make it a real feature source. The
combined-mix source kind has been selected, and the owner froze the
normalized-autocorrelation voicing threshold at `0.45` on 2026-08-16. A
separate formal builder may bind the accepted audio/synchronization contract,
map absolute mic-A ticks into the common analysis window, and write an ordinary
v3 feature manifest only through a new formal builder and independent review.

## Frozen signal algorithm

All processing is causal and deterministic:

1. Decode a complete mono, uncompressed PCM16 WAV. The implementation rejects
   short payloads, changed bytes, non-finite in-memory PCM, and unsupported
   channel/sample formats.
2. Convert to 16 kHz. Version 1 accepts 16 kHz or an integer multiple up to
   192 kHz (the corpus source is 48 kHz). Downsampling uses a one-sided causal
   Kaiser-windowed FIR followed by integer decimation. The filter has
   `64 × factor + 1` taps, at least 80 dB rejection from the 8 kHz target
   Nyquist boundary through every accepted source Nyquist, and an uncompensated
   2 ms causal group delay recorded as runtime latency. It does not use
   `resample_poly`'s centred offline alignment. Samples whose filter history
   reaches before the real recording are marked invalid; an incomplete tail
   target sample is discarded.
3. At 25 Hz tick `t`, compute `array_energy = log(max(RMS, 1e-6))` from exactly
   the completed trailing 640 samples, corresponding to `(t-40 ms, t]`.
4. Compute pitch from exactly the completed trailing 1,280 samples,
   corresponding to `(t-80 ms, t]`. Remove the mean, divide autocorrelation at
   every lag by the zero-lag window energy, retain local peaks spanning
   50--500 Hz, and select the highest peak (shortest lag on an exact tie).
   `voiced=1` when that peak is at least `0.45`; then
   `f0_proxy=log2(F0/100 Hz)`. Otherwise `voiced=0` and `f0_proxy=0`.
5. A missing 40/80 ms history, resampler warm-up, or source boundary produces
   `NaN` plus `energy_valid=false` / `pitch_valid=false`. Padding is never
   interpreted as silence or unvoiced audio. Formal assembly must crop to rows
   where both masks are true rather than zero-fill them.

The generic candidate artifact uses source-local 25 Hz ticks. It makes no claim
about the mic-A shared-time offset; that mapping belongs to the real-corpus
preflight and later source-approved formal builder.

## Real-corpus preflight (not a formal feature artifact)

`src/hri_turn_taking/data/real_acoustic_features.py` and
`tools/build_real_acoustic_features.py` close the source and timeline wiring in
read-only memory. They:

1. fully replay the canonical audio contract and the canonical
   `vap_real_analysis_base` parent;
2. decode the complete four-channel 48 kHz PCM16 `master_aligned.wav`;
3. convert each int16 frame to `float64`, compute its four-channel arithmetic
   mean, and divide by `32768`, with no intermediate PCM16 quantization and no
   gold-channel time corrections;
4. run the causal extractor on the complete master before selecting the frozen
   absolute mic-A 25 Hz ticks; and
5. require every one of the 78,582 common-window rows to have finite energy/F0,
   binary `voiced`, and both validity masks true.

The preflight has no writer or publish option. Its report is hard-coded to
`assembly_eligible=false`, `counted_input_ready=false`, and
`parameter_status=owner_approval_pending`. It is evidence that the real source
and time axis can drive the extractor; it is not evidence that `0.45` has been
approved or that a v3 feature source exists.

### Fixture-only assembly-v3 scaffold

The public entry point is
`build_read_only_artifact_scaffold(scene, *, authority=None)`. It accepts only
an inventory scene identifier and an optional code-owned
`RealAcousticAuthority`; it rejects scenes outside
`analysis_base.EXPECTED_TICK_RANGES`, then performs one fresh
`derive_real_acoustic_preflight` replay before constructing anything. A caller
cannot supply a `RealAcousticPreflight`, feature frame, or report. The fresh
preflight can become four immutable **in-memory-only** payloads: an independent
source CSV, an evidence JSON object, a feature CSV, and a feature-manifest
preview. This exercises the exact v3 identity-source shape needed later:
global columns have JSON-null participant and slot bindings;
`(session_id,timestamp_s)` keys and values replay exactly; the feature bytes
bind the source/evidence hashes; and causality is fixed to trailing windows
with zero lookahead.

The frame schema is fixed. `session_id` has exact object dtype and every value
is an exact string equal to the canonical scene; `timestamp_s`,
`array_energy`, `f0_proxy`, and `voiced` have exact `float64` dtype. Boolean,
integer, extension-string, and other coercible dtypes are rejected. The strict
CSV parser consumes the complete payload, serialises the parsed frame again via
the canonical `_csv_payload` representation, and requires byte equality. It
therefore rejects trailing blank records, extra whitespace, and alternative
float spellings. Source and feature frames are compared column by column using
exact dtype, shape, and bitwise array hash; this distinguishes `+0.0` from
`-0.0` and never relies on `DataFrame.equals`.

The strict path uses JSON-type-exact recursive comparison throughout. A JSON
boolean is never accepted as integer `1`, and integer `0` is never accepted for
the declared floating-point `0.0` lookahead. Parsing rejects duplicate keys,
the named `NaN`/`Infinity` constants, and finite-looking numeric tokens whose
conversion overflows (for example `1e999`) at any nested depth.

The embedded preflight report is not treated as an opaque note. Its complete
format-v1 key set is replayed and hash-bound inside the evidence. Validation
requires the frozen source-selection status and kind; exact analysis-base data
and provenance paths/hashes plus key schema; exact audio-contract path/hash and
identity; the master path/hash, WAV header, complete-PCM schema and mix replay;
the current complete extractor parameter tree; the causal and validity
contracts; and every feature array's dtype, shape, row count, hash, and values
recomputed from the independently parsed source CSV. The feature CSV must in
turn match every source column's dtype, shape, and bitwise value hash, and the
manifest binds both byte payloads.

The build-time trust boundary is source-backed: every public build freshly
reopens and validates the selected analysis, audio, and master inputs through
`derive_real_acoustic_preflight`. Exact dataclass identity alone is never a
substitute for that replay. The load-time boundary is deliberately narrower:
the strict in-memory loader does not reopen source paths or derive a new
preflight; it validates only the four supplied immutable, nonpublishable byte
payloads, including the complete embedded report and all cross-bindings. A
strict in-memory load therefore cannot promote or refresh the scaffold.

This scaffold is intentionally rejected by the production assembler:

- its manifest kind is `vap_feature_source_read_only_scaffold`, **not** the
  accepted `vap_feature_source` kind;
- the accepted v3 kind/version appear only inside a
  `target_contract.status=preview_only_not_emitted` description;
- evidence and manifest both retain `assembly_eligible=false` and
  `counted_input_ready=false`;
- the owner freeze token is null and format v1 states that such a token is not
  supported; `promotion_implemented=false` and a new independent review is
  mandatory;
- neither builder nor loader accepts a destination path, and the module exports
  no write/publish API.

Therefore even an owner-token string supplied by a caller cannot unlock this
format. A future formal artifact requires an explicit owner decision freezing
the voicing threshold, a new reviewed format, code-owned publication paths, and
separate publication-transaction review. The current scaffold must not be
written to `data/`, passed to `vap-assemble`, or counted as a completed input.

## Integrity and causality gates

This publication paragraph applies only to the generic candidate NPZ evidence;
the real-corpus preflight above has no publisher. Candidate publication is a
single-file, atomic no-clobber hard link from a unique staged
file. The builder validates and deterministically replays the staged file,
publishes without replacement, validates/replays the published file again,
and removes only its own link after any `BaseException`. Loading always
re-hashes and fully decodes the bound WAV before replaying every stored array.
Strict format/type checks reject a bool/float masquerading as an integer
version or parameter.

The regression suite includes:

- perturbing every source sample after tick `t` at both 16 and 48 kHz, while
  requiring every output through `t` to remain bit-identical;
- changing the final sample whose interval ends at `t`, proving the right edge
  is included while the following sample is excluded;
- 40/80 ms boundary and causal-resampler warm-up cases;
- silence, 50/100/200/500 Hz tones, NaN/complex/non-mono/bad-rate inputs;
- truncated WAV payload, source replacement, array/metadata/version tampering,
  concurrent destination creation, and post-publication interruption.

## Approved real-data source

The core source is the deterministic equal-weight mono average of the four
channels in `master_aligned.wav`, after the audio contract has replayed the
channel bytes against A/B/C/R and rejected any unproven channel-order claim.
This selects a complete combined-mix surrogate without reading participant
identity from channel index.

The choice is deliberately conservative about semantics. The upstream run
record says that `master_aligned.wav` was manually muxed with the participant
video, after which `participants_rgb_audio_edited.wav` was extracted from that
MP4. The latter is therefore a shorter, encoded/resampled derivative used for
Transcript timing, not independent evidence of a physical room microphone.
It remains the reference-coverage authority but is not the model-input source.

No gold-tier per-channel timing correction is applied before this mix. Those
corrections exist only to place the four lavalier gold-label channels on the
mic-A axis. Every counted manifest must state the resulting deployment domain
shift: training acoustics come from an equal-weight close-talk-lavalier mix,
whereas Furhat would observe a physical far-field microphone.
