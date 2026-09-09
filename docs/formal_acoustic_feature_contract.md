# Formal causal acoustic feature contract

This contract covers the counted real-data acoustic inputs `array_energy`,
`f0_proxy`, and `voiced`. The canonical output root is
`data/processed/acoustic_features/`.

## Frozen method

- The source is the complete canonical four-channel `master_aligned.wav`.
- Every int16 sample is converted to float64; each source frame is the
  equal-weight mean of all four channels divided by 32768.
- The complete source is causally resampled and extracted before the canonical
  analysis ticks are selected. Gold per-channel time corrections are not used.
- Energy and F0 use the right-aligned causal windows frozen in
  `docs/methodology.md`. Normalized autocorrelation uses the owner-approved
  voicing threshold 0.45. Every analysis tick must have valid energy and pitch,
  finite values, binary `voiced`, and `f0_proxy == 0` when unvoiced.
- Publication assumes the owner-approved local, trusted, single-user and
  single-writer boundary. Hashes are reproducibility/integrity evidence, not a
  defence against a process able to replace trusted code and every bound file.
- Counted publication is available only through the exact singleton returned by
  `default_formal_acoustic_authority()`. It rechecks the code-owned analysis-base,
  audio-source, frozen-inventory, output, and review roots. Caller-constructed or
  fixture authorities always have scope `noncanonical_fixture_non_counted`; they
  may exercise pending construction but reviewed seal/load rejects them.

## Review gate

`--build-pending` writes, without overwrite, exactly four per-scene files:

- `<scene>.source.csv`
- `<scene>.evidence.json`
- `<scene>.csv`
- `<scene>.pending.manifest.json`

The pending manifest has kind `vap_feature_source_review_pending`, status
`review_pending`, and is deliberately not accepted by `vap-assemble`. The
builder never creates or approves a review receipt.

An independent reviewer writes `<scene>.review.json` with these exact keys:

```json
{
  "artifact_kind": "vap_real_causal_acoustic_review_receipt",
  "format_version": 1,
  "scene": "05_15__2",
  "decision": "accepted",
  "independent_review": true,
  "reviewer_id": "reviewer-name",
  "reviewed_at_utc": "2026-08-16T12:00:00Z",
  "review_scope": [
    "canonical_source_timeline_and_mix_binding",
    "causal_extractor_and_threshold_contract",
    "artifact_type_hash_and_replay_gates",
    "assembly_v3_manifest_conformance",
    "trusted_local_single_user_single_writer_boundary"
  ],
  "reviewed_bundle": {
    "source_csv": {"path": "ABSOLUTE_PATH", "sha256": "...", "size_bytes": 0},
    "evidence_json": {"path": "ABSOLUTE_PATH", "sha256": "...", "size_bytes": 0},
    "feature_csv": {"path": "ABSOLUTE_PATH", "sha256": "...", "size_bytes": 0},
    "pending_manifest": {"path": "ABSOLUTE_PATH", "sha256": "...", "size_bytes": 0}
  },
  "notes": ""
}
```

Every reviewed-bundle binding has exactly `path`, `sha256`, and `size_bytes`.
The paths, byte sizes, and SHA-256 values must describe the current canonical
pending bundle exactly. `--seal-reviewed` validates this external receipt and,
only under the default canonical production authority, creates
`<scene>.manifest.json`, the assembly feature-manifest v3 sidecar.
That sidecar binds the exact keys, ordered columns, typed values, source files,
causality parameters, runtime, threshold decision, analysis base and sidecar,
audio contract, master bytes/header/permutation, and extractor implementation.

## Operations

Run one scene at a time:

```powershell
python tools/build_formal_acoustic_features.py 05_15__2 --build-pending
python tools/build_formal_acoustic_features.py 05_15__2 --audit-pending
python tools/build_formal_acoustic_features.py 05_15__2 --seal-reviewed
python tools/build_formal_acoustic_features.py 05_15__2 --audit-reviewed
python tools/build_formal_acoustic_features.py 05_15__2 --raw-reaudit
```

The two audit modes are sealed downstream loads and do not decode the full WAV.
`--raw-reaudit` recomputes the bundle from the canonical raw sources. This
minimal delivery intentionally has no all-scenes command or completion
inventory: each scene remains uncounted until its own independent review seal
exists.
