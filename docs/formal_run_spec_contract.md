# Fixture-only formal-run specification binding

> Status: isolated in-memory scaffold; permanently non-counted.
> This contract does not approve a formal run, publish an artifact, register a
> backend, or authorize real data.

## Purpose

`src/hri_turn_taking/vap/formal_run_spec.py` closes one narrow design gap: it
makes the candidate execution configuration hashable and lets a manifest-like
mapping be checked against that exact configuration. It is deliberately
separate from `outer_experiment.py`, the backend registry, the run index, and
all publishers.

The module binds all of the following in one strict JSON object:

- condition ID, feature-set name, encoder, and the code-owned ordered active
  feature columns;
- the complete fixture `FrozenTorchSpec` and its SHA-256 identity;
- model, VAP label, AdamW, training-window, epoch-budget, batch, and
  warmup/cosine scheduler configurations;
- permanent non-counted blockers and the decisions still pending from the
  owner.

The duplicated model, label, optimizer, and window fields are intentional.
Validation re-derives them from `FrozenTorchSpec`; a caller cannot change one
copy while leaving the other copy untouched.

## Trust and eligibility boundary

Every spec and manifest-binding payload contains exact, code-owned values:

```text
fixture_only = true
counted_eligible = false
publisher_available = false
writer_available = false
```

There is no writer, loader from disk, real-data entry point, registry hook, or
training function in this module. A valid payload is only an in-memory
foundation for later reviewed integration. It cannot be cited as a counted
experiment manifest.

The hard blockers remain:

1. fixture-only contract;
2. no outer-run-manifest integration;
3. no registered production backend;
4. no implemented formal batch/scheduler execution path;
5. no formal data-readiness authorization;
6. no atomic formal run index; and
7. no publisher.

The payload also records three owner-pending decisions: the formal
`max_epochs` value, calibration/decision thresholds, and the filesystem trust
boundary. Thresholds are intentionally absent from the spec schema.

## Caller-supplied values are candidates, not approvals

`build_formal_run_spec(...)` has no defaults. The caller must explicitly pass
the condition, `FrozenTorchSpec`, model, label, optimizer, window, scheduler,
`max_epochs`, and `batch_size` values. This prevents the binding layer from
silently choosing a training budget. It does not turn the supplied values into
owner-approved settings; `formal_max_epochs_approval_pending` remains in every
payload.

The optimizer mapping spells out PyTorch AdamW defaults that the fixture
backend currently inherits (`betas`, `eps`, and boolean/nullable switches), as
well as the no-decay grouping rule, gradient clipping, and zero auxiliary-VAD
weight. The scheduler mapping records its step unit, total-step calculation,
warmup rounding, cosine rule, and minimum learning-rate scale. This module
validates those declarations but does not execute them.

## Exact JSON and hash contract

Both payload kinds require exact key sets and exact JSON types. In particular:

- booleans never satisfy integer fields;
- `12.0` never satisfies an integer `max_epochs=12`;
- `0` never satisfies a float field such as `dropout=0.0`;
- NaN and infinity are rejected;
- extra and missing keys are rejected;
- active columns must exactly equal `active_feature_columns(feature_set)` in
  canonical schema order;
- label values and types must exactly match the bound `FrozenTorchSpec`;
- all redundant SHA-256 values are replayed before acceptance.

Canonical serialization uses UTF-8 JSON with sorted keys, compact separators,
Unicode preserved, and non-finite values forbidden. `spec_sha256` hashes the
entire spec body except the self-hash field. `binding_sha256` follows the same
rule for the manifest-like view.

## Manifest-like consistency check

`build_manifest_spec_binding(spec)` produces a second in-memory view with the
field names a future run manifest needs. `validate_manifest_spec_binding(
manifest, spec)` checks:

- both permanent fixture/non-counted bases;
- the formal-spec and `FrozenTorchSpec` identities;
- condition and ordered active columns;
- model, label, optimizer, window, budget, batch, and scheduler values with
  exact types; and
- the manifest-binding self-hash.

Even a tampered manifest whose attacker recomputes `binding_sha256` is rejected
when any bound value differs from the supplied spec.

## Promotion requirements

Promotion is intentionally outside this module. A future version must be a
separate reviewed change that integrates the binding into the outer run
manifest, implements batch/warmup-cosine execution, binds environment/source
evidence, adds the atomic run index, resolves the owner filesystem decision,
registers a production backend, and adds a reviewed publisher. The current
fixture schema must not be relabelled or reused as counted evidence.
