# Production PyTorch backend v3 scaffold contract

> Status: **implemented for deterministic synthetic fixtures only; permanently
> non-counted in this version.** Last updated: 2026-08-15.

`src/hri_turn_taking/vap/production_backend.py` is an executable rehearsal of
the future PyTorch backend boundary used by `outer_experiment.py`. It is not a
formal experiment backend. It is deliberately absent from
`_PRODUCTION_BACKEND_REGISTRY`, is not exposed by the CLI, accepts only a
caller-declared fixture authority, and stamps every output with
`counted_eligible=false`. The declaration is not yet a repository-generator
replay proof, so it cannot authenticate that arbitrary tensors are synthetic.

This separation is intentional: model-fitting correctness can be tested before
the unresolved publication trust boundary, environment capture, run index, D3
pose-shift reassembly, and owner-approved calibration policy are allowed to
affect a dissertation result.

## 1. Frozen specification

`FrozenTorchSpec` is the only model/optimisation specification accepted by the
scaffold. Its canonical JSON binds:

- the exact transformed schema: seven shared participant features and five
  global features, four slots, four future bins, 33 model inputs;
- `session_id` plus integer `tick_25hz` as frame identity;
- model width/layers/heads/dropout, Transformer or GRU encoder and attention
  window;
- the complete VAP label configuration and its hash;
- training window/stride, inference chunk size, learning rate, weight decay and
  gradient clip;
- CPU-only deterministic algorithms and seeded epoch-wise window order; and
- `aux_vad_weight` as the exact JSON float `0.0`.

The artifact stores both `input_schema_sha256` and `spec_sha256`. Loaders rebuild
the schema and hashes instead of trusting stored values. Unexpected fields,
wrong JSON types, reordered feature names, non-finite numbers, or a changed hash
are rejected.

## 2. Session and time identity

Every fixture session needs a `SessionTickBinding` before fitting. A binding
contains:

```text
session_id
data_kind = synthetic_fixture
tick_25hz = contiguous integer ticks
session_content_sha256
```

`session_content_sha256` covers the session/group identity, exact ticks, array
dtypes/shapes, and the bytes of features, labels, VAD, validity and presence.
`TickAuthorityV3` requires unique lexicographically sorted sessions. The backend
rechecks these hashes before and after each fit/evaluation, so a caller cannot
silently exchange a session or shift its time axis.

This v3 authority rejects an explicitly declared `data_kind=real`, but does not
prove that a caller did not relabel arbitrary tensors as a fixture. A later
synthetic path must replay the repository generator, and a real authority must be built
from the audited assembled-table provenance and exact stored 25 Hz ticks; merely
renaming this fixture authority is not an acceptable promotion path.

## 3. Training boundary

The outer runner supplies both `FeatureNormalizer` and per-bin `pos_weight`.
The backend validates, copies, hashes and uses them; it never calls
`FeatureNormalizer.fit` or recomputes class weights. Inputs and statistics are
rehashed after fitting.

Training uses only the participant-wise four-bin VAP loss. The current-VAD
tensor is not read by the loss, so changing VAD while holding features/labels
fixed produces identical learned weights. Tests exercise this isolation.

Epochs are one-based. Window starts are deterministically permuted from
`(seed, epoch)`. When the regular stride misses the end of a session, an
overlapping final window is added so the tail is covered. Inner selection runs
full-session inference for every validation session after every epoch and
requires strict per-bin mAP to be evaluable; it fails closed instead of silently
selecting on a different metric. The first best epoch wins ties. Final fitting
starts from scratch with the same seed and executes exactly the runner-selected
number of epochs.

The fitting core deliberately uses one window per optimiser step and a constant
learning rate. Formal batch size and warmup/cosine scheduling are not yet
implemented, and the outer run manifest does not yet pass an exact expected
spec hash to the backend. Both are hard blockers; registration must not treat
the fixture optimiser as the frozen dissertation optimiser.

Transformer inference uses the existing exact chunk-with-left-context path;
GRU inference uses one full forward pass. Every prediction artifact carries all
session ticks, including the final tick, and separate target-availability masks.

## 4. Strict artifact schemas

The backend writes only into runner-created directories using exclusive
no-overwrite file creation. The outer orchestrator remains responsible for its
own staging/publication state machine; this module does not modify that logic.

| Artifact | v3 content and validation |
|---|---|
| inner/final checkpoint | weights-only-loadable PyTorch mapping; exact model-state keys/shapes/dtypes; spec/tick/statistic/history hashes; role-specific split identity; from-scratch flag |
| history | contiguous one-based epochs; exact numeric fields; inner full-session mAP selection or fixed-epoch final role, never a mixture |
| inner OOF / outer predictions | strict JSON; embedded complete tick authority with per-record exact replay of `session_id + tick_25hz + content hash`; present/valid masks; finite `[0,1]` probabilities; binary targets; checkpoint hash |
| calibration | exact three-inner-fold OOF bindings; thresholds remain `null` with `not_estimated_owner_approval_pending` |
| metrics | recursively validated balanced/support-aware frame and event diagnostics from the one saved outer prediction pass; threshold status explicitly remains diagnostic fixed 0.5 |

JSON readers reject duplicate keys, `NaN`/`Infinity`, booleans masquerading as
numbers, wrong shapes, extra fields and role/identity mismatches. Checkpoints use
`torch.load(..., weights_only=True)` and reject any model-state deviation.

## 5. Counted-result gate

Every v3 scaffold artifact contains the same immutable blockers:

1. `backend_not_registered`;
2. `fixture_only_tick_authority`;
3. `synthetic_generator_replay_not_implemented`;
4. `outer_run_manifest_spec_binding_not_implemented`;
5. `formal_batch_scheduler_not_implemented`;
6. `environment_capture_not_implemented`;
7. `source_state_capture_not_implemented`;
8. `atomic_run_index_not_implemented`;
9. `d3_pose_shift_reassembly_not_implemented`;
10. `calibration_threshold_policy_owner_approval_pending`.

`ProductionEvidenceGateV3` defines exact future bindings for environment,
source state, atomic run index and D3 pose-shift reassembly. In v3 every binding
must be `null` and the status must be `interface_only_not_implemented`; the
object cannot self-assert that evidence exists. Implementing and independently
validating those bindings requires a later reviewed version bump.

Registration is a separate source-controlled decision. The class does not
inherit `BackendContractSmokeBackend`; therefore the current outer runner
rejects it as absent from its registry. Adding the class to a registry without
first replacing the fixture authority and closing every evidence gate would
still be scientifically invalid.

## 6. Required work before any real/countable use

- create a provenance-replayed real `session_id/tick_25hz` authority;
- bind the frozen outer run's exact spec hash to the backend instance;
- implement and freeze the formal batch/scheduler behaviour;
- capture and hash Python/PyTorch/CPU/package environment and git source state;
- implement the controlled atomic run index under the owner-approved filesystem
  trust model;
- implement both prespecified D3 pose shifts for every head-pose condition;
- freeze the calibration/threshold policy before outer-label access;
- independently review checkpoint reload/prediction equivalence and one-pass
  outer-test access; and
- only then make a reviewed registry/version change. Registration must not
  retroactively promote any v3 artifact.

## 7. Verification

Focused verification:

```powershell
pytest -q tests/test_production_backend.py
ruff check --no-cache src/hri_turn_taking/vap/production_backend.py tests/test_production_backend.py
```

The focused suite covers exact spec hashes, fixture-only tick/content binding,
tail-window coverage, no statistic refit, one-based history, full-session OOF,
auxiliary-VAD isolation, exact calibration coverage, outer train/test
disjointness, the complete inner/calibration/final/outer scaffold lifecycle,
shifted-tick replay rejection, recursive metrics/type tamper rejection,
one-pass instance-level outer access, and absence from the code-owned registry.
