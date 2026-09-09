# H2 matched replay: isolated model-seed sensitivity contract

Status: implementation and tests only until an independent cold review accepts
the version-2 runner. No seed-42/73 replay is authorised by this document.

## Why this is a separate implementation

The completed seed-13 freeze, trace seal and evaluation bind
`src/hri_turn_taking/vap/h2_matched_replay.py` at SHA-256
`30511969e2a5e3ca8fe3a138fef12494da3d0700ee5364a26538aba49248fd3c`.
Changing that file would make the existing seed-13 cold audit fail even if its
outputs were not overwritten. Version 2 therefore leaves that source and
`outputs/h2_matched_replay_posthoc_v1/` byte-for-byte unchanged.

The extension imports the version-1 causal policy, event, metric, and E-only
acoustic primitives. Every version-2 artifact binds both the new implementation
and the exact version-1 dependency hash.

## Fixed scientific boundary

- Model seeds are exactly 42 and 73. Seed 13 remains the immutable primary
  five-epoch result and the earlier 15-epoch sensitivity result.
- The calibration algorithm seed remains exactly 13. It is the fixed
  `LogisticRegression` setting and is not a VAP model initialisation.
- Five and 15 epochs use the already completed A+H nested-LOGO saved
  probabilities. No model is trained or loaded for inference.
- Each outer fold is calibrated only from its three group-OOF inner partitions.
  Outer labels and event outcomes cannot enter detector, threshold, silence-run,
  or predictive-policy selection.
- Trace generation reads only the freeze-owned gold-free projection
  (`session_id`, `group_id`, 25 Hz ticks, probabilities, and static present
  mask) plus sealed local E-only `array_energy/voiced/f0_proxy` descendants.
- Raw WAV replay, F-drive dereference, floor-holder input, physical robot/TTS,
  action feedback, far-field robot-ear audio, and live latency are absent.
- All outputs remain post-outcome, exploratory, non-production and non-counted.
  H2 is never marked established by the runner.

## Frozen sources

The CLI does not accept matrix or aggregate overrides. For each seed/budget the
module freezes the known matrix-index and independently audited aggregate
hashes, then revalidates the canonical run/split, exact-eight corpus, every A+H
unit/result, 12 inner OOF predictions, 12 paired inner checkpoints, four final
checkpoints and four outer saved predictions.

| Budget | Seed | Matrix-index SHA-256 | Aggregate SHA-256 |
|---|---:|---|---|
| 5 epochs | 42 | `c0f4467c655d94a4a61528c1815c48e4a53da79fa2a8ac757af0dd5e5ddfcff7` | `7cdd0d126569c073773066419f8fe9e5d71af74480d480d7d8683ff7f973609f` |
| 5 epochs | 73 | `f6b8811a5428d80073d89ece6003cbaadc5d3253fbf43d81ece661b43b544a3a` | `527e28f6f300415c2c5c7e6d8f20739433d077106de252ae796bd0b231fa168f` |
| 15 epochs | 42 | `187b98b965b35b8290649da66c4ba34c59c9f957d07af2af525da694740b233a` | `4acaaca61a3b080271014fef8b3e1381478351174ab4800e6c2f270f3333dc50` |
| 15 epochs | 73 | `1f2a7691351314a796e1502e4671b7ccb445cf7c90a00c14bca0ff48cd05167b` | `dba7f5f0ed256acd5d0195a444709e4fe1efb3eced9df0cd809ac4d32b187514` |

The aggregate's four A+H outer-prediction hashes must exactly equal the matrix
artifacts in outer-fold order. Mixing seed 42 with 73 or five with 15 epochs
fails closed.

## Per-seed create-once phases

For each extension seed:

1. `freeze` validates all frozen sources and writes four gold-free outer views
   for each budget. Its immediate cold audit independently reopens the
   canonical matrix/aggregate sources, reconstructs the complete expected
   freeze tree, and requires an exact file-inventory and byte-for-byte match;
   merely rewriting a view and updating its local seals cannot pass.
2. `calibrate-and-seal-traces` fits inner-only detector and policy calibration
   and writes outer traces without opening outer gold. Its audit reopens the
   bound inner OOF artifacts and sealed E-only acoustic descendants, reruns the
   hash-bound version-1 calibration, and requires the saved calibration to have
   the exact canonical schema and JSON bytes before replaying each trace. A
   modified calibration remains invalid even if its trace-seal binding is
   recomputed.
3. `evaluate` opens the bound outer artifacts only after the trace seal exists.
4. `audit-seed` replays the completed prefix from canonical bytes.

Every budget/policy/window must expose the identical 303 unique robot-entry
opportunities (111 positive, 192 negative), including matching session, group,
fold, gap ticks, previous/next slots and label. The reactive calibration and
frame trace must be exact across five/15-epoch budgets.

Publication is atomic/create-once. The trace publication audit deliberately
uses the already cold-audited, freeze-owned projection so that it does not
reopen legacy outer predictions during trace generation; a later explicit
`audit-seed` performs the full canonical-source freeze reconstruction. If an
immediate audit fails, cleanup is allowed only when the final directory is
still the same just-created directory identity; a concurrently replaced owner
is never deleted.

## Three-seed aggregate

The final aggregate binds:

- immutable seed-13 evaluation manifest
  `f0bf3c63d7db9f1d70627dab9455d0b51bcc6d7aae11d86813192cecfb506f1f`;
- immutable seed-13 event CSV
  `535895be9277bc3f1dec49676b629e80db83c1605a60510fa1fa33c780637518`;
- immutable seed-13 metrics
  `58302f9e74868fb1ad02bffbe9c20a6f38b85c35d9babe2b62fa4d8b612cb853`;
- independently audited seed-42 and seed-73 evaluations.

It requires full opportunity metadata and the reactive calibration, frame
traces, and event decisions to be exact across all three model seeds. It reports
each predictive result plus arithmetic mean and sample standard deviation
(`ddof=1`). Both-hit predictive lead remains descriptive. No significance,
production, deployment or live-latency claim is generated.

## Commands after independent code acceptance

Use new paths; never point at the seed-13 output tree:

```powershell
python tools/run_h2_multiseed_replay.py freeze --seed 42 `
  --output outputs/h2_matched_replay_multiseed_20260819/seed42/freeze
python tools/run_h2_multiseed_replay.py calibrate-and-seal-traces `
  --freeze outputs/h2_matched_replay_multiseed_20260819/seed42/freeze `
  --output outputs/h2_matched_replay_multiseed_20260819/seed42/traces
python tools/run_h2_multiseed_replay.py evaluate `
  --freeze outputs/h2_matched_replay_multiseed_20260819/seed42/freeze `
  --traces outputs/h2_matched_replay_multiseed_20260819/seed42/traces `
  --output outputs/h2_matched_replay_multiseed_20260819/seed42/evaluation
```

Repeat for seed 73, run `audit-seed` for both, then create and cold-audit one
new aggregate directory with `aggregate` and `audit-aggregate`.
