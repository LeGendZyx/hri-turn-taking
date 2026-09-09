# Exact-eight voiced-only reactive diagnostic

## Status and boundary

This is a post-hoc exploratory diagnostic over the exact eight-scene,
four-group deadline corpus. It is not a trained model, a complete speech-VAD
baseline, a production result, or a repository-counted experiment. It cannot
by itself establish H2 or a real latency advantage.

The trigger phase reads only the reviewed causal acoustic `voiced` stream. It
seals all 64,299 frame-level trigger rows before evaluation-only gold VAD is
opened. A trigger is emitted once, on the 38th consecutive unvoiced 25 Hz tick
(the rounded 1.5 s threshold). The evaluation then applies the same robot-entry
opportunity contract as the VAP event scorer.

## Immutable artifact

The complete create-once artifact is stored at:

`tmp/reactive_diag_preflight_r1/`

Despite the historical `preflight` directory name, this directory contains the
complete result. Its manifests bind absolute paths, so it must not be renamed,
moved, or treated as disposable workspace cache.

| File | SHA-256 |
|---|---|
| `trigger_frames.csv` | `c7c71cf3fb0954499303406db14766522c593111b2af23259cddb80258953354` |
| `trigger_manifest.json` | `74bcf97d4ee505801c59ed062c115dcb5456762e53e699d4eb2ed093572d9d9e` |
| `robot_entry_opportunities.csv` | `92de19251a815553b6bac5ebd890b3089a3ab5f8f8882d0e6faa05a7dc717285` |
| `metrics.json` | `46d8f669ea8bd5e7008f2a18d80e1ce9524bab753bf00d55edd8a8fcd3c823af` |
| `evaluation_manifest.json` | `f55e315d4b7257fac63cb07e6a397c9d28b9df4aa211e36518b61da1f8bc126b` |

## Result

The frozen event contract contains 303 opportunities: 111 positive
demonstrator entries and 192 negative skipped gaps. The fixed rule produced
TP/FP/FN/TN = 5/6/106/186, positive recall 0.0450, positive miss rate 0.9550,
negative false-entry rate 0.03125, F1 0.0820, and balanced accuracy 0.5069.
Five positive hits had mean/median gap latency 1.232/1.16 s.

There were 40 rising triggers globally: 4 occurred during gold speech, 11 in
eligible gaps, and 25 in other or non-eligible silence. The appropriate
interpretation is that the voiced-only 1.5 s rule was conservative and missed
most demonstrator entries. No calibrated predictive policy or causal
reactive-silence comparison was run, so H2 remains unestablished.

## Verification

- robot-entry support and the existing scorer cross-check: 303 / 111 / 192,
  passed;
- independent artifact and semantic review: accepted;
- focused tests: 11 passed;
- repository suite after integration: 914 passed, 2 non-failing pandas
  performance warnings;
- repository Ruff check: passed.
