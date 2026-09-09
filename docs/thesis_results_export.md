# Thesis result export

## Purpose

`tools/export_thesis_results.py` converts one completed deadline-8 primary
aggregate into paper-ready derivatives. It is deliberately downstream of
`tools/run_thesis_outer.py aggregate`: it does not load a model, run inference,
recompute training, edit the aggregate, or modify `report/main.tex`.

The exporter accepts only the exact current A / H / A+H, seed-13, four-fold,
eight-scene aggregate schema. It rejects duplicate JSON keys, NaN/Infinity,
unexpected fields, changed corpus/fold identities, incorrect source hashes,
inconsistent supports, invalid NA semantics, and paired differences that do not
match pooled frame mAP. The input bytes are hash-bound and checked again before
the output bundle is published.

## Command

Run this only after the twelve-unit matrix has been successfully aggregated:

```powershell
.\.venv\Scripts\python.exe tools\export_thesis_results.py `
  --aggregate <primary-aggregate.json> `
  --output <new-paper-results-directory>
```

The output path is create-once. A complete bundle is first rendered in a
sibling staging directory and then renamed into place; an existing destination
is never overwritten.

## Bundle contents

- `real_corpus_results_summary.json`: compact, support-aware values and claim
  boundaries for later prose generation.
- `real_corpus_results_table.tex`: standalone LaTeX table snippet. Event AP
  cells include positive/negative support; non-evaluable values render as a
  dash and retain their machine-readable reason in JSON.
- `real_corpus_fold_metrics.csv`: one row for each of 12 condition/fold units,
  preserving held-out-group dispersion rather than substituting a fold mean
  for pooled OOF metrics.
- `real_corpus_results.pdf` and `.png`: a pooled metric overview plus paired
  per-held-out-group frame-mAP trajectories. Lines are descriptive pairings,
  not confidence intervals.
- `export_manifest.json`: SHA-256 and size bindings for the input, generator,
  and every derivative.

The generated table and figure remain separate from the dissertation source.
After reviewing them, include them explicitly from `report/main.tex`; this tool
never makes that scientific/editorial decision automatically.

## Interpretation boundary

Use the wording **deadline non-production nested-LOGO evaluation**. The primary
scores pool all saved out-of-fold examples; they are not averages of four fold
scores. A+H-minus-A and A+H-minus-H are descriptive paired differences, not
significance tests. F1 and balanced accuracy use the fixed diagnostic threshold
0.5 and are not calibrated robot-policy claims.

This first bundle covers one seed and the central D3 primary analysis only. It
does not claim production or repository-counted status, D3 sensitivity,
multi-seed replication, ASD/Gaze/Facial results, or a physical Furhat result.
Those absences must remain explicit when the table is discussed in the thesis.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_export_thesis_results.py -q `
  -p no:cacheprovider
.\.venv\Scripts\ruff.exe check tools\export_thesis_results.py `
  tests\test_export_thesis_results.py
```

The fixture tests verify input immutability, complete output inventory, source
hash binding, no-clobber behavior, high-resolution figure generation,
duplicate-key rejection, semantic tamper rejection, and an absent participant
slot with explicit zero support.
