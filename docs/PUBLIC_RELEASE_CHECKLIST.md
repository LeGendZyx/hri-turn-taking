# Public release checklist

The repository may be pushed to its private remote while these items remain
open. Do not change the GitHub visibility to public until every item below has
been reviewed.

- Confirm that the ethics and data-sharing approval permits publication of the
  de-identified Transcript repair constants and exact scene/time metadata used
  by the audit code. If it does not, move those constants to ignored private
  manifests and keep only synthetic fixtures in the public test suite.
- Review the HPE inventories and operational status ledgers for metadata that
  should remain private. They contain no media, but they bind exact corpus
  scenes, timing quality, and local annotation artifacts.
- Replace or remove any remaining workstation-specific source defaults. Public
  code should use `HRI_CORPUS_ROOT` and `HRI_HPE_DELIVERY_ROOT`.
- Choose and add a software licence. A public GitHub repository without a
  licence does not grant general permission to reuse the code.
- Confirm that the author name in `pyproject.toml` is intended to be public.
- Decide whether the personal email recorded in earlier Git history is
  acceptable. New commits use a GitHub noreply address; changing old commits
  would require a separately reviewed history rewrite and force push.
- Test installation, CLI help, synthetic generation, `pytest`, and Ruff from a
  fresh clone that contains none of the ignored local data or result files.
- Re-run a secret scan over the complete Git history immediately before the
  visibility change.

The dissertation sources, presentation exports, recordings, annotations,
checkpoints, generated results, and recovery snapshots must remain excluded.
