"""Render thesis artifacts from one separately audited, SHA-pinned aggregate.

This module is deliberately a renderer only, not a scientific validator.  The
accepted input bytes were audited independently; this tool only enforces their
exact SHA-256 identity and copies already-computed metrics into a figure and a
LaTeX table.  It never trains a model, runs inference, or recomputes metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGGREGATE = (
    REPOSITORY_ROOT / "outputs" / "thesis_deadline_20260816" / "aggregate_seed13_v1.json"
)
EXPECTED_AGGREGATE_SHA256 = "76373fd75f0685a324f05eb1adba2785d646d575c4d0b75dcde46ed6ef648995"
DEFAULT_PDF = REPOSITORY_ROOT / "report" / "Figures" / "real_corpus_fold_dispersion.pdf"
DEFAULT_PNG = REPOSITORY_ROOT / "report" / "Figures" / "real_corpus_fold_dispersion.png"
DEFAULT_TABLE = REPOSITORY_ROOT / "report" / "Tables" / "real_corpus_ah_results.tex"
DEFAULT_MANIFEST = REPOSITORY_ROOT / "report" / "real_corpus_results_render_manifest.json"

CONDITIONS = ("A", "H", "A+H")
FOLDS = ("outer_01", "outer_02", "outer_03", "outer_04")
GROUP_COLOURS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
GROUP_MARKERS = ("o", "s", "^", "D")
FIXED_PDF_TIME = datetime(2026, 8, 17, tzinfo=timezone.utc)


class FrozenInputHashError(ValueError):
    """Raised when the renderer is not given the separately audited bytes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pinned_aggregate(path: Path) -> dict[str, Any]:
    observed = _sha256(path)
    if observed != EXPECTED_AGGREGATE_SHA256:
        raise FrozenInputHashError(
            "Renderer input is not the independently audited aggregate: "
            f"expected {EXPECTED_AGGREGATE_SHA256}, observed {observed}."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def _event_support(event: dict[str, Any]) -> str:
    return f"{int(event['n_positive']):,} / {int(event['n_negative']):,}"


def _table_text(aggregate: dict[str, Any]) -> str:
    rows: list[tuple[str, str, tuple[Any, Any, Any]]] = []
    conditions = aggregate["conditions"]

    frame = conditions["A"]["pooled"]["frame"]
    frame_support = f"{int(frame['n_positive']):,} / {int(frame['n_negative']):,}"
    rows.extend(
        [
            (
                "Frame mAP",
                frame_support,
                tuple(conditions[c]["pooled"]["frame"]["mAP"] for c in CONDITIONS),
            ),
            (
                "Participant-macro frame mAP",
                "4 participant slots",
                tuple(
                    conditions[c]["pooled"]["frame"]["participant_macro_mAP"] for c in CONDITIONS
                ),
            ),
            (
                "Frame balanced accuracy",
                frame_support,
                tuple(conditions[c]["pooled"]["frame"]["balanced_accuracy"] for c in CONDITIONS),
            ),
            (
                "Frame micro F1",
                frame_support,
                tuple(conditions[c]["pooled"]["frame"]["f1"] for c in CONDITIONS),
            ),
        ]
    )

    event_specs = (
        ("Shift/hold AP", "shift_hold", "average_precision"),
        ("Shift-prediction AP", "shift_prediction", "average_precision"),
        ("Overlap-resolution AP", "overlap_resolution", "average_precision"),
        ("Robot-entry AP", "robot_entry", "average_precision"),
    )
    for label, event_key, metric_key in event_specs:
        first_event = conditions["A"]["pooled"]["events"][event_key]
        rows.append(
            (
                label,
                _event_support(first_event),
                tuple(conditions[c]["pooled"]["events"][event_key][metric_key] for c in CONDITIONS),
            )
        )

    next_speaker = conditions["A"]["pooled"]["events"]["next_speaker"]
    participant_support = "/".join(
        str(int(next_speaker["per_participant"][participant]["n_positive"]))
        for participant in ("P1", "P2", "P3", "P4")
    )
    rows.append(
        (
            "Next-speaker macro OvR AP",
            f"$n={int(next_speaker['n_events']):,}$ ({participant_support})",
            tuple(
                conditions[c]["pooled"]["events"]["next_speaker"]["participant_macro_ovr_ap"]
                for c in CONDITIONS
            ),
        )
    )

    body = "\n".join(
        f"{label} & {support} & " + " & ".join(_format_metric(value) for value in values) + r" \\"
        for label, support, values in rows
    )
    return (
        "% Renderer-only derivative of the independently audited aggregate.\n"
        f"% Frozen aggregate SHA-256: {EXPECTED_AGGREGATE_SHA256}\n"
        "% Values are copied from pooled fields; no metric is recomputed here.\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Pooled group-disjoint real-corpus results for audio (A), "
        "head pose (H), and their fusion (A+H). Support is positive / negative; "
        "next-speaker support gives total events followed by positive counts for "
        "P1/P2/P3/P4. AP and mAP are threshold-free. Balanced accuracy and F1 use "
        "the fixed diagnostic threshold of 0.5.}\n"
        "\\label{table:real_corpus_ah_results}\n"
        "\\begin{tabular}{llrrr}\n"
        "\\toprule\n"
        "Metric & Support & A & H & A+H \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _render_figure(aggregate: dict[str, Any], pdf_path: Path, png_path: Path) -> None:
    conditions = aggregate["conditions"]
    fig, axis = plt.subplots(figsize=(6.4, 3.9), constrained_layout=True)
    x_positions = range(len(CONDITIONS))

    for fold, colour, marker in zip(FOLDS, GROUP_COLOURS, GROUP_MARKERS, strict=True):
        group = aggregate["paired_fold_group_ids"][fold]
        values = [conditions[c]["per_fold"][fold]["frame"]["mAP"] for c in CONDITIONS]
        axis.plot(
            x_positions,
            values,
            color=colour,
            marker=marker,
            markersize=5.8,
            linewidth=1.7,
            label=group,
        )

    axis.set_xticks(tuple(x_positions), CONDITIONS)
    axis.set_xlabel("Input condition")
    axis.set_ylabel("Frame mAP")
    axis.set_ylim(0.20, 0.44)
    axis.set_yticks((0.20, 0.25, 0.30, 0.35, 0.40))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    legend = axis.legend(
        title="Held-out recording group",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.25,
    )
    legend.get_title().set_fontsize(8.5)
    axis.tick_params(labelsize=8.5)
    axis.xaxis.label.set_size(9)
    axis.yaxis.label.set_size(9)

    pdf_metadata = {
        "Title": "Real-corpus outer-fold frame mAP dispersion",
        "Author": "Predictive Turn-Taking MSc Project",
        "Subject": "Renderer-only derivative of a separately audited aggregate",
        "Keywords": f"aggregate-sha256:{EXPECTED_AGGREGATE_SHA256}",
        "Creator": "tools/render_audited_results.py",
        "CreationDate": FIXED_PDF_TIME,
        "ModDate": FIXED_PDF_TIME,
    }
    fig.savefig(pdf_path, format="pdf", metadata=pdf_metadata)
    fig.savefig(
        png_path,
        format="png",
        dpi=300,
        metadata={
            "Software": "tools/render_audited_results.py",
            "Source": EXPECTED_AGGREGATE_SHA256,
        },
    )
    plt.close(fig)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_figure(aggregate: dict[str, Any], pdf_path: Path, png_path: Path) -> tuple[Path, Path]:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".real_corpus_figure.", dir=pdf_path.parent) as raw:
        stage = Path(raw)
        staged_pdf = stage / pdf_path.name
        staged_png = stage / png_path.name
        _render_figure(aggregate, staged_pdf, staged_png)
        os.replace(staged_pdf, pdf_path)
        os.replace(staged_png, png_path)
    return pdf_path, png_path


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def render_results(
    aggregate_path: Path,
    pdf_path: Path,
    png_path: Path,
    table_path: Path,
    manifest_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render values from the exact frozen bytes without validating or recomputing them."""
    output_paths = (pdf_path, png_path, table_path, manifest_path)
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to replace existing renderer outputs: {names}")

    aggregate = _load_pinned_aggregate(aggregate_path)
    _atomic_figure(aggregate, pdf_path, png_path)
    _atomic_text(table_path, _table_text(aggregate))

    rendered = (pdf_path, png_path, table_path)
    manifest: dict[str, Any] = {
        "artifact_kind": "thesis_real_results_renderer_manifest",
        "format_version": 1,
        "renderer_role": "renderer_only_not_validator",
        "scientific_validation": "external_independent_audit_of_exact_input_bytes",
        "metric_computation": "none_values_copied_from_frozen_aggregate",
        "input": {
            "path": _display_path(aggregate_path),
            "sha256": EXPECTED_AGGREGATE_SHA256,
        },
        "renderer": {
            "path": _display_path(Path(__file__)),
            "sha256": _sha256(Path(__file__)),
        },
        "figure": {
            "metric_path": "conditions.*.per_fold.*.frame.mAP",
            "conditions": list(CONDITIONS),
            "paired_fold_ids": list(FOLDS),
            "paired_group_ids": [aggregate["paired_fold_group_ids"][fold] for fold in FOLDS],
            "png_dpi": 300,
        },
        "table": {
            "metric_source": "conditions.*.pooled.frame_and_events",
            "threshold_policy": aggregate["threshold_policy"],
        },
        "outputs": {
            _display_path(path): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in rendered
        },
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render, without validating or recomputing, the exact independently audited "
            "real-corpus aggregate."
        )
    )
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = render_results(
        args.aggregate,
        args.pdf,
        args.png,
        args.table,
        args.manifest,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
