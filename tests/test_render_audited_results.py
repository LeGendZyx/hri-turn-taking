from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from tools.render_audited_results import (
    EXPECTED_AGGREGATE_SHA256,
    FrozenInputHashError,
    render_results,
)

ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "outputs" / "thesis_deadline_20260816" / "aggregate_seed13_v1.json"
pytestmark = pytest.mark.skipif(
    not AGGREGATE.is_file(),
    reason="requires the private, independently audited result aggregate",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / "Figures" / "real_corpus_fold_dispersion.pdf",
        root / "Figures" / "real_corpus_fold_dispersion.png",
        root / "Tables" / "real_corpus_ah_results.tex",
        root / "real_corpus_results_render_manifest.json",
    )


def test_repository_aggregate_is_the_exact_independently_audited_input() -> None:
    assert _sha256(AGGREGATE) == EXPECTED_AGGREGATE_SHA256


def test_renderer_emits_figure_table_and_bound_manifest(tmp_path: Path) -> None:
    pdf, png, table, manifest_path = _paths(tmp_path)
    manifest = render_results(AGGREGATE, pdf, png, table, manifest_path)

    assert pdf.read_bytes().startswith(b"%PDF-")
    assert pdf.stat().st_size > 10_000
    with Image.open(png) as image:
        assert image.width >= 1_800
        assert image.height >= 1_000
        dpi = image.info["dpi"]
        assert dpi[0] == pytest.approx(300, abs=0.1)
        assert dpi[1] == pytest.approx(300, abs=0.1)

    latex = table.read_text(encoding="utf-8")
    assert "\\label{table:real_corpus_ah_results}" in latex
    assert "Frame mAP & 217,146 / 805,366 & 0.232 & 0.309 & 0.263" in latex
    assert "Participant-macro frame mAP & 4 participant slots & 0.247 & 0.239 & 0.208" in latex
    assert "Frame micro F1 & 217,146 / 805,366 & 0.331 & 0.350 & 0.359" in latex
    assert "Frame F1 &" not in latex
    assert "Shift/hold AP & 312 / 166 & 0.583 & 0.664 & 0.690" in latex
    assert "Overlap-resolution AP & 35 / 34 & 0.706 & 0.576 & 0.578" in latex
    assert "Next-speaker macro OvR AP & $n=479$ (177/101/149/52)" in latex

    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest == manifest
    assert manifest["renderer_role"] == "renderer_only_not_validator"
    assert manifest["input"]["sha256"] == EXPECTED_AGGREGATE_SHA256
    assert manifest["metric_computation"] == "none_values_copied_from_frozen_aggregate"
    assert manifest["figure"]["paired_group_ids"] == ["03_20", "03_26", "05_14", "05_15"]
    assert manifest["figure"]["png_dpi"] == 300
    for output in (pdf, png, table):
        binding = manifest["outputs"][str(output.resolve())]
        assert binding["sha256"] == _sha256(output)
        assert binding["size_bytes"] == output.stat().st_size


def test_renderer_rejects_any_nonidentical_input_before_writing(tmp_path: Path) -> None:
    altered = tmp_path / "altered.json"
    altered.write_bytes(AGGREGATE.read_bytes() + b"\n")
    outputs = _paths(tmp_path / "out")

    with pytest.raises(FrozenInputHashError, match="independently audited aggregate"):
        render_results(altered, *outputs)

    assert not any(path.exists() for path in outputs)


def test_renderer_is_no_clobber_by_default(tmp_path: Path) -> None:
    outputs = _paths(tmp_path)
    outputs[0].parent.mkdir(parents=True)
    outputs[0].write_bytes(b"owner content")

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        render_results(AGGREGATE, *outputs)

    assert outputs[0].read_bytes() == b"owner content"
