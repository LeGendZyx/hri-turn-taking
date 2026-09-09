from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG
from hri_turn_taking.vap.outer_experiment import (
    D3_40MS_DESCRIPTIVE_SESSION_IDS,
    PAIRED_D3_80MS_SESSION_IDS,
    PRIMARY_D3_SESSION_IDS,
    PRIMARY_SESSION_TO_GROUP,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_figure_generator() -> ModuleType:
    """Load the repository CLI independently of cwd and pytest entry point."""

    path = REPO_ROOT / "tools" / "generate_thesis_figures.py"
    spec = importlib.util.spec_from_file_location("thesis_figure_generator_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load thesis figure generator from {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclass annotation resolution needs the executing module registered.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


figures = _load_figure_generator()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_d3_figure_records_match_formal_inventory_and_protocol() -> None:
    records = figures.load_d3_records(REPO_ROOT)
    inventory = json.loads(
        (REPO_ROOT / "configs" / "hpe_delivery_inventory.json").read_text(encoding="utf-8")
    )
    by_scene = {entry["session_id"]: entry for entry in inventory["deliveries"]}

    assert tuple(record.session_id for record in records) == PRIMARY_D3_SESSION_IDS
    for record in records:
        source = by_scene[record.session_id]
        assert record.group_id == PRIMARY_SESSION_TO_GROUP[record.session_id]
        assert record.offset_s == source["d3_6cam"]["offset_s"]
        assert record.quality_status == source["d3_6cam"]["quality_status"]
        assert record.error_bound_ms == source["d3_6cam"]["error_bound_ms"]
    assert tuple(record.session_id for record in records if record.error_bound_ms <= 80) == (
        PAIRED_D3_80MS_SESSION_IDS
    )
    assert tuple(record.session_id for record in records if record.error_bound_ms <= 40) == (
        D3_40MS_DESCRIPTIVE_SESSION_IDS
    )


def test_figure_bundle_is_complete_high_resolution_and_reproducible(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = figures.generate_all_figures(first_dir, repo_root=REPO_ROOT)
    second = figures.generate_all_figures(second_dir, repo_root=REPO_ROOT)

    assert tuple(first["outputs"]) == figures.FIGURE_STEMS
    assert first == second
    assert first["label_schematic"]["bin_edges_s"] == list(
        DEFAULT_LABEL_CONFIG.bin_edges_s
    )
    assert first["d3"]["paired_le80_session_ids"] == list(PAIRED_D3_80MS_SESSION_IDS)
    assert first["d3"]["descriptive_le40_session_ids"] == list(
        D3_40MS_DESCRIPTIVE_SESSION_IDS
    )
    assert len(first["nested_logo"]["outer_folds"]) == 4
    assert {fold["test_group"] for fold in first["nested_logo"]["outer_folds"]} == set(
        PRIMARY_SESSION_TO_GROUP.values()
    )

    for stem in figures.FIGURE_STEMS:
        for suffix in ("pdf", "png"):
            first_path = first_dir / f"{stem}.{suffix}"
            second_path = second_dir / f"{stem}.{suffix}"
            assert first_path.is_file()
            assert first_path.stat().st_size > 5_000
            assert _sha256(first_path) == _sha256(second_path)
            assert first["outputs"][stem][suffix]["sha256"] == _sha256(first_path)
        raster = mpimg.imread(first_dir / f"{stem}.png")
        assert raster.shape[0] >= 1_200
        assert raster.shape[1] >= 2_100
        assert raster.shape[2] in {3, 4}

    manifest_path = first_dir / "thesis_figures.manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == first
    assert first["source_bindings"]["configs/hpe_delivery_inventory.json"]["sha256"] == (
        _sha256(REPO_ROOT / "configs" / "hpe_delivery_inventory.json")
    )
    assert first["source_bindings"]["tools/generate_thesis_figures.py"]["sha256"] == (
        _sha256(REPO_ROOT / "tools" / "generate_thesis_figures.py")
    )


def test_timeline_raster_uses_participants_as_rows_and_time_as_columns() -> None:
    schematic = figures.label_schematic_data()
    figure = figures.make_label_timeline_figure()
    try:
        rendered_raster = np.asarray(figure.axes[1].images[0].get_array())
        np.testing.assert_array_equal(rendered_raster, schematic["vad"].T)
        assert rendered_raster.shape == (3, DEFAULT_LABEL_CONFIG.horizon_frames)
    finally:
        plt.close(figure)
