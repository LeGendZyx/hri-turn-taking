from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from types import ModuleType

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest

from hri_turn_taking.vap.outer_experiment import (
    DEADLINE_CORPUS_EXCLUSIONS,
    DEADLINE_SESSION_TO_GROUP,
)
from hri_turn_taking.vap.thesis_backend import (
    BACKEND_ID,
    CONDITION_FEATURE_SETS,
    COUNTED_BLOCKERS,
    RESULT_ROLE,
    THRESHOLD_POLICY,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("A", "H", "A+H")
FOLDS = ("outer_01", "outer_02", "outer_03", "outer_04")


def _load_exporter() -> ModuleType:
    path = REPO_ROOT / "tools" / "export_thesis_results.py"
    spec = importlib.util.spec_from_file_location("thesis_results_exporter_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load thesis results exporter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exporter = _load_exporter()


def _ap_summary(score: float, *, n_positive: int, n_negative: int, f1: bool) -> dict:
    payload = {
        "n_positive": n_positive,
        "n_negative": n_negative,
        "average_precision": score,
        "ap_na_reason": None,
    }
    if f1:
        payload["f1"] = 0.6
    return payload


def _frame(score: float) -> dict:
    bins = {
        f"bin{index}": _ap_summary(score, n_positive=4, n_negative=6, f1=True)
        for index in range(4)
    }
    sample_counts = {"P1": 3, "P2": 3, "P3": 2, "P4": 2}
    participants = {
        participant: {
            "in_scope": True,
            "n_samples": count,
            "mAP": score,
            "mAP_na_reason": None,
            "bins": {
                f"bin{index}": _ap_summary(
                    score,
                    n_positive=1,
                    n_negative=count - 1,
                    f1=False,
                )
                for index in range(4)
            },
        }
        for participant, count in sample_counts.items()
    }
    return {
        "mAP": score,
        "mAP_na_reason": None,
        "mAP_evaluable_bins": score,
        "n_bins": 4,
        "n_bins_evaluable_ap": 4,
        "micro_AP": score,
        "micro_AP_na_reason": None,
        "macro_f1_bins": 0.6,
        "balanced_accuracy": 0.62,
        "balanced_accuracy_na_reason": None,
        "f1": 0.6,
        "micro_f1": 0.6,
        "positive_rate": 0.4,
        "n_positive": 16,
        "n_negative": 24,
        "bins": bins,
        "participants": participants,
        "participant_support_available": True,
        "participant_macro_mAP": score,
        "participant_macro_mAP_na_reason": None,
        **{f"AP_bin{index}": score for index in range(4)},
        **{f"F1_bin{index}": 0.6 for index in range(4)},
    }


def _single_class_frame() -> dict:
    frame = _frame(0.6)
    frame.update(
        mAP=None,
        mAP_na_reason="one_or_more_bins_single_class",
        mAP_evaluable_bins=None,
        n_bins_evaluable_ap=0,
        micro_AP=None,
        micro_AP_na_reason="no_positive",
        macro_f1_bins=0.0,
        balanced_accuracy=None,
        balanced_accuracy_na_reason="no_positive",
        f1=0.0,
        micro_f1=0.0,
        positive_rate=0.0,
        n_positive=0,
        n_negative=40,
        participant_macro_mAP=None,
        participant_macro_mAP_na_reason=(
            "one_or_more_in_scope_participants_or_bins_not_evaluable"
        ),
    )
    for index in range(4):
        frame["bins"][f"bin{index}"].update(
            n_positive=0,
            n_negative=10,
            average_precision=None,
            ap_na_reason="no_positive",
            f1=0.0,
        )
        frame[f"AP_bin{index}"] = None
        frame[f"F1_bin{index}"] = 0.0
    for participant in frame["participants"].values():
        participant["mAP"] = None
        participant["mAP_na_reason"] = "one_or_more_bins_single_class"
        for item in participant["bins"].values():
            item.update(
                n_positive=0,
                n_negative=participant["n_samples"],
                average_precision=None,
                ap_na_reason="no_positive",
            )
    return frame


def _event(score: float) -> dict:
    return {
        "n_events": 5,
        "n_positive": 2,
        "n_negative": 3,
        "balanced_accuracy": 0.6,
        "balanced_accuracy_na_reason": None,
        "f1": 0.6,
        "f1_na_reason": None,
        "average_precision": score,
        "average_precision_na_reason": None,
    }


def _next_speaker(score: float) -> dict:
    positives = {"P1": 2, "P2": 1, "P3": 1, "P4": 1}
    return {
        "n_events": 5,
        "n_participants_in_scope": 4,
        "n_participants_evaluable_ap": 4,
        "participant_macro_ovr_ap": score,
        "participant_macro_ovr_ap_na_reason": None,
        "balanced_recall": 0.5,
        "mrr": 0.7,
        "top1_accuracy": 0.6,
        "tie_policy": "uniform_credit_among_max_scores",
        "tie_rate": 0.0,
        "per_participant": {
            participant: {
                "in_scope": True,
                "n_positive": n_positive,
                "n_negative": 5 - n_positive,
                "ovr_average_precision": score,
                "tie_aware_recall": 0.5,
                "ap_na_reason": None,
                "recall_na_reason": None,
            }
            for participant, n_positive in positives.items()
        },
        "na_reason": None,
    }


def _metrics(score: float) -> dict:
    return {
        "threshold_policy": THRESHOLD_POLICY,
        "frame": _frame(score),
        "events": {
            name: _event(score)
            for name in ("shift_hold", "shift_prediction", "overlap_resolution", "robot_entry")
        }
        | {"next_speaker": _next_speaker(score)},
    }


def _folds() -> tuple[dict[str, list[str]], dict[str, str]]:
    sessions = {
        "outer_01": ["03_20__4"],
        "outer_02": ["03_26__3"],
        "outer_03": ["05_14__2", "05_14__3", "05_14__4"],
        "outer_04": ["05_15__2", "05_15__3", "05_15__6"],
    }
    groups = {fold: DEADLINE_SESSION_TO_GROUP[values[0]] for fold, values in sessions.items()}
    return sessions, groups


def _condition(condition_id: str, score: float) -> dict:
    sessions, groups = _folds()
    digest = {"A": "a", "H": "b", "A+H": "c"}[condition_id] * 64
    return {
        "artifact_kind": "vap_thesis_outer_aggregate",
        "format_version": 1,
        "backend_id": BACKEND_ID,
        "result_role": RESULT_ROLE,
        "data_kind": "real",
        "production_eligible": False,
        "counted_eligible": False,
        "counted_blockers": list(COUNTED_BLOCKERS),
        "condition_id": condition_id,
        "feature_set": CONDITION_FEATURE_SETS[condition_id],
        "seed": 13,
        "real_session_authority_sha256": "d" * 64,
        "source_manifest_sha256": "e" * 64,
        "run_manifest_sha256": "f" * 64,
        "outer_fold_ids": list(FOLDS),
        "session_ids": sorted(DEADLINE_SESSION_TO_GROUP),
        "fold_session_ids": sessions,
        "fold_group_ids": groups,
        "prediction_artifact_sha256": [digest] * 4,
        "aggregation_source": "saved_outer_probabilities_only_no_model_inference",
        "threshold_policy": THRESHOLD_POLICY,
        "pooled": _metrics(score),
        "per_fold": {
            fold: _metrics(score - 0.03 + index * 0.02)
            for index, fold in enumerate(FOLDS)
        },
    }


def _aggregate() -> dict:
    sessions, groups = _folds()
    conditions = {
        "A": _condition("A", 0.6),
        "H": _condition("H", 0.5),
        "A+H": _condition("A+H", 0.7),
    }
    return {
        "artifact_kind": "vap_thesis_primary_matrix_aggregate",
        "format_version": 1,
        "result_role": RESULT_ROLE,
        "data_kind": "real",
        "production_eligible": False,
        "counted_eligible": False,
        "analysis_variant": "primary",
        "included_session_ids": list(DEADLINE_SESSION_TO_GROUP),
        "group_ids": sorted(set(DEADLINE_SESSION_TO_GROUP.values())),
        "corpus_exclusions": dict(DEADLINE_CORPUS_EXCLUSIONS),
        "seed": 13,
        "aggregation_source": "saved_outer_probabilities_only_no_model_inference",
        "threshold_policy": THRESHOLD_POLICY,
        "run_manifest_sha256": "f" * 64,
        "paired_fold_session_ids": sessions,
        "paired_fold_group_ids": groups,
        "conditions": conditions,
        "paired_comparisons": {
            "A+H_minus_A": {
                "pooled_frame_mAP_difference": 0.1,
                "interpretation": "descriptive_paired_difference_not_a_significance_test",
            },
            "A+H_minus_H": {
                "pooled_frame_mAP_difference": 0.2,
                "interpretation": "descriptive_paired_difference_not_a_significance_test",
            },
        },
    }


def _write_aggregate(path: Path, payload: dict) -> bytes:
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return raw


def test_export_is_read_only_complete_and_paper_ready(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.json"
    original = _write_aggregate(aggregate, _aggregate())
    output = tmp_path / "paper_results"

    manifest_path = exporter.export_results(aggregate, output)

    assert aggregate.read_bytes() == original
    assert manifest_path == output / "export_manifest.json"
    assert {path.name for path in output.iterdir()} == {
        "export_manifest.json",
        "real_corpus_results_summary.json",
        "real_corpus_results_table.tex",
        "real_corpus_fold_metrics.csv",
        "real_corpus_results.pdf",
        "real_corpus_results.png",
    }
    summary = json.loads((output / "real_corpus_results_summary.json").read_text(encoding="utf-8"))
    assert summary["source_aggregate"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert summary["conditions"]["A+H"]["frame"]["mAP"] == 0.7
    assert summary["claim_boundary"]["one_seed_only"] is True
    assert "_validated_aggregate" not in summary
    table = (output / "real_corpus_results_table.tex").read_text(encoding="utf-8")
    assert "Pooled out-of-fold performance" in table
    assert "0.700" in table
    assert "descriptive" in table
    rows = (output / "real_corpus_fold_metrics.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 13
    raster = mpimg.imread(output / "real_corpus_results.png")
    assert raster.shape[0] >= 1_000
    assert raster.shape[1] >= 2_000
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_mutation"] is False
    assert manifest["checkpoint_or_model_loaded"] is False
    assert manifest["report_main_tex_modified"] is False
    assert manifest["outputs"]["real_corpus_results.pdf"]["size_bytes"] > 5_000

    with pytest.raises(FileExistsError, match="overwrite"):
        exporter.export_results(aggregate, output)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["conditions"]["H"]["fold_group_ids"].__setitem__(
                "outer_01", "05_15"
            ),
            "session/group mapping differs",
        ),
        (
            lambda payload: payload["paired_comparisons"]["A+H_minus_A"].__setitem__(
                "pooled_frame_mAP_difference", 0.3
            ),
            "value is inconsistent",
        ),
        (
            lambda payload: payload["conditions"]["A"]["pooled"]["frame"].__setitem__(
                "mAP", 0.99
            ),
            "strict four-bin mean",
        ),
    ],
)
def test_export_fails_closed_on_semantic_tampering(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = copy.deepcopy(_aggregate())
    mutation(payload)
    aggregate = tmp_path / "aggregate.json"
    _write_aggregate(aggregate, payload)
    output = tmp_path / "must_not_exist"

    with pytest.raises(exporter.ThesisResultsExportError, match=message):
        exporter.export_results(aggregate, output)

    assert not output.exists()


def test_duplicate_json_key_is_rejected_before_output(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.json"
    raw = json.dumps(_aggregate(), allow_nan=False)
    aggregate.write_text(raw.replace("{", '{"artifact_kind":"forged",', 1), encoding="utf-8")
    output = tmp_path / "must_not_exist"

    with pytest.raises(exporter.ThesisResultsExportError, match="duplicate JSON key"):
        exporter.export_results(aggregate, output)

    assert not output.exists()


def test_frame_validator_accepts_a_schema_defined_absent_participant() -> None:
    frame = _frame(0.6)
    frame["participants"]["P3"]["n_samples"] = 4
    for item in frame["participants"]["P3"]["bins"].values():
        item["n_positive"] = 2
        item["n_negative"] = 2
    frame["participants"]["P4"] = {
        "in_scope": False,
        "n_samples": 0,
        "mAP": None,
        "mAP_na_reason": "not_in_scope",
        "bins": {
            f"bin{index}": {
                "n_positive": 0,
                "n_negative": 0,
                "average_precision": None,
                "ap_na_reason": "not_in_scope",
            }
            for index in range(4)
        },
    }

    assert exporter._validate_frame(frame, "fixture") is frame


@pytest.mark.parametrize(
    "payload",
    [
        {
            "n_positive": 0,
            "n_negative": 5,
            "average_precision": 0.4,
            "ap_na_reason": None,
        },
        {
            "n_positive": 2,
            "n_negative": 3,
            "average_precision": None,
            "ap_na_reason": "forged_reason",
        },
    ],
)
def test_ap_summary_support_fixes_value_and_na_reason(payload: dict) -> None:
    with pytest.raises(exporter.ThesisResultsExportError, match="AP/NA semantics"):
        exporter._validate_ap_summary(payload, "fixture", with_f1=False)


@pytest.mark.parametrize("case", ["empty", "single_class"])
def test_binary_event_support_fixes_metric_na_semantics(case: str) -> None:
    event = _event(0.6)
    if case == "empty":
        event.update(
            n_events=0,
            n_positive=0,
            n_negative=0,
            balanced_accuracy=None,
            balanced_accuracy_na_reason="no_positive_events",
            f1=None,
            f1_na_reason="no_positive_events",
            average_precision=None,
            average_precision_na_reason="no_positive_events",
        )
    else:
        event.update(
            n_positive=0,
            n_negative=5,
            balanced_accuracy=None,
            balanced_accuracy_na_reason="no_positive_events",
            f1=None,
            f1_na_reason="no_positive_events",
            average_precision=None,
            average_precision_na_reason="no_positive_events",
        )
    with pytest.raises(exporter.ThesisResultsExportError, match="metric semantics"):
        exporter._validate_binary_event(event, "fixture")


@pytest.mark.parametrize("case", ["positive_rate", "participant_bin"])
def test_frame_support_summaries_fail_closed(case: str) -> None:
    frame = _frame(0.6)
    if case == "positive_rate":
        frame["positive_rate"] = 0.5
        message = "positive_rate differs"
    else:
        frame["participants"]["P1"]["bins"]["bin0"].update(
            n_positive=2,
            n_negative=1,
        )
        message = "participant-bin class supports differ"
    with pytest.raises(exporter.ThesisResultsExportError, match=message):
        exporter._validate_frame(frame, "fixture")


def test_next_speaker_positive_supports_must_sum_to_event_count() -> None:
    event = _next_speaker(0.6)
    event["per_participant"]["P1"].update(n_positive=1, n_negative=4)
    with pytest.raises(exporter.ThesisResultsExportError, match="positive supports"):
        exporter._validate_next_speaker(event, "fixture")


def test_na_frame_map_still_has_complete_legend_without_warning() -> None:
    aggregate = _aggregate()
    for condition in aggregate["conditions"].values():
        condition["pooled"]["frame"]["mAP"] = None
    summary = {
        "conditions": {
            condition_id: exporter._paper_condition_summary(
                aggregate["conditions"][condition_id]
            )
            for condition_id in CONDITIONS
        },
        "_validated_aggregate": aggregate,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure = exporter.make_results_figure(summary)
    try:
        legend = figure.axes[0].get_legend()
        assert [text.get_text() for text in legend.get_texts()] == list(CONDITIONS)
        assert caught == []
    finally:
        plt.close(figure)


@pytest.mark.parametrize("case", ["mixed_micro_ap", "single_balanced_accuracy"])
def test_frame_class_support_fixes_micro_ap_and_balanced_accuracy(case: str) -> None:
    if case == "mixed_micro_ap":
        frame = _frame(0.6)
        frame["micro_AP"] = None
        frame["micro_AP_na_reason"] = "forged_reason"
        message = "mixed-class micro AP/BA semantics"
    else:
        frame = _single_class_frame()
        assert exporter._validate_frame(frame, "valid_single_class") is frame
        frame["balanced_accuracy"] = 0.5
        frame["balanced_accuracy_na_reason"] = None
        message = "single-class micro AP/BA semantics"
    with pytest.raises(exporter.ThesisResultsExportError, match=message):
        exporter._validate_frame(frame, "fixture")


def test_frame_macro_f1_must_equal_bin_mean() -> None:
    frame = _frame(0.6)
    frame["macro_f1_bins"] = 0.7
    with pytest.raises(exporter.ThesisResultsExportError, match="differs from bin F1 mean"):
        exporter._validate_frame(frame, "fixture")


def test_frame_f1_aliases_must_be_finite() -> None:
    frame = _frame(0.6)
    frame["f1"] = None
    frame["micro_f1"] = None
    with pytest.raises(exporter.ThesisResultsExportError, match="finite score"):
        exporter._validate_frame(frame, "fixture")


def test_all_na_fold_keeps_explicit_group_legend_without_warning() -> None:
    aggregate = _aggregate()
    for condition in aggregate["conditions"].values():
        condition["per_fold"]["outer_01"]["frame"]["mAP"] = None
    summary = {
        "conditions": {
            condition_id: exporter._paper_condition_summary(
                aggregate["conditions"][condition_id]
            )
            for condition_id in CONDITIONS
        },
        "_validated_aggregate": aggregate,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure = exporter.make_results_figure(summary)
    try:
        legend = figure.axes[1].get_legend()
        expected_groups = [
            aggregate["paired_fold_group_ids"][fold_id] for fold_id in FOLDS
        ]
        assert [text.get_text() for text in legend.get_texts()] == expected_groups
        assert caught == []
    finally:
        plt.close(figure)
