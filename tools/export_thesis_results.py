"""Export a validated thesis aggregate into paper-ready, read-only derivatives.

The input aggregate is never modified.  The destination is create-once and is
published only after the complete bundle has been rendered in a sibling staging
directory.  This utility deliberately consumes the saved-probability aggregate;
it does not load a checkpoint, run inference, or edit ``report/main.tex``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from hri_turn_taking.vap.outer_experiment import (  # noqa: E402
    DEADLINE_CORPUS_EXCLUSIONS,
    DEADLINE_SESSION_TO_GROUP,
)
from hri_turn_taking.vap.thesis_backend import (  # noqa: E402
    BACKEND_ID,
    CONDITION_FEATURE_SETS,
    COUNTED_BLOCKERS,
    RESULT_ROLE,
    THRESHOLD_POLICY,
)

EXPORT_FORMAT_VERSION = 1
SOURCE_ARTIFACT_KIND = "vap_thesis_primary_matrix_aggregate"
CONDITION_AGGREGATE_KIND = "vap_thesis_outer_aggregate"
CONDITIONS = ("A", "H", "A+H")
FOLDS = ("outer_01", "outer_02", "outer_03", "outer_04")
EXPECTED_SEED = 13
EXPECTED_SESSIONS = tuple(DEADLINE_SESSION_TO_GROUP)
EXPECTED_GROUPS = tuple(sorted(set(DEADLINE_SESSION_TO_GROUP.values())))
FIXED_ARTIFACT_TIME = datetime(2026, 8, 16, tzinfo=UTC)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
DARK = "#222222"
MID_GREY = "#777777"
LIGHT_GREY = "#E6E6E6"
WHITE = "#FFFFFF"
CONDITION_COLOURS = {"A": BLUE, "H": ORANGE, "A+H": GREEN}


class ThesisResultsExportError(ValueError):
    """The aggregate or export request violates the paper-export contract."""


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ThesisResultsExportError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_strict_json(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ThesisResultsExportError(f"cannot read aggregate {path}: {exc}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(raw) != before.st_size:
        raise ThesisResultsExportError("aggregate changed while it was read")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ThesisResultsExportError(f"non-standard JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ThesisResultsExportError(f"aggregate is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ThesisResultsExportError("aggregate JSON root must be an object")
    return payload, {
        "sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
        "identity": identity_before,
    }


def _require_exact_keys(payload: object, expected: set[str], context: str) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != expected:
        observed = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ThesisResultsExportError(
            f"{context} fields are not exact; observed={observed!r}"
        )
    return payload


def _require_int(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ThesisResultsExportError(f"{context} must be an integer >= {minimum}")
    return value


def _require_score(value: object, context: str, *, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThesisResultsExportError(f"{context} must be a finite score or null")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ThesisResultsExportError(f"{context} must be in [0,1]")
    return result


def _require_na_pair(value: float | None, reason: object, context: str) -> None:
    if value is None:
        if not isinstance(reason, str) or not reason:
            raise ThesisResultsExportError(f"{context} null value lacks an NA reason")
    elif reason is not None:
        raise ThesisResultsExportError(f"{context} finite value must have a null NA reason")


def _same_optional_number(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _validate_ap_summary(
    payload: object,
    context: str,
    *,
    with_f1: bool,
    allow_empty: bool = False,
) -> dict[str, object]:
    expected = {"n_positive", "n_negative", "average_precision", "ap_na_reason"}
    if with_f1:
        expected.add("f1")
    result = _require_exact_keys(payload, expected, context)
    n_positive = _require_int(result["n_positive"], f"{context}.n_positive")
    n_negative = _require_int(result["n_negative"], f"{context}.n_negative")
    support = n_positive + n_negative
    if support <= 0 and not allow_empty:
        raise ThesisResultsExportError(f"{context} has no support")
    ap = _require_score(result["average_precision"], f"{context}.average_precision")
    if support == 0:
        expected_reason = "not_in_scope"
    elif n_positive == 0:
        expected_reason = "no_positive"
    elif n_negative == 0:
        expected_reason = "no_negative"
    else:
        expected_reason = None
    if (expected_reason is None and (ap is None or result["ap_na_reason"] is not None)) or (
        expected_reason is not None
        and (ap is not None or result["ap_na_reason"] != expected_reason)
    ):
        raise ThesisResultsExportError(f"{context} AP/NA semantics differ from support")
    if with_f1:
        _require_score(result["f1"], f"{context}.f1", allow_none=False)
    return result


def _validate_frame(payload: object, context: str) -> dict[str, object]:
    expected = {
        "mAP",
        "mAP_na_reason",
        "mAP_evaluable_bins",
        "n_bins",
        "n_bins_evaluable_ap",
        "micro_AP",
        "micro_AP_na_reason",
        "macro_f1_bins",
        "balanced_accuracy",
        "balanced_accuracy_na_reason",
        "f1",
        "micro_f1",
        "positive_rate",
        "n_positive",
        "n_negative",
        "bins",
        "participants",
        "participant_support_available",
        "participant_macro_mAP",
        "participant_macro_mAP_na_reason",
        *(f"AP_bin{index}" for index in range(4)),
        *(f"F1_bin{index}" for index in range(4)),
    }
    result = _require_exact_keys(payload, expected, context)
    if result["participant_support_available"] is not True:
        raise ThesisResultsExportError(f"{context} lacks participant support")
    if _require_int(result["n_bins"], f"{context}.n_bins") != 4:
        raise ThesisResultsExportError(f"{context} must contain exactly four bins")
    n_evaluable = _require_int(
        result["n_bins_evaluable_ap"], f"{context}.n_bins_evaluable_ap"
    )
    if n_evaluable > 4:
        raise ThesisResultsExportError(f"{context} has too many evaluable bins")
    bins = _require_exact_keys(
        result["bins"], {f"bin{index}" for index in range(4)}, f"{context}.bins"
    )
    bin_ap: list[float] = []
    bin_f1: list[float] = []
    bin_positive_support: list[int] = []
    bin_negative_support: list[int] = []
    expected_samples: int | None = None
    for index in range(4):
        summary = _validate_ap_summary(
            bins[f"bin{index}"], f"{context}.bins.bin{index}", with_f1=True
        )
        support = int(summary["n_positive"]) + int(summary["n_negative"])
        bin_positive_support.append(int(summary["n_positive"]))
        bin_negative_support.append(int(summary["n_negative"]))
        if expected_samples is None:
            expected_samples = support
        elif support != expected_samples:
            raise ThesisResultsExportError(f"{context} bin supports differ")
        ap = summary["average_precision"]
        if ap is not None:
            bin_ap.append(float(ap))
        bin_f1.append(float(summary["f1"]))
        if not _same_optional_number(result[f"AP_bin{index}"], ap):
            raise ThesisResultsExportError(f"{context}.AP_bin{index} differs from bins")
        if not _same_optional_number(result[f"F1_bin{index}"], summary["f1"]):
            raise ThesisResultsExportError(f"{context}.F1_bin{index} differs from bins")
    if len(bin_ap) != n_evaluable:
        raise ThesisResultsExportError(f"{context} evaluable-bin count is inconsistent")
    strict_map = _require_score(result["mAP"], f"{context}.mAP")
    _require_na_pair(strict_map, result["mAP_na_reason"], f"{context}.mAP")
    expected_map = sum(bin_ap) / 4 if len(bin_ap) == 4 else None
    if not _same_optional_number(strict_map, expected_map):
        raise ThesisResultsExportError(f"{context}.mAP is not the strict four-bin mean")
    diagnostic_map = _require_score(
        result["mAP_evaluable_bins"], f"{context}.mAP_evaluable_bins"
    )
    expected_diagnostic = sum(bin_ap) / len(bin_ap) if bin_ap else None
    if not _same_optional_number(diagnostic_map, expected_diagnostic):
        raise ThesisResultsExportError(f"{context} evaluable-bin mAP is inconsistent")
    for key in (
        "micro_AP",
        "macro_f1_bins",
        "balanced_accuracy",
        "f1",
        "micro_f1",
        "positive_rate",
        "participant_macro_mAP",
    ):
        value = _require_score(result[key], f"{context}.{key}")
        reason_key = {
            "micro_AP": "micro_AP_na_reason",
            "balanced_accuracy": "balanced_accuracy_na_reason",
            "participant_macro_mAP": "participant_macro_mAP_na_reason",
        }.get(key)
        if reason_key is not None:
            _require_na_pair(value, result[reason_key], f"{context}.{key}")
    macro_f1 = _require_score(
        result["macro_f1_bins"], f"{context}.macro_f1_bins", allow_none=False
    )
    frame_f1 = _require_score(result["f1"], f"{context}.f1", allow_none=False)
    micro_f1 = _require_score(
        result["micro_f1"], f"{context}.micro_f1", allow_none=False
    )
    if not _same_optional_number(macro_f1, sum(bin_f1) / 4):
        raise ThesisResultsExportError(f"{context}.macro_f1_bins differs from bin F1 mean")
    if not _same_optional_number(frame_f1, micro_f1):
        raise ThesisResultsExportError(f"{context}.f1 and micro_f1 differ")
    n_positive = _require_int(result["n_positive"], f"{context}.n_positive")
    n_negative = _require_int(result["n_negative"], f"{context}.n_negative")
    if expected_samples is None or n_positive + n_negative != 4 * expected_samples:
        raise ThesisResultsExportError(f"{context} flattened support is inconsistent")
    if n_positive != sum(bin_positive_support) or n_negative != sum(bin_negative_support):
        raise ThesisResultsExportError(f"{context} flattened class supports differ from bins")
    micro_ap = _require_score(result["micro_AP"], f"{context}.micro_AP")
    balanced_accuracy = _require_score(
        result["balanced_accuracy"], f"{context}.balanced_accuracy"
    )
    if n_positive == 0 or n_negative == 0:
        expected_reason = "no_positive" if n_positive == 0 else "no_negative"
        if (
            micro_ap is not None
            or result["micro_AP_na_reason"] != expected_reason
            or balanced_accuracy is not None
            or result["balanced_accuracy_na_reason"] != expected_reason
        ):
            raise ThesisResultsExportError(
                f"{context} single-class micro AP/BA semantics differ from support"
            )
    elif (
        micro_ap is None
        or result["micro_AP_na_reason"] is not None
        or balanced_accuracy is None
        or result["balanced_accuracy_na_reason"] is not None
    ):
        raise ThesisResultsExportError(
            f"{context} mixed-class micro AP/BA semantics differ from support"
        )
    expected_positive_rate = n_positive / (n_positive + n_negative)
    if not _same_optional_number(result["positive_rate"], expected_positive_rate):
        raise ThesisResultsExportError(f"{context}.positive_rate differs from support")

    participants = _require_exact_keys(
        result["participants"], {f"P{index}" for index in range(1, 5)},
        f"{context}.participants"
    )
    participant_maps: list[float] = []
    participant_sample_total = 0
    participant_bin_positive = [0, 0, 0, 0]
    participant_bin_negative = [0, 0, 0, 0]
    all_in_scope_evaluable = True
    for participant_id, participant in participants.items():
        participant_payload = _require_exact_keys(
            participant,
            {"in_scope", "n_samples", "mAP", "mAP_na_reason", "bins"},
            f"{context}.participants.{participant_id}",
        )
        if not isinstance(participant_payload["in_scope"], bool):
            raise ThesisResultsExportError(f"{context}.{participant_id}.in_scope is not bool")
        n_samples = _require_int(
            participant_payload["n_samples"], f"{context}.{participant_id}.n_samples"
        )
        participant_sample_total += n_samples
        if participant_payload["in_scope"] is not (n_samples > 0):
            raise ThesisResultsExportError(f"{context}.{participant_id} scope/support differ")
        participant_bins = _require_exact_keys(
            participant_payload["bins"],
            {f"bin{index}" for index in range(4)},
            f"{context}.{participant_id}.bins",
        )
        values: list[float] = []
        for index in range(4):
            summary = _validate_ap_summary(
                participant_bins[f"bin{index}"],
                f"{context}.{participant_id}.bins.bin{index}",
                with_f1=False,
                allow_empty=True,
            )
            if int(summary["n_positive"]) + int(summary["n_negative"]) != n_samples:
                raise ThesisResultsExportError(
                    f"{context}.{participant_id}.bin{index} support differs"
                )
            participant_bin_positive[index] += int(summary["n_positive"])
            participant_bin_negative[index] += int(summary["n_negative"])
            if summary["average_precision"] is not None:
                values.append(float(summary["average_precision"]))
        participant_map = _require_score(
            participant_payload["mAP"], f"{context}.{participant_id}.mAP"
        )
        _require_na_pair(
            participant_map,
            participant_payload["mAP_na_reason"],
            f"{context}.{participant_id}.mAP",
        )
        expected_participant_map = sum(values) / 4 if len(values) == 4 and n_samples else None
        if not _same_optional_number(participant_map, expected_participant_map):
            raise ThesisResultsExportError(f"{context}.{participant_id}.mAP is inconsistent")
        if participant_payload["in_scope"] and participant_map is None:
            all_in_scope_evaluable = False
        if participant_map is not None:
            participant_maps.append(participant_map)
    if participant_sample_total != expected_samples:
        raise ThesisResultsExportError(
            f"{context} participant supports do not sum to frame support"
        )
    if (
        participant_bin_positive != bin_positive_support
        or participant_bin_negative != bin_negative_support
    ):
        raise ThesisResultsExportError(
            f"{context} participant-bin class supports differ from pooled bins"
        )
    expected_participant_macro = (
        sum(participant_maps) / len(participant_maps)
        if participant_maps and all_in_scope_evaluable
        else None
    )
    if not _same_optional_number(result["participant_macro_mAP"], expected_participant_macro):
        raise ThesisResultsExportError(f"{context}.participant_macro_mAP is inconsistent")
    return result


def _validate_binary_event(payload: object, context: str) -> dict[str, object]:
    result = _require_exact_keys(
        payload,
        {
            "n_events",
            "n_positive",
            "n_negative",
            "balanced_accuracy",
            "balanced_accuracy_na_reason",
            "f1",
            "f1_na_reason",
            "average_precision",
            "average_precision_na_reason",
        },
        context,
    )
    n_events = _require_int(result["n_events"], f"{context}.n_events")
    n_positive = _require_int(result["n_positive"], f"{context}.n_positive")
    n_negative = _require_int(result["n_negative"], f"{context}.n_negative")
    if n_positive + n_negative != n_events:
        raise ThesisResultsExportError(f"{context} event support is inconsistent")
    balanced_accuracy = _require_score(
        result["balanced_accuracy"], f"{context}.balanced_accuracy"
    )
    f1 = _require_score(result["f1"], f"{context}.f1")
    average_precision = _require_score(
        result["average_precision"], f"{context}.average_precision"
    )
    if n_events == 0:
        if any(value is not None for value in (balanced_accuracy, f1, average_precision)) or any(
            result[key] != "no_events"
            for key in (
                "balanced_accuracy_na_reason",
                "f1_na_reason",
                "average_precision_na_reason",
            )
        ):
            raise ThesisResultsExportError(f"{context} empty-event metric semantics differ")
    elif n_positive == 0 or n_negative == 0:
        expected_reason = "no_positive_events" if n_positive == 0 else "no_negative_events"
        if (
            balanced_accuracy is not None
            or result["balanced_accuracy_na_reason"] != expected_reason
            or average_precision is not None
            or result["average_precision_na_reason"] != expected_reason
            or f1 is None
            or result["f1_na_reason"] is not None
        ):
            raise ThesisResultsExportError(f"{context} single-class metric semantics differ")
    elif (
        balanced_accuracy is None
        or f1 is None
        or average_precision is None
        or any(
            result[key] is not None
            for key in (
                "balanced_accuracy_na_reason",
                "f1_na_reason",
                "average_precision_na_reason",
            )
        )
    ):
        raise ThesisResultsExportError(f"{context} mixed-class metric semantics differ")
    return result


def _validate_next_speaker(payload: object, context: str) -> dict[str, object]:
    result = _require_exact_keys(
        payload,
        {
            "n_events",
            "n_participants_in_scope",
            "n_participants_evaluable_ap",
            "participant_macro_ovr_ap",
            "participant_macro_ovr_ap_na_reason",
            "balanced_recall",
            "mrr",
            "top1_accuracy",
            "tie_policy",
            "tie_rate",
            "per_participant",
            "na_reason",
        },
        context,
    )
    n_events = _require_int(result["n_events"], f"{context}.n_events")
    n_in_scope = _require_int(
        result["n_participants_in_scope"], f"{context}.n_participants_in_scope"
    )
    n_evaluable = _require_int(
        result["n_participants_evaluable_ap"], f"{context}.n_participants_evaluable_ap"
    )
    if n_in_scope > 4 or n_evaluable > n_in_scope:
        raise ThesisResultsExportError(f"{context} participant counts are inconsistent")
    if result["tie_policy"] != "uniform_credit_among_max_scores":
        raise ThesisResultsExportError(f"{context} tie policy differs from the evaluator")
    overall_scores = {
        key: _require_score(result[key], f"{context}.{key}")
        for key in (
            "participant_macro_ovr_ap",
            "balanced_recall",
            "mrr",
            "top1_accuracy",
            "tie_rate",
        )
    }
    if n_events == 0:
        if result["na_reason"] != "no_events":
            raise ThesisResultsExportError(f"{context} empty result lacks no_events reason")
    elif result["na_reason"] is not None:
        raise ThesisResultsExportError(f"{context} non-empty result has an NA reason")
    participants = _require_exact_keys(
        result["per_participant"],
        {f"P{index}" for index in range(1, 5)},
        f"{context}.per_participant",
    )
    observed_in_scope = 0
    observed_evaluable = 0
    participant_ap_values: list[float] = []
    participant_recall_values: list[float] = []
    all_in_scope_ap_evaluable = True
    total_participant_positives = 0
    for participant_id, participant in participants.items():
        item = _require_exact_keys(
            participant,
            {
                "in_scope",
                "n_positive",
                "n_negative",
                "ovr_average_precision",
                "tie_aware_recall",
                "ap_na_reason",
                "recall_na_reason",
            },
            f"{context}.{participant_id}",
        )
        if not isinstance(item["in_scope"], bool):
            raise ThesisResultsExportError(f"{context}.{participant_id}.in_scope is not bool")
        n_positive = _require_int(
            item["n_positive"], f"{context}.{participant_id}.n_positive"
        )
        n_negative = _require_int(
            item["n_negative"], f"{context}.{participant_id}.n_negative"
        )
        participant_support = n_positive + n_negative
        total_participant_positives += n_positive
        if participant_support > n_events or item["in_scope"] is not (participant_support > 0):
            raise ThesisResultsExportError(
                f"{context}.{participant_id} scope/support is inconsistent"
            )
        ap = _require_score(
            item["ovr_average_precision"], f"{context}.{participant_id}.ovr_average_precision"
        )
        recall = _require_score(
            item["tie_aware_recall"], f"{context}.{participant_id}.tie_aware_recall"
        )
        if not item["in_scope"]:
            expected_ap_reason = "not_in_scope"
        elif n_positive == 0:
            expected_ap_reason = "no_positive_events"
        elif n_negative == 0:
            expected_ap_reason = "no_negative_events"
        else:
            expected_ap_reason = None
        if (
            expected_ap_reason is None
            and (ap is None or item["ap_na_reason"] is not None)
        ) or (
            expected_ap_reason is not None
            and (ap is not None or item["ap_na_reason"] != expected_ap_reason)
        ):
            raise ThesisResultsExportError(
                f"{context}.{participant_id} AP/NA semantics differ from support"
            )
        if (
            n_positive == 0
            and (recall is not None or item["recall_na_reason"] != "no_positive_events")
        ) or (n_positive > 0 and (recall is None or item["recall_na_reason"] is not None)):
            raise ThesisResultsExportError(
                f"{context}.{participant_id} recall/NA semantics differ from support"
            )
        observed_in_scope += int(item["in_scope"])
        observed_evaluable += int(ap is not None)
        if item["in_scope"] and ap is None:
            all_in_scope_ap_evaluable = False
        if ap is not None:
            participant_ap_values.append(ap)
        if recall is not None:
            participant_recall_values.append(recall)
    if observed_in_scope != n_in_scope or observed_evaluable != n_evaluable:
        raise ThesisResultsExportError(f"{context} participant summary counts differ")
    if total_participant_positives != n_events:
        raise ThesisResultsExportError(
            f"{context} participant positive supports do not sum to event count"
        )
    expected_macro_ap = (
        sum(participant_ap_values) / len(participant_ap_values)
        if participant_ap_values and all_in_scope_ap_evaluable
        else None
    )
    if not _same_optional_number(result["participant_macro_ovr_ap"], expected_macro_ap):
        raise ThesisResultsExportError(f"{context} participant macro AP is inconsistent")
    expected_macro_reason = (
        None
        if expected_macro_ap is not None
        else (
            "no_events"
            if n_events == 0
            else "one_or_more_in_scope_participants_single_class"
        )
    )
    if result["participant_macro_ovr_ap_na_reason"] != expected_macro_reason:
        raise ThesisResultsExportError(f"{context} participant macro AP reason is inconsistent")
    expected_balanced_recall = (
        sum(participant_recall_values) / len(participant_recall_values)
        if participant_recall_values
        else None
    )
    if not _same_optional_number(result["balanced_recall"], expected_balanced_recall):
        raise ThesisResultsExportError(f"{context} balanced recall is inconsistent")
    if n_events == 0:
        if any(
            overall_scores[key] is not None
            for key in ("mrr", "top1_accuracy", "tie_rate")
        ):
            raise ThesisResultsExportError(f"{context} empty-event summary scores must be null")
    elif any(
        overall_scores[key] is None
        for key in ("balanced_recall", "mrr", "top1_accuracy", "tie_rate")
    ):
        raise ThesisResultsExportError(f"{context} non-empty summary scores must be finite")
    return result


def _validate_metrics(payload: object, context: str) -> dict[str, object]:
    result = _require_exact_keys(
        payload, {"threshold_policy", "frame", "events"}, context
    )
    if result["threshold_policy"] != THRESHOLD_POLICY:
        raise ThesisResultsExportError(f"{context} threshold policy differs")
    _validate_frame(result["frame"], f"{context}.frame")
    events = _require_exact_keys(
        result["events"],
        {"shift_hold", "shift_prediction", "overlap_resolution", "robot_entry", "next_speaker"},
        f"{context}.events",
    )
    for name in ("shift_hold", "shift_prediction", "overlap_resolution", "robot_entry"):
        _validate_binary_event(events[name], f"{context}.events.{name}")
    _validate_next_speaker(events["next_speaker"], f"{context}.events.next_speaker")
    return result


def _validate_fold_assignments(
    fold_sessions: object,
    fold_groups: object,
    context: str,
) -> tuple[dict[str, object], dict[str, object]]:
    sessions = _require_exact_keys(fold_sessions, set(FOLDS), f"{context}.fold_session_ids")
    groups = _require_exact_keys(fold_groups, set(FOLDS), f"{context}.fold_group_ids")
    flattened: list[str] = []
    observed_groups: list[str] = []
    for fold_id in FOLDS:
        values = sessions[fold_id]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) for item in values)
            or len(values) != len(set(values))
        ):
            raise ThesisResultsExportError(f"{context}.{fold_id} sessions are malformed")
        group_id = groups[fold_id]
        if not isinstance(group_id, str) or group_id not in EXPECTED_GROUPS:
            raise ThesisResultsExportError(f"{context}.{fold_id} group is invalid")
        if any(DEADLINE_SESSION_TO_GROUP.get(item) != group_id for item in values):
            raise ThesisResultsExportError(f"{context}.{fold_id} session/group mapping differs")
        flattened.extend(values)
        observed_groups.append(group_id)
    if tuple(sorted(flattened)) != tuple(sorted(EXPECTED_SESSIONS)):
        raise ThesisResultsExportError(f"{context} does not partition the exact eight scenes")
    if tuple(sorted(observed_groups)) != EXPECTED_GROUPS:
        raise ThesisResultsExportError(f"{context} does not hold out every group exactly once")
    return sessions, groups


def _validate_condition(
    payload: object,
    condition_id: str,
    paired_sessions: dict[str, object],
    paired_groups: dict[str, object],
    run_manifest_sha256: str,
) -> dict[str, object]:
    expected = {
        "artifact_kind",
        "format_version",
        "backend_id",
        "result_role",
        "data_kind",
        "production_eligible",
        "counted_eligible",
        "counted_blockers",
        "condition_id",
        "feature_set",
        "seed",
        "real_session_authority_sha256",
        "source_manifest_sha256",
        "run_manifest_sha256",
        "outer_fold_ids",
        "session_ids",
        "fold_session_ids",
        "fold_group_ids",
        "prediction_artifact_sha256",
        "aggregation_source",
        "threshold_policy",
        "pooled",
        "per_fold",
    }
    result = _require_exact_keys(payload, expected, f"conditions.{condition_id}")
    base_expected = {
        "artifact_kind": CONDITION_AGGREGATE_KIND,
        "format_version": 1,
        "backend_id": BACKEND_ID,
        "result_role": RESULT_ROLE,
        "data_kind": "real",
        "production_eligible": False,
        "counted_eligible": False,
        "counted_blockers": list(COUNTED_BLOCKERS),
        "condition_id": condition_id,
        "feature_set": CONDITION_FEATURE_SETS[condition_id],
        "seed": EXPECTED_SEED,
        "run_manifest_sha256": run_manifest_sha256,
        "outer_fold_ids": list(FOLDS),
        "session_ids": sorted(EXPECTED_SESSIONS),
        "aggregation_source": "saved_outer_probabilities_only_no_model_inference",
        "threshold_policy": THRESHOLD_POLICY,
    }
    for key, expected_value in base_expected.items():
        if result[key] != expected_value:
            raise ThesisResultsExportError(
                f"conditions.{condition_id}.{key} differs from the export contract"
            )
    for key in ("real_session_authority_sha256", "source_manifest_sha256"):
        if not _is_sha256(result[key]):
            raise ThesisResultsExportError(f"conditions.{condition_id}.{key} is not SHA-256")
    hashes = result["prediction_artifact_sha256"]
    if (
        not isinstance(hashes, list)
        or len(hashes) != 4
        or not all(_is_sha256(item) for item in hashes)
    ):
        raise ThesisResultsExportError(
            f"conditions.{condition_id} must bind four prediction artifacts"
        )
    condition_sessions, condition_groups = _validate_fold_assignments(
        result["fold_session_ids"], result["fold_group_ids"], f"conditions.{condition_id}"
    )
    if condition_sessions != paired_sessions or condition_groups != paired_groups:
        raise ThesisResultsExportError(f"conditions.{condition_id} is not paired by fold")
    _validate_metrics(result["pooled"], f"conditions.{condition_id}.pooled")
    per_fold = _require_exact_keys(
        result["per_fold"], set(FOLDS), f"conditions.{condition_id}.per_fold"
    )
    for fold_id in FOLDS:
        _validate_metrics(per_fold[fold_id], f"conditions.{condition_id}.per_fold.{fold_id}")
    return result


def validate_aggregate(payload: object) -> dict[str, object]:
    """Fail closed on any aggregate that is not the exact deadline-8 primary matrix."""

    expected = {
        "artifact_kind",
        "format_version",
        "result_role",
        "data_kind",
        "production_eligible",
        "counted_eligible",
        "analysis_variant",
        "included_session_ids",
        "group_ids",
        "corpus_exclusions",
        "seed",
        "aggregation_source",
        "threshold_policy",
        "run_manifest_sha256",
        "paired_fold_session_ids",
        "paired_fold_group_ids",
        "conditions",
        "paired_comparisons",
    }
    result = _require_exact_keys(payload, expected, "aggregate")
    fixed = {
        "artifact_kind": SOURCE_ARTIFACT_KIND,
        "format_version": 1,
        "result_role": RESULT_ROLE,
        "data_kind": "real",
        "production_eligible": False,
        "counted_eligible": False,
        "analysis_variant": "primary",
        "included_session_ids": list(EXPECTED_SESSIONS),
        "group_ids": list(EXPECTED_GROUPS),
        "corpus_exclusions": dict(DEADLINE_CORPUS_EXCLUSIONS),
        "seed": EXPECTED_SEED,
        "aggregation_source": "saved_outer_probabilities_only_no_model_inference",
        "threshold_policy": THRESHOLD_POLICY,
    }
    for key, expected_value in fixed.items():
        if result[key] != expected_value:
            raise ThesisResultsExportError(f"aggregate.{key} differs from the deadline contract")
    if not _is_sha256(result["run_manifest_sha256"]):
        raise ThesisResultsExportError("aggregate.run_manifest_sha256 is not SHA-256")
    paired_sessions, paired_groups = _validate_fold_assignments(
        result["paired_fold_session_ids"], result["paired_fold_group_ids"], "aggregate.paired"
    )
    conditions = _require_exact_keys(result["conditions"], set(CONDITIONS), "conditions")
    for condition_id in CONDITIONS:
        _validate_condition(
            conditions[condition_id],
            condition_id,
            paired_sessions,
            paired_groups,
            str(result["run_manifest_sha256"]),
        )
    comparisons = _require_exact_keys(
        result["paired_comparisons"],
        {"A+H_minus_A", "A+H_minus_H"},
        "paired_comparisons",
    )
    for baseline in ("A", "H"):
        key = f"A+H_minus_{baseline}"
        comparison = _require_exact_keys(
            comparisons[key],
            {"pooled_frame_mAP_difference", "interpretation"},
            f"paired_comparisons.{key}",
        )
        if comparison["interpretation"] != "descriptive_paired_difference_not_a_significance_test":
            raise ThesisResultsExportError(f"paired_comparisons.{key} interpretation differs")
        fused = conditions["A+H"]["pooled"]["frame"]["mAP"]
        reference = conditions[baseline]["pooled"]["frame"]["mAP"]
        expected_difference = None if fused is None or reference is None else fused - reference
        if not _same_optional_number(
            comparison["pooled_frame_mAP_difference"], expected_difference
        ):
            raise ThesisResultsExportError(f"paired_comparisons.{key} value is inconsistent")
    return result


def _paper_condition_summary(condition: dict[str, object]) -> dict[str, object]:
    pooled = condition["pooled"]
    frame = pooled["frame"]
    events = pooled["events"]
    return {
        "feature_set": condition["feature_set"],
        "frame": {
            "mAP": frame["mAP"],
            "mAP_na_reason": frame["mAP_na_reason"],
            "macro_f1_bins_at_0.5": frame["macro_f1_bins"],
            "balanced_accuracy_at_0.5": frame["balanced_accuracy"],
            "f1_at_0.5": frame["f1"],
            "participant_macro_mAP": frame["participant_macro_mAP"],
            "n_positive": frame["n_positive"],
            "n_negative": frame["n_negative"],
            "bins": {
                f"bin{index}": {
                    "average_precision": frame["bins"][f"bin{index}"]["average_precision"],
                    "f1_at_0.5": frame["bins"][f"bin{index}"]["f1"],
                    "n_positive": frame["bins"][f"bin{index}"]["n_positive"],
                    "n_negative": frame["bins"][f"bin{index}"]["n_negative"],
                    "ap_na_reason": frame["bins"][f"bin{index}"]["ap_na_reason"],
                }
                for index in range(4)
            },
        },
        "events": {
            name: {
                "average_precision": events[name]["average_precision"],
                "balanced_accuracy_at_0.5": events[name]["balanced_accuracy"],
                "f1_at_0.5": events[name]["f1"],
                "n_events": events[name]["n_events"],
                "n_positive": events[name]["n_positive"],
                "n_negative": events[name]["n_negative"],
                "average_precision_na_reason": events[name]["average_precision_na_reason"],
            }
            for name in ("shift_hold", "shift_prediction", "overlap_resolution", "robot_entry")
        }
        | {
            "next_speaker": {
                "participant_macro_ovr_ap": events["next_speaker"]["participant_macro_ovr_ap"],
                "balanced_recall": events["next_speaker"]["balanced_recall"],
                "mrr": events["next_speaker"]["mrr"],
                "top1_accuracy": events["next_speaker"]["top1_accuracy"],
                "tie_rate": events["next_speaker"]["tie_rate"],
                "n_events": events["next_speaker"]["n_events"],
                "ap_na_reason": events["next_speaker"][
                    "participant_macro_ovr_ap_na_reason"
                ],
            }
        },
        "per_fold_frame_mAP": {
            fold_id: condition["per_fold"][fold_id]["frame"]["mAP"] for fold_id in FOLDS
        },
    }


def build_paper_summary(
    aggregate: dict[str, object],
    source_path: Path,
    source_snapshot: dict[str, object],
) -> dict[str, object]:
    """Create the compact, support-aware JSON used by the table and plot."""

    return {
        "artifact_kind": "vap_thesis_results_paper_summary",
        "format_version": EXPORT_FORMAT_VERSION,
        "source_aggregate": {
            "path": str(source_path),
            "sha256": source_snapshot["sha256"],
            "size_bytes": source_snapshot["size_bytes"],
            "artifact_kind": aggregate["artifact_kind"],
            "run_manifest_sha256": aggregate["run_manifest_sha256"],
        },
        "scope": {
            "result_role": aggregate["result_role"],
            "production_eligible": aggregate["production_eligible"],
            "counted_eligible": aggregate["counted_eligible"],
            "analysis_variant": aggregate["analysis_variant"],
            "seed": aggregate["seed"],
            "included_session_ids": aggregate["included_session_ids"],
            "group_ids": aggregate["group_ids"],
            "corpus_exclusions": aggregate["corpus_exclusions"],
            "threshold_policy": aggregate["threshold_policy"],
            "aggregation_source": aggregate["aggregation_source"],
        },
        "claim_boundary": {
            "paper_label": "deadline non-production nested-LOGO evaluation",
            "pooled_estimator": "pooled saved out-of-fold predictions; not a mean of fold scores",
            "paired_differences": "descriptive only; no significance test",
            "fixed_threshold_metrics": "diagnostic at 0.5; not calibrated robot-policy claims",
            "one_seed_only": True,
            "d3_sensitivity_included": False,
        },
        "fold_session_ids": aggregate["paired_fold_session_ids"],
        "fold_group_ids": aggregate["paired_fold_group_ids"],
        "conditions": {
            condition_id: _paper_condition_summary(aggregate["conditions"][condition_id])
            for condition_id in CONDITIONS
        },
        "paired_comparisons": aggregate["paired_comparisons"],
    }


def _format_metric(value: object) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def _latex_metric(value: object) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def _latex_event_cell(event: dict[str, object]) -> str:
    score = _latex_metric(event["average_precision"])
    return f"{score} [{event['n_positive']}/{event['n_negative']}]"


def render_latex_table(summary: dict[str, object], source_sha256: str) -> str:
    """Return a standalone table snippet; callers decide where to include it."""

    conditions = summary["conditions"]
    rows: list[tuple[str, list[str]]] = []
    rows.append(("Frame mAP", [_latex_metric(conditions[c]["frame"]["mAP"]) for c in CONDITIONS]))
    rows.append(
        (
            "Frame macro F1 at 0.5",
            [_latex_metric(conditions[c]["frame"]["macro_f1_bins_at_0.5"]) for c in CONDITIONS],
        )
    )
    rows.append(
        (
            "Frame balanced accuracy at 0.5",
            [_latex_metric(conditions[c]["frame"]["balanced_accuracy_at_0.5"]) for c in CONDITIONS],
        )
    )
    for index, interval in enumerate(("0--0.2 s", "0.2--0.6 s", "0.6--1.2 s", "1.2--2.0 s")):
        rows.append(
            (
                f"Future-bin AP ({interval})",
                [
                    _latex_metric(
                        conditions[c]["frame"]["bins"][f"bin{index}"]["average_precision"]
                    )
                    for c in CONDITIONS
                ],
            )
        )
    labels = {
        "shift_hold": "Shift/hold AP",
        "shift_prediction": "Shift-prediction AP",
        "overlap_resolution": "Overlap-resolution AP",
        "robot_entry": "Robot-entry AP",
    }
    for event_name, label in labels.items():
        rows.append(
            (
                label,
                [_latex_event_cell(conditions[c]["events"][event_name]) for c in CONDITIONS],
            )
        )
    rows.extend(
        (
            (
                "Next-speaker macro OvR AP",
                [
                    _latex_metric(
                        conditions[c]["events"]["next_speaker"]["participant_macro_ovr_ap"]
                    )
                    for c in CONDITIONS
                ],
            ),
            (
                "Next-speaker top-1 accuracy",
                [
                    _latex_metric(conditions[c]["events"]["next_speaker"]["top1_accuracy"])
                    for c in CONDITIONS
                ],
            ),
        )
    )
    lines = [
        "% Generated by tools/export_thesis_results.py; do not edit values by hand.",
        f"% Source aggregate SHA-256: {source_sha256}",
        "\\begin{table}[tbp]",
        "\\centering",
        "\\caption{Pooled out-of-fold performance on the eight-scene real-corpus deadline matrix.}",
        "\\label{table:real_corpus_ah_results}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Metric & Audio (A) & Head pose (H) & Audio + head pose (A+H) \\\\",
        "\\midrule",
    ]
    for label, values in rows:
        lines.append(f"{label} & " + " & ".join(values) + " \\\\")
    lines.extend(
        (
            "\\bottomrule",
            "\\end{tabular}",
            "\\par\\smallskip",
            "\\begin{minipage}{0.96\\linewidth}\\footnotesize",
            "Event AP cells report AP [positive/negative support]. Values are pooled from one "
            "saved "
            "out-of-fold prediction pass per condition and outer fold, not averaged across folds. "
            "Dashes denote non-evaluable metrics with explicit reasons in the accompanying JSON. "
            "The comparison is descriptive, uses one seed, and is a deadline non-production "
            "nested-LOGO evaluation; it is not a significance test or a calibrated robot-policy "
            "claim.",
            "\\end{minipage}",
            "\\end{table}",
            "",
        )
    )
    return "\n".join(lines)


def write_fold_csv(summary: dict[str, object], path: Path) -> None:
    fields = (
        "fold_id",
        "test_group",
        "condition_id",
        "frame_mAP",
        "frame_macro_f1_at_0.5",
        "shift_hold_AP",
        "shift_prediction_AP",
        "overlap_resolution_AP",
        "robot_entry_AP",
        "next_speaker_macro_ovr_AP",
        "next_speaker_top1_accuracy",
    )
    aggregate = summary["_validated_aggregate"]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for fold_id in FOLDS:
            for condition_id in CONDITIONS:
                metrics = aggregate["conditions"][condition_id]["per_fold"][fold_id]
                frame = metrics["frame"]
                events = metrics["events"]
                row = {
                    "fold_id": fold_id,
                    "test_group": aggregate["paired_fold_group_ids"][fold_id],
                    "condition_id": condition_id,
                    "frame_mAP": _format_metric(frame["mAP"]),
                    "frame_macro_f1_at_0.5": _format_metric(frame["macro_f1_bins"]),
                    "shift_hold_AP": _format_metric(events["shift_hold"]["average_precision"]),
                    "shift_prediction_AP": _format_metric(
                        events["shift_prediction"]["average_precision"]
                    ),
                    "overlap_resolution_AP": _format_metric(
                        events["overlap_resolution"]["average_precision"]
                    ),
                    "robot_entry_AP": _format_metric(events["robot_entry"]["average_precision"]),
                    "next_speaker_macro_ovr_AP": _format_metric(
                        events["next_speaker"]["participant_macro_ovr_ap"]
                    ),
                    "next_speaker_top1_accuracy": _format_metric(
                        events["next_speaker"]["top1_accuracy"]
                    ),
                }
                writer.writerow(row)


def make_results_figure(summary: dict[str, object]) -> matplotlib.figure.Figure:
    """Plot pooled metrics and paired fold dispersion without inferential error bars."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    conditions = summary["conditions"]
    metrics = (
        ("Frame\nmAP", lambda item: item["frame"]["mAP"]),
        ("Shift/\nhold AP", lambda item: item["events"]["shift_hold"]["average_precision"]),
        (
            "Shift-\npred. AP",
            lambda item: item["events"]["shift_prediction"]["average_precision"],
        ),
        (
            "Overlap\nAP",
            lambda item: item["events"]["overlap_resolution"]["average_precision"],
        ),
        ("Robot\nentry AP", lambda item: item["events"]["robot_entry"]["average_precision"]),
        (
            "Next spk.\nmacro AP",
            lambda item: item["events"]["next_speaker"]["participant_macro_ovr_ap"],
        ),
    )
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.65),
        gridspec_kw={"width_ratios": [1.55, 1]},
    )
    width = 0.24
    x_values = list(range(len(metrics)))
    for condition_index, condition_id in enumerate(CONDITIONS):
        offset = (condition_index - 1) * width
        values = [getter(conditions[condition_id]) for _, getter in metrics]
        for metric_index, value in enumerate(values):
            x = x_values[metric_index] + offset
            if value is None:
                left.text(x, 0.025, "NA", ha="center", va="bottom", rotation=90, fontsize=6.5)
            else:
                left.bar(
                    x,
                    float(value),
                    width=width * 0.9,
                    color=CONDITION_COLOURS[condition_id],
                )
    left.set_xticks(x_values, [label for label, _ in metrics])
    left.set_ylim(0, 1.05)
    left.set_ylabel("Score")
    left.set_title("(a) Pooled saved OOF predictions")
    left.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)
    left.legend(
        handles=[
            Patch(facecolor=CONDITION_COLOURS[condition_id], label=condition_id)
            for condition_id in CONDITIONS
        ],
        frameon=False,
        ncol=3,
        loc="upper center",
    )

    aggregate = summary["_validated_aggregate"]
    markers = ("o", "s", "^", "D")
    for fold_index, fold_id in enumerate(FOLDS):
        values = [
            aggregate["conditions"][condition_id]["per_fold"][fold_id]["frame"]["mAP"]
            for condition_id in CONDITIONS
        ]
        finite_x = [index for index, value in enumerate(values) if value is not None]
        finite_y = [float(values[index]) for index in finite_x]
        if finite_x:
            right.plot(
                finite_x,
                finite_y,
                color=MID_GREY,
                alpha=0.72,
                linewidth=1.0,
                marker=markers[fold_index],
                markersize=4.5,
            )
        for index, value in enumerate(values):
            if value is None:
                right.text(index, 0.025 + 0.025 * fold_index, "NA", ha="center", fontsize=6)
    right.set_xticks(range(3), CONDITIONS)
    right.set_xlim(-0.35, 2.35)
    right.set_ylim(0, 1.05)
    right.set_ylabel("Frame mAP")
    right.set_title("(b) Paired held-out-group dispersion")
    right.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)
    right.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=MID_GREY,
                linewidth=1.0,
                marker=markers[fold_index],
                markersize=4.5,
                label=aggregate["paired_fold_group_ids"][fold_id],
            )
            for fold_index, fold_id in enumerate(FOLDS)
        ],
        title="Test group",
        frameon=False,
        loc="best",
        fontsize=6.8,
        title_fontsize=7,
    )
    fig.text(
        0.5,
        0.012,
        "Descriptive comparison only (one seed); lines pair identical held-out groups and are "
        "not confidence intervals.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=DARK,
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.19, top=0.91, wspace=0.28)
    return fig


def _save_figure(fig: matplotlib.figure.Figure, output_dir: Path) -> dict[str, object]:
    created = FIXED_ARTIFACT_TIME
    metadata = {
        "pdf": {
            "Title": "Real-corpus A H A+H outer results",
            "Author": "Yixuan Zhuang",
            "Creator": "tools/export_thesis_results.py",
            "CreationDate": created,
            "ModDate": created,
        },
        "png": {
            "Title": "Real-corpus A H A+H outer results",
            "Author": "Yixuan Zhuang",
            "Software": "tools/export_thesis_results.py",
            "Creation Time": created.isoformat(),
        },
    }
    outputs: dict[str, object] = {}
    for suffix in ("pdf", "png"):
        path = output_dir / f"real_corpus_results.{suffix}"
        fig.savefig(
            path,
            format=suffix,
            dpi=300 if suffix == "png" else None,
            facecolor=WHITE,
            metadata=metadata[suffix],
        )
        outputs[suffix] = {
            "file": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    plt.close(fig)
    return outputs


def _write_json_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        handle.write("\n")


def _write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _current_snapshot(path: Path) -> tuple[tuple[int, int, int, int], str]:
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    return identity, _sha256_file(path)


def export_results(aggregate_path: str | Path, output_directory: str | Path) -> Path:
    """Validate one aggregate and create a no-clobber paper-derivative bundle."""

    source = Path(aggregate_path).resolve(strict=True)
    if not source.is_file():
        raise ThesisResultsExportError(f"aggregate is not a regular file: {source}")
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite results export: {destination}")
    aggregate, snapshot = _read_strict_json(source)
    validate_aggregate(aggregate)
    paper_summary = build_paper_summary(aggregate, source, snapshot)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.stage"
    stage.mkdir()
    try:
        summary_path = stage / "real_corpus_results_summary.json"
        _write_json_exclusive(summary_path, paper_summary)
        table_path = stage / "real_corpus_results_table.tex"
        _write_text_exclusive(
            table_path,
            render_latex_table(paper_summary, str(snapshot["sha256"])),
        )
        internal_summary = dict(paper_summary)
        internal_summary["_validated_aggregate"] = aggregate
        fold_path = stage / "real_corpus_fold_metrics.csv"
        write_fold_csv(internal_summary, fold_path)
        figure_outputs = _save_figure(make_results_figure(internal_summary), stage)
        expected_identity = tuple(snapshot["identity"])
        current_identity, current_hash = _current_snapshot(source)
        if current_identity != expected_identity or current_hash != snapshot["sha256"]:
            raise ThesisResultsExportError("aggregate changed before export publication")
        generator = Path(__file__).resolve()
        output_paths = (summary_path, table_path, fold_path)
        manifest = {
            "artifact_kind": "vap_thesis_results_export_bundle",
            "format_version": EXPORT_FORMAT_VERSION,
            "generator": {
                "path": str(generator),
                "sha256": _sha256_file(generator),
            },
            "source_aggregate": {
                "path": str(source),
                "sha256": snapshot["sha256"],
                "size_bytes": snapshot["size_bytes"],
                "run_manifest_sha256": aggregate["run_manifest_sha256"],
            },
            "outputs": {
                path.name: {
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in output_paths
            }
            | {
                f"real_corpus_results.{suffix}": binding
                for suffix, binding in figure_outputs.items()
            },
            "input_mutation": False,
            "checkpoint_or_model_loaded": False,
            "report_main_tex_modified": False,
        }
        _write_json_exclusive(stage / "export_manifest.json", manifest)
        if destination.exists():
            raise FileExistsError(f"results export appeared concurrently: {destination}")
        stage.rename(destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return destination / "export_manifest.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = export_results(args.aggregate, args.output)
    print(json.dumps({"export_manifest": str(manifest)}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
