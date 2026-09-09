"""Run the non-production, real-data MSc A/H/A+H outer evaluation.

The command is intentionally separate from the production registry.  It uses
the audited-real loader and the existing four-group nested-LOGO orchestrator,
but every result remains explicitly non-production and non-counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from hri_turn_taking.data.assembly import load_and_audit_vap_source
from hri_turn_taking.vap.dataset import FeatureNormalizer, sessions_from_dataframe
from hri_turn_taking.vap.feature_sets import apply_feature_set
from hri_turn_taking.vap.outer_experiment import (
    DEADLINE_CORPUS_EXCLUSIONS,
    DEADLINE_EXPECTED_N_ROWS,
    DEADLINE_SESSION_TO_GROUP,
    FROZEN_SEEDS,
    InnerFitRequest,
    OuterCondition,
    freeze_outer_protocol,
    prepare_outer_condition,
    prepared_outer_authority_inputs,
    run_outer_unit,
    validate_outer_run_manifest,
    validate_outer_split_manifest,
)
from hri_turn_taking.vap.thesis_backend import (
    CONDITION_FEATURE_SETS,
    RESULT_ROLE,
    ThesisBackendSpec,
    ThesisRunBinding,
    ThesisTorchBackend,
    aggregate_outer_prediction_artifacts,
    build_real_session_authority,
    load_metrics_artifact,
    load_prediction_artifact,
    thesis_model_config,
    thesis_trainer_config,
)

CONDITIONS = (
    OuterCondition("A", "audio_only"),
    OuterCondition("H", "head_pose_only"),
    OuterCondition("A+H", "audio_plus_head_pose"),
)
OUTER_FOLD_IDS = ("outer_01", "outer_02", "outer_03", "outer_04")
SAFE_CONDITION_DIR = {"A": "A", "H": "H", "A+H": "A_plus_H"}
EXPECTED_PRIMARY_SESSIONS = tuple(DEADLINE_SESSION_TO_GROUP)
EXPECTED_GROUP_IDS = tuple(sorted(set(DEADLINE_SESSION_TO_GROUP.values())))
CORPUS_EXCLUSIONS = dict(DEADLINE_CORPUS_EXCLUSIONS)


class ThesisRunnerError(ValueError):
    """A deadline runner preflight, matrix, or aggregation gate failed."""


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ThesisRunnerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ThesisRunnerError(f"non-standard JSON constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ThesisRunnerError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ThesisRunnerError(f"JSON root must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        handle.write("\n")


def _load_protocol(protocol_directory: str | Path) -> tuple[dict[str, object], dict[str, object]]:
    root = Path(protocol_directory).resolve()
    split = _read_json(root / "split_manifest.json")
    run = _read_json(root / "run_manifest.json")
    validate_outer_split_manifest(split)
    validate_outer_run_manifest(run, split)
    expected = [
        {
            "condition_id": item.condition_id,
            "feature_set": item.feature_set,
            "encoder": item.encoder,
        }
        for item in CONDITIONS
    ]
    if split.get("conditions") != expected:
        raise ThesisRunnerError("protocol does not freeze the exact A/H/A+H matrix")
    if (
        split.get("source_session_to_group") is None
        or tuple(sorted(split["source_session_to_group"])) != EXPECTED_PRIMARY_SESSIONS
    ):
        raise ThesisRunnerError("protocol does not bind the exact eight-scene deadline corpus")
    if split.get("corpus_exclusions") != CORPUS_EXCLUSIONS:
        raise ThesisRunnerError("protocol does not bind the two owner-pending exclusions")
    return split, run


def freeze_protocol(
    source: str | Path,
    destination: str | Path,
    *,
    max_epochs: int,
    spec: ThesisBackendSpec | None = None,
) -> Path:
    selected_spec = spec or ThesisBackendSpec()
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ThesisRunnerError("max_epochs must be a positive integer")
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {destination_path}")
    bundle = freeze_outer_protocol(
        source,
        destination_path,
        conditions=CONDITIONS,
        max_epochs=max_epochs,
        trainer_config=thesis_trainer_config(selected_spec),
        model_config=thesis_model_config(selected_spec),
        production_backend=None,
        excluded_session_ids=tuple(CORPUS_EXCLUSIONS),
    )
    _, run = _load_protocol(bundle.directory)
    ThesisRunBinding.from_run_manifest(run, selected_spec)
    return bundle.directory


def _load_authority(source: str | Path, condition_id: str) -> tuple[object, object, object]:
    feature_set = CONDITION_FEATURE_SETS[condition_id]
    frame, report = load_and_audit_vap_source(
        Path(source).resolve(),
        data_kind="real",
        feature_set=feature_set,
        excluded_session_ids=tuple(CORPUS_EXCLUSIONS),
    )
    if (
        tuple(sorted(report.session_ids)) != EXPECTED_PRIMARY_SESSIONS
        or tuple(sorted(report.group_ids)) != EXPECTED_GROUP_IDS
        or report.n_rows != DEADLINE_EXPECTED_N_ROWS
    ):
        raise ThesisRunnerError(
            "audited source is not the exact 64,299-row/eight-scene/four-group deadline corpus"
        )
    if not report.target_ready or report.selected_feature_ready is not True:
        raise ThesisRunnerError(
            f"condition {condition_id} is not train-ready: "
            f"{[*report.target_blockers, *report.selected_feature_blockers]}"
        )
    authority = build_real_session_authority(
        frame,
        report.to_dict(),
        condition_id=condition_id,
    )
    return frame, report, authority


def _fit_pos_weight(sessions: tuple[object, ...], max_weight: float = 10.0) -> np.ndarray:
    positives = np.zeros(4, dtype=np.float64)
    totals = np.zeros(4, dtype=np.float64)
    for session in sessions:
        mask = session.valid[:, None, None] & (session.present[None, :, None] > 0)
        expanded = np.broadcast_to(mask, session.labels.shape)
        positives += (session.labels * expanded).sum(axis=(0, 1))
        totals += expanded.sum(axis=(0, 1))
    negatives = totals - positives
    result = np.ones(4, dtype=np.float32)
    available = positives > 0
    result[available] = np.minimum(negatives[available] / positives[available], max_weight).astype(
        np.float32
    )
    return result


def _process_rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except (ImportError, OSError):
        return None


def benchmark_inner(
    source: str | Path,
    output: str | Path,
    *,
    condition_id: str = "A+H",
    seed: int = 13,
    spec: ThesisBackendSpec | None = None,
) -> dict[str, object]:
    """Time one predeclared inner fit without touching an outer-test group."""

    if condition_id not in CONDITION_FEATURE_SETS:
        raise ThesisRunnerError("benchmark condition must be A, H, or A+H")
    if seed not in FROZEN_SEEDS:
        raise ThesisRunnerError(f"benchmark seed must be one of {FROZEN_SEEDS}")
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite benchmark directory: {root}")
    selected_spec = spec or ThesisBackendSpec()
    frame, _, authority = _load_authority(source, condition_id)
    masked = apply_feature_set(frame, CONDITION_FEATURE_SETS[condition_id])
    sessions = tuple(sessions_from_dataframe(masked))
    # This is outer_01/inner_01 from the outcome-blind frozen group order:
    # train 05_14+05_15, validate 03_26, and never touch outer-test 03_20.
    train = tuple(session for session in sessions if session.group_id in {"05_14", "05_15"})
    validation = tuple(session for session in sessions if session.group_id == "03_26")
    touches_outer = any(session.group_id == "03_20" for session in (*train, *validation))
    if not train or not validation or touches_outer:
        raise ThesisRunnerError("benchmark partition differs from outer_01/inner_01")
    normalizer = FeatureNormalizer.fit(list(train))
    pos_weight = _fit_pos_weight(train)
    binding = ThesisRunBinding.benchmark(selected_spec)
    backend = ThesisTorchBackend(selected_spec, authority, binding)
    root.mkdir(parents=True)
    inner_dir = root / "inner_01"
    inner_dir.mkdir()
    request = InnerFitRequest(
        condition_id=condition_id,
        feature_set=CONDITION_FEATURE_SETS[condition_id],
        encoder=selected_spec.encoder,
        outer_fold_id="outer_01",
        inner_fold_id="inner_01",
        seed=seed,
        max_epochs=1,
        train_sessions=train,
        validation_sessions=validation,
        normalizer=normalizer,
        pos_weight=pos_weight,
        output_dir=inner_dir,
    )
    rss_before = _process_rss_mb()
    tracemalloc.start()
    started = time.perf_counter()
    backend.fit_inner(request)
    elapsed = time.perf_counter() - started
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _process_rss_mb()
    payload = {
        "artifact_kind": "vap_thesis_timing_benchmark",
        "format_version": 1,
        "result_role": "timing_only_not_a_model_result",
        "data_kind": "real",
        "counted_eligible": False,
        "included_session_ids": list(EXPECTED_PRIMARY_SESSIONS),
        "group_ids": list(EXPECTED_GROUP_IDS),
        "corpus_exclusions": CORPUS_EXCLUSIONS,
        "condition_id": condition_id,
        "outer_test_group_excluded": "03_20",
        "train_group_ids": ["05_14", "05_15"],
        "inner_validation_group_ids": ["03_26"],
        "epochs": 1,
        "elapsed_seconds": elapsed,
        "python_peak_memory_mb": python_peak / (1024 * 1024),
        "process_rss_before_mb": rss_before,
        "process_rss_after_mb": rss_after,
    }
    _write_json_exclusive(root / "benchmark_summary.json", payload)
    return payload


def benchmark_console_payload(payload: dict[str, object]) -> dict[str, object]:
    """Return only duration/memory values; scientific scores are never printed."""

    return {
        "elapsed_seconds": payload["elapsed_seconds"],
        "python_peak_memory_mb": payload["python_peak_memory_mb"],
        "process_rss_before_mb": payload["process_rss_before_mb"],
        "process_rss_after_mb": payload["process_rss_after_mb"],
    }


def _unit_directory(root: Path, condition_id: str, fold_id: str) -> Path:
    return root / SAFE_CONDITION_DIR[condition_id] / fold_id


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_matrix_index(
    root: Path,
    index: dict[str, object],
) -> dict[tuple[str, str], dict[str, object]]:
    required = {
        "artifact_kind",
        "format_version",
        "result_role",
        "data_kind",
        "production_eligible",
        "counted_eligible",
        "analysis_variant",
        "seed",
        "condition_ids",
        "outer_fold_ids",
        "expected_unit_count",
        "completed_unit_count",
        "protocol_directory",
        "run_manifest_sha256",
        "units",
        "included_session_ids",
        "group_ids",
        "corpus_exclusions",
    }
    if set(index) != required:
        raise ThesisRunnerError("matrix index fields are not exact")
    if (
        index["artifact_kind"] != "vap_thesis_primary_matrix_index"
        or index["format_version"] != 1
        or isinstance(index["format_version"], bool)
        or index["result_role"] != RESULT_ROLE
        or index["data_kind"] != "real"
        or index["production_eligible"] is not False
        or index["counted_eligible"] is not False
        or index["analysis_variant"] != "primary"
        or index["condition_ids"] != list(CONDITION_FEATURE_SETS)
        or index["outer_fold_ids"] != list(OUTER_FOLD_IDS)
        or index["expected_unit_count"] != 12
        or index["completed_unit_count"] != 12
        or index["included_session_ids"] != list(EXPECTED_PRIMARY_SESSIONS)
        or index["group_ids"] != list(EXPECTED_GROUP_IDS)
        or index["corpus_exclusions"] != CORPUS_EXCLUSIONS
        or not _is_sha256(index["run_manifest_sha256"])
    ):
        raise ThesisRunnerError("matrix index identity is invalid or attempted promotion")
    seed = index["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in FROZEN_SEEDS:
        raise ThesisRunnerError("matrix index seed is invalid")
    units = index["units"]
    if not isinstance(units, list) or len(units) != 12:
        raise ThesisRunnerError("matrix index must bind exactly twelve units")
    expected_pairs = [
        (condition.condition_id, fold_id) for condition in CONDITIONS for fold_id in OUTER_FOLD_IDS
    ]
    result: dict[tuple[str, str], dict[str, object]] = {}
    for record, (condition_id, fold_id) in zip(units, expected_pairs, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "condition_id",
            "outer_fold_id",
            "seed",
            "result_manifest",
            "result_manifest_sha256",
            "elapsed_seconds",
        }:
            raise ThesisRunnerError("matrix unit record fields are not exact")
        expected_result = (
            _unit_directory(root, condition_id, fold_id) / "result_manifest.json"
        ).resolve()
        elapsed = record["elapsed_seconds"]
        if (
            record["condition_id"] != condition_id
            or record["outer_fold_id"] != fold_id
            or record["seed"] != seed
            or record["result_manifest"] != str(expected_result)
            or not _is_sha256(record["result_manifest_sha256"])
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise ThesisRunnerError("matrix unit record identity is invalid")
        if not expected_result.is_file():
            raise ThesisRunnerError("matrix unit result manifest is missing")
        if _sha256_file(expected_result) != record["result_manifest_sha256"]:
            raise ThesisRunnerError("matrix unit result manifest hash mismatch")
        result[(condition_id, fold_id)] = record
    return result


def run_matrix(
    protocol_directory: str | Path,
    source: str | Path,
    output: str | Path,
    *,
    seed: int,
    spec: ThesisBackendSpec | None = None,
) -> Path:
    """Execute exactly primary x one seed x A/H/A+H x four outer folds."""

    if seed not in FROZEN_SEEDS:
        raise ThesisRunnerError(f"seed must be one of {FROZEN_SEEDS}")
    selected_spec = spec or ThesisBackendSpec()
    _, run = _load_protocol(protocol_directory)
    binding = ThesisRunBinding.from_run_manifest(run, selected_spec)
    source_path = Path(source).resolve()
    if str(source_path) != run.get("source"):
        raise ThesisRunnerError("execution source differs from frozen protocol")
    root = Path(output).resolve()
    planned = [
        _unit_directory(root, condition.condition_id, fold_id)
        for condition in CONDITIONS
        for fold_id in OUTER_FOLD_IDS
    ]
    # Whole-matrix preflight occurs before any fit.  Partial reruns require a new
    # output root; this command never deletes, renames, or auto-suffixes results.
    if root.exists() or any(path.exists() for path in planned):
        raise FileExistsError(f"refusing to overwrite or resume matrix: {root}")
    prepared_conditions = {}
    authorities = {}
    for condition in CONDITIONS:
        prepared = prepare_outer_condition(
            protocol_directory,
            source=source_path,
            condition_id=condition.condition_id,
        )
        frame, report = prepared_outer_authority_inputs(prepared)
        if (
            tuple(sorted(report.session_ids)) != EXPECTED_PRIMARY_SESSIONS
            or tuple(sorted(report.group_ids)) != EXPECTED_GROUP_IDS
            or report.n_rows != DEADLINE_EXPECTED_N_ROWS
            or not report.target_ready
            or report.selected_feature_ready is not True
        ):
            raise ThesisRunnerError(
                f"prepared condition {condition.condition_id} is not the exact "
                "train-ready 64,299-row/eight-scene/four-group deadline corpus"
            )
        prepared_conditions[condition.condition_id] = prepared
        authorities[condition.condition_id] = build_real_session_authority(
            frame,
            report.to_dict(),
            condition_id=condition.condition_id,
        )
    unit_records = []
    for condition in CONDITIONS:
        for fold_id in OUTER_FOLD_IDS:
            unit_path = _unit_directory(root, condition.condition_id, fold_id)
            backend = ThesisTorchBackend(
                selected_spec,
                authorities[condition.condition_id],
                binding,
            )
            started = time.perf_counter()
            result = run_outer_unit(
                protocol_directory,
                unit_path,
                source=source_path,
                analysis_variant="primary",
                condition_id=condition.condition_id,
                outer_fold_id=fold_id,
                seed=seed,
                backend=backend,
                prepared=prepared_conditions[condition.condition_id],
            )
            unit_records.append(
                {
                    "condition_id": condition.condition_id,
                    "outer_fold_id": fold_id,
                    "seed": seed,
                    "result_manifest": str(result.resolve()),
                    "result_manifest_sha256": _sha256_file(result),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    index_path = root / "matrix_index.json"
    _write_json_exclusive(
        index_path,
        {
            "artifact_kind": "vap_thesis_primary_matrix_index",
            "format_version": 1,
            "result_role": RESULT_ROLE,
            "data_kind": "real",
            "production_eligible": False,
            "counted_eligible": False,
            "analysis_variant": "primary",
            "seed": seed,
            "condition_ids": list(CONDITION_FEATURE_SETS),
            "outer_fold_ids": list(OUTER_FOLD_IDS),
            "expected_unit_count": 12,
            "completed_unit_count": len(unit_records),
            "protocol_directory": str(Path(protocol_directory).resolve()),
            "run_manifest_sha256": binding.run_manifest_sha256,
            "included_session_ids": list(EXPECTED_PRIMARY_SESSIONS),
            "group_ids": list(EXPECTED_GROUP_IDS),
            "corpus_exclusions": CORPUS_EXCLUSIONS,
            "units": unit_records,
        },
    )
    return index_path


def _bound_artifact_path(
    unit_directory: Path,
    bindings: dict[str, object],
    role: str,
) -> Path:
    binding = bindings.get(role)
    if not isinstance(binding, dict) or set(binding) != {
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        raise ThesisRunnerError(f"outer {role} binding is malformed")
    relative = binding["relative_path"]
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ThesisRunnerError(f"outer {role} relative path is invalid")
    path = (unit_directory / relative).resolve()
    try:
        path.relative_to(unit_directory.resolve())
    except ValueError as exc:
        raise ThesisRunnerError(f"outer {role} path escapes its unit") from exc
    size = binding["size_bytes"]
    if (
        not path.is_file()
        or not _is_sha256(binding["sha256"])
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or path.stat().st_size != size
        or _sha256_file(path) != binding["sha256"]
    ):
        raise ThesisRunnerError(f"outer {role} bytes differ from result manifest")
    return path


def _prediction_path_from_result(
    unit_directory: Path,
    *,
    condition_id: str,
    fold_id: str,
    seed: int,
    run_manifest_sha256: str,
) -> Path:
    result = _read_json(unit_directory / "result_manifest.json")
    expected_registration = {
        "backend_id": "backend_contract_smoke",
        "backend_version": 1,
        "result_role": "backend_contract_smoke",
        "counted_eligible": False,
    }
    if (
        result.get("result_role") != "backend_contract_smoke"
        or result.get("execution_complete") is not True
        or result.get("counted_experiment_complete") is not False
        or result.get("analysis_variant") != "primary"
        or result.get("condition_id") != condition_id
        or result.get("feature_set") != CONDITION_FEATURE_SETS[condition_id]
        or result.get("outer_fold_id") != fold_id
        or result.get("seed") != seed
        or result.get("outer_evaluation_count") != 1
        or isinstance(result.get("outer_evaluation_count"), bool)
        or result.get("run_manifest_sha256") != run_manifest_sha256
        or result.get("backend_registration") != expected_registration
        or result.get("corpus_exclusions") != CORPUS_EXCLUSIONS
    ):
        raise ThesisRunnerError("outer result identity is not the expected non-counted unit")
    bindings = result.get("artifact_bindings")
    if not isinstance(bindings, dict):
        raise ThesisRunnerError("outer artifact bindings are malformed")
    prediction_path = _bound_artifact_path(unit_directory, bindings, "outer_predictions")
    metrics_path = _bound_artifact_path(unit_directory, bindings, "outer_metrics")
    prediction = load_prediction_artifact(prediction_path)
    metrics = load_metrics_artifact(
        metrics_path,
        prediction_path=prediction_path,
    )
    if (
        prediction["condition_id"] != condition_id
        or prediction["outer_fold_id"] != fold_id
        or prediction["seed"] != seed
        or prediction["run_manifest_sha256"] != run_manifest_sha256
        or metrics["condition_id"] != condition_id
        or metrics["outer_fold_id"] != fold_id
        or metrics["seed"] != seed
        or metrics["run_manifest_sha256"] != run_manifest_sha256
        or metrics["prediction_sha256"] != _sha256_file(prediction_path)
        or metrics["checkpoint_sha256"] != result.get("checkpoint_sha256")
        or prediction["checkpoint_sha256"] != result.get("checkpoint_sha256")
        or [record["session_id"] for record in prediction["sessions"]]
        != result.get("outer_test_session_ids")
        or any(
            DEADLINE_SESSION_TO_GROUP.get(record["session_id"]) != record["group_id"]
            for record in prediction["sessions"]
        )
        or metrics["real_session_authority_sha256"] != prediction["real_session_authority_sha256"]
        or metrics["source_manifest_sha256"] != prediction["source_manifest_sha256"]
    ):
        raise ThesisRunnerError("backend artifact semantics differ from the outer result")
    return prediction_path


def _metric_value(payload: dict[str, object], path: tuple[str, ...]) -> float | None:
    value: object = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def aggregate_matrix(matrix_directory: str | Path, output: str | Path) -> Path:
    """Pool the twelve saved prediction artifacts; no model is loaded or run."""

    root = Path(matrix_directory).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {output_path}")
    index = _read_json(root / "matrix_index.json")
    _validate_matrix_index(root, index)
    seed = index["seed"]
    result_files = sorted(root.glob("*/*/result_manifest.json"))
    expected_result_files = {
        (_unit_directory(root, condition.condition_id, fold_id) / "result_manifest.json").resolve()
        for condition in CONDITIONS
        for fold_id in OUTER_FOLD_IDS
    }
    if {path.resolve() for path in result_files} != expected_result_files:
        raise ThesisRunnerError("matrix directory must contain exactly twelve result manifests")
    summaries: dict[str, dict[str, object]] = {}
    for condition in CONDITIONS:
        paths = tuple(
            _prediction_path_from_result(
                _unit_directory(root, condition.condition_id, fold_id),
                condition_id=condition.condition_id,
                fold_id=fold_id,
                seed=seed,
                run_manifest_sha256=index["run_manifest_sha256"],
            )
            for fold_id in OUTER_FOLD_IDS
        )
        summary = aggregate_outer_prediction_artifacts(paths)
        if summary["seed"] != seed or summary["condition_id"] != condition.condition_id:
            raise ThesisRunnerError("aggregate identity differs from matrix index")
        if tuple(summary["session_ids"]) != EXPECTED_PRIMARY_SESSIONS:
            raise ThesisRunnerError("aggregate does not cover the exact eight-scene corpus")
        if summary["run_manifest_sha256"] != index["run_manifest_sha256"]:
            raise ThesisRunnerError("aggregate run identity differs from matrix index")
        summaries[condition.condition_id] = summary
    paired_fold_sessions = {
        json.dumps(summary["fold_session_ids"], sort_keys=True) for summary in summaries.values()
    }
    paired_fold_groups = {
        json.dumps(summary["fold_group_ids"], sort_keys=True) for summary in summaries.values()
    }
    if len(paired_fold_sessions) != 1 or len(paired_fold_groups) != 1:
        raise ThesisRunnerError("A/H/A+H do not use identical outer fold/session assignments")
    comparisons = {}
    for left, right in (("A", "A+H"), ("H", "A+H")):
        left_map = _metric_value(summaries[left], ("pooled", "frame", "mAP"))
        right_map = _metric_value(summaries[right], ("pooled", "frame", "mAP"))
        comparisons[f"{right}_minus_{left}"] = {
            "pooled_frame_mAP_difference": (
                None if left_map is None or right_map is None else right_map - left_map
            ),
            "interpretation": "descriptive_paired_difference_not_a_significance_test",
        }
    payload = {
        "artifact_kind": "vap_thesis_primary_matrix_aggregate",
        "format_version": 1,
        "result_role": RESULT_ROLE,
        "data_kind": "real",
        "production_eligible": False,
        "counted_eligible": False,
        "analysis_variant": "primary",
        "included_session_ids": list(EXPECTED_PRIMARY_SESSIONS),
        "group_ids": list(EXPECTED_GROUP_IDS),
        "corpus_exclusions": CORPUS_EXCLUSIONS,
        "seed": seed,
        "aggregation_source": "saved_outer_probabilities_only_no_model_inference",
        "threshold_policy": "diagnostic_fixed_0.5_not_calibrated",
        "run_manifest_sha256": index["run_manifest_sha256"],
        "paired_fold_session_ids": summaries["A"]["fold_session_ids"],
        "paired_fold_group_ids": summaries["A"]["fold_group_ids"],
        "conditions": summaries,
        "paired_comparisons": comparisons,
    }
    _write_json_exclusive(output_path, payload)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze the no-clobber real outer protocol")
    freeze.add_argument("--data", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--max-epochs", required=True, type=int)

    benchmark = subparsers.add_parser(
        "benchmark-inner", help="time one inner fit without outer-test access"
    )
    benchmark.add_argument("--data", required=True, type=Path)
    benchmark.add_argument("--output", required=True, type=Path)
    benchmark.add_argument("--condition", choices=tuple(CONDITION_FEATURE_SETS), default="A+H")
    benchmark.add_argument("--seed", choices=FROZEN_SEEDS, type=int, default=13)

    run = subparsers.add_parser("run-matrix", help="run primary A/H/A+H x four folds")
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--data", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--seed", choices=FROZEN_SEEDS, type=int, required=True)

    aggregate = subparsers.add_parser(
        "aggregate", help="pool saved outer probabilities without inference"
    )
    aggregate.add_argument("--run-dir", required=True, type=Path)
    aggregate.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        path = freeze_protocol(args.data, args.output, max_epochs=args.max_epochs)
        print(json.dumps({"frozen_protocol": str(path)}, allow_nan=False))
    elif args.command == "benchmark-inner":
        payload = benchmark_inner(
            args.data,
            args.output,
            condition_id=args.condition,
            seed=args.seed,
        )
        print(
            json.dumps(
                benchmark_console_payload(payload),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    elif args.command == "run-matrix":
        path = run_matrix(
            args.protocol,
            args.data,
            args.output,
            seed=args.seed,
        )
        print(json.dumps({"matrix_index": str(path)}, allow_nan=False))
    else:
        path = aggregate_matrix(args.run_dir, args.output)
        print(json.dumps({"aggregate": str(path)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
