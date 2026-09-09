"""Fail-closed orchestration for the counted four-group outer experiment.

This module deliberately does not replace :func:`train_vap_model`.  The latter
remains a development/inner-validation runner.  Here, a small backend protocol
separates the frozen split and artifact state machine from the concrete PyTorch
training implementation.  That lets the leakage rules be tested quickly before
any real-data fit is allowed.

The only currently registered backend is a code-owned contract-smoke backend,
so every executable unit is explicitly non-counted.  A future production
backend must be added to the source-controlled registry and pass the remaining
environment/run-index gates before any sealed unit may receive
``result_role=counted_outer_test``.  A protocol bundle, an in-progress staging
directory, a checkpoint, or an inner-validation output never receives that role.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat as stat_module
import statistics
import uuid
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from hri_turn_taking.data.assembly import ReadinessReport, load_and_audit_vap_source
from hri_turn_taking.vap.dataset import FeatureNormalizer, SessionTensors, sessions_from_dataframe
from hri_turn_taking.vap.feature_sets import (
    active_feature_columns,
    apply_feature_set,
    get_feature_set,
)
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG, VAPLabelConfig

PROTOCOL_ID = "four_group_nested_logo_v2"
DEADLINE_PROTOCOL_ID = "four_group_nested_logo_deadline_8_v1"
SPLIT_MANIFEST_KIND = "vap_counted_outer_split"
RUN_MANIFEST_KIND = "vap_counted_outer_run"
RESULT_MANIFEST_KIND = "vap_counted_outer_result"
MANIFEST_FORMAT_VERSION = 2
RESULT_FORMAT_VERSION = 2
DEADLINE_MANIFEST_FORMAT_VERSION = 3
FROZEN_SEEDS = (13, 42, 73)
FROZEN_GROUP_COUNT = 4
FROZEN_AUX_VAD_WEIGHT = 0.0

PRIMARY_D3_SESSION_IDS = (
    "03_20__4",
    "03_26__3",
    "05_14__2",
    "05_14__3",
    "05_14__4",
    "05_15__2",
    "05_15__3",
    "05_15__4",
    "05_15__5",
    "05_15__6",
)
PRIMARY_SESSION_TO_GROUP = {
    "03_20__4": "03_20",
    "03_26__3": "03_26",
    "05_14__2": "05_14",
    "05_14__3": "05_14",
    "05_14__4": "05_14",
    "05_15__2": "05_15",
    "05_15__3": "05_15",
    "05_15__4": "05_15",
    "05_15__5": "05_15",
    "05_15__6": "05_15",
}
DEADLINE_CORPUS_EXCLUSIONS = {
    "05_15__4": "no A speech in Transcript; pending owner manual review",
    "05_15__5": "A/R extremely sparse; pending owner manual review",
}
DEADLINE_SESSION_TO_GROUP = {
    session_id: group_id
    for session_id, group_id in PRIMARY_SESSION_TO_GROUP.items()
    if session_id not in DEADLINE_CORPUS_EXCLUSIONS
}
DEADLINE_EXPECTED_N_ROWS = 64_299
PAIRED_D3_80MS_SESSION_IDS = (
    "03_20__4",
    "03_26__3",
    "05_14__2",
    "05_15__2",
    "05_15__3",
    "05_15__5",
)
D3_40MS_DESCRIPTIVE_SESSION_IDS = (
    "03_20__4",
    "03_26__3",
    "05_15__2",
)
D3_FROZEN_VALUES = {
    "03_20__4": {
        "offset_s": -6.7334,
        "status": "ok",
        "self_check_ok": True,
        "error_bound_ms": 33.5,
    },
    "03_26__3": {
        "offset_s": -1.5665,
        "status": "ok",
        "self_check_ok": True,
        "error_bound_ms": 33.5,
    },
    "05_14__2": {
        "offset_s": -5.9873,
        "status": "degraded",
        "self_check_ok": True,
        "error_bound_ms": 69.6,
    },
    "05_14__3": {
        "offset_s": -5.0551,
        "status": "degraded",
        "self_check_ok": True,
        "error_bound_ms": 101.3,
    },
    "05_14__4": {
        "offset_s": -4.8509,
        "status": "degraded",
        "self_check_ok": True,
        "error_bound_ms": 104.4,
    },
    "05_15__2": {
        "offset_s": -3.7621,
        "status": "degraded",
        "self_check_ok": True,
        "error_bound_ms": 36.2,
    },
    "05_15__3": {
        "offset_s": -6.5999,
        "status": "degraded",
        "self_check_ok": True,
        "error_bound_ms": 56.4,
    },
    "05_15__4": {
        "offset_s": -6.7,
        "status": "degraded",
        "self_check_ok": True,
        "error_bound_ms": 87.2,
    },
    "05_15__5": {
        "offset_s": -10.7654,
        "status": "degraded",
        "self_check_ok": True,
        "error_bound_ms": 43.7,
    },
    "05_15__6": {
        "offset_s": -4.864,
        "status": "degraded",
        "self_check_ok": True,
        "error_bound_ms": 101.3,
    },
}
ANALYSIS_VARIANTS = ("primary", "d3_le80")
INNER_CHECKPOINT_FILENAME = "inner_checkpoint.artifact"
INNER_OOF_FILENAME = "inner_oof_predictions.artifact"
CALIBRATION_FILENAME = "calibration.artifact"
FINAL_CHECKPOINT_FILENAME = "final_checkpoint.artifact"
OUTER_PREDICTIONS_FILENAME = "outer_predictions.artifact"
OUTER_METRICS_FILENAME = "outer_metrics.artifact"


class OuterExperimentGateError(ValueError):
    """A formal outer-test integrity or protocol gate failed."""


@dataclass(frozen=True)
class _CorpusProfile:
    protocol_id: str
    format_version: int
    session_to_group: tuple[tuple[str, str], ...]
    corpus_exclusions: tuple[tuple[str, str], ...]
    d3_decision_status: str
    expected_n_rows: int | None

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self.session_to_group)

    @property
    def exclusions(self) -> dict[str, str]:
        return dict(self.corpus_exclusions)

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(self.mapping)


_PRIMARY_CORPUS_PROFILE = _CorpusProfile(
    protocol_id=PROTOCOL_ID,
    format_version=MANIFEST_FORMAT_VERSION,
    session_to_group=tuple(PRIMARY_SESSION_TO_GROUP.items()),
    corpus_exclusions=(),
    d3_decision_status="frozen_before_outcome_inspection",
    expected_n_rows=None,
)
_DEADLINE_CORPUS_PROFILE = _CorpusProfile(
    protocol_id=DEADLINE_PROTOCOL_ID,
    format_version=DEADLINE_MANIFEST_FORMAT_VERSION,
    session_to_group=tuple(DEADLINE_SESSION_TO_GROUP.items()),
    corpus_exclusions=tuple(DEADLINE_CORPUS_EXCLUSIONS.items()),
    d3_decision_status="owner_revised_after_target_qc_before_model_outcome_inspection",
    expected_n_rows=DEADLINE_EXPECTED_N_ROWS,
)


class BackendContractSmokeBackend:
    """Marker for code-path tests; this class can never issue a counted result."""


@dataclass(frozen=True)
class _BackendRegistration:
    backend_type: type
    backend_id: str
    backend_version: int
    result_role: str
    counted_eligible: bool


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    sha256: str
    mode: int
    size_bytes: int
    device: int
    inode: int
    link_count: int
    mtime_ns: int
    ctime_ns: int
    change_time_token: int
    file_attributes: int


@dataclass(frozen=True)
class _OpenHandleState:
    mode: int
    device: int
    inode: int
    link_count: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    change_time_token: int
    file_attributes: int


@dataclass(frozen=True)
class _HeldSnapshotHandle:
    path: Path
    fd: int
    opening_state: _OpenHandleState
    ancestor_guard: tuple[tuple[Path, int, int, int, int, int], ...]
    expected: _FileSnapshot | None
    require_single_link: bool


@dataclass(frozen=True)
class _ResultValidationSnapshot:
    immutable_files: tuple[_FileSnapshot, ...]
    role_artifacts: tuple[_FileSnapshot, ...]
    readiness: ReadinessReport
    d3_records: dict[str, dict[str, object]]


_SMOKE_BACKEND_REGISTRATION = _BackendRegistration(
    backend_type=BackendContractSmokeBackend,
    backend_id="backend_contract_smoke",
    backend_version=1,
    result_role="backend_contract_smoke",
    counted_eligible=False,
)
# A future production backend must be added here by reviewed source-code change.
# Runtime objects cannot register themselves or self-assert counted eligibility.
_PRODUCTION_BACKEND_REGISTRY: tuple[_BackendRegistration, ...] = ()


@dataclass(frozen=True)
class OuterCondition:
    """One paired model condition in the frozen protocol."""

    condition_id: str
    feature_set: str
    encoder: str = "transformer"


@dataclass(frozen=True)
class InnerFitRequest:
    condition_id: str
    feature_set: str
    encoder: str
    outer_fold_id: str
    inner_fold_id: str
    seed: int
    max_epochs: int
    train_sessions: tuple[SessionTensors, ...]
    validation_sessions: tuple[SessionTensors, ...]
    normalizer: FeatureNormalizer
    pos_weight: np.ndarray
    output_dir: Path


@dataclass(frozen=True)
class InnerFitReceipt:
    condition_id: str
    outer_fold_id: str
    inner_fold_id: str
    seed: int
    best_epoch: int
    train_session_ids: tuple[str, ...]
    validation_session_ids: tuple[str, ...]
    normalizer_fit_session_ids: tuple[str, ...]
    pos_weight_fit_session_ids: tuple[str, ...]
    checkpoint_path: Path
    oof_predictions_path: Path


@dataclass(frozen=True)
class CalibrationRequest:
    condition_id: str
    outer_fold_id: str
    seed: int
    inner_fold_ids: tuple[str, ...]
    oof_prediction_paths: tuple[Path, ...]
    oof_session_ids: tuple[str, ...]
    output_dir: Path


@dataclass(frozen=True)
class CalibrationReceipt:
    condition_id: str
    outer_fold_id: str
    seed: int
    inner_fold_ids: tuple[str, ...]
    oof_prediction_paths: tuple[Path, ...]
    calibration_path: Path


@dataclass(frozen=True)
class FinalFitRequest:
    condition_id: str
    feature_set: str
    encoder: str
    outer_fold_id: str
    seed: int
    fixed_epochs: int
    train_sessions: tuple[SessionTensors, ...]
    normalizer: FeatureNormalizer
    pos_weight: np.ndarray
    calibration_path: Path
    output_dir: Path


@dataclass(frozen=True)
class FinalFitReceipt:
    condition_id: str
    outer_fold_id: str
    seed: int
    fixed_epochs: int
    train_session_ids: tuple[str, ...]
    normalizer_fit_session_ids: tuple[str, ...]
    pos_weight_fit_session_ids: tuple[str, ...]
    initialized_from_scratch: bool
    checkpoint_path: Path


@dataclass(frozen=True)
class OuterEvaluationRequest:
    condition_id: str
    outer_fold_id: str
    seed: int
    checkpoint_path: Path
    checkpoint_sha256: str
    calibration_path: Path
    test_sessions: tuple[SessionTensors, ...]
    output_dir: Path


@dataclass(frozen=True)
class OuterEvaluationReceipt:
    condition_id: str
    outer_fold_id: str
    seed: int
    checkpoint_sha256: str
    evaluation_session_ids: tuple[str, ...]
    evaluation_count: int
    predictions_path: Path
    metrics_path: Path


class OuterTrainingBackend(Protocol):
    """Trusted training code behind the independently testable split gate."""

    def fit_inner(self, request: InnerFitRequest) -> InnerFitReceipt: ...

    def calibrate_inner_oof(self, request: CalibrationRequest) -> CalibrationReceipt: ...

    def fit_final(self, request: FinalFitRequest) -> FinalFitReceipt: ...

    def evaluate_outer(self, request: OuterEvaluationRequest) -> OuterEvaluationReceipt: ...


@dataclass(frozen=True)
class OuterProtocolBundle:
    directory: Path
    split_manifest_path: Path
    run_manifest_path: Path
    split_manifest_sha256: str
    run_manifest_sha256: str


@dataclass(frozen=True, init=False, eq=False)
class PreparedOuterCondition:
    """One process-local, factory-sealed audited condition for deadline reuse.

    The object is intentionally not serialisable authority. It only avoids
    repeating the expensive deterministic source replay inside one trusted
    single-writer ``run-matrix`` process; every unit still re-hashes the frozen
    source bindings and runs the normal stage/final artifact validators.
    """

    protocol_directory: Path
    source: Path
    condition: OuterCondition
    split_manifest_sha256: str
    run_manifest_sha256: str
    readiness_sha256: str
    d3_records_sha256: str
    sessions_sha256: str
    frame_df: object = field(repr=False)
    readiness: ReadinessReport = field(repr=False)
    sessions: tuple[SessionTensors, ...] = field(repr=False)
    _seal: object = field(repr=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("PreparedOuterCondition must be created by prepare_outer_condition")


_PREPARED_OUTER_CONDITION_SEAL = object()
_PREPARED_OUTER_CONDITIONS: weakref.WeakValueDictionary[int, PreparedOuterCondition] = (
    weakref.WeakValueDictionary()
)


def freeze_outer_protocol(
    source: str | Path,
    destination: str | Path,
    *,
    conditions: tuple[OuterCondition, ...],
    max_epochs: int,
    trainer_config: dict[str, object],
    model_config: dict[str, object],
    production_backend: dict[str, object] | None = None,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    excluded_session_ids: tuple[str, ...] = (),
) -> OuterProtocolBundle:
    """Audit real data and atomically publish the outcome-blind formal protocol.

    This function only freezes inputs and partitions.  It performs no fitting and
    cannot create a counted result.  Every condition must see the exact same source
    snapshot and exact same sessions.
    """
    checked_conditions = _validate_conditions(conditions)
    checked_max_epochs = _exact_positive_int(max_epochs, "max_epochs")
    checked_exclusions = _checked_excluded_session_ids(excluded_session_ids)
    source_path = Path(source).resolve()

    reference_frame = None
    reference_report = None
    readiness_by_condition: dict[str, dict[str, object]] = {}
    for condition in checked_conditions:
        frame_df, report = load_and_audit_vap_source(
            source_path,
            data_kind="real",
            feature_set=condition.feature_set,
            excluded_session_ids=checked_exclusions,
        )
        _require_ready(report, condition)
        if reference_report is None:
            reference_frame = frame_df
            reference_report = report
        else:
            if _source_snapshot(report) != _source_snapshot(reference_report):
                raise OuterExperimentGateError(
                    "all conditions must bind the exact same audited source snapshot"
                )
            if not frame_df.equals(reference_frame):
                raise OuterExperimentGateError(
                    "source dataframe changed while the paired protocol was being frozen"
                )
        readiness_by_condition[condition.condition_id] = _readiness_binding(report)

    if reference_frame is None or reference_report is None:
        raise OuterExperimentGateError("at least one condition is required")
    session_to_group = _session_group_map(reference_frame)
    source_bindings = _bindings_from_report(reference_report)
    d3_records = _d3_records_from_report(reference_report)
    split_manifest = build_outer_split_manifest(
        session_to_group,
        source_bindings=source_bindings,
        conditions=checked_conditions,
        d3_records=d3_records,
    )
    run_manifest = build_outer_run_manifest(
        split_manifest,
        source_path=source_path,
        max_epochs=checked_max_epochs,
        trainer_config=trainer_config,
        model_config=model_config,
        production_backend=production_backend,
        label_config=label_config,
        readiness_by_condition=readiness_by_condition,
    )
    return publish_outer_protocol(destination, split_manifest, run_manifest)


def build_outer_split_manifest(
    session_to_group: dict[str, str],
    *,
    source_bindings: list[dict[str, object]],
    conditions: tuple[OuterCondition, ...],
    d3_records: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Build both executable variants of the frozen nested LOGO partition."""
    checked_conditions = _validate_conditions(conditions)
    mapping = _validate_session_group_map(session_to_group)
    profile = _corpus_profile_from_mapping(mapping)
    checked_d3 = _validate_d3_records(d3_records, source_bindings, profile=profile)
    variant_mappings = {
        "primary": dict(mapping),
        "d3_le80": {
            session_id: mapping[session_id]
            for session_id in PAIRED_D3_80MS_SESSION_IDS
            if session_id in mapping
        },
    }
    variants = {
        name: {
            "analysis_variant": name,
            "session_to_group": dict(sorted(variant_mapping.items())),
            "group_count": FROZEN_GROUP_COUNT,
            "outer_folds": _build_logo_folds(variant_mapping),
        }
        for name, variant_mapping in variant_mappings.items()
    }

    manifest: dict[str, object] = {
        "artifact_kind": SPLIT_MANIFEST_KIND,
        "format_version": profile.format_version,
        "protocol_id": profile.protocol_id,
        "result_role": "protocol_only_not_a_result",
        "source_session_to_group": dict(sorted(mapping.items())),
        "source_bindings": _validate_source_bindings_payload(source_bindings),
        "d3_records": checked_d3,
        "condition_ids": [condition.condition_id for condition in checked_conditions],
        "conditions": [asdict(condition) for condition in checked_conditions],
        "seeds": list(FROZEN_SEEDS),
        "aux_vad_weight": FROZEN_AUX_VAD_WEIGHT,
        "selection": {
            "inner_scheme": "leave_one_group_out_over_three_outer_train_groups",
            "best_epoch_indexing": "one_based",
            "fixed_epoch_rule": "median_of_three_inner_best_epochs",
            "threshold_and_calibration_source": "pooled_inner_oof_only",
            "final_fit": "same_seed_from_scratch_on_all_outer_train_groups",
            "outer_evaluation": "once_after_sealed_final_checkpoint",
        },
        "d3_sensitivity": _d3_sensitivity_payload(checked_d3, checked_conditions, profile=profile),
        "analysis_variants": variants,
    }
    if profile.corpus_exclusions:
        manifest["corpus_exclusions"] = profile.exclusions
    validate_outer_split_manifest(manifest)
    return manifest


def build_outer_run_manifest(
    split_manifest: dict[str, object],
    *,
    source_path: str | Path,
    max_epochs: int,
    trainer_config: dict[str, object],
    model_config: dict[str, object],
    production_backend: dict[str, object] | None = None,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    readiness_by_condition: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build the immutable training configuration paired with a split manifest."""
    validate_outer_split_manifest(split_manifest)
    profile = _corpus_profile_from_protocol_id(split_manifest["protocol_id"])
    max_epochs = _exact_positive_int(max_epochs, "max_epochs")
    trainer = _strict_json_mapping(trainer_config, "trainer_config")
    model = _strict_json_mapping(model_config, "model_config")
    source = Path(source_path).resolve()
    backend_policy = _build_backend_policy(production_backend)
    if trainer.get("aux_vad_weight") != FROZEN_AUX_VAD_WEIGHT or isinstance(
        trainer.get("aux_vad_weight"), bool
    ):
        raise OuterExperimentGateError("trainer_config.aux_vad_weight must be exactly 0.0")
    if "random_seed" in trainer:
        raise OuterExperimentGateError(
            "trainer_config must not freeze one random_seed; the protocol supplies all seeds"
        )
    if "epochs" in trainer:
        raise OuterExperimentGateError(
            "trainer_config must not contain epochs; max_epochs and selected fixed epoch are sealed"
        )
    readiness = _strict_json_mapping(readiness_by_condition or {}, "readiness_by_condition")
    condition_ids = split_manifest["condition_ids"]
    if set(readiness) != set(condition_ids):
        raise OuterExperimentGateError("readiness bindings must cover every condition exactly")
    _validate_readiness_bindings(readiness, split_manifest)
    payload: dict[str, object] = {
        "artifact_kind": RUN_MANIFEST_KIND,
        "format_version": profile.format_version,
        "protocol_id": profile.protocol_id,
        "result_role": "protocol_only_not_a_result",
        "counted_experiment_complete": False,
        "split_manifest_sha256": _json_sha256(split_manifest),
        "source": str(source),
        "seeds": list(FROZEN_SEEDS),
        "condition_ids": list(condition_ids),
        "max_epochs": max_epochs,
        "aux_vad_weight": FROZEN_AUX_VAD_WEIGHT,
        "trainer_config": trainer,
        "model_config": model,
        "backend_policy": backend_policy,
        "label_config": _validate_label_config(
            _strict_json_mapping(asdict(label_config), "label_config")
        ),
        "readiness_by_condition": readiness,
        "implementation_status": {
            "protocol_orchestrator": "implemented",
            "production_torch_backend": backend_policy["production_status"],
            "environment_capture": "not_implemented_blocker",
            "run_index": "not_implemented_blocker",
            "pose_time_shift_reassembly": "not_implemented_blocker",
        },
    }
    if profile.corpus_exclusions:
        payload["corpus_exclusions"] = profile.exclusions
    validate_outer_run_manifest(payload, split_manifest)
    return payload


def publish_outer_protocol(
    destination: str | Path,
    split_manifest: dict[str, object],
    run_manifest: dict[str, object],
) -> OuterProtocolBundle:
    """Atomically publish a no-clobber protocol bundle."""
    validate_outer_split_manifest(split_manifest)
    validate_outer_run_manifest(run_manifest, split_manifest)
    _verify_source_bindings(split_manifest["source_bindings"])
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite protocol bundle: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stage = destination_path.with_name(f".{destination_path.name}.{token}.stage")
    try:
        _create_owned_stage(stage, token)
        _write_json(stage / "split_manifest.json", split_manifest)
        _write_json(stage / "run_manifest.json", run_manifest)
        _load_protocol_from_directory(stage)
        os.rename(stage, destination_path)
        try:
            loaded_split, loaded_run = _load_protocol_from_directory(destination_path)
            _verify_source_bindings(loaded_split["source_bindings"])
            return OuterProtocolBundle(
                directory=destination_path,
                split_manifest_path=destination_path / "split_manifest.json",
                run_manifest_path=destination_path / "run_manifest.json",
                split_manifest_sha256=_json_sha256(loaded_split),
                run_manifest_sha256=_json_sha256(loaded_run),
            )
        except BaseException:
            _rollback_owned_path(destination_path, token)
            raise
    except BaseException:
        _remove_call_owned_stage(stage, token)
        _rollback_owned_path(destination_path, token)
        raise
    finally:
        _remove_call_owned_stage(stage, token)


def prepare_outer_condition(
    protocol_directory: str | Path,
    *,
    source: str | Path,
    condition_id: str,
) -> PreparedOuterCondition:
    """Replay and seal one condition once for same-process deadline units.

    This is a process-local optimisation, not a persisted cache. The factory
    binds the exact protocol/run manifests, resolved source, readiness, D3
    records, and condition-masked session tensors. Objects not issued by this
    factory are rejected even if their attributes resemble a prepared object.
    """

    protocol_path = Path(protocol_directory).resolve()
    split_manifest, run_manifest = _load_protocol_from_directory(protocol_path)
    condition = _condition_from_manifest(split_manifest, condition_id)
    resolved_source = Path(source).resolve()
    if str(resolved_source) != run_manifest["source"]:
        raise OuterExperimentGateError("prepared source differs from the frozen run source")
    source_bindings = split_manifest["source_bindings"]
    _verify_source_bindings(source_bindings)
    frame_df, sessions, readiness = _load_condition_material(
        resolved_source,
        condition,
        run_manifest,
    )
    expected_readiness = run_manifest["readiness_by_condition"].get(condition.condition_id)
    readiness_binding = _readiness_binding(readiness)
    if not _type_exact_equal(readiness_binding, expected_readiness):
        raise OuterExperimentGateError("prepared readiness snapshot differs from the frozen run")
    fresh_d3 = _d3_records_from_report(readiness)
    if not _type_exact_equal(fresh_d3, split_manifest["d3_records"]):
        raise OuterExperimentGateError("prepared D3 provenance differs from the frozen split")
    _verify_loaded_sessions(sessions, split_manifest)
    _verify_source_bindings(source_bindings)

    prepared = object.__new__(PreparedOuterCondition)
    values = {
        "protocol_directory": protocol_path,
        "source": resolved_source,
        "condition": condition,
        "split_manifest_sha256": _json_sha256(split_manifest),
        "run_manifest_sha256": _json_sha256(run_manifest),
        "readiness_sha256": _json_sha256(readiness_binding),
        "d3_records_sha256": _json_sha256(fresh_d3),
        "sessions_sha256": _sessions_sha256(sessions),
        "frame_df": frame_df,
        "readiness": readiness,
        "sessions": tuple(sessions),
        "_seal": _PREPARED_OUTER_CONDITION_SEAL,
    }
    for name, value in values.items():
        object.__setattr__(prepared, name, value)
    _PREPARED_OUTER_CONDITIONS[id(prepared)] = prepared
    return prepared


def prepared_outer_authority_inputs(
    prepared: PreparedOuterCondition,
) -> tuple[object, ReadinessReport]:
    """Return the sealed audited frame/report pair for authority construction."""

    split_manifest, run_manifest = _load_protocol_from_directory(
        _registered_prepared(prepared).protocol_directory
    )
    condition = _condition_from_manifest(split_manifest, prepared.condition.condition_id)
    _verify_source_bindings(split_manifest["source_bindings"])
    _prepared_sessions_and_readiness(
        prepared,
        protocol_path=prepared.protocol_directory,
        source=prepared.source,
        condition=condition,
        split_manifest=split_manifest,
        run_manifest=run_manifest,
    )
    projected_sessions = _sessions_from_condition_frame(
        prepared.frame_df,
        condition,
        run_manifest,
    )
    if _sessions_sha256(projected_sessions) != prepared.sessions_sha256:
        raise OuterExperimentGateError(
            "prepared authority frame no longer projects to the sealed sessions"
        )
    return prepared.frame_df, prepared.readiness


def run_outer_unit(
    protocol_directory: str | Path,
    output_directory: str | Path,
    *,
    source: str | Path,
    analysis_variant: str,
    condition_id: str,
    outer_fold_id: str,
    seed: int,
    backend: OuterTrainingBackend,
    prepared: PreparedOuterCondition | None = None,
) -> Path:
    """Run and atomically seal one backend-contract or future counted unit.

    The backend receives only the role-appropriate sessions and runner-fitted
    statistics. The source and checkpoint are re-hashed after the single outer
    evaluation. ``prepared=None`` retains the original fresh-replay semantics;
    a factory-sealed prepared condition may be reused only inside the same
    process. Any exception rolls back only this invocation's staging/output.
    """
    protocol_path = Path(protocol_directory).resolve()
    split_manifest, run_manifest = _load_protocol_from_directory(protocol_path)
    seed = _exact_positive_int(seed, "seed")
    if seed not in FROZEN_SEEDS:
        raise OuterExperimentGateError(f"seed must be one of {FROZEN_SEEDS}")
    registration = _resolve_backend_registration(backend, run_manifest)
    condition = _condition_from_manifest(split_manifest, condition_id)
    _authorize_result_role(registration, condition, run_manifest)
    variant = _variant_from_manifest(split_manifest, analysis_variant)
    fold = _fold_from_variant(variant, outer_fold_id)
    source_bindings = split_manifest["source_bindings"]
    resolved_source = Path(source).resolve()
    if str(resolved_source) != run_manifest["source"]:
        raise OuterExperimentGateError("execution source differs from the frozen run source")
    _verify_source_bindings(source_bindings)

    if prepared is None:
        sessions, readiness = _load_condition_sessions(
            resolved_source,
            condition,
            run_manifest,
        )
    else:
        sessions, readiness = _prepared_sessions_and_readiness(
            prepared,
            protocol_path=protocol_path,
            source=resolved_source,
            condition=condition,
            split_manifest=split_manifest,
            run_manifest=run_manifest,
        )
    if not _type_exact_equal(
        _readiness_binding(readiness),
        run_manifest["readiness_by_condition"].get(condition_id),
    ):
        raise OuterExperimentGateError("current readiness snapshot differs from the frozen run")
    fresh_d3 = _d3_records_from_report(readiness)
    if not _type_exact_equal(fresh_d3, split_manifest["d3_records"]):
        raise OuterExperimentGateError("fresh D3 provenance differs from the frozen split")
    _verify_loaded_sessions(sessions, split_manifest)
    _verify_source_bindings(source_bindings)
    master_sessions_sha256 = _sessions_sha256(sessions)

    output_path = Path(output_directory).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite outer-test unit: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stage = output_path.with_name(f".{output_path.name}.{token}.stage")
    try:
        _create_owned_stage(stage, token)
        by_id = {session.session_id: session for session in sessions}
        inner_receipts: list[InnerFitReceipt] = []
        inner_statistics: list[dict[str, object]] = []
        for inner in fold["inner_folds"]:
            inner_dir = stage / inner["inner_fold_id"]
            inner_dir.mkdir()
            train_master = _select_sessions(by_id, inner["train_session_ids"])
            validation_master = _select_sessions(by_id, inner["validation_session_ids"])
            normalizer = FeatureNormalizer.fit(list(train_master))
            pos_weight = _fit_pos_weight(train_master)
            inner_statistics.append(
                _statistics_record(inner["inner_fold_id"], train_master, normalizer, pos_weight)
            )
            request = InnerFitRequest(
                condition_id=condition.condition_id,
                feature_set=condition.feature_set,
                encoder=condition.encoder,
                outer_fold_id=outer_fold_id,
                inner_fold_id=inner["inner_fold_id"],
                seed=seed,
                max_epochs=run_manifest["max_epochs"],
                train_sessions=_clone_sessions(train_master),
                validation_sessions=_clone_sessions(validation_master),
                normalizer=_clone_normalizer(normalizer),
                pos_weight=pos_weight.copy(),
                output_dir=inner_dir,
            )
            receipt = backend.fit_inner(request)
            _assert_master_sessions_unchanged(sessions, master_sessions_sha256)
            _validate_inner_receipt(receipt, request)
            inner_receipts.append(receipt)

        best_epochs = [
            _exact_positive_int(item.best_epoch, "best_epoch") for item in inner_receipts
        ]
        selected_epoch = int(statistics.median(best_epochs))
        calibration_dir = stage / "calibration"
        calibration_dir.mkdir()
        sealed_oof = {
            str(item.oof_predictions_path): _capture_file_snapshot(
                item.oof_predictions_path,
                require_single_link=True,
            ).sha256
            for item in inner_receipts
        }
        calibration_request = CalibrationRequest(
            condition_id=condition.condition_id,
            outer_fold_id=outer_fold_id,
            seed=seed,
            inner_fold_ids=tuple(item.inner_fold_id for item in inner_receipts),
            oof_prediction_paths=tuple(item.oof_predictions_path for item in inner_receipts),
            oof_session_ids=tuple(
                session_id
                for inner in fold["inner_folds"]
                for session_id in inner["validation_session_ids"]
            ),
            output_dir=calibration_dir,
        )
        calibration = backend.calibrate_inner_oof(calibration_request)
        _assert_master_sessions_unchanged(sessions, master_sessions_sha256)
        _verify_path_hashes(sealed_oof, "inner OOF prediction")
        _validate_calibration_receipt(calibration, calibration_request)
        calibration_sha256 = _capture_file_snapshot(
            calibration.calibration_path,
            require_single_link=True,
        ).sha256

        final_train_master = _select_sessions(by_id, fold["outer_train_session_ids"])
        final_normalizer = FeatureNormalizer.fit(list(final_train_master))
        final_pos_weight = _fit_pos_weight(final_train_master)
        final_statistics = _statistics_record(
            "outer_train_final", final_train_master, final_normalizer, final_pos_weight
        )
        final_dir = stage / "final_fit"
        final_dir.mkdir()
        final_request = FinalFitRequest(
            condition_id=condition.condition_id,
            feature_set=condition.feature_set,
            encoder=condition.encoder,
            outer_fold_id=outer_fold_id,
            seed=seed,
            fixed_epochs=selected_epoch,
            train_sessions=_clone_sessions(final_train_master),
            normalizer=_clone_normalizer(final_normalizer),
            pos_weight=final_pos_weight.copy(),
            calibration_path=calibration.calibration_path,
            output_dir=final_dir,
        )
        final = backend.fit_final(final_request)
        _assert_master_sessions_unchanged(sessions, master_sessions_sha256)
        if (
            _capture_file_snapshot(
                calibration.calibration_path,
                require_single_link=True,
            ).sha256
            != calibration_sha256
        ):
            raise OuterExperimentGateError("sealed calibration changed during final fit")
        _validate_final_receipt(final, final_request)
        checkpoint_sha256 = _capture_file_snapshot(
            final.checkpoint_path,
            require_single_link=True,
        ).sha256

        # Selection is now closed.  Recheck data immediately before the sole test pass.
        _verify_source_bindings(source_bindings)
        outer_sessions_master = _select_sessions(by_id, fold["outer_test_session_ids"])
        evaluation_dir = stage / "outer_evaluation"
        evaluation_dir.mkdir()
        evaluation_request = OuterEvaluationRequest(
            condition_id=condition.condition_id,
            outer_fold_id=outer_fold_id,
            seed=seed,
            checkpoint_path=final.checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            calibration_path=calibration.calibration_path,
            test_sessions=_clone_sessions(outer_sessions_master),
            output_dir=evaluation_dir,
        )
        evaluation = backend.evaluate_outer(evaluation_request)
        _assert_master_sessions_unchanged(sessions, master_sessions_sha256)
        _validate_evaluation_receipt(evaluation, evaluation_request)
        if (
            _capture_file_snapshot(
                final.checkpoint_path,
                require_single_link=True,
            ).sha256
            != checkpoint_sha256
        ):
            raise OuterExperimentGateError("sealed final checkpoint changed during evaluation")
        if (
            _capture_file_snapshot(
                calibration.calibration_path,
                require_single_link=True,
            ).sha256
            != calibration_sha256
        ):
            raise OuterExperimentGateError("sealed calibration changed during outer evaluation")
        _verify_source_bindings(source_bindings)
        if not _type_exact_equal(_d3_records_from_report(readiness), fresh_d3):
            raise OuterExperimentGateError("D3 provenance changed during execution")

        result_manifest = _build_result_manifest(
            split_manifest=split_manifest,
            run_manifest=run_manifest,
            protocol_path=protocol_path,
            registration=registration,
            analysis_variant=analysis_variant,
            condition=condition,
            fold=fold,
            seed=seed,
            best_epochs=best_epochs,
            selected_epoch=selected_epoch,
            inner_receipts=inner_receipts,
            inner_statistics=inner_statistics,
            calibration=calibration,
            final_statistics=final_statistics,
            final=final,
            evaluation=evaluation,
            checkpoint_sha256=checkpoint_sha256,
            calibration_sha256=calibration_sha256,
            master_sessions_sha256=master_sessions_sha256,
            d3_records=fresh_d3,
            stage=stage,
        )
        _write_json(stage / "result_manifest.json", result_manifest)
        if prepared is None:
            _validate_result_directory(stage)
        else:
            _validate_result_directory(stage, prepared=prepared)
        os.rename(stage, output_path)
        try:
            final_snapshot = (
                _validate_result_directory(output_path)
                if prepared is None
                else _validate_result_directory(output_path, prepared=prepared)
            )
            # Keep the whole validated file set open through the return boundary.
            # On Windows these handles deny WRITE/DELETE sharing as a set; on
            # POSIX the final handle/path-state pass closes the detectable
            # mutation window under the documented trusted-filesystem boundary.
            with _held_result_snapshot(final_snapshot):
                return output_path / "result_manifest.json"
        except BaseException:
            _rollback_owned_path(output_path, token)
            raise
    except BaseException:
        _remove_call_owned_stage(stage, token)
        _rollback_owned_path(output_path, token)
        raise
    finally:
        _remove_call_owned_stage(stage, token)


def validate_outer_split_manifest(manifest: dict[str, object]) -> None:
    """Reject any split mutation that changes the frozen scientific protocol."""
    if not isinstance(manifest, dict):
        raise OuterExperimentGateError("split manifest must be a JSON object")
    profile = _corpus_profile_from_protocol_id(manifest.get("protocol_id"))
    expected_keys = {
        "artifact_kind",
        "format_version",
        "protocol_id",
        "result_role",
        "source_session_to_group",
        "source_bindings",
        "d3_records",
        "condition_ids",
        "conditions",
        "seeds",
        "aux_vad_weight",
        "selection",
        "d3_sensitivity",
        "analysis_variants",
    }
    if profile.corpus_exclusions:
        expected_keys.add("corpus_exclusions")
    if set(manifest) != expected_keys:
        raise OuterExperimentGateError("split manifest must use the exact version-2 fields")
    _require_exact(manifest, "artifact_kind", SPLIT_MANIFEST_KIND)
    _require_exact_int_field(manifest, "format_version", profile.format_version)
    _require_exact(manifest, "protocol_id", profile.protocol_id)
    if profile.corpus_exclusions and not _type_exact_equal(
        manifest.get("corpus_exclusions"), profile.exclusions
    ):
        raise OuterExperimentGateError("corpus exclusions differ from the code-owned profile")
    _require_exact(manifest, "result_role", "protocol_only_not_a_result")
    if not _type_exact_equal(manifest.get("seeds"), list(FROZEN_SEEDS)):
        raise OuterExperimentGateError("split manifest seeds differ from the frozen seeds")
    aux = manifest.get("aux_vad_weight")
    if isinstance(aux, bool) or not isinstance(aux, (int, float)) or float(aux) != 0.0:
        raise OuterExperimentGateError("aux_vad_weight must be the frozen value 0.0")

    mapping = _validate_session_group_map(manifest.get("source_session_to_group"))
    if mapping != profile.mapping:
        raise OuterExperimentGateError("split manifest primary session/group binding changed")
    source_bindings = _validate_source_bindings_payload(manifest.get("source_bindings"))
    d3_records = _validate_d3_records(manifest.get("d3_records"), source_bindings, profile=profile)

    raw_conditions = manifest.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise OuterExperimentGateError("conditions must be a non-empty JSON array")
    conditions = _validate_conditions(
        tuple(_condition_from_payload(item) for item in raw_conditions)
    )
    expected_ids = [item.condition_id for item in conditions]
    if manifest.get("condition_ids") != expected_ids:
        raise OuterExperimentGateError("condition_ids must exactly match conditions in order")

    selection = manifest.get("selection")
    expected_selection = {
        "inner_scheme": "leave_one_group_out_over_three_outer_train_groups",
        "best_epoch_indexing": "one_based",
        "fixed_epoch_rule": "median_of_three_inner_best_epochs",
        "threshold_and_calibration_source": "pooled_inner_oof_only",
        "final_fit": "same_seed_from_scratch_on_all_outer_train_groups",
        "outer_evaluation": "once_after_sealed_final_checkpoint",
    }
    if selection != expected_selection:
        raise OuterExperimentGateError("selection contract differs from the frozen protocol")
    if not _type_exact_equal(
        manifest.get("d3_sensitivity"),
        _d3_sensitivity_payload(d3_records, conditions, profile=profile),
    ):
        raise OuterExperimentGateError("D3 sensitivity contract differs from the frozen plan")

    variants = manifest.get("analysis_variants")
    if not isinstance(variants, dict) or set(variants) != set(ANALYSIS_VARIANTS):
        raise OuterExperimentGateError("analysis_variants must contain primary and d3_le80")
    expected_variant_mappings = {
        "primary": mapping,
        "d3_le80": {
            session_id: mapping[session_id]
            for session_id in PAIRED_D3_80MS_SESSION_IDS
            if session_id in mapping
        },
    }
    for name in ANALYSIS_VARIANTS:
        variant = variants[name]
        expected_variant_keys = {
            "analysis_variant",
            "session_to_group",
            "group_count",
            "outer_folds",
        }
        if not isinstance(variant, dict) or set(variant) != expected_variant_keys:
            raise OuterExperimentGateError("analysis variant must use exact version-2 fields")
        if variant["analysis_variant"] != name:
            raise OuterExperimentGateError("analysis variant name/key mismatch")
        _require_exact_int_field(variant, "group_count", FROZEN_GROUP_COUNT)
        variant_mapping = _validate_session_group_map(variant["session_to_group"])
        if variant_mapping != expected_variant_mappings[name]:
            raise OuterExperimentGateError(f"{name} session membership differs from frozen plan")
        _validate_logo_folds(variant_mapping, variant.get("outer_folds"))


def _validate_logo_folds(mapping: dict[str, str], folds: object) -> None:
    groups = tuple(sorted(set(mapping.values())))
    if len(groups) != FROZEN_GROUP_COUNT:
        raise OuterExperimentGateError("each analysis variant must retain exactly four groups")
    if not isinstance(folds, list) or len(folds) != FROZEN_GROUP_COUNT:
        raise OuterExperimentGateError("there must be exactly four outer folds")
    seen_test_groups: list[str] = []
    for index, fold in enumerate(folds, start=1):
        expected_fold_keys = {
            "outer_fold_id",
            "outer_train_group_ids",
            "outer_test_group_ids",
            "outer_train_session_ids",
            "outer_test_session_ids",
            "inner_folds",
        }
        if not isinstance(fold, dict) or set(fold) != expected_fold_keys:
            raise OuterExperimentGateError("every outer fold must be a JSON object")
        if fold.get("outer_fold_id") != f"outer_{index:02d}":
            raise OuterExperimentGateError("outer fold IDs/order are not canonical")
        test_groups = _string_list(fold.get("outer_test_group_ids"), "outer_test_group_ids")
        train_groups = _string_list(fold.get("outer_train_group_ids"), "outer_train_group_ids")
        if len(test_groups) != 1 or set(train_groups) | set(test_groups) != set(groups):
            raise OuterExperimentGateError("outer train/test groups must partition all groups")
        if set(train_groups) & set(test_groups) or len(train_groups) != 3:
            raise OuterExperimentGateError("outer train/test group leakage")
        seen_test_groups.extend(test_groups)
        _require_session_list(
            fold, "outer_train_session_ids", _sessions_for_groups(mapping, tuple(train_groups))
        )
        _require_session_list(
            fold, "outer_test_session_ids", _sessions_for_groups(mapping, tuple(test_groups))
        )
        inner_folds = fold.get("inner_folds")
        if not isinstance(inner_folds, list) or len(inner_folds) != 3:
            raise OuterExperimentGateError("each outer fold must contain three inner folds")
        seen_validation: list[str] = []
        for inner_index, inner in enumerate(inner_folds, start=1):
            expected_inner_keys = {
                "inner_fold_id",
                "train_group_ids",
                "validation_group_ids",
                "train_session_ids",
                "validation_session_ids",
            }
            if not isinstance(inner, dict) or set(inner) != expected_inner_keys:
                raise OuterExperimentGateError("every inner fold must be a JSON object")
            if inner.get("inner_fold_id") != f"inner_{inner_index:02d}":
                raise OuterExperimentGateError("inner fold IDs/order are not canonical")
            validation_groups = _string_list(
                inner.get("validation_group_ids"), "validation_group_ids"
            )
            inner_train = _string_list(inner.get("train_group_ids"), "train_group_ids")
            if len(validation_groups) != 1 or set(validation_groups) - set(train_groups):
                raise OuterExperimentGateError("inner validation must be one outer-train group")
            if set(inner_train) | set(validation_groups) != set(train_groups):
                raise OuterExperimentGateError(
                    "inner train/validation do not partition outer train"
                )
            if set(inner_train) & set(validation_groups) or len(inner_train) != 2:
                raise OuterExperimentGateError("inner train/validation group leakage")
            seen_validation.extend(validation_groups)
            _require_session_list(
                inner, "train_session_ids", _sessions_for_groups(mapping, tuple(inner_train))
            )
            _require_session_list(
                inner,
                "validation_session_ids",
                _sessions_for_groups(mapping, tuple(validation_groups)),
            )
        if sorted(seen_validation) != sorted(train_groups):
            raise OuterExperimentGateError("each outer-train group must validate exactly once")
    if sorted(seen_test_groups) != list(groups):
        raise OuterExperimentGateError("each group must be outer test exactly once")


def validate_outer_run_manifest(
    manifest: dict[str, object], split_manifest: dict[str, object]
) -> None:
    validate_outer_split_manifest(split_manifest)
    profile = _corpus_profile_from_protocol_id(split_manifest["protocol_id"])
    if not isinstance(manifest, dict):
        raise OuterExperimentGateError("run manifest must be a JSON object")
    expected_keys = {
        "artifact_kind",
        "format_version",
        "protocol_id",
        "result_role",
        "counted_experiment_complete",
        "split_manifest_sha256",
        "source",
        "seeds",
        "condition_ids",
        "max_epochs",
        "aux_vad_weight",
        "trainer_config",
        "model_config",
        "backend_policy",
        "label_config",
        "readiness_by_condition",
        "implementation_status",
    }
    if profile.corpus_exclusions:
        expected_keys.add("corpus_exclusions")
    if set(manifest) != expected_keys:
        raise OuterExperimentGateError("run manifest must use the exact version-2 fields")
    _require_exact(manifest, "artifact_kind", RUN_MANIFEST_KIND)
    _require_exact_int_field(manifest, "format_version", profile.format_version)
    _require_exact(manifest, "protocol_id", profile.protocol_id)
    if profile.corpus_exclusions and not _type_exact_equal(
        manifest.get("corpus_exclusions"), profile.exclusions
    ):
        raise OuterExperimentGateError("run corpus exclusions differ from the split profile")
    _require_exact(manifest, "result_role", "protocol_only_not_a_result")
    if manifest.get("counted_experiment_complete") is not False:
        raise OuterExperimentGateError("protocol run manifest cannot claim counted completion")
    if manifest.get("split_manifest_sha256") != _json_sha256(split_manifest):
        raise OuterExperimentGateError("run manifest does not bind this exact split manifest")
    source_value = manifest.get("source")
    if not isinstance(source_value, str):
        raise OuterExperimentGateError("run source must be a normalized absolute path")
    source_path = Path(source_value)
    if not source_path.is_absolute() or source_path != source_path.resolve():
        raise OuterExperimentGateError("run source must be a normalized absolute path")
    if not _type_exact_equal(manifest.get("seeds"), list(FROZEN_SEEDS)):
        raise OuterExperimentGateError("run manifest seed set changed")
    if manifest.get("condition_ids") != split_manifest["condition_ids"]:
        raise OuterExperimentGateError("run/split conditions differ")
    _exact_positive_int(manifest.get("max_epochs"), "max_epochs")
    if manifest.get("aux_vad_weight") != FROZEN_AUX_VAD_WEIGHT or isinstance(
        manifest.get("aux_vad_weight"), bool
    ):
        raise OuterExperimentGateError("run aux_vad_weight must be exactly 0.0")
    trainer = _strict_json_mapping(manifest.get("trainer_config"), "trainer_config")
    if trainer.get("aux_vad_weight") != FROZEN_AUX_VAD_WEIGHT or isinstance(
        trainer.get("aux_vad_weight"), bool
    ):
        raise OuterExperimentGateError("trainer_config.aux_vad_weight must be exactly 0.0")
    if "random_seed" in trainer or "epochs" in trainer:
        raise OuterExperimentGateError("seed/epochs are controlled only by the outer protocol")
    _strict_json_mapping(manifest.get("model_config"), "model_config")
    backend_policy = _validate_backend_policy(manifest.get("backend_policy"))
    _validate_label_config(manifest.get("label_config"))
    readiness = _strict_json_mapping(
        manifest.get("readiness_by_condition"), "readiness_by_condition"
    )
    if set(readiness) != set(split_manifest["condition_ids"]):
        raise OuterExperimentGateError("readiness bindings must cover all conditions exactly")
    _validate_readiness_bindings(readiness, split_manifest)
    expected_status = {
        "protocol_orchestrator": "implemented",
        "production_torch_backend": backend_policy["production_status"],
        "environment_capture": "not_implemented_blocker",
        "run_index": "not_implemented_blocker",
        "pose_time_shift_reassembly": "not_implemented_blocker",
    }
    if manifest.get("implementation_status") != expected_status:
        raise OuterExperimentGateError("implementation status must expose every remaining blocker")


def _sessions_from_condition_frame(
    frame_df: object,
    condition: OuterCondition,
    run_manifest: dict[str, object],
) -> list[SessionTensors]:
    experiment_df = apply_feature_set(frame_df, condition.feature_set)
    label_config = VAPLabelConfig(**run_manifest["label_config"])
    return sessions_from_dataframe(experiment_df, label_config=label_config)


def _load_condition_material(
    source: Path,
    condition: OuterCondition,
    run_manifest: dict[str, object],
) -> tuple[object, list[SessionTensors], ReadinessReport]:
    profile = _corpus_profile_from_protocol_id(run_manifest.get("protocol_id"))
    frame_df, report = load_and_audit_vap_source(
        source,
        data_kind="real",
        feature_set=condition.feature_set,
        excluded_session_ids=tuple(profile.exclusions),
    )
    _require_ready(report, condition)
    sessions = _sessions_from_condition_frame(frame_df, condition, run_manifest)
    return frame_df, sessions, report


def _load_condition_sessions(
    source: Path, condition: OuterCondition, run_manifest: dict[str, object]
) -> tuple[list[SessionTensors], ReadinessReport]:
    _, sessions, report = _load_condition_material(source, condition, run_manifest)
    return sessions, report


def _registered_prepared(value: object) -> PreparedOuterCondition:
    if type(value) is not PreparedOuterCondition:
        raise OuterExperimentGateError(
            "prepared condition must be an exact factory-issued PreparedOuterCondition"
        )
    prepared = value
    if (
        _PREPARED_OUTER_CONDITIONS.get(id(prepared)) is not prepared
        or prepared._seal is not _PREPARED_OUTER_CONDITION_SEAL
    ):
        raise OuterExperimentGateError(
            "prepared condition is not a factory-issued prepare_outer_condition object"
        )
    return prepared


def _prepared_sessions_and_readiness(
    value: object,
    *,
    protocol_path: Path,
    source: Path,
    condition: OuterCondition,
    split_manifest: dict[str, object],
    run_manifest: dict[str, object],
) -> tuple[list[SessionTensors], ReadinessReport]:
    prepared = _registered_prepared(value)
    if prepared.protocol_directory != protocol_path:
        raise OuterExperimentGateError("prepared condition belongs to a different protocol")
    if prepared.source != source:
        raise OuterExperimentGateError("prepared condition belongs to a different source")
    if type(prepared.condition) is not OuterCondition or prepared.condition != condition:
        raise OuterExperimentGateError("prepared condition identity differs from this unit")
    if prepared.split_manifest_sha256 != _json_sha256(split_manifest):
        raise OuterExperimentGateError("prepared split-manifest binding differs")
    if prepared.run_manifest_sha256 != _json_sha256(run_manifest):
        raise OuterExperimentGateError("prepared run-manifest binding differs")
    if not isinstance(prepared.readiness, ReadinessReport):
        raise OuterExperimentGateError("prepared readiness object is invalid")
    try:
        readiness_source = Path(prepared.readiness.source).resolve()
    except (OSError, TypeError, ValueError) as exc:
        raise OuterExperimentGateError("prepared readiness source is invalid") from exc
    if readiness_source != source:
        raise OuterExperimentGateError("prepared readiness belongs to a different source")
    readiness_binding = _readiness_binding(prepared.readiness)
    expected_readiness = run_manifest["readiness_by_condition"].get(condition.condition_id)
    if prepared.readiness_sha256 != _json_sha256(readiness_binding) or not _type_exact_equal(
        readiness_binding, expected_readiness
    ):
        raise OuterExperimentGateError("prepared readiness differs from the frozen run")
    fresh_d3 = _d3_records_from_report(prepared.readiness)
    if prepared.d3_records_sha256 != _json_sha256(fresh_d3) or not _type_exact_equal(
        fresh_d3, split_manifest["d3_records"]
    ):
        raise OuterExperimentGateError("prepared D3 provenance differs from the frozen split")
    if not isinstance(prepared.sessions, tuple):
        raise OuterExperimentGateError("prepared sessions must be a sealed tuple")
    sessions = list(prepared.sessions)
    _verify_loaded_sessions(sessions, split_manifest)
    if prepared.sessions_sha256 != _sessions_sha256(sessions):
        raise OuterExperimentGateError("prepared session tensors changed after sealing")
    return sessions, prepared.readiness


def _build_result_manifest(
    *,
    split_manifest: dict[str, object],
    run_manifest: dict[str, object],
    protocol_path: Path,
    registration: _BackendRegistration,
    analysis_variant: str,
    condition: OuterCondition,
    fold: dict[str, object],
    seed: int,
    best_epochs: list[int],
    selected_epoch: int,
    inner_receipts: list[InnerFitReceipt],
    inner_statistics: list[dict[str, object]],
    calibration: CalibrationReceipt,
    final_statistics: dict[str, object],
    final: FinalFitReceipt,
    evaluation: OuterEvaluationReceipt,
    checkpoint_sha256: str,
    calibration_sha256: str,
    master_sessions_sha256: str,
    d3_records: dict[str, dict[str, object]],
    stage: Path,
) -> dict[str, object]:
    artifact_bindings = {
        "inner_checkpoints": [
            {
                "inner_fold_id": item.inner_fold_id,
                **_artifact_binding(item.checkpoint_path, stage),
            }
            for item in inner_receipts
        ],
        "inner_oof_predictions": [
            {
                "inner_fold_id": item.inner_fold_id,
                **_artifact_binding(item.oof_predictions_path, stage),
            }
            for item in inner_receipts
        ],
        "calibration": _artifact_binding(calibration.calibration_path, stage),
        "final_checkpoint": _artifact_binding(final.checkpoint_path, stage),
        "outer_predictions": _artifact_binding(evaluation.predictions_path, stage),
        "outer_metrics": _artifact_binding(evaluation.metrics_path, stage),
    }
    _validate_physical_artifact_uniqueness(
        tuple(
            stage / binding["relative_path"]
            for value in artifact_bindings.values()
            for binding in (value if isinstance(value, list) else [value])
        )
    )
    if artifact_bindings["final_checkpoint"]["sha256"] != checkpoint_sha256:
        raise OuterExperimentGateError("final checkpoint hash changed before manifest sealing")
    if artifact_bindings["calibration"]["sha256"] != calibration_sha256:
        raise OuterExperimentGateError("calibration hash changed before manifest sealing")
    if final_statistics["fit_session_ids"] != list(final.normalizer_fit_session_ids):
        raise OuterExperimentGateError("final statistics scope differs from backend receipt")
    remaining_gates = _remaining_execution_gates(registration, condition, run_manifest)
    profile = _corpus_profile_from_protocol_id(split_manifest["protocol_id"])
    payload: dict[str, object] = {
        "artifact_kind": RESULT_MANIFEST_KIND,
        "format_version": profile.format_version,
        "protocol_id": profile.protocol_id,
        "result_role": registration.result_role,
        "execution_complete": True,
        "counted_experiment_complete": registration.counted_eligible,
        "remaining_counted_gates": remaining_gates,
        "protocol_directory": str(protocol_path),
        "split_manifest_sha256": _json_sha256(split_manifest),
        "run_manifest_sha256": _json_sha256(run_manifest),
        "analysis_variant": analysis_variant,
        "condition_id": condition.condition_id,
        "feature_set": condition.feature_set,
        "active_feature_columns": active_feature_columns(condition.feature_set),
        "uses_head_pose": _condition_uses_head_pose(condition),
        "encoder": condition.encoder,
        "backend_registration": _registration_payload(registration),
        "outer_fold_id": fold["outer_fold_id"],
        "seed": seed,
        "selection": {
            "rule": "median_of_three_inner_best_epochs",
            "inner_best_epochs": [
                {
                    "inner_fold_id": receipt.inner_fold_id,
                    "best_epoch": best_epoch,
                }
                for receipt, best_epoch in zip(inner_receipts, best_epochs, strict=True)
            ],
            "selected_fixed_epoch": selected_epoch,
            "calibration_source": "pooled_inner_oof_only",
            "calibration_inner_fold_ids": list(calibration.inner_fold_ids),
            "calibration_oof_session_ids": [
                session_id
                for inner in fold["inner_folds"]
                for session_id in inner["validation_session_ids"]
            ],
        },
        "statistics": {
            "inner": inner_statistics,
            "final": final_statistics,
        },
        "outer_test_session_ids": list(evaluation.evaluation_session_ids),
        "outer_evaluation_count": evaluation.evaluation_count,
        "master_sessions_sha256_before": master_sessions_sha256,
        "master_sessions_sha256_after": master_sessions_sha256,
        "d3_records": d3_records,
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": calibration_sha256,
        "source_bindings": split_manifest["source_bindings"],
        "artifact_bindings": artifact_bindings,
    }
    if profile.corpus_exclusions:
        payload["corpus_exclusions"] = profile.exclusions
    return payload


def _validate_result_directory(
    directory: Path,
    *,
    prepared: PreparedOuterCondition | None = None,
) -> _ResultValidationSnapshot:
    payload, result_file_snapshot = _read_json_with_snapshot(directory / "result_manifest.json")
    profile = _corpus_profile_from_protocol_id(payload.get("protocol_id"))
    expected_result_keys = {
        "artifact_kind",
        "format_version",
        "protocol_id",
        "result_role",
        "execution_complete",
        "counted_experiment_complete",
        "remaining_counted_gates",
        "protocol_directory",
        "split_manifest_sha256",
        "run_manifest_sha256",
        "analysis_variant",
        "condition_id",
        "feature_set",
        "active_feature_columns",
        "uses_head_pose",
        "encoder",
        "backend_registration",
        "outer_fold_id",
        "seed",
        "selection",
        "statistics",
        "outer_test_session_ids",
        "outer_evaluation_count",
        "master_sessions_sha256_before",
        "master_sessions_sha256_after",
        "d3_records",
        "checkpoint_sha256",
        "calibration_sha256",
        "source_bindings",
        "artifact_bindings",
    }
    if profile.corpus_exclusions:
        expected_result_keys.add("corpus_exclusions")
    if set(payload) != expected_result_keys:
        raise OuterExperimentGateError("result manifest must use the exact version-2 fields")
    entry_artifact_snapshots = _snapshot_declared_role_artifacts(
        directory,
        payload.get("artifact_bindings"),
    )
    _require_exact(payload, "artifact_kind", RESULT_MANIFEST_KIND)
    _require_exact_int_field(payload, "format_version", profile.format_version)
    _require_exact(payload, "protocol_id", profile.protocol_id)
    if profile.corpus_exclusions and not _type_exact_equal(
        payload.get("corpus_exclusions"), profile.exclusions
    ):
        raise OuterExperimentGateError("result corpus exclusions differ from its profile")
    if payload.get("execution_complete") is not True:
        raise OuterExperimentGateError("result execution_complete must be true")
    seed = _exact_positive_int(payload.get("seed"), "result seed")
    if payload.get("outer_evaluation_count") != 1 or isinstance(
        payload.get("outer_evaluation_count"), bool
    ):
        raise OuterExperimentGateError("a counted unit requires exactly one outer evaluation")
    protocol_value = payload.get("protocol_directory")
    if not isinstance(protocol_value, str):
        raise OuterExperimentGateError("result protocol_directory must be a string")
    protocol_path = Path(protocol_value)
    if not protocol_path.is_absolute() or protocol_path != protocol_path.resolve():
        raise OuterExperimentGateError("result protocol_directory must be normalized and absolute")
    split, split_file_snapshot = _read_json_with_snapshot(protocol_path / "split_manifest.json")
    run, run_file_snapshot = _read_json_with_snapshot(protocol_path / "run_manifest.json")
    validate_outer_split_manifest(split)
    validate_outer_run_manifest(run, split)
    if split["protocol_id"] != profile.protocol_id:
        raise OuterExperimentGateError("result protocol profile differs from frozen split")
    split_digest = _sha256_string(
        payload.get("split_manifest_sha256"), "result split_manifest_sha256"
    )
    run_digest = _sha256_string(payload.get("run_manifest_sha256"), "result run_manifest_sha256")
    if split_digest != _json_sha256(split):
        raise OuterExperimentGateError("result no longer binds its split manifest")
    if run_digest != _json_sha256(run):
        raise OuterExperimentGateError("result no longer binds its run manifest")
    registration = _registration_from_result(payload, run)
    if payload.get("result_role") != registration.result_role:
        raise OuterExperimentGateError("result role differs from the code-owned backend registry")
    if payload.get("counted_experiment_complete") is not registration.counted_eligible:
        raise OuterExperimentGateError("counted status differs from backend registry eligibility")
    condition = _condition_from_manifest(split, payload.get("condition_id"))
    expected_gates = _remaining_execution_gates(registration, condition, run)
    if payload.get("remaining_counted_gates") != expected_gates:
        raise OuterExperimentGateError("remaining counted gates are incomplete or reordered")
    if not _type_exact_equal(payload.get("source_bindings"), split["source_bindings"]):
        raise OuterExperimentGateError("result source bindings differ from the protocol")
    result_sources = _validate_source_bindings_payload(payload["source_bindings"])
    source_snapshots = _snapshot_source_bindings(result_sources)
    if payload.get("feature_set") != condition.feature_set:
        raise OuterExperimentGateError("result feature_set differs from its condition")
    if payload.get("encoder") != condition.encoder:
        raise OuterExperimentGateError("result encoder differs from its condition")
    if payload.get("active_feature_columns") != active_feature_columns(condition.feature_set):
        raise OuterExperimentGateError("result active feature columns differ from feature set")
    uses_head_pose = _condition_uses_head_pose(condition)
    if payload.get("uses_head_pose") is not uses_head_pose:
        raise OuterExperimentGateError("result head-pose sensitivity flag is incorrect")
    variant = _variant_from_manifest(split, payload.get("analysis_variant"))
    fold = _fold_from_variant(variant, payload.get("outer_fold_id"))
    if seed not in run["seeds"]:
        raise OuterExperimentGateError("result seed is absent from the frozen run")
    if payload.get("outer_test_session_ids") != fold["outer_test_session_ids"]:
        raise OuterExperimentGateError("result outer-test sessions differ from the frozen fold")

    selection = payload.get("selection")
    expected_selection_keys = {
        "rule",
        "inner_best_epochs",
        "selected_fixed_epoch",
        "calibration_source",
        "calibration_inner_fold_ids",
        "calibration_oof_session_ids",
    }
    if not isinstance(selection, dict) or set(selection) != expected_selection_keys:
        raise OuterExperimentGateError("result selection must use the exact version-2 fields")
    if selection["rule"] != "median_of_three_inner_best_epochs":
        raise OuterExperimentGateError("result epoch selection rule changed")
    inner_epoch_records = selection["inner_best_epochs"]
    if not isinstance(inner_epoch_records, list) or len(inner_epoch_records) != 3:
        raise OuterExperimentGateError("result must bind exactly three inner best epochs")
    expected_inner_ids = [inner["inner_fold_id"] for inner in fold["inner_folds"]]
    epochs: list[int] = []
    for record, expected_inner_id in zip(inner_epoch_records, expected_inner_ids, strict=True):
        if not isinstance(record, dict) or set(record) != {"inner_fold_id", "best_epoch"}:
            raise OuterExperimentGateError("malformed inner best-epoch record")
        if record["inner_fold_id"] != expected_inner_id:
            raise OuterExperimentGateError("inner best-epoch fold order changed")
        epochs.append(_exact_positive_int(record["best_epoch"], "inner_best_epoch"))
    if any(value > run["max_epochs"] for value in epochs):
        raise OuterExperimentGateError("result inner best epoch exceeds max_epochs")
    selected_epoch = _exact_positive_int(selection["selected_fixed_epoch"], "selected_fixed_epoch")
    if selected_epoch != int(statistics.median(epochs)):
        raise OuterExperimentGateError("result selected epoch is not the frozen median")
    if selection["calibration_source"] != "pooled_inner_oof_only":
        raise OuterExperimentGateError("calibration source must be pooled inner OOF only")
    if selection["calibration_inner_fold_ids"] != expected_inner_ids:
        raise OuterExperimentGateError("calibration inner-fold scope changed")
    expected_oof_sessions = [
        session_id
        for inner in fold["inner_folds"]
        for session_id in inner["validation_session_ids"]
    ]
    if selection["calibration_oof_session_ids"] != expected_oof_sessions:
        raise OuterExperimentGateError("calibration OOF session scope changed")

    source_path = Path(run["source"])
    if prepared is None:
        sessions, readiness = _load_condition_sessions(source_path, condition, run)
    else:
        sessions, readiness = _prepared_sessions_and_readiness(
            prepared,
            protocol_path=protocol_path,
            source=source_path,
            condition=condition,
            split_manifest=split,
            run_manifest=run,
        )
    expected_readiness = run["readiness_by_condition"][condition.condition_id]
    if not _type_exact_equal(_readiness_binding(readiness), expected_readiness):
        raise OuterExperimentGateError("fresh result readiness differs from frozen run")
    _verify_loaded_sessions(sessions, split)
    fresh_d3 = _d3_records_from_report(readiness)
    if not _type_exact_equal(fresh_d3, split["d3_records"]):
        raise OuterExperimentGateError("fresh result D3 provenance differs from frozen split")
    if not _type_exact_equal(payload.get("d3_records"), fresh_d3):
        raise OuterExperimentGateError("result D3 records differ from fresh provenance")
    master_hash = _sessions_sha256(sessions)
    before_hash = _sha256_string(
        payload.get("master_sessions_sha256_before"), "master_sessions_sha256_before"
    )
    after_hash = _sha256_string(
        payload.get("master_sessions_sha256_after"), "master_sessions_sha256_after"
    )
    if before_hash != master_hash:
        raise OuterExperimentGateError("result pre-backend master-session hash differs")
    if after_hash != master_hash:
        raise OuterExperimentGateError("result post-backend master-session hash differs")
    by_id = {session.session_id: session for session in sessions}
    statistics_payload = payload.get("statistics")
    if not isinstance(statistics_payload, dict) or set(statistics_payload) != {"inner", "final"}:
        raise OuterExperimentGateError("statistics must contain exact inner/final records")
    inner_statistics = statistics_payload["inner"]
    if not isinstance(inner_statistics, list) or len(inner_statistics) != 3:
        raise OuterExperimentGateError("statistics must contain three inner records")
    for record, inner in zip(inner_statistics, fold["inner_folds"], strict=True):
        train_sessions = _select_sessions(by_id, inner["train_session_ids"])
        _validate_statistics_record(record, inner["inner_fold_id"], train_sessions)
    final_train = _select_sessions(by_id, fold["outer_train_session_ids"])
    _validate_statistics_record(statistics_payload["final"], "outer_train_final", final_train)

    bindings = payload.get("artifact_bindings")
    expected_binding_keys = {
        "inner_checkpoints",
        "inner_oof_predictions",
        "calibration",
        "final_checkpoint",
        "outer_predictions",
        "outer_metrics",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_binding_keys:
        raise OuterExperimentGateError("result artifact_bindings must be a JSON object")
    if (
        not isinstance(bindings["inner_checkpoints"], list)
        or not isinstance(bindings["inner_oof_predictions"], list)
        or len(bindings["inner_checkpoints"]) != 3
        or len(bindings["inner_oof_predictions"]) != 3
    ):
        raise OuterExperimentGateError("result must bind three inner checkpoints and OOF files")
    for key in ("inner_checkpoints", "inner_oof_predictions"):
        if [item.get("inner_fold_id") for item in bindings[key] if isinstance(item, dict)] != (
            expected_inner_ids
        ):
            raise OuterExperimentGateError(f"{key} fold order differs from selection")
    expected_relative_paths = {
        "inner_checkpoints": [
            f"{inner_fold_id}/{INNER_CHECKPOINT_FILENAME}" for inner_fold_id in expected_inner_ids
        ],
        "inner_oof_predictions": [
            f"{inner_fold_id}/{INNER_OOF_FILENAME}" for inner_fold_id in expected_inner_ids
        ],
        "calibration": f"calibration/{CALIBRATION_FILENAME}",
        "final_checkpoint": f"final_fit/{FINAL_CHECKPOINT_FILENAME}",
        "outer_predictions": f"outer_evaluation/{OUTER_PREDICTIONS_FILENAME}",
        "outer_metrics": f"outer_evaluation/{OUTER_METRICS_FILENAME}",
    }
    for key, expected_paths in expected_relative_paths.items():
        value = bindings[key]
        actual_paths = (
            [item.get("relative_path") for item in value]
            if isinstance(value, list)
            else value.get("relative_path")
        )
        if actual_paths != expected_paths:
            raise OuterExperimentGateError(f"{key} paths differ from the fixed role contract")
    if not isinstance(bindings["final_checkpoint"], dict) or not isinstance(
        bindings["calibration"], dict
    ):
        raise OuterExperimentGateError(
            "result final-checkpoint/calibration bindings must be objects"
        )
    checkpoint_digest = _sha256_string(payload.get("checkpoint_sha256"), "result checkpoint_sha256")
    calibration_digest = _sha256_string(
        payload.get("calibration_sha256"), "result calibration_sha256"
    )
    if checkpoint_digest != bindings["final_checkpoint"].get("sha256"):
        raise OuterExperimentGateError("result checkpoint hash binding is inconsistent")
    if calibration_digest != bindings["calibration"].get("sha256"):
        raise OuterExperimentGateError("result calibration hash binding is inconsistent")
    artifact_snapshots: list[_FileSnapshot] = []
    for value in bindings.values():
        entries = value if isinstance(value, list) else [value]
        for binding in entries:
            expected_artifact_keys = {
                "relative_path",
                "sha256",
                "size_bytes",
            }
            if isinstance(value, list):
                expected_artifact_keys.add("inner_fold_id")
            if not isinstance(binding, dict) or set(binding) != expected_artifact_keys:
                raise OuterExperimentGateError("malformed result artifact binding")
            relative = binding.get("relative_path")
            if not isinstance(relative, str):
                raise OuterExperimentGateError("artifact relative_path must be a string")
            digest = _sha256_string(binding.get("sha256"), "result artifact sha256")
            size = _exact_nonnegative_int(binding.get("size_bytes"), "result artifact size_bytes")
            candidate = directory / relative
            if candidate.is_symlink():
                raise OuterExperimentGateError(
                    f"result artifact must not be a symlink: {candidate}"
                )
            artifact = _contained_path(directory, candidate)
            normalized_relative = artifact.relative_to(directory.resolve())
            if isinstance(value, list):
                inner_fold_id = binding["inner_fold_id"]
                if not normalized_relative.parts or normalized_relative.parts[0] != inner_fold_id:
                    raise OuterExperimentGateError(
                        "inner artifact path is not owned by its declared fold"
                    )
            artifact_snapshot = _capture_file_snapshot(
                artifact,
                require_single_link=True,
            )
            artifact_snapshots.append(artifact_snapshot)
            if artifact_snapshot.sha256 != digest:
                raise OuterExperimentGateError(f"result artifact hash changed: {artifact}")
            if artifact_snapshot.size_bytes != size:
                raise OuterExperimentGateError(f"result artifact size changed: {artifact}")
    _validate_physical_artifact_uniqueness(tuple(snapshot.path for snapshot in artifact_snapshots))
    if {snapshot.path: snapshot for snapshot in artifact_snapshots} != {
        snapshot.path: snapshot for snapshot in entry_artifact_snapshots
    }:
        raise OuterExperimentGateError("role artifact changed during result validation")
    validation_snapshot = _ResultValidationSnapshot(
        immutable_files=(
            result_file_snapshot,
            split_file_snapshot,
            run_file_snapshot,
            *source_snapshots,
        ),
        role_artifacts=entry_artifact_snapshots,
        readiness=readiness,
        d3_records=fresh_d3,
    )
    _revalidate_result_snapshot(validation_snapshot)
    return validation_snapshot


def _validate_inner_receipt(receipt: InnerFitReceipt, request: InnerFitRequest) -> None:
    _receipt_identity(receipt, request, include_inner=True)
    _exact_positive_int(receipt.best_epoch, "best_epoch")
    if receipt.best_epoch > request.max_epochs:
        raise OuterExperimentGateError("inner best_epoch exceeds the frozen maximum")
    expected_train = _session_ids(request.train_sessions)
    expected_val = _session_ids(request.validation_sessions)
    if (
        receipt.train_session_ids != expected_train
        or receipt.validation_session_ids != expected_val
    ):
        raise OuterExperimentGateError("inner backend returned a mismatched session partition")
    if receipt.normalizer_fit_session_ids != expected_train:
        raise OuterExperimentGateError("inner normalizer fit scope differs from inner train")
    if receipt.pos_weight_fit_session_ids != expected_train:
        raise OuterExperimentGateError("inner pos_weight fit scope differs from inner train")
    _exact_role_artifact(
        receipt.checkpoint_path,
        request.output_dir,
        INNER_CHECKPOINT_FILENAME,
        "inner checkpoint",
    )
    _exact_role_artifact(
        receipt.oof_predictions_path,
        request.output_dir,
        INNER_OOF_FILENAME,
        "inner OOF predictions",
    )
    if receipt.checkpoint_path.resolve() == receipt.oof_predictions_path.resolve():
        raise OuterExperimentGateError("inner checkpoint and OOF predictions must be distinct")


def _validate_calibration_receipt(receipt: CalibrationReceipt, request: CalibrationRequest) -> None:
    _receipt_identity(receipt, request, include_inner=False)
    if receipt.inner_fold_ids != request.inner_fold_ids:
        raise OuterExperimentGateError("calibration omitted or reordered an inner fold")
    if receipt.oof_prediction_paths != request.oof_prediction_paths:
        raise OuterExperimentGateError("calibration must use exactly the pooled inner OOF files")
    _exact_role_artifact(
        receipt.calibration_path,
        request.output_dir,
        CALIBRATION_FILENAME,
        "inner-OOF calibration",
    )


def _validate_final_receipt(receipt: FinalFitReceipt, request: FinalFitRequest) -> None:
    _receipt_identity(receipt, request, include_inner=False)
    if receipt.fixed_epochs != request.fixed_epochs or isinstance(receipt.fixed_epochs, bool):
        raise OuterExperimentGateError("final fit did not use the selected fixed epoch")
    expected = _session_ids(request.train_sessions)
    if receipt.train_session_ids != expected:
        raise OuterExperimentGateError("final fit session scope differs from outer train")
    if (
        receipt.normalizer_fit_session_ids != expected
        or receipt.pos_weight_fit_session_ids != expected
    ):
        raise OuterExperimentGateError("final statistics must be fit on outer-train sessions only")
    if receipt.initialized_from_scratch is not True:
        raise OuterExperimentGateError("final fit must initialize from scratch")
    _exact_role_artifact(
        receipt.checkpoint_path,
        request.output_dir,
        FINAL_CHECKPOINT_FILENAME,
        "final checkpoint",
    )


def _validate_evaluation_receipt(
    receipt: OuterEvaluationReceipt, request: OuterEvaluationRequest
) -> None:
    _receipt_identity(receipt, request, include_inner=False)
    if receipt.checkpoint_sha256 != request.checkpoint_sha256:
        raise OuterExperimentGateError("outer evaluation did not bind the sealed checkpoint bytes")
    if receipt.evaluation_session_ids != _session_ids(request.test_sessions):
        raise OuterExperimentGateError("outer evaluation session scope differs from outer test")
    if receipt.evaluation_count != 1 or isinstance(receipt.evaluation_count, bool):
        raise OuterExperimentGateError("outer evaluation must occur exactly once")
    _exact_role_artifact(
        receipt.predictions_path,
        request.output_dir,
        OUTER_PREDICTIONS_FILENAME,
        "outer predictions",
    )
    _exact_role_artifact(
        receipt.metrics_path,
        request.output_dir,
        OUTER_METRICS_FILENAME,
        "outer metrics",
    )
    if receipt.predictions_path.resolve() == receipt.metrics_path.resolve():
        raise OuterExperimentGateError("outer predictions and metrics must be distinct")


def _receipt_identity(receipt: object, request: object, *, include_inner: bool) -> None:
    receipt_seed = _exact_positive_int(receipt.seed, "backend receipt seed")
    request_seed = _exact_positive_int(request.seed, "backend request seed")
    if receipt_seed != request_seed:
        raise OuterExperimentGateError("backend receipt seed mismatch")
    for field_name in ("condition_id", "outer_fold_id", "seed"):
        if getattr(receipt, field_name) != getattr(request, field_name):
            raise OuterExperimentGateError(f"backend receipt {field_name} mismatch")
    if include_inner and receipt.inner_fold_id != request.inner_fold_id:
        raise OuterExperimentGateError("backend receipt inner_fold_id mismatch")


def _fit_pos_weight(sessions: tuple[SessionTensors, ...], max_weight: float = 10.0) -> np.ndarray:
    if not sessions:
        raise OuterExperimentGateError("cannot fit pos_weight on an empty training partition")
    positives = np.zeros(sessions[0].labels.shape[-1], dtype=np.float64)
    totals = np.zeros_like(positives)
    for session in sessions:
        present = session.present > 0
        labels = session.labels[session.valid][:, present, :]
        positives += labels.sum(axis=(0, 1))
        totals += labels.shape[0] * labels.shape[1]
    positives = np.clip(positives, 1.0, None)
    return np.clip((totals - positives) / positives, 1.0, max_weight).astype(np.float32)


def _clone_sessions(sessions: tuple[SessionTensors, ...]) -> tuple[SessionTensors, ...]:
    return tuple(
        SessionTensors(
            session_id=session.session_id,
            features=session.features.copy(),
            labels=session.labels.copy(),
            vad=session.vad.copy(),
            valid=session.valid.copy(),
            present=session.present.copy(),
            group_id=session.group_id,
        )
        for session in sessions
    )


def _clone_normalizer(normalizer: FeatureNormalizer) -> FeatureNormalizer:
    return FeatureNormalizer(
        participant_mean=normalizer.participant_mean.copy(),
        participant_std=normalizer.participant_std.copy(),
        global_mean=normalizer.global_mean.copy(),
        global_std=normalizer.global_std.copy(),
    )


def _sessions_sha256(sessions: list[SessionTensors]) -> str:
    digest = hashlib.sha256(b"vap-session-tensors-v1")
    for session in sessions:
        for value in (session.session_id, session.group_id):
            if not isinstance(value, str):
                raise OuterExperimentGateError("master session identity/group must be strings")
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        for name in ("features", "labels", "vad", "valid", "present"):
            array = np.asarray(getattr(session, name))
            if array.dtype.hasobject:
                raise OuterExperimentGateError("master session arrays must not use object dtype")
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                raise OuterExperimentGateError("master session arrays must be finite")
            contiguous = np.ascontiguousarray(array)
            descriptor = json.dumps(
                {"name": name, "dtype": contiguous.dtype.str, "shape": contiguous.shape},
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(descriptor).to_bytes(8, "little"))
            digest.update(descriptor)
            digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _assert_master_sessions_unchanged(sessions: list[SessionTensors], expected_sha256: str) -> None:
    if _sessions_sha256(sessions) != expected_sha256:
        raise OuterExperimentGateError("backend mutated the runner-owned master session tensors")


def _verify_path_hashes(bindings: dict[str, str], role: str) -> None:
    for path_value, expected_sha256 in bindings.items():
        path = Path(path_value)
        if _capture_file_snapshot(path, require_single_link=True).sha256 != expected_sha256:
            raise OuterExperimentGateError(f"sealed {role} changed during backend execution")


def _statistics_record(
    scope_id: str,
    sessions: tuple[SessionTensors, ...],
    normalizer: FeatureNormalizer,
    pos_weight: np.ndarray,
) -> dict[str, object]:
    return {
        "scope_id": scope_id,
        "fit_session_ids": list(_session_ids(sessions)),
        "normalizer": normalizer.to_dict(),
        "pos_weight": np.asarray(pos_weight).tolist(),
    }


def _finite_vector(value: object, name: str, expected_length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != expected_length:
        raise OuterExperimentGateError(f"{name} must be a JSON array of length {expected_length}")
    numbers = []
    for item in value:
        numbers.append(_finite_number(item, name))
    return np.asarray(numbers, dtype=np.float64)


def _validate_statistics_record(
    value: object,
    expected_scope_id: str,
    sessions: tuple[SessionTensors, ...],
) -> None:
    expected_keys = {"scope_id", "fit_session_ids", "normalizer", "pos_weight"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise OuterExperimentGateError("statistics record must use exact version-2 fields")
    if value["scope_id"] != expected_scope_id:
        raise OuterExperimentGateError("statistics scope_id differs from the frozen fold")
    if value["fit_session_ids"] != list(_session_ids(sessions)):
        raise OuterExperimentGateError("statistics fit-session scope differs from training data")
    expected_normalizer = FeatureNormalizer.fit(list(sessions))
    expected_normalizer_payload = expected_normalizer.to_dict()
    normalizer = value["normalizer"]
    if not isinstance(normalizer, dict) or set(normalizer) != set(expected_normalizer_payload):
        raise OuterExperimentGateError("normalizer must use the exact four-array schema")
    for name, expected_values in expected_normalizer_payload.items():
        actual = _finite_vector(normalizer[name], f"normalizer.{name}", len(expected_values))
        expected = np.asarray(expected_values, dtype=np.float64)
        if not np.array_equal(actual, expected):
            raise OuterExperimentGateError(
                f"normalizer.{name} differs from a fresh training-scope recomputation"
            )
    expected_pos_weight = _fit_pos_weight(sessions).astype(np.float64)
    actual_pos_weight = _finite_vector(
        value["pos_weight"], "pos_weight", expected_pos_weight.shape[0]
    )
    if not np.array_equal(actual_pos_weight, expected_pos_weight):
        raise OuterExperimentGateError(
            "pos_weight differs from a fresh training-scope recomputation"
        )


def _d3_sensitivity_payload(
    d3_records: dict[str, dict[str, object]],
    conditions: tuple[OuterCondition, ...],
    *,
    profile: _CorpusProfile,
) -> dict[str, object]:
    session_ids = profile.session_ids
    paired = [
        session_id for session_id in session_ids if d3_records[session_id]["error_bound_ms"] <= 80.0
    ]
    expected_paired = [
        session_id for session_id in PAIRED_D3_80MS_SESSION_IDS if session_id in profile.mapping
    ]
    if paired != expected_paired:
        raise OuterExperimentGateError("fresh <=80 ms D3 membership differs from frozen plan")
    descriptive = [
        session_id for session_id in session_ids if d3_records[session_id]["error_bound_ms"] <= 40.0
    ]
    expected_descriptive = [
        session_id
        for session_id in D3_40MS_DESCRIPTIVE_SESSION_IDS
        if session_id in profile.mapping
    ]
    if descriptive != expected_descriptive:
        raise OuterExperimentGateError("fresh <=40 ms D3 membership differs from frozen plan")
    pose_condition_ids = [
        condition.condition_id for condition in conditions if _condition_uses_head_pose(condition)
    ]
    pose_feature_sets = sorted(
        {condition.feature_set for condition in conditions if _condition_uses_head_pose(condition)}
    )
    return {
        "decision_status": profile.d3_decision_status,
        "primary": {
            "session_ids": list(session_ids),
            "offset_policy": "published_self_checked_central_offset",
        },
        "paired_subset": {
            "rule": "exclude_error_bound_ms_strictly_greater_than_80",
            "threshold_ms": 80.0,
            "session_ids": paired,
            "requires_same_four_groups": True,
        },
        "pose_time_shift_metric_range": {
            "condition_ids": pose_condition_ids,
            "feature_sets": pose_feature_sets,
            "session_ids": list(session_ids),
            "per_session_error_bound_ms": {
                session_id: d3_records[session_id]["error_bound_ms"] for session_id in session_ids
            },
            "directions": ["minus_error_bound_ms", "plus_error_bound_ms"],
            "recalibration": False,
            "selection_use": "forbidden",
            "implementation_status": "blocked_until_versioned_pose_reassembly_exists",
        },
        "descriptive_only_subset": {
            "rule": "error_bound_ms_less_than_or_equal_to_40",
            "session_ids": descriptive,
            "reason": "retains_only_three_sessions_and_is_not_a_counted_paired_subset",
        },
    }


def _d3_provenance_specs(report: ReadinessReport) -> tuple[tuple[Path, str], ...]:
    specs: list[tuple[Path, str]] = []
    for artifact in report.source_artifacts:
        provenance_value = artifact.get("provenance_path")
        provenance_sha = artifact.get("provenance_sha256")
        if not isinstance(provenance_value, str) or not isinstance(provenance_sha, str):
            raise OuterExperimentGateError("real source lacks a bound provenance sidecar")
        # Keep the declared lexical path intact until the snapshot primitive has
        # lstat'ed it.  Path.resolve() here would follow a symlink/reparse point
        # before the immutable-file gate had a chance to reject it.
        provenance_path = _absolute_snapshot_path(Path(provenance_value))
        specs.append((provenance_path, provenance_sha))
    return tuple(specs)


def _d3_records_from_payloads(
    specs: tuple[tuple[Path, str], ...],
    payloads: dict[Path, dict[str, object]],
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for provenance_path, provenance_sha in specs:
        try:
            provenance = payloads[provenance_path]
        except KeyError as exc:
            raise OuterExperimentGateError(
                f"D3 provenance was not captured from the held file set: {provenance_path}"
            ) from exc
        session_id = _normalised_string(provenance.get("session_id"), "D3 session_id")
        group_id = _normalised_string(provenance.get("group_id"), "D3 group_id")
        summary = provenance.get("offset_quality")
        if not isinstance(summary, dict):
            raise OuterExperimentGateError("fresh provenance has no D3 offset_quality object")
        stream = _normalised_string(summary.get("stream"), "D3 stream")
        offset_s = _finite_number(summary.get("offset_s"), "D3 offset_s")
        error_bound_ms = _finite_number(summary.get("error_bound_ms"), "D3 error_bound_ms")
        if summary.get("self_check_ok") is not True:
            raise OuterExperimentGateError("fresh D3 self_check_ok must be true")
        status = summary.get("status")
        if status not in {"ok", "degraded"}:
            raise OuterExperimentGateError("fresh D3 status must be ok/degraded")
        if session_id in records:
            raise OuterExperimentGateError(f"duplicate D3 provenance for {session_id}")
        records[session_id] = {
            "session_id": session_id,
            "group_id": group_id,
            "stream": stream,
            "offset_s": offset_s,
            "status": status,
            "self_check_ok": True,
            "error_bound_ms": error_bound_ms,
            "provenance_path": str(provenance_path),
            "provenance_sha256": provenance_sha,
        }
    return dict(sorted(records.items()))


def _d3_records_from_report(report: ReadinessReport) -> dict[str, dict[str, object]]:
    specs = _d3_provenance_specs(report)
    payloads: dict[Path, dict[str, object]] = {}
    for provenance_path, provenance_sha in specs:
        provenance, provenance_snapshot = _read_json_with_snapshot(provenance_path)
        if provenance_snapshot.sha256 != provenance_sha:
            raise OuterExperimentGateError("fresh provenance hash differs from readiness")
        payloads[provenance_path] = provenance
    return _d3_records_from_payloads(specs, payloads)


def _validate_d3_records(
    value: object,
    source_bindings: list[dict[str, object]],
    *,
    profile: _CorpusProfile,
) -> dict[str, dict[str, object]]:
    session_ids = profile.session_ids
    if not isinstance(value, dict) or tuple(sorted(value)) != tuple(session_ids):
        raise OuterExperimentGateError("D3 records must exactly cover the corpus profile")
    provenance_pairs = {
        (binding["path"], binding["sha256"])
        for binding in source_bindings
        if binding["role"] == "assembled_provenance"
    }
    checked: dict[str, dict[str, object]] = {}
    expected_keys = {
        "session_id",
        "group_id",
        "stream",
        "offset_s",
        "status",
        "self_check_ok",
        "error_bound_ms",
        "provenance_path",
        "provenance_sha256",
    }
    for session_id in session_ids:
        record = value[session_id]
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise OuterExperimentGateError("each D3 record must use exact version-2 fields")
        if record["session_id"] != session_id:
            raise OuterExperimentGateError("D3 record key/session mismatch")
        if record["group_id"] != profile.mapping[session_id]:
            raise OuterExperimentGateError("D3 record group differs from frozen identity")
        if record["stream"] != "6cam":
            raise OuterExperimentGateError("D3 record must bind the physical 6cam stream")
        core = {
            "offset_s": _finite_number(record["offset_s"], "D3 offset_s"),
            "status": record["status"],
            "self_check_ok": record["self_check_ok"],
            "error_bound_ms": _finite_number(record["error_bound_ms"], "D3 error_bound_ms"),
        }
        if not _type_exact_equal(core, D3_FROZEN_VALUES[session_id]):
            raise OuterExperimentGateError(
                f"D3 values for {session_id} differ from the pre-outcome freeze"
            )
        provenance_path = _normalised_string(record["provenance_path"], "D3 provenance_path")
        provenance_sha = _sha256_string(record["provenance_sha256"], "D3 provenance_sha256")
        if (provenance_path, provenance_sha) not in provenance_pairs:
            raise OuterExperimentGateError("D3 record is not bound to a source provenance file")
        checked[session_id] = dict(record)
    return checked


def _require_ready(report: ReadinessReport, condition: OuterCondition) -> None:
    if report.data_kind != "real" or not report.structural_ready:
        raise OuterExperimentGateError("counted outer protocol accepts audited real data only")
    if not report.target_ready:
        raise OuterExperimentGateError(
            f"condition {condition.condition_id!r} has no accepted gold target: "
            f"{list(report.target_blockers)}"
        )
    if report.feature_set != condition.feature_set or report.selected_feature_ready is not True:
        raise OuterExperimentGateError(
            f"condition {condition.condition_id!r} selected inputs are not ready: "
            f"{list(report.selected_feature_blockers)}"
        )


def _registration_payload(registration: _BackendRegistration) -> dict[str, object]:
    return {
        "backend_id": registration.backend_id,
        "backend_version": registration.backend_version,
        "result_role": registration.result_role,
        "counted_eligible": registration.counted_eligible,
    }


def _validate_registration_payload(value: object) -> dict[str, object]:
    expected_keys = {"backend_id", "backend_version", "result_role", "counted_eligible"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise OuterExperimentGateError("backend registration uses malformed fields")
    if not isinstance(value["counted_eligible"], bool):
        raise OuterExperimentGateError("backend counted_eligible must be an exact boolean")
    return {
        "backend_id": _normalised_string(value["backend_id"], "backend_id"),
        "backend_version": _exact_positive_int(value["backend_version"], "backend_version"),
        "result_role": _normalised_string(value["result_role"], "backend result_role"),
        "counted_eligible": value["counted_eligible"],
    }


def _build_backend_policy(production_backend: dict[str, object] | None) -> dict[str, object]:
    selected = None
    status = "not_registered_blocker"
    if production_backend is not None:
        requested = _validate_registration_payload(production_backend)
        matches = [
            registration
            for registration in _PRODUCTION_BACKEND_REGISTRY
            if _registration_payload(registration) == requested
        ]
        if len(matches) != 1:
            raise OuterExperimentGateError(
                "production backend is not present in the code-owned registry"
            )
        selected = requested
        status = "registered"
    return {
        "registry_version": 1,
        "smoke_backend": _registration_payload(_SMOKE_BACKEND_REGISTRATION),
        "production_backend": selected,
        "production_status": status,
    }


def _validate_backend_policy(value: object) -> dict[str, object]:
    expected_keys = {
        "registry_version",
        "smoke_backend",
        "production_backend",
        "production_status",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise OuterExperimentGateError("backend_policy must use exact version-2 fields")
    _require_exact_int_field(value, "registry_version", 1)
    if not _type_exact_equal(
        value["smoke_backend"], _registration_payload(_SMOKE_BACKEND_REGISTRATION)
    ):
        raise OuterExperimentGateError("smoke backend policy differs from code-owned registry")
    rebuilt = _build_backend_policy(value["production_backend"])
    if not _type_exact_equal(value, rebuilt):
        raise OuterExperimentGateError("backend policy differs from the code-owned registry")
    return rebuilt


def _resolve_backend_registration(
    backend: OuterTrainingBackend, run_manifest: dict[str, object]
) -> _BackendRegistration:
    policy = _validate_backend_policy(run_manifest["backend_policy"])
    if isinstance(backend, BackendContractSmokeBackend):
        return _SMOKE_BACKEND_REGISTRATION
    for registration in _PRODUCTION_BACKEND_REGISTRY:
        if type(backend) is registration.backend_type:
            if not _type_exact_equal(
                policy["production_backend"], _registration_payload(registration)
            ):
                raise OuterExperimentGateError(
                    "production backend is not the one frozen in the run manifest"
                )
            return registration
    raise OuterExperimentGateError("backend type is absent from the code-owned registry")


def _registration_from_result(
    payload: dict[str, object], run_manifest: dict[str, object]
) -> _BackendRegistration:
    claimed = _validate_registration_payload(payload.get("backend_registration"))
    smoke = _registration_payload(_SMOKE_BACKEND_REGISTRATION)
    if _type_exact_equal(claimed, smoke):
        return _SMOKE_BACKEND_REGISTRATION
    policy = _validate_backend_policy(run_manifest["backend_policy"])
    for registration in _PRODUCTION_BACKEND_REGISTRY:
        registration_value = _registration_payload(registration)
        if _type_exact_equal(claimed, registration_value) and _type_exact_equal(
            policy["production_backend"], registration_value
        ):
            return registration
    raise OuterExperimentGateError("result backend is absent from the code-owned registry")


def _remaining_execution_gates(
    registration: _BackendRegistration,
    condition: OuterCondition,
    run_manifest: dict[str, object],
) -> list[str]:
    gates = []
    if not registration.counted_eligible:
        gates.append("backend_contract_smoke_not_counted")
    status = run_manifest["implementation_status"]
    for key in ("production_torch_backend", "environment_capture", "run_index"):
        if status[key] != "implemented":
            gates.append(key)
    if (
        _condition_uses_head_pose(condition)
        and status["pose_time_shift_reassembly"] != "implemented"
    ):
        gates.append("pose_time_shift_reassembly")
    return gates


def _authorize_result_role(
    registration: _BackendRegistration,
    condition: OuterCondition,
    run_manifest: dict[str, object],
) -> None:
    if registration.counted_eligible:
        blockers = _remaining_execution_gates(registration, condition, run_manifest)
        if blockers:
            raise OuterExperimentGateError(
                f"production counted result is blocked by implementation gates: {blockers}"
            )


def _validate_label_config(value: object) -> dict[str, object]:
    payload = _strict_json_mapping(value, "label_config")
    expected_keys = {
        "frame_rate_hz",
        "bin_edges_s",
        "active_ratio",
        "now_bins",
        "future_bins",
    }
    if set(payload) != expected_keys:
        raise OuterExperimentGateError("label_config must use the exact version-1 fields")
    frame_rate = _exact_positive_int(payload["frame_rate_hz"], "label frame_rate_hz")

    def finite_numbers(raw: object, name: str) -> tuple[float, ...]:
        if not isinstance(raw, list) or not raw:
            raise OuterExperimentGateError(f"{name} must be a non-empty JSON array")
        result = []
        for item in raw:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise OuterExperimentGateError(f"{name} must contain only finite numbers")
            number = float(item)
            if not math.isfinite(number):
                raise OuterExperimentGateError(f"{name} must contain only finite numbers")
            result.append(number)
        return tuple(result)

    def exact_indices(raw: object, name: str) -> tuple[int, ...]:
        if not isinstance(raw, list) or not raw:
            raise OuterExperimentGateError(f"{name} must be a non-empty JSON array")
        return tuple(_exact_nonnegative_int(item, name) for item in raw)

    edges = finite_numbers(payload["bin_edges_s"], "label bin_edges_s")
    active_values = finite_numbers([payload["active_ratio"]], "label active_ratio")
    config = VAPLabelConfig(
        frame_rate_hz=frame_rate,
        bin_edges_s=edges,
        active_ratio=active_values[0],
        now_bins=exact_indices(payload["now_bins"], "label now_bins"),
        future_bins=exact_indices(payload["future_bins"], "label future_bins"),
    )
    canonical = _strict_json_mapping(asdict(config), "label_config")
    if not _type_exact_equal(payload, canonical):
        raise OuterExperimentGateError("label_config values/types are not canonical")
    return canonical


def _validate_readiness_bindings(
    readiness: dict[str, object], split_manifest: dict[str, object]
) -> None:
    mapping = split_manifest["source_session_to_group"]
    expected_sessions = sorted(mapping)
    expected_groups = sorted(set(mapping.values()))
    source_pairs = {
        (binding["path"], binding["sha256"]) for binding in split_manifest["source_bindings"]
    }
    conditions = {
        condition.condition_id: condition
        for condition in (_condition_from_payload(value) for value in split_manifest["conditions"])
    }
    expected_keys = {
        "data_kind",
        "n_rows",
        "session_ids",
        "group_ids",
        "source_artifacts",
        "target_ready",
        "feature_set",
        "selected_feature_ready",
        "d3_records",
    }
    expected_artifact_keys = {
        "data_path",
        "data_sha256",
        "provenance_path",
        "provenance_sha256",
        "artifact_kind",
        "format_version",
        "identity_authority",
        "identity_inventory_sha256",
    }
    reference_n_rows: int | None = None
    reference_artifacts: list[dict[str, object]] | None = None
    for condition_id, value in readiness.items():
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise OuterExperimentGateError(
                "each readiness binding must use the exact version-1 fields"
            )
        if value["data_kind"] != "real":
            raise OuterExperimentGateError("counted readiness must bind real data")
        n_rows = _exact_positive_int(value["n_rows"], "readiness n_rows")
        if reference_n_rows is None:
            reference_n_rows = n_rows
        elif n_rows != reference_n_rows:
            raise OuterExperimentGateError("paired conditions bind different row counts")
        if value["session_ids"] != expected_sessions or value["group_ids"] != expected_groups:
            raise OuterExperimentGateError(
                "readiness session/group scope differs from the frozen split"
            )
        if value["target_ready"] is not True or value["selected_feature_ready"] is not True:
            raise OuterExperimentGateError(
                "counted readiness must have accepted targets and inputs"
            )
        if value["feature_set"] != conditions[condition_id].feature_set:
            raise OuterExperimentGateError("readiness feature_set differs from its condition")
        artifacts = value["source_artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise OuterExperimentGateError(
                "readiness must bind at least one audited source artifact"
            )
        if reference_artifacts is None:
            reference_artifacts = artifacts
        elif not _type_exact_equal(artifacts, reference_artifacts):
            raise OuterExperimentGateError(
                "paired conditions bind different audited source metadata"
            )
        readiness_pairs: list[tuple[str, str]] = []
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != expected_artifact_keys:
                raise OuterExperimentGateError("malformed readiness source artifact")
            data_path = artifact.get("data_path")
            data_sha = artifact.get("data_sha256")
            if not isinstance(data_path, str) or not isinstance(data_sha, str):
                raise OuterExperimentGateError("readiness data source path/hash are required")
            _sha256_string(data_sha, "readiness data_sha256")
            _normalised_string(artifact.get("artifact_kind"), "readiness artifact_kind")
            _exact_positive_int(artifact.get("format_version"), "readiness format_version")
            _normalised_string(artifact.get("identity_authority"), "readiness identity_authority")
            _sha256_string(
                artifact.get("identity_inventory_sha256"),
                "readiness identity_inventory_sha256",
            )
            readiness_pairs.append((data_path, data_sha))
            provenance_path = artifact.get("provenance_path")
            provenance_sha = artifact.get("provenance_sha256")
            if provenance_path is not None or provenance_sha is not None:
                if not isinstance(provenance_path, str) or not isinstance(provenance_sha, str):
                    raise OuterExperimentGateError(
                        "readiness provenance path/hash must both be strings or both null"
                    )
                _sha256_string(provenance_sha, "readiness provenance_sha256")
                readiness_pairs.append((provenance_path, provenance_sha))
        if (
            len(readiness_pairs) != len(set(readiness_pairs))
            or set(readiness_pairs) != source_pairs
        ):
            raise OuterExperimentGateError(
                "readiness artifacts do not exactly match the split source bindings"
            )
        if not _type_exact_equal(value["d3_records"], split_manifest["d3_records"]):
            raise OuterExperimentGateError("readiness D3 records differ from the split")
    profile = _corpus_profile_from_protocol_id(split_manifest["protocol_id"])
    if profile.expected_n_rows is not None and reference_n_rows != profile.expected_n_rows:
        raise OuterExperimentGateError(
            f"deadline corpus must contain exactly {profile.expected_n_rows} rows"
        )


def _readiness_binding(report: ReadinessReport) -> dict[str, object]:
    return {
        "data_kind": report.data_kind,
        "n_rows": report.n_rows,
        "session_ids": list(report.session_ids),
        "group_ids": list(report.group_ids),
        "source_artifacts": [dict(item) for item in report.source_artifacts],
        "target_ready": report.target_ready,
        "feature_set": report.feature_set,
        "selected_feature_ready": report.selected_feature_ready,
        "d3_records": _d3_records_from_report(report),
    }


def _source_snapshot(report: ReadinessReport) -> dict[str, object]:
    return {
        "n_rows": report.n_rows,
        "session_ids": report.session_ids,
        "group_ids": report.group_ids,
        "source_artifacts": report.source_artifacts,
    }


def _bindings_from_report(report: ReadinessReport) -> list[dict[str, object]]:
    bindings: list[dict[str, object]] = []
    for index, artifact in enumerate(report.source_artifacts):
        data_path = Path(str(artifact["data_path"])).resolve()
        bindings.append(
            {
                "artifact_id": f"source_{index:03d}_data",
                "role": "assembled_csv",
                "path": str(data_path),
                "sha256": artifact["data_sha256"],
                "size_bytes": data_path.stat().st_size,
            }
        )
        if artifact.get("provenance_path") is not None:
            provenance = Path(str(artifact["provenance_path"])).resolve()
            bindings.append(
                {
                    "artifact_id": f"source_{index:03d}_provenance",
                    "role": "assembled_provenance",
                    "path": str(provenance),
                    "sha256": artifact["provenance_sha256"],
                    "size_bytes": provenance.stat().st_size,
                }
            )
    return _validate_source_bindings_payload(bindings)


def _validate_source_bindings_payload(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise OuterExperimentGateError("source_bindings must be a non-empty JSON array")
    result: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "artifact_id",
            "role",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise OuterExperimentGateError("each source binding must have the exact v1 fields")
        artifact_id = _normalised_string(item["artifact_id"], "artifact_id")
        role = _normalised_string(item["role"], "source role")
        path = _normalised_string(item["path"], "source path")
        path_object = Path(path)
        if not path_object.is_absolute() or path_object != path_object.resolve():
            raise OuterExperimentGateError("source binding paths must be normalized absolute paths")
        digest = _sha256_string(item["sha256"], "source sha256")
        size = _exact_nonnegative_int(item["size_bytes"], "source size_bytes")
        normalized_path = os.path.normcase(path)
        if artifact_id in seen_ids or normalized_path in seen_paths:
            raise OuterExperimentGateError("source binding IDs and paths must be unique")
        seen_ids.add(artifact_id)
        seen_paths.add(normalized_path)
        result.append(
            {
                "artifact_id": artifact_id,
                "role": role,
                "path": path,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return result


def _verify_source_bindings(bindings: object) -> None:
    _snapshot_source_bindings(bindings)


def _snapshot_source_bindings(bindings: object) -> tuple[_FileSnapshot, ...]:
    snapshots: list[_FileSnapshot] = []
    for binding in _validate_source_bindings_payload(bindings):
        path = Path(binding["path"])
        snapshot = _capture_file_snapshot(path)
        if snapshot.size_bytes != binding["size_bytes"] or snapshot.sha256 != binding["sha256"]:
            raise OuterExperimentGateError(f"bound source bytes changed: {path}")
        snapshots.append(snapshot)
    return tuple(snapshots)


def _validate_conditions(conditions: tuple[OuterCondition, ...]) -> tuple[OuterCondition, ...]:
    if not isinstance(conditions, tuple) or not conditions:
        raise OuterExperimentGateError("conditions must be a non-empty tuple")
    seen: set[str] = set()
    checked = []
    for condition in conditions:
        if not isinstance(condition, OuterCondition):
            raise OuterExperimentGateError("every condition must be an OuterCondition")
        condition_id = _normalised_string(condition.condition_id, "condition_id")
        feature_set = _normalised_string(condition.feature_set, "feature_set")
        encoder = _normalised_string(condition.encoder, "encoder")
        if condition_id in seen:
            raise OuterExperimentGateError("condition_id values must be unique")
        get_feature_set(feature_set)
        if encoder not in {"transformer", "gru"}:
            raise OuterExperimentGateError("encoder must be 'transformer' or 'gru'")
        seen.add(condition_id)
        checked.append(OuterCondition(condition_id, feature_set, encoder))
    return tuple(checked)


def _condition_from_payload(value: object) -> OuterCondition:
    if not isinstance(value, dict) or set(value) != {"condition_id", "feature_set", "encoder"}:
        raise OuterExperimentGateError("condition entries must use the exact version-2 schema")
    return OuterCondition(value["condition_id"], value["feature_set"], value["encoder"])


def _condition_from_manifest(manifest: dict[str, object], condition_id: str) -> OuterCondition:
    condition_id = _normalised_string(condition_id, "condition_id")
    for value in manifest["conditions"]:
        condition = _condition_from_payload(value)
        if condition.condition_id == condition_id:
            return condition
    raise OuterExperimentGateError(f"condition is not frozen in the protocol: {condition_id}")


def _condition_uses_head_pose(condition: OuterCondition) -> bool:
    return any(
        column.startswith("P") and column.endswith("_head_yaw_deg")
        for column in active_feature_columns(condition.feature_set)
    )


def _variant_from_manifest(
    manifest: dict[str, object], analysis_variant: object
) -> dict[str, object]:
    name = _normalised_string(analysis_variant, "analysis_variant")
    if name not in ANALYSIS_VARIANTS:
        raise OuterExperimentGateError(f"analysis_variant must be one of {ANALYSIS_VARIANTS}")
    return manifest["analysis_variants"][name]


def _fold_from_variant(variant: dict[str, object], fold_id: object) -> dict[str, object]:
    fold_id = _normalised_string(fold_id, "outer_fold_id")
    for fold in variant["outer_folds"]:
        if fold["outer_fold_id"] == fold_id:
            return fold
    raise OuterExperimentGateError(f"outer fold is not frozen in the protocol: {fold_id}")


def _build_logo_folds(mapping: dict[str, str]) -> list[dict[str, object]]:
    groups = tuple(sorted(set(mapping.values())))
    if len(groups) != FROZEN_GROUP_COUNT:
        raise OuterExperimentGateError("nested LOGO variants must contain exactly four groups")
    folds: list[dict[str, object]] = []
    for outer_number, outer_group in enumerate(groups, start=1):
        outer_train_groups = tuple(group for group in groups if group != outer_group)
        inner_folds = []
        for inner_number, validation_group in enumerate(outer_train_groups, start=1):
            inner_train_groups = tuple(
                group for group in outer_train_groups if group != validation_group
            )
            inner_folds.append(
                {
                    "inner_fold_id": f"inner_{inner_number:02d}",
                    "train_group_ids": list(inner_train_groups),
                    "validation_group_ids": [validation_group],
                    "train_session_ids": _sessions_for_groups(mapping, inner_train_groups),
                    "validation_session_ids": _sessions_for_groups(mapping, (validation_group,)),
                }
            )
        folds.append(
            {
                "outer_fold_id": f"outer_{outer_number:02d}",
                "outer_train_group_ids": list(outer_train_groups),
                "outer_test_group_ids": [outer_group],
                "outer_train_session_ids": _sessions_for_groups(mapping, outer_train_groups),
                "outer_test_session_ids": _sessions_for_groups(mapping, (outer_group,)),
                "inner_folds": inner_folds,
            }
        )
    return folds


def _session_group_map(frame_df: object) -> dict[str, str]:
    if not hasattr(frame_df, "groupby") or "group_id" not in frame_df.columns:
        raise OuterExperimentGateError("real source must expose provenance group_id")
    mapping: dict[str, str] = {}
    for session_id, group in frame_df.groupby("session_id", sort=True):
        values = group["group_id"].drop_duplicates().tolist()
        if len(values) != 1:
            raise OuterExperimentGateError(f"session {session_id!r} has inconsistent group_id")
        mapping[str(session_id)] = str(values[0])
    return _validate_session_group_map(mapping)


def _validate_session_group_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise OuterExperimentGateError("session_to_group must be a non-empty JSON object")
    mapping: dict[str, str] = {}
    for session_id, group_id in value.items():
        session = _normalised_string(session_id, "session_id")
        group = _normalised_string(group_id, "group_id")
        mapping[session] = group
    if len(mapping) != len(value):
        raise OuterExperimentGateError("session_to_group contains duplicate normalized IDs")
    return mapping


def _checked_excluded_session_ids(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or any(type(item) is not str for item in value):
        raise OuterExperimentGateError("excluded_session_ids must be an exact string tuple")
    if value == ():
        return ()
    expected = tuple(DEADLINE_CORPUS_EXCLUSIONS)
    if value != expected:
        raise OuterExperimentGateError(
            f"deadline exclusion must be exactly {list(expected)} in canonical order"
        )
    return value


def _corpus_profile_from_mapping(mapping: dict[str, str]) -> _CorpusProfile:
    if mapping == PRIMARY_SESSION_TO_GROUP:
        return _PRIMARY_CORPUS_PROFILE
    if mapping == DEADLINE_SESSION_TO_GROUP:
        return _DEADLINE_CORPUS_PROFILE
    raise OuterExperimentGateError(
        "session/group binding is neither the frozen ten-scene corpus nor the "
        "code-owned deadline-eight corpus"
    )


def _corpus_profile_from_protocol_id(value: object) -> _CorpusProfile:
    if value == PROTOCOL_ID and type(value) is str:
        return _PRIMARY_CORPUS_PROFILE
    if value == DEADLINE_PROTOCOL_ID and type(value) is str:
        return _DEADLINE_CORPUS_PROFILE
    raise OuterExperimentGateError("protocol_id does not select a code-owned corpus profile")


def _verify_loaded_sessions(
    sessions: list[SessionTensors], split_manifest: dict[str, object]
) -> None:
    if not sessions:
        raise OuterExperimentGateError("loaded source contains no sessions")
    loaded: dict[str, str] = {}
    for session in sessions:
        session_id = _normalised_string(session.session_id, "loaded session_id")
        group_id = _normalised_string(session.group_id, "loaded group_id")
        if session_id in loaded:
            raise OuterExperimentGateError(f"duplicate loaded session: {session_id}")
        loaded[session_id] = group_id
    if loaded != split_manifest["source_session_to_group"]:
        raise OuterExperimentGateError("loaded session/group binding differs from split manifest")


def _select_sessions(
    by_id: dict[str, SessionTensors], session_ids: object
) -> tuple[SessionTensors, ...]:
    ids = _string_list(session_ids, "partition session_ids")
    try:
        return tuple(by_id[session_id] for session_id in ids)
    except KeyError as exc:
        raise OuterExperimentGateError(
            f"partition references missing session: {exc.args[0]}"
        ) from exc


def _session_ids(sessions: tuple[SessionTensors, ...]) -> tuple[str, ...]:
    return tuple(session.session_id for session in sessions)


def _sessions_for_groups(mapping: dict[str, str], groups: tuple[str, ...]) -> list[str]:
    group_set = set(groups)
    return sorted(session_id for session_id, group_id in mapping.items() if group_id in group_set)


def _require_session_list(owner: dict[str, object], key: str, expected: list[str]) -> None:
    if owner.get(key) != expected:
        raise OuterExperimentGateError(f"{key} does not match the exact group/session binding")


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise OuterExperimentGateError(f"{name} must be a non-empty JSON array")
    values = [_normalised_string(item, name) for item in value]
    if len(values) != len(set(values)):
        raise OuterExperimentGateError(f"{name} must not contain duplicates")
    return values


def _normalised_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OuterExperimentGateError(f"{name} must be a non-empty normalized string")
    return value


def _exact_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OuterExperimentGateError(f"{name} must be an exact positive integer")
    return value


def _exact_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OuterExperimentGateError(f"{name} must be an exact non-negative integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OuterExperimentGateError(f"{name} must be a finite JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise OuterExperimentGateError(f"{name} must be a finite JSON number")
    return number


def _require_exact_int_field(owner: dict[str, object], key: str, expected: int) -> None:
    value = owner.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise OuterExperimentGateError(f"{key} must be the exact integer {expected}")


def _require_exact(owner: dict[str, object], key: str, expected: object) -> None:
    if owner.get(key) != expected or type(owner.get(key)) is not type(expected):
        raise OuterExperimentGateError(f"{key} must be exactly {expected!r}")


def _strict_json_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise OuterExperimentGateError(f"{name} must be a JSON object")
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise OuterExperimentGateError(f"{name} must contain strict JSON data") from exc
    if not isinstance(decoded, dict):
        raise OuterExperimentGateError(f"{name} must be a JSON object")
    return decoded


def _type_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _type_exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _type_exact_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _sha256_string(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OuterExperimentGateError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_SNAPSHOT_CHUNK_BYTES = 1024 * 1024


def _absolute_snapshot_path(path: Path) -> Path:
    # Do not call Path.resolve(): it follows a reparse point before we can reject it.
    return Path(os.path.abspath(os.fspath(path)))


def _is_reparse_stat(stat_result: os.stat_result) -> bool:
    return bool(getattr(stat_result, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


def _reject_non_regular_or_reparse(path: Path, stat_result: os.stat_result) -> None:
    if not stat_module.S_ISREG(stat_result.st_mode) or _is_reparse_stat(stat_result):
        raise OuterExperimentGateError(f"immutable file must be a regular non-reparse file: {path}")


def _windows_extended_path(path: Path) -> str:
    value = str(path)
    if value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def _open_snapshot_fd(path: Path) -> int:
    if os.name != "nt":
        # O_NONBLOCK prevents an lstat->open replacement with a FIFO/device from
        # hanging the gate.  The opened descriptor is independently required to
        # be a regular file before any read occurs.
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        return os.open(path, flags)

    import ctypes
    import msvcrt
    from ctypes import wintypes

    generic_read = 0x80000000
    file_share_read = 0x00000001
    open_existing = 3
    file_flag_sequential_scan = 0x08000000
    file_flag_open_reparse_point = 0x00200000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    handle = create_file(
        _windows_extended_path(path),
        generic_read,
        file_share_read,
        None,
        open_existing,
        file_flag_sequential_scan | file_flag_open_reparse_point,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle in (None, invalid_handle):
        error = ctypes.get_last_error()
        raise OSError(error, ctypes.FormatError(error), str(path))
    raw_handle = int(handle)
    try:
        return msvcrt.open_osfhandle(
            raw_handle,
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
    except BaseException as exc:
        if not close_handle(wintypes.HANDLE(raw_handle)):
            error = ctypes.get_last_error()
            exc.add_note(
                f"CreateFileW handle cleanup also failed: [{error}] {ctypes.FormatError(error)}"
            )
        raise


def _windows_change_time_and_attributes(fd: int) -> tuple[int, int]:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class FileBasicInfo(ctypes.Structure):
        _fields_ = [
            ("CreationTime", ctypes.c_longlong),
            ("LastAccessTime", ctypes.c_longlong),
            ("LastWriteTime", ctypes.c_longlong),
            ("ChangeTime", ctypes.c_longlong),
            ("FileAttributes", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_info = kernel32.GetFileInformationByHandleEx
    get_info.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    get_info.restype = wintypes.BOOL
    info = FileBasicInfo()
    raw_handle = msvcrt.get_osfhandle(fd)
    if not get_info(
        wintypes.HANDLE(raw_handle),
        0,  # FileBasicInfo
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        error = ctypes.get_last_error()
        raise OSError(error, ctypes.FormatError(error))
    return int(info.ChangeTime), int(info.FileAttributes)


def _open_handle_state(fd: int) -> _OpenHandleState:
    stat_result = os.fstat(fd)
    if os.name == "nt":
        change_time, attributes = _windows_change_time_and_attributes(fd)
    else:
        change_time = stat_result.st_ctime_ns
        attributes = getattr(stat_result, "st_file_attributes", 0)
    return _OpenHandleState(
        mode=stat_result.st_mode,
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        link_count=stat_result.st_nlink,
        size_bytes=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        ctime_ns=stat_result.st_ctime_ns,
        change_time_token=change_time,
        file_attributes=attributes,
    )


def _hash_open_handle(
    fd: int,
    *,
    capture_bytes: bool,
) -> tuple[str, bytes | None]:
    """Hash one already-open descriptor; optionally retain the exact hashed bytes."""
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture_bytes else None
    while True:
        block = os.read(fd, _SNAPSHOT_CHUNK_BYTES)
        if not block:
            break
        digest.update(block)
        if chunks is not None:
            chunks.append(block)
    return digest.hexdigest(), b"".join(chunks) if chunks is not None else None


def _path_identity_matches_handle(
    path_stat: os.stat_result,
    handle_state: _OpenHandleState,
) -> bool:
    return path_stat.st_dev == handle_state.device and path_stat.st_ino == handle_state.inode


def _path_state_matches_handle(
    path_stat: os.stat_result,
    handle_state: _OpenHandleState,
) -> bool:
    common = (
        _path_identity_matches_handle(path_stat, handle_state)
        and stat_module.S_IFMT(path_stat.st_mode) == stat_module.S_IFMT(handle_state.mode)
        and path_stat.st_nlink == handle_state.link_count
        and path_stat.st_size == handle_state.size_bytes
        and path_stat.st_mtime_ns == handle_state.mtime_ns
    )
    # Windows path stat and CRT fstat expose different historical meanings for
    # st_ctime.  The native handle ChangeTime token is compared before/after
    # hashing; only POSIX can compare path/handle ctime directly.
    return common and (os.name == "nt" or path_stat.st_ctime_ns == handle_state.ctime_ns)


def _ancestor_guard_states(
    path: Path,
) -> tuple[tuple[Path, int, int, int, int, int], ...]:
    """Capture lexical ancestor identities without following a reparse point."""
    states: list[tuple[Path, int, int, int, int, int]] = []
    for ancestor in path.parents:
        stat_result = ancestor.lstat()
        if (
            stat_module.S_ISLNK(stat_result.st_mode)
            or not stat_module.S_ISDIR(stat_result.st_mode)
            or _is_reparse_stat(stat_result)
        ):
            raise OuterExperimentGateError(
                f"immutable file ancestor must be a non-reparse directory: {ancestor}"
            )
        states.append(
            (
                ancestor,
                stat_result.st_dev,
                stat_result.st_ino,
                # Opening descendants can cause filesystem-dependent directory
                # timestamp churn on Windows.  WRITE/DELETE sharing is already
                # denied there; POSIX retains timestamp tokens to expose an ABA
                # directory-entry swap that open descriptors cannot prevent.
                0 if os.name == "nt" else stat_result.st_mtime_ns,
                0 if os.name == "nt" else stat_result.st_ctime_ns,
                getattr(stat_result, "st_file_attributes", 0),
            )
        )
    return tuple(states)


def _snapshot_from_state(path: Path, digest: str, state: _OpenHandleState) -> _FileSnapshot:
    return _FileSnapshot(
        path=path,
        sha256=digest,
        mode=state.mode,
        size_bytes=state.size_bytes,
        device=state.device,
        inode=state.inode,
        link_count=state.link_count,
        mtime_ns=state.mtime_ns,
        ctime_ns=state.ctime_ns,
        change_time_token=state.change_time_token,
        file_attributes=state.file_attributes,
    )


def _state_matches_snapshot(state: _OpenHandleState, snapshot: _FileSnapshot) -> bool:
    return (
        state.mode == snapshot.mode
        and state.device == snapshot.device
        and state.inode == snapshot.inode
        and state.link_count == snapshot.link_count
        and state.size_bytes == snapshot.size_bytes
        and state.mtime_ns == snapshot.mtime_ns
        and state.ctime_ns == snapshot.ctime_ns
        and state.change_time_token == snapshot.change_time_token
        and state.file_attributes == snapshot.file_attributes
    )


def _close_fd_after_failure(fd: int, failure: BaseException) -> None:
    try:
        os.close(fd)
    except OSError as close_error:
        failure.add_note(f"snapshot descriptor cleanup also failed: {close_error}")


def _close_held_snapshot_handles(
    handles: list[_HeldSnapshotHandle],
    failure: BaseException | None,
) -> None:
    close_errors: list[OSError] = []
    for held in reversed(handles):
        try:
            os.close(held.fd)
        except OSError as exc:
            close_errors.append(exc)
    if not close_errors:
        return
    if failure is not None:
        for close_error in close_errors:
            failure.add_note(f"snapshot descriptor cleanup also failed: {close_error}")
        return
    raise OuterExperimentGateError("cannot close immutable-file snapshot handle") from (
        close_errors[0]
    )


def _open_held_snapshot_handle(
    path: Path,
    *,
    expected: _FileSnapshot | None,
    require_single_link: bool,
) -> _HeldSnapshotHandle:
    path = _absolute_snapshot_path(path)
    fd: int | None = None
    try:
        ancestors = _ancestor_guard_states(path)
        path_before = path.lstat()
        _reject_non_regular_or_reparse(path, path_before)
        fd = _open_snapshot_fd(path)
        opening_state = _open_handle_state(fd)
        if (
            not stat_module.S_ISREG(opening_state.mode)
            or opening_state.file_attributes & _FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise OuterExperimentGateError(
                f"opened immutable-file handle is not a regular non-reparse file: {path}"
            )
        if not _path_state_matches_handle(path_before, opening_state):
            raise OuterExperimentGateError(f"file changed while being opened: {path}")
        if _ancestor_guard_states(path) != ancestors:
            raise OuterExperimentGateError(
                f"immutable-file ancestor changed while being opened: {path}"
            )
        if require_single_link and opening_state.link_count != 1:
            raise OuterExperimentGateError(
                f"role artifact must have exactly one physical link: {path}"
            )
        if expected is not None:
            if path != expected.path or not _state_matches_snapshot(opening_state, expected):
                raise OuterExperimentGateError(
                    f"validated file changed before final return: {path}"
                )
        held = _HeldSnapshotHandle(
            path=path,
            fd=fd,
            opening_state=opening_state,
            ancestor_guard=ancestors,
            expected=expected,
            require_single_link=require_single_link,
        )
        fd = None
        return held
    except OuterExperimentGateError as exc:
        if fd is not None:
            _close_fd_after_failure(fd, exc)
        raise
    except OSError as exc:
        if fd is not None:
            _close_fd_after_failure(fd, exc)
        raise OuterExperimentGateError(f"cannot open immutable-file snapshot: {path}") from exc
    except BaseException as exc:
        if fd is not None:
            _close_fd_after_failure(fd, exc)
        raise


def _verify_held_path_and_state(
    held: _HeldSnapshotHandle,
    state: _OpenHandleState,
) -> None:
    if not stat_module.S_ISREG(state.mode) or (
        state.file_attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise OuterExperimentGateError(
            f"held immutable-file handle stopped naming a regular file: {held.path}"
        )
    if state != held.opening_state:
        raise OuterExperimentGateError(f"held immutable file changed: {held.path}")
    try:
        path_state = held.path.lstat()
        _reject_non_regular_or_reparse(held.path, path_state)
        if not _path_state_matches_handle(path_state, state):
            raise OuterExperimentGateError(
                f"path no longer names the held physical file: {held.path}"
            )
        if _ancestor_guard_states(held.path) != held.ancestor_guard:
            raise OuterExperimentGateError(
                f"immutable-file ancestor changed while handles were held: {held.path}"
            )
    except OuterExperimentGateError:
        raise
    except OSError as exc:
        raise OuterExperimentGateError(
            f"cannot revalidate held immutable-file path: {held.path}"
        ) from exc
    if held.require_single_link and state.link_count != 1:
        raise OuterExperimentGateError(
            f"role artifact must have exactly one physical link: {held.path}"
        )
    if held.expected is not None and not _state_matches_snapshot(state, held.expected):
        raise OuterExperimentGateError(f"validated file changed before final return: {held.path}")


def _snapshot_held_handle(
    held: _HeldSnapshotHandle,
    *,
    capture_bytes: bool,
) -> tuple[_FileSnapshot, bytes | None]:
    before = _open_handle_state(held.fd)
    _verify_held_path_and_state(held, before)
    digest, captured = _hash_open_handle(held.fd, capture_bytes=capture_bytes)
    after = _open_handle_state(held.fd)
    if before != after:
        raise OuterExperimentGateError(
            f"held immutable file changed while it was being hashed: {held.path}"
        )
    _verify_held_path_and_state(held, after)
    snapshot = _snapshot_from_state(held.path, digest, after)
    if held.expected is not None and snapshot != held.expected:
        raise OuterExperimentGateError(f"validated file changed before final return: {held.path}")
    return snapshot, captured


def _final_check_held_handle(held: _HeldSnapshotHandle) -> None:
    _verify_held_path_and_state(held, _open_handle_state(held.fd))


def _snapshot_open_file(
    path: Path,
    *,
    capture_bytes: bool,
    require_single_link: bool,
) -> tuple[_FileSnapshot, bytes | None]:
    held: _HeldSnapshotHandle | None = None
    failure: BaseException | None = None
    try:
        held = _open_held_snapshot_handle(
            path,
            expected=None,
            require_single_link=require_single_link,
        )
        return _snapshot_held_handle(held, capture_bytes=capture_bytes)
    except OSError as exc:
        gate_error = OuterExperimentGateError(
            f"cannot snapshot immutable file: {_absolute_snapshot_path(path)}"
        )
        failure = gate_error
        raise gate_error from exc
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if held is not None:
            _close_held_snapshot_handles([held], failure)


def _capture_file_snapshot(
    path: Path,
    *,
    require_single_link: bool = False,
) -> _FileSnapshot:
    return _snapshot_open_file(
        path,
        capture_bytes=False,
        require_single_link=require_single_link,
    )[0]


def _assert_file_snapshot_unchanged(
    expected: _FileSnapshot,
    *,
    require_single_link: bool = False,
) -> None:
    current = _capture_file_snapshot(
        expected.path,
        require_single_link=require_single_link,
    )
    if current != expected:
        raise OuterExperimentGateError(
            f"validated file changed before final return: {expected.path}"
        )


def _validate_physical_artifact_uniqueness(paths: tuple[Path, ...]) -> None:
    resolved = tuple(Path(path).resolve() for path in paths)
    if len(resolved) != len(set(os.path.normcase(str(path)) for path in resolved)):
        raise OuterExperimentGateError("result artifact roles must bind distinct paths")
    for path in resolved:
        _capture_file_snapshot(path, require_single_link=True)
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            try:
                same_file = os.path.samefile(left, right)
            except OSError as exc:
                raise OuterExperimentGateError(
                    "cannot verify physical role-artifact identity"
                ) from exc
            if same_file:
                raise OuterExperimentGateError(
                    "distinct artifact roles must not share one physical file"
                )


def _validate_held_role_uniqueness(
    role_handles: tuple[_HeldSnapshotHandle, ...],
    *,
    refresh_states: bool = False,
) -> None:
    path_keys = [os.path.normcase(str(item.path)) for item in role_handles]
    if len(path_keys) != len(set(path_keys)):
        raise OuterExperimentGateError("result artifact roles must bind distinct paths")
    states: list[_OpenHandleState] = []
    for held in role_handles:
        state = _open_handle_state(held.fd) if refresh_states else held.opening_state
        if refresh_states:
            _verify_held_path_and_state(held, state)
        states.append(state)
    identities = [(state.device, state.inode) for state in states]
    if len(identities) != len(set(identities)):
        raise OuterExperimentGateError("distinct artifact roles must not share one physical file")


@contextmanager
def _held_result_snapshot(snapshot: _ResultValidationSnapshot) -> Iterator[None]:
    """Hold and verify the entire result dependency set through caller return.

    Windows handles deny WRITE and DELETE sharing for the entire collection.
    POSIX has no mandatory portable read lock: all descriptors remain open and
    receive a hash pass plus a final fstat/lstat/ancestor pass.  A malicious
    privileged writer scheduled after the final syscall can still mutate a path;
    that is outside this trusted-filesystem integrity gate and must not be
    described as cryptographic immutability.
    """
    role_paths = {item.path for item in snapshot.role_artifacts}
    expected_files = (*snapshot.immutable_files, *snapshot.role_artifacts)
    path_keys = [os.path.normcase(str(item.path)) for item in expected_files]
    if len(path_keys) != len(set(path_keys)):
        raise OuterExperimentGateError("final result snapshot must not repeat a logical file path")
    handles: list[_HeldSnapshotHandle] = []
    failure: BaseException | None = None
    try:
        for expected in expected_files:
            handles.append(
                _open_held_snapshot_handle(
                    expected.path,
                    expected=expected,
                    require_single_link=expected.path in role_paths,
                )
            )
        role_handles = tuple(item for item in handles if item.path in role_paths)
        _validate_held_role_uniqueness(role_handles)

        d3_specs = _d3_provenance_specs(snapshot.readiness)
        d3_paths = {path for path, _ in d3_specs}
        held_paths = {item.path for item in handles}
        if not d3_paths <= held_paths:
            raise OuterExperimentGateError(
                "D3 provenance is not contained in the held source-binding set"
            )
        d3_payloads: dict[Path, dict[str, object]] = {}
        for held in handles:
            file_snapshot, raw = _snapshot_held_handle(
                held,
                capture_bytes=held.path in d3_paths,
            )
            if held.path in d3_paths:
                if raw is None:
                    raise OuterExperimentGateError(
                        "held D3 provenance snapshot did not retain its bytes"
                    )
                expected_sha = dict(d3_specs)[held.path]
                if file_snapshot.sha256 != expected_sha:
                    raise OuterExperimentGateError("held D3 provenance hash differs from readiness")
                d3_payloads[held.path] = _parse_strict_json_bytes(raw, held.path)
        current_d3 = _d3_records_from_payloads(d3_specs, d3_payloads)
        if not _type_exact_equal(current_d3, snapshot.d3_records):
            raise OuterExperimentGateError("D3 provenance changed before final return")

        # Yield while every handle is still live.  For a `return` in the with
        # body, contextlib resumes here before the function actually returns.
        yield

        # Catch POSIX write/restore and late hard-link changes after the hash
        # pass.  Windows reaches this pass with mandatory WRITE/DELETE denial
        # still active for every file in the set.
        for held in handles:
            _final_check_held_handle(held)
        # One last fresh descriptor pass catches a hard-link count/identity
        # change that occurred after an earlier role's final path check.
        _validate_held_role_uniqueness(role_handles, refresh_states=True)
    except OSError as exc:
        gate_error = OuterExperimentGateError("cannot maintain the held final result snapshot")
        failure = gate_error
        raise gate_error from exc
    except BaseException as exc:
        failure = exc
        raise
    finally:
        _close_held_snapshot_handles(handles, failure)


def _revalidate_result_snapshot(snapshot: _ResultValidationSnapshot) -> None:
    with _held_result_snapshot(snapshot):
        pass


def _json_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _create_owned_stage(stage: Path, token: str) -> None:
    try:
        stage.mkdir(exist_ok=False)
    except FileExistsError:
        # A concurrent caller may own this path.  Without our token marker,
        # this invocation must never remove it.
        raise
    except BaseException:
        _remove_stage_after_create_failure(stage, token)
        raise
    try:
        _write_ownership_marker(stage, token)
    except BaseException:
        _remove_stage_after_create_failure(stage, token)
        raise


def _write_ownership_marker(stage: Path, token: str) -> None:
    owner = stage / "ownership.json"
    temporary = stage / f".ownership.{token}.tmp"
    encoded = (json.dumps({"publication_token": token}, allow_nan=False) + "\n").encode("utf-8")
    file_descriptor: int | None = None
    try:
        file_descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(file_descriptor, "wb") as handle:
            file_descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, owner)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        temporary.unlink(missing_ok=True)


def _remove_call_owned_stage(stage: Path, token: str) -> None:
    expected_suffix = f".{token}.stage"
    if not stage.name.endswith(expected_suffix) or not stage.exists():
        return
    if stage.is_symlink() or not stage.is_dir():
        return
    owner = stage / "ownership.json"
    if not owner.exists():
        return
    try:
        payload = _read_json(owner)
    except OuterExperimentGateError:
        return
    if payload != {"publication_token": token}:
        return
    shutil.rmtree(stage)


def _remove_stage_after_create_failure(stage: Path, token: str) -> None:
    """Clean only state attributable to this interrupted create operation."""
    expected_suffix = f".{token}.stage"
    if not stage.name.endswith(expected_suffix) or not stage.exists():
        return
    if stage.is_symlink() or not stage.is_dir():
        return
    owner = stage / "ownership.json"
    if owner.exists():
        try:
            payload = _read_json(owner)
        except OuterExperimentGateError:
            return
        if payload != {"publication_token": token}:
            return
        shutil.rmtree(stage)
        return
    # Prior to publishing ownership, our code can only have an empty directory
    # or its token-specific temporary marker.  Any other entry is foreign.
    allowed_partial = f".ownership.{token}.tmp"
    try:
        child_names = {child.name for child in stage.iterdir()}
    except OSError:
        return
    if child_names <= {allowed_partial}:
        shutil.rmtree(stage)


def _parse_strict_json_bytes(raw: bytes, path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise OuterExperimentGateError(f"non-finite JSON constant is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OuterExperimentGateError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OuterExperimentGateError(f"cannot read strict JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise OuterExperimentGateError(f"JSON artifact must contain an object: {path}")
    return value


def _read_json_with_snapshot(path: Path) -> tuple[dict[str, object], _FileSnapshot]:
    path = _absolute_snapshot_path(path)
    try:
        snapshot, raw = _snapshot_open_file(
            path,
            capture_bytes=True,
            require_single_link=False,
        )
        if raw is None:
            raise OuterExperimentGateError("strict JSON snapshot did not retain its bytes")
        value = _parse_strict_json_bytes(raw, path)
    except OSError as exc:
        raise OuterExperimentGateError(f"cannot read strict JSON artifact: {path}") from exc
    return value, snapshot


def _read_json(path: Path) -> dict[str, object]:
    return _read_json_with_snapshot(path)[0]


def _load_protocol_from_directory(
    directory: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    split = _read_json(directory / "split_manifest.json")
    run = _read_json(directory / "run_manifest.json")
    validate_outer_split_manifest(split)
    validate_outer_run_manifest(run, split)
    return split, run


def _artifact_binding(path: Path, stage: Path) -> dict[str, object]:
    artifact = _owned_file(path, stage)
    snapshot = _capture_file_snapshot(artifact, require_single_link=True)
    return {
        "relative_path": artifact.relative_to(stage.resolve()).as_posix(),
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
    }


def _snapshot_declared_role_artifacts(
    directory: Path,
    value: object,
) -> tuple[_FileSnapshot, ...]:
    if not isinstance(value, dict) or not value:
        raise OuterExperimentGateError("result artifact_bindings must be a JSON object")
    snapshots: list[_FileSnapshot] = []
    for role_value in value.values():
        entries = role_value if isinstance(role_value, list) else [role_value]
        if not entries:
            raise OuterExperimentGateError("artifact role list must not be empty")
        for binding in entries:
            if not isinstance(binding, dict):
                raise OuterExperimentGateError("malformed result artifact binding")
            relative = binding.get("relative_path")
            if not isinstance(relative, str):
                raise OuterExperimentGateError("artifact relative_path must be a string")
            artifact = _contained_path(directory, directory / relative)
            snapshots.append(_capture_file_snapshot(artifact, require_single_link=True))
    _validate_physical_artifact_uniqueness(tuple(item.path for item in snapshots))
    return tuple(snapshots)


def _exact_role_artifact(path: Path, output_dir: Path, filename: str, role: str) -> Path:
    artifact = _owned_file(path, output_dir)
    expected = (output_dir / filename).resolve()
    if artifact != expected:
        raise OuterExperimentGateError(f"{role} must use its fixed path {expected}, got {artifact}")
    _capture_file_snapshot(artifact, require_single_link=True)
    return artifact


def _owned_file(path: Path, stage: Path) -> Path:
    if path.is_symlink():
        raise OuterExperimentGateError(f"backend artifact must not be a symlink: {path}")
    resolved = _contained_path(stage, path)
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise OuterExperimentGateError(f"backend artifact is missing or empty: {path}")
    return resolved


def _contained_path(parent: Path, child: Path) -> Path:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    if child_resolved == parent_resolved or parent_resolved not in child_resolved.parents:
        raise OuterExperimentGateError(f"artifact escapes its owned output directory: {child}")
    return child_resolved


def _rollback_owned_path(path: Path, token: str) -> None:
    if not path.exists() or path.is_symlink() or not path.is_dir():
        return
    owner = path / "ownership.json"
    try:
        payload = _read_json(owner)
    except (OuterExperimentGateError, OSError):
        return
    if payload == {"publication_token": token}:
        shutil.rmtree(path)


__all__ = [
    "BackendContractSmokeBackend",
    "CalibrationReceipt",
    "CalibrationRequest",
    "FinalFitReceipt",
    "FinalFitRequest",
    "FROZEN_SEEDS",
    "InnerFitReceipt",
    "InnerFitRequest",
    "OuterCondition",
    "OuterEvaluationReceipt",
    "OuterEvaluationRequest",
    "OuterExperimentGateError",
    "OuterProtocolBundle",
    "OuterTrainingBackend",
    "build_outer_run_manifest",
    "build_outer_split_manifest",
    "freeze_outer_protocol",
    "publish_outer_protocol",
    "run_outer_unit",
    "validate_outer_run_manifest",
    "validate_outer_split_manifest",
]
