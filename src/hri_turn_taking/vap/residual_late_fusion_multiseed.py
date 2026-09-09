"""Frozen seed-42/73 replay of the saved-probability residual-fusion diagnostic.

Version 1 is the immutable seed-13 implementation and result.  This module
binds that implementation by raw SHA-256, adapts only the source seed identity,
and applies the same formula, grid, tie rule, inner-only selection, metrics and
descriptive criteria to the already completed 15-epoch seed-42/73 matrices.

No model is trained or loaded.  All outputs are post-outcome, exploratory,
non-production and non-counted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hri_turn_taking.vap import residual_late_fusion as v1

FORMAT_VERSION = 2
RESULT_ROLE = "posthoc_saved_probability_residual_late_fusion_multiseed_stability"
PROTOCOL_KIND = "vap_residual_late_fusion_multiseed_protocol"
FOLD_KIND = "vap_residual_late_fusion_multiseed_fold"
SEED_AGGREGATE_KIND = "vap_residual_late_fusion_seed_aggregate"
MULTISEED_AGGREGATE_KIND = "vap_residual_late_fusion_multiseed_aggregate"
DATA_KIND = "real"
PRODUCTION_ELIGIBLE = False
COUNTED_ELIGIBLE = False
ANALYSIS_TIMING = "post_outcome_exploratory_stability_replay"
COUNTED_BLOCKERS = (
    "posthoc_after_outer_outcomes_were_available",
    "saved_probability_diagnostic_not_trainable_architecture",
    "multiseed_replication_of_posthoc_seed13_diagnostic",
    "d3_sensitivity_not_executed",
)
ALLOWED_NEW_SEEDS = (42, 73)
ALL_REPORTING_SEEDS = (13, 42, 73)
A_SYMMETRY_ABS_TOLERANCE = 2e-7
A_SYMMETRY_ULP_TOLERANCE = 4
A_SYMMETRY_GATE_VERSION = 2

V1_IMPLEMENTATION_SHA256 = "aced7056d97adf396350d15c1901cd2281187e717ffc1ec07d8d331ee576e710"
V1_RESULT_ROOT = Path("outputs/residual_late_fusion_posthoc_15epoch_20260818/v1")
V1_AGGREGATE_SHA256 = "00a71a2dbe8692a3e8a5c714cbe0d89bb6b46439bcc5fc64d83520d6bc3438c6"
DEFAULT_OUTPUT = Path("outputs/residual_late_fusion_posthoc_15epoch_20260818/multiseed_v2")

PROTOCOL_FILENAME = "protocol.json"
FOLD_MANIFEST_FILENAME = v1.FOLD_MANIFEST_FILENAME
FUSED_PROBABILITIES_FILENAME = v1.FUSED_PROBABILITIES_FILENAME
SEED_AGGREGATE_FILENAME = "aggregate.json"
MULTISEED_AGGREGATE_FILENAME = "multiseed_aggregate.json"

METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "frame_mAP": ("frame", "mAP"),
    "participant_macro_mAP": ("frame", "participant_macro_mAP"),
    "shift_prediction_AP": ("events", "shift_prediction", "average_precision"),
    "overlap_resolution_AP": (
        "events",
        "overlap_resolution",
        "average_precision",
    ),
    "robot_entry_AP": ("events", "robot_entry", "average_precision"),
    "next_speaker_participant_macro_ovr_AP": (
        "events",
        "next_speaker",
        "participant_macro_ovr_ap",
    ),
    "next_speaker_top1_accuracy": (
        "events",
        "next_speaker",
        "top1_accuracy",
    ),
}


class ResidualLateFusionMultiseedError(ValueError):
    """A frozen-source, pairing, publication, or replay-audit gate failed."""


@dataclass(frozen=True)
class FrozenSeedContract:
    seed: int
    matrix_root: Path
    matrix_index_sha256: str
    training_aggregate: Path
    training_aggregate_sha256: str
    protocol_root: Path
    run_manifest_raw_sha256: str


SOURCE_CONTRACTS: dict[int, FrozenSeedContract] = {
    42: FrozenSeedContract(
        seed=42,
        matrix_root=Path("outputs/thesis_multiseed_15epoch_20260818/matrix_seed42_v1"),
        matrix_index_sha256=(
            "187b98b965b35b8290649da66c4ba34c59c9f957d07af2af525da694740b233a"
        ),
        training_aggregate=Path(
            "outputs/thesis_multiseed_15epoch_20260818/aggregate_seed42_v1.json"
        ),
        training_aggregate_sha256=(
            "4acaaca61a3b080271014fef8b3e1381478351174ab4800e6c2f270f3333dc50"
        ),
        protocol_root=Path("outputs/thesis_posthoc_15epoch_20260817/protocol_seed13_v1"),
        run_manifest_raw_sha256=(
            "890ce604f8a73ec68c6bb727600bcd1b57c22a9865b4a7088b5b696ee749dcec"
        ),
    ),
    73: FrozenSeedContract(
        seed=73,
        matrix_root=Path("outputs/thesis_multiseed_15epoch_20260818/matrix_seed73_v1"),
        matrix_index_sha256=(
            "1f2a7691351314a796e1502e4671b7ccb445cf7c90a00c14bca0ff48cd05167b"
        ),
        training_aggregate=Path(
            "outputs/thesis_multiseed_15epoch_20260818/aggregate_seed73_v1.json"
        ),
        training_aggregate_sha256=(
            "dba7f5f0ed256acd5d0195a444709e4fe1efb3eced9df0cd809ac4d32b187514"
        ),
        protocol_root=Path("outputs/thesis_posthoc_15epoch_20260817/protocol_seed13_v1"),
        run_manifest_raw_sha256=(
            "890ce604f8a73ec68c6bb727600bcd1b57c22a9865b4a7088b5b696ee749dcec"
        ),
    ),
}


@dataclass(frozen=True)
class ExternalBinding:
    absolute_path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "absolute_path": self.absolute_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class SeedSource:
    contract: FrozenSeedContract
    matrix: v1.SourceMatrix
    training_aggregate_path: Path
    training_aggregate_binding: ExternalBinding
    training_aggregate: dict[str, object]

    @property
    def seed(self) -> int:
        return self.contract.seed


@dataclass(frozen=True)
class FrozenInputs:
    sources: dict[int, SeedSource]
    v1_implementation_binding: ExternalBinding
    v1_aggregate_root: Path
    v1_aggregate_binding: ExternalBinding
    v1_aggregate: dict[str, object]


def _fail(message: str) -> None:
    raise ResidualLateFusionMultiseedError(message)


def _external_binding(path_value: str | Path) -> ExternalBinding:
    path = Path(path_value).resolve()
    if path.is_symlink() or not path.is_file():
        _fail(f"external artifact must be a regular non-symlink file: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ResidualLateFusionMultiseedError(
            f"cannot stat external artifact: {path}"
        ) from exc
    return ExternalBinding(str(path), v1._sha256_file(path), size)


def _verify_exact_external(
    path_value: str | Path,
    expected_sha256: str,
    *,
    role: str,
) -> ExternalBinding:
    binding = _external_binding(path_value)
    if binding.sha256 != expected_sha256:
        _fail(f"{role} differs from its frozen SHA-256 binding")
    return binding


def _verify_v1_core() -> ExternalBinding:
    implementation = Path(v1.__file__).resolve()
    binding = _verify_exact_external(
        implementation,
        V1_IMPLEMENTATION_SHA256,
        role="v1 residual-fusion implementation",
    )
    exact_constants = {
        "FORMAT_VERSION": 1,
        "LAMBDA_GRID": (0.0, 0.125, 0.25, 0.5, 1.0),
        "A_SYMMETRY_TOLERANCE": 1e-7,
        "SELECTION_METRIC": "group_equal_participant_macro_strict_mAP",
        "FORMULA": "sigmoid(logit(H_probability)+lambda*logit(A_probability))",
        "TIE_RULE": "highest_inner_group_equal_score_then_smallest_lambda",
        "SOURCE_MAX_EPOCHS": 15,
    }
    for name, expected in exact_constants.items():
        if getattr(v1, name) != expected or type(getattr(v1, name)) is not type(expected):
            _fail(f"v1 frozen core constant changed: {name}")
    if v1.SUCCESS_CRITERIA != {
        "participant_macro_mAP_selected_minus_lambda0_min": 0.02,
        "strictly_positive_outer_fold_count_min": 3,
        "shift_prediction_AP_selected_minus_lambda0_min": -0.02,
        "overlap_resolution_AP_selected_minus_lambda0_min": -0.02,
        "selection_independent_of_success_criteria": True,
    }:
        _fail("v1 descriptive criteria changed")
    return binding


def _validate_training_aggregate(
    contract: FrozenSeedContract,
) -> tuple[Path, ExternalBinding, dict[str, object]]:
    path = contract.training_aggregate.resolve()
    binding = _verify_exact_external(
        path,
        contract.training_aggregate_sha256,
        role=f"seed-{contract.seed} 15-epoch training aggregate",
    )
    payload = v1._read_json_strict(path)
    exact = {
        "artifact_kind": "vap_thesis_primary_matrix_aggregate",
        "format_version": 1,
        "result_role": v1.thesis_backend.RESULT_ROLE,
        "data_kind": "real",
        "production_eligible": False,
        "counted_eligible": False,
        "analysis_variant": "primary",
        "included_session_ids": list(v1.EXACT_EIGHT_SESSION_IDS),
        "group_ids": list(v1.EXACT_FOUR_GROUP_IDS),
        "corpus_exclusions": dict(v1.EXACT_CORPUS_EXCLUSIONS),
        "seed": contract.seed,
        "aggregation_source": "saved_outer_probabilities_only_no_model_inference",
        "run_manifest_sha256": v1.SOURCE_RUN_MANIFEST_SHA256,
    }
    for field, expected in exact.items():
        if payload.get(field) != expected or type(payload.get(field)) is not type(expected):
            _fail(f"seed-{contract.seed} training aggregate differs at {field}")
    conditions = payload.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(v1.MATRIX_CONDITIONS):
        _fail(f"seed-{contract.seed} training aggregate condition grid differs")
    return path, binding, payload


def _load_seed_source(contract: FrozenSeedContract) -> SeedSource:
    if contract.seed not in ALLOWED_NEW_SEEDS or type(contract.seed) is not int:
        _fail("multiseed v2 admits only new seeds 42 and 73")
    source_root = contract.matrix_root.resolve()
    if source_root != contract.matrix_root.resolve():  # pragma: no cover - dataclass guard
        _fail("source matrix root is not canonical")
    index_path = source_root / "matrix_index.json"
    if v1._sha256_file(index_path) != contract.matrix_index_sha256:
        _fail(f"seed-{contract.seed} matrix index differs from its frozen binding")
    index = v1._read_json_strict(index_path)
    required_values = {
        "artifact_kind": "vap_thesis_primary_matrix_index",
        "format_version": 1,
        "result_role": v1.thesis_backend.RESULT_ROLE,
        "data_kind": "real",
        "production_eligible": False,
        "counted_eligible": False,
        "analysis_variant": "primary",
        "seed": contract.seed,
        "condition_ids": list(v1.MATRIX_CONDITIONS),
        "outer_fold_ids": list(v1.OUTER_FOLD_IDS),
        "expected_unit_count": 12,
        "completed_unit_count": 12,
        "included_session_ids": list(v1.EXACT_EIGHT_SESSION_IDS),
        "group_ids": list(v1.EXACT_FOUR_GROUP_IDS),
        "corpus_exclusions": dict(v1.EXACT_CORPUS_EXCLUSIONS),
    }
    for field, expected in required_values.items():
        if index.get(field) != expected or type(index.get(field)) is not type(expected):
            _fail(f"seed-{contract.seed} source matrix index differs at {field}")

    declared_protocol = index.get("protocol_directory")
    protocol_root = contract.protocol_root.resolve()
    if (
        not isinstance(declared_protocol, str)
        or Path(declared_protocol) != protocol_root
        or Path(declared_protocol).resolve() != protocol_root
    ):
        _fail(f"seed-{contract.seed} source protocol directory differs")
    run_manifest_path = protocol_root / "run_manifest.json"
    if v1._sha256_file(run_manifest_path) != contract.run_manifest_raw_sha256:
        _fail(f"seed-{contract.seed} raw run manifest differs from its frozen binding")
    run_manifest = v1._read_json_strict(run_manifest_path)
    if (
        index.get("run_manifest_sha256") != v1.SOURCE_RUN_MANIFEST_SHA256
        or v1._canonical_json_sha256(run_manifest) != v1.SOURCE_RUN_MANIFEST_SHA256
    ):
        _fail(f"seed-{contract.seed} canonical run manifest binding differs")
    run_exact = {
        "artifact_kind": v1.SOURCE_RUN_MANIFEST_KIND,
        "format_version": v1.SOURCE_RUN_MANIFEST_VERSION,
        "protocol_id": v1.SOURCE_PROTOCOL_ID,
        "result_role": "protocol_only_not_a_result",
        "counted_experiment_complete": False,
        "split_manifest_sha256": v1.SOURCE_SPLIT_MANIFEST_SHA256,
        "seeds": list(v1.SOURCE_SEEDS),
        "condition_ids": list(v1.MATRIX_CONDITIONS),
        "max_epochs": 15,
        "aux_vad_weight": 0.0,
    }
    for field, expected in run_exact.items():
        if run_manifest.get(field) != expected or type(run_manifest.get(field)) is not type(
            expected
        ):
            _fail(f"seed-{contract.seed} run manifest differs at {field}")

    snapshots: dict[Path, v1.SourceFileSnapshot] = {}
    for path, single_link in (
        (Path(v1.__file__).resolve(), False),
        (index_path, False),
        (run_manifest_path, False),
        (contract.training_aggregate.resolve(), False),
    ):
        snapshot = v1._capture_source_file_snapshot(path, require_single_link=single_link)
        v1._add_source_file_snapshot(
            snapshots,
            snapshot,
            require_single_link=single_link,
        )

    prepared_conditions = v1._prepare_historical_conditions(protocol_root, run_manifest)
    units_value = index.get("units")
    if not isinstance(units_value, list) or len(units_value) != 12:
        _fail(f"seed-{contract.seed} source matrix must contain exactly 12 units")
    keyed: dict[tuple[str, str], dict[str, object]] = {}
    for item in units_value:
        if not isinstance(item, dict):
            _fail("source matrix unit must be an object")
        condition = item.get("condition_id")
        fold = item.get("outer_fold_id")
        if condition not in v1.MATRIX_CONDITIONS or fold not in v1.OUTER_FOLD_IDS:
            _fail("source matrix unit identity is invalid")
        if item.get("seed") != contract.seed or type(item.get("seed")) is not int:
            _fail(f"source matrix unit seed differs from {contract.seed}")
        key = (str(condition), str(fold))
        if key in keyed:
            _fail("source matrix unit identity is duplicated")
        keyed[key] = item
    expected_keys = {
        (condition, fold)
        for condition in v1.MATRIX_CONDITIONS
        for fold in v1.OUTER_FOLD_IDS
    }
    if set(keyed) != expected_keys:
        _fail("source matrix does not contain the exact 12-unit grid")

    units: dict[tuple[str, str], v1.SourceUnit] = {}
    result_bindings: dict[tuple[str, str], v1.FileBinding] = {}
    for condition in v1.MATRIX_CONDITIONS:
        for fold in v1.OUTER_FOLD_IDS:
            item = keyed[(condition, fold)]
            directory = (source_root / v1.CONDITION_DIRECTORIES[condition] / fold).resolve()
            result_path = directory / "result_manifest.json"
            declared_path = item.get("result_manifest")
            if not isinstance(declared_path, str) or Path(declared_path).resolve() != result_path:
                _fail("matrix index result_manifest path is not canonical")
            declared_hash = item.get("result_manifest_sha256")
            if not v1._is_sha256(declared_hash) or v1._sha256_file(result_path) != declared_hash:
                _fail("matrix index result-manifest hash changed")
            validation = v1._validate_historical_unit(
                directory,
                prepared=prepared_conditions[condition],
            )
            v1._add_historical_validation_snapshots(snapshots, validation)
            result = v1._read_json_strict(result_path)
            if (
                result.get("condition_id") != condition
                or result.get("outer_fold_id") != fold
                or result.get("seed") != contract.seed
                or type(result.get("seed")) is not int
                or result.get("run_manifest_sha256") != v1.SOURCE_RUN_MANIFEST_SHA256
            ):
                _fail("source result identity differs from matrix index")
            result_binding = v1._binding(result_path, source_root)
            result_bindings[(condition, fold)] = result_binding
            result_snapshot = v1._capture_source_file_snapshot(
                result_path,
                require_single_link=False,
            )
            v1._add_source_file_snapshot(
                snapshots,
                result_snapshot,
                require_single_link=False,
            )
            if condition not in v1.SOURCE_CONDITIONS:
                continue
            prediction_bindings = {
                inner: v1._artifact_binding_from_result(
                    result,
                    role="inner_oof_predictions",
                    inner_fold_id=inner,
                )
                for inner in v1.INNER_FOLD_IDS
            }
            prediction_bindings["outer"] = v1._artifact_binding_from_result(
                result,
                role="outer_predictions",
                inner_fold_id=None,
            )
            for binding in prediction_bindings.values():
                prediction_path = v1._verify_binding(binding, directory)
                prediction_snapshot = v1._capture_source_file_snapshot(
                    prediction_path,
                    require_single_link=True,
                )
                v1._add_source_file_snapshot(
                    snapshots,
                    prediction_snapshot,
                    require_single_link=True,
                )
            units[(condition, fold)] = v1.SourceUnit(
                condition_id=condition,
                outer_fold_id=fold,
                directory=directory,
                result_manifest_path=result_path,
                result_manifest_binding=result_binding,
                prediction_bindings=prediction_bindings,
            )
    ordered_snapshots = tuple(
        snapshots[path]
        for path in sorted(snapshots, key=lambda value: os.path.normcase(str(value)))
    )
    v1._revalidate_source_file_snapshots(ordered_snapshots)
    matrix = v1.SourceMatrix(
        root=source_root,
        index_path=index_path,
        index_binding=v1._binding(index_path, source_root),
        index=index,
        protocol_root=protocol_root,
        run_manifest_path=run_manifest_path,
        run_manifest_binding=v1._binding(run_manifest_path, protocol_root),
        run_manifest=run_manifest,
        units=units,
        result_manifest_bindings=result_bindings,
        source_file_snapshots=ordered_snapshots,
    )
    aggregate_path, aggregate_binding, aggregate = _validate_training_aggregate(contract)
    return SeedSource(
        contract=contract,
        matrix=matrix,
        training_aggregate_path=aggregate_path,
        training_aggregate_binding=aggregate_binding,
        training_aggregate=aggregate,
    )


def _load_frozen_inputs() -> FrozenInputs:
    v1_implementation = _verify_v1_core()
    if set(SOURCE_CONTRACTS) != set(ALLOWED_NEW_SEEDS):
        _fail("source contract set must be exactly seeds 42 and 73")
    sources = {seed: _load_seed_source(SOURCE_CONTRACTS[seed]) for seed in ALLOWED_NEW_SEEDS}
    v1_root = V1_RESULT_ROOT.resolve()
    v1_aggregate_path = v1_root / v1.AGGREGATE_FILENAME
    v1_aggregate_binding = _verify_exact_external(
        v1_aggregate_path,
        V1_AGGREGATE_SHA256,
        role="audited seed-13 v1 aggregate",
    )
    v1_aggregate = v1._read_json_strict(v1_aggregate_path)
    if (
        v1_aggregate.get("artifact_kind") != v1.AGGREGATE_KIND
        or v1_aggregate.get("format_version") != 1
        or v1_aggregate.get("seed") != 13
        or v1_aggregate.get("aggregation_source")
        != "saved_fused_outer_probabilities_only_no_model_inference"
    ):
        _fail("seed-13 v1 aggregate identity differs")
    return FrozenInputs(
        sources=sources,
        v1_implementation_binding=v1_implementation,
        v1_aggregate_root=v1_root,
        v1_aggregate_binding=v1_aggregate_binding,
        v1_aggregate=v1_aggregate,
    )


def _revalidate_frozen_inputs(inputs: FrozenInputs) -> None:
    current_v1 = _verify_v1_core()
    if current_v1 != inputs.v1_implementation_binding:
        _fail("v1 implementation binding changed during replay")
    current_aggregate = _verify_exact_external(
        inputs.v1_aggregate_root / v1.AGGREGATE_FILENAME,
        V1_AGGREGATE_SHA256,
        role="audited seed-13 v1 aggregate",
    )
    if current_aggregate != inputs.v1_aggregate_binding:
        _fail("seed-13 v1 aggregate binding changed during replay")
    for source in inputs.sources.values():
        v1._revalidate_source_matrix(source.matrix)
        current_training = _verify_exact_external(
            source.training_aggregate_path,
            source.contract.training_aggregate_sha256,
            role=f"seed-{source.seed} training aggregate",
        )
        if current_training != source.training_aggregate_binding:
            _fail(f"seed-{source.seed} training aggregate changed during replay")


def _participant_symmetry_diagnostics(
    payload: dict[str, object],
) -> tuple[float, int]:
    """Return maximum participant spread in absolute units and float32 ULPs."""

    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        _fail("A source payload has no sessions for symmetry validation")
    observed_abs = 0.0
    observed_ulp = 0
    for record in sessions:
        if not isinstance(record, dict):
            _fail("A source session is malformed for symmetry validation")
        probability = np.asarray(record.get("probabilities"), dtype=np.float32)
        present = np.asarray(record.get("present"), dtype=bool)
        if (
            probability.ndim != 3
            or probability.shape[1:] != (4, 4)
            or present.shape != (4,)
            or not present.any()
            or not np.isfinite(probability).all()
            or ((probability < 0.0) | (probability > 1.0)).any()
        ):
            _fail("A source probability array is invalid for symmetry validation")
        values = np.ascontiguousarray(probability[:, present, :], dtype=np.float32)
        absolute_spread = np.ptp(values.astype(np.float64), axis=1)
        # Probabilities are non-negative, so IEEE-754 uint32 ordering is monotonic.
        bit_values = values.view(np.uint32).astype(np.uint64)
        ulp_spread = np.ptp(bit_values, axis=1)
        observed_abs = max(observed_abs, float(absolute_spread.max(initial=0.0)))
        observed_ulp = max(observed_ulp, int(ulp_spread.max(initial=0)))
    return observed_abs, observed_ulp


def _validate_paired_payloads(
    a_payload: dict[str, object],
    h_payload: dict[str, object],
    *,
    seed: int,
    outer_fold_id: str,
    inner_fold_id: str | None,
) -> float:
    if seed not in ALLOWED_NEW_SEEDS:
        _fail("paired payload validation admits only seeds 42 and 73")
    if a_payload.get("condition_id") != "A" or h_payload.get("condition_id") != "H":
        _fail("paired sources must be conditions A and H")
    expected_inner = inner_fold_id is not None
    for payload in (a_payload, h_payload):
        if payload.get("outer_fold_id") != outer_fold_id or payload.get("seed") != seed:
            _fail("paired source fold/seed identity differs")
        if expected_inner and payload.get("inner_fold_id") != inner_fold_id:
            _fail("paired inner OOF identity differs")
        if payload.get("run_manifest_sha256") != a_payload.get("run_manifest_sha256"):
            _fail("paired sources use different run manifests")
        if payload.get("frame_rate_hz") != 25:
            _fail("paired source frame rate differs from 25 Hz")
        if payload.get("label_config") != a_payload.get("label_config"):
            _fail("paired source label configurations differ")
        if payload.get("threshold_policy") != a_payload.get("threshold_policy"):
            _fail("paired source threshold policies differ")
    a_sessions = a_payload.get("sessions")
    h_sessions = h_payload.get("sessions")
    if not isinstance(a_sessions, list) or not isinstance(h_sessions, list):
        _fail("paired source sessions are malformed")
    if len(a_sessions) != len(h_sessions) or not a_sessions:
        _fail("paired A/H source session counts differ")
    for a_record, h_record in zip(a_sessions, h_sessions, strict=True):
        if not isinstance(a_record, dict) or not isinstance(h_record, dict):
            _fail("paired source session record is malformed")
        for field in ("session_id", "group_id", "n_frames", "tick_25hz"):
            if a_record.get(field) != h_record.get(field) or type(a_record.get(field)) is not type(
                h_record.get(field)
            ):
                _fail(f"A/H paired {field} values differ")
        try:
            v1._exact_array_field(a_record["labels"], h_record["labels"], "labels", np.float32)
            v1._exact_array_field(a_record["vad"], h_record["vad"], "vad", np.float32)
            v1._exact_array_field(a_record["valid"], h_record["valid"], "valid", np.int8)
            v1._exact_array_field(a_record["present"], h_record["present"], "present", np.int8)
            v1._exact_array_field(
                np.asarray(a_record["tick_25hz"], dtype=np.int64),
                np.asarray(h_record["tick_25hz"], dtype=np.int64),
                "ticks",
                np.int64,
            )
        except (KeyError, v1.ResidualLateFusionError) as exc:
            raise ResidualLateFusionMultiseedError(str(exc)) from exc
    try:
        symmetry, ulp_distance = _participant_symmetry_diagnostics(a_payload)
    except (KeyError, TypeError, ValueError, v1.ResidualLateFusionError) as exc:
        raise ResidualLateFusionMultiseedError("A participant symmetry is invalid") from exc
    if (
        not math.isfinite(symmetry)
        or symmetry > A_SYMMETRY_ABS_TOLERANCE
        or ulp_distance > A_SYMMETRY_ULP_TOLERANCE
    ):
        _fail(
            "A participant symmetry exceeds the v2 float32 numerical-invariance gate: "
            f"observed_abs={symmetry:.17g}, observed_ulp={ulp_distance}, "
            f"abs_limit={A_SYMMETRY_ABS_TOLERANCE:.17g}, "
            f"ulp_limit={A_SYMMETRY_ULP_TOLERANCE}"
        )
    return symmetry


def _paired_predictions(
    source: SeedSource,
    outer_fold_id: str,
    inner_fold_id: str | None,
) -> v1.PairedPredictions:
    key = inner_fold_id if inner_fold_id is not None else "outer"
    a_unit = source.matrix.units[("A", outer_fold_id)]
    h_unit = source.matrix.units[("H", outer_fold_id)]
    a_path = v1._verify_binding(a_unit.prediction_bindings[key], a_unit.directory)
    h_path = v1._verify_binding(h_unit.prediction_bindings[key], h_unit.directory)
    a_payload = v1._load_prediction(a_path, inner_fold_id)
    h_payload = v1._load_prediction(h_path, inner_fold_id)
    symmetry = _validate_paired_payloads(
        a_payload,
        h_payload,
        seed=source.seed,
        outer_fold_id=outer_fold_id,
        inner_fold_id=inner_fold_id,
    )
    return v1.PairedPredictions(
        a_payload=a_payload,
        h_payload=h_payload,
        a_path=a_path,
        h_path=h_path,
        a_binding=v1._binding(a_path, source.matrix.root),
        h_binding=v1._binding(h_path, source.matrix.root),
        a_symmetry_max_abs_diff=symmetry,
    )


def _recompute_expected_fold(
    source: SeedSource,
    outer_fold_id: str,
) -> tuple[
    float,
    list[dict[str, object]],
    tuple[v1.PairedPredictions, ...],
    v1.PairedPredictions,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    inner_pairs = tuple(
        _paired_predictions(source, outer_fold_id, inner_fold_id)
        for inner_fold_id in v1.INNER_FOLD_IDS
    )
    selected_lambda, candidates = v1._select_lambda(inner_pairs)
    outer_pair = _paired_predictions(source, outer_fold_id, None)
    selected = v1._fuse_pair(outer_pair, selected_lambda)
    lambda0 = v1._fuse_pair(outer_pair, 0.0)
    return selected_lambda, candidates, inner_pairs, outer_pair, selected, lambda0


def _protocol_payload(inputs: FrozenInputs) -> dict[str, object]:
    source_records: dict[str, object] = {}
    for seed in ALLOWED_NEW_SEEDS:
        source = inputs.sources[seed]
        source_records[str(seed)] = {
            "seed": seed,
            "matrix_root": str(source.matrix.root),
            "matrix_index": source.matrix.index_binding.to_dict(),
            "training_aggregate": source.training_aggregate_binding.to_dict(),
            "protocol_root": str(source.matrix.protocol_root),
            "run_manifest": source.matrix.run_manifest_binding.to_dict(),
            "run_manifest_canonical_json_sha256": v1.SOURCE_RUN_MANIFEST_SHA256,
            "run_manifest_raw_sha256": source.matrix.run_manifest_binding.sha256,
            "max_epochs": 15,
            "included_session_ids": list(v1.EXACT_EIGHT_SESSION_IDS),
            "group_ids": list(v1.EXACT_FOUR_GROUP_IDS),
            "corpus_exclusions": dict(v1.EXACT_CORPUS_EXCLUSIONS),
        }
    return {
        "artifact_kind": PROTOCOL_KIND,
        "format_version": FORMAT_VERSION,
        "result_role": RESULT_ROLE,
        "data_kind": DATA_KIND,
        "production_eligible": PRODUCTION_ELIGIBLE,
        "counted_eligible": COUNTED_ELIGIBLE,
        "counted_blockers": list(COUNTED_BLOCKERS),
        "analysis_timing": ANALYSIS_TIMING,
        "confirmatory_hypothesis_verdict_eligible": False,
        "diagnostic_boundary": (
            "saved-probability residual-fusion stability replay; no training, checkpoint, "
            "model inference, or trainable gated-fusion architecture"
        ),
        "v1_frozen_implementation": inputs.v1_implementation_binding.to_dict(),
        "seed13_audited_v1_aggregate": inputs.v1_aggregate_binding.to_dict(),
        "new_seed_ids": list(ALLOWED_NEW_SEEDS),
        "all_reporting_seed_ids": list(ALL_REPORTING_SEEDS),
        "sources": source_records,
        "source_conditions": list(v1.SOURCE_CONDITIONS),
        "historical_A_plus_H_context": {
            "condition_id": "A+H",
            "role": "context_only",
            "used_in_lambda_selection": False,
            "used_in_primary_comparison": False,
        },
        "outer_fold_ids": list(v1.OUTER_FOLD_IDS),
        "inner_fold_ids_per_outer": list(v1.INNER_FOLD_IDS),
        "lambda_grid": list(v1.LAMBDA_GRID),
        "formula": v1.FORMULA,
        "selection_metric": v1.SELECTION_METRIC,
        "selection_aggregation": "equal_mean_of_three_inner_group_scores_not_frame_pooled",
        "tie_rule": v1.TIE_RULE,
        "outer_gold_used_in_lambda_selection": False,
        "matched_control": {
            "lambda": 0.0,
            "definition": "same_formula_exact_H_copy",
            "required_array_equal_to_H": True,
            "required_canonical_probability_bytes_equal_to_H": True,
        },
        "a_participant_symmetry_gate": {
            "gate_version": A_SYMMETRY_GATE_VERSION,
            "max_abs_diff": A_SYMMETRY_ABS_TOLERANCE,
            "max_float32_ulp_distance": A_SYMMETRY_ULP_TOLERANCE,
            "rule": "both_limits_must_pass",
            "scope": "present_participants_per_frame_and_future_bin",
            "gate_only_no_probability_lambda_metric_or_criteria_change": True,
            "repair_round1_source_scan": {
                "scope": "all_32_seed42_73_A_inner_and_outer_probability_artifacts",
                "gold_metrics_read": False,
                "seed42_max_abs_diff": 8.940696716308594e-08,
                "seed42_max_float32_ulp_distance": 3,
                "seed73_max_abs_diff": 1.1920928955078125e-07,
                "seed73_max_float32_ulp_distance": 4,
            },
        },
        "descriptive_success_criteria": v1.SUCCESS_CRITERIA,
        "criteria_timing_caveat": (
            "criteria were recorded after historical outer outcomes existed; they are "
            "descriptive, not preregistered or confirmatory"
        ),
        "multiseed_summary_rule": "arithmetic_mean_and_sample_SD_across_three_seeds",
        "significance_testing_permitted": False,
    }


def _fold_payload(
    *,
    source: SeedSource,
    outer_fold_id: str,
    selected_lambda: float,
    candidates: list[dict[str, object]],
    inner_pairs: tuple[v1.PairedPredictions, ...],
    outer_pair: v1.PairedPredictions,
    fused_binding: v1.FileBinding,
    selected_probabilities: tuple[np.ndarray, ...],
    lambda0_probabilities: tuple[np.ndarray, ...],
    selected_metrics: dict[str, object],
    lambda0_metrics: dict[str, object],
) -> dict[str, object]:
    inner_symmetry = {}
    for pair in inner_pairs:
        inner_fold_id = pair.h_payload.get("inner_fold_id")
        if not isinstance(inner_fold_id, str):
            _fail("inner pair has no symmetry-audit fold identity")
        observed_abs, observed_ulp = _participant_symmetry_diagnostics(pair.a_payload)
        inner_symmetry[inner_fold_id] = {
            "max_abs_diff": observed_abs,
            "max_float32_ulp_distance": observed_ulp,
        }
    outer_abs, outer_ulp = _participant_symmetry_diagnostics(outer_pair.a_payload)
    payload = v1._fold_payload(
        source=source.matrix,
        outer_fold_id=outer_fold_id,
        selected_lambda=selected_lambda,
        candidates=candidates,
        inner_pairs=inner_pairs,
        outer_pair=outer_pair,
        fused_binding=fused_binding,
        selected_probabilities=selected_probabilities,
        lambda0_probabilities=lambda0_probabilities,
        selected_metrics=selected_metrics,
        lambda0_metrics=lambda0_metrics,
    )
    payload.update(
        {
            "artifact_kind": FOLD_KIND,
            "format_version": FORMAT_VERSION,
            "result_role": RESULT_ROLE,
            "counted_blockers": list(COUNTED_BLOCKERS),
            "analysis_timing": ANALYSIS_TIMING,
            "seed": source.seed,
            "v1_implementation_sha256": V1_IMPLEMENTATION_SHA256,
            "source_training_aggregate": source.training_aggregate_binding.to_dict(),
            "a_participant_symmetry": {
                "gate_version": A_SYMMETRY_GATE_VERSION,
                "max_abs_diff_limit": A_SYMMETRY_ABS_TOLERANCE,
                "max_float32_ulp_distance_limit": A_SYMMETRY_ULP_TOLERANCE,
                "rule": "both_limits_must_pass",
                "inner_observed": inner_symmetry,
                "outer_observed": {
                    "max_abs_diff": outer_abs,
                    "max_float32_ulp_distance": outer_ulp,
                },
                "all_within_gate": True,
                "gate_only_no_probability_lambda_metric_or_criteria_change": True,
            },
        }
    )
    return payload


def _seed_aggregate_payload(
    *,
    seed: int,
    protocol_binding: v1.FileBinding,
    fold_records: list[tuple[dict[str, object], v1.FileBinding, v1.FileBinding]],
    selected_metrics: dict[str, object],
    lambda0_metrics: dict[str, object],
) -> dict[str, object]:
    payload = v1._aggregate_payload(
        protocol_binding,
        fold_records,
        selected_metrics,
        lambda0_metrics,
    )
    payload.update(
        {
            "artifact_kind": SEED_AGGREGATE_KIND,
            "format_version": FORMAT_VERSION,
            "result_role": RESULT_ROLE,
            "counted_blockers": list(COUNTED_BLOCKERS),
            "analysis_timing": ANALYSIS_TIMING,
            "seed": seed,
            "v1_implementation_sha256": V1_IMPLEMENTATION_SHA256,
            "stability_role": "one_of_two_new_seed_replays_of_frozen_seed13_diagnostic",
        }
    )
    return payload


def _required_metric(metrics: dict[str, object], path: tuple[str, ...]) -> float:
    value: object = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value:
            _fail(f"required multiseed metric is missing: {'/'.join(path)}")
        value = value[key]
    if type(value) is not float or not math.isfinite(value):
        _fail(f"required multiseed metric is not a finite float: {'/'.join(path)}")
    return value


def _metric_vector(metrics: dict[str, object]) -> dict[str, float]:
    return {name: _required_metric(metrics, path) for name, path in METRIC_PATHS.items()}


def _sample_summary(values: list[float]) -> dict[str, object]:
    if len(values) != 3 or any(not math.isfinite(value) for value in values):
        _fail("multiseed summary requires exactly three finite values")
    return {
        "n_seeds": 3,
        "mean": float(math.fsum(values) / 3.0),
        "sample_SD": float(statistics.stdev(values)),
    }


def _multiseed_aggregate_payload(
    *,
    inputs: FrozenInputs,
    protocol_binding: v1.FileBinding,
    seed_aggregates: dict[int, tuple[dict[str, object], v1.FileBinding]],
) -> dict[str, object]:
    all_aggregates = {13: inputs.v1_aggregate}
    all_aggregates.update({seed: value[0] for seed, value in seed_aggregates.items()})
    per_seed: dict[str, object] = {}
    selected_vectors: dict[int, dict[str, float]] = {}
    control_vectors: dict[int, dict[str, float]] = {}
    delta_vectors: dict[int, dict[str, float]] = {}
    for seed in ALL_REPORTING_SEEDS:
        aggregate = all_aggregates[seed]
        pooled = aggregate.get("pooled")
        if not isinstance(pooled, dict):
            _fail(f"seed-{seed} aggregate has no pooled metrics")
        selected = pooled.get("selected")
        control = pooled.get("matched_lambda0")
        if not isinstance(selected, dict) or not isinstance(control, dict):
            _fail(f"seed-{seed} aggregate pooled metrics are malformed")
        selected_vector = _metric_vector(selected)
        control_vector = _metric_vector(control)
        delta_vector = {
            name: float(selected_vector[name] - control_vector[name])
            for name in METRIC_PATHS
        }
        selected_vectors[seed] = selected_vector
        control_vectors[seed] = control_vector
        delta_vectors[seed] = delta_vector
        folds = aggregate.get("per_fold")
        success = aggregate.get("success_evaluation")
        if not isinstance(folds, dict) or set(folds) != set(v1.OUTER_FOLD_IDS):
            _fail(f"seed-{seed} aggregate fold grid differs")
        if not isinstance(success, dict):
            _fail(f"seed-{seed} aggregate success evaluation is missing")
        per_seed[str(seed)] = {
            "seed": seed,
            "selected": selected_vector,
            "matched_lambda0": control_vector,
            "selected_minus_matched_lambda0": delta_vector,
            "selected_lambdas": {
                fold: folds[fold]["selected_lambda"] for fold in v1.OUTER_FOLD_IDS
            },
            "descriptive_criteria_met": success.get("descriptive_criteria_met"),
            "descriptive_verdict": success.get("verdict"),
            "confirmatory_hypothesis_verdict_eligible": False,
        }
    summaries: dict[str, object] = {}
    for role, vectors in (
        ("selected", selected_vectors),
        ("matched_lambda0", control_vectors),
        ("selected_minus_matched_lambda0", delta_vectors),
    ):
        summaries[role] = {
            name: _sample_summary([vectors[seed][name] for seed in ALL_REPORTING_SEEDS])
            for name in METRIC_PATHS
        }
    return {
        "artifact_kind": MULTISEED_AGGREGATE_KIND,
        "format_version": FORMAT_VERSION,
        "result_role": RESULT_ROLE,
        "data_kind": DATA_KIND,
        "production_eligible": False,
        "counted_eligible": False,
        "counted_blockers": list(COUNTED_BLOCKERS),
        "analysis_timing": ANALYSIS_TIMING,
        "confirmatory_hypothesis_verdict_eligible": False,
        "aggregation_source": (
            "audited_seed13_v1_plus_saved_probability_seed42_73_replays_no_model_inference"
        ),
        "protocol": protocol_binding.to_dict(),
        "v1_frozen_implementation": inputs.v1_implementation_binding.to_dict(),
        "seed_aggregate_bindings": {
            "13": inputs.v1_aggregate_binding.to_dict(),
            **{
                str(seed): binding.to_dict()
                for seed, (_aggregate, binding) in seed_aggregates.items()
            },
        },
        "seed_ids": list(ALL_REPORTING_SEEDS),
        "per_seed": per_seed,
        "mean_and_sample_SD": summaries,
        "sample_SD_denominator": "n_minus_1",
        "significance_tests_run": False,
        "significance_claims_permitted": False,
        "interpretation_role": (
            "descriptive_cross_seed_stability_of_a_post_outcome_saved_probability_diagnostic"
        ),
        "hypothesis_boundary": "does_not_change_H1_or_test_H2",
    }


def _expected_output_files() -> set[str]:
    files = {PROTOCOL_FILENAME, MULTISEED_AGGREGATE_FILENAME}
    for seed in ALLOWED_NEW_SEEDS:
        seed_dir = f"seed_{seed}"
        files.add(f"{seed_dir}/{SEED_AGGREGATE_FILENAME}")
        for fold in v1.OUTER_FOLD_IDS:
            files.add(f"{seed_dir}/{fold}/{FOLD_MANIFEST_FILENAME}")
            files.add(f"{seed_dir}/{fold}/{FUSED_PROBABILITIES_FILENAME}")
    return files


def _actual_output_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _build_seed(
    stage: Path,
    source: SeedSource,
    protocol_binding: v1.FileBinding,
) -> tuple[dict[str, object], v1.FileBinding]:
    fold_records: list[tuple[dict[str, object], v1.FileBinding, v1.FileBinding]] = []
    selected_sessions: list[dict[str, Any]] = []
    selected_probabilities: list[np.ndarray] = []
    lambda0_sessions: list[dict[str, Any]] = []
    lambda0_probabilities: list[np.ndarray] = []
    observed_groups: set[str] = set()
    observed_sessions: set[str] = set()
    seed_dir = stage / f"seed_{source.seed}"
    for outer_fold_id in v1.OUTER_FOLD_IDS:
        (
            selected_lambda,
            candidates,
            inner_pairs,
            outer_pair,
            selected,
            lambda0,
        ) = _recompute_expected_fold(source, outer_fold_id)
        fold_dir = seed_dir / outer_fold_id
        fused_path = fold_dir / FUSED_PROBABILITIES_FILENAME
        packed, offsets = v1._pack_probabilities(selected)
        v1._write_npz_exclusive(fused_path, packed, offsets)
        fused_binding = v1._binding(fused_path, stage)
        selected_metrics = v1._metrics(outer_pair.h_payload, selected)
        lambda0_metrics = v1._metrics(outer_pair.h_payload, lambda0)
        manifest = _fold_payload(
            source=source,
            outer_fold_id=outer_fold_id,
            selected_lambda=selected_lambda,
            candidates=candidates,
            inner_pairs=inner_pairs,
            outer_pair=outer_pair,
            fused_binding=fused_binding,
            selected_probabilities=selected,
            lambda0_probabilities=lambda0,
            selected_metrics=selected_metrics,
            lambda0_metrics=lambda0_metrics,
        )
        group_id = manifest["held_out_group_id"]
        if not isinstance(group_id, str) or group_id in observed_groups:
            _fail("outer group appears in more than one fold")
        observed_groups.add(group_id)
        session_ids = manifest["outer_session_ids"]
        if not isinstance(session_ids, list) or any(
            not isinstance(value, str) or value in observed_sessions for value in session_ids
        ):
            _fail("outer session appears in more than one fold")
        observed_sessions.update(session_ids)
        manifest_path = fold_dir / FOLD_MANIFEST_FILENAME
        v1._write_json_exclusive(manifest_path, manifest)
        manifest_binding = v1._binding(manifest_path, stage)
        fold_records.append((manifest, manifest_binding, fused_binding))
        sessions = v1.thesis_backend._sessions_from_prediction(outer_pair.h_payload)
        selected_sessions.extend(sessions)
        selected_probabilities.extend(selected)
        lambda0_sessions.extend(sessions)
        lambda0_probabilities.extend(lambda0)
    if observed_groups != set(v1.EXACT_FOUR_GROUP_IDS):
        _fail("outer folds do not cover the exact four groups")
    if observed_sessions != set(v1.EXACT_EIGHT_SESSION_IDS):
        _fail("outer folds do not cover the exact eight sessions")
    selected_pooled = v1.thesis_backend._metrics(
        tuple(selected_sessions),
        tuple(selected_probabilities),
    )
    lambda0_pooled = v1.thesis_backend._metrics(
        tuple(lambda0_sessions),
        tuple(lambda0_probabilities),
    )
    aggregate = _seed_aggregate_payload(
        seed=source.seed,
        protocol_binding=protocol_binding,
        fold_records=fold_records,
        selected_metrics=selected_pooled,
        lambda0_metrics=lambda0_pooled,
    )
    aggregate_path = seed_dir / SEED_AGGREGATE_FILENAME
    v1._write_json_exclusive(aggregate_path, aggregate)
    return aggregate, v1._binding(aggregate_path, stage)


def _audit_multiseed(
    output_root_value: str | Path,
    *,
    validated_inputs: FrozenInputs | None,
) -> dict[str, object]:
    output_root = Path(output_root_value).resolve()
    if not output_root.is_dir() or output_root.is_symlink():
        _fail("multiseed residual-fusion output must be a real directory")
    if _actual_output_files(output_root) != _expected_output_files():
        _fail("multiseed residual-fusion output file set differs")
    inputs = _load_frozen_inputs() if validated_inputs is None else validated_inputs
    _revalidate_frozen_inputs(inputs)
    protocol_path = output_root / PROTOCOL_FILENAME
    protocol = v1._read_json_strict(protocol_path)
    expected_protocol = _protocol_payload(inputs)
    if v1._canonical_json(protocol) != v1._canonical_json(expected_protocol):
        _fail("multiseed protocol differs from frozen-source replay")
    protocol_binding = v1._binding(protocol_path, output_root)
    seed_aggregates: dict[int, tuple[dict[str, object], v1.FileBinding]] = {}
    observed_symmetry = 0.0
    observed_symmetry_ulp = 0
    for seed in ALLOWED_NEW_SEEDS:
        source = inputs.sources[seed]
        fold_records: list[tuple[dict[str, object], v1.FileBinding, v1.FileBinding]] = []
        selected_sessions: list[dict[str, Any]] = []
        selected_probabilities: list[np.ndarray] = []
        lambda0_sessions: list[dict[str, Any]] = []
        lambda0_probabilities: list[np.ndarray] = []
        for outer_fold_id in v1.OUTER_FOLD_IDS:
            (
                selected_lambda,
                candidates,
                inner_pairs,
                outer_pair,
                expected_selected,
                expected_lambda0,
            ) = _recompute_expected_fold(source, outer_fold_id)
            fold_dir = output_root / f"seed_{seed}" / outer_fold_id
            fused_path = fold_dir / FUSED_PROBABILITIES_FILENAME
            lengths = tuple(record["n_frames"] for record in outer_pair.h_payload["sessions"])
            saved_selected = v1._load_saved_probabilities(fused_path, lengths)
            if not all(
                np.array_equal(saved, expected)
                for saved, expected in zip(saved_selected, expected_selected, strict=True)
            ):
                _fail(f"seed-{seed} saved fused probabilities differ from formula replay")
            fused_binding = v1._binding(fused_path, output_root)
            selected_metrics = v1._metrics(outer_pair.h_payload, saved_selected)
            lambda0_metrics = v1._metrics(outer_pair.h_payload, expected_lambda0)
            expected_manifest = _fold_payload(
                source=source,
                outer_fold_id=outer_fold_id,
                selected_lambda=selected_lambda,
                candidates=candidates,
                inner_pairs=inner_pairs,
                outer_pair=outer_pair,
                fused_binding=fused_binding,
                selected_probabilities=saved_selected,
                lambda0_probabilities=expected_lambda0,
                selected_metrics=selected_metrics,
                lambda0_metrics=lambda0_metrics,
            )
            manifest_path = fold_dir / FOLD_MANIFEST_FILENAME
            manifest = v1._read_json_strict(manifest_path)
            if v1._canonical_json(manifest) != v1._canonical_json(expected_manifest):
                _fail(f"seed-{seed} fold manifest replay differs: {outer_fold_id}")
            manifest_binding = v1._binding(manifest_path, output_root)
            fold_records.append((manifest, manifest_binding, fused_binding))
            observed_symmetry = max(
                observed_symmetry,
                outer_pair.a_symmetry_max_abs_diff,
                *(pair.a_symmetry_max_abs_diff for pair in inner_pairs),
            )
            observed_symmetry_ulp = max(
                observed_symmetry_ulp,
                _participant_symmetry_diagnostics(outer_pair.a_payload)[1],
                *(
                    _participant_symmetry_diagnostics(pair.a_payload)[1]
                    for pair in inner_pairs
                ),
            )
            sessions = v1.thesis_backend._sessions_from_prediction(outer_pair.h_payload)
            selected_sessions.extend(sessions)
            selected_probabilities.extend(saved_selected)
            lambda0_sessions.extend(sessions)
            lambda0_probabilities.extend(expected_lambda0)
        selected_pooled = v1.thesis_backend._metrics(
            tuple(selected_sessions),
            tuple(selected_probabilities),
        )
        lambda0_pooled = v1.thesis_backend._metrics(
            tuple(lambda0_sessions),
            tuple(lambda0_probabilities),
        )
        expected_aggregate = _seed_aggregate_payload(
            seed=seed,
            protocol_binding=protocol_binding,
            fold_records=fold_records,
            selected_metrics=selected_pooled,
            lambda0_metrics=lambda0_pooled,
        )
        aggregate_path = output_root / f"seed_{seed}" / SEED_AGGREGATE_FILENAME
        aggregate = v1._read_json_strict(aggregate_path)
        if v1._canonical_json(aggregate) != v1._canonical_json(expected_aggregate):
            _fail(f"seed-{seed} aggregate differs from saved-probability replay")
        seed_aggregates[seed] = (aggregate, v1._binding(aggregate_path, output_root))
    expected_multiseed = _multiseed_aggregate_payload(
        inputs=inputs,
        protocol_binding=protocol_binding,
        seed_aggregates=seed_aggregates,
    )
    multiseed_path = output_root / MULTISEED_AGGREGATE_FILENAME
    multiseed = v1._read_json_strict(multiseed_path)
    if v1._canonical_json(multiseed) != v1._canonical_json(expected_multiseed):
        _fail("three-seed aggregate differs from seed-level replay")
    _revalidate_frozen_inputs(inputs)
    return {
        "audit_passed": True,
        "output_root": str(output_root),
        "multiseed_aggregate_path": str(multiseed_path),
        "multiseed_aggregate_sha256": v1._sha256_file(multiseed_path),
        "seed_aggregate_sha256": {
            str(seed): seed_aggregates[seed][1].sha256 for seed in ALLOWED_NEW_SEEDS
        },
        "selected_lambdas": {
            str(seed): multiseed["per_seed"][str(seed)]["selected_lambdas"]
            for seed in ALL_REPORTING_SEEDS
        },
        "a_participant_symmetry_observed_max_abs_diff": observed_symmetry,
        "a_participant_symmetry_observed_max_float32_ulp_distance": (
            observed_symmetry_ulp
        ),
        "a_participant_symmetry_max_abs_diff_limit": A_SYMMETRY_ABS_TOLERANCE,
        "a_participant_symmetry_max_float32_ulp_distance_limit": (
            A_SYMMETRY_ULP_TOLERANCE
        ),
        "significance_tests_run": False,
    }


def _publish_with_audit(
    output_root_value: str | Path = DEFAULT_OUTPUT,
) -> tuple[Path, dict[str, object]]:
    output_root = Path(output_root_value).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite multiseed residual fusion: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.stage"
    if stage.exists():
        _fail("unexpected staging path collision")
    inputs = _load_frozen_inputs()
    published = False
    published_identity: tuple[int, int] | None = None
    try:
        stage.mkdir()
        protocol = _protocol_payload(inputs)
        protocol_path = stage / PROTOCOL_FILENAME
        v1._write_json_exclusive(protocol_path, protocol)
        protocol_binding = v1._binding(protocol_path, stage)
        seed_aggregates = {
            seed: _build_seed(stage, inputs.sources[seed], protocol_binding)
            for seed in ALLOWED_NEW_SEEDS
        }
        multiseed = _multiseed_aggregate_payload(
            inputs=inputs,
            protocol_binding=protocol_binding,
            seed_aggregates=seed_aggregates,
        )
        multiseed_path = stage / MULTISEED_AGGREGATE_FILENAME
        v1._write_json_exclusive(multiseed_path, multiseed)
        _audit_multiseed(stage, validated_inputs=inputs)
        if output_root.exists():
            raise FileExistsError(
                f"refusing to overwrite multiseed residual fusion: {output_root}"
            )
        published_identity = v1._directory_identity(stage)
        os.rename(stage, output_root)
        published = True
        receipt = _audit_multiseed(output_root, validated_inputs=inputs)
        return output_root / MULTISEED_AGGREGATE_FILENAME, receipt
    except BaseException as exc:
        if stage.exists() and stage.is_dir() and not stage.is_symlink():
            shutil.rmtree(stage)
        if published:
            try:
                current_identity = v1._directory_identity(output_root)
            except v1.ResidualLateFusionError:
                current_identity = None
            if current_identity != published_identity:
                raise ResidualLateFusionMultiseedError(
                    "post-rename audit failed and output identity changed; refusing to delete "
                    "an unowned replacement"
                ) from exc
            shutil.rmtree(output_root)
        raise


def publish_residual_late_fusion_multiseed(
    output_root_value: str | Path = DEFAULT_OUTPUT,
) -> Path:
    """Create the frozen seed-42/73 replay once and return its aggregate."""

    aggregate_path, _receipt = _publish_with_audit(output_root_value)
    return aggregate_path


def audit_residual_late_fusion_multiseed(
    output_root_value: str | Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Independently cold-audit both sources and replay every saved result."""

    return _audit_multiseed(output_root_value, validated_inputs=None)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Run or audit frozen seed-42/73 residual late-fusion stability replay."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Create the exact frozen replay once.")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    audit = commands.add_parser("audit", help="Cold-audit an existing replay.")
    audit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "run":
        aggregate, receipt = _publish_with_audit(args.output)
        receipt["multiseed_aggregate_path"] = str(aggregate)
    else:
        receipt = audit_residual_late_fusion_multiseed(args.output)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
