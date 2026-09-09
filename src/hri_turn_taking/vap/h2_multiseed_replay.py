"""Isolated seed-42/73 extension of the sealed H2 matched replay.

The completed seed-13 replay binds :mod:`h2_matched_replay` by SHA-256.  This
module therefore does not modify or replace that implementation.  It reuses
its causal policy, event, metric, and E-only acoustic primitives through an
explicitly hash-bound dependency and owns a separate version-2 artifact
contract for model seeds 42 and 73.

The phase boundary remains strict: ``freeze`` is the only phase that projects
the legacy outer prediction (which embeds gold) into a self-contained
gold-free view; ``calibrate_and_seal_traces`` may use inner OOF gold but never
outer gold; ``evaluate`` opens outer gold only after the trace seal exists.
All publication is create-once and post-hoc/non-production/non-counted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

from hri_turn_taking.vap import h2_matched_replay as v1
from hri_turn_taking.vap import thesis_backend
from hri_turn_taking.vap.outer_experiment import (
    DEADLINE_CORPUS_EXCLUSIONS,
    DEADLINE_EXPECTED_N_ROWS,
    DEADLINE_PROTOCOL_ID,
    DEADLINE_SESSION_TO_GROUP,
    validate_outer_run_manifest,
    validate_outer_split_manifest,
)

SUPPORTED_MODEL_SEEDS = (42, 73)
ALL_REPLAY_MODEL_SEEDS = (13, 42, 73)
CALIBRATION_ALGORITHM_SEED = 13
V1_IMPLEMENTATION_SHA256 = (
    "30511969e2a5e3ca8fe3a138fef12494da3d0700ee5364a26538aba49248fd3c"
)

RESULT_ROLE = "posthoc_exploratory_h2_matched_replay_multiseed"
PRODUCTION_ELIGIBLE = False
COUNTED_ELIGIBLE = False
FRAME_RATE_HZ = v1.FRAME_RATE_HZ
OUTER_FOLD_IDS = v1.OUTER_FOLD_IDS
INNER_FOLD_IDS = v1.INNER_FOLD_IDS
ANALYSIS_ORDER = ("primary_5ep", "sensitivity_15ep")
BUDGET_EPOCHS = {"primary_5ep": 5, "sensitivity_15ep": 15}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACOUSTIC_ROOT = v1.ACOUSTIC_ROOT
SEED13_ROOT = PROJECT_ROOT / "outputs" / "h2_matched_replay_posthoc_v1"
SEED13_EVALUATION_ROOT = SEED13_ROOT / "evaluation"
SEED13_EVALUATION_SHA256 = (
    "f0bf3c63d7db9f1d70627dab9455d0b51bcc6d7aae11d86813192cecfb506f1f"
)
SEED13_EVENT_SHA256 = (
    "535895be9277bc3f1dec49676b629e80db83c1605a60510fa1fa33c780637518"
)
SEED13_METRICS_SHA256 = (
    "58302f9e74868fb1ad02bffbe9c20a6f38b85c35d9babe2b62fa4d8b612cb853"
)

FREEZE_FILENAME = "freeze_manifest.json"
PROJECTION_SEAL_FILENAME = "projection_seal.json"
TRACE_SEAL_FILENAME = "trace_seal_manifest.json"
EVALUATION_FILENAME = "evaluation_manifest.json"
EVENT_FILENAME = "matched_robot_entry_events.csv"
METRICS_FILENAME = "matched_metrics.json"
AGGREGATE_FILENAME = "multiseed_aggregate_manifest.json"
AGGREGATE_METRICS_FILENAME = "multiseed_metrics.json"

TRACE_COLUMNS = ("model_seed", *v1.TRACE_COLUMNS)
EVENT_COLUMNS = ("model_seed", *v1.EVENT_COLUMNS)


@dataclass(frozen=True)
class FrozenSourceSpec:
    """One exact already-audited matrix/aggregate pair."""

    matrix: Path
    matrix_index_sha256: str
    aggregate: Path
    aggregate_sha256: str
    max_epochs: int
    run_manifest_payload_sha256: str


CANONICAL_SEED_SOURCES: dict[int, dict[str, FrozenSourceSpec]] = {
    42: {
        "primary_5ep": FrozenSourceSpec(
            matrix=(
                PROJECT_ROOT
                / "outputs"
                / "thesis_multiseed_5epoch_20260818"
                / "matrix_seed42_v1"
            ),
            matrix_index_sha256=(
                "c0f4467c655d94a4a61528c1815c48e4a53da79fa2a8ac757af0dd5e5ddfcff7"
            ),
            aggregate=(
                PROJECT_ROOT
                / "outputs"
                / "thesis_multiseed_5epoch_20260818"
                / "aggregate_seed42_v1.json"
            ),
            aggregate_sha256=(
                "7cdd0d126569c073773066419f8fe9e5d71af74480d480d7d8683ff7f973609f"
            ),
            max_epochs=5,
            run_manifest_payload_sha256=(
                "b1a5c2385c6b39eb75abc389a7194528918c8341b14a20218844818fed72918b"
            ),
        ),
        "sensitivity_15ep": FrozenSourceSpec(
            matrix=(
                PROJECT_ROOT
                / "outputs"
                / "thesis_multiseed_15epoch_20260818"
                / "matrix_seed42_v1"
            ),
            matrix_index_sha256=(
                "187b98b965b35b8290649da66c4ba34c59c9f957d07af2af525da694740b233a"
            ),
            aggregate=(
                PROJECT_ROOT
                / "outputs"
                / "thesis_multiseed_15epoch_20260818"
                / "aggregate_seed42_v1.json"
            ),
            aggregate_sha256=(
                "4acaaca61a3b080271014fef8b3e1381478351174ab4800e6c2f270f3333dc50"
            ),
            max_epochs=15,
            run_manifest_payload_sha256=(
                "a82c31bd6e4e052462dd0452ccd65bf11e92313871edd74b517360e5d40f1e74"
            ),
        ),
    },
    73: {
        "primary_5ep": FrozenSourceSpec(
            matrix=(
                PROJECT_ROOT
                / "outputs"
                / "thesis_multiseed_5epoch_20260818"
                / "matrix_seed73_v1"
            ),
            matrix_index_sha256=(
                "f6b8811a5428d80073d89ece6003cbaadc5d3253fbf43d81ece661b43b544a3a"
            ),
            aggregate=(
                PROJECT_ROOT
                / "outputs"
                / "thesis_multiseed_5epoch_20260818"
                / "aggregate_seed73_v1.json"
            ),
            aggregate_sha256=(
                "527e28f6f300415c2c5c7e6d8f20739433d077106de252ae796bd0b231fa168f"
            ),
            max_epochs=5,
            run_manifest_payload_sha256=(
                "b1a5c2385c6b39eb75abc389a7194528918c8341b14a20218844818fed72918b"
            ),
        ),
        "sensitivity_15ep": FrozenSourceSpec(
            matrix=(
                PROJECT_ROOT
                / "outputs"
                / "thesis_multiseed_15epoch_20260818"
                / "matrix_seed73_v1"
            ),
            matrix_index_sha256=(
                "1f2a7691351314a796e1502e4671b7ccb445cf7c90a00c14bca0ff48cd05167b"
            ),
            aggregate=(
                PROJECT_ROOT
                / "outputs"
                / "thesis_multiseed_15epoch_20260818"
                / "aggregate_seed73_v1.json"
            ),
            aggregate_sha256=(
                "dba7f5f0ed256acd5d0195a444709e4fe1efb3eced9df0cd809ac4d32b187514"
            ),
            max_epochs=15,
            run_manifest_payload_sha256=(
                "a82c31bd6e4e052462dd0452ccd65bf11e92313871edd74b517360e5d40f1e74"
            ),
        ),
    },
}


class H2MultiSeedReplayError(ValueError):
    """A version-2 source, leakage, publication, or audit gate failed."""


def _fail(message: str) -> None:
    raise H2MultiSeedReplayError(message)


def _require_model_seed(value: object, *, extension_only: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("model_seed must be one exact integer")
    allowed = SUPPORTED_MODEL_SEEDS if extension_only else ALL_REPLAY_MODEL_SEEDS
    if value not in allowed:
        _fail(f"model_seed must be one of {allowed}")
    return value


def _sha256(path: Path) -> str:
    return v1._sha256_file(path)


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_bindings() -> dict[str, object]:
    v1_path = Path(v1.__file__).resolve()
    if _sha256(v1_path) != V1_IMPLEMENTATION_SHA256:
        _fail("seed-13 v1 implementation bytes changed; refusing v2 reuse")
    return {
        "v2": v1._file_binding(Path(__file__)),
        "v1_causal_primitives": v1._file_binding(v1_path),
        "expected_v1_sha256": V1_IMPLEMENTATION_SHA256,
        "reused_primitives": [
            "load_e_only_acoustic_descendant",
            "fit_cross_fitted_speech_detector",
            "select_speech_threshold",
            "replay_matched_policy",
            "score_trace_events",
            "validate_exact_event_support",
        ],
    }


def _verify_implementation_bindings(payload: object) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "v2",
        "v1_causal_primitives",
        "expected_v1_sha256",
        "reused_primitives",
    }:
        _fail("implementation dependency bindings are not exact")
    v2_path = v1.verify_file_binding(payload["v2"], role="H2 v2 implementation")
    v1_path = v1.verify_file_binding(
        payload["v1_causal_primitives"], role="H2 v1 primitive dependency"
    )
    if (
        v2_path != Path(__file__).resolve()
        or v1_path != Path(v1.__file__).resolve()
        or payload["expected_v1_sha256"] != V1_IMPLEMENTATION_SHA256
        or _sha256(v1_path) != V1_IMPLEMENTATION_SHA256
        or payload["reused_primitives"]
        != [
            "load_e_only_acoustic_descendant",
            "fit_cross_fitted_speech_detector",
            "select_speech_threshold",
            "replay_matched_policy",
            "score_trace_events",
            "validate_exact_event_support",
        ]
    ):
        _fail("implementation dependency identity differs")


def _owned_tree_unchanged(path: Path, identity: tuple[int, int]) -> bool:
    try:
        observed = os.lstat(path)
    except OSError:
        return False
    return (
        not path.is_symlink()
        and path.is_dir()
        and (observed.st_dev, observed.st_ino) == identity
    )


def _publish_with_owned_rollback(
    destination: str | Path,
    files: dict[Path, bytes],
    audit: Callable[[Path], object],
) -> Path:
    """Publish once and remove only the same just-created tree on audit failure."""

    published = v1._publish_directory_create_once(destination, files)
    observed = os.lstat(published)
    identity = (observed.st_dev, observed.st_ino)
    try:
        audit(published)
    except Exception:
        if _owned_tree_unchanged(published, identity):
            shutil.rmtree(published)
        raise
    return published


def _source_spec(model_seed: int, budget_id: str) -> FrozenSourceSpec:
    seed = _require_model_seed(model_seed)
    if budget_id not in ANALYSIS_ORDER:
        _fail("budget_id is not one of the exact H2 analysis budgets")
    spec = CANONICAL_SEED_SOURCES[seed][budget_id]
    if spec.max_epochs != BUDGET_EPOCHS[budget_id]:
        _fail("configured source epoch budget differs")
    return spec


def _load_bound_aggregate(
    spec: FrozenSourceSpec,
    *,
    model_seed: int,
) -> dict[str, object]:
    path = v1._readable_local_path(spec.aggregate, "independently audited aggregate")
    if path != spec.aggregate.resolve() or _sha256(path) != spec.aggregate_sha256:
        _fail("independently audited aggregate path/hash differs from frozen source")
    payload = v1._read_json(path)
    required = {
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
    condition = payload.get("conditions", {}).get("A+H")
    if (
        set(payload) != required
        or payload["artifact_kind"] != "vap_thesis_primary_matrix_aggregate"
        or payload["format_version"] != 1
        or payload["result_role"] != thesis_backend.RESULT_ROLE
        or payload["data_kind"] != "real"
        or payload["production_eligible"] is not False
        or payload["counted_eligible"] is not False
        or payload["analysis_variant"] != "primary"
        or payload["seed"] != model_seed
        or payload["included_session_ids"] != list(DEADLINE_SESSION_TO_GROUP)
        or payload["group_ids"] != sorted(set(DEADLINE_SESSION_TO_GROUP.values()))
        or payload["corpus_exclusions"] != dict(DEADLINE_CORPUS_EXCLUSIONS)
        or payload["aggregation_source"]
        != "saved_outer_probabilities_only_no_model_inference"
        or payload["run_manifest_sha256"] != spec.run_manifest_payload_sha256
        or not isinstance(condition, dict)
        or condition.get("condition_id") != "A+H"
        or condition.get("feature_set") != "audio_plus_head_pose"
        or condition.get("seed") != model_seed
        or condition.get("run_manifest_sha256") != spec.run_manifest_payload_sha256
        or condition.get("aggregation_source")
        != "saved_outer_probabilities_only_no_model_inference"
    ):
        _fail("independently audited aggregate identity differs")
    return payload


def _validate_matrix_index(
    spec: FrozenSourceSpec,
    *,
    budget_id: str,
    model_seed: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    matrix_root = spec.matrix.resolve()
    if matrix_root.is_symlink() or not matrix_root.is_dir():
        _fail("frozen matrix root must be one existing non-symlink directory")
    index_path = v1._readable_local_path(matrix_root / "matrix_index.json", "matrix index")
    if _sha256(index_path) != spec.matrix_index_sha256:
        _fail("matrix index bytes differ from the frozen multi-seed source")
    index = v1._read_json(index_path)
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
        "included_session_ids",
        "group_ids",
        "corpus_exclusions",
        "units",
    }
    if set(index) != required:
        _fail(f"{budget_id} matrix index fields are not exact")
    if (
        index["artifact_kind"] != "vap_thesis_primary_matrix_index"
        or index["format_version"] != 1
        or index["result_role"] != thesis_backend.RESULT_ROLE
        or index["data_kind"] != "real"
        or index["production_eligible"] is not False
        or index["counted_eligible"] is not False
        or index["analysis_variant"] != "primary"
        or index["seed"] != model_seed
        or index["condition_ids"] != ["A", "H", "A+H"]
        or index["outer_fold_ids"] != list(OUTER_FOLD_IDS)
        or index["expected_unit_count"] != 12
        or index["completed_unit_count"] != 12
        or index["included_session_ids"] != list(DEADLINE_SESSION_TO_GROUP)
        or index["group_ids"] != sorted(set(DEADLINE_SESSION_TO_GROUP.values()))
        or index["corpus_exclusions"] != dict(DEADLINE_CORPUS_EXCLUSIONS)
        or index["run_manifest_sha256"] != spec.run_manifest_payload_sha256
    ):
        _fail(f"{budget_id} matrix identity is invalid or seed-mixed")

    protocol = v1._reject_f_path(index["protocol_directory"], "outer protocol")
    split_path = v1._readable_local_path(protocol / "split_manifest.json", "split manifest")
    run_path = v1._readable_local_path(protocol / "run_manifest.json", "run manifest")
    split = v1._read_json(split_path)
    run = v1._read_json(run_path)
    try:
        validate_outer_split_manifest(split)
        validate_outer_run_manifest(run, split)
    except ValueError as exc:
        raise H2MultiSeedReplayError(f"{budget_id} outer protocol is invalid: {exc}") from exc
    if (
        split.get("protocol_id") != DEADLINE_PROTOCOL_ID
        or split.get("source_session_to_group") != dict(DEADLINE_SESSION_TO_GROUP)
        or split.get("corpus_exclusions") != dict(DEADLINE_CORPUS_EXCLUSIONS)
        or run.get("max_epochs") != spec.max_epochs
        or run.get("seeds") != list(ALL_REPLAY_MODEL_SEEDS)
        or _canonical_json_sha256(run) != spec.run_manifest_payload_sha256
    ):
        _fail(f"{budget_id} canonical run/split contract differs")
    aggregate = _load_bound_aggregate(spec, model_seed=model_seed)
    return index, split, run, aggregate


def _validate_inner_checkpoint_scope(
    checkpoint: dict[str, object],
    prediction: dict[str, object],
    *,
    checkpoint_sha256: str,
    fold: dict[str, object],
    inner: dict[str, object],
    model_seed: int,
) -> None:
    train_ids = inner["train_session_ids"]
    validation_ids = inner["validation_session_ids"]
    outer_ids = fold["outer_test_session_ids"]
    if (
        checkpoint.get("condition_id") != "A+H"
        or checkpoint.get("outer_fold_id") != fold["outer_fold_id"]
        or checkpoint.get("inner_fold_id") != inner["inner_fold_id"]
        or checkpoint.get("fit_role") != "inner_selection"
        or checkpoint.get("seed") != model_seed
        or checkpoint.get("train_session_ids") != train_ids
        or set(train_ids) & set(validation_ids)
        or set(train_ids) & set(outer_ids)
        or set(validation_ids) & set(outer_ids)
        or prediction.get("seed") != model_seed
        or prediction.get("checkpoint_sha256") != checkpoint_sha256
    ):
        _fail("inner checkpoint/OOF scope or model_seed differs")
    groups = {DEADLINE_SESSION_TO_GROUP[scene] for scene in train_ids}
    if groups != set(inner["train_group_ids"]):
        _fail("inner checkpoint does not use exactly the other two groups")


def _validate_final_checkpoint_scope(
    checkpoint: dict[str, object],
    prediction: dict[str, object],
    *,
    checkpoint_sha256: str,
    fold: dict[str, object],
    model_seed: int,
) -> None:
    train_ids = fold["outer_train_session_ids"]
    test_ids = fold["outer_test_session_ids"]
    if (
        checkpoint.get("condition_id") != "A+H"
        or checkpoint.get("outer_fold_id") != fold["outer_fold_id"]
        or checkpoint.get("inner_fold_id") is not None
        or checkpoint.get("fit_role") != "fixed_epoch_outer_train_from_scratch"
        or checkpoint.get("seed") != model_seed
        or checkpoint.get("train_session_ids") != train_ids
        or set(train_ids) & set(test_ids)
        or prediction.get("seed") != model_seed
        or prediction.get("checkpoint_sha256") != checkpoint_sha256
    ):
        _fail("final checkpoint/outer scope or model_seed differs")


def _gold_free_probability_view(
    prediction: dict[str, object],
    *,
    budget_id: str,
    max_epochs: int,
    model_seed: int,
) -> dict[str, object]:
    sessions = prediction.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        _fail("saved prediction has no sessions")
    views = [v1._validate_probability_session(record) for record in sessions]
    if (
        prediction.get("condition_id") != "A+H"
        or prediction.get("seed") != model_seed
        or prediction.get("frame_rate_hz") != FRAME_RATE_HZ
    ):
        _fail("saved outer prediction identity differs before projection")
    return {
        "artifact_kind": "vap_h2_multiseed_gold_free_outer_probability_view",
        "format_version": 2,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "budget_id": budget_id,
        "max_epochs": max_epochs,
        "condition_id": "A+H",
        "outer_fold_id": prediction.get("outer_fold_id"),
        "model_seed": model_seed,
        "frame_rate_hz": FRAME_RATE_HZ,
        "checkpoint_sha256": prediction.get("checkpoint_sha256"),
        "run_manifest_sha256": prediction.get("run_manifest_sha256"),
        "allowed_session_fields": [
            "session_id",
            "group_id",
            "n_frames",
            "tick_25hz",
            "probabilities",
            "present",
        ],
        "forbidden_fields": ["labels", "vad", "valid", "features", "floor_holder"],
        "sessions": [
            {
                "session_id": view.session_id,
                "group_id": view.group_id,
                "n_frames": int(view.ticks.size),
                "tick_25hz": view.ticks.tolist(),
                "probabilities": view.probabilities.tolist(),
                "present": view.present.tolist(),
            }
            for view in views
        ],
    }


def load_gold_free_probability_view(
    path_value: str | Path,
    *,
    expected_model_seed: int,
) -> tuple[v1.GoldFreeSessionView, ...]:
    seed = _require_model_seed(expected_model_seed)
    payload = v1._read_json(path_value)
    required = {
        "artifact_kind",
        "format_version",
        "result_role",
        "production_eligible",
        "counted_eligible",
        "budget_id",
        "max_epochs",
        "condition_id",
        "outer_fold_id",
        "model_seed",
        "frame_rate_hz",
        "checkpoint_sha256",
        "run_manifest_sha256",
        "allowed_session_fields",
        "forbidden_fields",
        "sessions",
    }
    if (
        set(payload) != required
        or payload["artifact_kind"]
        != "vap_h2_multiseed_gold_free_outer_probability_view"
        or payload["format_version"] != 2
        or payload["result_role"] != RESULT_ROLE
        or payload["production_eligible"] is not False
        or payload["counted_eligible"] is not False
        or payload["condition_id"] != "A+H"
        or payload["model_seed"] != seed
        or payload["frame_rate_hz"] != FRAME_RATE_HZ
    ):
        _fail("gold-free view identity differs or attempted promotion")
    allowed = {
        "session_id",
        "group_id",
        "n_frames",
        "tick_25hz",
        "probabilities",
        "present",
    }
    if payload["allowed_session_fields"] != list(allowed):
        # Set order is intentionally not relied upon below; require the emitted order.
        expected = [
            "session_id",
            "group_id",
            "n_frames",
            "tick_25hz",
            "probabilities",
            "present",
        ]
        if payload["allowed_session_fields"] != expected:
            _fail("gold-free view allowed-field declaration differs")
    if payload["forbidden_fields"] != [
        "labels",
        "vad",
        "valid",
        "features",
        "floor_holder",
    ]:
        _fail("gold-free view forbidden-field declaration differs")
    records = payload["sessions"]
    if not isinstance(records, list) or not records:
        _fail("gold-free view has no sessions")
    result = []
    for record in records:
        if not isinstance(record, dict) or set(record) != allowed:
            _fail("gold-free session fields are not exact")
        synthetic = {
            **record,
            "labels": np.zeros((record["n_frames"], 4, 4)).tolist(),
            "vad": np.zeros((record["n_frames"], 4)).tolist(),
            "valid": np.ones(record["n_frames"]).tolist(),
        }
        result.append(v1._validate_probability_session(synthetic))
    return tuple(result)


def _freeze_budget(
    spec: FrozenSourceSpec,
    *,
    destination: Path,
    budget_id: str,
    model_seed: int,
) -> tuple[dict[str, object], dict[Path, bytes]]:
    index, split, run, aggregate = _validate_matrix_index(
        spec, budget_id=budget_id, model_seed=model_seed
    )
    folds = v1._folds_by_id(split)
    units = v1._matrix_unit_records(index)
    sources = v1._source_artifacts_by_scene(run)
    files: dict[Path, bytes] = {}
    fold_records: dict[str, object] = {}
    outer_hashes: list[str] = []
    for fold_id in OUTER_FOLD_IDS:
        unit = units[("A+H", fold_id)]
        result_path = v1._readable_local_path(
            unit.get("result_manifest"), "A+H result manifest"
        )
        expected_result = (
            spec.matrix / "A_plus_H" / fold_id / "result_manifest.json"
        ).resolve()
        if (
            result_path != expected_result
            or unit.get("seed") != model_seed
            or unit.get("result_manifest_sha256") != _sha256(result_path)
        ):
            _fail(f"{budget_id}/{fold_id} result binding or model_seed differs")
        result = v1._read_json(result_path)
        fold = folds[fold_id]
        if (
            result.get("result_role") != "backend_contract_smoke"
            or result.get("counted_experiment_complete") is not False
            or result.get("analysis_variant") != "primary"
            or result.get("condition_id") != "A+H"
            or result.get("feature_set") != "audio_plus_head_pose"
            or result.get("outer_fold_id") != fold_id
            or result.get("seed") != model_seed
            or result.get("outer_test_session_ids") != fold["outer_test_session_ids"]
            or result.get("outer_evaluation_count") != 1
            or result.get("run_manifest_sha256") != spec.run_manifest_payload_sha256
        ):
            _fail(f"{budget_id}/{fold_id} result identity differs")
        bindings = result.get("artifact_bindings")
        if not isinstance(bindings, dict):
            _fail(f"{budget_id}/{fold_id} result lacks artifact bindings")
        outer_path = v1._bound_result_artifact(
            result_path.parent, bindings.get("outer_predictions"), role="outer prediction"
        )
        outer = thesis_backend.load_prediction_artifact(outer_path)
        if (
            outer.get("condition_id") != "A+H"
            or outer.get("outer_fold_id") != fold_id
            or outer.get("seed") != model_seed
            or outer.get("run_manifest_sha256") != spec.run_manifest_payload_sha256
            or [record["session_id"] for record in outer["sessions"]]
            != fold["outer_test_session_ids"]
        ):
            _fail(f"{budget_id}/{fold_id} outer prediction identity differs")
        outer_hashes.append(_sha256(outer_path))
        view = _gold_free_probability_view(
            outer,
            budget_id=budget_id,
            max_epochs=spec.max_epochs,
            model_seed=model_seed,
        )
        view_bytes = v1._pretty_json_bytes(view)
        view_relative = Path("outer_probability_views") / budget_id / f"{fold_id}.json"
        files[view_relative] = view_bytes

        inner_bindings = bindings.get("inner_oof_predictions")
        checkpoint_bindings = bindings.get("inner_checkpoints")
        if not isinstance(inner_bindings, list) or len(inner_bindings) != 3:
            _fail(f"{budget_id}/{fold_id} must bind three inner OOF artifacts")
        if not isinstance(checkpoint_bindings, list) or len(checkpoint_bindings) != 3:
            _fail(f"{budget_id}/{fold_id} must bind three inner checkpoints")
        frozen_inner = []
        for inner, binding, checkpoint_binding in zip(
            fold["inner_folds"], inner_bindings, checkpoint_bindings, strict=True
        ):
            inner_id = inner["inner_fold_id"]
            path = v1._bound_result_artifact(
                result_path.parent,
                binding,
                role="inner OOF prediction",
                inner_fold_id=inner_id,
            )
            payload = thesis_backend._load_prediction_artifact(
                path, expected_inner_fold_id=inner_id
            )
            checkpoint_path = v1._bound_result_artifact(
                result_path.parent,
                checkpoint_binding,
                role="inner checkpoint",
                inner_fold_id=inner_id,
            )
            checkpoint = thesis_backend.load_checkpoint_artifact(checkpoint_path)
            if (
                payload.get("condition_id") != "A+H"
                or payload.get("outer_fold_id") != fold_id
                or payload.get("seed") != model_seed
                or payload.get("run_manifest_sha256") != spec.run_manifest_payload_sha256
                or [record["session_id"] for record in payload["sessions"]]
                != inner["validation_session_ids"]
            ):
                _fail(f"{budget_id}/{fold_id}/{inner_id} inner OOF identity differs")
            _validate_inner_checkpoint_scope(
                checkpoint,
                payload,
                checkpoint_sha256=str(checkpoint_binding["sha256"]),
                fold=fold,
                inner=inner,
                model_seed=model_seed,
            )
            frozen_inner.append(
                {
                    "inner_fold_id": inner_id,
                    "train_group_ids": inner["train_group_ids"],
                    "train_session_ids": inner["train_session_ids"],
                    "validation_group_ids": inner["validation_group_ids"],
                    "validation_session_ids": inner["validation_session_ids"],
                    "artifact": v1._file_binding(path),
                    "checkpoint_provenance": v1._file_binding(checkpoint_path),
                }
            )
        final_path = v1._bound_result_artifact(
            result_path.parent, bindings.get("final_checkpoint"), role="final checkpoint"
        )
        final_checkpoint = thesis_backend.load_checkpoint_artifact(final_path)
        _validate_final_checkpoint_scope(
            final_checkpoint,
            outer,
            checkpoint_sha256=_sha256(final_path),
            fold=fold,
            model_seed=model_seed,
        )
        fold_records[fold_id] = {
            "outer_train_group_ids": fold["outer_train_group_ids"],
            "outer_test_group_ids": fold["outer_test_group_ids"],
            "outer_train_session_ids": fold["outer_train_session_ids"],
            "outer_test_session_ids": fold["outer_test_session_ids"],
            "result_manifest": v1._file_binding(result_path),
            "inner_oof": frozen_inner,
            "final_checkpoint_provenance": v1._file_binding(final_path),
            "outer_prediction_provenance": v1._file_binding(outer_path),
            "gold_free_outer_view": v1._planned_binding(
                destination / view_relative, view_bytes
            ),
        }

    aggregate_hashes = aggregate["conditions"]["A+H"].get(
        "prediction_artifact_sha256"
    )
    if aggregate_hashes != outer_hashes:
        _fail("aggregate A+H outer hashes differ from the four matrix artifacts")
    protocol = Path(index["protocol_directory"])
    return (
        {
            "budget_id": budget_id,
            "max_epochs": spec.max_epochs,
            "model_seed": model_seed,
            "matrix_index": v1._file_binding(spec.matrix / "matrix_index.json"),
            "independently_audited_aggregate": v1._file_binding(spec.aggregate),
            "protocol_directory": str(protocol.resolve()),
            "split_manifest": v1._file_binding(protocol / "split_manifest.json"),
            "run_manifest": v1._file_binding(protocol / "run_manifest.json"),
            "run_manifest_payload_sha256": spec.run_manifest_payload_sha256,
            "assembled_source_artifacts": sources,
            "outer_prediction_sha256_in_fold_order": outer_hashes,
            "folds": fold_records,
        },
        files,
    )


def _build_canonical_freeze_files(
    destination: Path,
    *,
    model_seed: int,
) -> dict[Path, bytes]:
    """Reconstruct one freeze tree from its canonical matrix/aggregate sources."""

    seed = _require_model_seed(model_seed)
    records: dict[str, object] = {}
    files: dict[Path, bytes] = {}
    for budget_id in ANALYSIS_ORDER:
        record, budget_files = _freeze_budget(
            _source_spec(seed, budget_id),
            destination=destination,
            budget_id=budget_id,
            model_seed=seed,
        )
        records[budget_id] = record
        files.update(budget_files)
    if records["primary_5ep"]["assembled_source_artifacts"] != records[
        "sensitivity_15ep"
    ]["assembled_source_artifacts"]:
        _fail("5/15-epoch sources are not identical")

    projection = {
        "artifact_kind": "vap_h2_multiseed_gold_free_projection_seal",
        "format_version": 2,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "model_seed": seed,
        "self_contained": True,
        "source_predictions_are_opaque_provenance": True,
        "analysis_order": list(ANALYSIS_ORDER),
        "views": {
            budget_id: {
                fold_id: fold["gold_free_outer_view"]
                for fold_id, fold in records[budget_id]["folds"].items()
            }
            for budget_id in ANALYSIS_ORDER
        },
    }
    projection_bytes = v1._pretty_json_bytes(projection)
    files[Path(PROJECTION_SEAL_FILENAME)] = projection_bytes
    manifest = {
        "artifact_kind": "vap_h2_multiseed_seed_freeze",
        "format_version": 2,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
        "analysis_order": list(ANALYSIS_ORDER),
        "condition_id": "A+H",
        "model_seed": seed,
        "calibration_algorithm_seed": CALIBRATION_ALGORITHM_SEED,
        "corpus": {
            "included_session_ids": list(DEADLINE_SESSION_TO_GROUP),
            "session_to_group": dict(DEADLINE_SESSION_TO_GROUP),
            "row_count": DEADLINE_EXPECTED_N_ROWS,
            "corpus_exclusions": dict(DEADLINE_CORPUS_EXCLUSIONS),
        },
        "budgets": records,
        "projection_seal": v1._planned_binding(
            destination / PROJECTION_SEAL_FILENAME, projection_bytes
        ),
        "outer_boundary": {
            "strict_saved_artifact_validated_before_projection": True,
            "trace_phase_allowed_fields": [
                "session_id",
                "group_id",
                "tick_25hz",
                "probabilities",
                "present",
            ],
            "trace_phase_forbidden_fields": ["labels", "vad", "valid", "floor_holder"],
            "outer_gold_opened_during_freeze_projection": True,
            "reason": "legacy saved artifact embeds gold; projection discards it",
        },
        "acoustic_boundary": {
            "loader": "E_only_sealed_acoustic_descendant",
            "raw_source_replay": "skipped",
            "F_drive_paths_dereferenced": False,
        },
        "implementation": _implementation_bindings(),
    }
    files[Path(FREEZE_FILENAME)] = v1._pretty_json_bytes(manifest)
    return files


def _freeze_tree_bytes(root: Path) -> dict[Path, bytes]:
    observed: dict[Path, bytes] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            _fail("canonical freeze tree must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            _fail("canonical freeze tree contains a non-regular entry")
        relative = path.relative_to(root)
        try:
            observed[relative] = path.read_bytes()
        except OSError as exc:
            raise H2MultiSeedReplayError(
                f"cannot read canonical freeze artifact {relative}: {exc}"
            ) from exc
    return observed


def _cold_audit_freeze_against_canonical_sources(
    root: Path,
    freeze: dict[str, object],
) -> None:
    """Reopen frozen sources and require an exact canonical freeze reconstruction."""

    expected = _build_canonical_freeze_files(
        root,
        model_seed=_require_model_seed(freeze.get("model_seed")),
    )
    observed = _freeze_tree_bytes(root)
    if set(observed) != set(expected):
        _fail("published freeze file inventory differs from canonical reconstruction")
    for relative, expected_bytes in expected.items():
        if observed[relative] != expected_bytes:
            _fail(
                "published freeze bytes differ from canonical source reconstruction: "
                f"{relative.as_posix()}"
            )


def freeze_h2_multiseed_seed(destination: str | Path, *, model_seed: int) -> Path:
    """Freeze one exact seed-42/73 A+H pair and publish gold-free outer views."""

    seed = _require_model_seed(model_seed)
    output = v1._reject_f_path(destination, "H2 multi-seed freeze output")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite H2 multi-seed freeze: {output}")
    files = _build_canonical_freeze_files(output, model_seed=seed)
    published = _publish_with_owned_rollback(
        output,
        files,
        lambda root: audit_h2_multiseed_seed(root),
    )
    return published / FREEZE_FILENAME


def _load_freeze(path_value: str | Path) -> tuple[Path, dict[str, object]]:
    root = v1._reject_f_path(path_value, "H2 multi-seed freeze directory")
    if root.is_symlink() or not root.is_dir():
        _fail("H2 multi-seed freeze must be one existing non-symlink directory")
    manifest = v1._read_json(root / FREEZE_FILENAME)
    required = {
        "artifact_kind",
        "format_version",
        "result_role",
        "production_eligible",
        "counted_eligible",
        "h2_established",
        "analysis_order",
        "condition_id",
        "model_seed",
        "calibration_algorithm_seed",
        "corpus",
        "budgets",
        "projection_seal",
        "outer_boundary",
        "acoustic_boundary",
        "implementation",
    }
    seed = _require_model_seed(manifest.get("model_seed"))
    if (
        set(manifest) != required
        or manifest["artifact_kind"] != "vap_h2_multiseed_seed_freeze"
        or manifest["format_version"] != 2
        or manifest["result_role"] != RESULT_ROLE
        or manifest["production_eligible"] is not False
        or manifest["counted_eligible"] is not False
        or manifest["h2_established"] is not False
        or manifest["analysis_order"] != list(ANALYSIS_ORDER)
        or manifest["condition_id"] != "A+H"
        or manifest["calibration_algorithm_seed"] != CALIBRATION_ALGORITHM_SEED
        or manifest["corpus"]
        != {
            "included_session_ids": list(DEADLINE_SESSION_TO_GROUP),
            "session_to_group": dict(DEADLINE_SESSION_TO_GROUP),
            "row_count": DEADLINE_EXPECTED_N_ROWS,
            "corpus_exclusions": dict(DEADLINE_CORPUS_EXCLUSIONS),
        }
    ):
        _fail("H2 multi-seed freeze identity differs or attempted promotion")
    _verify_implementation_bindings(manifest["implementation"])
    budgets = manifest["budgets"]
    if not isinstance(budgets, dict) or tuple(budgets) != ANALYSIS_ORDER:
        _fail("H2 multi-seed freeze budgets are not exact")
    for budget_id in ANALYSIS_ORDER:
        budget = budgets[budget_id]
        if (
            not isinstance(budget, dict)
            or budget.get("budget_id") != budget_id
            or budget.get("max_epochs") != BUDGET_EPOCHS[budget_id]
            or budget.get("model_seed") != seed
            or budget.get("run_manifest_payload_sha256")
            != _source_spec(seed, budget_id).run_manifest_payload_sha256
        ):
            _fail(f"{budget_id} frozen identity differs")
        for role in (
            "matrix_index",
            "independently_audited_aggregate",
            "split_manifest",
            "run_manifest",
        ):
            v1._validate_opaque_binding(budget.get(role), role=f"{budget_id} {role}")
        hashes = budget.get("outer_prediction_sha256_in_fold_order")
        if (
            not isinstance(hashes, list)
            or len(hashes) != 4
            or not all(v1._is_sha256(value) for value in hashes)
        ):
            _fail(f"{budget_id} outer prediction hash list differs")
        folds = budget.get("folds")
        if not isinstance(folds, dict) or tuple(folds) != OUTER_FOLD_IDS:
            _fail(f"{budget_id} frozen folds differ")
        for fold_id, fold in folds.items():
            if not isinstance(fold, dict):
                _fail(f"{budget_id}/{fold_id} frozen fold is malformed")
            for role in (
                "result_manifest",
                "final_checkpoint_provenance",
                "outer_prediction_provenance",
            ):
                v1._validate_opaque_binding(fold.get(role), role=role)
            view_path = v1.verify_file_binding(
                fold.get("gold_free_outer_view"), role="gold-free outer view"
            )
            if root not in view_path.parents:
                _fail("gold-free view is not owned by the freeze directory")
            views = load_gold_free_probability_view(
                view_path, expected_model_seed=seed
            )
            if [view.session_id for view in views] != fold["outer_test_session_ids"]:
                _fail("gold-free view sessions differ from the frozen fold")
            inner = fold.get("inner_oof")
            if not isinstance(inner, list) or len(inner) != 3:
                _fail(f"{budget_id}/{fold_id} lacks three inner OOF bindings")
            for record, inner_id in zip(inner, INNER_FOLD_IDS, strict=True):
                if not isinstance(record, dict) or record.get("inner_fold_id") != inner_id:
                    _fail("inner OOF order differs")
                v1._validate_opaque_binding(record.get("artifact"), role="inner OOF")
                v1._validate_opaque_binding(
                    record.get("checkpoint_provenance"), role="inner checkpoint"
                )
    if budgets["primary_5ep"]["assembled_source_artifacts"] != budgets[
        "sensitivity_15ep"
    ]["assembled_source_artifacts"]:
        _fail("5/15-epoch frozen assembled sources differ")

    projection_path = v1.verify_file_binding(
        manifest["projection_seal"], role="gold-free projection seal"
    )
    if projection_path != (root / PROJECTION_SEAL_FILENAME).resolve():
        _fail("projection seal is not owned by the freeze directory")
    projection = v1._read_json(projection_path)
    if (
        set(projection)
        != {
            "artifact_kind",
            "format_version",
            "result_role",
            "production_eligible",
            "counted_eligible",
            "model_seed",
            "self_contained",
            "source_predictions_are_opaque_provenance",
            "analysis_order",
            "views",
        }
        or projection["artifact_kind"]
        != "vap_h2_multiseed_gold_free_projection_seal"
        or projection["format_version"] != 2
        or projection["result_role"] != RESULT_ROLE
        or projection["production_eligible"] is not False
        or projection["counted_eligible"] is not False
        or projection["model_seed"] != seed
        or projection["self_contained"] is not True
        or projection["source_predictions_are_opaque_provenance"] is not True
        or projection["analysis_order"] != list(ANALYSIS_ORDER)
    ):
        _fail("gold-free projection seal identity differs")
    expected_views = {
        budget_id: {
            fold_id: fold["gold_free_outer_view"]
            for fold_id, fold in budgets[budget_id]["folds"].items()
        }
        for budget_id in ANALYSIS_ORDER
    }
    if projection["views"] != expected_views:
        _fail("projection seal does not bind the exact frozen views")
    return root, manifest


def _calibration_session(
    record: dict[str, object],
    acoustic: v1.EOnlyAcousticSession,
) -> v1.CalibrationSession:
    view = v1._validate_probability_session(record)
    vad = np.asarray(record["vad"], dtype=np.float64)
    valid = np.asarray(record["valid"], dtype=bool)
    if vad.shape != (view.ticks.size, 4) or valid.shape != (view.ticks.size,):
        _fail("inner OOF gold shape differs from probability rows")
    if not np.array_equal(view.ticks, acoustic.ticks):
        _fail(f"inner OOF/acoustic keys differ for {view.session_id}")
    return v1.CalibrationSession(
        session_id=view.session_id,
        group_id=view.group_id,
        ticks=view.ticks,
        probabilities=view.probabilities,
        vad=vad,
        valid=valid,
        present=view.present,
        acoustic_features=acoustic.features,
    )


def _load_inner_calibration_sessions(
    budget: dict[str, object],
    fold_id: str,
    *,
    model_seed: int,
    acoustic_root: Path,
) -> tuple[tuple[v1.CalibrationSession, ...], dict[str, v1.EOnlyAcousticSession]]:
    fold = budget["folds"][fold_id]
    sources = budget["assembled_source_artifacts"]
    sessions: list[v1.CalibrationSession] = []
    acoustics: dict[str, v1.EOnlyAcousticSession] = {}
    for inner_record in fold["inner_oof"]:
        inner_id = inner_record["inner_fold_id"]
        path = v1.verify_file_binding(inner_record["artifact"], role="inner OOF artifact")
        payload = thesis_backend._load_prediction_artifact(
            path, expected_inner_fold_id=inner_id
        )
        if (
            payload.get("outer_fold_id") != fold_id
            or payload.get("condition_id") != "A+H"
            or payload.get("seed") != model_seed
            or payload.get("run_manifest_sha256")
            != budget["run_manifest_payload_sha256"]
        ):
            _fail("inner OOF identity differs during calibration")
        for record in payload["sessions"]:
            scene = record["session_id"]
            if scene not in fold["outer_train_session_ids"]:
                _fail("outer-test session attempted to enter inner calibration")
            if scene not in acoustics:
                acoustics[scene] = v1.load_e_only_acoustic_descendant(
                    scene,
                    assembled_source_artifact=sources[scene],
                    acoustic_root=acoustic_root,
                )
            sessions.append(_calibration_session(record, acoustics[scene]))
    if (
        [session.session_id for session in sessions] != fold["outer_train_session_ids"]
        or {session.group_id for session in sessions}
        != set(fold["outer_train_group_ids"])
        or {session.group_id for session in sessions}
        & set(fold["outer_test_group_ids"])
    ):
        _fail("inner calibration scope is not exactly the three outer-train groups")
    return tuple(sessions), acoustics


def _build_canonical_fold_calibration(
    budget: dict[str, object],
    *,
    budget_id: str,
    fold_id: str,
    model_seed: int,
    acoustic_root: Path,
) -> tuple[dict[str, object], dict[str, v1.EOnlyAcousticSession]]:
    """Refit one calibration from its bound inner OOF and E-only descendants."""

    seed = _require_model_seed(model_seed)
    fold = budget["folds"][fold_id]
    sessions, inner_acoustics = _load_inner_calibration_sessions(
        budget,
        fold_id,
        model_seed=seed,
        acoustic_root=acoustic_root,
    )
    calibration = v1._calibrate_fold(sessions)
    estimator = calibration["speech_detector"]["final_model"]["estimator"]
    if estimator.get("random_state") != CALIBRATION_ALGORITHM_SEED:
        _fail("speech-detector calibration algorithm seed differs from 13")
    calibration.update(
        {
            "model_seed": seed,
            "calibration_algorithm_seed": CALIBRATION_ALGORITHM_SEED,
            "budget_id": budget_id,
            "max_epochs": budget["max_epochs"],
            "outer_fold_id": fold_id,
            "outer_train_group_ids": fold["outer_train_group_ids"],
            "outer_test_group_ids": fold["outer_test_group_ids"],
            "outer_train_session_ids": fold["outer_train_session_ids"],
            "outer_test_session_ids": fold["outer_test_session_ids"],
            "inner_oof_sources": [item["artifact"] for item in fold["inner_oof"]],
            "inner_acoustic_sources": {
                scene: artifact.bindings for scene, artifact in inner_acoustics.items()
            },
            "outer_group_used_for_fit": False,
        }
    )
    return calibration, inner_acoustics


def _reactive_calibration_signature(calibration: dict[str, object]) -> dict[str, object]:
    return {
        "speech_detector": calibration["speech_detector"],
        "speech_threshold": calibration["speech_threshold"],
        "reactive_policy": calibration["reactive_policy"],
    }


def _trace_frame(
    views: tuple[v1.GoldFreeSessionView, ...],
    acoustics: dict[str, v1.EOnlyAcousticSession],
    calibration: dict[str, object],
    *,
    fold_id: str,
    model_seed: int,
) -> pd.DataFrame:
    frame = v1._outer_trace_frame(views, acoustics, calibration, fold_id=fold_id)
    frame.insert(0, "model_seed", model_seed)
    return frame.loc[:, TRACE_COLUMNS]


def _reactive_trace_signature(frame: pd.DataFrame) -> bytes:
    columns = [
        "session_id",
        "group_id",
        "outer_fold_id",
        "tick_25hz",
        "timestamp_s",
        "speech_probability",
        "speech_detected",
        "reactive_silent_run_frames",
        "reactive_ready",
        "reactive_trigger",
    ]
    return v1._csv_bytes(frame.loc[:, columns])


def calibrate_and_seal_h2_multiseed_traces(
    freeze_directory: str | Path,
    destination: str | Path,
    *,
    acoustic_root: str | Path = ACOUSTIC_ROOT,
) -> Path:
    """Fit inner-only calibration and seal one seed's gold-free outer traces."""

    freeze_root, freeze = _load_freeze(freeze_directory)
    seed = freeze["model_seed"]
    output = v1._reject_f_path(destination, "H2 multi-seed trace output")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite H2 multi-seed traces: {output}")
    local_acoustic = v1._reject_f_path(acoustic_root, "acoustic descendant root")
    files: dict[Path, bytes] = {}
    records: dict[str, object] = {}
    reactive_calibrations: dict[str, dict[str, object]] = {}
    reactive_traces: dict[str, dict[str, bytes]] = {}
    for budget_id in ANALYSIS_ORDER:
        budget = freeze["budgets"][budget_id]
        budget_records: dict[str, object] = {}
        reactive_calibrations[budget_id] = {}
        reactive_traces[budget_id] = {}
        for fold_id in OUTER_FOLD_IDS:
            fold = budget["folds"][fold_id]
            calibration, _ = _build_canonical_fold_calibration(
                budget,
                budget_id=budget_id,
                fold_id=fold_id,
                model_seed=seed,
                acoustic_root=local_acoustic,
            )
            reactive_calibrations[budget_id][fold_id] = _reactive_calibration_signature(
                calibration
            )
            calibration_rel = Path("calibration") / budget_id / f"{fold_id}.json"
            calibration_bytes = v1._pretty_json_bytes(calibration)
            files[calibration_rel] = calibration_bytes

            view_path = v1.verify_file_binding(
                fold["gold_free_outer_view"], role="gold-free outer view"
            )
            views = load_gold_free_probability_view(
                view_path, expected_model_seed=seed
            )
            if [view.session_id for view in views] != fold["outer_test_session_ids"]:
                _fail("gold-free outer sessions differ from held-out split")
            outer_acoustics = {
                view.session_id: v1.load_e_only_acoustic_descendant(
                    view.session_id,
                    assembled_source_artifact=budget["assembled_source_artifacts"][
                        view.session_id
                    ],
                    acoustic_root=local_acoustic,
                )
                for view in views
            }
            trace = _trace_frame(
                views,
                outer_acoustics,
                calibration,
                fold_id=fold_id,
                model_seed=seed,
            )
            reactive_traces[budget_id][fold_id] = _reactive_trace_signature(trace)
            trace_rel = Path("traces") / budget_id / f"{fold_id}.csv"
            trace_bytes = v1._csv_bytes(trace)
            files[trace_rel] = trace_bytes
            budget_records[fold_id] = {
                "calibration": v1._planned_binding(
                    output / calibration_rel, calibration_bytes
                ),
                "trace": {
                    **v1._planned_binding(output / trace_rel, trace_bytes),
                    "row_count": len(trace),
                    "ordered_columns": list(TRACE_COLUMNS),
                },
                "gold_free_outer_view": fold["gold_free_outer_view"],
                "outer_acoustic_sources": {
                    scene: artifact.bindings
                    for scene, artifact in outer_acoustics.items()
                },
            }
        records[budget_id] = budget_records

    for fold_id in OUTER_FOLD_IDS:
        if (
            reactive_calibrations["primary_5ep"][fold_id]
            != reactive_calibrations["sensitivity_15ep"][fold_id]
            or reactive_traces["primary_5ep"][fold_id]
            != reactive_traces["sensitivity_15ep"][fold_id]
        ):
            _fail("reactive calibration/trace differs across epoch budgets")
    seal = {
        "artifact_kind": "vap_h2_multiseed_trace_seal",
        "format_version": 2,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
        "model_seed": seed,
        "calibration_algorithm_seed": CALIBRATION_ALGORITHM_SEED,
        "freeze_manifest": v1._file_binding(freeze_root / FREEZE_FILENAME),
        "analysis_order": list(ANALYSIS_ORDER),
        "trace_generation_boundary": {
            "outer_gold_opened": False,
            "outer_fields_consumed": [
                "session_id",
                "group_id",
                "tick_25hz",
                "probabilities",
                "present",
            ],
            "floor_holder_consumed": False,
            "raw_audio_replayed": False,
            "F_drive_paths_dereferenced": False,
        },
        "policies": {
            "reactive": "causal_acoustic_speech_detector_plus_selected_silence_run",
            "predictive": "DemonstrationPolicy_P1_dual_threshold",
            "min_consecutive_frames": v1.MIN_CONSECUTIVE_FRAMES,
            "refractory_frames": v1.REFRACTORY_FRAMES,
        },
        "reactive_exact_across_epoch_budgets": True,
        "budgets": records,
        "implementation": _implementation_bindings(),
    }
    files[Path(TRACE_SEAL_FILENAME)] = v1._pretty_json_bytes(seal)
    published = _publish_with_owned_rollback(
        output,
        files,
        lambda root: _audit_trace_publication(
            freeze_root,
            root,
            acoustic_root=local_acoustic,
        ),
    )
    return published / TRACE_SEAL_FILENAME


def _verify_trace_binding(binding: object, *, role: str) -> Path:
    if not isinstance(binding, dict) or set(binding) != {
        "path",
        "sha256",
        "size_bytes",
        "row_count",
        "ordered_columns",
    }:
        _fail(f"{role} trace binding fields are not exact")
    return v1.verify_file_binding(
        {key: binding[key] for key in ("path", "sha256", "size_bytes")},
        role=role,
    )


def _load_trace_seal(
    path_value: str | Path,
    *,
    freeze_root: Path,
    freeze: dict[str, object],
    acoustic_root: Path,
) -> tuple[Path, dict[str, object]]:
    root = v1._reject_f_path(path_value, "H2 multi-seed trace directory")
    if root.is_symlink() or not root.is_dir():
        _fail("H2 multi-seed trace directory must be a non-symlink directory")
    seal = v1._read_json(root / TRACE_SEAL_FILENAME)
    required = {
        "artifact_kind",
        "format_version",
        "result_role",
        "production_eligible",
        "counted_eligible",
        "h2_established",
        "model_seed",
        "calibration_algorithm_seed",
        "freeze_manifest",
        "analysis_order",
        "trace_generation_boundary",
        "policies",
        "reactive_exact_across_epoch_budgets",
        "budgets",
        "implementation",
    }
    seed = freeze["model_seed"]
    expected_boundary = {
        "outer_gold_opened": False,
        "outer_fields_consumed": [
            "session_id",
            "group_id",
            "tick_25hz",
            "probabilities",
            "present",
        ],
        "floor_holder_consumed": False,
        "raw_audio_replayed": False,
        "F_drive_paths_dereferenced": False,
    }
    if (
        set(seal) != required
        or seal["artifact_kind"] != "vap_h2_multiseed_trace_seal"
        or seal["format_version"] != 2
        or seal["result_role"] != RESULT_ROLE
        or seal["production_eligible"] is not False
        or seal["counted_eligible"] is not False
        or seal["h2_established"] is not False
        or seal["model_seed"] != seed
        or seal["calibration_algorithm_seed"] != CALIBRATION_ALGORITHM_SEED
        or seal["analysis_order"] != list(ANALYSIS_ORDER)
        or seal["trace_generation_boundary"] != expected_boundary
        or seal["reactive_exact_across_epoch_budgets"] is not True
    ):
        _fail("H2 multi-seed trace identity/boundary differs")
    frozen = v1.verify_file_binding(seal["freeze_manifest"], role="trace-bound freeze")
    if frozen != (freeze_root / FREEZE_FILENAME).resolve():
        _fail("trace seal binds a different freeze")
    _verify_implementation_bindings(seal["implementation"])
    budgets = seal["budgets"]
    if not isinstance(budgets, dict) or tuple(budgets) != ANALYSIS_ORDER:
        _fail("trace seal budgets are not exact")
    reactive_calibrations: dict[str, dict[str, object]] = {}
    reactive_traces: dict[str, dict[str, bytes]] = {}
    for budget_id in ANALYSIS_ORDER:
        folds = budgets[budget_id]
        if not isinstance(folds, dict) or tuple(folds) != OUTER_FOLD_IDS:
            _fail(f"{budget_id} trace folds are not exact")
        reactive_calibrations[budget_id] = {}
        reactive_traces[budget_id] = {}
        for fold_id, record in folds.items():
            if not isinstance(record, dict) or set(record) != {
                "calibration",
                "trace",
                "gold_free_outer_view",
                "outer_acoustic_sources",
            }:
                _fail(f"{budget_id}/{fold_id} trace record fields are not exact")
            calibration_path = v1.verify_file_binding(
                record["calibration"], role="inner-only calibration"
            )
            trace_path = _verify_trace_binding(record["trace"], role="sealed trace")
            if root not in calibration_path.parents or root not in trace_path.parents:
                _fail("trace/calibration is not owned by the trace directory")
            calibration = v1._read_json(calibration_path)
            frozen_budget = freeze["budgets"][budget_id]
            frozen_fold = frozen_budget["folds"][fold_id]
            expected_calibration, _ = _build_canonical_fold_calibration(
                frozen_budget,
                budget_id=budget_id,
                fold_id=fold_id,
                model_seed=seed,
                acoustic_root=acoustic_root,
            )
            try:
                observed_calibration_bytes = calibration_path.read_bytes()
            except OSError as exc:
                raise H2MultiSeedReplayError(
                    f"cannot read sealed calibration bytes: {exc}"
                ) from exc
            if set(calibration) != set(expected_calibration):
                _fail("sealed calibration schema differs from canonical inner refit")
            if observed_calibration_bytes != v1._pretty_json_bytes(expected_calibration):
                _fail("sealed calibration bytes differ from canonical inner OOF/acoustic refit")
            estimator = (
                calibration.get("speech_detector", {})
                .get("final_model", {})
                .get("estimator", {})
            )
            if (
                calibration.get("calibration_scope")
                != "three_inner_group_OOF_only_no_outer_test"
                or calibration.get("outer_group_used_for_fit") is not False
                or calibration.get("model_seed") != seed
                or calibration.get("calibration_algorithm_seed")
                != CALIBRATION_ALGORITHM_SEED
                or estimator.get("random_state") != CALIBRATION_ALGORITHM_SEED
                or calibration.get("budget_id") != budget_id
                or calibration.get("max_epochs") != frozen_budget["max_epochs"]
                or calibration.get("outer_fold_id") != fold_id
                or calibration.get("outer_train_group_ids")
                != frozen_fold["outer_train_group_ids"]
                or calibration.get("outer_test_group_ids")
                != frozen_fold["outer_test_group_ids"]
                or calibration.get("outer_train_session_ids")
                != frozen_fold["outer_train_session_ids"]
                or calibration.get("outer_test_session_ids")
                != frozen_fold["outer_test_session_ids"]
                or calibration.get("inner_oof_sources")
                != [item["artifact"] for item in frozen_fold["inner_oof"]]
                or set(calibration.get("outer_train_group_ids", []))
                & set(calibration.get("outer_test_group_ids", []))
            ):
                _fail("calibration is not exact inner-only seed-bound calibration")
            reactive_calibrations[budget_id][fold_id] = _reactive_calibration_signature(
                calibration
            )
            if record["gold_free_outer_view"] != frozen_fold["gold_free_outer_view"]:
                _fail("trace does not bind its frozen gold-free view")
            view_path = v1.verify_file_binding(
                frozen_fold["gold_free_outer_view"], role="trace replay view"
            )
            if freeze_root not in view_path.parents:
                _fail("trace replay view is not freeze-owned")
            views = load_gold_free_probability_view(
                view_path, expected_model_seed=seed
            )
            acoustics = {
                view.session_id: v1.load_e_only_acoustic_descendant(
                    view.session_id,
                    assembled_source_artifact=frozen_budget[
                        "assembled_source_artifacts"
                    ][view.session_id],
                    acoustic_root=acoustic_root,
                )
                for view in views
            }
            replayed_sources = {
                scene: acoustic.bindings for scene, acoustic in acoustics.items()
            }
            if replayed_sources != record["outer_acoustic_sources"]:
                _fail("trace acoustic descendants differ from sealed bindings")
            replayed = _trace_frame(
                views,
                acoustics,
                calibration,
                fold_id=fold_id,
                model_seed=seed,
            )
            replayed_bytes = v1._csv_bytes(replayed)
            try:
                observed_bytes = trace_path.read_bytes()
                trace = pd.read_csv(trace_path)
            except (OSError, pd.errors.ParserError) as exc:
                raise H2MultiSeedReplayError(f"cannot read sealed trace: {exc}") from exc
            if observed_bytes != replayed_bytes:
                _fail("sealed trace bytes differ from calibration/view/acoustic replay")
            if (
                tuple(trace.columns) != TRACE_COLUMNS
                or len(trace) != record["trace"]["row_count"]
                or record["trace"]["ordered_columns"] != list(TRACE_COLUMNS)
                or set(trace["model_seed"].astype(int)) != {seed}
                or "floor_holder" in trace.columns
            ):
                _fail("sealed trace schema, seed, or row count differs")
            reactive_traces[budget_id][fold_id] = _reactive_trace_signature(trace)
    for fold_id in OUTER_FOLD_IDS:
        if (
            reactive_calibrations["primary_5ep"][fold_id]
            != reactive_calibrations["sensitivity_15ep"][fold_id]
            or reactive_traces["primary_5ep"][fold_id]
            != reactive_traces["sensitivity_15ep"][fold_id]
        ):
            _fail("audited reactive calibration/trace differs across budgets")
    return root, seal


def _audit_trace_publication(
    freeze_directory: str | Path,
    trace_directory: str | Path,
    *,
    acoustic_root: Path,
) -> None:
    """Audit a just-created trace without reopening legacy outer predictions."""

    freeze_root, freeze = _load_freeze(freeze_directory)
    _load_trace_seal(
        trace_directory,
        freeze_root=freeze_root,
        freeze=freeze,
        acoustic_root=acoustic_root,
    )


def _gold_session(record: dict[str, object]) -> v1.CalibrationSession:
    view = v1._validate_probability_session(record)
    vad = np.asarray(record["vad"], dtype=np.float64)
    valid = np.asarray(record["valid"], dtype=bool)
    if vad.shape != (view.ticks.size, 4) or valid.shape != (view.ticks.size,):
        _fail("outer evaluation gold shape differs from probabilities")
    return v1.CalibrationSession(
        session_id=view.session_id,
        group_id=view.group_id,
        ticks=view.ticks,
        probabilities=view.probabilities,
        vad=vad,
        valid=valid,
        present=view.present,
        acoustic_features=np.zeros((view.ticks.size, 3), dtype=np.float64),
    )


def _trace_for_session(frame: pd.DataFrame, session: v1.CalibrationSession) -> pd.DataFrame:
    rows = frame.loc[frame["session_id"].astype(str) == session.session_id]
    if rows.empty:
        _fail(f"sealed trace lacks outer session {session.session_id}")
    if not np.array_equal(
        rows["tick_25hz"].to_numpy(dtype=np.int64), session.ticks
    ):
        _fail("sealed trace keys differ from evaluation-only gold")
    return rows


def _opportunity_signature(records: list[dict[str, object]]) -> tuple[tuple[object, ...], ...]:
    fields = (
        "event_id",
        "session_id",
        "group_id",
        "event_index",
        "gap_start_tick",
        "gap_end_tick_exclusive",
        "previous_slot",
        "next_slot",
        "robot_entry_label",
    )
    keyed: dict[str, tuple[object, ...]] = {}
    for record in records:
        value = tuple(record[field] for field in fields)
        event_id = str(record["event_id"])
        if event_id in keyed and keyed[event_id] != value:
            _fail("one event ID has inconsistent opportunity metadata")
        keyed[event_id] = value
    return tuple(keyed[key] for key in sorted(keyed))


def _rebuild_evaluation_payloads(
    freeze: dict[str, object],
    seal: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], set[str]]:
    seed = freeze["model_seed"]
    all_records: list[dict[str, object]] = []
    pooled: dict[str, object] = {}
    reference_ids: set[str] | None = None
    reference_opportunities: tuple[tuple[object, ...], ...] | None = None
    for budget_id in ANALYSIS_ORDER:
        budget = freeze["budgets"][budget_id]
        budget_metrics: dict[str, object] = {}
        fold_inputs: dict[str, tuple[dict[str, object], pd.DataFrame]] = {}
        for fold_id in OUTER_FOLD_IDS:
            fold = budget["folds"][fold_id]
            prediction_path = v1.verify_file_binding(
                fold["outer_prediction_provenance"],
                role="evaluation-only outer prediction",
            )
            prediction = thesis_backend.load_prediction_artifact(prediction_path)
            if (
                prediction.get("condition_id") != "A+H"
                or prediction.get("outer_fold_id") != fold_id
                or prediction.get("seed") != seed
                or prediction.get("run_manifest_sha256")
                != budget["run_manifest_payload_sha256"]
                or prediction.get("checkpoint_sha256")
                != fold["final_checkpoint_provenance"]["sha256"]
                or [record["session_id"] for record in prediction["sessions"]]
                != fold["outer_test_session_ids"]
            ):
                _fail("evaluation-only outer prediction differs from freeze")
            trace_path = _verify_trace_binding(
                seal["budgets"][budget_id][fold_id]["trace"],
                role="sealed trace",
            )
            try:
                trace = pd.read_csv(trace_path)
            except (OSError, pd.errors.ParserError) as exc:
                raise H2MultiSeedReplayError(f"cannot parse sealed trace: {exc}") from exc
            fold_inputs[fold_id] = (prediction, trace)
        for policy_id, trigger_column in (
            ("reactive", "reactive_trigger"),
            ("predictive_demonstration_P1", "predictive_trigger"),
        ):
            policy_metrics: dict[str, object] = {}
            for window_id, pre_gap_frames in (
                ("primary_pre_gap_0.4s_to_gap_end", v1.PRIMARY_PRE_GAP_FRAMES),
                ("sensitivity_gap_only", 0),
            ):
                records: list[dict[str, object]] = []
                fold_metrics: dict[str, object] = {}
                for fold_id in OUTER_FOLD_IDS:
                    prediction, trace = fold_inputs[fold_id]
                    fold_records: list[dict[str, object]] = []
                    for source_record in prediction["sessions"]:
                        session = _gold_session(source_record)
                        rows = _trace_for_session(trace, session)
                        event_rows = v1.score_trace_events(
                            session,
                            rows[trigger_column].to_numpy(dtype=np.int8),
                            pre_gap_frames=pre_gap_frames,
                            policy_id=policy_id,
                        )
                        for event in event_rows:
                            event.update(
                                {
                                    "model_seed": seed,
                                    "budget_id": budget_id,
                                    "max_epochs": budget["max_epochs"],
                                    "outer_fold_id": fold_id,
                                    "window_id": window_id,
                                }
                            )
                        fold_records.extend(event_rows)
                    fold_metrics[fold_id] = v1._binary_metrics(fold_records)
                    records.extend(fold_records)
                v1.validate_exact_event_support(records)
                ids = {str(record["event_id"]) for record in records}
                opportunities = _opportunity_signature(records)
                if reference_ids is None:
                    reference_ids = ids
                    reference_opportunities = opportunities
                elif ids != reference_ids or opportunities != reference_opportunities:
                    _fail("budgets/policies/windows do not score identical opportunities")
                policy_metrics[window_id] = {
                    "pooled": v1._binary_metrics(records),
                    "outer_folds": fold_metrics,
                }
                all_records.extend(records)
            budget_metrics[policy_id] = policy_metrics
        pooled[budget_id] = budget_metrics
    if reference_ids is None:
        _fail("H2 multi-seed evaluation produced no opportunities")
    event_frame = pd.DataFrame(all_records, columns=EVENT_COLUMNS)
    metrics = {
        "artifact_kind": "vap_h2_multiseed_seed_metrics",
        "format_version": 2,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
        "model_seed": seed,
        "calibration_algorithm_seed": CALIBRATION_ALGORITHM_SEED,
        "primary_budget": "primary_5ep",
        "sensitivity_budget": "sensitivity_15ep",
        "primary_window": "primary_pre_gap_0.4s_to_gap_end",
        "sensitivity_window": "sensitivity_gap_only",
        "event_support_per_budget_policy_window": {
            "total": v1.EXPECTED_N_EVENTS,
            "positive": v1.EXPECTED_N_POSITIVE,
            "negative": v1.EXPECTED_N_NEGATIVE,
        },
        "same_full_opportunity_metadata_across_all_comparisons": True,
        "results": pooled,
        "claim_boundary": (
            "posthoc matched offline replay; no robot action feedback, TTS startup, "
            "far-field robot-ear audio, or live deployment"
        ),
    }
    return event_frame, metrics, reference_ids


def evaluate_h2_multiseed_seed(
    freeze_directory: str | Path,
    trace_directory: str | Path,
    destination: str | Path,
    *,
    acoustic_root: str | Path = ACOUSTIC_ROOT,
) -> Path:
    """Open outer gold only after one seed's trace seal and score matched events."""

    freeze_root, freeze = _load_freeze(freeze_directory)
    local_acoustic = v1._reject_f_path(acoustic_root, "acoustic descendant root")
    trace_root, seal = _load_trace_seal(
        trace_directory,
        freeze_root=freeze_root,
        freeze=freeze,
        acoustic_root=local_acoustic,
    )
    output = v1._reject_f_path(destination, "H2 multi-seed evaluation output")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite H2 multi-seed evaluation: {output}")
    events, metrics, event_ids = _rebuild_evaluation_payloads(freeze, seal)
    event_bytes = v1._csv_bytes(events)
    metrics_bytes = v1._pretty_json_bytes(metrics)
    evaluation = {
        "artifact_kind": "vap_h2_multiseed_seed_evaluation",
        "format_version": 2,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
        "model_seed": freeze["model_seed"],
        "calibration_algorithm_seed": CALIBRATION_ALGORITHM_SEED,
        "freeze_manifest": v1._file_binding(freeze_root / FREEZE_FILENAME),
        "trace_seal_manifest": v1._file_binding(trace_root / TRACE_SEAL_FILENAME),
        "outer_gold_opened_only_after_trace_seal": True,
        "event_csv": {
            **v1._planned_binding(output / EVENT_FILENAME, event_bytes),
            "row_count": len(events),
            "ordered_columns": list(EVENT_COLUMNS),
        },
        "metrics": v1._planned_binding(output / METRICS_FILENAME, metrics_bytes),
        "legacy_voiced_diagnostic": v1._legacy_crosscheck(event_ids),
        "scientific_boundary": {
            "same_303_event_ids_and_metadata_across_all_comparisons": True,
            "main_window": "pre_gap_0.4s_to_gap_end",
            "gap_only_is_sensitivity": True,
            "outer_threshold_tuning": False,
            "live_robot_or_TTS_latency_simulated": False,
            "production_eligible": False,
            "counted_eligible": False,
        },
        "implementation": _implementation_bindings(),
    }
    files = {
        Path(EVENT_FILENAME): event_bytes,
        Path(METRICS_FILENAME): metrics_bytes,
        Path(EVALUATION_FILENAME): v1._pretty_json_bytes(evaluation),
    }
    published = _publish_with_owned_rollback(
        output,
        files,
        lambda root: audit_h2_multiseed_seed(
            freeze_root,
            trace_directory=trace_root,
            evaluation_directory=root,
            acoustic_root=local_acoustic,
        ),
    )
    return published / EVALUATION_FILENAME


def audit_h2_multiseed_seed(
    freeze_directory: str | Path,
    *,
    trace_directory: str | Path | None = None,
    evaluation_directory: str | Path | None = None,
    acoustic_root: str | Path = ACOUSTIC_ROOT,
) -> dict[str, object]:
    """Cold replay-audit any completed prefix for one extension seed."""

    freeze_root, freeze = _load_freeze(freeze_directory)
    _cold_audit_freeze_against_canonical_sources(freeze_root, freeze)
    local_acoustic = v1._reject_f_path(acoustic_root, "acoustic descendant root")
    result: dict[str, object] = {
        "model_seed": freeze["model_seed"],
        "calibration_algorithm_seed": CALIBRATION_ALGORITHM_SEED,
        "freeze": "passed",
        "trace_seal": "not_requested",
        "evaluation": "not_requested",
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
    }
    trace_root: Path | None = None
    trace_seal: dict[str, object] | None = None
    if trace_directory is not None:
        trace_root, trace_seal = _load_trace_seal(
            trace_directory,
            freeze_root=freeze_root,
            freeze=freeze,
            acoustic_root=local_acoustic,
        )
        result["trace_seal"] = "passed"
    if evaluation_directory is not None:
        if trace_root is None or trace_seal is None:
            _fail("evaluation audit requires the validated trace directory")
        evaluation_root = v1._reject_f_path(
            evaluation_directory, "H2 multi-seed evaluation directory"
        )
        if evaluation_root.is_symlink() or not evaluation_root.is_dir():
            _fail("evaluation directory must be an existing non-symlink directory")
        manifest = v1._read_json(evaluation_root / EVALUATION_FILENAME)
        required = {
            "artifact_kind",
            "format_version",
            "result_role",
            "production_eligible",
            "counted_eligible",
            "h2_established",
            "model_seed",
            "calibration_algorithm_seed",
            "freeze_manifest",
            "trace_seal_manifest",
            "outer_gold_opened_only_after_trace_seal",
            "event_csv",
            "metrics",
            "legacy_voiced_diagnostic",
            "scientific_boundary",
            "implementation",
        }
        if (
            set(manifest) != required
            or manifest["artifact_kind"] != "vap_h2_multiseed_seed_evaluation"
            or manifest["format_version"] != 2
            or manifest["result_role"] != RESULT_ROLE
            or manifest["production_eligible"] is not False
            or manifest["counted_eligible"] is not False
            or manifest["h2_established"] is not False
            or manifest["model_seed"] != freeze["model_seed"]
            or manifest["calibration_algorithm_seed"] != CALIBRATION_ALGORITHM_SEED
            or manifest["outer_gold_opened_only_after_trace_seal"] is not True
        ):
            _fail("evaluation identity differs or attempted promotion")
        frozen = v1.verify_file_binding(
            manifest["freeze_manifest"], role="evaluation freeze"
        )
        sealed = v1.verify_file_binding(
            manifest["trace_seal_manifest"], role="evaluation trace seal"
        )
        if (
            frozen != (freeze_root / FREEZE_FILENAME).resolve()
            or sealed != (trace_root / TRACE_SEAL_FILENAME).resolve()
        ):
            _fail("evaluation binds different freeze/trace artifacts")
        _verify_implementation_bindings(manifest["implementation"])
        expected_boundary = {
            "same_303_event_ids_and_metadata_across_all_comparisons": True,
            "main_window": "pre_gap_0.4s_to_gap_end",
            "gap_only_is_sensitivity": True,
            "outer_threshold_tuning": False,
            "live_robot_or_TTS_latency_simulated": False,
            "production_eligible": False,
            "counted_eligible": False,
        }
        if manifest["scientific_boundary"] != expected_boundary:
            _fail("evaluation scientific boundary differs")
        event_binding = manifest["event_csv"]
        if not isinstance(event_binding, dict) or set(event_binding) != {
            "path",
            "sha256",
            "size_bytes",
            "row_count",
            "ordered_columns",
        }:
            _fail("evaluation event binding fields are not exact")
        event_path = v1.verify_file_binding(
            {key: event_binding[key] for key in ("path", "sha256", "size_bytes")},
            role="H2 multi-seed event CSV",
        )
        metrics_path = v1.verify_file_binding(manifest["metrics"], role="H2 metrics")
        if (
            evaluation_root not in event_path.parents
            or evaluation_root not in metrics_path.parents
        ):
            _fail("evaluation outputs are not directory-owned")
        rebuilt_events, rebuilt_metrics, event_ids = _rebuild_evaluation_payloads(
            freeze, trace_seal
        )
        try:
            observed_events = event_path.read_bytes()
            observed_metrics = metrics_path.read_bytes()
        except OSError as exc:
            raise H2MultiSeedReplayError(f"cannot read evaluation outputs: {exc}") from exc
        if observed_events != v1._csv_bytes(rebuilt_events):
            _fail("event CSV differs from sealed-trace/outer-gold replay")
        if observed_metrics != v1._pretty_json_bytes(rebuilt_metrics):
            _fail("metric bytes differ from sealed-trace/outer-gold replay")
        if (
            event_binding["row_count"] != len(rebuilt_events)
            or event_binding["ordered_columns"] != list(EVENT_COLUMNS)
            or manifest["legacy_voiced_diagnostic"] != v1._legacy_crosscheck(event_ids)
        ):
            _fail("evaluation manifest differs from canonical replay")
        result["evaluation"] = "passed"
    result["analysis_order"] = list(ANALYSIS_ORDER)
    return result


@dataclass(frozen=True)
class _EvaluationSource:
    model_seed: int
    events: pd.DataFrame
    metrics: dict[str, object]
    reactive_calibrations: dict[str, dict[str, object]]
    reactive_traces: dict[str, dict[str, str]]
    bindings: dict[str, object]


def _trace_invariants(
    trace_manifest_path: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, str]]]:
    seal = v1._read_json(trace_manifest_path)
    budgets = seal.get("budgets")
    if not isinstance(budgets, dict) or tuple(budgets) != ANALYSIS_ORDER:
        _fail("aggregate source trace budgets differ")
    calibrations: dict[str, dict[str, object]] = {}
    traces: dict[str, dict[str, str]] = {}
    for budget_id in ANALYSIS_ORDER:
        folds = budgets[budget_id]
        if not isinstance(folds, dict) or tuple(folds) != OUTER_FOLD_IDS:
            _fail("aggregate source trace folds differ")
        calibrations[budget_id] = {}
        traces[budget_id] = {}
        for fold_id, record in folds.items():
            calibration_path = v1.verify_file_binding(
                record["calibration"], role="aggregate reactive calibration"
            )
            calibration = v1._read_json(calibration_path)
            calibrations[budget_id][fold_id] = _reactive_calibration_signature(
                calibration
            )
            trace_path = v1._verify_trace_binding(
                record["trace"], role="aggregate reactive trace"
            )
            try:
                trace = pd.read_csv(trace_path)
            except (OSError, pd.errors.ParserError) as exc:
                raise H2MultiSeedReplayError(
                    f"cannot read aggregate reactive trace: {exc}"
                ) from exc
            traces[budget_id][fold_id] = hashlib.sha256(
                _reactive_trace_signature(trace)
            ).hexdigest()
    return calibrations, traces


def _load_seed13_source(
    evaluation_root: Path,
    *,
    acoustic_root: Path,
) -> _EvaluationSource:
    expected_root = SEED13_EVALUATION_ROOT.resolve()
    if evaluation_root.resolve() != expected_root:
        _fail("seed13 aggregate source is not the immutable completed evaluation")
    evaluation_path = expected_root / v1.EVALUATION_FILENAME
    event_path = expected_root / v1.EVENT_FILENAME
    metrics_path = expected_root / v1.METRICS_FILENAME
    if (
        _sha256(evaluation_path) != SEED13_EVALUATION_SHA256
        or _sha256(event_path) != SEED13_EVENT_SHA256
        or _sha256(metrics_path) != SEED13_METRICS_SHA256
    ):
        _fail("immutable seed13 evaluation bytes differ")
    v1.audit_h2_matched_replay(
        SEED13_ROOT / "freeze",
        trace_directory=SEED13_ROOT / "traces",
        evaluation_directory=expected_root,
        acoustic_root=acoustic_root,
    )
    try:
        events = pd.read_csv(event_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise H2MultiSeedReplayError(f"cannot parse seed13 event CSV: {exc}") from exc
    if tuple(events.columns) != v1.EVENT_COLUMNS:
        _fail("seed13 event schema differs")
    events.insert(0, "model_seed", 13)
    metrics = v1._read_json(metrics_path)
    if (
        metrics.get("result_role") != v1.RESULT_ROLE
        or metrics.get("h2_established") is not False
        or metrics.get("event_support_per_budget_policy_window")
        != {
            "total": v1.EXPECTED_N_EVENTS,
            "positive": v1.EXPECTED_N_POSITIVE,
            "negative": v1.EXPECTED_N_NEGATIVE,
        }
    ):
        _fail("seed13 metrics identity/support differs")
    trace_manifest = SEED13_ROOT / "traces" / v1.TRACE_SEAL_FILENAME
    calibrations, traces = _trace_invariants(trace_manifest)
    return _EvaluationSource(
        model_seed=13,
        events=events,
        metrics=metrics,
        reactive_calibrations=calibrations,
        reactive_traces=traces,
        bindings={
            "evaluation_manifest": v1._file_binding(evaluation_path),
            "event_csv": v1._file_binding(event_path),
            "metrics": v1._file_binding(metrics_path),
            "freeze_manifest": v1._file_binding(
                SEED13_ROOT / "freeze" / v1.FREEZE_FILENAME
            ),
            "trace_seal_manifest": v1._file_binding(trace_manifest),
            "implementation": v1._file_binding(Path(v1.__file__)),
        },
    )


def _load_extension_source(
    evaluation_root: Path,
    *,
    expected_seed: int,
    acoustic_root: Path,
) -> _EvaluationSource:
    manifest_path = evaluation_root / EVALUATION_FILENAME
    manifest = v1._read_json(manifest_path)
    if manifest.get("model_seed") != expected_seed:
        _fail("extension evaluation model_seed differs")
    freeze_path = v1.verify_file_binding(
        manifest.get("freeze_manifest"), role="aggregate extension freeze"
    )
    trace_path = v1.verify_file_binding(
        manifest.get("trace_seal_manifest"), role="aggregate extension trace seal"
    )
    audit_h2_multiseed_seed(
        freeze_path.parent,
        trace_directory=trace_path.parent,
        evaluation_directory=evaluation_root,
        acoustic_root=acoustic_root,
    )
    event_binding = manifest["event_csv"]
    event_path = v1.verify_file_binding(
        {key: event_binding[key] for key in ("path", "sha256", "size_bytes")},
        role="aggregate extension event CSV",
    )
    metrics_path = v1.verify_file_binding(
        manifest["metrics"], role="aggregate extension metrics"
    )
    try:
        events = pd.read_csv(event_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise H2MultiSeedReplayError(
            f"cannot parse extension event CSV: {exc}"
        ) from exc
    if (
        tuple(events.columns) != EVENT_COLUMNS
        or set(events["model_seed"].astype(int)) != {expected_seed}
    ):
        _fail("extension event schema/model_seed differs")
    metrics = v1._read_json(metrics_path)
    if (
        metrics.get("artifact_kind") != "vap_h2_multiseed_seed_metrics"
        or metrics.get("model_seed") != expected_seed
        or metrics.get("h2_established") is not False
    ):
        _fail("extension metric identity differs")
    calibrations, traces = _trace_invariants(trace_path)
    return _EvaluationSource(
        model_seed=expected_seed,
        events=events,
        metrics=metrics,
        reactive_calibrations=calibrations,
        reactive_traces=traces,
        bindings={
            "evaluation_manifest": v1._file_binding(manifest_path),
            "event_csv": v1._file_binding(event_path),
            "metrics": v1._file_binding(metrics_path),
            "freeze_manifest": v1._file_binding(freeze_path),
            "trace_seal_manifest": v1._file_binding(trace_path),
            "implementation": _implementation_bindings(),
        },
    )


def _load_evaluation_sources(
    seed42_evaluation: str | Path,
    seed73_evaluation: str | Path,
    *,
    seed13_evaluation: str | Path = SEED13_EVALUATION_ROOT,
    acoustic_root: str | Path = ACOUSTIC_ROOT,
) -> dict[int, _EvaluationSource]:
    local_acoustic = v1._reject_f_path(acoustic_root, "acoustic descendant root")
    sources = {
        13: _load_seed13_source(
            v1._reject_f_path(seed13_evaluation, "seed13 evaluation"),
            acoustic_root=local_acoustic,
        ),
        42: _load_extension_source(
            v1._reject_f_path(seed42_evaluation, "seed42 evaluation"),
            expected_seed=42,
            acoustic_root=local_acoustic,
        ),
        73: _load_extension_source(
            v1._reject_f_path(seed73_evaluation, "seed73 evaluation"),
            expected_seed=73,
            acoustic_root=local_acoustic,
        ),
    }
    return sources


def _finite_summary(values: list[object]) -> dict[str, object]:
    if len(values) != 3:
        _fail("three-seed summary requires exactly three values")
    if any(
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        return {"mean": None, "sample_sd": None, "na_reason": "non_finite_or_missing"}
    numeric = [float(value) for value in values]
    return {
        "mean": float(mean(numeric)),
        "sample_sd": float(stdev(numeric)),
        "na_reason": None,
    }


def _paired_both_hit_lead(
    events: pd.DataFrame,
    *,
    budget_id: str,
    window_id: str,
) -> dict[str, object]:
    scope = events.loc[
        (events["budget_id"] == budget_id) & (events["window_id"] == window_id)
    ]
    reactive = scope.loc[
        scope["policy_id"] == "reactive",
        ["event_id", "robot_entry_label", "triggered", "latency_s"],
    ].rename(
        columns={
            "triggered": "reactive_triggered",
            "latency_s": "reactive_latency_s",
        }
    )
    predictive = scope.loc[
        scope["policy_id"] == "predictive_demonstration_P1",
        ["event_id", "robot_entry_label", "triggered", "latency_s"],
    ].rename(
        columns={
            "robot_entry_label": "predictive_label",
            "triggered": "predictive_triggered",
            "latency_s": "predictive_latency_s",
        }
    )
    paired = reactive.merge(predictive, on="event_id", how="inner", validate="one_to_one")
    if len(paired) != v1.EXPECTED_N_EVENTS or not (
        paired["robot_entry_label"] == paired["predictive_label"]
    ).all():
        _fail("paired lead derivation does not contain the same 303 labels")
    both = paired.loc[
        (paired["robot_entry_label"] == 1)
        & (paired["reactive_triggered"] == 1)
        & (paired["predictive_triggered"] == 1)
    ].copy()
    if both.empty:
        return {
            "n_both_hit_positive": 0,
            "predictive_lead_mean_s": None,
            "predictive_lead_median_s": None,
        }
    lead = (
        both["reactive_latency_s"].astype(float)
        - both["predictive_latency_s"].astype(float)
    )
    return {
        "n_both_hit_positive": int(len(both)),
        "predictive_lead_mean_s": float(lead.mean()),
        "predictive_lead_median_s": float(lead.median()),
    }


def _build_multiseed_metrics(
    sources: dict[int, _EvaluationSource],
) -> dict[str, object]:
    if tuple(sources) != ALL_REPLAY_MODEL_SEEDS:
        _fail("multi-seed aggregate sources must be ordered 13/42/73")
    reference_opportunities: tuple[tuple[object, ...], ...] | None = None
    reference_reactive_events: bytes | None = None
    reference_calibration: dict[str, dict[str, object]] | None = None
    reference_traces: dict[str, dict[str, str]] | None = None
    for seed in ALL_REPLAY_MODEL_SEEDS:
        source = sources[seed]
        records = source.events.to_dict(orient="records")
        opportunities = _opportunity_signature(records)
        if len(opportunities) != v1.EXPECTED_N_EVENTS:
            _fail("one seed does not expose exactly 303 unique opportunities")
        reactive = source.events.loc[
            source.events["policy_id"] == "reactive", list(v1.EVENT_COLUMNS)
        ]
        reactive_bytes = v1._csv_bytes(reactive)
        if reference_opportunities is None:
            reference_opportunities = opportunities
            reference_reactive_events = reactive_bytes
            reference_calibration = source.reactive_calibrations
            reference_traces = source.reactive_traces
        elif (
            opportunities != reference_opportunities
            or reactive_bytes != reference_reactive_events
            or source.reactive_calibrations != reference_calibration
            or source.reactive_traces != reference_traces
        ):
            _fail("opportunities or reactive replay are not exact across model seeds")

    results: dict[str, object] = {}
    metric_fields = (
        "positive_recall",
        "negative_false_entry_rate",
        "balanced_accuracy",
        "positive_hit_latency_mean_s",
    )
    for budget_id in ANALYSIS_ORDER:
        budget_result: dict[str, object] = {}
        for window_id in (
            "primary_pre_gap_0.4s_to_gap_end",
            "sensitivity_gap_only",
        ):
            per_seed: dict[str, object] = {}
            leads: dict[str, object] = {}
            deltas: dict[str, object] = {}
            for seed in ALL_REPLAY_MODEL_SEEDS:
                metrics = sources[seed].metrics["results"][budget_id]
                reactive = metrics["reactive"][window_id]["pooled"]
                predictive = metrics["predictive_demonstration_P1"][window_id]["pooled"]
                per_seed[str(seed)] = {
                    "reactive": reactive,
                    "predictive_demonstration_P1": predictive,
                }
                deltas[str(seed)] = {
                    field: (
                        None
                        if reactive.get(field) is None or predictive.get(field) is None
                        else float(predictive[field]) - float(reactive[field])
                    )
                    for field in metric_fields
                }
                leads[str(seed)] = _paired_both_hit_lead(
                    sources[seed].events,
                    budget_id=budget_id,
                    window_id=window_id,
                )
            predictive_summary = {
                field: _finite_summary(
                    [
                        per_seed[str(seed)]["predictive_demonstration_P1"].get(field)
                        for seed in ALL_REPLAY_MODEL_SEEDS
                    ]
                )
                for field in metric_fields
            }
            delta_summary = {
                field: _finite_summary(
                    [deltas[str(seed)][field] for seed in ALL_REPLAY_MODEL_SEEDS]
                )
                for field in metric_fields
            }
            budget_result[window_id] = {
                "per_seed": per_seed,
                "reactive_exact_across_model_seeds": True,
                "reactive_reference": per_seed["13"]["reactive"],
                "predictive_mean_sample_sd": predictive_summary,
                "predictive_minus_reactive_mean_sample_sd": delta_summary,
                "predictive_balanced_accuracy_better_seed_count": sum(
                    deltas[str(seed)]["balanced_accuracy"] > 0
                    for seed in ALL_REPLAY_MODEL_SEEDS
                ),
                "paired_both_hit_positive_lead_descriptive": leads,
            }
        results[budget_id] = budget_result
    return {
        "artifact_kind": "vap_h2_matched_replay_three_seed_metrics",
        "format_version": 2,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
        "primary_result_remains_seed13_5epoch": True,
        "model_seeds": list(ALL_REPLAY_MODEL_SEEDS),
        "calibration_algorithm_seed": CALIBRATION_ALGORITHM_SEED,
        "event_support_per_seed_budget_policy_window": {
            "total": v1.EXPECTED_N_EVENTS,
            "positive": v1.EXPECTED_N_POSITIVE,
            "negative": v1.EXPECTED_N_NEGATIVE,
        },
        "same_full_opportunity_metadata_across_model_seeds": True,
        "reactive_calibration_trace_and_events_exact_across_model_seeds": True,
        "statistics": "arithmetic_mean_and_sample_standard_deviation_ddof_1",
        "results": results,
        "claim_boundary": (
            "post-outcome descriptive model-initialisation sensitivity from saved "
            "probabilities; no significance test, physical robot, TTS, action feedback, "
            "far-field robot-ear audio, or live latency"
        ),
    }


def aggregate_h2_multiseed(
    destination: str | Path,
    *,
    seed42_evaluation: str | Path,
    seed73_evaluation: str | Path,
    seed13_evaluation: str | Path = SEED13_EVALUATION_ROOT,
    acoustic_root: str | Path = ACOUSTIC_ROOT,
) -> Path:
    """Bind the immutable seed13 result and audited seed42/73 evaluations."""

    output = v1._reject_f_path(destination, "H2 three-seed aggregate output")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite H2 multi-seed aggregate: {output}")
    sources = _load_evaluation_sources(
        seed42_evaluation,
        seed73_evaluation,
        seed13_evaluation=seed13_evaluation,
        acoustic_root=acoustic_root,
    )
    metrics = _build_multiseed_metrics(sources)
    metrics_bytes = v1._pretty_json_bytes(metrics)
    manifest = {
        "artifact_kind": "vap_h2_matched_replay_three_seed_aggregate",
        "format_version": 2,
        "result_role": RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
        "primary_result_remains_seed13_5epoch": True,
        "model_seeds": list(ALL_REPLAY_MODEL_SEEDS),
        "calibration_algorithm_seed": CALIBRATION_ALGORITHM_SEED,
        "aggregation_source": "sealed_saved_probability_evaluations_no_new_inference",
        "sources": {str(seed): sources[seed].bindings for seed in ALL_REPLAY_MODEL_SEEDS},
        "metrics": v1._planned_binding(
            output / AGGREGATE_METRICS_FILENAME, metrics_bytes
        ),
        "invariants": {
            "same_303_opportunities_and_full_metadata_across_seeds": True,
            "reactive_calibration_trace_and_events_exact_across_seeds": True,
            "outer_threshold_tuning": False,
            "production_eligible": False,
            "counted_eligible": False,
        },
        "implementation": _implementation_bindings(),
    }
    files = {
        Path(AGGREGATE_METRICS_FILENAME): metrics_bytes,
        Path(AGGREGATE_FILENAME): v1._pretty_json_bytes(manifest),
    }
    published = _publish_with_owned_rollback(
        output,
        files,
        lambda root: audit_h2_multiseed_aggregate(
            root,
            seed42_evaluation=seed42_evaluation,
            seed73_evaluation=seed73_evaluation,
            seed13_evaluation=seed13_evaluation,
            acoustic_root=acoustic_root,
        ),
    )
    return published / AGGREGATE_FILENAME


def audit_h2_multiseed_aggregate(
    aggregate_directory: str | Path,
    *,
    seed42_evaluation: str | Path,
    seed73_evaluation: str | Path,
    seed13_evaluation: str | Path = SEED13_EVALUATION_ROOT,
    acoustic_root: str | Path = ACOUSTIC_ROOT,
) -> dict[str, object]:
    """Cold-rebuild the three-seed summary and require canonical bytes."""

    root = v1._reject_f_path(aggregate_directory, "H2 multi-seed aggregate directory")
    if root.is_symlink() or not root.is_dir():
        _fail("H2 multi-seed aggregate must be an existing non-symlink directory")
    manifest = v1._read_json(root / AGGREGATE_FILENAME)
    required = {
        "artifact_kind",
        "format_version",
        "result_role",
        "production_eligible",
        "counted_eligible",
        "h2_established",
        "primary_result_remains_seed13_5epoch",
        "model_seeds",
        "calibration_algorithm_seed",
        "aggregation_source",
        "sources",
        "metrics",
        "invariants",
        "implementation",
    }
    if (
        set(manifest) != required
        or manifest["artifact_kind"] != "vap_h2_matched_replay_three_seed_aggregate"
        or manifest["format_version"] != 2
        or manifest["result_role"] != RESULT_ROLE
        or manifest["production_eligible"] is not False
        or manifest["counted_eligible"] is not False
        or manifest["h2_established"] is not False
        or manifest["primary_result_remains_seed13_5epoch"] is not True
        or manifest["model_seeds"] != list(ALL_REPLAY_MODEL_SEEDS)
        or manifest["calibration_algorithm_seed"] != CALIBRATION_ALGORITHM_SEED
        or manifest["aggregation_source"]
        != "sealed_saved_probability_evaluations_no_new_inference"
        or manifest["invariants"]
        != {
            "same_303_opportunities_and_full_metadata_across_seeds": True,
            "reactive_calibration_trace_and_events_exact_across_seeds": True,
            "outer_threshold_tuning": False,
            "production_eligible": False,
            "counted_eligible": False,
        }
    ):
        _fail("H2 three-seed aggregate identity differs or attempted promotion")
    _verify_implementation_bindings(manifest["implementation"])
    sources = _load_evaluation_sources(
        seed42_evaluation,
        seed73_evaluation,
        seed13_evaluation=seed13_evaluation,
        acoustic_root=acoustic_root,
    )
    expected_bindings = {
        str(seed): sources[seed].bindings for seed in ALL_REPLAY_MODEL_SEEDS
    }
    if manifest["sources"] != expected_bindings:
        _fail("H2 aggregate source bindings differ")
    metrics_path = v1.verify_file_binding(
        manifest["metrics"], role="H2 multi-seed aggregate metrics"
    )
    if root not in metrics_path.parents:
        _fail("H2 aggregate metrics are not directory-owned")
    rebuilt = _build_multiseed_metrics(sources)
    try:
        observed = metrics_path.read_bytes()
    except OSError as exc:
        raise H2MultiSeedReplayError(f"cannot read aggregate metrics: {exc}") from exc
    if observed != v1._pretty_json_bytes(rebuilt):
        _fail("H2 three-seed metrics differ from cold source replay")
    return {
        "aggregate": "passed",
        "model_seeds": list(ALL_REPLAY_MODEL_SEEDS),
        "same_303_opportunities": True,
        "reactive_exact_across_model_seeds": True,
        "production_eligible": False,
        "counted_eligible": False,
        "h2_established": False,
    }


__all__ = [
    "ACOUSTIC_ROOT",
    "AGGREGATE_FILENAME",
    "AGGREGATE_METRICS_FILENAME",
    "CALIBRATION_ALGORITHM_SEED",
    "CANONICAL_SEED_SOURCES",
    "EVALUATION_FILENAME",
    "EVENT_FILENAME",
    "FREEZE_FILENAME",
    "H2MultiSeedReplayError",
    "METRICS_FILENAME",
    "SEED13_EVALUATION_ROOT",
    "SUPPORTED_MODEL_SEEDS",
    "TRACE_SEAL_FILENAME",
    "aggregate_h2_multiseed",
    "audit_h2_multiseed_aggregate",
    "audit_h2_multiseed_seed",
    "calibrate_and_seal_h2_multiseed_traces",
    "evaluate_h2_multiseed_seed",
    "freeze_h2_multiseed_seed",
    "load_gold_free_probability_view",
]
