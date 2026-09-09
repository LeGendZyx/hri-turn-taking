"""Frozen nested-LOGO runner for the one-tick previous-VAD oracle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from hri_turn_taking.data.assembly import ReadinessReport, load_and_audit_vap_source
from hri_turn_taking.vap.oracle_prev_vad_backend import (
    BACKEND_ID,
    CHECKPOINT_FILENAME,
    CONDITION_IDS,
    EXACT_EIGHT_GROUP_BY_SCENE,
    EXACT_EIGHT_GROUPS,
    EXACT_EIGHT_TICK_RANGES,
    EXACT_EIGHT_TOTAL_ROWS,
    FEATURE_SCHEMA,
    GLOBAL_WIDTH,
    PARTICIPANT_WIDTH,
    PREDICTIONS_FILENAME,
    PREDICTIONS_SIDECAR_FILENAME,
    RESULT_FILENAME,
    LoadedOracleUnit,
    OracleBackendSpec,
    OraclePosthocTorchBackend,
    OracleUnitReceipt,
    OracleUnitRequest,
    build_oracle_condition_sessions,
    fit_oracle_training_normalizer,
    initial_model_state_sha256,
    load_oracle_checkpoint_artifact,
    load_saved_oracle_unit,
    oracle_condition_metadata,
    oracle_label_config,
    oracle_metrics_from_predictions,
    oracle_model_config,
    oracle_trainer_spec,
)
from hri_turn_taking.vap.outer_experiment import (
    DEADLINE_CORPUS_EXCLUSIONS,
    DEADLINE_PROTOCOL_ID,
    validate_outer_run_manifest,
    validate_outer_split_manifest,
)
from hri_turn_taking.vap.thesis_backend import (
    ThesisBackendSpec,
    thesis_model_config,
    thesis_trainer_config,
)

FORMAT_VERSION = 1
PROTOCOL_ID = "previous_vad_oracle_exact8_nested_logo_v1"
PROTOCOL_FILENAME = "protocol.json"
PROTOCOL_BINDING_FILENAME = "protocol.binding.json"
MATRIX_INDEX_FILENAME = "matrix.index.json"
AGGREGATE_KIND = "vap_previous_vad_oracle_saved_probability_aggregate"
FORMAL_MATRIX_KIND = "vap_previous_vad_oracle_outer_matrix"
FIXTURE_MATRIX_KIND = "vap_previous_vad_oracle_fixture_matrix_never_real"
CONDITIONS = tuple(CONDITION_IDS)
OUTER_FOLD_IDS = tuple(f"outer_{index:02d}" for index in range(1, 5))
SEED = 13
MAX_EPOCHS = 15
AUX_VAD_WEIGHT = 0.0
EXCLUDED_OWNER_HOLD_SCENES = dict(DEADLINE_CORPUS_EXCLUSIONS)
CANONICAL_SPLIT_FILENAME = "split_manifest.json"
CANONICAL_RUN_FILENAME = "run_manifest.json"


class OracleOuterExperimentError(ValueError):
    """A frozen protocol, matrix, or aggregate gate failed closed."""


class _UnitRunner(Protocol):
    def run_unit(self, request: OracleUnitRequest) -> OracleUnitReceipt: ...


@dataclass(frozen=True)
class OracleOuterProtocolArtifact:
    directory: Path
    protocol_path: Path
    binding_path: Path
    protocol: dict[str, object]
    protocol_sha256: str


@dataclass(frozen=True)
class OracleOuterMatrixArtifact:
    directory: Path
    index_path: Path
    index: dict[str, object]
    units: dict[tuple[str, str], LoadedOracleUnit]


@dataclass(frozen=True)
class OracleOuterAggregateArtifact:
    path: Path
    payload: dict[str, object]
    sha256: str


@dataclass(frozen=True)
class OracleSourceArtifact:
    source: Path
    frame: pd.DataFrame
    frames: dict[str, pd.DataFrame]
    readiness: ReadinessReport
    readiness_payload: dict[str, object]


@dataclass(frozen=True)
class CanonicalProtocolArtifact:
    directory: Path
    split_path: Path
    run_path: Path
    split: dict[str, object]
    run: dict[str, object]


def _fail(message: str) -> None:
    raise OracleOuterExperimentError(message)


def _boundary() -> dict[str, object]:
    return {
        "posthoc_exploratory": True,
        "oracle_gold_input": True,
        "robot_observable": False,
        "deployable": False,
        "production_eligible": False,
        "counted_eligible": False,
        "formal_result_eligible": False,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_bytes(value: object) -> bytes:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return text.encode("utf-8")


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OracleOuterExperimentError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                OracleOuterExperimentError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OracleOuterExperimentError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"JSON artifact must contain an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def _file_binding(
    path: Path,
    *,
    relative_to: Path | None = None,
) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        _fail(f"artifact must be a regular non-symlink file: {path}")
    recorded = path.relative_to(relative_to).as_posix() if relative_to else str(path)
    return {
        "path": recorded,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_relative_binding(
    value: object,
    root: Path,
    expected_relative_path: str,
) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        _fail(f"binding fields are not exact for {expected_relative_path}")
    if value["path"] != expected_relative_path:
        _fail(f"binding path differs for {expected_relative_path}")
    path = root / Path(expected_relative_path)
    if not path.is_file() or path.is_symlink():
        _fail(f"bound file is missing or invalid: {expected_relative_path}")
    if type(value["size_bytes"]) is not int or value["size_bytes"] != path.stat().st_size:
        _fail(f"bound file size differs: {expected_relative_path}")
    if value["sha256"] != _sha256_file(path):
        _fail(f"bound file SHA-256 differs: {expected_relative_path}")
    return path


def _validate_boundary(value: dict[str, object]) -> None:
    for key, expected in _boundary().items():
        if value.get(key) is not expected:
            _fail(f"eligibility boundary {key!r} must be {expected!r}")


def _audit_source(source: str | Path) -> OracleSourceArtifact:
    source_path = Path(source).resolve()
    if not source_path.is_dir() or source_path.is_symlink():
        _fail(f"audited real source is missing or invalid: {source_path}")
    try:
        frame, readiness = load_and_audit_vap_source(
            source_path,
            data_kind="real",
            feature_set="head_pose_only",
            excluded_session_ids=tuple(EXCLUDED_OWNER_HOLD_SCENES),
        )
    except Exception as exc:
        raise OracleOuterExperimentError(f"real source audit failed: {exc}") from exc
    payload = readiness.to_dict()
    required = {
        "data_kind": "real",
        "feature_set": "head_pose_only",
        "n_rows": EXACT_EIGHT_TOTAL_ROWS,
        "target_ready": True,
        "selected_feature_ready": True,
    }
    for key, expected in required.items():
        if payload.get(key) != expected or type(payload.get(key)) is not type(expected):
            _fail(f"source readiness field {key!r} differs from exact-eight H-only")
    if tuple(sorted(readiness.session_ids)) != tuple(EXACT_EIGHT_TICK_RANGES):
        _fail("source readiness does not contain the exact eight sessions")
    if tuple(sorted(readiness.group_ids)) != EXACT_EIGHT_GROUPS:
        _fail("source readiness does not contain the exact four groups")
    if len(frame) != EXACT_EIGHT_TOTAL_ROWS:
        _fail("audited source dataframe does not contain 64,299 rows")
    frames: dict[str, pd.DataFrame] = {}
    for scene, (start, end) in EXACT_EIGHT_TICK_RANGES.items():
        rows = frame.loc[frame["session_id"].astype(str) == scene].sort_values(
            "timestamp_s"
        )
        if len(rows) != end - start:
            _fail(f"audited source row count differs for {scene}")
        expected_ticks = np.arange(start, end, dtype=np.int64)
        expected_timestamps = expected_ticks.astype(np.float64) / 25.0
        timestamps = rows["timestamp_s"].to_numpy(dtype=np.float64)
        if not np.array_equal(timestamps, expected_timestamps):
            _fail(f"audited source timestamp grid differs for {scene}")
        if "tick_25hz" in rows:
            ticks = rows["tick_25hz"].to_numpy()
            if not pd.api.types.is_integer_dtype(ticks.dtype):
                _fail(f"audited source tick_25hz dtype differs for {scene}")
            if not np.array_equal(ticks, expected_ticks):
                _fail(f"audited source tick range differs for {scene}")
        else:
            rows = rows.copy()
            rows.insert(
                rows.columns.get_loc("timestamp_s") + 1,
                "tick_25hz",
                expected_ticks,
            )
        groups = rows["group_id"].astype(str).unique().tolist()
        if groups != [EXACT_EIGHT_GROUP_BY_SCENE[scene]]:
            _fail(f"audited source group differs for {scene}")
        frames[scene] = rows.reset_index(drop=True)
    return OracleSourceArtifact(
        source=source_path,
        frame=frame,
        frames=frames,
        readiness=readiness,
        readiness_payload=payload,
    )


def _audit_canonical_protocol(
    protocol_directory: str | Path,
    source: OracleSourceArtifact,
) -> CanonicalProtocolArtifact:
    directory = Path(protocol_directory).resolve()
    if not directory.is_dir() or directory.is_symlink():
        _fail(f"canonical 15-epoch protocol is missing or invalid: {directory}")
    split_path = directory / CANONICAL_SPLIT_FILENAME
    run_path = directory / CANONICAL_RUN_FILENAME
    split = _strict_json(split_path)
    run = _strict_json(run_path)
    try:
        validate_outer_split_manifest(split)
        validate_outer_run_manifest(run, split)
    except Exception as exc:
        raise OracleOuterExperimentError(
            f"canonical outer protocol validation failed: {exc}"
        ) from exc
    expected_conditions = [
        {"condition_id": "A", "feature_set": "audio_only", "encoder": "transformer"},
        {"condition_id": "H", "feature_set": "head_pose_only", "encoder": "transformer"},
        {
            "condition_id": "A+H",
            "feature_set": "audio_plus_head_pose",
            "encoder": "transformer",
        },
    ]
    if (
        split.get("protocol_id") != DEADLINE_PROTOCOL_ID
        or split.get("conditions") != expected_conditions
        or split.get("source_session_to_group") != dict(EXACT_EIGHT_GROUP_BY_SCENE)
        or split.get("corpus_exclusions") != EXCLUDED_OWNER_HOLD_SCENES
    ):
        _fail("canonical split is not the exact-eight primary A/H/A+H protocol")
    spec = ThesisBackendSpec()
    if (
        run.get("protocol_id") != DEADLINE_PROTOCOL_ID
        or run.get("source") != str(source.source)
        or run.get("seeds") != [13, 42, 73]
        or run.get("condition_ids") != ["A", "H", "A+H"]
        or type(run.get("max_epochs")) is not int
        or run.get("max_epochs") != MAX_EPOCHS
        or type(run.get("aux_vad_weight")) is not float
        or run.get("aux_vad_weight") != AUX_VAD_WEIGHT
        or run.get("trainer_config") != thesis_trainer_config(spec)
        or run.get("model_config") != thesis_model_config(spec)
        or run.get("corpus_exclusions") != EXCLUDED_OWNER_HOLD_SCENES
    ):
        _fail("canonical run is not the exact-eight 15-epoch source protocol")
    readiness = run.get("readiness_by_condition")
    h_ready = readiness.get("H") if isinstance(readiness, dict) else None
    if not isinstance(h_ready, dict) or (
        h_ready.get("data_kind") != "real"
        or h_ready.get("feature_set") != "head_pose_only"
        or h_ready.get("n_rows") != EXACT_EIGHT_TOTAL_ROWS
        or h_ready.get("selected_feature_ready") is not True
        or tuple(h_ready.get("session_ids", ())) != tuple(EXACT_EIGHT_TICK_RANGES)
        or tuple(h_ready.get("group_ids", ())) != EXACT_EIGHT_GROUPS
    ):
        _fail("canonical run H readiness differs from the fresh exact-eight source")
    return CanonicalProtocolArtifact(
        directory=directory,
        split_path=split_path,
        run_path=run_path,
        split=split,
        run=run,
    )


def _scene_to_group(artifact: OracleSourceArtifact) -> dict[str, str]:
    return dict(EXACT_EIGHT_GROUP_BY_SCENE)


def _source_binding(artifact: OracleSourceArtifact) -> dict[str, object]:
    payload = {
        "source": str(artifact.source),
        "readiness_sha256": hashlib.sha256(
            _canonical_json(artifact.readiness_payload).encode("utf-8")
        ).hexdigest(),
        "session_ids": list(EXACT_EIGHT_TICK_RANGES),
        "group_ids": list(EXACT_EIGHT_GROUPS),
        "total_rows": EXACT_EIGHT_TOTAL_ROWS,
        "tick_ranges": {
            scene: [start, end]
            for scene, (start, end) in EXACT_EIGHT_TICK_RANGES.items()
        },
    }
    payload["source_snapshot_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def _canonical_protocol_binding(
    artifact: CanonicalProtocolArtifact,
) -> dict[str, object]:
    return {
        "directory": str(artifact.directory),
        "split_manifest": _file_binding(artifact.split_path),
        "run_manifest": _file_binding(artifact.run_path),
        "split_canonical_sha256": hashlib.sha256(
            _canonical_json(artifact.split).encode("utf-8")
        ).hexdigest(),
        "run_canonical_sha256": hashlib.sha256(
            _canonical_json(artifact.run).encode("utf-8")
        ).hexdigest(),
    }


def _sessions_for_groups(groups: tuple[str, ...]) -> list[str]:
    group_set = set(groups)
    return [
        scene
        for scene, group in EXACT_EIGHT_GROUP_BY_SCENE.items()
        if group in group_set
    ]


def _outer_folds(initial_state_sha256: str) -> list[dict[str, object]]:
    folds = []
    for index, test_group in enumerate(EXACT_EIGHT_GROUPS, start=1):
        train_groups = tuple(group for group in EXACT_EIGHT_GROUPS if group != test_group)
        inner_folds = []
        for inner_index, validation_group in enumerate(train_groups, start=1):
            inner_train = tuple(
                group for group in train_groups if group != validation_group
            )
            inner_folds.append(
                {
                    "inner_fold_id": f"inner_{inner_index:02d}",
                    "training_groups": list(inner_train),
                    "validation_group": validation_group,
                    "training_sessions": _sessions_for_groups(inner_train),
                    "validation_sessions": _sessions_for_groups((validation_group,)),
                }
            )
        folds.append(
            {
                "fold_id": f"outer_{index:02d}",
                "outer_train_groups": list(train_groups),
                "outer_test_group": test_group,
                "outer_train_sessions": _sessions_for_groups(train_groups),
                "outer_test_sessions": _sessions_for_groups((test_group,)),
                "inner_folds": inner_folds,
                "condition_initial_state_sha256": {
                    condition: initial_state_sha256 for condition in CONDITIONS
                },
            }
        )
    return folds


def _protocol_payload(
    source: OracleSourceArtifact,
    canonical: CanonicalProtocolArtifact,
    *,
    initial_state_sha256: str | None = None,
) -> dict[str, object]:
    backend_spec = OracleBackendSpec()
    initial_state = initial_state_sha256 or initial_model_state_sha256(
        backend_spec, SEED
    )
    if not _is_sha256(initial_state):
        _fail("protocol initial-state SHA-256 is invalid")
    return {
        "artifact_kind": "vap_previous_vad_oracle_outer_protocol",
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "result_role": "posthoc_exploratory_only",
        "source_snapshot": _source_binding(source),
        "canonical_exact8_15epoch_protocol": _canonical_protocol_binding(canonical),
        "source_contract": {
            "data_kind": "real",
            "base_feature_set": "head_pose_only",
            "target_ready": True,
            "selected_feature_ready": True,
            "gold_input_source": "human-reviewed-Transcript-derived VAD",
            "gold_input_timing": "same_session_previous_tick_only",
            "gold_input_delay_ticks": 1,
            "gold_input_delay_seconds": 0.04,
        },
        "corpus": {
            "exact_scene_inventory": list(EXACT_EIGHT_TICK_RANGES),
            "session_to_group": dict(EXACT_EIGHT_GROUP_BY_SCENE),
            "excluded_owner_hold_scenes": dict(EXCLUDED_OWNER_HOLD_SCENES),
            "total_rows": EXACT_EIGHT_TOTAL_ROWS,
            "group_ids": list(EXACT_EIGHT_GROUPS),
            "n_groups": 4,
        },
        "conditions": [
            oracle_condition_metadata(condition) for condition in CONDITIONS
        ],
        "schema_dimensions": {
            "feature_schema": FEATURE_SCHEMA,
            "conceptual_raw_input_columns": 36,
            "transformed_input_dimensions": 37,
            "participant_features_per_token": PARTICIPANT_WIDTH,
            "global_features_per_token": GLOBAL_WIDTH,
            "token_input_dimensions": PARTICIPANT_WIDTH + GLOBAL_WIDTH,
        },
        "training": {
            "seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "aux_vad_weight": AUX_VAD_WEIGHT,
            "epoch_selection": "median_inner_logo_selected_epoch",
            "outer_test_access": "one_saved_probability_pass_per_condition_fold",
        },
        "backend": {
            "backend_id": BACKEND_ID,
            "implementation": "OraclePosthocTorchBackend",
            "backend_spec": backend_spec.to_dict(),
            "model_config": oracle_model_config(backend_spec),
            "trainer_spec": oracle_trainer_spec(backend_spec),
            "label_config": oracle_label_config(),
            "initial_state_strategy": {
                "algorithm": (
                    "torch_manual_seed_13_before_each_cpu_model_construction_"
                    "then_canonical_state_dict_sha256"
                ),
                "seed": SEED,
                "sha256": initial_state,
                "reset_before_every_inner_and_final_fit": True,
                "paired_conditions_share_initial_state": True,
            },
        },
        "outer_folds": _outer_folds(initial_state),
        "primary_comparison": "O1_minus_O0_one_tick_previous_VAD_increment",
        "comparison_status": "descriptive_posthoc_no_success_threshold",
        **_boundary(),
    }


def freeze_oracle_outer_protocol(
    source: str | Path,
    canonical_protocol: str | Path,
    destination: str | Path,
) -> OracleOuterProtocolArtifact:
    """Freeze O0/O1 × four folds before this diagnostic sees outer outcomes."""

    audited_source = _audit_source(source)
    canonical = _audit_canonical_protocol(canonical_protocol, audited_source)
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite Oracle protocol: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}-stage-",
            dir=destination_path.parent,
        )
    ).resolve()
    try:
        protocol = _protocol_payload(audited_source, canonical)
        protocol_path = stage / PROTOCOL_FILENAME
        _write_exclusive(protocol_path, _json_bytes(protocol))
        binding = {
            "artifact_kind": "vap_previous_vad_oracle_outer_protocol_binding",
            "format_version": FORMAT_VERSION,
            "protocol": _file_binding(protocol_path, relative_to=stage),
            **_boundary(),
        }
        _write_exclusive(stage / PROTOCOL_BINDING_FILENAME, _json_bytes(binding))
        if destination_path.exists():
            raise FileExistsError(
                f"refusing to overwrite Oracle protocol: {destination_path}"
            )
        _load_protocol(stage, audited_source)
        stage.rename(destination_path)
        try:
            audited_final = _load_protocol(destination_path, audited_source)
        except BaseException:
            if destination_path.exists() and not destination_path.is_symlink():
                shutil.rmtree(destination_path)
            raise
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return audited_final


def _load_protocol(
    protocol_directory: str | Path,
    source: OracleSourceArtifact,
    *,
    verify_initial_state: bool = True,
) -> OracleOuterProtocolArtifact:
    directory = Path(protocol_directory).resolve()
    if not directory.is_dir() or directory.is_symlink():
        _fail(f"Oracle protocol directory is missing or invalid: {directory}")
    if {path.name for path in directory.iterdir()} != {
        PROTOCOL_FILENAME,
        PROTOCOL_BINDING_FILENAME,
    }:
        _fail("Oracle protocol directory file set is not exact")
    protocol_path = directory / PROTOCOL_FILENAME
    binding_path = directory / PROTOCOL_BINDING_FILENAME
    protocol = _strict_json(protocol_path)
    binding = _strict_json(binding_path)
    expected_binding_keys = {
        "artifact_kind",
        "format_version",
        "protocol",
        *_boundary(),
    }
    if (
        set(binding) != expected_binding_keys
        or
        binding.get("artifact_kind")
        != "vap_previous_vad_oracle_outer_protocol_binding"
        or binding.get("format_version") != FORMAT_VERSION
    ):
        _fail("Oracle protocol binding kind/version is invalid")
    _validate_boundary(binding)
    _validate_relative_binding(binding.get("protocol"), directory, PROTOCOL_FILENAME)
    backend = protocol.get("backend")
    strategy = backend.get("initial_state_strategy") if isinstance(backend, dict) else None
    declared_initial_state = strategy.get("sha256") if isinstance(strategy, dict) else None
    if not _is_sha256(declared_initial_state):
        _fail("Oracle protocol has no valid initial-state SHA-256")
    if verify_initial_state and declared_initial_state != initial_model_state_sha256(
        OracleBackendSpec(), SEED
    ):
        _fail("Oracle protocol initial-state SHA-256 cannot be replayed")
    canonical_binding = protocol.get("canonical_exact8_15epoch_protocol")
    canonical_directory = (
        canonical_binding.get("directory")
        if isinstance(canonical_binding, dict)
        else None
    )
    if not isinstance(canonical_directory, str):
        _fail("Oracle protocol has no canonical 15-epoch protocol directory")
    canonical = _audit_canonical_protocol(canonical_directory, source)
    expected = _protocol_payload(
        source,
        canonical,
        initial_state_sha256=declared_initial_state,
    )
    if _canonical_json(protocol) != _canonical_json(expected):
        _fail("Oracle protocol failed exact frozen replay")
    _validate_boundary(protocol)
    protocol_sha256 = _sha256_file(protocol_path)
    return OracleOuterProtocolArtifact(
        directory=directory,
        protocol_path=protocol_path,
        binding_path=binding_path,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )


def _fold_by_id(protocol: dict[str, object], fold_id: str) -> dict[str, object]:
    folds = protocol.get("outer_folds")
    if not isinstance(folds, list):
        _fail("protocol outer_folds is invalid")
    matches = [
        fold
        for fold in folds
        if isinstance(fold, dict) and fold.get("fold_id") == fold_id
    ]
    if len(matches) != 1:
        _fail(f"protocol has no unique fold {fold_id}")
    return matches[0]


def _unit_relative(condition: str, fold_id: str, filename: str) -> str:
    return f"{condition}/{fold_id}/{filename}"


def _unit_record(
    root: Path,
    condition: str,
    fold: dict[str, object],
) -> dict[str, object]:
    fold_id = str(fold["fold_id"])
    unit = root / condition / fold_id
    return {
        "condition_id": condition,
        "fold_id": fold_id,
        "outer_test_group": fold["outer_test_group"],
        "outer_test_sessions": fold["outer_test_sessions"],
        "initial_state_sha256": fold["condition_initial_state_sha256"][condition],
        "artifacts": {
            "checkpoint": _file_binding(
                unit / CHECKPOINT_FILENAME, relative_to=root
            ),
            "predictions": _file_binding(
                unit / PREDICTIONS_FILENAME, relative_to=root
            ),
            "predictions_sidecar": _file_binding(
                unit / PREDICTIONS_SIDECAR_FILENAME, relative_to=root
            ),
            "result": _file_binding(unit / RESULT_FILENAME, relative_to=root),
        },
    }


def _matrix_index(
    root: Path,
    protocol: OracleOuterProtocolArtifact,
    source: OracleSourceArtifact,
    *,
    fixture_only: bool,
) -> dict[str, object]:
    units = []
    for condition in CONDITIONS:
        for fold_id in OUTER_FOLD_IDS:
            units.append(
                _unit_record(root, condition, _fold_by_id(protocol.protocol, fold_id))
            )
    return {
        "artifact_kind": FIXTURE_MATRIX_KIND if fixture_only else FORMAL_MATRIX_KIND,
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "result_role": (
            "fixture_only_never_formal"
            if fixture_only
            else "posthoc_exploratory_only"
        ),
        "fixture_only": fixture_only,
        "source_snapshot": _source_binding(source),
        "protocol": _file_binding(protocol.protocol_path),
        "conditions": list(CONDITIONS),
        "outer_fold_ids": list(OUTER_FOLD_IDS),
        "planned_units": 8,
        "completed_units": 8,
        "units": units,
        **_boundary(),
    }


def _fold_initial_state(fold: dict[str, object], condition: str) -> str:
    values = fold.get("condition_initial_state_sha256")
    if not isinstance(values, dict) or set(values) != set(CONDITIONS):
        _fail("protocol fold initial-state map is invalid")
    value = values.get(condition)
    if not _is_sha256(value):
        _fail("protocol fold initial-state SHA-256 is invalid")
    return value


def _run_oracle_outer_matrix_with_runner(
    protocol: str | Path,
    source_path: str | Path,
    output: str | Path,
    *,
    runner: _UnitRunner,
    fixture_only: bool,
) -> OracleOuterMatrixArtifact:
    if not fixture_only and type(runner) is not OraclePosthocTorchBackend:
        _fail("formal Oracle matrix requires the exact frozen Torch backend")
    source = _audit_source(source_path)
    frozen = _load_protocol(protocol, source)
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Oracle outer matrix: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-stage-", dir=destination.parent)
    ).resolve()
    try:
        for condition in CONDITIONS:
            sessions = build_oracle_condition_sessions(
                source.frames,
                condition,
                scene_to_group=_scene_to_group(source),
            )
            for fold_id in OUTER_FOLD_IDS:
                fold = _fold_by_id(frozen.protocol, fold_id)
                request = OracleUnitRequest(
                    condition_id=condition,
                    fold_id=fold_id,
                    outer_test_group=str(fold["outer_test_group"]),
                    outer_train_groups=tuple(str(value) for value in fold["outer_train_groups"]),
                    inner_validation_groups=tuple(
                        str(value) for value in fold["outer_train_groups"]
                    ),
                    sessions=sessions,
                    output_directory=stage / condition / fold_id,
                    source_snapshot_sha256=str(
                        _source_binding(source)["source_snapshot_sha256"]
                    ),
                    protocol_sha256=frozen.protocol_sha256,
                    expected_initial_state_sha256=_fold_initial_state(fold, condition),
                    seed=SEED,
                    max_epochs=MAX_EPOCHS,
                    aux_vad_weight=AUX_VAD_WEIGHT,
                    fixture_only=fixture_only,
                )
                receipt = runner.run_unit(request)
                if receipt.output_directory.resolve() != request.output_directory.resolve():
                    _fail("unit runner returned a receipt outside its assigned directory")
        index = _matrix_index(
            stage,
            frozen,
            source,
            fixture_only=fixture_only,
        )
        _write_exclusive(stage / MATRIX_INDEX_FILENAME, _json_bytes(index))
        _audit_matrix(
            stage,
            source,
            frozen,
            fixture_only=fixture_only,
            load_checkpoints=True,
        )
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite Oracle outer matrix: {destination}"
            )
        stage.rename(destination)
        try:
            audited_final = _audit_matrix(
                destination,
                source,
                frozen,
                fixture_only=fixture_only,
                load_checkpoints=True,
            )
        except BaseException:
            if destination.exists() and not destination.is_symlink():
                shutil.rmtree(destination)
            raise
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return OracleOuterMatrixArtifact(
        directory=destination,
        index_path=destination / MATRIX_INDEX_FILENAME,
        index=audited_final.index,
        units=audited_final.units,
    )


def run_oracle_outer_matrix(
    protocol: str | Path,
    source: str | Path,
    output: str | Path,
) -> OracleOuterMatrixArtifact:
    """Run the real eight-unit post-hoc matrix with the frozen backend."""

    runner = OraclePosthocTorchBackend(OracleBackendSpec())
    return _run_oracle_outer_matrix_with_runner(
        protocol,
        source,
        output,
        runner=runner,
        fixture_only=False,
    )


def _run_oracle_outer_fixture_matrix(
    protocol: str | Path,
    source: str | Path,
    output: str | Path,
    *,
    fixture_runner: _UnitRunner,
) -> OracleOuterMatrixArtifact:
    """Test-only harness whose artifacts are permanently rejected as formal."""

    return _run_oracle_outer_matrix_with_runner(
        protocol,
        source,
        output,
        runner=fixture_runner,
        fixture_only=True,
    )


def _validate_index_artifact_bindings(
    root: Path,
    record: dict[str, object],
    condition: str,
    fold_id: str,
) -> None:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "checkpoint",
        "predictions",
        "predictions_sidecar",
        "result",
    }:
        _fail("matrix unit artifact bindings are not exact")
    expected_names = {
        "checkpoint": CHECKPOINT_FILENAME,
        "predictions": PREDICTIONS_FILENAME,
        "predictions_sidecar": PREDICTIONS_SIDECAR_FILENAME,
        "result": RESULT_FILENAME,
    }
    for role, filename in expected_names.items():
        _validate_relative_binding(
            artifacts[role],
            root,
            _unit_relative(condition, fold_id, filename),
        )


def _audit_matrix(
    matrix_directory: str | Path,
    source: OracleSourceArtifact,
    protocol: OracleOuterProtocolArtifact | None = None,
    *,
    fixture_only: bool,
    load_checkpoints: bool,
) -> OracleOuterMatrixArtifact:
    root = Path(matrix_directory).resolve()
    if not root.is_dir() or root.is_symlink():
        _fail(f"Oracle matrix directory is missing or invalid: {root}")
    if {path.name for path in root.iterdir()} != {
        *CONDITIONS,
        MATRIX_INDEX_FILENAME,
    }:
        _fail("Oracle matrix root file set is not exact")
    index_path = root / MATRIX_INDEX_FILENAME
    index = _strict_json(index_path)
    expected_kind = FIXTURE_MATRIX_KIND if fixture_only else FORMAL_MATRIX_KIND
    expected_role = (
        "fixture_only_never_formal" if fixture_only else "posthoc_exploratory_only"
    )
    if (
        index.get("artifact_kind") != expected_kind
        or index.get("format_version") != FORMAT_VERSION
        or index.get("protocol_id") != PROTOCOL_ID
        or index.get("result_role") != expected_role
        or index.get("fixture_only") is not fixture_only
    ):
        _fail("Oracle matrix index kind/version/protocol is invalid")
    _validate_boundary(index)
    if _canonical_json(index.get("source_snapshot")) != _canonical_json(
        _source_binding(source)
    ):
        _fail("Oracle matrix source binding differs from the audited source")
    protocol_binding = index.get("protocol")
    if not isinstance(protocol_binding, dict):
        _fail("Oracle matrix protocol binding is invalid")
    protocol_path_value = protocol_binding.get("path")
    if not isinstance(protocol_path_value, str):
        _fail("Oracle matrix protocol path is invalid")
    bound_protocol_path = Path(protocol_path_value).resolve()
    if not bound_protocol_path.is_file() or bound_protocol_path.is_symlink():
        _fail("Oracle matrix bound protocol is unavailable")
    if (
        protocol_binding.get("sha256") != _sha256_file(bound_protocol_path)
        or protocol_binding.get("size_bytes") != bound_protocol_path.stat().st_size
    ):
        _fail("Oracle matrix protocol binding differs")
    if protocol is None:
        protocol = _load_protocol(
            bound_protocol_path.parent,
            source,
            verify_initial_state=load_checkpoints,
        )
    if protocol.protocol_path != bound_protocol_path:
        _fail("Oracle matrix protocol path differs from loaded authority")
    if index.get("conditions") != list(CONDITIONS):
        _fail("Oracle matrix condition inventory differs")
    if index.get("outer_fold_ids") != list(OUTER_FOLD_IDS):
        _fail("Oracle matrix fold inventory differs")
    if index.get("planned_units") != 8 or index.get("completed_units") != 8:
        _fail("Oracle matrix is not a complete 8-unit run")
    expected_index = _matrix_index(
        root,
        protocol,
        source,
        fixture_only=fixture_only,
    )
    if _canonical_json(index) != _canonical_json(expected_index):
        _fail("Oracle matrix index failed exact hash-bound replay")
    records = index.get("units")
    if not isinstance(records, list) or len(records) != 8:
        _fail("Oracle matrix unit records are invalid")
    expected_pairs = [
        (condition, fold_id)
        for condition in CONDITIONS
        for fold_id in OUTER_FOLD_IDS
    ]
    units: dict[tuple[str, str], LoadedOracleUnit] = {}
    audited_normalizers: dict[tuple[str, str], dict[str, object]] = {}
    sessions_by_condition = {
        condition: build_oracle_condition_sessions(
            source.frames,
            condition,
            scene_to_group=_scene_to_group(source),
        )
        for condition in CONDITIONS
    }
    for record, (condition, fold_id) in zip(records, expected_pairs, strict=True):
        if not isinstance(record, dict):
            _fail("Oracle matrix unit record is not an object")
        fold = _fold_by_id(protocol.protocol, fold_id)
        expected_header = {
            "condition_id": condition,
            "fold_id": fold_id,
            "outer_test_group": fold["outer_test_group"],
            "outer_test_sessions": fold["outer_test_sessions"],
            "initial_state_sha256": _fold_initial_state(fold, condition),
        }
        for key, expected in expected_header.items():
            if record.get(key) != expected:
                _fail(f"Oracle matrix unit field {key!r} differs")
        _validate_index_artifact_bindings(root, record, condition, fold_id)
        sessions = tuple(
            session
            for session in sessions_by_condition[condition]
            if session.group_id == fold["outer_test_group"]
        )
        expected_initial_state = _fold_initial_state(fold, condition)
        loaded = load_saved_oracle_unit(
            root / condition / fold_id,
            sessions,
            condition_id=condition,
            fold_id=fold_id,
            outer_test_group=str(fold["outer_test_group"]),
            source_snapshot_sha256=str(
                _source_binding(source)["source_snapshot_sha256"]
            ),
            protocol_sha256=protocol.protocol_sha256,
            expected_initial_state_sha256=expected_initial_state,
            fixture_only=fixture_only,
        )
        if load_checkpoints:
            checkpoint = load_oracle_checkpoint_artifact(
                root / condition / fold_id / CHECKPOINT_FILENAME,
                condition_id=condition,
                fold_id=fold_id,
                outer_test_group=str(fold["outer_test_group"]),
                source_snapshot_sha256=str(
                    _source_binding(source)["source_snapshot_sha256"]
                ),
                protocol_sha256=protocol.protocol_sha256,
                expected_initial_state_sha256=expected_initial_state,
                fixture_only=fixture_only,
            )
            payload = checkpoint["payload"]
            if not isinstance(payload, dict):
                _fail("Oracle checkpoint core payload is invalid after loading")
            if payload["selected_final_epochs"] != loaded.result[
                "selected_final_epochs"
            ]:
                _fail("checkpoint/result selected final epochs differ")
            if payload["final_history"] != loaded.result["final_history"]:
                _fail("checkpoint/result final histories differ")
            if payload["initial_state_sha256"] != loaded.result[
                "initial_state_sha256"
            ]:
                _fail("checkpoint/result initial-state hashes differ")
            expected_normalizer = fit_oracle_training_normalizer(
                sessions_by_condition[condition],
                tuple(str(value) for value in fold["outer_train_groups"]),
            ).to_dict()
            if _canonical_json(payload["normalizer"]) != _canonical_json(
                expected_normalizer
            ):
                _fail(
                    f"checkpoint normalizer differs from outer-training replay for "
                    f"{condition}/{fold_id}"
                )
            audited_normalizers[(condition, fold_id)] = dict(payload["normalizer"])
        units[(condition, fold_id)] = loaded
    for fold_id in OUTER_FOLD_IDS:
        initial_hashes = {
            units[(condition, fold_id)].result["initial_state_sha256"]
            for condition in CONDITIONS
        }
        if len(initial_hashes) != 1:
            _fail(f"paired conditions do not share one initial state in {fold_id}")
        if load_checkpoints and _canonical_json(
            audited_normalizers[("O0", fold_id)]
        ) != _canonical_json(audited_normalizers[("O1", fold_id)]):
            _fail(f"paired conditions do not share one normalizer in {fold_id}")
    return OracleOuterMatrixArtifact(
        directory=root,
        index_path=index_path,
        index=index,
        units=units,
    )


def audit_oracle_outer_matrix(
    matrix_directory: str | Path,
    source: str | Path,
) -> OracleOuterMatrixArtifact:
    """Replay source authority and audit all eight saved-probability units."""

    audited_source = _audit_source(source)
    return _audit_matrix(
        matrix_directory,
        audited_source,
        fixture_only=False,
        load_checkpoints=True,
    )


def _metric_at(metrics: dict[str, object], path: tuple[str, ...]) -> float | None:
    value: object = metrics
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _comparison(
    metrics: dict[str, dict[str, object]],
    left: str,
    right: str,
) -> dict[str, float | None]:
    paths = {
        "frame_mAP": ("frame", "mAP"),
        "frame_micro_AP": ("frame", "micro_AP"),
        "frame_macro_f1_bins": ("frame", "macro_f1_bins"),
        "shift_hold_AP": ("events", "shift_hold", "average_precision"),
        "shift_prediction_AP": (
            "events",
            "shift_prediction",
            "average_precision",
        ),
        "overlap_resolution_AP": (
            "events",
            "overlap_resolution",
            "average_precision",
        ),
        "robot_entry_AP": ("events", "robot_entry", "average_precision"),
        "next_speaker_macro_AP": (
            "events",
            "next_speaker",
            "participant_macro_ovr_ap",
        ),
    }
    result: dict[str, float | None] = {}
    for name, path in paths.items():
        left_value = _metric_at(metrics[left], path)
        right_value = _metric_at(metrics[right], path)
        result[name] = (
            left_value - right_value
            if left_value is not None and right_value is not None
            else None
        )
    return result


def _aggregate_payload(
    audited: OracleOuterMatrixArtifact,
    audited_source: OracleSourceArtifact,
) -> dict[str, object]:
    """Rebuild the aggregate entirely from audited saved probabilities."""

    metrics: dict[str, dict[str, object]] = {}
    condition_sources: dict[str, list[dict[str, object]]] = {}
    for condition in CONDITIONS:
        sessions = build_oracle_condition_sessions(
            audited_source.frames,
            condition,
            scene_to_group=_scene_to_group(audited_source),
        )
        by_session: dict[str, tuple[object, object]] = {}
        unit_sources = []
        for fold_id in OUTER_FOLD_IDS:
            fold_group = EXACT_EIGHT_GROUPS[OUTER_FOLD_IDS.index(fold_id)]
            outer_sessions = tuple(
                session for session in sessions if session.group_id == fold_group
            )
            loaded = audited.units[(condition, fold_id)]
            for session, probability in zip(
                outer_sessions, loaded.predictions, strict=True
            ):
                if session.session_id in by_session:
                    _fail(f"outer session appears twice for {condition}: {session.session_id}")
                by_session[session.session_id] = (session, probability)
            unit_sources.append(
                {
                    "fold_id": fold_id,
                    "result_sha256": _sha256_file(
                        audited.directory / condition / fold_id / RESULT_FILENAME
                    ),
                    "predictions_sha256": _sha256_file(
                        audited.directory / condition / fold_id / PREDICTIONS_FILENAME
                    ),
                    "predictions_sidecar_sha256": _sha256_file(
                        audited.directory
                        / condition
                        / fold_id
                        / PREDICTIONS_SIDECAR_FILENAME
                    ),
                }
            )
        if tuple(by_session) != tuple(EXACT_EIGHT_TICK_RANGES):
            _fail(f"pooled session inventory/order differs for {condition}")
        pooled_sessions = [by_session[scene][0] for scene in EXACT_EIGHT_TICK_RANGES]
        pooled_predictions = [by_session[scene][1] for scene in EXACT_EIGHT_TICK_RANGES]
        metrics[condition] = oracle_metrics_from_predictions(
            pooled_sessions, pooled_predictions
        )
        condition_sources[condition] = unit_sources
    return {
        "artifact_kind": AGGREGATE_KIND,
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "result_role": "posthoc_exploratory_saved_probability_aggregate",
        "aggregation_method": "pooled_out_of_group_saved_probabilities_no_inference",
        "source_snapshot": _source_binding(audited_source),
        "matrix_index": _file_binding(audited.index_path),
        "exact_scene_inventory": list(EXACT_EIGHT_TICK_RANGES),
        "total_rows": EXACT_EIGHT_TOTAL_ROWS,
        "group_ids": list(EXACT_EIGHT_GROUPS),
        "conditions": list(CONDITIONS),
        "condition_sources": condition_sources,
        "metrics": metrics,
        "comparisons": {
            "primary_O1_minus_O0": _comparison(metrics, "O1", "O0"),
        },
        "interpretation_boundary": (
            "single-seed post-hoc gold-input upper bound; cannot establish deployable "
            "ASD/VAD performance, H1/H2, causal sensing equivalence, or significance"
        ),
        **_boundary(),
    }


def _audit_aggregate_file(
    aggregate: str | Path,
    audited: OracleOuterMatrixArtifact,
    audited_source: OracleSourceArtifact,
) -> OracleOuterAggregateArtifact:
    path = Path(aggregate).resolve()
    if path.suffix.lower() != ".json":
        _fail("Oracle aggregate must be a .json file")
    if not path.is_file() or path.is_symlink():
        _fail(f"Oracle aggregate is missing or invalid: {path}")
    payload = _strict_json(path)
    expected = _aggregate_payload(audited, audited_source)
    if _canonical_json(payload) != _canonical_json(expected):
        _fail("Oracle aggregate differs from saved-probability replay")
    _validate_boundary(payload)
    return OracleOuterAggregateArtifact(
        path=path,
        payload=payload,
        sha256=_sha256_file(path),
    )


def audit_oracle_outer_aggregate(
    aggregate: str | Path,
    matrix: str | Path,
    source: str | Path,
) -> OracleOuterAggregateArtifact:
    """Independently replay-audit an existing saved-probability aggregate."""

    audited_source = _audit_source(source)
    audited = _audit_matrix(
        matrix,
        audited_source,
        fixture_only=False,
        load_checkpoints=False,
    )
    return _audit_aggregate_file(aggregate, audited, audited_source)


def _published_file_identity(path: Path) -> tuple[int, int, int, str]:
    if not path.is_file() or path.is_symlink():
        _fail(f"published Oracle aggregate is missing or invalid: {path}")
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, _sha256_file(path)


def aggregate_oracle_outer_matrix(
    matrix: str | Path,
    source: str | Path,
    output: str | Path,
) -> Path:
    """Publish a replay-audited aggregate without loading a model checkpoint."""

    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".json":
        _fail("Oracle aggregate output must be a .json file")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Oracle aggregate: {output_path}")
    audited_source = _audit_source(source)
    audited = _audit_matrix(
        matrix,
        audited_source,
        fixture_only=False,
        load_checkpoints=False,
    )
    payload = _aggregate_payload(audited, audited_source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}-stage-",
            dir=output_path.parent,
        )
    ).resolve()
    stage_path = stage / output_path.name
    published = False
    published_identity: tuple[int, int, int, str] | None = None
    try:
        _write_exclusive(stage_path, _json_bytes(payload))
        _audit_aggregate_file(stage_path, audited, audited_source)
        try:
            os.link(stage_path, output_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite Oracle aggregate: {output_path}"
            ) from exc
        published = True
        published_identity = _published_file_identity(output_path)
        audited_final = _audit_aggregate_file(
            output_path,
            audited,
            audited_source,
        )
        stage_path.unlink()
        stage.rmdir()
        return audited_final.path
    except BaseException as exc:
        rollback_error: OracleOuterExperimentError | None = None
        if published:
            try:
                current_identity = _published_file_identity(output_path)
            except OracleOuterExperimentError:
                current_identity = None
            if current_identity != published_identity:
                rollback_error = OracleOuterExperimentError(
                    "post-publication aggregate audit failed and output identity "
                    "changed; refusing to delete an unowned replacement"
                )
            else:
                try:
                    output_path.unlink()
                except OSError as unlink_exc:
                    rollback_error = OracleOuterExperimentError(
                        "post-publication aggregate audit failed and the owned "
                        "output could not be removed"
                    )
                    rollback_error.__cause__ = unlink_exc
        stage_cleanup_error: Exception | None = None
        if stage.exists() and stage.is_dir() and not stage.is_symlink():
            try:
                shutil.rmtree(stage)
            except Exception as cleanup_exc:
                stage_cleanup_error = cleanup_exc
        if rollback_error is not None:
            raise rollback_error from exc
        if stage_cleanup_error is not None:
            raise OracleOuterExperimentError(
                "Oracle aggregate destination was rolled back but stage cleanup failed"
            ) from exc
        raise


__all__ = [
    "AGGREGATE_KIND",
    "AUX_VAD_WEIGHT",
    "CONDITIONS",
    "FORMAT_VERSION",
    "OracleOuterExperimentError",
    "OracleOuterAggregateArtifact",
    "OracleOuterMatrixArtifact",
    "OracleOuterProtocolArtifact",
    "MATRIX_INDEX_FILENAME",
    "MAX_EPOCHS",
    "OUTER_FOLD_IDS",
    "PROTOCOL_ID",
    "SEED",
    "aggregate_oracle_outer_matrix",
    "audit_oracle_outer_aggregate",
    "audit_oracle_outer_matrix",
    "freeze_oracle_outer_protocol",
    "run_oracle_outer_matrix",
]
