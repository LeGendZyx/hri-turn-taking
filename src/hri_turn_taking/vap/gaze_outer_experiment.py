"""Frozen, isolated nested-LOGO runner for the schema-v3 Gaze post-hoc study."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hri_turn_taking.data.gaze_posthoc import (
    EXACT_EIGHT_GROUP_BY_SCENE,
    EXACT_EIGHT_GROUPS,
    EXACT_EIGHT_TICK_RANGES,
    EXACT_EIGHT_TOTAL_ROWS,
    EXCLUDED_OWNER_HOLD_SCENES,
    REAL_MATRIX_ARTIFACT_KIND,
    REAL_PERMITTED_USE,
    RealGazePosthocMatrixArtifact,
    audit_real_posthoc_matrix,
)
from hri_turn_taking.vap.feature_sets import (
    GAZE_POSTHOC_CONDITION_IDS,
    gaze_posthoc_feature_set_metadata,
)
from hri_turn_taking.vap.gaze_posthoc_backend import (
    BACKEND_ID,
    CHECKPOINT_FILENAME,
    PREDICTIONS_FILENAME,
    PREDICTIONS_SIDECAR_FILENAME,
    RESULT_FILENAME,
    GazeBackendSpec,
    GazePosthocTorchBackend,
    GazeUnitReceipt,
    GazeUnitRequest,
    LoadedGazeUnit,
    build_gaze_condition_sessions,
    gaze_label_config,
    gaze_metrics_from_predictions,
    gaze_model_config,
    gaze_trainer_spec,
    initial_model_state_sha256,
    load_gaze_checkpoint_artifact,
    load_saved_gaze_unit,
)

FORMAT_VERSION = 1
PROTOCOL_ID = "gaze_posthoc_exact8_nested_logo_v1"
PROTOCOL_FILENAME = "protocol.json"
PROTOCOL_BINDING_FILENAME = "protocol.binding.json"
MATRIX_INDEX_FILENAME = "matrix.index.json"
AGGREGATE_KIND = "vap_gaze_posthoc_saved_probability_aggregate"
FORMAL_MATRIX_KIND = "vap_gaze_posthoc_outer_matrix"
FIXTURE_MATRIX_KIND = "vap_gaze_posthoc_outer_matrix_fixture_never_formal"
CONDITIONS = tuple(GAZE_POSTHOC_CONDITION_IDS)
OUTER_FOLD_IDS = tuple(f"outer_{index:02d}" for index in range(1, 5))
SEED = 13
MAX_EPOCHS = 15
AUX_VAD_WEIGHT = 0.0


class GazeOuterExperimentError(ValueError):
    """A frozen protocol, matrix, or aggregate gate failed closed."""


class _UnitRunner(Protocol):
    def run_unit(self, request: GazeUnitRequest) -> GazeUnitReceipt: ...


@dataclass(frozen=True)
class GazeOuterProtocolArtifact:
    directory: Path
    protocol_path: Path
    binding_path: Path
    protocol: dict[str, object]
    protocol_sha256: str


@dataclass(frozen=True)
class GazeOuterMatrixArtifact:
    directory: Path
    index_path: Path
    index: dict[str, object]
    units: dict[tuple[str, str], LoadedGazeUnit]


def _fail(message: str) -> None:
    raise GazeOuterExperimentError(message)


def _boundary() -> dict[str, object]:
    return {
        "posthoc_exploratory": True,
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
            raise GazeOuterExperimentError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                GazeOuterExperimentError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GazeOuterExperimentError(f"cannot read strict JSON {path}: {exc}") from exc
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


def _validate_source_artifact(
    artifact: RealGazePosthocMatrixArtifact,
) -> RealGazePosthocMatrixArtifact:
    manifest = artifact.manifest
    required = {
        "artifact_kind": REAL_MATRIX_ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "permitted_use": REAL_PERMITTED_USE,
        "posthoc_exploratory_training_ready": True,
        "formal_outer_runner_admitted": False,
        "assembly_eligible": False,
        "production_eligible": False,
        "counted_eligible": False,
        "upstream_exact_regeneration_available": False,
        "implementation_counterfactual_causality_verified": False,
        "source_corpus_scene_count": 10,
        "selected_scene_count": 8,
        "total_rows": EXACT_EIGHT_TOTAL_ROWS,
        "group_ids": list(EXACT_EIGHT_GROUPS),
        "n_groups": 4,
        "exact_scene_inventory": list(EXACT_EIGHT_TICK_RANGES),
        "excluded_owner_hold_scenes": dict(EXCLUDED_OWNER_HOLD_SCENES),
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            _fail(f"audited source matrix field {key!r} differs from the frozen contract")
    schema = manifest.get("schema")
    if not isinstance(schema, dict):
        _fail("audited source matrix has no schema object")
    expected_schema = {
        "feature_schema": "multiparty_v3_gaze_validity",
        "raw_input_width": 36,
        "transformed_input_width": 37,
        "token_input_width": 13,
    }
    for key, expected in expected_schema.items():
        if schema.get(key) != expected:
            _fail(f"audited source matrix schema field {key!r} differs")
    scene_records = manifest.get("scenes")
    if not isinstance(scene_records, dict) or tuple(scene_records) != tuple(
        EXACT_EIGHT_TICK_RANGES
    ):
        _fail("audited source matrix scene-record inventory/order differs")
    for scene, expected_group in EXACT_EIGHT_GROUP_BY_SCENE.items():
        record = scene_records.get(scene)
        if not isinstance(record, dict) or type(record.get("group_id")) is not str:
            _fail(f"audited source matrix scene {scene!r} has no exact group_id")
        if record["group_id"] != expected_group:
            _fail(f"audited source matrix scene {scene!r} group_id differs")
    if tuple(artifact.frames) != tuple(EXACT_EIGHT_TICK_RANGES):
        _fail("audited source frames do not preserve exact-eight order")
    if sum(len(frame) for frame in artifact.frames.values()) != EXACT_EIGHT_TOTAL_ROWS:
        _fail("audited source frames do not total 64,299 rows")
    if not artifact.manifest_path.is_file() or artifact.manifest_path.is_symlink():
        _fail("audited source matrix manifest path is invalid")
    if artifact.manifest_path.resolve() != (
        artifact.output_dir.resolve() / "matrix.manifest.json"
    ):
        _fail("audited source matrix manifest is outside its output directory")
    if manifest.get("output_directory") != str(artifact.output_dir.resolve()):
        _fail("audited source matrix output-directory binding differs")
    return artifact


def _audit_source(source_matrix: str | Path) -> RealGazePosthocMatrixArtifact:
    return _validate_source_artifact(audit_real_posthoc_matrix(source_matrix))


def _scene_to_group(
    artifact: RealGazePosthocMatrixArtifact,
) -> dict[str, str]:
    scene_records = artifact.manifest["scenes"]
    if not isinstance(scene_records, dict):
        _fail("audited source matrix scene records are invalid")
    return {
        scene: scene_records[scene]["group_id"]
        for scene in EXACT_EIGHT_TICK_RANGES
    }


def _source_binding(
    artifact: RealGazePosthocMatrixArtifact,
) -> dict[str, object]:
    return {
        "output_directory": str(artifact.output_dir.resolve()),
        "matrix_manifest": _file_binding(artifact.manifest_path.resolve()),
        "canonical_manifest_sha256": hashlib.sha256(
            _canonical_json(artifact.manifest).encode("utf-8")
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
    source: RealGazePosthocMatrixArtifact,
    *,
    initial_state_sha256: str | None = None,
) -> dict[str, object]:
    backend_spec = GazeBackendSpec()
    initial_state = initial_state_sha256 or initial_model_state_sha256(
        backend_spec, SEED
    )
    if not _is_sha256(initial_state):
        _fail("protocol initial-state SHA-256 is invalid")
    return {
        "artifact_kind": "vap_gaze_posthoc_outer_protocol",
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "result_role": "posthoc_exploratory_only",
        "source_matrix": _source_binding(source),
        "source_contract": {
            "permitted_use": REAL_PERMITTED_USE,
            "formal_outer_runner_admitted": False,
            "upstream_exact_regeneration_available": False,
            "implementation_counterfactual_causality_verified": False,
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
            gaze_posthoc_feature_set_metadata(condition) for condition in CONDITIONS
        ],
        "schema_dimensions": {
            "feature_schema": "multiparty_v3_gaze_validity",
            "raw_input_columns": 36,
            "transformed_input_dimensions": 37,
            "participant_features_per_token": 8,
            "global_features_per_token": 5,
            "token_input_dimensions": 13,
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
            "implementation": "GazePosthocTorchBackend",
            "backend_spec": backend_spec.to_dict(),
            "model_config": gaze_model_config(backend_spec),
            "trainer_spec": gaze_trainer_spec(backend_spec),
            "label_config": gaze_label_config(),
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
        "primary_comparison": "CG_minus_CM_gaze_content_given_observability",
        "secondary_comparisons": [
            "CM_minus_C0_observability_or_missingness",
            "CG_minus_C0_total_gaze_increment",
        ],
        **_boundary(),
    }


def freeze_gaze_outer_protocol(
    source_matrix: str | Path,
    destination: str | Path,
) -> GazeOuterProtocolArtifact:
    """Freeze the exact C0/CM/CG × four-group protocol once."""

    source = _audit_source(source_matrix)
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite Gaze protocol: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}-stage-",
            dir=destination_path.parent,
        )
    ).resolve()
    try:
        protocol = _protocol_payload(source)
        protocol_path = stage / PROTOCOL_FILENAME
        _write_exclusive(protocol_path, _json_bytes(protocol))
        binding = {
            "artifact_kind": "vap_gaze_posthoc_outer_protocol_binding",
            "format_version": FORMAT_VERSION,
            "protocol": _file_binding(protocol_path, relative_to=stage),
            **_boundary(),
        }
        _write_exclusive(stage / PROTOCOL_BINDING_FILENAME, _json_bytes(binding))
        if destination_path.exists():
            raise FileExistsError(
                f"refusing to overwrite Gaze protocol: {destination_path}"
            )
        stage.rename(destination_path)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return _load_protocol(destination_path, source)


def _load_protocol(
    protocol_directory: str | Path,
    source: RealGazePosthocMatrixArtifact,
    *,
    verify_initial_state: bool = True,
) -> GazeOuterProtocolArtifact:
    directory = Path(protocol_directory).resolve()
    if not directory.is_dir() or directory.is_symlink():
        _fail(f"Gaze protocol directory is missing or invalid: {directory}")
    if {path.name for path in directory.iterdir()} != {
        PROTOCOL_FILENAME,
        PROTOCOL_BINDING_FILENAME,
    }:
        _fail("Gaze protocol directory file set is not exact")
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
        != "vap_gaze_posthoc_outer_protocol_binding"
        or binding.get("format_version") != FORMAT_VERSION
    ):
        _fail("Gaze protocol binding kind/version is invalid")
    _validate_boundary(binding)
    _validate_relative_binding(binding.get("protocol"), directory, PROTOCOL_FILENAME)
    backend = protocol.get("backend")
    strategy = backend.get("initial_state_strategy") if isinstance(backend, dict) else None
    declared_initial_state = strategy.get("sha256") if isinstance(strategy, dict) else None
    if not _is_sha256(declared_initial_state):
        _fail("Gaze protocol has no valid initial-state SHA-256")
    if verify_initial_state and declared_initial_state != initial_model_state_sha256(
        GazeBackendSpec(), SEED
    ):
        _fail("Gaze protocol initial-state SHA-256 cannot be replayed")
    expected = _protocol_payload(
        source,
        initial_state_sha256=declared_initial_state,
    )
    if _canonical_json(protocol) != _canonical_json(expected):
        _fail("Gaze protocol failed exact frozen replay")
    _validate_boundary(protocol)
    protocol_sha256 = _sha256_file(protocol_path)
    return GazeOuterProtocolArtifact(
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
    protocol: GazeOuterProtocolArtifact,
    source: RealGazePosthocMatrixArtifact,
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
        "source_matrix": _source_binding(source),
        "protocol": _file_binding(protocol.protocol_path),
        "conditions": list(CONDITIONS),
        "outer_fold_ids": list(OUTER_FOLD_IDS),
        "planned_units": 12,
        "completed_units": 12,
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


def _run_gaze_outer_matrix_with_runner(
    protocol: str | Path,
    source_matrix: str | Path,
    output: str | Path,
    *,
    runner: _UnitRunner,
    fixture_only: bool,
) -> GazeOuterMatrixArtifact:
    if not fixture_only and type(runner) is not GazePosthocTorchBackend:
        _fail("formal Gaze matrix requires the exact frozen Torch backend")
    source = _audit_source(source_matrix)
    frozen = _load_protocol(protocol, source)
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Gaze outer matrix: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-stage-", dir=destination.parent)
    ).resolve()
    try:
        for condition in CONDITIONS:
            sessions = build_gaze_condition_sessions(
                source.frames,
                condition,
                scene_to_group=_scene_to_group(source),
            )
            for fold_id in OUTER_FOLD_IDS:
                fold = _fold_by_id(frozen.protocol, fold_id)
                request = GazeUnitRequest(
                    condition_id=condition,
                    fold_id=fold_id,
                    outer_test_group=str(fold["outer_test_group"]),
                    outer_train_groups=tuple(str(value) for value in fold["outer_train_groups"]),
                    inner_validation_groups=tuple(
                        str(value) for value in fold["outer_train_groups"]
                    ),
                    sessions=sessions,
                    output_directory=stage / condition / fold_id,
                    source_matrix_manifest_sha256=_sha256_file(source.manifest_path),
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
        audited_stage = _audit_matrix(
            stage,
            source,
            frozen,
            fixture_only=fixture_only,
            load_checkpoints=True,
        )
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite Gaze outer matrix: {destination}"
            )
        stage.rename(destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return GazeOuterMatrixArtifact(
        directory=destination,
        index_path=destination / MATRIX_INDEX_FILENAME,
        index=audited_stage.index,
        units=audited_stage.units,
    )


def run_gaze_outer_matrix(
    protocol: str | Path,
    source_matrix: str | Path,
    output: str | Path,
) -> GazeOuterMatrixArtifact:
    """Run the formal 12-unit matrix with the one frozen production backend."""

    runner = GazePosthocTorchBackend(GazeBackendSpec())
    return _run_gaze_outer_matrix_with_runner(
        protocol,
        source_matrix,
        output,
        runner=runner,
        fixture_only=False,
    )


def _run_gaze_outer_fixture_matrix(
    protocol: str | Path,
    source_matrix: str | Path,
    output: str | Path,
    *,
    fixture_runner: _UnitRunner,
) -> GazeOuterMatrixArtifact:
    """Test-only harness whose artifacts are permanently rejected as formal."""

    return _run_gaze_outer_matrix_with_runner(
        protocol,
        source_matrix,
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
    source: RealGazePosthocMatrixArtifact,
    protocol: GazeOuterProtocolArtifact | None = None,
    *,
    fixture_only: bool,
    load_checkpoints: bool,
) -> GazeOuterMatrixArtifact:
    root = Path(matrix_directory).resolve()
    if not root.is_dir() or root.is_symlink():
        _fail(f"Gaze matrix directory is missing or invalid: {root}")
    if {path.name for path in root.iterdir()} != {
        *CONDITIONS,
        MATRIX_INDEX_FILENAME,
    }:
        _fail("Gaze matrix root file set is not exact")
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
        _fail("Gaze matrix index kind/version/protocol is invalid")
    _validate_boundary(index)
    if _canonical_json(index.get("source_matrix")) != _canonical_json(
        _source_binding(source)
    ):
        _fail("Gaze matrix source binding differs from the audited source")
    protocol_binding = index.get("protocol")
    if not isinstance(protocol_binding, dict):
        _fail("Gaze matrix protocol binding is invalid")
    protocol_path_value = protocol_binding.get("path")
    if not isinstance(protocol_path_value, str):
        _fail("Gaze matrix protocol path is invalid")
    bound_protocol_path = Path(protocol_path_value).resolve()
    if not bound_protocol_path.is_file() or bound_protocol_path.is_symlink():
        _fail("Gaze matrix bound protocol is unavailable")
    if (
        protocol_binding.get("sha256") != _sha256_file(bound_protocol_path)
        or protocol_binding.get("size_bytes") != bound_protocol_path.stat().st_size
    ):
        _fail("Gaze matrix protocol binding differs")
    if protocol is None:
        protocol = _load_protocol(
            bound_protocol_path.parent,
            source,
            verify_initial_state=load_checkpoints,
        )
    if protocol.protocol_path != bound_protocol_path:
        _fail("Gaze matrix protocol path differs from loaded authority")
    if index.get("conditions") != list(CONDITIONS):
        _fail("Gaze matrix condition inventory differs")
    if index.get("outer_fold_ids") != list(OUTER_FOLD_IDS):
        _fail("Gaze matrix fold inventory differs")
    if index.get("planned_units") != 12 or index.get("completed_units") != 12:
        _fail("Gaze matrix is not a complete 12-unit run")
    expected_index = _matrix_index(
        root,
        protocol,
        source,
        fixture_only=fixture_only,
    )
    if _canonical_json(index) != _canonical_json(expected_index):
        _fail("Gaze matrix index failed exact hash-bound replay")
    records = index.get("units")
    if not isinstance(records, list) or len(records) != 12:
        _fail("Gaze matrix unit records are invalid")
    expected_pairs = [
        (condition, fold_id)
        for condition in CONDITIONS
        for fold_id in OUTER_FOLD_IDS
    ]
    units: dict[tuple[str, str], LoadedGazeUnit] = {}
    sessions_by_condition = {
        condition: build_gaze_condition_sessions(
            source.frames,
            condition,
            scene_to_group=_scene_to_group(source),
        )
        for condition in CONDITIONS
    }
    for record, (condition, fold_id) in zip(records, expected_pairs, strict=True):
        if not isinstance(record, dict):
            _fail("Gaze matrix unit record is not an object")
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
                _fail(f"Gaze matrix unit field {key!r} differs")
        _validate_index_artifact_bindings(root, record, condition, fold_id)
        sessions = tuple(
            session
            for session in sessions_by_condition[condition]
            if session.group_id == fold["outer_test_group"]
        )
        expected_initial_state = _fold_initial_state(fold, condition)
        loaded = load_saved_gaze_unit(
            root / condition / fold_id,
            sessions,
            condition_id=condition,
            fold_id=fold_id,
            outer_test_group=str(fold["outer_test_group"]),
            source_matrix_manifest_sha256=_sha256_file(source.manifest_path),
            protocol_sha256=protocol.protocol_sha256,
            expected_initial_state_sha256=expected_initial_state,
            fixture_only=fixture_only,
        )
        if load_checkpoints:
            checkpoint = load_gaze_checkpoint_artifact(
                root / condition / fold_id / CHECKPOINT_FILENAME,
                condition_id=condition,
                fold_id=fold_id,
                outer_test_group=str(fold["outer_test_group"]),
                source_matrix_manifest_sha256=_sha256_file(source.manifest_path),
                protocol_sha256=protocol.protocol_sha256,
                expected_initial_state_sha256=expected_initial_state,
                fixture_only=fixture_only,
            )
            payload = checkpoint["payload"]
            if not isinstance(payload, dict):
                _fail("Gaze checkpoint core payload is invalid after loading")
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
        units[(condition, fold_id)] = loaded
    for fold_id in OUTER_FOLD_IDS:
        initial_hashes = {
            units[(condition, fold_id)].result["initial_state_sha256"]
            for condition in CONDITIONS
        }
        if len(initial_hashes) != 1:
            _fail(f"paired conditions do not share one initial state in {fold_id}")
    return GazeOuterMatrixArtifact(
        directory=root,
        index_path=index_path,
        index=index,
        units=units,
    )


def audit_gaze_outer_matrix(
    matrix_directory: str | Path,
    source_matrix: str | Path,
) -> GazeOuterMatrixArtifact:
    """Replay the source authority and audit all 12 saved-probability units."""

    source = _audit_source(source_matrix)
    return _audit_matrix(
        matrix_directory,
        source,
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


def aggregate_gaze_outer_matrix(
    matrix: str | Path,
    source_matrix: str | Path,
    output: str | Path,
) -> Path:
    """Pool audited saved probabilities; never load a checkpoint or infer."""

    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".json":
        _fail("Gaze aggregate output must be a .json file")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Gaze aggregate: {output_path}")
    source = _audit_source(source_matrix)
    audited = _audit_matrix(
        matrix,
        source,
        fixture_only=False,
        load_checkpoints=False,
    )
    metrics: dict[str, dict[str, object]] = {}
    condition_sources: dict[str, list[dict[str, object]]] = {}
    for condition in CONDITIONS:
        sessions = build_gaze_condition_sessions(
            source.frames,
            condition,
            scene_to_group=_scene_to_group(source),
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
        metrics[condition] = gaze_metrics_from_predictions(
            pooled_sessions, pooled_predictions
        )
        condition_sources[condition] = unit_sources
    payload = {
        "artifact_kind": AGGREGATE_KIND,
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "result_role": "posthoc_exploratory_saved_probability_aggregate",
        "aggregation_method": "pooled_out_of_group_saved_probabilities_no_inference",
        "source_matrix": _source_binding(source),
        "matrix_index": _file_binding(audited.index_path),
        "exact_scene_inventory": list(EXACT_EIGHT_TICK_RANGES),
        "total_rows": EXACT_EIGHT_TOTAL_ROWS,
        "group_ids": list(EXACT_EIGHT_GROUPS),
        "conditions": list(CONDITIONS),
        "condition_sources": condition_sources,
        "metrics": metrics,
        "comparisons": {
            "primary_CG_minus_CM": _comparison(metrics, "CG", "CM"),
            "secondary_CM_minus_C0": _comparison(metrics, "CM", "C0"),
            "secondary_CG_minus_C0": _comparison(metrics, "CG", "C0"),
        },
        "interpretation_boundary": (
            "single-seed post-hoc exploratory evidence; cannot alone establish H1, "
            "causal deployment equivalence, or statistical significance"
        ),
        **_boundary(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive(output_path, _json_bytes(payload))
    return output_path


__all__ = [
    "AGGREGATE_KIND",
    "AUX_VAD_WEIGHT",
    "CONDITIONS",
    "FORMAT_VERSION",
    "GazeOuterExperimentError",
    "GazeOuterMatrixArtifact",
    "GazeOuterProtocolArtifact",
    "MATRIX_INDEX_FILENAME",
    "MAX_EPOCHS",
    "OUTER_FOLD_IDS",
    "PROTOCOL_ID",
    "SEED",
    "aggregate_gaze_outer_matrix",
    "audit_gaze_outer_matrix",
    "freeze_gaze_outer_protocol",
    "run_gaze_outer_matrix",
]
