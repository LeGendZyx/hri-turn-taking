"""Isolated saved-probability residual late-fusion diagnostic.

This module never trains or loads a model.  It consumes the immutable seed-13,
15-epoch A and H nested-LOGO saved probabilities, selects one residual audio
weight from the three paired inner OOF groups, and applies that already-selected
weight once to the held-out outer probabilities.

The diagnostic is deliberately post-hoc, non-production and non-counted.  It is
not an implementation of a trainable gated-fusion architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hri_turn_taking.vap import outer_experiment, thesis_backend

FORMAT_VERSION = 1
RESULT_ROLE = "posthoc_saved_probability_residual_late_fusion_diagnostic"
PROTOCOL_KIND = "vap_residual_late_fusion_protocol"
FOLD_KIND = "vap_residual_late_fusion_fold"
AGGREGATE_KIND = "vap_residual_late_fusion_aggregate"
DATA_KIND = "real"
PRODUCTION_ELIGIBLE = False
COUNTED_ELIGIBLE = False
ANALYSIS_TIMING = "post_outcome_exploratory"
COUNTED_BLOCKERS = (
    "posthoc_after_outer_outcomes_were_available",
    "saved_probability_diagnostic_not_trainable_architecture",
    "single_seed_13_only",
    "d3_sensitivity_not_executed",
)
SOURCE_CONDITIONS = ("A", "H")
MATRIX_CONDITIONS = ("A", "H", "A+H")
CONDITION_DIRECTORIES = {"A": "A", "H": "H", "A+H": "A_plus_H"}
OUTER_FOLD_IDS = ("outer_01", "outer_02", "outer_03", "outer_04")
INNER_FOLD_IDS = ("inner_01", "inner_02", "inner_03")
EXACT_EIGHT_SESSION_IDS = (
    "03_20__4",
    "03_26__3",
    "05_14__2",
    "05_14__3",
    "05_14__4",
    "05_15__2",
    "05_15__3",
    "05_15__6",
)
EXACT_FOUR_GROUP_IDS = ("03_20", "03_26", "05_14", "05_15")
EXACT_CORPUS_EXCLUSIONS = {
    "05_15__4": "no A speech in Transcript; pending owner manual review",
    "05_15__5": "A/R extremely sparse; pending owner manual review",
}
SOURCE_RUN_MANIFEST_KIND = "vap_counted_outer_run"
SOURCE_RUN_MANIFEST_VERSION = 3
SOURCE_PROTOCOL_ID = "four_group_nested_logo_deadline_8_v1"
SOURCE_MAX_EPOCHS = 15
SOURCE_SEEDS = (13, 42, 73)
SOURCE_RUN_MANIFEST_SHA256 = "a82c31bd6e4e052462dd0452ccd65bf11e92313871edd74b517360e5d40f1e74"
SOURCE_SPLIT_MANIFEST_SHA256 = "d5fb3f4e233cfbfdff5e9aa8a8af60dfb9b1f55b5f503a0755b68852bdc9c311"
LAMBDA_GRID = (0.0, 0.125, 0.25, 0.5, 1.0)
A_SYMMETRY_TOLERANCE = 1e-7
SELECTION_METRIC = "group_equal_participant_macro_strict_mAP"
FORMULA = "sigmoid(logit(H_probability)+lambda*logit(A_probability))"
TIE_RULE = "highest_inner_group_equal_score_then_smallest_lambda"
DEFAULT_SOURCE = Path("outputs/thesis_posthoc_15epoch_20260817/matrix_seed13_v1")
DEFAULT_OUTPUT = Path("outputs/residual_late_fusion_posthoc_20260818")
PROTOCOL_FILENAME = "protocol.json"
FOLD_MANIFEST_FILENAME = "fold_manifest.json"
FUSED_PROBABILITIES_FILENAME = "fused_probabilities.npz"
AGGREGATE_FILENAME = "aggregate.json"

SUCCESS_CRITERIA = {
    "participant_macro_mAP_selected_minus_lambda0_min": 0.02,
    "strictly_positive_outer_fold_count_min": 3,
    "shift_prediction_AP_selected_minus_lambda0_min": -0.02,
    "overlap_resolution_AP_selected_minus_lambda0_min": -0.02,
    "selection_independent_of_success_criteria": True,
}


class ResidualLateFusionError(ValueError):
    """A source, pairing, publication, or replay-audit gate failed."""


@dataclass(frozen=True)
class FileBinding:
    relative_path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class SourceUnit:
    condition_id: str
    outer_fold_id: str
    directory: Path
    result_manifest_path: Path
    result_manifest_binding: FileBinding
    prediction_bindings: dict[str, FileBinding]


@dataclass(frozen=True)
class SourceFileSnapshot:
    """One fully validated immutable dependency retained for cheap replay checks."""

    snapshot: object
    require_single_link: bool


@dataclass(frozen=True)
class SourceMatrix:
    root: Path
    index_path: Path
    index_binding: FileBinding
    index: dict[str, object]
    protocol_root: Path
    run_manifest_path: Path
    run_manifest_binding: FileBinding
    run_manifest: dict[str, object]
    units: dict[tuple[str, str], SourceUnit]
    result_manifest_bindings: dict[tuple[str, str], FileBinding]
    source_file_snapshots: tuple[SourceFileSnapshot, ...]


@dataclass(frozen=True)
class PairedPredictions:
    a_payload: dict[str, object]
    h_payload: dict[str, object]
    a_path: Path
    h_path: Path
    a_binding: FileBinding
    h_binding: FileBinding
    a_symmetry_max_abs_diff: float


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ResidualLateFusionError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ResidualLateFusionError(f"non-finite JSON constant is forbidden: {value}")


def _read_json_strict(path_value: str | Path) -> dict[str, object]:
    path = Path(path_value).resolve()
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResidualLateFusionError(f"cannot read strict JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ResidualLateFusionError(f"JSON artifact root must be an object: {path}")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path_value: str | Path) -> str:
    path = Path(path_value).resolve()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ResidualLateFusionError(f"cannot hash artifact: {path}") from exc
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _binding(path_value: str | Path, root: Path) -> FileBinding:
    path = Path(path_value).resolve()
    root = root.resolve()
    try:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
    except (OSError, ValueError) as exc:
        raise ResidualLateFusionError(f"artifact is not a regular child of {root}: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise ResidualLateFusionError(f"artifact must be a regular non-symlink file: {path}")
    return FileBinding(relative, _sha256_file(path), size)


def _verify_binding(binding: FileBinding, root: Path) -> Path:
    path = (root.resolve() / binding.relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ResidualLateFusionError("artifact binding escapes its declared root") from exc
    observed = _binding(path, root)
    if observed != binding:
        raise ResidualLateFusionError(f"bound artifact bytes changed: {path}")
    return path


def _directory_identity(path_value: str | Path) -> tuple[int, int]:
    """Return a rename-stable identity for one real directory."""

    path = Path(path_value)
    try:
        stat = path.stat()
    except OSError as exc:
        raise ResidualLateFusionError(f"cannot identify publication directory: {path}") from exc
    if path.is_symlink() or not path.is_dir() or stat.st_ino == 0:
        raise ResidualLateFusionError(f"publication directory has no safe identity: {path}")
    return int(stat.st_dev), int(stat.st_ino)


def _binding_from_dict(value: object) -> FileBinding:
    if not isinstance(value, dict) or set(value) != {
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        raise ResidualLateFusionError("file binding must use exact version-1 fields")
    relative = value["relative_path"]
    digest = value["sha256"]
    size = value["size_bytes"]
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ResidualLateFusionError("file binding relative_path is invalid")
    if not _is_sha256(digest):
        raise ResidualLateFusionError("file binding SHA-256 is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ResidualLateFusionError("file binding size is invalid")
    return FileBinding(relative, digest, size)


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise
    except OSError as exc:
        raise ResidualLateFusionError(f"cannot publish artifact: {path}") from exc


def _write_npz_exclusive(path: Path, probabilities: np.ndarray, offsets: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            np.savez_compressed(handle, probabilities=probabilities, session_offsets=offsets)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise
    except OSError as exc:
        raise ResidualLateFusionError(f"cannot publish probability artifact: {path}") from exc


def _validate_historical_unit(path: Path, *, prepared: object) -> object:
    """Run the existing read-only source/result validator without changing it."""

    return outer_experiment._validate_result_directory(path.resolve(), prepared=prepared)


def _prepare_historical_conditions(
    protocol_root: Path,
    run_manifest: dict[str, object],
) -> dict[str, object]:
    """Replay each expensive condition source exactly once for its four folds."""

    source_value = run_manifest.get("source")
    if not isinstance(source_value, str):
        raise ResidualLateFusionError("source run manifest has no canonical source path")
    source_path = Path(source_value)
    if not source_path.is_absolute() or source_path != source_path.resolve():
        raise ResidualLateFusionError("source run manifest source path is not canonical")
    prepared: dict[str, object] = {}
    try:
        for condition in MATRIX_CONDITIONS:
            prepared[condition] = outer_experiment.prepare_outer_condition(
                protocol_root,
                source=source_path,
                condition_id=condition,
            )
    except (OSError, TypeError, ValueError, outer_experiment.OuterExperimentGateError) as exc:
        raise ResidualLateFusionError("cannot prepare the frozen historical source matrix") from exc
    return prepared


def _capture_source_file_snapshot(
    path: Path,
    *,
    require_single_link: bool,
) -> object:
    try:
        return outer_experiment._capture_file_snapshot(
            path.resolve(),
            require_single_link=require_single_link,
        )
    except (OSError, outer_experiment.OuterExperimentGateError) as exc:
        raise ResidualLateFusionError(
            f"cannot snapshot validated source dependency: {path}"
        ) from exc


def _add_source_file_snapshot(
    records: dict[Path, SourceFileSnapshot],
    snapshot: object,
    *,
    require_single_link: bool,
) -> None:
    path_value = getattr(snapshot, "path", None)
    if not isinstance(path_value, Path) or not path_value.is_absolute():
        raise ResidualLateFusionError("historical validator returned an invalid file snapshot")
    path = path_value.resolve()
    existing = records.get(path)
    if existing is not None and existing.snapshot != snapshot:
        raise ResidualLateFusionError(f"source dependency changed during validation: {path}")
    records[path] = SourceFileSnapshot(
        snapshot=snapshot,
        require_single_link=(
            require_single_link or (existing.require_single_link if existing else False)
        ),
    )


def _add_historical_validation_snapshots(
    records: dict[Path, SourceFileSnapshot],
    validation_snapshot: object,
) -> None:
    """Merge one result validator's source/result/role-artifact snapshot set."""

    if validation_snapshot is None:  # Narrow fixture seam; production validator never returns None.
        return
    immutable_files = getattr(validation_snapshot, "immutable_files", None)
    role_artifacts = getattr(validation_snapshot, "role_artifacts", None)
    if not isinstance(immutable_files, tuple) or not isinstance(role_artifacts, tuple):
        raise ResidualLateFusionError("historical validator returned no immutable snapshot set")
    for snapshot in immutable_files:
        _add_source_file_snapshot(records, snapshot, require_single_link=False)
    for snapshot in role_artifacts:
        _add_source_file_snapshot(records, snapshot, require_single_link=True)


def _revalidate_source_file_snapshots(snapshots: tuple[SourceFileSnapshot, ...]) -> None:
    """Freshly hash every unique dependency once and reject concurrent mutation."""

    try:
        for record in snapshots:
            outer_experiment._assert_file_snapshot_unchanged(
                record.snapshot,
                require_single_link=record.require_single_link,
            )
    except (OSError, outer_experiment.OuterExperimentGateError) as exc:
        raise ResidualLateFusionError("validated source dependency snapshot changed") from exc


def _revalidate_source_matrix(source: SourceMatrix) -> None:
    _revalidate_source_file_snapshots(source.source_file_snapshots)


def _load_canonical_source_run_manifest(
    index: dict[str, object],
) -> tuple[Path, Path, FileBinding, dict[str, object]]:
    """Bind the source matrix to the exact immutable 15-epoch protocol."""

    declared_root = index.get("protocol_directory")
    if not isinstance(declared_root, str) or not Path(declared_root).is_absolute():
        raise ResidualLateFusionError("source protocol_directory must be absolute")
    protocol_root = Path(declared_root).resolve()
    if (
        Path(declared_root) != protocol_root
        or not protocol_root.is_dir()
        or protocol_root.is_symlink()
    ):
        raise ResidualLateFusionError("source protocol_directory is not canonical")
    run_manifest_path = protocol_root / "run_manifest.json"
    if run_manifest_path.is_symlink() or not run_manifest_path.is_file():
        raise ResidualLateFusionError("canonical source run_manifest.json is missing")
    run_manifest = _read_json_strict(run_manifest_path)
    observed_hash = _canonical_json_sha256(run_manifest)
    declared_hash = index.get("run_manifest_sha256")
    if declared_hash != SOURCE_RUN_MANIFEST_SHA256 or observed_hash != SOURCE_RUN_MANIFEST_SHA256:
        raise ResidualLateFusionError("source matrix is not bound to the frozen 15-epoch run")
    exact_values = {
        "artifact_kind": SOURCE_RUN_MANIFEST_KIND,
        "format_version": SOURCE_RUN_MANIFEST_VERSION,
        "protocol_id": SOURCE_PROTOCOL_ID,
        "result_role": "protocol_only_not_a_result",
        "counted_experiment_complete": False,
        "split_manifest_sha256": SOURCE_SPLIT_MANIFEST_SHA256,
        "seeds": list(SOURCE_SEEDS),
        "condition_ids": list(MATRIX_CONDITIONS),
        "max_epochs": SOURCE_MAX_EPOCHS,
        "aux_vad_weight": 0.0,
    }
    for field, expected in exact_values.items():
        observed = run_manifest.get(field)
        if observed != expected or type(observed) is not type(expected):
            raise ResidualLateFusionError(
                f"source run manifest differs from the frozen 15-epoch contract at {field}"
            )
    return (
        protocol_root,
        run_manifest_path,
        _binding(run_manifest_path, protocol_root),
        run_manifest,
    )


def _artifact_binding_from_result(
    result: dict[str, object],
    *,
    role: str,
    inner_fold_id: str | None,
) -> FileBinding:
    artifacts = result.get("artifact_bindings")
    if not isinstance(artifacts, dict):
        raise ResidualLateFusionError("source result has no artifact_bindings object")
    if inner_fold_id is None:
        value = artifacts.get(role)
        if not isinstance(value, dict):
            raise ResidualLateFusionError(f"source result has no {role} binding")
    else:
        values = artifacts.get(role)
        if not isinstance(values, list):
            raise ResidualLateFusionError(f"source result has no {role} list")
        matches = [
            item
            for item in values
            if isinstance(item, dict) and item.get("inner_fold_id") == inner_fold_id
        ]
        if len(matches) != 1:
            raise ResidualLateFusionError(
                f"source result must bind {inner_fold_id} exactly once for {role}"
            )
        value = matches[0]
    expected = {"relative_path", "sha256", "size_bytes"}
    if inner_fold_id is not None:
        expected.add("inner_fold_id")
    if set(value) != expected:
        raise ResidualLateFusionError("source prediction binding fields are not exact")
    return _binding_from_dict(
        {
            "relative_path": value["relative_path"],
            "sha256": value["sha256"],
            "size_bytes": value["size_bytes"],
        }
    )


def _load_source_matrix(source_root_value: str | Path) -> SourceMatrix:
    source_root = Path(source_root_value).resolve()
    index_path = source_root / "matrix_index.json"
    source_file_snapshots: dict[Path, SourceFileSnapshot] = {}
    _add_source_file_snapshot(
        source_file_snapshots,
        _capture_source_file_snapshot(index_path, require_single_link=False),
        require_single_link=False,
    )
    index = _read_json_strict(index_path)
    required_values = {
        "artifact_kind": "vap_thesis_primary_matrix_index",
        "format_version": 1,
        "result_role": thesis_backend.RESULT_ROLE,
        "data_kind": "real",
        "production_eligible": False,
        "counted_eligible": False,
        "analysis_variant": "primary",
        "seed": 13,
        "condition_ids": list(MATRIX_CONDITIONS),
        "outer_fold_ids": list(OUTER_FOLD_IDS),
        "expected_unit_count": 12,
        "completed_unit_count": 12,
    }
    for field, expected in required_values.items():
        if index.get(field) != expected or type(index.get(field)) is not type(expected):
            raise ResidualLateFusionError(f"source matrix index differs at {field}")
    (
        protocol_root,
        run_manifest_path,
        run_manifest_binding,
        run_manifest,
    ) = _load_canonical_source_run_manifest(index)
    _add_source_file_snapshot(
        source_file_snapshots,
        _capture_source_file_snapshot(run_manifest_path, require_single_link=False),
        require_single_link=False,
    )
    prepared_conditions = _prepare_historical_conditions(protocol_root, run_manifest)
    included = index.get("included_session_ids")
    groups = index.get("group_ids")
    if (
        included != list(EXACT_EIGHT_SESSION_IDS)
        or type(included) is not list
        or groups != list(EXACT_FOUR_GROUP_IDS)
        or type(groups) is not list
        or index.get("corpus_exclusions") != EXACT_CORPUS_EXCLUSIONS
        or type(index.get("corpus_exclusions")) is not dict
    ):
        raise ResidualLateFusionError("source matrix exact-eight identity is invalid")
    units_value = index.get("units")
    if not isinstance(units_value, list) or len(units_value) != 12:
        raise ResidualLateFusionError("source matrix index must contain exactly 12 units")
    keyed: dict[tuple[str, str], dict[str, object]] = {}
    for item in units_value:
        if not isinstance(item, dict):
            raise ResidualLateFusionError("source matrix unit must be an object")
        condition = item.get("condition_id")
        fold = item.get("outer_fold_id")
        if condition not in MATRIX_CONDITIONS or fold not in OUTER_FOLD_IDS:
            raise ResidualLateFusionError("source matrix unit identity is invalid")
        key = (condition, fold)
        if key in keyed:
            raise ResidualLateFusionError("source matrix unit identity is duplicated")
        if item.get("seed") != 13 or isinstance(item.get("seed"), bool):
            raise ResidualLateFusionError("source matrix unit seed differs from 13")
        keyed[key] = item
    expected_keys = {
        (condition, fold) for condition in MATRIX_CONDITIONS for fold in OUTER_FOLD_IDS
    }
    if set(keyed) != expected_keys:
        raise ResidualLateFusionError("source matrix does not contain the exact 12-unit grid")

    units: dict[tuple[str, str], SourceUnit] = {}
    result_manifest_bindings: dict[tuple[str, str], FileBinding] = {}
    for condition in MATRIX_CONDITIONS:
        for fold in OUTER_FOLD_IDS:
            item = keyed[(condition, fold)]
            directory = (source_root / CONDITION_DIRECTORIES[condition] / fold).resolve()
            result_path = directory / "result_manifest.json"
            declared_path = item.get("result_manifest")
            if not isinstance(declared_path, str) or Path(declared_path).resolve() != result_path:
                raise ResidualLateFusionError("matrix index result_manifest path is not canonical")
            declared_hash = item.get("result_manifest_sha256")
            if not _is_sha256(declared_hash) or _sha256_file(result_path) != declared_hash:
                raise ResidualLateFusionError("matrix index result-manifest hash changed")
            validation_snapshot = _validate_historical_unit(
                directory,
                prepared=prepared_conditions[condition],
            )
            _add_historical_validation_snapshots(source_file_snapshots, validation_snapshot)
            result = _read_json_strict(result_path)
            if (
                result.get("condition_id") != condition
                or result.get("outer_fold_id") != fold
                or result.get("seed") != 13
                or isinstance(result.get("seed"), bool)
                or result.get("run_manifest_sha256") != index["run_manifest_sha256"]
            ):
                raise ResidualLateFusionError("source result identity differs from matrix index")
            result_binding = _binding(result_path, source_root)
            result_manifest_bindings[(condition, fold)] = result_binding
            _add_source_file_snapshot(
                source_file_snapshots,
                _capture_source_file_snapshot(result_path, require_single_link=False),
                require_single_link=False,
            )
            if condition not in SOURCE_CONDITIONS:
                continue
            prediction_bindings = {
                inner: _artifact_binding_from_result(
                    result,
                    role="inner_oof_predictions",
                    inner_fold_id=inner,
                )
                for inner in INNER_FOLD_IDS
            }
            prediction_bindings["outer"] = _artifact_binding_from_result(
                result,
                role="outer_predictions",
                inner_fold_id=None,
            )
            for binding in prediction_bindings.values():
                prediction_path = _verify_binding(binding, directory)
                _add_source_file_snapshot(
                    source_file_snapshots,
                    _capture_source_file_snapshot(
                        prediction_path,
                        require_single_link=True,
                    ),
                    require_single_link=True,
                )
            units[(condition, fold)] = SourceUnit(
                condition_id=condition,
                outer_fold_id=fold,
                directory=directory,
                result_manifest_path=result_path,
                result_manifest_binding=result_binding,
                prediction_bindings=prediction_bindings,
            )
    ordered_source_snapshots = tuple(
        source_file_snapshots[path]
        for path in sorted(source_file_snapshots, key=lambda item: os.path.normcase(str(item)))
    )
    _revalidate_source_file_snapshots(ordered_source_snapshots)
    return SourceMatrix(
        root=source_root,
        index_path=index_path,
        index_binding=_binding(index_path, source_root),
        index=index,
        protocol_root=protocol_root,
        run_manifest_path=run_manifest_path,
        run_manifest_binding=run_manifest_binding,
        run_manifest=run_manifest,
        units=units,
        result_manifest_bindings=result_manifest_bindings,
        source_file_snapshots=ordered_source_snapshots,
    )


def _probability_arrays(payload: dict[str, object]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(record["probabilities"], dtype=np.float32) for record in payload["sessions"]
    )


def _participant_symmetry_max(payload: dict[str, object]) -> float:
    observed = 0.0
    for record, probabilities in zip(
        payload["sessions"], _probability_arrays(payload), strict=True
    ):
        present = np.asarray(record["present"], dtype=bool)
        if not present.any():
            raise ResidualLateFusionError("A source record has no present participant")
        spread = np.ptp(probabilities[:, present, :].astype(np.float64), axis=1)
        observed = max(observed, float(spread.max(initial=0.0)))
    return observed


def _exact_array_field(left: object, right: object, field: str, dtype: Any) -> None:
    try:
        left_array = np.asarray(left, dtype=dtype)
        right_array = np.asarray(right, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ResidualLateFusionError(f"paired {field} field is not numeric") from exc
    if left_array.shape != right_array.shape or not np.array_equal(left_array, right_array):
        raise ResidualLateFusionError(f"A/H paired {field} values differ")


def _validate_paired_payloads(
    a_payload: dict[str, object],
    h_payload: dict[str, object],
    *,
    outer_fold_id: str,
    inner_fold_id: str | None,
) -> float:
    if a_payload.get("condition_id") != "A" or h_payload.get("condition_id") != "H":
        raise ResidualLateFusionError("paired sources must be conditions A and H")
    expected_inner = inner_fold_id is not None
    for payload in (a_payload, h_payload):
        if payload.get("outer_fold_id") != outer_fold_id or payload.get("seed") != 13:
            raise ResidualLateFusionError("paired source fold/seed identity differs")
        if expected_inner and payload.get("inner_fold_id") != inner_fold_id:
            raise ResidualLateFusionError("paired inner OOF identity differs")
        if payload.get("run_manifest_sha256") != a_payload.get("run_manifest_sha256"):
            raise ResidualLateFusionError("paired sources use different run manifests")
        if payload.get("frame_rate_hz") != 25:
            raise ResidualLateFusionError("paired source frame rate differs from 25 Hz")
        if payload.get("label_config") != a_payload.get("label_config"):
            raise ResidualLateFusionError("paired source label configurations differ")
        if payload.get("threshold_policy") != a_payload.get("threshold_policy"):
            raise ResidualLateFusionError("paired source threshold policies differ")
    a_sessions = a_payload.get("sessions")
    h_sessions = h_payload.get("sessions")
    if not isinstance(a_sessions, list) or not isinstance(h_sessions, list):
        raise ResidualLateFusionError("paired source sessions are malformed")
    if len(a_sessions) != len(h_sessions) or not a_sessions:
        raise ResidualLateFusionError("paired A/H source session counts differ")
    for a_record, h_record in zip(a_sessions, h_sessions, strict=True):
        for field in ("session_id", "group_id", "n_frames", "tick_25hz"):
            if a_record[field] != h_record[field] or type(a_record[field]) is not type(
                h_record[field]
            ):
                raise ResidualLateFusionError(f"A/H paired {field} values differ")
        _exact_array_field(a_record["labels"], h_record["labels"], "labels", np.float32)
        _exact_array_field(a_record["vad"], h_record["vad"], "vad", np.float32)
        _exact_array_field(a_record["valid"], h_record["valid"], "valid", np.int8)
        _exact_array_field(a_record["present"], h_record["present"], "present", np.int8)
        _exact_array_field(
            np.asarray(a_record["tick_25hz"], dtype=np.int64),
            np.asarray(h_record["tick_25hz"], dtype=np.int64),
            "ticks",
            np.int64,
        )
    symmetry = _participant_symmetry_max(a_payload)
    if not math.isfinite(symmetry) or symmetry > A_SYMMETRY_TOLERANCE:
        raise ResidualLateFusionError(
            f"A participant symmetry exceeds the frozen 1e-7 tolerance: observed={symmetry:.17g}"
        )
    return symmetry


def _load_prediction(path: Path, inner_fold_id: str | None) -> dict[str, object]:
    try:
        if inner_fold_id is None:
            return thesis_backend.load_prediction_artifact(path)
        return thesis_backend._load_prediction_artifact(
            path,
            expected_inner_fold_id=inner_fold_id,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ResidualLateFusionError(f"invalid thesis saved-probability source: {path}") from exc


def _paired_predictions(
    source: SourceMatrix,
    outer_fold_id: str,
    inner_fold_id: str | None,
) -> PairedPredictions:
    key = inner_fold_id if inner_fold_id is not None else "outer"
    a_unit = source.units[("A", outer_fold_id)]
    h_unit = source.units[("H", outer_fold_id)]
    a_local_binding = a_unit.prediction_bindings[key]
    h_local_binding = h_unit.prediction_bindings[key]
    a_path = _verify_binding(a_local_binding, a_unit.directory)
    h_path = _verify_binding(h_local_binding, h_unit.directory)
    a_payload = _load_prediction(a_path, inner_fold_id)
    h_payload = _load_prediction(h_path, inner_fold_id)
    symmetry = _validate_paired_payloads(
        a_payload,
        h_payload,
        outer_fold_id=outer_fold_id,
        inner_fold_id=inner_fold_id,
    )
    return PairedPredictions(
        a_payload=a_payload,
        h_payload=h_payload,
        a_path=a_path,
        h_path=h_path,
        a_binding=_binding(a_path, source.root),
        h_binding=_binding(h_path, source.root),
        a_symmetry_max_abs_diff=symmetry,
    )


def fuse_probability_arrays(
    h_probability: np.ndarray,
    a_probability: np.ndarray,
    lambda_value: float,
) -> np.ndarray:
    """Apply the frozen residual logit formula to one paired probability array."""

    h = np.asarray(h_probability, dtype=np.float32)
    a = np.asarray(a_probability, dtype=np.float32)
    if h.shape != a.shape or h.ndim != 3 or h.shape[1:] != (4, 4):
        raise ResidualLateFusionError("fusion inputs must be paired [frames,4,4] arrays")
    if type(lambda_value) is not float or lambda_value not in LAMBDA_GRID:
        raise ResidualLateFusionError("lambda must be one frozen grid value")
    if not np.isfinite(h).all() or not np.isfinite(a).all():
        raise ResidualLateFusionError("fusion probabilities must be finite")
    if ((h <= 0.0) | (h >= 1.0) | (a <= 0.0) | (a >= 1.0)).any():
        raise ResidualLateFusionError("exact logit fusion requires probabilities strictly in (0,1)")
    if lambda_value == 0.0:
        control = h.copy()
        if not np.array_equal(control, h):
            raise ResidualLateFusionError("lambda=0 control is not an exact H copy")
        return control
    h64 = h.astype(np.float64)
    a64 = a.astype(np.float64)
    logits = np.log(h64) - np.log1p(-h64)
    logits += lambda_value * (np.log(a64) - np.log1p(-a64))
    fused = np.empty_like(logits)
    positive = logits >= 0.0
    fused[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_value = np.exp(logits[~positive])
    fused[~positive] = exp_value / (1.0 + exp_value)
    result = fused.astype(np.float32)
    if not np.isfinite(result).all() or ((result <= 0.0) | (result >= 1.0)).any():
        raise ResidualLateFusionError("fused probabilities are not finite values in (0,1)")
    return result


def _fuse_pair(pair: PairedPredictions, lambda_value: float) -> tuple[np.ndarray, ...]:
    a_arrays = _probability_arrays(pair.a_payload)
    h_arrays = _probability_arrays(pair.h_payload)
    return tuple(
        fuse_probability_arrays(h, a, lambda_value) for h, a in zip(h_arrays, a_arrays, strict=True)
    )


def _participant_macro_strict_map(
    h_payload: dict[str, object],
    probabilities: tuple[np.ndarray, ...],
) -> float:
    sessions = thesis_backend._sessions_from_prediction(h_payload)
    metrics = thesis_backend._frame_metrics(sessions, probabilities)
    value = metrics.get("participant_macro_mAP")
    if type(value) is not float or not math.isfinite(value):
        raise ResidualLateFusionError("strict participant-macro inner mAP is unavailable")
    return value


def _select_lambda(
    inner_pairs: tuple[PairedPredictions, ...],
) -> tuple[float, list[dict[str, object]]]:
    if len(inner_pairs) != 3:
        raise ResidualLateFusionError("lambda selection requires exactly three paired inner OOFs")
    group_ids: list[str] = []
    session_ids: set[str] = set()
    for pair in inner_pairs:
        groups = {record["group_id"] for record in pair.h_payload["sessions"]}
        if len(groups) != 1:
            raise ResidualLateFusionError("one inner OOF artifact must contain exactly one group")
        group_ids.append(next(iter(groups)))
        current_sessions = {record["session_id"] for record in pair.h_payload["sessions"]}
        if session_ids & current_sessions:
            raise ResidualLateFusionError("inner OOF session appears in more than one fold")
        session_ids.update(current_sessions)
    if len(set(group_ids)) != 3:
        raise ResidualLateFusionError("inner OOF selection must cover three distinct groups")

    candidates: list[dict[str, object]] = []
    selected_lambda = LAMBDA_GRID[0]
    selected_score = -math.inf
    for lambda_value in LAMBDA_GRID:
        inner_scores = []
        for pair, group_id in zip(inner_pairs, group_ids, strict=True):
            score = _participant_macro_strict_map(
                pair.h_payload,
                _fuse_pair(pair, lambda_value),
            )
            inner_scores.append(
                {
                    "inner_fold_id": pair.h_payload["inner_fold_id"],
                    "group_id": group_id,
                    "participant_macro_strict_mAP": score,
                }
            )
        group_equal_score = float(
            math.fsum(item["participant_macro_strict_mAP"] for item in inner_scores)
            / len(inner_scores)
        )
        candidates.append(
            {
                "lambda": lambda_value,
                "inner_group_scores": inner_scores,
                "group_equal_participant_macro_strict_mAP": group_equal_score,
            }
        )
        # Grid iteration is ascending, so exact score ties retain the smaller lambda.
        if group_equal_score > selected_score:
            selected_score = group_equal_score
            selected_lambda = lambda_value
    return selected_lambda, candidates


def _canonical_probability_bytes_sha256(probabilities: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    for probability in probabilities:
        canonical = np.ascontiguousarray(probability, dtype="<f4")
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _pack_probabilities(probabilities: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    if not probabilities:
        raise ResidualLateFusionError("no probabilities to pack")
    offsets = [0]
    for value in probabilities:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 3 or array.shape[1:] != (4, 4):
            raise ResidualLateFusionError("saved probability has unexpected shape")
        offsets.append(offsets[-1] + array.shape[0])
    return np.concatenate(probabilities, axis=0), np.asarray(offsets, dtype=np.int64)


def _load_saved_probabilities(
    path: Path,
    expected_lengths: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    try:
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {"probabilities", "session_offsets"}:
                raise ResidualLateFusionError("saved probability NPZ fields are not exact")
            probabilities = np.asarray(saved["probabilities"])
            offsets = np.asarray(saved["session_offsets"])
    except (OSError, ValueError) as exc:
        raise ResidualLateFusionError(f"cannot read saved probability artifact: {path}") from exc
    expected_offsets = np.asarray(
        [0, *np.cumsum(np.asarray(expected_lengths, dtype=np.int64)).tolist()],
        dtype=np.int64,
    )
    if probabilities.dtype != np.float32 or probabilities.shape != (
        int(expected_offsets[-1]),
        4,
        4,
    ):
        raise ResidualLateFusionError("saved fused probability shape/dtype differs")
    if offsets.dtype != np.int64 or not np.array_equal(offsets, expected_offsets):
        raise ResidualLateFusionError("saved probability session offsets differ")
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities <= 0.0) | (probabilities >= 1.0)).any()
    ):
        raise ResidualLateFusionError("saved fused probabilities are outside (0,1)")
    return tuple(
        probabilities[left:right].copy()
        for left, right in zip(offsets[:-1], offsets[1:], strict=True)
    )


def _metrics(
    h_payload: dict[str, object],
    probabilities: tuple[np.ndarray, ...],
) -> dict[str, object]:
    return thesis_backend._metrics(
        thesis_backend._sessions_from_prediction(h_payload),
        probabilities,
    )


def _source_binding_records(source: SourceMatrix, outer_fold_id: str) -> list[dict[str, object]]:
    records = []
    for condition in SOURCE_CONDITIONS:
        unit = source.units[(condition, outer_fold_id)]
        records.append(
            {
                "role": f"{condition}_result_manifest",
                **unit.result_manifest_binding.to_dict(),
            }
        )
        for inner_fold_id in INNER_FOLD_IDS:
            local = unit.prediction_bindings[inner_fold_id]
            path = _verify_binding(local, unit.directory)
            records.append(
                {
                    "role": f"{condition}_{inner_fold_id}_oof_probabilities",
                    **_binding(path, source.root).to_dict(),
                }
            )
        local = unit.prediction_bindings["outer"]
        path = _verify_binding(local, unit.directory)
        records.append(
            {
                "role": f"{condition}_outer_probabilities",
                **_binding(path, source.root).to_dict(),
            }
        )
    return records


def _protocol_payload(source: SourceMatrix) -> dict[str, object]:
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
            "saved-probability residual diagnostic; no training, checkpoint, model inference, "
            "or trainable gated-fusion architecture"
        ),
        "source_matrix_root": str(source.root),
        "source_matrix_index": source.index_binding.to_dict(),
        "source_protocol_root": str(source.protocol_root),
        "source_run_manifest": source.run_manifest_binding.to_dict(),
        "source_run_manifest_contract": {
            "artifact_kind": SOURCE_RUN_MANIFEST_KIND,
            "format_version": SOURCE_RUN_MANIFEST_VERSION,
            "protocol_id": SOURCE_PROTOCOL_ID,
            "max_epochs": SOURCE_MAX_EPOCHS,
            "seeds": list(SOURCE_SEEDS),
            "condition_ids": list(MATRIX_CONDITIONS),
            "canonical_json_sha256": SOURCE_RUN_MANIFEST_SHA256,
            "raw_file_sha256": source.run_manifest_binding.sha256,
        },
        "source_conditions": list(SOURCE_CONDITIONS),
        "historical_A_plus_H_context": {
            "condition_id": "A+H",
            "role": "context_only",
            "used_in_lambda_selection": False,
            "used_in_primary_comparison": False,
        },
        "seed": 13,
        "outer_fold_ids": list(OUTER_FOLD_IDS),
        "inner_fold_ids_per_outer": list(INNER_FOLD_IDS),
        "lambda_grid": list(LAMBDA_GRID),
        "formula": FORMULA,
        "selection_metric": SELECTION_METRIC,
        "selection_aggregation": "equal_mean_of_three_inner_group_scores_not_frame_pooled",
        "tie_rule": TIE_RULE,
        "outer_gold_used_in_lambda_selection": False,
        "matched_control": {
            "lambda": 0.0,
            "definition": "same_formula_exact_H_copy",
            "required_array_equal_to_H": True,
            "required_canonical_probability_bytes_equal_to_H": True,
        },
        "primary_comparison": "selected_lambda_minus_matched_lambda0",
        "a_participant_symmetry_tolerance_max_abs_diff": A_SYMMETRY_TOLERANCE,
        "descriptive_success_criteria_recorded_before_this_diagnostic_outer_replay": (
            SUCCESS_CRITERIA
        ),
        "criteria_timing_caveat": (
            "criteria were recorded after historical outer outcomes existed; they are descriptive, "
            "not preregistered or confirmatory"
        ),
    }


def _fold_payload(
    *,
    source: SourceMatrix,
    outer_fold_id: str,
    selected_lambda: float,
    candidates: list[dict[str, object]],
    inner_pairs: tuple[PairedPredictions, ...],
    outer_pair: PairedPredictions,
    fused_binding: FileBinding,
    selected_probabilities: tuple[np.ndarray, ...],
    lambda0_probabilities: tuple[np.ndarray, ...],
    selected_metrics: dict[str, object],
    lambda0_metrics: dict[str, object],
) -> dict[str, object]:
    h_probabilities = _probability_arrays(outer_pair.h_payload)
    if not all(
        np.array_equal(control, source_h)
        for control, source_h in zip(lambda0_probabilities, h_probabilities, strict=True)
    ):
        raise ResidualLateFusionError("lambda=0 is not an exact-copy H matched control")
    h_hash = _canonical_probability_bytes_sha256(h_probabilities)
    lambda0_hash = _canonical_probability_bytes_sha256(lambda0_probabilities)
    if h_hash != lambda0_hash:
        raise ResidualLateFusionError("lambda=0 canonical probability bytes differ from H")
    outer_groups = {record["group_id"] for record in outer_pair.h_payload["sessions"]}
    if len(outer_groups) != 1:
        raise ResidualLateFusionError("one outer fold must contain exactly one held-out group")
    inner_groups = {
        record["group_id"] for pair in inner_pairs for record in pair.h_payload["sessions"]
    }
    if len(inner_groups) != 3 or inner_groups & outer_groups:
        raise ResidualLateFusionError("inner and outer group scopes are not disjoint 3+1 LOGO")
    return {
        "artifact_kind": FOLD_KIND,
        "format_version": FORMAT_VERSION,
        "result_role": RESULT_ROLE,
        "data_kind": DATA_KIND,
        "production_eligible": False,
        "counted_eligible": False,
        "counted_blockers": list(COUNTED_BLOCKERS),
        "analysis_timing": ANALYSIS_TIMING,
        "confirmatory_hypothesis_verdict_eligible": False,
        "outer_fold_id": outer_fold_id,
        "held_out_group_id": next(iter(outer_groups)),
        "outer_session_ids": [record["session_id"] for record in outer_pair.h_payload["sessions"]],
        "seed": 13,
        "formula": FORMULA,
        "lambda_grid": list(LAMBDA_GRID),
        "selection_metric": SELECTION_METRIC,
        "selection_aggregation": "equal_mean_of_three_inner_group_scores_not_frame_pooled",
        "tie_rule": TIE_RULE,
        "outer_gold_used_in_lambda_selection": False,
        "selected_lambda": selected_lambda,
        "inner_selection_candidates": candidates,
        "matched_control": {
            "lambda": 0.0,
            "definition": "same_formula_exact_H_copy",
            "array_equal_to_H": True,
            "canonical_probability_bytes_equal_to_H": True,
            "h_source_probability_bytes_sha256": h_hash,
            "lambda0_probability_bytes_sha256": lambda0_hash,
        },
        "a_participant_symmetry": {
            "tolerance_max_abs_diff": A_SYMMETRY_TOLERANCE,
            "inner_observed_max_abs_diff": {
                pair.h_payload["inner_fold_id"]: pair.a_symmetry_max_abs_diff
                for pair in inner_pairs
            },
            "outer_observed_max_abs_diff": outer_pair.a_symmetry_max_abs_diff,
            "all_within_tolerance": True,
        },
        "source_alignment": {
            "ticks": "exact",
            "labels": "array_equal",
            "vad": "array_equal",
            "valid": "array_equal",
            "present": "array_equal",
            "session_and_group_identity": "exact",
        },
        "source_bindings": _source_binding_records(source, outer_fold_id),
        "fused_probability_artifact": fused_binding.to_dict(),
        "selected_probability_bytes_sha256": _canonical_probability_bytes_sha256(
            selected_probabilities
        ),
        "outer_metrics": {
            "selected": selected_metrics,
            "matched_lambda0": lambda0_metrics,
        },
        "primary_comparison": "selected_lambda_minus_matched_lambda0",
        "historical_A_plus_H_role": "context_only_not_loaded_or_compared",
    }


def _metric_at(metrics: dict[str, object], *path: str) -> float | None:
    value: object = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value if type(value) is float and math.isfinite(value) else None


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def _success_evaluation(
    selected: dict[str, object],
    control: dict[str, object],
    fold_deltas: dict[str, float | None],
) -> dict[str, object]:
    participant_delta = _difference(
        _metric_at(selected, "frame", "participant_macro_mAP"),
        _metric_at(control, "frame", "participant_macro_mAP"),
    )
    shift_delta = _difference(
        _metric_at(selected, "events", "shift_prediction", "average_precision"),
        _metric_at(control, "events", "shift_prediction", "average_precision"),
    )
    overlap_delta = _difference(
        _metric_at(selected, "events", "overlap_resolution", "average_precision"),
        _metric_at(control, "events", "overlap_resolution", "average_precision"),
    )
    positive_folds = sum(value is not None and value > 0.0 for value in fold_deltas.values())
    components = {
        "participant_macro_delta_at_least_0.02": (
            participant_delta is not None and participant_delta >= 0.02
        ),
        "at_least_3_of_4_outer_folds_strictly_positive": positive_folds >= 3,
        "shift_prediction_AP_delta_not_below_minus_0.02": (
            shift_delta is not None and shift_delta >= -0.02
        ),
        "overlap_resolution_AP_delta_not_below_minus_0.02": (
            overlap_delta is not None and overlap_delta >= -0.02
        ),
    }
    criteria_met = all(components.values())
    return {
        "criteria": SUCCESS_CRITERIA,
        "observed": {
            "participant_macro_mAP_selected_minus_lambda0": participant_delta,
            "strictly_positive_outer_fold_count": positive_folds,
            "shift_prediction_AP_selected_minus_lambda0": shift_delta,
            "overlap_resolution_AP_selected_minus_lambda0": overlap_delta,
        },
        "components": components,
        "interpretation_role": "descriptive_post_outcome_threshold_check",
        "verdict": ("descriptive_criteria_met" if criteria_met else "descriptive_criteria_not_met"),
        "descriptive_criteria_met": criteria_met,
        "confirmatory_hypothesis_verdict_eligible": False,
        "lambda_selection_was_not_changed_by_this_verdict": True,
    }


def _aggregate_payload(
    protocol_binding: FileBinding,
    fold_records: list[tuple[dict[str, object], FileBinding, FileBinding]],
    selected_metrics: dict[str, object],
    lambda0_metrics: dict[str, object],
) -> dict[str, object]:
    per_fold: dict[str, dict[str, object]] = {}
    fold_deltas: dict[str, float | None] = {}
    for manifest, manifest_binding, probability_binding in fold_records:
        fold_id = manifest["outer_fold_id"]
        selected_value = _metric_at(
            manifest["outer_metrics"]["selected"], "frame", "participant_macro_mAP"
        )
        control_value = _metric_at(
            manifest["outer_metrics"]["matched_lambda0"],
            "frame",
            "participant_macro_mAP",
        )
        delta = _difference(selected_value, control_value)
        fold_deltas[fold_id] = delta
        per_fold[fold_id] = {
            "held_out_group_id": manifest["held_out_group_id"],
            "selected_lambda": manifest["selected_lambda"],
            "participant_macro_mAP_selected": selected_value,
            "participant_macro_mAP_lambda0": control_value,
            "participant_macro_mAP_selected_minus_lambda0": delta,
            "fold_manifest": manifest_binding.to_dict(),
            "fused_probability_artifact": probability_binding.to_dict(),
        }
    return {
        "artifact_kind": AGGREGATE_KIND,
        "format_version": FORMAT_VERSION,
        "result_role": RESULT_ROLE,
        "data_kind": DATA_KIND,
        "production_eligible": False,
        "counted_eligible": False,
        "counted_blockers": list(COUNTED_BLOCKERS),
        "analysis_timing": ANALYSIS_TIMING,
        "confirmatory_hypothesis_verdict_eligible": False,
        "aggregation_source": "saved_fused_outer_probabilities_only_no_model_inference",
        "protocol": protocol_binding.to_dict(),
        "outer_fold_ids": list(OUTER_FOLD_IDS),
        "seed": 13,
        "formula": FORMULA,
        "primary_comparison": "selected_lambda_minus_matched_lambda0",
        "matched_control": "lambda0_same_formula_exact_H_copy",
        "per_fold": per_fold,
        "pooled": {
            "selected": selected_metrics,
            "matched_lambda0": lambda0_metrics,
        },
        "primary_comparison_metrics": {
            "participant_macro_mAP": _difference(
                _metric_at(selected_metrics, "frame", "participant_macro_mAP"),
                _metric_at(lambda0_metrics, "frame", "participant_macro_mAP"),
            ),
            "shift_prediction_AP": _difference(
                _metric_at(selected_metrics, "events", "shift_prediction", "average_precision"),
                _metric_at(lambda0_metrics, "events", "shift_prediction", "average_precision"),
            ),
            "overlap_resolution_AP": _difference(
                _metric_at(selected_metrics, "events", "overlap_resolution", "average_precision"),
                _metric_at(lambda0_metrics, "events", "overlap_resolution", "average_precision"),
            ),
        },
        "success_evaluation": _success_evaluation(
            selected_metrics,
            lambda0_metrics,
            fold_deltas,
        ),
        "historical_A_plus_H_context": {
            "role": "context_only",
            "used_in_selection": False,
            "used_in_primary_comparison": False,
        },
        "diagnostic_boundary": (
            "post-hoc residual fusion of saved probabilities; not a newly trained gated model"
        ),
    }


def _expected_output_files() -> set[str]:
    result = {PROTOCOL_FILENAME, AGGREGATE_FILENAME}
    for fold in OUTER_FOLD_IDS:
        result.add(f"{fold}/{FOLD_MANIFEST_FILENAME}")
        result.add(f"{fold}/{FUSED_PROBABILITIES_FILENAME}")
    return result


def _actual_output_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _recompute_expected_fold(
    source: SourceMatrix,
    outer_fold_id: str,
) -> tuple[
    float,
    list[dict[str, object]],
    tuple[PairedPredictions, ...],
    PairedPredictions,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    inner_pairs = tuple(
        _paired_predictions(source, outer_fold_id, inner_fold_id)
        for inner_fold_id in INNER_FOLD_IDS
    )
    selected_lambda, candidates = _select_lambda(inner_pairs)
    # Only inner prediction payloads enter lambda selection.  The historical
    # integrity validator may already have read outer artifact bytes, but the
    # outer prediction payload is first passed to diagnostic computation here.
    outer_pair = _paired_predictions(source, outer_fold_id, None)
    outer_group = {record["group_id"] for record in outer_pair.h_payload["sessions"]}
    inner_groups = {
        record["group_id"] for pair in inner_pairs for record in pair.h_payload["sessions"]
    }
    if len(outer_group) != 1 or len(inner_groups) != 3 or outer_group & inner_groups:
        raise ResidualLateFusionError("paired nested-LOGO group partition is invalid")
    selected = _fuse_pair(outer_pair, selected_lambda)
    lambda0 = _fuse_pair(outer_pair, 0.0)
    return selected_lambda, candidates, inner_pairs, outer_pair, selected, lambda0


def _publish_residual_late_fusion_with_audit(
    source_root_value: str | Path = DEFAULT_SOURCE,
    output_root_value: str | Path = DEFAULT_OUTPUT,
) -> tuple[Path, dict[str, object]]:
    """Create the isolated diagnostic exactly once, then replay-audit it."""

    output_root = Path(output_root_value).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite residual-fusion output: {output_root}")
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stage = parent / f".{output_root.name}.{token}.stage"
    stage.mkdir(exist_ok=False)
    published_this_call = False
    published_identity: tuple[int, int] | None = None
    try:
        source = _load_source_matrix(source_root_value)
        protocol_path = stage / PROTOCOL_FILENAME
        _write_json_exclusive(protocol_path, _protocol_payload(source))
        protocol_binding = _binding(protocol_path, stage)
        fold_records: list[tuple[dict[str, object], FileBinding, FileBinding]] = []
        selected_sessions = []
        selected_probabilities = []
        lambda0_sessions = []
        lambda0_probabilities = []
        observed_outer_groups: set[str] = set()
        observed_outer_sessions: set[str] = set()
        for outer_fold_id in OUTER_FOLD_IDS:
            (
                selected_lambda,
                candidates,
                inner_pairs,
                outer_pair,
                selected,
                lambda0,
            ) = _recompute_expected_fold(source, outer_fold_id)
            fold_dir = stage / outer_fold_id
            packed, offsets = _pack_probabilities(selected)
            fused_path = fold_dir / FUSED_PROBABILITIES_FILENAME
            _write_npz_exclusive(fused_path, packed, offsets)
            fused_binding = _binding(fused_path, stage)
            selected_metrics = _metrics(outer_pair.h_payload, selected)
            lambda0_metrics = _metrics(outer_pair.h_payload, lambda0)
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
            fold_group = manifest["held_out_group_id"]
            if fold_group in observed_outer_groups:
                raise ResidualLateFusionError("outer group appears in more than one fold")
            observed_outer_groups.add(fold_group)
            for session_id in manifest["outer_session_ids"]:
                if session_id in observed_outer_sessions:
                    raise ResidualLateFusionError("outer session appears in more than one fold")
                observed_outer_sessions.add(session_id)
            manifest_path = fold_dir / FOLD_MANIFEST_FILENAME
            _write_json_exclusive(manifest_path, manifest)
            manifest_binding = _binding(manifest_path, stage)
            fold_records.append((manifest, manifest_binding, fused_binding))
            sessions = thesis_backend._sessions_from_prediction(outer_pair.h_payload)
            selected_sessions.extend(sessions)
            selected_probabilities.extend(selected)
            lambda0_sessions.extend(sessions)
            lambda0_probabilities.extend(lambda0)
        if observed_outer_groups != set(source.index["group_ids"]):
            raise ResidualLateFusionError("outer folds do not cover the four indexed groups")
        if observed_outer_sessions != set(source.index["included_session_ids"]):
            raise ResidualLateFusionError("outer folds do not cover the exact indexed sessions")
        selected_pooled = thesis_backend._metrics(
            tuple(selected_sessions), tuple(selected_probabilities)
        )
        lambda0_pooled = thesis_backend._metrics(
            tuple(lambda0_sessions), tuple(lambda0_probabilities)
        )
        aggregate = _aggregate_payload(
            protocol_binding,
            fold_records,
            selected_pooled,
            lambda0_pooled,
        )
        _write_json_exclusive(stage / AGGREGATE_FILENAME, aggregate)
        _audit_residual_late_fusion(stage, validated_source=source)
        if output_root.exists():
            raise FileExistsError(f"refusing to overwrite residual-fusion output: {output_root}")
        published_identity = _directory_identity(stage)
        os.rename(stage, output_root)
        published_this_call = True
        post_rename_audit = _audit_residual_late_fusion(
            output_root,
            validated_source=source,
        )
        return output_root / AGGREGATE_FILENAME, post_rename_audit
    except BaseException as exc:
        if stage.exists() and stage.is_dir() and not stage.is_symlink():
            shutil.rmtree(stage)
        if published_this_call:
            try:
                current_identity = _directory_identity(output_root)
            except ResidualLateFusionError:
                current_identity = None
            if current_identity != published_identity:
                raise ResidualLateFusionError(
                    "post-rename audit failed and output identity changed; "
                    "refusing to delete an unowned replacement"
                ) from exc
            shutil.rmtree(output_root)
        raise


def publish_residual_late_fusion(
    source_root_value: str | Path = DEFAULT_SOURCE,
    output_root_value: str | Path = DEFAULT_OUTPUT,
) -> Path:
    """Create once and return the aggregate path after the cached post-rename audit."""

    aggregate_path, _ = _publish_residual_late_fusion_with_audit(
        source_root_value,
        output_root_value,
    )
    return aggregate_path


def _audit_residual_late_fusion(
    output_root_value: str | Path,
    *,
    validated_source: SourceMatrix | None,
) -> dict[str, object]:
    """Rebuild selection and all metrics from bound source/saved probabilities."""

    output_root = Path(output_root_value).resolve()
    if not output_root.is_dir() or output_root.is_symlink():
        raise ResidualLateFusionError("residual-fusion output must be a real directory")
    actual_files = _actual_output_files(output_root)
    expected_files = _expected_output_files()
    if actual_files != expected_files:
        raise ResidualLateFusionError(
            f"residual-fusion output file set differs: expected={sorted(expected_files)}, "
            f"actual={sorted(actual_files)}"
        )
    protocol_path = output_root / PROTOCOL_FILENAME
    protocol = _read_json_strict(protocol_path)
    source_root = protocol.get("source_matrix_root")
    if not isinstance(source_root, str) or not Path(source_root).is_absolute():
        raise ResidualLateFusionError("protocol source_matrix_root must be absolute")
    resolved_source_root = Path(source_root).resolve()
    if validated_source is None:
        source = _load_source_matrix(resolved_source_root)
    else:
        source = validated_source
        if source.root != resolved_source_root:
            raise ResidualLateFusionError("cached source matrix differs from output protocol")
        _revalidate_source_matrix(source)
    expected_protocol = _protocol_payload(source)
    if _canonical_json(protocol) != _canonical_json(expected_protocol):
        raise ResidualLateFusionError("protocol differs from source-replayed expectation")
    protocol_binding = _binding(protocol_path, output_root)

    fold_records: list[tuple[dict[str, object], FileBinding, FileBinding]] = []
    selected_sessions = []
    selected_probabilities = []
    lambda0_sessions = []
    lambda0_probabilities = []
    observed_symmetry = 0.0
    for outer_fold_id in OUTER_FOLD_IDS:
        (
            selected_lambda,
            candidates,
            inner_pairs,
            outer_pair,
            expected_selected,
            expected_lambda0,
        ) = _recompute_expected_fold(source, outer_fold_id)
        manifest_path = output_root / outer_fold_id / FOLD_MANIFEST_FILENAME
        fused_path = output_root / outer_fold_id / FUSED_PROBABILITIES_FILENAME
        manifest = _read_json_strict(manifest_path)
        lengths = tuple(record["n_frames"] for record in outer_pair.h_payload["sessions"])
        saved_selected = _load_saved_probabilities(fused_path, lengths)
        if not all(
            np.array_equal(saved, expected)
            for saved, expected in zip(saved_selected, expected_selected, strict=True)
        ):
            raise ResidualLateFusionError("saved fused probabilities differ from formula replay")
        fused_binding = _binding(fused_path, output_root)
        selected_metrics = _metrics(outer_pair.h_payload, saved_selected)
        lambda0_metrics = _metrics(outer_pair.h_payload, expected_lambda0)
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
        if _canonical_json(manifest) != _canonical_json(expected_manifest):
            raise ResidualLateFusionError(f"fold manifest replay differs: {outer_fold_id}")
        manifest_binding = _binding(manifest_path, output_root)
        fold_records.append((manifest, manifest_binding, fused_binding))
        observed_symmetry = max(
            observed_symmetry,
            outer_pair.a_symmetry_max_abs_diff,
            *(pair.a_symmetry_max_abs_diff for pair in inner_pairs),
        )
        sessions = thesis_backend._sessions_from_prediction(outer_pair.h_payload)
        selected_sessions.extend(sessions)
        selected_probabilities.extend(saved_selected)
        lambda0_sessions.extend(sessions)
        lambda0_probabilities.extend(expected_lambda0)
    selected_pooled = thesis_backend._metrics(
        tuple(selected_sessions), tuple(selected_probabilities)
    )
    lambda0_pooled = thesis_backend._metrics(tuple(lambda0_sessions), tuple(lambda0_probabilities))
    expected_aggregate = _aggregate_payload(
        protocol_binding,
        fold_records,
        selected_pooled,
        lambda0_pooled,
    )
    aggregate_path = output_root / AGGREGATE_FILENAME
    aggregate = _read_json_strict(aggregate_path)
    if _canonical_json(aggregate) != _canonical_json(expected_aggregate):
        raise ResidualLateFusionError("aggregate differs from saved-probability replay")
    _revalidate_source_matrix(source)
    return {
        "audit_passed": True,
        "output_root": str(output_root),
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": _sha256_file(aggregate_path),
        "selected_lambdas": {
            fold_id: aggregate["per_fold"][fold_id]["selected_lambda"] for fold_id in OUTER_FOLD_IDS
        },
        "a_participant_symmetry_observed_max_abs_diff": observed_symmetry,
        "a_participant_symmetry_tolerance": A_SYMMETRY_TOLERANCE,
        "success_evaluation": aggregate["success_evaluation"],
    }


def audit_residual_late_fusion(
    output_root_value: str | Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Independently audit an existing result with a fresh full source validation."""

    return _audit_residual_late_fusion(
        output_root_value,
        validated_source=None,
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Run or audit the isolated saved-probability residual late-fusion diagnostic."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Create the diagnostic exactly once.")
    run_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    audit_parser = subparsers.add_parser("audit", help="Replay-audit an existing diagnostic.")
    audit_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "run":
        aggregate_path, result = _publish_residual_late_fusion_with_audit(
            args.source,
            args.output,
        )
        result["aggregate_path"] = str(aggregate_path)
    else:
        result = audit_residual_late_fusion(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
