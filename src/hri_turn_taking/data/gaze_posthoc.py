"""Nullable 2D Gaze post-hoc fixture and frozen real-matrix contracts.

The sealed source preflight remains read-only and permanently non-publishing.
This module separately provides an owner-authorised exact-eight publisher that
binds the current preflight evidence to the formal A/H/A+H bases under the
parallel schema v3.  The derived matrix is usable only for post-hoc exploratory
training: upstream exact regeneration and implementation counterfactual
causality remain unverified, and production/counting/formal-run eligibility is
always false.

Nothing here registers the matrix with ``vap-assemble``, ``vap-train`` or a
formal outer runner.  The older synthetic fixture API remains available for
schema, mask and normaliser tests.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hri_turn_taking.data import analysis_base, gaze_source_readiness
from hri_turn_taking.data.multiparty import (
    FEATURE_SCHEMA_V2,
    FEATURE_SCHEMA_V3_GAZE_VALIDITY,
    GLOBAL_FEATURE_COLUMNS,
    MAX_PARTICIPANTS,
    input_feature_columns,
    participant_column,
    validate_multiparty_dataframe,
)
from hri_turn_taking.vap.dataset import (
    TRANSFORMED_GLOBAL_FEATURES,
    transformed_participant_features,
)
from hri_turn_taking.vap.feature_sets import (
    GAZE_POSTHOC_CONDITION_IDS,
    gaze_posthoc_feature_set_metadata,
)

ARTIFACT_KIND = "vap_gaze_posthoc_schema_v3_fixture"
READINESS_ARTIFACT_KIND = "vap_gaze_posthoc_schema_v3_fixture_readiness"
FORMAT_VERSION = 1
MANIFEST_SUFFIX = ".gaze-posthoc-fixture.json"
SLOTS = tuple(f"P{index}" for index in range(1, MAX_PARTICIPANTS + 1))

REAL_SCENE_ARTIFACT_KIND = "vap_real_gaze_posthoc_schema_v3_scene"
REAL_MATRIX_ARTIFACT_KIND = "vap_real_gaze_posthoc_schema_v3_matrix"
REAL_SCENE_PROVENANCE_SUFFIX = ".provenance.json"
REAL_MATRIX_MANIFEST_NAME = "matrix.manifest.json"
REAL_PERMITTED_USE = "posthoc_exploratory_training"
GAZE_README_RELATIVE_PATH = Path("gaze_features_export") / "README.md"
GAZE_README_SHA256 = "9c7c4a298705228867c363fb6e16a570570c7e5a24a7e0ba3082481c7468d64a"
GAZE_LLE_METHOD_TAG = "v1.0.0"
GAZE_LLE_METHOD_COMMIT = "2c4fedac7159a0ffd9e40d96d4f58f4dd85aec6a"

# Frozen before this post-hoc publisher exists.  The two held scenes stay in the
# immutable ten-scene source corpus and are excluded only from this exploratory
# view; a caller cannot substitute a different member or time range.
EXACT_EIGHT_TICK_RANGES: dict[str, tuple[int, int]] = {
    "03_20__4": (376, 5332),
    "03_26__3": (315, 4636),
    "05_14__2": (601, 7351),
    "05_14__3": (624, 12338),
    "05_14__4": (687, 12383),
    "05_15__2": (656, 6973),
    "05_15__3": (586, 6398),
    "05_15__6": (629, 13362),
}
EXACT_EIGHT_GROUP_BY_SCENE = {
    "03_20__4": "03_20",
    "03_26__3": "03_26",
    "05_14__2": "05_14",
    "05_14__3": "05_14",
    "05_14__4": "05_14",
    "05_15__2": "05_15",
    "05_15__3": "05_15",
    "05_15__6": "05_15",
}
EXACT_EIGHT_TOTAL_ROWS = 64_299
EXACT_EIGHT_GROUPS = ("03_20", "03_26", "05_14", "05_15")
EXCLUDED_OWNER_HOLD_SCENES = {
    "05_15__4": "no A speech in Transcript; pending owner manual review",
    "05_15__5": "A/R extremely sparse; pending owner manual review",
}

_AUTHORITY_EXACT_EIGHT = tuple(
    scene
    for scene in analysis_base.EXPECTED_TICK_RANGES
    if scene not in EXCLUDED_OWNER_HOLD_SCENES
)
if (
    tuple(EXACT_EIGHT_TICK_RANGES) != _AUTHORITY_EXACT_EIGHT
    or any(
        EXACT_EIGHT_TICK_RANGES[scene] != analysis_base.EXPECTED_TICK_RANGES[scene]
        for scene in _AUTHORITY_EXACT_EIGHT
    )
    or sum(end - start for start, end in EXACT_EIGHT_TICK_RANGES.values())
    != EXACT_EIGHT_TOTAL_ROWS
):
    raise RuntimeError("Gaze post-hoc exact-eight profile differs from corpus authority")

UPSTREAM_REGENERATION_GAPS = (
    "actual_B_L_inout_variant_unrecorded",
    "actual_YOLO_commit_weights_and_configuration_unrecorded",
    "actual_Gaze_LLE_commit_weights_and_configuration_unrecorded",
)

_REQUIRED_PREFLIGHT_BINDING_POINTERS = frozenset(
    {
        "analysis_base",
        "analysis_base/provenance",
        "sealed_identity/inventory",
        "sealed_identity/crosswalk_manifest",
        "sealed_identity/identity_manifest",
        "sealed_identity/tracklet_observations",
        "source",
        "source/metadata",
    }
)

REAL_PUBLICATION_BLOCKERS = (
    "current_preflight_is_read_only_and_has_no_writer_or_assembly_authority",
    "gaze_extractor_algorithm_is_not_hash_bound",
    "gaze_extractor_version_or_weights_are_not_hash_bound",
    "gaze_extractor_model_input_causality_is_not_established",
)


class GazePosthocGateError(ValueError):
    """A schema, source, replay, or real-publication gate failed closed."""


class GazePosthocRecoveryError(GazePosthocGateError):
    """Final audit failed after publication and safe rollback was not possible."""


@dataclass(frozen=True)
class GazePosthocFixtureArtifact:
    csv_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class RealGazePosthocMatrixArtifact:
    output_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    frames: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class _Binding:
    path: Path
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _fail(message: str) -> None:
    raise GazePosthocGateError(message)


def _identity(stat: Any) -> tuple[int, int, int, int]:
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _stable_read(path: str | Path, label: str) -> tuple[bytes, _Binding]:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        _fail(f"{label} is missing, not a regular file, or is a symlink: {resolved}")
    try:
        before = resolved.stat()
        payload = resolved.read_bytes()
        after = resolved.stat()
    except OSError as exc:
        raise GazePosthocGateError(f"could not read {label}: {resolved}") from exc
    if _identity(before) != _identity(after) or len(payload) != after.st_size:
        _fail(f"{label} changed while it was read")
    return payload, _Binding(
        path=resolved,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _parse_csv(payload: bytes, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(payload), float_precision="round_trip")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise GazePosthocGateError(f"could not parse {label} as CSV: {exc}") from exc


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GazePosthocGateError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if type(value) is not dict:
        _fail(f"{label} must contain a JSON object")
    return value


def _validate_evidence(frame: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    required = {
        "session_id",
        "timestamp_s",
        "slot",
        "gaze_robot_valid",
        "gaze_robot",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        _fail(f"nullable Gaze evidence is missing columns: {missing}")
    evidence = frame.loc[:, sorted(required)].copy()
    if evidence.empty:
        _fail("nullable Gaze evidence must not be empty")
    if evidence["session_id"].isna().any() or evidence["timestamp_s"].isna().any():
        _fail("nullable Gaze evidence has missing canonical keys")
    if not evidence["slot"].isin(SLOTS).all():
        _fail("nullable Gaze evidence contains a non-canonical participant slot")
    key_columns = ["session_id", "timestamp_s", "slot"]
    if evidence.duplicated(key_columns).any():
        _fail("nullable Gaze evidence has duplicate session/timestamp/slot keys")
    base_keys = base.loc[:, ["session_id", "timestamp_s"]]
    if base_keys.duplicated().any():
        _fail("schema-v2 base has duplicate canonical keys")
    expected = {
        (str(session), float(timestamp), slot)
        for session, timestamp in base_keys.itertuples(index=False, name=None)
        for slot in SLOTS
    }
    actual = {
        (str(session), float(timestamp), str(slot))
        for session, timestamp, slot in evidence[key_columns].itertuples(
            index=False, name=None
        )
    }
    if actual != expected:
        _fail("nullable Gaze evidence does not exactly cover every base tick and slot")
    validity = evidence["gaze_robot_valid"]
    if validity.isna().any() or not validity.astype(float).isin((0.0, 1.0)).all():
        _fail("gaze_robot_valid must be an exact non-null binary field")
    validity_bool = validity.astype(float).eq(1.0)
    values = pd.to_numeric(evidence["gaze_robot"], errors="coerce")
    if values[~validity_bool].notna().any() or values[validity_bool].isna().any():
        _fail("gaze_robot must be present iff gaze_robot_valid=1")
    if not values[validity_bool].isin((0.0, 1.0)).all():
        _fail("valid gaze_robot evidence must be binary 0/1")
    robot = evidence["slot"].eq("P1")
    if validity_bool[robot].any() or values[robot].notna().any():
        _fail("P1/R must remain invalid and nullable in source Gaze evidence")
    evidence["gaze_robot_valid"] = validity_bool.astype(float)
    evidence["gaze_robot"] = values
    return evidence


def build_schema_v3_frame(base_frame: pd.DataFrame, evidence_frame: pd.DataFrame) -> pd.DataFrame:
    """Join exact nullable evidence onto one validated schema-v2 table in memory."""

    try:
        validate_multiparty_dataframe(base_frame, feature_schema=FEATURE_SCHEMA_V2)
    except (TypeError, ValueError) as exc:
        raise GazePosthocGateError(f"schema-v2 base failed validation: {exc}") from exc
    if any(participant_column(index, "gaze_robot_valid") in base_frame for index in range(1, 5)):
        _fail("schema-v2 base already contains a gaze_robot_valid column")
    evidence = _validate_evidence(evidence_frame, base_frame)
    output = base_frame.copy(deep=True)
    base_index = pd.MultiIndex.from_frame(output[["session_id", "timestamp_s"]])
    for index, slot in enumerate(SLOTS, start=1):
        rows = evidence.loc[evidence["slot"].eq(slot)].set_index(
            ["session_id", "timestamp_s"]
        )
        rows = rows.reindex(base_index)
        valid = rows["gaze_robot_valid"].to_numpy(dtype=float)
        values = rows["gaze_robot"].fillna(0.0).to_numpy(dtype=float)
        present = output[participant_column(index, "present")].to_numpy(dtype=float)
        if (valid > present).any():
            _fail(f"{slot} has valid Gaze evidence where the base slot is absent")
        value_column = participant_column(index, "gaze_robot")
        valid_column = participant_column(index, "gaze_robot_valid")
        output.loc[:, value_column] = values
        output.insert(output.columns.get_loc(value_column) + 1, valid_column, valid)
    try:
        validate_multiparty_dataframe(
            output, feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY
        )
    except (TypeError, ValueError) as exc:
        raise GazePosthocGateError(f"constructed schema-v3 table failed validation: {exc}") from exc
    return output


def current_real_publication_blockers(
    preflight_report: Mapping[str, object] | None = None,
) -> list[str]:
    """Return the non-bypassable blockers for current real Gaze publication."""

    blockers = list(REAL_PUBLICATION_BLOCKERS)
    if preflight_report is not None:
        if preflight_report.get("assembly_eligible") is not False:
            blockers.append("preflight_does_not_preserve_assembly_eligible_false")
        if preflight_report.get("counted_input_ready") is not False:
            blockers.append("preflight_does_not_preserve_counted_input_ready_false")
        if preflight_report.get("writer_available") is not False:
            blockers.append("preflight_does_not_preserve_writer_available_false")
        limits = preflight_report.get("scientific_limits")
        if not isinstance(limits, Mapping) or limits.get("model_input_promoted") is not False:
            blockers.append("preflight_model_input_limit_is_missing_or_inconsistent")
    return blockers


def require_current_real_publication_ready(
    preflight_report: Mapping[str, object] | None = None,
) -> None:
    """Always fail until a separately reviewed real publisher contract exists."""

    blockers = current_real_publication_blockers(preflight_report)
    _fail("real Gaze publication is blocked: " + "; ".join(blockers))


def _schema_contract() -> dict[str, object]:
    participant = transformed_participant_features(FEATURE_SCHEMA_V3_GAZE_VALIDITY)
    raw_columns = input_feature_columns(feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY)
    transformed_width = MAX_PARTICIPANTS * len(participant) + len(
        TRANSFORMED_GLOBAL_FEATURES
    )
    return {
        "feature_schema": FEATURE_SCHEMA_V3_GAZE_VALIDITY,
        "raw_input_columns": raw_columns,
        "raw_input_width": len(raw_columns),
        "transformed_participant_features": participant,
        "transformed_global_features": list(TRANSFORMED_GLOBAL_FEATURES),
        "transformed_input_width": transformed_width,
        "token_input_width": len(participant) + len(TRANSFORMED_GLOBAL_FEATURES),
        "global_raw_columns": list(GLOBAL_FEATURE_COLUMNS),
    }


def publish_fixture_artifact(
    base_csv: str | Path,
    evidence_csv: str | Path,
    output_csv: str | Path,
) -> GazePosthocFixtureArtifact:
    """Publish a replayable fixture artifact; never a real or trainable artifact."""

    output_path = Path(output_csv).resolve()
    manifest_path = output_path.with_name(output_path.name + MANIFEST_SUFFIX)
    if output_path.suffix.lower() != ".csv":
        _fail("fixture output path must have a .csv suffix")
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite Gaze post-hoc fixture artifact(s): "
            + ", ".join(str(path) for path in existing)
        )
    base_payload, base_binding = _stable_read(base_csv, "schema-v2 fixture base")
    evidence_payload, evidence_binding = _stable_read(
        evidence_csv, "nullable Gaze fixture evidence"
    )
    frame = build_schema_v3_frame(
        _parse_csv(base_payload, "schema-v2 fixture base"),
        _parse_csv(evidence_payload, "nullable Gaze fixture evidence"),
    )
    output_payload = _csv_bytes(frame)
    output_binding = _Binding(
        path=output_path,
        sha256=hashlib.sha256(output_payload).hexdigest(),
        size_bytes=len(output_payload),
    )
    manifest: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "fixture_only": True,
        "posthoc": True,
        "production_eligible": False,
        "counted_eligible": False,
        "real_training_ready": False,
        "base_source": base_binding.to_dict(),
        "gaze_evidence_source": evidence_binding.to_dict(),
        "output": {
            **output_binding.to_dict(),
            "columns": list(frame.columns),
            "dtypes": {column: str(frame[column].dtype) for column in frame},
            "n_rows": len(frame),
        },
        "schema": _schema_contract(),
        "conditions": {
            condition: gaze_posthoc_feature_set_metadata(condition)
            for condition in GAZE_POSTHOC_CONDITION_IDS
        },
        "real_publication_blockers": list(REAL_PUBLICATION_BLOCKERS),
    }
    manifest_payload = _json_bytes(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_created = False
    try:
        with output_path.open("xb") as handle:
            handle.write(output_payload)
        output_created = True
        with manifest_path.open("xb") as handle:
            handle.write(manifest_payload)
    except BaseException:
        # Only remove the exact output this call created; never touch sources.
        if output_created and output_path.exists() and not manifest_path.exists():
            output_path.unlink()
        raise
    return audit_fixture_artifact(output_path)


def _assert_binding(value: object, actual: _Binding, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        _fail(f"{label} binding is malformed")
    if (
        value.get("path") != str(actual.path)
        or value.get("sha256") != actual.sha256
        or type(value.get("size_bytes")) is not int
        or value.get("size_bytes") != actual.size_bytes
    ):
        _fail(f"{label} bytes/path differ from the manifest binding")


def audit_fixture_artifact(output_csv: str | Path) -> GazePosthocFixtureArtifact:
    """Re-hash sources/output and replay schema-v3 construction exactly."""

    output_path = Path(output_csv).resolve()
    manifest_path = output_path.with_name(output_path.name + MANIFEST_SUFFIX)
    output_payload, output_binding = _stable_read(output_path, "Gaze fixture output")
    manifest_payload, _manifest_binding = _stable_read(
        manifest_path, "Gaze fixture manifest"
    )
    manifest = _strict_json(manifest_payload, "Gaze fixture manifest")
    expected_top = {
        "artifact_kind",
        "format_version",
        "fixture_only",
        "posthoc",
        "production_eligible",
        "counted_eligible",
        "real_training_ready",
        "base_source",
        "gaze_evidence_source",
        "output",
        "schema",
        "conditions",
        "real_publication_blockers",
    }
    if set(manifest) != expected_top:
        _fail("Gaze fixture manifest fields are not exact")
    if (
        manifest.get("artifact_kind") != ARTIFACT_KIND
        or type(manifest.get("format_version")) is not int
        or manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("fixture_only") is not True
        or manifest.get("posthoc") is not True
        or manifest.get("production_eligible") is not False
        or manifest.get("counted_eligible") is not False
        or manifest.get("real_training_ready") is not False
        or manifest.get("schema") != _schema_contract()
        or manifest.get("conditions")
        != {
            condition: gaze_posthoc_feature_set_metadata(condition)
            for condition in GAZE_POSTHOC_CONDITION_IDS
        }
        or manifest.get("real_publication_blockers") != list(REAL_PUBLICATION_BLOCKERS)
    ):
        _fail("Gaze fixture manifest contract is inconsistent")
    base_path = Path(str(manifest["base_source"].get("path")))
    evidence_path = Path(str(manifest["gaze_evidence_source"].get("path")))
    base_payload, base_binding = _stable_read(base_path, "bound schema-v2 fixture base")
    evidence_payload, evidence_binding = _stable_read(
        evidence_path, "bound nullable Gaze fixture evidence"
    )
    _assert_binding(manifest["base_source"], base_binding, "base source")
    _assert_binding(manifest["gaze_evidence_source"], evidence_binding, "Gaze evidence")
    output_record = manifest["output"]
    if not isinstance(output_record, Mapping):
        _fail("Gaze fixture output binding is malformed")
    _assert_binding(
        {key: output_record.get(key) for key in ("path", "sha256", "size_bytes")},
        output_binding,
        "fixture output",
    )
    replay = build_schema_v3_frame(
        _parse_csv(base_payload, "bound schema-v2 fixture base"),
        _parse_csv(evidence_payload, "bound nullable Gaze fixture evidence"),
    )
    if _csv_bytes(replay) != output_payload:
        _fail("Gaze fixture output cannot be reproduced byte-for-byte from bound sources")
    parsed_output = _parse_csv(output_payload, "Gaze fixture output")
    if (
        output_record.get("columns") != list(parsed_output.columns)
        or output_record.get("dtypes")
        != {column: str(parsed_output[column].dtype) for column in parsed_output}
        or output_record.get("n_rows") != len(parsed_output)
    ):
        _fail("Gaze fixture output schema/row metadata differs from its bytes")
    return GazePosthocFixtureArtifact(output_path, manifest_path, manifest)


def fixture_readiness(output_csv: str | Path) -> dict[str, object]:
    """Return hash-bound fixture readiness without granting real training status."""

    artifact = audit_fixture_artifact(output_csv)
    manifest_payload, manifest_binding = _stable_read(
        artifact.manifest_path, "Gaze fixture manifest"
    )
    del manifest_payload
    return {
        "artifact_kind": READINESS_ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "fixture_only": True,
        "schema_fixture_ready": True,
        "hash_replay_ready": True,
        "condition_fixture_ready": {
            condition: True for condition in GAZE_POSTHOC_CONDITION_IDS
        },
        "manifest_sha256": manifest_binding.sha256,
        "posthoc": True,
        "production_eligible": False,
        "counted_eligible": False,
        "real_training_ready": False,
        "real_publication_blockers": list(REAL_PUBLICATION_BLOCKERS),
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_json(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _binding_for_payload(path: Path, payload: bytes) -> _Binding:
    return _Binding(
        path=path.resolve(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_method_evidence(root: Path) -> dict[str, object]:
    readme_path = (root / GAZE_README_RELATIVE_PATH).resolve()
    _payload, readme = _stable_read(readme_path, "frozen Gaze export README")
    if readme.sha256 != GAZE_README_SHA256:
        _fail("Gaze export README differs from the owner-frozen SHA-256")
    return {
        "documented_method_scope": "per_frame_no_future_method_evidence_only",
        "repository_readme": readme.to_dict(),
        "gaze_lle_official_method_reference": {
            "tag": GAZE_LLE_METHOD_TAG,
            "commit": GAZE_LLE_METHOD_COMMIT,
        },
        "upstream_exact_regeneration_available": False,
        "implementation_counterfactual_causality_verified": False,
        "unrecorded_actual_export_details": list(UPSTREAM_REGENERATION_GAPS),
    }


def _extract_consumed_bindings(report: Mapping[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    def visit(value: object, pointer: str) -> None:
        if isinstance(value, Mapping):
            if {"path", "sha256", "size_bytes"} <= set(value):
                path_value = value.get("path")
                digest = value.get("sha256")
                size = value.get("size_bytes")
                if (
                    type(path_value) is not str
                    or not Path(path_value).is_absolute()
                    or not _is_sha256(digest)
                    or type(size) is not int
                    or size < 0
                ):
                    _fail(f"preflight consumed binding is malformed at {pointer}")
                records.append(
                    {
                        "report_pointer": pointer,
                        "path": str(Path(path_value).resolve()),
                        "sha256": digest,
                        "size_bytes": size,
                    }
                )
            for key, nested in value.items():
                child = f"{pointer}/{key}" if pointer else str(key)
                visit(nested, child)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{pointer}/{index}")

    visit(report, "")
    pointers = {str(record["report_pointer"]) for record in records}
    missing = sorted(_REQUIRED_PREFLIGHT_BINDING_POINTERS - pointers)
    if missing:
        _fail(f"preflight report omits required consumed bindings: {missing}")
    records.sort(key=lambda record: (str(record["report_pointer"]), str(record["path"])))
    for record in records:
        _payload, actual = _stable_read(
            str(record["path"]),
            f"preflight consumed file {record['report_pointer']}",
        )
        if (
            actual.sha256 != record["sha256"]
            or actual.size_bytes != record["size_bytes"]
            or str(actual.path) != record["path"]
        ):
            _fail(
                "preflight consumed file differs from its report binding at "
                f"{record['report_pointer']}"
            )
    return records


def _formal_base_paths(root: Path, scene: str) -> tuple[Path, Path]:
    base_root = (root / "data" / "processed" / "formal_a_h_ah_v1").resolve()
    csv_path = (base_root / f"{scene}.csv").resolve()
    provenance_path = (base_root / f"{scene}.provenance.json").resolve()
    if csv_path.parent != base_root or provenance_path.parent != base_root:
        _fail("formal base path escaped its code-owned repository directory")
    return csv_path, provenance_path


def _load_formal_base(
    root: Path,
    scene: str,
) -> tuple[pd.DataFrame, dict[str, object], np.ndarray]:
    start, end = EXACT_EIGHT_TICK_RANGES[scene]
    expected_rows = end - start
    expected_group = EXACT_EIGHT_GROUP_BY_SCENE[scene]
    csv_path, provenance_path = _formal_base_paths(root, scene)
    csv_payload, csv_binding = _stable_read(csv_path, f"{scene} formal schema-v2 base")
    provenance_payload, provenance_binding = _stable_read(
        provenance_path,
        f"{scene} formal base provenance",
    )
    provenance = _strict_json(provenance_payload, f"{scene} formal base provenance")
    frame = _parse_csv(csv_payload, f"{scene} formal schema-v2 base")
    try:
        validate_multiparty_dataframe(frame, feature_schema=FEATURE_SCHEMA_V2)
    except (TypeError, ValueError) as exc:
        raise GazePosthocGateError(
            f"{scene} formal schema-v2 base failed validation: {exc}"
        ) from exc
    parsed_dtypes = {column: str(frame[column].dtype) for column in frame}
    tick_record = provenance.get("output_tick_range")
    if (
        provenance.get("artifact_kind") != "vap_real_assembled_session"
        or provenance.get("format_version") != 2
        or provenance.get("data_kind") != "real"
        or provenance.get("session_id") != scene
        or provenance.get("group_id") != expected_group
        or provenance.get("trainable") is not True
        or provenance.get("csv_sha256") != csv_binding.sha256
        or provenance.get("columns") != list(frame.columns)
        or provenance.get("dtypes") != parsed_dtypes
        or provenance.get("n_rows") != expected_rows
        or provenance.get("n_frames") != expected_rows
        or tick_record != {"start_inclusive": start, "end_exclusive": end}
    ):
        _fail(f"{scene} formal base/provenance contract is inconsistent")
    if len(frame) != expected_rows:
        _fail(f"{scene} formal base row count differs from its frozen tick range")
    if not frame["session_id"].eq(scene).all():
        _fail(f"{scene} formal base contains a different session_id")
    ticks = np.arange(start, end, dtype=np.int64)
    expected_timestamps = ticks.astype(np.float64) / 25.0
    timestamps = pd.to_numeric(frame["timestamp_s"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.array_equal(timestamps, expected_timestamps):
        _fail(f"{scene} formal base timestamps are not the exact frozen 25-Hz ticks")
    return (
        frame,
        {
            "csv": csv_binding.to_dict(),
            "provenance": provenance_binding.to_dict(),
            "artifact_kind": provenance["artifact_kind"],
            "format_version": provenance["format_version"],
            "data_kind": provenance["data_kind"],
            "session_id": scene,
            "group_id": expected_group,
            "n_rows": expected_rows,
            "tick_start_inclusive": start,
            "tick_end_exclusive": end,
            "columns": list(frame.columns),
            "dtypes": parsed_dtypes,
        },
        ticks,
    )


def _coverage_from_evidence(evidence: pd.DataFrame) -> dict[str, object]:
    slots: dict[str, object] = {}
    for slot in SLOTS:
        rows = evidence.loc[evidence["slot"].eq(slot)]
        valid = rows["gaze_robot_valid"].astype(bool)
        values = pd.to_numeric(rows["gaze_robot"], errors="coerce")
        present = rows["slot_present"].astype(bool)
        slots[slot] = {
            "n_rows": int(len(rows)),
            "present_ticks": int(present.sum()),
            "valid_ticks": int(valid.sum()),
            "invalid_ticks": int((~valid).sum()),
            "positive_ticks": int(values[valid].eq(1.0).sum()),
            "negative_ticks": int(values[valid].eq(0.0).sum()),
        }
    return {"slots": slots}


def _coverage_from_output(frame: pd.DataFrame) -> dict[str, object]:
    slots: dict[str, object] = {}
    for index, slot in enumerate(SLOTS, start=1):
        valid_column = participant_column(index, "gaze_robot_valid")
        value_column = participant_column(index, "gaze_robot")
        present_column = participant_column(index, "present")
        valid = frame[valid_column].eq(1.0)
        values = pd.to_numeric(frame[value_column], errors="coerce")
        if values.isna().any() or values[~valid].ne(0.0).any():
            _fail(f"{slot} invalid Gaze values must be explicit zero beside validity=0")
        if not values[valid].isin((0.0, 1.0)).all():
            _fail(f"{slot} valid Gaze values must be binary")
        slots[slot] = {
            "n_rows": int(len(frame)),
            "present_ticks": int(frame[present_column].eq(1.0).sum()),
            "valid_ticks": int(valid.sum()),
            "invalid_ticks": int((~valid).sum()),
            "positive_ticks": int(values[valid].eq(1.0).sum()),
            "negative_ticks": int(values[valid].eq(0.0).sum()),
        }
    if (
        slots["P1"]["valid_ticks"] != 0
        or frame["P1_gaze_robot_valid"].ne(0.0).any()
        or frame["P1_gaze_robot"].ne(0.0).any()
    ):
        _fail("P1/R must remain validity=0 and value=0 in the schema-v3 output")
    return {"slots": slots}


def _validate_preflight_result(
    scene: str,
    result: gaze_source_readiness.GazeSourcePreflight,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if (
        type(result) is not gaze_source_readiness.GazeSourcePreflight
        or result.scene != scene
        or type(result.report) is not dict
        or type(result.evidence_df) is not pd.DataFrame
    ):
        _fail(f"{scene} sealed Gaze preflight returned the wrong result type")
    report = result.report
    analysis = report.get("analysis_base")
    evidence_record = report.get("evidence")
    time_mapping = report.get("time_mapping")
    limits = report.get("scientific_limits")
    start, end = analysis_base.EXPECTED_TICK_RANGES[scene]
    if (
        report.get("artifact_kind") != gaze_source_readiness.ARTIFACT_KIND
        or report.get("format_version") != gaze_source_readiness.FORMAT_VERSION
        or report.get("scene") != scene
        or report.get("assembly_eligible") is not False
        or report.get("counted_input_ready") is not False
        or report.get("writer_available") is not False
        or report.get("artifact_or_feature_published") is not False
        or not isinstance(analysis, Mapping)
        or analysis.get("tick_start_inclusive") != start
        or analysis.get("tick_end_exclusive") != end
        or analysis.get("n_ticks") != end - start
        or not isinstance(evidence_record, Mapping)
        or evidence_record.get("n_rows") != (end - start) * len(SLOTS)
        or not isinstance(time_mapping, Mapping)
        or time_mapping.get("past_only") is not True
        or time_mapping.get("nearest_frame") is not False
        or time_mapping.get("interpolation") is not False
        or time_mapping.get("forward_fill") is not False
        or time_mapping.get("extrapolation") is not False
        or not isinstance(limits, Mapping)
        or limits.get("robot_slot_gaze_observed") is not False
        or limits.get("model_input_promoted") is not False
    ):
        _fail(f"{scene} source preflight boundary/tick contract is inconsistent")
    evidence = result.evidence_df.copy(deep=True)
    if (
        evidence_record.get("columns") != list(evidence.columns)
        or evidence_record.get("parsed_dtypes")
        != {column: str(evidence[column].dtype) for column in evidence}
        or len(evidence) != (end - start) * len(SLOTS)
    ):
        _fail(f"{scene} source preflight evidence metadata differs from its frame")
    required = {
        "session_id",
        "tick",
        "timestamp_s",
        "participant",
        "slot",
        "slot_present",
        "gaze_robot_valid",
        "gaze_robot",
    }
    if not required <= set(evidence):
        _fail(f"{scene} source preflight evidence omits required nullable fields")
    if evidence.duplicated(["tick", "slot"]).any():
        _fail(f"{scene} source preflight evidence has duplicate tick/slot keys")
    participant_by_slot = {"P1": "R", "P2": "A", "P3": "B", "P4": "C"}
    ticks = np.arange(start, end, dtype=np.int64)
    expected_timestamps = ticks.astype(np.float64) / 25.0
    slot_presence = analysis.get("slot_presence")
    if (
        not isinstance(slot_presence, Mapping)
        or set(slot_presence) != set(SLOTS)
        or any(type(slot_presence[slot]) is not bool for slot in SLOTS)
    ):
        _fail(f"{scene} source preflight slot-presence contract is malformed")
    for slot in SLOTS:
        rows = evidence.loc[evidence["slot"].eq(slot)].sort_values("tick")
        row_ticks = pd.to_numeric(rows["tick"], errors="coerce").to_numpy()
        row_timestamps = pd.to_numeric(rows["timestamp_s"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        expected_presence = np.full(len(ticks), slot_presence[slot], dtype=bool)
        if (
            len(rows) != len(ticks)
            or not rows["session_id"].eq(scene).all()
            or not rows["participant"].eq(participant_by_slot[slot]).all()
            or not np.array_equal(row_ticks, ticks)
            or not np.array_equal(row_timestamps, expected_timestamps)
            or not np.array_equal(rows["slot_present"].astype(bool), expected_presence)
        ):
            _fail(f"{scene} {slot} preflight coverage drifted from the formal base")
        valid = rows["gaze_robot_valid"].astype(bool)
        values = pd.to_numeric(rows["gaze_robot"], errors="coerce")
        if values[~valid].notna().any() or values[valid].isna().any():
            _fail(f"{scene} {slot} nullable Gaze value/validity semantics are invalid")
        if not values[valid].isin((0.0, 1.0)).all():
            _fail(f"{scene} {slot} valid Gaze evidence is not binary")
        if slot == "P1" and (valid.any() or values.notna().any()):
            _fail(f"{scene} P1/R source Gaze must remain invalid and missing")
    consumed = _extract_consumed_bindings(report)
    return (
        evidence,
        {
            "artifact_kind": report["artifact_kind"],
            "format_version": report["format_version"],
            "scene": scene,
            "tick_start_inclusive": start,
            "tick_end_exclusive": end,
            "n_ticks": end - start,
            "report_sha256": _sha256_json(report),
            "consumed_bindings": consumed,
            "source_boundary": {
                "assembly_eligible": False,
                "counted_input_ready": False,
                "writer_available": False,
                "artifact_or_feature_published": False,
                "model_input_promoted": False,
            },
            "past_only_time_mapping": True,
        },
    )


def _validate_included_preflight_against_base(
    scene: str,
    base: pd.DataFrame,
    ticks: np.ndarray,
    evidence: pd.DataFrame,
) -> None:
    if (
        scene not in EXACT_EIGHT_TICK_RANGES
        or EXACT_EIGHT_TICK_RANGES[scene]
        != analysis_base.EXPECTED_TICK_RANGES[scene]
    ):
        _fail(f"{scene} is not in the authority-derived exact-eight profile")
    for index, slot in enumerate(SLOTS, start=1):
        rows = evidence.loc[evidence["slot"].eq(slot)].sort_values("tick")
        row_ticks = pd.to_numeric(rows["tick"], errors="coerce").to_numpy()
        expected_presence = base[participant_column(index, "present")].eq(1.0).to_numpy()
        if (
            not np.array_equal(row_ticks, ticks)
            or not np.array_equal(rows["slot_present"].astype(bool), expected_presence)
        ):
            _fail(f"{scene} {slot} preflight coverage drifted from the formal base")


def _audit_full_preflight_matrix() -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    try:
        results = {
            scene: gaze_source_readiness.audit_gaze_source_preflight(scene)
            for scene in analysis_base.EXPECTED_TICK_RANGES
        }
        gaze_source_readiness.validate_frozen_preflight_matrix(results)
    except gaze_source_readiness.GazeSourceReadinessError as exc:
        raise GazePosthocGateError(
            "full ten-scene Gaze source preflight replay failed"
        ) from exc
    evidence_by_scene: dict[str, pd.DataFrame] = {}
    records: dict[str, dict[str, object]] = {}
    for scene, result in results.items():
        evidence, record = _validate_preflight_result(scene, result)
        evidence_by_scene[scene] = evidence
        records[scene] = record
    matrix_record: dict[str, object] = {
        "artifact_kind": gaze_source_readiness.MATRIX_ARTIFACT_KIND,
        "format_version": gaze_source_readiness.FORMAT_VERSION,
        "frozen_matrix_replay_validated": True,
        "scene_inventory": list(analysis_base.EXPECTED_TICK_RANGES),
        "n_scenes": len(results),
        "total_ticks": analysis_base.EXPECTED_TOTAL_ROWS,
        "total_evidence_rows": analysis_base.EXPECTED_TOTAL_ROWS * len(SLOTS),
        "source_boundary": {
            "assembly_eligible": False,
            "counted_input_ready": False,
            "writer_available": False,
            "artifact_or_feature_published": False,
        },
        "scenes": records,
    }
    return evidence_by_scene, records, matrix_record


def _build_real_scene(
    root: Path,
    scene: str,
    output_path: Path,
    method_evidence: Mapping[str, object],
    evidence: pd.DataFrame,
    preflight_record: Mapping[str, object],
) -> tuple[bytes, dict[str, object], pd.DataFrame]:
    base, base_record, ticks = _load_formal_base(root, scene)
    _validate_included_preflight_against_base(scene, base, ticks, evidence)
    output = build_schema_v3_frame(
        base,
        evidence[
            [
                "session_id",
                "timestamp_s",
                "slot",
                "gaze_robot_valid",
                "gaze_robot",
            ]
        ],
    )
    source_coverage = _coverage_from_evidence(evidence)
    output_coverage = _coverage_from_output(output)
    if source_coverage != output_coverage:
        _fail(f"{scene} Gaze coverage changed while joining the schema-v3 table")
    output_payload = _csv_bytes(output)
    parsed_output = _parse_csv(output_payload, f"{scene} schema-v3 output")
    try:
        validate_multiparty_dataframe(
            parsed_output,
            feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY,
        )
    except (TypeError, ValueError) as exc:
        raise GazePosthocGateError(
            f"{scene} serialized schema-v3 output failed validation: {exc}"
        ) from exc
    output_binding = _binding_for_payload(output_path, output_payload)
    start, end = EXACT_EIGHT_TICK_RANGES[scene]
    sidecar: dict[str, object] = {
        "artifact_kind": REAL_SCENE_ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": scene,
        "group_id": EXACT_EIGHT_GROUP_BY_SCENE[scene],
        "permitted_use": REAL_PERMITTED_USE,
        "posthoc_exploratory_training_ready": True,
        "formal_outer_runner_admitted": False,
        "assembly_eligible": False,
        "production_eligible": False,
        "counted_eligible": False,
        "upstream_exact_regeneration_available": False,
        "implementation_counterfactual_causality_verified": False,
        "method_evidence": dict(method_evidence),
        "base": base_record,
        "source_preflight": dict(preflight_record),
        "schema": _schema_contract(),
        "coverage": {
            **source_coverage,
            "no_coverage_drift": True,
            "invalid_source_missing_output_value_zero_with_validity_mask": True,
            "p1_source_missing_output_value_zero_with_validity_zero": True,
        },
        "output": {
            **output_binding.to_dict(),
            "columns": list(parsed_output.columns),
            "dtypes": {
                column: str(parsed_output[column].dtype) for column in parsed_output
            },
            "n_rows": len(parsed_output),
            "tick_start_inclusive": start,
            "tick_end_exclusive": end,
            "group_id": EXACT_EIGHT_GROUP_BY_SCENE[scene],
        },
    }
    return output_payload, sidecar, parsed_output


def _scene_provenance_path(output_dir: Path, scene: str) -> Path:
    return output_dir / f"{scene}{REAL_SCENE_PROVENANCE_SUFFIX}"


def _matrix_manifest(
    output_dir: Path,
    method_evidence: Mapping[str, object],
    scene_records: Mapping[str, Mapping[str, object]],
    preflight_matrix: Mapping[str, object],
) -> dict[str, object]:
    expected_scenes = list(EXACT_EIGHT_TICK_RANGES)
    authority_scenes = [
        scene
        for scene in analysis_base.EXPECTED_TICK_RANGES
        if scene not in EXCLUDED_OWNER_HOLD_SCENES
    ]
    if (
        expected_scenes != authority_scenes
        or list(scene_records) != expected_scenes
        or preflight_matrix.get("scene_inventory")
        != list(analysis_base.EXPECTED_TICK_RANGES)
        or preflight_matrix.get("frozen_matrix_replay_validated") is not True
    ):
        _fail("Gaze post-hoc matrix does not preserve the frozen exact-eight order")
    total_rows = sum(
        int(scene_records[scene]["n_rows"]) for scene in expected_scenes
    )
    groups = sorted(
        {str(scene_records[scene]["group_id"]) for scene in expected_scenes}
    )
    if total_rows != EXACT_EIGHT_TOTAL_ROWS or groups != list(EXACT_EIGHT_GROUPS):
        _fail("Gaze post-hoc matrix row/group totals differ from the frozen contract")
    return {
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
        "method_evidence": dict(method_evidence),
        "source_corpus_scene_count": len(analysis_base.EXPECTED_TICK_RANGES),
        "selected_scene_count": len(expected_scenes),
        "exact_scene_inventory": expected_scenes,
        "excluded_owner_hold_scenes": dict(EXCLUDED_OWNER_HOLD_SCENES),
        "total_rows": total_rows,
        "group_ids": groups,
        "n_groups": len(groups),
        "schema": _schema_contract(),
        "source_preflight_matrix": dict(preflight_matrix),
        "output_directory": str(output_dir.resolve()),
        "output_layout": {
            "scene_csv": "<scene>.csv",
            "scene_provenance": f"<scene>{REAL_SCENE_PROVENANCE_SUFFIX}",
            "matrix_manifest": REAL_MATRIX_MANIFEST_NAME,
        },
        "scenes": {scene: dict(scene_records[scene]) for scene in expected_scenes},
    }


def _scene_matrix_record(
    scene: str,
    sidecar: Mapping[str, object],
    provenance_binding: _Binding,
) -> dict[str, object]:
    output = sidecar.get("output")
    if not isinstance(output, Mapping):
        _fail(f"{scene} sidecar has no output binding")
    return {
        "group_id": EXACT_EIGHT_GROUP_BY_SCENE[scene],
        "n_rows": EXACT_EIGHT_TICK_RANGES[scene][1] - EXACT_EIGHT_TICK_RANGES[scene][0],
        "tick_start_inclusive": EXACT_EIGHT_TICK_RANGES[scene][0],
        "tick_end_exclusive": EXACT_EIGHT_TICK_RANGES[scene][1],
        "csv": {
            key: output[key] for key in ("path", "sha256", "size_bytes")
        },
        "provenance": provenance_binding.to_dict(),
        "source_preflight_report_sha256": sidecar["source_preflight"]["report_sha256"],
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def _payload_identity(payload: bytes) -> tuple[str, int]:
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _published_tree_matches_created_payloads(
    output_dir: Path,
    created_payloads: Mapping[str, tuple[str, int]],
) -> bool:
    if output_dir.is_symlink() or not output_dir.is_dir():
        return False
    try:
        paths = list(output_dir.iterdir())
        if (
            {path.name for path in paths} != _expected_real_matrix_files()
            or set(created_payloads) != _expected_real_matrix_files()
        ):
            return False
        for name, (expected_sha256, expected_size) in created_payloads.items():
            _payload, binding = _stable_read(
                output_dir / name,
                f"newly published rollback candidate {name}",
            )
            if (
                binding.sha256 != expected_sha256
                or binding.size_bytes != expected_size
            ):
                return False
    except (OSError, GazePosthocGateError):
        return False
    return True


def publish_real_posthoc_matrix(
    output_dir: str | Path,
) -> RealGazePosthocMatrixArtifact:
    """Create the fixed exact-eight exploratory Gaze matrix exactly once.

    The caller controls only the new output directory.  Scene membership,
    formal bases, source preflight authority, method evidence and all
    eligibility boundaries are code-owned.
    """

    supplied = Path(output_dir)
    if supplied.is_symlink():
        _fail("real Gaze post-hoc output directory must not be a symlink")
    output_path = supplied.resolve()
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite real Gaze post-hoc directory: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = _repository_root()
    (
        preflight_evidence,
        preflight_records,
        preflight_matrix,
    ) = _audit_full_preflight_matrix()
    method_evidence = _load_method_evidence(root)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".gaze-posthoc-schema-v3-stage-",
            dir=output_path.parent,
        )
    ).resolve()
    scene_records: dict[str, Mapping[str, object]] = {}
    created_payloads: dict[str, tuple[str, int]] = {}
    try:
        for scene in EXACT_EIGHT_TICK_RANGES:
            final_csv = output_path / f"{scene}.csv"
            output_payload, sidecar, _frame = _build_real_scene(
                root,
                scene,
                final_csv,
                method_evidence,
                preflight_evidence[scene],
                preflight_records[scene],
            )
            sidecar_payload = _json_bytes(sidecar)
            final_provenance = _scene_provenance_path(output_path, scene)
            sidecar_binding = _binding_for_payload(final_provenance, sidecar_payload)
            _write_exclusive(staging / f"{scene}.csv", output_payload)
            _write_exclusive(
                _scene_provenance_path(staging, scene),
                sidecar_payload,
            )
            created_payloads[f"{scene}.csv"] = _payload_identity(output_payload)
            created_payloads[
                f"{scene}{REAL_SCENE_PROVENANCE_SUFFIX}"
            ] = _payload_identity(sidecar_payload)
            scene_records[scene] = _scene_matrix_record(
                scene,
                sidecar,
                sidecar_binding,
            )
        manifest = _matrix_manifest(
            output_path,
            method_evidence,
            scene_records,
            preflight_matrix,
        )
        manifest_payload = _json_bytes(manifest)
        _write_exclusive(
            staging / REAL_MATRIX_MANIFEST_NAME,
            manifest_payload,
        )
        created_payloads[REAL_MATRIX_MANIFEST_NAME] = _payload_identity(
            manifest_payload
        )
        if output_path.exists():
            raise FileExistsError(
                f"refusing to overwrite real Gaze post-hoc directory: {output_path}"
            )
        staging.rename(output_path)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    try:
        return audit_real_posthoc_matrix(output_path)
    except BaseException as audit_error:
        if _published_tree_matches_created_payloads(output_path, created_payloads):
            try:
                shutil.rmtree(output_path)
            except OSError as rollback_error:
                raise GazePosthocRecoveryError(
                    "final audit failed and the unchanged new output could not be "
                    f"removed; preserve for owner recovery: {output_path}"
                ) from rollback_error
            if output_path.exists():
                raise GazePosthocRecoveryError(
                    "final audit failed and safe rollback did not remove the complete "
                    f"new output; preserve for owner recovery: {output_path}"
                ) from audit_error
            raise
        raise GazePosthocRecoveryError(
            "final audit failed after publication, but the exact17 output tree no "
            "longer matches this call's payload ledger; it was preserved for owner "
            f"recovery: {output_path}"
        ) from audit_error


def _expected_real_matrix_files() -> set[str]:
    names = {REAL_MATRIX_MANIFEST_NAME}
    for scene in EXACT_EIGHT_TICK_RANGES:
        names.add(f"{scene}.csv")
        names.add(f"{scene}{REAL_SCENE_PROVENANCE_SUFFIX}")
    return names


def audit_real_posthoc_matrix(
    output_dir: str | Path,
) -> RealGazePosthocMatrixArtifact:
    """Strictly re-hash and replay the fixed real post-hoc matrix."""

    supplied = Path(output_dir)
    if supplied.is_symlink():
        _fail("real Gaze post-hoc directory must not be a symlink")
    output_path = supplied.resolve()
    if not output_path.is_dir():
        _fail(f"real Gaze post-hoc directory is missing: {output_path}")
    actual_names = {path.name for path in output_path.iterdir()}
    if actual_names != _expected_real_matrix_files():
        _fail("real Gaze post-hoc directory file set is not exact")
    (
        preflight_evidence,
        preflight_records,
        preflight_matrix,
    ) = _audit_full_preflight_matrix()
    manifest_path = output_path / REAL_MATRIX_MANIFEST_NAME
    manifest_payload, _manifest_binding = _stable_read(
        manifest_path,
        "real Gaze post-hoc matrix manifest",
    )
    manifest = _strict_json(
        manifest_payload,
        "real Gaze post-hoc matrix manifest",
    )
    if (
        manifest.get("artifact_kind") != REAL_MATRIX_ARTIFACT_KIND
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        _fail("real Gaze post-hoc matrix manifest kind/version is invalid")
    root = _repository_root()
    method_evidence = _load_method_evidence(root)
    scene_records: dict[str, Mapping[str, object]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for scene in EXACT_EIGHT_TICK_RANGES:
        csv_path = output_path / f"{scene}.csv"
        provenance_path = _scene_provenance_path(output_path, scene)
        output_payload, output_binding = _stable_read(
            csv_path,
            f"{scene} real Gaze post-hoc output",
        )
        provenance_payload, provenance_binding = _stable_read(
            provenance_path,
            f"{scene} real Gaze post-hoc provenance",
        )
        sidecar = _strict_json(
            provenance_payload,
            f"{scene} real Gaze post-hoc provenance",
        )
        output_record = sidecar.get("output")
        if not isinstance(output_record, Mapping):
            _fail(f"{scene} real Gaze post-hoc output binding is malformed")
        _assert_binding(
            {
                key: output_record.get(key)
                for key in ("path", "sha256", "size_bytes")
            },
            output_binding,
            f"{scene} real Gaze post-hoc output",
        )
        replay_payload, expected_sidecar, _replay_frame = _build_real_scene(
            root,
            scene,
            csv_path,
            method_evidence,
            preflight_evidence[scene],
            preflight_records[scene],
        )
        if replay_payload != output_payload:
            _fail(f"{scene} schema-v3 output failed byte-for-byte source replay")
        expected_provenance_payload = _json_bytes(expected_sidecar)
        if provenance_payload != expected_provenance_payload or sidecar != expected_sidecar:
            _fail(f"{scene} provenance failed exact canonical replay")
        parsed_output = _parse_csv(
            output_payload,
            f"{scene} audited schema-v3 output",
        )
        try:
            validate_multiparty_dataframe(
                parsed_output,
                feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY,
            )
        except (TypeError, ValueError) as exc:
            raise GazePosthocGateError(
                f"{scene} audited schema-v3 output failed validation: {exc}"
            ) from exc
        frames[scene] = parsed_output
        scene_records[scene] = _scene_matrix_record(
            scene,
            sidecar,
            provenance_binding,
        )
    expected_manifest = _matrix_manifest(
        output_path,
        method_evidence,
        scene_records,
        preflight_matrix,
    )
    if manifest != expected_manifest or manifest_payload != _json_bytes(expected_manifest):
        _fail("real Gaze post-hoc matrix manifest failed exact canonical replay")
    if sum(len(frame) for frame in frames.values()) != EXACT_EIGHT_TOTAL_ROWS:
        _fail("audited real Gaze post-hoc frame total differs from 64,299")
    return RealGazePosthocMatrixArtifact(
        output_dir=output_path,
        manifest_path=manifest_path,
        manifest=manifest,
        frames=frames,
    )


__all__ = [
    "ARTIFACT_KIND",
    "EXACT_EIGHT_GROUPS",
    "EXACT_EIGHT_GROUP_BY_SCENE",
    "EXACT_EIGHT_TICK_RANGES",
    "EXACT_EIGHT_TOTAL_ROWS",
    "EXCLUDED_OWNER_HOLD_SCENES",
    "FORMAT_VERSION",
    "GAZE_LLE_METHOD_COMMIT",
    "GAZE_LLE_METHOD_TAG",
    "GAZE_README_SHA256",
    "GazePosthocFixtureArtifact",
    "GazePosthocGateError",
    "GazePosthocRecoveryError",
    "READINESS_ARTIFACT_KIND",
    "REAL_MATRIX_ARTIFACT_KIND",
    "REAL_MATRIX_MANIFEST_NAME",
    "REAL_PERMITTED_USE",
    "REAL_PUBLICATION_BLOCKERS",
    "REAL_SCENE_ARTIFACT_KIND",
    "RealGazePosthocMatrixArtifact",
    "audit_fixture_artifact",
    "audit_real_posthoc_matrix",
    "build_schema_v3_frame",
    "current_real_publication_blockers",
    "fixture_readiness",
    "publish_fixture_artifact",
    "publish_real_posthoc_matrix",
    "require_current_real_publication_ready",
]
