"""Read-only Facial delivery evidence and readiness inventory.

This module inventories the ten repository-local Facial CSV deliveries against
the sealed identity and canonical analysis-grid authorities.  It deliberately
does not derive MAR, aggregate duplicate detections, interpolate sparse rows,
write feature files, or expose an assembly/publishing path.

The upstream README says that a 98-point detector was used internally, but the
delivered ``openface_features.csv`` files contain only sixteen scalar columns.
Those bytes therefore cannot reproduce MAR.  ``mouth_motion.csv`` is recorded
as disabled evidence and is never read as a MAR substitute.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hri_turn_taking.data import analysis_base, identity_annotation
from hri_turn_taking.data import hpe_transcript_crosswalk as crosswalk_module

ARTIFACT_KIND = "vap_real_facial_source_readiness"
MATRIX_ARTIFACT_KIND = "vap_real_facial_source_readiness_matrix"
FORMAT_VERSION = 1
FRAME_RATE_HZ = 25
OUTPUT_ROLE = "read_only_nullable_evidence_inventory_only"
PARTICIPANT_TO_SLOT = {"R": "P1", "A": "P2", "B": "P3", "C": "P4"}
EXPECTED_OPENFACE_COLUMNS = (
    "frame_index",
    "timestamp_sec",
    "person_idx",
    "match_iou",
    "face_conf",
    "emotion_argmax",
    "gaze_yaw",
    "gaze_pitch",
    "au_0",
    "au_1",
    "au_2",
    "au_3",
    "au_4",
    "au_5",
    "au_6",
    "au_7",
)
EXPECTED_MOUTH_MOTION_COLUMNS = ("frame_index", "person_idx", "mouth_motion")
AU_COLUMNS = tuple(f"au_{index}" for index in range(8))
EMOTION_COLUMNS = ("emotion_argmax",)
GAZE_COLUMNS = ("gaze_yaw", "gaze_pitch")
QUALITY_COLUMNS = ("match_iou", "face_conf")
_SCENE_RE = re.compile(r"^[0-9]{2}_[0-9]{2}__[0-9]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class FacialSourceReadinessError(ValueError):
    """A canonical source, identity, or analysis authority failed closed."""


@dataclass(frozen=True, kw_only=True)
class FacialSourceAuthority:
    """Code-owned roots for the repository-local read-only inventory."""

    repository_root: Path
    facial_root: Path
    analysis_base_root: Path
    crosswalk_inventory: Path
    crosswalk_root: Path
    identity_root: Path
    enforce_frozen_inventory: bool = True

    def __post_init__(self) -> None:
        for field in (
            "repository_root",
            "facial_root",
            "analysis_base_root",
            "crosswalk_inventory",
            "crosswalk_root",
            "identity_root",
        ):
            value = getattr(self, field)
            if not isinstance(value, Path):
                raise FacialSourceReadinessError(f"{field} must be a pathlib.Path")
            object.__setattr__(self, field, _lexical_absolute(value))
        if type(self.enforce_frozen_inventory) is not bool:
            raise FacialSourceReadinessError("enforce_frozen_inventory must be an exact boolean")


@dataclass(frozen=True, kw_only=True)
class _SceneDocuments:
    analysis: pd.DataFrame
    analysis_binding: dict[str, Any]
    analysis_provenance: dict[str, Any]
    sealed_identity: dict[str, Any]
    confirmed_bridge: pd.DataFrame
    source_fps: Fraction
    accepted_window: tuple[int, int]


def default_source_authority() -> FacialSourceAuthority:
    root = Path(__file__).resolve().parents[3]
    return FacialSourceAuthority(
        repository_root=root,
        facial_root=root / "facial_features_export",
        analysis_base_root=root / "data" / "processed" / "analysis_base",
        crosswalk_inventory=(root / "configs" / "hpe_transcript_crosswalk_inventory.json"),
        crosswalk_root=(root / "data" / "annotations" / "hpe_transcript_crosswalk"),
        identity_root=root / "data" / "annotations" / "identity",
    )


def _fail(message: str) -> None:
    raise FacialSourceReadinessError(message)


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _file_identity(path: Path, label: str) -> tuple[int, int, int, int, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise FacialSourceReadinessError(f"{label} cannot be lstat'ed: {exc}") from exc
    attributes = int(getattr(info, "st_file_attributes", 0))
    if stat.S_ISLNK(info.st_mode) or attributes & _REPARSE_POINT or not stat.S_ISREG(info.st_mode):
        _fail(f"{label} must be a regular non-link non-reparse file")
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        attributes,
    )


def _secure_existing_file(path: Path, *, root: Path, label: str) -> Path:
    lexical_root = _lexical_absolute(root)
    lexical_path = _lexical_absolute(path)
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise FacialSourceReadinessError(f"{label} escapes its authority root") from exc
    current = lexical_root
    chain = [current]
    for part in relative.parts:
        current = current / part
        chain.append(current)
    for index, candidate in enumerate(chain):
        try:
            info = candidate.lstat()
        except OSError as exc:
            raise FacialSourceReadinessError(
                f"{label} path component is unavailable: {candidate}"
            ) from exc
        attributes = int(getattr(info, "st_file_attributes", 0))
        if stat.S_ISLNK(info.st_mode) or attributes & _REPARSE_POINT:
            _fail(f"{label} path contains a link/reparse point: {candidate}")
        leaf = index == len(chain) - 1
        if leaf and not stat.S_ISREG(info.st_mode):
            _fail(f"{label} leaf is not a regular file")
        if not leaf and not stat.S_ISDIR(info.st_mode):
            _fail(f"{label} ancestor is not a directory: {candidate}")
    return lexical_path


def _stable_read(path: Path, *, root: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    secured = _secure_existing_file(path, root=root, label=label)
    before = _file_identity(secured, label)
    try:
        raw = secured.read_bytes()
    except OSError as exc:
        raise FacialSourceReadinessError(f"{label} is unreadable: {exc}") from exc
    after = _file_identity(secured, label)
    if before != after or len(raw) != before[2]:
        _fail(f"{label} changed during its stable read")
    return raw, {
        "path": str(secured),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FacialSourceReadinessError(f"{label} is not UTF-8") from exc

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda token: _fail(f"{label} contains forbidden JSON constant {token}"),
        )
    except json.JSONDecodeError as exc:
        raise FacialSourceReadinessError(f"{label} is invalid JSON") from exc
    if type(value) is not dict:
        _fail(f"{label} must contain one JSON object")
    return value


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _exact_path(value: Any, expected: Path, label: str) -> None:
    if type(value) is not str or _lexical_absolute(value) != _lexical_absolute(expected):
        _fail(f"{label} is not the code-owned canonical path")


def _binding_matches(
    declared: Any,
    actual: dict[str, Any],
    *,
    expected_path: Path,
    label: str,
) -> None:
    if type(declared) is not dict:
        _fail(f"{label} binding is not an exact object")
    _exact_path(declared.get("path"), expected_path, f"{label}.path")
    digest = declared.get("sha256")
    if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
        _fail(f"{label}.sha256 is malformed")
    if digest != actual["sha256"]:
        _fail(f"{label} differs from its sealed SHA-256")
    if "size_bytes" in declared and declared["size_bytes"] != actual["size_bytes"]:
        _fail(f"{label} differs from its sealed byte size")


def _read_csv_bytes(
    raw: bytes,
    label: str,
    *,
    dtype: dict[str, Any] | None = None,
) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(raw), dtype=dtype)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise FacialSourceReadinessError(f"{label} is not a readable CSV: {exc}") from exc


def _normalised_dtype(value: str) -> str:
    if value in {"object", "str", "string"}:
        return "text"
    if re.fullmatch(r"int(8|16|32|64)", value):
        return "integer"
    if re.fullmatch(r"float(16|32|64)", value):
        return "float"
    if value in {"bool", "boolean"}:
        return "boolean"
    return value


def _validate_analysis_dtypes(recorded: Any, frame: pd.DataFrame) -> None:
    columns = list(frame.columns)
    actual = {column: str(dtype) for column, dtype in frame.dtypes.items()}
    if (
        type(recorded) is not dict
        or set(recorded) != set(columns)
        or any(type(key) is not str for key in recorded)
        or any(type(value) is not str or not value for value in recorded.values())
        or any(
            _normalised_dtype(recorded[column]) != _normalised_dtype(actual[column])
            for column in columns
        )
    ):
        _fail("canonical analysis parsed dtypes differ from provenance")


def _schema_binding_matches(
    declared: Any,
    frame: pd.DataFrame,
    *,
    label: str,
) -> None:
    if type(declared) is not dict:
        _fail(f"{label} binding is malformed")
    if declared.get("columns") != list(frame.columns):
        _fail(f"{label} ordered columns differ from their sealed binding")
    if declared.get("n_rows") != len(frame):
        _fail(f"{label} row count differs from its sealed binding")
    recorded = declared.get("parsed_dtypes")
    actual = [str(value) for value in frame.dtypes]
    if (
        type(recorded) is not list
        or len(recorded) != len(actual)
        or any(type(value) is not str or not value for value in recorded)
        or [_normalised_dtype(value) for value in recorded]
        != [_normalised_dtype(value) for value in actual]
    ):
        _fail(f"{label} parsed dtypes differ from their sealed binding")


def _validate_scene(scene: str, authority: FacialSourceAuthority) -> None:
    if type(scene) is not str or _SCENE_RE.fullmatch(scene) is None:
        _fail("scene must be one normalised date__session identifier")
    if authority.enforce_frozen_inventory and scene not in analysis_base.EXPECTED_TICK_RANGES:
        _fail(f"scene is outside the frozen ten-scene inventory: {scene!r}")


def _source_paths(scene: str, authority: FacialSourceAuthority) -> tuple[Path, Path]:
    _validate_scene(scene, authority)
    dataset, session = scene.rsplit("__", maxsplit=1)
    directory = authority.facial_root / dataset / f"session_{session}"
    return directory / "openface_features.csv", directory / "mouth_motion.csv"


def _load_inventory(authority: FacialSourceAuthority) -> dict[str, Any]:
    try:
        inventory = crosswalk_module.load_sealed_crosswalk_inventory(
            authority.crosswalk_inventory,
            repository_root=authority.repository_root,
        )
    except (OSError, ValueError) as exc:
        raise FacialSourceReadinessError(
            f"sealed crosswalk inventory replay failed: {exc}"
        ) from exc
    scenes = inventory.get("scenes")
    if type(scenes) is not dict:
        _fail("sealed crosswalk inventory has no scene map")
    if authority.enforce_frozen_inventory and set(scenes) != set(
        analysis_base.EXPECTED_TICK_RANGES
    ):
        _fail("sealed identity inventory differs from the frozen ten-scene matrix")
    return inventory


def _load_sealed_identity(
    scene: str,
    authority: FacialSourceAuthority,
    inventory: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, Fraction, tuple[int, int]]:
    record = inventory["scenes"].get(scene)
    if type(record) is not dict:
        _fail(f"scene {scene!r} is absent from the sealed identity inventory")
    manifest_path = authority.crosswalk_root / scene / crosswalk_module.MANIFEST_NAME
    raw_crosswalk, crosswalk_actual = _stable_read(
        manifest_path,
        root=authority.crosswalk_root,
        label="sealed crosswalk manifest",
    )
    file_binding = record.get("files", {}).get(crosswalk_module.MANIFEST_NAME)
    if type(file_binding) is not dict:
        _fail("sealed inventory lacks the scene crosswalk-manifest binding")
    if (
        file_binding.get("sha256") != crosswalk_actual["sha256"]
        or file_binding.get("size_bytes") != crosswalk_actual["size_bytes"]
    ):
        _fail("crosswalk manifest differs from its sealed inventory binding")
    crosswalk = _strict_json(raw_crosswalk, "sealed crosswalk manifest")
    if (
        crosswalk.get("artifact_kind") != crosswalk_module.ARTIFACT_KIND
        or crosswalk.get("format_version") != crosswalk_module.FORMAT_VERSION
        or crosswalk.get("scene") != scene
        or crosswalk.get("state") != "finalized"
    ):
        _fail("sealed crosswalk manifest has unsupported identity/state semantics")
    accepted = crosswalk.get("accepted_window")
    record_window = record.get("accepted_window")
    if type(accepted) is not dict or not _json_equal(accepted, record_window):
        _fail("sealed scene accepted window differs from its inventory")
    start = accepted.get("local_start_inclusive")
    end = accepted.get("local_end_exclusive")
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        _fail("sealed accepted local frame window is malformed")

    gaze = crosswalk.get("sources", {}).get("gaze")
    if type(gaze) is not dict:
        _fail("sealed crosswalk lacks the Gaze identity source binding")
    identity_dir = authority.identity_root / scene
    identity_path = identity_dir / "identity_annotations.json"
    observations_path = identity_dir / "tracklet_observations.csv"
    raw_identity, identity_actual = _stable_read(
        identity_path,
        root=authority.identity_root,
        label="sealed identity manifest",
    )
    _binding_matches(
        gaze.get("manifest"),
        identity_actual,
        expected_path=identity_path,
        label="sealed identity manifest",
    )
    identity = _strict_json(raw_identity, "sealed identity manifest")
    try:
        loaded_identity = identity_annotation.load_annotation_manifest(identity_path)
    except (OSError, ValueError) as exc:
        raise FacialSourceReadinessError(
            f"sealed identity manifest structure failed replay: {exc}"
        ) from exc
    if not _json_equal(identity, loaded_identity):
        _fail("identity manifest parse changed exact JSON types or values")
    group, scenario = scene.rsplit("__", maxsplit=1)
    declared_video = identity.get("source", {}).get("declared_video")
    scope = identity.get("annotation_scope")
    if (
        identity.get("dataset") != group
        or str(identity.get("session")) != scenario
        or type(declared_video) is not dict
        or type(scope) is not dict
        or scope.get("gaze_local_start_frame_inclusive") != start
        or scope.get("gaze_local_end_frame_exclusive") != end
        or declared_video.get("camera_dir") != accepted.get("camera")
    ):
        _fail("sealed identity manifest scene/camera/window binding changed")
    fps_value = declared_video.get("fps")
    if (
        isinstance(fps_value, bool)
        or not isinstance(fps_value, (int, float))
        or not math.isfinite(float(fps_value))
        or float(fps_value) <= 0
    ):
        _fail("sealed identity source FPS is malformed")
    source_fps = Fraction(str(fps_value))

    raw_observations, observations_actual = _stable_read(
        observations_path,
        root=authority.identity_root,
        label="sealed tracklet observations",
    )
    observations_declared = gaze.get("tracklet_observations")
    _binding_matches(
        observations_declared,
        observations_actual,
        expected_path=observations_path,
        label="sealed tracklet observations",
    )
    observations = _read_csv_bytes(
        raw_observations,
        "sealed tracklet observations",
        dtype={"source_person_idx": str, "tracklet_id": str},
    )
    _schema_binding_matches(
        observations_declared,
        observations,
        label="sealed tracklet observations",
    )
    identity_observations_binding = identity.get("tracklet_observations")
    _binding_matches(
        identity_observations_binding,
        observations_actual,
        expected_path=observations_path,
        label="identity-manifest tracklet observations",
    )
    _schema_binding_matches(
        identity_observations_binding,
        observations,
        label="identity-manifest tracklet observations",
    )
    required_observation_columns = {
        "observation_id",
        "frame_index",
        "source_kind",
        "source_person_idx",
        "tracklet_id",
    }
    if not required_observation_columns.issubset(observations.columns):
        _fail("sealed tracklet observations omit exact identity keys")
    for column in ("observation_id", "frame_index"):
        values = pd.to_numeric(observations[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            _fail(f"sealed tracklet {column} is not finite integer-valued")
        observations[column] = values.astype(np.int64)
    if not np.array_equal(
        observations["observation_id"].to_numpy(dtype=np.int64),
        np.arange(len(observations), dtype=np.int64),
    ):
        _fail("sealed observation_id is not the ordered 0..N-1 sequence")
    annotations = identity.get("annotations")
    if type(annotations) is not list or not annotations:
        _fail("sealed identity manifest has no annotations")
    annotation_ids = [item.get("tracklet_id") for item in annotations if type(item) is dict]
    if (
        len(annotation_ids) != len(annotations)
        or len(set(annotation_ids)) != len(annotation_ids)
        or set(annotation_ids) != set(observations["tracklet_id"].astype(str))
    ):
        _fail("sealed identity annotations and tracklet rows disagree")
    if any(item.get("status") == "unreviewed" for item in annotations):
        _fail("sealed identity artifact still contains an unreviewed tracklet")
    try:
        bridge, conflicts = identity_annotation.build_identity_bridge(observations, identity)
    except (KeyError, TypeError, ValueError) as exc:
        raise FacialSourceReadinessError(f"sealed identity bridge rebuild failed: {exc}") from exc
    if not conflicts.empty:
        _fail("sealed identity bridge rebuild contains same-frame participant conflicts")
    confirmed = bridge.loc[
        (bridge["source_kind"] == "participant_detection")
        & (bridge["annotation_status"] == "confirmed")
        & bridge["participant"].isin(PARTICIPANT_TO_SLOT)
    ].copy()
    if confirmed.empty:
        _fail("sealed identity bridge has no confirmed participant detections")
    confirmed["source_person_idx"] = pd.to_numeric(confirmed["source_person_idx"], errors="raise")
    numeric_person = confirmed["source_person_idx"].to_numpy(dtype=float)
    if (
        not np.isfinite(numeric_person).all()
        or not np.equal(numeric_person, np.floor(numeric_person)).all()
    ):
        _fail("confirmed Facial identity keys require integer person_idx values")
    confirmed["source_person_idx"] = numeric_person.astype(np.int64)
    if confirmed.duplicated(["frame_index", "source_person_idx"]).any():
        _fail("confirmed identity bridge is not unique on (frame_index, person_idx)")
    sealed_identity = {
        "replay_mode": ("sealed_repository_bytes_and_confirmed_bridge_rebuild_without_raw_media"),
        "inventory": {
            "path": str(authority.crosswalk_inventory),
            "sha256": hashlib.sha256(authority.crosswalk_inventory.read_bytes()).hexdigest(),
        },
        "crosswalk_manifest": crosswalk_actual,
        "identity_manifest": identity_actual,
        "tracklet_observations": {
            **observations_actual,
            "columns": list(observations.columns),
            "parsed_dtypes": [str(value) for value in observations.dtypes],
            "n_rows": int(len(observations)),
        },
        "confirmed_bridge_rows": int(len(confirmed)),
        "rebuilt_identity_conflicts": int(len(conflicts)),
        "accepted_window": dict(accepted),
    }
    return sealed_identity, confirmed, source_fps, (start, end)


def _load_analysis(
    scene: str,
    authority: FacialSourceAuthority,
    *,
    accepted_window: tuple[int, int],
    source_fps: Fraction,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    csv_path = authority.analysis_base_root / f"{scene}.csv"
    provenance_path = authority.analysis_base_root / f"{scene}.provenance.json"
    raw_csv, csv_actual = _stable_read(
        csv_path,
        root=authority.analysis_base_root,
        label="canonical analysis-base CSV",
    )
    raw_provenance, provenance_actual = _stable_read(
        provenance_path,
        root=authority.analysis_base_root,
        label="canonical analysis-base provenance",
    )
    provenance = _strict_json(raw_provenance, "canonical analysis-base provenance")
    if (
        provenance.get("artifact_kind") != analysis_base.ARTIFACT_KIND
        or provenance.get("session_id") != scene
        or provenance.get("trainable") is not False
    ):
        _fail("canonical analysis base has unsupported artifact/readiness semantics")
    _exact_path(
        provenance.get("analysis_base_csv"),
        csv_path,
        "analysis_base_csv",
    )
    if provenance.get("csv_sha256") != csv_actual["sha256"]:
        _fail("canonical analysis CSV differs from its provenance SHA-256")
    frame = _read_csv_bytes(raw_csv, "canonical analysis-base CSV")
    if provenance.get("columns") != list(frame.columns):
        _fail("canonical analysis ordered columns differ from provenance")
    if provenance.get("n_rows") != len(frame) or provenance.get("n_frames") != len(frame):
        _fail("canonical analysis row count differs from provenance")
    _validate_analysis_dtypes(provenance.get("dtypes"), frame)
    if "session_id" not in frame or "timestamp_s" not in frame:
        _fail("canonical analysis base omits exact session/timestamp keys")
    if not frame["session_id"].eq(scene).all():
        _fail("canonical analysis base contains a foreign session key")
    timestamps = pd.to_numeric(frame["timestamp_s"], errors="raise").to_numpy(float)
    scaled = timestamps * FRAME_RATE_HZ
    ticks = np.rint(scaled).astype(np.int64)
    if not np.isfinite(timestamps).all() or not np.allclose(scaled, ticks, rtol=0.0, atol=1e-9):
        _fail("canonical analysis timestamps are not exact 25 Hz tick values")
    expected_range = analysis_base.EXPECTED_TICK_RANGES.get(scene)
    declared_range = provenance.get("output_tick_range")
    if (
        type(declared_range) is not dict
        or type(declared_range.get("start_inclusive")) is not int
        or type(declared_range.get("end_exclusive")) is not int
    ):
        _fail("canonical analysis output_tick_range is malformed")
    start = declared_range["start_inclusive"]
    end = declared_range["end_exclusive"]
    if authority.enforce_frozen_inventory and (start, end) != expected_range:
        _fail("canonical analysis ticks differ from the frozen matrix")
    if not np.array_equal(ticks, np.arange(start, end, dtype=np.int64)):
        _fail("canonical analysis rows are not the complete ordered half-open tick range")
    contract = provenance.get("analysis_window_contract")
    if type(contract) is not dict:
        _fail("canonical analysis base lacks its analysis-window contract")
    contract_ticks = contract.get("authorized_tick_range")
    if (
        type(contract_ticks) is not dict
        or contract_ticks.get("start_inclusive") != start
        or contract_ticks.get("end_exclusive") != end
        or contract_ticks.get("n_ticks") != end - start
        or contract_ticks.get("frame_rate_hz") != FRAME_RATE_HZ
    ):
        _fail("analysis-window contract differs from the exact CSV tick grid")
    accepted = contract.get("accepted_pose_window")
    if (
        type(accepted) is not dict
        or accepted.get("local_start_inclusive") != accepted_window[0]
        or accepted.get("local_end_exclusive") != accepted_window[1]
        or accepted.get("scene") != scene
    ):
        _fail("analysis-window contract differs from the sealed identity window")
    recorded_fps = provenance.get("source_frame_rate_hz")
    if (
        isinstance(recorded_fps, bool)
        or not isinstance(recorded_fps, (int, float))
        or Fraction(str(recorded_fps)) != source_fps
    ):
        _fail("analysis source FPS differs from the sealed identity source FPS")
    participant_mapping = provenance.get("participant_to_slot")
    if participant_mapping != PARTICIPANT_TO_SLOT:
        _fail("analysis participant-to-slot mapping changed")
    for _participant, slot in PARTICIPANT_TO_SLOT.items():
        column = f"{slot}_present"
        if column not in frame:
            _fail(f"canonical analysis base omits {column}")
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
            _fail(f"canonical analysis {column} is not a binary presence mask")
        if len(set(values.tolist())) != 1:
            _fail(f"canonical analysis {column} is not static within the session")
    d3 = contract.get("d3_quality")
    offset = d3.get("offset_s") if type(d3) is dict else None
    if (
        isinstance(offset, bool)
        or not isinstance(offset, (int, float))
        or not math.isfinite(float(offset))
    ):
        _fail("analysis-window contract has no finite 6cam D3 offset")
    binding = {
        **csv_actual,
        "provenance_path": provenance_actual["path"],
        "provenance_size_bytes": provenance_actual["size_bytes"],
        "provenance_sha256": provenance_actual["sha256"],
        "columns": list(frame.columns),
        "parsed_dtypes": [str(value) for value in frame.dtypes],
        "tick_start_inclusive": start,
        "tick_end_exclusive": end,
        "n_ticks": end - start,
        "d3_offset_s": float(offset),
    }
    return frame, binding, provenance


def _load_documents(
    scene: str,
    authority: FacialSourceAuthority,
    inventory: dict[str, Any],
) -> _SceneDocuments:
    sealed_identity, confirmed, source_fps, accepted_window = _load_sealed_identity(
        scene, authority, inventory
    )
    analysis, analysis_binding, provenance = _load_analysis(
        scene,
        authority,
        accepted_window=accepted_window,
        source_fps=source_fps,
    )
    return _SceneDocuments(
        analysis=analysis,
        analysis_binding=analysis_binding,
        analysis_provenance=provenance,
        sealed_identity=sealed_identity,
        confirmed_bridge=confirmed,
        source_fps=source_fps,
        accepted_window=accepted_window,
    )


def past_only_frame_indices(
    ticks: np.ndarray,
    *,
    source_fps: Fraction,
    d3_offset_s: Fraction,
) -> np.ndarray:
    """Map shared 25 Hz ticks to the current camera frame with exact floor.

    ``shared = local + offset``.  No round-to-nearest, interpolation, or
    forward fill is used.  The returned camera frame is the frame whose
    half-open exposure interval contains the analysis tick.
    """

    if not isinstance(source_fps, Fraction) or source_fps <= 0:
        _fail("source_fps must be a positive Fraction")
    if not isinstance(d3_offset_s, Fraction):
        _fail("d3_offset_s must be a Fraction")
    values = np.asarray(ticks)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        _fail("ticks must be one integer vector")
    output = np.empty(len(values), dtype=np.int64)
    for index, tick in enumerate(values.tolist()):
        local_position = (Fraction(int(tick), FRAME_RATE_HZ) - d3_offset_s) * source_fps
        output[index] = math.floor(local_position)
    return output


def _optional_csv(
    path: Path,
    *,
    root: Path,
    label: str,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    canonical = _lexical_absolute(path)
    if not canonical.exists():
        return None, {
            "path": str(canonical),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
            "columns": None,
            "parsed_dtypes": None,
            "n_rows": 0,
            "parse_error": None,
        }
    raw, binding = _stable_read(canonical, root=root, label=label)
    try:
        frame = _read_csv_bytes(raw, label)
    except FacialSourceReadinessError as exc:
        return None, {
            **binding,
            "exists": True,
            "columns": None,
            "parsed_dtypes": None,
            "n_rows": 0,
            "parse_error": str(exc),
        }
    return frame, {
        **binding,
        "exists": True,
        "columns": list(frame.columns),
        "parsed_dtypes": [str(value) for value in frame.dtypes],
        "n_rows": int(len(frame)),
        "parse_error": None,
    }


def _coerce_facial_source(
    frame: pd.DataFrame,
    *,
    source_fps: Fraction,
) -> tuple[pd.DataFrame, list[str]]:
    blockers: list[str] = []
    if list(frame.columns) != list(EXPECTED_OPENFACE_COLUMNS):
        blockers.append("openface_ordered_schema_is_not_the_bound_16_column_export")
        return frame.copy(), blockers
    result = frame.copy()
    numeric_columns = list(EXPECTED_OPENFACE_COLUMNS)
    for column in numeric_columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError):
            blockers.append(f"openface_column_is_not_numeric:{column}")
            return result, blockers
    values = result[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        blockers.append("openface_contains_nan_or_infinite_values")
    for column in ("frame_index", "person_idx", "emotion_argmax"):
        column_values = result[column].to_numpy(dtype=float)
        if not np.equal(column_values, np.floor(column_values)).all():
            blockers.append(f"openface_column_is_not_integer_valued:{column}")
    if blockers:
        return result, blockers
    result["frame_index"] = result["frame_index"].astype(np.int64)
    result["person_idx"] = result["person_idx"].astype(np.int64)
    result["emotion_argmax"] = result["emotion_argmax"].astype(np.int64)
    if (result["frame_index"] < 0).any() or (result["person_idx"] < 0).any():
        blockers.append("openface_key_contains_a_negative_value")
    if not result["emotion_argmax"].between(0, 7).all():
        blockers.append("emotion_argmax_is_outside_the_documented_0_to_7_range")
    if not result["match_iou"].between(0.0, 1.0).all():
        blockers.append("match_iou_is_outside_0_to_1")
    if not result["face_conf"].between(0.0, 1.0).all():
        blockers.append("face_conf_is_outside_0_to_1")
    expected_times = np.asarray(
        [float(Fraction(int(value), 1) / source_fps) for value in result["frame_index"]]
    )
    if not np.allclose(
        result["timestamp_sec"].to_numpy(dtype=float),
        expected_times,
        rtol=0.0,
        atol=1e-6,
    ):
        blockers.append("timestamp_sec_is_not_frame_index_divided_by_bound_source_fps")
    return result, blockers


def _column_stats(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in columns:
        if column not in frame:
            result[column] = {
                "present": False,
                "finite_rows": 0,
                "null_rows": int(len(frame)),
                "min": None,
                "max": None,
            }
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        finite = np.isfinite(numeric)
        result[column] = {
            "present": True,
            "finite_rows": int(finite.sum()),
            "null_rows": int((~finite).sum()),
            "min": None if not finite.any() else float(numeric[finite].min()),
            "max": None if not finite.any() else float(numeric[finite].max()),
        }
    return result


def _landmark_inventory(columns: list[str]) -> dict[str, Any]:
    conventions = {
        "x_N_y_N": (re.compile(r"^x_([0-9]+)$"), re.compile(r"^y_([0-9]+)$")),
        "landmark_N_xy": (
            re.compile(r"^landmark_([0-9]+)_x$"),
            re.compile(r"^landmark_([0-9]+)_y$"),
        ),
        "landmark_xy_N": (
            re.compile(r"^landmark_x_([0-9]+)$"),
            re.compile(r"^landmark_y_([0-9]+)$"),
        ),
    }
    detected: dict[str, dict[str, list[int]]] = {}
    complete = False
    for name, (x_pattern, y_pattern) in conventions.items():
        x_indices = sorted(
            int(match.group(1))
            for column in columns
            if (match := x_pattern.fullmatch(column)) is not None
        )
        y_indices = sorted(
            int(match.group(1))
            for column in columns
            if (match := y_pattern.fullmatch(column)) is not None
        )
        if x_indices or y_indices:
            detected[name] = {"x_indices": x_indices, "y_indices": y_indices}
        if x_indices == list(range(98)) and y_indices == list(range(98)):
            complete = True
    return {
        "recognised_coordinate_conventions": detected,
        "complete_98_point_xy_export_present": complete,
        "required_for_mar": (
            "reproducible 98-point x/y coordinates including documented mouth points"
        ),
    }


def _feature_families(frame: pd.DataFrame) -> dict[str, Any]:
    emotion_counts: dict[str, int] = {}
    if "emotion_argmax" in frame:
        numeric = pd.to_numeric(frame["emotion_argmax"], errors="coerce")
        finite = numeric[np.isfinite(numeric.to_numpy(float))]
        emotion_counts = {
            str(int(key)): int(value)
            for key, value in sorted(Counter(finite.astype(int).tolist()).items())
        }
    common = {
        "value_contract": "nullable_evidence_only",
        "model_ready": False,
        "assembly_eligible": False,
        "zero_fill_missing_forbidden": True,
    }
    return {
        "action_units": {
            **common,
            "columns": list(AU_COLUMNS),
            "column_stats": _column_stats(frame, AU_COLUMNS),
            "semantic_index_mapping_verified": False,
        },
        "emotion": {
            **common,
            "columns": list(EMOTION_COLUMNS),
            "column_stats": _column_stats(frame, EMOTION_COLUMNS),
            "observed_argmax_counts": emotion_counts,
            "probability_or_logit_vector_exported": False,
            "human_validated_affect_labels": False,
        },
        "facial_gaze_angles": {
            **common,
            "columns": list(GAZE_COLUMNS),
            "column_stats": _column_stats(frame, GAZE_COLUMNS),
            "approved_model_derivation": False,
        },
    }


def _empty_coverage(analysis: pd.DataFrame, reason: str) -> dict[str, Any]:
    participants: dict[str, Any] = {}
    for participant, slot in PARTICIPANT_TO_SLOT.items():
        present = int(pd.to_numeric(analysis[f"{slot}_present"]).sum())
        absent = int(len(analysis) - present)
        participants[participant] = {
            "slot": slot,
            "present_tick_count": present,
            "absent_tick_count": absent,
            "valid_tick_count": 0,
            "missing_present_tick_count": present,
            "coverage_of_present_ticks": None if present == 0 else 0.0,
            "value_state": "absent" if present == 0 else "all_null",
            "reason": "slot_absent" if present == 0 else reason,
            "zero_filled": False,
        }
    return {
        "participant_ticks": participants,
        "ticks_with_any_valid_participant": 0,
        "ticks_with_all_present_participants_valid": 0,
        "complete_strict_visual_tick_count": 0,
    }


def _coverage(
    *,
    analysis: pd.DataFrame,
    facial: pd.DataFrame,
    confirmed: pd.DataFrame,
    camera_frames: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity_keys = confirmed.loc[
        :, ["frame_index", "source_person_idx", "participant", "slot"]
    ].rename(columns={"source_person_idx": "person_idx"})
    merged = facial.merge(
        identity_keys,
        how="left",
        on=["frame_index", "person_idx"],
        validate="many_to_one",
        sort=False,
    )
    key_counts = facial.groupby(["frame_index", "person_idx"], sort=False).size()
    duplicate_counts = key_counts[key_counts > 1]
    participant_frame_counts = (
        merged.loc[merged["participant"].isin(PARTICIPANT_TO_SLOT)]
        .groupby(["frame_index", "participant"], sort=False)
        .size()
    )
    participants: dict[str, Any] = {}
    valid_arrays: dict[str, np.ndarray] = {}
    for participant, slot in PARTICIPANT_TO_SLOT.items():
        present_mask = pd.to_numeric(analysis[f"{slot}_present"]).to_numpy(float) == 1.0
        evidence_counts = np.asarray(
            [
                int(participant_frame_counts.get((int(frame), participant), 0))
                for frame in camera_frames
            ],
            dtype=np.int64,
        )
        valid = present_mask & (evidence_counts == 1)
        valid_arrays[participant] = valid
        present = int(present_mask.sum())
        absent = int((~present_mask).sum())
        valid_count = int(valid.sum())
        missing = present - valid_count
        joined_rows = int((merged["participant"] == participant).sum())
        joined_unique_keys = int(
            merged.loc[merged["participant"] == participant, ["frame_index", "person_idx"]]
            .drop_duplicates()
            .shape[0]
        )
        duplicate_groups = int(
            sum(
                1
                for (frame_index, person_idx), _count in duplicate_counts.items()
                if bool(
                    (
                        (identity_keys["frame_index"] == frame_index)
                        & (identity_keys["person_idx"] == person_idx)
                        & (identity_keys["participant"] == participant)
                    ).any()
                )
            )
        )
        if present == 0:
            state = "absent"
            reason = "slot_absent_not_an_invalid_observation"
        elif joined_rows == 0:
            state = "all_null"
            reason = (
                "robot_identity_has_no_facial_source_row"
                if participant == "R"
                else "no_exact_confirmed_identity_join"
            )
        elif valid_count == present:
            state = "valid_on_all_present_ticks"
            reason = None
        else:
            state = "nullable_partial"
            reason = "sparse_or_missing_or_duplicate_exact_key"
        participants[participant] = {
            "slot": slot,
            "present_tick_count": present,
            "absent_tick_count": absent,
            "source_rows_joined_to_confirmed_identity": joined_rows,
            "source_unique_keys_joined_to_confirmed_identity": joined_unique_keys,
            "duplicate_key_groups_invalidated": duplicate_groups,
            "valid_tick_count": valid_count,
            "missing_present_tick_count": missing,
            "coverage_of_present_ticks": (None if present == 0 else valid_count / present),
            "value_state": state,
            "reason": reason,
            "zero_filled": False,
        }
    stacked = np.vstack([valid_arrays[name] for name in PARTICIPANT_TO_SLOT])
    present = np.vstack(
        [
            pd.to_numeric(analysis[f"{slot}_present"]).to_numpy(float) == 1.0
            for slot in PARTICIPANT_TO_SLOT.values()
        ]
    )
    any_valid = stacked.any(axis=0)
    all_present_valid = np.all((~present) | stacked, axis=0)
    any_present = present.any(axis=0)
    all_present_valid &= any_present
    join_inventory = {
        "accepted_source_rows": int(len(facial)),
        "unique_frame_person_keys": int(len(key_counts)),
        "duplicate_key_group_count": int(len(duplicate_counts)),
        "rows_in_duplicate_key_groups": int(duplicate_counts.sum()),
        "duplicate_groups_are_wholly_invalid": True,
        "confirmed_identity_joined_rows": int(merged["participant"].notna().sum()),
        "confirmed_identity_joined_unique_keys": int(
            merged.loc[merged["participant"].notna(), ["frame_index", "person_idx"]]
            .drop_duplicates()
            .shape[0]
        ),
        "rows_without_exact_confirmed_identity": int(merged["participant"].isna().sum()),
        "identity_join_key": ["frame_index", "person_idx"],
        "join_kind": "exact_same_frame_no_interpolation",
    }
    return (
        {
            "participant_ticks": participants,
            "ticks_with_any_valid_participant": int(any_valid.sum()),
            "ticks_with_all_present_participants_valid": int(all_present_valid.sum()),
            "complete_strict_visual_tick_count": 0,
        },
        join_inventory,
    )


def _mouth_motion_inventory(
    path: Path,
    authority: FacialSourceAuthority,
) -> dict[str, Any]:
    frame, binding = _optional_csv(
        path,
        root=authority.facial_root,
        label="disabled mouth_motion CSV",
    )
    schema_matches = frame is not None and list(frame.columns) == list(
        EXPECTED_MOUTH_MOTION_COLUMNS
    )
    return {
        **binding,
        "ordered_schema_matches_documented_export": schema_matches,
        "disabled": True,
        "consumed_for_values": False,
        "is_mar": False,
        "assembly_eligible": False,
        "reason": "mouth_motion_is_not_mar_and_is_forbidden_by_the_project_contract",
    }


def _build_report(
    scene: str,
    documents: _SceneDocuments,
    authority: FacialSourceAuthority,
) -> dict[str, Any]:
    openface_path, mouth_path = _source_paths(scene, authority)
    raw_source, source_binding = _optional_csv(
        openface_path,
        root=authority.facial_root,
        label="canonical Facial openface CSV",
    )
    mouth = _mouth_motion_inventory(mouth_path, authority)
    ticks = np.arange(
        documents.analysis_binding["tick_start_inclusive"],
        documents.analysis_binding["tick_end_exclusive"],
        dtype=np.int64,
    )
    d3_offset = Fraction(str(documents.analysis_binding["d3_offset_s"]))
    camera_frames = past_only_frame_indices(
        ticks,
        source_fps=documents.source_fps,
        d3_offset_s=d3_offset,
    )
    start, end = documents.accepted_window
    if ((camera_frames < start) | (camera_frames >= end)).any():
        _fail("canonical analysis tick maps outside the sealed Facial identity window")

    dynamic_blockers: list[str] = []
    source_frame = pd.DataFrame(columns=EXPECTED_OPENFACE_COLUMNS)
    accepted = source_frame
    source_schema_matches = False
    feature_families = _feature_families(source_frame)
    join_inventory = {
        "accepted_source_rows": 0,
        "unique_frame_person_keys": 0,
        "duplicate_key_group_count": 0,
        "rows_in_duplicate_key_groups": 0,
        "duplicate_groups_are_wholly_invalid": True,
        "confirmed_identity_joined_rows": 0,
        "confirmed_identity_joined_unique_keys": 0,
        "rows_without_exact_confirmed_identity": 0,
        "identity_join_key": ["frame_index", "person_idx"],
        "join_kind": "exact_same_frame_no_interpolation",
    }
    if raw_source is None:
        dynamic_blockers.append(
            "facial_source_missing"
            if source_binding["parse_error"] is None
            else "facial_source_unreadable"
        )
        coverage = _empty_coverage(documents.analysis, dynamic_blockers[0])
    else:
        source_frame, schema_blockers = _coerce_facial_source(
            raw_source,
            source_fps=documents.source_fps,
        )
        dynamic_blockers.extend(schema_blockers)
        source_schema_matches = not schema_blockers and list(source_frame.columns) == list(
            EXPECTED_OPENFACE_COLUMNS
        )
        if source_schema_matches:
            accepted_mask = source_frame["frame_index"].between(start, end - 1, inclusive="both")
            accepted = source_frame.loc[accepted_mask].copy()
            feature_families = _feature_families(accepted)
            coverage, join_inventory = _coverage(
                analysis=documents.analysis,
                facial=accepted,
                confirmed=documents.confirmed_bridge,
                camera_frames=camera_frames,
            )
        else:
            coverage = _empty_coverage(documents.analysis, "facial_schema_invalid")
    source_binding = {
        **source_binding,
        "canonical_repository_relative_path": (
            str(openface_path.relative_to(authority.repository_root)).replace("\\", "/")
            if openface_path.is_relative_to(authority.repository_root)
            else None
        ),
        "expected_ordered_columns": list(EXPECTED_OPENFACE_COLUMNS),
        "ordered_schema_matches_bound_16_column_export": source_schema_matches,
        "source_fps": float(documents.source_fps),
        "accepted_local_frame_window": {
            "start_inclusive": start,
            "end_exclusive": end,
        },
        "rows_in_accepted_window": int(len(accepted)),
        "rows_outside_accepted_window": int(len(source_frame) - len(accepted)),
    }
    landmark_inventory = _landmark_inventory([] if raw_source is None else list(raw_source.columns))
    fixed_blockers = [
        "mar_unavailable_no_exported_98_point_landmark_coordinates",
        "mouth_motion_forbidden_not_mar",
        "robot_participant_has_no_facial_source_row",
        "au_emotion_and_facial_gaze_are_nullable_evidence_not_model_ready",
        "no_approved_derivation_or_modality_specific_formal_validity_schema",
        "read_only_evidence_has_no_writer_or_promotion_api",
    ]
    return {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": scene,
        "output_role": OUTPUT_ROLE,
        "read_only": True,
        "assembly_eligible": False,
        "counted_input_ready": False,
        "feature_publication_allowed": False,
        "writer_available": False,
        "raw_video_opened_or_hydrated": False,
        "external_media_read": False,
        "analysis_base": documents.analysis_binding,
        "sealed_identity": documents.sealed_identity,
        "source": source_binding,
        "time_mapping": {
            "axis": "mic_A_shared_seconds",
            "rule": "floor((analysis_tick/25 - d3_offset_s) * source_fps)",
            "exact_rational_arithmetic": True,
            "past_only": True,
            "round_to_nearest": False,
            "interpolation": False,
            "forward_fill_or_hold": False,
            "analysis_tick_count": int(len(ticks)),
            "mapped_camera_frame_min": int(camera_frames.min()),
            "mapped_camera_frame_max": int(camera_frames.max()),
        },
        "column_inventory": {
            "keys": ["frame_index", "timestamp_sec", "person_idx"],
            "quality": list(QUALITY_COLUMNS),
            "action_units": list(AU_COLUMNS),
            "emotion": list(EMOTION_COLUMNS),
            "facial_gaze_angles": list(GAZE_COLUMNS),
            "landmarks": landmark_inventory,
        },
        "feature_families": feature_families,
        "identity_join_inventory": join_inventory,
        "coverage": coverage,
        "nullable_validity_contract": {
            "valid_true_only_when": (
                "one exact Facial row joins one confirmed same-frame identity key "
                "at the past-only analysis frame"
            ),
            "duplicate_frame_person_group": "invalid_entire_group",
            "no_exact_bridge": "valid_false_value_null",
            "missing_source_row": "valid_false_value_null",
            "absent_slot": "absent_not_invalid_and_value_null",
            "zero_fill_missing": False,
            "aggregation_or_row_selection": None,
        },
        "mar": {
            "readiness": False,
            "reproducible_from_bound_export": False,
            "bound_export_column_count": (None if raw_source is None else len(raw_source.columns)),
            "bound_export_is_exact_16_column_evidence": source_schema_matches,
            "complete_98_point_xy_export_present": landmark_inventory[
                "complete_98_point_xy_export_present"
            ],
            "mouth_motion_substitution_allowed": False,
            "reason": (
                "the bound 16-column export has no landmark coordinates; "
                "mouth_motion is a different disabled signal"
            ),
        },
        "disabled_mouth_motion": mouth,
        "readiness_blockers": [*dynamic_blockers, *fixed_blockers],
    }


def audit_facial_source_readiness(
    scene: str,
    *,
    authority: FacialSourceAuthority | None = None,
) -> dict[str, Any]:
    """Return one in-memory, permanently non-publishable readiness report."""

    selected = default_source_authority() if authority is None else authority
    if type(selected) is not FacialSourceAuthority:
        _fail("authority has the wrong exact type")
    _validate_scene(scene, selected)
    inventory = _load_inventory(selected)
    documents = _load_documents(scene, selected, inventory)
    return _build_report(scene, documents, selected)


def audit_facial_source_matrix(
    *,
    authority: FacialSourceAuthority | None = None,
) -> dict[str, Any]:
    """Audit the exact frozen scene set while reading each sealed inventory once."""

    selected = default_source_authority() if authority is None else authority
    if type(selected) is not FacialSourceAuthority:
        _fail("authority has the wrong exact type")
    inventory = _load_inventory(selected)
    scenes = (
        tuple(analysis_base.EXPECTED_TICK_RANGES)
        if selected.enforce_frozen_inventory
        else tuple(sorted(inventory["scenes"]))
    )
    reports = {
        scene: _build_report(
            scene,
            _load_documents(scene, selected, inventory),
            selected,
        )
        for scene in scenes
    }
    for report in reports.values():
        if (
            report["assembly_eligible"] is not False
            or report["counted_input_ready"] is not False
            or report["writer_available"] is not False
            or report["mar"]["readiness"] is not False
        ):
            _fail("Facial readiness matrix contains an impossible promotion claim")
    return {
        "artifact_kind": MATRIX_ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "output_role": OUTPUT_ROLE,
        "read_only": True,
        "assembly_eligible": False,
        "counted_input_ready": False,
        "feature_publication_allowed": False,
        "writer_available": False,
        "raw_video_opened_or_hydrated": False,
        "external_media_read": False,
        "n_scenes": len(reports),
        "n_analysis_ticks": sum(report["analysis_base"]["n_ticks"] for report in reports.values()),
        "n_sources_present": sum(int(report["source"]["exists"]) for report in reports.values()),
        "n_exact_16_column_sources": sum(
            int(report["source"]["ordered_schema_matches_bound_16_column_export"])
            for report in reports.values()
        ),
        "mar_ready_scenes": 0,
        "scenes": reports,
    }


__all__ = [
    "ARTIFACT_KIND",
    "EXPECTED_OPENFACE_COLUMNS",
    "FORMAT_VERSION",
    "FacialSourceAuthority",
    "FacialSourceReadinessError",
    "audit_facial_source_matrix",
    "audit_facial_source_readiness",
    "default_source_authority",
    "past_only_frame_indices",
]
