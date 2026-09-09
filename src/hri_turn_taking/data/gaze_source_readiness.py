"""Read-only nullable 2D Gaze evidence preflight for the frozen real corpus.

This module deliberately stops before feature publication.  It replays the
repository-local analysis-base, sealed crosswalk, identity, and Gaze source
documents, then constructs nullable evidence in memory on the exact canonical
25 Hz ticks.  It has no writer, assembly manifest, or counted-input API.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hri_turn_taking.data import analysis_base, identity_annotation, multiparty
from hri_turn_taking.data import hpe_transcript_crosswalk as crosswalk

ARTIFACT_KIND = "vap_real_nullable_2d_gaze_evidence_preflight"
MATRIX_ARTIFACT_KIND = "vap_real_nullable_2d_gaze_evidence_preflight_matrix"
FORMAT_VERSION = 1
FRAME_RATE_HZ = 25
OUTPUT_ROLE = "read_only_nullable_2d_gaze_evidence"
TRUST_MODEL = "sealed_repo_small_file_replay_nonpublishing_only"

PARTICIPANT_TO_SLOT = {"R": "P1", "A": "P2", "B": "P3", "C": "P4"}
SLOT_TO_PARTICIPANT = {slot: participant for participant, slot in PARTICIPANT_TO_SLOT.items()}
GAZE_TARGETS = frozenset({"robot", "participant", "elsewhere"})
RAW_GAZE_COLUMNS = (
    "frame_index",
    "timestamp_sec",
    "robot_detected",
    "robot_x0",
    "robot_y0",
    "robot_x1",
    "robot_y1",
    "person_idx",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "gaze_target",
    "gaze_peak_x",
    "gaze_peak_y",
    "inout_score",
)
EVIDENCE_COLUMNS = (
    "session_id",
    "tick",
    "timestamp_s",
    "source_frame_index",
    "source_timestamp_s",
    "participant",
    "slot",
    "slot_present",
    "source_observation_id",
    "gaze_target_valid",
    "gaze_target",
    "gaze_target_reason",
    "gaze_robot_valid",
    "gaze_robot",
    "gaze_robot_reason",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class GazeSourceReadinessError(ValueError):
    """A source, identity, timing, or nullable-evidence gate failed closed."""


@dataclass(frozen=True)
class GazeSourcePreflight:
    """One in-memory, permanently non-publishable scene result."""

    scene: str
    evidence_df: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class _FileBinding:
    path: Path
    sha256: str
    size_bytes: int

    def report(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class _AnalysisSnapshot:
    frame: pd.DataFrame
    provenance: dict[str, Any]
    ticks: np.ndarray
    slot_presence: dict[str, bool]
    csv: _FileBinding
    sidecar: _FileBinding


@dataclass(frozen=True)
class _GazeSnapshot:
    bridge: pd.DataFrame
    source_fps: int
    accepted_window: dict[str, Any]
    bindings: dict[str, _FileBinding]
    reports: dict[str, Any]


def _fail(message: str) -> None:
    raise GazeSourceReadinessError(message)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _file_identity(stat: Any) -> tuple[int, int, int, int]:
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _stable_read(path: Path, label: str) -> tuple[bytes, _FileBinding]:
    if not path.is_file() or path.is_symlink():
        _fail(f"{label} is missing, not a regular file, or is a symlink: {path}")
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise GazeSourceReadinessError(f"could not read {label}: {path}") from exc
    if _file_identity(before) != _file_identity(after) or len(payload) != after.st_size:
        _fail(f"{label} changed while it was read")
    return payload, _FileBinding(
        path=path.resolve(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GazeSourceReadinessError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if type(value) is not dict:
        _fail(f"{label} must contain a JSON object")
    return value


def _strict_csv(payload: bytes, label: str, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(payload), **kwargs)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise GazeSourceReadinessError(f"could not parse {label} as CSV: {exc}") from exc


def _assert_binding(
    declared: Any,
    actual: _FileBinding,
    label: str,
    *,
    require_size: bool,
) -> None:
    if type(declared) is not dict:
        _fail(f"{label} binding must be an object")
    path_value = declared.get("path")
    digest = declared.get("sha256")
    if type(path_value) is not str or Path(path_value).resolve() != actual.path:
        _fail(f"{label} binding does not name the canonical source path")
    if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
        _fail(f"{label} binding has a malformed SHA-256")
    if digest != actual.sha256:
        _fail(f"{label} bytes differ from their sealed SHA-256")
    if require_size:
        size = declared.get("size_bytes")
        if type(size) is not int or size < 0 or size != actual.size_bytes:
            _fail(f"{label} size differs from its sealed binding")


def _assert_inventory_file_binding(declared: Any, actual: _FileBinding, label: str) -> None:
    if type(declared) is not dict or set(declared) != {"sha256", "size_bytes"}:
        _fail(f"{label} inventory binding is malformed")
    if (
        type(declared["sha256"]) is not str
        or _SHA256_RE.fullmatch(declared["sha256"]) is None
        or declared["sha256"] != actual.sha256
        or type(declared["size_bytes"]) is not int
        or declared["size_bytes"] != actual.size_bytes
    ):
        _fail(f"{label} differs from the sealed inventory")


def _exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        _fail(f"{label} must be an exact integer (boolean/float rejected)")
    if minimum is not None and value < minimum:
        _fail(f"{label} must be at least {minimum}")
    return value


def _exact_source_fps(value: Any, label: str) -> int:
    if type(value) not in {int, float} or type(value) is bool:
        _fail(f"{label} must be an integer-valued number")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or int(number) not in {30, 60}:
        _fail(f"{label} must be exactly 30 or 60 Hz")
    return int(number)


def _finite_fraction(value: Any, label: str) -> Fraction:
    if type(value) not in {str, int, float} or type(value) is bool:
        _fail(f"{label} must be a finite decimal scalar")
    if type(value) is float and not math.isfinite(value):
        _fail(f"{label} must be finite")
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise GazeSourceReadinessError(f"{label} is not a finite decimal scalar") from exc


def past_only_frame_indices(
    ticks: list[int] | tuple[int, ...] | np.ndarray,
    *,
    source_fps: int,
    d3_offset_s: str | int | float,
    accepted_start: int,
    accepted_end: int,
) -> np.ndarray:
    """Map 25 Hz shared ticks to the latest source frame using rational floor."""

    if type(source_fps) is not int or source_fps not in {30, 60}:
        _fail("source_fps must be the exact integer 30 or 60")
    start = _exact_int(accepted_start, "accepted_start", minimum=0)
    end = _exact_int(accepted_end, "accepted_end", minimum=1)
    if end <= start:
        _fail("accepted source-frame window must be non-empty")
    if type(ticks) not in {list, tuple, np.ndarray}:
        _fail("ticks must be a list, tuple, or ndarray")
    values = list(ticks)
    if not values or any(type(value) not in {int, np.int64} for value in values):
        _fail("ticks must be non-empty exact integers")
    integer_ticks = [int(value) for value in values]
    if any(value < 0 for value in integer_ticks) or any(
        right <= left for left, right in zip(integer_ticks, integer_ticks[1:], strict=False)
    ):
        _fail("ticks must be non-negative, strictly increasing, and unique")
    offset = _finite_fraction(d3_offset_s, "d3_offset_s")
    frames = np.asarray(
        [
            ((Fraction(tick, FRAME_RATE_HZ) - offset) * source_fps).__floor__()
            for tick in integer_ticks
        ],
        dtype=np.int64,
    )
    if np.any(frames < start) or np.any(frames >= end):
        _fail("past-only Gaze mapping escapes the accepted half-open source window")
    for tick, frame in zip(integer_ticks, frames, strict=True):
        target = Fraction(tick, FRAME_RATE_HZ)
        current = Fraction(int(frame), source_fps) + offset
        following = Fraction(int(frame) + 1, source_fps) + offset
        if current > target or following <= target:
            _fail("past-only rational Gaze source-frame invariant failed")
    return frames


def _load_analysis_snapshot(scene: str, root: Path) -> _AnalysisSnapshot:
    csv_path = (root / "data" / "processed" / "analysis_base" / f"{scene}.csv").resolve()
    sidecar_path = csv_path.with_suffix(".provenance.json")
    csv_payload, csv_binding = _stable_read(csv_path, "canonical analysis-base CSV")
    sidecar_payload, sidecar_binding = _stable_read(
        sidecar_path, "canonical analysis-base provenance"
    )
    provenance = _strict_json(sidecar_payload, "analysis-base provenance")
    if provenance.get("artifact_kind") != analysis_base.ARTIFACT_KIND:
        _fail("Gaze preflight requires a canonical vap_real_analysis_base")
    if provenance.get("format_version") != 2 or provenance.get("session_id") != scene:
        _fail("analysis-base format/session differs from the frozen contract")
    if provenance.get("analysis_base_csv") != str(csv_path):
        _fail("analysis-base sidecar is not anchored to its canonical path")
    if provenance.get("csv_sha256") != csv_binding.sha256:
        _fail("analysis-base CSV hash differs from its provenance")
    frame = _strict_csv(csv_payload, "analysis-base CSV", float_precision="round_trip")
    columns = list(frame.columns)
    actual_dtypes = {column: str(frame[column].dtype) for column in columns}
    if provenance.get("columns") != columns or provenance.get("dtypes") != actual_dtypes:
        _fail("analysis-base ordered schema/dtypes differ from provenance")
    if provenance.get("n_rows") != len(frame):
        _fail("analysis-base row count differs from provenance")
    canonical_columns = [
        *multiparty.METADATA_COLUMNS,
        *multiparty.vad_columns(),
        *multiparty.present_columns(),
        *multiparty.input_feature_columns(),
    ]
    if len(columns) != len(canonical_columns) or set(columns) != set(canonical_columns):
        _fail("analysis-base CSV is not the exact canonical multiparty schema")
    try:
        multiparty.validate_multiparty_dataframe(frame)
    except (TypeError, ValueError) as exc:
        raise GazeSourceReadinessError(
            f"analysis-base multiparty value contract failed: {exc}"
        ) from exc

    start, end = analysis_base.EXPECTED_TICK_RANGES[scene]
    if provenance.get("output_tick_range") != {
        "start_inclusive": start,
        "end_exclusive": end,
    }:
        _fail("analysis-base output ticks differ from the frozen scene range")
    ticks = np.arange(start, end, dtype=np.int64)
    if len(frame) != len(ticks):
        _fail("analysis-base CSV row count differs from its frozen tick interval")
    if not np.array_equal(frame["session_id"].to_numpy(dtype=object), [scene] * len(ticks)):
        _fail("analysis-base session_id keys are not constant and canonical")
    if frame.duplicated(["session_id", "timestamp_s"]).any():
        _fail("analysis-base contains duplicate canonical keys")
    expected_time = ticks.astype(np.float64) / FRAME_RATE_HZ
    if not np.array_equal(frame["timestamp_s"].to_numpy(dtype=np.float64), expected_time):
        _fail("analysis-base timestamps do not exactly equal integer tick / 25")
    if provenance.get("participant_to_slot") != PARTICIPANT_TO_SLOT:
        _fail("analysis-base participant_to_slot differs from R=P1,A=P2,B=P3,C=P4")

    slot_presence: dict[str, bool] = {}
    for slot in SLOT_TO_PARTICIPANT:
        column = f"{slot}_present"
        if column not in frame:
            _fail(f"analysis-base lacks static presence column {column}")
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isin(values, (0.0, 1.0)).all() or np.unique(values).size != 1:
            _fail(f"analysis-base {column} must be binary and static")
        slot_presence[slot] = bool(values[0])
    placeholders = provenance.get("placeholder_columns")
    if type(placeholders) is not list:
        _fail("analysis-base placeholder_columns is malformed")
    required_placeholders = {
        f"{slot}_{feature}"
        for slot in SLOT_TO_PARTICIPANT
        for feature in ("gaze_robot", "gazed_at")
    }
    if not required_placeholders.issubset(placeholders):
        _fail("analysis base unexpectedly claims an existing formal Gaze input")
    return _AnalysisSnapshot(
        frame=frame,
        provenance=provenance,
        ticks=ticks,
        slot_presence=slot_presence,
        csv=csv_binding,
        sidecar=sidecar_binding,
    )


def _resolved_repo_path(root: Path, value: Any, label: str) -> Path:
    if type(value) is not str or not value:
        _fail(f"{label} must be a non-empty path string")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GazeSourceReadinessError(f"{label} must remain inside the repository") from exc
    return resolved


def _portable_binding_matches(left: Any, right: Any) -> bool:
    if type(left) is not dict or type(right) is not dict:
        return False
    for key in ("path", "sha256", "columns", "n_rows"):
        if left.get(key) != right.get(key):
            return False
    left_types = left.get("parsed_dtypes")
    right_types = right.get("parsed_dtypes")
    return (
        type(left_types) is list
        and type(right_types) is list
        and len(left_types) == len(right_types)
        and all(type(value) is str and value for value in [*left_types, *right_types])
    )


def _assert_source_replay(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    if len(expected) != len(actual):
        _fail("tracklet observations cannot be replayed from the bound Gaze CSV")
    missing = [column for column in expected.columns if column not in actual]
    if missing:
        _fail(f"tracklet observations omit Gaze-derived columns: {missing}")
    for column in expected.columns:
        left = expected[column]
        right = actual[column]
        if pd.api.types.is_numeric_dtype(left.dtype):
            try:
                left_values = pd.to_numeric(left, errors="raise").to_numpy(dtype=float)
                right_values = pd.to_numeric(right, errors="raise").to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise GazeSourceReadinessError(
                    f"tracklet numeric source column changed: {column}"
                ) from exc
            if not np.isclose(
                left_values,
                right_values,
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ).all():
                _fail(f"tracklet observations differ from Gaze source column {column}")
        else:
            left_text = left.astype("string").fillna("")
            right_text = right.astype("string").fillna("")
            if not left_text.equals(right_text):
                _fail(f"tracklet observations differ from Gaze source column {column}")


def _validate_final_mapping(
    frame: pd.DataFrame, expected_hpe_persons: Any, scene: str
) -> list[dict[str, Any]]:
    expected_columns = [
        "hpe_person",
        "participant",
        "slot",
        "review_status",
        "evidence_sample_ids",
    ]
    if list(frame.columns) != expected_columns or frame.empty:
        _fail(f"sealed final crosswalk schema is malformed for {scene}")
    people = pd.to_numeric(frame["hpe_person"], errors="raise")
    if not np.equal(people, np.floor(people)).all() or people.duplicated().any():
        _fail(f"sealed final crosswalk has invalid/duplicate HPE people for {scene}")
    actual_people = sorted(int(value) for value in people.tolist())
    if type(expected_hpe_persons) is not list or actual_people != expected_hpe_persons:
        _fail(f"sealed final crosswalk HPE-person set differs for {scene}")
    if not (frame["review_status"] == "confirmed").all():
        _fail(f"sealed final crosswalk contains an unconfirmed mapping for {scene}")
    records: list[dict[str, Any]] = []
    for hpe_person, participant, slot in zip(
        people.tolist(), frame["participant"], frame["slot"], strict=True
    ):
        if participant not in PARTICIPANT_TO_SLOT or PARTICIPANT_TO_SLOT[participant] != slot:
            _fail(f"sealed final crosswalk violates canonical slot semantics for {scene}")
        records.append(
            {
                "hpe_person": int(hpe_person),
                "participant": str(participant),
                "slot": str(slot),
            }
        )
    if set(frame["participant"]) != set(PARTICIPANT_TO_SLOT):
        _fail(f"sealed final crosswalk does not cover A/B/C/R for {scene}")
    return records


def _load_gaze_snapshot(
    scene: str,
    root: Path,
    analysis: _AnalysisSnapshot,
) -> _GazeSnapshot:
    inventory_path = (root / crosswalk.DEFAULT_SEALED_INVENTORY_PATH).resolve()
    inventory_payload, inventory_binding = _stable_read(
        inventory_path, "sealed crosswalk inventory"
    )
    inventory_raw = _strict_json(inventory_payload, "sealed crosswalk inventory")
    try:
        inventory = crosswalk.load_sealed_crosswalk_inventory(inventory_path, repository_root=root)
    except (OSError, ValueError, crosswalk.HPETranscriptCrosswalkError) as exc:
        raise GazeSourceReadinessError(
            f"sealed crosswalk inventory failed repo-local replay: {exc}"
        ) from exc
    if inventory != inventory_raw:
        _fail("sealed crosswalk inventory changed or parsed ambiguously")
    scene_record = inventory.get("scenes", {}).get(scene)
    if type(scene_record) is not dict:
        _fail(f"scene {scene} is absent from the sealed crosswalk inventory")
    artifact_dir = _resolved_repo_path(
        root, scene_record.get("artifact_dir"), "sealed crosswalk artifact directory"
    )
    if artifact_dir != (root / "data" / "annotations" / "hpe_transcript_crosswalk" / scene):
        _fail("sealed crosswalk artifact directory is not canonical for the scene")

    manifest_path = artifact_dir / crosswalk.MANIFEST_NAME
    manifest_payload, manifest_binding = _stable_read(manifest_path, "sealed crosswalk manifest")
    _assert_inventory_file_binding(
        scene_record.get("files", {}).get(crosswalk.MANIFEST_NAME),
        manifest_binding,
        "sealed crosswalk manifest",
    )
    crosswalk_manifest = _strict_json(manifest_payload, "sealed crosswalk manifest")
    if (
        crosswalk_manifest.get("artifact_kind") != crosswalk.ARTIFACT_KIND
        or crosswalk_manifest.get("format_version") != crosswalk.FORMAT_VERSION
        or crosswalk_manifest.get("scene") != scene
        or crosswalk_manifest.get("state") != "finalized"
    ):
        _fail("sealed crosswalk manifest is not a finalized artifact for the scene")
    accepted = scene_record.get("accepted_window")
    if crosswalk_manifest.get("accepted_window") != accepted:
        _fail("sealed crosswalk accepted window differs from the inventory")

    final_path = artifact_dir / crosswalk.FINAL_MAPPING_NAME
    final_payload, final_binding = _stable_read(final_path, "sealed final crosswalk CSV")
    _assert_inventory_file_binding(
        scene_record.get("files", {}).get(crosswalk.FINAL_MAPPING_NAME),
        final_binding,
        "sealed final crosswalk CSV",
    )
    _assert_binding(
        crosswalk_manifest.get("final_mapping"),
        final_binding,
        "crosswalk final_mapping",
        require_size=False,
    )
    final_frame = _strict_csv(final_payload, "sealed final crosswalk CSV")
    mapping_records = _validate_final_mapping(
        final_frame, scene_record.get("expected_hpe_persons"), scene
    )

    conflict_path = artifact_dir / crosswalk.CONFLICTS_NAME
    conflict_payload, conflict_binding = _stable_read(
        conflict_path, "sealed crosswalk conflict CSV"
    )
    _assert_inventory_file_binding(
        scene_record.get("files", {}).get(crosswalk.CONFLICTS_NAME),
        conflict_binding,
        "sealed crosswalk conflict CSV",
    )
    _assert_binding(
        crosswalk_manifest.get("conflict_report"),
        conflict_binding,
        "crosswalk conflict_report",
        require_size=False,
    )
    if not _strict_csv(conflict_payload, "sealed crosswalk conflict CSV").empty:
        _fail("sealed crosswalk conflict report is non-empty")

    gaze_sources = crosswalk_manifest.get("sources", {}).get("gaze")
    if type(gaze_sources) is not dict:
        _fail("sealed crosswalk manifest lacks the Gaze identity source chain")
    identity_declared = gaze_sources.get("manifest")
    if type(identity_declared) is not dict:
        _fail("sealed crosswalk Gaze identity-manifest binding is malformed")
    identity_path = _resolved_repo_path(
        root, identity_declared.get("path"), "sealed Gaze identity manifest"
    )
    canonical_identity_dir = root / "data" / "annotations" / "identity" / scene
    if identity_path != (canonical_identity_dir / "identity_annotations.json").resolve():
        _fail("sealed Gaze identity manifest is not at the canonical scene path")
    identity_payload, identity_binding = _stable_read(identity_path, "Gaze identity manifest")
    _assert_binding(
        identity_declared, identity_binding, "sealed Gaze identity manifest", require_size=False
    )
    identity_manifest_raw = _strict_json(identity_payload, "Gaze identity manifest")
    try:
        identity_manifest = identity_annotation.load_annotation_manifest(identity_path)
    except (OSError, ValueError, identity_annotation.IdentityAnnotationError) as exc:
        raise GazeSourceReadinessError(f"Gaze identity manifest failed validation: {exc}") from exc
    if identity_manifest != identity_manifest_raw:
        _fail("Gaze identity manifest changed or parsed ambiguously")
    day, scenario = scene.rsplit("__", maxsplit=1)
    if (
        identity_manifest.get("dataset") != day
        or str(identity_manifest.get("session")) != scenario
        or identity_manifest.get("label_to_slot") != PARTICIPANT_TO_SLOT
    ):
        _fail("Gaze identity manifest scene/slot semantics differ from sealed authority")
    scope = identity_manifest.get("annotation_scope")
    if type(scope) is not dict:
        _fail("Gaze identity annotation scope is malformed")
    source_fps = _exact_source_fps(scope.get("source_fps"), "identity source_fps")
    if (
        scope.get("review_status") != "confirmed"
        or scope.get("gaze_local_start_frame_inclusive") != accepted.get("local_start_inclusive")
        or scope.get("gaze_local_end_frame_exclusive") != accepted.get("local_end_exclusive")
    ):
        _fail("Gaze identity scope differs from the sealed accepted window")

    tracklet_declared = gaze_sources.get("tracklet_observations")
    identity_tracklet = identity_manifest.get("tracklet_observations")
    if not _portable_binding_matches(tracklet_declared, identity_tracklet):
        _fail("crosswalk and identity manifests disagree on tracklet observations")
    tracklet_path = _resolved_repo_path(
        root, tracklet_declared.get("path"), "sealed tracklet observations"
    )
    if tracklet_path != (canonical_identity_dir / "tracklet_observations.csv").resolve():
        _fail("sealed tracklet observations are not at the canonical scene path")
    tracklet_payload, tracklet_binding = _stable_read(tracklet_path, "sealed tracklet observations")
    _assert_binding(
        tracklet_declared,
        tracklet_binding,
        "sealed tracklet observations",
        require_size=False,
    )
    try:
        observations = identity_annotation.read_tracklet_observations(tracklet_path)
        identity_annotation._validate_tracklet_summaries(observations, identity_manifest)
    except (OSError, ValueError, identity_annotation.IdentityAnnotationError) as exc:
        raise GazeSourceReadinessError(f"tracklet observations failed validation: {exc}") from exc
    if list(observations.columns) != tracklet_declared.get("columns"):
        _fail("tracklet ordered schema differs from its sealed binding")
    if len(observations) != tracklet_declared.get("n_rows"):
        _fail("tracklet row count differs from its sealed binding")

    source = identity_manifest.get("source")
    if type(source) is not dict:
        _fail("Gaze identity manifest lacks its source binding")
    gaze_path = _resolved_repo_path(root, source.get("gaze_csv"), "bound raw Gaze CSV")
    gaze_root = (root / "gaze_features_export").resolve()
    try:
        gaze_path.relative_to(gaze_root)
    except ValueError as exc:
        raise GazeSourceReadinessError(
            "bound raw Gaze CSV is outside gaze_features_export"
        ) from exc
    gaze_payload, gaze_binding = _stable_read(gaze_path, "bound raw Gaze CSV")
    if source.get("gaze_csv_sha256") != gaze_binding.sha256:
        _fail("raw Gaze CSV differs from the identity source hash")
    raw_gaze = _strict_csv(gaze_payload, "bound raw Gaze CSV", low_memory=False)
    if tuple(raw_gaze.columns) != RAW_GAZE_COLUMNS:
        _fail("raw Gaze CSV ordered schema differs from the frozen 16-column export")
    participant_mask = raw_gaze["person_idx"].notna()
    non_null_targets = raw_gaze.loc[participant_mask, "gaze_target"].dropna()
    if not non_null_targets.isin(GAZE_TARGETS).all():
        _fail("raw Gaze CSV contains a target outside robot/participant/elsewhere")

    metadata_path = _resolved_repo_path(
        root, source.get("session_metadata"), "bound Gaze session metadata"
    )
    if metadata_path != gaze_path.with_name("session_metadata.json"):
        _fail("Gaze session metadata is not the bound CSV sibling")
    metadata_payload, metadata_binding = _stable_read(metadata_path, "bound Gaze session metadata")
    if source.get("session_metadata_sha256") != metadata_binding.sha256:
        _fail("Gaze session metadata differs from the identity source hash")
    metadata = _strict_json(metadata_payload, "bound Gaze session metadata")
    stream = metadata.get("stream")
    if (
        type(stream) is not dict
        or _exact_source_fps(stream.get("fps"), "metadata fps") != source_fps
    ):
        _fail("Gaze metadata FPS differs from the sealed identity timing authority")
    declared_video = source.get("declared_video")
    if (
        type(declared_video) is not dict
        or _exact_source_fps(declared_video.get("fps"), "declared video fps") != source_fps
        or declared_video.get("camera_dir") != accepted.get("camera")
        or declared_video.get("dataset") != day
        or str(declared_video.get("session")) != scenario
    ):
        _fail("Gaze declared-video identity differs from the sealed accepted source")

    try:
        raw_observations = identity_annotation.load_gaze_observations(gaze_path)
    except (OSError, ValueError, identity_annotation.IdentityAnnotationError) as exc:
        raise GazeSourceReadinessError(f"raw Gaze CSV failed strict replay: {exc}") from exc
    start = _exact_int(accepted.get("local_start_inclusive"), "accepted Gaze start", minimum=0)
    end = _exact_int(accepted.get("local_end_exclusive"), "accepted Gaze end", minimum=1)
    expected_observations = raw_observations.loc[
        (raw_observations["frame_index"] >= start) & (raw_observations["frame_index"] < end)
    ].copy()
    if expected_observations.empty:
        _fail("sealed Gaze window contains no raw observations")
    expected_observations = expected_observations.reset_index(drop=True)
    expected_observations["observation_id"] = np.arange(len(expected_observations), dtype=int)
    _assert_source_replay(expected_observations, observations)

    bridge, conflicts = identity_annotation.build_identity_bridge(observations, identity_manifest)
    if not conflicts.empty:
        _fail("Gaze identity bridge has duplicate/conflicting canonical identities")
    confirmed = bridge.loc[
        (bridge["annotation_status"] == "confirmed") & bridge["participant"].notna()
    ]
    duplicate = confirmed.duplicated(["frame_index", "participant"], keep=False)
    if duplicate.any():
        _fail("Gaze identity bridge has duplicate (frame, canonical participant) rows")
    if (
        not confirmed.loc[confirmed["participant"] == "R", "source_kind"]
        .eq("robot_candidate")
        .all()
        or not confirmed.loc[confirmed["participant"] != "R", "source_kind"]
        .eq("participant_detection")
        .all()
    ):
        _fail("Gaze identity source kinds do not preserve R versus non-R semantics")
    expected_slots = confirmed["participant"].map(PARTICIPANT_TO_SLOT)
    if not confirmed["slot"].equals(expected_slots):
        _fail("Gaze identity bridge slots differ from canonical participant semantics")

    analysis_identity = analysis.provenance.get("identity_contract")
    if type(analysis_identity) is not dict:
        _fail("analysis base lacks its sealed identity contract")
    analysis_inventory = analysis_identity.get("inventory")
    if (
        type(analysis_inventory) is not dict
        or Path(str(analysis_inventory.get("path"))).resolve() != inventory_path
        or analysis_inventory.get("sha256") != inventory_binding.sha256
        or analysis_identity.get("scene") != scene
        or analysis_identity.get("accepted_window") != accepted
    ):
        _fail("analysis-base identity binding differs from the sealed inventory/window")
    normalized_analysis_mapping = sorted(
        (
            int(record["hpe_person"]),
            str(record["participant"]),
            str(record["slot"]),
        )
        for record in analysis_identity.get("person_mapping", [])
    )
    normalized_mapping = sorted(
        (record["hpe_person"], record["participant"], record["slot"]) for record in mapping_records
    )
    if normalized_analysis_mapping != normalized_mapping:
        _fail("analysis-base identity mapping differs from the sealed final crosswalk")

    bindings = {
        "analysis_csv": analysis.csv,
        "analysis_provenance": analysis.sidecar,
        "sealed_crosswalk_inventory": inventory_binding,
        "crosswalk_manifest": manifest_binding,
        "final_crosswalk_csv": final_binding,
        "crosswalk_conflict_csv": conflict_binding,
        "identity_manifest": identity_binding,
        "tracklet_observations": tracklet_binding,
        "gaze_csv": gaze_binding,
        "gaze_metadata": metadata_binding,
    }
    reports = {
        "inventory": inventory_binding.report(),
        "crosswalk_manifest": manifest_binding.report(),
        "final_mapping": mapping_records,
        "identity_manifest": identity_binding.report(),
        "tracklet_observations": {
            **tracklet_binding.report(),
            "columns": list(observations.columns),
            "parsed_dtypes": {column: str(observations[column].dtype) for column in observations},
            "n_rows": int(len(observations)),
        },
        "source": {
            **gaze_binding.report(),
            "columns": list(raw_gaze.columns),
            "parsed_dtypes": {column: str(raw_gaze[column].dtype) for column in raw_gaze},
            "n_rows": int(len(raw_gaze)),
            "metadata": metadata_binding.report(),
        },
    }
    return _GazeSnapshot(
        bridge=bridge,
        source_fps=source_fps,
        accepted_window=dict(accepted),
        bindings=bindings,
        reports=reports,
    )


def derive_nullable_gaze_evidence(
    *,
    scene: str,
    ticks: np.ndarray,
    source_frames: np.ndarray,
    source_fps: int,
    bridge: pd.DataFrame,
    slot_presence: Mapping[str, bool],
) -> pd.DataFrame:
    """Create a four-slot nullable evidence grid from validated source rows.

    Missing/uncertain rows close to ``valid=false`` and NA.  A duplicate raw or
    canonical identity key is a whole-scene error, never an averaging rule.
    """

    if type(scene) is not str or not scene:
        _fail("scene must be a non-empty string")
    if type(ticks) is not np.ndarray or type(source_frames) is not np.ndarray:
        _fail("ticks and source_frames must be ndarrays")
    if ticks.dtype != np.int64 or source_frames.dtype != np.int64:
        _fail("ticks and source_frames must use int64")
    if ticks.ndim != 1 or source_frames.ndim != 1 or len(ticks) != len(source_frames):
        _fail("ticks and source_frames must be aligned one-dimensional arrays")
    _exact_source_fps(source_fps, "source_fps")
    if set(slot_presence) != set(SLOT_TO_PARTICIPANT) or any(
        type(value) is not bool for value in slot_presence.values()
    ):
        _fail("slot_presence must provide exact booleans for P1..P4")
    required = {
        "observation_id",
        "frame_index",
        "source_kind",
        "source_person_idx",
        "gaze_target",
        "participant",
        "slot",
        "annotation_status",
    }
    missing = sorted(required.difference(bridge.columns))
    if missing:
        _fail(f"identity bridge lacks evidence columns: {missing}")
    raw_participants = bridge.loc[bridge["source_kind"] == "participant_detection"]
    raw_duplicate = raw_participants.duplicated(["frame_index", "source_person_idx"], keep=False)
    if raw_duplicate.any():
        _fail("raw duplicate (frame_index, person_idx) invalidates the whole scene")
    confirmed = bridge.loc[
        (bridge["annotation_status"] == "confirmed") & bridge["participant"].notna()
    ].copy()
    canonical_duplicate = confirmed.duplicated(["frame_index", "participant"], keep=False)
    if canonical_duplicate.any():
        _fail("duplicate (frame, canonical participant) invalidates the whole scene")
    non_null_targets = confirmed.loc[
        confirmed["source_kind"] == "participant_detection", "gaze_target"
    ].dropna()
    if not non_null_targets.isin(GAZE_TARGETS).all():
        _fail("confirmed Gaze evidence contains an unsupported target category")

    lookup = {
        (int(row.frame_index), str(row.participant)): row
        for row in confirmed.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    for tick, source_frame in zip(ticks.tolist(), source_frames.tolist(), strict=True):
        robot_reference = lookup.get((int(source_frame), "R"))
        robot_reference_valid = (
            robot_reference is not None and str(robot_reference.source_kind) == "robot_candidate"
        )
        for slot, participant in SLOT_TO_PARTICIPANT.items():
            present = slot_presence[slot]
            record: dict[str, Any] = {
                "session_id": scene,
                "tick": int(tick),
                "timestamp_s": float(Fraction(int(tick), FRAME_RATE_HZ)),
                "source_frame_index": int(source_frame),
                "source_timestamp_s": float(Fraction(int(source_frame), source_fps)),
                "participant": participant,
                "slot": slot,
                "slot_present": present,
                "source_observation_id": pd.NA,
                "gaze_target_valid": False,
                "gaze_target": pd.NA,
                "gaze_target_reason": "absent_slot" if not present else "",
                "gaze_robot_valid": False,
                "gaze_robot": np.nan,
                "gaze_robot_reason": "absent_slot" if not present else "",
            }
            if not present:
                records.append(record)
                continue
            if participant == "R":
                record["gaze_target_reason"] = "robot_slot_gaze_not_observed"
                record["gaze_robot_reason"] = "robot_slot_gaze_not_observed"
                records.append(record)
                continue
            source_row = lookup.get((int(source_frame), participant))
            if source_row is None or str(source_row.source_kind) != "participant_detection":
                record["gaze_target_reason"] = "missing_confirmed_participant_observation"
                record["gaze_robot_reason"] = "missing_confirmed_participant_observation"
                records.append(record)
                continue
            record["source_observation_id"] = int(source_row.observation_id)
            target = source_row.gaze_target
            if pd.isna(target) or not str(target).strip():
                record["gaze_target_reason"] = "missing_gaze_target"
                record["gaze_robot_reason"] = "missing_gaze_target"
                records.append(record)
                continue
            target = str(target)
            if target not in GAZE_TARGETS:
                _fail("confirmed Gaze evidence contains an unsupported target category")
            record["gaze_target_valid"] = True
            record["gaze_target"] = target
            record["gaze_target_reason"] = "valid_2d_category"
            if not robot_reference_valid:
                record["gaze_robot_reason"] = "missing_confirmed_robot_bbox_reference"
                records.append(record)
                continue
            record["gaze_robot_valid"] = True
            record["gaze_robot"] = float(target == "robot")
            record["gaze_robot_reason"] = "valid_nullable_binary_derivation"
            records.append(record)

    evidence = pd.DataFrame.from_records(records, columns=EVIDENCE_COLUMNS)
    evidence["slot_present"] = evidence["slot_present"].astype(bool)
    evidence["gaze_target_valid"] = evidence["gaze_target_valid"].astype(bool)
    evidence["gaze_robot_valid"] = evidence["gaze_robot_valid"].astype(bool)
    evidence["source_observation_id"] = evidence["source_observation_id"].astype("Int64")
    if len(evidence) != len(ticks) * len(SLOT_TO_PARTICIPANT):
        _fail("nullable Gaze evidence grid is incomplete")
    invalid_target = ~evidence["gaze_target_valid"]
    invalid_binary = ~evidence["gaze_robot_valid"]
    if evidence.loc[invalid_target, "gaze_target"].notna().any():
        _fail("invalid Gaze target evidence must remain NA")
    if evidence.loc[invalid_binary, "gaze_robot"].notna().any():
        _fail("invalid gaze_robot evidence must remain NaN, never zero-filled")
    valid_binary = evidence.loc[evidence["gaze_robot_valid"], "gaze_robot"]
    if not valid_binary.isin((0.0, 1.0)).all():
        _fail("valid gaze_robot evidence must be binary")
    robot_rows = evidence["participant"] == "R"
    if (
        evidence.loc[robot_rows, "gaze_target_valid"].any()
        or evidence.loc[robot_rows, "gaze_robot_valid"].any()
        or evidence.loc[robot_rows, "gaze_target"].notna().any()
        or evidence.loc[robot_rows, "gaze_robot"].notna().any()
    ):
        _fail("R has no source Gaze row and must stay invalid/NA")
    absent = ~evidence["slot_present"]
    if (
        evidence.loc[absent, "gaze_target_valid"].any()
        or evidence.loc[absent, "gaze_robot_valid"].any()
        or evidence.loc[absent, "gaze_target"].notna().any()
        or evidence.loc[absent, "gaze_robot"].notna().any()
    ):
        _fail("absent slots must stay distinct from observed negative Gaze")
    return evidence


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _coverage_report(evidence: pd.DataFrame) -> dict[str, Any]:
    participants: dict[str, Any] = {}
    for participant, slot in PARTICIPANT_TO_SLOT.items():
        rows = evidence.loc[evidence["participant"] == participant]
        present = int(rows["slot_present"].sum())
        target_valid = int(rows["gaze_target_valid"].sum())
        binary_valid = int(rows["gaze_robot_valid"].sum())
        target_counts = Counter(
            str(value) for value in rows.loc[rows["gaze_target_valid"], "gaze_target"]
        )
        participants[participant] = {
            "slot": slot,
            "total_ticks": int(len(rows)),
            "present_ticks": present,
            "gaze_target_valid_ticks": target_valid,
            "gaze_target_coverage_of_present": _ratio(target_valid, present),
            "gaze_robot_valid_ticks": binary_valid,
            "gaze_robot_coverage_of_present": _ratio(binary_valid, present),
            "gaze_target_counts": {
                category: int(target_counts.get(category, 0)) for category in sorted(GAZE_TARGETS)
            },
            "gaze_target_invalid_reasons": {
                str(reason): int(count)
                for reason, count in sorted(
                    Counter(rows.loc[~rows["gaze_target_valid"], "gaze_target_reason"]).items()
                )
            },
            "gaze_robot_invalid_reasons": {
                str(reason): int(count)
                for reason, count in sorted(
                    Counter(rows.loc[~rows["gaze_robot_valid"], "gaze_robot_reason"]).items()
                )
            },
        }
    non_robot_present = evidence["slot_present"] & (evidence["participant"] != "R")
    denominator = int(non_robot_present.sum())
    return {
        "participants": participants,
        "eligible_nonrobot_present_ticks": denominator,
        "gaze_target_valid_ticks": int(evidence.loc[non_robot_present, "gaze_target_valid"].sum()),
        "gaze_target_coverage": _ratio(
            int(evidence.loc[non_robot_present, "gaze_target_valid"].sum()), denominator
        ),
        "gaze_robot_valid_ticks": int(evidence.loc[non_robot_present, "gaze_robot_valid"].sum()),
        "gaze_robot_coverage": _ratio(
            int(evidence.loc[non_robot_present, "gaze_robot_valid"].sum()), denominator
        ),
    }


def _d3_offset(provenance: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    top = provenance.get("offset_quality")
    contract = provenance.get("analysis_window_contract")
    nested = contract.get("d3_quality") if type(contract) is dict else None
    if type(top) is not dict or type(nested) is not dict or top != nested:
        _fail("analysis-base D3 quality differs between its bound contract copies")
    if (
        top.get("convention") != "shared_time = stream_local_time + offset_s"
        or top.get("stream") != "6cam"
        or top.get("self_check_ok") is not True
    ):
        _fail("analysis-base D3 timing convention/source/self-check is invalid")
    _finite_fraction(top.get("offset_s"), "D3 offset_s")
    return top["offset_s"], dict(top)


def _assert_bindings_unchanged(bindings: Mapping[str, _FileBinding]) -> None:
    for label, initial in bindings.items():
        _payload, current = _stable_read(initial.path, label)
        if current != initial:
            _fail(f"source document changed during Gaze preflight: {label}")


def audit_gaze_source_preflight(scene: str) -> GazeSourcePreflight:
    """Replay one canonical scene and derive diagnostic nullable evidence.

    There is intentionally no authority override: callers cannot turn a fixture
    tree into a formal-looking report.  The function has no output path and
    writes no file.
    """

    if type(scene) is not str or scene not in analysis_base.EXPECTED_TICK_RANGES:
        _fail("scene is outside the frozen ten-scene dissertation inventory")
    root = _repository_root()
    analysis = _load_analysis_snapshot(scene, root)
    gaze = _load_gaze_snapshot(scene, root, analysis)
    offset, d3 = _d3_offset(analysis.provenance)
    start = _exact_int(
        gaze.accepted_window.get("local_start_inclusive"),
        "accepted Gaze start",
        minimum=0,
    )
    end = _exact_int(
        gaze.accepted_window.get("local_end_exclusive"),
        "accepted Gaze end",
        minimum=1,
    )
    source_frames = past_only_frame_indices(
        analysis.ticks,
        source_fps=gaze.source_fps,
        d3_offset_s=offset,
        accepted_start=start,
        accepted_end=end,
    )
    evidence = derive_nullable_gaze_evidence(
        scene=scene,
        ticks=analysis.ticks,
        source_frames=source_frames,
        source_fps=gaze.source_fps,
        bridge=gaze.bridge,
        slot_presence=analysis.slot_presence,
    )
    coverage = _coverage_report(evidence)
    first_tick = int(analysis.ticks[0])
    last_tick = int(analysis.ticks[-1])
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": scene,
        "output_role": OUTPUT_ROLE,
        "trust_model": TRUST_MODEL,
        "assembly_eligible": False,
        "counted_input_ready": False,
        "writer_available": False,
        "artifact_or_feature_published": False,
        "analysis_base": {
            **analysis.csv.report(),
            "provenance": analysis.sidecar.report(),
            "artifact_kind": analysis.provenance["artifact_kind"],
            "tick_start_inclusive": first_tick,
            "tick_end_exclusive": last_tick + 1,
            "n_ticks": int(len(analysis.ticks)),
            "slot_presence": dict(analysis.slot_presence),
        },
        "sealed_identity": {
            "inventory": gaze.reports["inventory"],
            "crosswalk_manifest": gaze.reports["crosswalk_manifest"],
            "final_mapping": gaze.reports["final_mapping"],
            "identity_manifest": gaze.reports["identity_manifest"],
            "tracklet_observations": gaze.reports["tracklet_observations"],
        },
        "source": {
            **gaze.reports["source"],
            "source_fps": gaze.source_fps,
            "accepted_window": dict(gaze.accepted_window),
            "allowed_gaze_targets": sorted(GAZE_TARGETS),
        },
        "time_mapping": {
            "analysis_axis": "mic_A_shared_seconds",
            "source_axis": "sealed_6cam_logical_frame_index",
            "d3": d3,
            "rule": "floor((tick / 25 - d3_offset_s) * source_fps)",
            "arithmetic": "exact_rational_Fraction_from_decimal_scalar",
            "past_only": True,
            "nearest_frame": False,
            "interpolation": False,
            "forward_fill": False,
            "extrapolation": False,
            "source_frame_start_inclusive": int(source_frames[0]),
            "source_frame_end_inclusive": int(source_frames[-1]),
            "unique_source_frames": int(np.unique(source_frames).size),
        },
        "evidence": {
            "n_rows": int(len(evidence)),
            "rows_per_analysis_tick": 4,
            "columns": list(evidence.columns),
            "parsed_dtypes": {column: str(evidence[column].dtype) for column in evidence},
            "nullable_missing_is_zero": False,
            "coverage": coverage,
        },
        "scientific_limits": {
            "evidence_kind": "nullable_2d_categorical_gaze_target",
            "supported_targets": sorted(GAZE_TARGETS),
            "participant_target_identity_resolved": False,
            "participant_wise_gazed_at_derived": False,
            "three_dimensional_intersection_claimed": False,
            "robot_slot_gaze_observed": False,
            "robot_slot_invalid_means_negative_gaze": False,
            "absent_slot_invalid_means_negative_gaze": False,
            "model_input_promoted": False,
        },
    }
    _assert_bindings_unchanged(gaze.bindings)
    return GazeSourcePreflight(scene=scene, evidence_df=evidence, report=report)


def validate_frozen_preflight_matrix(
    results: Mapping[str, GazeSourcePreflight],
) -> None:
    """Require the exact ten-scene/tick matrix without granting readiness."""

    if set(results) != set(analysis_base.EXPECTED_TICK_RANGES):
        _fail("Gaze preflight matrix does not exactly cover the frozen ten scenes")
    total_ticks = 0
    for scene, (start, end) in analysis_base.EXPECTED_TICK_RANGES.items():
        result = results[scene]
        if type(result) is not GazeSourcePreflight or result.scene != scene:
            _fail(f"Gaze preflight result has the wrong scene/type for {scene}")
        if len(result.evidence_df) != (end - start) * 4:
            _fail(f"Gaze evidence grid row count differs for {scene}")
        if (
            result.report.get("assembly_eligible") is not False
            or result.report.get("counted_input_ready") is not False
            or result.report.get("writer_available") is not False
        ):
            _fail("a Gaze preflight result cannot claim publication/counting readiness")
        total_ticks += end - start
    if total_ticks != analysis_base.EXPECTED_TOTAL_ROWS:
        _fail("Gaze preflight total differs from the frozen 78,582 analysis ticks")


__all__ = [
    "ARTIFACT_KIND",
    "FORMAT_VERSION",
    "GAZE_TARGETS",
    "GazeSourcePreflight",
    "GazeSourceReadinessError",
    "MATRIX_ARTIFACT_KIND",
    "audit_gaze_source_preflight",
    "derive_nullable_gaze_evidence",
    "past_only_frame_indices",
    "validate_frozen_preflight_matrix",
]
