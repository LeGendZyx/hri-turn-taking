"""Conservative bbox tracklets and auditable manual identity annotations.

The gaze export's ``person_idx`` is explicitly a within-frame detection index.
This module therefore never uses it to link observations over time.  Instead it
builds short, conservative bounding-box tracklets which a human reviewer can
label as A/B/C/R.  Source exports are never modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ARTIFACT_KIND = "vap_manual_identity_annotation"
FORMAT_VERSION = 2
LEGACY_FORMAT_VERSION = 1
REVIEW_ANCHOR_INTERVAL_S = 5.0
PARTICIPANT_LABELS = ("A", "B", "C", "R")
LABEL_TO_SLOT = {"R": "P1", "A": "P2", "B": "P3", "C": "P4"}

_ATOMIC_REPLACE_ATTEMPTS = 10
_ATOMIC_REPLACE_INITIAL_DELAY_S = 0.05
_ATOMIC_REPLACE_MAX_DELAY_S = 0.5

_BBOX_COLUMNS = ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")
_ROBOT_BBOX_COLUMNS = ("robot_x0", "robot_y0", "robot_x1", "robot_y1")
_REQUIRED_GAZE_COLUMNS = {
    "frame_index",
    "timestamp_sec",
    "robot_detected",
    "person_idx",
    *_BBOX_COLUMNS,
    *_ROBOT_BBOX_COLUMNS,
}


class IdentityAnnotationError(ValueError):
    """Raised when an identity annotation artifact is unsafe or inconsistent."""


@dataclass(frozen=True)
class TrackerConfig:
    """Conservative gates for linking detections into reviewable tracklets."""

    max_gap_s: float = 0.15
    min_iou: float = 0.05
    strong_center_distance_ratio: float = 0.45
    max_center_distance_ratio: float = 1.0
    max_area_ratio: float = 2.5
    max_link_cost: float = 1.15
    guard_link_cost: float = 0.75
    crowding_iou_threshold: float = 0.25
    crowding_center_distance_ratio: float = 0.80
    ambiguity_margin: float = 0.08

    def validate(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise IdentityAnnotationError("tracker parameters must all be finite")
        if self.max_gap_s <= 0:
            raise IdentityAnnotationError("max_gap_s must be positive")
        if not 0 <= self.min_iou <= 1:
            raise IdentityAnnotationError("min_iou must be in [0, 1]")
        if self.strong_center_distance_ratio < 0:
            raise IdentityAnnotationError("strong_center_distance_ratio must be non-negative")
        if self.max_center_distance_ratio <= self.strong_center_distance_ratio:
            raise IdentityAnnotationError(
                "max_center_distance_ratio must exceed strong_center_distance_ratio"
            )
        if self.max_area_ratio <= 1:
            raise IdentityAnnotationError("max_area_ratio must exceed 1")
        if self.max_link_cost <= 0 or self.ambiguity_margin < 0:
            raise IdentityAnnotationError("link-cost gates must be non-negative")
        if not 0 < self.guard_link_cost <= self.max_link_cost:
            raise IdentityAnnotationError("guard_link_cost must be in (0, max_link_cost]")
        if not 0 < self.crowding_iou_threshold <= 1:
            raise IdentityAnnotationError("crowding_iou_threshold must be in (0, 1]")
        if self.crowding_center_distance_ratio <= 0:
            raise IdentityAnnotationError(
                "crowding_center_distance_ratio must be positive"
            )


TRACKER_ALGORITHM = "crowding_quarantine_v2"

_LEGACY_AUTOMATIC_MISSING_POLICY = (
    "crowding_guard_and_singleton_participant_tracklets_are_uncertain"
)
_AUTOMATIC_MISSING_POLICY_KIND = "conservative_tracklet_quarantine"
_AUTOMATIC_MISSING_POLICY_VERSION = 2
_EARLY_V2_NO_AUTOMATIC_QUARANTINE_KIND = "legacy_no_automatic_quarantine"
_SHORT_TRACKLET_DURATION_MEASURE = "max(timestamp_sec)-min(timestamp_sec)"
_SHORT_TRACKLET_BOUNDARY_TOLERANCE_S = 1e-12
_EARLY_V2_REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "annotation_scope",
        "annotations",
        "artifact_kind",
        "created_at",
        "dataset",
        "format_version",
        "label_to_slot",
        "review_policy",
        "scope_counts",
        "session",
        "source",
        "tracker_config",
        "tracklet_observations",
        "updated_at",
    }
)
_EARLY_V2_ALLOWED_TOP_LEVEL_KEYS = _EARLY_V2_REQUIRED_TOP_LEVEL_KEYS | {"video"}
_LEGACY_V1_REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "annotations",
        "artifact_kind",
        "created_at",
        "dataset",
        "format_version",
        "label_to_slot",
        "review_policy",
        "session",
        "source",
        "tracker_config",
        "tracklet_observations",
        "updated_at",
    }
)
_LEGACY_V1_ALLOWED_TOP_LEVEL_KEYS = _LEGACY_V1_REQUIRED_TOP_LEVEL_KEYS | {"video"}


def _validate_short_tracklet_max_duration_s(value: float) -> float:
    if isinstance(value, bool):
        raise IdentityAnnotationError(
            "short_tracklet_max_duration_s must be a finite non-negative number"
        )
    try:
        duration_s = float(value)
    except (TypeError, ValueError) as exc:
        raise IdentityAnnotationError(
            "short_tracklet_max_duration_s must be a finite non-negative number"
        ) from exc
    if not math.isfinite(duration_s) or duration_s < 0:
        raise IdentityAnnotationError(
            "short_tracklet_max_duration_s must be a finite non-negative number"
        )
    return duration_s


def _automatic_missing_policy(short_tracklet_max_duration_s: float) -> dict[str, Any]:
    duration_s = _validate_short_tracklet_max_duration_s(
        short_tracklet_max_duration_s
    )
    return {
        "kind": _AUTOMATIC_MISSING_POLICY_KIND,
        "version": _AUTOMATIC_MISSING_POLICY_VERSION,
        "rules": [
            "all_crowding_guard_tracklets",
            "singleton_participant_tracklets",
            "short_participant_tracklets",
        ],
        "short_tracklet_max_duration_s": duration_s,
        "short_tracklet_duration_measure": _SHORT_TRACKLET_DURATION_MEASURE,
        "short_tracklet_duration_comparison": "<=",
        "short_tracklet_boundary_tolerance_s": _SHORT_TRACKLET_BOUNDARY_TOLERANCE_S,
        "identity_effect": "uncertain_missing",
    }


def _early_v2_no_automatic_quarantine_policy() -> dict[str, Any]:
    return {
        "kind": _EARLY_V2_NO_AUTOMATIC_QUARANTINE_KIND,
        "version": 1,
        "rules": [],
        "identity_effect": "preserve_existing_manual_state",
    }


def _normalise_automatic_missing_policy(value: Any) -> dict[str, Any]:
    if value == _LEGACY_AUTOMATIC_MISSING_POLICY:
        return _automatic_missing_policy(0.0)
    if not isinstance(value, dict):
        raise IdentityAnnotationError(
            "manifest automatic_missing_policy is missing or unsupported"
        )
    duration_s = _validate_short_tracklet_max_duration_s(
        value.get("short_tracklet_max_duration_s")
    )
    expected = _automatic_missing_policy(duration_s)
    if value != expected:
        raise IdentityAnnotationError(
            "manifest automatic_missing_policy is malformed or unsupported"
        )
    return expected


def _is_recognised_early_v2_manifest(manifest: dict[str, Any]) -> bool:
    """Recognise the narrow pre-auto-quarantine v2 schema used by 03_26__3."""

    annotations = manifest.get("annotations")
    source = manifest.get("source")
    return bool(
        manifest.get("format_version") == FORMAT_VERSION
        and set(manifest).issubset(_EARLY_V2_ALLOWED_TOP_LEVEL_KEYS)
        and _EARLY_V2_REQUIRED_TOP_LEVEL_KEYS.issubset(manifest)
        and "automatic_missing_policy" not in manifest
        and "tracker_algorithm" not in manifest
        and isinstance(annotations, list)
        and all(
            isinstance(item, dict)
            and "auto_quarantined" not in item
            and "auto_quarantine_reason" not in item
            for item in annotations
        )
        and isinstance(manifest.get("tracker_config"), dict)
        and manifest.get("review_policy")
        == {
            "periodic_anchor_interval_s": REVIEW_ANCHOR_INTERVAL_S,
            "gap_anchor_rule": "both sides when gap > max(0.075 s, 2.5x median)",
        }
        and isinstance(source, dict)
        and {
            "gaze_csv",
            "gaze_csv_sha256",
            "session_metadata",
            "session_metadata_sha256",
            "declared_video",
        }.issubset(source)
    )


def _is_recognised_legacy_v1_manifest(manifest: dict[str, Any]) -> bool:
    """Recognise the narrow original full-session v1 schema used by migration."""

    annotations = manifest.get("annotations")
    source = manifest.get("source")
    return bool(
        manifest.get("format_version") == LEGACY_FORMAT_VERSION
        and set(manifest).issubset(_LEGACY_V1_ALLOWED_TOP_LEVEL_KEYS)
        and _LEGACY_V1_REQUIRED_TOP_LEVEL_KEYS.issubset(manifest)
        and "automatic_missing_policy" not in manifest
        and "tracker_algorithm" not in manifest
        and isinstance(annotations, list)
        and all(
            isinstance(item, dict)
            and "auto_quarantined" not in item
            and "auto_quarantine_reason" not in item
            for item in annotations
        )
        and isinstance(manifest.get("tracker_config"), dict)
        and manifest.get("review_policy")
        == {
            "periodic_anchor_interval_s": REVIEW_ANCHOR_INTERVAL_S,
            "gap_anchor_rule": "both sides when gap > max(0.075 s, 2.5x median)",
        }
        and isinstance(source, dict)
        and {
            "gaze_csv",
            "gaze_csv_sha256",
            "session_metadata",
            "session_metadata_sha256",
            "declared_video",
        }.issubset(source)
    )


def _is_explicit_legacy_no_quarantine_migration(manifest: dict[str, Any]) -> bool:
    migration = manifest.get("migration")
    return bool(
        manifest.get("format_version") == FORMAT_VERSION
        and manifest.get("automatic_missing_policy")
        == _early_v2_no_automatic_quarantine_policy()
        and manifest.get("tracker_algorithm") == TRACKER_ALGORITHM
        and isinstance(migration, dict)
        and migration.get("action") == "legacy_full_video_to_hpe_core"
        and migration.get("from_format_version") == LEGACY_FORMAT_VERSION
    )


def _manifest_automatic_missing_policy(manifest: dict[str, Any]) -> dict[str, Any]:
    if "automatic_missing_policy" not in manifest:
        if _is_recognised_early_v2_manifest(manifest) or _is_recognised_legacy_v1_manifest(
            manifest
        ):
            return _early_v2_no_automatic_quarantine_policy()
        raise IdentityAnnotationError(
            "manifest automatic_missing_policy is missing or unsupported"
        )
    value = manifest["automatic_missing_policy"]
    if value == _early_v2_no_automatic_quarantine_policy():
        if _is_explicit_legacy_no_quarantine_migration(manifest):
            return _early_v2_no_automatic_quarantine_policy()
        raise IdentityAnnotationError(
            "legacy no-quarantine policy lacks a verified migration record"
        )
    return _normalise_automatic_missing_policy(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise IdentityAnnotationError(f"{label} must be an exact positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IdentityAnnotationError(
            f"{label} must be an exact positive integer"
        ) from exc
    try:
        exact = float(value) == number
    except (TypeError, ValueError, OverflowError):
        exact = False
    if number <= 0 or not exact:
        raise IdentityAnnotationError(f"{label} must be an exact positive integer")
    return number


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise IdentityAnnotationError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IdentityAnnotationError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise IdentityAnnotationError(f"{label} must be a finite number")
    return number


def _ceil_frame(seconds: float, fps: float, label: str) -> int:
    """Match HPE's Decimal ceiling for a half-open time boundary."""

    try:
        product = Decimal(str(seconds)) * Decimal(str(fps))
        return int(product.to_integral_value(rounding=ROUND_CEILING))
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise IdentityAnnotationError(f"cannot convert {label} to a frame index") from exc


def load_hpe_annotation_scope(
    profile_path: str | Path,
    run_metadata_path: str | Path,
    *,
    dataset: str,
    session: str,
    declared_video: dict[str, Any],
) -> dict[str, Any]:
    """Bind one gaze export to an owner-confirmed HPE formal-dialogue window.

    HPE profiles use the concatenated six-camera logical frame axis.  A gaze
    export can instead cover one physical AVI segment whose local frame zero is
    later on that axis (notably ``05_14`` ``(1).avi`` files).  The HPE extractor
    marker is therefore required to prove the segment's logical origin.
    """

    profile = Path(profile_path).resolve()
    run_metadata = Path(run_metadata_path).resolve()
    for path, label in ((profile, "HPE geometry profile"), (run_metadata, "HPE run metadata")):
        if not path.is_file():
            raise IdentityAnnotationError(f"{label} does not exist: {path}")
    try:
        with profile.open("r", encoding="utf-8") as handle:
            profile_doc = json.load(handle)
        with run_metadata.open("r", encoding="utf-8") as handle:
            run_doc = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityAnnotationError("cannot read HPE core-window provenance") from exc
    if not isinstance(profile_doc, dict) or not isinstance(run_doc, dict):
        raise IdentityAnnotationError("HPE provenance roots must be JSON objects")
    if profile_doc.get("review_status") != "confirmed":
        raise IdentityAnnotationError("HPE geometry profile is not owner-confirmed")
    if profile_doc.get("day") != str(dataset) or str(profile_doc.get("scenario")) != str(
        session
    ):
        raise IdentityAnnotationError("HPE geometry profile identifies a different scene")
    if profile.stem != f"{dataset}__{session}":
        raise IdentityAnnotationError("HPE geometry profile filename identifies a different scene")
    schema_version = _exact_positive_int(
        profile_doc.get("schema_version"), "HPE profile schema_version"
    )
    window = profile_doc.get("core_window_sec")
    if not isinstance(window, dict):
        raise IdentityAnnotationError("HPE profile core_window_sec must be a JSON object")
    start_sec = _finite_number(
        window.get("start_inclusive"), "core_window_sec.start_inclusive"
    )
    end_sec = _finite_number(
        window.get("end_exclusive"), "core_window_sec.end_exclusive"
    )
    profile_fps = _finite_number(profile_doc.get("source_fps"), "HPE source_fps")
    declared_fps = _finite_number(declared_video.get("fps"), "gaze metadata fps")
    if start_sec < 0 or end_sec <= start_sec:
        raise IdentityAnnotationError("HPE core window must be a non-empty non-negative range")
    if profile_fps <= 0 or not math.isclose(
        profile_fps, declared_fps, rel_tol=0.0, abs_tol=1e-9
    ):
        raise IdentityAnnotationError("HPE and gaze source FPS differ")
    logical_start = _ceil_frame(start_sec, profile_fps, "core start")
    logical_end = _ceil_frame(end_sec, profile_fps, "core end")
    if logical_end <= logical_start:
        raise IdentityAnnotationError("HPE core window contains no logical frame")

    if not math.isclose(
        _finite_number(run_doc.get("source_fps"), "HPE run source_fps"),
        declared_fps,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise IdentityAnnotationError("HPE run metadata and gaze FPS differ")
    for key, declared_key in (("width", "width"), ("height", "height")):
        if _exact_positive_int(run_doc.get(key), f"HPE run {key}") != _exact_positive_int(
            declared_video.get(declared_key), f"gaze metadata {declared_key}"
        ):
            raise IdentityAnnotationError(f"HPE run and gaze video {key} differ")
    segments_value = run_doc.get("segments")
    if segments_value is None:
        # Early HPE single-video markers (including the formal 03_20 runs) predate
        # ``segments`` and bind the one physical source through ``input``.  This
        # fallback is deliberately limited to a complete, start-at-zero run; a
        # partial extraction cannot prove the gaze video's logical frame origin.
        legacy_input = run_doc.get("input")
        if not isinstance(legacy_input, str) or not legacy_input.strip():
            raise IdentityAnnotationError(
                "HPE run metadata must bind either segments or one legacy input path"
            )
        params = run_doc.get("params")
        if not isinstance(params, dict):
            raise IdentityAnnotationError("legacy HPE run metadata lacks params")
        start_frame = params.get("start_frame")
        max_frames = params.get("max_frames")
        if start_frame != 0 or max_frames is not None:
            raise IdentityAnnotationError(
                "legacy HPE run must cover the complete source from frame zero"
            )
        segments = [legacy_input]
        run_source_contract = "legacy_single_input_complete_v1"
    else:
        if not isinstance(segments_value, list) or not segments_value or not all(
            isinstance(value, str) and value.strip() for value in segments_value
        ):
            raise IdentityAnnotationError(
                "HPE run metadata segments must be non-empty paths"
            )
        segments = list(segments_value)
        if run_doc.get("input") != segments[0]:
            raise IdentityAnnotationError("HPE run input must equal its first segment")
        input_files = run_doc.get("input_files")
        params = run_doc.get("params")
        if (
            not isinstance(params, dict)
            or isinstance(params.get("start_frame"), bool)
            or params.get("start_frame") != 0
            or params.get("max_frames") is not None
        ):
            raise IdentityAnnotationError(
                "HPE segmented run must cover the complete source from frame zero"
            )
        if input_files is None and run_doc.get("declared_expected_frames") is None:
            if len(segments) != 1 or run_doc.get("segments_decoded_frames") is not None:
                raise IdentityAnnotationError(
                    "legacy HPE segmented marker must describe one complete source"
                )
            run_source_contract = "legacy_single_segment_complete_v1"
        else:
            if (
                not isinstance(input_files, list)
                or len(input_files) != len(segments)
                or any(
                    not isinstance(item, dict) or item.get("path") != segment
                    for item, segment in zip(input_files, segments, strict=True)
                )
            ):
                raise IdentityAnnotationError(
                    "HPE run input_files do not bind its segments"
                )
            run_source_contract = "segments_v2"
    expected_camera = str(declared_video.get("camera_dir", ""))
    expected_source_session = str(declared_video.get("session", ""))
    for segment in segments:
        segment_path = Path(segment)
        if (
            segment_path.parent.name.casefold() != expected_camera.casefold()
            or segment_path.parent.parent.name != expected_source_session
            or len(segment_path.parents) < 4
            or segment_path.parents[3].name != str(dataset)
        ):
            raise IdentityAnnotationError(
                "HPE segment path identifies a different day/session/camera"
            )
    decoded = run_doc.get("segments_decoded_frames")
    processed = _exact_positive_int(
        run_doc.get("processed_frames"), "HPE processed_frames"
    )
    if decoded is None:
        if len(segments) != 1:
            raise IdentityAnnotationError("multi-segment HPE run lacks decoded frame counts")
        decoded_frames = [processed]
    else:
        if not isinstance(decoded, list) or len(decoded) != len(segments):
            raise IdentityAnnotationError("HPE segment frame counts do not match segments")
        decoded_frames = [
            _exact_positive_int(value, "HPE segment decoded frame count") for value in decoded
        ]
        if sum(decoded_frames) != processed:
            raise IdentityAnnotationError("HPE segment frame counts do not sum to processed_frames")
    declared_expected = run_doc.get("declared_expected_frames")
    if run_source_contract == "segments_v2":
        if (
            _exact_positive_int(
                declared_expected, "HPE declared_expected_frames"
            )
            != processed
        ):
            raise IdentityAnnotationError(
                "HPE declared_expected_frames differs from processed_frames"
            )

    basename = str(declared_video.get("basename", ""))
    matching_indices = [
        index for index, segment in enumerate(segments) if Path(segment).name == basename
    ]
    if len(matching_indices) != 1:
        raise IdentityAnnotationError(
            "gaze source basename does not uniquely match the HPE run segments"
        )
    segment_index = matching_indices[0]
    segment_start = sum(decoded_frames[:segment_index])
    segment_frames = decoded_frames[segment_index]
    gaze_frames = _exact_positive_int(
        declared_video.get("frame_count"), "gaze metadata frame_count"
    )
    if gaze_frames != segment_frames:
        raise IdentityAnnotationError("gaze frame count differs from its HPE source segment")
    segment_end = segment_start + segment_frames
    overlap_start = max(logical_start, segment_start)
    overlap_end = min(logical_end, segment_end)
    if overlap_end <= overlap_start:
        raise IdentityAnnotationError("gaze source segment does not overlap the HPE core window")
    local_start = overlap_start - segment_start
    local_end = overlap_end - segment_start

    return {
        "kind": "hpe_confirmed_core_window",
        "profile_path": str(profile),
        "profile_sha256": sha256_file(profile),
        "profile_schema_version": schema_version,
        "run_metadata_path": str(run_metadata),
        "run_metadata_sha256": sha256_file(run_metadata),
        "run_source_contract": run_source_contract,
        "day": str(dataset),
        "scenario": str(session),
        "review_status": "confirmed",
        "source_fps": profile_fps,
        "core_start_sec_inclusive": start_sec,
        "core_end_sec_exclusive": end_sec,
        "core_start_logical_frame_inclusive": logical_start,
        "core_end_logical_frame_exclusive": logical_end,
        "gaze_segment_index": segment_index,
        "gaze_segment_basename": basename,
        "gaze_segment_logical_start_frame": segment_start,
        "gaze_segment_frame_count": segment_frames,
        "gaze_local_start_frame_inclusive": local_start,
        "gaze_local_end_frame_exclusive": local_end,
        "core_coverage_fraction": (overlap_end - overlap_start)
        / (logical_end - logical_start),
        "filter_rule": "local_start <= frame_index < local_end",
    }


def load_export_metadata(gaze_csv: Path) -> tuple[Path, str, dict[str, Any]]:
    metadata_path = gaze_csv.with_name("session_metadata.json")
    if not metadata_path.is_file():
        raise IdentityAnnotationError(
            f"required sibling session_metadata.json is missing beside {gaze_csv.name}"
        )
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityAnnotationError(f"cannot read session metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise IdentityAnnotationError("session metadata root must be a JSON object")
    stream = metadata.get("stream")
    input_section = metadata.get("input")
    if not isinstance(stream, dict) or not isinstance(input_section, dict):
        raise IdentityAnnotationError("session metadata requires object-valued input and stream")

    declared_path = input_section.get("video_path")
    if not isinstance(declared_path, str) or not declared_path.strip():
        raise IdentityAnnotationError(
            "session metadata input.video_path must be a non-empty string"
        )
    numeric_fields = {
        "frame_count": stream.get("total_frames_in_video"),
        "fps": stream.get("fps"),
        "width": stream.get("frame_width"),
        "height": stream.get("frame_height"),
    }
    if any(isinstance(value, bool) for value in numeric_fields.values()):
        raise IdentityAnnotationError("session metadata video dimensions must be numeric, not bool")
    try:
        frame_count = int(numeric_fields["frame_count"])
        fps = float(numeric_fields["fps"])
        width = int(numeric_fields["width"])
        height = int(numeric_fields["height"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise IdentityAnnotationError("session metadata has invalid video dimensions") from exc
    if (
        frame_count <= 0
        or width <= 0
        or height <= 0
        or not math.isfinite(fps)
        or fps <= 0
        or float(numeric_fields["frame_count"]) != frame_count
        or float(numeric_fields["width"]) != width
        or float(numeric_fields["height"]) != height
    ):
        raise IdentityAnnotationError("session metadata has invalid video dimensions")
    source_parts = PurePosixPath(declared_path.replace("\\", "/")).parts
    if len(source_parts) < 5 or source_parts[0].casefold() != "data":
        raise IdentityAnnotationError(
            "session metadata input.video_path must follow "
            "Data/<dataset>/.../<session>/<camera>/<file>"
        )
    declared_video = {
        "source_path": declared_path,
        "basename": Path(declared_path).name,
        "dataset": source_parts[1],
        "session": source_parts[-3],
        "camera_dir": source_parts[-2],
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }
    return metadata_path.resolve(), sha256_file(metadata_path), declared_video


def _normalise_person_idx(value: Any) -> str:
    if pd.isna(value):
        raise IdentityAnnotationError("person_idx may not be missing")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise IdentityAnnotationError("person_idx may not be empty")
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _validate_bbox_values(frame: pd.DataFrame, columns: tuple[str, str, str, str]) -> None:
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise IdentityAnnotationError(f"non-finite bbox values in columns {columns}")
    if np.any(values[:, 2] <= values[:, 0]) or np.any(values[:, 3] <= values[:, 1]):
        raise IdentityAnnotationError(f"degenerate bbox values in columns {columns}")


def load_gaze_observations(
    path: str | Path,
    *,
    include_robot: bool = True,
    start_frame_inclusive: int | None = None,
    end_frame_exclusive: int | None = None,
) -> pd.DataFrame:
    """Load gaze detections and optionally deduplicate the per-frame robot bbox.

    Participant and robot candidates remain separate ``source_kind`` values so
    the heuristic robot box can never silently become a confirmed R identity.
    """

    source = Path(path).resolve()
    if not source.is_file():
        raise IdentityAnnotationError(f"gaze CSV does not exist: {source}")
    raw = pd.read_csv(source)
    missing = sorted(_REQUIRED_GAZE_COLUMNS.difference(raw.columns))
    if missing:
        raise IdentityAnnotationError(f"gaze CSV is missing required columns: {missing}")
    if raw.empty:
        raise IdentityAnnotationError("gaze CSV is empty")

    numeric_columns = ["frame_index", "timestamp_sec"]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    if not np.isfinite(raw[numeric_columns].to_numpy(dtype=float)).all():
        raise IdentityAnnotationError("frame and timestamp values must be finite")
    if not np.equal(raw["frame_index"], np.floor(raw["frame_index"])).all():
        raise IdentityAnnotationError("frame_index must contain integers")
    raw["frame_index"] = raw["frame_index"].astype(int)
    if (raw["frame_index"] < 0).any() or (raw["timestamp_sec"] < 0).any():
        raise IdentityAnnotationError("frame_index and timestamp_sec must be non-negative")
    frame_times = raw.groupby("frame_index", sort=False)["timestamp_sec"].agg(["min", "max"])
    if ((frame_times["max"] - frame_times["min"]).abs() > 1e-6).any():
        raise IdentityAnnotationError("rows for one frame disagree on timestamp_sec")
    if (start_frame_inclusive is None) != (end_frame_exclusive is None):
        raise IdentityAnnotationError("both gaze frame-window boundaries are required")
    if start_frame_inclusive is not None and end_frame_exclusive is not None:
        if (
            isinstance(start_frame_inclusive, bool)
            or isinstance(end_frame_exclusive, bool)
            or int(start_frame_inclusive) != start_frame_inclusive
            or int(end_frame_exclusive) != end_frame_exclusive
            or start_frame_inclusive < 0
            or end_frame_exclusive <= start_frame_inclusive
        ):
            raise IdentityAnnotationError("gaze frame window must be a non-empty integer range")
        raw = raw.loc[
            (raw["frame_index"] >= int(start_frame_inclusive))
            & (raw["frame_index"] < int(end_frame_exclusive))
        ].copy()
        if raw.empty:
            raise IdentityAnnotationError("gaze CSV has no rows in the HPE core window")

    participant_present = raw["person_idx"].notna()
    participant_bbox_present = raw.loc[:, list(_BBOX_COLUMNS)].notna()
    incomplete_participant = participant_present & ~participant_bbox_present.all(axis=1)
    stray_participant_bbox = ~participant_present & participant_bbox_present.any(axis=1)
    if incomplete_participant.any() or stray_participant_bbox.any():
        raise IdentityAnnotationError(
            "person_idx and participant bbox must be either jointly present or jointly missing"
        )
    participant_raw = raw.loc[participant_present].copy()
    for column in _BBOX_COLUMNS:
        participant_raw[column] = pd.to_numeric(participant_raw[column], errors="raise")
    _validate_bbox_values(participant_raw, _BBOX_COLUMNS)

    participant_raw["source_person_idx"] = participant_raw["person_idx"].map(
        _normalise_person_idx
    )
    duplicate_keys = participant_raw.duplicated(
        ["frame_index", "source_person_idx"], keep=False
    )
    if duplicate_keys.any():
        examples = participant_raw.loc[
            duplicate_keys, ["frame_index", "source_person_idx"]
        ].head(5)
        raise IdentityAnnotationError(
            "duplicate (frame_index, person_idx) observations: "
            f"{examples.to_dict(orient='records')}"
        )

    optional_columns = [
        column
        for column in ("gaze_target", "gaze_peak_x", "gaze_peak_y", "inout_score")
        if column in raw.columns
    ]
    participant = participant_raw.loc[
        :,
        ["frame_index", "timestamp_sec", "source_person_idx", *_BBOX_COLUMNS, *optional_columns],
    ].copy()
    participant.insert(2, "source_kind", "participant_detection")

    parts = [participant] if not participant.empty else []
    if include_robot:
        robot_rows: list[dict[str, Any]] = []
        for frame_index, group in raw.groupby("frame_index", sort=True):
            detected_values = pd.to_numeric(group["robot_detected"], errors="raise").to_numpy(
                dtype=float
            )
            if not np.isfinite(detected_values).all() or len(np.unique(detected_values)) != 1:
                raise IdentityAnnotationError(
                    f"frame {frame_index} has inconsistent robot_detected values"
                )
            detected = float(detected_values[0])
            if detected not in (0.0, 1.0):
                raise IdentityAnnotationError("robot_detected must be binary")
            robot_values = group.loc[:, list(_ROBOT_BBOX_COLUMNS)]
            if detected == 0.0:
                if robot_values.notna().any(axis=None):
                    raise IdentityAnnotationError(
                        f"frame {frame_index} has a robot bbox while robot_detected is 0"
                    )
                continue
            if robot_values.isna().any(axis=None):
                raise IdentityAnnotationError(
                    f"frame {frame_index} is missing a detected robot bbox"
                )
            robot_values = robot_values.apply(pd.to_numeric, errors="raise")
            if robot_values.nunique(dropna=False).max() != 1:
                raise IdentityAnnotationError(
                    f"frame {frame_index} has inconsistent repeated robot bbox values"
                )
            bbox = robot_values.iloc[0].to_numpy(dtype=float)
            if not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise IdentityAnnotationError(f"frame {frame_index} has an invalid robot bbox")
            robot_rows.append(
                {
                    "frame_index": int(frame_index),
                    "timestamp_sec": float(group["timestamp_sec"].iloc[0]),
                    "source_kind": "robot_candidate",
                    "source_person_idx": "__robot__",
                    "bbox_x0": bbox[0],
                    "bbox_y0": bbox[1],
                    "bbox_x1": bbox[2],
                    "bbox_y1": bbox[3],
                    "gaze_target": "",
                    "gaze_peak_x": np.nan,
                    "gaze_peak_y": np.nan,
                    "inout_score": np.nan,
                }
            )
        if robot_rows:
            parts.append(pd.DataFrame(robot_rows))

    if not parts:
        raise IdentityAnnotationError("gaze CSV contains no participant or robot observations")
    observations = pd.concat(parts, ignore_index=True, sort=False)
    observations = observations.sort_values(
        ["frame_index", "source_kind", "source_person_idx"], kind="stable"
    ).reset_index(drop=True)
    observations.insert(0, "observation_id", np.arange(len(observations), dtype=int))
    return observations


def _scope_frame_window(scope: dict[str, Any]) -> tuple[int, int] | None:
    kind = scope.get("kind")
    if kind == "full_gaze_export":
        return None
    if kind != "hpe_confirmed_core_window":
        raise IdentityAnnotationError("unsupported identity annotation scope")
    start = scope.get("gaze_local_start_frame_inclusive")
    end = scope.get("gaze_local_end_frame_exclusive")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise IdentityAnnotationError("annotation scope has an invalid local frame window")
    return start, end


def _filter_tracked_observations_to_scope(
    observations: pd.DataFrame, scope: dict[str, Any]
) -> pd.DataFrame:
    window = _scope_frame_window(scope)
    if window is None:
        output = observations.copy()
    else:
        start, end = window
        output = observations.loc[
            (observations["frame_index"] >= start)
            & (observations["frame_index"] < end)
        ].copy()
    if output.empty:
        raise IdentityAnnotationError("annotation scope contains no gaze observations")
    output = output.reset_index(drop=True)
    output["observation_id"] = np.arange(len(output), dtype=int)
    return output


def _bbox_array(row: pd.Series) -> np.ndarray:
    return row.loc[list(_BBOX_COLUMNS)].to_numpy(dtype=float)


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def _frame_has_crowding(
    output: pd.DataFrame, indices: list[int], config: TrackerConfig
) -> bool:
    for first_position, first_index in enumerate(indices):
        first_bbox = _bbox_array(output.loc[first_index])
        for second_index in indices[first_position + 1 :]:
            second_bbox = _bbox_array(output.loc[second_index])
            first_diagonal = math.hypot(
                first_bbox[2] - first_bbox[0], first_bbox[3] - first_bbox[1]
            )
            second_diagonal = math.hypot(
                second_bbox[2] - second_bbox[0], second_bbox[3] - second_bbox[1]
            )
            centre_ratio = float(
                np.linalg.norm(_bbox_center(first_bbox) - _bbox_center(second_bbox))
            ) / max((first_diagonal + second_diagonal) / 2, 1e-6)
            if (
                _bbox_iou(first_bbox, second_bbox) >= config.crowding_iou_threshold
                or centre_ratio <= config.crowding_center_distance_ratio
            ):
                return True
    return False


def _assignments_invert_spatial_order(
    assignments: list[tuple[int, int, float]],
    active_ids: list[str],
    active: dict[str, tuple[float, np.ndarray]],
    indices: list[int],
    output: pd.DataFrame,
) -> bool:
    for first_position, (first_row, first_column, _) in enumerate(assignments):
        first_previous = active[active_ids[first_row]][1]
        first_current = _bbox_array(output.loc[indices[first_column]])
        for second_row, second_column, _ in assignments[first_position + 1 :]:
            second_previous = active[active_ids[second_row]][1]
            second_current = _bbox_array(output.loc[indices[second_column]])
            previous_centres = (_bbox_center(first_previous), _bbox_center(second_previous))
            current_centres = (_bbox_center(first_current), _bbox_center(second_current))
            for axis in (0, 1):
                previous_difference = previous_centres[0][axis] - previous_centres[1][axis]
                current_difference = current_centres[0][axis] - current_centres[1][axis]
                previous_sizes = (
                    first_previous[axis + 2] - first_previous[axis],
                    second_previous[axis + 2] - second_previous[axis],
                )
                current_sizes = (
                    first_current[axis + 2] - first_current[axis],
                    second_current[axis + 2] - second_current[axis],
                )
                separation_gate = 0.15 * float(
                    np.mean([*previous_sizes, *current_sizes])
                )
                if (
                    abs(previous_difference) > separation_gate
                    and abs(current_difference) > separation_gate
                    and previous_difference * current_difference < 0
                ):
                    return True
    return False


def _link_cost(first: np.ndarray, second: np.ndarray, config: TrackerConfig) -> float:
    first_center = np.array([(first[0] + first[2]) / 2, (first[1] + first[3]) / 2])
    second_center = np.array([(second[0] + second[2]) / 2, (second[1] + second[3]) / 2])
    first_diag = math.hypot(first[2] - first[0], first[3] - first[1])
    second_diag = math.hypot(second[2] - second[0], second[3] - second[1])
    centre_ratio = float(np.linalg.norm(first_center - second_center)) / max(
        (first_diag + second_diag) / 2, 1e-6
    )
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    area_ratio = max(first_area, second_area) / max(min(first_area, second_area), 1e-6)
    iou = _bbox_iou(first, second)
    if centre_ratio > config.max_center_distance_ratio:
        return math.inf
    if area_ratio > config.max_area_ratio:
        return math.inf
    if iou < config.min_iou and centre_ratio > config.strong_center_distance_ratio:
        return math.inf
    area_penalty = min(abs(math.log(area_ratio)), 1.0)
    return 0.60 * (1.0 - iou) + 0.30 * centre_ratio + 0.10 * area_penalty


def _is_ambiguous(costs: np.ndarray, row: int, column: int, margin: float) -> bool:
    if margin <= 0:
        return False
    selected = costs[row, column]
    row_values = np.sort(costs[row, np.isfinite(costs[row])])
    column_values = np.sort(costs[np.isfinite(costs[:, column]), column])
    row_ambiguous = len(row_values) > 1 and row_values[1] - selected < margin
    column_ambiguous = len(column_values) > 1 and column_values[1] - selected < margin
    return bool(row_ambiguous or column_ambiguous)


def build_tracklets(
    observations: pd.DataFrame, config: TrackerConfig | None = None
) -> pd.DataFrame:
    """Link bbox observations without consulting the unstable person index."""

    tracker_config = config or TrackerConfig()
    tracker_config.validate()
    required = {"observation_id", "frame_index", "timestamp_sec", "source_kind", *_BBOX_COLUMNS}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise IdentityAnnotationError(f"observations are missing required columns: {missing}")

    output = observations.copy()
    output["tracklet_id"] = ""
    output["link_cost"] = np.nan
    output["link_confidence"] = 0.0
    output["link_flags"] = ""
    counters = {"participant_detection": 0, "robot_candidate": 0}
    prefixes = {"participant_detection": "p", "robot_candidate": "r"}

    for source_kind, kind_frame in output.groupby("source_kind", sort=False):
        if source_kind not in prefixes:
            raise IdentityAnnotationError(f"unsupported source_kind: {source_kind!r}")
        active: dict[str, tuple[float, np.ndarray]] = {}
        for _, frame in kind_frame.groupby("frame_index", sort=True):
            indices = frame.index.to_list()
            frame_time = float(frame["timestamp_sec"].iloc[0])
            active = {
                tracklet_id: state
                for tracklet_id, state in active.items()
                if 0 < frame_time - state[0] <= tracker_config.max_gap_s + 1e-9
            }
            active_ids = sorted(active)
            costs = np.full((len(active_ids), len(indices)), np.inf, dtype=float)
            for row_index, tracklet_id in enumerate(active_ids):
                _, previous_bbox = active[tracklet_id]
                for column_index, observation_index in enumerate(indices):
                    costs[row_index, column_index] = _link_cost(
                        previous_bbox, _bbox_array(output.loc[observation_index]), tracker_config
                    )

            assignments: list[tuple[int, int, float]] = []
            if costs.size and np.isfinite(costs).any():
                safe_costs = np.where(np.isfinite(costs), costs, 1e6)
                row_indices, column_indices = linear_sum_assignment(safe_costs)
                for row_index, column_index in zip(row_indices, column_indices, strict=True):
                    cost = costs[row_index, column_index]
                    if not math.isfinite(cost) or cost > tracker_config.max_link_cost:
                        continue
                    assignments.append((row_index, column_index, float(cost)))

            guard_flag: str | None = None
            if source_kind == "participant_detection" and _frame_has_crowding(
                output, indices, tracker_config
            ):
                guard_flag = "crowding_guard"
            elif any(
                _is_ambiguous(costs, row, column, tracker_config.ambiguity_margin)
                for row, column, _ in assignments
            ):
                guard_flag = "ambiguity_guard"
            elif any(cost > tracker_config.guard_link_cost for _, _, cost in assignments):
                guard_flag = "high_cost_guard"
            elif source_kind == "participant_detection" and _assignments_invert_spatial_order(
                assignments, active_ids, active, indices, output
            ):
                guard_flag = "order_inversion_guard"

            if guard_flag is not None:
                active.clear()
                for observation_index in indices:
                    counters[source_kind] += 1
                    tracklet_id = f"{prefixes[source_kind]}{counters[source_kind]:06d}"
                    output.at[observation_index, "tracklet_id"] = tracklet_id
                    output.at[observation_index, "link_confidence"] = 0.0
                    output.at[observation_index, "link_flags"] = guard_flag
                continue

            matched_observations: set[int] = set()
            matched_tracks: set[str] = set()
            for row_index, column_index, cost in assignments:
                observation_index = indices[column_index]
                tracklet_id = active_ids[row_index]
                output.at[observation_index, "tracklet_id"] = tracklet_id
                output.at[observation_index, "link_cost"] = cost
                output.at[observation_index, "link_confidence"] = max(0.0, 1.0 - cost)
                output.at[observation_index, "link_flags"] = "linked"
                active[tracklet_id] = (
                    frame_time,
                    _bbox_array(output.loc[observation_index]),
                )
                matched_observations.add(observation_index)
                matched_tracks.add(tracklet_id)

            for observation_index in indices:
                if observation_index in matched_observations:
                    continue
                counters[source_kind] += 1
                tracklet_id = f"{prefixes[source_kind]}{counters[source_kind]:06d}"
                output.at[observation_index, "tracklet_id"] = tracklet_id
                output.at[observation_index, "link_confidence"] = 1.0
                output.at[observation_index, "link_flags"] = "new_tracklet"
                active[tracklet_id] = (
                    frame_time,
                    _bbox_array(output.loc[observation_index]),
                )
            for tracklet_id in matched_tracks:
                if tracklet_id not in active:
                    raise AssertionError("matched track unexpectedly absent")

    if (output["tracklet_id"] == "").any():
        raise AssertionError("every observation must receive a tracklet")
    return output


def summarise_tracklets(
    observations: pd.DataFrame,
    *,
    short_tracklet_max_duration_s: float = 0.0,
) -> list[dict[str, Any]]:
    short_duration_s = _validate_short_tracklet_max_duration_s(
        short_tracklet_max_duration_s
    )
    summaries: list[dict[str, Any]] = []
    for tracklet_id, frame in observations.groupby("tracklet_id", sort=True):
        ordered = frame.sort_values(["frame_index", "observation_id"], kind="stable")
        observation_frames = ordered["frame_index"].astype(int).tolist()
        required_review_frames = {
            observation_frames[0],
            observation_frames[len(observation_frames) // 2],
            observation_frames[-1],
        }
        guarded = ordered[~ordered["link_flags"].isin({"linked", "new_tracklet"})]
        required_review_frames.update(guarded["frame_index"].astype(int).tolist())
        last_periodic_time = float(ordered["timestamp_sec"].iloc[0])
        for _, observation in ordered.iloc[1:-1].iterrows():
            observation_time = float(observation["timestamp_sec"])
            if observation_time - last_periodic_time >= REVIEW_ANCHOR_INTERVAL_S:
                required_review_frames.add(int(observation["frame_index"]))
                last_periodic_time = observation_time
        timestamp_values = ordered["timestamp_sec"].to_numpy(dtype=float)
        if len(timestamp_values) > 1:
            gaps = np.diff(timestamp_values)
            positive_gaps = gaps[gaps > 0]
            if len(positive_gaps):
                gap_threshold = max(0.075, 2.5 * float(np.median(positive_gaps)))
                for gap_index in np.flatnonzero(gaps > gap_threshold):
                    required_review_frames.add(int(observation_frames[gap_index]))
                    required_review_frames.add(int(observation_frames[gap_index + 1]))
        source_kind = str(frame["source_kind"].iloc[0])
        tracklet_duration_s = float(timestamp_values.max() - timestamp_values.min())
        crowding_quarantine = bool(ordered["link_flags"].eq("crowding_guard").all())
        singleton_quarantine = bool(
            len(ordered) == 1 and source_kind == "participant_detection"
        )
        short_fragment_quarantine = bool(
            source_kind == "participant_detection"
            and not singleton_quarantine
            and tracklet_duration_s
            <= short_duration_s + _SHORT_TRACKLET_BOUNDARY_TOLERANCE_S
        )
        auto_quarantined = (
            crowding_quarantine or singleton_quarantine or short_fragment_quarantine
        )
        quarantine_reason = (
            "crowding_guard"
            if crowding_quarantine
            else "singleton_fragment"
            if singleton_quarantine
            else "short_fragment"
            if short_fragment_quarantine
            else None
        )
        summaries.append(
            {
                "tracklet_id": str(tracklet_id),
                "source_kind": source_kind,
                "start_frame": int(frame["frame_index"].min()),
                "end_frame": int(frame["frame_index"].max()),
                "start_time_s": float(frame["timestamp_sec"].min()),
                "end_time_s": float(frame["timestamp_sec"].max()),
                "observation_count": int(len(frame)),
                "required_review_frames": sorted(required_review_frames),
                "reviewed_anchor_frames": [],
                "label": None,
                "status": "uncertain" if auto_quarantined else "unreviewed",
                "note": (
                    f"automatic {quarantine_reason} quarantine; identity intentionally missing"
                    if quarantine_reason is not None
                    else ""
                ),
                "auto_quarantined": auto_quarantined,
                "auto_quarantine_reason": quarantine_reason,
                "updated_at": None,
            }
        )
    return summaries


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    """Tolerate short-lived Windows locks without weakening atomic replacement."""

    delay_s = _ATOMIC_REPLACE_INITIAL_DELAY_S
    for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt + 1 == _ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(delay_s)
            delay_s = min(delay_s * 2, _ATOMIC_REPLACE_MAX_DELAY_S)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".json.tmp", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    # Keep the complete temporary JSON if every retry fails.  It is recovery
    # evidence; normal resume remains bound to the official destination only.
    _replace_with_retry(temporary, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", suffix=".csv.tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_tracklet_observations(path: str | Path) -> pd.DataFrame:
    observations_path = Path(path)
    if not observations_path.is_file():
        raise IdentityAnnotationError(
            f"tracklet observations do not exist: {observations_path}"
        )
    observations = pd.read_csv(
        observations_path,
        dtype={"source_person_idx": str, "tracklet_id": str},
    )
    required = {
        "observation_id",
        "frame_index",
        "timestamp_sec",
        "source_kind",
        "source_person_idx",
        "tracklet_id",
        "link_cost",
        "link_confidence",
        "link_flags",
        *_BBOX_COLUMNS,
    }
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise IdentityAnnotationError(
            f"tracklet observations are missing required columns: {missing}"
        )
    if observations.empty:
        raise IdentityAnnotationError("tracklet observations may not be empty")
    for column in ("observation_id", "frame_index", "timestamp_sec", *_BBOX_COLUMNS):
        observations[column] = pd.to_numeric(observations[column], errors="raise")
    numeric = observations.loc[
        :, ["observation_id", "frame_index", "timestamp_sec", *_BBOX_COLUMNS]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise IdentityAnnotationError("tracklet observation keys and boxes must be finite")
    if not np.equal(observations["observation_id"], np.floor(observations["observation_id"])).all():
        raise IdentityAnnotationError("observation_id must contain integers")
    if not np.equal(observations["frame_index"], np.floor(observations["frame_index"])).all():
        raise IdentityAnnotationError("frame_index must contain integers")
    observations["observation_id"] = observations["observation_id"].astype(int)
    observations["frame_index"] = observations["frame_index"].astype(int)
    expected_ids = np.arange(len(observations), dtype=int)
    if not np.array_equal(observations["observation_id"].to_numpy(), expected_ids):
        raise IdentityAnnotationError("observation_id must be the ordered 0..N-1 sequence")
    if (observations["frame_index"] < 0).any() or (observations["timestamp_sec"] < 0).any():
        raise IdentityAnnotationError("observation frame and timestamp must be non-negative")
    _validate_bbox_values(observations, _BBOX_COLUMNS)
    if not observations["source_kind"].isin(
        {"participant_detection", "robot_candidate"}
    ).all():
        raise IdentityAnnotationError("tracklet observations contain an invalid source_kind")
    if observations["tracklet_id"].isna().any() or (
        observations["tracklet_id"].str.strip() == ""
    ).any():
        raise IdentityAnnotationError("tracklet_id may not be empty")
    if observations.groupby("tracklet_id")["source_kind"].nunique().max() != 1:
        raise IdentityAnnotationError("one tracklet may not mix source kinds")
    return observations


def validate_annotation_scope(manifest: dict[str, Any]) -> dict[str, Any]:
    scope = manifest.get("annotation_scope")
    if not isinstance(scope, dict):
        raise IdentityAnnotationError("manifest annotation_scope must be a JSON object")
    window = _scope_frame_window(scope)
    if window is None:
        if scope != {"kind": "full_gaze_export"}:
            raise IdentityAnnotationError("full-export annotation scope has extra fields")
        return scope
    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("declared_video"), dict):
        raise IdentityAnnotationError("manifest lacks declared video for HPE scope replay")
    profile_path = scope.get("profile_path")
    run_metadata_path = scope.get("run_metadata_path")
    if not isinstance(profile_path, str) or not isinstance(run_metadata_path, str):
        raise IdentityAnnotationError("HPE annotation scope lacks provenance paths")
    authority = scope.get("authority_binding")
    if authority is not None:
        required_authority = {
            "kind",
            "mapping_path",
            "mapping_sha256",
            "validation_path",
            "validation_sha256",
            "camera",
            "expected_run_metadata_sha256",
        }
        if (
            not isinstance(authority, dict)
            or set(authority) != required_authority
            or authority.get("kind")
            not in {
                "hpe_confirmed_delivery_validation_v1",
                "hpe_confirmed_scene_excluded_camera_axis_v1",
            }
            or authority.get("camera")
            != source["declared_video"].get("camera_dir")
            or authority.get("expected_run_metadata_sha256")
            != scope.get("run_metadata_sha256")
        ):
            raise IdentityAnnotationError("HPE annotation authority binding is malformed")
        for path_key, hash_key, label in (
            ("mapping_path", "mapping_sha256", "delivery mapping"),
            ("validation_path", "validation_sha256", "validation marker"),
        ):
            value = authority.get(path_key)
            digest = authority.get(hash_key)
            if (
                not isinstance(value, str)
                or not isinstance(digest, str)
                or not Path(value).is_file()
                or sha256_file(Path(value)) != digest
            ):
                raise IdentityAnnotationError(
                    f"HPE {label} changed since annotation preparation"
                )
    current = load_hpe_annotation_scope(
        profile_path,
        run_metadata_path,
        dataset=str(manifest.get("dataset")),
        session=str(manifest.get("session")),
        declared_video=source["declared_video"],
    )
    recorded_base = dict(scope)
    recorded_base.pop("authority_binding", None)
    if current != recorded_base:
        raise IdentityAnnotationError(
            "HPE annotation scope changed since preparation; start a new revision"
        )
    return scope


def annotation_frame_window(manifest: dict[str, Any]) -> tuple[int, int] | None:
    """Return the validated local half-open frame window for the artifact."""

    return _scope_frame_window(validate_annotation_scope(manifest))


def _validate_observations_against_source(
    observations: pd.DataFrame, manifest: dict[str, Any]
) -> None:
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise IdentityAnnotationError("manifest source binding must be a JSON object")
    gaze_csv_value = source.get("gaze_csv")
    if not isinstance(gaze_csv_value, str):
        raise IdentityAnnotationError("manifest source is missing gaze_csv")
    gaze_csv = Path(gaze_csv_value)
    if not gaze_csv.is_file():
        raise IdentityAnnotationError(f"bound gaze CSV does not exist: {gaze_csv}")
    if source.get("gaze_csv_sha256") != sha256_file(gaze_csv):
        raise IdentityAnnotationError("bound gaze CSV SHA-256 differs from the manifest")
    metadata_path, metadata_hash, declared_video = load_export_metadata(gaze_csv)
    if (
        source.get("session_metadata") != str(metadata_path)
        or source.get("session_metadata_sha256") != metadata_hash
        or source.get("declared_video") != declared_video
    ):
        raise IdentityAnnotationError(
            "bound session metadata changed since annotation preparation"
        )
    scope = validate_annotation_scope(manifest)
    full_expected = load_gaze_observations(gaze_csv)
    expected = _filter_tracked_observations_to_scope(full_expected, scope)
    counts = manifest.get("scope_counts")
    if (
        not isinstance(counts, dict)
        or counts.get("full_observations") != len(full_expected)
        or counts.get("scoped_observations") != len(expected)
        or counts.get("excluded_observations") != len(full_expected) - len(expected)
    ):
        raise IdentityAnnotationError("manifest scope_counts cannot be replayed from source")
    if len(expected) != len(observations):
        raise IdentityAnnotationError(
            "tracklet observations row count cannot be replayed from the gaze CSV"
        )
    missing = [column for column in expected.columns if column not in observations.columns]
    if missing:
        raise IdentityAnnotationError(
            f"tracklet observations omit source-derived columns: {missing}"
        )
    for column in expected.columns:
        expected_values = expected[column]
        actual_values = observations[column]
        if pd.api.types.is_numeric_dtype(expected_values.dtype):
            try:
                expected_numeric = pd.to_numeric(expected_values, errors="raise").to_numpy(
                    dtype=float
                )
                actual_numeric = pd.to_numeric(actual_values, errors="raise").to_numpy(
                    dtype=float
                )
            except (TypeError, ValueError) as exc:
                raise IdentityAnnotationError(
                    f"source-derived numeric column changed: {column}"
                ) from exc
            equal = np.isclose(
                expected_numeric,
                actual_numeric,
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )
            if not bool(equal.all()):
                raise IdentityAnnotationError(
                    f"tracklet observations differ from gaze source column: {column}"
                )
        else:
            expected_text = expected_values.astype("string").fillna("")
            actual_text = actual_values.astype("string").fillna("")
            if not expected_text.equals(actual_text):
                raise IdentityAnnotationError(
                    f"tracklet observations differ from gaze source column: {column}"
                )


def _validate_tracklet_summaries(
    observations: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    automatic_missing_policy: dict[str, Any] | None = None,
) -> None:
    policy = automatic_missing_policy or _manifest_automatic_missing_policy(manifest)
    legacy_no_quarantine = (
        policy.get("kind") == _EARLY_V2_NO_AUTOMATIC_QUARANTINE_KIND
    )
    short_duration_s = (
        0.0
        if legacy_no_quarantine
        else float(policy["short_tracklet_max_duration_s"])
    )
    expected = {
        item["tracklet_id"]: item
        for item in summarise_tracklets(
            observations,
            short_tracklet_max_duration_s=short_duration_s,
        )
    }
    actual = annotation_lookup(manifest)
    if set(expected) != set(actual):
        raise IdentityAnnotationError("tracklet observations and annotations disagree")
    exact_fields = (
        "source_kind",
        "start_frame",
        "end_frame",
        "observation_count",
        "required_review_frames",
    )
    for tracklet_id, expected_item in expected.items():
        actual_item = actual[tracklet_id]
        for field in exact_fields:
            if actual_item.get(field) != expected_item[field]:
                raise IdentityAnnotationError(
                    f"tracklet summary {field} differs for {tracklet_id}"
                )
        if not legacy_no_quarantine:
            for field in ("auto_quarantined", "auto_quarantine_reason"):
                if actual_item.get(field) != expected_item[field]:
                    raise IdentityAnnotationError(
                        f"tracklet summary {field} differs for {tracklet_id}"
                    )
        for field in ("start_time_s", "end_time_s"):
            try:
                matches = math.isclose(
                    float(actual_item.get(field)),
                    float(expected_item[field]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise IdentityAnnotationError(
                    f"tracklet summary {field} differs for {tracklet_id}"
                )


def _tracklet_observations_binding(path: Path) -> dict[str, Any]:
    observations = read_tracklet_observations(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "columns": list(observations.columns),
        "parsed_dtypes": [str(observations[column].dtype) for column in observations.columns],
        "n_rows": int(len(observations)),
    }


def validate_tracklet_observations(
    path: str | Path, manifest: dict[str, Any]
) -> pd.DataFrame:
    observations_path = Path(path)
    if not observations_path.is_file():
        raise IdentityAnnotationError(
            f"tracklet observations do not exist: {observations_path}"
        )
    binding = manifest.get("tracklet_observations")
    if not isinstance(binding, dict):
        raise IdentityAnnotationError("manifest is missing the tracklet-observations binding")
    if binding.get("path") != str(observations_path.resolve()):
        raise IdentityAnnotationError("tracklet observations path differs from the manifest")
    if binding.get("sha256") != sha256_file(observations_path):
        raise IdentityAnnotationError("tracklet observations SHA-256 differs from the manifest")
    observations = read_tracklet_observations(observations_path)
    actual_columns = list(observations.columns)
    actual_dtypes = [str(observations[column].dtype) for column in observations.columns]
    if binding.get("columns") != actual_columns:
        raise IdentityAnnotationError("tracklet observations columns differ from the manifest")
    if binding.get("parsed_dtypes") != actual_dtypes:
        raise IdentityAnnotationError("tracklet observation dtypes differ from the manifest")
    if binding.get("n_rows") != len(observations):
        raise IdentityAnnotationError("tracklet observation row count differs from the manifest")
    counts = manifest.get("scope_counts")
    if not isinstance(counts, dict):
        raise IdentityAnnotationError("manifest is missing scope_counts")
    expected_scoped = int(len(observations))
    expected_tracklets = int(observations["tracklet_id"].nunique())
    if (
        counts.get("scoped_observations") != expected_scoped
        or counts.get("scoped_tracklets") != expected_tracklets
        or not isinstance(counts.get("full_observations"), int)
        or isinstance(counts.get("full_observations"), bool)
        or not isinstance(counts.get("excluded_observations"), int)
        or isinstance(counts.get("excluded_observations"), bool)
        or counts["full_observations"]
        != expected_scoped + counts["excluded_observations"]
        or counts["excluded_observations"] < 0
    ):
        raise IdentityAnnotationError("manifest scope_counts differ from observations")
    _validate_observations_against_source(observations, manifest)
    _validate_tracklet_summaries(observations, manifest)
    return observations


def validate_tracklet_observations_portable(
    path: str | Path, manifest: dict[str, Any]
) -> pd.DataFrame:
    """Replay a sealed artifact without trusting pandas dtype display names.

    Pandas 2 reports parsed text columns as ``object`` while pandas 3 reports
    the same immutable CSV bytes as ``str``.  A sealed downstream consumer
    therefore keeps every security- and semantics-relevant check (canonical
    path, exact SHA-256, ordered columns, row counts, source replay, scope
    counts and tracklet summaries) while treating recorded dtype spellings as
    environment metadata.  The interactive review/finalisation path continues
    to use :func:`validate_tracklet_observations` and remains exact-version
    strict.
    """

    observations_path = Path(path)
    if not observations_path.is_file():
        raise IdentityAnnotationError(
            f"tracklet observations do not exist: {observations_path}"
        )
    binding = manifest.get("tracklet_observations")
    if not isinstance(binding, dict):
        raise IdentityAnnotationError("manifest is missing the tracklet-observations binding")
    if binding.get("path") != str(observations_path.resolve()):
        raise IdentityAnnotationError("tracklet observations path differs from the manifest")
    if binding.get("sha256") != sha256_file(observations_path):
        raise IdentityAnnotationError("tracklet observations SHA-256 differs from the manifest")
    observations = read_tracklet_observations(observations_path)
    columns = list(observations.columns)
    if binding.get("columns") != columns:
        raise IdentityAnnotationError("tracklet observations columns differ from the manifest")
    if binding.get("n_rows") != len(observations):
        raise IdentityAnnotationError("tracklet observation row count differs from the manifest")
    recorded_dtypes = binding.get("parsed_dtypes")
    if (
        not isinstance(recorded_dtypes, list)
        or len(recorded_dtypes) != len(columns)
        or any(not isinstance(value, str) or not value for value in recorded_dtypes)
    ):
        raise IdentityAnnotationError("tracklet observation dtype record is malformed")
    counts = manifest.get("scope_counts")
    expected_scoped = int(len(observations))
    expected_tracklets = int(observations["tracklet_id"].nunique())
    if (
        not isinstance(counts, dict)
        or counts.get("scoped_observations") != expected_scoped
        or counts.get("scoped_tracklets") != expected_tracklets
        or not isinstance(counts.get("full_observations"), int)
        or isinstance(counts.get("full_observations"), bool)
        or not isinstance(counts.get("excluded_observations"), int)
        or isinstance(counts.get("excluded_observations"), bool)
        or counts["full_observations"]
        != expected_scoped + counts["excluded_observations"]
        or counts["excluded_observations"] < 0
    ):
        raise IdentityAnnotationError("manifest scope_counts differ from observations")
    _validate_observations_against_source(observations, manifest)
    _validate_tracklet_summaries(observations, manifest)
    return observations


def _validate_legacy_observations_for_migration(
    path: Path,
    manifest: dict[str, Any],
    *,
    automatic_missing_policy: dict[str, Any],
) -> pd.DataFrame:
    """Validate a v1 artifact without depending on pandas dtype display names.

    Pandas 2 and 3 can parse the same immutable CSV bytes as ``object`` versus
    ``str`` and Windows builds may choose ``int32`` versus ``int64``.  A legacy
    migration therefore keeps the security-relevant checks (path, SHA-256,
    ordered columns, row count, full source replay and track summaries) while
    treating the recorded dtype spellings as environment metadata.  The new v2
    artifact is rebound to the current canonical parser after migration.
    """

    binding = manifest.get("tracklet_observations")
    if not isinstance(binding, dict):
        raise IdentityAnnotationError("legacy manifest lacks observations binding")
    if binding.get("path") != str(path.resolve()):
        raise IdentityAnnotationError("tracklet observations path differs from the manifest")
    if binding.get("sha256") != sha256_file(path):
        raise IdentityAnnotationError("tracklet observations SHA-256 differs from the manifest")
    observations = read_tracklet_observations(path)
    columns = list(observations.columns)
    if binding.get("columns") != columns:
        raise IdentityAnnotationError("tracklet observations columns differ from the manifest")
    if binding.get("n_rows") != len(observations):
        raise IdentityAnnotationError("tracklet observation row count differs from the manifest")
    recorded_dtypes = binding.get("parsed_dtypes")
    if not isinstance(recorded_dtypes, list) or len(recorded_dtypes) != len(columns) or not all(
        isinstance(value, str) and value for value in recorded_dtypes
    ):
        raise IdentityAnnotationError("legacy observation dtype record is malformed")
    _validate_observations_against_source(observations, manifest)
    _validate_tracklet_summaries(
        observations,
        manifest,
        automatic_missing_policy=automatic_missing_policy,
    )
    return observations


def save_annotation_state(
    observations_path: str | Path,
    manifest_path: str | Path,
    observations: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    _validate_observations_against_source(observations, manifest)
    _validate_tracklet_summaries(observations, manifest)
    counts = manifest.get("scope_counts")
    if not isinstance(counts, dict):
        raise IdentityAnnotationError("manifest is missing scope_counts")
    counts["scoped_observations"] = int(len(observations))
    counts["scoped_tracklets"] = int(observations["tracklet_id"].nunique())
    observations_destination = Path(observations_path)
    _atomic_write_csv(observations_destination, observations)
    manifest["tracklet_observations"] = _tracklet_observations_binding(
        observations_destination
    )
    save_annotation_manifest(manifest_path, manifest)


def prepare_annotation_artifact(
    gaze_csv: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    session: str,
    config: TrackerConfig | None = None,
    annotation_scope: dict[str, Any] | None = None,
    short_tracklet_max_duration_s: float = 0.0,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Create or validate resumable tracklet and annotation files."""

    source = Path(gaze_csv).resolve()
    if not source.is_file():
        raise IdentityAnnotationError(f"gaze CSV does not exist: {source}")
    destination = Path(output_dir).resolve()
    observations_path = destination / "tracklet_observations.csv"
    manifest_path = destination / "identity_annotations.json"
    source_hash = sha256_file(source)
    metadata_path, metadata_hash, declared_video = load_export_metadata(source)
    if str(dataset) != declared_video["dataset"] or str(session) != declared_video["session"]:
        raise IdentityAnnotationError(
            "dataset/session arguments disagree with the source session metadata"
        )
    tracker_config = config or TrackerConfig()
    tracker_config.validate()
    expected_automatic_missing_policy = _automatic_missing_policy(
        short_tracklet_max_duration_s
    )
    requested_scope = annotation_scope or {"kind": "full_gaze_export"}
    _scope_frame_window(requested_scope)
    if requested_scope.get("kind") == "hpe_confirmed_core_window":
        scope_probe = {
            "dataset": str(dataset),
            "session": str(session),
            "source": {"declared_video": declared_video},
            "annotation_scope": requested_scope,
        }
        validate_annotation_scope(scope_probe)
    elif requested_scope != {"kind": "full_gaze_export"}:
        raise IdentityAnnotationError("invalid full-export annotation scope")
    if observations_path.exists() or manifest_path.exists():
        if overwrite:
            if not observations_path.is_file() or not manifest_path.is_file():
                raise IdentityAnnotationError("partial annotation artifact cannot be overwritten")
        else:
            if not observations_path.is_file() or not manifest_path.is_file():
                raise IdentityAnnotationError("annotation artifact is incomplete")
            manifest = load_annotation_manifest(manifest_path)
            if manifest.get("dataset") != str(dataset) or manifest.get("session") != str(session):
                raise IdentityAnnotationError(
                    "annotation dataset/session does not match the requested export"
                )
            recorded_source = manifest.get("source", {}).get("gaze_csv")
            if not isinstance(recorded_source, str) or Path(recorded_source).resolve() != source:
                raise IdentityAnnotationError(
                    "annotation artifact is bound to a different gaze CSV path"
                )
            recorded_hash = manifest.get("source", {}).get("gaze_csv_sha256")
            if recorded_hash != source_hash:
                raise IdentityAnnotationError(
                    "gaze CSV changed since annotation preparation; start a new revision"
                )
            recorded_metadata_hash = manifest.get("source", {}).get(
                "session_metadata_sha256"
            )
            if recorded_metadata_hash != metadata_hash:
                raise IdentityAnnotationError(
                    "session metadata changed since annotation preparation; start a new revision"
                )
            if manifest.get("tracker_config") != asdict(tracker_config):
                raise IdentityAnnotationError(
                    "tracker settings differ from this annotation revision; use a new revision"
                )
            if manifest.get("annotation_scope") != requested_scope:
                raise IdentityAnnotationError(
                    "annotation scope differs from this revision; migrate or use a new revision"
                )
            if manifest.get("review_policy") != {
                "periodic_anchor_interval_s": REVIEW_ANCHOR_INTERVAL_S,
                "gap_anchor_rule": "both sides when gap > max(0.075 s, 2.5x median)",
            }:
                raise IdentityAnnotationError(
                    "review-anchor policy differs from this annotation revision"
                )
            recorded_automatic_missing_policy = _manifest_automatic_missing_policy(
                manifest
            )
            early_v2_without_quarantine = (
                recorded_automatic_missing_policy.get("kind")
                == _EARLY_V2_NO_AUTOMATIC_QUARANTINE_KIND
            )
            if (
                early_v2_without_quarantine
                and float(short_tracklet_max_duration_s) != 0.0
            ) or (
                not early_v2_without_quarantine
                and recorded_automatic_missing_policy
                != expected_automatic_missing_policy
            ):
                raise IdentityAnnotationError(
                    "automatic missing policy differs from this annotation revision; "
                    "use a new revision"
                )
            validate_tracklet_observations(observations_path, manifest)
            return observations_path, manifest_path

    full_source_observations = load_gaze_observations(source)
    scoped_source_observations = _filter_tracked_observations_to_scope(
        full_source_observations, requested_scope
    )
    observations = build_tracklets(scoped_source_observations, tracker_config)
    if int(observations["frame_index"].max()) >= declared_video["frame_count"]:
        raise IdentityAnnotationError("gaze observations exceed the metadata video frame count")
    expected_times = observations["frame_index"].to_numpy(dtype=float) / declared_video["fps"]
    source_times = observations["timestamp_sec"].to_numpy(dtype=float)
    if np.max(np.abs(source_times - expected_times), initial=0.0) > max(
        1e-6, 0.01 / declared_video["fps"]
    ):
        raise IdentityAnnotationError(
            "gaze timestamps disagree with session metadata FPS/frame indices"
        )
    boxes = observations.loc[:, list(_BBOX_COLUMNS)].to_numpy(dtype=float)
    if (
        np.any(boxes[:, 0] < 0)
        or np.any(boxes[:, 1] < 0)
        or np.any(boxes[:, 2] > declared_video["width"])
        or np.any(boxes[:, 3] > declared_video["height"])
    ):
        raise IdentityAnnotationError("gaze bounding boxes exceed metadata video dimensions")
    manifest = {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "dataset": str(dataset),
        "session": str(session),
        "source": {
            "gaze_csv": str(source),
            "gaze_csv_sha256": source_hash,
            "gaze_csv_size": source.stat().st_size,
            "gaze_csv_mtime_ns": source.stat().st_mtime_ns,
            "session_metadata": str(metadata_path),
            "session_metadata_sha256": metadata_hash,
            "declared_video": declared_video,
        },
        "tracker_config": asdict(tracker_config),
        "tracker_algorithm": TRACKER_ALGORITHM,
        "automatic_missing_policy": expected_automatic_missing_policy,
        "annotation_scope": requested_scope,
        "scope_counts": {
            "full_observations": int(len(full_source_observations)),
            "scoped_observations": int(len(observations)),
            "excluded_observations": int(
                len(full_source_observations) - len(observations)
            ),
            "scoped_tracklets": int(observations["tracklet_id"].nunique()),
        },
        "review_policy": {
            "periodic_anchor_interval_s": REVIEW_ANCHOR_INTERVAL_S,
            "gap_anchor_rule": "both sides when gap > max(0.075 s, 2.5x median)",
        },
        "label_to_slot": LABEL_TO_SLOT,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "annotations": summarise_tracklets(
            observations,
            short_tracklet_max_duration_s=short_tracklet_max_duration_s,
        ),
    }
    _atomic_write_csv(observations_path, observations)
    manifest["tracklet_observations"] = _tracklet_observations_binding(observations_path)
    _atomic_write_json(manifest_path, manifest)
    return observations_path, manifest_path


def _load_annotation_manifest_versions(
    path: str | Path, *, allowed_versions: set[int]
) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise IdentityAnnotationError("unexpected identity annotation artifact_kind")
    if manifest.get("format_version") not in allowed_versions:
        if manifest.get("format_version") == LEGACY_FORMAT_VERSION:
            raise IdentityAnnotationError(
                "legacy full-video identity artifact requires explicit HPE-core migration"
            )
        raise IdentityAnnotationError("unsupported identity annotation format_version")
    annotations = manifest.get("annotations")
    if not isinstance(annotations, list):
        raise IdentityAnnotationError("identity annotations must be a list")
    identifiers = [item.get("tracklet_id") for item in annotations if isinstance(item, dict)]
    if len(identifiers) != len(annotations) or len(set(identifiers)) != len(identifiers):
        raise IdentityAnnotationError("identity annotations have invalid/duplicate tracklet IDs")
    _manifest_automatic_missing_policy(manifest)
    for item in annotations:
        tracklet_id = item.get("tracklet_id")
        source_kind = item.get("source_kind")
        status = item.get("status")
        label = item.get("label")
        if not isinstance(tracklet_id, str) or not tracklet_id:
            raise IdentityAnnotationError("tracklet_id must be a non-empty string")
        if source_kind not in {"participant_detection", "robot_candidate"}:
            raise IdentityAnnotationError(f"invalid source_kind for {tracklet_id}")
        if status not in {"confirmed", "uncertain", "ignored", "unreviewed"}:
            raise IdentityAnnotationError(f"invalid annotation status for {tracklet_id}")
        if label is not None and label not in PARTICIPANT_LABELS:
            raise IdentityAnnotationError(f"invalid participant label for {tracklet_id}")
        if (status == "confirmed") != (label is not None):
            raise IdentityAnnotationError(
                f"only confirmed annotations may carry a participant label: {tracklet_id}"
            )
        auto_quarantined = item.get("auto_quarantined", False)
        quarantine_reason = item.get("auto_quarantine_reason")
        if not isinstance(auto_quarantined, bool):
            raise IdentityAnnotationError(
                f"auto_quarantined must be boolean for {tracklet_id}"
            )
        if auto_quarantined and (status != "uncertain" or label is not None):
            raise IdentityAnnotationError(
                f"auto-quarantined tracklet must remain uncertain: {tracklet_id}"
            )
        if auto_quarantined != (
            quarantine_reason
            in {"crowding_guard", "singleton_fragment", "short_fragment"}
        ):
            raise IdentityAnnotationError(
                f"invalid auto-quarantine reason for {tracklet_id}"
            )
        required_frames = item.get("required_review_frames")
        reviewed_frames = item.get("reviewed_anchor_frames")
        if (
            not isinstance(required_frames, list)
            or not required_frames
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in required_frames
            )
            or required_frames != sorted(set(required_frames))
        ):
            raise IdentityAnnotationError(f"invalid required review anchors for {tracklet_id}")
        if (
            not isinstance(reviewed_frames, list)
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in reviewed_frames
            )
            or reviewed_frames != sorted(set(reviewed_frames))
            or not set(reviewed_frames).issubset(required_frames)
        ):
            raise IdentityAnnotationError(f"invalid reviewed anchors for {tracklet_id}")
        if status == "confirmed" and not set(required_frames).issubset(reviewed_frames):
            raise IdentityAnnotationError(
                f"confirmed annotation has unreviewed anchors: {tracklet_id}"
            )
    if manifest.get("format_version") == FORMAT_VERSION:
        scope = manifest.get("annotation_scope")
        if not isinstance(scope, dict):
            raise IdentityAnnotationError("v2 identity manifest requires annotation_scope")
        _scope_frame_window(scope)
    return manifest


def load_annotation_manifest(path: str | Path) -> dict[str, Any]:
    return _load_annotation_manifest_versions(path, allowed_versions={FORMAT_VERSION})


def migrate_legacy_annotation_to_hpe_core(
    output_dir: str | Path,
    annotation_scope: dict[str, Any],
    *,
    expected_tracker_config: TrackerConfig | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Explicitly migrate a v1 full-video artifact into a scoped v2 revision.

    The original directory is copied to a timestamped sibling first.  Tracks
    wholly inside the formal window retain their human review state.  Any track
    clipped by a boundary is reset to unreviewed because its required anchors
    change; its previous decision remains in the migration record and backup.
    """

    if annotation_scope.get("kind") != "hpe_confirmed_core_window":
        raise IdentityAnnotationError("legacy migration requires an HPE core scope")
    destination = Path(output_dir).resolve()
    observations_path = destination / "tracklet_observations.csv"
    manifest_path = destination / "identity_annotations.json"
    if not observations_path.is_file() or not manifest_path.is_file():
        raise IdentityAnnotationError("legacy annotation artifact is incomplete")
    legacy = _load_annotation_manifest_versions(
        manifest_path, allowed_versions={LEGACY_FORMAT_VERSION}
    )
    legacy_automatic_missing_policy = _manifest_automatic_missing_policy(legacy)
    legacy_probe = dict(legacy)
    legacy_probe["annotation_scope"] = {"kind": "full_gaze_export"}
    legacy_probe["scope_counts"] = {
        "full_observations": legacy["tracklet_observations"]["n_rows"],
        "scoped_observations": legacy["tracklet_observations"]["n_rows"],
        "excluded_observations": 0,
        "scoped_tracklets": len(legacy["annotations"]),
    }
    observations = _validate_legacy_observations_for_migration(
        observations_path,
        legacy_probe,
        automatic_missing_policy=legacy_automatic_missing_policy,
    )
    scoped_probe = dict(legacy)
    scoped_probe["annotation_scope"] = annotation_scope
    validate_annotation_scope(scoped_probe)

    tracker_config_raw = legacy.get("tracker_config")
    if not isinstance(tracker_config_raw, dict):
        raise IdentityAnnotationError("legacy artifact lacks tracker_config")
    try:
        tracker_config = TrackerConfig(**tracker_config_raw)
    except TypeError as exc:
        raise IdentityAnnotationError("legacy tracker_config is unsupported") from exc
    tracker_config.validate()
    if expected_tracker_config is not None:
        expected_tracker_config.validate()
        if asdict(tracker_config) != asdict(expected_tracker_config):
            raise IdentityAnnotationError(
                "requested tracker settings differ from the legacy artifact"
            )
    source_path = Path(str(legacy["source"]["gaze_csv"]))
    metadata_path, metadata_hash, declared_video = load_export_metadata(source_path)
    source_binding = legacy.get("source")
    if not isinstance(source_binding, dict):
        raise IdentityAnnotationError("legacy artifact lacks source binding")
    if (
        source_binding.get("session_metadata") != str(metadata_path)
        or source_binding.get("session_metadata_sha256") != metadata_hash
        or source_binding.get("declared_video") != declared_video
    ):
        raise IdentityAnnotationError(
            "legacy session metadata changed since annotation preparation"
        )
    full_source = load_gaze_observations(source_path)
    scoped_source = _filter_tracked_observations_to_scope(
        full_source, annotation_scope
    )
    scoped = build_tracklets(scoped_source, tracker_config)
    old_lookup = annotation_lookup(legacy)
    new_summaries = summarise_tracklets(scoped)
    if (
        legacy_automatic_missing_policy.get("kind")
        != _EARLY_V2_NO_AUTOMATIC_QUARANTINE_KIND
    ):
        raise IdentityAnnotationError("legacy v1 migration policy is unsupported")
    for summary in new_summaries:
        summary.update(
            {
                "label": None,
                "status": "unreviewed",
                "note": "",
                "auto_quarantined": False,
                "auto_quarantine_reason": None,
            }
        )
    exact_fields = (
        "source_kind",
        "start_frame",
        "end_frame",
        "observation_count",
        "required_review_frames",
    )
    key_columns = ("frame_index", "source_kind", "source_person_idx")

    def row_key(row: pd.Series) -> tuple[int, str, str]:
        return (int(row["frame_index"]), str(row["source_kind"]), str(row["source_person_idx"]))

    old_key_to_tracklet: dict[tuple[int, str, str], str] = {}
    old_tracklet_keys: dict[str, frozenset[tuple[int, str, str]]] = {}
    for tracklet_id, group in observations.groupby("tracklet_id", sort=False):
        keys = frozenset(row_key(row) for _, row in group.loc[:, list(key_columns)].iterrows())
        old_tracklet_keys[str(tracklet_id)] = keys
        for key in keys:
            if key in old_key_to_tracklet:
                raise IdentityAnnotationError("legacy observations have duplicate source keys")
            old_key_to_tracklet[key] = str(tracklet_id)

    carried: list[dict[str, Any]] = []
    reset: list[dict[str, Any]] = []
    for summary in new_summaries:
        new_tracklet_id = str(summary["tracklet_id"])
        group = scoped.loc[scoped["tracklet_id"].astype(str) == new_tracklet_id]
        keys = frozenset(row_key(row) for _, row in group.loc[:, list(key_columns)].iterrows())
        candidates = sorted(
            {
                old_key_to_tracklet[key]
                for key in keys
                if key in old_key_to_tracklet
            }
        )
        old_tracklet_id = candidates[0] if len(candidates) == 1 else None
        old = old_lookup.get(old_tracklet_id) if old_tracklet_id is not None else None
        exact_membership = (
            old_tracklet_id is not None
            and old_tracklet_keys[old_tracklet_id] == keys
        )
        unchanged = bool(
            exact_membership
            and old is not None
            and all(old.get(field) == summary[field] for field in exact_fields)
        )
        if unchanged and old is not None and old_tracklet_id is not None:
            summary.update(
                {
                    "label": old.get("label"),
                    "status": old.get("status", "unreviewed"),
                    "note": old.get("note", ""),
                    "reviewed_anchor_frames": list(
                        old.get("reviewed_anchor_frames", [])
                    ),
                    "updated_at": old.get("updated_at", _utc_now()),
                }
            )
            carried.append(
                {
                    "old_tracklet_id": old_tracklet_id,
                    "new_tracklet_id": new_tracklet_id,
                    "status": old.get("status", "unreviewed"),
                    "label": old.get("label"),
                }
            )
        else:
            reset.append(
                {
                    "new_tracklet_id": new_tracklet_id,
                    "candidate_old_tracklet_ids": candidates,
                    "previous_annotations": [
                        {
                            "old_tracklet_id": candidate,
                            "status": old_lookup[candidate].get(
                                "status", "unreviewed"
                            ),
                            "label": old_lookup[candidate].get("label"),
                            "note": old_lookup[candidate].get("note", ""),
                            "reviewed_anchor_frames": list(
                                old_lookup[candidate].get(
                                    "reviewed_anchor_frames", []
                                )
                            ),
                        }
                        for candidate in candidates
                    ],
                    "reason": "membership_or_summary_changed_at_scope_boundary",
                }
            )
    scoped_keys = {
        row_key(row) for _, row in scoped.loc[:, list(key_columns)].iterrows()
    }
    dropped = [
        {
            "tracklet_id": tracklet_id,
            "previous_label": item.get("label"),
            "previous_status": item.get("status", "unreviewed"),
        }
        for tracklet_id, item in old_lookup.items()
        if old_tracklet_keys[tracklet_id].isdisjoint(scoped_keys)
    ]

    migration_summary: dict[str, Any] = {
        "backup_path": None,
        "carried_tracklets": len(carried),
        "reset_tracklets": len(reset),
        "dropped_tracklets": len(dropped),
        "scoped_observations": len(scoped),
        "scoped_tracklets": len(new_summaries),
    }
    if dry_run:
        return migration_summary

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = destination.parent / f"{destination.name}_full_v1_backup_{timestamp}"
    if backup.exists():
        raise IdentityAnnotationError(f"migration backup already exists: {backup}")
    legacy_hashes = {
        path.name: sha256_file(path)
        for path in destination.iterdir()
        if path.is_file()
    }
    shutil.copytree(destination, backup)
    for name, digest in legacy_hashes.items():
        copied = backup / name
        if not copied.is_file() or sha256_file(copied) != digest:
            raise IdentityAnnotationError(f"migration backup verification failed: {name}")

    migrated = dict(legacy)
    migrated["format_version"] = FORMAT_VERSION
    migrated["tracker_algorithm"] = TRACKER_ALGORITHM
    migrated["automatic_missing_policy"] = (
        _early_v2_no_automatic_quarantine_policy()
    )
    migrated["annotation_scope"] = annotation_scope
    migrated["scope_counts"] = {
        "full_observations": int(len(observations)),
        "scoped_observations": int(len(scoped)),
        "excluded_observations": int(len(observations) - len(scoped)),
        "full_tracklets": int(len(old_lookup)),
        "scoped_tracklets": int(len(new_summaries)),
    }
    migrated["annotations"] = new_summaries
    previous_structural_events = migrated.pop("structural_events", [])
    migrated["migration"] = {
        "timestamp": _utc_now(),
        "action": "legacy_full_video_to_hpe_core",
        "from_format_version": LEGACY_FORMAT_VERSION,
        "backup_path": str(backup),
        "legacy_artifact_sha256": legacy_hashes,
        "carried_tracklets": carried,
        "reset_tracklets": reset,
        "dropped_tracklets": dropped,
        "legacy_structural_events": previous_structural_events,
    }
    _atomic_write_csv(observations_path, scoped)
    migrated["tracklet_observations"] = _tracklet_observations_binding(
        observations_path
    )
    save_annotation_manifest(manifest_path, migrated)
    # Never leave a full-session derived bridge beside the new scoped manifest.
    # The verified backup retains the old files; the current directory receives
    # freshly derived scoped outputs immediately.
    for derived_name in ("identity_bridge.csv", "identity_conflicts.csv"):
        (destination / derived_name).unlink(missing_ok=True)
    export_identity_bridge(observations_path, manifest_path, destination)
    audit_path = destination / "annotation_audit.jsonl"
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": _utc_now(),
                    "action": "legacy_full_video_to_hpe_core",
                    "backup_path": str(backup),
                    "carried_tracklet_count": len(carried),
                    "reset_tracklet_count": len(reset),
                    "dropped_tracklet_count": len(dropped),
                },
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    migration_summary["backup_path"] = backup
    return migration_summary


def save_annotation_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _atomic_write_json(Path(path), manifest)


def set_tracklet_annotation(
    manifest: dict[str, Any],
    tracklet_id: str,
    *,
    label: str | None,
    status: str,
    note: str = "",
) -> dict[str, Any]:
    if label is not None and label not in PARTICIPANT_LABELS:
        raise IdentityAnnotationError(f"unsupported participant label: {label!r}")
    if status not in {"confirmed", "uncertain", "ignored", "unreviewed"}:
        raise IdentityAnnotationError(f"unsupported annotation status: {status!r}")
    if status == "confirmed" and label is None:
        raise IdentityAnnotationError("confirmed annotations require a participant label")
    if status != "confirmed" and label is not None:
        raise IdentityAnnotationError("only confirmed annotations may carry a participant label")
    for item in manifest["annotations"]:
        if item["tracklet_id"] == tracklet_id:
            if item.get("auto_quarantined", False) and (
                status != "uncertain" or label is not None
            ):
                raise IdentityAnnotationError(
                    "auto-quarantined tracklets must remain missing"
                )
            if status == "confirmed":
                missing_anchors = sorted(
                    set(item["required_review_frames"])
                    - set(item["reviewed_anchor_frames"])
                )
                if missing_anchors:
                    raise IdentityAnnotationError(
                        f"review required anchor frames before confirming: {missing_anchors}"
                    )
            previous = dict(item)
            item.update(
                {
                    "label": label,
                    "status": status,
                    "note": note,
                    "updated_at": _utc_now(),
                }
            )
            return previous
    raise IdentityAnnotationError(f"unknown tracklet_id: {tracklet_id}")


def mark_review_anchor(
    manifest: dict[str, Any], tracklet_id: str, frame_index: int
) -> tuple[list[int], list[int]]:
    for item in manifest["annotations"]:
        if item["tracklet_id"] != tracklet_id:
            continue
        required = list(item["required_review_frames"])
        if int(frame_index) not in required:
            raise IdentityAnnotationError(
                f"frame {frame_index} is not a required review anchor for {tracklet_id}"
            )
        reviewed = sorted({*item["reviewed_anchor_frames"], int(frame_index)})
        item["reviewed_anchor_frames"] = reviewed
        item["updated_at"] = _utc_now()
        remaining = sorted(set(required) - set(reviewed))
        return reviewed, remaining
    raise IdentityAnnotationError(f"unknown tracklet_id: {tracklet_id}")


def append_audit_event(
    path: str | Path,
    *,
    tracklet_id: str,
    previous: dict[str, Any],
    current: dict[str, Any],
    action: str,
) -> None:
    event = {
        "timestamp": _utc_now(),
        "action": action,
        "tracklet_id": tracklet_id,
        "previous": {
            "label": previous.get("label"),
            "status": previous.get("status"),
            "note": previous.get("note", ""),
        },
        "current": {
            "label": current.get("label"),
            "status": current.get("status"),
            "note": current.get("note", ""),
        },
    }
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def append_split_audit_event(
    path: str | Path,
    *,
    source_tracklet_id: str,
    new_tracklet_id: str,
    split_frame: int,
) -> None:
    event = {
        "timestamp": _utc_now(),
        "action": "split_tracklet",
        "source_tracklet_id": source_tracklet_id,
        "new_tracklet_id": new_tracklet_id,
        "split_frame": int(split_frame),
    }
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def append_anchor_audit_event(
    path: str | Path,
    *,
    tracklet_id: str,
    frame_index: int,
    remaining_frames: list[int],
) -> None:
    event = {
        "timestamp": _utc_now(),
        "action": "review_anchor",
        "tracklet_id": tracklet_id,
        "frame_index": int(frame_index),
        "remaining_frames": list(remaining_frames),
    }
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def annotation_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["tracklet_id"]): item for item in manifest["annotations"]}


def build_identity_bridge(
    observations: pd.DataFrame, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a per-observation identity bridge and all same-frame conflicts."""

    lookup = annotation_lookup(manifest)
    bridge = observations.copy()
    bridge["participant"] = bridge["tracklet_id"].map(
        lambda value: lookup[str(value)].get("label")
    )
    bridge["slot"] = bridge["participant"].map(LABEL_TO_SLOT)
    bridge["annotation_status"] = bridge["tracklet_id"].map(
        lambda value: lookup[str(value)].get("status")
    )
    confirmed = bridge[
        (bridge["annotation_status"] == "confirmed") & bridge["participant"].notna()
    ]
    conflict_mask = confirmed.duplicated(["frame_index", "participant"], keep=False)
    conflicts = confirmed.loc[conflict_mask].sort_values(
        ["frame_index", "participant", "tracklet_id"], kind="stable"
    )
    return bridge, conflicts


def export_identity_bridge(
    observations_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    manifest = load_annotation_manifest(manifest_path)
    observations = validate_tracklet_observations(observations_path, manifest)
    bridge, conflicts = build_identity_bridge(observations, manifest)
    destination = Path(output_dir)
    bridge_path = destination / "identity_bridge.csv"
    conflicts_path = destination / "identity_conflicts.csv"
    _atomic_write_csv(bridge_path, bridge)
    _atomic_write_csv(conflicts_path, conflicts)
    return bridge_path, conflicts_path


def split_tracklet(
    observations: pd.DataFrame,
    manifest: dict[str, Any],
    tracklet_id: str,
    split_frame: int,
) -> tuple[pd.DataFrame, str]:
    mask = observations["tracklet_id"].astype(str) == tracklet_id
    selected = observations.loc[mask]
    if selected.empty:
        raise IdentityAnnotationError(f"unknown tracklet_id: {tracklet_id}")
    old_items = [
        item for item in manifest["annotations"] if item["tracklet_id"] == tracklet_id
    ]
    if len(old_items) != 1:
        raise IdentityAnnotationError("manifest does not contain exactly one source tracklet")
    if old_items[0].get("auto_quarantined", False):
        raise IdentityAnnotationError(
            "auto-quarantined tracklets cannot be manually split"
        )
    automatic_missing_policy = _manifest_automatic_missing_policy(manifest)
    legacy_no_quarantine = (
        automatic_missing_policy.get("kind")
        == _EARLY_V2_NO_AUTOMATIC_QUARANTINE_KIND
    )
    short_duration_s = (
        0.0
        if legacy_no_quarantine
        else float(automatic_missing_policy["short_tracklet_max_duration_s"])
    )
    before = selected["frame_index"] < split_frame
    after = selected["frame_index"] >= split_frame
    if not before.any() or not after.any():
        raise IdentityAnnotationError("split frame must leave observations on both sides")
    suffix = 1
    existing = set(observations["tracklet_id"].astype(str))
    while True:
        new_id = f"{tracklet_id}_s{suffix}"
        if new_id not in existing:
            break
        suffix += 1
    output = observations.copy()
    split_mask = mask & (output["frame_index"] >= split_frame)
    output.loc[split_mask, "tracklet_id"] = new_id
    first_index = output.loc[split_mask].sort_values("frame_index", kind="stable").index[0]
    output.at[first_index, "link_cost"] = np.nan
    output.at[first_index, "link_confidence"] = 1.0
    output.at[first_index, "link_flags"] = "manual_split"

    old_summary = summarise_tracklets(
        output[output["tracklet_id"] == tracklet_id],
        short_tracklet_max_duration_s=short_duration_s,
    )[0]
    new_summary = summarise_tracklets(
        output[output["tracklet_id"] == new_id],
        short_tracklet_max_duration_s=short_duration_s,
    )[0]
    if legacy_no_quarantine:
        for summary in (old_summary, new_summary):
            summary.update(
                {
                    "status": "unreviewed",
                    "note": "",
                    "auto_quarantined": False,
                    "auto_quarantine_reason": None,
                }
            )
    manifest["annotations"] = [
        old_summary if item["tracklet_id"] == tracklet_id else item
        for item in manifest["annotations"]
    ]
    manifest["annotations"].append(new_summary)
    manifest["annotations"].sort(key=lambda item: item["tracklet_id"])
    manifest.setdefault("structural_events", []).append(
        {
            "timestamp": _utc_now(),
            "action": "split_tracklet",
            "source_tracklet_id": tracklet_id,
            "new_tracklet_id": new_id,
            "split_frame": int(split_frame),
        }
    )
    return output, new_id
