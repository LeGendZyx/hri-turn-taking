"""Human-reviewed HPE-person to Transcript-identity crosswalks.

The HPE pipeline already supplies stable, reviewed ``person`` identifiers.  This
module deliberately does not repeat tracking and never treats ``world_person``
or a camera-local ``person_id`` as the crosswalk key.  It samples visual evidence
for each final HPE ``person`` and records independent A/B/C/R decisions.

All accepted scene/camera/window choices come from the live identity ledger.
The upstream HPE participant map is retained only as a diagnostic proposal; it
is never copied into a human decision.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hri_turn_taking.data.identity_annotation import (
    LABEL_TO_SLOT,
    PARTICIPANT_LABELS,
    IdentityAnnotationError,
    _atomic_write_csv,
    _atomic_write_json,
    annotation_frame_window,
    build_identity_bridge,
    load_annotation_manifest,
    sha256_file,
    validate_tracklet_observations,
    validate_tracklet_observations_portable,
)

ARTIFACT_KIND = "vap_hpe_transcript_identity_crosswalk"
FORMAT_VERSION = 1
SAMPLE_FORMAT_VERSION = 1
SEALED_INVENTORY_KIND = "vap_hpe_transcript_crosswalk_inventory"
SEALED_INVENTORY_VERSION = 1
SEALED_DOWNSTREAM_KIND = "vap_hpe_transcript_identity_downstream_contract"
SEALED_DOWNSTREAM_VERSION = 1
DEFAULT_LEDGER_PATH = Path("docs/IDENTITY_DATA_STATUS.md")
DEFAULT_SEALED_INVENTORY_PATH = Path(
    "configs/hpe_transcript_crosswalk_inventory.json"
)
DEFAULT_SEALED_LEDGER_PATH = Path(
    "configs/hpe_transcript_crosswalk_ledger_snapshot.md"
)

MANIFEST_NAME = "hpe_transcript_crosswalk.json"
SAMPLES_NAME = "review_samples.csv"
FINAL_MAPPING_NAME = "hpe_transcript_crosswalk.csv"
CONFLICTS_NAME = "crosswalk_conflicts.csv"
AUDIT_NAME = "crosswalk_audit.jsonl"
SEALED_ARTIFACT_FILES = frozenset(
    {
        MANIFEST_NAME,
        SAMPLES_NAME,
        FINAL_MAPPING_NAME,
        CONFLICTS_NAME,
        AUDIT_NAME,
    }
)

_CANONICAL_LABELS = frozenset(PARTICIPANT_LABELS)
_SPARSE_VIDEO_CHUNK_SIZE = 1024 * 1024
_SPARSE_VIDEO_ALGORITHM = "sha256(size + first/middle/last 1MiB with offsets)"
_CLOUD_RECALL_FLAGS = (
    getattr(stat, "FILE_ATTRIBUTE_OFFLINE", 0x1000)
    | getattr(stat, "FILE_ATTRIBUTE_RECALL_ON_OPEN", 0x40000)
    | getattr(stat, "FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS", 0x400000)
)
_HPE_BBOX_COLUMNS = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
_GAZE_BBOX_COLUMNS = ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")
_SAMPLE_COLUMNS = (
    "sample_id",
    "hpe_person",
    "phase",
    "logical_frame",
    "gaze_local_frame",
    "hpe_camera",
    "gaze_camera",
    "hpe_bbox_json",
    "all_hpe_boxes_json",
    "gaze_confirmed_boxes_json",
    "hpe_image_source_json",
    "gaze_image_source_json",
)


class HPETranscriptCrosswalkError(ValueError):
    """Raised when crosswalk evidence or state is unsafe or inconsistent."""


@dataclass(frozen=True)
class AcceptedIdentityWindow:
    """One accepted scene row parsed from the live identity ledger."""

    scene: str
    camera: str
    local_start_inclusive: int
    local_end_exclusive: int

    @property
    def local_window(self) -> tuple[int, int]:
        return (self.local_start_inclusive, self.local_end_exclusive)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_markdown_cell(value: str) -> str:
    return value.strip().strip("`").strip()


def parse_accepted_identity_ledger(
    path: str | Path = DEFAULT_LEDGER_PATH,
) -> dict[str, AcceptedIdentityWindow]:
    """Parse only the accepted-window table from the live identity ledger.

    This intentionally does not infer a scene from filenames or from historical
    examples elsewhere in the repository.
    """

    ledger_path = Path(path).resolve()
    if not ledger_path.is_file():
        raise HPETranscriptCrosswalkError(
            f"identity-status ledger does not exist: {ledger_path}"
        )
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    try:
        section_start = next(
            index
            for index, line in enumerate(lines)
            if line.strip() == "## Accepted identity windows"
        )
    except StopIteration as exc:
        raise HPETranscriptCrosswalkError(
            "identity-status ledger lacks the Accepted identity windows section"
        ) from exc

    header_index: int | None = None
    header: list[str] = []
    for index in range(section_start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("## "):
            break
        if stripped.startswith("|") and "Scene" in stripped and "Camera" in stripped:
            header_index = index
            header = [cell.strip() for cell in stripped.strip("|").split("|")]
            break
    if header_index is None:
        raise HPETranscriptCrosswalkError(
            "accepted identity section lacks its canonical Markdown table"
        )
    required_headers = {
        "Scene",
        "Camera",
        "Accepted local frame window",
        "Identity status",
    }
    if not required_headers.issubset(header):
        raise HPETranscriptCrosswalkError(
            "accepted identity table is missing required columns"
        )
    indices = {name: header.index(name) for name in required_headers}

    accepted: dict[str, AcceptedIdentityWindow] = {}
    for line in lines[header_index + 2 :]:
        stripped = line.strip()
        if stripped.startswith("##"):
            break
        if not stripped.startswith("|"):
            if accepted and stripped:
                break
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != len(header):
            raise HPETranscriptCrosswalkError(
                "accepted identity table contains a malformed row"
            )
        status = _normalise_markdown_cell(cells[indices["Identity status"]])
        if not status.lower().startswith("accepted"):
            continue
        scene = _normalise_markdown_cell(cells[indices["Scene"]])
        camera = _normalise_markdown_cell(cells[indices["Camera"]])
        window = _normalise_markdown_cell(
            cells[indices["Accepted local frame window"]]
        )
        if not scene or "__" not in scene:
            raise HPETranscriptCrosswalkError(
                f"accepted identity row has an invalid scene: {scene!r}"
            )
        if not camera.startswith("Cam_") or not camera[4:].isdigit():
            raise HPETranscriptCrosswalkError(
                f"accepted identity row has an invalid camera: {camera!r}"
            )
        if not (window.startswith("[") and window.endswith(")") and "," in window):
            raise HPETranscriptCrosswalkError(
                f"accepted identity row has an invalid half-open window: {window!r}"
            )
        start_text, end_text = window[1:-1].split(",", maxsplit=1)
        try:
            start = int(start_text.strip())
            end = int(end_text.strip())
        except ValueError as exc:
            raise HPETranscriptCrosswalkError(
                f"accepted identity row has a non-integer window: {window!r}"
            ) from exc
        if start < 0 or end <= start:
            raise HPETranscriptCrosswalkError(
                f"accepted identity row has an invalid window: {window!r}"
            )
        if scene in accepted:
            raise HPETranscriptCrosswalkError(
                f"accepted identity table contains duplicate scene {scene!r}"
            )
        accepted[scene] = AcceptedIdentityWindow(scene, camera, start, end)
    if not accepted:
        raise HPETranscriptCrosswalkError(
            "identity-status ledger contains no accepted identity windows"
        )
    return accepted


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise HPETranscriptCrosswalkError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HPETranscriptCrosswalkError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise HPETranscriptCrosswalkError(f"{label} must contain a JSON object")
    return value


def _binding(path: Path, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }
    if frame is not None:
        result.update(
            {
                "columns": list(frame.columns),
                "parsed_dtypes": [str(dtype) for dtype in frame.dtypes],
                "n_rows": int(len(frame)),
            }
        )
    return result


def _resolve_inside(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise HPETranscriptCrosswalkError(f"{label} path is missing")
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise HPETranscriptCrosswalkError(f"{label} escapes the HPE repository") from exc
    return resolved


def _coerce_integer_series(
    values: pd.Series, *, label: str, allow_zero: bool
) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise HPETranscriptCrosswalkError(f"{label} must be numeric integers") from exc
    array = numeric.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise HPETranscriptCrosswalkError(f"{label} must be finite integers")
    integers = numeric.astype("int64")
    minimum = 0 if allow_zero else 1
    if (integers < minimum).any():
        raise HPETranscriptCrosswalkError(
            f"{label} values must be >= {minimum}"
        )
    return integers


def _normalise_participant_map(value: Any, label: str) -> dict[int, str]:
    if not isinstance(value, dict):
        raise HPETranscriptCrosswalkError(f"{label} must be a JSON object")
    output: dict[int, str] = {}
    for raw_person, raw_participant in value.items():
        if isinstance(raw_person, bool):
            raise HPETranscriptCrosswalkError(f"{label} has an invalid person key")
        try:
            person = int(raw_person)
        except (TypeError, ValueError) as exc:
            raise HPETranscriptCrosswalkError(
                f"{label} has a non-integer person key"
            ) from exc
        if str(person) != str(raw_person).strip() or person < 1:
            raise HPETranscriptCrosswalkError(f"{label} has an invalid person key")
        if raw_participant not in _CANONICAL_LABELS:
            raise HPETranscriptCrosswalkError(
                f"{label} has an invalid participant label"
            )
        if person in output:
            raise HPETranscriptCrosswalkError(f"{label} repeats person {person}")
        output[person] = str(raw_participant)
    if not output:
        raise HPETranscriptCrosswalkError(f"{label} may not be empty")
    return output


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise HPETranscriptCrosswalkError(f"{label} does not exist: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise HPETranscriptCrosswalkError(f"could not read {label}: {path}") from exc


def _validate_participant_map_csv(
    path: Path, proposal: dict[int, str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _load_csv(path, "HPE participant_map.csv")
    if list(frame.columns) != ["person", "participant"]:
        raise HPETranscriptCrosswalkError(
            "HPE participant_map.csv must have exactly person,participant columns"
        )
    if frame.empty or frame["person"].duplicated().any():
        raise HPETranscriptCrosswalkError(
            "HPE participant_map.csv must contain unique person rows"
        )
    people = _coerce_integer_series(
        frame["person"], label="participant_map.csv person", allow_zero=False
    )
    csv_map = dict(zip(people.tolist(), frame["participant"].astype(str), strict=True))
    if csv_map != proposal:
        raise HPETranscriptCrosswalkError(
            "HPE mapping.json participant_map differs from participant_map.csv"
        )
    return frame, _binding(path, frame)


def _validate_hpe_authority(
    *,
    scene: str,
    accepted: AcceptedIdentityWindow,
    delivery_dir: Path,
    hpe_repo_root: Path,
) -> dict[str, Any]:
    delivery = delivery_dir.resolve()
    root = hpe_repo_root.resolve()
    try:
        delivery.relative_to(root)
    except ValueError as exc:
        raise HPETranscriptCrosswalkError(
            "HPE delivery must reside inside the declared HPE repository"
        ) from exc
    canonical_delivery = (root / "outputs" / "delivery" / scene).resolve()
    if delivery != canonical_delivery:
        raise HPETranscriptCrosswalkError(
            "HPE delivery is not the canonical outputs/delivery scene directory"
        )
    if delivery.name != scene:
        raise HPETranscriptCrosswalkError(
            "HPE delivery directory name differs from the accepted scene"
        )
    mapping_path = delivery / "mapping.json"
    mapping = _load_json_object(mapping_path, "HPE mapping.json")
    group, scenario = scene.rsplit("__", maxsplit=1)
    if (
        mapping.get("delivery") != scene
        or str(mapping.get("group")) != group
        or str(mapping.get("scenario")) != scenario
        or mapping.get("review_status") != "confirmed"
        or mapping.get("mapping_mode") not in {None, "seat", "person_exact"}
    ):
        raise HPETranscriptCrosswalkError(
            "HPE mapping is not confirmed authority for this scene"
        )
    proposal = _normalise_participant_map(
        mapping.get("participant_map"), "HPE mapping participant_map"
    )
    _, participant_map_binding = _validate_participant_map_csv(
        delivery / "participant_map.csv", proposal
    )
    sources = mapping.get("sources")
    if not isinstance(sources, dict):
        raise HPETranscriptCrosswalkError("HPE mapping sources must be a JSON object")
    identity_dir = _resolve_inside(
        root, sources.get("identity_dir"), "HPE identity directory"
    )
    validation_dir = _resolve_inside(
        root, sources.get("validation_dir"), "HPE validation directory"
    )
    identity_complete_path = identity_dir / "identity_complete.json"
    identity_complete = _load_json_object(
        identity_complete_path, "HPE identity_complete.json"
    )
    declared_validation = _resolve_inside(
        root, identity_complete.get("validation"), "identity-complete validation"
    )
    if declared_validation != validation_dir:
        raise HPETranscriptCrosswalkError(
            "identity_complete validation path differs from mapping.json"
        )
    validation_path = validation_dir / "validation_complete.json"
    validation = _load_json_object(validation_path, "HPE validation_complete.json")
    validation_hash = sha256_file(validation_path)
    if identity_complete.get("validation_marker_sha256") != validation_hash:
        raise HPETranscriptCrosswalkError(
            "HPE validation marker changed since identity completion"
        )
    cardinality = validation.get("geometry_cardinality_gate")
    if (
        not isinstance(cardinality, dict)
        or cardinality.get("production_pass") is not True
        or cardinality.get("coverage_complete") is not True
    ):
        raise HPETranscriptCrosswalkError(
            "HPE validation marker is not a complete production-pass authority"
        )

    core_hashes = identity_complete.get("core_sha256")
    if not isinstance(core_hashes, dict):
        raise HPETranscriptCrosswalkError("HPE identity_complete lacks core_sha256")
    identity_world_path = identity_dir / "world_persons.csv"
    persons_path = identity_dir / "persons.json"
    if (
        core_hashes.get("world_persons.csv") != sha256_file(identity_world_path)
        or core_hashes.get("persons.json") != sha256_file(persons_path)
    ):
        raise HPETranscriptCrosswalkError(
            "HPE identity core files changed since identity completion"
        )
    persons = _load_json_object(persons_path, "HPE persons.json")
    identity_scope = identity_complete.get("identity_scope")
    if not isinstance(identity_scope, dict) or persons.get("identity_scope") != identity_scope:
        raise HPETranscriptCrosswalkError(
            "HPE persons.json identity scope differs from identity_complete"
        )
    try:
        identity_start = int(identity_scope["start_inclusive"])
        identity_end = int(identity_scope["end_exclusive"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HPETranscriptCrosswalkError("HPE identity scope is malformed") from exc
    if identity_start < 0 or identity_end <= identity_start:
        raise HPETranscriptCrosswalkError("HPE identity scope is invalid")

    delivery_world_path = delivery / "world_persons.csv"
    raw_delivery_world = _load_csv(delivery_world_path, "delivery world_persons.csv")
    delivery_world = raw_delivery_world.copy()
    required_world = {"frame_index", "person"}
    if not required_world.issubset(delivery_world.columns):
        raise HPETranscriptCrosswalkError(
            "delivery world_persons.csv lacks frame_index/person"
        )
    delivery_world = delivery_world.copy()
    delivery_world["frame_index"] = _coerce_integer_series(
        delivery_world["frame_index"],
        label="delivery world_persons frame_index",
        allow_zero=True,
    )
    delivery_world["person"] = _coerce_integer_series(
        delivery_world["person"],
        label="delivery world_persons final person",
        allow_zero=False,
    )
    return {
        "mapping": mapping,
        "proposal": proposal,
        "identity_dir": identity_dir,
        "validation_dir": validation_dir,
        "validation": validation,
        "hpe_repo_root": root,
        "identity_scope": (identity_start, identity_end),
        "delivery_world": delivery_world,
        "bindings": {
            "mapping": _binding(mapping_path),
            "participant_map": participant_map_binding,
            "identity_complete": _binding(identity_complete_path),
            "identity_world_persons": _binding(
                identity_world_path, _load_csv(identity_world_path, "identity world_persons")
            ),
            "persons": _binding(persons_path),
            "validation_complete": _binding(validation_path),
            "delivery_world_persons": _binding(
                delivery_world_path, raw_delivery_world
            ),
        },
        "accepted": asdict(accepted),
    }


def _validate_gaze_reference(
    *,
    scene: str,
    accepted: AcceptedIdentityWindow,
    identity_dir: Path,
    portable_dtypes: bool = False,
) -> dict[str, Any]:
    directory = identity_dir.resolve()
    manifest_path = directory / "identity_annotations.json"
    observations_path = directory / "tracklet_observations.csv"
    try:
        manifest = load_annotation_manifest(manifest_path)
        observations = (
            validate_tracklet_observations_portable(observations_path, manifest)
            if portable_dtypes
            else validate_tracklet_observations(observations_path, manifest)
        )
    except IdentityAnnotationError as exc:
        raise HPETranscriptCrosswalkError(
            f"Gaze identity artifact failed canonical replay: {exc}"
        ) from exc
    group, scenario = scene.rsplit("__", maxsplit=1)
    if str(manifest.get("dataset")) != group or str(manifest.get("session")) != scenario:
        raise HPETranscriptCrosswalkError(
            "Gaze identity artifact dataset/session differs from the accepted scene"
        )
    source = manifest.get("source")
    declared_video = source.get("declared_video") if isinstance(source, dict) else None
    if (
        not isinstance(declared_video, dict)
        or declared_video.get("camera_dir") != accepted.camera
    ):
        raise HPETranscriptCrosswalkError(
            "Gaze identity artifact camera differs from the live ledger"
        )
    local_window = annotation_frame_window(manifest)
    if local_window != accepted.local_window:
        raise HPETranscriptCrosswalkError(
            "Gaze identity artifact window differs from the live ledger"
        )
    scope = manifest.get("annotation_scope")
    if not isinstance(scope, dict):
        raise HPETranscriptCrosswalkError("Gaze identity annotation scope is malformed")
    try:
        core_start = int(scope["core_start_logical_frame_inclusive"])
        core_end = int(scope["core_end_logical_frame_exclusive"])
        segment_logical_start = int(scope["gaze_segment_logical_start_frame"])
        local_start = int(scope["gaze_local_start_frame_inclusive"])
        local_end = int(scope["gaze_local_end_frame_exclusive"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HPETranscriptCrosswalkError(
            "Gaze identity artifact lacks logical/local window binding"
        ) from exc
    if (local_start, local_end) != accepted.local_window:
        raise HPETranscriptCrosswalkError(
            "Gaze identity artifact local window differs from the ledger"
        )
    logical_start = segment_logical_start + local_start
    logical_end = segment_logical_start + local_end
    if logical_start < core_start or logical_end > core_end:
        raise HPETranscriptCrosswalkError(
            "accepted Gaze logical window is outside the confirmed HPE core"
        )
    annotations = manifest.get("annotations")
    if not isinstance(annotations, list) or any(
        isinstance(item, dict) and item.get("status") == "unreviewed"
        for item in annotations
    ):
        raise HPETranscriptCrosswalkError(
            "Gaze identity artifact still has unreviewed manual tracklets"
        )
    bridge, conflicts = build_identity_bridge(observations, manifest)
    if not conflicts.empty:
        raise HPETranscriptCrosswalkError(
            "Gaze identity artifact has same-frame canonical identity conflicts"
        )
    confirmed = bridge.loc[
        (bridge["annotation_status"] == "confirmed")
        & bridge["participant"].isin(_CANONICAL_LABELS)
    ].copy()
    if confirmed.empty or set(confirmed["participant"].astype(str)) != _CANONICAL_LABELS:
        raise HPETranscriptCrosswalkError(
            "Gaze identity artifact lacks confirmed references for A/B/C/R"
        )
    confirmed["frame_index"] = _coerce_integer_series(
        confirmed["frame_index"], label="Gaze bridge frame_index", allow_zero=True
    )
    for column in _GAZE_BBOX_COLUMNS:
        confirmed[column] = pd.to_numeric(confirmed[column], errors="raise")
    boxes = confirmed.loc[:, list(_GAZE_BBOX_COLUMNS)].to_numpy(dtype=float)
    if (
        not np.isfinite(boxes).all()
        or not (confirmed["bbox_x1"] > confirmed["bbox_x0"]).all()
        or not (confirmed["bbox_y1"] > confirmed["bbox_y0"]).all()
    ):
        raise HPETranscriptCrosswalkError(
            "Gaze confirmed reference contains invalid bounding boxes"
        )
    return {
        "manifest": manifest,
        "confirmed": confirmed,
        "logical_window": (logical_start, logical_end),
        "local_window": (local_start, local_end),
        "frame_offset": logical_start - local_start,
        "bindings": {
            "manifest": _binding(manifest_path),
            "tracklet_observations": _binding(
                observations_path,
                _load_csv(observations_path, "Gaze tracklet observations"),
            ),
        },
    }


def _load_relabeled_camera(
    identity_dir: Path, camera: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = identity_dir / f"relabeled_{camera}.csv"
    raw_frame = _load_csv(path, f"HPE relabeled {camera} CSV")
    required = {"frame_index", "person", *_HPE_BBOX_COLUMNS}
    if not required.issubset(raw_frame.columns):
        raise HPETranscriptCrosswalkError(
            f"HPE relabeled {camera} CSV lacks required frame/person/bbox columns"
        )
    frame = raw_frame.copy()
    frame["frame_index"] = _coerce_integer_series(
        frame["frame_index"],
        label=f"HPE relabeled {camera} frame_index",
        allow_zero=True,
    )
    try:
        person_numeric = pd.to_numeric(frame["person"], errors="coerce")
    except (TypeError, ValueError) as exc:
        raise HPETranscriptCrosswalkError(
            f"HPE relabeled {camera} person column is malformed"
        ) from exc
    finite_integer = person_numeric.notna() & np.isfinite(person_numeric)
    finite_integer &= person_numeric.eq(np.floor(person_numeric))
    frame["person"] = person_numeric.where(finite_integer).astype("Int64")
    for column in _HPE_BBOX_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["hpe_camera"] = camera
    return frame, _binding(path, raw_frame)


def _as_string_list(value: Any, label: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = list(value)
    else:
        raise HPETranscriptCrosswalkError(f"{label} must be a string or string list")
    if not values or any(not item.strip() for item in values):
        raise HPETranscriptCrosswalkError(f"{label} may not be empty")
    return values


def _camera_review_catalog(
    hpe: Mapping[str, Any], camera: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load hash-bound sparse stills plus the fallback raw-video contract."""

    root = Path(hpe["hpe_repo_root"])
    validation = hpe["validation"]
    if not isinstance(validation, dict):
        raise HPETranscriptCrosswalkError("HPE validation marker is malformed")
    csv_root = _resolve_inside(root, validation.get("csv_root"), "HPE pose CSV root")
    pose_sources = validation.get("pose_sources")
    camera_sources = pose_sources.get("cameras") if isinstance(pose_sources, dict) else None
    pose_source = camera_sources.get(camera) if isinstance(camera_sources, dict) else None
    if not isinstance(pose_source, dict):
        raise HPETranscriptCrosswalkError(
            f"HPE validation marker has no pose source for {camera}"
        )
    metadata_name = pose_source.get("metadata")
    if (
        not isinstance(metadata_name, str)
        or Path(metadata_name).name != metadata_name
        or not metadata_name.lower().endswith(".json")
    ):
        raise HPETranscriptCrosswalkError(
            f"HPE validation has an invalid metadata filename for {camera}"
        )
    metadata_path = (csv_root / camera / metadata_name).resolve()
    try:
        metadata_path.relative_to(root)
    except ValueError as exc:
        raise HPETranscriptCrosswalkError("HPE run metadata escapes the repository") from exc
    metadata = _load_json_object(metadata_path, f"HPE {camera} run metadata")
    metadata_hash = sha256_file(metadata_path)
    if pose_source.get("metadata_sha256") != metadata_hash:
        raise HPETranscriptCrosswalkError(
            f"HPE {camera} run metadata changed since validation"
        )
    try:
        fps = float(metadata["source_fps"])
        width = int(metadata["width"])
        height = int(metadata["height"])
        processed_frames = int(metadata["processed_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HPETranscriptCrosswalkError(
            f"HPE {camera} run metadata lacks video dimensions/timing"
        ) from exc
    if (
        not math.isfinite(fps)
        or fps <= 0
        or width <= 0
        or height <= 0
        or processed_frames <= 0
    ):
        raise HPETranscriptCrosswalkError(
            f"HPE {camera} run metadata has invalid video dimensions/timing"
        )
    scene = str(hpe["accepted"]["scene"])
    group, scenario = scene.rsplit("__", maxsplit=1)
    if (
        str(pose_source.get("bound_day")) != group
        or str(pose_source.get("bound_scenario")) != scenario
        or pose_source.get("bound_camera") != camera
        or not math.isclose(float(pose_source.get("source_fps")), fps, rel_tol=0.0, abs_tol=1e-9)
        or int(pose_source.get("processed_frames")) != processed_frames
    ):
        raise HPETranscriptCrosswalkError(
            f"HPE validation/run-metadata identity differs for {camera}"
        )

    stills: dict[int, dict[str, Any]] = {}
    review = metadata.get("review_artifacts")
    files = review.get("files") if isinstance(review, dict) else None
    if isinstance(files, dict):
        pattern = re.compile(r"^review_source_frame_NON_PRODUCTION_(\d+)\.jpg$")
        for name, raw_binding in files.items():
            match = pattern.fullmatch(str(name))
            if match is None:
                continue
            if not isinstance(raw_binding, dict):
                raise HPETranscriptCrosswalkError(
                    f"HPE {camera} review-still binding is malformed"
                )
            still_path = _resolve_inside(
                root, raw_binding.get("path"), f"HPE {camera} review still"
            )
            try:
                size_bytes = int(raw_binding["size_bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HPETranscriptCrosswalkError(
                    f"HPE {camera} review-still size binding is malformed"
                ) from exc
            digest = raw_binding.get("sha256")
            if (
                not still_path.is_file()
                or still_path.stat().st_size != size_bytes
                or not isinstance(digest, str)
                or sha256_file(still_path) != digest
            ):
                raise HPETranscriptCrosswalkError(
                    f"HPE {camera} review still changed since extraction"
                )
            frame_index = int(match.group(1))
            stills[frame_index] = {
                "kind": "still",
                "path": str(still_path),
                "size_bytes": size_bytes,
                "sha256": digest,
                "camera": camera,
                "frame_index": frame_index,
                "logical_frame": frame_index,
            }

    segments = _as_string_list(metadata.get("segments"), f"HPE {camera} segments")
    input_files = metadata.get("input_files")
    if isinstance(input_files, dict) and len(segments) == 1:
        input_files = [input_files]
    if not isinstance(input_files, list) or len(input_files) != len(segments):
        raise HPETranscriptCrosswalkError(
            f"HPE {camera} input-file bindings do not cover its segments"
        )
    raw_counts = metadata.get("segments_decoded_frames")
    if raw_counts is None and len(segments) == 1:
        decoded_counts = [processed_frames]
    elif isinstance(raw_counts, list) and len(raw_counts) == len(segments):
        try:
            decoded_counts = [int(value) for value in raw_counts]
        except (TypeError, ValueError) as exc:
            raise HPETranscriptCrosswalkError(
                f"HPE {camera} segment frame counts are malformed"
            ) from exc
    else:
        decoded_counts = []
    video_segments: list[dict[str, Any]] = []
    for index, (segment, file_binding) in enumerate(
        zip(segments, input_files, strict=True)
    ):
        if not isinstance(file_binding, dict) or file_binding.get("path") != segment:
            raise HPETranscriptCrosswalkError(
                f"HPE {camera} segment and input-file bindings disagree"
            )
        try:
            size_bytes = int(file_binding["size_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HPETranscriptCrosswalkError(
                f"HPE {camera} input-file size binding is malformed"
            ) from exc
        decoded_frames = decoded_counts[index] if decoded_counts else None
        if decoded_frames is not None and decoded_frames <= 0:
            raise HPETranscriptCrosswalkError(
                f"HPE {camera} segment frame count must be positive"
            )
        video_segments.append(
            {
                "path": str(Path(segment)),
                "size_bytes": size_bytes,
                "mtime_ns": int(file_binding.get("mtime_ns", 0)),
                "decoded_frames": decoded_frames,
                "video_identity": {
                    "basename": Path(segment).name,
                    "camera_dir": camera,
                    "fps": fps,
                    "frame_count": decoded_frames,
                    "width": width,
                    "height": height,
                    "size_bytes": size_bytes,
                },
            }
        )
    return (
        {
            "camera": camera,
            "stills": stills,
            "segments": video_segments,
            "run_metadata_path": str(metadata_path),
            "run_metadata_sha256": metadata_hash,
        },
        _binding(metadata_path),
    )


def _hpe_image_source(catalog: Mapping[str, Any], logical_frame: int) -> dict[str, Any]:
    still = catalog["stills"].get(logical_frame)
    if still is not None:
        return dict(still)
    segments = catalog["segments"]
    cursor = 0
    for segment_index, segment in enumerate(segments):
        count = segment.get("decoded_frames")
        if count is None:
            raise HPETranscriptCrosswalkError(
                "raw-video fallback needs per-segment decoded frame counts"
            )
        if logical_frame < cursor + int(count):
            local_frame = logical_frame - cursor
            return {
                "kind": "video",
                "camera": catalog["camera"],
                "logical_frame": logical_frame,
                "frame_index": local_frame,
                "segment_index": segment_index,
                "segment_local_frame": local_frame,
                "path": segment["path"],
                "run_metadata_path": catalog["run_metadata_path"],
                "run_metadata_sha256": catalog["run_metadata_sha256"],
                "video_identity": segment["video_identity"],
            }
        cursor += int(count)
    raise HPETranscriptCrosswalkError(
        f"logical frame {logical_frame} is outside HPE raw-video segments"
    )


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _valid_hpe_rows(
    frame: pd.DataFrame,
    *,
    expected_people: set[int],
    logical_window: tuple[int, int],
) -> pd.DataFrame:
    start, end = logical_window
    numeric_boxes = frame.loc[:, list(_HPE_BBOX_COLUMNS)].to_numpy(dtype=float)
    finite_boxes = np.isfinite(numeric_boxes).all(axis=1)
    valid_boxes = (
        finite_boxes
        & (frame["bbox_x2"].to_numpy(dtype=float) > frame["bbox_x1"].to_numpy(dtype=float))
        & (frame["bbox_y2"].to_numpy(dtype=float) > frame["bbox_y1"].to_numpy(dtype=float))
    )
    mask = (
        frame["person"].notna()
        & frame["person"].isin(expected_people)
        & frame["frame_index"].ge(start)
        & frame["frame_index"].lt(end)
        & valid_boxes
    )
    output = frame.loc[mask].copy()
    output["person"] = output["person"].astype(int)
    duplicate = output.duplicated(["frame_index", "person"], keep=False)
    return output.loc[~duplicate].copy()


def expected_hpe_persons(
    delivery_world_persons: pd.DataFrame, logical_window: tuple[int, int]
) -> list[int]:
    """Return final HPE ``person`` keys present in a strict half-open window."""

    if not {"frame_index", "person"}.issubset(delivery_world_persons.columns):
        raise HPETranscriptCrosswalkError(
            "delivery world_persons lacks frame_index/final person"
        )
    start, end = logical_window
    if start < 0 or end <= start:
        raise HPETranscriptCrosswalkError("logical accepted window is invalid")
    frames = _coerce_integer_series(
        delivery_world_persons["frame_index"],
        label="delivery world_persons frame_index",
        allow_zero=True,
    )
    people = _coerce_integer_series(
        delivery_world_persons["person"],
        label="delivery world_persons final person",
        allow_zero=False,
    )
    mask = frames.ge(start) & frames.lt(end)
    return sorted(people.loc[mask].unique().tolist())


def _sample_phases(count: int) -> list[str]:
    if count < 3:
        raise HPETranscriptCrosswalkError(
            "crosswalk review requires at least early/middle/late samples"
        )
    if count == 3:
        return ["early", "middle", "late"]
    return ["early", *[f"middle_{index}" for index in range(1, count - 1)], "late"]


def build_review_samples(
    *,
    accepted: AcceptedIdentityWindow,
    logical_window: tuple[int, int],
    expected_people: Iterable[int],
    relabeled_by_camera: Mapping[str, pd.DataFrame],
    gaze_confirmed: pd.DataFrame,
    gaze_frame_offset: int,
    gaze_declared_video: Mapping[str, Any],
    gaze_video_identity: Mapping[str, Any] | None,
    relabeled_bindings: Mapping[str, Mapping[str, Any]],
    review_catalogs: Mapping[str, Mapping[str, Any]],
    world_presence: pd.DataFrame,
    samples_per_person: int = 3,
) -> pd.DataFrame:
    """Choose deterministic early/middle/late visual evidence per HPE person."""

    phases = _sample_phases(samples_per_person)
    expected = sorted({int(person) for person in expected_people})
    if not expected or expected[0] < 1:
        raise HPETranscriptCrosswalkError("expected HPE persons must be positive")
    gaze_counts = gaze_confirmed.groupby("frame_index")["participant"].nunique()
    # All four labels on a reference frame make a visual identity comparison
    # self-contained; a lone confirmed box is not enough evidence.
    gaze_frames = set(gaze_counts.loc[gaze_counts == len(_CANONICAL_LABELS)].index.astype(int))
    world_keys = {
        (int(row["frame_index"]), int(row["person"]))
        for _, row in world_presence.loc[:, ["frame_index", "person"]]
        .drop_duplicates()
        .iterrows()
    }
    valid_by_camera: dict[str, pd.DataFrame] = {}
    for camera, frame in relabeled_by_camera.items():
        valid = _valid_hpe_rows(
            frame,
            expected_people=set(expected),
            logical_window=logical_window,
        )
        valid_by_camera[camera] = valid.loc[
            [
                (int(row["frame_index"]), int(row["person"])) in world_keys
                for _, row in valid.iterrows()
            ]
        ].copy()

    records: list[dict[str, Any]] = []
    ranked_cameras: list[tuple[int, int, str, pd.DataFrame]] = []
    for camera, frame in valid_by_camera.items():
        candidate = frame.copy()
        candidate["gaze_local_frame"] = candidate["frame_index"] - int(gaze_frame_offset)
        candidate = candidate.loc[candidate["gaze_local_frame"].isin(gaze_frames)]
        per_person_counts = candidate.groupby("person")["frame_index"].nunique()
        if all(int(per_person_counts.get(person, 0)) >= samples_per_person for person in expected):
            still_frames = set(review_catalogs[camera]["stills"])
            still_counts = (
                candidate.loc[candidate["frame_index"].isin(still_frames)]
                .groupby("person")["frame_index"]
                .nunique()
            )
            all_still = all(
                int(still_counts.get(person, 0)) >= samples_per_person for person in expected
            )
            rank = 0 if camera == accepted.camera else 1
            ranked_cameras.append(
                (rank, 0 if all_still else 1, camera, candidate)
            )
    if not ranked_cameras:
        raise HPETranscriptCrosswalkError(
            "no single HPE camera provides enough valid anchors for every expected person"
        )
    _, still_policy_rank, selected_camera, selected_camera_rows = sorted(
        ranked_cameras,
        key=lambda item: (
            item[0],
            item[1],
            -int(item[3]["frame_index"].nunique()),
            item[2],
        ),
    )[0]
    catalog = review_catalogs[selected_camera]
    still_frames = set(catalog["stills"])

    for person in expected:
        camera = selected_camera
        person_rows = selected_camera_rows.loc[
            selected_camera_rows["person"] == person
        ].copy()
        if still_policy_rank == 0:
            person_rows = person_rows.loc[person_rows["frame_index"].isin(still_frames)]
        frames = sorted(person_rows["frame_index"].astype(int).unique().tolist())
        chunks = np.array_split(np.asarray(frames, dtype=int), samples_per_person)
        selected_frames: list[int] = []
        for chunk in chunks:
            if len(chunk) == 0:
                raise HPETranscriptCrosswalkError(
                    f"HPE person {person} cannot provide distinct review anchors"
                )
            centre = (float(chunk[0]) + float(chunk[-1])) / 2.0
            candidates = person_rows.loc[person_rows["frame_index"].isin(chunk)].copy()
            face_matched = pd.to_numeric(
                candidates.get("face_matched", pd.Series(0, index=candidates.index)),
                errors="coerce",
            ).fillna(0.0)
            confidence = pd.to_numeric(
                candidates.get("confidence", pd.Series(0, index=candidates.index)),
                errors="coerce",
            ).fillna(0.0)
            area = (candidates["bbox_x2"] - candidates["bbox_x1"]) * (
                candidates["bbox_y2"] - candidates["bbox_y1"]
            )
            candidates = candidates.assign(
                _face_matched=face_matched,
                _confidence=confidence,
                _area=area,
                _centre_distance=(candidates["frame_index"] - centre).abs(),
            ).sort_values(
                ["_face_matched", "_confidence", "_area", "_centre_distance", "frame_index"],
                ascending=[False, False, False, True, True],
                kind="stable",
            )
            selected_frames.append(int(candidates.iloc[0]["frame_index"]))
        if len(set(selected_frames)) != samples_per_person:
            raise HPETranscriptCrosswalkError(
                f"HPE person {person} cannot provide distinct review anchors"
            )
        camera_frame = valid_by_camera[camera]
        for phase, logical_frame in zip(phases, selected_frames, strict=True):
            selected = person_rows.loc[
                person_rows["frame_index"] == logical_frame
            ].iloc[0]
            gaze_local_frame = int(logical_frame - gaze_frame_offset)
            all_hpe = camera_frame.loc[camera_frame["frame_index"] == logical_frame]
            hpe_boxes = [
                {
                    "hpe_person": int(row["person"]),
                    "bbox": [float(row[column]) for column in _HPE_BBOX_COLUMNS],
                }
                for _, row in all_hpe.sort_values("person", kind="stable").iterrows()
            ]
            gaze_rows = gaze_confirmed.loc[
                gaze_confirmed["frame_index"] == gaze_local_frame
            ].sort_values(["participant", "source_kind", "tracklet_id"], kind="stable")
            gaze_boxes = [
                {
                    "participant": str(row["participant"]),
                    "slot": str(row["slot"]),
                    "source_kind": str(row["source_kind"]),
                    "tracklet_id": str(row["tracklet_id"]),
                    "bbox": [float(row[column]) for column in _GAZE_BBOX_COLUMNS],
                }
                for _, row in gaze_rows.iterrows()
            ]
            if not gaze_boxes:
                raise HPETranscriptCrosswalkError(
                    "sample selection produced a frame without confirmed Gaze reference"
                )
            relabeled_binding = relabeled_bindings[camera]
            sample_id = f"person_{person:03d}_{phase}"
            hpe_source = _hpe_image_source(catalog, logical_frame)
            hpe_source["relabeled_csv_path"] = relabeled_binding["path"]
            hpe_source["relabeled_csv_sha256"] = relabeled_binding["sha256"]
            if camera == accepted.camera:
                gaze_source = {"kind": "same_as_hpe"}
            else:
                identity = (
                    dict(gaze_video_identity)
                    if gaze_video_identity is not None
                    else dict(gaze_declared_video)
                )
                path = identity.get("last_verified_path") or identity.get("first_bound_path")
                gaze_source = {
                    "kind": "video",
                    "camera": accepted.camera,
                    "frame_index": gaze_local_frame,
                    "local_frame": gaze_local_frame,
                    "path": path,
                    "video_identity": identity,
                    "declared_video": dict(gaze_declared_video),
                }
            records.append(
                {
                    "sample_id": sample_id,
                    "hpe_person": person,
                    "phase": phase,
                    "logical_frame": logical_frame,
                    "gaze_local_frame": gaze_local_frame,
                    "hpe_camera": camera,
                    "gaze_camera": accepted.camera,
                    "hpe_bbox_json": _compact_json(
                        [float(selected[column]) for column in _HPE_BBOX_COLUMNS]
                    ),
                    "all_hpe_boxes_json": _compact_json(hpe_boxes),
                    "gaze_confirmed_boxes_json": _compact_json(gaze_boxes),
                    "hpe_image_source_json": _compact_json(hpe_source),
                    "gaze_image_source_json": _compact_json(gaze_source),
                }
            )
    samples = pd.DataFrame.from_records(records, columns=_SAMPLE_COLUMNS)
    samples = samples.sort_values(
        ["hpe_person", "logical_frame", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    if samples["sample_id"].duplicated().any():
        raise HPETranscriptCrosswalkError("sample IDs are not unique")
    return samples


def _normalised_upstream_proposal(proposal: Mapping[int, str]) -> dict[str, str]:
    return {str(person): proposal[person] for person in sorted(proposal)}


def _atomic_write_audit(path: Path, event: dict[str, Any]) -> None:
    existing = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        for line in existing.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HPETranscriptCrosswalkError(
                    "crosswalk audit contains invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise HPETranscriptCrosswalkError(
                    "crosswalk audit events must be JSON objects"
                )
    payload = existing + json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".jsonl.tmp", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def prepare_crosswalk_artifact(
    *,
    scene: str,
    delivery_dir: str | Path,
    hpe_repo_root: str | Path,
    gaze_identity_dir: str | Path,
    output_dir: str | Path,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    hpe_cameras: Iterable[str] | None = None,
    samples_per_person: int = 3,
) -> tuple[Path, Path]:
    """Prepare or safely resume one independent crosswalk review artifact."""

    ledger = Path(ledger_path).resolve()
    accepted_all = parse_accepted_identity_ledger(ledger)
    if scene not in accepted_all:
        raise HPETranscriptCrosswalkError(
            f"scene {scene!r} is not accepted in the live identity ledger"
        )
    accepted = accepted_all[scene]
    hpe = _validate_hpe_authority(
        scene=scene,
        accepted=accepted,
        delivery_dir=Path(delivery_dir),
        hpe_repo_root=Path(hpe_repo_root),
    )
    gaze = _validate_gaze_reference(
        scene=scene,
        accepted=accepted,
        identity_dir=Path(gaze_identity_dir),
    )
    logical_window = gaze["logical_window"]
    identity_start, identity_end = hpe["identity_scope"]
    if logical_window[0] < identity_start or logical_window[1] > identity_end:
        raise HPETranscriptCrosswalkError(
            "accepted Gaze logical window is outside the confirmed HPE identity scope"
        )
    delivery_world = hpe["delivery_world"]
    scoped_world = delivery_world.loc[
        delivery_world["frame_index"].ge(logical_window[0])
        & delivery_world["frame_index"].lt(logical_window[1])
    ].copy()
    expected_people = expected_hpe_persons(delivery_world, logical_window)
    if len(expected_people) < len(_CANONICAL_LABELS):
        raise HPETranscriptCrosswalkError(
            "accepted HPE window contains fewer final persons than A/B/C/R"
        )
    if hpe_cameras is None:
        pose_sources = hpe["validation"].get("pose_sources")
        validated_cameras = (
            set(pose_sources.get("cameras", {}))
            if isinstance(pose_sources, dict)
            and isinstance(pose_sources.get("cameras"), dict)
            else set()
        )
        requested_cameras = sorted(
            path.stem.removeprefix("relabeled_")
            for path in hpe["identity_dir"].glob("relabeled_Cam_*.csv")
            if path.is_file() and path.stem.removeprefix("relabeled_") in validated_cameras
        )
    else:
        requested_cameras = list(dict.fromkeys(hpe_cameras))
    if not requested_cameras or any(
        not isinstance(camera, str)
        or not camera.startswith("Cam_")
        or not camera[4:].isdigit()
        for camera in requested_cameras
    ):
        raise HPETranscriptCrosswalkError("HPE review cameras are malformed")
    relabeled: dict[str, pd.DataFrame] = {}
    relabeled_bindings: dict[str, dict[str, Any]] = {}
    review_catalogs: dict[str, dict[str, Any]] = {}
    run_metadata_bindings: dict[str, dict[str, Any]] = {}

    def load_camera(camera: str) -> None:
        frame, binding = _load_relabeled_camera(hpe["identity_dir"], camera)
        catalog, run_binding = _camera_review_catalog(hpe, camera)
        relabeled[camera] = frame
        relabeled_bindings[camera] = binding
        review_catalogs[camera] = catalog
        run_metadata_bindings[camera] = run_binding

    def make_samples() -> pd.DataFrame:
        return build_review_samples(
            accepted=accepted,
            logical_window=logical_window,
            expected_people=expected_people,
            relabeled_by_camera=relabeled,
            gaze_confirmed=gaze["confirmed"],
            gaze_frame_offset=gaze["frame_offset"],
            gaze_declared_video=gaze["manifest"]["source"]["declared_video"],
            gaze_video_identity=(
                gaze["manifest"].get("video")
                if isinstance(gaze["manifest"].get("video"), dict)
                else None
            ),
            relabeled_bindings=relabeled_bindings,
            review_catalogs=review_catalogs,
            world_presence=scoped_world,
            samples_per_person=samples_per_person,
        )

    samples: pd.DataFrame | None = None
    # The accepted Gaze camera is always preferred.  If it alone satisfies the
    # contract, do not parse unrelated multi-million-row camera CSVs.
    if accepted.camera in requested_cameras:
        load_camera(accepted.camera)
        try:
            samples = make_samples()
        except HPETranscriptCrosswalkError as exc:
            if "no single HPE camera provides enough valid anchors" not in str(exc):
                raise
    if samples is None:
        for camera in requested_cameras:
            if camera not in relabeled:
                load_camera(camera)
        samples = make_samples()
    now = _utc_now()
    destination = Path(output_dir).resolve()
    samples_path = destination / SAMPLES_NAME
    manifest_path = destination / MANIFEST_NAME
    requested_manifest = {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": scene,
        "accepted_window": {
            **asdict(accepted),
            "logical_start_inclusive": int(logical_window[0]),
            "logical_end_exclusive": int(logical_window[1]),
            "filter_rule": "start <= frame_index < end",
        },
        "expected_hpe_persons": expected_people,
        "sampling_policy": {
            "format_version": SAMPLE_FORMAT_VERSION,
            "samples_per_person": samples_per_person,
            "phases": _sample_phases(samples_per_person),
            "selection": "deterministic_per_person_time_chunks_then_quality_rank",
            "camera_preference": (
                "accepted_gaze_camera_then_all_hash_bound_stills_then_"
                "max_eligible_frames_then_name"
            ),
            "bbox_rule": "finite_and_strictly_positive_width_height",
            "frame_scope": "strict_half_open_accepted_window",
            "reference_rule": "confirmed_gaze_identity_only",
        },
        "sources": {
            "ledger": _binding(ledger),
            "hpe_repo_root": str(Path(hpe_repo_root).resolve()),
            "delivery_dir": str(Path(delivery_dir).resolve()),
            "hpe": hpe["bindings"],
            "relabeled_cameras": relabeled_bindings,
            "hpe_run_metadata": run_metadata_bindings,
            "gaze_identity_dir": str(Path(gaze_identity_dir).resolve()),
            "gaze": gaze["bindings"],
        },
        "upstream_proposal": _normalised_upstream_proposal(hpe["proposal"]),
        "sample_decisions": {
            sample_id: {
                "status": "unreviewed",
                "label": None,
                "note": "",
                "updated_at": None,
            }
            for sample_id in samples["sample_id"].astype(str)
        },
        "state": "in_review",
        "review_video_identities": None,
        "upstream_mismatch": None,
        "conflict_report": None,
        "final_mapping": None,
        "created_at": now,
        "updated_at": now,
    }

    existing = [samples_path.exists(), manifest_path.exists()]
    if any(existing):
        if not all(existing):
            raise HPETranscriptCrosswalkError(
                "crosswalk artifact is partial; preserve it and start a new output revision"
            )
        # This call already replayed all current HPE/Gaze authority above and
        # reconstructed the requested bindings.  A second full replay would be
        # redundant; the local state plus the exact comparable below is enough.
        current, current_samples = load_crosswalk_review_state(
            destination, verify_bound_video_bytes=True
        )
        comparable = dict(current)
        for key in (
            "sample_decisions",
            "state",
            "review_video_identities",
            "upstream_mismatch",
            "conflict_report",
            "final_mapping",
            "created_at",
            "updated_at",
            "review_samples",
        ):
            comparable.pop(key, None)
        requested_comparable = dict(requested_manifest)
        for key in (
            "sample_decisions",
            "state",
            "review_video_identities",
            "upstream_mismatch",
            "conflict_report",
            "final_mapping",
            "created_at",
            "updated_at",
        ):
            requested_comparable.pop(key, None)
        if comparable != requested_comparable or not current_samples.equals(samples):
            raise HPETranscriptCrosswalkError(
                "requested crosswalk preparation differs from the existing revision"
            )
        return samples_path, manifest_path

    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(samples_path, samples)
    requested_manifest["review_samples"] = _binding(
        samples_path, pd.read_csv(samples_path)
    )
    _atomic_write_json(manifest_path, requested_manifest)
    _atomic_write_audit(
        destination / AUDIT_NAME,
        {
            "timestamp": now,
            "action": "prepare",
            "scene": scene,
            "expected_hpe_persons": expected_people,
            "sample_ids": samples["sample_id"].astype(str).tolist(),
        },
    )
    return samples_path, manifest_path


def _portable_dtype_family(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    name = value.strip().lower()
    if name == "object" or name == "str" or name.startswith("string"):
        return "text"
    if re.fullmatch(r"int(8|16|32|64)", name):
        return "signed_integer"
    if re.fullmatch(r"uint(8|16|32|64)", name):
        return "unsigned_integer"
    if re.fullmatch(r"float(16|32|64|128)", name):
        return "floating"
    if name in {"bool", "boolean"}:
        return "boolean"
    return name


def _portable_dtypes_match(recorded: Any, actual: list[str]) -> bool:
    return (
        isinstance(recorded, list)
        and len(recorded) == len(actual)
        and all(
            _portable_dtype_family(left) == _portable_dtype_family(right)
            and _portable_dtype_family(left) is not None
            for left, right in zip(recorded, actual, strict=True)
        )
    )


def _portable_tabular_bindings_equal(left: Any, right: Any) -> bool:
    required = {"path", "sha256", "columns", "parsed_dtypes", "n_rows"}
    if (
        not isinstance(left, dict)
        or not isinstance(right, dict)
        or set(left) != required
        or set(right) != required
    ):
        return False
    return (
        Path(str(left["path"])).resolve() == Path(str(right["path"])).resolve()
        and left["sha256"] == right["sha256"]
        and left["columns"] == right["columns"]
        and left["n_rows"] == right["n_rows"]
        and _portable_dtypes_match(left["parsed_dtypes"], right["parsed_dtypes"])
    )


def _require_canonical_authority_bindings(
    *,
    recorded_hpe: Mapping[str, Any],
    canonical_hpe: Mapping[str, Any],
    recorded_gaze: Mapping[str, Any],
    canonical_gaze: Mapping[str, Any],
    portable_dtypes: bool,
) -> None:
    """Reject byte-identical copies that are not owned by canonical sources."""

    hpe_tabular_keys = {
        "participant_map",
        "identity_world_persons",
        "delivery_world_persons",
    }
    hpe_matches = set(recorded_hpe) == set(canonical_hpe)
    if hpe_matches:
        for key, canonical in canonical_hpe.items():
            recorded = recorded_hpe[key]
            if key in hpe_tabular_keys and portable_dtypes:
                matches = _portable_tabular_bindings_equal(recorded, canonical)
            else:
                matches = recorded == canonical
            if not matches:
                hpe_matches = False
                break
    if not hpe_matches:
        raise HPETranscriptCrosswalkError(
            "manifest HPE bindings differ from canonical delivery authority"
        )

    gaze_matches = set(recorded_gaze) == set(canonical_gaze)
    if gaze_matches:
        for key, canonical in canonical_gaze.items():
            recorded = recorded_gaze[key]
            if key == "tracklet_observations" and portable_dtypes:
                matches = _portable_tabular_bindings_equal(recorded, canonical)
            else:
                matches = recorded == canonical
            if not matches:
                gaze_matches = False
                break
    if not gaze_matches:
        raise HPETranscriptCrosswalkError(
            "manifest Gaze bindings differ from canonical identity authority"
        )


def _validate_tabular_binding(
    binding: Any, label: str, *, portable_dtypes: bool = False
) -> pd.DataFrame:
    if not isinstance(binding, dict):
        raise HPETranscriptCrosswalkError(f"{label} binding is malformed")
    required = {"path", "sha256", "columns", "parsed_dtypes", "n_rows"}
    if set(binding) != required:
        raise HPETranscriptCrosswalkError(f"{label} binding fields are malformed")
    path = Path(str(binding["path"]))
    frame = _load_csv(path, label)
    if sha256_file(path) != binding["sha256"]:
        raise HPETranscriptCrosswalkError(f"{label} SHA-256 changed")
    actual_dtypes = [str(dtype) for dtype in frame.dtypes]
    dtype_matches = (
        _portable_dtypes_match(binding["parsed_dtypes"], actual_dtypes)
        if portable_dtypes
        else actual_dtypes == binding["parsed_dtypes"]
    )
    if (
        list(frame.columns) != binding["columns"]
        or not dtype_matches
        or len(frame) != binding["n_rows"]
    ):
        raise HPETranscriptCrosswalkError(f"{label} schema or row count changed")
    return frame


def _validate_plain_binding(binding: Any, label: str) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise HPETranscriptCrosswalkError(f"{label} binding is malformed")
    path = Path(str(binding["path"]))
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise HPETranscriptCrosswalkError(f"{label} source changed")
    return path


def _validate_manifest_structure(manifest: dict[str, Any]) -> None:
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise HPETranscriptCrosswalkError("not an HPE/Transcript crosswalk artifact")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise HPETranscriptCrosswalkError(
            "unsupported HPE/Transcript crosswalk format version"
        )
    expected = manifest.get("expected_hpe_persons")
    if (
        not isinstance(expected, list)
        or not expected
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in expected
        )
        or expected != sorted(set(expected))
    ):
        raise HPETranscriptCrosswalkError("manifest expected_hpe_persons is malformed")
    decisions = manifest.get("sample_decisions")
    if not isinstance(decisions, dict):
        raise HPETranscriptCrosswalkError("manifest sample_decisions is malformed")
    for sample_id, decision in decisions.items():
        if not isinstance(sample_id, str) or not isinstance(decision, dict):
            raise HPETranscriptCrosswalkError("manifest sample decision is malformed")
        status = decision.get("status")
        label = decision.get("label")
        if status not in {"unreviewed", "confirmed", "uncertain"}:
            raise HPETranscriptCrosswalkError("manifest has an invalid sample status")
        if status == "confirmed" and label not in _CANONICAL_LABELS:
            raise HPETranscriptCrosswalkError(
                "confirmed sample decisions require an A/B/C/R label"
            )
        if status != "confirmed" and label is not None:
            raise HPETranscriptCrosswalkError(
                "only confirmed sample decisions may carry a label"
            )


def _iter_sample_sources(samples: pd.DataFrame) -> Iterable[tuple[str, str, dict[str, Any]]]:
    for _, row in samples.iterrows():
        sample_id = str(row["sample_id"])
        for side in ("hpe", "gaze"):
            raw = row[f"{side}_image_source_json"]
            try:
                source = json.loads(str(raw))
            except json.JSONDecodeError as exc:
                raise HPETranscriptCrosswalkError(
                    f"{sample_id} {side} image source is invalid JSON"
                ) from exc
            if not isinstance(source, dict) or not isinstance(source.get("kind"), str):
                raise HPETranscriptCrosswalkError(
                    f"{sample_id} {side} image source is malformed"
                )
            yield sample_id, side, source


def _validate_review_sources(
    manifest: dict[str, Any], samples: pd.DataFrame, *, require_video_binding: bool
) -> None:
    expected_videos: list[dict[str, Any]] = []
    for sample_id, side, source in _iter_sample_sources(samples):
        kind = source["kind"]
        if kind == "same_as_hpe":
            if side != "gaze" or set(source) != {"kind"}:
                raise HPETranscriptCrosswalkError(
                    f"{sample_id} has a malformed same_as_hpe source"
                )
            continue
        if kind == "still":
            path_value = source.get("path")
            digest = source.get("sha256")
            try:
                size_bytes = int(source["size_bytes"])
                frame_index = int(source["frame_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HPETranscriptCrosswalkError(
                    f"{sample_id} {side} still binding is malformed"
                ) from exc
            path = Path(path_value) if isinstance(path_value, str) else Path("")
            if (
                frame_index < 0
                or not path.is_file()
                or path.stat().st_size != size_bytes
                or not isinstance(digest, str)
                or sha256_file(path) != digest
            ):
                raise HPETranscriptCrosswalkError(
                    f"{sample_id} {side} review still changed"
                )
            continue
        if kind != "video":
            raise HPETranscriptCrosswalkError(
                f"{sample_id} {side} has unsupported image-source kind {kind!r}"
            )
        identity = source.get("video_identity")
        required = {"basename", "camera_dir", "fps", "frame_count", "width", "height"}
        if not isinstance(identity, dict) or not required.issubset(identity):
            raise HPETranscriptCrosswalkError(
                f"{sample_id} {side} video identity is incomplete"
            )
        try:
            values = {
                "fps": float(identity["fps"]),
                "frame_count": int(identity["frame_count"]),
                "width": int(identity["width"]),
                "height": int(identity["height"]),
            }
            frame_index = int(source["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HPETranscriptCrosswalkError(
                f"{sample_id} {side} video identity is malformed"
            ) from exc
        if (
            not math.isfinite(values["fps"])
            or values["fps"] <= 0
            or values["frame_count"] <= 0
            or values["width"] <= 0
            or values["height"] <= 0
            or frame_index < 0
            or frame_index >= values["frame_count"]
        ):
            raise HPETranscriptCrosswalkError(
                f"{sample_id} {side} video frame/identity is invalid"
            )
        expected_videos.append(identity)

    recorded = manifest.get("review_video_identities")
    if not expected_videos:
        if recorded not in (None, {}):
            raise HPETranscriptCrosswalkError(
                "review video bindings exist but no sample requires a raw video"
            )
        return
    if not require_video_binding and recorded is None:
        return
    if not isinstance(recorded, dict) or not recorded:
        raise HPETranscriptCrosswalkError(
            "raw-video review sources require bound preflight identities"
        )
    actual_values = list(recorded.values())
    for expected in expected_videos:
        matches = [actual for actual in actual_values if _video_identity_matches(expected, actual)]
        if len(matches) != 1:
            raise HPETranscriptCrosswalkError(
                "bound review video does not uniquely match a sample source"
            )


def _video_identity_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    path = actual.get("path")
    if not isinstance(path, str):
        return False
    if Path(path).name != str(expected.get("basename")):
        return False
    if Path(path).parent.name != str(expected.get("camera_dir")):
        return False
    try:
        if not math.isclose(
            float(actual["fps"]), float(expected["fps"]), rel_tol=0.0, abs_tol=1e-3
        ):
            return False
        for key in ("frame_count", "width", "height"):
            if int(actual[key]) != int(expected[key]):
                return False
        if "size_bytes" in expected and int(actual["size_bytes"]) != int(
            expected["size_bytes"]
        ):
            return False
        if "sparse_sha256" in expected and actual.get("sparse_sha256") != expected.get(
            "sparse_sha256"
        ):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return isinstance(actual.get("sparse_sha256"), str)


def _windows_file_attributes(path: Path, timeout_s: float = 3.0) -> int:
    result: list[int] = []

    def query() -> None:
        result.append(int(ctypes.windll.kernel32.GetFileAttributesW(str(path))))

    worker = threading.Thread(target=query, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise HPETranscriptCrosswalkError(
            "timed out querying a review-video cloud placeholder; make it local first"
        )
    if not result:
        raise HPETranscriptCrosswalkError(
            f"could not query review-video attributes: {path}"
        )
    return result[0]


def _validate_local_review_video_without_opening(path: Path) -> None:
    """Reject an Offline placeholder before stat/open can hydrate it."""

    if os.name == "nt":
        attributes = _windows_file_attributes(path)
        if attributes in {-1, 0xFFFFFFFF}:
            raise HPETranscriptCrosswalkError(f"bound review video does not exist: {path}")
        if attributes & getattr(stat, "FILE_ATTRIBUTE_DIRECTORY", 0x10):
            raise HPETranscriptCrosswalkError(f"bound review video is a directory: {path}")
        if attributes & _CLOUD_RECALL_FLAGS:
            raise HPETranscriptCrosswalkError(
                "bound review video is an Offline cloud placeholder; "
                "make it local before resume/finalize"
            )
        return
    if not path.is_file():
        raise HPETranscriptCrosswalkError(f"bound review video does not exist: {path}")


def _sparse_video_fingerprint(
    path: Path, chunk_size: int = _SPARSE_VIDEO_CHUNK_SIZE
) -> str:
    size = path.stat().st_size
    offsets = sorted(
        {0, max(0, size // 2 - chunk_size // 2), max(0, size - chunk_size)}
    )
    digest = hashlib.sha256()
    digest.update(f"size={size};chunk={chunk_size};".encode())
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            payload = handle.read(chunk_size)
            digest.update(f"offset={offset};length={len(payload)};".encode())
            digest.update(payload)
    return digest.hexdigest()


def _verify_bound_review_video_bytes(
    manifest: Mapping[str, Any], samples: pd.DataFrame
) -> None:
    expected_videos = [
        source["video_identity"]
        for _, _, source in _iter_sample_sources(samples)
        if source["kind"] == "video"
    ]
    recorded = manifest.get("review_video_identities")
    if not expected_videos:
        if recorded not in (None, {}):
            raise HPETranscriptCrosswalkError(
                "review video bindings exist but no sample requires a raw video"
            )
        return
    if not isinstance(recorded, dict) or not recorded:
        raise HPETranscriptCrosswalkError(
            "raw-video review sources require bound preflight identities"
        )
    unique_expected = {
        (
            str(value.get("basename")),
            str(value.get("camera_dir")),
            float(value.get("fps")),
            int(value.get("frame_count")),
            int(value.get("width")),
            int(value.get("height")),
            int(value.get("size_bytes", -1)),
            str(value.get("sparse_sha256", "")),
        )
        for value in expected_videos
    }
    if len(recorded) != len(unique_expected):
        raise HPETranscriptCrosswalkError(
            "review video bindings do not exactly cover unique sample sources"
        )
    for key, raw in recorded.items():
        if not isinstance(key, str) or not isinstance(raw, dict):
            raise HPETranscriptCrosswalkError("review video binding is malformed")
        path_value = raw.get("path")
        if not isinstance(path_value, str):
            raise HPETranscriptCrosswalkError("review video binding lacks a path")
        path = Path(path_value).resolve()
        if str(path) != str(Path(key).resolve()):
            raise HPETranscriptCrosswalkError(
                "review video binding key differs from its path"
            )
        _validate_local_review_video_without_opening(path)
        try:
            expected_size = int(raw["size_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HPETranscriptCrosswalkError(
                "review video size binding is malformed"
            ) from exc
        if path.stat().st_size != expected_size:
            raise HPETranscriptCrosswalkError(
                "bound review video size changed since human preflight"
            )
        if raw.get("fingerprint_algorithm") != _SPARSE_VIDEO_ALGORITHM:
            raise HPETranscriptCrosswalkError(
                "review video fingerprint algorithm is missing or unsupported"
            )
        digest = raw.get("sparse_sha256")
        if not isinstance(digest, str) or _sparse_video_fingerprint(path) != digest:
            raise HPETranscriptCrosswalkError(
                "bound review video sparse fingerprint changed since human preflight"
            )
        if not any(_video_identity_matches(expected, raw) for expected in expected_videos):
            raise HPETranscriptCrosswalkError(
                "bound review video no longer matches a sample source identity"
            )


def _replay_nonledger_sources(
    manifest: dict[str, Any],
    accepted: AcceptedIdentityWindow,
    *,
    portable_dtypes: bool = False,
) -> None:
    """Replay HPE/Gaze authority using a separately trusted accepted window."""

    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise HPETranscriptCrosswalkError("manifest sources are malformed")
    scene = str(manifest.get("scene"))
    recorded_window = manifest.get("accepted_window")
    if not isinstance(recorded_window, dict):
        raise HPETranscriptCrosswalkError("manifest accepted_window is malformed")
    for key, value in asdict(accepted).items():
        if recorded_window.get(key) != value:
            raise HPETranscriptCrosswalkError(
                "live ledger scene/camera/window differs from the crosswalk"
            )

    hpe_bindings = sources.get("hpe")
    if not isinstance(hpe_bindings, dict):
        raise HPETranscriptCrosswalkError("manifest HPE source bindings are malformed")
    for key in (
        "mapping",
        "identity_complete",
        "persons",
        "validation_complete",
    ):
        _validate_plain_binding(hpe_bindings.get(key), f"HPE {key}")
    for key in (
        "participant_map",
        "identity_world_persons",
        "delivery_world_persons",
    ):
        _validate_tabular_binding(
            hpe_bindings.get(key), f"HPE {key}", portable_dtypes=portable_dtypes
        )
    relabeled = sources.get("relabeled_cameras")
    if not isinstance(relabeled, dict) or not relabeled:
        raise HPETranscriptCrosswalkError("manifest relabeled camera bindings are malformed")
    validated_relabeled: dict[str, pd.DataFrame] = {}
    for camera, binding in relabeled.items():
        if (
            not isinstance(camera, str)
            or not re.fullmatch(r"Cam_[1-9][0-9]*", camera)
        ):
            raise HPETranscriptCrosswalkError(
                "manifest relabeled camera name is malformed"
            )
        validated_relabeled[camera] = _validate_tabular_binding(
            binding,
            f"HPE relabeled {camera}",
            portable_dtypes=portable_dtypes,
        )
    run_metadata = sources.get("hpe_run_metadata")
    if not isinstance(run_metadata, dict) or set(run_metadata) != set(relabeled):
        raise HPETranscriptCrosswalkError("manifest HPE run-metadata bindings are malformed")
    for camera, binding in run_metadata.items():
        _validate_plain_binding(binding, f"HPE {camera} run metadata")

    gaze_bindings = sources.get("gaze")
    if not isinstance(gaze_bindings, dict):
        raise HPETranscriptCrosswalkError("manifest Gaze bindings are malformed")
    gaze_manifest_path = _validate_plain_binding(
        gaze_bindings.get("manifest"), "Gaze identity manifest"
    )
    _validate_tabular_binding(
        gaze_bindings.get("tracklet_observations"),
        "Gaze tracklet observations",
        portable_dtypes=portable_dtypes,
    )
    try:
        gaze_manifest = load_annotation_manifest(gaze_manifest_path)
        observations_path = Path(
            str(gaze_bindings["tracklet_observations"]["path"])
        )
        validated_observations = (
            validate_tracklet_observations_portable(observations_path, gaze_manifest)
            if portable_dtypes
            else validate_tracklet_observations(observations_path, gaze_manifest)
        )
        bridge, conflicts = build_identity_bridge(validated_observations, gaze_manifest)
    except IdentityAnnotationError as exc:
        raise HPETranscriptCrosswalkError(
            f"Gaze identity artifact failed canonical replay: {exc}"
        ) from exc
    if not conflicts.empty:
        raise HPETranscriptCrosswalkError("Gaze canonical conflicts appeared on replay")
    if any(
        isinstance(item, dict) and item.get("status") == "unreviewed"
        for item in gaze_manifest.get("annotations", [])
    ):
        raise HPETranscriptCrosswalkError("Gaze unreviewed tracklets appeared on replay")
    confirmed = bridge.loc[
        (bridge["annotation_status"] == "confirmed")
        & bridge["participant"].isin(_CANONICAL_LABELS)
    ]
    if set(confirmed["participant"].astype(str)) != _CANONICAL_LABELS:
        raise HPETranscriptCrosswalkError("Gaze confirmed A/B/C/R reference changed")

    # Re-run the authority chain and require the recorded bindings to be the
    # canonical files owned by that chain.  A byte-identical copy elsewhere is
    # not provenance for this delivery.
    hpe = _validate_hpe_authority(
        scene=scene,
        accepted=accepted,
        delivery_dir=Path(str(sources.get("delivery_dir"))),
        hpe_repo_root=Path(str(sources.get("hpe_repo_root"))),
    )
    gaze = _validate_gaze_reference(
        scene=scene,
        accepted=accepted,
        identity_dir=Path(str(sources.get("gaze_identity_dir"))),
        portable_dtypes=portable_dtypes,
    )
    declared_directories = {
        "HPE repository": (sources.get("hpe_repo_root"), hpe["hpe_repo_root"]),
        "HPE delivery": (
            sources.get("delivery_dir"),
            Path(hpe["bindings"]["mapping"]["path"]).parent,
        ),
        "Gaze identity": (
            sources.get("gaze_identity_dir"),
            Path(gaze["bindings"]["manifest"]["path"]).parent,
        ),
    }
    for label, (recorded, expected) in declared_directories.items():
        if (
            not isinstance(recorded, str)
            or not recorded.strip()
            or Path(recorded).resolve() != Path(expected).resolve()
        ):
            raise HPETranscriptCrosswalkError(
                f"manifest {label} directory is disconnected from canonical authority"
            )
    _require_canonical_authority_bindings(
        recorded_hpe=hpe_bindings,
        canonical_hpe=hpe["bindings"],
        recorded_gaze=gaze_bindings,
        canonical_gaze=gaze["bindings"],
        portable_dtypes=portable_dtypes,
    )
    for camera, binding in relabeled.items():
        expected_relabeled = hpe["identity_dir"] / f"relabeled_{camera}.csv"
        if Path(str(binding["path"])).resolve() != expected_relabeled.resolve():
            raise HPETranscriptCrosswalkError(
                f"manifest HPE relabeled {camera} is outside canonical identity authority"
            )
        canonical_relabeled = _binding(
            expected_relabeled,
            validated_relabeled[camera],
        )
        if portable_dtypes:
            relabeled_matches = _portable_tabular_bindings_equal(
                binding, canonical_relabeled
            )
        else:
            relabeled_matches = binding == canonical_relabeled
        if not relabeled_matches:
            raise HPETranscriptCrosswalkError(
                f"manifest HPE relabeled {camera} differs from canonical identity authority"
            )
        _, canonical_run_binding = _camera_review_catalog(hpe, camera)
        if run_metadata[camera] != canonical_run_binding:
            raise HPETranscriptCrosswalkError(
                f"manifest HPE {camera} run metadata differs from canonical validation"
            )


def _replay_sources(manifest: dict[str, Any]) -> None:
    """Strict review/re-audit replay, including the mutable live ledger bytes."""

    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise HPETranscriptCrosswalkError("manifest sources are malformed")
    ledger_path = _validate_plain_binding(sources.get("ledger"), "identity ledger")
    scene = str(manifest.get("scene"))
    accepted = parse_accepted_identity_ledger(ledger_path).get(scene)
    if accepted is None:
        raise HPETranscriptCrosswalkError("scene is no longer accepted in the live ledger")
    _replay_nonledger_sources(manifest, accepted)


def _artifact_paths(
    path: str | Path, manifest_path: str | Path | None = None
) -> tuple[Path, Path]:
    supplied = Path(path).resolve()
    if manifest_path is not None:
        return supplied, Path(manifest_path).resolve()
    if supplied.is_dir():
        return supplied / SAMPLES_NAME, supplied / MANIFEST_NAME
    if supplied.name == SAMPLES_NAME:
        return supplied, supplied.parent / MANIFEST_NAME
    return supplied.parent / SAMPLES_NAME, supplied


def load_crosswalk_artifact(
    path: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load a crosswalk and replay every bound source before returning it."""

    manifest, samples = load_crosswalk_review_state(
        path, manifest_path, verify_bound_video_bytes=True
    )
    _replay_sources(manifest)
    return manifest, samples


def load_crosswalk_review_state(
    path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    verify_bound_video_bytes: bool = False,
    portable_dtypes: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load immutable samples plus mutable decisions without upstream replay.

    This is the trusted-code UI refresh path.  Entry/resume and finalisation must
    still use :func:`load_crosswalk_artifact`, which performs the full HPE/Gaze
    authority replay.
    """

    samples_path, resolved_manifest_path = _artifact_paths(path, manifest_path)
    manifest = _load_json_object(resolved_manifest_path, "crosswalk manifest")
    _validate_manifest_structure(manifest)
    samples = _validate_tabular_binding(
        manifest.get("review_samples"),
        "crosswalk review samples",
        portable_dtypes=portable_dtypes,
    )
    if tuple(samples.columns) != _SAMPLE_COLUMNS:
        raise HPETranscriptCrosswalkError("crosswalk review-sample schema is unsupported")
    sample_ids = samples["sample_id"].astype(str)
    if sample_ids.duplicated().any() or set(sample_ids) != set(manifest["sample_decisions"]):
        raise HPETranscriptCrosswalkError(
            "crosswalk decisions do not exactly cover review samples"
        )
    expected = set(manifest["expected_hpe_persons"])
    sample_people = set(
        _coerce_integer_series(
            samples["hpe_person"], label="review sample hpe_person", allow_zero=False
        ).tolist()
    )
    if sample_people != expected:
        raise HPETranscriptCrosswalkError(
            "review samples do not exactly cover expected HPE persons"
        )
    window = manifest["accepted_window"]
    logical_frames = _coerce_integer_series(
        samples["logical_frame"], label="review sample logical_frame", allow_zero=True
    )
    start = int(window["logical_start_inclusive"])
    end = int(window["logical_end_exclusive"])
    if not (logical_frames.ge(start) & logical_frames.lt(end)).all():
        raise HPETranscriptCrosswalkError(
            "review sample escaped the strict half-open accepted window"
        )
    _validate_review_sources(
        manifest,
        samples,
        require_video_binding=any(
            decision.get("status") != "unreviewed"
            for decision in manifest["sample_decisions"].values()
        )
        or manifest.get("state") == "finalized",
    )
    if verify_bound_video_bytes and manifest.get("review_video_identities") is not None:
        _verify_bound_review_video_bytes(manifest, samples)
    final = manifest.get("final_mapping")
    conflict = manifest.get("conflict_report")
    if manifest.get("state") == "finalized":
        _validate_tabular_binding(
            final,
            "final HPE/Transcript crosswalk",
            portable_dtypes=portable_dtypes,
        )
        conflict_frame = _validate_tabular_binding(
            conflict,
            "crosswalk conflict report",
            portable_dtypes=portable_dtypes,
        )
        if not conflict_frame.empty:
            raise HPETranscriptCrosswalkError(
                "finalized crosswalk has a non-empty conflict report"
            )
    return manifest, samples


def bind_review_video_identities(
    path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind UI-preflighted raw videos before the first human decision.

    The UI computes sparse content fingerprints without allowing implicit cloud
    hydration.  Rebinding a review revision to different bytes is forbidden.
    """

    manifest, samples = load_crosswalk_review_state(
        path, manifest_path, verify_bound_video_bytes=True
    )
    _, resolved_manifest_path = _artifact_paths(path, manifest_path)
    if manifest.get("state") == "finalized":
        raise HPETranscriptCrosswalkError("a finalized crosswalk is immutable")
    normalised: dict[str, dict[str, Any]] = {}
    required_fields = {
        "path",
        "fps",
        "frame_count",
        "width",
        "height",
        "size_bytes",
        "fingerprint_algorithm",
        "sparse_sha256",
    }
    for key, raw in bindings.items():
        if not isinstance(key, str) or not isinstance(raw, Mapping):
            raise HPETranscriptCrosswalkError("review video binding is malformed")
        value = dict(raw)
        if not required_fields.issubset(value):
            raise HPETranscriptCrosswalkError(
                "review video binding lacks metadata or sparse SHA-256"
            )
        if value["fingerprint_algorithm"] != _SPARSE_VIDEO_ALGORITHM:
            raise HPETranscriptCrosswalkError(
                "review video fingerprint algorithm is unsupported"
            )
        resolved = str(Path(str(value["path"])).resolve())
        if str(Path(key).resolve()) != resolved:
            raise HPETranscriptCrosswalkError(
                "review video binding key differs from its recorded path"
            )
        value["path"] = resolved
        if resolved in normalised:
            raise HPETranscriptCrosswalkError("review video binding repeats a path")
        normalised[resolved] = value
    probe = dict(manifest)
    probe["review_video_identities"] = normalised
    _validate_review_sources(probe, samples, require_video_binding=True)
    expected_videos = [
        source["video_identity"]
        for _, _, source in _iter_sample_sources(samples)
        if source["kind"] == "video"
    ]
    if not expected_videos:
        if normalised:
            raise HPETranscriptCrosswalkError(
                "review video preflight contains an unrelated source"
            )
        # Still-only reviews need no mutable video binding or audit event.
        return
    for actual in normalised.values():
        if not any(_video_identity_matches(expected, actual) for expected in expected_videos):
            raise HPETranscriptCrosswalkError(
                "review video preflight contains an unrelated source"
            )
    existing = manifest.get("review_video_identities")
    if existing is not None and existing != normalised:
        raise HPETranscriptCrosswalkError(
            "review video bytes differ from the existing revision binding"
        )
    if existing == normalised:
        return
    now = _utc_now()
    manifest["review_video_identities"] = normalised
    manifest["updated_at"] = now
    _atomic_write_json(resolved_manifest_path, manifest)
    _atomic_write_audit(
        resolved_manifest_path.parent / AUDIT_NAME,
        {
            "timestamp": now,
            "action": "bind_review_video_identities",
            "bindings": normalised,
        },
    )


def set_sample_decision(
    path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    sample_id: str,
    label: str | None = None,
    participant: str | None = None,
    status: str,
    note: str = "",
) -> dict[str, Any]:
    """Atomically record one independent sample decision and audit it."""

    if label is not None and participant is not None and label != participant:
        raise HPETranscriptCrosswalkError("label and participant aliases disagree")
    label = participant if participant is not None else label
    manifest, samples = load_crosswalk_review_state(path, manifest_path)
    _validate_review_sources(manifest, samples, require_video_binding=True)
    _, resolved_manifest_path = _artifact_paths(path, manifest_path)
    if manifest.get("state") == "finalized":
        raise HPETranscriptCrosswalkError("a finalized crosswalk is immutable")
    if sample_id not in manifest["sample_decisions"]:
        raise HPETranscriptCrosswalkError(f"unknown review sample: {sample_id}")
    if status not in {"unreviewed", "confirmed", "uncertain"}:
        raise HPETranscriptCrosswalkError(f"unsupported sample status: {status!r}")
    if status == "confirmed" and label not in _CANONICAL_LABELS:
        raise HPETranscriptCrosswalkError(
            "confirmed sample decisions require an A/B/C/R label"
        )
    if status != "confirmed" and label is not None:
        raise HPETranscriptCrosswalkError(
            "only confirmed sample decisions may carry a label"
        )
    previous = dict(manifest["sample_decisions"][sample_id])
    current = {
        "status": status,
        "label": label,
        "note": str(note),
        "updated_at": _utc_now(),
    }
    manifest["sample_decisions"][sample_id] = current
    manifest["updated_at"] = current["updated_at"]
    _atomic_write_json(resolved_manifest_path, manifest)
    _atomic_write_audit(
        resolved_manifest_path.parent / AUDIT_NAME,
        {
            "timestamp": current["updated_at"],
            "action": "set_sample_decision",
            "sample_id": sample_id,
            "previous": previous,
            "current": current,
        },
    )
    return previous


def _derive_person_labels(
    manifest: Mapping[str, Any], samples: pd.DataFrame
) -> dict[int, str]:
    output: dict[int, str] = {}
    for person, group in samples.groupby("hpe_person", sort=True):
        decisions = [
            manifest["sample_decisions"][sample_id]
            for sample_id in group["sample_id"].astype(str)
        ]
        if any(decision["status"] != "confirmed" for decision in decisions):
            raise HPETranscriptCrosswalkError(
                f"HPE person {person} still has an unconfirmed review anchor"
            )
        labels = {str(decision["label"]) for decision in decisions}
        if len(labels) != 1:
            raise HPETranscriptCrosswalkError(
                f"HPE person {person} has inconsistent sample decisions"
            )
        output[int(person)] = labels.pop()
    return output


def _same_frame_label_conflicts(
    world: pd.DataFrame,
    *,
    labels: Mapping[int, str],
    logical_window: tuple[int, int],
) -> pd.DataFrame:
    start, end = logical_window
    scoped = world.loc[
        world["frame_index"].ge(start)
        & world["frame_index"].lt(end)
        & world["person"].isin(labels)
    ].loc[:, ["frame_index", "person"]].drop_duplicates()
    scoped["participant"] = scoped["person"].map(labels)
    mask = scoped.duplicated(["frame_index", "participant"], keep=False)
    return scoped.loc[mask].sort_values(
        ["frame_index", "participant", "person"], kind="stable"
    )


def finalize_crosswalk(
    path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    upstream_mismatch_acknowledgement: str | None = None,
) -> Path:
    """Validate all decisions and export the confirmed crosswalk mapping."""

    manifest, samples = load_crosswalk_artifact(path, manifest_path)
    _validate_review_sources(manifest, samples, require_video_binding=True)
    _, resolved_manifest_path = _artifact_paths(path, manifest_path)
    if manifest.get("state") == "finalized":
        final = manifest.get("final_mapping")
        _validate_tabular_binding(final, "final HPE/Transcript crosswalk")
        return Path(str(final["path"]))
    labels = _derive_person_labels(manifest, samples)
    expected = set(manifest["expected_hpe_persons"])
    if set(labels) != expected:
        raise HPETranscriptCrosswalkError(
            "final decisions do not exactly cover expected HPE persons"
        )
    if set(labels.values()) != _CANONICAL_LABELS:
        raise HPETranscriptCrosswalkError(
            "final crosswalk labels must contain exactly A/B/C/R"
        )
    world_binding = manifest["sources"]["hpe"]["delivery_world_persons"]
    world = _validate_tabular_binding(world_binding, "delivery world_persons")
    world = world.copy()
    world["frame_index"] = _coerce_integer_series(
        world["frame_index"], label="delivery world frame_index", allow_zero=True
    )
    world["person"] = _coerce_integer_series(
        world["person"], label="delivery final person", allow_zero=False
    )
    window = manifest["accepted_window"]
    logical_window = (
        int(window["logical_start_inclusive"]),
        int(window["logical_end_exclusive"]),
    )
    conflicts = _same_frame_label_conflicts(
        world, labels=labels, logical_window=logical_window
    )
    conflict_output = conflicts.rename(columns={"person": "hpe_person"}).loc[
        :, ["frame_index", "participant", "hpe_person"]
    ]
    conflict_path = resolved_manifest_path.parent / CONFLICTS_NAME
    _atomic_write_csv(conflict_path, conflict_output)
    manifest["conflict_report"] = _binding(
        conflict_path, pd.read_csv(conflict_path)
    )
    if not conflicts.empty:
        first = conflicts.iloc[0]
        now = _utc_now()
        manifest["updated_at"] = now
        _atomic_write_json(resolved_manifest_path, manifest)
        _atomic_write_audit(
            resolved_manifest_path.parent / AUDIT_NAME,
            {
                "timestamp": now,
                "action": "finalize_blocked_same_frame_conflict",
                "n_conflict_rows": int(len(conflicts)),
                "conflict_report_sha256": manifest["conflict_report"]["sha256"],
            },
        )
        raise HPETranscriptCrosswalkError(
            "two HPE persons assigned the same canonical label coexist at frame "
            f"{int(first['frame_index'])}"
        )

    proposal = {
        int(person): str(label)
        for person, label in manifest["upstream_proposal"].items()
        if int(person) in expected
    }
    mismatches = [
        {
            "hpe_person": person,
            "reviewed_label": labels[person],
            "upstream_label": proposal.get(person),
        }
        for person in sorted(expected)
        if proposal.get(person) != labels[person]
    ]
    acknowledgement = (
        upstream_mismatch_acknowledgement.strip()
        if isinstance(upstream_mismatch_acknowledgement, str)
        else ""
    )
    if mismatches and not acknowledgement:
        raise HPETranscriptCrosswalkError(
            "reviewed labels differ from the upstream proposal; "
            "explicit acknowledgement is required"
        )

    rows: list[dict[str, Any]] = []
    for person in sorted(expected):
        evidence = samples.loc[samples["hpe_person"] == person, "sample_id"].astype(str)
        rows.append(
            {
                "hpe_person": person,
                "participant": labels[person],
                "slot": LABEL_TO_SLOT[labels[person]],
                "review_status": "confirmed",
                "evidence_sample_ids": ";".join(evidence),
            }
        )
    mapping = pd.DataFrame.from_records(rows)
    final_path = resolved_manifest_path.parent / FINAL_MAPPING_NAME
    _atomic_write_csv(final_path, mapping)
    final_binding = _binding(final_path, pd.read_csv(final_path))
    now = _utc_now()
    manifest["state"] = "finalized"
    manifest["upstream_mismatch"] = {
        "differences": mismatches,
        "acknowledgement": acknowledgement if mismatches else None,
    }
    manifest["final_mapping"] = final_binding
    manifest["updated_at"] = now
    _atomic_write_json(resolved_manifest_path, manifest)
    _atomic_write_audit(
        resolved_manifest_path.parent / AUDIT_NAME,
        {
            "timestamp": now,
            "action": "finalize",
            "labels": {str(person): labels[person] for person in sorted(labels)},
            "upstream_mismatch": mismatches,
            "acknowledgement": acknowledgement if mismatches else None,
            "final_mapping_sha256": final_binding["sha256"],
        },
    )
    return final_path


def _canonical_repo_relative(root: Path, path: Path, label: str) -> str:
    root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise HPETranscriptCrosswalkError(
            f"{label} must remain inside the repository"
        ) from exc
    if not relative.parts:
        raise HPETranscriptCrosswalkError(f"{label} cannot be the repository root")
    return relative.as_posix()


def _resolve_inventory_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise HPETranscriptCrosswalkError(
            f"{label} must be a canonical repository-relative POSIX path"
        )
    candidate = Path(value)
    if candidate.is_absolute():
        raise HPETranscriptCrosswalkError(f"{label} must be repository-relative")
    resolved = (root.resolve() / candidate).resolve()
    canonical = _canonical_repo_relative(root, resolved, label)
    if canonical != value:
        raise HPETranscriptCrosswalkError(f"{label} is not a canonical path")
    return resolved


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise HPETranscriptCrosswalkError(f"{label} SHA-256 is malformed")
    return value


def _accepted_snapshot(value: AcceptedIdentityWindow) -> dict[str, Any]:
    return asdict(value)


def _parse_accepted_snapshot(value: Any, *, scene: str) -> AcceptedIdentityWindow:
    required = {
        "scene",
        "camera",
        "local_start_inclusive",
        "local_end_exclusive",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise HPETranscriptCrosswalkError(
            f"sealed accepted window for {scene} is malformed"
        )
    if value.get("scene") != scene:
        raise HPETranscriptCrosswalkError(
            f"sealed accepted window scene differs from {scene}"
        )
    camera = value.get("camera")
    start = value.get("local_start_inclusive")
    end = value.get("local_end_exclusive")
    if (
        not isinstance(camera, str)
        or re.fullmatch(r"Cam_[1-9][0-9]*", camera) is None
        or isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise HPETranscriptCrosswalkError(
            f"sealed accepted window for {scene} has invalid values"
        )
    return AcceptedIdentityWindow(scene, camera, start, end)


def _expected_person_list(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(person, bool) or not isinstance(person, int) or person < 1
            for person in value
        )
        or value != sorted(set(value))
    ):
        raise HPETranscriptCrosswalkError(f"{label} is malformed")
    return value


def _validate_inventory_structure(
    inventory: dict[str, Any], *, repository_root: Path
) -> tuple[Path, dict[str, AcceptedIdentityWindow], dict[str, dict[str, Any]]]:
    top_fields = {
        "artifact_kind",
        "format_version",
        "trust_model",
        "artifact_root",
        "ledger_snapshot",
        "scenes",
    }
    if set(inventory) != top_fields:
        raise HPETranscriptCrosswalkError("sealed inventory fields are malformed")
    if inventory.get("artifact_kind") != SEALED_INVENTORY_KIND:
        raise HPETranscriptCrosswalkError("not a sealed crosswalk inventory")
    if inventory.get("format_version") != SEALED_INVENTORY_VERSION:
        raise HPETranscriptCrosswalkError("unsupported sealed inventory version")
    if inventory.get("trust_model") != "version_controlled_exact_hash_inventory":
        raise HPETranscriptCrosswalkError("sealed inventory trust model is unsupported")
    artifact_root = _resolve_inventory_path(
        repository_root, inventory.get("artifact_root"), "sealed artifact root"
    )

    ledger = inventory.get("ledger_snapshot")
    if not isinstance(ledger, dict) or set(ledger) != {
        "source_path",
        "sealed_path",
        "sha256",
        "accepted_windows",
    }:
        raise HPETranscriptCrosswalkError("sealed ledger snapshot is malformed")
    # The live source path is syntax-checked but deliberately not opened: that
    # status document is allowed to evolve after sealing.  A byte-for-byte copy
    # beside the inventory remains the immutable re-audit input.
    _resolve_inventory_path(
        repository_root, ledger.get("source_path"), "sealed ledger source path"
    )
    _resolve_inventory_path(
        repository_root, ledger.get("sealed_path"), "immutable ledger snapshot path"
    )
    _require_sha256(ledger.get("sha256"), "sealed ledger")
    raw_windows = ledger.get("accepted_windows")
    if not isinstance(raw_windows, dict) or not raw_windows:
        raise HPETranscriptCrosswalkError("sealed accepted-window snapshot is empty")
    accepted: dict[str, AcceptedIdentityWindow] = {}
    for scene, raw in raw_windows.items():
        if not isinstance(scene, str) or not scene:
            raise HPETranscriptCrosswalkError("sealed scene name is malformed")
        accepted[scene] = _parse_accepted_snapshot(raw, scene=scene)

    scenes = inventory.get("scenes")
    if not isinstance(scenes, dict) or set(scenes) != set(accepted):
        raise HPETranscriptCrosswalkError(
            "sealed scene set differs from the accepted-window snapshot"
        )
    for scene, raw in scenes.items():
        if not isinstance(raw, dict) or set(raw) != {
            "artifact_dir",
            "accepted_window",
            "expected_hpe_persons",
            "files",
        }:
            raise HPETranscriptCrosswalkError(
                f"sealed scene record for {scene} is malformed"
            )
        expected_dir = f"{inventory['artifact_root'].rstrip('/')}/{scene}"
        if raw.get("artifact_dir") != expected_dir:
            raise HPETranscriptCrosswalkError(
                f"sealed artifact path for {scene} is not canonical"
            )
        _resolve_inventory_path(
            repository_root, raw["artifact_dir"], f"sealed artifact {scene}"
        )
        window = raw.get("accepted_window")
        required_window = {
            "scene",
            "camera",
            "local_start_inclusive",
            "local_end_exclusive",
            "logical_start_inclusive",
            "logical_end_exclusive",
            "filter_rule",
        }
        if not isinstance(window, dict) or set(window) != required_window:
            raise HPETranscriptCrosswalkError(
                f"sealed full window for {scene} is malformed"
            )
        for key, expected_value in asdict(accepted[scene]).items():
            if window.get(key) != expected_value:
                raise HPETranscriptCrosswalkError(
                    f"sealed full window for {scene} differs from its snapshot"
                )
        logical_start = window.get("logical_start_inclusive")
        logical_end = window.get("logical_end_exclusive")
        if (
            isinstance(logical_start, bool)
            or not isinstance(logical_start, int)
            or isinstance(logical_end, bool)
            or not isinstance(logical_end, int)
            or logical_start < 0
            or logical_end <= logical_start
            or logical_end - logical_start
            != accepted[scene].local_end_exclusive
            - accepted[scene].local_start_inclusive
            or window.get("filter_rule") != "start <= frame_index < end"
        ):
            raise HPETranscriptCrosswalkError(
                f"sealed logical window for {scene} is invalid"
            )
        _expected_person_list(
            raw.get("expected_hpe_persons"),
            f"sealed expected HPE persons for {scene}",
        )
        files = raw.get("files")
        if not isinstance(files, dict) or set(files) != SEALED_ARTIFACT_FILES:
            raise HPETranscriptCrosswalkError(
                f"sealed file set for {scene} is not the exact five-file contract"
            )
        for filename, binding in files.items():
            if not isinstance(binding, dict) or set(binding) != {
                "size_bytes",
                "sha256",
            }:
                raise HPETranscriptCrosswalkError(
                    f"sealed binding for {scene}/{filename} is malformed"
                )
            size = binding.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise HPETranscriptCrosswalkError(
                    f"sealed size for {scene}/{filename} is malformed"
                )
            _require_sha256(binding.get("sha256"), f"sealed {scene}/{filename}")
    return artifact_root, accepted, scenes


def _validate_sealed_files(
    *, artifact_root: Path, scenes: Mapping[str, Mapping[str, Any]]
) -> None:
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise HPETranscriptCrosswalkError("sealed artifact root is missing or unsafe")
    entries = list(artifact_root.iterdir())
    if any(not entry.is_dir() or entry.is_symlink() for entry in entries):
        raise HPETranscriptCrosswalkError(
            "sealed artifact root contains a non-scene or symlink entry"
        )
    if {entry.name for entry in entries} != set(scenes):
        raise HPETranscriptCrosswalkError(
            "sealed artifact root has an extra or missing scene"
        )
    for scene, record in scenes.items():
        scene_dir = artifact_root / scene
        files = list(scene_dir.iterdir())
        if any(not path.is_file() or path.is_symlink() for path in files):
            raise HPETranscriptCrosswalkError(
                f"sealed scene {scene} contains a non-file or symlink entry"
            )
        if {path.name for path in files} != SEALED_ARTIFACT_FILES:
            raise HPETranscriptCrosswalkError(
                f"sealed scene {scene} has an extra or missing artifact file"
            )
        for filename, binding in record["files"].items():
            path = scene_dir / filename
            if (
                path.stat().st_size != binding["size_bytes"]
                or sha256_file(path) != binding["sha256"]
            ):
                raise HPETranscriptCrosswalkError(
                    f"sealed artifact bytes changed: {scene}/{filename}"
                )


def load_sealed_crosswalk_inventory(
    inventory_path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Load the version-controlled inventory and verify all ten five-file sets.

    The current live ledger is intentionally not opened.  Its accepted-window
    snapshot was checked when the inventory was generated and is now part of
    the version-controlled trust anchor.
    """

    root = Path(repository_root).resolve()
    supplied = Path(inventory_path)
    path = (
        supplied.resolve()
        if supplied.is_absolute()
        else (root / supplied).resolve()
    )
    _canonical_repo_relative(root, path, "sealed inventory")
    inventory = _load_json_object(path, "sealed crosswalk inventory")
    artifact_root, _, scenes = _validate_inventory_structure(
        inventory, repository_root=root
    )
    ledger = inventory["ledger_snapshot"]
    sealed_ledger = _resolve_inventory_path(
        root, ledger["sealed_path"], "immutable ledger snapshot path"
    )
    if (
        not sealed_ledger.is_file()
        or sealed_ledger.is_symlink()
        or sha256_file(sealed_ledger) != ledger["sha256"]
    ):
        raise HPETranscriptCrosswalkError(
            "immutable ledger snapshot is missing or its bytes changed"
        )
    parsed_snapshot = parse_accepted_identity_ledger(sealed_ledger)
    if {
        scene: _accepted_snapshot(window)
        for scene, window in sorted(parsed_snapshot.items())
    } != ledger["accepted_windows"]:
        raise HPETranscriptCrosswalkError(
            "immutable ledger bytes differ from the accepted-window snapshot"
        )
    _validate_sealed_files(artifact_root=artifact_root, scenes=scenes)
    return inventory


def _require_bound_artifact_path(
    binding: Any,
    *,
    expected_path: Path,
    label: str,
    portable_dtypes: bool = False,
) -> pd.DataFrame:
    if not isinstance(binding, dict) or Path(str(binding.get("path"))).resolve() != (
        expected_path.resolve()
    ):
        raise HPETranscriptCrosswalkError(f"{label} points outside its sealed scene")
    return _validate_tabular_binding(
        binding, label, portable_dtypes=portable_dtypes
    )


def _load_audit_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise HPETranscriptCrosswalkError("sealed crosswalk audit is missing")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise HPETranscriptCrosswalkError(
                f"sealed crosswalk audit has a blank line at {line_number}"
            )
        try:
            value = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant {token}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise HPETranscriptCrosswalkError(
                f"sealed crosswalk audit line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise HPETranscriptCrosswalkError(
                f"sealed crosswalk audit line {line_number} is not an object"
            )
        events.append(value)
    if not events:
        raise HPETranscriptCrosswalkError("sealed crosswalk audit is empty")
    return events


def _validate_decision_record(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "status",
        "label",
        "note",
        "updated_at",
    }:
        raise HPETranscriptCrosswalkError(f"{label} is malformed")
    status = value["status"]
    participant = value["label"]
    if status not in {"unreviewed", "confirmed", "uncertain"}:
        raise HPETranscriptCrosswalkError(f"{label} has an invalid status")
    if status == "confirmed" and participant not in _CANONICAL_LABELS:
        raise HPETranscriptCrosswalkError(f"{label} has an invalid confirmed label")
    if status != "confirmed" and participant is not None:
        raise HPETranscriptCrosswalkError(f"{label} carries an invalid label")
    if not isinstance(value["note"], str):
        raise HPETranscriptCrosswalkError(f"{label} note is malformed")
    updated = value["updated_at"]
    if updated is not None and (not isinstance(updated, str) or not updated):
        raise HPETranscriptCrosswalkError(f"{label} timestamp is malformed")
    return value


def _validate_final_audit(
    *,
    artifact_dir: Path,
    manifest: dict[str, Any],
    samples: pd.DataFrame,
    labels: Mapping[int, str],
    mismatches: list[dict[str, Any]],
    acknowledgement: str | None,
) -> None:
    events = _load_audit_events(artifact_dir / AUDIT_NAME)
    prepare = events[0]
    if set(prepare) != {
        "timestamp",
        "action",
        "scene",
        "expected_hpe_persons",
        "sample_ids",
    } or prepare.get("action") != "prepare":
        raise HPETranscriptCrosswalkError("sealed audit does not start with prepare")
    sample_ids = samples["sample_id"].astype(str).tolist()
    if (
        prepare.get("scene") != manifest["scene"]
        or prepare.get("expected_hpe_persons") != manifest["expected_hpe_persons"]
        or prepare.get("sample_ids") != sample_ids
    ):
        raise HPETranscriptCrosswalkError("sealed audit prepare event changed")
    replayed = {
        sample_id: {
            "status": "unreviewed",
            "label": None,
            "note": "",
            "updated_at": None,
        }
        for sample_id in sample_ids
    }
    recorded_video_bindings: dict[str, Any] | None = None
    final_events = 0
    for index, event in enumerate(events[1:], 1):
        action = event.get("action")
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp:
            raise HPETranscriptCrosswalkError("sealed audit timestamp is malformed")
        if action == "set_sample_decision":
            if set(event) != {
                "timestamp",
                "action",
                "sample_id",
                "previous",
                "current",
            }:
                raise HPETranscriptCrosswalkError("sealed decision audit event is malformed")
            sample_id = event.get("sample_id")
            if sample_id not in replayed:
                raise HPETranscriptCrosswalkError("sealed audit names an unknown sample")
            previous = _validate_decision_record(
                event.get("previous"), "sealed audit previous decision"
            )
            current = _validate_decision_record(
                event.get("current"), "sealed audit current decision"
            )
            if previous != replayed[sample_id] or current["updated_at"] != timestamp:
                raise HPETranscriptCrosswalkError("sealed decision audit chain is broken")
            replayed[sample_id] = dict(current)
        elif action == "bind_review_video_identities":
            if set(event) != {"timestamp", "action", "bindings"}:
                raise HPETranscriptCrosswalkError("sealed video audit event is malformed")
            if recorded_video_bindings is not None:
                raise HPETranscriptCrosswalkError("sealed audit repeats a video binding")
            bindings = event.get("bindings")
            if not isinstance(bindings, dict) or not bindings:
                raise HPETranscriptCrosswalkError("sealed video audit binding is malformed")
            recorded_video_bindings = bindings
        elif action == "finalize":
            final_events += 1
            if index != len(events) - 1 or set(event) != {
                "timestamp",
                "action",
                "labels",
                "upstream_mismatch",
                "acknowledgement",
                "final_mapping_sha256",
            }:
                raise HPETranscriptCrosswalkError(
                    "sealed audit finalize event is not the exact final event"
                )
            expected_labels = {
                str(person): labels[person] for person in sorted(labels)
            }
            if (
                event.get("labels") != expected_labels
                or event.get("upstream_mismatch") != mismatches
                or event.get("acknowledgement") != acknowledgement
                or event.get("final_mapping_sha256")
                != manifest["final_mapping"]["sha256"]
                or manifest.get("updated_at") != timestamp
            ):
                raise HPETranscriptCrosswalkError("sealed audit finalize event changed")
        else:
            raise HPETranscriptCrosswalkError(
                f"sealed audit contains unsupported action {action!r}"
            )
    if final_events != 1 or events[-1].get("action") != "finalize":
        raise HPETranscriptCrosswalkError("sealed audit is not closed by one finalize event")
    if replayed != manifest["sample_decisions"]:
        raise HPETranscriptCrosswalkError(
            "sealed audit decisions differ from the finalized manifest"
        )
    if recorded_video_bindings != manifest.get("review_video_identities"):
        if not (
            recorded_video_bindings is None
            and manifest.get("review_video_identities") in (None, {})
        ):
            raise HPETranscriptCrosswalkError(
                "sealed video audit differs from the finalized manifest"
            )


def _validate_finalized_crosswalk_semantics(
    *,
    artifact_dir: Path,
    manifest: dict[str, Any],
    samples: pd.DataFrame,
    accepted: AcceptedIdentityWindow,
    scene_record: Mapping[str, Any],
) -> pd.DataFrame:
    scene = accepted.scene
    if manifest.get("scene") != scene or manifest.get("state") != "finalized":
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} is not the matching finalized artifact"
        )
    recorded_window = manifest.get("accepted_window")
    if recorded_window != scene_record.get("accepted_window"):
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} window differs from the inventory"
        )
    for key, expected_value in asdict(accepted).items():
        if recorded_window.get(key) != expected_value:
            raise HPETranscriptCrosswalkError(
                f"sealed scene {scene} escaped its accepted window"
            )
    expected = _expected_person_list(
        manifest.get("expected_hpe_persons"), "manifest expected HPE persons"
    )
    if expected != scene_record.get("expected_hpe_persons"):
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} person universe differs from the inventory"
        )
    if Path(str(manifest["review_samples"].get("path"))).resolve() != (
        artifact_dir / SAMPLES_NAME
    ).resolve():
        raise HPETranscriptCrosswalkError(
            "sealed review-sample binding points outside its scene"
        )
    mapping = _require_bound_artifact_path(
        manifest.get("final_mapping"),
        expected_path=artifact_dir / FINAL_MAPPING_NAME,
        label="sealed final mapping",
        portable_dtypes=True,
    )
    conflicts = _require_bound_artifact_path(
        manifest.get("conflict_report"),
        expected_path=artifact_dir / CONFLICTS_NAME,
        label="sealed conflict report",
        portable_dtypes=True,
    )
    if tuple(mapping.columns) != (
        "hpe_person",
        "participant",
        "slot",
        "review_status",
        "evidence_sample_ids",
    ):
        raise HPETranscriptCrosswalkError("sealed final mapping schema changed")
    if tuple(conflicts.columns) != ("frame_index", "participant", "hpe_person"):
        raise HPETranscriptCrosswalkError("sealed conflict-report schema changed")
    if not conflicts.empty:
        raise HPETranscriptCrosswalkError("sealed conflict report is not empty")

    policy = manifest.get("sampling_policy")
    policy_fields = {
        "format_version",
        "samples_per_person",
        "phases",
        "selection",
        "camera_preference",
        "bbox_rule",
        "frame_scope",
        "reference_rule",
    }
    if not isinstance(policy, dict) or set(policy) != policy_fields:
        raise HPETranscriptCrosswalkError("sealed sampling policy is malformed")
    count = policy.get("samples_per_person")
    phases = policy.get("phases")
    if (
        policy.get("format_version") != SAMPLE_FORMAT_VERSION
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(phases, list)
        or len(phases) != count
        or any(not isinstance(phase, str) or not phase for phase in phases)
        or len(set(phases)) != len(phases)
    ):
        raise HPETranscriptCrosswalkError("sealed sampling policy values are invalid")

    sample_people = _coerce_integer_series(
        samples["hpe_person"], label="sealed sample hpe_person", allow_zero=False
    )
    logical_frames = _coerce_integer_series(
        samples["logical_frame"], label="sealed sample logical_frame", allow_zero=True
    )
    local_frames = _coerce_integer_series(
        samples["gaze_local_frame"],
        label="sealed sample gaze_local_frame",
        allow_zero=True,
    )
    logical_start = int(recorded_window["logical_start_inclusive"])
    logical_end = int(recorded_window["logical_end_exclusive"])
    local_start = accepted.local_start_inclusive
    local_end = accepted.local_end_exclusive
    if not (
        logical_frames.ge(logical_start) & logical_frames.lt(logical_end)
    ).all() or not (local_frames.ge(local_start) & local_frames.lt(local_end)).all():
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} has a sample outside its half-open window"
        )
    expected_offset = logical_start - local_start
    if not (logical_frames - local_frames == expected_offset).all():
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} has inconsistent local/logical sample frames"
        )
    if not samples["gaze_camera"].eq(accepted.camera).all():
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} changed its Gaze reference camera"
        )
    samples = samples.copy()
    samples["hpe_person"] = sample_people
    for person, group in samples.groupby("hpe_person", sort=True):
        if len(group) != count or group["phase"].astype(str).tolist() != phases:
            raise HPETranscriptCrosswalkError(
                f"sealed HPE person {person} does not have the frozen sample phases"
            )

    world_binding = manifest.get("sources", {}).get("hpe", {}).get(
        "delivery_world_persons"
    )
    world = _validate_tabular_binding(
        world_binding,
        "sealed delivery world_persons",
        portable_dtypes=True,
    )
    if not {"frame_index", "person"}.issubset(world.columns):
        raise HPETranscriptCrosswalkError(
            "sealed delivery world_persons lost frame_index/person"
        )
    world = world.copy()
    world["frame_index"] = _coerce_integer_series(
        world["frame_index"], label="sealed world frame_index", allow_zero=True
    )
    world["person"] = _coerce_integer_series(
        world["person"], label="sealed world final person", allow_zero=False
    )
    derived_expected = expected_hpe_persons(world, (logical_start, logical_end))
    if derived_expected != expected:
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} expected-person universe changed"
        )
    world_keys = set(
        zip(world["frame_index"].tolist(), world["person"].tolist(), strict=True)
    )
    if any(
        (int(frame), int(person)) not in world_keys
        for frame, person in zip(logical_frames, sample_people, strict=True)
    ):
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} has a sample without delivery-world presence"
        )

    for sample_id, decision in manifest["sample_decisions"].items():
        _validate_decision_record(decision, f"sealed decision {sample_id}")
    labels = _derive_person_labels(manifest, samples)
    if sorted(labels) != expected or set(labels.values()) != _CANONICAL_LABELS:
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} lacks the exact expected people or A/B/C/R labels"
        )
    derived_conflicts = _same_frame_label_conflicts(
        world, labels=labels, logical_window=(logical_start, logical_end)
    )
    if not derived_conflicts.empty:
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} has a same-frame canonical-label conflict"
        )

    mapping_people = _coerce_integer_series(
        mapping["hpe_person"], label="sealed mapping hpe_person", allow_zero=False
    )
    if mapping_people.tolist() != expected:
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} mapping does not exactly cover expected people"
        )
    for index, row in mapping.iterrows():
        person = int(mapping_people.iloc[index])
        evidence = samples.loc[
            samples["hpe_person"].eq(person), "sample_id"
        ].astype(str).tolist()
        if (
            row["participant"] != labels[person]
            or row["slot"] != LABEL_TO_SLOT[labels[person]]
            or row["review_status"] != "confirmed"
            or str(row["evidence_sample_ids"]).split(";") != evidence
        ):
            raise HPETranscriptCrosswalkError(
                f"sealed mapping row for HPE person {person} changed"
            )

    proposal = {
        int(person): str(label)
        for person, label in manifest.get("upstream_proposal", {}).items()
        if int(person) in expected
    }
    mismatches = [
        {
            "hpe_person": person,
            "reviewed_label": labels[person],
            "upstream_label": proposal.get(person),
        }
        for person in expected
        if proposal.get(person) != labels[person]
    ]
    mismatch_record = manifest.get("upstream_mismatch")
    if not isinstance(mismatch_record, dict) or set(mismatch_record) != {
        "differences",
        "acknowledgement",
    }:
        raise HPETranscriptCrosswalkError("sealed upstream mismatch record is malformed")
    acknowledgement = mismatch_record.get("acknowledgement")
    if mismatch_record.get("differences") != mismatches or (
        mismatches
        and (not isinstance(acknowledgement, str) or not acknowledgement.strip())
    ) or (not mismatches and acknowledgement is not None):
        raise HPETranscriptCrosswalkError("sealed upstream mismatch record changed")

    expected_video_identities = [
        source["video_identity"]
        for _, _, source in _iter_sample_sources(samples)
        if source["kind"] == "video"
    ]
    recorded_videos = manifest.get("review_video_identities")
    unique_expected_videos = {
        (
            str(value.get("basename")),
            str(value.get("camera_dir")),
            float(value.get("fps")),
            int(value.get("frame_count")),
            int(value.get("width")),
            int(value.get("height")),
            int(value.get("size_bytes", -1)),
            str(value.get("sparse_sha256", "")),
        )
        for value in expected_video_identities
    }
    if expected_video_identities and (
        not isinstance(recorded_videos, dict)
        or len(recorded_videos) != len(unique_expected_videos)
    ):
        raise HPETranscriptCrosswalkError(
            "sealed review-video bindings do not exactly cover sample sources"
        )

    _validate_final_audit(
        artifact_dir=artifact_dir,
        manifest=manifest,
        samples=samples,
        labels=labels,
        mismatches=mismatches,
        acknowledgement=acknowledgement,
    )
    return mapping


def _sealed_scene_record(
    *, repository_root: Path, artifact_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for filename in sorted(SEALED_ARTIFACT_FILES):
        path = artifact_dir / filename
        files[filename] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "artifact_dir": _canonical_repo_relative(
            repository_root, artifact_dir, "sealed scene artifact"
        ),
        "accepted_window": manifest["accepted_window"],
        "expected_hpe_persons": manifest["expected_hpe_persons"],
        "files": files,
    }


def build_sealed_crosswalk_inventory(
    *,
    repository_root: str | Path,
    artifact_root: str | Path = "data/annotations/hpe_transcript_crosswalk",
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    sealed_ledger_path: str | Path = DEFAULT_SEALED_LEDGER_PATH,
) -> dict[str, Any]:
    """Strictly re-audit every accepted final artifact and build a seal payload.

    This function requires raw review videos to be local because it deliberately
    uses :func:`load_crosswalk_artifact`.  Merely loading the returned sealed
    inventory later does not require those AVI files.
    """

    root = Path(repository_root).resolve()
    artifacts = (
        Path(artifact_root).resolve()
        if Path(artifact_root).is_absolute()
        else (root / artifact_root).resolve()
    )
    ledger = (
        Path(ledger_path).resolve()
        if Path(ledger_path).is_absolute()
        else (root / ledger_path).resolve()
    )
    sealed_ledger = (
        Path(sealed_ledger_path).resolve()
        if Path(sealed_ledger_path).is_absolute()
        else (root / sealed_ledger_path).resolve()
    )
    artifact_relative = _canonical_repo_relative(root, artifacts, "artifact root")
    ledger_relative = _canonical_repo_relative(root, ledger, "identity ledger")
    sealed_ledger_relative = _canonical_repo_relative(
        root, sealed_ledger, "immutable ledger snapshot"
    )
    if ledger == sealed_ledger:
        raise HPETranscriptCrosswalkError(
            "immutable ledger snapshot must differ from the mutable live ledger"
        )
    accepted = parse_accepted_identity_ledger(ledger)
    if not artifacts.is_dir() or artifacts.is_symlink():
        raise HPETranscriptCrosswalkError("crosswalk artifact root is missing or unsafe")
    entries = list(artifacts.iterdir())
    if any(not entry.is_dir() or entry.is_symlink() for entry in entries):
        raise HPETranscriptCrosswalkError(
            "crosswalk artifact root contains a non-scene or symlink entry"
        )
    if {entry.name for entry in entries} != set(accepted):
        raise HPETranscriptCrosswalkError(
            "crosswalk artifact scenes do not exactly match the accepted ledger"
        )

    inventory: dict[str, Any] = {
        "artifact_kind": SEALED_INVENTORY_KIND,
        "format_version": SEALED_INVENTORY_VERSION,
        "trust_model": "version_controlled_exact_hash_inventory",
        "artifact_root": artifact_relative,
        "ledger_snapshot": {
            "source_path": ledger_relative,
            "sealed_path": sealed_ledger_relative,
            "sha256": sha256_file(ledger),
            "accepted_windows": {
                scene: _accepted_snapshot(accepted[scene]) for scene in sorted(accepted)
            },
        },
        "scenes": {},
    }
    for scene in sorted(accepted):
        artifact_dir = artifacts / scene
        files = list(artifact_dir.iterdir())
        if (
            any(not path.is_file() or path.is_symlink() for path in files)
            or {path.name for path in files} != SEALED_ARTIFACT_FILES
        ):
            raise HPETranscriptCrosswalkError(
                f"scene {scene} is not the exact five-file artifact"
            )
        manifest, samples = load_crosswalk_artifact(artifact_dir)
        manifest_ledger = manifest.get("sources", {}).get("ledger")
        if (
            not isinstance(manifest_ledger, dict)
            or set(manifest_ledger) != {"path", "sha256"}
            or Path(str(manifest_ledger.get("path"))).resolve() != ledger
            or manifest_ledger.get("sha256") != inventory["ledger_snapshot"]["sha256"]
        ):
            raise HPETranscriptCrosswalkError(
                f"scene {scene} was not prepared from the ledger being sealed"
            )
        scene_record = _sealed_scene_record(
            repository_root=root, artifact_dir=artifact_dir, manifest=manifest
        )
        _validate_finalized_crosswalk_semantics(
            artifact_dir=artifact_dir,
            manifest=manifest,
            samples=samples,
            accepted=accepted[scene],
            scene_record=scene_record,
        )
        inventory["scenes"][scene] = scene_record
    _validate_inventory_structure(inventory, repository_root=root)
    _validate_sealed_files(artifact_root=artifacts, scenes=inventory["scenes"])
    return inventory


def write_sealed_crosswalk_inventory(
    inventory: Mapping[str, Any],
    output_path: str | Path,
    *,
    repository_root: str | Path,
    overwrite: bool = False,
) -> Path:
    """Atomically write a previously built inventory after structural validation."""

    root = Path(repository_root).resolve()
    output = Path(output_path).resolve()
    _canonical_repo_relative(root, output, "sealed inventory output")
    if output.exists() and not overwrite:
        raise HPETranscriptCrosswalkError(
            f"sealed inventory already exists (use explicit overwrite): {output}"
        )
    value = dict(inventory)
    artifact_root, _, scenes = _validate_inventory_structure(
        value, repository_root=root
    )
    ledger = value["ledger_snapshot"]
    sealed_ledger = _resolve_inventory_path(
        root, ledger["sealed_path"], "immutable ledger snapshot path"
    )
    if (
        not sealed_ledger.is_file()
        or sealed_ledger.is_symlink()
        or sha256_file(sealed_ledger) != ledger["sha256"]
    ):
        raise HPETranscriptCrosswalkError(
            "immutable ledger snapshot must be written before its inventory"
        )
    _validate_sealed_files(artifact_root=artifact_root, scenes=scenes)
    _atomic_write_json(output, value)
    return output


def write_sealed_crosswalk_bundle(
    inventory: Mapping[str, Any],
    output_path: str | Path,
    *,
    repository_root: str | Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Publish the immutable ledger and inventory as a rollback-safe pair.

    A filesystem cannot atomically replace two paths.  Both complete payloads
    are therefore staged first; if either publication fails, the exact previous
    bytes are restored (or both new paths are removed).  This prevents a failed
    overwrite from invalidating an already-published inventory.
    """

    root = Path(repository_root).resolve()
    value = dict(inventory)
    artifact_root, _, scenes = _validate_inventory_structure(
        value, repository_root=root
    )
    _validate_sealed_files(artifact_root=artifact_root, scenes=scenes)
    ledger = value["ledger_snapshot"]
    source = _resolve_inventory_path(
        root, ledger["source_path"], "sealed ledger source path"
    )
    sealed = _resolve_inventory_path(
        root, ledger["sealed_path"], "immutable ledger snapshot path"
    )
    output = Path(output_path).resolve()
    _canonical_repo_relative(root, output, "sealed inventory output")
    if output in {source, sealed}:
        raise HPETranscriptCrosswalkError(
            "sealed inventory, live ledger, and ledger snapshot paths must differ"
        )
    for path in (output, sealed):
        try:
            path.relative_to(artifact_root)
        except ValueError:
            pass
        else:
            raise HPETranscriptCrosswalkError(
                "sealed bundle outputs cannot live inside the exact artifact root"
            )
    if not source.is_file() or source.is_symlink():
        raise HPETranscriptCrosswalkError(
            "live ledger source is missing or unsafe"
        )
    for path in (sealed, output):
        if path.exists() and (not path.is_file() or path.is_symlink()):
            raise HPETranscriptCrosswalkError(
                f"sealed bundle output is not a regular file: {path}"
            )
        if path.exists() and not overwrite:
            raise HPETranscriptCrosswalkError(
                f"sealed bundle output already exists (use explicit overwrite): {path}"
            )
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != ledger["sha256"]:
        raise HPETranscriptCrosswalkError(
            "live ledger bytes changed after the inventory was built"
        )
    try:
        inventory_payload = (
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HPETranscriptCrosswalkError(
            "sealed inventory cannot be serialized as strict JSON"
        ) from exc

    sealed.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    try:
        for destination, suffix, content in (
            (sealed, ".md.tmp", payload),
            (output, ".json.tmp", inventory_payload),
        ):
            with tempfile.NamedTemporaryFile(
                "wb", suffix=suffix, dir=destination.parent, delete=False
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                staged.append(Path(handle.name))
        if hashlib.sha256(staged[0].read_bytes()).hexdigest() != ledger["sha256"]:
            raise HPETranscriptCrosswalkError(
                "staged immutable ledger snapshot differs from its binding"
            )
        staged_inventory = json.loads(staged[1].read_text(encoding="utf-8"))
        if staged_inventory != value:
            raise HPETranscriptCrosswalkError(
                "staged sealed inventory differs from the validated value"
            )

        previous = {
            sealed: sealed.read_bytes() if sealed.exists() else None,
            output: output.read_bytes() if output.exists() else None,
        }
        try:
            os.replace(staged[0], sealed)
            staged.pop(0)
            os.replace(staged[0], output)
            staged.pop(0)
            if (
                sha256_file(sealed) != ledger["sha256"]
                or json.loads(output.read_text(encoding="utf-8")) != value
            ):
                raise HPETranscriptCrosswalkError(
                    "sealed bundle post-publication verification failed"
                )
        except (OSError, json.JSONDecodeError, HPETranscriptCrosswalkError) as exc:
            rollback_errors: list[str] = []
            for destination, old_payload in previous.items():
                try:
                    if old_payload is None:
                        destination.unlink(missing_ok=True)
                    else:
                        destination.write_bytes(old_payload)
                except OSError as rollback_exc:
                    rollback_errors.append(f"{destination}: {rollback_exc}")
            if rollback_errors:
                raise HPETranscriptCrosswalkError(
                    "sealed bundle publication failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from exc
            raise HPETranscriptCrosswalkError(
                "sealed bundle publication failed; previous state was restored"
            ) from exc
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    return output, sealed


def _load_sealed_scene(
    *,
    scene: str,
    inventory: dict[str, Any],
    repository_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    artifact_root, accepted, scenes = _validate_inventory_structure(
        inventory, repository_root=repository_root
    )
    if scene not in scenes:
        raise HPETranscriptCrosswalkError(f"scene {scene!r} is not in the sealed inventory")
    artifact_dir = artifact_root / scene
    manifest, samples = load_crosswalk_review_state(
        artifact_dir,
        verify_bound_video_bytes=False,
        portable_dtypes=True,
    )
    source_ledger = manifest.get("sources", {}).get("ledger")
    ledger_snapshot = inventory["ledger_snapshot"]
    expected_ledger_path = _resolve_inventory_path(
        repository_root,
        ledger_snapshot["source_path"],
        "sealed ledger source path",
    )
    if (
        not isinstance(source_ledger, dict)
        or set(source_ledger) != {"path", "sha256"}
        or Path(str(source_ledger.get("path"))).resolve() != expected_ledger_path
        or source_ledger.get("sha256") != ledger_snapshot["sha256"]
    ):
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} was not prepared from the frozen ledger snapshot"
        )
    # This replay intentionally bypasses only the now-mutable ledger file and
    # raw AVI bytes.  All HPE/Gaze bindings and their semantic authority chains
    # remain mandatory.
    _replay_nonledger_sources(
        manifest, accepted[scene], portable_dtypes=True
    )
    mapping = _validate_finalized_crosswalk_semantics(
        artifact_dir=artifact_dir,
        manifest=manifest,
        samples=samples,
        accepted=accepted[scene],
        scene_record=scenes[scene],
    )
    return manifest, samples, mapping


def load_sealed_crosswalk(
    scene: str,
    inventory_path: str | Path = DEFAULT_SEALED_INVENTORY_PATH,
    *,
    repository_root: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Load one sealed final crosswalk without opening its raw review AVI."""

    root = Path(repository_root).resolve()
    supplied_inventory = Path(inventory_path)
    resolved_inventory = (
        supplied_inventory.resolve()
        if supplied_inventory.is_absolute()
        else (root / supplied_inventory).resolve()
    )
    inventory = load_sealed_crosswalk_inventory(
        resolved_inventory, repository_root=root
    )
    return _load_sealed_scene(
        scene=scene, inventory=inventory, repository_root=root
    )


def _load_bound_json_object(binding: Any, label: str) -> tuple[Path, dict[str, Any]]:
    """Read one hash-bound JSON object from exactly the bytes that are parsed."""

    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise HPETranscriptCrosswalkError(f"{label} binding is malformed")
    path_value = binding.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise HPETranscriptCrosswalkError(f"{label} path is malformed")
    _require_sha256(binding.get("sha256"), label)
    path = Path(path_value).resolve()
    if not path.is_file() or path.is_symlink():
        raise HPETranscriptCrosswalkError(f"{label} does not exist or is unsafe: {path}")
    try:
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HPETranscriptCrosswalkError(f"could not read {label}: {path}") from exc
    if digest != binding["sha256"]:
        raise HPETranscriptCrosswalkError(f"{label} bytes changed after sealing")
    if not isinstance(value, dict):
        raise HPETranscriptCrosswalkError(f"{label} must contain a JSON object")
    return path, value


def _finite_positive_json_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HPETranscriptCrosswalkError(f"{label} must be a JSON number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise HPETranscriptCrosswalkError(f"{label} must be finite and positive")
    return number


def _resolve_hpe_logical_timing_authority(
    *,
    manifest: Mapping[str, Any],
    samples: pd.DataFrame,
    validation_binding: Any,
) -> dict[str, Any]:
    """Resolve FPS from the HPE cameras that actually prove the logical axis.

    The accepted identity camera is a Gaze camera.  In the 05_15 deliveries it
    is Cam_1, which was deliberately excluded from production HPE pose output.
    Therefore it is not an HPE timing authority.  Crosswalk samples instead
    bind one or more production HPE cameras at the same logical frame indices;
    their hash-bound run metadata are the valid authority and must agree.
    """

    _, validation = _load_bound_json_object(
        validation_binding, "HPE validation timing authority"
    )
    if "hpe_camera" not in samples.columns or samples.empty:
        raise HPETranscriptCrosswalkError(
            "sealed crosswalk has no HPE evidence camera for timing"
        )
    raw_cameras = samples["hpe_camera"].tolist()
    if any(
        not isinstance(camera, str)
        or not re.fullmatch(r"Cam_[1-9][0-9]*", camera)
        for camera in raw_cameras
    ):
        raise HPETranscriptCrosswalkError(
            "sealed crosswalk HPE evidence camera is malformed"
        )
    evidence_cameras = sorted(set(raw_cameras))

    sources = manifest.get("sources")
    run_bindings = (
        sources.get("hpe_run_metadata") if isinstance(sources, dict) else None
    )
    if not isinstance(run_bindings, dict):
        raise HPETranscriptCrosswalkError(
            "sealed crosswalk lacks bound HPE run metadata"
        )
    pose_sources = validation.get("pose_sources")
    camera_records = (
        pose_sources.get("cameras") if isinstance(pose_sources, dict) else None
    )
    if not isinstance(camera_records, dict):
        raise HPETranscriptCrosswalkError(
            "sealed HPE validation lacks production pose cameras"
        )

    authorities: list[dict[str, Any]] = []
    fps_values: list[float] = []
    for camera in evidence_cameras:
        camera_record = camera_records.get(camera)
        if not isinstance(camera_record, dict):
            raise HPETranscriptCrosswalkError(
                f"sealed HPE validation has no production timing source for {camera}"
            )
        run_binding = run_bindings.get(camera)
        run_path, run_metadata = _load_bound_json_object(
            run_binding, f"HPE {camera} run-metadata timing authority"
        )
        if (
            camera_record.get("bound_camera") != camera
            or camera_record.get("metadata") != run_path.name
            or camera_record.get("metadata_sha256") != run_binding["sha256"]
        ):
            raise HPETranscriptCrosswalkError(
                f"sealed HPE validation/run-metadata timing identity differs for {camera}"
            )
        validation_fps = _finite_positive_json_number(
            camera_record.get("source_fps"),
            f"HPE validation source_fps for {camera}",
        )
        metadata_fps = _finite_positive_json_number(
            run_metadata.get("source_fps"),
            f"HPE run-metadata source_fps for {camera}",
        )
        if not math.isclose(
            validation_fps, metadata_fps, rel_tol=0.0, abs_tol=1e-9
        ):
            raise HPETranscriptCrosswalkError(
                f"sealed HPE timing FPS differs for {camera}"
            )
        fps_values.append(metadata_fps)
        authorities.append(
            {
                "camera": camera,
                "source_fps": metadata_fps,
                "run_metadata": dict(run_binding),
            }
        )
    source_fps = fps_values[0]
    if any(
        not math.isclose(value, source_fps, rel_tol=0.0, abs_tol=1e-9)
        for value in fps_values[1:]
    ):
        raise HPETranscriptCrosswalkError(
            "HPE evidence cameras disagree on the logical-frame FPS"
        )
    return {
        "axis": "hpe_delivery_logical_frame_index",
        "source_fps": source_fps,
        "evidence_cameras": authorities,
        "validation_complete": dict(validation_binding),
    }


def load_sealed_crosswalk_contract(
    scene: str,
    inventory_path: str | Path = DEFAULT_SEALED_INVENTORY_PATH,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Return the stable JSON-shaped contract consumed by intake/assembly."""

    root = Path(repository_root).resolve()
    supplied_inventory = Path(inventory_path)
    resolved_inventory = (
        supplied_inventory.resolve()
        if supplied_inventory.is_absolute()
        else (root / supplied_inventory).resolve()
    )
    inventory = load_sealed_crosswalk_inventory(
        resolved_inventory, repository_root=root
    )
    manifest, samples, mapping = _load_sealed_scene(
        scene=scene, inventory=inventory, repository_root=root
    )
    validation_binding = manifest["sources"]["hpe"]["validation_complete"]
    timing_authority = _resolve_hpe_logical_timing_authority(
        manifest=manifest,
        samples=samples,
        validation_binding=validation_binding,
    )
    source_fps = timing_authority["source_fps"]
    mapping_records = [
        {
            "hpe_person": int(row.hpe_person),
            "participant": str(row.participant),
            "slot": str(row.slot),
        }
        for row in mapping.itertuples(index=False)
    ]
    scene_record = inventory["scenes"][scene]
    return {
        "artifact_kind": SEALED_DOWNSTREAM_KIND,
        "format_version": SEALED_DOWNSTREAM_VERSION,
        "inventory": {
            "path": str(resolved_inventory),
            "repository_path": _canonical_repo_relative(
                root, resolved_inventory, "sealed inventory"
            ),
            "sha256": sha256_file(resolved_inventory),
            "artifact_kind": inventory["artifact_kind"],
            "format_version": inventory["format_version"],
        },
        "scene": scene,
        "accepted_window": dict(manifest["accepted_window"]),
        "source_fps": source_fps,
        "hpe_logical_timing_authority": timing_authority,
        "expected_hpe_persons": list(manifest["expected_hpe_persons"]),
        "person_mapping": mapping_records,
        "artifact_dir": scene_record["artifact_dir"],
        "artifact_files": {
            name: dict(binding)
            for name, binding in scene_record["files"].items()
        },
        "ledger_snapshot_sha256": inventory["ledger_snapshot"]["sha256"],
        "upstream_proposal": dict(manifest["upstream_proposal"]),
        "upstream_mismatch": dict(manifest["upstream_mismatch"]),
        "delivery_world_persons": dict(
            manifest["sources"]["hpe"]["delivery_world_persons"]
        ),
        "hpe_validation_authority": dict(validation_binding),
    }


def reaudit_sealed_crosswalk(
    scene: str,
    inventory_path: str | Path = DEFAULT_SEALED_INVENTORY_PATH,
    *,
    repository_root: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Full raw-evidence re-audit driven only by the immutable sealed ledger.

    Unlike :func:`load_sealed_crosswalk`, this verifies the bound raw AVI bytes.
    Unlike the original :func:`load_crosswalk_artifact`, it remains reproducible
    after the live status ledger evolves.  No caller-supplied ledger override is
    accepted; the version-controlled inventory selects the exact snapshot.
    """

    root = Path(repository_root).resolve()
    inventory = load_sealed_crosswalk_inventory(
        inventory_path, repository_root=root
    )
    artifact_root, accepted, scenes = _validate_inventory_structure(
        inventory, repository_root=root
    )
    if scene not in scenes:
        raise HPETranscriptCrosswalkError(f"scene {scene!r} is not in the sealed inventory")
    ledger = inventory["ledger_snapshot"]
    sealed_ledger = _resolve_inventory_path(
        root, ledger["sealed_path"], "immutable ledger snapshot path"
    )
    snapshot_windows = parse_accepted_identity_ledger(sealed_ledger)
    if snapshot_windows.get(scene) != accepted[scene]:
        raise HPETranscriptCrosswalkError(
            f"immutable ledger snapshot differs for sealed scene {scene}"
        )
    artifact_dir = artifact_root / scene
    manifest, samples = load_crosswalk_review_state(
        artifact_dir,
        verify_bound_video_bytes=True,
        portable_dtypes=True,
    )
    source_ledger = manifest.get("sources", {}).get("ledger")
    source_ledger_path = _resolve_inventory_path(
        root, ledger["source_path"], "sealed ledger source path"
    )
    if (
        not isinstance(source_ledger, dict)
        or set(source_ledger) != {"path", "sha256"}
        or Path(str(source_ledger.get("path"))).resolve() != source_ledger_path
        or source_ledger.get("sha256") != ledger["sha256"]
    ):
        raise HPETranscriptCrosswalkError(
            f"sealed scene {scene} was not prepared from the immutable ledger bytes"
        )
    _replay_nonledger_sources(
        manifest,
        snapshot_windows[scene],
        portable_dtypes=True,
    )
    mapping = _validate_finalized_crosswalk_semantics(
        artifact_dir=artifact_dir,
        manifest=manifest,
        samples=samples,
        accepted=snapshot_windows[scene],
        scene_record=scenes[scene],
    )
    return manifest, samples, mapping


def load_all_sealed_crosswalks(
    inventory_path: str | Path = DEFAULT_SEALED_INVENTORY_PATH,
    *,
    repository_root: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load every sealed final mapping under one already-verified inventory."""

    root = Path(repository_root).resolve()
    inventory = load_sealed_crosswalk_inventory(
        inventory_path, repository_root=root
    )
    return {
        scene: _load_sealed_scene(
            scene=scene, inventory=inventory, repository_root=root
        )[2]
        for scene in sorted(inventory["scenes"])
    }
