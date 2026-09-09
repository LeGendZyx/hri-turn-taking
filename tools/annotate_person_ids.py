"""Interactive A/B/C/R annotation for unstable gaze-export detection IDs.

Run from the repository root.  The script intentionally keeps OpenCV as a
lazy, annotation-only dependency so importing the core training package does
not require a video runtime.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import stat
import sys
import threading
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hri_turn_taking.data.identity_annotation import (  # noqa: E402
    IdentityAnnotationError,
    TrackerConfig,
    annotation_frame_window,
    annotation_lookup,
    append_anchor_audit_event,
    append_audit_event,
    append_split_audit_event,
    build_identity_bridge,
    export_identity_bridge,
    load_annotation_manifest,
    load_export_metadata,
    load_hpe_annotation_scope,
    mark_review_anchor,
    migrate_legacy_annotation_to_hpe_core,
    prepare_annotation_artifact,
    save_annotation_manifest,
    save_annotation_state,
    set_tracklet_annotation,
    sha256_file,
    split_tracklet,
    validate_tracklet_observations,
)

WINDOW_NAME = "Manual participant identity annotation"
_CLOUD_RECALL_FLAGS = (
    getattr(stat, "FILE_ATTRIBUTE_OFFLINE", 0x1000)
    | getattr(stat, "FILE_ATTRIBUTE_RECALL_ON_OPEN", 0x40000)
    | getattr(stat, "FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS", 0x400000)
)
LABEL_COLOURS = {
    "A": (80, 220, 80),
    "B": (255, 150, 40),
    "C": (220, 80, 220),
    "R": (40, 200, 255),
    None: (80, 80, 255),
    "ignored": (150, 150, 150),
}


def _windows_file_attributes(path: Path, timeout_s: float = 3.0) -> int:
    result: list[int] = []

    def query() -> None:
        result.append(int(ctypes.windll.kernel32.GetFileAttributesW(str(path))))

    worker = threading.Thread(target=query, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise IdentityAnnotationError(
            "timed out querying the OneDrive video placeholder; make the file local first"
        )
    if not result:
        raise IdentityAnnotationError(f"could not query video attributes: {path}")
    return result[0]


def _is_offline_placeholder(path: Path) -> bool:
    if os.name != "nt":
        return False
    attributes = _windows_file_attributes(path)
    if attributes == 0xFFFFFFFF:
        return False
    return bool(attributes & _CLOUD_RECALL_FLAGS)


def _validate_video_file_without_opening(path: Path, *, allow_offline: bool) -> None:
    if os.name == "nt":
        attributes = _windows_file_attributes(path)
        if attributes == 0xFFFFFFFF:
            raise IdentityAnnotationError(f"video does not exist: {path}")
        directory_flag = getattr(stat, "FILE_ATTRIBUTE_DIRECTORY", 0x10)
        if attributes & directory_flag:
            raise IdentityAnnotationError(f"video path is a directory: {path}")
        is_offline = bool(attributes & _CLOUD_RECALL_FLAGS)
    else:
        if not path.is_file():
            raise IdentityAnnotationError(f"video does not exist: {path}")
        is_offline = False
    if is_offline and not allow_offline:
        raise IdentityAnnotationError(
            "video is a OneDrive Offline placeholder. In Explorer choose "
            "'Always keep on this device' first; the tool refuses an implicit "
            "multi-GB download."
        )


def _infer_dataset_session(gaze_csv: Path) -> tuple[str, str]:
    session_dir = gaze_csv.parent.name
    if not session_dir.startswith("session_"):
        raise IdentityAnnotationError(
            "cannot infer session; pass --dataset and --session explicitly"
        )
    return gaze_csv.parent.parent.name, session_dir.removeprefix("session_")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise IdentityAnnotationError(f"{label} does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityAnnotationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise IdentityAnnotationError(f"{label} root must be a JSON object")
    return value


def _resolve_inside_hpe_root(hpe_root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = hpe_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(hpe_root.resolve())
    except ValueError as exc:
        raise IdentityAnnotationError(f"{label} escapes the HPE repository") from exc
    return resolved


def _find_hpe_run_metadata(
    hpe_root: Path,
    *,
    profile: Path,
    dataset: str,
    session: str,
    declared_video: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Follow the confirmed delivery chain to the exact production marker."""

    scene = f"{dataset}__{session}"
    mapping_path = (hpe_root / "outputs" / "delivery" / scene / "mapping.json").resolve()
    mapping = _load_json_object(mapping_path, "HPE delivery mapping")
    if (
        mapping.get("review_status") != "confirmed"
        or mapping.get("delivery") != scene
        or mapping.get("group") != dataset
        or str(mapping.get("scenario")) != session
    ):
        raise IdentityAnnotationError("HPE delivery mapping is not confirmed for this scene")
    sources = mapping.get("sources")
    if not isinstance(sources, dict) or not isinstance(
        sources.get("validation_dir"), str
    ):
        raise IdentityAnnotationError("HPE mapping lacks a validation_dir binding")
    validation_dir = _resolve_inside_hpe_root(
        hpe_root, sources["validation_dir"], "HPE validation_dir"
    )
    validation_path = (validation_dir / "validation_complete.json").resolve()
    validation = _load_json_object(validation_path, "HPE validation marker")

    gate = validation.get("geometry_cardinality_gate")
    profile_binding = gate.get("profile") if isinstance(gate, dict) else None
    if (
        not isinstance(gate, dict)
        or gate.get("day") != dataset
        or str(gate.get("scenario")) != session
        or gate.get("production_pass") is not True
        or gate.get("coverage_complete") is not True
        or not isinstance(profile_binding, dict)
        or not isinstance(profile_binding.get("repo_path"), str)
        or not isinstance(profile_binding.get("sha256"), str)
    ):
        raise IdentityAnnotationError("HPE validation lacks a passing geometry core gate")
    bound_profile = _resolve_inside_hpe_root(
        hpe_root, profile_binding["repo_path"], "HPE validated geometry profile"
    )
    if (
        bound_profile != profile.resolve()
        or profile_binding["sha256"] != sha256_file(profile.resolve())
    ):
        raise IdentityAnnotationError(
            "HPE validation does not bind the selected geometry profile"
        )

    pose_sources = validation.get("pose_sources")
    camera = str(declared_video["camera_dir"])
    cameras = pose_sources.get("cameras") if isinstance(pose_sources, dict) else None
    entry = cameras.get(camera) if isinstance(cameras, dict) else None
    if (
        not isinstance(pose_sources, dict)
        or pose_sources.get("batch_schema") != "v2_gate"
        or pose_sources.get("recording_day") != dataset
        or str(pose_sources.get("scenario")) != session
        or not isinstance(cameras, dict)
    ):
        raise IdentityAnnotationError("HPE validation pose sources identify another scene")
    authority_kind = "hpe_confirmed_delivery_validation_v1"
    if isinstance(entry, dict):
        if (
            entry.get("bound_day") != dataset
            or str(entry.get("bound_scenario")) != session
            or entry.get("bound_camera") != camera
        ):
            raise IdentityAnnotationError(
                "HPE validation pose source identifies another stream"
            )
        try:
            entry_fps = float(entry.get("source_fps"))
            entry_frames = int(entry.get("processed_frames"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise IdentityAnnotationError(
                "HPE validation pose-source counts are invalid"
            ) from exc
        extraction_scope = entry.get("extraction_scope")
        if (
            isinstance(entry.get("source_fps"), bool)
            or not math.isfinite(entry_fps)
            or not math.isclose(
                entry_fps,
                float(declared_video["fps"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or isinstance(entry.get("processed_frames"), bool)
            or entry_frames <= 0
            or float(entry.get("processed_frames")) != entry_frames
            or not isinstance(extraction_scope, dict)
            or isinstance(extraction_scope.get("start_frame"), bool)
            or extraction_scope.get("start_frame") != 0
            or extraction_scope.get("max_frames") is not None
        ):
            raise IdentityAnnotationError("HPE validated pose source is not a complete run")
        csv_root_value = validation.get("csv_root")
        metadata_name = entry.get("metadata")
        expected_hash = entry.get("metadata_sha256")
        if (
            not isinstance(csv_root_value, str)
            or not isinstance(metadata_name, str)
            or Path(metadata_name).name != metadata_name
            or not isinstance(expected_hash, str)
        ):
            raise IdentityAnnotationError(
                "HPE validation lacks a safe run-metadata binding"
            )
        csv_root = _resolve_inside_hpe_root(
            hpe_root, csv_root_value, "HPE pose csv_root"
        )
        run_metadata = (csv_root / camera / metadata_name).resolve()
        try:
            run_metadata.relative_to(hpe_root.resolve())
        except ValueError as exc:
            raise IdentityAnnotationError("HPE run metadata escapes the repository") from exc
        if not run_metadata.is_file() or sha256_file(run_metadata) != expected_hash:
            raise IdentityAnnotationError(
                "HPE run metadata differs from validation_complete"
            )
        run_doc = _load_json_object(run_metadata, "HPE run metadata")
        try:
            run_fps = float(run_doc.get("source_fps"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise IdentityAnnotationError("HPE run metadata has invalid FPS") from exc
        if run_doc.get("processed_frames") != entry_frames or not math.isclose(
            run_fps, entry_fps, rel_tol=0.0, abs_tol=1e-9
        ):
            raise IdentityAnnotationError(
                "HPE run metadata disagrees with validation_complete"
            )
    else:
        # Some confirmed scenes intentionally exclude the gaze camera from pose
        # reconstruction (05_15 uses Cam_1 for gaze but validates Cam_2..Cam_6).
        # The scene/profile authority still comes from delivery+validation; the
        # exact excluded-camera video axis is bound independently to its complete
        # extractor marker and rechecked by ``load_hpe_annotation_scope``.
        required_cameras = gate.get("required_active_cameras")
        if (
            not isinstance(required_cameras, list)
            or not required_cameras
            or camera in required_cameras
        ):
            raise IdentityAnnotationError(
                "gaze camera is missing from the validated HPE pose sources"
            )
        run_metadata = (
            hpe_root
            / "outputs_v2"
            / dataset
            / "6cams"
            / session
            / camera
            / "run_metadata.json"
        ).resolve()
        try:
            run_metadata.relative_to(hpe_root.resolve())
        except ValueError as exc:
            raise IdentityAnnotationError("HPE camera-axis marker escapes the repository") from exc
        if not run_metadata.is_file():
            raise IdentityAnnotationError(
                "excluded gaze camera lacks an exact HPE video-axis marker"
            )
        expected_hash = sha256_file(run_metadata)
        authority_kind = "hpe_confirmed_scene_excluded_camera_axis_v1"
    return run_metadata, {
        "kind": authority_kind,
        "mapping_path": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "validation_path": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "camera": camera,
        "expected_run_metadata_sha256": expected_hash,
    }


def _resolve_hpe_scope(
    *,
    hpe_root: Path,
    profile_arg: Path | None,
    dataset: str,
    session: str,
    declared_video: dict[str, Any],
) -> dict[str, Any]:
    profile = (
        profile_arg.resolve()
        if profile_arg is not None
        else (
            hpe_root
            / "calibration"
            / "scenario_geometry_profiles"
            / f"{dataset}__{session}.json"
        ).resolve()
    )
    run_metadata, authority_binding = _find_hpe_run_metadata(
        hpe_root,
        profile=profile,
        dataset=dataset,
        session=session,
        declared_video=declared_video,
    )
    scope = load_hpe_annotation_scope(
        profile,
        run_metadata,
        dataset=dataset,
        session=session,
        declared_video=declared_video,
    )
    scope["authority_binding"] = authority_binding
    return scope


def _require_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise IdentityAnnotationError(
            "OpenCV is required for the video UI. Install the project vision extra "
            "or use a Python environment that already provides cv2."
        ) from exc
    return cv2


def _validate_video_path_identity(video_path: Path, declared_video: dict[str, Any]) -> None:
    expected_name = str(declared_video["basename"])
    expected_camera = str(declared_video["camera_dir"])
    expected_session = str(declared_video["session"])
    if video_path.name.casefold() != expected_name.casefold():
        raise IdentityAnnotationError(
            f"video basename does not match session metadata: expected {expected_name!r}"
        )
    if video_path.parent.name.casefold() != expected_camera.casefold():
        raise IdentityAnnotationError(
            f"video camera directory does not match session metadata: expected {expected_camera!r}"
        )
    if video_path.parent.parent.name.casefold() != expected_session.casefold():
        raise IdentityAnnotationError(
            "video session directory does not match session metadata: "
            f"expected {expected_session!r}"
        )


def _sparse_video_fingerprint(path: Path, chunk_size: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    offsets = sorted({0, max(0, size // 2 - chunk_size // 2), max(0, size - chunk_size)})
    digest = hashlib.sha256()
    digest.update(f"size={size};chunk={chunk_size};".encode())
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            payload = handle.read(chunk_size)
            digest.update(f"offset={offset};length={len(payload)};".encode())
            digest.update(payload)
    return digest.hexdigest()


def _preflight_video(video_path: Path, declared_video: dict[str, Any]) -> dict[str, Any]:
    _validate_video_path_identity(video_path, declared_video)
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IdentityAnnotationError(f"OpenCV cannot open video: {video_path}")
    try:
        actual = {
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        }
    finally:
        capture.release()
    if not math.isfinite(actual["fps"]) or actual["fps"] <= 0:
        raise IdentityAnnotationError("video reports an invalid FPS")
    fps_tolerance = max(1e-3, abs(float(declared_video["fps"])) * 1e-5)
    if abs(actual["fps"] - float(declared_video["fps"])) > fps_tolerance:
        raise IdentityAnnotationError(
            f"video FPS {actual['fps']} does not match metadata {declared_video['fps']}"
        )
    for field in ("frame_count", "width", "height"):
        if actual[field] != int(declared_video[field]):
            raise IdentityAnnotationError(
                f"video {field} {actual[field]} does not match metadata {declared_video[field]}"
            )
    return {
        "identity_version": 1,
        "declared_source_path": declared_video["source_path"],
        "basename": declared_video["basename"],
        "camera_dir": declared_video["camera_dir"],
        **actual,
        "size_bytes": video_path.stat().st_size,
        "fingerprint_algorithm": "sha256(size + first/middle/last 1MiB with offsets)",
        "sparse_sha256": _sparse_video_fingerprint(video_path),
    }


def _bind_video_identity(
    manifest: dict[str, Any], video_identity: dict[str, Any], video_path: Path
) -> None:
    recorded_video = manifest.get("video")
    if recorded_video is not None:
        if not isinstance(recorded_video, dict):
            raise IdentityAnnotationError("manifest video binding must be a JSON object")
        identity_fields = {
            "identity_version",
            "declared_source_path",
            "basename",
            "camera_dir",
            "fps",
            "frame_count",
            "width",
            "height",
            "size_bytes",
            "fingerprint_algorithm",
            "sparse_sha256",
        }
        if any(recorded_video.get(key) != video_identity.get(key) for key in identity_fields):
            raise IdentityAnnotationError(
                "selected video differs from the video bound to these annotations"
            )
    else:
        manifest["video"] = dict(video_identity)
        manifest["video"]["first_bound_path"] = str(video_path.resolve())
    manifest["video"]["last_verified_path"] = str(video_path.resolve())


def _preflight_existing_video_binding(
    output_dir: Path, video_identity: dict[str, Any], video_path: Path
) -> None:
    """Compare an existing binding without mutating the annotation artifact."""

    manifest_path = output_dir / "identity_annotations.json"
    if not manifest_path.is_file():
        return
    manifest = _load_json_object(manifest_path, "identity annotation manifest")
    if manifest.get("video") is None:
        return
    probe = {"video": dict(manifest["video"])} if isinstance(manifest["video"], dict) else {
        "video": manifest["video"]
    }
    _bind_video_identity(probe, video_identity, video_path)


def _resolve_start_frame(
    requested: int | None,
    *,
    annotation_scope: dict[str, Any],
    declared_video: dict[str, Any],
) -> int:
    if annotation_scope.get("kind") == "hpe_confirmed_core_window":
        lower = int(annotation_scope["gaze_local_start_frame_inclusive"])
        upper = int(annotation_scope["gaze_local_end_frame_exclusive"])
    elif annotation_scope == {"kind": "full_gaze_export"}:
        lower = 0
        upper = int(declared_video["frame_count"])
    else:
        raise IdentityAnnotationError("cannot determine annotation frame window")
    result = lower if requested is None else requested
    if not lower <= result < upper:
        raise IdentityAnnotationError(
            f"--start-frame must fall inside the local annotation window [{lower}, {upper})"
        )
    return result


class IdentityAnnotator:
    def __init__(
        self,
        *,
        video_path: Path,
        observations_path: Path,
        manifest_path: Path,
        output_dir: Path,
        video_identity: dict[str, Any],
        start_frame: int = 0,
    ) -> None:
        self.cv2 = _require_cv2()
        self.video_path = video_path
        self.observations_path = observations_path
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.audit_path = output_dir / "annotation_audit.jsonl"
        self.manifest = load_annotation_manifest(manifest_path)
        self.frame_window = annotation_frame_window(self.manifest)
        self.observations = validate_tracklet_observations(
            observations_path, self.manifest
        )
        self.annotations = annotation_lookup(self.manifest)
        self.undo_stack: list[tuple[str, dict[str, Any]]] = []
        self.frame_rows: dict[int, pd.DataFrame] = {}
        self._refresh_indices()

        self.capture = self.cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise IdentityAnnotationError(f"OpenCV cannot open video: {video_path}")
        self.fps = float(self.capture.get(self.cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(self.cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(self.cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise IdentityAnnotationError("video reports an invalid FPS")
        if self.frame_count <= 0 or self.width <= 0 or self.height <= 0:
            raise IdentityAnnotationError("video reports invalid dimensions/frame count")
        if self.frame_window is None:
            self.minimum_frame = 0
            self.maximum_frame_exclusive = self.frame_count
        else:
            self.minimum_frame, self.maximum_frame_exclusive = self.frame_window
            if self.maximum_frame_exclusive > self.frame_count:
                raise IdentityAnnotationError("HPE local core window exceeds video length")
        if video_path.stat().st_size != int(video_identity["size_bytes"]):
            raise IdentityAnnotationError("video changed after preflight")
        if (
            abs(self.fps - float(video_identity["fps"])) > 1e-6
            or self.frame_count != int(video_identity["frame_count"])
            or self.width != int(video_identity["width"])
            or self.height != int(video_identity["height"])
        ):
            raise IdentityAnnotationError("video stream changed after preflight")
        maximum_source_frame = int(self.observations["frame_index"].max())
        if maximum_source_frame >= self.frame_count:
            raise IdentityAnnotationError(
                f"gaze frame {maximum_source_frame} exceeds video frame count "
                f"{self.frame_count}; this is probably a split/version mismatch"
            )
        expected_times = self.observations["frame_index"].to_numpy(dtype=float) / self.fps
        source_times = self.observations["timestamp_sec"].to_numpy(dtype=float)
        if abs(source_times - expected_times).max(initial=0.0) > max(0.02, 0.75 / self.fps):
            raise IdentityAnnotationError(
                "gaze timestamps disagree with video FPS/frame indices; refuse unsafe overlay"
            )
        boxes = self.observations.loc[
            :, ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]
        ].to_numpy(dtype=float)
        if (
            (boxes[:, 0] < 0).any()
            or (boxes[:, 1] < 0).any()
            or (boxes[:, 2] > self.width).any()
            or (boxes[:, 3] > self.height).any()
        ):
            raise IdentityAnnotationError("annotation boxes exceed the selected video dimensions")

        _bind_video_identity(self.manifest, video_identity, video_path)
        save_annotation_manifest(self.manifest_path, self.manifest)

        self.current_frame = min(
            max(int(start_frame), self.minimum_frame),
            self.maximum_frame_exclusive - 1,
        )
        self.selected_tracklet: str | None = None
        self.playing = False
        self.show_help = True
        self.status_message = "Click a box, then press A/B/C/R. Press H for help."
        self.display_scale = 1.0
        self.dirty = True
        self.last_display = None

    def _refresh_indices(self) -> None:
        self.frame_rows = {
            int(frame_index): frame.copy()
            for frame_index, frame in self.observations.groupby("frame_index", sort=False)
        }
        self.annotations = annotation_lookup(self.manifest)

    def _annotation(self, tracklet_id: str) -> dict[str, Any]:
        return self.annotations[str(tracklet_id)]

    def _read_frame(self, frame_index: int) -> Any:
        self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = self.capture.read()
        if not ok:
            raise IdentityAnnotationError(f"could not decode video frame {frame_index}")
        return frame

    def _display_size(self) -> tuple[int, int, float]:
        scale = min(1.0, 1600 / self.width, 900 / self.height)
        return int(round(self.width * scale)), int(round(self.height * scale)), scale

    def _colour_for(self, tracklet_id: str) -> tuple[int, int, int]:
        annotation = self._annotation(tracklet_id)
        if annotation.get("status") == "ignored":
            return LABEL_COLOURS["ignored"]
        return LABEL_COLOURS.get(annotation.get("label"), LABEL_COLOURS[None])

    def _draw_text(
        self, image: Any, text: str, x: int, y: int, colour: tuple[int, int, int]
    ) -> None:
        font = self.cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        thickness = 1
        (width, height), baseline = self.cv2.getTextSize(text, font, scale, thickness)
        y = max(height + baseline + 2, y)
        self.cv2.rectangle(
            image, (x, y - height - baseline - 3), (x + width + 4, y + 2), (20, 20, 20), -1
        )
        self.cv2.putText(
            image,
            text,
            (x + 2, y - baseline),
            font,
            scale,
            colour,
            thickness,
            self.cv2.LINE_AA,
        )

    def _render(self) -> Any:
        frame = self._read_frame(self.current_frame)
        rows = self.frame_rows.get(self.current_frame)
        if rows is not None:
            for _, row in rows.iterrows():
                tracklet_id = str(row["tracklet_id"])
                annotation = self._annotation(tracklet_id)
                colour = self._colour_for(tracklet_id)
                x0, y0, x1, y1 = [int(round(float(row[column]))) for column in (
                    "bbox_x0",
                    "bbox_y0",
                    "bbox_x1",
                    "bbox_y1",
                )]
                selected = tracklet_id == self.selected_tracklet
                thickness = 5 if selected else (3 if row["source_kind"] == "robot_candidate" else 2)
                self.cv2.rectangle(frame, (x0, y0), (x1, y1), colour, thickness)
                label = annotation.get("label") or annotation.get("status", "unreviewed")
                prefix = "ROBOT?" if row["source_kind"] == "robot_candidate" else "face"
                self._draw_text(
                    frame,
                    f"{prefix} {tracklet_id} raw={row['source_person_idx']} -> {label}",
                    x0,
                    y0 - 4,
                    colour,
                )
                gaze_x = pd.to_numeric(pd.Series([row.get("gaze_peak_x")]), errors="coerce").iloc[0]
                gaze_y = pd.to_numeric(pd.Series([row.get("gaze_peak_y")]), errors="coerce").iloc[0]
                if pd.notna(gaze_x) and pd.notna(gaze_y):
                    centre = ((x0 + x1) // 2, (y0 + y1) // 2)
                    peak = (int(round(float(gaze_x))), int(round(float(gaze_y))))
                    self.cv2.line(frame, centre, peak, colour, 1)
                    self.cv2.circle(frame, peak, 4, colour, -1)

        manual_items = [
            item
            for item in self.manifest["annotations"]
            if not item.get("auto_quarantined", False)
        ]
        reviewed = sum(
            item.get("status") in {"confirmed", "ignored", "uncertain"}
            for item in manual_items
        )
        total = len(manual_items)
        auto_missing = sum(
            bool(item.get("auto_quarantined", False))
            for item in self.manifest["annotations"]
        )
        participant_items = [
            item
            for item in self.manifest["annotations"]
            if item.get("source_kind") == "participant_detection"
        ]
        participant_rows = sum(int(item["observation_count"]) for item in participant_items)
        resolved_rows = sum(
            int(item["observation_count"])
            for item in participant_items
            if item.get("status") in {"confirmed", "ignored"}
        )
        confirmed_rows = sum(
            int(item["observation_count"])
            for item in participant_items
            if item.get("status") == "confirmed"
        )
        resolved_percent = 100 * resolved_rows / max(participant_rows, 1)
        confirmed_percent = 100 * confirmed_rows / max(participant_rows, 1)
        header = (
            f"frame={self.current_frame}  "
            f"core=[{self.minimum_frame},{self.maximum_frame_exclusive})  "
            f"time={self.current_frame / self.fps:.3f}s  "
            f"manual reviewed={reviewed}/{total}  auto-missing={auto_missing}  "
            f"face rows: resolved={resolved_percent:.1f}% "
            f"confirmed={confirmed_percent:.1f}%  selected={self.selected_tracklet or '-'}"
        )
        self._draw_text(frame, header, 10, 28, (255, 255, 255))
        self._draw_text(frame, self.status_message, 10, 54, (255, 255, 255))

        if self.show_help:
            help_lines = [
                "Mouse: select bbox   A/B/C/R: confirm identity   U: uncertain   I: ignore",
                "V: verify required anchor   N/P: next/previous unreviewed tracklet",
                "[/]: selected tracklet start/end   X: split selected tracklet here",
                "Left/Right: 1 frame   Up/Down: 1 second   Space: play/pause",
                "Z: undo last label   H: toggle help   Q/Esc: save and quit",
            ]
            for index, line in enumerate(help_lines):
                self._draw_text(frame, line, 10, 82 + 25 * index, (240, 240, 240))

        display_width, display_height, self.display_scale = self._display_size()
        if self.display_scale != 1.0:
            frame = self.cv2.resize(
                frame, (display_width, display_height), interpolation=self.cv2.INTER_AREA
            )
        self.last_display = frame
        return frame

    def _seek(self, frame_index: int) -> None:
        minimum_frame = getattr(self, "minimum_frame", 0)
        maximum_frame_exclusive = getattr(
            self, "maximum_frame_exclusive", self.frame_count
        )
        self.current_frame = min(
            max(int(frame_index), minimum_frame),
            maximum_frame_exclusive - 1,
        )
        if self.selected_tracklet is not None and not self._tracklet_visible(
            self.selected_tracklet, self.current_frame
        ):
            self.selected_tracklet = None
        self.dirty = True

    def _tracklet_visible(self, tracklet_id: str, frame_index: int) -> bool:
        rows = self.frame_rows.get(int(frame_index))
        if rows is None:
            return False
        return bool((rows["tracklet_id"].astype(str) == str(tracklet_id)).any())

    def _select_at(self, x: int, y: int) -> None:
        source_x = x / max(self.display_scale, 1e-9)
        source_y = y / max(self.display_scale, 1e-9)
        rows = self.frame_rows.get(self.current_frame)
        if rows is None:
            self.selected_tracklet = None
            self.status_message = "no selectable box in this frame"
            self.dirty = True
            return
        candidates: list[tuple[float, str]] = []
        for _, row in rows.iterrows():
            x0, y0, x1, y1 = (float(row[column]) for column in (
                "bbox_x0",
                "bbox_y0",
                "bbox_x1",
                "bbox_y1",
            ))
            if x0 <= source_x <= x1 and y0 <= source_y <= y1:
                candidates.append(((x1 - x0) * (y1 - y0), str(row["tracklet_id"])))
        if candidates:
            self.selected_tracklet = min(candidates)[1]
            annotation = self._annotation(self.selected_tracklet)
            self.status_message = (
                f"selected {self.selected_tracklet}: frames {annotation['start_frame']}-"
                f"{annotation['end_frame']}, {annotation['observation_count']} observations"
            )
        else:
            self.selected_tracklet = None
            self.status_message = "click missed all boxes; selection cleared"
        self.dirty = True

    def _mouse_callback(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == self.cv2.EVENT_LBUTTONDOWN:
            self._select_at(x, y)

    def _save_annotation(self, label: str | None, status: str) -> None:
        if self.selected_tracklet is None:
            self.status_message = "Select a bbox first."
            self.dirty = True
            return
        tracklet_id = self.selected_tracklet
        if not self._tracklet_visible(tracklet_id, self.current_frame):
            self.selected_tracklet = None
            self.status_message = "selected tracklet is not visible; select a box again"
            self.dirty = True
            return
        try:
            previous = set_tracklet_annotation(
                self.manifest, tracklet_id, label=label, status=status
            )
        except IdentityAnnotationError as exc:
            self.status_message = str(exc)
            self.dirty = True
            return
        self.undo_stack.append((tracklet_id, previous))
        save_annotation_manifest(self.manifest_path, self.manifest)
        current = self._annotation(tracklet_id)
        append_audit_event(
            self.audit_path,
            tracklet_id=tracklet_id,
            previous=previous,
            current=current,
            action="set_annotation",
        )
        _, conflicts = build_identity_bridge(self.observations, self.manifest)
        suffix = f"; {len(conflicts)} conflicting rows" if not conflicts.empty else ""
        self.status_message = f"saved {tracklet_id} -> {label or status}{suffix}"
        self.dirty = True

    def _mark_anchor(self) -> None:
        if self.selected_tracklet is None:
            self.status_message = "Select a visible tracklet first."
            self.dirty = True
            return
        tracklet_id = self.selected_tracklet
        if not self._tracklet_visible(tracklet_id, self.current_frame):
            self.selected_tracklet = None
            self.status_message = "selected tracklet is not visible; selection cleared"
            self.dirty = True
            return
        already_reviewed = self.current_frame in self._annotation(tracklet_id)[
            "reviewed_anchor_frames"
        ]
        try:
            _, remaining = mark_review_anchor(
                self.manifest, tracklet_id, self.current_frame
            )
        except IdentityAnnotationError as exc:
            required = self._annotation(tracklet_id)["required_review_frames"]
            self.status_message = f"{exc}; required={required}"
            self.dirty = True
            return
        save_annotation_manifest(self.manifest_path, self.manifest)
        reviewed_frame = self.current_frame
        if not already_reviewed:
            append_anchor_audit_event(
                self.audit_path,
                tracklet_id=tracklet_id,
                frame_index=reviewed_frame,
                remaining_frames=remaining,
            )
        if remaining:
            next_frame = next(
                (frame for frame in remaining if frame > reviewed_frame), remaining[0]
            )
            self._seek(next_frame)
            self.status_message = (
                f"verified {tracklet_id} at {reviewed_frame}; next anchor={next_frame}, "
                f"remaining={len(remaining)}"
            )
        else:
            self.status_message = f"all required anchors verified for {tracklet_id}"
        self.dirty = True

    def _undo(self) -> None:
        if not self.undo_stack:
            self.status_message = "Nothing to undo."
            self.dirty = True
            return
        tracklet_id, previous = self.undo_stack.pop()
        current = dict(self._annotation(tracklet_id))
        set_tracklet_annotation(
            self.manifest,
            tracklet_id,
            label=previous.get("label"),
            status=previous.get("status", "unreviewed"),
            note=previous.get("note", ""),
        )
        save_annotation_manifest(self.manifest_path, self.manifest)
        append_audit_event(
            self.audit_path,
            tracklet_id=tracklet_id,
            previous=current,
            current=self._annotation(tracklet_id),
            action="undo",
        )
        self.status_message = f"undid annotation for {tracklet_id}"
        self.dirty = True

    def _jump_unreviewed(self, direction: int) -> None:
        items = sorted(
            self.manifest["annotations"],
            key=lambda item: (
                -int(item["observation_count"]),
                int(item["start_frame"]),
                item["tracklet_id"],
            ),
        )
        if not any(item.get("status") == "unreviewed" for item in items):
            self.status_message = "All tracklets have been reviewed."
            self.dirty = True
            return
        current_index = -1 if direction > 0 else 0
        for index, item in enumerate(items):
            if item["tracklet_id"] == self.selected_tracklet:
                current_index = index
                break
        target = None
        for offset in range(1, len(items) + 1):
            candidate = items[(current_index + direction * offset) % len(items)]
            if candidate.get("status") == "unreviewed":
                target = candidate
                break
        if target is None:
            raise AssertionError("an unreviewed tracklet was expected")
        self.selected_tracklet = target["tracklet_id"]
        target_frames = self.observations.loc[
            self.observations["tracklet_id"].astype(str) == self.selected_tracklet,
            "frame_index",
        ].sort_values(kind="stable")
        self._seek(int(target_frames.iloc[len(target_frames) // 2]))
        self.status_message = (
            f"unreviewed {self.selected_tracklet}: frames {target['start_frame']}-"
            f"{target['end_frame']}, {target['observation_count']} observations"
        )

    def _jump_selected_edge(self, *, end: bool) -> None:
        if self.selected_tracklet is None:
            self.status_message = "Select a tracklet first."
            self.dirty = True
            return
        frames = self.observations.loc[
            self.observations["tracklet_id"].astype(str) == self.selected_tracklet,
            "frame_index",
        ]
        self.playing = False
        self._seek(int(frames.max() if end else frames.min()))
        edge = "end" if end else "start"
        self.status_message = f"{self.selected_tracklet} {edge}"

    def _split_selected(self) -> None:
        if self.selected_tracklet is None:
            self.status_message = "Select a tracklet before splitting."
            self.dirty = True
            return
        source_tracklet = self.selected_tracklet
        try:
            self.observations, new_id = split_tracklet(
                self.observations, self.manifest, source_tracklet, self.current_frame
            )
        except IdentityAnnotationError as exc:
            self.status_message = str(exc)
            self.dirty = True
            return
        save_annotation_state(
            self.observations_path,
            self.manifest_path,
            self.observations,
            self.manifest,
        )
        append_split_audit_event(
            self.audit_path,
            source_tracklet_id=source_tracklet,
            new_tracklet_id=new_id,
            split_frame=self.current_frame,
        )
        self._refresh_indices()
        self.selected_tracklet = new_id
        self.status_message = f"split {source_tracklet}; selected new {new_id}"
        self.dirty = True

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord(" "):
            self.playing = not self.playing
        elif key in (81, 2424832, ord(",")):
            self.playing = False
            self._seek(self.current_frame - 1)
        elif key in (83, 2555904, ord(".")):
            self.playing = False
            self._seek(self.current_frame + 1)
        elif key in (82, 2490368):
            self.playing = False
            self._seek(self.current_frame - round(self.fps))
        elif key in (84, 2621440):
            self.playing = False
            self._seek(self.current_frame + round(self.fps))
        elif key in tuple(map(ord, "abcrABCR")):
            self._save_annotation(chr(key).upper(), "confirmed")
        elif key in (ord("u"), ord("U")):
            self._save_annotation(None, "uncertain")
        elif key in (ord("i"), ord("I")):
            self._save_annotation(None, "ignored")
        elif key in (ord("n"), ord("N")):
            self._jump_unreviewed(1)
        elif key in (ord("p"), ord("P")):
            self._jump_unreviewed(-1)
        elif key in (ord("v"), ord("V")):
            self._mark_anchor()
        elif key == ord("["):
            self._jump_selected_edge(end=False)
        elif key == ord("]"):
            self._jump_selected_edge(end=True)
        elif key in (ord("x"), ord("X")):
            self._split_selected()
        elif key in (ord("z"), ord("Z")):
            self._undo()
        elif key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
            self.dirty = True
        return True

    def run(self) -> None:
        self.cv2.namedWindow(WINDOW_NAME, self.cv2.WINDOW_AUTOSIZE)
        self.cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)
        try:
            running = True
            while running:
                if self.dirty or self.playing:
                    display = self._render()
                    self.cv2.imshow(WINDOW_NAME, display)
                    self.dirty = False
                delay_ms = max(1, round(1000 / self.fps)) if self.playing else 30
                key = self.cv2.waitKeyEx(delay_ms)
                if key != -1:
                    running = self._handle_key(key)
                if self.playing:
                    if self.current_frame + 1 >= self.maximum_frame_exclusive:
                        self.playing = False
                    else:
                        self._seek(self.current_frame + 1)
                if self.cv2.getWindowProperty(WINDOW_NAME, self.cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            save_annotation_manifest(self.manifest_path, self.manifest)
            bridge_path, conflicts_path = export_identity_bridge(
                self.observations_path, self.manifest_path, self.output_dir
            )
            self.capture.release()
            self.cv2.destroyAllWindows()
            print(f"Saved identity bridge: {bridge_path}")
            print(f"Saved conflict report: {conflicts_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build bbox tracklets and manually label them as A/B/C/R."
    )
    parser.add_argument("--gaze-csv", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--session")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--max-gap-s", type=float, default=0.15)
    parser.add_argument(
        "--crowding-center-distance-ratio",
        type=float,
        default=0.80,
        help="bbox-centre crowding gate; changing it requires a new revision",
    )
    parser.add_argument(
        "--short-tracklet-max-duration-s",
        type=float,
        default=0.0,
        help=(
            "automatically leave participant tracklets at or below this timestamp span "
            "missing; changing it requires a new revision"
        ),
    )
    parser.add_argument(
        "--hpe-root",
        type=Path,
        default=REPOSITORY_ROOT.parent / "Head_pose_extraction",
    )
    parser.add_argument("--hpe-profile", type=Path)
    parser.add_argument(
        "--allow-full-session",
        action="store_true",
        help="disable the formal HPE core window (smoke/debug only)",
    )
    revision_group = parser.add_mutually_exclusive_group()
    revision_group.add_argument(
        "--migrate-hpe-core",
        action="store_true",
        help="back up and explicitly migrate an existing legacy v1 artifact",
    )
    parser.add_argument("--prepare-only", action="store_true")
    revision_group.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-offline-video",
        action="store_true",
        help="allow OpenCV to hydrate a OneDrive Offline placeholder (not recommended)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        gaze_csv = args.gaze_csv.resolve()
        inferred_dataset, inferred_session = _infer_dataset_session(gaze_csv)
        dataset = str(args.dataset or inferred_dataset)
        session = str(args.session or inferred_session)
        if dataset != inferred_dataset or session != inferred_session:
            raise IdentityAnnotationError(
                "--dataset/--session may not override the gaze export directory identity"
            )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else (REPOSITORY_ROOT / "data" / "annotations" / "identity" / f"{dataset}__{session}")
        )
        _, _, declared_video = load_export_metadata(gaze_csv)
        if args.allow_full_session:
            if args.hpe_profile is not None:
                raise IdentityAnnotationError(
                    "--allow-full-session cannot be combined with HPE provenance arguments"
                )
            if args.migrate_hpe_core:
                raise IdentityAnnotationError(
                    "--migrate-hpe-core requires an HPE core window"
                )
            annotation_scope = {"kind": "full_gaze_export"}
        else:
            annotation_scope = _resolve_hpe_scope(
                hpe_root=args.hpe_root.resolve(),
                profile_arg=args.hpe_profile,
                dataset=dataset,
                session=session,
                declared_video=declared_video,
            )
            print(
                "HPE formal window: "
                f"logical=[{annotation_scope['core_start_logical_frame_inclusive']},"
                f"{annotation_scope['core_end_logical_frame_exclusive']}) "
                f"local=[{annotation_scope['gaze_local_start_frame_inclusive']},"
                f"{annotation_scope['gaze_local_end_frame_exclusive']}) "
                f"coverage={100 * annotation_scope['core_coverage_fraction']:.1f}%",
                flush=True,
            )
        start_frame = _resolve_start_frame(
            args.start_frame,
            annotation_scope=annotation_scope,
            declared_video=declared_video,
        )
        tracker_config = TrackerConfig(
            max_gap_s=args.max_gap_s,
            crowding_center_distance_ratio=args.crowding_center_distance_ratio,
        )
        tracker_config.validate()
        if (
            not math.isfinite(args.short_tracklet_max_duration_s)
            or args.short_tracklet_max_duration_s < 0
        ):
            raise IdentityAnnotationError(
                "--short-tracklet-max-duration-s must be finite and non-negative"
            )
        video_path: Path | None = None
        video_identity: dict[str, Any] | None = None
        if not args.prepare_only:
            if args.video is None:
                raise IdentityAnnotationError("--video is required unless --prepare-only is used")
            video_path = Path(os.path.abspath(os.fspath(args.video.expanduser())))
            print("Validating video identity without modifying annotations...", flush=True)
            _validate_video_file_without_opening(
                video_path, allow_offline=args.allow_offline_video
            )
            video_identity = _preflight_video(video_path, declared_video)
            _preflight_existing_video_binding(
                output_dir, video_identity, video_path
            )

        # Migration is the first state-changing operation.  All available
        # argument/configuration/video preflights above must succeed before the
        # existing human annotation revision is rewritten.
        if args.migrate_hpe_core:
            migration = migrate_legacy_annotation_to_hpe_core(
                output_dir,
                annotation_scope,
                expected_tracker_config=tracker_config,
            )
            print(
                "Migrated legacy annotation: "
                f"carried={migration['carried_tracklets']}, "
                f"reset={migration['reset_tracklets']}, "
                f"dropped={migration['dropped_tracklets']}",
                flush=True,
            )
            print(f"Legacy backup: {migration['backup_path']}", flush=True)

        print(
            "Preparing/resuming conservative tracklets; a first pass over a long session "
            "can take several minutes...",
            flush=True,
        )
        observations_path, manifest_path = prepare_annotation_artifact(
            gaze_csv,
            output_dir,
            dataset=dataset,
            session=session,
            config=tracker_config,
            annotation_scope=annotation_scope,
            short_tracklet_max_duration_s=args.short_tracklet_max_duration_s,
            overwrite=args.overwrite,
        )
        manifest = load_annotation_manifest(manifest_path)
        observations = validate_tracklet_observations(observations_path, manifest)
        print(f"Prepared {len(observations):,} observations")
        print(f"Prepared {len(manifest['annotations']):,} conservative tracklets")
        print(f"Artifact: {output_dir}")
        if args.prepare_only:
            export_identity_bridge(observations_path, manifest_path, output_dir)
            return 0
        if video_path is None or video_identity is None:
            raise AssertionError("video preflight must complete before annotation")
        IdentityAnnotator(
            video_path=video_path,
            observations_path=observations_path,
            manifest_path=manifest_path,
            output_dir=output_dir,
            video_identity=video_identity,
            start_frame=start_frame,
        ).run()
        return 0
    except IdentityAnnotationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
