"""Human review UI for the HPE-person to Transcript A/B/C/R crosswalk.

The data/provenance layer lives in
``hri_turn_taking.data.hpe_transcript_crosswalk``.  This script deliberately
keeps OpenCV lazy so artifact preparation and the normal training environment
do not require a video dependency.

The reviewer is shown two neutral views of the same logical instant:

* left: the already-confirmed Gaze identities (A/B/C/R), and
* right: HPE ``person`` numbers, with the current person highlighted.

The upstream HPE ``participant_map.csv`` is intentionally not shown or used as
an annotation default.  It is compared only by the core finalisation gate.
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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hri_turn_taking.data.hpe_transcript_crosswalk import (  # noqa: E402
    HPETranscriptCrosswalkError,
    bind_review_video_identities,
    finalize_crosswalk,
    load_crosswalk_review_state,
    prepare_crosswalk_artifact,
    set_sample_decision,
)

WINDOW_NAME = "HPE person -> Transcript A/B/C/R review"
_CAMERA_NAME_PREFIX = "Cam_"
_CLOUD_RECALL_FLAGS = (
    getattr(stat, "FILE_ATTRIBUTE_OFFLINE", 0x1000)
    | getattr(stat, "FILE_ATTRIBUTE_RECALL_ON_OPEN", 0x40000)
    | getattr(stat, "FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS", 0x400000)
)
_COLOURS = {
    "A": (80, 220, 80),
    "B": (255, 150, 40),
    "C": (220, 80, 220),
    "R": (40, 200, 255),
    "hpe": (180, 180, 180),
    "target": (40, 40, 255),
    "uncertain": (80, 80, 255),
}


def _require_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise HPETranscriptCrosswalkError(
            "OpenCV is required for the review UI. Install the annotation extra "
            "or use the interpreter recorded by the accepted artifact."
        ) from exc
    return cv2


def _windows_file_attributes(path: Path, timeout_s: float = 3.0) -> int:
    result: list[int] = []

    def query() -> None:
        result.append(int(ctypes.windll.kernel32.GetFileAttributesW(str(path))))

    worker = threading.Thread(target=query, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise HPETranscriptCrosswalkError(
            "timed out querying the OneDrive video placeholder; make it local first"
        )
    if not result:
        raise HPETranscriptCrosswalkError(f"could not query video attributes: {path}")
    return result[0]


def _validate_local_file_without_opening(path: Path) -> None:
    """Reject missing/directories/cloud placeholders without hydrating the file."""

    if os.name == "nt":
        attributes = _windows_file_attributes(path)
        # ctypes defaults to a signed C ``int`` return value, so Win32's
        # INVALID_FILE_ATTRIBUTES may arrive as either -1 or 0xFFFFFFFF.
        if attributes in {-1, 0xFFFFFFFF}:
            raise HPETranscriptCrosswalkError(f"video does not exist: {path}")
        if attributes & getattr(stat, "FILE_ATTRIBUTE_DIRECTORY", 0x10):
            raise HPETranscriptCrosswalkError(f"video path is a directory: {path}")
        if attributes & _CLOUD_RECALL_FLAGS:
            raise HPETranscriptCrosswalkError(
                "video is a OneDrive Offline placeholder. In Explorer choose "
                "'Always keep on this device' first; this tool will not trigger an "
                "implicit multi-GB download."
            )
    elif not path.is_file():
        raise HPETranscriptCrosswalkError(f"video does not exist: {path}")


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


def _normalise_camera(value: object) -> str:
    camera = str(value)
    if (
        not camera.startswith(_CAMERA_NAME_PREFIX)
        or not camera.removeprefix(_CAMERA_NAME_PREFIX).isdigit()
    ):
        raise HPETranscriptCrosswalkError(
            f"camera must use the Cam_N convention, got {camera!r}"
        )
    return camera


def _parse_hpe_video_assignments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        camera, separator, raw_path = value.partition("=")
        if not separator or not raw_path.strip():
            raise HPETranscriptCrosswalkError(
                "--hpe-video must use Cam_N=PATH, for example --hpe-video Cam_2=video.avi"
            )
        camera = _normalise_camera(camera.strip())
        if camera in result:
            raise HPETranscriptCrosswalkError(
                f"--hpe-video repeats camera {camera}; supply one exact video per camera"
            )
        result[camera] = Path(os.path.abspath(os.fspath(Path(raw_path.strip()).expanduser())))
    return result


def _json_object(value: object, label: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise HPETranscriptCrosswalkError(f"{label} must be a JSON object")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HPETranscriptCrosswalkError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise HPETranscriptCrosswalkError(f"{label} must be a JSON object")
    return parsed


def _source_kind(source: dict[str, Any]) -> str:
    kind = source.get("kind")
    if not isinstance(kind, str):
        raise HPETranscriptCrosswalkError("image source lacks a string kind")
    aliases = {
        "hash_bound_still": "still",
        "bound_still": "still",
        "image": "still",
        "still": "still",
        "video_frame": "video",
        "video": "video",
        "same_as_hpe": "same_as_hpe",
        "same_camera_gaze_declared_frame": "same_as_hpe",
        "cross_camera_gaze_declared_frame": "video",
    }
    try:
        return aliases[kind]
    except KeyError as exc:
        raise HPETranscriptCrosswalkError(f"unsupported image source kind: {kind!r}") from exc


def _expected_video_identity(source: dict[str, Any]) -> dict[str, Any]:
    candidate = source.get("video_identity", source.get("declared_video", source))
    if not isinstance(candidate, dict):
        raise HPETranscriptCrosswalkError("video source lacks an identity object")
    required = ("basename", "camera_dir", "fps", "frame_count", "width", "height")
    missing = [key for key in required if key not in candidate]
    if missing:
        raise HPETranscriptCrosswalkError(
            f"video source identity lacks required fields: {', '.join(missing)}"
        )
    return candidate


def _default_bound_video_path(source: dict[str, Any]) -> Path | None:
    """Return an already-bound path, without touching it or triggering recall."""

    identity = _expected_video_identity(source)
    for container in (source, identity):
        for key in (
            "last_verified_path",
            "first_bound_path",
            "local_path",
            "video_path",
            "path",
            "source_path",
            "declared_source_path",
        ):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    return None


def _validate_video_path_identity(path: Path, expected: dict[str, Any]) -> None:
    basename = str(expected["basename"])
    camera = _normalise_camera(expected["camera_dir"])
    if path.name.casefold() != basename.casefold():
        raise HPETranscriptCrosswalkError(
            f"video basename mismatch: expected {basename!r}, got {path.name!r}"
        )
    if path.parent.name.casefold() != camera.casefold():
        raise HPETranscriptCrosswalkError(
            f"video camera directory mismatch: expected {camera!r}, got {path.parent.name!r}"
        )


def _preflight_video(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Fully validate one raw video before any human-decision mutation."""

    _validate_local_file_without_opening(path)
    _validate_video_path_identity(path, expected)
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise HPETranscriptCrosswalkError(f"OpenCV cannot open video: {path}")
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
        raise HPETranscriptCrosswalkError("video reports an invalid FPS")
    fps_tolerance = max(1e-3, abs(float(expected["fps"])) * 1e-5)
    if abs(actual["fps"] - float(expected["fps"])) > fps_tolerance:
        raise HPETranscriptCrosswalkError(
            f"video FPS {actual['fps']} does not match bound metadata {expected['fps']}"
        )
    for field in ("frame_count", "width", "height"):
        if actual[field] != int(expected[field]):
            raise HPETranscriptCrosswalkError(
                f"video {field} {actual[field]} does not match bound metadata {expected[field]}"
            )
    size_bytes = path.stat().st_size
    if "size_bytes" in expected and size_bytes != int(expected["size_bytes"]):
        raise HPETranscriptCrosswalkError(
            f"video size {size_bytes} does not match bound metadata {expected['size_bytes']}"
        )
    sparse_sha256 = _sparse_video_fingerprint(path)
    expected_sparse = expected.get("sparse_sha256")
    if isinstance(expected_sparse, str) and sparse_sha256 != expected_sparse:
        raise HPETranscriptCrosswalkError(
            "video sparse fingerprint does not match the previously reviewed source"
        )
    return {
        "path": str(path.resolve()),
        "basename": str(expected["basename"]),
        "camera_dir": str(expected["camera_dir"]),
        **actual,
        "size_bytes": size_bytes,
        "fingerprint_algorithm": "sha256(size + first/middle/last 1MiB with offsets)",
        "sparse_sha256": sparse_sha256,
    }


@dataclass(frozen=True)
class _ResolvedSource:
    kind: str
    source: dict[str, Any]
    path: Path | None
    frame_index: int | None
    camera: str | None


def _resolve_sample_source(
    row: pd.Series | dict[str, Any],
    side: str,
    *,
    gaze_video: Path | None,
    hpe_videos: dict[str, Path],
) -> _ResolvedSource:
    """Resolve a sample side without opening an image or video.

    ``same_as_hpe`` is important for scenes where Gaze and HPE share Cam_2;
    it prevents the caller from needlessly hydrating a second copy.  For
    05_15, the left source is a Cam_1 video and the right source is an HPE
    camera, so the two paths remain distinct.
    """

    if side not in {"gaze", "hpe"}:
        raise ValueError("side must be gaze or hpe")
    value = row[f"{side}_image_source_json"]
    source = _json_object(value, f"{side}_image_source_json")
    kind = _source_kind(source)
    if kind == "same_as_hpe":
        if side != "gaze":
            raise HPETranscriptCrosswalkError("same_as_hpe is valid only for the gaze panel")
        resolved = _resolve_sample_source(
            row, "hpe", gaze_video=gaze_video, hpe_videos=hpe_videos
        )
        return _ResolvedSource(
            "same_as_hpe",
            source,
            resolved.path,
            resolved.frame_index,
            resolved.camera,
        )
    if kind == "still":
        raw_path = source.get("path")
        if not isinstance(raw_path, str):
            raise HPETranscriptCrosswalkError(f"{side} still source lacks path")
        local_frame_key = f"{side}_local_frame"
        default_frame = row[local_frame_key] if local_frame_key in row else row["logical_frame"]
        camera_value = source.get("camera", row.get("hpe_camera"))
        return _ResolvedSource(
            kind,
            source,
            Path(raw_path).resolve(),
            int(source.get("frame_index", default_frame)),
            str(camera_value) if camera_value is not None else None,
        )
    expected = _expected_video_identity(source)
    camera = _normalise_camera(expected["camera_dir"])
    explicit_path = gaze_video if side == "gaze" else hpe_videos.get(camera)
    path = explicit_path or _default_bound_video_path(source)
    if path is None:
        option = "--gaze-video" if side == "gaze" else f"--hpe-video {camera}=PATH"
        raise HPETranscriptCrosswalkError(
            f"sample {row['sample_id']} needs a raw {camera} video; supply {option}"
        )
    default_frame_key = "gaze_local_frame" if side == "gaze" else "hpe_local_frame"
    frame_value = source.get(
        "frame_index",
        row.get(default_frame_key, row.get("logical_frame")),
    )
    return _ResolvedSource(kind, source, path, int(frame_value), camera)


def _preflight_sample_videos(
    samples: pd.DataFrame,
    *,
    gaze_video: Path | None,
    hpe_videos: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    """Preflight every distinct required raw video, without mutating artifacts."""

    required: dict[Path, dict[str, Any]] = {}
    for _, row in samples.iterrows():
        for side in ("hpe", "gaze"):
            resolved = _resolve_sample_source(
                row, side, gaze_video=gaze_video, hpe_videos=hpe_videos
            )
            if resolved.kind not in {"video", "same_as_hpe"}:
                continue
            # A same_as_hpe source inherits the HPE source identity, not its own
            # tiny marker object.
            if resolved.kind == "same_as_hpe":
                source = _json_object(
                    row["hpe_image_source_json"], "hpe_image_source_json"
                )
                if _source_kind(source) != "video":
                    continue
            else:
                source = resolved.source
            expected = _expected_video_identity(source)
            assert resolved.path is not None
            prior = required.get(resolved.path)
            if prior is not None and prior != expected:
                raise HPETranscriptCrosswalkError(
                    f"one video path is claimed by conflicting source identities: {resolved.path}"
                )
            required[resolved.path] = expected
    return {
        str(path.resolve()): _preflight_video(path, expected)
        for path, expected in required.items()
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class _FrameReader:
    def __init__(self, cv2: Any) -> None:
        self.cv2 = cv2
        self.captures: dict[Path, Any] = {}

    def close(self) -> None:
        for capture in self.captures.values():
            capture.release()
        self.captures.clear()

    def read(self, resolved: _ResolvedSource) -> tuple[Any, dict[str, Any]]:
        if resolved.path is None:
            raise HPETranscriptCrosswalkError("resolved image source has no path")
        if resolved.kind == "still":
            expected_hash = resolved.source.get("sha256")
            if not isinstance(expected_hash, str):
                raise HPETranscriptCrosswalkError(
                    "a still may be displayed only when it has an immutable SHA-256 binding"
                )
            try:
                size_bytes = resolved.path.stat().st_size
                actual_hash = _sha256_file(resolved.path)
            except OSError as exc:
                raise HPETranscriptCrosswalkError(
                    f"cannot read hash-bound review still: {resolved.path}"
                ) from exc
            expected_size = resolved.source.get("size_bytes")
            if expected_size is not None and size_bytes != int(expected_size):
                raise HPETranscriptCrosswalkError(
                    f"review still size differs from its binding: {resolved.path}"
                )
            if actual_hash != expected_hash:
                raise HPETranscriptCrosswalkError(
                    f"review still differs from its hash binding: {resolved.path}"
                )
            image = self.cv2.imread(str(resolved.path), self.cv2.IMREAD_COLOR)
            if image is None:
                raise HPETranscriptCrosswalkError(
                    f"OpenCV cannot decode review still: {resolved.path}"
                )
            return image, {
                "kind": "still",
                "path": str(resolved.path),
                "frame_index": resolved.frame_index,
            }
        capture = self.captures.get(resolved.path)
        if capture is None:
            capture = self.cv2.VideoCapture(str(resolved.path))
            if not capture.isOpened():
                raise HPETranscriptCrosswalkError(
                    f"OpenCV cannot open preflighted video: {resolved.path}"
                )
            self.captures[resolved.path] = capture
        if resolved.frame_index is None or resolved.frame_index < 0:
            raise HPETranscriptCrosswalkError("video sample has an invalid frame index")
        capture.set(self.cv2.CAP_PROP_POS_FRAMES, float(resolved.frame_index))
        ok, image = capture.read()
        decoded = int(round(capture.get(self.cv2.CAP_PROP_POS_FRAMES))) - 1
        if not ok or image is None:
            raise HPETranscriptCrosswalkError(
                f"cannot decode frame {resolved.frame_index} from {resolved.path}"
            )
        if decoded != resolved.frame_index:
            raise HPETranscriptCrosswalkError(
                f"video seek decoded frame {decoded}, expected exact frame {resolved.frame_index}"
            )
        return image, {
            "kind": "video",
            "path": str(resolved.path),
            "requested_frame": resolved.frame_index,
            "decoded_frame": decoded,
        }


def _boxes(value: object, label: str) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise HPETranscriptCrosswalkError(f"{label} is invalid JSON") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise HPETranscriptCrosswalkError(f"{label} must be a JSON array of objects")
    return value


def _bbox_coordinates(box: dict[str, Any]) -> tuple[int, int, int, int]:
    for keys in (
        ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"),
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
        ("x0", "y0", "x1", "y1"),
    ):
        if all(key in box for key in keys):
            return tuple(int(round(float(box[key]))) for key in keys)  # type: ignore[return-value]
    bbox = box.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return tuple(int(round(float(value))) for value in bbox)  # type: ignore[return-value]
    raise HPETranscriptCrosswalkError("bbox object lacks four coordinates")


def _draw_boxes(
    cv2: Any,
    image: Any,
    boxes: list[dict[str, Any]],
    *,
    panel: str,
    target_person: str,
) -> Any:
    result = image.copy()
    for box in boxes:
        x0, y0, x1, y1 = _bbox_coordinates(box)
        if panel == "gaze":
            label = str(box.get("participant", box.get("label", "?")))
            colour = _COLOURS.get(label, _COLOURS["uncertain"])
        else:
            person = str(box.get("hpe_person", box.get("person", "?")))
            label = f"HPE {person}"
            colour = _COLOURS["target"] if person == target_person else _COLOURS["hpe"]
        thickness = 4 if panel == "hpe" and colour == _COLOURS["target"] else 2
        cv2.rectangle(result, (x0, y0), (x1, y1), colour, thickness)
        cv2.putText(
            result,
            label,
            (x0, max(24, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colour,
            2,
            cv2.LINE_AA,
        )
    return result


def _fit_panel(cv2: Any, image: Any, width: int = 760, height: int = 600) -> Any:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
    )
    canvas = cv2.copyMakeBorder(
        resized,
        0,
        height - resized.shape[0],
        0,
        width - resized.shape[1],
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )
    return canvas


def _expanded_crop(image: Any, box: dict[str, Any], expansion: float = 2.5) -> Any:
    x0, y0, x1, y1 = _bbox_coordinates(box)
    centre_x = (x0 + x1) / 2
    centre_y = (y0 + y1) / 2
    half_width = max(1.0, (x1 - x0) * expansion / 2)
    half_height = max(1.0, (y1 - y0) * expansion / 2)
    left = max(0, int(math.floor(centre_x - half_width)))
    top = max(0, int(math.floor(centre_y - half_height)))
    right = min(image.shape[1], int(math.ceil(centre_x + half_width)))
    bottom = min(image.shape[0], int(math.ceil(centre_y + half_height)))
    if right <= left or bottom <= top:
        raise HPETranscriptCrosswalkError("cannot crop an empty bounding box")
    return image[top:bottom, left:right].copy()


def _crop_tile(
    cv2: Any,
    image: Any | None,
    *,
    label: str,
    width: int,
    height: int,
    colour: tuple[int, int, int],
) -> Any:
    import numpy as np

    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    if image is not None and image.size:
        scale = min(width / image.shape[1], (height - 32) / image.shape[0])
        resized = cv2.resize(
            image,
            (
                max(1, round(image.shape[1] * scale)),
                max(1, round(image.shape[0] * scale)),
            ),
        )
        x = (width - resized.shape[1]) // 2
        y = 30 + (height - 30 - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    else:
        cv2.putText(
            canvas,
            "MISSING (no exact box)",
            (8, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            _COLOURS["uncertain"],
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        colour,
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), colour, 1)
    return canvas


def _crop_strip(
    cv2: Any,
    *,
    gaze_image: Any,
    gaze_boxes: list[dict[str, Any]],
    hpe_image: Any,
    hpe_boxes: list[dict[str, Any]],
    target_person: str,
) -> Any:
    gaze_tiles = []
    for participant in ("A", "B", "C", "R"):
        exact = next(
            (
                box
                for box in gaze_boxes
                if str(box.get("participant", box.get("label", ""))) == participant
            ),
            None,
        )
        crop = _expanded_crop(gaze_image, exact) if exact is not None else None
        gaze_tiles.append(
            _crop_tile(
                cv2,
                crop,
                label=f"Gaze {participant}",
                width=190,
                height=210,
                colour=_COLOURS[participant],
            )
        )
    target_box = next(
        (
            box
            for box in hpe_boxes
            if str(box.get("hpe_person", box.get("person", ""))) == target_person
        ),
        None,
    )
    target_crop = _expanded_crop(hpe_image, target_box) if target_box is not None else None
    target_tile = _crop_tile(
        cv2,
        target_crop,
        label=f"TARGET HPE {target_person} (2.5x bbox crop)",
        width=760,
        height=210,
        colour=_COLOURS["target"],
    )
    return cv2.hconcat([*gaze_tiles, target_tile])


class CrosswalkReviewer:
    def __init__(
        self,
        *,
        samples_path: Path,
        manifest_path: Path,
        samples: pd.DataFrame,
        manifest: dict[str, Any],
        gaze_video: Path | None,
        hpe_videos: dict[str, Path],
    ) -> None:
        self.cv2 = _require_cv2()
        self.samples_path = samples_path
        self.manifest_path = manifest_path
        self.samples = samples.reset_index(drop=True)
        self.manifest = manifest
        self.gaze_video = gaze_video
        self.hpe_videos = hpe_videos
        self.reader = _FrameReader(self.cv2)
        self.index = self._first_unreviewed_index()
        self.undo_stack: list[tuple[str, str | None, str, str]] = []
        self.status_message = (
            "A/B/C/R=confirm sample, U=uncertain, N/P=navigate, "
            "Z=undo, F=finalize, Q=save exit"
        )
        self.finalize_armed = False
        self.show_help = False
        self.display_evidence: dict[str, dict[str, Any]] = {}

    def _decision(self, sample_id: str) -> tuple[str | None, str]:
        decisions = self.manifest.get("sample_decisions", self.manifest.get("decisions", {}))
        value = decisions.get(sample_id, {}) if isinstance(decisions, dict) else {}
        if not isinstance(value, dict):
            return None, "unreviewed"
        label = value.get("label") or value.get("participant")
        status = str(value.get("status", value.get("annotation_status", "unreviewed")))
        return str(label) if label is not None else None, status

    def _first_unreviewed_index(self) -> int:
        for index, row in self.samples.iterrows():
            if self._decision(str(row["sample_id"]))[1] == "unreviewed":
                return int(index)
        return 0

    def _reload(self) -> None:
        self.manifest, self.samples = load_crosswalk_review_state(self.manifest_path)
        self.samples = self.samples.reset_index(drop=True)

    def _set_decision(self, label: str | None, status: str) -> None:
        row = self.samples.iloc[self.index]
        sample_id = str(row["sample_id"])
        # The displayed raw-video frame numbers are included in the audit note;
        # still sources are already hash-bound by the immutable sample row.
        displayed = self.display_evidence.get(sample_id)
        if displayed is None:
            raise HPETranscriptCrosswalkError(
                "sample cannot be labelled before its exact evidence frame is displayed"
            )
        note = json.dumps(
            {
                "review": "manual_dual_panel",
                "logical_frame": int(row["logical_frame"]),
                "displayed_evidence": displayed,
            },
            sort_keys=True,
        )
        previous = set_sample_decision(
            self.manifest_path,
            sample_id=sample_id,
            label=label,
            status=status,
            note=note,
        )
        self.undo_stack.append(
            (
                sample_id,
                previous.get("label"),
                str(previous.get("status", "unreviewed")),
                str(previous.get("note", "")),
            )
        )
        self._reload()
        self.finalize_armed = False
        self.status_message = f"saved {sample_id}: {label or status}"
        self._navigate(1, unreviewed_only=True)

    def _navigate(self, step: int, *, unreviewed_only: bool = False) -> None:
        count = len(self.samples)
        if count == 0:
            return
        for offset in range(1, count + 1):
            candidate = (self.index + step * offset) % count
            if not unreviewed_only:
                self.index = candidate
                return
            sample_id = str(self.samples.iloc[candidate]["sample_id"])
            if self._decision(sample_id)[1] == "unreviewed":
                self.index = candidate
                return
        self.status_message = "all samples have a decision; press F to finalize"

    def _undo(self) -> None:
        if not self.undo_stack:
            self.status_message = "nothing to undo from this run"
            return
        sample_id, label, status, note = self.undo_stack.pop()
        set_sample_decision(
            self.manifest_path,
            sample_id=sample_id,
            label=label,
            status=status,
            note=note,
        )
        self._reload()
        matches = self.samples.index[self.samples["sample_id"].astype(str) == sample_id]
        if len(matches):
            self.index = int(matches[0])
        self.finalize_armed = False
        self.status_message = f"restored {sample_id}: {label or status}"

    def _finalize(self) -> bool:
        acknowledgement = "I independently confirmed the Transcript semantic crosswalk"
        try:
            # Startup preflight is not enough: a local/cloud-backed video can be
            # removed or replaced while the review window is open.  Recompute
            # every required identity immediately before *each* finalize
            # attempt (including both presses of the mismatch acknowledgement).
            video_identities = _preflight_sample_videos(
                self.samples,
                gaze_video=self.gaze_video,
                hpe_videos=self.hpe_videos,
            )
            bind_review_video_identities(
                self.samples_path,
                self.manifest_path,
                bindings=video_identities,
            )
            output = finalize_crosswalk(
                self.manifest_path,
                upstream_mismatch_acknowledgement=(
                    acknowledgement if self.finalize_armed else None
                ),
            )
        except HPETranscriptCrosswalkError as exc:
            message = str(exc)
            lowered = message.casefold()
            mismatch_gate = (
                "upstream" in lowered
                and (
                    "mismatch" in lowered
                    or "differ" in lowered
                    or "acknowledgement" in lowered
                )
            )
            if mismatch_gate:
                if not self.finalize_armed:
                    self.finalize_armed = True
                    self.status_message = (
                        "Crosswalk differs from upstream HPE proposal. Inspect the panels; "
                        "press F again to explicitly confirm your independent review."
                    )
                    return False
            self.finalize_armed = False
            self.status_message = message
            return False
        print(f"Finalized HPE/Transcript crosswalk: {output}", flush=True)
        self.status_message = f"finalized: {output}"
        return True

    def _render(self) -> Any:
        row = self.samples.iloc[self.index]
        hpe_source = _resolve_sample_source(
            row, "hpe", gaze_video=self.gaze_video, hpe_videos=self.hpe_videos
        )
        gaze_source = _resolve_sample_source(
            row, "gaze", gaze_video=self.gaze_video, hpe_videos=self.hpe_videos
        )
        hpe_image, hpe_decoded = self.reader.read(hpe_source)
        if gaze_source.kind == "same_as_hpe":
            gaze_image = hpe_image.copy()
            gaze_decoded = dict(hpe_decoded)
            gaze_decoded["kind"] = "same_as_hpe"
        else:
            gaze_image, gaze_decoded = self.reader.read(gaze_source)
        target_person = str(row["hpe_person"])
        gaze_boxes = _boxes(row["gaze_confirmed_boxes_json"], "gaze_confirmed_boxes_json")
        hpe_boxes = _boxes(row["all_hpe_boxes_json"], "all_hpe_boxes_json")
        sample_id = str(row["sample_id"])
        self.display_evidence[sample_id] = {
            "gaze": gaze_decoded,
            "hpe": hpe_decoded,
        }
        gaze_panel = _draw_boxes(
            self.cv2, gaze_image, gaze_boxes, panel="gaze", target_person=target_person
        )
        hpe_panel = _draw_boxes(
            self.cv2, hpe_image, hpe_boxes, panel="hpe", target_person=target_person
        )
        gaze_panel = _fit_panel(self.cv2, gaze_panel)
        hpe_panel = _fit_panel(self.cv2, hpe_panel)
        full_views = self.cv2.hconcat([gaze_panel, hpe_panel])
        crops = _crop_strip(
            self.cv2,
            gaze_image=gaze_image,
            gaze_boxes=gaze_boxes,
            hpe_image=hpe_image,
            hpe_boxes=hpe_boxes,
            target_person=target_person,
        )
        canvas = self.cv2.vconcat([full_views, crops])
        canvas = self.cv2.copyMakeBorder(
            canvas, 70, 110, 0, 0, self.cv2.BORDER_CONSTANT, value=(20, 20, 20)
        )
        label, status = self._decision(sample_id)
        title = (
            f"LEFT: confirmed Gaze A/B/C/R    RIGHT: neutral HPE persons | "
            f"target HPE {target_person} | sample {self.index + 1}/{len(self.samples)} "
            f"({row['phase']}) | current={label or status}"
        )
        self.cv2.putText(
            canvas,
            title,
            (20, 38),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (235, 235, 235),
            2,
            self.cv2.LINE_AA,
        )
        evidence = (
            f"logical={int(row['logical_frame'])} | gaze={gaze_decoded} | hpe={hpe_decoded}"
        )
        self.cv2.putText(
            canvas,
            evidence[:180],
            (20, canvas.shape[0] - 66),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (190, 190, 190),
            1,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            canvas,
            self.status_message[:180],
            (20, canvas.shape[0] - 28),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 220, 255) if self.finalize_armed else (235, 235, 235),
            1,
            self.cv2.LINE_AA,
        )
        if self.show_help:
            help_lines = (
                "A/B/C/R: confirm THIS sample (never prefilled from HPE)",
                "U: uncertain/missing   N/P: next/previous   Z: undo",
                "F: finalize (press twice only when upstream proposal differs)",
                "H: hide help   Q/Esc: safe exit without finalizing",
            )
            for line_number, line in enumerate(help_lines):
                self.cv2.putText(
                    canvas,
                    line,
                    (20, 78 + 28 * line_number),
                    self.cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    self.cv2.LINE_AA,
                )
        return canvas

    def run(self) -> None:
        self.cv2.namedWindow(WINDOW_NAME, self.cv2.WINDOW_AUTOSIZE)
        try:
            while True:
                self.cv2.imshow(WINDOW_NAME, self._render())
                key = self.cv2.waitKeyEx(0)
                if key in (ord("q"), ord("Q"), 27):
                    print("Review state is safely preserved; crosswalk was not finalized.")
                    return
                if key in tuple(map(ord, "abcrABCR")):
                    self._set_decision(chr(key).upper(), "confirmed")
                elif key in (ord("u"), ord("U")):
                    self._set_decision(None, "uncertain")
                elif key in (ord("n"), ord("N")):
                    self.finalize_armed = False
                    self._navigate(1)
                elif key in (ord("p"), ord("P")):
                    self.finalize_armed = False
                    self._navigate(-1)
                elif key in (ord("z"), ord("Z")):
                    self._undo()
                elif key in (ord("h"), ord("H")):
                    self.show_help = not self.show_help
                elif key in (ord("f"), ord("F")) and self._finalize():
                    return
                if self.cv2.getWindowProperty(WINDOW_NAME, self.cv2.WND_PROP_VISIBLE) < 1:
                    print("Review window closed; state is safely preserved.")
                    return
        finally:
            self.reader.close()
            self.cv2.destroyAllWindows()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review HPE person numbers against canonical Transcript A/B/C/R identities."
    )
    parser.add_argument("--scene", required=True, help="accepted scene, e.g. 03_20__4")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPOSITORY_ROOT / "docs" / "IDENTITY_DATA_STATUS.md",
    )
    parser.add_argument(
        "--hpe-root",
        type=Path,
        default=REPOSITORY_ROOT.parent / "Head_pose_extraction",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "annotations" / "hpe_transcript_crosswalk",
    )
    parser.add_argument("--gaze-video", type=Path)
    parser.add_argument(
        "--hpe-video",
        action="append",
        default=[],
        metavar="Cam_N=PATH",
        help="raw HPE-camera video fallback; repeat once per required camera",
    )
    parser.add_argument(
        "--hpe-camera",
        action="append",
        default=[],
        metavar="Cam_N",
        help="limit HPE evidence cameras (default: core chooses a supported camera)",
    )
    parser.add_argument("--samples-per-person", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        hpe_root = args.hpe_root.resolve()
        ledger = args.ledger.resolve()
        output_dir = (args.output_root / args.scene).resolve()
        delivery_dir = (hpe_root / "outputs" / "delivery" / args.scene).resolve()
        gaze_identity_dir = (
            REPOSITORY_ROOT / "data" / "annotations" / "identity" / args.scene
        ).resolve()
        hpe_cameras = [_normalise_camera(value) for value in args.hpe_camera]
        if len(set(hpe_cameras)) != len(hpe_cameras):
            raise HPETranscriptCrosswalkError("--hpe-camera contains a duplicate camera")
        if args.samples_per_person < 3:
            raise HPETranscriptCrosswalkError("--samples-per-person must be at least 3")
        hpe_videos = _parse_hpe_video_assignments(args.hpe_video)
        gaze_video = (
            Path(os.path.abspath(os.fspath(args.gaze_video.expanduser())))
            if args.gaze_video is not None
            else None
        )

        print(
            f"Validating accepted HPE/Gaze sources for {args.scene}; "
            "large CSV hashes can take one or two minutes...",
            flush=True,
        )
        samples_path, manifest_path = prepare_crosswalk_artifact(
            scene=args.scene,
            ledger_path=ledger,
            delivery_dir=delivery_dir,
            hpe_repo_root=hpe_root,
            gaze_identity_dir=gaze_identity_dir,
            output_dir=output_dir,
            hpe_cameras=hpe_cameras or None,
            samples_per_person=args.samples_per_person,
        )
        # ``prepare_crosswalk_artifact`` has already replayed the complete
        # HPE/Gaze authority chain (including on resume).  Reload only the
        # immutable local review state here so startup does not hash every
        # large upstream CSV a second time.
        manifest, samples = load_crosswalk_review_state(manifest_path)
        print(
            f"Prepared/resumed {len(samples)} review samples in {output_dir}",
            flush=True,
        )
        if args.prepare_only:
            return 0

        # This is deliberately before the first call that can change a human
        # sample decision.  Any missing/offline/wrong video therefore leaves
        # the existing annotation state byte-for-byte unchanged.
        video_identities = _preflight_sample_videos(
            samples,
            gaze_video=gaze_video,
            hpe_videos=hpe_videos,
        )
        bind_review_video_identities(
            samples_path,
            manifest_path,
            bindings=video_identities,
        )
        manifest, samples = load_crosswalk_review_state(samples_path, manifest_path)
        CrosswalkReviewer(
            samples_path=samples_path,
            manifest_path=manifest_path,
            samples=samples,
            manifest=manifest,
            gaze_video=gaze_video,
            hpe_videos=hpe_videos,
        ).run()
        return 0
    except HPETranscriptCrosswalkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
