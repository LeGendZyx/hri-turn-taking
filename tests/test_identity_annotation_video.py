from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import tools.annotate_person_ids as annotator

from hri_turn_taking.data.identity_annotation import IdentityAnnotationError


class _FakeCapture:
    def __init__(self, properties: dict[int, float], *, opened: bool = True) -> None:
        self.properties = properties
        self.opened = opened
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV
        return self.opened

    def get(self, key: int) -> float:
        return self.properties[key]

    def release(self) -> None:
        self.released = True


class _FakeCV2:
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_COUNT = 2
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, *, fps: float = 30.0, frames: int = 100, width: int = 320) -> None:
        self.capture = _FakeCapture(
            {
                self.CAP_PROP_FPS: fps,
                self.CAP_PROP_FRAME_COUNT: float(frames),
                self.CAP_PROP_FRAME_WIDTH: float(width),
                self.CAP_PROP_FRAME_HEIGHT: 180.0,
            }
        )

    def VideoCapture(self, _path: str) -> _FakeCapture:  # noqa: N802 - mirrors OpenCV
        return self.capture


def _declared() -> dict[str, object]:
    return {
        "source_path": "Data/05_15/6cams/2/Cam_1/correct.avi",
        "basename": "correct.avi",
        "dataset": "05_15",
        "session": "2",
        "camera_dir": "Cam_1",
        "fps": 30.0,
        "frame_count": 100,
        "width": 320,
        "height": 180,
    }


def _video_path(tmp_path: Path, *, session: str = "2", camera: str = "Cam_1") -> Path:
    path = tmp_path / session / camera / "correct.avi"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video-bytes" * 100)
    return path


def test_find_hpe_run_metadata_follows_confirmed_delivery_chain(tmp_path: Path) -> None:
    profile = tmp_path / "calibration" / "scenario_geometry_profiles" / "05_15__2.json"
    profile.parent.mkdir(parents=True)
    profile.write_text("{}", encoding="utf-8")
    marker = tmp_path / "review_artifacts" / "formal" / "pose" / "Cam_1"
    marker.mkdir(parents=True)
    marker_path = marker / "run_metadata.json"
    marker_path.write_text(
        json.dumps(
            {
                "source_fps": 30.0,
                "processed_frames": 100,
            }
        ),
        encoding="utf-8",
    )
    validation_dir = tmp_path / "review_artifacts" / "formal" / "validation"
    validation_dir.mkdir(parents=True)
    validation = {
        "csv_root": "review_artifacts/formal/pose",
        "geometry_cardinality_gate": {
            "profile": {
                "repo_path": "calibration/scenario_geometry_profiles/05_15__2.json",
                "sha256": annotator.sha256_file(profile),
            },
            "day": "05_15",
            "scenario": "2",
            "production_pass": True,
            "coverage_complete": True,
        },
        "pose_sources": {
            "batch_schema": "v2_gate",
            "recording_day": "05_15",
            "scenario": "2",
            "cameras": {
                "Cam_1": {
                    "metadata": "run_metadata.json",
                    "metadata_sha256": annotator.sha256_file(marker_path),
                    "bound_day": "05_15",
                    "bound_scenario": "2",
                    "bound_camera": "Cam_1",
                    "source_fps": 30.0,
                    "processed_frames": 100,
                    "extraction_scope": {"start_frame": 0, "max_frames": None},
                }
            },
        },
    }
    validation_path = validation_dir / "validation_complete.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    delivery = tmp_path / "outputs" / "delivery" / "05_15__2"
    delivery.mkdir(parents=True)
    (delivery / "mapping.json").write_text(
        json.dumps(
            {
                "review_status": "confirmed",
                "delivery": "05_15__2",
                "group": "05_15",
                "scenario": "2",
                "sources": {
                    "validation_dir": "review_artifacts/formal/validation"
                },
            }
        ),
        encoding="utf-8",
    )

    result, authority = annotator._find_hpe_run_metadata(
        tmp_path,
        profile=profile,
        dataset="05_15",
        session="2",
        declared_video=_declared(),
    )

    assert result == marker_path.resolve()
    assert authority["validation_path"] == str(validation_path.resolve())

    validation["geometry_cardinality_gate"]["required_active_cameras"] = ["Cam_2"]
    validation["pose_sources"]["cameras"] = {}
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    axis_marker = (
        tmp_path
        / "outputs_v2"
        / "05_15"
        / "6cams"
        / "2"
        / "Cam_1"
        / "run_metadata.json"
    )
    axis_marker.parent.mkdir(parents=True)
    axis_marker.write_text(
        json.dumps(
            {
                "input": "F:/share/05_15/6cams/2/Cam_1/correct.avi",
                "segments": ["F:/share/05_15/6cams/2/Cam_1/correct.avi"],
                "segments_decoded_frames": None,
                "source_fps": 30.0,
                "processed_frames": 100,
                "width": 320,
                "height": 180,
                "params": {"start_frame": 0, "max_frames": None},
            }
        ),
        encoding="utf-8",
    )

    excluded_result, excluded_authority = annotator._find_hpe_run_metadata(
        tmp_path,
        profile=profile,
        dataset="05_15",
        session="2",
        declared_video=_declared(),
    )

    assert excluded_result == axis_marker.resolve()
    assert (
        excluded_authority["kind"]
        == "hpe_confirmed_scene_excluded_camera_axis_v1"
    )


def test_migration_and_overwrite_flags_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotate_person_ids.py",
            "--gaze-csv",
            "gaze.csv",
            "--migrate-hpe-core",
            "--overwrite",
        ],
    )

    with pytest.raises(SystemExit):
        annotator._parse_args()


def test_start_frame_is_validated_against_half_open_scope_before_ui() -> None:
    scope = {
        "kind": "hpe_confirmed_core_window",
        "gaze_local_start_frame_inclusive": 10,
        "gaze_local_end_frame_exclusive": 20,
    }

    assert (
        annotator._resolve_start_frame(
            None, annotation_scope=scope, declared_video=_declared()
        )
        == 10
    )
    with pytest.raises(IdentityAnnotationError, match=r"\[10, 20\)"):
        annotator._resolve_start_frame(
            20, annotation_scope=scope, declared_video=_declared()
        )


@pytest.mark.parametrize(
    ("session", "camera", "name", "message"),
    [
        ("2", "Cam_1", "wrong.avi", "basename"),
        ("2", "Cam_2", "correct.avi", "camera directory"),
        ("3", "Cam_1", "correct.avi", "session directory"),
    ],
)
def test_path_identity_rejects_wrong_segment_camera_or_session(
    tmp_path: Path, session: str, camera: str, name: str, message: str
) -> None:
    path = tmp_path / session / camera / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video")

    with pytest.raises(IdentityAnnotationError, match=message):
        annotator._validate_video_path_identity(path, _declared())


@pytest.mark.parametrize(
    ("fake", "message"),
    [
        (_FakeCV2(fps=25.0), "FPS"),
        (_FakeCV2(frames=101), "frame_count"),
        (_FakeCV2(width=640), "width"),
    ],
)
def test_preflight_rejects_stream_metadata_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakeCV2,
    message: str,
) -> None:
    video = _video_path(tmp_path)
    monkeypatch.setattr(annotator, "_require_cv2", lambda: fake)

    with pytest.raises(IdentityAnnotationError, match=message):
        annotator._preflight_video(video, _declared())
    assert fake.capture.released


def test_preflight_binds_sparse_content_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_path(tmp_path)
    fake = _FakeCV2()
    monkeypatch.setattr(annotator, "_require_cv2", lambda: fake)

    first = annotator._preflight_video(video, _declared())
    second = annotator._preflight_video(video, _declared())

    assert first["sparse_sha256"] == second["sparse_sha256"]
    assert first["size_bytes"] == video.stat().st_size
    assert "full" not in str(first["fingerprint_algorithm"]).lower()


def test_offline_placeholder_is_rejected_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_path(tmp_path)
    monkeypatch.setattr(
        annotator,
        "_windows_file_attributes",
        lambda _path: annotator._CLOUD_RECALL_FLAGS,
    )

    with pytest.raises(IdentityAnnotationError, match="Offline placeholder"):
        annotator._validate_video_file_without_opening(video, allow_offline=False)

    annotator._validate_video_file_without_opening(video, allow_offline=True)


def test_resume_rejects_a_different_bound_video(tmp_path: Path) -> None:
    video = _video_path(tmp_path)
    identity = {
        "identity_version": 1,
        "declared_source_path": _declared()["source_path"],
        "basename": video.name,
        "camera_dir": video.parent.name,
        "fps": 30.0,
        "frame_count": 100,
        "width": 320,
        "height": 180,
        "size_bytes": video.stat().st_size,
        "fingerprint_algorithm": "sparse-test",
        "sparse_sha256": "first",
    }
    manifest: dict[str, object] = {}
    annotator._bind_video_identity(manifest, identity, video)

    replacement = dict(identity)
    replacement["sparse_sha256"] = "second"
    with pytest.raises(IdentityAnnotationError, match="differs from the video bound"):
        annotator._bind_video_identity(manifest, replacement, video)


def test_existing_video_binding_preflight_is_read_only(tmp_path: Path) -> None:
    output = tmp_path / "annotation"
    output.mkdir()
    video = _video_path(tmp_path)
    recorded = {
        "identity_version": 1,
        "declared_source_path": _declared()["source_path"],
        "basename": video.name,
        "camera_dir": video.parent.name,
        "fps": 30.0,
        "frame_count": 100,
        "width": 320,
        "height": 180,
        "size_bytes": video.stat().st_size,
        "fingerprint_algorithm": "sparse-test",
        "sparse_sha256": "first",
    }
    manifest_path = output / "identity_annotations.json"
    manifest_path.write_text(json.dumps({"video": recorded}), encoding="utf-8")
    before = manifest_path.read_bytes()
    replacement = dict(recorded)
    replacement["sparse_sha256"] = "second"

    with pytest.raises(IdentityAnnotationError, match="differs from the video bound"):
        annotator._preflight_existing_video_binding(output, replacement, video)

    assert manifest_path.read_bytes() == before


def test_relocated_identical_video_binding_is_allowed(tmp_path: Path) -> None:
    first_path = _video_path(tmp_path / "first")
    second_path = _video_path(tmp_path / "second")
    identity = {
        "identity_version": 1,
        "declared_source_path": _declared()["source_path"],
        "basename": first_path.name,
        "camera_dir": first_path.parent.name,
        "fps": 30.0,
        "frame_count": 100,
        "width": 320,
        "height": 180,
        "size_bytes": first_path.stat().st_size,
        "fingerprint_algorithm": "sparse-test",
        "sparse_sha256": "same",
    }
    manifest: dict[str, object] = {}

    annotator._bind_video_identity(manifest, identity, first_path)
    annotator._bind_video_identity(manifest, identity, second_path)

    assert manifest["video"]["first_bound_path"] == str(first_path.resolve())
    assert manifest["video"]["last_verified_path"] == str(second_path.resolve())


def test_seek_clears_a_tracklet_that_is_not_visible() -> None:
    ui = annotator.IdentityAnnotator.__new__(annotator.IdentityAnnotator)
    ui.current_frame = 0
    ui.frame_count = 3
    ui.selected_tracklet = "p1"
    ui.frame_rows = {
        0: pd.DataFrame({"tracklet_id": ["p1"]}),
        1: pd.DataFrame({"tracklet_id": ["p2"]}),
    }
    ui.dirty = False

    ui._seek(1)

    assert ui.selected_tracklet is None
    assert ui.dirty


def test_clicking_blank_space_clears_the_previous_selection() -> None:
    ui = annotator.IdentityAnnotator.__new__(annotator.IdentityAnnotator)
    ui.current_frame = 0
    ui.display_scale = 1.0
    ui.selected_tracklet = "p1"
    ui.status_message = ""
    ui.dirty = False
    ui.frame_rows = {
        0: pd.DataFrame(
            {
                "tracklet_id": ["p1"],
                "bbox_x0": [10.0],
                "bbox_y0": [10.0],
                "bbox_x1": [30.0],
                "bbox_y1": [30.0],
            }
        )
    }

    ui._select_at(100, 100)

    assert ui.selected_tracklet is None
    assert "selection cleared" in ui.status_message
