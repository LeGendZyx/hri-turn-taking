from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import tools.review_hpe_transcript_crosswalk as reviewer

from hri_turn_taking.data.hpe_transcript_crosswalk import HPETranscriptCrosswalkError


class _FakeCapture:
    def __init__(
        self,
        properties: dict[int, float],
        *,
        opened: bool = True,
        decoded_frame: int | None = None,
    ) -> None:
        self.properties = properties
        self.opened = opened
        self.released = False
        self.decoded_frame = decoded_frame
        self.requested_frame: int | None = None

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV
        return self.opened

    def get(self, key: int) -> float:
        if key == _FakeCV2.CAP_PROP_POS_FRAMES and self.requested_frame is not None:
            decoded = self.requested_frame if self.decoded_frame is None else self.decoded_frame
            return float(decoded + 1)
        return self.properties[key]

    def set(self, key: int, value: float) -> bool:
        assert key == _FakeCV2.CAP_PROP_POS_FRAMES
        self.requested_frame = int(value)
        return True

    def read(self) -> tuple[bool, object]:
        return True, object()

    def release(self) -> None:
        self.released = True


class _FakeCV2:
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_COUNT = 2
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_FRAMES = 5

    def __init__(
        self,
        *,
        fps: float = 30.0,
        frames: int = 100,
        width: int = 320,
        height: int = 180,
        decoded_frame: int | None = None,
    ) -> None:
        self.capture = _FakeCapture(
            {
                self.CAP_PROP_FPS: fps,
                self.CAP_PROP_FRAME_COUNT: float(frames),
                self.CAP_PROP_FRAME_WIDTH: float(width),
                self.CAP_PROP_FRAME_HEIGHT: float(height),
            },
            decoded_frame=decoded_frame,
        )

    def VideoCapture(self, _path: str) -> _FakeCapture:  # noqa: N802 - mirrors OpenCV
        return self.capture


def _identity(camera: str, basename: str = "correct.avi") -> dict[str, object]:
    return {
        "basename": basename,
        "camera_dir": camera,
        "fps": 30.0,
        "frame_count": 100,
        "width": 320,
        "height": 180,
    }


def _video_path(
    tmp_path: Path,
    *,
    camera: str,
    basename: str = "correct.avi",
) -> Path:
    path = tmp_path / camera / basename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video-bytes" * 100)
    return path


def _video_source(camera: str, frame: int) -> str:
    return json.dumps(
        {
            "kind": "video",
            "frame_index": frame,
            "video_identity": _identity(camera),
        }
    )


def _sample_row(
    *,
    gaze_source: str,
    hpe_source: str,
    hpe_camera: str = "Cam_2",
) -> dict[str, object]:
    return {
        "sample_id": "person1-middle",
        "hpe_person": "1",
        "phase": "middle",
        "logical_frame": 50,
        "gaze_local_frame": 50,
        "hpe_camera": hpe_camera,
        "hpe_bbox_json": "{}",
        "all_hpe_boxes_json": "[]",
        "gaze_confirmed_boxes_json": "[]",
        "gaze_image_source_json": gaze_source,
        "hpe_image_source_json": hpe_source,
    }


def test_parse_args_exposes_required_review_inputs() -> None:
    args = reviewer._parse_args(
        [
            "--scene",
            "05_15__2",
            "--prepare-only",
            "--ledger",
            "ledger.md",
            "--hpe-root",
            "hpe",
            "--output-root",
            "crosswalks",
            "--gaze-video",
            "gaze.avi",
            "--hpe-video",
            "Cam_2=hpe.avi",
            "--hpe-video",
            "Cam_3=hpe3.avi",
        ]
    )

    assert args.scene == "05_15__2"
    assert args.prepare_only
    assert args.hpe_video == ["Cam_2=hpe.avi", "Cam_3=hpe3.avi"]


def test_hpe_video_assignments_reject_duplicate_or_malformed_camera() -> None:
    with pytest.raises(HPETranscriptCrosswalkError, match="repeats camera"):
        reviewer._parse_hpe_video_assignments(
            ["Cam_2=first.avi", "Cam_2=second.avi"]
        )
    with pytest.raises(HPETranscriptCrosswalkError, match="Cam_N convention"):
        reviewer._parse_hpe_video_assignments(["6_cam=video.avi"])
    with pytest.raises(HPETranscriptCrosswalkError, match="Cam_N=PATH"):
        reviewer._parse_hpe_video_assignments(["Cam_2"])


def test_offline_video_fails_before_annotation_artifact_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_path(tmp_path, camera="Cam_2")
    artifact = tmp_path / "crosswalk_manifest.json"
    artifact.write_bytes(b"unchanged human decisions")
    before = artifact.read_bytes()
    samples = pd.DataFrame(
        [
            _sample_row(
                gaze_source=json.dumps({"kind": "same_as_hpe"}),
                hpe_source=_video_source("Cam_2", 50),
            )
        ]
    )
    monkeypatch.setattr(reviewer.os, "name", "nt")
    monkeypatch.setattr(
        reviewer,
        "_windows_file_attributes",
        lambda _path: reviewer._CLOUD_RECALL_FLAGS,
    )

    with pytest.raises(HPETranscriptCrosswalkError, match="Offline placeholder"):
        reviewer._preflight_sample_videos(
            samples,
            gaze_video=None,
            hpe_videos={"Cam_2": video},
        )

    assert artifact.read_bytes() == before


def test_wrong_video_identity_fails_before_annotation_artifact_changes(
    tmp_path: Path,
) -> None:
    wrong = _video_path(tmp_path, camera="Cam_2", basename="wrong.avi")
    artifact = tmp_path / "crosswalk_manifest.json"
    artifact.write_bytes(b"unchanged human decisions")
    before = artifact.read_bytes()
    samples = pd.DataFrame(
        [
            _sample_row(
                gaze_source=json.dumps({"kind": "same_as_hpe"}),
                hpe_source=_video_source("Cam_2", 50),
            )
        ]
    )

    with pytest.raises(HPETranscriptCrosswalkError, match="basename mismatch"):
        reviewer._preflight_sample_videos(
            samples,
            gaze_video=None,
            hpe_videos={"Cam_2": wrong},
        )

    assert artifact.read_bytes() == before


def test_same_as_hpe_reuses_one_exact_frame_and_one_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hpe_video = _video_path(tmp_path, camera="Cam_2")
    row = _sample_row(
        gaze_source=json.dumps({"kind": "same_as_hpe"}),
        hpe_source=_video_source("Cam_2", 47),
    )
    gaze = reviewer._resolve_sample_source(
        row,
        "gaze",
        gaze_video=None,
        hpe_videos={"Cam_2": hpe_video},
    )
    hpe = reviewer._resolve_sample_source(
        row,
        "hpe",
        gaze_video=None,
        hpe_videos={"Cam_2": hpe_video},
    )

    assert gaze.kind == "same_as_hpe"
    assert gaze.path == hpe.path == hpe_video
    assert gaze.frame_index == hpe.frame_index == 47

    calls: list[Path] = []
    monkeypatch.setattr(
        reviewer,
        "_preflight_video",
        lambda path, _expected: calls.append(path) or {"path": str(path)},
    )
    result = reviewer._preflight_sample_videos(
        pd.DataFrame([row]),
        gaze_video=None,
        hpe_videos={"Cam_2": hpe_video},
    )

    assert calls == [hpe_video]
    assert list(result) == [str(hpe_video.resolve())]


def test_same_as_hpe_still_needs_no_raw_video_preflight(tmp_path: Path) -> None:
    still = tmp_path / "review_source_frame_00047.jpg"
    row = _sample_row(
        gaze_source=json.dumps({"kind": "same_as_hpe"}),
        hpe_source=json.dumps(
            {
                "kind": "still",
                "path": str(still),
                "frame_index": 47,
                "camera": "Cam_2",
                "size_bytes": 10,
                "sha256": "0" * 64,
            }
        ),
    )

    result = reviewer._preflight_sample_videos(
        pd.DataFrame([row]),
        gaze_video=None,
        hpe_videos={},
    )

    assert result == {}


def test_cross_camera_scene_uses_gaze_cam1_and_hpe_cam2() -> None:
    gaze_video = Path("root/2/Cam_1/correct.avi").resolve()
    hpe_video = Path("root/2/Cam_2/correct.avi").resolve()
    row = _sample_row(
        gaze_source=_video_source("Cam_1", 41),
        hpe_source=_video_source("Cam_2", 41),
    )

    gaze = reviewer._resolve_sample_source(
        row,
        "gaze",
        gaze_video=gaze_video,
        hpe_videos={"Cam_2": hpe_video},
    )
    hpe = reviewer._resolve_sample_source(
        row,
        "hpe",
        gaze_video=gaze_video,
        hpe_videos={"Cam_2": hpe_video},
    )

    assert (gaze.camera, gaze.path, gaze.frame_index) == ("Cam_1", gaze_video, 41)
    assert (hpe.camera, hpe.path, hpe.frame_index) == ("Cam_2", hpe_video, 41)


def test_bound_default_video_path_is_used_when_cli_path_is_omitted(
    tmp_path: Path,
) -> None:
    bound = _video_path(tmp_path, camera="Cam_1")
    source = json.loads(_video_source("Cam_1", 41))
    source["video_identity"]["last_verified_path"] = str(bound)
    row = _sample_row(
        gaze_source=json.dumps(source),
        hpe_source=_video_source("Cam_2", 41),
    )

    gaze = reviewer._resolve_sample_source(
        row,
        "gaze",
        gaze_video=None,
        hpe_videos={"Cam_2": Path("hpe.avi")},
    )

    assert gaze.path == bound.resolve()


def test_finalize_requires_a_second_keypress_for_upstream_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acknowledgements: list[str | None] = []
    preflight_calls: list[bool] = []

    def fake_finalize(
        _path: Path, *, upstream_mismatch_acknowledgement: str | None
    ) -> Path:
        acknowledgements.append(upstream_mismatch_acknowledgement)
        if upstream_mismatch_acknowledgement is None:
            raise HPETranscriptCrosswalkError(
                "reviewed labels differ from the upstream proposal; "
                "explicit acknowledgement is required"
            )
        return tmp_path / "hpe_transcript_crosswalk.csv"

    monkeypatch.setattr(reviewer, "finalize_crosswalk", fake_finalize)
    monkeypatch.setattr(
        reviewer,
        "_preflight_sample_videos",
        lambda *_args, **_kwargs: preflight_calls.append(True) or {},
    )
    monkeypatch.setattr(
        reviewer,
        "bind_review_video_identities",
        lambda *_args, **_kwargs: None,
    )
    ui = reviewer.CrosswalkReviewer.__new__(reviewer.CrosswalkReviewer)
    ui.samples_path = tmp_path / "review_samples.csv"
    ui.manifest_path = tmp_path / "crosswalk_manifest.json"
    ui.samples = pd.DataFrame()
    ui.gaze_video = None
    ui.hpe_videos = {}
    ui.finalize_armed = False
    ui.status_message = ""

    assert not ui._finalize()
    assert ui.finalize_armed
    assert "press F again" in ui.status_message
    assert ui._finalize()
    assert acknowledgements[0] is None
    assert acknowledgements[1]
    assert len(preflight_calls) == 2


def test_finalize_rejects_video_deleted_after_startup_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_path(tmp_path, camera="Cam_2")
    samples = pd.DataFrame(
        [
            _sample_row(
                gaze_source=json.dumps({"kind": "same_as_hpe"}),
                hpe_source=_video_source("Cam_2", 50),
            )
        ]
    )
    monkeypatch.setattr(reviewer, "_require_cv2", lambda: _FakeCV2())
    startup = reviewer._preflight_sample_videos(
        samples,
        gaze_video=None,
        hpe_videos={"Cam_2": video},
    )
    assert startup
    video.unlink()
    monkeypatch.setattr(
        reviewer,
        "bind_review_video_identities",
        lambda *_args, **_kwargs: pytest.fail("deleted video must fail before binding"),
    )
    monkeypatch.setattr(
        reviewer,
        "finalize_crosswalk",
        lambda *_args, **_kwargs: pytest.fail("deleted video must not finalize"),
    )
    ui = reviewer.CrosswalkReviewer.__new__(reviewer.CrosswalkReviewer)
    ui.samples_path = tmp_path / "review_samples.csv"
    ui.manifest_path = tmp_path / "crosswalk_manifest.json"
    ui.samples = samples
    ui.gaze_video = None
    ui.hpe_videos = {"Cam_2": video}
    ui.finalize_armed = True
    ui.status_message = ""

    assert not ui._finalize()
    assert not ui.finalize_armed
    assert "does not exist" in ui.status_message


def test_finalize_rejects_video_replaced_after_startup_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_path(tmp_path, camera="Cam_2")
    samples = pd.DataFrame(
        [
            _sample_row(
                gaze_source=json.dumps({"kind": "same_as_hpe"}),
                hpe_source=_video_source("Cam_2", 50),
            )
        ]
    )
    monkeypatch.setattr(reviewer, "_require_cv2", lambda: _FakeCV2())
    startup = reviewer._preflight_sample_videos(
        samples,
        gaze_video=None,
        hpe_videos={"Cam_2": video},
    )
    original_size = video.stat().st_size
    video.write_bytes(b"X" * original_size)
    assert reviewer._sparse_video_fingerprint(video) != next(
        iter(startup.values())
    )["sparse_sha256"]
    bind_calls: list[dict[str, dict[str, object]]] = []

    def reject_replacement(
        _path: Path,
        _manifest_path: Path,
        *,
        bindings: dict[str, dict[str, object]],
    ) -> None:
        bind_calls.append(bindings)
        if bindings != startup:
            raise HPETranscriptCrosswalkError(
                "review video bytes differ from the existing revision binding"
            )

    monkeypatch.setattr(reviewer, "bind_review_video_identities", reject_replacement)
    monkeypatch.setattr(
        reviewer,
        "finalize_crosswalk",
        lambda *_args, **_kwargs: pytest.fail("replaced video must not finalize"),
    )
    ui = reviewer.CrosswalkReviewer.__new__(reviewer.CrosswalkReviewer)
    ui.samples_path = tmp_path / "review_samples.csv"
    ui.manifest_path = tmp_path / "crosswalk_manifest.json"
    ui.samples = samples
    ui.gaze_video = None
    ui.hpe_videos = {"Cam_2": video}
    ui.finalize_armed = True
    ui.status_message = ""

    assert not ui._finalize()
    assert bind_calls
    assert not ui.finalize_armed
    assert "bytes differ" in ui.status_message


def test_preflight_checks_stream_metadata_and_sparse_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_path(tmp_path, camera="Cam_2")
    fake = _FakeCV2()
    monkeypatch.setattr(reviewer, "_require_cv2", lambda: fake)

    first = reviewer._preflight_video(video, _identity("Cam_2"))
    expected = {**_identity("Cam_2"), **first}
    second = reviewer._preflight_video(video, expected)

    assert second["sparse_sha256"] == first["sparse_sha256"]
    assert fake.capture.released


def test_frame_reader_rejects_inexact_video_seek(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_path(tmp_path, camera="Cam_2")
    fake = _FakeCV2(decoded_frame=48)
    reader = reviewer._FrameReader(fake)
    resolved = reviewer._ResolvedSource(
        "video",
        json.loads(_video_source("Cam_2", 47)),
        video,
        47,
        "Cam_2",
    )

    with pytest.raises(HPETranscriptCrosswalkError, match="decoded frame 48"):
        reader.read(resolved)

    reader.close()
    assert fake.capture.released
