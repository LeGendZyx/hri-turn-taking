from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import hri_turn_taking.data.identity_annotation as identity_annotation
from hri_turn_taking.data.identity_annotation import (
    ARTIFACT_KIND,
    FORMAT_VERSION,
    IdentityAnnotationError,
    TrackerConfig,
    build_identity_bridge,
    build_tracklets,
    export_identity_bridge,
    load_annotation_manifest,
    load_export_metadata,
    load_gaze_observations,
    load_hpe_annotation_scope,
    mark_review_anchor,
    migrate_legacy_annotation_to_hpe_core,
    prepare_annotation_artifact,
    save_annotation_manifest,
    save_annotation_state,
    set_tracklet_annotation,
    split_tracklet,
    summarise_tracklets,
    validate_tracklet_observations,
)


def test_manifest_save_retries_a_transient_windows_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "identity_annotations.json"
    real_replace = identity_annotation.os.replace
    replace_calls = 0

    def flaky_replace(source: Path, target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls < 3:
            raise PermissionError(5, "transient destination lock")
        real_replace(source, target)

    monkeypatch.setattr(identity_annotation.os, "replace", flaky_replace)
    monkeypatch.setattr(identity_annotation.time, "sleep", lambda _seconds: None)

    manifest: dict[str, object] = {"annotations": []}
    save_annotation_manifest(destination, manifest)

    assert replace_calls == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == manifest
    assert list(tmp_path.glob("*.json.tmp")) == []


def test_manifest_save_keeps_recovery_json_after_a_persistent_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "identity_annotations.json"
    destination.write_text('{"old": true}\n', encoding="utf-8")
    replace_calls = 0

    def locked_replace(_source: Path, _target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        raise PermissionError(5, "persistent destination lock")

    monkeypatch.setattr(identity_annotation.os, "replace", locked_replace)
    monkeypatch.setattr(identity_annotation.time, "sleep", lambda _seconds: None)

    manifest: dict[str, object] = {"annotations": []}
    with pytest.raises(PermissionError, match="persistent destination lock"):
        save_annotation_manifest(destination, manifest)

    assert replace_calls == identity_annotation._ATOMIC_REPLACE_ATTEMPTS
    assert json.loads(destination.read_text(encoding="utf-8")) == {"old": True}
    recovery_files = list(tmp_path.glob("*.json.tmp"))
    assert len(recovery_files) == 1
    assert json.loads(recovery_files[0].read_text(encoding="utf-8")) == manifest


def _row(
    frame: int,
    person_idx: int | None,
    bbox: tuple[float, float, float, float] | None,
    *,
    robot_detected: int = 1,
) -> dict[str, object]:
    robot_bbox = (200.0, 30.0, 240.0, 90.0) if robot_detected else (None,) * 4
    participant_bbox = bbox if bbox is not None else (None,) * 4
    return {
        "frame_index": frame,
        "timestamp_sec": frame * 0.04,
        "robot_detected": robot_detected,
        "robot_x0": robot_bbox[0],
        "robot_y0": robot_bbox[1],
        "robot_x1": robot_bbox[2],
        "robot_y1": robot_bbox[3],
        "person_idx": person_idx,
        "bbox_x0": participant_bbox[0],
        "bbox_y0": participant_bbox[1],
        "bbox_x1": participant_bbox[2],
        "bbox_y1": participant_bbox[3],
        "gaze_target": "participant" if person_idx is not None else "unknown",
        "gaze_peak_x": 100.0 if person_idx is not None else None,
        "gaze_peak_y": 50.0 if person_idx is not None else None,
        "inout_score": 0.9 if person_idx is not None else None,
    }


def _swapped_detection_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in range(3):
        left_idx, right_idx = ((0, 1) if frame % 2 == 0 else (1, 0))
        rows.extend(
            [
                _row(frame, left_idx, (10 + frame, 10, 30 + frame, 40)),
                _row(frame, right_idx, (110 + frame, 10, 130 + frame, 40)),
            ]
        )
    return rows


def _write_gaze(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    maximum_frame = max(int(row["frame_index"]) for row in rows)
    (path.parent / "session_metadata.json").write_text(
        json.dumps(
            {
                "input": {"video_path": "Data/test/1/Cam_2/test_video.avi"},
                "stream": {
                    "total_frames_in_video": maximum_frame + 1,
                    "fps": 25.0,
                    "frame_width": 320,
                    "frame_height": 180,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_hpe_scope(
    root: Path,
    source: Path,
    *,
    start_sec: float,
    end_sec: float,
    segments: list[str] | None = None,
    decoded_frames: list[int] | None = None,
) -> dict[str, object]:
    metadata = json.loads((source.parent / "session_metadata.json").read_text("utf-8"))
    stream = metadata["stream"]
    profile = root / "test__1.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "day": "test",
                "scenario": 1,
                "core_window_sec": {
                    "start_inclusive": start_sec,
                    "end_exclusive": end_sec,
                },
                "source_fps": stream["fps"],
                "review_status": "confirmed",
            }
        ),
        encoding="utf-8",
    )
    segment_paths = segments or [
        f"F:/share/test/6cams/1/Cam_2/{Path(metadata['input']['video_path']).name}"
    ]
    frame_counts = decoded_frames
    processed_frames = (
        sum(frame_counts) if frame_counts is not None else stream["total_frames_in_video"]
    )
    run_metadata = root / "run_metadata.json"
    run_metadata.write_text(
        json.dumps(
            {
                "input": segment_paths[0],
                "segments": segment_paths,
                "input_files": [{"path": value} for value in segment_paths],
                "segments_decoded_frames": frame_counts,
                "source_fps": stream["fps"],
                "declared_expected_frames": processed_frames,
                "processed_frames": processed_frames,
                "width": stream["frame_width"],
                "height": stream["frame_height"],
                "params": {"start_frame": 0, "max_frames": None},
            }
        ),
        encoding="utf-8",
    )
    _, _, declared_video = load_export_metadata(source)
    return load_hpe_annotation_scope(
        profile,
        run_metadata,
        dataset="test",
        session="1",
        declared_video=declared_video,
    )


def _manifest(observations: pd.DataFrame) -> dict[str, object]:
    return {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "automatic_missing_policy": (
            "crowding_guard_and_singleton_participant_tracklets_are_uncertain"
        ),
        "annotations": summarise_tracklets(observations),
    }


def test_tracklets_ignore_swapping_raw_person_indices(tmp_path: Path) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())

    observations = build_tracklets(load_gaze_observations(source))
    participants = observations[observations["source_kind"] == "participant_detection"]
    robot = observations[observations["source_kind"] == "robot_candidate"]

    assert participants["tracklet_id"].nunique() == 2
    assert robot["tracklet_id"].nunique() == 1
    for _, tracklet in participants.groupby("tracklet_id"):
        assert len(tracklet) == 3
        assert set(tracklet["source_person_idx"]) == {"0", "1"}


def test_crossing_people_are_cut_before_a_tracklet_can_mix_identities() -> None:
    rows: list[dict[str, object]] = []
    positions = {"A": [0, 40, 80], "B": [100, 60, 20]}
    for frame in range(3):
        for identity in ("A", "B"):
            x0 = positions[identity][frame]
            rows.append(
                {
                    "observation_id": len(rows),
                    "frame_index": frame,
                    "timestamp_sec": frame * 0.04,
                    "source_kind": "participant_detection",
                    "source_person_idx": str(len(rows) % 2),
                    "bbox_x0": x0,
                    "bbox_y0": 0,
                    "bbox_x1": x0 + 60,
                    "bbox_y1": 100,
                    "true_identity": identity,
                }
            )

    tracked = build_tracklets(pd.DataFrame(rows))

    assert all(
        tracklet["true_identity"].nunique() == 1
        for _, tracklet in tracked.groupby("tracklet_id")
    )
    assert set(tracked.loc[tracked["frame_index"] == 1, "link_flags"]) == {
        "crowding_guard"
    }
    assert set(tracked.loc[tracked["frame_index"] == 2, "link_flags"]) == {
        "crowding_guard"
    }


def test_prolonged_crowding_is_quarantined_without_mixing_identities() -> None:
    rows: list[dict[str, object]] = []
    for frame in range(100):
        for identity, x0 in (("A", frame * 0.1), ("B", 20 + frame * 0.1)):
            rows.append(
                {
                    "observation_id": len(rows),
                    "frame_index": frame,
                    "timestamp_sec": frame * 0.04,
                    "source_kind": "participant_detection",
                    "source_person_idx": str(len(rows) % 2),
                    "bbox_x0": x0,
                    "bbox_y0": 0,
                    "bbox_x1": x0 + 60,
                    "bbox_y1": 100,
                    "true_identity": identity,
                }
            )

    tracked = build_tracklets(pd.DataFrame(rows))

    assert tracked["tracklet_id"].nunique() == 200
    assert tracked["link_flags"].eq("crowding_guard").all()
    assert all(
        tracklet["true_identity"].nunique() == 1
        for _, tracklet in tracked.groupby("tracklet_id")
    )
    summaries = summarise_tracklets(tracked)
    assert all(item["auto_quarantined"] for item in summaries)
    assert all(item["status"] == "uncertain" for item in summaries)
    with pytest.raises(IdentityAnnotationError, match="must remain missing"):
        set_tracklet_annotation(
            {"annotations": summaries},
            summaries[0]["tracklet_id"],
            label="A",
            status="confirmed",
        )


def test_short_participant_quarantine_uses_timestamp_span_and_skips_robot() -> None:
    rows: list[dict[str, object]] = []

    def add_tracklet(
        tracklet_id: str,
        source_kind: str,
        timestamps: list[float],
    ) -> None:
        for offset, timestamp_s in enumerate(timestamps):
            rows.append(
                {
                    "observation_id": len(rows),
                    "frame_index": 10 * len(rows),
                    "timestamp_sec": timestamp_s,
                    "source_kind": source_kind,
                    "source_person_idx": "0" if source_kind == "participant_detection" else "robot",
                    "tracklet_id": tracklet_id,
                    "link_cost": None if offset == 0 else 0.0,
                    "link_confidence": 1.0,
                    "link_flags": "new_tracklet" if offset == 0 else "linked",
                }
            )

    # Observation count is deliberately uninformative: eight dense observations
    # fit inside the threshold, while two sparse observations exceed it.
    add_tracklet("p_dense", "participant_detection", [index / 70 for index in range(8)])
    add_tracklet("p_edge_inside", "participant_detection", [0.0, 0.1000000000005])
    add_tracklet("p_edge_outside", "participant_detection", [0.0, 0.100000000002])
    add_tracklet("p_sparse", "participant_detection", [0.0, 0.100001])
    add_tracklet("r_short", "robot_candidate", [0.0, 0.1])

    observations = pd.DataFrame(rows)
    default = {item["tracklet_id"]: item for item in summarise_tracklets(observations)}
    summaries = {
        item["tracklet_id"]: item
        for item in summarise_tracklets(
            observations,
            short_tracklet_max_duration_s=0.1,
        )
    }

    assert default["p_dense"]["status"] == "unreviewed"
    assert summaries["p_dense"]["status"] == "uncertain"
    assert summaries["p_dense"]["auto_quarantined"] is True
    assert summaries["p_dense"]["auto_quarantine_reason"] == "short_fragment"
    assert summaries["p_edge_inside"]["status"] == "uncertain"
    assert summaries["p_edge_outside"]["status"] == "unreviewed"
    assert summaries["p_sparse"]["status"] == "unreviewed"
    assert summaries["r_short"]["status"] == "unreviewed"
    with pytest.raises(IdentityAnnotationError, match="must remain missing"):
        set_tracklet_annotation(
            {"annotations": list(summaries.values())},
            "p_dense",
            label="A",
            status="confirmed",
        )


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf"), True])
def test_short_tracklet_duration_threshold_must_be_finite_and_non_negative(
    value: float,
) -> None:
    observations = pd.DataFrame(
        [
            {
                "observation_id": 0,
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "source_kind": "participant_detection",
                "source_person_idx": "0",
                "tracklet_id": "p000001",
                "link_cost": None,
                "link_confidence": 1.0,
                "link_flags": "new_tracklet",
            }
        ]
    )

    with pytest.raises(IdentityAnnotationError, match="finite non-negative"):
        summarise_tracklets(
            observations,
            short_tracklet_max_duration_s=value,
        )


def test_empty_frame_sentinel_is_not_treated_as_a_person(tmp_path: Path) -> None:
    rows = [_row(10, None, None, robot_detected=0), *_swapped_detection_rows()]
    source = _write_gaze(tmp_path / "gaze.csv", rows)

    observations = load_gaze_observations(source)

    assert 10 not in set(observations["frame_index"])
    assert len(observations) == 9


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"person_idx": None, "bbox_x0": 10.0}, "jointly present"),
        ({"robot_detected": 0, "robot_x0": 200.0}, "while robot_detected is 0"),
    ],
)
def test_malformed_missing_detection_rows_fail_closed(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    row = _row(0, None, None, robot_detected=0)
    row.update(mutation)
    source = _write_gaze(tmp_path / "gaze.csv", [row])

    with pytest.raises(IdentityAnnotationError, match=message):
        load_gaze_observations(source)


def test_prepare_is_resumable_but_rejects_changed_source(tmp_path: Path) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())
    output = tmp_path / "annotation"
    observations_path, manifest_path = prepare_annotation_artifact(
        source, output, dataset="test", session="1"
    )

    resumed = prepare_annotation_artifact(source, output, dataset="test", session="1")
    manifest = load_annotation_manifest(manifest_path)

    assert resumed == (observations_path, manifest_path)
    assert manifest["source"]["gaze_csv"] == str(source.resolve())

    with source.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(IdentityAnnotationError, match="changed since annotation preparation"):
        prepare_annotation_artifact(source, output, dataset="test", session="1")


def test_hpe_scope_maps_segment_local_frames_and_uses_half_open_window(
    tmp_path: Path,
) -> None:
    rows = [_row(frame, 0, (10, 10, 30, 40)) for frame in range(10)]
    source = _write_gaze(tmp_path / "gaze.csv", rows)
    metadata = json.loads((tmp_path / "session_metadata.json").read_text("utf-8"))
    metadata["input"]["video_path"] = "Data/test/1/Cam_2/second.avi"
    metadata["stream"]["total_frames_in_video"] = 10
    (tmp_path / "session_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    scope = _write_hpe_scope(
        tmp_path,
        source,
        start_sec=0.8,
        end_sec=2.4,
        segments=[
            "F:/share/test/6cams/1/Cam_2/first.avi",
            "F:/share/test/6cams/1/Cam_2/second.avi",
        ],
        decoded_frames=[20, 10],
    )

    assert scope["core_start_logical_frame_inclusive"] == 20
    assert scope["core_end_logical_frame_exclusive"] == 60
    assert scope["gaze_segment_logical_start_frame"] == 20
    assert scope["gaze_local_start_frame_inclusive"] == 0
    assert scope["gaze_local_end_frame_exclusive"] == 10
    assert scope["core_coverage_fraction"] == pytest.approx(0.25)


def test_hpe_scope_accepts_only_a_complete_legacy_single_input_run(
    tmp_path: Path,
) -> None:
    source = _write_gaze(
        tmp_path / "gaze.csv",
        [_row(frame, 0, (10, 10, 30, 40)) for frame in range(10)],
    )
    metadata = json.loads((tmp_path / "session_metadata.json").read_text("utf-8"))
    metadata["input"]["video_path"] = "Data/test/6cams/1/Cam_2/test_video.avi"
    (tmp_path / "session_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    profile = tmp_path / "test__1.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "day": "test",
                "scenario": 1,
                "core_window_sec": {
                    "start_inclusive": 0.08,
                    "end_exclusive": 0.24,
                },
                "source_fps": 25.0,
                "review_status": "confirmed",
            }
        ),
        encoding="utf-8",
    )
    run_metadata = tmp_path / "run_metadata.json"
    run_doc = {
        "input": "F:/share/test/6cams/1/Cam_2/test_video.avi",
        "source_fps": 25.0,
        "processed_frames": 10,
        "width": 320,
        "height": 180,
        "params": {"start_frame": 0, "max_frames": None},
    }
    run_metadata.write_text(json.dumps(run_doc), encoding="utf-8")
    _, _, declared_video = load_export_metadata(source)

    scope = load_hpe_annotation_scope(
        profile,
        run_metadata,
        dataset="test",
        session="1",
        declared_video=declared_video,
    )

    assert scope["run_source_contract"] == "legacy_single_input_complete_v1"
    assert scope["gaze_local_start_frame_inclusive"] == 2
    assert scope["gaze_local_end_frame_exclusive"] == 6

    run_doc["segments"] = [run_doc["input"]]
    run_metadata.write_text(json.dumps(run_doc), encoding="utf-8")
    segmented_scope = load_hpe_annotation_scope(
        profile,
        run_metadata,
        dataset="test",
        session="1",
        declared_video=declared_video,
    )
    assert (
        segmented_scope["run_source_contract"]
        == "legacy_single_segment_complete_v1"
    )

    run_doc["params"]["start_frame"] = 1
    run_metadata.write_text(json.dumps(run_doc), encoding="utf-8")
    with pytest.raises(IdentityAnnotationError, match="complete source from frame zero"):
        load_hpe_annotation_scope(
            profile,
            run_metadata,
            dataset="test",
            session="1",
            declared_video=declared_video,
        )


def test_prepare_filters_before_tracking_and_binds_profile_changes(
    tmp_path: Path,
) -> None:
    rows = [_row(frame, 0, (10 + frame, 10, 30 + frame, 40)) for frame in range(10)]
    source = _write_gaze(tmp_path / "gaze.csv", rows)
    scope = _write_hpe_scope(
        tmp_path,
        source,
        start_sec=0.08,
        end_sec=0.24,
    )
    output = tmp_path / "annotation"

    observations_path, manifest_path = prepare_annotation_artifact(
        source,
        output,
        dataset="test",
        session="1",
        annotation_scope=scope,
    )
    manifest = load_annotation_manifest(manifest_path)
    observations = validate_tracklet_observations(observations_path, manifest)

    assert set(observations["frame_index"]) == {2, 3, 4, 5}
    first_rows = observations.groupby("tracklet_id", sort=False).head(1)
    assert first_rows["link_cost"].isna().all()
    assert manifest["scope_counts"] == {
        "full_observations": 20,
        "scoped_observations": 8,
        "excluded_observations": 12,
        "scoped_tracklets": 2,
    }

    profile = Path(scope["profile_path"])
    profile.write_text(profile.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(IdentityAnnotationError, match="scope changed"):
        prepare_annotation_artifact(
            source,
            output,
            dataset="test",
            session="1",
            annotation_scope=scope,
        )


def test_scope_validation_replays_source_counts_metadata_and_run_marker(
    tmp_path: Path,
) -> None:
    source = _write_gaze(
        tmp_path / "gaze.csv",
        [_row(frame, 0, (10, 10, 30, 40)) for frame in range(10)],
    )
    scope = _write_hpe_scope(tmp_path, source, start_sec=0.08, end_sec=0.24)
    output = tmp_path / "annotation"
    observations_path, manifest_path = prepare_annotation_artifact(
        source,
        output,
        dataset="test",
        session="1",
        annotation_scope=scope,
    )
    manifest = load_annotation_manifest(manifest_path)

    manifest["scope_counts"]["full_observations"] += 100
    manifest["scope_counts"]["excluded_observations"] += 100
    with pytest.raises(IdentityAnnotationError, match="cannot be replayed"):
        validate_tracklet_observations(observations_path, manifest)

    clean = load_annotation_manifest(manifest_path)
    metadata_path = Path(clean["source"]["session_metadata"])
    metadata_path.write_text(metadata_path.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(IdentityAnnotationError, match="session metadata changed"):
        validate_tracklet_observations(observations_path, clean)

    metadata_path.write_text(metadata_path.read_text("utf-8").rstrip() + "\n", encoding="utf-8")
    clean["source"]["session_metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    run_metadata = Path(scope["run_metadata_path"])
    run_metadata.write_text(run_metadata.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(IdentityAnnotationError, match="scope changed"):
        validate_tracklet_observations(observations_path, clean)


def test_legacy_migration_carries_only_exact_inside_tracklets(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for frame in range(10):
        rows.append(_row(frame, 0, (10 + frame, 10, 30 + frame, 40)))
        if 2 <= frame < 8:
            rows.append(_row(frame, 1, (110 + frame, 10, 130 + frame, 40)))
        if frame < 2:
            rows.append(_row(frame, 2, (50 + frame, 80, 70 + frame, 110)))
    source = _write_gaze(tmp_path / "gaze.csv", rows)
    output = tmp_path / "annotation"
    observations_path, manifest_path = prepare_annotation_artifact(
        source, output, dataset="test", session="1"
    )
    manifest = load_annotation_manifest(manifest_path)
    observations = validate_tracklet_observations(observations_path, manifest)
    inside_tracklet = str(
        observations.loc[
            (observations["source_kind"] == "participant_detection")
            & (observations["source_person_idx"] == "1"),
            "tracklet_id",
        ].iloc[0]
    )
    inside_annotation = next(
        item for item in manifest["annotations"] if item["tracklet_id"] == inside_tracklet
    )
    for frame in inside_annotation["required_review_frames"]:
        mark_review_anchor(manifest, inside_tracklet, frame)
    set_tracklet_annotation(
        manifest, inside_tracklet, label="B", status="confirmed"
    )
    crossing_tracklet = str(
        observations.loc[
            (observations["source_kind"] == "participant_detection")
            & (observations["source_person_idx"] == "0"),
            "tracklet_id",
        ].iloc[0]
    )
    crossing_annotation = next(
        item
        for item in manifest["annotations"]
        if item["tracklet_id"] == crossing_tracklet
    )
    for frame in crossing_annotation["required_review_frames"]:
        mark_review_anchor(manifest, crossing_tracklet, frame)
    set_tracklet_annotation(
        manifest, crossing_tracklet, label="A", status="confirmed"
    )
    manifest["format_version"] = 1
    manifest.pop("annotation_scope")
    manifest.pop("scope_counts")
    manifest.pop("automatic_missing_policy")
    manifest.pop("tracker_algorithm")
    for item in manifest["annotations"]:
        item.pop("auto_quarantined")
        item.pop("auto_quarantine_reason")
    manifest["tracklet_observations"]["parsed_dtypes"] = [
        "int32" if str(value).startswith("int") else "object" if value == "str" else value
        for value in manifest["tracklet_observations"]["parsed_dtypes"]
    ]
    save_annotation_manifest(manifest_path, manifest)
    scope = _write_hpe_scope(
        tmp_path,
        source,
        start_sec=0.08,
        end_sec=0.32,
    )

    before_failed_preflight = manifest_path.read_bytes()
    with pytest.raises(IdentityAnnotationError, match="tracker settings differ"):
        migrate_legacy_annotation_to_hpe_core(
            output,
            scope,
            expected_tracker_config=TrackerConfig(max_gap_s=0.3),
        )
    assert manifest_path.read_bytes() == before_failed_preflight

    result = migrate_legacy_annotation_to_hpe_core(output, scope)
    migrated = load_annotation_manifest(manifest_path)
    scoped = validate_tracklet_observations(observations_path, migrated)

    assert result["carried_tracklets"] >= 1
    assert result["reset_tracklets"] >= 1
    assert result["dropped_tracklets"] >= 1
    assert Path(result["backup_path"]).is_dir()
    assert migrated["automatic_missing_policy"] == {
        "kind": "legacy_no_automatic_quarantine",
        "version": 1,
        "rules": [],
        "identity_effect": "preserve_existing_manual_state",
    }
    assert set(scoped["frame_index"]) == set(range(2, 8))
    confirmed = [
        item for item in migrated["annotations"] if item["status"] == "confirmed"
    ]
    assert [(item["label"], item["observation_count"]) for item in confirmed] == [
        ("B", 6)
    ]
    reset_previous = [
        previous
        for item in migrated["migration"]["reset_tracklets"]
        for previous in item["previous_annotations"]
    ]
    assert any(
        item["old_tracklet_id"] == crossing_tracklet
        and item["status"] == "confirmed"
        and item["label"] == "A"
        for item in reset_previous
    )
    bridge = pd.read_csv(output / "identity_bridge.csv")
    assert set(bridge["frame_index"]) == set(range(2, 8))


def test_prepare_rejects_identity_override_without_creating_output(tmp_path: Path) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())
    output = tmp_path / "annotation"

    with pytest.raises(IdentityAnnotationError, match="disagree with the source"):
        prepare_annotation_artifact(source, output, dataset="other", session="1")

    assert not output.exists()


def test_resume_rejects_changed_tracker_settings(tmp_path: Path) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())
    output = tmp_path / "annotation"
    prepare_annotation_artifact(source, output, dataset="test", session="1")

    with pytest.raises(IdentityAnnotationError, match="tracker settings differ"):
        prepare_annotation_artifact(
            source,
            output,
            dataset="test",
            session="1",
            config=TrackerConfig(max_gap_s=0.3),
        )


def test_short_tracklet_policy_is_manifest_bound_and_legacy_default_resumes(
    tmp_path: Path,
) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())
    short_output = tmp_path / "short_annotation"
    observations_path, manifest_path = prepare_annotation_artifact(
        source,
        short_output,
        dataset="test",
        session="1",
        short_tracklet_max_duration_s=0.1,
    )
    manifest = load_annotation_manifest(manifest_path)
    observations = validate_tracklet_observations(observations_path, manifest)

    assert manifest["automatic_missing_policy"] == {
        "kind": "conservative_tracklet_quarantine",
        "version": 2,
        "rules": [
            "all_crowding_guard_tracklets",
            "singleton_participant_tracklets",
            "short_participant_tracklets",
        ],
        "short_tracklet_max_duration_s": 0.1,
        "short_tracklet_duration_measure": "max(timestamp_sec)-min(timestamp_sec)",
        "short_tracklet_duration_comparison": "<=",
        "short_tracklet_boundary_tolerance_s": 1e-12,
        "identity_effect": "uncertain_missing",
    }
    participant_ids = set(
        observations.loc[
            observations["source_kind"] == "participant_detection", "tracklet_id"
        ]
    )
    participant_annotations = [
        item for item in manifest["annotations"] if item["tracklet_id"] in participant_ids
    ]
    assert participant_annotations
    assert all(
        item["auto_quarantine_reason"] == "short_fragment"
        for item in participant_annotations
    )
    assert all(item["status"] == "uncertain" for item in participant_annotations)
    assert prepare_annotation_artifact(
        source,
        short_output,
        dataset="test",
        session="1",
        short_tracklet_max_duration_s=0.1,
    ) == (observations_path, manifest_path)
    with pytest.raises(IdentityAnnotationError, match="automatic missing policy differs"):
        prepare_annotation_artifact(
            source,
            short_output,
            dataset="test",
            session="1",
            short_tracklet_max_duration_s=0.2,
        )

    legacy_output = tmp_path / "legacy_default_annotation"
    _, legacy_manifest_path = prepare_annotation_artifact(
        source,
        legacy_output,
        dataset="test",
        session="1",
    )
    legacy_manifest = load_annotation_manifest(legacy_manifest_path)
    legacy_manifest["automatic_missing_policy"] = (
        "crowding_guard_and_singleton_participant_tracklets_are_uncertain"
    )
    save_annotation_manifest(legacy_manifest_path, legacy_manifest)

    prepare_annotation_artifact(
        source,
        legacy_output,
        dataset="test",
        session="1",
    )
    with pytest.raises(IdentityAnnotationError, match="automatic missing policy differs"):
        prepare_annotation_artifact(
            source,
            legacy_output,
            dataset="test",
            session="1",
            short_tracklet_max_duration_s=0.1,
        )


def test_recognised_early_v2_without_auto_quarantine_resumes_only_at_zero(
    tmp_path: Path,
) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())
    early_output = tmp_path / "early_v2_annotation"
    observations_path, manifest_path = prepare_annotation_artifact(
        source,
        early_output,
        dataset="test",
        session="1",
    )
    early_manifest = load_annotation_manifest(manifest_path)
    early_manifest.pop("automatic_missing_policy")
    early_manifest.pop("tracker_algorithm")
    for item in early_manifest["annotations"]:
        item.pop("auto_quarantined")
        item.pop("auto_quarantine_reason")
    save_annotation_manifest(manifest_path, early_manifest)

    loaded = load_annotation_manifest(manifest_path)
    assert "automatic_missing_policy" not in loaded
    assert prepare_annotation_artifact(
        source,
        early_output,
        dataset="test",
        session="1",
    ) == (observations_path, manifest_path)
    with pytest.raises(IdentityAnnotationError, match="automatic missing policy differs"):
        prepare_annotation_artifact(
            source,
            early_output,
            dataset="test",
            session="1",
            short_tracklet_max_duration_s=0.1,
        )

    modern_output = tmp_path / "modern_annotation"
    _, modern_manifest_path = prepare_annotation_artifact(
        source,
        modern_output,
        dataset="test",
        session="1",
    )
    modern_manifest = load_annotation_manifest(modern_manifest_path)
    modern_manifest.pop("automatic_missing_policy")
    save_annotation_manifest(modern_manifest_path, modern_manifest)

    with pytest.raises(IdentityAnnotationError, match="automatic_missing_policy is missing"):
        load_annotation_manifest(modern_manifest_path)


def test_validation_replays_auto_quarantine_fields_from_manifest_policy(
    tmp_path: Path,
) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())
    output = tmp_path / "annotation"
    observations_path, manifest_path = prepare_annotation_artifact(
        source,
        output,
        dataset="test",
        session="1",
        short_tracklet_max_duration_s=0.1,
    )
    manifest = load_annotation_manifest(manifest_path)
    short_item = next(
        item
        for item in manifest["annotations"]
        if item["auto_quarantine_reason"] == "short_fragment"
    )
    short_item.update(
        {
            "auto_quarantined": False,
            "auto_quarantine_reason": None,
            "status": "unreviewed",
            "note": "",
        }
    )
    save_annotation_manifest(manifest_path, manifest)

    structurally_valid = load_annotation_manifest(manifest_path)
    with pytest.raises(IdentityAnnotationError, match="summary auto_quarantined differs"):
        validate_tracklet_observations(observations_path, structurally_valid)


def test_tampered_tracklet_observations_fail_resume_and_export(tmp_path: Path) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())
    output = tmp_path / "annotation"
    observations_path, manifest_path = prepare_annotation_artifact(
        source, output, dataset="test", session="1"
    )
    tampered = pd.read_csv(observations_path)
    tampered.loc[0, "source_person_idx"] = "999"
    tampered.loc[0, "bbox_x0"] = 199
    tampered.loc[0, "bbox_x1"] = 30
    tampered.to_csv(observations_path, index=False)

    with pytest.raises(IdentityAnnotationError, match="SHA-256 differs"):
        prepare_annotation_artifact(source, output, dataset="test", session="1")
    with pytest.raises(IdentityAnnotationError, match="SHA-256 differs"):
        export_identity_bridge(observations_path, manifest_path, output)


def test_save_cannot_rebind_tampered_source_derived_fields(tmp_path: Path) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())
    output = tmp_path / "annotation"
    observations_path, manifest_path = prepare_annotation_artifact(
        source, output, dataset="test", session="1"
    )
    manifest = load_annotation_manifest(manifest_path)
    observations = validate_tracklet_observations(observations_path, manifest)
    observations.loc[0, "source_person_idx"] = "999"
    observations.loc[0, "bbox_x0"] = float(observations.loc[0, "bbox_x0"]) + 1
    original_hash = manifest["tracklet_observations"]["sha256"]

    with pytest.raises(IdentityAnnotationError, match="gaze source column"):
        save_annotation_state(
            observations_path, manifest_path, observations, manifest
        )

    reloaded_manifest = load_annotation_manifest(manifest_path)
    assert reloaded_manifest["tracklet_observations"]["sha256"] == original_hash
    validate_tracklet_observations(observations_path, reloaded_manifest)


def test_save_recomputes_tracklet_summary_contract(tmp_path: Path) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())
    output = tmp_path / "annotation"
    observations_path, manifest_path = prepare_annotation_artifact(
        source, output, dataset="test", session="1"
    )
    manifest = load_annotation_manifest(manifest_path)
    observations = validate_tracklet_observations(observations_path, manifest)
    manifest["annotations"][0]["observation_count"] += 1

    with pytest.raises(IdentityAnnotationError, match="summary observation_count"):
        save_annotation_state(
            observations_path, manifest_path, observations, manifest
        )


def test_split_state_updates_binding_and_embeds_structural_event(tmp_path: Path) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())
    output = tmp_path / "annotation"
    observations_path, manifest_path = prepare_annotation_artifact(
        source, output, dataset="test", session="1"
    )
    manifest = load_annotation_manifest(manifest_path)
    observations = validate_tracklet_observations(observations_path, manifest)
    tracklet_id = next(
        item["tracklet_id"]
        for item in manifest["annotations"]
        if item["source_kind"] == "participant_detection"
        and item["observation_count"] >= 3
    )

    split_observations, new_id = split_tracklet(
        observations, manifest, tracklet_id, split_frame=2
    )
    save_annotation_state(
        observations_path, manifest_path, split_observations, manifest
    )
    reloaded_manifest = load_annotation_manifest(manifest_path)
    validated = validate_tracklet_observations(observations_path, reloaded_manifest)

    assert new_id in set(validated["tracklet_id"])
    assert reloaded_manifest["structural_events"][-1]["new_tracklet_id"] == new_id


def test_manual_split_reapplies_short_policy_and_auto_tracks_cannot_split(
    tmp_path: Path,
) -> None:
    source = _write_gaze(
        tmp_path / "gaze.csv",
        [_row(frame, 0, (10 + frame, 10, 30 + frame, 40)) for frame in range(6)],
    )
    output = tmp_path / "annotation"
    observations_path, manifest_path = prepare_annotation_artifact(
        source,
        output,
        dataset="test",
        session="1",
        short_tracklet_max_duration_s=0.1,
    )
    manifest = load_annotation_manifest(manifest_path)
    observations = validate_tracklet_observations(observations_path, manifest)
    tracklet_id = next(
        item["tracklet_id"]
        for item in manifest["annotations"]
        if item["source_kind"] == "participant_detection"
    )

    split_observations, new_id = split_tracklet(
        observations,
        manifest,
        tracklet_id,
        split_frame=4,
    )
    old_item = next(
        item for item in manifest["annotations"] if item["tracklet_id"] == tracklet_id
    )
    new_item = next(
        item for item in manifest["annotations"] if item["tracklet_id"] == new_id
    )

    assert old_item["status"] == "unreviewed"
    assert old_item["auto_quarantined"] is False
    assert new_item["status"] == "uncertain"
    assert new_item["auto_quarantined"] is True
    assert new_item["auto_quarantine_reason"] == "short_fragment"
    with pytest.raises(IdentityAnnotationError, match="cannot be manually split"):
        split_tracklet(
            split_observations,
            manifest,
            new_id,
            split_frame=5,
        )

    save_annotation_state(
        observations_path,
        manifest_path,
        split_observations,
        manifest,
    )
    reloaded = load_annotation_manifest(manifest_path)
    validate_tracklet_observations(observations_path, reloaded)


def test_conflicts_are_explicit_and_manual_split_is_auditable(tmp_path: Path) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())
    observations = build_tracklets(load_gaze_observations(source))
    manifest = _manifest(observations)
    participant_ids = sorted(
        observations.loc[
            observations["source_kind"] == "participant_detection", "tracklet_id"
        ].unique()
    )
    for tracklet_id in participant_ids:
        annotation = next(
            item for item in manifest["annotations"] if item["tracklet_id"] == tracklet_id
        )
        for frame_index in annotation["required_review_frames"]:
            mark_review_anchor(manifest, str(tracklet_id), frame_index)
        set_tracklet_annotation(
            manifest, str(tracklet_id), label="A", status="confirmed"
        )

    _, conflicts = build_identity_bridge(observations, manifest)
    split_observations, new_id = split_tracklet(
        observations, manifest, str(participant_ids[0]), split_frame=2
    )

    assert len(conflicts) == 6
    assert new_id.endswith("_s1")
    assert set(
        split_observations.loc[split_observations["frame_index"] >= 2, "tracklet_id"]
    ).issuperset({new_id})
    new_annotation = next(
        item for item in manifest["annotations"] if item["tracklet_id"] == new_id
    )
    assert new_annotation["status"] == "uncertain"
    assert new_annotation["label"] is None
    assert new_annotation["auto_quarantined"] is True
    assert new_annotation["auto_quarantine_reason"] == "singleton_fragment"
    old_annotation = next(
        item
        for item in manifest["annotations"]
        if item["tracklet_id"] == participant_ids[0]
    )
    assert old_annotation["status"] == "unreviewed"
    assert old_annotation["label"] is None


def test_confirmation_requires_all_review_anchors(tmp_path: Path) -> None:
    source = _write_gaze(tmp_path / "gaze.csv", _swapped_detection_rows())
    observations = build_tracklets(load_gaze_observations(source))
    manifest = _manifest(observations)
    tracklet = next(
        item
        for item in manifest["annotations"]
        if item["source_kind"] == "participant_detection"
    )

    with pytest.raises(IdentityAnnotationError, match="review required anchor"):
        set_tracklet_annotation(
            manifest, tracklet["tracklet_id"], label="A", status="confirmed"
        )

    for frame_index in tracklet["required_review_frames"]:
        mark_review_anchor(manifest, tracklet["tracklet_id"], frame_index)
    set_tracklet_annotation(
        manifest, tracklet["tracklet_id"], label="A", status="confirmed"
    )

    assert tracklet["status"] == "confirmed"


def test_long_tracklets_require_periodic_review_anchors() -> None:
    observations = pd.DataFrame(
        [
            {
                "observation_id": frame,
                "frame_index": frame,
                "timestamp_sec": frame / 25,
                "source_kind": "participant_detection",
                "source_person_idx": "0",
                "bbox_x0": 10,
                "bbox_y0": 10,
                "bbox_x1": 30,
                "bbox_y1": 40,
                "tracklet_id": "p000001",
                "link_cost": None if frame == 0 else 0.0,
                "link_confidence": 1.0,
                "link_flags": "new_tracklet" if frame == 0 else "linked",
            }
            for frame in range(301)
        ]
    )

    summary = summarise_tracklets(observations)[0]

    assert {0, 125, 150, 250, 300}.issubset(summary["required_review_frames"])


def test_manifest_rejects_duplicate_tracklet_ids(tmp_path: Path) -> None:
    path = tmp_path / "identity_annotations.json"
    path.write_text(
        json.dumps(
            {
                "artifact_kind": ARTIFACT_KIND,
                "format_version": FORMAT_VERSION,
                "annotations": [{"tracklet_id": "p1"}, {"tracklet_id": "p1"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(IdentityAnnotationError, match="duplicate tracklet IDs"):
        load_annotation_manifest(path)
