from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import hri_turn_taking.data.hpe_transcript_crosswalk as crosswalk
from hri_turn_taking.data.hpe_transcript_crosswalk import (
    AcceptedIdentityWindow,
    HPETranscriptCrosswalkError,
    bind_review_video_identities,
    build_review_samples,
    build_sealed_crosswalk_inventory,
    expected_hpe_persons,
    finalize_crosswalk,
    load_crosswalk_artifact,
    load_crosswalk_review_state,
    load_sealed_crosswalk,
    load_sealed_crosswalk_contract,
    load_sealed_crosswalk_inventory,
    parse_accepted_identity_ledger,
    reaudit_sealed_crosswalk,
    set_sample_decision,
)


def _write_ledger(path: Path) -> Path:
    path.write_text(
        """# Status

## Accepted identity windows

| Scene | Camera | Accepted local frame window | Identity status |
|---|---|---:|---|
| `05_14__3` | Cam_2 | `[10,20)` | accepted, **first segment only** |
| `not_ready__1` | Cam_1 | `[0,30)` | pending |

### Observation-weighted coverage by canonical participant
""",
        encoding="utf-8",
    )
    return path


def test_ledger_and_expected_people_use_strict_end_exclusive_final_person(
    tmp_path: Path,
) -> None:
    accepted = parse_accepted_identity_ledger(_write_ledger(tmp_path / "ledger.md"))

    assert list(accepted) == ["05_14__3"]
    assert accepted["05_14__3"].local_window == (10, 20)
    world = pd.DataFrame(
        {
            "frame_index": [9, 10, 19, 20],
            "world_person": [91, 92, 93, 94],
            "person": [8, 1, 2, 7],
        }
    )

    # The 05_14-style second-segment/end-boundary person is excluded, and the
    # tempting world_person column is never used as the identity key.
    assert expected_hpe_persons(world, (10, 20)) == [1, 2]


def _relabeled_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame in range(10, 20):
        for person in range(1, 5):
            rows.append(
                {
                    "frame_index": frame,
                    "person": person,
                    "bbox_x1": person * 10,
                    "bbox_y1": 10,
                    "bbox_x2": person * 10 + 8,
                    "bbox_y2": 24,
                    "face_matched": 1,
                    "confidence": 0.5 + frame / 100,
                }
            )
    return pd.DataFrame(rows)


def _gaze_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    slots = {"R": "P1", "A": "P2", "B": "P3", "C": "P4"}
    for frame in range(10, 20):
        for index, participant in enumerate(("A", "B", "C", "R")):
            rows.append(
                {
                    "frame_index": frame,
                    "participant": participant,
                    "slot": slots[participant],
                    "source_kind": "participant_detection",
                    "tracklet_id": f"{participant}-{frame}",
                    "bbox_x0": index * 10,
                    "bbox_y0": 0,
                    "bbox_x1": index * 10 + 8,
                    "bbox_y1": 12,
                }
            )
    return pd.DataFrame(rows)


def test_duplicate_hpe_frame_person_boxes_are_excluded_as_ambiguous() -> None:
    frame = pd.DataFrame(
        [
            # Two candidate boxes for the same final person at one instant are
            # ambiguous evidence; neither may be selected for manual review.
            (10, 1, 0, 0, 10, 10),
            (10, 1, 20, 0, 30, 10),
            (11, 1, 0, 0, 10, 10),
            (10, 2, 40, 0, 50, 10),
        ],
        columns=["frame_index", "person", *crosswalk._HPE_BBOX_COLUMNS],
    )

    valid = crosswalk._valid_hpe_rows(
        frame,
        expected_people={1, 2},
        logical_window=(10, 12),
    )

    assert set(zip(valid["frame_index"], valid["person"], strict=True)) == {
        (11, 1),
        (10, 2),
    }


def test_sampling_is_deterministic_clear_and_bound_to_world_presence(
    tmp_path: Path,
) -> None:
    relabeled_path = tmp_path / "relabeled_Cam_2.csv"
    relabeled = _relabeled_rows()
    relabeled.to_csv(relabeled_path, index=False)
    stills: dict[int, dict[str, object]] = {}
    for frame in range(10, 20):
        image = tmp_path / f"frame_{frame}.jpg"
        image.write_bytes(f"frame-{frame}".encode())
        stills[frame] = {
            "kind": "still",
            "path": str(image),
            "size_bytes": image.stat().st_size,
            "sha256": crosswalk.sha256_file(image),
            "camera": "Cam_2",
            "frame_index": frame,
            "logical_frame": frame,
        }
    binding = crosswalk._binding(relabeled_path, pd.read_csv(relabeled_path))
    kwargs = {
        "accepted": AcceptedIdentityWindow("05_14__3", "Cam_2", 10, 20),
        "logical_window": (10, 20),
        "expected_people": [1, 2, 3, 4],
        "relabeled_by_camera": {"Cam_2": relabeled},
        "gaze_confirmed": _gaze_rows(),
        "gaze_frame_offset": 0,
        "gaze_declared_video": {"camera_dir": "Cam_2"},
        "gaze_video_identity": None,
        "relabeled_bindings": {"Cam_2": binding},
        "review_catalogs": {
            "Cam_2": {
                "camera": "Cam_2",
                "stills": stills,
                "segments": [],
                "run_metadata_path": str(tmp_path / "run_metadata.json"),
                "run_metadata_sha256": "unused",
            }
        },
        "world_presence": pd.DataFrame(
            [(frame, person) for frame in range(10, 20) for person in range(1, 5)],
            columns=["frame_index", "person"],
        ),
        "samples_per_person": 3,
    }

    first = build_review_samples(**kwargs)
    second = build_review_samples(**kwargs)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["phase"]) == {"early", "middle", "late"}
    assert first["logical_frame"].between(10, 19).all()
    assert set(first["hpe_camera"]) == {"Cam_2"}
    assert all(json.loads(value)["kind"] == "still" for value in first["hpe_image_source_json"])
    assert all(
        json.loads(value) == {"kind": "same_as_hpe"}
        for value in first["gaze_image_source_json"]
    )


def test_partial_first_segment_derives_logical_window_from_segment_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_dir = tmp_path / "identity"
    identity_dir.mkdir()
    (identity_dir / "identity_annotations.json").write_text("{}", encoding="utf-8")
    (identity_dir / "tracklet_observations.csv").write_text("dummy\n1\n", encoding="utf-8")
    manifest = {
        "dataset": "05_14",
        "session": "3",
        "source": {"declared_video": {"camera_dir": "Cam_2"}},
        "annotation_scope": {
            "core_start_logical_frame_inclusive": 10,
            "core_end_logical_frame_exclusive": 100,
            "gaze_segment_logical_start_frame": 0,
            "gaze_local_start_frame_inclusive": 10,
            "gaze_local_end_frame_exclusive": 50,
        },
        "annotations": [{"status": "confirmed"}],
    }
    bridge = _gaze_rows().loc[lambda frame: frame["frame_index"] == 10].copy()
    bridge["annotation_status"] = "confirmed"
    observations = bridge.drop(columns=["participant", "slot", "annotation_status"])
    monkeypatch.setattr(crosswalk, "load_annotation_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        crosswalk, "validate_tracklet_observations", lambda _path, _manifest: observations
    )
    monkeypatch.setattr(crosswalk, "annotation_frame_window", lambda _manifest: (10, 50))
    monkeypatch.setattr(
        crosswalk,
        "build_identity_bridge",
        lambda _observations, _manifest: (bridge, bridge.iloc[0:0]),
    )

    result = crosswalk._validate_gaze_reference(
        scene="05_14__3",
        accepted=AcceptedIdentityWindow("05_14__3", "Cam_2", 10, 50),
        identity_dir=identity_dir,
    )

    assert result["logical_window"] == (10, 50)
    assert result["frame_offset"] == 0


def _artifact(
    tmp_path: Path,
    *,
    world_rows: list[tuple[int, int]],
    people: list[int],
    labels: dict[int, str],
    proposal: dict[int, str] | None = None,
    unreviewed: tuple[int, str] | None = None,
) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    still = tmp_path / "evidence.jpg"
    still.write_bytes(b"immutable evidence")
    still_source = json.dumps(
        {
            "kind": "still",
            "path": str(still),
            "size_bytes": still.stat().st_size,
            "sha256": crosswalk.sha256_file(still),
            "camera": "Cam_2",
            "frame_index": 1,
        },
        sort_keys=True,
    )
    sample_rows: list[dict[str, object]] = []
    decisions: dict[str, dict[str, object]] = {}
    for person in people:
        for phase_index, phase in enumerate(("early", "middle", "late")):
            sample_id = f"person_{person:03d}_{phase}"
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "hpe_person": person,
                    "phase": phase,
                    "logical_frame": 1 + phase_index,
                    "gaze_local_frame": 1 + phase_index,
                    "hpe_camera": "Cam_2",
                    "gaze_camera": "Cam_2",
                    "hpe_bbox_json": "[0,0,2,2]",
                    "all_hpe_boxes_json": "[]",
                    "gaze_confirmed_boxes_json": "[]",
                    "hpe_image_source_json": still_source,
                    "gaze_image_source_json": '{"kind":"same_as_hpe"}',
                }
            )
            status = "unreviewed" if unreviewed == (person, phase) else "confirmed"
            decisions[sample_id] = {
                "status": status,
                "label": None if status == "unreviewed" else labels[person],
                "note": "",
                "updated_at": None,
            }
    samples = pd.DataFrame(sample_rows, columns=crosswalk._SAMPLE_COLUMNS)
    samples_path = tmp_path / crosswalk.SAMPLES_NAME
    samples.to_csv(samples_path, index=False)
    world = pd.DataFrame(world_rows, columns=["frame_index", "person"])
    world_path = tmp_path / "world_persons.csv"
    world.to_csv(world_path, index=False)
    manifest = {
        "artifact_kind": crosswalk.ARTIFACT_KIND,
        "format_version": crosswalk.FORMAT_VERSION,
        "scene": "test__1",
        "accepted_window": {
            "scene": "test__1",
            "camera": "Cam_2",
            "local_start_inclusive": 0,
            "local_end_exclusive": 10,
            "logical_start_inclusive": 0,
            "logical_end_exclusive": 10,
        },
        "expected_hpe_persons": people,
        "sources": {
            "hpe": {
                "delivery_world_persons": crosswalk._binding(
                    world_path, pd.read_csv(world_path)
                )
            }
        },
        "upstream_proposal": {
            str(person): label
            for person, label in (proposal or labels).items()
        },
        "sample_decisions": decisions,
        "state": "in_review",
        "review_video_identities": None,
        "upstream_mismatch": None,
        "conflict_report": None,
        "final_mapping": None,
        "review_samples": crosswalk._binding(samples_path, pd.read_csv(samples_path)),
    }
    manifest_path = tmp_path / crosswalk.MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return samples_path, manifest_path, still


def _replace_still_with_bound_video(
    samples_path: Path, manifest_path: Path
) -> Path:
    video = samples_path.parent / "Cam_2" / "review.avi"
    video.parent.mkdir()
    video.write_bytes(b"a" * (4 * 1024 * 1024))
    identity = {
        "basename": video.name,
        "camera_dir": "Cam_2",
        "fps": 30.0,
        "frame_count": 10,
        "width": 320,
        "height": 180,
        "size_bytes": video.stat().st_size,
    }
    source = json.dumps(
        {
            "kind": "video",
            "path": str(video),
            "camera": "Cam_2",
            "frame_index": 1,
            "video_identity": identity,
        },
        sort_keys=True,
    )
    samples = pd.read_csv(samples_path)
    samples["hpe_image_source_json"] = source
    samples["gaze_image_source_json"] = '{"kind":"same_as_hpe"}'
    samples.to_csv(samples_path, index=False)
    actual = {
        "path": str(video.resolve()),
        "fps": 30.0,
        "frame_count": 10,
        "width": 320,
        "height": 180,
        "size_bytes": video.stat().st_size,
        "fingerprint_algorithm": crosswalk._SPARSE_VIDEO_ALGORITHM,
        "sparse_sha256": crosswalk._sparse_video_fingerprint(video),
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_samples"] = crosswalk._binding(
        samples_path, pd.read_csv(samples_path)
    )
    manifest["review_video_identities"] = {str(video.resolve()): actual}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return video


def test_many_to_one_nonoverlap_is_allowed_and_all_artifacts_are_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "A", 3: "B", 4: "C", 5: "R"}
    samples, manifest, _ = _artifact(
        tmp_path,
        world_rows=[(1, 1), (2, 2), (1, 3), (1, 4), (1, 5)],
        people=[1, 2, 3, 4, 5],
        labels=labels,
    )
    monkeypatch.setattr(crosswalk, "_replay_sources", lambda _manifest: None)

    output = finalize_crosswalk(samples, manifest)
    loaded = pd.read_csv(output)
    state = json.loads(manifest.read_text(encoding="utf-8"))

    assert list(loaded["participant"]).count("A") == 2
    assert state["state"] == "finalized"
    assert state["conflict_report"]["n_rows"] == 0
    assert (tmp_path / crosswalk.CONFLICTS_NAME).is_file()


def test_same_frame_many_to_one_is_blocked_with_conflict_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "A", 3: "B", 4: "C", 5: "R"}
    samples, manifest, _ = _artifact(
        tmp_path,
        world_rows=[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
        people=[1, 2, 3, 4, 5],
        labels=labels,
    )
    monkeypatch.setattr(crosswalk, "_replay_sources", lambda _manifest: None)

    with pytest.raises(HPETranscriptCrosswalkError, match="coexist at frame 1"):
        finalize_crosswalk(samples, manifest)

    conflicts = pd.read_csv(tmp_path / crosswalk.CONFLICTS_NAME)
    assert set(conflicts["hpe_person"]) == {1, 2}
    assert json.loads(manifest.read_text(encoding="utf-8"))["state"] == "in_review"


def test_finalize_rejects_unreviewed_anchor_and_missing_canonical_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "B", 3: "C", 4: "R"}
    samples, manifest, _ = _artifact(
        tmp_path / "unreviewed",
        world_rows=[(1, person) for person in labels],
        people=list(labels),
        labels=labels,
        unreviewed=(1, "middle"),
    )
    monkeypatch.setattr(crosswalk, "_replay_sources", lambda _manifest: None)
    with pytest.raises(HPETranscriptCrosswalkError, match="unconfirmed review anchor"):
        finalize_crosswalk(samples, manifest)

    missing = {1: "A", 2: "A", 3: "B", 4: "C"}
    samples, manifest, _ = _artifact(
        tmp_path / "missing_label",
        world_rows=[(1, person) for person in missing],
        people=list(missing),
        labels=missing,
    )
    with pytest.raises(HPETranscriptCrosswalkError, match="exactly A/B/C/R"):
        finalize_crosswalk(samples, manifest)


def test_load_rejects_missing_expected_hpe_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "B", 3: "C", 4: "R"}
    samples, manifest_path, _ = _artifact(
        tmp_path,
        world_rows=[(1, person) for person in labels],
        people=list(labels),
        labels=labels,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expected_hpe_persons"] = [1, 2, 3, 4, 5]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(crosswalk, "_replay_sources", lambda _manifest: None)

    with pytest.raises(HPETranscriptCrosswalkError, match="exactly cover expected"):
        load_crosswalk_artifact(samples, manifest_path)


def test_upstream_mismatch_requires_explicit_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "B", 3: "C", 4: "R"}
    proposal = {1: "B", 2: "A", 3: "C", 4: "R"}
    samples, manifest, _ = _artifact(
        tmp_path,
        world_rows=[(1, person) for person in labels],
        people=list(labels),
        labels=labels,
        proposal=proposal,
    )
    monkeypatch.setattr(crosswalk, "_replay_sources", lambda _manifest: None)

    with pytest.raises(HPETranscriptCrosswalkError, match="explicit acknowledgement"):
        finalize_crosswalk(samples, manifest)
    output = finalize_crosswalk(
        samples,
        manifest,
        upstream_mismatch_acknowledgement="independently checked all visual anchors",
    )

    assert output.is_file()
    state = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(state["upstream_mismatch"]["differences"]) == 2


def test_source_or_sample_tamper_blocks_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "B", 3: "C", 4: "R"}
    samples, manifest, still = _artifact(
        tmp_path,
        world_rows=[(1, person) for person in labels],
        people=list(labels),
        labels=labels,
    )
    monkeypatch.setattr(crosswalk, "_replay_sources", lambda _manifest: None)
    still.write_bytes(b"tampered evidence")
    with pytest.raises(HPETranscriptCrosswalkError, match="review still changed"):
        load_crosswalk_artifact(samples, manifest)

    still.write_bytes(b"immutable evidence")
    with samples.open("a", encoding="utf-8") as handle:
        handle.write("tamper")
    with pytest.raises(HPETranscriptCrosswalkError, match="SHA-256 changed"):
        load_crosswalk_artifact(samples, manifest)


def test_decision_refresh_is_local_but_finalize_still_full_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "B", 3: "C", 4: "R"}
    samples, manifest, _ = _artifact(
        tmp_path,
        world_rows=[(1, person) for person in labels],
        people=list(labels),
        labels=labels,
    )

    def reject_full_replay(_manifest: dict[str, object]) -> None:
        raise HPETranscriptCrosswalkError("full replay sentinel")

    monkeypatch.setattr(crosswalk, "_replay_sources", reject_full_replay)
    set_sample_decision(
        samples,
        manifest,
        sample_id="person_001_middle",
        label="A",
        status="confirmed",
    )
    state, _ = load_crosswalk_review_state(samples, manifest)
    assert state["sample_decisions"]["person_001_middle"]["label"] == "A"
    with pytest.raises(HPETranscriptCrosswalkError, match="full replay sentinel"):
        finalize_crosswalk(samples, manifest)


def test_full_load_rejects_deleted_bound_review_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "B", 3: "C", 4: "R"}
    samples, manifest, _ = _artifact(
        tmp_path,
        world_rows=[(1, person) for person in labels],
        people=list(labels),
        labels=labels,
    )
    video = _replace_still_with_bound_video(samples, manifest)
    monkeypatch.setattr(crosswalk, "_replay_sources", lambda _manifest: None)
    load_crosswalk_artifact(samples, manifest)
    video.unlink()

    with pytest.raises(HPETranscriptCrosswalkError, match="does not exist"):
        load_crosswalk_artifact(samples, manifest)


def test_full_load_rejects_same_size_sparse_region_video_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "B", 3: "C", 4: "R"}
    samples, manifest, _ = _artifact(
        tmp_path,
        world_rows=[(1, person) for person in labels],
        people=list(labels),
        labels=labels,
    )
    video = _replace_still_with_bound_video(samples, manifest)
    monkeypatch.setattr(crosswalk, "_replay_sources", lambda _manifest: None)
    with video.open("r+b") as handle:
        handle.seek(100)
        handle.write(b"tampered-without-size-change")

    with pytest.raises(HPETranscriptCrosswalkError, match="sparse fingerprint changed"):
        load_crosswalk_artifact(samples, manifest)


def test_full_load_rejects_unrelated_extra_video_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "B", 3: "C", 4: "R"}
    samples, manifest_path, _ = _artifact(
        tmp_path,
        world_rows=[(1, person) for person in labels],
        people=list(labels),
        labels=labels,
    )
    video = _replace_still_with_bound_video(samples, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extra = dict(manifest["review_video_identities"][str(video.resolve())])
    extra_path = tmp_path / "Cam_3" / "unrelated.avi"
    extra["path"] = str(extra_path.resolve())
    manifest["review_video_identities"][str(extra_path.resolve())] = extra
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(crosswalk, "_replay_sources", lambda _manifest: None)

    with pytest.raises(HPETranscriptCrosswalkError, match="exactly cover"):
        load_crosswalk_artifact(samples, manifest_path)


def test_still_only_artifact_rejects_unrelated_video_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "B", 3: "C", 4: "R"}
    samples, manifest_path, _ = _artifact(
        tmp_path,
        world_rows=[(1, person) for person in labels],
        people=list(labels),
        labels=labels,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_video_identities"] = {"unrelated": {"path": "unrelated.avi"}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(crosswalk, "_replay_sources", lambda _manifest: None)

    with pytest.raises(HPETranscriptCrosswalkError, match="no sample requires"):
        load_crosswalk_artifact(samples, manifest_path)


def test_offline_placeholder_is_rejected_before_video_open_or_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "B", 3: "C", 4: "R"}
    samples, manifest, _ = _artifact(
        tmp_path,
        world_rows=[(1, person) for person in labels],
        people=list(labels),
        labels=labels,
    )
    _replace_still_with_bound_video(samples, manifest)
    monkeypatch.setattr(crosswalk, "_replay_sources", lambda _manifest: None)
    monkeypatch.setattr(
        crosswalk,
        "_windows_file_attributes",
        lambda _path: crosswalk._CLOUD_RECALL_FLAGS,
    )
    monkeypatch.setattr(
        crosswalk,
        "_sparse_video_fingerprint",
        lambda _path: pytest.fail("offline placeholder must not be opened"),
    )

    with pytest.raises(HPETranscriptCrosswalkError, match="Offline cloud placeholder"):
        load_crosswalk_artifact(samples, manifest)


def test_video_bind_is_fast_local_but_finalize_still_full_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "B", 3: "C", 4: "R"}
    samples, manifest_path, _ = _artifact(
        tmp_path,
        world_rows=[(1, person) for person in labels],
        people=list(labels),
        labels=labels,
    )
    video = _replace_still_with_bound_video(samples, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bindings = dict(manifest["review_video_identities"])
    manifest["review_video_identities"] = None
    for decision in manifest["sample_decisions"].values():
        decision["status"] = "unreviewed"
        decision["label"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def reject_full_replay(_manifest: dict[str, object]) -> None:
        raise HPETranscriptCrosswalkError("full replay sentinel")

    monkeypatch.setattr(crosswalk, "_replay_sources", reject_full_replay)
    bind_review_video_identities(samples, manifest_path, bindings=bindings)
    state, _ = load_crosswalk_review_state(samples, manifest_path)
    assert list(state["review_video_identities"]) == [str(video.resolve())]
    with pytest.raises(HPETranscriptCrosswalkError, match="full replay sentinel"):
        finalize_crosswalk(samples, manifest_path)


def test_still_only_empty_video_bind_is_fast_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {1: "A", 2: "B", 3: "C", 4: "R"}
    samples, manifest_path, _ = _artifact(
        tmp_path,
        world_rows=[(1, person) for person in labels],
        people=list(labels),
        labels=labels,
    )
    before = manifest_path.read_bytes()

    def reject_full_replay(_manifest: dict[str, object]) -> None:
        raise HPETranscriptCrosswalkError("bind must not perform a full replay")

    monkeypatch.setattr(crosswalk, "_replay_sources", reject_full_replay)
    bind_review_video_identities(samples, manifest_path, bindings={})

    assert manifest_path.read_bytes() == before


def _sealed_ledger(path: Path, *, scene: str = "05_14__3") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Status

## Accepted identity windows

| Scene | Camera | Accepted local frame window | Identity status |
|---|---|---:|---|
| `{scene}` | Cam_2 | `[0,10)` | accepted |

### Observation-weighted coverage by canonical participant
""",
        encoding="utf-8",
    )
    return path


def _sealed_fixture(
    tmp_path: Path,
    *,
    scene: str = "05_14__3",
    people: list[int] | None = None,
    raw_video: bool = False,
) -> dict[str, Path]:
    people = people or [1, 2, 3, 4]
    labels = (
        {1: "R", 2: "B", 3: "A", 4: "C", 5: "A", 6: "C"}
        if people == [1, 2, 3, 4, 5, 6]
        else {
            person: ("A", "B", "C", "R")[(person - 1) % 4]
            for person in people
        }
    )
    frames_by_person = {
        person: ((4, 5, 6) if person in {5, 6} else (1, 2, 3))
        for person in people
    }
    repo = tmp_path / "repo"
    scene_dir = repo / "data" / "annotations" / "hpe_transcript_crosswalk" / scene
    upstream = repo / "upstream"
    scene_dir.mkdir(parents=True)
    upstream.mkdir()
    live_ledger = _sealed_ledger(repo / "docs" / "IDENTITY_DATA_STATUS.md", scene=scene)
    immutable_ledger = repo / "configs" / "hpe_transcript_crosswalk_ledger_snapshot.md"
    immutable_ledger.parent.mkdir()
    immutable_ledger.write_bytes(live_ledger.read_bytes())

    still = upstream / "evidence.jpg"
    still.write_bytes(b"sealed immutable review still")
    video = upstream / "Cam_2" / "review.avi"
    if raw_video:
        video.parent.mkdir()
        video.write_bytes(b"v" * (4 * 1024 * 1024))
        identity = {
            "basename": video.name,
            "camera_dir": "Cam_2",
            "fps": 30.0,
            "frame_count": 10,
            "width": 320,
            "height": 180,
            "size_bytes": video.stat().st_size,
        }
        hpe_source = {
            "kind": "video",
            "path": str(video.resolve()),
            "camera": "Cam_2",
            "frame_index": 1,
            "video_identity": identity,
        }
        video_binding = {
            "path": str(video.resolve()),
            "fps": 30.0,
            "frame_count": 10,
            "width": 320,
            "height": 180,
            "size_bytes": video.stat().st_size,
            "fingerprint_algorithm": crosswalk._SPARSE_VIDEO_ALGORITHM,
            "sparse_sha256": crosswalk._sparse_video_fingerprint(video),
        }
        review_videos: dict[str, object] | None = {
            str(video.resolve()): video_binding
        }
    else:
        hpe_source = {
            "kind": "still",
            "path": str(still.resolve()),
            "size_bytes": still.stat().st_size,
            "sha256": crosswalk.sha256_file(still),
            "camera": "Cam_2",
            "frame_index": 1,
        }
        review_videos = None

    rows: list[dict[str, object]] = []
    decisions: dict[str, dict[str, object]] = {}
    audit: list[dict[str, object]] = []
    sample_ids: list[str] = []
    phases = ["early", "middle", "late"]
    for person in people:
        for frame, phase in zip(frames_by_person[person], phases, strict=True):
            sample_id = f"person_{person:03d}_{phase}"
            sample_ids.append(sample_id)
            rows.append(
                {
                    "sample_id": sample_id,
                    "hpe_person": person,
                    "phase": phase,
                    "logical_frame": frame,
                    "gaze_local_frame": frame,
                    "hpe_camera": "Cam_2",
                    "gaze_camera": "Cam_2",
                    "hpe_bbox_json": "[0,0,2,2]",
                    "all_hpe_boxes_json": "[]",
                    "gaze_confirmed_boxes_json": "[]",
                    "hpe_image_source_json": json.dumps(hpe_source, sort_keys=True),
                    "gaze_image_source_json": '{"kind":"same_as_hpe"}',
                }
            )
            decisions[sample_id] = {
                "status": "confirmed",
                "label": labels[person],
                "note": f"reviewed {sample_id}",
                "updated_at": f"decision-{sample_id}",
            }
    samples = pd.DataFrame.from_records(rows, columns=crosswalk._SAMPLE_COLUMNS)
    samples_path = scene_dir / crosswalk.SAMPLES_NAME
    samples.to_csv(samples_path, index=False)

    world = pd.DataFrame(
        [
            (frame, person)
            for person in people
            for frame in frames_by_person[person]
        ],
        columns=["frame_index", "person"],
    )
    world_path = upstream / "world_persons.csv"
    world.to_csv(world_path, index=False)
    run_metadata_path = upstream / "run_metadata.json"
    run_metadata_path.write_text(
        json.dumps({"source_fps": 30.0}), encoding="utf-8"
    )
    validation_path = upstream / "validation_complete.json"
    validation_path.write_text(
        json.dumps(
            {
                "pose_sources": {
                    "cameras": {
                        "Cam_2": {
                            "bound_camera": "Cam_2",
                            "metadata": run_metadata_path.name,
                            "metadata_sha256": crosswalk.sha256_file(run_metadata_path),
                            "source_fps": 30.0,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    mapping = pd.DataFrame(
        [
            {
                "hpe_person": person,
                "participant": labels[person],
                "slot": crosswalk.LABEL_TO_SLOT[labels[person]],
                "review_status": "confirmed",
                "evidence_sample_ids": ";".join(
                    f"person_{person:03d}_{phase}" for phase in phases
                ),
            }
            for person in people
        ]
    )
    mapping_path = scene_dir / crosswalk.FINAL_MAPPING_NAME
    mapping.to_csv(mapping_path, index=False)
    conflicts = pd.DataFrame(
        columns=["frame_index", "participant", "hpe_person"]
    )
    conflicts_path = scene_dir / crosswalk.CONFLICTS_NAME
    conflicts.to_csv(conflicts_path, index=False)

    audit.append(
        {
            "timestamp": "prepare-time",
            "action": "prepare",
            "scene": scene,
            "expected_hpe_persons": people,
            "sample_ids": sample_ids,
        }
    )
    if review_videos is not None:
        audit.append(
            {
                "timestamp": "bind-time",
                "action": "bind_review_video_identities",
                "bindings": review_videos,
            }
        )
    for sample_id in sample_ids:
        audit.append(
            {
                "timestamp": decisions[sample_id]["updated_at"],
                "action": "set_sample_decision",
                "sample_id": sample_id,
                "previous": {
                    "status": "unreviewed",
                    "label": None,
                    "note": "",
                    "updated_at": None,
                },
                "current": decisions[sample_id],
            }
        )
    audit.append(
        {
            "timestamp": "final-time",
            "action": "finalize",
            "labels": {str(person): labels[person] for person in people},
            "upstream_mismatch": [],
            "acknowledgement": None,
            "final_mapping_sha256": crosswalk.sha256_file(mapping_path),
        }
    )
    audit_path = scene_dir / crosswalk.AUDIT_NAME
    audit_path.write_text(
        "".join(json.dumps(event) + "\n" for event in audit), encoding="utf-8"
    )
    manifest = {
        "artifact_kind": crosswalk.ARTIFACT_KIND,
        "format_version": crosswalk.FORMAT_VERSION,
        "scene": scene,
        "accepted_window": {
            "scene": scene,
            "camera": "Cam_2",
            "local_start_inclusive": 0,
            "local_end_exclusive": 10,
            "logical_start_inclusive": 0,
            "logical_end_exclusive": 10,
            "filter_rule": "start <= frame_index < end",
        },
        "expected_hpe_persons": people,
        "sampling_policy": {
            "format_version": crosswalk.SAMPLE_FORMAT_VERSION,
            "samples_per_person": 3,
            "phases": phases,
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
            "ledger": crosswalk._binding(live_ledger),
            "hpe": {
                "delivery_world_persons": crosswalk._binding(
                    world_path, pd.read_csv(world_path)
                ),
                "validation_complete": crosswalk._binding(validation_path),
            },
            "hpe_run_metadata": {
                "Cam_2": crosswalk._binding(run_metadata_path),
            },
        },
        "upstream_proposal": {str(person): labels[person] for person in people},
        "sample_decisions": decisions,
        "state": "finalized",
        "review_video_identities": review_videos,
        "upstream_mismatch": {"differences": [], "acknowledgement": None},
        "conflict_report": crosswalk._binding(
            conflicts_path, pd.read_csv(conflicts_path)
        ),
        "final_mapping": crosswalk._binding(mapping_path, pd.read_csv(mapping_path)),
        "review_samples": crosswalk._binding(samples_path, pd.read_csv(samples_path)),
        "created_at": "prepare-time",
        "updated_at": "final-time",
    }
    manifest_path = scene_dir / crosswalk.MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    inventory = {
        "artifact_kind": crosswalk.SEALED_INVENTORY_KIND,
        "format_version": crosswalk.SEALED_INVENTORY_VERSION,
        "trust_model": "version_controlled_exact_hash_inventory",
        "artifact_root": "data/annotations/hpe_transcript_crosswalk",
        "ledger_snapshot": {
            "source_path": "docs/IDENTITY_DATA_STATUS.md",
            "sealed_path": "configs/hpe_transcript_crosswalk_ledger_snapshot.md",
            "sha256": crosswalk.sha256_file(live_ledger),
            "accepted_windows": {
                scene: {
                    "scene": scene,
                    "camera": "Cam_2",
                    "local_start_inclusive": 0,
                    "local_end_exclusive": 10,
                }
            },
        },
        "scenes": {},
    }
    inventory["scenes"][scene] = {
        "artifact_dir": f"data/annotations/hpe_transcript_crosswalk/{scene}",
        "accepted_window": manifest["accepted_window"],
        "expected_hpe_persons": people,
        "files": {},
    }
    for filename in crosswalk.SEALED_ARTIFACT_FILES:
        path = scene_dir / filename
        inventory["scenes"][scene]["files"][filename] = {
            "size_bytes": path.stat().st_size,
            "sha256": crosswalk.sha256_file(path),
        }
    inventory_path = repo / "configs" / "hpe_transcript_crosswalk_inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    return {
        "repo": repo,
        "scene_dir": scene_dir,
        "inventory": inventory_path,
        "live_ledger": live_ledger,
        "immutable_ledger": immutable_ledger,
        "video": video,
        "world": world_path,
        "run_metadata": run_metadata_path,
    }


def _refresh_sealed_inventory(paths: dict[str, Path]) -> None:
    inventory = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    scene = paths["scene_dir"].name
    for filename in crosswalk.SEALED_ARTIFACT_FILES:
        path = paths["scene_dir"] / filename
        inventory["scenes"][scene]["files"][filename] = {
            "size_bytes": path.stat().st_size,
            "sha256": crosswalk.sha256_file(path),
        }
    paths["inventory"].write_text(json.dumps(inventory), encoding="utf-8")


def test_sealed_downstream_allows_offline_video_but_strict_still_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _sealed_fixture(tmp_path, raw_video=True)
    monkeypatch.setattr(
        crosswalk, "_replay_nonledger_sources", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        crosswalk,
        "_windows_file_attributes",
        lambda _path: crosswalk._CLOUD_RECALL_FLAGS,
    )
    monkeypatch.setattr(
        crosswalk,
        "_sparse_video_fingerprint",
        lambda _path: pytest.fail("sealed downstream must not open the raw AVI"),
    )

    _, _, mapping = load_sealed_crosswalk(
        "05_14__3", paths["inventory"], repository_root=paths["repo"]
    )
    assert mapping["participant"].tolist() == ["A", "B", "C", "R"]
    with pytest.raises(HPETranscriptCrosswalkError, match="Offline cloud placeholder"):
        load_crosswalk_artifact(paths["scene_dir"])


def test_sealed_downstream_contract_exposes_only_validated_consumption_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _sealed_fixture(tmp_path)
    monkeypatch.setattr(
        crosswalk, "_replay_nonledger_sources", lambda *_args, **_kwargs: None
    )

    contract = load_sealed_crosswalk_contract(
        "05_14__3", paths["inventory"], repository_root=paths["repo"]
    )

    assert contract["artifact_kind"] == crosswalk.SEALED_DOWNSTREAM_KIND
    assert contract["inventory"]["sha256"] == crosswalk.sha256_file(
        paths["inventory"]
    )
    assert contract["inventory"]["repository_path"].startswith("configs/")
    assert contract["accepted_window"]["logical_end_exclusive"] == 10
    assert contract["source_fps"] == 30.0
    assert contract["hpe_logical_timing_authority"] == {
        "axis": "hpe_delivery_logical_frame_index",
        "source_fps": 30.0,
        "evidence_cameras": [
            {
                "camera": "Cam_2",
                "source_fps": 30.0,
                "run_metadata": crosswalk._binding(paths["run_metadata"]),
            }
        ],
        "validation_complete": contract["hpe_validation_authority"],
    }
    assert contract["person_mapping"] == [
        {"hpe_person": 1, "participant": "A", "slot": "P2"},
        {"hpe_person": 2, "participant": "B", "slot": "P3"},
        {"hpe_person": 3, "participant": "C", "slot": "P4"},
        {"hpe_person": 4, "participant": "R", "slot": "P1"},
    ]
    assert set(contract["artifact_files"]) == crosswalk.SEALED_ARTIFACT_FILES
    assert contract["delivery_world_persons"]["sha256"] == crosswalk.sha256_file(
        paths["world"]
    )


def test_sealed_downstream_accepts_portable_text_dtype_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pandas 2 ``object`` and pandas 3 ``str`` are parser metadata, not bytes."""

    paths = _sealed_fixture(tmp_path)
    manifest_path = paths["scene_dir"] / crosswalk.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("review_samples", "final_mapping", "conflict_report"):
        manifest[key]["parsed_dtypes"] = [
            "str" if value == "object" else value
            for value in manifest[key]["parsed_dtypes"]
        ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_sealed_inventory(paths)
    monkeypatch.setattr(
        crosswalk,
        "_replay_nonledger_sources",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(HPETranscriptCrosswalkError, match="schema or row count"):
        crosswalk.load_crosswalk_review_state(paths["scene_dir"])
    reaudit_sealed_crosswalk(
        "05_14__3", paths["inventory"], repository_root=paths["repo"]
    )
    contract = load_sealed_crosswalk_contract(
        "05_14__3", paths["inventory"], repository_root=paths["repo"]
    )

    assert contract["source_fps"] == 30.0


def test_canonical_authority_rejects_byte_identical_delivery_copy(
    tmp_path: Path,
) -> None:
    canonical_path = tmp_path / "outputs" / "delivery" / "scene" / "world_persons.csv"
    copied_path = tmp_path / "copied" / "world_persons.csv"
    canonical_path.parent.mkdir(parents=True)
    copied_path.parent.mkdir(parents=True)
    frame = pd.DataFrame({"frame_index": [1], "person": [1]})
    frame.to_csv(canonical_path, index=False)
    copied_path.write_bytes(canonical_path.read_bytes())
    canonical = crosswalk._binding(canonical_path, pd.read_csv(canonical_path))
    copied = crosswalk._binding(copied_path, pd.read_csv(copied_path))

    with pytest.raises(HPETranscriptCrosswalkError, match="canonical delivery"):
        crosswalk._require_canonical_authority_bindings(
            recorded_hpe={"delivery_world_persons": copied},
            canonical_hpe={"delivery_world_persons": canonical},
            recorded_gaze={},
            canonical_gaze={},
            portable_dtypes=True,
        )


def test_sealed_timing_uses_bound_hpe_camera_when_gaze_cam1_is_excluded(
    tmp_path: Path,
) -> None:
    """05_15 uses Gaze Cam_1, while production HPE timing begins at Cam_2."""

    paths = _sealed_fixture(tmp_path, scene="05_15__3")
    manifest_path = paths["scene_dir"] / crosswalk.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["accepted_window"]["camera"] = "Cam_1"
    samples = pd.read_csv(paths["scene_dir"] / crosswalk.SAMPLES_NAME)
    authority = crosswalk._resolve_hpe_logical_timing_authority(
        manifest=manifest,
        samples=samples,
        validation_binding=manifest["sources"]["hpe"]["validation_complete"],
    )

    assert manifest["accepted_window"]["camera"] == "Cam_1"
    assert authority["source_fps"] == 30.0
    assert [
        item["camera"]
        for item in authority["evidence_cameras"]
    ] == ["Cam_2"]


def test_live_ledger_update_preserves_sealed_load_and_snapshot_reaudit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _sealed_fixture(tmp_path)
    monkeypatch.setattr(
        crosswalk, "_replay_nonledger_sources", lambda *_args, **_kwargs: None
    )
    paths["live_ledger"].write_text("# later status wording\n", encoding="utf-8")

    load_sealed_crosswalk(
        "05_14__3", paths["inventory"], repository_root=paths["repo"]
    )
    reaudit_sealed_crosswalk(
        "05_14__3", paths["inventory"], repository_root=paths["repo"]
    )
    with pytest.raises(HPETranscriptCrosswalkError, match="identity ledger source changed"):
        load_crosswalk_artifact(paths["scene_dir"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unfinished", "not the matching finalized"),
        ("conflict", "non-empty conflict report"),
        ("audit", "not closed by one finalize"),
        ("slot", "mapping row"),
        ("person", "expected-person universe changed"),
    ],
)
def test_sealed_semantic_tamper_is_rejected_even_after_hash_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    paths = _sealed_fixture(tmp_path)
    monkeypatch.setattr(
        crosswalk, "_replay_nonledger_sources", lambda *_args, **_kwargs: None
    )
    manifest_path = paths["scene_dir"] / crosswalk.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "unfinished":
        manifest["state"] = "in_review"
    elif mutation == "conflict":
        conflict_path = paths["scene_dir"] / crosswalk.CONFLICTS_NAME
        pd.DataFrame([(1, "A", 1)], columns=["frame_index", "participant", "hpe_person"]).to_csv(
            conflict_path, index=False
        )
        manifest["conflict_report"] = crosswalk._binding(
            conflict_path, pd.read_csv(conflict_path)
        )
    elif mutation == "audit":
        audit_path = paths["scene_dir"] / crosswalk.AUDIT_NAME
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        audit_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    elif mutation == "slot":
        mapping_path = paths["scene_dir"] / crosswalk.FINAL_MAPPING_NAME
        mapping = pd.read_csv(mapping_path)
        mapping.loc[0, "slot"] = "P4"
        mapping.to_csv(mapping_path, index=False)
        manifest["final_mapping"] = crosswalk._binding(
            mapping_path, pd.read_csv(mapping_path)
        )
        audit_path = paths["scene_dir"] / crosswalk.AUDIT_NAME
        events = [json.loads(line) for line in audit_path.read_text().splitlines()]
        events[-1]["final_mapping_sha256"] = manifest["final_mapping"]["sha256"]
        audit_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )
    elif mutation == "person":
        manifest["expected_hpe_persons"] = [1, 2, 3]
        manifest["sample_decisions"] = {
            key: value
            for key, value in manifest["sample_decisions"].items()
            if not key.startswith("person_004_")
        }
        samples_path = paths["scene_dir"] / crosswalk.SAMPLES_NAME
        samples = pd.read_csv(samples_path).loc[lambda frame: frame["hpe_person"] != 4]
        samples.to_csv(samples_path, index=False)
        manifest["review_samples"] = crosswalk._binding(
            samples_path, pd.read_csv(samples_path)
        )
        inventory = json.loads(paths["inventory"].read_text(encoding="utf-8"))
        inventory["scenes"]["05_14__3"]["expected_hpe_persons"] = [1, 2, 3]
        paths["inventory"].write_text(json.dumps(inventory), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_sealed_inventory(paths)

    with pytest.raises(HPETranscriptCrosswalkError, match=message):
        load_sealed_crosswalk(
            "05_14__3", paths["inventory"], repository_root=paths["repo"]
        )


def test_sealed_inventory_rejects_extra_missing_scene_or_file(tmp_path: Path) -> None:
    paths = _sealed_fixture(tmp_path)
    extra = paths["scene_dir"] / "unsealed.txt"
    extra.write_text("extra", encoding="utf-8")
    with pytest.raises(HPETranscriptCrosswalkError, match="extra or missing artifact file"):
        load_sealed_crosswalk_inventory(
            paths["inventory"], repository_root=paths["repo"]
        )
    extra.unlink()
    (paths["scene_dir"] / crosswalk.CONFLICTS_NAME).unlink()
    with pytest.raises(HPETranscriptCrosswalkError, match="extra or missing artifact file"):
        load_sealed_crosswalk_inventory(
            paths["inventory"], repository_root=paths["repo"]
        )

    other = _sealed_fixture(tmp_path / "other")
    extra_scene = other["repo"] / "data" / "annotations" / "hpe_transcript_crosswalk" / "extra"
    extra_scene.mkdir()
    with pytest.raises(HPETranscriptCrosswalkError, match="extra or missing scene"):
        load_sealed_crosswalk_inventory(
            other["inventory"], repository_root=other["repo"]
        )


def test_sealed_05_14_end_exclusive_and_05_15_3_person_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _sealed_fixture(tmp_path / "window")
    monkeypatch.setattr(
        crosswalk, "_replay_nonledger_sources", lambda *_args, **_kwargs: None
    )
    manifest_path = paths["scene_dir"] / crosswalk.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples_path = paths["scene_dir"] / crosswalk.SAMPLES_NAME
    samples = pd.read_csv(samples_path)
    samples.loc[0, ["logical_frame", "gaze_local_frame"]] = 10
    samples.to_csv(samples_path, index=False)
    manifest["review_samples"] = crosswalk._binding(samples_path, pd.read_csv(samples_path))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_sealed_inventory(paths)
    with pytest.raises(HPETranscriptCrosswalkError, match="half-open"):
        load_sealed_crosswalk(
            "05_14__3", paths["inventory"], repository_root=paths["repo"]
        )

    people = [1, 2, 3, 4, 5, 6]
    paths = _sealed_fixture(tmp_path / "universe", scene="05_15__3", people=people)
    manifest_path = paths["scene_dir"] / crosswalk.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expected_hpe_persons"] = [1, 2, 3, 4, 5]
    manifest["sample_decisions"] = {
        sample_id: decision
        for sample_id, decision in manifest["sample_decisions"].items()
        if not sample_id.startswith("person_006_")
    }
    manifest["upstream_proposal"].pop("6")
    samples_path = paths["scene_dir"] / crosswalk.SAMPLES_NAME
    samples = pd.read_csv(samples_path).loc[lambda frame: frame["hpe_person"] != 6]
    samples.to_csv(samples_path, index=False)
    manifest["review_samples"] = crosswalk._binding(samples_path, pd.read_csv(samples_path))
    mapping_path = paths["scene_dir"] / crosswalk.FINAL_MAPPING_NAME
    mapping = pd.read_csv(mapping_path).loc[lambda frame: frame["hpe_person"] != 6]
    mapping.to_csv(mapping_path, index=False)
    manifest["final_mapping"] = crosswalk._binding(
        mapping_path, pd.read_csv(mapping_path)
    )
    audit_path = paths["scene_dir"] / crosswalk.AUDIT_NAME
    events = [json.loads(line) for line in audit_path.read_text().splitlines()]
    events = [
        event
        for event in events
        if not str(event.get("sample_id", "")).startswith("person_006_")
    ]
    events[0]["expected_hpe_persons"] = [1, 2, 3, 4, 5]
    events[0]["sample_ids"] = samples["sample_id"].astype(str).tolist()
    events[-1]["labels"].pop("6")
    events[-1]["final_mapping_sha256"] = manifest["final_mapping"]["sha256"]
    audit_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    inventory = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    inventory["scenes"]["05_15__3"]["expected_hpe_persons"] = [1, 2, 3, 4, 5]
    paths["inventory"].write_text(json.dumps(inventory), encoding="utf-8")
    _refresh_sealed_inventory(paths)
    with pytest.raises(HPETranscriptCrosswalkError, match="expected-person universe changed"):
        load_sealed_crosswalk(
            "05_15__3", paths["inventory"], repository_root=paths["repo"]
        )


def test_inventory_builder_uses_strict_raw_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _sealed_fixture(tmp_path)

    def strict_sentinel(*_args: object, **_kwargs: object) -> None:
        raise HPETranscriptCrosswalkError("strict raw replay sentinel")

    monkeypatch.setattr(crosswalk, "load_crosswalk_artifact", strict_sentinel)
    with pytest.raises(HPETranscriptCrosswalkError, match="strict raw replay sentinel"):
        build_sealed_crosswalk_inventory(
            repository_root=paths["repo"],
            artifact_root="data/annotations/hpe_transcript_crosswalk",
            ledger_path="docs/IDENTITY_DATA_STATUS.md",
            sealed_ledger_path="configs/hpe_transcript_crosswalk_ledger_snapshot.md",
        )


def test_sealed_inventory_and_source_authority_are_not_shallow_hash_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _sealed_fixture(tmp_path)
    manifest_path = paths["scene_dir"] / crosswalk.MANIFEST_NAME
    manifest_path.write_text(manifest_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(HPETranscriptCrosswalkError, match="artifact bytes changed"):
        load_sealed_crosswalk_inventory(
            paths["inventory"], repository_root=paths["repo"]
        )

    paths = _sealed_fixture(tmp_path / "authority")

    def authority_sentinel(*_args: object, **_kwargs: object) -> None:
        raise HPETranscriptCrosswalkError("HPE/Gaze authority sentinel")

    monkeypatch.setattr(crosswalk, "_replay_nonledger_sources", authority_sentinel)
    with pytest.raises(HPETranscriptCrosswalkError, match="authority sentinel"):
        load_sealed_crosswalk(
            "05_14__3", paths["inventory"], repository_root=paths["repo"]
        )


def test_sealed_inventory_rejects_changed_immutable_ledger_snapshot(
    tmp_path: Path,
) -> None:
    paths = _sealed_fixture(tmp_path)
    paths["immutable_ledger"].write_text("tampered snapshot", encoding="utf-8")

    with pytest.raises(HPETranscriptCrosswalkError, match="ledger snapshot"):
        load_sealed_crosswalk_inventory(
            paths["inventory"], repository_root=paths["repo"]
        )


@pytest.mark.parametrize("overwrite", [False, True])
def test_sealed_bundle_second_publish_failure_leaves_no_partial_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    paths = _sealed_fixture(tmp_path)
    inventory = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    output = paths["repo"] / "configs" / "new_inventory.json"
    sealed = paths["repo"] / "configs" / "new_ledger_snapshot.md"
    inventory["ledger_snapshot"]["sealed_path"] = (
        "configs/new_ledger_snapshot.md"
    )
    old_output = b"old inventory bytes\n" if overwrite else None
    old_sealed = b"old ledger bytes\n" if overwrite else None
    if overwrite:
        output.write_bytes(old_output)
        sealed.write_bytes(old_sealed)

    real_replace = crosswalk.os.replace
    calls = 0

    def fail_second_publish(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected inventory publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(crosswalk.os, "replace", fail_second_publish)
    with pytest.raises(HPETranscriptCrosswalkError, match="previous state was restored"):
        crosswalk.write_sealed_crosswalk_bundle(
            inventory,
            output,
            repository_root=paths["repo"],
            overwrite=overwrite,
        )

    if overwrite:
        assert output.read_bytes() == old_output
        assert sealed.read_bytes() == old_sealed
    else:
        assert not output.exists()
        assert not sealed.exists()
    assert not list(output.parent.glob("*.tmp"))
