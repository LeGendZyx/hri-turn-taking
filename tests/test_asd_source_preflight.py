from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hri_turn_taking.data import analysis_base, assembly, audio_contract
from hri_turn_taking.data import asd_source_preflight as asd
from hri_turn_taking.data import hpe_transcript_crosswalk as crosswalk_module

SCENE = "03_20__4"


def _authority(tmp_path: Path) -> asd.ASDSourceAuthority:
    root = tmp_path / "repo"
    return asd.ASDSourceAuthority(
        repository_root=root,
        corpus_root=tmp_path / "corpus",
        analysis_base_root=root / "data" / "processed" / "analysis_base",
        crosswalk_inventory=root / "configs" / "inventory.json",
        crosswalk_root=root / "data" / "annotations" / "crosswalk",
        audio_contract_root=root / "data" / "annotations" / "audio_contract",
    )


def _documents(
    tmp_path: Path,
    *,
    scene: str = SCENE,
    offset_s: float = -6.7334,
) -> tuple[asd.ASDSourceAuthority, asd.ASDSourceDocuments]:
    authority = _authority(tmp_path)
    start, end = analysis_base.EXPECTED_TICK_RANGES[scene]
    participant_camera = "Cam_1" if scene.startswith("05_15__") else "Cam_2"
    fps = 60 if scene.startswith("05_14__") else 30
    accepted = {
        "scene": scene,
        "camera": participant_camera,
        "local_start_inclusive": 1800 if fps == 60 else 600,
        "local_end_exclusive": 29913 if scene == "05_14__3" else (
            30008 if scene == "05_14__4" else 6600
        ),
    }
    if scene == "05_14__3":
        offset_s = -5.04
    mapping = [
        {"hpe_person": 1, "participant": "R", "slot": "P1"},
        {"hpe_person": 3, "participant": "A", "slot": "P2"},
        {"hpe_person": 2, "participant": "B", "slot": "P3"},
        {"hpe_person": 4, "participant": "C", "slot": "P4"},
    ]
    crosswalk = {
        "artifact_kind": crosswalk_module.SEALED_DOWNSTREAM_KIND,
        "format_version": crosswalk_module.SEALED_DOWNSTREAM_VERSION,
        "scene": scene,
        "accepted_window": accepted,
        "source_fps": float(fps),
        "person_mapping": mapping,
    }
    path = authority.analysis_base_root / f"{scene}.csv"
    path.parent.mkdir(parents=True)
    path.write_text("fixture", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "session_id": [scene] * (end - start),
            "timestamp_s": np.arange(start, end, dtype=np.float64) / 25.0,
        }
    )
    dataset = assembly._DatasetFile(
        path=path,
        frame_df=frame,
        provenance={
            "artifact_kind": analysis_base.ARTIFACT_KIND,
            "session_id": scene,
            "trainable": False,
            "output_tick_range": {
                "start_inclusive": start,
                "end_exclusive": end,
            },
            "identity_contract": crosswalk,
            "offset_quality": {"offset_s": offset_s},
        },
        data_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    day, scenario = scene.rsplit("__", maxsplit=1)
    basename = f"{participant_camera}_Take fixture.avi"
    relative = f"Data/{day}/6cams/{scenario}/{participant_camera}/{basename}"
    participant_video = authority.corpus_root.joinpath(*Path(relative).parts[1:])
    participant_video.parent.mkdir(parents=True)
    participant_video.write_bytes(b"fixture participant AVI")
    robot_video = participant_video.parent.parent / "Cam_6" / basename.replace(
        f"{participant_camera}_", "Cam_6_", 1
    )
    robot_video.parent.mkdir(parents=True)
    robot_video.write_bytes(b"fixture robot AVI")
    first_segment_frames = accepted["local_end_exclusive"] + 100
    identity = {
        "artifact_kind": "vap_manual_identity_annotation",
        "format_version": 2,
        "source": {
            "declared_video": {
                "source_path": relative,
                "camera_dir": participant_camera,
                "frame_count": first_segment_frames,
                "fps": float(fps),
                "dataset": day,
                "session": scenario,
            }
        },
        "annotation_scope": {
            "review_status": "confirmed",
            "source_fps": float(fps),
            "gaze_local_start_frame_inclusive": accepted["local_start_inclusive"],
            "gaze_local_end_frame_exclusive": accepted["local_end_exclusive"],
        },
    }
    validation = {
        "pose_sources": {
            "cameras": {
                "Cam_6": {
                    "bound_camera": "Cam_6",
                    "source_fps": float(fps),
                    "processed_frames": first_segment_frames,
                }
            }
        }
    }
    audio_authority = audio_contract.AudioSourceAuthority(
        corpus_root=authority.corpus_root,
        hpe_delivery_root=authority.repository_root.parent
        / "Head_pose_extraction"
        / "outputs"
        / "delivery",
    )
    paths = audio_contract.canonical_source_paths(scene, audio_authority)
    sources = {
        speaker: {"path": str(paths[speaker])} for speaker in ("A", "B", "C", "R")
    }
    sources["master"] = {
        "path": str(paths["master"]),
        "sha256": "c" * 64,
        "size_bytes": 80_044,
        "wave_header": {
            "channels": 4,
            "sample_width_bytes": 2,
            "sample_rate_hz": 48_000,
            "compression_type": "NONE",
            "n_frames": 10_000,
        },
        "pcm_payload": {
            "complete": True,
            "decoded_frames": 10_000,
            "frame_aligned": True,
            "frame_size_bytes": 8,
        },
    }
    audio = {
        "artifact_kind": audio_contract.ARTIFACT_KIND,
        "format_version": audio_contract.FORMAT_VERSION,
        "scene": scene,
        "output_role": audio_contract.OUTPUT_ROLE,
        "counted_gold_ready": False,
        "master_sample_replay": {
            "channel_index_to_speaker": ["A", "B", "C", "R"],
            "exact_permutation": True,
            "identity_rule": "sample_replay_only_never_channel_index_or_report",
        },
        "sources": sources,
    }
    documents = asd.ASDSourceDocuments(
        analysis=dataset,
        crosswalk=crosswalk,
        crosswalk_manifest={},
        validation_manifest=validation,
        identity_manifest=identity,
        audio_report=audio,
        bindings={
            "fixture": {
                "path": str(path.resolve()),
                "size_bytes": len(path.read_bytes()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        },
    )
    return authority, documents


def _probe(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": True,
        "path_gate_pass": True,
        "path_kind": "regular_file",
        "declared_size_bytes": path.stat().st_size,
        "bytes_status": "bytes_unverified_no_hash_read",
        "ready_for_extraction": False,
        "file_opened_or_hydrated": False,
    }


def test_frozen_camera_matrix_covers_exactly_ten_scenes() -> None:
    assert set(analysis_base.EXPECTED_TICK_RANGES) == {
        "03_20__4", "03_26__3", "05_14__2", "05_14__3", "05_14__4",
        "05_15__2", "05_15__3", "05_15__4", "05_15__5", "05_15__6",
    }
    for scene in analysis_base.EXPECTED_TICK_RANGES:
        plan = asd.camera_plan(scene)
        assert plan["R"] == "Cam_6"
        expected = "Cam_1" if scene.startswith("05_15__") else "Cam_2"
        assert {plan[label] for label in "ABC"} == {expected}


def test_past_only_rational_mapping_uses_floor_not_nearest() -> None:
    frames = asd.past_only_frame_indices(
        [0, 1, 2], source_fps=30, d3_offset_s=0, accepted_start=0, accepted_end=10
    )
    assert frames.tolist() == [0, 1, 2]
    # tick 1 is 0.04 s: frame 1 at 1/30 is allowed; frame 2 is in the future.
    assert FractionLike(frames[1], 30) <= 1 / 25
    assert FractionLike(frames[1] + 1, 30) > 1 / 25


def FractionLike(numerator: int, denominator: int) -> float:
    return int(numerator) / denominator


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_fps": True}, "source_fps"),
        ({"d3_offset_s": float("nan")}, "finite"),
        ({"accepted_start": False}, "accepted_start"),
        ({"accepted_end": 10.0}, "accepted_end"),
    ],
)
def test_resampling_rejects_bool_float_and_nan(kwargs: dict[str, object], message: str) -> None:
    options: dict[str, object] = {
        "source_fps": 30,
        "d3_offset_s": 0,
        "accepted_start": 0,
        "accepted_end": 10,
    }
    options.update(kwargs)
    with pytest.raises(asd.ASDSourcePreflightError, match=message):
        asd.past_only_frame_indices([0, 1], **options)  # type: ignore[arg-type]


def test_frame_boundary_escape_fails_closed() -> None:
    with pytest.raises(asd.ASDSourcePreflightError, match="escapes"):
        asd.past_only_frame_indices(
            [0, 1], source_fps=30, d3_offset_s=0, accepted_start=1, accepted_end=10
        )


def test_report_is_nonpublishable_and_records_stock_noncausality(tmp_path: Path) -> None:
    authority, documents = _documents(tmp_path)
    report = asd._build_preflight_report(
        SCENE, documents, authority=authority, media_probe=_probe
    )
    assert report["assembly_eligible"] is False
    assert report["artifact_kind"] == "vap_real_asd_source_preflight_fixture"
    assert report["fixture_only"] is True
    assert report["counted_input_ready"] is False
    assert report["stock_asd_executed"] is False
    assert report["raw_video_opened_or_hydrated"] is False
    r_source = report["source_plan"]["participant_sources"]["R"]
    assert r_source["face_head_association"]["r_source_valid"] is False
    assert any("R Cam6" in blocker for blocker in report["readiness_blockers"])
    assert report["upstream_models"]["Light-ASD"]["commit"] == asd.LIGHT_ASD_COMMIT
    assert report["upstream_models"]["LR-ASD"]["commit"] == asd.LR_ASD_COMMIT
    assert report["upstream_models"]["Light-ASD"]["stock_noncausal"] is True
    tests = report["future_causal_acceptance_tests"]
    assert set(tests) == {
        "overall_pass", "tests", "any_failure_blocks_feature_publication",
    }
    assert tests["overall_pass"] is False
    assert all(result["pass"] is False for result in tests["tests"].values())
    assert all(
        result["status"] == "not_run_no_model_output"
        for result in tests["tests"].values()
    )
    assert report["source_plan"]["audio_input"]["single_lavalier_input_forbidden"] is True


def test_first_segment_only_is_explicit_and_later_segment_is_excluded(tmp_path: Path) -> None:
    authority, documents = _documents(tmp_path, scene="05_14__3")
    report = asd._build_preflight_report(
        "05_14__3", documents, authority=authority, media_probe=_probe
    )
    window = report["source_plan"]["accepted_source_window"]
    assert window["first_segment_only"] is True
    assert window["later_segments_excluded_not_zero_filled"] is True


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda d: d.identity_manifest["source"]["declared_video"].update(
            {"camera_dir": "Cam_1"}), "camera/FPS"),
        (lambda d: d.identity_manifest["source"]["declared_video"].update(
            {"frame_count": 6500}), "escapes"),
        (lambda d: d.crosswalk["person_mapping"][0].update({"slot": "P2"}),
         "participant-to-slot"),
        (lambda d: d.audio_report["sources"]["master"].update(
            {"path": d.audio_report["sources"]["A"]["path"]}), "audio master"),
    ],
)
def test_camera_segment_identity_and_audio_leakage_counterexamples(
    tmp_path: Path,
    mutator: object,
    message: str,
) -> None:
    authority, original = _documents(tmp_path)
    documents = copy.deepcopy(original)
    mutator(documents)  # type: ignore[operator]
    with pytest.raises(asd.ASDSourcePreflightError, match=message):
        asd._build_preflight_report(
            SCENE, documents, authority=authority, media_probe=_probe
        )


def test_analysis_path_copy_is_rejected_even_when_contents_match(tmp_path: Path) -> None:
    authority, original = _documents(tmp_path)
    copied_path = tmp_path / "copy" / f"{SCENE}.csv"
    copied_path.parent.mkdir()
    copied_path.write_bytes(original.analysis.path.read_bytes())
    copied = assembly._DatasetFile(
        path=copied_path,
        frame_df=original.analysis.frame_df.copy(),
        provenance=copy.deepcopy(original.analysis.provenance),
        data_sha256=original.analysis.data_sha256,
        provenance_sha256=original.analysis.provenance_sha256,
    )
    documents = asd.ASDSourceDocuments(
        analysis=copied,
        crosswalk=copy.deepcopy(original.crosswalk),
        crosswalk_manifest={},
        validation_manifest=copy.deepcopy(original.validation_manifest),
        identity_manifest=copy.deepcopy(original.identity_manifest),
        audio_report=copy.deepcopy(original.audio_report),
        bindings=copy.deepcopy(original.bindings),
    )
    with pytest.raises(asd.ASDSourcePreflightError, match="copied away"):
        asd._build_preflight_report(
            SCENE, documents, authority=authority, media_probe=_probe
        )


def test_cam6_without_face_bbox_schema_is_explicitly_blocked() -> None:
    frame = pd.DataFrame({"frame_index": [1], "person": [1], "face_matched": [1]})
    coverage = asd._association_coverage(
        np.array([1, 2], dtype=np.int64),
        relabeled=frame,
        hpe_person=1,
        binding_available=True,
    )
    assert coverage["status"] == "blocked_missing_face_bbox_or_association_schema"
    assert coverage["valid"] == 0
    assert coverage["missing"] == 2
    assert coverage["missing_is_zero_filled"] is False
    assert coverage["r_source_valid"] is False


def test_cam6_coverage_partitions_valid_missing_and_ambiguous_ticks() -> None:
    rows = pd.DataFrame(
        {
            "frame_index": [1, 2, 2],
            "person": [1, 1, 1],
            "face_matched": [1, 1, 1],
            "face_bbox_x1": [0.0, 0.0, 1.0],
            "face_bbox_y1": [0.0, 0.0, 1.0],
            "face_bbox_x2": [10.0, 10.0, 11.0],
            "face_bbox_y2": [10.0, 10.0, 11.0],
            "face_association_reason": ["matched_unique", "matched", "matched"],
        }
    )
    coverage = asd._association_coverage(
        np.array([1, 2, 3], dtype=np.int64),
        relabeled=rows,
        hpe_person=1,
        binding_available=True,
    )
    assert coverage["valid"] == 1
    assert coverage["ambiguous"] == 1
    assert coverage["missing"] == 1
    assert coverage["r_source_valid"] is False


def test_fixture_probe_cannot_claim_ready_or_opened_media(tmp_path: Path) -> None:
    authority, documents = _documents(tmp_path)

    def malicious(path: Path) -> dict[str, object]:
        report = _probe(path)
        report["ready_for_extraction"] = True
        report["file_opened_or_hydrated"] = True
        return report

    with pytest.raises(asd.ASDSourcePreflightError, match="cannot claim ready/opened"):
        asd._build_preflight_report(
            SCENE, documents, authority=authority, media_probe=malicious
        )


def test_binding_schema_rejects_extra_or_missing_fields(tmp_path: Path) -> None:
    authority, original = _documents(tmp_path)
    documents = copy.deepcopy(original)
    documents.bindings["fixture"]["forged"] = True
    with pytest.raises(asd.ASDSourcePreflightError, match="non-exact schema"):
        asd._build_preflight_report(
            SCENE, documents, authority=authority, media_probe=_probe
        )


def test_identity_source_path_cannot_be_copied_to_another_scenario(tmp_path: Path) -> None:
    authority, original = _documents(tmp_path)
    documents = copy.deepcopy(original)
    declared = documents.identity_manifest["source"]["declared_video"]
    declared["source_path"] = declared["source_path"].replace("/4/", "/5/")
    with pytest.raises(asd.ASDSourcePreflightError, match="exact scene/camera"):
        asd._build_preflight_report(
            SCENE, documents, authority=authority, media_probe=_probe
        )


def test_media_probe_rejects_directory_without_following_or_opening(tmp_path: Path) -> None:
    report = asd._media_probe(tmp_path)
    assert report["path_gate_pass"] is False
    assert report["ready_for_extraction"] is False
    assert report["file_opened_or_hydrated"] is False
    assert report["bytes_status"] == "rejected_without_following_or_opening"


def test_audio_payload_forgery_is_rejected_by_private_fixture_seam(tmp_path: Path) -> None:
    authority, original = _documents(tmp_path)
    documents = copy.deepcopy(original)
    documents.audio_report["sources"]["master"]["pcm_payload"][
        "decoded_frames"
    ] = 9_999
    with pytest.raises(asd.ASDSourcePreflightError, match="PCM payload"):
        asd._build_preflight_report(
            SCENE, documents, authority=authority, media_probe=_probe
        )


def test_simulated_parent_junction_is_rejected_before_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "authority"
    parent = root / "parent"
    parent.mkdir(parents=True)
    leaf = parent / "source.json"
    leaf.write_text("{}", encoding="utf-8")
    original = asd._lstat

    def fake_lstat(path: Path) -> object:
        info = original(path)
        if path == parent:
            return SimpleNamespace(
                st_mode=info.st_mode,
                st_dev=info.st_dev,
                st_ino=info.st_ino,
                st_size=info.st_size,
                st_mtime_ns=info.st_mtime_ns,
                st_file_attributes=0x400,
            )
        return info

    monkeypatch.setattr(asd, "_lstat", fake_lstat)
    with pytest.raises(asd.ASDSourcePreflightError, match="junction/reparse"):
        asd._secure_existing_path(leaf, root=root, label="simulated source")


@pytest.mark.parametrize("field", ["data_sha256", "provenance_sha256"])
def test_analysis_or_provenance_change_after_gate_is_rejected(
    tmp_path: Path, field: str
) -> None:
    _authority_value, documents = _documents(tmp_path)
    gated = documents.analysis
    analysis_binding = {"sha256": gated.data_sha256}
    provenance_binding = {"sha256": gated.provenance_sha256}
    if field == "data_sha256":
        analysis_binding["sha256"] = "0" * 64
    else:
        provenance_binding["sha256"] = "0" * 64
    with pytest.raises(asd.ASDSourcePreflightError, match="stable-read SHA"):
        asd._assert_gated_snapshot_hashes(
            gated, analysis_binding, provenance_binding
        )


def test_nonexistent_self_binding_is_rejected(tmp_path: Path) -> None:
    authority, original = _documents(tmp_path)
    documents = copy.deepcopy(original)
    documents.bindings["fixture"]["path"] = str(tmp_path / "missing.bin")
    with pytest.raises(asd.ASDSourcePreflightError, match="unavailable"):
        asd._build_preflight_report(
            SCENE, documents, authority=authority, media_probe=_probe
        )


def test_lying_safe_probe_byte_state_is_rejected(tmp_path: Path) -> None:
    authority, documents = _documents(tmp_path)

    def liar(path: Path) -> dict[str, object]:
        report = _probe(path)
        report["bytes_status"] = "cryptographically_verified"
        return report

    with pytest.raises(asd.ASDSourcePreflightError, match="forbidden byte state"):
        asd._build_preflight_report(
            SCENE, documents, authority=authority, media_probe=liar
        )


def test_production_report_rejects_injected_probe(tmp_path: Path) -> None:
    authority, documents = _documents(tmp_path)
    with pytest.raises(asd.ASDSourcePreflightError, match="cannot inject"):
        asd._build_preflight_report(
            SCENE,
            documents,
            authority=authority,
            media_probe=_probe,
            fixture_only=False,
        )
