from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hri_turn_taking.data import facial_source_readiness as facial

SCENE = "03_20__4"


def _authority(tmp_path: Path) -> facial.FacialSourceAuthority:
    root = tmp_path / "repo"
    return facial.FacialSourceAuthority(
        repository_root=root,
        facial_root=root / "facial_features_export",
        analysis_base_root=root / "data" / "processed" / "analysis_base",
        crosswalk_inventory=root / "configs" / "inventory.json",
        crosswalk_root=root / "data" / "annotations" / "crosswalk",
        identity_root=root / "data" / "annotations" / "identity",
        enforce_frozen_inventory=False,
    )


def _documents() -> facial._SceneDocuments:
    analysis = pd.DataFrame(
        {
            "session_id": [SCENE] * 4,
            "timestamp_s": np.arange(4, dtype=float) / 25.0,
            "P1_present": [1, 1, 1, 1],
            "P2_present": [1, 1, 1, 1],
            "P3_present": [1, 1, 1, 1],
            "P4_present": [0, 0, 0, 0],
        }
    )
    confirmed = pd.DataFrame(
        {
            "frame_index": [0, 1, 2],
            "source_person_idx": [0, 0, 1],
            "participant": ["A", "A", "B"],
            "slot": ["P2", "P2", "P3"],
        }
    )
    return facial._SceneDocuments(
        analysis=analysis,
        analysis_binding={
            "path": "analysis.csv",
            "sha256": "0" * 64,
            "provenance_path": "analysis.provenance.json",
            "provenance_sha256": "1" * 64,
            "tick_start_inclusive": 0,
            "tick_end_exclusive": 4,
            "n_ticks": 4,
            "d3_offset_s": 0.0,
        },
        analysis_provenance={},
        sealed_identity={"replay_mode": "fixture"},
        confirmed_bridge=confirmed,
        source_fps=Fraction(25, 1),
        accepted_window=(0, 4),
    )


def _facial_row(frame_index: int, person_idx: int, *, au: float = 0.1) -> dict[str, float]:
    return {
        "frame_index": frame_index,
        "timestamp_sec": frame_index / 25.0,
        "person_idx": person_idx,
        "match_iou": 0.5,
        "face_conf": 0.9,
        "emotion_argmax": 0,
        "gaze_yaw": 0.01,
        "gaze_pitch": -0.02,
        **{f"au_{index}": au + index / 100.0 for index in range(8)},
    }


def _write_sources(authority: facial.FacialSourceAuthority) -> None:
    directory = authority.facial_root / "03_20" / "session_4"
    directory.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            _facial_row(0, 0),
            _facial_row(1, 0),
            _facial_row(1, 0, au=0.2),
            _facial_row(2, 1),
            _facial_row(3, 9),
        ],
        columns=facial.EXPECTED_OPENFACE_COLUMNS,
    )
    frame.to_csv(directory / "openface_features.csv", index=False)
    pd.DataFrame(
        {
            "frame_index": [0, 1],
            "person_idx": [0, 0],
            "mouth_motion": [9.0, 8.0],
        }
    ).to_csv(directory / "mouth_motion.csv", index=False)


def test_past_only_mapping_uses_exact_floor_and_half_open_frame() -> None:
    ticks = np.arange(0, 20, dtype=np.int64)
    fps = Fraction(30, 1)
    offset = Fraction(-17, 100)
    frames = facial.past_only_frame_indices(
        ticks,
        source_fps=fps,
        d3_offset_s=offset,
    )

    for tick, frame in zip(ticks.tolist(), frames.tolist(), strict=True):
        shared = Fraction(tick, 25)
        frame_start = Fraction(frame, 1) / fps + offset
        frame_end = Fraction(frame + 1, 1) / fps + offset
        assert frame_start <= shared < frame_end


def test_duplicate_group_is_invalid_and_absent_differs_from_missing(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    _write_sources(authority)

    report = facial._build_report(SCENE, _documents(), authority)

    assert report["assembly_eligible"] is False
    assert report["counted_input_ready"] is False
    assert report["writer_available"] is False
    assert report["identity_join_inventory"]["duplicate_key_group_count"] == 1
    assert report["identity_join_inventory"]["rows_in_duplicate_key_groups"] == 2
    participant = report["coverage"]["participant_ticks"]
    assert participant["A"]["valid_tick_count"] == 1
    assert participant["A"]["duplicate_key_groups_invalidated"] == 1
    assert participant["R"]["value_state"] == "all_null"
    assert participant["R"]["reason"] == "robot_identity_has_no_facial_source_row"
    assert participant["R"]["zero_filled"] is False
    assert participant["C"]["value_state"] == "absent"
    assert participant["C"]["absent_tick_count"] == 4
    assert participant["C"]["missing_present_tick_count"] == 0


def test_mouth_motion_is_hash_bound_but_never_consumed_as_mar(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    _write_sources(authority)

    report = facial._build_report(SCENE, _documents(), authority)

    mouth = report["disabled_mouth_motion"]
    assert mouth["exists"] is True
    assert mouth["sha256"]
    assert mouth["disabled"] is True
    assert mouth["consumed_for_values"] is False
    assert mouth["is_mar"] is False
    assert report["mar"]["readiness"] is False
    assert report["mar"]["reproducible_from_bound_export"] is False
    assert report["mar"]["bound_export_column_count"] == 16
    assert report["mar"]["complete_98_point_xy_export_present"] is False


def test_missing_entire_facial_root_returns_fail_closed_coverage(tmp_path: Path) -> None:
    authority = _authority(tmp_path)

    report = facial._build_report(SCENE, _documents(), authority)

    assert report["source"]["exists"] is False
    assert report["source"]["sha256"] is None
    assert report["source"]["rows_in_accepted_window"] == 0
    assert report["readiness_blockers"][0] == "facial_source_missing"
    assert all(
        item["valid_tick_count"] == 0 for item in report["coverage"]["participant_ticks"].values()
    )
    assert report["coverage"]["participant_ticks"]["C"]["value_state"] == "absent"
    json.dumps(report, allow_nan=False)


def test_extra_landmark_like_columns_do_not_silently_upgrade_bound_schema(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)
    _write_sources(authority)
    path = authority.facial_root / "03_20" / "session_4" / "openface_features.csv"
    frame = pd.read_csv(path)
    for index in range(98):
        frame[f"x_{index}"] = float(index)
        frame[f"y_{index}"] = float(index)
    frame.to_csv(path, index=False)

    report = facial._build_report(SCENE, _documents(), authority)

    assert report["source"]["ordered_schema_matches_bound_16_column_export"] is False
    assert report["mar"]["readiness"] is False
    assert report["mar"]["reproducible_from_bound_export"] is False
    assert report["mar"]["complete_98_point_xy_export_present"] is True
    assert (
        "openface_ordered_schema_is_not_the_bound_16_column_export" in report["readiness_blockers"]
    )


def test_analysis_dtype_map_requires_exact_keys_and_values() -> None:
    frame = _documents().analysis
    recorded = {column: str(dtype) for column, dtype in frame.dtypes.items()}

    facial._validate_analysis_dtypes(recorded, frame)

    invalid_maps = []
    missing = dict(recorded)
    missing.pop("timestamp_s")
    invalid_maps.append(missing)
    invalid_maps.append({**recorded, "unexpected": "float64"})
    invalid_maps.append({**recorded, "timestamp_s": "int64"})
    invalid_maps.append({**recorded, "timestamp_s": 1})
    for invalid in invalid_maps:
        with pytest.raises(
            facial.FacialSourceReadinessError,
            match="parsed dtypes differ from provenance",
        ):
            facial._validate_analysis_dtypes(invalid, frame)


def test_current_repository_scene_matches_documented_evidence_when_present() -> None:
    authority = facial.default_source_authority()
    source, _mouth = facial._source_paths(SCENE, authority)
    if not source.is_file():
        pytest.skip("repository-local Facial delivery is intentionally not packaged")

    report = facial.audit_facial_source_readiness(SCENE)

    assert report["source"]["rows_in_accepted_window"] == 2_579
    assert report["identity_join_inventory"]["confirmed_identity_joined_rows"] == 2_478
    assert report["identity_join_inventory"]["duplicate_key_group_count"] == 6
    assert report["coverage"]["participant_ticks"]["R"]["valid_tick_count"] == 0
    assert report["mar"]["readiness"] is False
    assert report["disabled_mouth_motion"]["consumed_for_values"] is False
