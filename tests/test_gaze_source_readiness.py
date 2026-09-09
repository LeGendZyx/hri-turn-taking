from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hri_turn_taking.data import gaze_source_readiness as gaze


def _bridge(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults: dict[str, object] = {
        "observation_id": 0,
        "frame_index": 0,
        "source_kind": "participant_detection",
        "source_person_idx": "0",
        "gaze_target": "elsewhere",
        "participant": "A",
        "slot": "P2",
        "annotation_status": "confirmed",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _robot(frame: int, observation_id: int) -> dict[str, object]:
    return {
        "observation_id": observation_id,
        "frame_index": frame,
        "source_kind": "robot_candidate",
        "source_person_idx": "__robot__",
        "gaze_target": pd.NA,
        "participant": "R",
        "slot": "P1",
        "annotation_status": "confirmed",
    }


def _participant(
    participant: str,
    frame: int,
    observation_id: int,
    target: object,
    *,
    person_idx: str | None = None,
) -> dict[str, object]:
    return {
        "observation_id": observation_id,
        "frame_index": frame,
        "source_kind": "participant_detection",
        "source_person_idx": person_idx or participant,
        "gaze_target": target,
        "participant": participant,
        "slot": gaze.PARTICIPANT_TO_SLOT[participant],
        "annotation_status": "confirmed",
    }


def _evidence(
    bridge: pd.DataFrame,
    *,
    frames: tuple[int, ...] = (0, 1),
    presence: dict[str, bool] | None = None,
) -> pd.DataFrame:
    return gaze.derive_nullable_gaze_evidence(
        scene="fixture__1",
        ticks=np.arange(len(frames), dtype=np.int64),
        source_frames=np.asarray(frames, dtype=np.int64),
        source_fps=30,
        bridge=bridge,
        slot_presence=presence or {"P1": True, "P2": True, "P3": True, "P4": True},
    )


def test_past_only_mapping_uses_exact_floor_not_nearest() -> None:
    frames = gaze.past_only_frame_indices(
        [0, 1, 2],
        source_fps=30,
        d3_offset_s=0,
        accepted_start=0,
        accepted_end=10,
    )

    assert frames.tolist() == [0, 1, 2]
    # Shared tick 1 is 0.04 s. Source frame 1 at 1/30 s is past; frame 2 is future.
    assert frames[1] / 30 <= 1 / 25 < (frames[1] + 1) / 30


def test_past_only_mapping_uses_decimal_offset_as_rational() -> None:
    frames = gaze.past_only_frame_indices(
        [376, 377],
        source_fps=30,
        d3_offset_s=-6.7334,
        accepted_start=600,
        accepted_end=6600,
    )

    assert frames.tolist() == [653, 654]


@pytest.mark.parametrize(
    ("ticks", "kwargs"),
    [
        ([0, 0], {}),
        ([0.0, 1.0], {}),
        ([0, 1], {"source_fps": 25}),
        ([0, 1], {"accepted_start": 2}),
    ],
)
def test_past_only_mapping_rejects_invalid_or_out_of_window_inputs(
    ticks: list[object], kwargs: dict[str, int]
) -> None:
    options = {
        "source_fps": 30,
        "d3_offset_s": 0,
        "accepted_start": 0,
        "accepted_end": 10,
        **kwargs,
    }
    with pytest.raises(gaze.GazeSourceReadinessError):
        gaze.past_only_frame_indices(ticks, **options)  # type: ignore[arg-type]


def test_nullable_evidence_keeps_r_invalid_and_derives_non_r_binary() -> None:
    bridge = _bridge(
        [
            _robot(0, 0),
            _participant("A", 0, 1, "robot"),
            _participant("B", 0, 2, "participant"),
            _participant("C", 0, 3, "elsewhere"),
            _robot(1, 4),
            _participant("A", 1, 5, "elsewhere"),
        ]
    )

    evidence = _evidence(bridge)
    robot = evidence[evidence["participant"] == "R"]
    assert not robot["gaze_target_valid"].any()
    assert not robot["gaze_robot_valid"].any()
    assert robot["gaze_target"].isna().all()
    assert robot["gaze_robot"].isna().all()
    assert set(robot["gaze_robot_reason"]) == {"robot_slot_gaze_not_observed"}

    first_tick = evidence[evidence["tick"] == 0].set_index("participant")
    assert first_tick.loc["A", "gaze_robot"] == 1.0
    assert first_tick.loc["B", "gaze_robot"] == 0.0
    assert first_tick.loc["C", "gaze_robot"] == 0.0
    assert first_tick.loc[["A", "B", "C"], "gaze_robot_valid"].all()


def test_missing_robot_reference_preserves_category_but_invalidates_binary() -> None:
    evidence = _evidence(
        _bridge([_participant("A", 0, 0, "elsewhere")]),
        frames=(0,),
    )
    row = evidence.loc[evidence["participant"] == "A"].iloc[0]

    assert bool(row["gaze_target_valid"])
    assert row["gaze_target"] == "elsewhere"
    assert not bool(row["gaze_robot_valid"])
    assert pd.isna(row["gaze_robot"])
    assert row["gaze_robot_reason"] == "missing_confirmed_robot_bbox_reference"


def test_missing_or_uncertain_identity_is_invalid_not_zero() -> None:
    bridge = _bridge(
        [
            _robot(0, 0),
            {
                **_participant("A", 0, 1, "robot"),
                "participant": pd.NA,
                "slot": pd.NA,
                "annotation_status": "uncertain",
            },
        ]
    )
    evidence = _evidence(bridge, frames=(0,))
    row = evidence.loc[evidence["participant"] == "A"].iloc[0]

    assert not bool(row["gaze_target_valid"])
    assert not bool(row["gaze_robot_valid"])
    assert pd.isna(row["gaze_target"])
    assert pd.isna(row["gaze_robot"])
    assert row["gaze_target_reason"] == "missing_confirmed_participant_observation"


def test_absent_slot_is_distinct_from_observed_negative() -> None:
    bridge = _bridge([_robot(0, 0), _participant("C", 0, 1, "elsewhere")])
    presence = {"P1": True, "P2": True, "P3": True, "P4": False}
    evidence = _evidence(bridge, frames=(0,), presence=presence)
    row = evidence.loc[evidence["participant"] == "C"].iloc[0]

    assert not bool(row["slot_present"])
    assert not bool(row["gaze_target_valid"])
    assert not bool(row["gaze_robot_valid"])
    assert pd.isna(row["gaze_target"])
    assert pd.isna(row["gaze_robot"])
    assert row["gaze_target_reason"] == "absent_slot"


def test_raw_duplicate_frame_person_invalidates_whole_scene() -> None:
    bridge = _bridge(
        [
            _robot(0, 0),
            _participant("A", 0, 1, "robot", person_idx="7"),
            {
                **_participant("B", 0, 2, "elsewhere", person_idx="7"),
                "annotation_status": "uncertain",
                "participant": pd.NA,
                "slot": pd.NA,
            },
        ]
    )

    with pytest.raises(gaze.GazeSourceReadinessError, match="raw duplicate"):
        _evidence(bridge, frames=(0,))


def test_canonical_duplicate_invalidates_whole_scene() -> None:
    bridge = _bridge(
        [
            _robot(0, 0),
            _participant("A", 0, 1, "robot", person_idx="1"),
            _participant("A", 0, 2, "elsewhere", person_idx="2"),
        ]
    )

    with pytest.raises(gaze.GazeSourceReadinessError, match="canonical participant"):
        _evidence(bridge, frames=(0,))


def test_unsupported_target_invalidates_whole_scene() -> None:
    bridge = _bridge([_robot(0, 0), _participant("A", 0, 1, "specific_B")])

    with pytest.raises(gaze.GazeSourceReadinessError, match="unsupported target"):
        _evidence(bridge, frames=(0,))


def test_coverage_report_does_not_use_r_or_absent_slots_as_denominator() -> None:
    bridge = _bridge(
        [
            _robot(0, 0),
            _participant("A", 0, 1, "robot"),
            _participant("B", 0, 2, "elsewhere"),
        ]
    )
    evidence = _evidence(
        bridge,
        frames=(0,),
        presence={"P1": True, "P2": True, "P3": True, "P4": False},
    )

    coverage = gaze._coverage_report(evidence)
    assert coverage["eligible_nonrobot_present_ticks"] == 2
    assert coverage["gaze_robot_valid_ticks"] == 2
    assert coverage["gaze_robot_coverage"] == 1.0
    assert coverage["participants"]["R"]["gaze_robot_valid_ticks"] == 0
    assert coverage["participants"]["C"]["present_ticks"] == 0


def test_report_contract_cannot_be_upgraded_by_matrix_validation() -> None:
    results = {}
    for scene, (start, end) in gaze.analysis_base.EXPECTED_TICK_RANGES.items():
        evidence = pd.DataFrame(index=range((end - start) * 4))
        results[scene] = gaze.GazeSourcePreflight(
            scene=scene,
            evidence_df=evidence,
            report={
                "assembly_eligible": False,
                "counted_input_ready": False,
                "writer_available": False,
            },
        )

    gaze.validate_frozen_preflight_matrix(results)
    tampered = copy.copy(results)
    first = next(iter(tampered))
    tampered[first] = gaze.GazeSourcePreflight(
        scene=first,
        evidence_df=tampered[first].evidence_df,
        report={
            "assembly_eligible": True,
            "counted_input_ready": False,
            "writer_available": False,
        },
    )
    with pytest.raises(gaze.GazeSourceReadinessError, match="cannot claim"):
        gaze.validate_frozen_preflight_matrix(tampered)


def test_tool_exposes_no_output_or_publish_option() -> None:
    tool = Path("tools/audit_gaze_source_readiness.py").read_text(encoding="utf-8")

    assert 'add_argument("--output"' not in tool
    assert 'add_argument("--publish"' not in tool
    assert '"writer_available": False' in tool


def test_current_repository_scene_replays_with_canonical_crosswalk_names() -> None:
    scene = "03_20__4"
    repository_inputs = (
        Path("configs/hpe_transcript_crosswalk_inventory.json"),
        Path("data/processed/analysis_base") / f"{scene}.csv",
        Path("data/annotations/hpe_transcript_crosswalk")
        / scene
        / "hpe_transcript_crosswalk.csv",
    )
    if not all(path.is_file() for path in repository_inputs):
        pytest.skip("canonical repository Gaze evidence is not present")

    result = gaze.audit_gaze_source_preflight(scene)

    assert result.report["scene"] == scene
    assert result.report["assembly_eligible"] is False
    assert result.report["counted_input_ready"] is False
    assert result.report["writer_available"] is False
