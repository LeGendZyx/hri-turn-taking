from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from hri_turn_taking.vap import reactive_diagnostic as diagnostic
from hri_turn_taking.vap.events import robot_entry_scores
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG


def test_replay_uses_38_ticks_and_one_rising_trigger_per_silent_run() -> None:
    voiced = np.asarray([1, *([0] * 45), 1, *([0] * 38)], dtype=np.int8)

    silent_run, ready, trigger, state = diagnostic.replay_voiced(voiced)

    assert silent_run[37] == 37
    assert ready[37] == 0
    assert trigger[37] == 0
    assert silent_run[38] == 38
    assert ready[38] == 1
    assert trigger[38] == 1
    assert ready[39:46].tolist() == [1] * 7
    assert trigger[39:46].tolist() == [0] * 7
    assert trigger[-1] == 1
    assert int(trigger.sum()) == 2
    assert state.silent_run_frames == 38


def test_replay_full_equals_chunked_with_carried_state() -> None:
    voiced = np.asarray([1, *([0] * 50), 1, *([0] * 41)], dtype=np.int8)
    full_run, full_ready, full_trigger, full_state = diagnostic.replay_voiced(voiced)

    run_a, ready_a, trigger_a, state_a = diagnostic.replay_voiced(voiced[:23])
    run_b, ready_b, trigger_b, state_b = diagnostic.replay_voiced(
        voiced[23:], state=state_a
    )

    assert np.array_equal(np.concatenate([run_a, run_b]), full_run)
    assert np.array_equal(np.concatenate([ready_a, ready_b]), full_ready)
    assert np.array_equal(np.concatenate([trigger_a, trigger_b]), full_trigger)
    assert state_b == full_state


def test_replay_is_future_causal_and_new_session_resets_state() -> None:
    original = np.zeros(70, dtype=np.int8)
    changed_future = original.copy()
    changed_future[55:] = 1

    original_outputs = diagnostic.replay_voiced(original)
    changed_outputs = diagnostic.replay_voiced(changed_future)
    for original_value, changed_value in zip(
        original_outputs[:3], changed_outputs[:3], strict=True
    ):
        assert np.array_equal(original_value[:55], changed_value[:55])

    _, _, continuous_trigger, _ = diagnostic.replay_voiced(np.zeros(40, dtype=np.int8))
    _, _, first_session, _ = diagnostic.replay_voiced(np.zeros(20, dtype=np.int8))
    _, _, second_session, _ = diagnostic.replay_voiced(np.zeros(20, dtype=np.int8))
    assert int(continuous_trigger.sum()) == 1
    assert int(first_session.sum()) == 0
    assert int(second_session.sum()) == 0


def test_trigger_api_has_no_gold_or_prediction_input() -> None:
    trigger_signature = inspect.signature(diagnostic.generate_trigger_artifact)
    replay_signature = inspect.signature(diagnostic.replay_voiced)

    assert set(trigger_signature.parameters) == {"protocol_directory", "output_directory"}
    assert "vad" not in replay_signature.parameters
    assert "labels" not in replay_signature.parameters
    assert "predictions" not in replay_signature.parameters

    voiced = np.asarray([1, *([0] * 38)], dtype=np.int8)
    trigger_before = diagnostic.replay_voiced(voiced)[2].tobytes()
    gold_vad = np.zeros((len(voiced), 2), dtype=np.float32)
    gold_vad[:, 0] = 1
    gold_vad[:] = 1 - gold_vad
    trigger_after = diagnostic.replay_voiced(voiced)[2].tobytes()
    assert trigger_after == trigger_before


def test_robot_entry_opportunity_helper_matches_existing_scorer() -> None:
    vad = np.zeros((25, 2), dtype=np.float32)
    vad[:10, 1] = 1
    vad[15:, 0] = 1
    present = np.ones(2, dtype=np.float32)
    valid = np.ones(25, dtype=bool)

    opportunities = diagnostic.extract_robot_entry_opportunities(
        vad, present=present, valid=valid
    )
    probabilities = np.zeros((25, 2, 4), dtype=np.float32)
    labels, scores = robot_entry_scores(
        probabilities,
        vad,
        robot_slot=0,
        present=present,
        valid=valid,
        label_config=DEFAULT_LABEL_CONFIG,
    )

    assert len(opportunities) == 1
    assert opportunities[0].gap_start_f == 10
    assert opportunities[0].gap_end_f == 15
    assert opportunities[0].previous_speaker == 1
    assert opportunities[0].next_speaker == 0
    assert opportunities[0].label == 1
    assert labels == [opportunity.label for opportunity in opportunities]
    assert len(scores) == len(opportunities)


def test_robot_entry_opportunity_requires_full_gap_validity_and_present_robot() -> None:
    vad = np.zeros((25, 2), dtype=np.float32)
    vad[:10, 1] = 1
    vad[15:, 0] = 1
    valid = np.ones(25, dtype=bool)
    valid[12] = False

    assert diagnostic.extract_robot_entry_opportunities(
        vad, present=np.ones(2), valid=valid
    ) == []
    assert diagnostic.extract_robot_entry_opportunities(
        vad, present=np.asarray([0, 1]), valid=np.ones(25)
    ) == []


def test_file_binding_detects_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"voiced\n0\n")
    binding = {
        "path": str(source.resolve()),
        "sha256": diagnostic._sha256_file(source),
        "size_bytes": source.stat().st_size,
    }

    assert diagnostic.verify_file_binding(binding, role="test source") == source.resolve()
    source.write_bytes(b"voiced\n1\n")
    with pytest.raises(diagnostic.ReactiveDiagnosticError, match="frozen byte binding"):
        diagnostic.verify_file_binding(binding, role="test source")


def test_exclusive_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    diagnostic._write_bytes_exclusive(output, b"first")

    with pytest.raises(FileExistsError):
        diagnostic._write_bytes_exclusive(output, b"second")
    assert output.read_bytes() == b"first"


@pytest.mark.parametrize(
    "voiced",
    [
        np.asarray([0.0, np.nan]),
        np.asarray([0.0, 2.0]),
        np.asarray([[0.0, 1.0]]),
    ],
)
def test_replay_rejects_nonfinite_nonbinary_or_nonvector_voiced(voiced: np.ndarray) -> None:
    with pytest.raises(diagnostic.ReactiveDiagnosticError):
        diagnostic.replay_voiced(voiced)
