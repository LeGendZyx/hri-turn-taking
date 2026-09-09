from __future__ import annotations

import copy
import json
import os
import wave
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hri_turn_taking.data import gold_vad_candidate as candidate
from hri_turn_taking.data import transcript_fs_guard
from hri_turn_taking.data.transcript_acceptance import (
    AcceptedTranscriptReference,
    TranscriptEvent,
)


def _accepted(events: tuple[TranscriptEvent, ...], *, end_tick: int = 125):
    intervals = {
        speaker: tuple(
            (event.start_frame, event.end_frame)
            for event in events
            if event.speaker == speaker
        )
        for speaker in candidate.SPEAKERS
    }
    return AcceptedTranscriptReference(
        scene="fixture__1",
        manifest_path=Path("acceptance.json"),
        output_role="coarse_transcript_activity_reference",
        counted_gold_ready=False,
        transcript_offset_sec=Decimal("0"),
        timecode_quantization_error_bound_ms=Decimal("1"),
        offset_lag_quantization_error_bound_ms=Decimal("1"),
        transcript_side_error_bound_ms=Decimal("2"),
        hpe_stream="6cam",
        hpe_offset_sec=Decimal("0"),
        hpe_error_bound_ms=Decimal("20"),
        combined_error_bound_ms=Decimal("22"),
        hpe_quality_status="ok",
        events=events,
        speaker_intervals=intervals,
        source_bindings={},
        source_normalizations=(),
        authorized_start_tick=0,
        authorized_end_tick=end_tick,
        scope_summary={},
        speaker_prevalence={},
        excluded_event_inventory=(),
    )


def _channels(duration_s: float = 5.0) -> dict[str, np.ndarray]:
    n = int(duration_s * 48_000)
    return {speaker: np.zeros(n, dtype=np.float64) for speaker in candidate.SPEAKERS}


def _add_speech(
    channels: dict[str, np.ndarray], speaker: str, start_s: float, end_s: float
) -> None:
    start = int(start_s * 48_000)
    end = int(end_s * 48_000)
    t = np.arange(end - start, dtype=np.float64) / 48_000
    carrier = 0.45 * np.sin(2 * np.pi * 180 * t)
    for item in candidate.SPEAKERS:
        scale = 1.0 if item == speaker else 0.02
        channels[item][start:end] += carrier * scale


def _computed_bundle(tmp_path: Path) -> tuple[dict, Path]:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source-bytes")
    binding = candidate._binding_from_path(source)
    events = tuple(
        TranscriptEvent(
            source_index=index,
            start_frame=15 + (index % 5),
            end_frame=45 + (index % 5),
            speaker=candidate.SPEAKERS[index % 4],
            text=f"event {index}",
        )
        for index in range(60)
    )
    accepted = _accepted(events)
    snapshot = SimpleNamespace(
        scene="fixture__1",
        start_tick=0,
        end_tick=125,
        corrections_s={speaker: Decimal("0") for speaker in candidate.SPEAKERS},
        source_bindings=(binding,),
        analysis_csv=tmp_path / "fixture.csv",
        bridge_path=tmp_path / "bridge",
        gold_path=tmp_path / "gold.json",
        bridge=SimpleNamespace(acceptance=accepted),
    )
    channels = _channels()
    _add_speech(channels, "A", 0.45, 1.8)
    _add_speech(channels, "B", 0.7, 1.6)
    _add_speech(channels, "C", 0.8, 1.7)
    _add_speech(channels, "R", 0.6, 1.9)
    return candidate._compute_payload(snapshot, channels), source


def test_rms_is_right_aligned_and_future_perturbation_does_not_change_past() -> None:
    samples = np.zeros(48_000 * 3, dtype=np.float64)
    samples[48_000:96_000] = 0.25
    ends, before = candidate._rms_db(samples, window_ms=20)
    changed = samples.copy()
    changed[96_000:] = 0.8
    changed_ends, after = candidate._rms_db(changed, window_ms=20)
    assert np.array_equal(ends, changed_ends)
    assert np.array_equal(before[ends <= 96_000], after[ends <= 96_000])
    # The sample ending exactly at t is included, while samples after t are not.
    edge = np.zeros(960 * 2, dtype=np.float64)
    edge[959] = 1.0
    _ends, edge_db = candidate._rms_db(edge, window_ms=20)
    assert edge_db[0] > 20 * np.log10(candidate.DB_EPSILON)


def test_channel_relative_detector_rejects_uniform_bleed_and_detects_owner() -> None:
    channels = _channels(3.0)
    _add_speech(channels, "A", 1.0, 2.0)
    view = candidate._detector_view(channels, start_s=0.0, window_ms=20)
    region = (view.right_edges_s >= 1.1) & (view.right_edges_s <= 1.9)
    assert view.speech["A"][region].mean() > 0.95
    assert not view.speech["B"][region].any()
    assert not view.speech["C"][region].any()
    assert not view.speech["R"][region].any()


def test_hysteresis_thresholds_gap_fill_and_minimum_duration() -> None:
    enter = np.zeros(30, dtype=bool)
    hold = np.zeros(30, dtype=bool)
    enter[2] = True
    enter[9] = True
    hold[2:12] = True
    hold[6:9] = False  # 30 ms, filled.
    mask = candidate._hysteresis(enter, hold)
    assert mask[2:12].all()
    enter[:] = False
    hold[:] = False
    enter[2] = True
    hold[2:6] = True  # 40 ms, removed by 60-ms minimum.
    assert not candidate._hysteresis(enter, hold).any()


def test_boundary_snap_overlap_and_edge_flags() -> None:
    channels = _channels(4.0)
    _add_speech(channels, "A", 1.0, 2.0)
    primary = candidate._detector_view(channels, start_s=0.0, window_ms=20)
    diagnostic = candidate._detector_view(channels, start_s=0.0, window_ms=40)
    accepted = _accepted(
        (
            TranscriptEvent(0, 33, 57, "A", "a"),
            TranscriptEvent(1, 45, 63, "B", "b"),
        ),
        end_tick=100,
    )
    item = candidate._candidate_event(
        accepted.events[0],
        accepted=accepted,
        primary=primary,
        diagnostic=diagnostic,
        formal_start=0.0,
        formal_end=4.0,
        transcript_overlap=True,
    )
    assert item["suggestion"]["start_s"] == pytest.approx(1.0, abs=0.011)
    assert item["suggestion"]["end_s_exclusive"] == pytest.approx(2.0, abs=0.011)
    assert "overlap" in item["flags"]
    assert item["automatic_promotion"] is False


def test_strict_missed_probe_is_flag_only_never_promoted() -> None:
    channels = _channels(4.0)
    _add_speech(channels, "A", 2.0, 2.25)
    primary = candidate._detector_view(channels, start_s=0.0, window_ms=20)
    accepted = _accepted((), end_tick=100)
    missed = candidate._missed_candidates(
        primary, accepted=accepted, formal_start=0.0, formal_end=4.0
    )
    assert missed
    assert all(item["candidate_type"] == "missed_token_strict_probe" for item in missed)
    assert all(item["automatic_promotion"] is False for item in missed)
    assert all(item["flags"] == ["candidate_only"] for item in missed)


def test_missed_probe_subtracts_same_speaker_half_open_mask_before_minimum() -> None:
    edges = np.arange(1, 31, dtype=np.float64) / 100.0
    zeros = {speaker: np.zeros(len(edges), dtype=np.float64) for speaker in candidate.SPEAKERS}
    false = {speaker: np.zeros(len(edges), dtype=bool) for speaker in candidate.SPEAKERS}
    strong = {speaker: values.copy() for speaker, values in false.items()}
    strong["A"][:] = True
    own_minus = {speaker: values.copy() for speaker, values in zeros.items()}
    own_minus["A"][:] = 9.0
    relative = {speaker: values.copy() for speaker, values in zeros.items()}
    relative["A"][:] = 9.0
    calibrated = {speaker: values.copy() for speaker, values in zeros.items()}
    calibrated["A"][:] = 12.0
    view = candidate._DetectorView(
        window_ms=20,
        right_edges_s=edges,
        calibrated_db=calibrated,
        relative_db=relative,
        own_minus_max_other_db=own_minus,
        speech=false,
        strong=strong,
        floors_db={speaker: 0.0 for speaker in candidate.SPEAKERS},
    )
    accepted = _accepted((TranscriptEvent(0, 3, 6, "A", "coarse"),), end_tick=10)
    missed = candidate._missed_candidates(
        view, accepted=accepted, formal_start=0.0, formal_end=0.4
    )
    a_items = [item for item in missed if item["speaker"] == "A"]
    assert len(a_items) == 2
    assert all(item["automatic_promotion"] is False for item in a_items)
    assert all(item["suggestion"]["duration_ms"] >= 80 for item in a_items)
    assert a_items[0]["evidence"]["transcript_subtraction"]["cut_right"] is True
    assert a_items[1]["evidence"]["transcript_subtraction"]["cut_left"] is True
    for item in a_items:
        interval = (
            item["suggestion"]["start_s"],
            item["suggestion"]["end_s_exclusive"],
        )
        assert candidate._overlap(interval, (0.1, 0.2)) == 0


def test_pcm_decode_uses_channel_local_shared_minus_correction_without_padding(
    tmp_path: Path,
) -> None:
    bindings = []
    sources = {}
    originals = {}
    for speaker_index, speaker in enumerate(candidate.SPEAKERS):
        path = tmp_path / f"{speaker}.wav"
        values = (
            (np.arange(48_000, dtype=np.int32) + speaker_index * 17) % 20_000 - 10_000
        ).astype("<i2")
        originals[speaker] = values
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(48_000)
            stream.writeframes(values.tobytes())
        binding = candidate._binding_from_path(path)
        bindings.append(binding)
        sources[speaker] = {
            **binding,
            "wave_header": {
                "channels": 1,
                "sample_width_bytes": 2,
                "sample_rate_hz": 48_000,
                "n_frames": 48_000,
                "compression_type": "NONE",
            },
        }
    corrections = {
        "A": Decimal("0"),
        "B": Decimal("0.01"),
        "C": Decimal("-0.01"),
        "R": Decimal("0"),
    }
    snapshot = SimpleNamespace(
        start_tick=5,
        end_tick=20,
        corrections_s=corrections,
        audio_report={"sources": sources},
    )
    with ExitStack() as stack:
        handles = {
            Path(binding["path"]): stack.enter_context(
                transcript_fs_guard.guarded_read(binding["path"])
            )
            for binding in bindings
        }
        decoded = candidate._decode_formal_channels(snapshot, handles)
    assert all(len(values) == 15 * 1920 for values in decoded.values())
    for speaker in candidate.SPEAKERS:
        expected_index = candidate._ceil_decimal_samples(
            Decimal("0.2") - corrections[speaker], 48_000
        )
        assert decoded[speaker][0] == originals[speaker][expected_index] / 32768.0

    snapshot.corrections_s["A"] = Decimal("-1")
    with ExitStack() as stack:
        handles = {
            Path(binding["path"]): stack.enter_context(
                transcript_fs_guard.guarded_read(binding["path"])
            )
            for binding in bindings
        }
        with pytest.raises(candidate.GoldVadCandidateError, match="coverage"):
            candidate._decode_formal_channels(snapshot, handles)


def test_channel_mapping_records_exact_samples_and_fractional_ceil_residuals() -> None:
    snapshot = SimpleNamespace(
        start_tick=5,
        end_tick=20,
        corrections_s={
            "A": Decimal("0.00001"),
            "B": Decimal("-0.00001"),
            "C": Decimal("0"),
            "R": Decimal("0.01"),
        },
    )
    mapping = candidate._channel_sample_mapping(snapshot)
    assert mapping["A"]["start_sample_inclusive"] == 9600
    assert mapping["A"]["end_sample_exclusive"] == 38400
    assert mapping["A"]["start_ceil_residual_samples"] == "0.48"
    assert mapping["A"]["end_ceil_residual_samples"] == "0.48"
    assert mapping["B"]["start_sample_inclusive"] == 9601
    assert mapping["B"]["start_ceil_residual_samples"] == "0.52"
    assert mapping["C"]["start_ceil_residual_samples"] == "0"


def test_joint_endpoint_and_cross_event_conflicts_retain_coarse_boundaries() -> None:
    decision = candidate._joint_boundary_decision(
        1.0,
        1.5,
        (1.4, 1.2, 0, 1),
        formal_start=0.0,
        formal_end=3.0,
    )
    assert decision == (
        1.0,
        1.5,
        "transcript_boundaries_retained_joint_order_conflict",
        True,
    )

    first = _manual_candidate(1, speaker="A", flags=[])
    second = _manual_candidate(2, speaker="A", flags=[])
    first["transcript_interval"] = {"start_s": 1.0, "end_s_exclusive": 1.5}
    second["transcript_interval"] = {"start_s": 2.0, "end_s_exclusive": 2.5}
    first["suggestion"].update(
        {"method": "snap", "start_s": 1.0, "end_s_exclusive": 2.1}
    )
    second["suggestion"].update(
        {"method": "snap", "start_s": 1.9, "end_s_exclusive": 2.5}
    )
    candidate._annotate_cross_event_conflicts(
        [first, second], formal_start=0.0, formal_end=3.0
    )
    assert first["suggestion"]["start_s"] == 1.0
    assert first["suggestion"]["end_s_exclusive"] == 1.5
    assert second["suggestion"]["start_s"] == 2.0
    assert second["suggestion"]["end_s_exclusive"] == 2.5
    assert {"ambiguity", "order_conflict", "refined_same_speaker_overlap"} <= set(
        first["flags"]
    )

    reused_first = copy.deepcopy(first)
    reused_second = copy.deepcopy(second)
    for item in (reused_first, reused_second):
        item["flags"] = []
        item["evidence"] = {}
        item["suggestion"].update({"start_s": 1.7, "end_s_exclusive": 1.9})
    candidate._annotate_cross_event_conflicts(
        [reused_first, reused_second], formal_start=0.0, formal_end=3.0
    )
    assert "reuse_conflict" in reused_first["flags"]
    assert "reuse_conflict" in reused_second["flags"]


def test_payload_has_permanent_candidate_status_and_20_40_diagnostics(tmp_path: Path) -> None:
    bundle, _source = _computed_bundle(tmp_path)
    candidate._validate_bundle_shape(bundle)
    assert bundle["candidate_only"] is True
    assert bundle["assembly_eligible"] is False
    assert bundle["counted_gold_ready"] is False
    assert bundle["source_selection"] == "owner_approval_pending"
    assert bundle["scientific_boundary"]["produces_p1_to_p4_vad"] is False
    assert bundle["diagnostics"]["per_speaker"]["A"]["diagnostic_20_40_agreement"] >= 0
    assert bundle["channel_time_mapping"]["per_channel_sample_mapping"]["A"][
        "start_sample_inclusive"
    ] == 0


def test_deep_type_failure_is_normalized_to_candidate_error() -> None:
    malformed = SimpleNamespace(start_tick="bad", end_tick=1)
    with pytest.raises(candidate.GoldVadCandidateError, match="candidate evidence computation"):
        candidate._compute_payload(malformed, {})


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("counted_gold_ready",), True),
        (("assembly_eligible",), True),
        (("candidate_only",), False),
        (("source_selection",), "approved"),
        (("parameters", "enter", "relative_db_gte"), "2.9"),
    ],
)
def test_status_and_parameter_tampering_fail_closed(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    bundle, _source = _computed_bundle(tmp_path)
    altered = copy.deepcopy(bundle)
    target = altered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(candidate.GoldVadCandidateError):
        candidate._validate_bundle_shape(altered)


def test_source_change_and_reparse_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    binding = candidate._binding_from_path(source)
    bundle = {"sources": {"bindings": [binding]}}
    candidate._verify_bound_sources(bundle)
    source.write_bytes(b"changed")
    with pytest.raises(candidate.GoldVadCandidateError, match="source changed"):
        candidate._verify_bound_sources(bundle)
    if os.name != "nt":
        real = tmp_path / "real.bin"
        real.write_bytes(b"real")
        link = tmp_path / "link.bin"
        link.symlink_to(real)
        with pytest.raises((candidate.GoldVadCandidateError, ValueError)):
            candidate._binding_from_path(link)


def _manual_candidate(index: int, *, speaker: str, flags: list[str]) -> dict:
    return {
        "candidate_id": f"c{index:03d}",
        "candidate_type": (
            "missed_token_strict_probe"
            if "candidate_only" in flags
            else "transcript_anchored_boundary"
        ),
        "source_index": None if "candidate_only" in flags else index,
        "speaker": speaker,
        "text": None if "candidate_only" in flags else "text",
        "transcript_interval": (
            None if "candidate_only" in flags else {"start_s": 1.0, "end_s_exclusive": 1.5}
        ),
        "suggestion": {"start_s": 1.0, "end_s_exclusive": 1.5},
        "diagnostic_40ms": None,
        "evidence": {"margin": 1.0},
        "flags": flags,
        "automatic_promotion": False,
    }


def test_blind_missed_clips_use_neutral_fixed_scene_grid_not_boundary_padding() -> None:
    first = _manual_candidate(100, speaker="A", flags=["candidate_only"])
    second = _manual_candidate(101, speaker="A", flags=["candidate_only"])
    first["suggestion"] = {"start_s": 0.15, "end_s_exclusive": 0.35}
    second["suggestion"] = {"start_s": 0.6, "end_s_exclusive": 0.8}

    first_row = candidate._blind_row(
        first,
        audit_id="audit_000001",
        seed="a" * 64,
        formal_start=0.0,
        formal_end=12.0,
    )
    second_row = candidate._blind_row(
        second,
        audit_id="audit_000002",
        seed="a" * 64,
        formal_start=0.0,
        formal_end=12.0,
    )
    assert first_row["clip_interval"] == second_row["clip_interval"] == {
        "start_s": 0.0,
        "end_s_exclusive": 4.0,
    }
    assert first_row["coarse_event"] is None
    assert second_row["coarse_event"] is None
    # Two distinct hidden intervals map to one clip, so no fixed +/- padding
    # inversion can recover either suggested boundary from the queue.
    assert first_row["clip_interval"]["start_s"] != first["suggestion"]["start_s"] - 1
    assert second_row["clip_interval"]["end_s_exclusive"] != (
        second["suggestion"]["end_s_exclusive"] + 1
    )


def test_blind_missed_clip_formal_edges_remain_half_open_and_candidate_independent() -> None:
    near_start = _manual_candidate(102, speaker="B", flags=["candidate_only"])
    near_end = _manual_candidate(103, speaker="B", flags=["candidate_only"])
    near_start["suggestion"] = {"start_s": 0.3, "end_s_exclusive": 0.5}
    near_end["suggestion"] = {"start_s": 10.1, "end_s_exclusive": 10.3}

    start_clip = candidate._neutral_missed_clip(
        near_start, seed="b" * 64, formal_start=0.3, formal_end=10.3
    )
    end_clip = candidate._neutral_missed_clip(
        near_end, seed="b" * 64, formal_start=0.3, formal_end=10.3
    )
    assert start_clip == {"start_s": 0.3, "end_s_exclusive": 4.3}
    assert end_clip == {"start_s": 6.3, "end_s_exclusive": 10.3}
    assert start_clip["end_s_exclusive"] - start_clip["start_s"] == pytest.approx(4.0)
    assert end_clip["end_s_exclusive"] - end_clip["start_s"] == pytest.approx(4.0)
    assert 0.3 <= near_start["suggestion"]["start_s"] < near_start["suggestion"][
        "end_s_exclusive"
    ] <= 10.3
    assert 0.3 <= near_end["suggestion"]["start_s"] < near_end["suggestion"][
        "end_s_exclusive"
    ] <= 10.3


def test_stratified_plan_is_deterministic_covers_all_flags_and_participants() -> None:
    items = []
    flag_cycle = [[], ["low_margin"], ["overlap"], ["short"], ["transcript_active_low"]]
    for index in range(80):
        items.append(
            _manual_candidate(
                index,
                speaker=candidate.SPEAKERS[index % 4],
                flags=flag_cycle[index % len(flag_cycle)],
            )
        )
    items.extend(
        _manual_candidate(100 + index, speaker="A", flags=["candidate_only"])
        for index in range(6)
    )
    first_queue, first_inventory = candidate._build_audit_plan(
        items, seed="a" * 64, formal_start=0.0, formal_end=10.0
    )
    second_queue, second_inventory = candidate._build_audit_plan(
        items, seed="a" * 64, formal_start=0.0, formal_end=10.0
    )
    assert (first_queue, first_inventory) == (second_queue, second_inventory)
    selected = {item["candidate_id"] for item in first_inventory["entries"]}
    flagged = {item["candidate_id"] for item in items if item["flags"]}
    assert len(selected) >= 50
    assert flagged <= selected
    expected_fields = {
        "audit_id",
        "speaker_channel",
        "coarse_event",
        "clip_interval",
        "suggestion_hidden_until_manual_decision",
        "hidden_suggestion_sha256",
    }
    assert all(set(item) == expected_fields for item in first_queue["items"])
    assert [item["audit_id"] for item in first_queue["items"]] == [
        f"audit_{index:06d}" for index in range(1, len(first_queue["items"]) + 1)
    ]
    forbidden = {
        "candidate_type",
        "audit_reasons",
        "stratum",
        "low_margin",
        "big_shift",
        "ambiguity",
    }
    assert all(not (set(item) & forbidden) for item in first_queue["items"])
    missed_audit_ids = {
        entry["audit_id"]
        for entry in first_inventory["entries"]
        if entry["candidate_id"].startswith("c10")
    }
    assert all(
        row["clip_interval"]["end_s_exclusive"]
        - row["clip_interval"]["start_s"]
        == 4.0
        for row in first_queue["items"]
        if row["audit_id"] in missed_audit_ids
    )
    for speaker in candidate.SPEAKERS:
        assert sum(item["speaker_channel"] == speaker for item in first_queue["items"]) >= 8


def test_blind_decisions_must_be_complete_before_reveal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _source = _computed_bundle(tmp_path)
    bundle_path = tmp_path / "candidate.json"
    bundle_path.write_bytes(candidate._canonical_json_bytes(bundle))
    monkeypatch.setattr(candidate, "load_gold_vad_candidate", lambda _path: bundle)
    decisions_path = tmp_path / "decisions.json"
    candidate.create_blind_audit_decisions(bundle_path, decisions_path)
    with pytest.raises(candidate.GoldVadCandidateError, match="listened"):
        candidate.reveal_blind_audit(bundle_path, decisions_path, tmp_path / "reveal.json")

    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    for item in decisions["decisions"]:
        item.update(
            {
                "listened": True,
                "speech_present": True,
                "speaker_correct": True,
                    "manual_start_s": "1",
                "manual_end_s_exclusive": "1.2",
                "note": "Blind listening decision made before reveal.",
            }
        )
    completed = tmp_path / "completed.json"
    completed.write_bytes(candidate._canonical_json_bytes(decisions))
    reveal_path = tmp_path / "reveal.json"
    candidate.reveal_blind_audit(bundle_path, completed, reveal_path)
    reveal = json.loads(reveal_path.read_text(encoding="utf-8"))
    assert reveal["manual_decisions_preceded_reveal"] is True
    assert reveal["automatic_label_promotion"] is False
    assert reveal["counted_gold_ready"] is False
    assert all("revealed_selection_inventory" in item for item in reveal["items"])


def test_no_clobber_and_baseexception_cleanup_are_foreign_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_bytes(b"foreign")
    with pytest.raises(candidate.GoldVadCandidateError, match="overwrite"):
        candidate._write_no_clobber_bytes(destination, b"ours")
    assert destination.read_bytes() == b"foreign"

    destination.unlink()

    def interrupt(_staged: Path, _destination: Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(candidate.os, "link", interrupt)
    with pytest.raises(KeyboardInterrupt):
        candidate._write_no_clobber_bytes(destination, b"ours")
    assert not destination.exists()
    assert not list(tmp_path.glob(".artifact.json.tmp.*"))

    monkeypatch.undo()

    real_link = candidate.os.link

    def link_then_interrupt(staged: Path, published: Path) -> None:
        real_link(staged, published)
        raise KeyboardInterrupt

    monkeypatch.setattr(candidate.os, "link", link_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        candidate._write_no_clobber_bytes(destination, b"ours")
    assert not destination.exists()
    assert not list(tmp_path.glob(".artifact.json.tmp.*"))

    monkeypatch.undo()

    def foreign_replacement(path: Path) -> None:
        path.unlink()
        path.write_bytes(b"foreign-after-publication")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        candidate._write_no_clobber_bytes(
            destination, b"ours", post_validate=foreign_replacement
        )
    assert destination.read_bytes() == b"foreign-after-publication"
    assert not list(tmp_path.glob(".artifact.json.tmp.*"))


def test_candidate_algorithm_never_emits_vad_columns_or_assembly_manifest(
    tmp_path: Path,
) -> None:
    bundle, _source = _computed_bundle(tmp_path)
    raw = candidate._canonical_json_bytes(bundle).decode("utf-8")
    assert '"P1_vad"' not in raw
    assert '"feature_manifest"' not in raw
    assert bundle["scientific_boundary"]["writes_assembly_manifest"] is False
