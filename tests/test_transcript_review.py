from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import wave
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from hri_turn_taking.data import audio_contract as audio_contract_module
from hri_turn_taking.data import audio_gold_alignment as gold_alignment_module
from hri_turn_taking.data import transcript_bridge as bridge_module
from hri_turn_taking.data import transcript_fs_guard as fs_guard
from hri_turn_taking.data import transcript_review as review_module
from hri_turn_taking.data import transcript_review_ui as ui_module
from hri_turn_taking.data.transcript_review import (
    TranscriptReviewError,
)


@pytest.fixture(autouse=True)
def _isolated_full_replay_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep fixtures small while proving review calls both full-replay loaders."""

    def load_audio(path: str | Path, *, authority: Any) -> dict[str, Any]:
        raw = Path(path).read_bytes()
        report = audio_contract_module._strict_json_loads(raw, "audio contract")
        audio_contract_module._validate_contract_shape(report)
        audio_contract_module._verify_sources_from_report(report, authority=authority)
        expected_empty = (
            "master_sample_replay",
            "transcript_timing",
            "hpe_mic_a_anchor",
            "audit_parameters",
            "reference_to_equal_weight_master",
            "inter_lavalier_to_mic_a",
            "analysis_window_contract",
            "gates",
        )
        if any(report[key] != {} for key in expected_empty):
            raise audio_contract_module.AudioContractError(
                "fixture full replay rejected forged derived evidence"
            )
        return report

    def load_gold(path: str | Path, *, authority: Any) -> dict[str, Any]:
        del authority
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if report.get("artifact_kind") != gold_alignment_module.ARTIFACT_KIND:
            raise gold_alignment_module.AudioGoldAlignmentError("fixture kind mismatch")
        if report.get("format_version") != gold_alignment_module.FORMAT_VERSION:
            raise gold_alignment_module.AudioGoldAlignmentError("fixture version mismatch")
        if report.get("gates") != {"alignment_prerequisite_pass": True}:
            raise gold_alignment_module.AudioGoldAlignmentError(
                "fixture full replay rejected forged gates"
            )
        return report

    monkeypatch.setattr(
        audio_contract_module, "_load_audio_contract_with_authority", load_audio
    )
    monkeypatch.setattr(gold_alignment_module, "_load_with_authority", load_gold)


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )
    return path


def _write_wave(path: Path, *, channels: int, rate: int, frames: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * channels * frames)
    return path


def _offset(group: str, scenario: str, *, duration_s: int = 2) -> dict[str, Any]:
    return {
        "session": scenario,
        "timeline_reference": "participants_rgb_audio_edited.wav",
        "frame_reference": "participants_rgb_audio_edited.mp4",
        "fps": 30.0,
        "offset_sec": 0.0,
        "offset_samples_processing_sr": 0,
        "lag_frames_rms": 0,
        "master_sr": 16000,
        "reference_sr": 16000,
        "processing_sr": 16000,
        "reference_duration_sec": float(duration_s),
        "interpretation": {
            "offset_positive": "crop beginning of A/B/C/R aligned wavs",
            "offset_negative": "pad silence at beginning of A/B/C/R aligned wavs",
        },
        "speaker_alignment": {
            speaker: {
                "src": f"{group}/{scenario}/audio/aligned_audio/{speaker}_aligned.wav",
                "dst": f"{group}/{scenario}/audio/aligned_audio/{speaker}_edited.wav",
                "sr": 16000,
                "offset_sec": 0.0,
                "shift_samples": 0,
                "action": "no shift",
                "output_len_samples": duration_s * 16000,
                "output_duration_sec": float(duration_s),
            }
            for speaker in review_module.SPEAKERS
        },
    }


def _event(
    start: str,
    end: str,
    speaker: str = "A",
    text: str = "speech",
) -> dict[str, str]:
    return {
        "start_timecode": start,
        "end_timecode": end,
        "speaker": speaker,
        "text": text,
    }


def _case(tmp_path: Path) -> tuple[
    review_module.ReviewSourceAuthority,
    dict[str, Any],
    Path,
]:
    scene = "03_20__4"
    group, scenario = scene.rsplit("__", 1)
    corpus = tmp_path / "corpus"
    hpe = tmp_path / "hpe"
    contracts = tmp_path / "contracts"
    authority = review_module.ReviewSourceAuthority(corpus, hpe, contracts)
    transcript_dir = corpus / group / "transcript" / scenario
    timeline = _write_json(
        transcript_dir / "timeline_video.json",
        [
            _event("00:00:00:03", "00:00:00:10", "A", "first"),
            _event("00:00:00:12", "00:00:00:20", "B", "second"),
        ],
    )
    offset = _write_json(transcript_dir / "offset.json", _offset(group, scenario))
    assert offset.read_bytes().find(b"30.0") == 148

    audio_authority = audio_contract_module.AudioSourceAuthority(corpus, hpe)
    paths = audio_contract_module.canonical_source_paths(scene, audio_authority)
    for speaker in review_module.SPEAKERS:
        _write_wave(paths[speaker], channels=1, rate=48000, frames=4 * 48000)
    _write_wave(paths["master"], channels=4, rate=48000, frames=4 * 48000)
    _write_wave(paths["reference"], channels=1, rate=16000, frames=2 * 16000)
    _write_json(paths["alignment_report"], {"fixture": True})
    _write_json(paths["hpe_offset"], {"streams": {"mic_A": 0.0, "6cam": -0.1}})
    _write_json(
        paths["hpe_offset_quality"],
        {
            "convention": "shared_time = stream_local_time + offset_s",
            "timebase_keys": {
                "mic_A": "A_aligned.wav t=0",
                "6cam": "world timestamp",
            },
            "quality": {
                "mic_A": {
                    "offset_s": 0.0,
                    "status": "ok",
                    "error_bound_ms": 0.0,
                },
                "6cam": {
                    "offset_s": -0.1,
                    "status": "ok",
                    "error_bound_ms": 33.5,
                    "self_check_ok": True,
                },
            },
        },
    )
    sources = audio_contract_module._bind_sources(paths)
    report = {
        "artifact_kind": audio_contract_module.ARTIFACT_KIND,
        "format_version": audio_contract_module.FORMAT_VERSION,
        "scene": scene,
        "output_role": audio_contract_module.OUTPUT_ROLE,
        "counted_gold_ready": False,
        "trust_model": "trusted_code_integrity_and_reproducibility_not_signature",
        "sources": sources,
        "master_sample_replay": {},
        "transcript_timing": {},
        "hpe_mic_a_anchor": {},
        "audit_parameters": {},
        "reference_to_equal_weight_master": {},
        "inter_lavalier_to_mic_a": {},
        "analysis_window_contract": {},
        "gates": {},
    }
    _write_json(contracts / f"{scene}.audio_contract.json", report)
    gold_root = contracts.parent / "audio_gold_alignment"
    _write_json(
        gold_root / f"{scene}.audio_gold_alignment.json",
        {
            "artifact_kind": gold_alignment_module.ARTIFACT_KIND,
            "format_version": gold_alignment_module.FORMAT_VERSION,
            "scene": scene,
            "authorization": {
                "gold_vad_use_authorized": True,
                "authorized_tick_range": {
                    "frame_rate_hz": 25,
                    "start_inclusive": 1,
                    "end_exclusive": 50,
                    "n_ticks": 49,
                },
            },
            "mapping_contract": {
                "channels": {
                    "A": {"correction_s": "0"},
                    "B": {"correction_s": "0.04"},
                    "C": {"correction_s": "-0.02"},
                    "R": {"correction_s": "0"},
                }
            },
            "gates": {"alignment_prerequisite_pass": True},
        },
    )
    plan = {
        "mechanical_repairs": [],
        "semantic_candidates": [
            review_module._candidate(
                "fixture-e0-end",
                source_index=0,
                field="end_timecode",
                expected="00:00:00:10",
                replacement="00:00:00:11",
                evidence="Owner must listen to the proposed one-frame extension.",
                listening_window_s=("0", "1"),
            )
        ],
        "event_flags": {0: ["semantic_boundary_candidate"]},
        "scene_flags": ["semantic_boundary_decision_required"],
    }
    assert timeline.is_file()
    return authority, plan, timeline


def _write_draft(path: Path, value: dict[str, Any]) -> Path:
    path.write_bytes(review_module._canonical_json_bytes(value))
    return path


def _mark_all_reviewed(
    draft: dict[str, Any],
    *,
    candidate_status: str = "accepted",
) -> dict[str, Any]:
    value = copy.deepcopy(draft)
    for item in value["events"]:
        item["review"] = {
            "reviewed": True,
            "speaker": {"action": "keep", "replacement": None},
            "start_timecode": {"action": "keep", "replacement": None},
            "end_timecode": {"action": "keep", "replacement": None},
            "note": "reviewed all four channels",
        }
    candidate = value["semantic_candidates"][0]
    candidate["decision"] = {
        "status": candidate_status,
        "note": "listened to A/B/C/R and made an explicit decision",
    }
    if candidate_status == "accepted":
        event = next(
            item
            for item in value["events"]
            if item["source_index"] == candidate["source_index"]
        )
        event["review"][candidate["field"]] = {
            "action": "change",
            "replacement": candidate["replacement"],
        }
    return value


def test_draft_covers_every_event_and_clip_binds_all_lavs(tmp_path: Path) -> None:
    authority, plan, _timeline = _case(tmp_path)
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    assert draft["review_status"] == "draft"
    assert draft["counted_gold_ready"] is False
    assert len(draft["events"]) == len(draft["clip_plan"]) == 2
    assert all(item["review"]["reviewed"] is False for item in draft["events"])
    clip = draft["clip_plan"][0]
    assert clip["interval_basis"] == "semantic_candidate_for_listening_only"
    assert clip["candidate_ids_for_listening_only"] == ["fixture-e0-end"]
    assert set(clip["per_speaker_corrected_audio_read_only"]) == set(
        review_module.SPEAKERS
    )
    # local = shared - correction: +40 ms moves B earlier, -20 ms moves C later.
    per_speaker = clip["per_speaker_corrected_audio_read_only"]
    assert Decimal(per_speaker["B"]["event_local_start_s"]) == (
        Decimal(clip["event_shared_start_s"]) - Decimal("0.04")
    )
    assert Decimal(per_speaker["C"]["event_local_start_s"]) == (
        Decimal(clip["event_shared_start_s"]) + Decimal("0.02")
    )
    assert draft["semantic_candidates"][0]["decision"] == {
        "status": "unresolved",
        "note": "",
    }


def test_forged_hash_missing_duplicate_and_bool_index_fail_closed(tmp_path: Path) -> None:
    authority, plan, _timeline = _case(tmp_path)
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    cases: list[dict[str, Any]] = []
    forged = copy.deepcopy(draft)
    forged["sources"]["timeline"]["sha256"] = "0" * 64
    cases.append(forged)
    missing = copy.deepcopy(draft)
    missing["events"].pop()
    cases.append(missing)
    duplicate = copy.deepcopy(draft)
    duplicate["events"][1] = copy.deepcopy(duplicate["events"][0])
    cases.append(duplicate)
    bool_index = copy.deepcopy(draft)
    bool_index["events"][0]["source_index"] = True
    cases.append(bool_index)
    bool_version = copy.deepcopy(draft)
    bool_version["format_version"] = True
    cases.append(bool_version)
    for index, value in enumerate(cases):
        path = _write_draft(tmp_path / f"attack-{index}.json", value)
        with pytest.raises(TranscriptReviewError):
            review_module._load_review_draft_with_authority(
                path, authority=authority, plan=plan
            )


def test_source_mutation_fails_even_if_shallow_hash_is_refreshed(tmp_path: Path) -> None:
    authority, plan, timeline = _case(tmp_path)
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    path = _write_draft(tmp_path / "draft.json", draft)
    events = json.loads(timeline.read_text(encoding="utf-8"))
    events[0]["text"] = "changed after draft"
    _write_json(timeline, events)
    with pytest.raises(TranscriptReviewError, match="changed|replay|match"):
        review_module._load_review_draft_with_authority(
            path, authority=authority, plan=plan
        )

    # Refreshing only the shallow file hash cannot refresh event-level review evidence.
    draft["sources"]["timeline"]["size_bytes"] = timeline.stat().st_size
    draft["sources"]["timeline"]["sha256"] = hashlib.sha256(
        timeline.read_bytes()
    ).hexdigest()
    _write_draft(path, draft)
    with pytest.raises(TranscriptReviewError, match="source snapshot|replay|changed"):
        review_module._load_review_draft_with_authority(
            path, authority=authority, plan=plan
        )


def test_unresolved_candidate_and_invalid_boundary_never_publish(tmp_path: Path) -> None:
    authority, plan, _timeline = _case(tmp_path)
    base = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    unresolved = _mark_all_reviewed(base)
    unresolved["semantic_candidates"][0]["decision"] = {
        "status": "unresolved",
        "note": "",
    }
    unresolved["events"][0]["review"]["end_timecode"] = {
        "action": "keep",
        "replacement": None,
    }
    unresolved_path = _write_draft(tmp_path / "unresolved.json", unresolved)
    unresolved_output = tmp_path / "should-not-exist.json"
    with pytest.raises(TranscriptReviewError, match="unresolved"):
        review_module._finalize_review_with_authority(
            unresolved_path,
            unresolved_output,
            reviewer="owner",
            reviewed_at="2026-08-15T12:00:00+00:00",
            method="manual four-channel listening",
            authority=authority,
            plan=plan,
        )
    assert not unresolved_output.exists()

    invalid = _mark_all_reviewed(base, candidate_status="rejected")
    invalid["events"][0]["review"]["end_timecode"] = {
        "action": "change",
        "replacement": invalid["events"][0]["source"]["start_timecode"],
    }
    invalid_path = _write_draft(tmp_path / "invalid-boundary.json", invalid)
    invalid_output = tmp_path / "invalid-final.json"
    with pytest.raises(TranscriptReviewError, match="strictly positive"):
        review_module._finalize_review_with_authority(
            invalid_path,
            invalid_output,
            reviewer="owner",
            reviewed_at="2026-08-15T12:00:00+00:00",
            method="manual four-channel listening",
            authority=authority,
            plan=plan,
        )
    assert not invalid_output.exists()

    beyond = _mark_all_reviewed(base, candidate_status="rejected")
    beyond["events"][0]["review"]["end_timecode"] = {
        "action": "change",
        "replacement": "00:00:03:00",
    }
    beyond_path = _write_draft(tmp_path / "beyond-duration.json", beyond)
    beyond_output = tmp_path / "beyond-duration-final.json"
    with pytest.raises(TranscriptReviewError, match="beyond the reference duration"):
        review_module._finalize_review_with_authority(
            beyond_path,
            beyond_output,
            reviewer="owner",
            reviewed_at="2026-08-15T12:00:00+00:00",
            method="manual four-channel listening",
            authority=authority,
            plan=plan,
        )
    assert not beyond_output.exists()


def test_final_review_is_coarse_hash_bound_and_no_clobber(tmp_path: Path) -> None:
    authority, plan, _timeline = _case(tmp_path)
    base = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    reviewed = _mark_all_reviewed(base)
    draft_path = _write_draft(tmp_path / "reviewed-draft.json", reviewed)
    output = tmp_path / "final.json"
    review_module._finalize_review_with_authority(
        draft_path,
        output,
        reviewer="owner",
        reviewed_at="2026-08-15T12:00:00+00:00",
        method="manual four-channel listening",
        authority=authority,
        plan=plan,
    )
    loaded = review_module._load_final_review_with_authority(
        output, authority=authority, plan=plan
    )
    assert loaded["review_status"] == "confirmed"
    assert loaded["output_role"] == review_module.OUTPUT_ROLE
    assert loaded["counted_gold_ready"] is False
    assert loaded["manual_review"]["all_in_scope_events_reviewed"] is True
    before = output.read_bytes()
    with pytest.raises(TranscriptReviewError, match="overwrite"):
        review_module._finalize_review_with_authority(
            draft_path,
            output,
            reviewer="owner",
            reviewed_at="2026-08-15T12:00:00+00:00",
            method="manual four-channel listening",
            authority=authority,
            plan=plan,
        )
    assert output.read_bytes() == before


def test_atomic_draft_publish_rolls_back_after_postcheck_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    destination = tmp_path / "draft.json"

    def reject(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise TranscriptReviewError("injected post-publication verification failure")

    monkeypatch.setattr(review_module, "_load_review_draft_with_authority", reject)
    with pytest.raises(TranscriptReviewError, match="injected"):
        review_module._write_review_draft_with_authority(
            "03_20__4", destination, authority=authority, plan=plan
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".draft.json.tmp.*.json"))


def test_05_15_6_exact_byte_preimage_repair_is_narrow() -> None:
    operation = copy.deepcopy(
        review_module.FORMAL_REVIEW_PLANS["05_15__6"]["mechanical_repairs"][0]
    )
    expected = operation["expected"].encode("utf-8")
    replacement = operation["replacement"].encode("utf-8")
    ordinary = [
        _event("00:00:00:00", "00:00:00:01", text=f"event {index}")
        for index in range(93)
    ]
    prefix = json.dumps(ordinary, ensure_ascii=False)[:-1].encode("utf-8")
    last_prefix = (
        b', {"start_timecode":"00:00:00:00","end_timecode":"00:00:00:01",'
        b'"speaker":"A",'
    )
    padding = b" " * (operation["byte_offset"] - len(prefix) - len(last_prefix))
    assert len(padding) >= 0
    raw = prefix + last_prefix + padding + expected + b"}]"
    assert raw[operation["byte_offset"] : operation["byte_offset"] + len(expected)] == expected
    timeline, ids = review_module.acceptance_module._apply_repair_operations(
        raw, {"operations": [operation]}
    )
    assert timeline[93]["text"] == "after we visited Louvre Museum."
    assert ids == ("05-15-6-e93-missing-text-quote",)

    wrong = copy.deepcopy(operation)
    wrong["byte_offset"] += 1
    with pytest.raises(
        review_module.acceptance_module.TranscriptAcceptanceError,
        match="exact byte preimage",
    ):
        review_module.acceptance_module._apply_repair_operations(
            raw, {"operations": [wrong]}
        )
    assert replacement in raw.replace(expected, replacement)


@pytest.mark.parametrize("field", ["master_sample_replay", "gates"])
def test_audio_contract_requires_full_derived_and_gate_replay(
    tmp_path: Path, field: str
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    contract = authority.audio_contract_root / "03_20__4.audio_contract.json"
    forged = json.loads(contract.read_text(encoding="utf-8"))
    forged[field] = {"forged_pass": True}
    _write_json(contract, forged)
    with pytest.raises(TranscriptReviewError, match="full replay|forged derived"):
        review_module._build_review_draft_with_authority(
            "03_20__4", authority=authority, plan=plan
        )


@pytest.mark.parametrize("version", [1, 2])
def test_stale_review_draft_fails_closed(tmp_path: Path, version: int) -> None:
    authority, plan, _timeline = _case(tmp_path)
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    draft["format_version"] = version
    path = _write_draft(tmp_path / f"old-v{version}.json", draft)
    with pytest.raises(TranscriptReviewError, match=rf"review-v{version}.*recreate"):
        review_module._load_review_draft_with_authority(
            path, authority=authority, plan=plan
        )


def test_gold_window_scopes_queue_excludes_e146_like_invalid_and_tracks_prevalence(
    tmp_path: Path,
) -> None:
    authority, _plan, timeline = _case(tmp_path)
    events = [
        _event("00:00:00:00", "00:00:00:01", "A", "before"),
        _event("00:00:00:00", "00:00:00:03", "A", "cross start"),
        _event("00:00:00:03", "00:00:00:10", "B", "inside"),
        _event("00:00:00:27", "00:00:01:03", "C", "cross end"),
        _event("00:00:01:24", "00:00:01:18", "R", "e146-like outside"),
        _event("00:00:00:30", "00:00:01:10", "R", "FF30 outside"),
    ]
    _write_json(timeline, events)
    gold_path = authority.gold_alignment_root / "03_20__4.audio_gold_alignment.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["authorization"]["authorized_tick_range"] = {
        "frame_rate_hz": 25,
        "start_inclusive": 1,
        "end_exclusive": 25,
        "n_ticks": 24,
    }
    _write_json(gold_path, gold)
    plan = {
        "mechanical_repairs": [
            review_module._field_repair(
                "fixture-ff30-carry",
                source_index=5,
                field="start_timecode",
                expected="00:00:00:30",
                replacement="00:00:01:00",
                operation="normalize_timecode",
                evidence="Unique 30/1 carry.",
            )
        ],
        "semantic_candidates": [
            review_module._candidate(
                "fixture-e146-outside",
                source_index=4,
                field="start_timecode",
                expected="00:00:01:24",
                replacement="00:00:01:15",
                evidence="Outside-window semantic candidate must remain inventory only.",
                listening_window_s=("1.4", "1.9"),
            )
        ],
        "event_flags": {4: ["semantic_boundary_candidate"]},
        "scene_flags": ["speaker_collapse_fixture"],
    }
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    assert draft["scope"] == {
        "schema": "transcript_gold_window_scope_v1",
        "frame_rate_hz": 25,
        "authorized_start_tick_inclusive": 1,
        "authorized_end_tick_exclusive": 25,
        "source_event_count": 6,
        "in_scope_pre_review_event_count": 3,
        "excluded_event_count": 3,
        "crossing_event_count": 2,
        "source_count_conserved": True,
    }
    assert [item["source_index"] for item in draft["events"]] == [1, 2, 3]
    clips = {item["source_index"]: item for item in draft["clip_plan"]}
    assert clips[1]["event_shared_start_s"] == "0.04"
    assert clips[3]["event_shared_end_s_exclusive"] == "1"
    assert clips[1]["extract_shared_start_s"] == "0.04"
    assert clips[3]["extract_shared_end_s_exclusive"] == "1"
    assert draft["semantic_candidates"] == []
    excluded = {item["source_index"]: item for item in draft["excluded_event_inventory"]}
    assert excluded[4]["exclusion_reason"] == "nonpositive_interval_wholly_after_window"
    assert excluded[4]["semantic_candidates"][0]["candidate_id"] == (
        "fixture-e146-outside"
    )
    assert excluded[5]["source"]["start_timecode"] == "00:00:01:00"
    assert "source_pre_review_speaker_collapse_fixture" in draft["scene_flags"]
    assert draft["speaker_prevalence"]["source_pre_review"]["R"]["event_count"] == 2
    assert draft["speaker_prevalence"]["in_scope_pre_review"]["R"]["event_count"] == 0

    reviewed = copy.deepcopy(draft)
    for item in reviewed["events"]:
        item["review"] = {
            "reviewed": True,
            "speaker": {"action": "keep", "replacement": None},
            "start_timecode": {"action": "keep", "replacement": None},
            "end_timecode": {"action": "keep", "replacement": None},
            "note": "listened to all four corrected channels",
        }
    reviewed["events"][1]["review"]["speaker"] = {
        "action": "change",
        "replacement": "R",
    }
    reviewed["events"][1]["review"]["note"] = "B was actually R on corrected audio"
    final = review_module._final_review_from_draft(
        reviewed,
        draft_binding={
            "path": str(tmp_path / "not-published.json"),
            "size_bytes": 0,
            "sha256": "0" * 64,
            "media_type": "application/json",
        },
        reviewer="owner",
        reviewed_at="2026-08-15T12:00:00+00:00",
        method="manual four-channel listening",
    )
    assert final["manual_review"]["all_in_scope_events_reviewed"] is True
    assert final["speaker_prevalence"]["accepted_post_review"]["R"]["event_count"] == 1
    assert final["scope"]["accepted_post_review_event_count"] == 3


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_atomic_publish_rolls_back_link_then_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    destination = tmp_path / "published.json"
    original_link = review_module.os.link

    def link_then_interrupt(source: Path, target: Path) -> None:
        original_link(source, target)
        raise interruption("injected after hard-link creation")

    monkeypatch.setattr(review_module.os, "link", link_then_interrupt)
    with pytest.raises(interruption):
        review_module._atomic_publish_no_clobber(
            destination,
            b"{}\n",
            pre_publish_check=lambda: None,
            post_publish_check=lambda _path: None,
        )
    assert not destination.exists()


def test_atomic_rollback_never_deletes_a_replacement_path(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published.json"

    def replace_then_fail(path: Path) -> None:
        path.unlink()
        path.write_bytes(b"foreign")
        raise KeyboardInterrupt("injected after replacement")

    with pytest.raises(KeyboardInterrupt):
        review_module._atomic_publish_no_clobber(
            destination,
            b"ours\n",
            pre_publish_check=lambda: None,
            post_publish_check=replace_then_fail,
        )
    assert destination.read_bytes() == b"foreign"


def _publish_fixture_final(
    tmp_path: Path,
    authority: review_module.ReviewSourceAuthority,
    plan: dict[str, Any],
) -> Path:
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    reviewed = _mark_all_reviewed(draft)
    draft_path = _write_draft(tmp_path / "reviewed-draft.json", reviewed)
    final_path = tmp_path / "reviewed-final.json"
    review_module._finalize_review_with_authority(
        draft_path,
        final_path,
        reviewer="test owner",
        reviewed_at="2026-08-15T12:00:00+00:00",
        method="manual four-channel listening",
        authority=authority,
        plan=plan,
    )
    return final_path


def test_final_flags_are_historical_review_triggers_not_live_requirements(
    tmp_path: Path,
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    final_path = _publish_fixture_final(tmp_path, authority, plan)
    final = review_module._load_final_review_with_authority(
        final_path, authority=authority, plan=plan
    )
    assert "review_trigger_semantic_boundary_decision_required" in final["scene_flags"]
    assert "semantic_boundary_decision_required" not in final["scene_flags"]
    assert all(
        flag.startswith(
            (
                "review_trigger_",
                "source_pre_review_",
                "in_scope_pre_review_",
                "accepted_post_review_",
            )
        )
        for flag in final["scene_flags"]
    )


def test_progress_never_reports_ready_for_a_cross_inconsistent_candidate(
    tmp_path: Path,
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    reviewed = _mark_all_reviewed(
        review_module._build_review_draft_with_authority(
            "03_20__4", authority=authority, plan=plan
        )
    )
    assert review_module.review_progress(reviewed)["ready_to_finalize"] is True
    reviewed["events"][0]["review"]["end_timecode"] = {
        "action": "keep",
        "replacement": None,
    }
    progress = review_module.review_progress(reviewed)
    assert progress["reviewed_in_scope_events"] == progress["in_scope_event_count"]
    assert progress["resolved_semantic_candidates"] == progress["n_semantic_candidates"]
    assert progress["decision_consistency_ok"] is False
    assert progress["ready_to_finalize"] is False


def test_final_bridge_round_trip_retains_audit_and_exact_window_raster(
    tmp_path: Path,
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    final_path = _publish_fixture_final(tmp_path, authority, plan)
    output = tmp_path / "bridge"
    bridge_module._publish_bundle_with_authority(
        final_path, output, authority=authority, plan=plan
    )
    loaded = bridge_module._load_bundle_with_authority(
        output, authority=authority, plan=plan
    )
    assert loaded.scene == "03_20__4"
    assert loaded.acceptance.counted_gold_ready is False
    assert loaded.receipt["review_audit"] == {
        "accepted_events_source": "confirmed_final_review_post_review_repairs_only",
        "accepted_events_derive_only_from_final_post_review": True,
        "source_event_count": 2,
        "in_scope_pre_review_event_count": 2,
        "accepted_post_review_event_count": 2,
        "excluded_event_count": 0,
        "crossing_event_count": 0,
        "source_count_conserved": True,
        "authorized_tick_range": {
            "frame_rate_hz": 25,
            "start_inclusive": 1,
            "end_exclusive": 50,
            "n_ticks": 49,
        },
        "repair_set_sha256": loaded.final_review["manual_review"][
            "repair_set_sha256"
        ],
        "semantic_candidate_decisions_sha256": review_module._canonical_json_sha256(
            loaded.final_review["semantic_candidate_decisions"]
        ),
        "event_reviews_sha256": review_module._canonical_json_sha256(
            loaded.final_review["event_reviews"]
        ),
        "final_scene_flags_sha256": review_module._canonical_json_sha256(
            loaded.final_review["scene_flags"]
        ),
    }
    rows = (output / bridge_module.ACTIVITY_FILENAME).read_text(
        encoding="utf-8"
    ).splitlines()
    assert rows[0] == "tick,timestamp_sec,A,B,C,R"
    assert len(rows) == 50
    assert rows[1].startswith("1,0.04,")
    assert rows[-1].startswith("49,1.96,")

    # A compact acceptance cannot be edited while retaining the bridge receipt.
    acceptance_path = output / bridge_module.ACCEPTANCE_FILENAME
    acceptance_path.write_bytes(acceptance_path.read_bytes() + b" ")
    with pytest.raises(bridge_module.TranscriptBridgeError, match="receipt"):
        bridge_module._load_bundle_with_authority(
            output, authority=authority, plan=plan
        )


@pytest.mark.parametrize("version", [1, 2])
def test_bridge_stale_review_final_fails_before_output(
    tmp_path: Path, version: int
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    final_path = _publish_fixture_final(tmp_path, authority, plan)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["format_version"] = version
    old_path = _write_json(tmp_path / f"old-final-v{version}.json", final)
    output = tmp_path / "must-not-exist"
    with pytest.raises(
        bridge_module.TranscriptBridgeError, match=rf"review-v{version}.*recreate"
    ):
        bridge_module._publish_bundle_with_authority(
            old_path, output, authority=authority, plan=plan
        )
    assert not output.exists()


def test_bridge_rolls_back_rename_then_base_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    final_path = _publish_fixture_final(tmp_path, authority, plan)
    output = tmp_path / "bridge"
    original_rename = bridge_module.os.rename

    def rename_then_interrupt(source: Path, destination: Path) -> None:
        original_rename(source, destination)
        raise KeyboardInterrupt("injected after directory publication")

    monkeypatch.setattr(bridge_module.os, "rename", rename_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        bridge_module._publish_bundle_with_authority(
            final_path, output, authority=authority, plan=plan
        )
    assert not output.exists()


def test_bridge_rollback_preserves_concurrent_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    final_path = _publish_fixture_final(tmp_path, authority, plan)
    output = (tmp_path / "bridge").resolve()
    original_load = bridge_module._load_bundle_with_authority

    def replace_after_publish(
        path: str | Path, *, authority: Any, plan: Any = None, **kwargs: Any
    ) -> Any:
        resolved = Path(path).resolve()
        if resolved == output:
            activity = resolved / bridge_module.ACTIVITY_FILENAME
            activity.write_bytes(b"foreign replacement")
            raise KeyboardInterrupt("injected after a concurrent replacement")
        return original_load(path, authority=authority, plan=plan, **kwargs)

    monkeypatch.setattr(
        bridge_module, "_load_bundle_with_authority", replace_after_publish
    )
    with pytest.raises(KeyboardInterrupt):
        bridge_module._publish_bundle_with_authority(
            final_path, output, authority=authority, plan=plan
        )
    assert (output / bridge_module.ACTIVITY_FILENAME).read_bytes() == (
        b"foreign replacement"
    )


def test_bridge_rollback_preserves_reused_foreign_staging_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    final_path = _publish_fixture_final(tmp_path, authority, plan)
    output = tmp_path / "bridge"
    original_rename = bridge_module.os.rename
    reused_stage: Path | None = None

    def rename_reuse_then_interrupt(source: Path, destination: Path) -> None:
        nonlocal reused_stage
        original_rename(source, destination)
        reused_stage = Path(source)
        reused_stage.mkdir()
        (reused_stage / "foreign.txt").write_text("keep", encoding="utf-8")
        raise KeyboardInterrupt("injected after foreign staging-name reuse")

    monkeypatch.setattr(bridge_module.os, "rename", rename_reuse_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="staging-name reuse"):
        bridge_module._publish_bundle_with_authority(
            final_path, output, authority=authority, plan=plan
        )
    assert not output.exists()
    assert reused_stage is not None
    assert (reused_stage / "foreign.txt").read_text(encoding="utf-8") == "keep"


def test_bridge_rejects_self_consistent_shallow_acceptance_or_raster_forgery(
    tmp_path: Path,
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    final_path = _publish_fixture_final(tmp_path, authority, plan)
    output = tmp_path / "bridge"
    bridge_module._publish_bundle_with_authority(
        final_path, output, authority=authority, plan=plan
    )
    receipt_path = output / bridge_module.RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    acceptance_path = output / bridge_module.ACCEPTANCE_FILENAME
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["timing"]["method"] = "forged but schema-valid method"
    acceptance_path.write_bytes(bridge_module._canonical_json_bytes(acceptance))
    receipt["acceptance"]["size_bytes"] = acceptance_path.stat().st_size
    receipt["acceptance"]["sha256"] = hashlib.sha256(
        acceptance_path.read_bytes()
    ).hexdigest()
    receipt_path.write_bytes(bridge_module._canonical_json_bytes(receipt))
    with pytest.raises(
        bridge_module.TranscriptBridgeError, match="deterministically replay"
    ):
        bridge_module._load_bundle_with_authority(
            output, authority=authority, plan=plan
        )

    # Rebuild a clean bundle, then forge CSV plus matching shallow receipt hash.
    clean = tmp_path / "bridge-clean"
    bridge_module._publish_bundle_with_authority(
        final_path, clean, authority=authority, plan=plan
    )
    receipt_path = clean / bridge_module.RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    activity_path = clean / bridge_module.ACTIVITY_FILENAME
    rows = activity_path.read_text(encoding="utf-8").splitlines()
    rows[1] = rows[1][:-1] + ("1" if rows[1][-1] == "0" else "0")
    activity_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")
    receipt["coarse_activity"]["size_bytes"] = activity_path.stat().st_size
    receipt["coarse_activity"]["sha256"] = hashlib.sha256(
        activity_path.read_bytes()
    ).hexdigest()
    receipt_path.write_bytes(bridge_module._canonical_json_bytes(receipt))
    with pytest.raises(bridge_module.TranscriptBridgeError, match="does not replay"):
        bridge_module._load_bundle_with_authority(
            clean, authority=authority, plan=plan
        )


def test_bridge_receipt_v1_requires_rebuild_from_review_v3(tmp_path: Path) -> None:
    authority, plan, _timeline = _case(tmp_path)
    final_path = _publish_fixture_final(tmp_path, authority, plan)
    output = tmp_path / "bridge"
    bridge_module._publish_bundle_with_authority(
        final_path, output, authority=authority, plan=plan
    )
    receipt_path = output / bridge_module.RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["format_version"] = 1
    receipt_path.write_bytes(bridge_module._canonical_json_bytes(receipt))
    with pytest.raises(
        bridge_module.TranscriptBridgeError, match="receipt v1.*rebuild.*review-v3"
    ):
        bridge_module._load_bundle_with_authority(
            output, authority=authority, plan=plan
        )


def test_ui_playback_does_not_mark_reviewed_and_temp_mix_is_authorised(
    tmp_path: Path,
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    source_bytes = {
        speaker: Path(draft["sources"]["lavalier_wavs"][speaker]["path"]).read_bytes()
        for speaker in review_module.SPEAKERS
    }
    reviews_before = copy.deepcopy([item["review"] for item in draft["events"]])
    output = tmp_path / "clips"
    output.mkdir()
    rendered = ui_module.render_review_clip(draft, 0, "mix", output)
    with wave.open(str(rendered), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 48000
        assert 0 < handle.getnframes() <= 2 * 48000
    assert draft["events"][0]["review"]["reviewed"] is False
    assert ui_module.adjacent_review_position(draft, 0, 1) == 1
    assert ui_module.next_priority_review_position(draft, 0) in {0, 1}
    assert [item["review"] for item in draft["events"]] == reviews_before
    assert all(
        Path(draft["sources"]["lavalier_wavs"][speaker]["path"]).read_bytes()
        == source_bytes[speaker]
        for speaker in review_module.SPEAKERS
    )


def test_ui_playback_rejects_same_size_same_mtime_wav_content_change(
    tmp_path: Path,
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    wav_path = Path(draft["sources"]["lavalier_wavs"]["A"]["path"])
    before = wav_path.stat()
    mutated = bytearray(wav_path.read_bytes())
    mutated[-1] ^= 1
    wav_path.write_bytes(mutated)
    os.utime(wav_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = wav_path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    output = tmp_path / "clips"
    output.mkdir()
    with pytest.raises(ui_module.TranscriptReviewUIError, match="SHA-256"):
        ui_module.render_review_clip(draft, 0, "A", output)


@pytest.mark.skipif(os.name != "nt", reason="Windows CreateFile share-mode contract")
@pytest.mark.parametrize("operation", ["write", "rename", "delete"])
def test_ui_playback_handle_denies_write_rename_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    target = Path(draft["sources"]["lavalier_wavs"]["A"]["path"])
    moved = target.with_name("A_moved.wav")
    real_hash = fs_guard.hash_guarded_read
    denied = False

    def attack_during_hash(handle: fs_guard.GuardedReadHandle) -> str:
        nonlocal denied
        try:
            if operation == "write":
                with target.open("r+b") as writer:
                    writer.write(b"broken")
            elif operation == "rename":
                target.rename(moved)
            else:
                target.unlink()
        except OSError:
            denied = True
        return real_hash(handle)

    monkeypatch.setattr(fs_guard, "hash_guarded_read", attack_during_hash)
    output = tmp_path / "clips"
    output.mkdir()
    ui_module.render_review_clip(draft, 0, "A", output)
    assert denied is True
    assert target.is_file()
    assert not moved.exists()


def test_ui_and_bridge_reject_reparse_ancestors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    wav_path = Path(draft["sources"]["lavalier_wavs"]["A"]["path"])
    guarded_ancestor = wav_path.parent
    real_lstat = Path.lstat

    class ReparseStat:
        def __init__(self, value: os.stat_result) -> None:
            for name in (
                "st_mode",
                "st_dev",
                "st_ino",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            ):
                setattr(self, name, getattr(value, name))
            self.st_file_attributes = (
                getattr(value, "st_file_attributes", 0)
                | fs_guard.FILE_ATTRIBUTE_REPARSE_POINT
            )

    def lstat_with_reparse(path: Path) -> os.stat_result | ReparseStat:
        value = real_lstat(path)
        return ReparseStat(value) if path == guarded_ancestor else value

    monkeypatch.setattr(Path, "lstat", lstat_with_reparse)
    output = tmp_path / "clips"
    output.mkdir()
    with pytest.raises(ui_module.TranscriptReviewUIError, match="ancestor|reparse"):
        ui_module.render_review_clip(draft, 0, "A", output)

    monkeypatch.setattr(Path, "lstat", real_lstat)
    final_path = _publish_fixture_final(tmp_path, authority, plan)
    bundle = tmp_path / "bridge"
    bridge_module._publish_bundle_with_authority(
        final_path, bundle, authority=authority, plan=plan
    )
    guarded_ancestor = bundle.parent
    monkeypatch.setattr(Path, "lstat", lstat_with_reparse)
    with pytest.raises(bridge_module.TranscriptBridgeError, match="reparse|unsafe"):
        bridge_module._load_bundle_with_authority(
            bundle, authority=authority, plan=plan
        )


def test_ui_candidate_and_event_decisions_require_explicit_consistency(
    tmp_path: Path,
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    candidate = draft["semantic_candidates"][0]
    with pytest.raises(ui_module.TranscriptReviewUIError, match="unresolved"):
        ui_module.apply_reviewed_event_decision(
            draft,
            0,
            speaker="A",
            start_timecode="00:00:00:03",
            end_timecode="00:00:00:10",
            note="",
        )
    ui_module.set_semantic_candidate_decision(
        draft,
        candidate["candidate_id"],
        status="accepted",
        note="heard the endpoint on all four channels",
    )
    with pytest.raises(ui_module.TranscriptReviewUIError, match="must be applied"):
        ui_module.apply_reviewed_event_decision(
            draft,
            0,
            speaker="A",
            start_timecode="00:00:00:03",
            end_timecode="00:00:00:10",
            note="",
        )
    ui_module.apply_reviewed_event_decision(
        draft,
        0,
        speaker="A",
        start_timecode="00:00:00:03",
        end_timecode="00:00:00:11",
        note="accepted the proposed one-frame endpoint after listening",
    )
    assert draft["events"][0]["review"]["reviewed"] is True
    assert draft["events"][0]["review"]["end_timecode"] == {
        "action": "change",
        "replacement": "00:00:00:11",
    }
    ui_module.set_semantic_candidate_decision(
        draft,
        candidate["candidate_id"],
        status="rejected",
        note="second listening rejected the proposal",
    )
    assert draft["events"][0]["review"] == review_module._blank_review()
    assert review_module.review_progress(draft)["ready_to_finalize"] is False


def test_ui_revision_save_is_fully_replayed_and_no_clobber(tmp_path: Path) -> None:
    authority, plan, _timeline = _case(tmp_path)
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    output = tmp_path / "draft.rev0001.json"
    ui_module._save_revision_with_authority(
        draft, output, authority=authority, plan=plan
    )
    assert review_module._load_review_draft_with_authority(
        output, authority=authority, plan=plan
    )["scene"] == "03_20__4"
    before = output.read_bytes()
    with pytest.raises(ui_module.TranscriptReviewUIError, match="overwrite"):
        ui_module._save_revision_with_authority(
            draft, output, authority=authority, plan=plan
        )
    assert output.read_bytes() == before
    forbidden = authority.corpus_root / "forbidden-revision.json"
    with pytest.raises(ui_module.TranscriptReviewUIError, match="source roots"):
        ui_module._save_revision_with_authority(
            draft, forbidden, authority=authority, plan=plan
        )
    assert not forbidden.exists()
    revisions = tmp_path / "revisions"
    revisions.mkdir()
    lexical_escape = revisions / ".." / "corpus" / "escaped-revision.json"
    with pytest.raises(ui_module.TranscriptReviewUIError, match="source roots"):
        ui_module._save_revision_with_authority(
            draft, lexical_escape, authority=authority, plan=plan
        )
    assert not (authority.corpus_root / "escaped-revision.json").exists()


def test_bridge_output_lexical_escape_into_source_root_is_rejected(
    tmp_path: Path,
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    final_path = _publish_fixture_final(tmp_path, authority, plan)
    staging_parent = tmp_path / "safe"
    staging_parent.mkdir()
    escaped = staging_parent / ".." / "corpus" / "escaped-bridge"
    with pytest.raises(bridge_module.TranscriptBridgeError, match="source roots"):
        bridge_module._publish_bundle_with_authority(
            final_path, escaped, authority=authority, plan=plan
        )
    assert not (authority.corpus_root / "escaped-bridge").exists()


def test_ui_offline_placeholder_preflight_fails_before_full_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    draft = review_module._build_review_draft_with_authority(
        "03_20__4", authority=authority, plan=plan
    )
    draft_path = _write_draft(tmp_path / "draft.json", draft)
    monkeypatch.setattr(
        ui_module,
        "_file_attributes",
        lambda _path: getattr(stat, "FILE_ATTRIBUTE_OFFLINE", 0x1000),
    )

    def must_not_replay(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("offline preflight must run before full source replay")

    monkeypatch.setattr(
        review_module, "_load_review_draft_with_authority", must_not_replay
    )
    with pytest.raises(ui_module.TranscriptReviewUIError, match="Offline placeholder"):
        ui_module._load_workspace_with_authority(
            draft_path, authority=authority, plan=plan
        )


def _formal_review_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority: review_module.ReviewSourceAuthority,
    plan: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    annotations = tmp_path / "formal" / "annotations"
    annotations.mkdir(parents=True)
    review_root = annotations / "transcript_review"
    bridge_root = annotations / "transcript_bridge"
    monkeypatch.setattr(review_module, "FORMAL_REVIEW_ROOT", review_root)
    monkeypatch.setattr(bridge_module, "FORMAL_BRIDGE_ROOT", bridge_root)
    monkeypatch.setattr(review_module, "default_source_authority", lambda: authority)
    monkeypatch.setitem(review_module.FORMAL_REVIEW_PLANS, "03_20__4", plan)

    scene_dir = review_root / "03_20__4"
    scene_dir.mkdir(parents=True)
    draft = _mark_all_reviewed(
        review_module._build_review_draft_with_authority(
            "03_20__4", authority=authority, plan=plan
        )
    )
    draft_path = _write_draft(scene_dir / "draft.rev0000.json", draft)
    final_path = scene_dir / "final.v3.json"
    review_module._finalize_review_with_authority(
        draft_path,
        final_path,
        reviewer="test owner",
        reviewed_at="2026-08-15T12:00:00+00:00",
        method="manual four-channel listening",
        authority=authority,
        plan=plan,
    )
    return review_root, bridge_root, draft_path, final_path


def test_production_review_and_bridge_paths_are_scene_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    review_root, bridge_root, draft_path, final_path = _formal_review_fixture(
        tmp_path, monkeypatch, authority, plan
    )

    assert review_module.load_review_draft(draft_path)["scene"] == "03_20__4"
    assert review_module.load_final_review(final_path)["scene"] == "03_20__4"
    copied_draft = tmp_path / "copied-draft.rev0000.json"
    copied_final = tmp_path / "copied-final.v3.json"
    shutil.copy2(draft_path, copied_draft)
    shutil.copy2(final_path, copied_final)
    with pytest.raises(TranscriptReviewError, match="canonical"):
        review_module.load_review_draft(copied_draft)
    with pytest.raises(TranscriptReviewError, match="exactly"):
        review_module.load_final_review(copied_final)

    published = bridge_module.publish_transcript_bridge_bundle(final_path)
    assert published == bridge_root / "03_20__4"
    assert bridge_module.load_transcript_bridge_bundle(published).scene == "03_20__4"

    copied_bundle = tmp_path / "copied-bridge"
    shutil.copytree(published, copied_bundle)
    with pytest.raises(bridge_module.TranscriptBridgeError, match="exactly"):
        bridge_module.load_transcript_bridge_bundle(copied_bundle)

    renamed_bundle = tmp_path / "renamed-bridge"
    published.rename(renamed_bundle)
    with pytest.raises(bridge_module.TranscriptBridgeError, match="exactly"):
        bridge_module.load_transcript_bridge_bundle(renamed_bundle)
    assert renamed_bundle.is_dir()


def test_publish_holds_canonical_final_and_draft_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    _review_root, bridge_root, draft_path, final_path = _formal_review_fixture(
        tmp_path, monkeypatch, authority, plan
    )
    private_final = tmp_path / "private-final.json"
    review_module._finalize_review_with_authority(
        draft_path,
        private_final,
        reviewer="replacement reviewer",
        reviewed_at="2026-08-15T12:01:00+00:00",
        method="manual four-channel listening",
        authority=authority,
        plan=plan,
    )
    original_publish = bridge_module._publish_bundle_with_authority

    def swap_after_preflight(*args: Any, **kwargs: Any) -> Path:
        os.replace(private_final, final_path)
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(
        bridge_module, "_publish_bundle_with_authority", swap_after_preflight
    )
    with pytest.raises(
        bridge_module.TranscriptBridgeError,
        match="canonical|changed|guarded|open|publication failed|authority",
    ):
        bridge_module.publish_transcript_bridge_bundle(final_path)
    assert not (bridge_root / "03_20__4").exists()


def test_bundle_load_holds_receipt_final_and_draft_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, plan, _timeline = _case(tmp_path)
    _review_root, _bridge_root, draft_path, final_path = _formal_review_fixture(
        tmp_path, monkeypatch, authority, plan
    )
    canonical_bundle = bridge_module.publish_transcript_bridge_bundle(final_path)

    private_draft = _write_draft(
        tmp_path / "private-draft.json",
        json.loads(draft_path.read_text(encoding="utf-8")),
    )
    private_final = tmp_path / "private-final.json"
    review_module._finalize_review_with_authority(
        private_draft,
        private_final,
        reviewer="test owner",
        reviewed_at="2026-08-15T12:00:00+00:00",
        method="manual four-channel listening",
        authority=authority,
        plan=plan,
    )
    private_bundle = tmp_path / "private-bridge"
    bridge_module._publish_bundle_with_authority(
        private_final, private_bundle, authority=authority, plan=plan
    )
    private_receipt = private_bundle / bridge_module.RECEIPT_FILENAME
    canonical_receipt = canonical_bundle / bridge_module.RECEIPT_FILENAME
    original_load = bridge_module._load_bundle_with_authority

    def swap_after_preflight(*args: Any, **kwargs: Any) -> Any:
        os.replace(private_receipt, canonical_receipt)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(
        bridge_module, "_load_bundle_with_authority", swap_after_preflight
    )
    with pytest.raises(
        bridge_module.TranscriptBridgeError, match="canonical|changed|guarded|open"
    ):
        bridge_module.load_transcript_bridge_bundle(canonical_bundle)
