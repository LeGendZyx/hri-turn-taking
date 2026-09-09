from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import hri_turn_taking.data.transcript_acceptance as transcript_acceptance_module
from hri_turn_taking.data import audio_gold_alignment as gold_alignment_module
from hri_turn_taking.data.transcript_acceptance import (
    ARTIFACT_KIND,
    FORMAT_VERSION,
    LEGACY_OFFSET_FPS_NORMALIZATION,
    LEGACY_OFFSET_FPS_NORMALIZATION_ID,
    OFFSET_LAG_QUANTIZATION_ERROR_BOUND_MS,
    OFFSET_LAG_QUANTIZATION_RULE,
    OFFSET_PROCESSING_RATE_HZ,
    OFFSET_RMS_HOP_SAMPLES,
    OUTPUT_ROLE,
    SHARED_TIME_FORMULA,
    SOURCE_NORMALIZATION_CONTRACT,
    TIMECODE_QUANTIZATION_ERROR_BOUND_MS,
    TRANSCRIPT_SIDE_ERROR_BOUND_MS,
    TranscriptAcceptanceError,
    TranscriptSourceAuthority,
    derive_manual_review_bindings,
    file_binding,
    parse_timecode_frames,
    rasterize_transcript_activity_reference,
)
from hri_turn_taking.data.transcript_acceptance import (
    load_transcript_acceptance as load_production_transcript_acceptance,
)


@pytest.fixture(autouse=True)
def _isolated_gold_full_replay(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(gold_alignment_module, "_load_with_authority", load_gold)


def load_transcript_acceptance(path: Path, *, authority: TranscriptSourceAuthority) -> Any:
    """Exercise the internal loader under an isolated test authority."""

    return transcript_acceptance_module._load_transcript_acceptance_with_authority(
        path,
        authority=authority,
    )


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def _event(start: str, end: str, speaker: str, text: str = "speech") -> dict[str, str]:
    return {
        "start_timecode": start,
        "end_timecode": end,
        "speaker": speaker,
        "text": text,
    }


def _offset(*, scene: str = "05_14__4", shift: int = 5120, lag: int = 10) -> dict[str, Any]:
    group, session = scene.rsplit("__", maxsplit=1)
    duration = 5.0
    alignment = {}
    for speaker in "ABCR":
        alignment[speaker] = {
            "src": f"{group}/{session}/audio/aligned_audio/{speaker}_aligned.wav",
            "dst": f"{group}/{session}/audio/aligned_audio/{speaker}_edited.wav",
            "sr": 16000,
            "offset_sec": 0.32,
            "shift_samples": shift,
            "action": f"crop {shift} samples",
            "output_len_samples": 80000,
            "output_duration_sec": duration,
        }
    return {
        "session": session,
        "timeline_reference": "participants_rgb_audio_edited.wav",
        "frame_reference": "participants_rgb_audio_edited.mp4",
        "fps": 30,
        "offset_sec": 0.32,
        "offset_samples_processing_sr": shift,
        "lag_frames_rms": lag,
        "master_sr": 16000,
        "reference_sr": 16000,
        "processing_sr": 16000,
        "reference_duration_sec": duration,
        "interpretation": {
            "offset_positive": "crop beginning of A/B/C/R aligned wavs",
            "offset_negative": "pad silence at beginning of A/B/C/R aligned wavs",
        },
        "speaker_alignment": alignment,
    }


def _hpe_offset() -> dict[str, Any]:
    return {"streams": {"mic_A": 0.0, "6cam": -0.1}}


def _hpe_quality() -> dict[str, Any]:
    return {
        "convention": "shared_time = stream_local_time + offset_s",
        "timebase_keys": {"mic_A": "A_aligned.wav t=0", "6cam": "world timestamp"},
        "quality": {
            "mic_A": {"offset_s": 0.0, "status": "ok", "error_bound_ms": 0.0},
            "6cam": {
                "offset_s": -0.1,
                "status": "ok",
                "error_bound_ms": 33.5,
                "self_check_ok": True,
            },
        },
    }


def _base_events() -> list[dict[str, str]]:
    # Source order is deliberately not chronological. A intervals overlap;
    # B overlaps A and must remain independently active.
    return [
        _event("00:00:00:15", "00:00:01:15", "B", "B overlap"),
        _event("00:00:00:00", "00:00:01:00", "A", "A first"),
        _event("00:00:00:20", "00:00:02:00", "A", "A continuation"),
        _event("00:00:02:00", "00:00:03:00", "C", "C"),
        _event("00:00:03:00", "00:00:04:00", "R", "R"),
    ]


def _case(
    tmp_path: Path,
    *,
    events: list[dict[str, str]] | None = None,
    timeline_raw: bytes | None = None,
    repairs: list[dict[str, Any]] | None = None,
    scene: str = "05_14__4",
) -> tuple[Path, dict[str, Any], dict[str, Path], TranscriptSourceAuthority]:
    group, scenario = scene.rsplit("__", maxsplit=1)
    corpus_root = tmp_path / "corpus"
    hpe_delivery_root = tmp_path / "hpe_delivery"
    transcript_dir = corpus_root / group / "transcript" / scenario
    transcript_dir.mkdir(parents=True)
    timeline = transcript_dir / "timeline_video.json"
    if timeline_raw is None:
        _write_json(timeline, _base_events() if events is None else events)
    else:
        timeline.write_bytes(timeline_raw)
    offset = _write_json(transcript_dir / "offset.json", _offset(scene=scene))
    reference = (
        corpus_root / group / "Processed" / scenario / "video" / "participants_rgb_audio_edited.wav"
    )
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"RIFF-reference-bytes")
    hpe_scene = hpe_delivery_root / scene
    hpe_scene.mkdir(parents=True)
    hpe_offset = _write_json(hpe_scene / "offset.json", _hpe_offset())
    hpe_quality = _write_json(hpe_scene / "offset_quality.json", _hpe_quality())
    gold_root = hpe_delivery_root.parent / "audio_gold_alignment"
    gold_root.mkdir(parents=True)
    gold_alignment = _write_json(
        gold_root / f"{scene}.audio_gold_alignment.json",
        {
            "artifact_kind": gold_alignment_module.ARTIFACT_KIND,
            "format_version": gold_alignment_module.FORMAT_VERSION,
            "scene": scene,
            "authorization": {
                "gold_vad_use_authorized": True,
                "authorized_tick_range": {
                    "frame_rate_hz": 25,
                    "start_inclusive": 8,
                    "end_exclusive": 133,
                    "n_ticks": 125,
                },
            },
            "gates": {"alignment_prerequisite_pass": True},
        },
    )
    paths = {
        "timeline": timeline,
        "offset": offset,
        "reference_wav": reference,
        "hpe_offset": hpe_offset,
        "hpe_offset_quality": hpe_quality,
        "gold_alignment": gold_alignment,
    }
    sources = {name: file_binding(path) for name, path in paths.items()}
    manifest: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "scene": scene,
        "output_role": OUTPUT_ROLE,
        "counted_gold_ready": False,
        "review_status": "confirmed",
        "unresolved_issues": [],
        "sources": sources,
        "source_normalizations": {
            "contract": SOURCE_NORMALIZATION_CONTRACT,
            "version": 1,
            "operations": [],
        },
        "timecode_contract": {
            "fps_numerator": 30,
            "fps_denominator": 1,
            "drop_frame": False,
            "frame_index_origin": 0,
        },
        "repairs": {"operations": [] if repairs is None else repairs},
        "manual_review": {
            "status": "confirmed",
            "reviewer": "test-owner",
            "reviewed_at": "2026-08-15T00:00:00+00:00",
            "timeline_sha256": sources["timeline"]["sha256"],
            "canonical_semantic_sha256": "0" * 64,
            "event_summary": {
                "n_events": 1,
                "speaker_event_counts": {speaker: 0 for speaker in "ABCR"},
                "minimum_start_frame": 0,
                "maximum_end_frame": 1,
            },
            "repair_set_sha256": "0" * 64,
            "repair_ids": [],
            "all_in_scope_events_reviewed": True,
            "speaker_assignments_reviewed": True,
            "time_boundaries_reviewed": True,
            "method": "audio-and-video manual review",
        },
        "timing": {
            "status": "confirmed",
            "method": "RMS offset plus HPE D3",
            "formula": SHARED_TIME_FORMULA,
            "transcript_offset_sec": "0.32",
            "timecode_quantization_rule": "half_frame_ceiling_to_whole_millisecond",
            "timecode_quantization_error_bound_ms": str(TIMECODE_QUANTIZATION_ERROR_BOUND_MS),
            "offset_lag_quantization_rule": OFFSET_LAG_QUANTIZATION_RULE,
            "offset_lag_hop_samples": OFFSET_RMS_HOP_SAMPLES,
            "offset_lag_processing_rate_hz": OFFSET_PROCESSING_RATE_HZ,
            "offset_lag_quantization_error_bound_ms": str(OFFSET_LAG_QUANTIZATION_ERROR_BOUND_MS),
            "transcript_side_error_bound_ms": str(TRANSCRIPT_SIDE_ERROR_BOUND_MS),
            "hpe_stream": "6cam",
            "hpe_offset_sec": "-0.1",
            "hpe_error_bound_ms": "33.5",
            "hpe_quality_status": "ok",
            "combined_error_bound_ms": "66.5",
            "uncertainty_combination": "conservative_sum",
            "uncertainty_accepted": True,
        },
        "scope": {},
        "speaker_prevalence": {},
        "excluded_event_inventory": [],
    }
    try:
        _refresh_review_bindings(manifest, timeline)
    except TranscriptAcceptanceError:
        # Invalid-input tests must reach the loader's fail-closed path.
        pass
    manifest_dir = tmp_path / "manifests" / scene
    manifest_dir.mkdir(parents=True)
    manifest_path = _write_json(manifest_dir / "transcript_acceptance.json", manifest)
    authority = TranscriptSourceAuthority(corpus_root, hpe_delivery_root)
    return manifest_path, manifest, paths, authority


def _rewrite_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _write_json(path, manifest)


def _rebind(manifest: dict[str, Any], name: str, path: Path) -> None:
    manifest["sources"][name] = file_binding(path)


def _legacy_fps_normalization(
    offset_path: Path, *, byte_offset: int | None = None
) -> dict[str, Any]:
    raw = offset_path.read_bytes()
    key_start = raw.index(b'"fps"')
    token_start = raw.index(b":", key_start) + 1
    while raw[token_start : token_start + 1] in {b" ", b"\t", b"\r", b"\n"}:
        token_start += 1
    return {
        "normalization_id": LEGACY_OFFSET_FPS_NORMALIZATION_ID,
        "operation": LEGACY_OFFSET_FPS_NORMALIZATION,
        "source": "offset",
        "json_path": "$.fps",
        "byte_offset": token_start if byte_offset is None else byte_offset,
        "expected_json_token": "30.0",
        "expected_json_type": "float",
        "canonical_json_type": "integer",
        "canonical_value": 30,
        "evidence": "legacy transcript.py serialized fps with float(fps)",
        "review_status": "confirmed",
    }


def _refresh_review_bindings(manifest: dict[str, Any], timeline: Path) -> None:
    gold_path = Path(manifest["sources"]["gold_alignment"]["path"])
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    ticks = gold["authorization"]["authorized_tick_range"]
    bindings = derive_manual_review_bindings(
        timeline.read_bytes(),
        manifest["repairs"],
        reference_duration_sec=Decimal("5"),
        transcript_offset_sec=Decimal(manifest["timing"]["transcript_offset_sec"]),
        authorized_start_tick=ticks["start_inclusive"],
        authorized_end_tick=ticks["end_exclusive"],
    )
    for key in (
        "timeline_sha256",
        "canonical_semantic_sha256",
        "event_summary",
        "repair_set_sha256",
        "repair_ids",
    ):
        manifest["manual_review"][key] = bindings[key]
    manifest["scope"] = bindings["scope_summary"]
    manifest["speaker_prevalence"] = bindings["speaker_prevalence"]
    manifest["excluded_event_inventory"] = bindings["excluded_event_inventory"]


def test_valid_reference_sorts_stably_unions_same_speaker_and_preserves_overlap(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest, _paths, authority = _case(tmp_path)

    accepted = load_transcript_acceptance(manifest_path, authority=authority)

    assert accepted.output_role == OUTPUT_ROLE
    assert accepted.counted_gold_ready is False
    assert accepted.timecode_quantization_error_bound_ms == Decimal("17")
    assert accepted.offset_lag_quantization_error_bound_ms == Decimal("16")
    assert accepted.transcript_side_error_bound_ms == Decimal("33")
    assert accepted.combined_error_bound_ms == Decimal("66.5")
    assert accepted.source_normalizations == ()
    assert [event.source_index for event in accepted.events] == [1, 0, 2, 3, 4]
    assert accepted.speaker_intervals["A"] == ((0, 60),)
    assert accepted.speaker_intervals["B"] == ((15, 45),)

    # Transcript offset is +0.32 s. A=[0.32, 2.32), B=[0.82, 1.82).
    timestamps = [Decimal(tick) / 25 for tick in range(8, 59)]
    reference = rasterize_transcript_activity_reference(accepted, timestamps)
    assert reference.counted_gold_ready is False
    assert reference.values["A"][0] == 1
    assert reference.values["A"][-1] == 0  # t=2.32 is the excluded endpoint
    overlap_index = timestamps.index(Decimal("0.84"))
    assert reference.values["A"][overlap_index] == 1
    assert reference.values["B"][overlap_index] == 1


def test_raw_invalid_json_requires_hash_bound_exact_preimage_repair(tmp_path: Path) -> None:
    raw = (
        b'[{"start_timecode":"00:00:00:00","end_timecode":"00:00:01:00",'
        b'"speaker":"A","text": after we visited Louvre Museum."}]'
    )
    manifest_path, manifest, paths, authority = _case(tmp_path, timeline_raw=raw)
    with pytest.raises(TranscriptAcceptanceError, match="not valid JSON"):
        load_transcript_acceptance(manifest_path, authority=authority)

    expected = '"text": after we visited Louvre Museum."'
    manifest["repairs"]["operations"] = [
        {
            "repair_id": "missing-opening-quote",
            "operation": "insert_missing_text_opening_quote",
            "source_index": 0,
            "field": "text",
            "byte_offset": raw.index(expected.encode()),
            "expected": expected,
            "replacement": '"text": "after we visited Louvre Museum."',
            "classification": "mechanical",
            "evidence": "single missing JSON quote; text bytes otherwise preserved",
            "review_status": "confirmed",
        }
    ]
    _refresh_review_bindings(manifest, paths["timeline"])
    _rewrite_manifest(manifest_path, manifest)
    accepted = load_transcript_acceptance(manifest_path, authority=authority)
    assert accepted.events[0].text == "after we visited Louvre Museum."

    manifest["repairs"]["operations"][0]["byte_offset"] += 1
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="exact byte preimage"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_whole_file_replacement_is_not_a_supported_repair(tmp_path: Path) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    raw_document = paths["timeline"].read_text(encoding="utf-8")
    manifest["repairs"] = {
        "operations": [
            {
                "repair_id": "rewrite-all",
                "operation": "insert_missing_text_opening_quote",
                "source_index": 0,
                "field": "text",
                "byte_offset": 0,
                "expected": raw_document,
                "replacement": '"' + raw_document,
                "classification": "mechanical",
                "evidence": "not a narrow operation",
                "review_status": "confirmed",
            }
        ]
    }
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="exact text key"):
        load_transcript_acceptance(manifest_path, authority=authority)


@pytest.mark.parametrize(
    "timecode",
    [
        "00:00:01:00 ",
        "00:00:01:30",
        "00:00:01:31",
        "00:00:60:00",
        "00：00：01：00",
    ],
)
def test_strict_timecode_rejects_known_malformed_forms(timecode: str) -> None:
    with pytest.raises(TranscriptAcceptanceError):
        parse_timecode_frames(timecode)


@pytest.mark.parametrize(
    ("malformed", "canonical"),
    [("00:00:00:30", "00:00:01:00"), ("00:00:01:00 ", "00:00:01:00")],
)
def test_ff30_and_trailing_space_need_explicit_exact_field_repairs(
    tmp_path: Path, malformed: str, canonical: str
) -> None:
    events = [_event("00:00:00:00", malformed, "A")]
    repairs = [
        {
            "repair_id": "carry-frame-30",
            "operation": "normalize_timecode",
            "source_index": 0,
            "field": "end_timecode",
            "expected": malformed,
            "replacement": canonical,
            "classification": "mechanical",
            "evidence": "unique mechanical normalization under the 30-fps contract",
            "review_status": "confirmed",
        }
    ]
    manifest_path, _manifest, _paths, authority = _case(tmp_path, events=events, repairs=repairs)
    accepted = load_transcript_acceptance(manifest_path, authority=authority)
    assert accepted.events[0].end_frame == 30


@pytest.mark.parametrize(
    "events",
    [
        [_event("00:00:02:00", "00:00:01:00", "A")],
        [_event("00:00:01:00", "00:00:01:00", "A")],
        [_event("00:00:00:00", "00:00:01:00", "R ")],
    ],
)
def test_negative_zero_and_whitespace_speaker_fail_closed(
    tmp_path: Path, events: list[dict[str, str]]
) -> None:
    manifest_path, _manifest, _paths, authority = _case(tmp_path, events=events)
    with pytest.raises(TranscriptAcceptanceError):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_field_repair_cannot_guess_or_target_the_same_field_twice(tmp_path: Path) -> None:
    events = [_event("00:00:02:00", "00:00:01:00", "A")]
    repair = {
        "repair_id": "manual-boundary",
        "operation": "correct_timecode_boundary",
        "source_index": 0,
        "field": "end_timecode",
        "expected": "00:00:09:99",
        "replacement": "00:00:03:00",
        "classification": "manual_audio",
        "evidence": "reviewed against lavalier audio",
        "review_status": "confirmed",
    }
    manifest_path, manifest, _paths, authority = _case(tmp_path, events=events)
    manifest["repairs"]["operations"] = [repair]
    with pytest.raises(TranscriptAcceptanceError, match="exact field preimage"):
        _rewrite_manifest(manifest_path, manifest)
        load_transcript_acceptance(manifest_path, authority=authority)

    repair["expected"] = "00:00:01:00"
    duplicate = dict(repair, repair_id="duplicate-boundary")
    manifest["repairs"]["operations"] = [repair, duplicate]
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="multiple repairs"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_manual_owner_text_correction_is_exact_and_trimmed() -> None:
    raw = json.dumps([_event("00:00:00:00", "00:00:01:00", "R", "Okay, xxxx.")]).encode("utf-8")
    repair = {
        "repair_id": "owner-text",
        "operation": "correct_text_content",
        "source_index": 0,
        "field": "text",
        "expected": "Okay, xxxx.",
        "replacement": "Okay, let's stop there.",
        "classification": "manual_owner",
        "evidence": "owner supplied the exact wording",
        "review_status": "confirmed",
    }
    timeline, repair_ids = transcript_acceptance_module._apply_repair_operations(
        raw, {"operations": [repair]}
    )
    assert repair_ids == ("owner-text",)
    assert timeline[0]["text"] == "Okay, let's stop there."

    invalid_classification = dict(repair, classification="manual_audio")
    with pytest.raises(TranscriptAcceptanceError, match="manual-owner"):
        transcript_acceptance_module._apply_repair_operations(
            raw, {"operations": [invalid_classification]}
        )
    padded = dict(repair, replacement=" Okay, let's stop there. ")
    with pytest.raises(TranscriptAcceptanceError, match="surrounding whitespace"):
        transcript_acceptance_module._apply_repair_operations(raw, {"operations": [padded]})
    wrong_preimage = dict(repair, expected="Okay, something else.")
    with pytest.raises(TranscriptAcceptanceError, match="exact field preimage"):
        transcript_acceptance_module._apply_repair_operations(raw, {"operations": [wrong_preimage]})


@pytest.mark.parametrize(
    ("key", "value", "pattern"),
    [
        ("review_status", "pending", "review_status"),
        ("unresolved_issues", ["negative interval"], "unresolved_issues"),
        ("counted_gold_ready", True, "never declare counted gold"),
    ],
)
def test_unconfirmed_unresolved_or_gold_promotion_is_rejected(
    tmp_path: Path, key: str, value: Any, pattern: str
) -> None:
    manifest_path, manifest, _paths, authority = _case(tmp_path)
    manifest[key] = value
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match=pattern):
        load_transcript_acceptance(manifest_path, authority=authority)


@pytest.mark.parametrize(
    "field",
    ["timeline_sha256", "canonical_semantic_sha256", "repair_set_sha256"],
)
def test_manual_review_binds_raw_semantics_and_repair_set(tmp_path: Path, field: str) -> None:
    manifest_path, manifest, _paths, authority = _case(tmp_path)
    manifest["manual_review"][field] = "0" * 64
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="manual review"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_manual_review_binds_exact_event_summary_and_repair_ids(tmp_path: Path) -> None:
    manifest_path, manifest, _paths, authority = _case(tmp_path)
    manifest["manual_review"]["event_summary"]["n_events"] = 5.0
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="exact JSON integer"):
        load_transcript_acceptance(manifest_path, authority=authority)

    manifest_path, manifest, _paths, authority = _case(tmp_path / "second")
    manifest["manual_review"]["repair_ids"] = ["invented"]
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="repair_ids"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_refreshing_raw_hash_does_not_refresh_manual_semantic_review(tmp_path: Path) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    events = _base_events()
    events[0]["text"] = "changed after review"
    _write_json(paths["timeline"], events)
    _rebind(manifest, "timeline", paths["timeline"])
    manifest["manual_review"]["timeline_sha256"] = manifest["sources"]["timeline"]["sha256"]
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="canonical event semantics"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_repair_evidence_change_requires_new_manual_repair_set_review(tmp_path: Path) -> None:
    repair = {
        "repair_id": "carry-frame-30",
        "operation": "normalize_timecode",
        "source_index": 0,
        "field": "end_timecode",
        "expected": "00:00:00:30",
        "replacement": "00:00:01:00",
        "classification": "mechanical",
        "evidence": "original evidence",
        "review_status": "confirmed",
    }
    manifest_path, manifest, _paths, authority = _case(
        tmp_path,
        events=[_event("00:00:00:00", "00:00:00:30", "A")],
        repairs=[repair],
    )
    manifest["repairs"]["operations"][0]["evidence"] = "changed evidence"
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="declared repair set"):
        load_transcript_acceptance(manifest_path, authority=authority)


@pytest.mark.parametrize(("shift", "lag"), [(5119, 10), (5120, 11)])
def test_offset_arithmetic_cannot_be_bypassed_by_refreshing_source_hash(
    tmp_path: Path, shift: int, lag: int
) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    _write_json(paths["offset"], _offset(shift=shift, lag=lag))
    _rebind(manifest, "offset", paths["offset"])
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="inconsistent"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_hpe_d3_and_conservative_uncertainty_are_replayed(tmp_path: Path) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    changed = _hpe_quality()
    changed["quality"]["6cam"]["error_bound_ms"] = 40.0
    _write_json(paths["hpe_offset_quality"], changed)
    _rebind(manifest, "hpe_offset_quality", paths["hpe_offset_quality"])
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="error bound"):
        load_transcript_acceptance(manifest_path, authority=authority)

    manifest["timing"]["hpe_error_bound_ms"] = "40.0"
    manifest["timing"]["combined_error_bound_ms"] = "66.5"
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="transcript-side 33 ms"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_timecode_uncertainty_cannot_be_self_reported(tmp_path: Path) -> None:
    manifest_path, manifest, _paths, authority = _case(tmp_path)
    manifest["timing"]["timecode_quantization_error_bound_ms"] = "1"
    manifest["timing"]["combined_error_bound_ms"] = "34.5"
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="frozen 30-fps contract"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_offset_lag_half_hop_and_transcript_side_bound_are_recomputed(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, _paths, authority = _case(tmp_path)
    manifest["timing"]["offset_lag_quantization_error_bound_ms"] = "0"
    manifest["timing"]["transcript_side_error_bound_ms"] = "17"
    manifest["timing"]["combined_error_bound_ms"] = "50.5"
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="RMS half-hop bound"):
        load_transcript_acceptance(manifest_path, authority=authority)


@pytest.mark.parametrize("field", ["offset_lag_hop_samples", "offset_lag_processing_rate_hz"])
def test_offset_lag_hop_and_rate_remain_exact_integer_contracts(tmp_path: Path, field: str) -> None:
    manifest_path, manifest, _paths, authority = _case(tmp_path)
    manifest["timing"][field] = float(manifest["timing"][field])
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="exact JSON integer"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_source_mutation_fails_even_when_json_remains_schema_valid(tmp_path: Path) -> None:
    manifest_path, _manifest, paths, authority = _case(tmp_path)
    events = _base_events()
    events[0]["text"] = "silently changed"
    _write_json(paths["timeline"], events)
    with pytest.raises(TranscriptAcceptanceError, match="manifest binding"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_scene_is_bound_to_canonical_source_authority(tmp_path: Path) -> None:
    manifest_path, manifest, _paths, authority = _case(tmp_path)
    # Same scenario number: offset.session alone would not detect this date swap.
    manifest["scene"] = "03_20__4"
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="canonical authority path"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_refreshing_hashes_for_an_arbitrary_source_copy_is_rejected(tmp_path: Path) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    copied = tmp_path / "copied_sources" / "timeline_video.json"
    copied.parent.mkdir(parents=True)
    shutil.copyfile(paths["timeline"], copied)
    _rebind(manifest, "timeline", copied)
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="canonical authority path"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_public_loader_has_no_source_authority_override(tmp_path: Path) -> None:
    manifest_path, _manifest, _paths, authority = _case(tmp_path)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        load_production_transcript_acceptance(  # type: ignore[call-arg]
            manifest_path,
            authority=authority,
        )


def test_cross_date_payload_copy_is_rejected_by_embedded_scene_paths(tmp_path: Path) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    new_scene = "03_20__4"
    new_group, new_scenario = new_scene.rsplit("__", maxsplit=1)
    copied_paths = {
        "timeline": (
            authority.corpus_root / new_group / "transcript" / new_scenario / "timeline_video.json"
        ),
        "offset": (authority.corpus_root / new_group / "transcript" / new_scenario / "offset.json"),
        "reference_wav": (
            authority.corpus_root
            / new_group
            / "Processed"
            / new_scenario
            / "video"
            / "participants_rgb_audio_edited.wav"
        ),
        "hpe_offset": authority.hpe_delivery_root / new_scene / "offset.json",
        "hpe_offset_quality": (authority.hpe_delivery_root / new_scene / "offset_quality.json"),
    }
    for name, target in copied_paths.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths[name], target)
        _rebind(manifest, name, target)
    manifest["scene"] = new_scene
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="paths do not match|canonical"):
        load_transcript_acceptance(manifest_path, authority=authority)


@pytest.mark.parametrize(
    ("location", "value", "pattern"),
    [
        ("format_version", 2.0, "exact JSON integer"),
        ("timecode_fps", 30.0, "exact JSON integer"),
    ],
)
def test_versions_and_fps_reject_float_encoded_integers(
    tmp_path: Path, location: str, value: float, pattern: str
) -> None:
    manifest_path, manifest, _paths, authority = _case(tmp_path)
    if location == "format_version":
        manifest["format_version"] = value
    else:
        manifest["timecode_contract"]["fps_numerator"] = value
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match=pattern):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_declared_exact_legacy_offset_fps_normalization_is_audited(tmp_path: Path) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    offset = _offset()
    offset["fps"] = 30.0
    _write_json(paths["offset"], offset)
    _rebind(manifest, "offset", paths["offset"])
    operation = _legacy_fps_normalization(paths["offset"])
    manifest["source_normalizations"]["operations"] = [operation]
    _rewrite_manifest(manifest_path, manifest)

    accepted = load_transcript_acceptance(manifest_path, authority=authority)

    assert accepted.source_normalizations == (operation,)


def test_undeclared_legacy_offset_fps_float_is_rejected(tmp_path: Path) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    offset = _offset()
    offset["fps"] = 30.0
    _write_json(paths["offset"], offset)
    _rebind(manifest, "offset", paths["offset"])
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="explicitly declared"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_source_normalization_version_is_exact_integer(tmp_path: Path) -> None:
    manifest_path, manifest, _paths, authority = _case(tmp_path)
    manifest["source_normalizations"]["version"] = 1.0
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="exact JSON integer"):
        load_transcript_acceptance(manifest_path, authority=authority)


@pytest.mark.parametrize("fps", [30.1, 29.97, True, "30.0", 30])
def test_legacy_fps_normalization_rejects_every_other_raw_type_or_value(
    tmp_path: Path, fps: Any
) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    offset = _offset()
    offset["fps"] = fps
    _write_json(paths["offset"], offset)
    _rebind(manifest, "offset", paths["offset"])
    manifest["source_normalizations"]["operations"] = [_legacy_fps_normalization(paths["offset"])]
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="raw JSON float 30.0"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_legacy_fps_normalization_requires_exact_token_and_byte_offset(tmp_path: Path) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    offset = _offset()
    offset["fps"] = 30.0
    _write_json(paths["offset"], offset)
    _rebind(manifest, "offset", paths["offset"])
    manifest["source_normalizations"]["operations"] = [
        _legacy_fps_normalization(paths["offset"], byte_offset=0)
    ]
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="exact byte preimage"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_legacy_fps_normalization_rejects_different_float_lexeme(tmp_path: Path) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    offset = _offset()
    offset["fps"] = 30.0
    _write_json(paths["offset"], offset)
    raw = paths["offset"].read_text(encoding="utf-8").replace('"fps": 30.0', '"fps": 30.00')
    paths["offset"].write_text(raw, encoding="utf-8")
    _rebind(manifest, "offset", paths["offset"])
    manifest["source_normalizations"]["operations"] = [_legacy_fps_normalization(paths["offset"])]
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="exact byte preimage"):
        load_transcript_acceptance(manifest_path, authority=authority)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_nonfinite_constants_globally(tmp_path: Path, constant: str) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    raw = paths["hpe_offset_quality"].read_text(encoding="utf-8")
    raw = raw.replace('"error_bound_ms": 33.5', f'"error_bound_ms": {constant}')
    paths["hpe_offset_quality"].write_text(raw, encoding="utf-8")
    _rebind(manifest, "hpe_offset_quality", paths["hpe_offset_quality"])
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="non-finite JSON constant"):
        load_transcript_acceptance(manifest_path, authority=authority)


@pytest.mark.parametrize("value", [True, 1, 1.0, None])
def test_hpe_timebase_values_must_be_exact_nonempty_strings(tmp_path: Path, value: Any) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    quality = _hpe_quality()
    quality["timebase_keys"]["6cam"] = value
    _write_json(paths["hpe_offset_quality"], quality)
    _rebind(manifest, "hpe_offset_quality", paths["hpe_offset_quality"])
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="non-empty string"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_hpe_timebase_nan_is_rejected_before_type_coercion(tmp_path: Path) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    quality = _hpe_quality()
    quality["timebase_keys"]["6cam"] = float("nan")
    _write_json(paths["hpe_offset_quality"], quality)
    _rebind(manifest, "hpe_offset_quality", paths["hpe_offset_quality"])
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="non-finite JSON constant NaN"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_final_source_reverification_closes_validation_to_use_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _manifest, paths, authority = _case(tmp_path)
    original = transcript_acceptance_module._validate_manual_review

    def mutate_after_semantic_validation(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        paths["timeline"].write_bytes(paths["timeline"].read_bytes() + b" ")

    monkeypatch.setattr(
        transcript_acceptance_module,
        "_validate_manual_review",
        mutate_after_semantic_validation,
    )
    with pytest.raises(TranscriptAcceptanceError, match="final re-verification"):
        load_transcript_acceptance(manifest_path, authority=authority)


@pytest.mark.parametrize(
    "timestamps",
    [
        [Decimal("0.00"), Decimal("0.08")],
        [Decimal("0.00"), Decimal("0.03")],
        [Decimal("0.00"), Decimal("NaN")],
    ],
)
def test_rasterization_requires_exact_consecutive_25hz_grid(
    tmp_path: Path, timestamps: list[Decimal]
) -> None:
    manifest_path, _manifest, _paths, authority = _case(tmp_path)
    accepted = load_transcript_acceptance(manifest_path, authority=authority)
    with pytest.raises(TranscriptAcceptanceError):
        rasterize_transcript_activity_reference(accepted, timestamps)


def test_rasterized_reference_cannot_be_confused_with_counted_gold(tmp_path: Path) -> None:
    manifest_path, _manifest, _paths, authority = _case(tmp_path)
    accepted = load_transcript_acceptance(manifest_path, authority=authority)
    values = rasterize_transcript_activity_reference(
        accepted, [Decimal(tick) / 25 for tick in range(8, 20)]
    )
    assert set(values.values) == {"A", "B", "C", "R"}
    assert all(array.dtype == np.int8 for array in values.values.values())
    assert values.output_role == OUTPUT_ROLE
    assert values.counted_gold_ready is False
    assert accepted.counted_gold_ready is False


def test_v2_acceptance_artifact_is_rejected_after_gold_window_upgrade(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, _paths, authority = _case(tmp_path)
    manifest["format_version"] = 2
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="kind/version"):
        load_transcript_acceptance(manifest_path, authority=authority)


def test_formal_acceptance_and_raster_are_strictly_gold_window_scoped(
    tmp_path: Path,
) -> None:
    events = [
        _event("00:00:00:00", "00:00:00:01", "A", "before"),
        _event("00:00:00:00", "00:00:00:03", "A", "cross start"),
        _event("00:00:00:03", "00:00:00:10", "B", "inside"),
        _event("00:00:00:18", "00:00:00:24", "C", "cross end"),
        _event("00:00:01:06", "00:00:01:03", "R", "invalid outside"),
    ]
    manifest_path, manifest, paths, authority = _case(tmp_path, events=events)
    gold_path = paths["gold_alignment"]
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["authorization"]["authorized_tick_range"] = {
        "frame_rate_hz": 25,
        "start_inclusive": 10,
        "end_exclusive": 25,
        "n_ticks": 15,
    }
    _write_json(gold_path, gold)
    _rebind(manifest, "gold_alignment", gold_path)
    _refresh_review_bindings(manifest, paths["timeline"])
    _rewrite_manifest(manifest_path, manifest)

    accepted = load_transcript_acceptance(manifest_path, authority=authority)
    assert [event.source_index for event in accepted.events] == [1, 2, 3]
    assert accepted.scope_summary["source_event_count"] == 5
    assert accepted.scope_summary["in_scope_pre_review_event_count"] == 3
    assert accepted.scope_summary["excluded_event_count"] == 2
    assert accepted.scope_summary["crossing_event_count"] == 2
    assert accepted.scope_summary["accepted_post_review_event_count"] == 3
    assert accepted.speaker_prevalence["source_pre_review"]["R"]["event_count"] == 1
    assert accepted.speaker_prevalence["in_scope_pre_review"]["R"]["event_count"] == 0
    assert accepted.speaker_prevalence["accepted_post_review"]["R"]["event_count"] == 0
    assert accepted.excluded_event_inventory[1]["exclusion_reason"] == (
        "nonpositive_interval_wholly_after_window"
    )

    raster = rasterize_transcript_activity_reference(
        accepted, [Decimal(tick) / 25 for tick in range(10, 25)]
    )
    assert raster.values["A"][0] == 1
    assert raster.values["C"][-1] == 1
    with pytest.raises(TranscriptAcceptanceError, match="gold-alignment window"):
        rasterize_transcript_activity_reference(
            accepted, [Decimal(tick) / 25 for tick in range(9, 25)]
        )
    with pytest.raises(TranscriptAcceptanceError, match="gold-alignment window"):
        rasterize_transcript_activity_reference(
            accepted, [Decimal(tick) / 25 for tick in range(10, 26)]
        )


def test_manual_repair_cannot_promote_an_excluded_event(tmp_path: Path) -> None:
    events = [
        _event("00:00:00:03", "00:00:00:10", "A", "inside"),
        _event("00:00:01:06", "00:00:01:03", "R", "invalid outside"),
    ]
    _manifest_path, manifest, paths, _authority = _case(tmp_path, events=events)
    gold_path = paths["gold_alignment"]
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["authorization"]["authorized_tick_range"] = {
        "frame_rate_hz": 25,
        "start_inclusive": 8,
        "end_exclusive": 25,
        "n_ticks": 17,
    }
    _write_json(gold_path, gold)
    _rebind(manifest, "gold_alignment", gold_path)
    manifest["repairs"] = {
        "operations": [
            {
                "repair_id": "outside-manual",
                "operation": "correct_timecode_boundary",
                "source_index": 1,
                "field": "start_timecode",
                "expected": "00:00:01:06",
                "replacement": "00:00:00:12",
                "classification": "manual_owner",
                "evidence": "attempted out-of-window promotion",
                "review_status": "confirmed",
            }
        ]
    }
    with pytest.raises(TranscriptAcceptanceError, match="outside the authorized window"):
        _refresh_review_bindings(manifest, paths["timeline"])


def test_forged_gold_alignment_gates_fail_even_after_rebinding(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, paths, authority = _case(tmp_path)
    gold_path = paths["gold_alignment"]
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["gates"] = {"alignment_prerequisite_pass": False}
    _write_json(gold_path, gold)
    _rebind(manifest, "gold_alignment", gold_path)
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(TranscriptAcceptanceError, match="full replay|prerequisite"):
        load_transcript_acceptance(manifest_path, authority=authority)
