from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest

import hri_turn_taking.data.audio_contract as audio_contract
from hri_turn_taking.data.audio_contract import (
    AlignmentAuditParameters,
    AudioContractError,
    AudioSourceAuthority,
)

SCENE = "03_20__4"
TEST_PARAMETERS = AlignmentAuditParameters(
    segment_seconds=1,
    minimum_segment_seconds=1,
    search_radius_ms=256,
    minimum_usable_segments=3,
)


def _write_wav(path: Path, values: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype="<i2")
    if array.ndim == 1:
        channels = 1
    else:
        channels = array.shape[1]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(array.tobytes())


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _truncate_tail(path: Path, n_bytes: int) -> None:
    original_size = path.stat().st_size
    assert 0 < n_bytes < original_size
    with path.open("r+b") as handle:
        handle.truncate(original_size - n_bytes)


def _fixture(
    tmp_path: Path,
    *,
    master_order: tuple[str, ...] = ("A", "C", "B", "R"),
    declared_order: tuple[str, ...] | None = ("A", "B", "C", "R"),
    actual_offset_hops: int = 10,
    declared_offset_hops: int | None = None,
) -> tuple[AudioSourceAuthority, dict[str, Path]]:
    authority = AudioSourceAuthority(tmp_path / "corpus", tmp_path / "hpe")
    paths = audio_contract.canonical_source_paths(SCENE, authority)
    hop = 1_536
    n_hops = 200
    n_samples = hop * n_hops
    rng = np.random.default_rng(2317)
    amplitude = rng.uniform(0.25, 0.95, size=n_hops).repeat(hop)
    sample = np.arange(n_samples, dtype=np.float64)
    tracks: dict[str, np.ndarray] = {}
    for index, speaker in enumerate(audio_contract.SPEAKERS):
        carrier = np.sin(2 * np.pi * (350 + index * 70) * sample / 48_000)
        common = np.sin(2 * np.pi * 470 * sample / 48_000)
        signal = amplitude * (0.8 * common + 0.2 * carrier) * (8_000 - index * 500)
        tracks[speaker] = np.rint(signal).clip(-32768, 32767).astype("<i2")
        _write_wav(paths[speaker], tracks[speaker], 48_000)
    master = np.stack([tracks[speaker] for speaker in master_order], axis=1)
    _write_wav(paths["master"], master, 48_000)

    mix = np.rint(master.astype(np.float64).mean(axis=1)).astype("<i2")
    start = actual_offset_hops * hop
    reference = mix[start : n_samples - hop][::3]
    _write_wav(paths["reference"], reference, 16_000)

    alignment_report: dict[str, object] = {
        "offsets": {"A": 0, "B": 0, "C": 0},
        "sample_rate": 48_000,
    }
    if declared_order is not None:
        alignment_report["channels"] = list(declared_order)
    _json_write(paths["alignment_report"], alignment_report)

    lag = actual_offset_hops if declared_offset_hops is None else declared_offset_hops
    offset_sec = lag * 512 / 16_000
    duration = len(reference) / 16_000
    shift = lag * 512
    action = (
        f"crop {shift} samples"
        if shift > 0
        else f"pad {-shift} samples"
        if shift < 0
        else "no shift"
    )
    speaker_alignment = {
        speaker: {
            "src": f"03_20/4/audio/aligned_audio/{speaker}_aligned.wav",
            "dst": f"03_20/4/audio/aligned_audio/{speaker}_edited.wav",
            "sr": 16_000,
            "offset_sec": offset_sec,
            "shift_samples": shift,
            "action": action,
            "output_len_samples": len(reference),
            "output_duration_sec": duration,
        }
        for speaker in audio_contract.SPEAKERS
    }
    _json_write(
        paths["transcript_offset"],
        {
            "session": "4",
            "timeline_reference": "participants_rgb_audio_edited.wav",
            "frame_reference": "participants_rgb_audio_edited.mp4",
            "fps": 30.0,
            "offset_sec": offset_sec,
            "offset_samples_processing_sr": lag * 512,
            "lag_frames_rms": lag,
            "master_sr": 16_000,
            "reference_sr": 16_000,
            "processing_sr": 16_000,
            "reference_duration_sec": duration,
            "interpretation": {
                "offset_positive": "crop beginning of A/B/C/R aligned wavs",
                "offset_negative": "pad silence at beginning of A/B/C/R aligned wavs",
            },
            "speaker_alignment": speaker_alignment,
        },
    )
    _json_write(paths["hpe_offset"], {"streams": {"mic_A": 0.0, "6cam": -1.0}})
    _json_write(
        paths["hpe_offset_quality"],
        {
            "convention": "shared_time = stream_local_time + offset_s",
            "timebase_keys": {
                "mic_A": "A_aligned.wav t=0",
                "6cam": "world_persons.csv",
            },
            "quality": {
                "mic_A": {"offset_s": 0.0, "error_bound_ms": 0.0, "status": "ok"},
                "6cam": {
                    "offset_s": -1.0,
                    "error_bound_ms": 20.0,
                    "status": "ok",
                    "self_check_ok": True,
                },
            },
        },
    )
    return authority, paths


def _audit(authority: AudioSourceAuthority) -> dict[str, object]:
    return audio_contract._audit_scene_with_authority(
        SCENE, authority=authority, parameters=TEST_PARAMETERS
    )


def test_master_identity_comes_from_exact_samples_not_declared_order(tmp_path: Path) -> None:
    authority, _paths = _fixture(tmp_path)
    report = _audit(authority)
    replay = report["master_sample_replay"]
    assert replay["channel_index_to_speaker"] == ["A", "C", "B", "R"]
    assert replay["declaration_status"] == "conflicts_with_sample_replay"
    assert replay["identity_rule"] == "sample_replay_only_never_channel_index_or_report"
    assert report["counted_gold_ready"] is False


def test_master_duplicate_channel_is_not_a_permutation(tmp_path: Path) -> None:
    authority, paths = _fixture(tmp_path)
    with wave.open(str(paths["master"]), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    master = np.frombuffer(raw, dtype="<i2").reshape(-1, 4).copy()
    master[:, 1] = master[:, 0]
    _write_wav(paths["master"], master, 48_000)
    with pytest.raises(AudioContractError, match="duplicated or missing named mono track"):
        _audit(authority)


def test_header_mismatch_is_rejected_before_quality_claim(tmp_path: Path) -> None:
    authority, paths = _fixture(tmp_path)
    with wave.open(str(paths["B"]), "rb") as handle:
        values = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").copy()
    _write_wav(paths["B"], values, 16_000)
    with pytest.raises(AudioContractError, match="sample_rate_hz must equal 48000"):
        _audit(authority)


def test_truncated_reference_payload_cannot_authorize_false_tail_coverage(
    tmp_path: Path,
) -> None:
    authority, paths = _fixture(tmp_path)
    valid_report = audio_contract._audit_scene_with_authority(SCENE, authority=authority)
    manifest = tmp_path / "stale-contract.json"
    _json_write(manifest, valid_report)
    declared_frames = valid_report["sources"]["reference"]["wave_header"]["n_frames"]
    _truncate_tail(paths["reference"], 16_000 * 2)
    assert audio_contract._wave_header(paths["reference"])["n_frames"] == declared_frames
    with pytest.raises(AudioContractError, match="ended before declared n_frames"):
        _audit(authority)
    false_tail_start = (
        float(
            valid_report["analysis_window_contract"]["nonpadding_intersection"]["end_s_exclusive"]
        )
        - 0.5
    )
    false_tail_end = false_tail_start + 0.25
    with pytest.raises(AudioContractError, match="bytes changed|ended before declared"):
        audio_contract._validate_analysis_window_with_authority(
            valid_report,
            false_tail_start,
            false_tail_end,
            authority=authority,
        )
    with pytest.raises(AudioContractError, match="bytes changed|ended before declared"):
        audio_contract._load_audio_contract_with_authority(manifest, authority=authority)


@pytest.mark.parametrize("source_name", ["A", "master", "reference"])
def test_every_wave_consumer_rejects_partial_pcm_frame(tmp_path: Path, source_name: str) -> None:
    authority, paths = _fixture(tmp_path)
    _truncate_tail(paths[source_name], 1)
    with pytest.raises(AudioContractError, match="partial|ended before declared"):
        _audit(authority)


@pytest.mark.parametrize("declared_lag", [11, -10])
def test_wrong_transcript_offset_hop_or_sign_fails_segmented_gate(
    tmp_path: Path, declared_lag: int
) -> None:
    authority, _paths = _fixture(tmp_path, declared_offset_hops=declared_lag)
    report = _audit(authority)
    reference = report["reference_to_equal_weight_master"]
    assert reference["quality_pass"] is False
    assert reference["failure_reasons"]


def test_periodic_envelope_is_rejected_as_ambiguous() -> None:
    periodic = np.tile(np.asarray([0.0, 1.0, 0.0, -1.0]), 500)
    result = audio_contract._segmented_alignment_audit(
        periodic,
        periodic,
        declared_lag_hops=0,
        parameters=AlignmentAuditParameters(
            segment_seconds=2,
            minimum_segment_seconds=1,
            search_radius_ms=256,
            minimum_usable_segments=3,
        ),
    )
    assert result["quality_pass"] is False
    assert "peak_uniqueness_below_limit" in result["failure_reasons"]


def test_time_varying_delay_is_rejected_as_drift() -> None:
    rng = np.random.default_rng(912)
    reference = rng.normal(size=3_000)
    index = np.arange(reference.size)
    delay = np.linspace(0.0, 7.0, reference.size)
    target = np.interp(index - delay, index, reference, left=0.0, right=0.0)
    result = audio_contract._segmented_alignment_audit(
        reference,
        target,
        declared_lag_hops=0,
        parameters=AlignmentAuditParameters(
            segment_seconds=4,
            minimum_segment_seconds=2,
            search_radius_ms=320,
            minimum_usable_segments=3,
            maximum_drift_span_ms=40.0,
        ),
    )
    assert result["quality_pass"] is False
    assert "drift_span_exceeds_limit" in result["failure_reasons"]


def test_strict_source_json_rejects_duplicate_nonfinite_and_bool(tmp_path: Path) -> None:
    authority, paths = _fixture(tmp_path)
    paths["alignment_report"].write_text(
        '{"sample_rate":48000,"sample_rate":48000}', encoding="utf-8"
    )
    with pytest.raises(AudioContractError, match="duplicate JSON key"):
        _audit(authority)

    authority, paths = _fixture(tmp_path / "nan")
    paths["hpe_offset"].write_text('{"streams":{"mic_A":NaN}}', encoding="utf-8")
    with pytest.raises(AudioContractError, match="non-finite"):
        _audit(authority)

    authority, paths = _fixture(tmp_path / "bool")
    _json_write(paths["hpe_offset"], {"streams": {"mic_A": False}})
    with pytest.raises(AudioContractError, match="JSON number"):
        _audit(authority)


def test_legacy_transcript_fps_allows_only_exact_float_30(tmp_path: Path) -> None:
    authority, paths = _fixture(tmp_path)
    offset = json.loads(paths["transcript_offset"].read_text(encoding="utf-8"))
    offset["fps"] = 30
    _json_write(paths["transcript_offset"], offset)
    with pytest.raises(AudioContractError, match="exact JSON float 30.0"):
        _audit(authority)


def test_hpe_exact_anchor_bundle_is_replayed_and_hash_mismatch_rejected(
    tmp_path: Path,
) -> None:
    authority, paths = _fixture(tmp_path)
    quality = json.loads(paths["hpe_offset_quality"].read_text(encoding="utf-8"))
    with wave.open(str(paths["A"]), "rb") as handle:
        frames = handle.getnframes()
    quality["rgb_d3_source_bundle"] = {
        "gate_version": "test_exact_anchor_v1",
        "delivery": SCENE,
        "anchor": {
            "path": str(paths["A"]),
            "sha256": audio_contract._sha256_file(paths["A"]),
            "sample_rate_hz": 48_000,
            "samples": frames,
            "duration_s": frames / 48_000,
        },
        "streams": {},
    }
    _json_write(paths["hpe_offset_quality"], quality)
    report = _audit(authority)
    assert report["hpe_mic_a_anchor"]["upstream_evidence_strength"] == ("exact_path_hash_header")
    quality["rgb_d3_source_bundle"]["anchor"]["sha256"] = "0" * 64
    _json_write(paths["hpe_offset_quality"], quality)
    with pytest.raises(AudioContractError, match="anchor hash disagrees"):
        _audit(authority)


@pytest.mark.parametrize("bad_version", [1.0, True])
def test_contract_version_requires_exact_integer(tmp_path: Path, bad_version: object) -> None:
    authority, _paths = _fixture(tmp_path)
    report = _audit(authority)
    report["format_version"] = bad_version
    manifest = tmp_path / "bad.json"
    _json_write(manifest, report)
    with pytest.raises(AudioContractError, match="exact JSON integer"):
        audio_contract._load_audio_contract_with_authority(manifest, authority=authority)


def test_bound_source_mutation_is_detected_by_final_rehash(tmp_path: Path) -> None:
    authority, paths = _fixture(tmp_path)
    report = _audit(authority)
    with paths["A"].open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(AudioContractError, match="bytes changed"):
        audio_contract._verify_sources_from_report(report, authority=authority)


def test_audit_closes_toctou_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    authority, paths = _fixture(tmp_path)
    original = audio_contract._verify_sources_from_report
    mutated = False

    def mutate_before_final_verify(
        report: dict[str, object], *, authority: AudioSourceAuthority
    ) -> None:
        nonlocal mutated
        if not mutated:
            with paths["alignment_report"].open("ab") as handle:
                handle.write(b" ")
            mutated = True
        original(report, authority=authority)

    monkeypatch.setattr(audio_contract, "_verify_sources_from_report", mutate_before_final_verify)
    with pytest.raises(AudioContractError, match="bytes changed"):
        _audit(authority)


def test_analysis_window_requires_real_half_open_coverage(tmp_path: Path) -> None:
    authority, _paths = _fixture(tmp_path)
    report = audio_contract._audit_scene_with_authority(SCENE, authority=authority)
    coverage = report["analysis_window_contract"]["nonpadding_intersection"]
    start = coverage["start_s"]
    end = coverage["end_s_exclusive"]
    accepted = audio_contract._validate_analysis_window_with_authority(
        report, start, end, authority=authority
    )
    assert accepted == {"start_s": start, "end_s_exclusive": end}
    with pytest.raises(AudioContractError, match="lacks real audio coverage"):
        audio_contract._validate_analysis_window_with_authority(
            report, 0.0, end, authority=authority
        )
    with pytest.raises(AudioContractError, match="positive half-open"):
        audio_contract._validate_analysis_window_with_authority(
            report, start, start, authority=authority
        )
    with pytest.raises(AudioContractError, match="must not be bool"):
        audio_contract._validate_analysis_window_with_authority(
            report, True, end, authority=authority
        )


def test_analysis_window_rejects_tampered_coverage_even_with_unchanged_sources(
    tmp_path: Path,
) -> None:
    authority, _paths = _fixture(tmp_path)
    report = audio_contract._audit_scene_with_authority(SCENE, authority=authority)
    tampered = json.loads(json.dumps(report))
    tampered["analysis_window_contract"]["nonpadding_intersection"]["start_s"] = "-999"
    tampered["analysis_window_contract"]["nonpadding_intersection"]["end_s_exclusive"] = "999"
    with pytest.raises(AudioContractError, match="fresh full audio replay"):
        audio_contract._validate_analysis_window_with_authority(
            tampered, -100.0, 100.0, authority=authority
        )


def test_contract_loader_replays_sources_and_rejects_path_substitution(tmp_path: Path) -> None:
    authority, _paths = _fixture(tmp_path)
    report = audio_contract._audit_scene_with_authority(SCENE, authority=authority)
    manifest = tmp_path / "contract.json"
    _json_write(manifest, report)
    loaded = audio_contract._load_audio_contract_with_authority(manifest, authority=authority)
    assert loaded["gates"]["counted_gold_ready"] is False
    copied = tmp_path / "copied_A.wav"
    copied.write_bytes(Path(report["sources"]["A"]["path"]).read_bytes())
    report["sources"]["A"]["path"] = str(copied)
    _json_write(manifest, report)
    with pytest.raises(AudioContractError, match="not canonical"):
        audio_contract._load_audio_contract_with_authority(manifest, authority=authority)


def test_no_clobber_writer_preserves_existing_file(tmp_path: Path) -> None:
    authority, _paths = _fixture(tmp_path)
    destination = tmp_path / "published.json"
    destination.write_bytes(b"owner data")
    with pytest.raises(AudioContractError, match="refusing to overwrite"):
        audio_contract._write_audio_contract_with_authority(SCENE, destination, authority=authority)
    assert destination.read_bytes() == b"owner data"


def test_writer_publishes_one_replayable_contract_without_temp_leak(tmp_path: Path) -> None:
    authority, _paths = _fixture(tmp_path)
    destination = tmp_path / "published.json"
    result = audio_contract._write_audio_contract_with_authority(
        SCENE, destination, authority=authority
    )
    assert result == destination.resolve()
    loaded = audio_contract._load_audio_contract_with_authority(destination, authority=authority)
    assert loaded["artifact_kind"] == audio_contract.ARTIFACT_KIND
    assert not list(tmp_path.glob(".published.json.tmp.*"))


def test_writer_rolls_back_its_link_on_base_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority, _paths = _fixture(tmp_path)
    destination = tmp_path / "published.json"

    def interrupt(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(audio_contract, "_load_audio_contract_with_authority", interrupt)
    with pytest.raises(KeyboardInterrupt):
        audio_contract._write_audio_contract_with_authority(SCENE, destination, authority=authority)
    assert not destination.exists()
    assert not list(tmp_path.glob(".published.json.tmp.*"))


def test_writer_does_not_delete_concurrent_unowned_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority, _paths = _fixture(tmp_path)
    destination = tmp_path / "published.json"

    def concurrent_link(_source: Path, target: Path) -> None:
        Path(target).write_bytes(b"concurrent owner")
        raise FileExistsError

    monkeypatch.setattr(audio_contract.os, "link", concurrent_link)
    with pytest.raises(FileExistsError):
        audio_contract._write_audio_contract_with_authority(SCENE, destination, authority=authority)
    assert destination.read_bytes() == b"concurrent owner"


def test_writer_rehashes_sources_immediately_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority, paths = _fixture(tmp_path)
    destination = tmp_path / "published.json"
    original = audio_contract._verify_sources_from_report
    calls = 0

    def mutate_on_prepublish(report: dict[str, object], *, authority: AudioSourceAuthority) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            with paths["alignment_report"].open("ab") as handle:
                handle.write(b" ")
        original(report, authority=authority)

    monkeypatch.setattr(audio_contract, "_verify_sources_from_report", mutate_on_prepublish)
    with pytest.raises(AudioContractError, match="bytes changed"):
        audio_contract._write_audio_contract_with_authority(SCENE, destination, authority=authority)
    assert not destination.exists()


def test_inter_lavalier_report_proposes_but_never_applies_correction(tmp_path: Path) -> None:
    authority, paths = _fixture(tmp_path)
    before = {speaker: audio_contract._sha256_file(paths[speaker]) for speaker in ("B", "C", "R")}
    report = _audit(authority)
    assert set(report["inter_lavalier_to_mic_a"]) == {"B", "C", "R"}
    for result in report["inter_lavalier_to_mic_a"].values():
        assert "proposed_target_correction_ms" in result
    after = {speaker: audio_contract._sha256_file(paths[speaker]) for speaker in ("B", "C", "R")}
    assert after == before
    assert report["gates"]["counted_gold_ready"] is False
    assert "not a gold-VAD" in report["gates"]["counted_gold_blocker"]


def test_nonfinite_contract_json_is_rejected(tmp_path: Path) -> None:
    authority, _paths = _fixture(tmp_path)
    manifest = tmp_path / "nan-contract.json"
    manifest.write_text('{"format_version":1,"x":NaN}', encoding="utf-8")
    with pytest.raises(AudioContractError, match="non-finite"):
        audio_contract._load_audio_contract_with_authority(manifest, authority=authority)


def test_synthetic_fixture_has_finite_sync_metrics(tmp_path: Path) -> None:
    authority, _paths = _fixture(tmp_path)
    report = _audit(authority)
    metric = report["reference_to_equal_weight_master"]["median_residual_ms"]
    assert metric is not None and math.isfinite(metric)
