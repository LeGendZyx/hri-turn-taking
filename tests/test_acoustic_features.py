from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import firwin, freqz

from hri_turn_taking.data import acoustic_features
from hri_turn_taking.data.acoustic_features import (
    AcousticFeatureError,
    build_acoustic_evidence,
    causal_resample_to_16khz,
    extract_causal_acoustic_features,
    load_acoustic_evidence,
)


def _tone(sample_rate: int, duration_s: float, frequency_hz: float = 200.0) -> np.ndarray:
    count = round(sample_rate * duration_s)
    time = np.arange(count, dtype=np.float64) / sample_rate
    return 0.5 * np.sin(2.0 * np.pi * frequency_hz * time)


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int = 16_000) -> None:
    clipped = np.clip(np.asarray(pcm, dtype=np.float64), -1.0, 1.0)
    int16 = np.rint(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(int16.tobytes())


def _rewrite_metadata(source: Path, destination: Path, update: object) -> None:
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(arrays["metadata_json"].tobytes().decode("utf-8"))
    update(metadata)  # type: ignore[operator]
    arrays["metadata_json"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    )
    with destination.open("wb") as handle:
        np.savez(handle, **arrays)


def test_frozen_parameters_are_explicit_and_versioned() -> None:
    parameters = acoustic_features.extractor_parameters()

    assert acoustic_features.FORMAT_VERSION == 1
    assert parameters["target_sample_rate_hz"] == 16_000
    assert parameters["frame_rate_hz"] == 25
    assert set(parameters["runtime_versions"]) == {"numpy", "scipy"}
    assert parameters["energy"]["window_ms"] == 40
    assert parameters["pitch"]["window_ms"] == 80
    assert parameters["pitch"]["minimum_hz"] == 50
    assert parameters["pitch"]["maximum_hz"] == 500
    assert parameters["pitch"]["voicing_threshold_inclusive"] == 0.45
    assert parameters["pitch"]["unvoiced_f0_proxy"] == 0.0
    assert parameters["resampler"]["minimum_stopband_attenuation_db"] == 80.0
    assert parameters["resampler"]["group_delay_ms"] == 2.0
    assert parameters["resampler"]["group_delay_policy"].startswith("not_compensated")


@pytest.mark.parametrize("factor", range(2, 13))
def test_causal_resampler_stopband_and_group_delay_are_bounded(factor: int) -> None:
    source_rate = acoustic_features.TARGET_SAMPLE_RATE_HZ * factor
    tap_count = acoustic_features.RESAMPLE_TAPS_PER_FACTOR * factor + 1
    taps = firwin(
        tap_count,
        cutoff=(acoustic_features.TARGET_SAMPLE_RATE_HZ / 2)
        * acoustic_features.RESAMPLE_CUTOFF_NYQUIST_RATIO,
        window=("kaiser", acoustic_features.RESAMPLE_KAISER_BETA),
        fs=source_rate,
        pass_zero="lowpass",
        scale=True,
    )
    frequencies, response = freqz(taps, worN=131_072, fs=source_rate)
    maximum_stopband = float(
        np.max(np.abs(response[frequencies >= acoustic_features.RESAMPLE_STOPBAND_START_HZ]))
    )
    attenuation_db = -20.0 * math.log10(max(maximum_stopband, np.finfo(float).tiny))
    group_delay_ms = ((tap_count - 1) / 2) / source_rate * 1_000.0

    assert attenuation_db >= acoustic_features.RESAMPLE_MINIMUM_STOPBAND_ATTENUATION_DB
    assert group_delay_ms == pytest.approx(acoustic_features.RESAMPLE_GROUP_DELAY_MS)


def test_direct_16khz_tone_has_nullable_start_then_expected_pitch() -> None:
    evidence = extract_causal_acoustic_features(_tone(16_000, 1.0), 16_000)

    assert evidence.tick_index[0] == 0
    assert not evidence.energy_valid[0]
    assert np.isnan(evidence.array_energy[0])
    assert evidence.energy_valid[1]
    assert not evidence.pitch_valid[1]
    assert np.isnan(evidence.f0_proxy[1])
    assert np.isnan(evidence.voiced[1])
    assert evidence.pitch_valid[2:].all()
    assert np.all(evidence.voiced[2:] == 1.0)
    assert np.allclose(evidence.f0_proxy[2:], math.log2(200.0 / 100.0), atol=1e-12)
    assert np.array_equal(evidence.model_input_valid, evidence.energy_valid & evidence.pitch_valid)


def test_valid_silence_uses_floor_and_unvoiced_zero_without_collision() -> None:
    evidence = extract_causal_acoustic_features(np.zeros(16_000), 16_000)

    assert evidence.pitch_valid[2:].all()
    assert np.all(evidence.voiced[2:] == 0.0)
    assert np.all(evidence.f0_proxy[2:] == 0.0)
    assert np.allclose(
        evidence.array_energy[evidence.energy_valid],
        math.log(acoustic_features.ENERGY_FLOOR),
    )


def test_exact_right_edge_includes_last_completed_sample_but_excludes_future() -> None:
    rng = np.random.default_rng(7)
    original = rng.normal(0.0, 0.1, 16_000)
    cutoff_tick = 10
    cutoff_sample = cutoff_tick * acoustic_features.SAMPLES_PER_TICK
    ticks = list(range(1, cutoff_tick + 1))

    future_changed = original.copy()
    future_changed[cutoff_sample:] = 0.9
    baseline = extract_causal_acoustic_features(original, 16_000, tick_indices=ticks)
    future = extract_causal_acoustic_features(future_changed, 16_000, tick_indices=ticks)
    for name in acoustic_features._ARRAY_NAMES:
        assert np.array_equal(
            getattr(baseline, name), getattr(future, name), equal_nan=True
        ), name

    boundary_changed = original.copy()
    boundary_changed[cutoff_sample - 1] = 1.0
    boundary = extract_causal_acoustic_features(
        boundary_changed, 16_000, tick_indices=[cutoff_tick]
    )
    assert boundary.array_energy[0] != baseline.array_energy[-1]


def test_48khz_resampler_is_causal_at_source_sample_boundary() -> None:
    rng = np.random.default_rng(8)
    original = rng.normal(0.0, 0.08, 48_000)
    cutoff_tick = 10
    source_cutoff = cutoff_tick * (48_000 // 25)
    ticks = list(range(1, cutoff_tick + 1))

    future_changed = original.copy()
    future_changed[source_cutoff:] = -0.8
    baseline = extract_causal_acoustic_features(original, 48_000, tick_indices=ticks)
    future = extract_causal_acoustic_features(future_changed, 48_000, tick_indices=ticks)
    for name in acoustic_features._ARRAY_NAMES:
        assert np.array_equal(
            getattr(baseline, name), getattr(future, name), equal_nan=True
        ), name

    boundary_changed = original.copy()
    boundary_changed[source_cutoff - 1] = 1.0
    boundary = extract_causal_acoustic_features(
        boundary_changed, 48_000, tick_indices=[cutoff_tick]
    )
    assert boundary.array_energy[0] != baseline.array_energy[-1]


def test_resampler_warmup_is_invalid_not_zero_padded_audio() -> None:
    evidence = extract_causal_acoustic_features(
        _tone(48_000, 1.0), 48_000, tick_indices=[1, 2, 3]
    )

    assert evidence.energy_valid.tolist() == [False, True, True]
    assert evidence.pitch_valid.tolist() == [False, False, True]
    assert np.isnan(evidence.array_energy[0])
    assert np.isnan(evidence.f0_proxy[:2]).all()
    assert evidence.voiced[2] == 1.0


@pytest.mark.parametrize("frequency", [50.0, 100.0, 200.0, 500.0])
def test_pitch_range_endpoints_and_interior(frequency: float) -> None:
    evidence = extract_causal_acoustic_features(
        _tone(16_000, 0.5, frequency), 16_000, tick_indices=[3]
    )
    assert evidence.pitch_valid[0]
    assert evidence.voiced[0] == 1.0
    estimated = 100.0 * (2.0 ** evidence.f0_proxy[0])
    assert estimated == pytest.approx(frequency, rel=0.02)


@pytest.mark.parametrize(
    ("pcm", "rate", "message"),
    [
        (np.array([0.0, np.nan]), 16_000, "NaN"),
        (np.zeros((2, 8)), 16_000, "one-dimensional"),
        (np.zeros(100), 44_100, "multiple of 16000"),
        (np.zeros(100), 208_000, "maximum"),
        (np.array([0, 1], dtype=np.int32), 16_000, "signed 16-bit"),
        (np.array([0.0 + 0.0j, 0.1 + 0.2j]), 16_000, "numeric"),
        (np.array([0.0, 1.1]), 16_000, "normalized"),
    ],
)
def test_bad_pcm_or_rate_fails_closed(pcm: np.ndarray, rate: int, message: str) -> None:
    with pytest.raises(AcousticFeatureError, match=message):
        causal_resample_to_16khz(pcm, rate)


@pytest.mark.parametrize("ticks", [[True], [1.0], [1, 1], [2, 1], [-1]])
def test_bad_tick_contract_fails_closed(ticks: list[object]) -> None:
    with pytest.raises(AcousticFeatureError, match="tick_indices"):
        extract_causal_acoustic_features(np.zeros(16_000), 16_000, tick_indices=ticks)


def test_numpy_integer_ticks_are_accepted_without_float_coercion() -> None:
    evidence = extract_causal_acoustic_features(
        np.zeros(16_000), 16_000, tick_indices=np.array([1, 2], dtype=np.int64)
    )
    assert evidence.tick_index.tolist() == [1, 2]


def test_candidate_artifact_round_trips_but_cannot_claim_readiness(tmp_path: Path) -> None:
    wav_path = tmp_path / "candidate.wav"
    output = tmp_path / "evidence.npz"
    _write_wav(wav_path, _tone(16_000, 0.5))

    built = build_acoustic_evidence(
        wav_path,
        output,
        candidate_label="synthetic_test_candidate",
    )
    loaded = load_acoustic_evidence(output)

    assert built.metadata == loaded.metadata
    assert loaded.metadata["artifact_kind"] == acoustic_features.ARTIFACT_KIND
    assert loaded.metadata["output_role"] == acoustic_features.OUTPUT_ROLE
    assert loaded.metadata["assembly_eligible"] is False
    assert loaded.metadata["counted_input_ready"] is False
    assert loaded.metadata["source_selection"]["status"] == "unapproved_candidate"
    assert not list(tmp_path.glob(".evidence.npz.tmp.*"))


def test_source_hash_is_rechecked_on_every_load(tmp_path: Path) -> None:
    wav_path = tmp_path / "candidate.wav"
    output = tmp_path / "evidence.npz"
    _write_wav(wav_path, _tone(16_000, 0.5))
    build_acoustic_evidence(wav_path, output, candidate_label="candidate")

    _write_wav(wav_path, _tone(16_000, 0.5, 250.0))

    with pytest.raises(AcousticFeatureError, match="source changed"):
        load_acoustic_evidence(output)


def test_source_replacement_during_replay_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = tmp_path / "candidate.wav"
    output = tmp_path / "evidence.npz"
    _write_wav(wav_path, _tone(16_000, 0.5))
    build_acoustic_evidence(wav_path, output, candidate_label="candidate")
    real_extract = acoustic_features.extract_causal_acoustic_features
    replaced = False

    def replace_after_extract(*args: object, **kwargs: object) -> object:
        nonlocal replaced
        result = real_extract(*args, **kwargs)
        if not replaced:
            replaced = True
            _write_wav(wav_path, _tone(16_000, 0.5, 250.0))
        return result

    monkeypatch.setattr(
        acoustic_features,
        "extract_causal_acoustic_features",
        replace_after_extract,
    )
    with pytest.raises(AcousticFeatureError, match="changed during"):
        load_acoustic_evidence(output)


@pytest.mark.parametrize("bad_version", [True, 1.0, 2])
def test_artifact_version_requires_exact_supported_integer(
    tmp_path: Path, bad_version: object
) -> None:
    wav_path = tmp_path / "candidate.wav"
    original = tmp_path / "evidence.npz"
    tampered = tmp_path / "tampered.npz"
    _write_wav(wav_path, _tone(16_000, 0.5))
    build_acoustic_evidence(wav_path, original, candidate_label="candidate")
    _rewrite_metadata(
        original,
        tampered,
        lambda metadata: metadata.__setitem__("format_version", bad_version),
    )

    with pytest.raises(AcousticFeatureError, match="format_version"):
        load_acoustic_evidence(tampered)


def test_extractor_metadata_uses_recursive_type_exact_comparison(tmp_path: Path) -> None:
    wav_path = tmp_path / "candidate.wav"
    original = tmp_path / "evidence.npz"
    tampered = tmp_path / "tampered.npz"
    _write_wav(wav_path, _tone(16_000, 0.5))
    build_acoustic_evidence(wav_path, original, candidate_label="candidate")

    def mutate(metadata: dict[str, object]) -> None:
        extractor = metadata["extractor"]
        assert isinstance(extractor, dict)
        extractor["frame_rate_hz"] = 25.0

    _rewrite_metadata(original, tampered, mutate)
    with pytest.raises(AcousticFeatureError, match="extractor parameters"):
        load_acoustic_evidence(tampered)


def test_stored_feature_tamper_is_rejected_by_array_hash(tmp_path: Path) -> None:
    wav_path = tmp_path / "candidate.wav"
    original = tmp_path / "evidence.npz"
    tampered = tmp_path / "tampered.npz"
    _write_wav(wav_path, _tone(16_000, 0.5))
    build_acoustic_evidence(wav_path, original, candidate_label="candidate")
    with np.load(original, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["array_energy"] = arrays["array_energy"].copy()
    arrays["array_energy"][2] += 1.0
    with tampered.open("wb") as handle:
        np.savez(handle, **arrays)

    with pytest.raises(AcousticFeatureError, match="arrays.array_energy"):
        load_acoustic_evidence(tampered)


def test_truncated_wav_payload_fails_as_short_read(tmp_path: Path) -> None:
    wav_path = tmp_path / "truncated.wav"
    output = tmp_path / "evidence.npz"
    _write_wav(wav_path, _tone(16_000, 0.5))
    raw = wav_path.read_bytes()
    wav_path.write_bytes(raw[:-200])

    with pytest.raises(AcousticFeatureError, match="ended before|complete mono"):
        build_acoustic_evidence(wav_path, output, candidate_label="candidate")
    assert not output.exists()


def test_too_short_source_cannot_publish_all_invalid_evidence(tmp_path: Path) -> None:
    wav_path = tmp_path / "short.wav"
    output = tmp_path / "evidence.npz"
    _write_wav(wav_path, _tone(16_000, 0.07))

    with pytest.raises(AcousticFeatureError, match="complete real 80 ms history"):
        build_acoustic_evidence(wav_path, output, candidate_label="candidate")
    assert not output.exists()


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    wav_path = tmp_path / "candidate.wav"
    output = tmp_path / "evidence.npz"
    _write_wav(wav_path, _tone(16_000, 0.5))
    output.write_bytes(b"owner data")

    with pytest.raises(AcousticFeatureError, match="refusing to overwrite"):
        build_acoustic_evidence(wav_path, output, candidate_label="candidate")
    assert output.read_bytes() == b"owner data"


def test_concurrent_destination_creation_wins_without_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = tmp_path / "candidate.wav"
    output = tmp_path / "evidence.npz"
    _write_wav(wav_path, _tone(16_000, 0.5))
    real_link = acoustic_features.os.link

    def race(stage: Path, destination: Path) -> None:
        destination.write_bytes(b"racer")
        real_link(stage, destination)

    monkeypatch.setattr(acoustic_features.os, "link", race)
    with pytest.raises(AcousticFeatureError, match="refusing to overwrite"):
        build_acoustic_evidence(wav_path, output, candidate_label="candidate")
    assert output.read_bytes() == b"racer"
    assert not list(tmp_path.glob(".evidence.npz.tmp.*"))


def test_post_publish_failure_rolls_back_only_own_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = tmp_path / "candidate.wav"
    output = tmp_path / "evidence.npz"
    _write_wav(wav_path, _tone(16_000, 0.5))
    real_loader = acoustic_features.load_acoustic_evidence
    calls = 0

    def fail_second(path: str | Path) -> acoustic_features.LoadedAcousticEvidence:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("post-publish interruption")
        return real_loader(path)

    monkeypatch.setattr(acoustic_features, "load_acoustic_evidence", fail_second)
    with pytest.raises(KeyboardInterrupt):
        build_acoustic_evidence(wav_path, output, candidate_label="candidate")
    assert not output.exists()
    assert not list(tmp_path.glob(".evidence.npz.tmp.*"))


def test_link_then_interrupt_rolls_back_own_published_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = tmp_path / "candidate.wav"
    output = tmp_path / "evidence.npz"
    _write_wav(wav_path, _tone(16_000, 0.5))
    real_link = acoustic_features.os.link

    def link_then_interrupt(stage: Path, destination: Path) -> None:
        real_link(stage, destination)
        raise KeyboardInterrupt("interrupted after atomic link")

    monkeypatch.setattr(acoustic_features.os, "link", link_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        build_acoustic_evidence(wav_path, output, candidate_label="candidate")
    assert not output.exists()
    assert not list(tmp_path.glob(".evidence.npz.tmp.*"))
