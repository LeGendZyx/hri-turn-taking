from __future__ import annotations

import copy
import hashlib
import inspect
import json
import runpy
import wave
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hri_turn_taking.data import acoustic_features, analysis_base, assembly, audio_contract
from hri_turn_taking.data import real_acoustic_features as real_acoustics

SCENE = "03_20__4"


def _sha256(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _write_master(
    path: Path,
    frames: np.ndarray,
    *,
    sample_rate_hz: int = real_acoustics.MASTER_SAMPLE_RATE_HZ,
    channels: int = real_acoustics.MASTER_CHANNELS,
    sample_width_bytes: int = real_acoustics.MASTER_SAMPLE_WIDTH_BYTES,
) -> None:
    values = np.asarray(frames)
    assert values.ndim == 2
    assert values.shape[1] == channels
    assert values.dtype == np.dtype("<i2") or values.dtype == np.dtype("int16")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width_bytes)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(values.astype("<i2", copy=False).tobytes(order="C"))


def _master_binding(
    path: Path,
    *,
    n_frames: int,
    channels: int = real_acoustics.MASTER_CHANNELS,
    sample_width_bytes: int = real_acoustics.MASTER_SAMPLE_WIDTH_BYTES,
    sample_rate_hz: int = real_acoustics.MASTER_SAMPLE_RATE_HZ,
    compression_type: str = "NONE",
) -> dict[str, Any]:
    digest, size = _sha256(path)
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": size,
        "wave_header": {
            "channels": channels,
            "sample_width_bytes": sample_width_bytes,
            "sample_rate_hz": sample_rate_hz,
            "compression_type": compression_type,
            "n_frames": n_frames,
        },
    }


def _dataset(
    tmp_path: Path,
    *,
    scene: str = SCENE,
    start: int = 3,
    end: int = 6,
    frame_scene: str | None = None,
    timestamp_delta: float = 0.0,
    provenance_scene: str | None = None,
    provenance_range: tuple[int, int] | None = None,
) -> assembly._DatasetFile:
    path = tmp_path / f"{scene}.csv"
    path.write_text("synthetic analysis base fixture\n", encoding="utf-8")
    ticks = np.arange(start, end, dtype=np.int64)
    chosen_range = (start, end) if provenance_range is None else provenance_range
    provenance = {
        "artifact_kind": analysis_base.ARTIFACT_KIND,
        "session_id": scene if provenance_scene is None else provenance_scene,
        "participant_to_slot": {"R": "P1", "A": "P2", "B": "P3", "C": "P4"},
        "output_tick_range": {
            "start_inclusive": chosen_range[0],
            "end_exclusive": chosen_range[1],
        },
    }
    frame = pd.DataFrame(
        {
            "session_id": np.full(
                ticks.size,
                scene if frame_scene is None else frame_scene,
                dtype=object,
            ),
            "timestamp_s": ticks.astype(np.float64) / real_acoustics.FRAME_RATE_HZ
            + timestamp_delta,
        }
    )
    return assembly._DatasetFile(
        path=path,
        frame_df=frame,
        provenance=provenance,
        data_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )


def _evidence(
    *,
    ticks: np.ndarray | None = None,
    array_energy: np.ndarray | None = None,
    energy_valid: np.ndarray | None = None,
    f0_proxy: np.ndarray | None = None,
    voiced: np.ndarray | None = None,
    pitch_valid: np.ndarray | None = None,
) -> acoustic_features.AcousticFeatureEvidence:
    selected_ticks = (
        np.array([3, 4, 5], dtype=np.int64)
        if ticks is None
        else np.asarray(ticks, dtype=np.int64)
    )
    count = selected_ticks.size
    return acoustic_features.AcousticFeatureEvidence(
        tick_index=selected_ticks,
        source_local_timestamp_s=selected_ticks.astype(np.float64) / 25.0,
        array_energy=(
            np.array([-5.0, -4.0, -3.0], dtype=np.float64)
            if array_energy is None
            else np.asarray(array_energy)
        ),
        energy_valid=(
            np.ones(count, dtype=bool)
            if energy_valid is None
            else np.asarray(energy_valid)
        ),
        f0_proxy=(
            np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if f0_proxy is None
            else np.asarray(f0_proxy)
        ),
        voiced=(
            np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if voiced is None
            else np.asarray(voiced)
        ),
        pitch_valid=(
            np.ones(count, dtype=bool)
            if pitch_valid is None
            else np.asarray(pitch_valid)
        ),
    )


def test_equal_weight_master_uses_exact_float64_mean_before_division(
    tmp_path: Path,
) -> None:
    frames = np.array(
        [
            [-32_768, -32_768, -32_768, -32_768],
            [32_767, 32_767, 32_767, 32_767],
            [1, 0, 0, 0],
            [-32_768, 32_767, 1, 0],
            [32_767, -32_768, -32_768, 32_767],
        ],
        dtype="<i2",
    )
    original = tmp_path / "master.wav"
    permuted = tmp_path / "master_permuted.wav"
    _write_master(original, frames)
    _write_master(permuted, frames[:, [3, 1, 0, 2]])

    pcm, replay = real_acoustics._read_equal_weight_master(
        original,
        binding=_master_binding(original, n_frames=len(frames)),
    )
    permuted_pcm, _ = real_acoustics._read_equal_weight_master(
        permuted,
        binding=_master_binding(permuted, n_frames=len(frames)),
    )
    expected = frames.astype(np.float64).mean(axis=1) / 32_768.0

    assert np.array_equal(pcm, expected)
    assert np.array_equal(permuted_pcm, expected)
    assert pcm[2] == 1.0 / (4 * 32_768)
    assert pcm.dtype == np.float64
    assert replay["input_dtype"] == "little_endian_signed_pcm16"
    assert replay["mix_arithmetic_dtype"] == "float64"
    assert replay["intermediate_pcm16_quantization"] is False
    assert replay["channel_order_invariant"] is True


def test_master_hash_mismatch_fails_before_decode(tmp_path: Path) -> None:
    path = tmp_path / "master.wav"
    frames = np.zeros((4, 4), dtype="<i2")
    _write_master(path, frames)
    binding = _master_binding(path, n_frames=len(frames))
    binding["sha256"] = "0" * 64

    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="bytes differ from the fully replayed audio contract",
    ):
        real_acoustics._read_equal_weight_master(path, binding=binding)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("channels", 3, "channels must equal 4"),
        ("channels", 4.0, "channels must equal 4"),
        ("sample_width_bytes", 1, "sample_width_bytes must equal 2"),
        ("sample_width_bytes", 2.0, "sample_width_bytes must equal 2"),
        ("sample_rate_hz", 16_000, "sample_rate_hz must equal 48000"),
        ("sample_rate_hz", 48_000.0, "sample_rate_hz must equal 48000"),
        ("compression_type", "ULAW", "compression_type must equal 'NONE'"),
        ("n_frames", True, "n_frames must be an exact integer"),
        ("n_frames", 4.0, "n_frames must be an exact integer"),
    ],
)
def test_master_declared_header_must_match_frozen_exact_types(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = tmp_path / "master.wav"
    frames = np.zeros((4, 4), dtype="<i2")
    _write_master(path, frames)
    binding = _master_binding(path, n_frames=len(frames))
    binding["wave_header"][field] = value

    with pytest.raises(real_acoustics.RealAcousticFeatureError, match=message):
        real_acoustics._read_equal_weight_master(path, binding=binding)


def test_master_actual_header_change_is_rejected_even_with_current_byte_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "three_channel.wav"
    frames = np.zeros((4, 3), dtype="<i2")
    _write_master(path, frames, channels=3)
    binding = _master_binding(path, n_frames=len(frames))

    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="WAV header changed after the audio-contract replay",
    ):
        real_acoustics._read_equal_weight_master(path, binding=binding)


def test_truncated_master_payload_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "master.wav"
    frames = np.arange(32, dtype=np.int16).reshape(8, 4).astype("<i2")
    _write_master(path, frames)
    path.write_bytes(path.read_bytes()[:-4])
    binding = _master_binding(path, n_frames=len(frames))

    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="partial multi-channel|unexpected PCM frame count|ended before",
    ):
        real_acoustics._read_equal_weight_master(path, binding=binding)


def test_analysis_base_accepts_only_exact_frozen_tick_session_and_time_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis_base, "EXPECTED_TICK_RANGES", {SCENE: (3, 6)})
    dataset = _dataset(tmp_path)

    ticks = real_acoustics._base_ticks(dataset, SCENE)

    assert np.array_equal(ticks, np.array([3, 4, 5], dtype=np.int64))


@pytest.mark.parametrize(
    ("dataset_kwargs", "message"),
    [
        ({"provenance_range": (2, 6)}, "ticks differ"),
        ({"provenance_scene": "03_20__5"}, "session_id does not match"),
        ({"frame_scene": "03_20__5"}, "session keys are not constant and canonical"),
        ({"timestamp_delta": 1e-12}, "timestamps do not exactly equal integer tick / 25"),
    ],
)
def test_analysis_base_rejects_shifted_tick_session_or_timestamp_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_kwargs: dict[str, Any],
    message: str,
) -> None:
    monkeypatch.setattr(analysis_base, "EXPECTED_TICK_RANGES", {SCENE: (3, 6)})
    dataset = _dataset(tmp_path, **dataset_kwargs)

    with pytest.raises(real_acoustics.RealAcousticFeatureError, match=message):
        real_acoustics._base_ticks(dataset, SCENE)


def test_feature_frame_accepts_only_complete_finite_binary_evidence(tmp_path: Path) -> None:
    frame = real_acoustics._feature_frame(_dataset(tmp_path), _evidence())

    assert list(frame.columns) == [
        "session_id",
        "timestamp_s",
        "array_energy",
        "f0_proxy",
        "voiced",
    ]
    assert frame["voiced"].tolist() == [0.0, 1.0, 0.0]
    assert frame.loc[frame["voiced"] == 0.0, "f0_proxy"].eq(0.0).all()


def test_evidence_contract_binds_exact_shape_dtype_ticks_and_timestamps() -> None:
    ticks = np.array([3, 4, 5], dtype=np.int64)

    real_acoustics._validate_evidence_contract(_evidence(ticks=ticks), ticks)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "source_local_timestamp_s",
            np.array([0.12, 0.16, 0.200000000001], dtype=np.float64),
            "timestamps do not exactly equal tick / 25",
        ),
        (
            "array_energy",
            np.array([-5.0, -4.0, -3.0], dtype=np.float32),
            "array_energy dtype differs",
        ),
        (
            "voiced",
            np.array([0.0, 1.0], dtype=np.float64),
            "voiced shape differs",
        ),
        (
            "tick_index",
            np.array([3, 4, 6], dtype=np.int64),
            "returned different tick keys",
        ),
    ],
)
def test_evidence_contract_rejects_schema_or_time_substitution(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    ticks = np.array([3, 4, 5], dtype=np.int64)
    evidence = replace(_evidence(ticks=ticks), **{field: value})

    with pytest.raises(real_acoustics.RealAcousticFeatureError, match=message):
        real_acoustics._validate_evidence_contract(evidence, ticks)


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        (
            _evidence(energy_valid=np.array([True, False, True])),
            "lacks complete real causal acoustic history",
        ),
        (
            _evidence(pitch_valid=np.array([True, False, True])),
            "lacks complete real causal acoustic history",
        ),
        (
            _evidence(array_energy=np.array([-5.0, np.nan, -3.0])),
            "array_energy contains a non-finite value",
        ),
        (
            _evidence(f0_proxy=np.array([0.0, np.inf, 0.0])),
            "f0_proxy contains a non-finite value",
        ),
        (
            _evidence(voiced=np.array([0.0, np.nan, 0.0])),
            "voiced contains a non-finite value",
        ),
        (
            _evidence(voiced=np.array([0.0, 0.5, 0.0])),
            "voiced values are not binary",
        ),
        (
            _evidence(f0_proxy=np.array([0.01, 1.0, 0.0])),
            "f0_proxy is non-zero on an unvoiced row",
        ),
    ],
)
def test_feature_frame_fails_closed_on_invalid_or_semantically_bad_evidence(
    tmp_path: Path,
    evidence: acoustic_features.AcousticFeatureEvidence,
    message: str,
) -> None:
    with pytest.raises(real_acoustics.RealAcousticFeatureError, match=message):
        real_acoustics._feature_frame(_dataset(tmp_path), evidence)


def _install_derive_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    after_extract: Callable[[], None] | None = None,
    returned_ticks: np.ndarray | None = None,
) -> tuple[real_acoustics.RealAcousticAuthority, Path, Path]:
    dataset = _dataset(tmp_path)
    manifest = tmp_path / f"{SCENE}.audio_contract.json"
    master = tmp_path / "master_aligned.wav"
    manifest.write_bytes(b"synthetic formal audio manifest")
    _write_master(master, np.zeros((48_000, 4), dtype="<i2"))
    manifest_hash, manifest_size = _sha256(manifest)
    master_hash, master_size = _sha256(master)
    master_header = {
        "channels": real_acoustics.MASTER_CHANNELS,
        "sample_width_bytes": real_acoustics.MASTER_SAMPLE_WIDTH_BYTES,
        "sample_rate_hz": real_acoustics.MASTER_SAMPLE_RATE_HZ,
        "n_frames": 48_000,
        "compression_type": "NONE",
    }
    master_source = {
        "path": str(master.resolve()),
        "size_bytes": master_size,
        "sha256": master_hash,
        "media_type": "audio/wav",
        "wave_header": master_header,
        "pcm_payload": {
            "frame_size_bytes": 8,
            "decoded_frames": 48_000,
            "frame_aligned": True,
            "complete": True,
        },
    }
    audio_report = {
        "artifact_kind": audio_contract.ARTIFACT_KIND,
        "format_version": audio_contract.FORMAT_VERSION,
        "master_sample_replay": {"exact_permutation": True},
        "sources": {"master": master_source},
    }
    authority = real_acoustics.RealAcousticAuthority(
        analysis=analysis_base.AnalysisBaseAuthority(
            pose_base_root=tmp_path / "pose",
            gold_alignment_root=tmp_path / "gold",
            audio_manifest_root=tmp_path,
            analysis_base_root=tmp_path,
            enforce_frozen_inventory=False,
        ),
        audio=audio_contract.AudioSourceAuthority(
            corpus_root=tmp_path / "corpus",
            hpe_delivery_root=tmp_path / "hpe",
        ),
    )

    monkeypatch.setattr(analysis_base, "EXPECTED_TICK_RANGES", {SCENE: (3, 6)})
    monkeypatch.setattr(
        real_acoustics,
        "_load_analysis_dataset",
        lambda scene, *, authority: dataset,
    )
    monkeypatch.setattr(
        real_acoustics,
        "_load_audio_report",
        lambda scene, *, authority: (
            audio_report,
            manifest,
            manifest_hash,
            manifest_size,
        ),
    )
    monkeypatch.setattr(
        audio_contract,
        "canonical_source_paths",
        lambda scene, authority: {"master": master},
    )
    if after_extract is not None or returned_ticks is not None:
        real_extract = acoustic_features.extract_causal_acoustic_features

        def extract(
            pcm: np.ndarray,
            source_rate_hz: int,
            *,
            tick_indices: np.ndarray,
        ) -> acoustic_features.AcousticFeatureEvidence:
            evidence = real_extract(
                pcm,
                source_rate_hz,
                tick_indices=tick_indices,
            )
            if after_extract is not None:
                after_extract()
            if returned_ticks is not None:
                evidence = replace(evidence, tick_index=returned_ticks)
            return evidence

        monkeypatch.setattr(acoustic_features, "extract_causal_acoustic_features", extract)
    return authority, manifest, master


def test_high_level_preflight_stays_owner_pending_and_non_publishable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _, _ = _install_derive_harness(tmp_path, monkeypatch)

    result = real_acoustics.derive_real_acoustic_preflight(SCENE, authority=authority)

    assert result.scene == SCENE
    assert len(result.feature_df) == 3
    assert result.report["artifact_kind"] == real_acoustics.ARTIFACT_KIND
    assert result.report["output_role"] == real_acoustics.OUTPUT_ROLE
    assert result.report["assembly_eligible"] is False
    assert result.report["counted_input_ready"] is False
    assert result.report["parameter_status"] == {
        "status": "owner_approval_pending",
        "parameter": "normalized_autocorrelation_voicing_threshold",
        "candidate_value": acoustic_features.VOICING_THRESHOLD,
        "blocker": "explicit owner freeze before formal feature publication",
    }
    assert result.report["source_selection"]["gold_channel_time_corrections_applied"] is False
    assert result.report["causality"]["full_master_processed_before_tick_selection"] is True
    assert result.report["validity"]["all_model_input_rows_valid"] is True
    assert result.report["validity"]["missing_rows_zero_filled"] is False


def test_high_level_preflight_rejects_extractor_tick_key_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _, _ = _install_derive_harness(
        tmp_path,
        monkeypatch,
        returned_ticks=np.array([3, 4, 6], dtype=np.int64),
    )

    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="extractor returned different tick keys",
    ):
        real_acoustics.derive_real_acoustic_preflight(SCENE, authority=authority)


def _read_only_scaffold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> real_acoustics.ReadOnlyAcousticArtifactScaffold:
    authority, _, _ = _install_derive_harness(tmp_path, monkeypatch)
    return real_acoustics.build_read_only_artifact_scaffold(
        SCENE,
        authority=authority,
    )


def _replace_json_payload(
    scaffold: real_acoustics.ReadOnlyAcousticArtifactScaffold,
    field: str,
    update: Callable[[dict[str, Any]], None],
) -> real_acoustics.ReadOnlyAcousticArtifactScaffold:
    raw = getattr(scaffold, field)
    value = json.loads(raw.decode("utf-8"))
    update(value)
    return replace(
        scaffold,
        **{
            field: json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        },
    )


def _replace_evidence_and_rebind_manifest(
    scaffold: real_acoustics.ReadOnlyAcousticArtifactScaffold,
    evidence_payload: bytes,
) -> real_acoustics.ReadOnlyAcousticArtifactScaffold:
    manifest = json.loads(scaffold.feature_manifest_json_payload.decode("utf-8"))
    evidence_binding = next(
        item
        for item in manifest["source_files"]
        if item["path"] == "payload:evidence_json"
    )
    evidence_binding["sha256"] = hashlib.sha256(evidence_payload).hexdigest()
    evidence_binding["byte_length"] = len(evidence_payload)
    return replace(
        scaffold,
        evidence_json_payload=evidence_payload,
        feature_manifest_json_payload=json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    )


def _replace_feature_and_rebind_manifest(
    scaffold: real_acoustics.ReadOnlyAcousticArtifactScaffold,
    feature_payload: bytes,
) -> real_acoustics.ReadOnlyAcousticArtifactScaffold:
    manifest = json.loads(scaffold.feature_manifest_json_payload.decode("utf-8"))
    manifest["feature_csv_sha256"] = hashlib.sha256(feature_payload).hexdigest()
    return replace(
        scaffold,
        feature_csv_payload=feature_payload,
        feature_manifest_json_payload=json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    )


def test_read_only_artifact_scaffold_strict_round_trip_is_reject_by_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffold = _read_only_scaffold(tmp_path, monkeypatch)

    loaded = real_acoustics.strict_replay_read_only_artifact_scaffold(scaffold)

    assert loaded.scene == SCENE
    assert scaffold.source_csv_payload == scaffold.feature_csv_payload
    assert loaded.feature_manifest["artifact_kind"] == (
        real_acoustics.SCAFFOLD_MANIFEST_KIND
    )
    assert loaded.feature_manifest["artifact_kind"] != assembly.FEATURE_MANIFEST_KIND
    assert loaded.feature_manifest["target_contract"] == {
        "artifact_kind": assembly.FEATURE_MANIFEST_KIND,
        "format_version": assembly.FEATURE_MANIFEST_FORMAT_VERSION,
        "status": "preview_only_not_emitted",
    }
    for payload in (loaded.evidence, loaded.feature_manifest):
        assert payload["assembly_eligible"] is False
        assert payload["counted_input_ready"] is False
        assert payload["parameter_status"]["candidate_value"] == 0.45
        assert payload["parameter_status"]["status"] == "owner_approval_pending"
        assert payload["promotion_gate"] == {
            "owner_freeze_token": None,
            "owner_freeze_supported_in_this_format": False,
            "independent_new_review_required": True,
            "promotion_implemented": False,
        }
    report_payload = json.dumps(
        loaded.evidence["preflight_report"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert loaded.evidence["preflight_report_binding"] == {
        "payload_id": "embedded_preflight_report",
        "sha256": hashlib.sha256(report_payload).hexdigest(),
        "byte_length": len(report_payload),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("causal", 1),
        ("lookahead_s", 0),
    ],
)
def test_read_only_manifest_causality_requires_type_exact_json_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    scaffold = _read_only_scaffold(tmp_path, monkeypatch)

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["causality"][field] = value

    tampered = _replace_json_payload(
        scaffold,
        "feature_manifest_json_payload",
        mutate,
    )
    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="causal model-input semantics",
    ):
        real_acoustics.strict_replay_read_only_artifact_scaffold(tampered)


def test_strict_json_rejects_overflow_to_infinity_after_evidence_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffold = _read_only_scaffold(tmp_path, monkeypatch)
    marker = b'"lookahead_s":0.0'
    assert scaffold.evidence_json_payload.count(marker) == 1
    evidence_payload = scaffold.evidence_json_payload.replace(
        marker,
        b'"lookahead_s":1e999',
        1,
    )
    tampered = _replace_evidence_and_rebind_manifest(scaffold, evidence_payload)

    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="forbidden non-finite number",
    ):
        real_acoustics.strict_replay_read_only_artifact_scaffold(tampered)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("source_selection", "source_kind"),
            "unapproved_source_substitution",
            "approved source/mix boundary",
        ),
        (
            ("extractor", "extractor_id"),
            "changed_extractor",
            "extractor contract",
        ),
        (
            ("master_mix_replay", "sha256"),
            "0" * 64,
            "master replay",
        ),
        (
            ("feature_arrays", "array_energy", "sha256"),
            "0" * 64,
            "feature_arrays",
        ),
    ],
)
def test_embedded_preflight_report_replays_complete_semantic_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    scaffold = _read_only_scaffold(tmp_path, monkeypatch)
    evidence = json.loads(scaffold.evidence_json_payload.decode("utf-8"))
    target: dict[str, Any] = evidence["preflight_report"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    evidence_payload = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    tampered = _replace_evidence_and_rebind_manifest(scaffold, evidence_payload)

    with pytest.raises(real_acoustics.RealAcousticFeatureError, match=message):
        real_acoustics.strict_replay_read_only_artifact_scaffold(tampered)


@pytest.mark.parametrize(
    "mutation",
    ["incomplete_report", "untrusted_paths_hashes", "arbitrary_values"],
)
def test_public_builder_rejects_every_manually_constructed_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    authority, _, _ = _install_derive_harness(tmp_path, monkeypatch)
    valid = real_acoustics.derive_real_acoustic_preflight(
        SCENE,
        authority=authority,
    )
    frame = valid.feature_df.copy(deep=True)
    report = copy.deepcopy(valid.report)
    if mutation == "incomplete_report":
        report = {"artifact_kind": real_acoustics.ARTIFACT_KIND}
    elif mutation == "untrusted_paths_hashes":
        report["analysis_base"]["path"] = str((tmp_path / "untrusted.csv").resolve())
        report["analysis_base"]["sha256"] = "0" * 64
        report["audio_contract"]["path"] = str((tmp_path / "untrusted.json").resolve())
        report["audio_contract"]["sha256"] = "0" * 64
    else:
        frame.loc[frame.index[0], "array_energy"] += 0.25
    manual = real_acoustics.RealAcousticPreflight(
        scene=valid.scene,
        feature_df=frame,
        report=report,
    )

    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="outside the frozen analysis inventory",
    ):
        real_acoustics.build_read_only_artifact_scaffold(
            manual,
            authority=authority,
        )


def test_public_builder_rejects_unknown_scene_before_derive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _, _ = _install_derive_harness(tmp_path, monkeypatch)

    def forbidden_derive(*args: object, **kwargs: object) -> None:
        raise AssertionError("derive must not run for an unknown scene")

    monkeypatch.setattr(
        real_acoustics,
        "derive_real_acoustic_preflight",
        forbidden_derive,
    )
    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="outside the frozen analysis inventory",
    ):
        real_acoustics.build_read_only_artifact_scaffold(
            "99_99__9",
            authority=authority,
        )


def test_public_builder_freshly_replays_authority_sources_on_every_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _, master = _install_derive_harness(tmp_path, monkeypatch)

    real_acoustics.build_read_only_artifact_scaffold(SCENE, authority=authority)
    master.write_bytes(b"untrusted replacement after the first build")

    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="master bytes differ from the fully replayed audio contract",
    ):
        real_acoustics.build_read_only_artifact_scaffold(SCENE, authority=authority)


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("session_id", pd.Series([SCENE] * 3, dtype="string")),
        ("timestamp_s", pd.Series([3, 4, 5], dtype=np.int64)),
        ("array_energy", pd.Series([True, False, True], dtype=bool)),
        ("f0_proxy", pd.Series([0, 1, 0], dtype=np.int64)),
        ("voiced", pd.Series([False, True, False], dtype=bool)),
    ],
)
def test_scaffold_frame_rejects_noncanonical_bool_int_or_string_dtype(
    tmp_path: Path,
    column: str,
    replacement: pd.Series,
) -> None:
    frame = real_acoustics._feature_frame(_dataset(tmp_path), _evidence())
    frame[column] = replacement

    with pytest.raises(real_acoustics.RealAcousticFeatureError, match="exact dtype"):
        real_acoustics._validate_scaffold_frame(frame, SCENE)


@pytest.mark.parametrize("mutation", ["trailing_blank", "whitespace", "float_spelling"])
def test_strict_csv_requires_complete_canonical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    scaffold = _read_only_scaffold(tmp_path, monkeypatch)
    payload = scaffold.source_csv_payload
    if mutation == "trailing_blank":
        payload += b"\n"
    elif mutation == "whitespace":
        payload = payload.replace(b",0.12,", b", 0.12,", 1)
    else:
        payload = payload.replace(b",0.12,", b",0.120,", 1)
    assert payload != scaffold.source_csv_payload
    tampered = replace(scaffold, source_csv_payload=payload)

    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="complete canonical CSV byte encoding|could not be parsed",
    ):
        real_acoustics.strict_replay_read_only_artifact_scaffold(tampered)


@pytest.mark.parametrize("mutation", ["signed_zero", "arbitrary_value"])
def test_source_feature_replay_uses_per_column_bitwise_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    scaffold = _read_only_scaffold(tmp_path, monkeypatch)
    feature = real_acoustics._strict_csv_payload(
        scaffold.feature_csv_payload,
        "feature fixture",
    )
    if mutation == "signed_zero":
        feature.loc[feature.index[0], "f0_proxy"] = -0.0
        assert np.signbit(feature.loc[feature.index[0], "f0_proxy"])
    else:
        feature.loc[feature.index[0], "array_energy"] += 0.25
    feature_payload = real_acoustics._csv_payload(feature)
    tampered = _replace_feature_and_rebind_manifest(scaffold, feature_payload)

    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="bitwise replay every typed independent source column",
    ):
        real_acoustics.strict_replay_read_only_artifact_scaffold(tampered)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("source_csv_payload", b"not,a,valid,source\n", "columns"),
        ("feature_csv_payload", b"not,a,valid,feature\n", "columns"),
        ("evidence_json_payload", b'{"x":1,"x":2}', "duplicate JSON key"),
        ("feature_manifest_json_payload", b"\xff", "strict UTF-8"),
    ],
)
def test_read_only_artifact_scaffold_rejects_each_tampered_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: bytes,
    message: str,
) -> None:
    scaffold = _read_only_scaffold(tmp_path, monkeypatch)
    tampered = replace(scaffold, **{field: replacement})

    with pytest.raises(real_acoustics.RealAcousticFeatureError, match=message):
        real_acoustics.strict_replay_read_only_artifact_scaffold(tampered)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("artifact_kind",), assembly.FEATURE_MANIFEST_KIND, "never use"),
        (("assembly_eligible",), True, "assembly eligibility"),
        (("counted_input_ready",), True, "counted-input readiness"),
        (
            ("promotion_gate", "owner_freeze_token"),
            "owner-supplied-but-unsupported",
            "owner token",
        ),
        (
            ("promotion_gate", "promotion_implemented"),
            True,
            "owner token",
        ),
        (
            ("parameter_status", "candidate_value"),
            0.46,
            "owner-pending 0.45",
        ),
    ],
)
def test_read_only_manifest_cannot_be_promoted_by_kind_flag_token_or_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    scaffold = _read_only_scaffold(tmp_path, monkeypatch)

    def mutate(manifest: dict[str, Any]) -> None:
        target: dict[str, Any] = manifest
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    tampered = _replace_json_payload(
        scaffold,
        "feature_manifest_json_payload",
        mutate,
    )
    with pytest.raises(real_acoustics.RealAcousticFeatureError, match=message):
        real_acoustics.strict_replay_read_only_artifact_scaffold(tampered)


def test_read_only_manifest_rejects_cross_column_identity_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffold = _read_only_scaffold(tmp_path, monkeypatch)

    def swap_source_column(manifest: dict[str, Any]) -> None:
        manifest["column_sources"]["array_energy"]["source_column"] = "f0_proxy"

    tampered = _replace_json_payload(
        scaffold,
        "feature_manifest_json_payload",
        swap_source_column,
    )
    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="column_sources",
    ):
        real_acoustics.strict_replay_read_only_artifact_scaffold(tampered)


def test_read_only_evidence_rejects_gold_correction_semantic_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaffold = _read_only_scaffold(tmp_path, monkeypatch)
    evidence = json.loads(scaffold.evidence_json_payload.decode("utf-8"))
    evidence["preflight_report"]["source_selection"][
        "gold_channel_time_corrections_applied"
    ] = True
    evidence_payload = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest = json.loads(scaffold.feature_manifest_json_payload.decode("utf-8"))
    manifest["source_files"][1]["sha256"] = hashlib.sha256(
        evidence_payload
    ).hexdigest()
    manifest["source_files"][1]["byte_length"] = len(evidence_payload)
    tampered = replace(
        scaffold,
        evidence_json_payload=evidence_payload,
        feature_manifest_json_payload=json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    )

    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="source/mix boundary",
    ):
        real_acoustics.strict_replay_read_only_artifact_scaffold(tampered)


def test_read_only_scaffold_has_no_writer_path_or_owner_token_api() -> None:
    build_parameters = inspect.signature(
        real_acoustics.build_read_only_artifact_scaffold
    ).parameters
    replay_parameters = inspect.signature(
        real_acoustics.strict_replay_read_only_artifact_scaffold
    ).parameters

    assert tuple(build_parameters) == ("scene", "authority")
    assert build_parameters["authority"].default is None
    assert tuple(replay_parameters) == ("scaffold",)
    assert "RealAcousticPreflight" not in real_acoustics.__all__
    assert not any(
        name.startswith(("write_", "publish_"))
        for name in real_acoustics.__all__
    )


@pytest.mark.parametrize("changed_source", ["manifest", "master"])
def test_high_level_preflight_final_rehash_rejects_source_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_source: str,
) -> None:
    source_holder: dict[str, Path] = {}

    def replace_source() -> None:
        source_holder[changed_source].write_bytes(b"replacement bytes after extraction")

    authority, manifest, master = _install_derive_harness(
        tmp_path,
        monkeypatch,
        after_extract=replace_source,
    )
    source_holder.update({"manifest": manifest, "master": master})
    expected = (
        "audio contract changed during real acoustic feature extraction"
        if changed_source == "manifest"
        else "master source changed during real acoustic feature extraction"
    )

    with pytest.raises(real_acoustics.RealAcousticFeatureError, match=expected):
        real_acoustics.derive_real_acoustic_preflight(SCENE, authority=authority)


def _frozen_matrix() -> dict[str, real_acoustics.RealAcousticPreflight]:
    return {
        scene: real_acoustics.RealAcousticPreflight(
            scene=scene,
            feature_df=pd.DataFrame(index=pd.RangeIndex(end - start)),
            report={"assembly_eligible": False},
        )
        for scene, (start, end) in analysis_base.EXPECTED_TICK_RANGES.items()
    }


def test_frozen_preflight_matrix_accepts_exact_ten_scenes_and_78582_rows() -> None:
    results = _frozen_matrix()

    real_acoustics.validate_frozen_preflight_matrix(results)

    assert len(results) == 10
    assert sum(len(result.feature_df) for result in results.values()) == 78_582


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_rows", "eligible"])
def test_frozen_preflight_matrix_rejects_inventory_row_or_readiness_substitution(
    mutation: str,
) -> None:
    results = _frozen_matrix()
    first = next(iter(results))
    if mutation == "missing":
        del results[first]
        expected = "exactly cover the ten scenes"
    elif mutation == "extra":
        results["99_99__9"] = replace(results[first], scene="99_99__9")
        expected = "exactly cover the ten scenes"
    elif mutation == "wrong_rows":
        result = results[first]
        results[first] = replace(result, feature_df=result.feature_df.iloc[:-1])
        expected = "row count changed"
    else:
        result = results[first]
        report = copy.deepcopy(result.report)
        report["assembly_eligible"] = True
        results[first] = replace(result, report=report)
        expected = "cannot claim assembly eligibility"

    with pytest.raises(real_acoustics.RealAcousticFeatureError, match=expected):
        real_acoustics.validate_frozen_preflight_matrix(results)


def test_frozen_preflight_matrix_rejects_changed_total_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _frozen_matrix()
    monkeypatch.setattr(analysis_base, "EXPECTED_TOTAL_ROWS", 78_583)

    with pytest.raises(
        real_acoustics.RealAcousticFeatureError,
        match="total differs from 78,582 rows",
    ):
        real_acoustics.validate_frozen_preflight_matrix(results)


def test_read_only_cli_exposes_no_publish_or_output_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = Path(__file__).resolve().parents[1] / "tools" / "build_real_acoustic_features.py"
    namespace = runpy.run_path(str(tool), run_name="real_acoustic_cli_contract_test")
    parser = namespace["_parser"]()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert options == {"-h", "--help", "--scene", "--all"}
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--scene", SCENE, "--publish", str(Path("artifact.csv"))])
    assert exc_info.value.code == 2
    assert "unrecognized arguments: --publish artifact.csv" in capsys.readouterr().err
