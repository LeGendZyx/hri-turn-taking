from __future__ import annotations

import copy
import os
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

import hri_turn_taking.data.audio_gold_alignment as gold

SCENE = "03_20__4"


def _audio_report(
    *,
    b_pass: bool = True,
    b_median: float = 7.0,
    c_pass: bool = False,
    c_median: float = -25.5,
    r_pass: bool = True,
    r_median: float = -2.0,
) -> dict:
    audits = {
        "B": {"quality_pass": b_pass, "median_residual_ms": b_median},
        "C": {"quality_pass": c_pass, "median_residual_ms": c_median},
        "R": {"quality_pass": r_pass, "median_residual_ms": r_median},
    }
    sources = {
        speaker: {
            "path": str(Path(f"{speaker}.wav").resolve()),
            "wave_header": {"n_frames": 48_000 * 100, "sample_rate_hz": 48_000},
        }
        for speaker in gold.SPEAKERS
    }
    return {
        "artifact_kind": gold.audio_contract.ARTIFACT_KIND,
        "format_version": gold.audio_contract.FORMAT_VERSION,
        "scene": SCENE,
        "counted_gold_ready": False,
        "audit_parameters": gold.audio_contract.DEFAULT_PARAMETERS.as_dict(),
        "gates": {
            "source_identity_pass": True,
            "master_exact_permutation_pass": True,
            "reference_segmented_sync_pass": True,
        },
        "inter_lavalier_to_mic_a": audits,
        "sources": sources,
        "analysis_window_contract": {
            "required_reference_coverage": {
                "start_s": "1.01",
                "end_s_exclusive": "99.99",
            }
        },
    }


def _pose_snapshot(tmp_path: Path, *, start: int = 25, end: int = 2500) -> gold._PoseSnapshot:
    return gold._PoseSnapshot(
        csv_path=tmp_path / f"{SCENE}.csv",
        sidecar_path=tmp_path / f"{SCENE}.provenance.json",
        csv_sha256="a" * 64,
        sidecar_sha256="b" * 64,
        provenance={
            "artifact_kind": gold.assembly.REAL_INTAKE_ARTIFACT_KIND,
            "format_version": gold.assembly.REAL_PROVENANCE_FORMAT_VERSION,
            "session_id": SCENE,
            "review_status": "confirmed",
            "identity_authority": "sealed_hpe_transcript_crosswalk",
            "frame_rate_hz": 25,
            "output_tick_range": {
                "start_inclusive": start,
                "end_exclusive": end,
            },
        },
        timestamp_s=np.arange(start, end, dtype=np.float64) / 25,
    )


def test_mapping_rule_keeps_passing_channels_identity_and_corrects_only_failures() -> None:
    mappings = gold._derive_channel_mappings(_audio_report(), correct_speakers={"C"})

    assert mappings["A"]["kind"] == "identity_anchor"
    assert mappings["B"]["kind"] == "identity"
    assert mappings["B"]["correction_s"] == "0"
    assert mappings["C"]["kind"] == "constant_shift"
    assert mappings["C"]["correction_s"] == "0.0255"
    assert mappings["R"]["kind"] == "identity"
    assert mappings["R"]["correction_s"] == "0"


@pytest.mark.parametrize("median", [None, float("nan"), float("inf"), True, "3"])
def test_mapping_rule_rejects_missing_or_nonfinite_median(median: object) -> None:
    report = _audio_report(c_median=-25.5)
    report["inter_lavalier_to_mic_a"]["C"]["median_residual_ms"] = median
    with pytest.raises(gold.AudioGoldAlignmentError, match="median residual"):
        gold._derive_channel_mappings(report)


def test_decimal_coverage_and_half_open_ticks_are_exact(tmp_path: Path) -> None:
    report = _audio_report(c_pass=False, c_median=-25.5)
    pose = _pose_snapshot(tmp_path)
    mappings = gold._derive_channel_mappings(report, correct_speakers={"C"})
    window = gold._derive_authorized_window(report, mappings, pose, SCENE)

    # Reference begins at 1.01 s, so the first legal 25-Hz tick is ceil(25.25)=26.
    assert window["authorized_tick_range"] == {
        "frame_rate_hz": 25,
        "start_inclusive": 26,
        "end_exclusive": 2500,
        "n_ticks": 2474,
    }
    assert window["authorized_shared_window"] == {
        "start_s": "1.01",
        "end_s_exclusive": "99.99",
    }
    assert window["consumer_inverse_formula"] == (
        "channel_local_time_s = shared_time_s - correction_s"
    )


def test_window_api_returns_decimal_inverse_map_and_rejects_end_tick() -> None:
    window = gold.GoldAlignmentWindow(
        scene=SCENE,
        start_tick_inclusive=25,
        end_tick_exclusive=30,
        correction_s={
            "A": Decimal("0"),
            "B": Decimal("0"),
            "C": Decimal("0.0255"),
            "R": Decimal("0"),
        },
    )

    assert window.start_s == Decimal("1")
    assert window.end_s_exclusive == Decimal("1.2")
    assert window.channel_local_time("C", 25) == Decimal("0.9745")
    with pytest.raises(gold.AudioGoldAlignmentError, match="half-open window"):
        window.channel_local_time("C", 30)


def test_fractional_replay_uses_correct_sign_and_wrong_sign_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(813)
    reference = rng.normal(size=5000)
    delay_hops = 2.0
    positions = np.arange(reference.size, dtype=np.float64) - delay_hops
    delayed = np.interp(positions, np.arange(reference.size), reference, left=0, right=0)
    envelopes = {"A": reference, "B": delayed, "C": reference, "R": reference}
    monkeypatch.setattr(
        gold.audio_contract,
        "_hop_rms_envelope",
        lambda path, mixdown: envelopes[path.stem],
    )
    report = _audio_report(c_pass=True)
    report["sources"] = {
        speaker: {"path": f"{speaker}.wav"} for speaker in gold.SPEAKERS
    }
    window = {
        "authorized_shared_window": {
            "start_s": "1",
            "end_s_exclusive": "150",
        }
    }
    mappings = gold._derive_channel_mappings(_audio_report(c_pass=True))
    mappings["B"] = {
        **mappings["B"],
        "kind": "constant_shift",
        "correction_s": "-0.064",
    }

    corrected = gold._corrected_formal_window_audit(report, mappings, window)
    assert corrected["channels_to_mic_A"]["B"]["quality_pass"] is True
    assert abs(corrected["channels_to_mic_A"]["B"]["median_residual_ms"]) < 0.01

    mappings["B"]["correction_s"] = "0.064"
    wrong_sign = gold._corrected_formal_window_audit(report, mappings, window)
    assert wrong_sign["channels_to_mic_A"]["B"]["quality_pass"] is False
    assert wrong_sign["all_channels_pass"] is False


def test_pose_grid_requires_exact_contiguous_ticks(tmp_path: Path) -> None:
    pose = _pose_snapshot(tmp_path)
    pose.timestamp_s[3] += 0.001
    with pytest.raises(gold.AudioGoldAlignmentError, match="contiguous 25-Hz"):
        gold._pose_tick_range(pose, SCENE)


def test_build_detects_audio_toctou_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = gold.GoldAlignmentAuthority(tmp_path / "audio", tmp_path / "pose")
    first = _audio_report()
    second = copy.deepcopy(first)
    second["inter_lavalier_to_mic_a"]["B"]["median_residual_ms"] = 8.0
    reports = iter((first, second))
    monkeypatch.setattr(gold, "_replay_audio_manifest", lambda path: next(reports))
    pose = _pose_snapshot(tmp_path)
    monkeypatch.setattr(gold, "_replay_pose_base", lambda path: pose)
    monkeypatch.setattr(
        gold.audio_contract,
        "_hop_rms_envelope",
        lambda path, mixdown: np.arange(5000, dtype=np.float64),
    )
    passed = {
        "quality_pass": True,
        "failure_reasons": [],
        "drift_span_ms": 0.0,
    }
    monkeypatch.setattr(
        gold,
        "_corrected_formal_window_audit",
        lambda report, mappings, window, envelopes: {
            "channels_to_mic_A": {
                "B": dict(passed),
                "C": dict(passed),
                "R": dict(passed),
            },
            "all_channels_pass": True,
        },
    )

    with pytest.raises(gold.AudioGoldAlignmentError, match="changed during"):
        gold._audit_scene_with_authority(SCENE, authority=authority)


def test_source_binding_rejects_forged_path_and_hash(tmp_path: Path) -> None:
    authority = gold.GoldAlignmentAuthority(tmp_path / "audio", tmp_path / "pose")
    audio, pose_csv, pose_sidecar = gold._canonical_paths(SCENE, authority)
    for path, payload in (
        (audio, b"audio"),
        (pose_csv, b"pose"),
        (pose_sidecar, b"sidecar"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    report = {
        "scene": SCENE,
        "sources": {
            "audio_contract": gold._file_binding(
                audio,
                artifact_kind=gold.audio_contract.ARTIFACT_KIND,
                format_version=gold.audio_contract.FORMAT_VERSION,
            ),
            "pose_csv": gold._file_binding(
                pose_csv,
                artifact_kind=gold.assembly.REAL_INTAKE_ARTIFACT_KIND,
                format_version=gold.assembly.REAL_PROVENANCE_FORMAT_VERSION,
            ),
            "pose_provenance": gold._file_binding(
                pose_sidecar,
                artifact_kind=gold.assembly.REAL_INTAKE_ARTIFACT_KIND,
                format_version=gold.assembly.REAL_PROVENANCE_FORMAT_VERSION,
            ),
        },
    }
    gold._verify_bound_sources(report, authority=authority)

    report["sources"]["audio_contract"]["path"] = str(tmp_path / "forged.json")
    with pytest.raises(gold.AudioGoldAlignmentError, match="canonical path"):
        gold._verify_bound_sources(report, authority=authority)


def test_writer_is_no_clobber_and_rolls_back_failed_postpublish_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = gold.GoldAlignmentAuthority(tmp_path / "audio", tmp_path / "pose")
    output = tmp_path / "out" / "map.json"
    minimal = {
        "artifact_kind": gold.ARTIFACT_KIND,
        "format_version": gold.FORMAT_VERSION,
        "scene": SCENE,
        "output_role": gold.OUTPUT_ROLE,
        "counted_gold_ready": False,
        "trust_model": gold.TRUST_MODEL,
        "sources": {},
        "mapping_contract": {},
        "authorization": {},
        "formal_window_replay": {},
        "gates": {"counted_gold_ready": False},
    }
    monkeypatch.setattr(gold, "_audit_scene_with_authority", lambda scene, authority: minimal)
    monkeypatch.setattr(gold, "_verify_bound_sources", lambda report, authority: None)
    monkeypatch.setattr(
        gold,
        "_load_with_authority",
        lambda path, authority: (_ for _ in ()).throw(
            gold.AudioGoldAlignmentError("postpublish replay failed")
        ),
    )

    with pytest.raises(gold.AudioGoldAlignmentError, match="postpublish"):
        gold._write_with_authority(SCENE, output, authority=authority)
    assert not output.exists()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("owner", encoding="utf-8")
    with pytest.raises(gold.AudioGoldAlignmentError, match="overwrite"):
        gold._write_with_authority(SCENE, output, authority=authority)
    assert output.read_text(encoding="utf-8") == "owner"


def test_writer_rolls_back_when_link_succeeds_then_interrupts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = gold.GoldAlignmentAuthority(tmp_path / "audio", tmp_path / "pose")
    output = tmp_path / "out" / "map.json"
    minimal = {
        "artifact_kind": gold.ARTIFACT_KIND,
        "format_version": gold.FORMAT_VERSION,
        "scene": SCENE,
        "output_role": gold.OUTPUT_ROLE,
        "counted_gold_ready": False,
        "trust_model": gold.TRUST_MODEL,
        "sources": {},
        "mapping_contract": {},
        "authorization": {},
        "formal_window_replay": {},
        "gates": {"counted_gold_ready": False},
    }
    monkeypatch.setattr(gold, "_audit_scene_with_authority", lambda scene, authority: minimal)
    monkeypatch.setattr(gold, "_verify_bound_sources", lambda report, authority: None)

    def link_then_interrupt(staged: Path, destination: Path) -> None:
        os.link(staged, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(gold, "_publish_no_clobber", link_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        gold._write_with_authority(SCENE, output, authority=authority)
    assert not output.exists()


def test_writer_never_deletes_a_replacement_owner_after_failed_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = gold.GoldAlignmentAuthority(tmp_path / "audio", tmp_path / "pose")
    output = tmp_path / "out" / "map.json"
    minimal = {
        "artifact_kind": gold.ARTIFACT_KIND,
        "format_version": gold.FORMAT_VERSION,
        "scene": SCENE,
        "output_role": gold.OUTPUT_ROLE,
        "counted_gold_ready": False,
        "trust_model": gold.TRUST_MODEL,
        "sources": {},
        "mapping_contract": {},
        "authorization": {},
        "formal_window_replay": {},
        "gates": {"counted_gold_ready": False},
    }
    monkeypatch.setattr(gold, "_audit_scene_with_authority", lambda scene, authority: minimal)
    monkeypatch.setattr(gold, "_verify_bound_sources", lambda report, authority: None)

    def replace_then_fail(path: Path, authority: gold.GoldAlignmentAuthority) -> dict:
        path.unlink()
        path.write_text("concurrent owner", encoding="utf-8")
        raise gold.AudioGoldAlignmentError("postpublish replay failed")

    monkeypatch.setattr(gold, "_load_with_authority", replace_then_fail)
    with pytest.raises(gold.AudioGoldAlignmentError, match="postpublish"):
        gold._write_with_authority(SCENE, output, authority=authority)
    assert output.read_text(encoding="utf-8") == "concurrent owner"


def test_shape_never_allows_counted_gold_or_linear_manual_claims() -> None:
    report = {
        "artifact_kind": gold.ARTIFACT_KIND,
        "format_version": gold.FORMAT_VERSION,
        "scene": SCENE,
        "output_role": gold.OUTPUT_ROLE,
        "counted_gold_ready": True,
        "trust_model": gold.TRUST_MODEL,
        "sources": {},
        "mapping_contract": {"linear_or_manual_maps_allowed": True},
        "authorization": {},
        "formal_window_replay": {},
        "gates": {"counted_gold_ready": True},
    }
    with pytest.raises(gold.AudioGoldAlignmentError, match="counted-gold"):
        gold._validate_report_shape(report)
    report["counted_gold_ready"] = False
    report["gates"]["counted_gold_ready"] = False
    with pytest.raises(gold.AudioGoldAlignmentError, match="manually supplied"):
        gold._validate_report_shape(report)
    report["mapping_contract"]["linear_or_manual_maps_allowed"] = False
    report["format_version"] = True
    with pytest.raises(gold.AudioGoldAlignmentError, match="kind or version"):
        gold._validate_report_shape(report)


def test_resolution_rule_accepts_exact_72_boundary_only_with_fine_pass() -> None:
    coarse = {
        "quality_pass": False,
        "failure_reasons": ["drift_span_exceeds_limit"],
        "drift_span_ms": -72.0,
    }
    fine_pass = {"quality_pass": True}
    assert gold._resolution_status(coarse, fine_pass) == "resolution_confirmed"

    coarse["drift_span_ms"] = 72.000001
    assert gold._resolution_status(coarse, fine_pass) == "failed"
    coarse["drift_span_ms"] = 72.0
    assert gold._resolution_status(coarse, {"quality_pass": False}) == "failed"
    coarse["failure_reasons"] = [
        "median_residual_exceeds_limit",
        "drift_span_exceeds_limit",
    ]
    assert gold._resolution_status(coarse, fine_pass) == "failed"
