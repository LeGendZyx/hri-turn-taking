from __future__ import annotations

import copy
import json
import os
import wave
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hri_turn_taking.data import transcript_derived_vad as module


def _event(start: str, end: str, speaker: str, text: str) -> dict[str, str]:
    return {
        "start_timecode": start,
        "end_timecode": end,
        "speaker": speaker,
        "text": text,
    }


def _timecode(frame: int) -> str:
    seconds, ff = divmod(frame, 30)
    minutes, ss = divmod(seconds, 60)
    hours, mm = divmod(minutes, 60)
    return f"{hours:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _write_wav(path: Path, *, seconds: int, rate: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = (np.arange(seconds * rate, dtype=np.int32) % 1000 - 500).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(values.tobytes())


def _authority(tmp_path: Path, *, production: bool = False) -> module.TranscriptDerivedVadAuthority:
    return module.TranscriptDerivedVadAuthority(
        corpus_root=tmp_path / "corpus",
        analysis_base_root=tmp_path / "analysis",
        gold_alignment_root=tmp_path / "gold",
        audio_contract_root=tmp_path / "audio_contract",
        annotation_root=tmp_path / "annotations",
        target_root=tmp_path / "targets",
        clip_root=tmp_path / "clips",
        _production_marker=module._PRODUCTION_TOKEN if production else None,
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scene: str = "03_20__4",
    production: bool = False,
) -> tuple[module.TranscriptDerivedVadAuthority, dict[str, Path]]:
    authority = _authority(tmp_path, production=production)
    group, scenario = scene.rsplit("__", 1)
    source_paths = module._canonical_source_paths(scene, authority)
    for path in source_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if scene == "05_15__3":
        n_events = 89
        events = [
            _event(_timecode(index * 2), _timecode(index * 2 + 1), "A", f"e{index}")
            for index in range(n_events)
        ]
        events[17] = _event("00:00:34:25", "00:00:34:25", "R", "Okay, xxxx.")
        events[88] = _event(
            "00:02:36:19",
            "00:02:28:15",
            "A",
            "Thank you, dear. For letting me ask first.",
        )
        start_tick, end_tick, duration = 0, 5000, 200
        mapping = {"R": "P1", "A": "P2", "B": "P3", "C": "P4"}
    else:
        events = [
            _event("00:00:00:01", "00:00:00:04", "A", "first"),
            _event("00:00:00:03", "00:00:00:06", "B", "overlap"),
            _event("00:00:00:05", "00:00:00:08", "R", "robot"),
        ]
        start_tick, end_tick, duration = 0, 10, 2
        mapping = {"R": "P1", "A": "P2", "B": "P3"}
    source_paths["timeline"].write_text(json.dumps(events), encoding="utf-8")
    source_paths["offset"].write_text('{"fixture":true}\n', encoding="utf-8")
    ticks = np.arange(start_tick, end_tick)
    present = {
        "P1_present": np.ones(len(ticks), dtype=np.int8),
        "P2_present": np.ones(len(ticks), dtype=np.int8),
        "P3_present": np.ones(len(ticks), dtype=np.int8),
        "P4_present": np.full(len(ticks), int("C" in mapping), dtype=np.int8),
    }
    frame = pd.DataFrame(
        {
            "session_id": [scene] * len(ticks),
            "timestamp_s": ticks.astype(float) / 25,
            **present,
        }
    )
    frame.to_csv(source_paths["analysis_csv"], index=False)
    source_paths["analysis_sidecar"].write_text("{}\n", encoding="utf-8")
    source_paths["gold_alignment"].write_text("{}\n", encoding="utf-8")
    source_paths["audio_contract"].write_text("{}\n", encoding="utf-8")
    lavs = {}
    for speaker in module.SPEAKERS:
        path = tmp_path / "audio" / f"{speaker}.wav"
        _write_wav(path, seconds=duration + 2)
        lavs[speaker] = path

    def snapshot_loader(
        requested_scene: str,
        *,
        authority: module.TranscriptDerivedVadAuthority,
    ) -> module._SourceSnapshot:
        assert requested_scene == scene
        bindings = []
        media = {
            "timeline": "application/json",
            "offset": "application/json",
            "analysis_csv": "text/csv",
            "analysis_sidecar": "application/json",
            "gold_alignment": "application/json",
            "audio_contract": "application/json",
        }
        for name, path in source_paths.items():
            binding, _ = module._guarded_binding(path, media_type=media[name])
            bindings.append(binding)
        for path in lavs.values():
            binding, _ = module._guarded_binding(path, media_type="audio/wav")
            bindings.append(binding)
        timeline_raw = source_paths["timeline"].read_bytes()
        return module._SourceSnapshot(
            scene=scene,
            timeline_path=source_paths["timeline"],
            timeline_raw=timeline_raw,
            offset_path=source_paths["offset"],
            offset_raw=source_paths["offset"].read_bytes(),
            analysis_csv=source_paths["analysis_csv"],
            analysis_sidecar=source_paths["analysis_sidecar"],
            frame_df=frame.copy(),
            analysis_provenance={
                "artifact_kind": "vap_real_analysis_base",
                "session_id": scene,
                "participant_to_slot": mapping,
            },
            gold_path=source_paths["gold_alignment"],
            gold_report={},
            audio_contract_path=source_paths["audio_contract"],
            audio_report={},
            transcript_offset_sec=Decimal("0"),
            reference_duration_sec=Decimal(duration),
            start_tick=start_tick,
            end_tick=end_tick,
            participant_to_slot=mapping,
            present_by_slot={
                slot: int(frame[f"{slot}_present"].iloc[0]) for slot in ("P1", "P2", "P3", "P4")
            },
            corrections_s={
                "A": Decimal("0"),
                "B": Decimal("0.04"),
                "C": Decimal("-0.02"),
                "R": Decimal("0"),
            },
            lavalier_paths=lavs,
            bindings=tuple(sorted(bindings, key=lambda item: item["path"].casefold())),
            authority_scope="production" if authority.production else "fixture",
        )

    monkeypatch.setattr(module, "_load_source_snapshot", snapshot_loader)
    return authority, source_paths


def _attest(scene: str, authority: module.TranscriptDerivedVadAuthority) -> Path:
    return module._write_owner_attestation_with_authority(
        scene,
        owner="Project Owner",
        attested_at="2026-08-16T10:00:00+01:00",
        authority=authority,
    )


def _decisions(
    authority: module.TranscriptDerivedVadAuthority,
    values: list[dict[str, Any]],
) -> Path:
    return module._write_semantic_decisions_with_authority(
        "05_15__3",
        reviewer="Project Owner",
        decided_at="2026-08-16T10:30:00+01:00",
        decisions=values,
        authority=authority,
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows OneDrive metadata contract")
def test_guarded_binding_tolerates_only_read_induced_windows_metadata_churn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b'{"exact":true}\n')
    real_file_state = module.transcript_fs_guard._file_state
    calls = 0

    def metadata_churn(fd: int) -> module.transcript_fs_guard.FileState:
        nonlocal calls
        state = real_file_state(fd)
        calls += 1
        if calls == 1:
            return state
        return replace(
            state,
            ctime_ns=state.ctime_ns + 100,
            change_time_token=state.change_time_token + 1,
        )

    monkeypatch.setattr(module.transcript_fs_guard, "_file_state", metadata_churn)
    binding, raw = module._guarded_binding(source, media_type="application/json")
    assert raw == b'{"exact":true}\n'
    assert binding == {
        "path": str(source.resolve()),
        "size_bytes": len(raw),
        "sha256": module._sha256_bytes(raw),
        "media_type": "application/json",
    }

    calls = 0

    def content_state_churn(fd: int) -> module.transcript_fs_guard.FileState:
        nonlocal calls
        state = real_file_state(fd)
        calls += 1
        return state if calls == 1 else replace(state, size_bytes=state.size_bytes + 1)

    monkeypatch.setattr(module.transcript_fs_guard, "_file_state", content_state_churn)
    with pytest.raises(module.TranscriptDerivedVadError, match="held file changed"):
        module._guarded_binding(source, media_type="application/json")


def test_owner_attested_half_open_target_and_assembly_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _paths = _fixture(tmp_path, monkeypatch)
    _attest("03_20__4", authority)
    result = module._write_target_bundle_with_authority("03_20__4", authority=authority)
    assert result.contract["research_label"] == module.RESEARCH_LABEL
    assert result.contract["claims"] == {
        "existing_human_annotation": True,
        "targeted_reaudit_only": True,
        "all_events_reaudited": False,
        "sample_perfect_lavalier_vad_claimed": False,
        "lavalier_boundary_refinement_applied": False,
    }
    trust = result.contract["trust_boundary"]
    assert trust["writer_source_replay"] == "one_fresh_full_replay_before_publication"
    assert (
        trust["writer_postpublication_verification"]
        == "local_four_file_exact_bytes_sha256_and_schema_only"
    )
    assert trust["writer_postpublication_fresh_source_replay_claimed"] is False
    assert trust["standalone_loader_fresh_source_replay_required"] is True
    assert trust["residual_concurrent_source_change_risk_accepted"] is True
    assert result.contract["counted_gold_ready"] is False  # fixture cannot masquerade
    assert result.target_df["P2_vad"].tolist()[:5] == [0, 1, 1, 1, 0]
    assert result.target_df["P3_vad"].tolist()[:6] == [0, 0, 0, 1, 1, 0]
    assert result.target_df["P1_vad"].tolist()[:7] == [0, 0, 0, 0, 0, 1, 1]
    assert set(result.target_df["P4_vad"]) == {0}
    assert result.source_df["P4_present"].eq(0).all()
    assert result.manifest["column_sources"]["P4_vad"]["participant"] is None
    loaded = module._load_target_bundle_with_authority(result.target_csv, authority=authority)
    assert loaded.target_df.equals(result.target_df)


def test_writer_uses_one_prepublish_snapshot_and_loader_replays_fresh_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _paths = _fixture(tmp_path, monkeypatch)
    _attest("03_20__4", authority)
    fixture_loader = module._load_source_snapshot
    snapshot_calls = 0

    def counted_loader(
        scene: str, *, authority: module.TranscriptDerivedVadAuthority
    ) -> module._SourceSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return fixture_loader(scene, authority=authority)

    monkeypatch.setattr(module, "_load_source_snapshot", counted_loader)
    result = module._write_target_bundle_with_authority("03_20__4", authority=authority)
    assert snapshot_calls == 1

    snapshot_calls = 0
    loaded = module._load_target_bundle_with_authority(result.target_csv, authority=authority)
    assert snapshot_calls == 1
    assert loaded.target_df.equals(result.target_df)


def test_writer_local_postpublish_verification_rejects_tampered_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _paths = _fixture(tmp_path, monkeypatch)
    _attest("03_20__4", authority)
    fixture_loader = module._load_source_snapshot
    snapshot_calls = 0

    def counted_loader(
        scene: str, *, authority: module.TranscriptDerivedVadAuthority
    ) -> module._SourceSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return fixture_loader(scene, authority=authority)

    monkeypatch.setattr(module, "_load_source_snapshot", counted_loader)
    local_verifier = module._verify_published_target_bundle_locally
    target_paths = module._target_paths("03_20__4", authority)

    def tamper_then_verify(*args: Any, **kwargs: Any) -> module.TranscriptDerivedVadResult:
        target_paths["target"].write_bytes(b"tampered\n")
        return local_verifier(*args, **kwargs)

    monkeypatch.setattr(module, "_verify_published_target_bundle_locally", tamper_then_verify)
    with pytest.raises(module.TranscriptDerivedVadError, match="target CSV differs from staged"):
        module._write_target_bundle_with_authority("03_20__4", authority=authority)

    assert snapshot_calls == 1
    assert target_paths["target"].read_bytes() == b"tampered\n"
    assert all(not target_paths[key].exists() for key in ("source", "contract", "manifest"))


def test_standalone_loader_rejects_source_change_after_local_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, paths = _fixture(tmp_path, monkeypatch)
    _attest("03_20__4", authority)
    result = module._write_target_bundle_with_authority("03_20__4", authority=authority)
    paths["timeline"].write_bytes(b"[]\n")

    with pytest.raises(module.TranscriptDerivedVadError, match="source bytes/paths"):
        module._load_target_bundle_with_authority(result.target_csv, authority=authority)


def test_three_person_scene_with_accepted_c_event_fails_before_raster_or_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, paths = _fixture(tmp_path, monkeypatch)
    timeline = json.loads(paths["timeline"].read_text(encoding="utf-8"))
    timeline.append(_event("00:00:00:02", "00:00:00:03", "C", "absent speaker"))
    paths["timeline"].write_text(json.dumps(timeline), encoding="utf-8")
    _attest("03_20__4", authority)

    def forbidden_raster(*_args: Any, **_kwargs: Any) -> dict[str, np.ndarray]:
        pytest.fail("rasterization must not run for an absent-speaker event")

    monkeypatch.setattr(module, "_rasterize", forbidden_raster)
    with pytest.raises(
        module.TranscriptDerivedVadError,
        match=r"absent participant.*source_index=3, speaker=C, canonical_slot=P4",
    ):
        module._write_target_bundle_with_authority("03_20__4", authority=authority)

    target_paths = module._target_paths("03_20__4", authority)
    assert all(not path.exists() for path in target_paths.values())


def test_attestation_is_truthful_type_exact_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, paths = _fixture(tmp_path, monkeypatch)
    attestation = _attest("03_20__4", authority)
    value = json.loads(attestation.read_text(encoding="utf-8"))
    value["annotation_claim"]["existing_human_annotation"] = 1
    attestation.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(module.TranscriptDerivedVadError, match="annotation_claim"):
        module._build_with_authority("03_20__4", authority=authority)

    attestation.unlink()
    _attest("03_20__4", authority)
    paths["timeline"].write_text("[]\n", encoding="utf-8")
    with pytest.raises(module.TranscriptDerivedVadError, match="source bytes/paths"):
        module._build_with_authority("03_20__4", authority=authority)


def test_old_review_and_reanchored_production_attestation_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _paths = _fixture(tmp_path, monkeypatch, production=True)
    attestation = _attest("03_20__4", authority)
    snapshot = module._load_source_snapshot("03_20__4", authority=authority)
    copied = tmp_path / "copied.json"
    copied.write_bytes(attestation.read_bytes())
    with pytest.raises(module.TranscriptDerivedVadError, match="copied or re-anchored"):
        module._load_attestation_with_snapshot(copied, snapshot=snapshot, authority=authority)
    value = json.loads(attestation.read_text(encoding="utf-8"))
    value["artifact_kind"] = "vap_transcript_manual_review_final"
    attestation.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(module.TranscriptDerivedVadError, match="role/version"):
        module._build_with_authority("03_20__4", authority=authority)


def test_four_candidate_contract_requires_explicit_valid_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _paths = _fixture(tmp_path, monkeypatch, scene="05_15__3")
    _attest("05_15__3", authority)
    catalog = module.SEMANTIC_CANDIDATES["05_15__3"]
    accepted = [
        {
            "candidate_id": candidate["candidate_id"],
            "status": "accepted",
            "replacement": None,
            "note": "listened to the short correction-aware clip",
        }
        for candidate in catalog
    ]
    decision_path = _decisions(authority, accepted)
    assert decision_path.name == "semantic_decisions.v2.json"
    assert json.loads(decision_path.read_text(encoding="utf-8"))["format_version"] == 2
    result = module._build_with_authority("05_15__3", authority=authority)
    assert result.contract["semantic_decisions"] == accepted
    assert result.contract["claims"]["all_events_reaudited"] is False
    assert result.contract["format_version"] == module.FORMAT_VERSION
    assert (
        result.contract["targeted_semantic_decisions"]["format_version"]
        == module.DECISIONS_FORMAT_VERSION
    )
    assert result.contract["owner_attestation"]["format_version"] == module.FORMAT_VERSION
    snapshot = module._load_source_snapshot("05_15__3", authority=authority)
    _events, _intervals, _material, repaired = module._accepted_material(snapshot, accepted)
    assert repaired[17] == {
        "start_timecode": "00:00:33:25",
        "end_timecode": "00:00:34:25",
        "speaker": "R",
        "text": "Okay, let's stop there.",
    }
    assert repaired[88]["end_timecode"] == "00:02:38:15"
    operations = module._semantic_operations("05_15__3", accepted)
    assert [(item["field"], item["operation"]) for item in operations] == [
        ("start_timecode", "correct_timecode_boundary"),
        ("text", "correct_text_content"),
        ("end_timecode", "correct_timecode_boundary"),
    ]

    decision_path.unlink()
    rejected = copy.deepcopy(accepted)
    rejected[1].update(status="rejected", note="proposal rejected", replacement=None)
    _decisions(authority, rejected)
    with pytest.raises(module.TranscriptDerivedVadError, match="explicit replacement"):
        module._build_with_authority("05_15__3", authority=authority)

    decision_path.unlink()
    replaced = copy.deepcopy(accepted)
    replacement_event = copy.deepcopy(catalog[1]["proposed_event"])
    replacement_event["end_timecode"] = "00:02:38:16"
    replaced[1].update(
        status="replaced",
        replacement=replacement_event,
        note="selected the next 30 fps boundary after listening",
    )
    _decisions(authority, replaced)
    assert (
        module._build_with_authority("05_15__3", authority=authority).contract[
            "semantic_decisions"
        ][1]["status"]
        == "replaced"
    )


def test_e76_owner_overlay_changes_end_and_speaker_together() -> None:
    candidate = module.SEMANTIC_CANDIDATES["05_15__2"][0]
    decisions = [
        {
            "candidate_id": candidate["candidate_id"],
            "status": "accepted",
            "replacement": None,
            "note": "owner confirmed speaker A and the corrected end boundary",
        }
    ]
    operations = module._semantic_operations("05_15__2", decisions)
    assert [
        (
            item["field"],
            item["expected"],
            item["replacement"],
            item["operation"],
        )
        for item in operations
    ] == [
        (
            "end_timecode",
            "00:03:27:02",
            "00:03:57:02",
            "correct_timecode_boundary",
        ),
        ("speaker", "B", "A", "correct_speaker_identity"),
    ]


def test_decision_bool_nan_preimage_and_manifest_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, paths = _fixture(tmp_path, monkeypatch, scene="05_15__3")
    _attest("05_15__3", authority)
    catalog = module.SEMANTIC_CANDIDATES["05_15__3"]
    invalid = [
        {
            "candidate_id": item["candidate_id"],
            "status": True,
            "replacement": None,
            "note": "invalid bool",
        }
        for item in catalog
    ]
    with pytest.raises(module.TranscriptDerivedVadError, match="status"):
        _decisions(authority, invalid)
    raw = json.loads(paths["timeline"].read_text(encoding="utf-8"))
    raw[17]["end_timecode"] = "00:00:34:26"
    paths["timeline"].write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(module.TranscriptDerivedVadError, match="source bytes/paths"):
        module._build_with_authority("05_15__3", authority=authority)

    # Strict JSON rejects NaN before any semantic interpretation.
    with pytest.raises(module.TranscriptDerivedVadError, match="non-finite"):
        module._strict_json_loads(b'{"status":NaN}', "decision")


def test_target_copy_hash_tick_and_manifest_source_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _paths = _fixture(tmp_path, monkeypatch, production=True)
    _attest("03_20__4", authority)
    result = module._write_target_bundle_with_authority("03_20__4", authority=authority)
    copied = tmp_path / "copy" / "03_20__4" / "vad_target.v1.csv"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(result.target_csv.read_bytes())
    with pytest.raises(module.TranscriptDerivedVadError, match="copied, renamed"):
        module._load_target_bundle_with_authority(copied, authority=authority)

    manifest = result.target_csv.with_suffix(".manifest.json")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["source_files"][0]["path"] = "../foreign.csv"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(module.TranscriptDerivedVadError, match="manifest differs"):
        module._load_target_bundle_with_authority(result.target_csv, authority=authority)


@pytest.mark.parametrize("mode, expected", [("separate", 4), ("mix", 1)])
def test_only_targeted_correction_mapped_short_clips_are_exported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected: int,
) -> None:
    authority, _paths = _fixture(tmp_path / mode, monkeypatch, scene="05_15__3")
    output = module._export_candidate_clips_with_authority(
        "05-15-3-e88-event",
        mode=mode,
        authority=authority,
    )
    wavs = sorted(output.glob("*.wav"))
    assert len(wavs) == expected
    receipt = json.loads((output / "receipt.v1.json").read_text(encoding="utf-8"))
    assert receipt["read_only_source_export"] is True
    assert receipt["source_audio_modified"] is False
    assert receipt["per_speaker"]["B"]["correction_s"] == "0.04"
    with pytest.raises(module.TranscriptDerivedVadError, match="refusing to overwrite"):
        module._export_candidate_clips_with_authority(
            "05-15-3-e88-event",
            mode=mode,
            authority=authority,
        )
    with pytest.raises(module.TranscriptDerivedVadError, match="four targeted"):
        module._export_candidate_clips_with_authority(
            "not-a-real-candidate", mode=mode, authority=authority
        )
