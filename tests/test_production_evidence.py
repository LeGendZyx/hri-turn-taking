from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hri_turn_taking.data import analysis_base, assembly
from hri_turn_taking.vap import production_evidence as evidence
from hri_turn_taking.vap.feature_sets import active_feature_columns


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_repository(root: Path) -> Path:
    files = {
        "AGENTS.md": "# fixture agent contract\n",
        "pyproject.toml": "[project]\nname = 'fixture'\nversion = '0.0.0'\n",
        "src/fixture/__init__.py": "VALUE = 1\n",
        "src/fixture/untracked.py": "DIRTY_OR_UNTRACKED = True\n",
        "configs/default.yaml": "seed: 42\n",
    }
    files.update({path: f"# {path}\n" for path in evidence.EXPLICIT_PROTOCOL_DOCS})
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / "data").mkdir()
    (root / "data/raw.wav").write_bytes(b"not source")
    (root / "report").mkdir()
    (root / "report/generated.tex").write_text("generated", encoding="utf-8")
    (root / "src/fixture/empty_source_namespace").mkdir()
    (root / "src/fixture/__pycache__").mkdir()
    return root


def test_source_state_seals_live_bytes_and_exact_scope(tmp_path: Path) -> None:
    root = _source_repository(tmp_path / "repository")
    snapshot = evidence.build_source_state_snapshot(root)

    assert evidence.validate_source_state_snapshot(snapshot, repository_root=root) == snapshot
    paths = [record["relative_path"] for record in snapshot["files"]]
    assert paths == sorted(paths)
    assert "src/fixture/untracked.py" in paths
    assert "data/raw.wav" not in paths
    assert "report/generated.tex" not in paths
    assert {
        ".",
        "src",
        "src/fixture",
        "src/fixture/empty_source_namespace",
        "configs",
        "docs",
    } <= set(snapshot["directories"])
    assert snapshot["scope"]["git_command_used"] is False
    assert snapshot["scope"]["dirty_and_untracked_source_bytes_in_scope"] is True

    (root / "src/fixture/untracked.py").write_text("CHANGED = True\n", encoding="utf-8")
    with pytest.raises(evidence.ProductionEvidenceGateError, match="fresh filesystem replay"):
        evidence.validate_source_state_snapshot(snapshot, repository_root=root)


def test_source_state_rejects_forbidden_and_non_code_entries(tmp_path: Path) -> None:
    root = _source_repository(tmp_path / "repository")
    snapshot = evidence.build_source_state_snapshot(root)
    forged = copy.deepcopy(snapshot)
    forged["files"][0]["relative_path"] = "data/raw.wav"
    with pytest.raises(evidence.ProductionEvidenceGateError, match="forbidden raw/output"):
        evidence.validate_source_state_snapshot(
            forged,
            repository_root=root,
            fresh_replay=False,
        )

    (root / "src/fixture/raw.wav").write_bytes(b"raw")
    with pytest.raises(evidence.ProductionEvidenceGateError, match="non-code/config suffix"):
        evidence.build_source_state_snapshot(root)


def test_source_state_rejects_root_tree_entry_and_ignored_reparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _source_repository(tmp_path / "repository").resolve()
    assert evidence._entry_is_forbidden_reparse(
        root / "ordinary",
        SimpleNamespace(st_file_attributes=0x400),
    )
    targets = (
        root,
        root / "src",
        root / "docs",
        root / "src/fixture/untracked.py",
        root / "src/fixture/__pycache__",
    )
    for target in targets:
        monkeypatch.setattr(
            evidence,
            "_entry_is_forbidden_reparse",
            lambda path, _info, target=target: path.resolve() == target,
        )
        with pytest.raises(
            evidence.ProductionEvidenceGateError,
            match="symlink, junction, or Windows reparse point",
        ):
            evidence.build_source_state_snapshot(root)


def test_source_state_rejects_real_linked_docs_parent(tmp_path: Path) -> None:
    root = _source_repository(tmp_path / "repository").resolve()
    docs = root / "docs"
    actual_docs = root / "actual_protocol_docs"
    docs.rename(actual_docs)
    if os.name == "nt":
        created = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(docs), str(actual_docs)],
            check=False,
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            pytest.skip(f"cannot create test junction: {created.stderr.strip()}")
    else:
        docs.symlink_to(actual_docs, target_is_directory=True)
    try:
        with pytest.raises(
            evidence.ProductionEvidenceGateError,
            match="symlink, junction, or Windows reparse point",
        ):
            evidence.build_source_state_snapshot(root)
    finally:
        if os.name == "nt":
            docs.rmdir()
        else:
            docs.unlink()


def _environment_core(snapshot: dict[str, object]) -> dict[str, object]:
    return {
        key: snapshot[key]
        for key in (
            "python",
            "operating_system",
            "packages",
            "cpu",
            "torch_runtime",
            "installed_distributions",
            "distribution_inventory_sha256",
            "inventory_method",
        )
    }


def test_environment_is_observed_exactly_and_freshly_replayed() -> None:
    snapshot = evidence.build_environment_snapshot()
    assert evidence.validate_environment_snapshot(snapshot) == snapshot
    assert Path(snapshot["python"]["executable"]).is_absolute()
    distributions = snapshot["installed_distributions"]
    assert distributions == sorted(
        distributions,
        key=lambda item: (item["name"], item["version"]),
    )
    assert snapshot["inventory_method"] == "importlib.metadata_no_pip_subprocess"

    self_report = copy.deepcopy(snapshot)
    self_report["torch_runtime"]["num_threads"] += 1
    self_report["environment_sha256"] = _canonical_hash(_environment_core(self_report))
    with pytest.raises(evidence.ProductionEvidenceGateError, match="fresh live replay"):
        evidence.validate_environment_snapshot(self_report)

    wrong_type = copy.deepcopy(snapshot)
    wrong_type["torch_runtime"]["num_threads"] = True
    wrong_type["environment_sha256"] = _canonical_hash(_environment_core(wrong_type))
    with pytest.raises(evidence.ProductionEvidenceGateError, match="exact integer"):
        evidence.validate_environment_snapshot(wrong_type, fresh_replay=False)


def _quality() -> dict[str, object]:
    return {
        "stream": "6cam",
        "offset_s": -3.25,
        "self_check_ok": True,
        "convention": "shared = timestamp_sec + offset_s",
        "status": "degraded",
        "error_bound_ms": 40.0,
        "timebase_keys": {"6cam": "timestamp_sec", "mic_A": "shared_seconds"},
    }


def _frame(scene: str, ticks: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "session_id": scene,
            "timestamp_s": ticks.astype(np.float64) / 25.0,
            "tick_25hz": ticks,
        }
    )
    for column in active_feature_columns("audio_plus_head_pose"):
        if column.endswith("_head_yaw_deg"):
            slot = int(column[1])
            frame[column] = ticks.astype(np.float64) + slot * 100.0
        elif column.endswith("_pose_valid"):
            frame[column] = np.ones(len(ticks), dtype=np.float64)
        elif column.endswith("_pose_quality"):
            frame[column] = np.full(len(ticks), 0.75, dtype=np.float64)
        elif column == "array_energy":
            frame[column] = ticks.astype(np.float64) + 10.0
        elif column == "f0_proxy":
            frame[column] = ticks.astype(np.float64) + 100.0
        elif column == "voiced":
            frame[column] = np.ones(len(ticks), dtype=np.float64)
        else:
            frame[column] = np.zeros(len(ticks), dtype=np.float64)
    frame["P1_vad"] = (ticks % 2).astype(np.float64)
    frame["transcript_marker"] = [f"word-{tick}" for tick in ticks]
    return frame


def _real_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, object],
    analysis_base.AnalysisBaseAuthority,
    pd.DataFrame,
    Path,
]:
    scene = "fixture__01"
    root = tmp_path / "analysis_base"
    root.mkdir()
    csv_path = (root / f"{scene}.csv").resolve()
    ticks = np.arange(10, dtype=np.int64)
    frame = _frame(scene, ticks)
    frame.to_csv(csv_path, index=False)
    provenance = {
        "artifact_kind": analysis_base.ARTIFACT_KIND,
        "format_version": 2,
        "session_id": scene,
        "group_id": "fixture_group",
        "analysis_window_contract": {
            "authorized_tick_range": {
                "frame_rate_hz": 25,
                "start_inclusive": 0,
                "end_exclusive": 10,
                "n_ticks": 10,
            },
            "d3_quality": _quality(),
        },
        "real_columns": [
            "array_energy",
            "f0_proxy",
            "voiced",
            *[
                column
                for column in frame.columns
                if column.startswith("P") and "pose_" in column
            ],
        ],
        "placeholder_columns": ["P1_vad"],
    }
    sidecar = csv_path.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    authority = analysis_base.AnalysisBaseAuthority(
        pose_base_root=tmp_path / "pose",
        gold_alignment_root=tmp_path / "gold",
        audio_manifest_root=tmp_path / "audio",
        analysis_base_root=root,
        enforce_frozen_inventory=False,
    )

    def fake_load(
        path: str | Path,
        *,
        data_kind: str,
        analysis_base_authority: analysis_base.AnalysisBaseAuthority,
    ) -> assembly._DatasetFile:
        resolved = Path(path).resolve()
        assert data_kind == "real"
        assert analysis_base_authority is authority
        return assembly._DatasetFile(
            path=resolved,
            frame_df=frame.copy(deep=True),
            provenance=json.loads(sidecar.read_text(encoding="utf-8")),
            data_sha256=_file_hash(resolved),
            provenance_sha256=_file_hash(sidecar),
        )

    def fake_readiness(
        source: str | Path,
        loaded: list[assembly._DatasetFile],
        *,
        data_kind: str,
        feature_set: str | None = None,
        group_delimiter: str | None = None,
    ) -> assembly.ReadinessReport:
        assert data_kind == "real"
        assert group_delimiter is None
        selected_ready = None if feature_set is None else feature_set == "audio_only"
        selected_blockers = (
            ()
            if selected_ready in {None, True}
            else (f"{scene}: fixture visual readiness is deliberately blocked",)
        )
        return assembly.ReadinessReport(
            source=str(source),
            data_kind="real",
            files=(str(loaded[0].path),),
            n_rows=len(frame),
            n_sessions=1,
            n_groups=1,
            session_ids=(scene,),
            group_ids=("fixture_group",),
            target_ready=False,
            target_blockers=(f"{scene}: counted gold VAD is not ready",),
            feature_set=feature_set,
            selected_feature_ready=selected_ready,
            selected_feature_blockers=selected_blockers,
        )

    monkeypatch.setattr(assembly, "_load_dataset_file", fake_load)
    monkeypatch.setattr(assembly, "_readiness_from_loaded", fake_readiness)
    snapshot = evidence.build_real_tick_authority_snapshot(
        (csv_path,),
        feature_sets=("audio_only", "audio_plus_head_pose"),
        authority=authority,
    )
    return snapshot, authority, frame, csv_path


def test_real_tick_authority_binds_bytes_ticks_identity_and_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, authority, _frame_df, csv_path = _real_fixture(tmp_path, monkeypatch)
    assert evidence.validate_real_tick_authority_snapshot(
        snapshot,
        authority=authority,
    ) == snapshot
    record = snapshot["records"][0]
    assert record["scene"] == "fixture__01"
    assert record["group_id"] == "fixture_group"
    assert record["tick_25hz"] == list(range(10))
    assert record["csv_sha256"] == _file_hash(csv_path)
    assert record["provenance_sha256"] == _file_hash(
        csv_path.with_suffix(".provenance.json")
    )
    assert record["dataframe_n_rows"] == 10
    assert record["dataframe_columns"] == _frame_df.columns.tolist()
    assert len(record["dataframe_content_sha256"]) == 64
    assert record["target_ready"] is False
    assert record["feature_readiness"]["audio_only"]["selected_feature_ready"] is True
    assert (
        record["feature_readiness"]["audio_plus_head_pose"]["selected_feature_ready"]
        is False
    )

    shifted = copy.deepcopy(snapshot)
    shifted_record = shifted["records"][0]
    shifted_record["tick_25hz"] = [tick + 100 for tick in shifted_record["tick_25hz"]]
    shifted_record["tick_start"] += 100
    shifted_record["tick_end_exclusive"] += 100
    shifted["inventory_sha256"] = _canonical_hash(
        {"feature_sets": shifted["feature_sets"], "records": shifted["records"]}
    )
    with pytest.raises(evidence.ProductionEvidenceGateError, match="fresh CSV/sidecar replay"):
        evidence.validate_real_tick_authority_snapshot(shifted, authority=authority)

    csv_path.write_bytes(csv_path.read_bytes() + b"\n")
    with pytest.raises(evidence.ProductionEvidenceGateError, match="fresh CSV/sidecar replay"):
        evidence.validate_real_tick_authority_snapshot(snapshot, authority=authority)


def test_tick_authority_bounds_reject_bool_even_when_bool_equals_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, _authority, _frame_df, _csv_path = _real_fixture(tmp_path, monkeypatch)
    for key, value in (
        ("tick_start", False),
        ("tick_end_exclusive", True),
        ("n_frames", True),
    ):
        forged = copy.deepcopy(snapshot)
        forged["records"][0][key] = value
        forged["inventory_sha256"] = _canonical_hash(
            {"feature_sets": forged["feature_sets"], "records": forged["records"]}
        )
        with pytest.raises(evidence.ProductionEvidenceGateError, match="exact integer"):
            evidence.validate_real_tick_authority_snapshot(
                forged,
                fresh_replay=False,
            )


def _contract_core(contract: dict[str, object]) -> dict[str, object]:
    base = {
        "artifact_kind",
        "format_version",
        "evidence_role",
        "counted_eligible",
        "publisher_available",
        "hard_blockers",
        "contract_sha256",
    }
    return {key: value for key, value in contract.items() if key not in base}


def test_d3_pose_reassembly_is_past_only_bounded_and_protects_gold_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tick_snapshot, authority, frame, _csv_path = _real_fixture(tmp_path, monkeypatch)
    contract = evidence.build_d3_reassembly_contract(
        tick_snapshot,
        scene="fixture__01",
        feature_set="audio_plus_head_pose",
        authority=authority,
    )
    result = evidence.reassemble_d3_variants(frame, contract)

    assert result.audit["output_tick_start"] == 1
    assert result.audit["output_tick_end_exclusive"] == 9
    assert result.audit["variant_source_tick_25hz"] == {
        "central": list(range(1, 9)),
        "minus": list(range(2, 10)),
        "plus": list(range(0, 8)),
    }
    assert result.variants["central"]["P1_head_yaw_deg"].tolist() == [
        float(value) for value in range(101, 109)
    ]
    assert result.variants["minus"]["P1_head_yaw_deg"].tolist() == [
        float(value) for value in range(102, 110)
    ]
    assert result.variants["plus"]["P1_head_yaw_deg"].tolist() == [
        float(value) for value in range(100, 108)
    ]
    for protected in ("array_energy", "f0_proxy", "voiced", "P1_vad", "timestamp_s"):
        central = result.variants["central"][protected]
        assert result.variants["minus"][protected].equals(central)
        assert result.variants["plus"][protected].equals(central)

    over_bound = copy.deepcopy(contract)
    over_bound["variant_aligned_source_delta_ms"]["plus"] = 80.0
    over_bound["contract_sha256"] = _canonical_hash(_contract_core(over_bound))
    with pytest.raises(evidence.ProductionEvidenceGateError, match="exactly bounded"):
        evidence.reassemble_d3_variants(frame, over_bound)

    missing_tick = frame.iloc[1:].reset_index(drop=True)
    with pytest.raises(evidence.ProductionEvidenceGateError, match="bound real tick authority"):
        evidence.reassemble_d3_variants(missing_tick, contract)


def test_d3_rejects_forged_full_frame_despite_inherited_source_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tick_snapshot, authority, frame, _csv_path = _real_fixture(tmp_path, monkeypatch)
    contract = evidence.build_d3_reassembly_contract(
        tick_snapshot,
        scene="fixture__01",
        feature_set="audio_plus_head_pose",
        authority=authority,
    )
    for column in ("array_energy", "P1_vad"):
        forged = frame.copy(deep=True)
        forged.loc[3, column] += 0.25
        with pytest.raises(
            evidence.ProductionEvidenceGateError,
            match="canonical CSV-loaded content/schema",
        ):
            evidence.reassemble_d3_variants(forged, contract)

    reordered = frame.loc[:, list(reversed(frame.columns))]
    with pytest.raises(
        evidence.ProductionEvidenceGateError,
        match="canonical CSV-loaded content/schema",
    ):
        evidence.reassemble_d3_variants(reordered, contract)

    non_finite = frame.copy(deep=True)
    non_finite.loc[2, "array_energy"] = np.nan
    with pytest.raises(evidence.ProductionEvidenceGateError, match="NaN/Inf"):
        evidence.reassemble_d3_variants(non_finite, contract)


def test_d3_tick_column_is_integer_and_exactly_matches_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tick_snapshot, authority, frame, _csv_path = _real_fixture(tmp_path, monkeypatch)
    contract = evidence.build_d3_reassembly_contract(
        tick_snapshot,
        scene="fixture__01",
        feature_set="audio_plus_head_pose",
        authority=authority,
    )
    float_ticks = frame.copy(deep=True)
    float_ticks["tick_25hz"] = float_ticks["tick_25hz"].astype(np.float64)
    with pytest.raises(evidence.ProductionEvidenceGateError, match="integer dtype"):
        evidence.reassemble_d3_variants(float_ticks, contract)

    mismatched_time = frame.copy(deep=True)
    mismatched_time.loc[4, "timestamp_s"] += 0.04
    with pytest.raises(evidence.ProductionEvidenceGateError, match="differs from tick_25hz"):
        evidence.reassemble_d3_variants(mismatched_time, contract)


def test_d3_audio_only_is_exact_identity_and_all_objects_remain_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tick_snapshot, authority, frame, _csv_path = _real_fixture(tmp_path, monkeypatch)
    contract = evidence.build_d3_reassembly_contract(
        tick_snapshot,
        scene="fixture__01",
        feature_set="audio_only",
        authority=authority,
    )
    result = evidence.reassemble_d3_variants(frame, contract)
    assert result.audit["identity_no_shift"] is True
    assert result.audit["variant_source_tick_25hz"] == {
        name: list(range(10)) for name in ("central", "minus", "plus")
    }
    for variant in result.variants.values():
        assert variant.equals(frame)

    environment = evidence.build_environment_snapshot()
    source = evidence.build_source_state_snapshot(Path(__file__).parents[1])
    for payload in (source, environment, tick_snapshot, contract):
        assert payload["counted_eligible"] is False
        assert payload["publisher_available"] is False
        assert payload["hard_blockers"] == list(evidence.EVIDENCE_BLOCKERS)
    assert all(
        token not in name
        for name in evidence.__all__
        for token in ("publish", "register", "train", "write")
    )
