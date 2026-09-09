"""Tests for exact-key real-data assembly and source readiness gates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hri_turn_taking.data.analysis_base as analysis_base_module
import hri_turn_taking.data.assembly as assembly_module
from hri_turn_taking.data.assembly import (
    AssemblyGateError,
    DataSourceGateError,
    assemble_session,
    audit_vap_source,
    load_and_audit_vap_source,
    load_vap_dataframe,
    write_synthetic_provenance,
)
from hri_turn_taking.data.intake import write_intake
from hri_turn_taking.data.multiparty import (
    SyntheticConfig,
    generate_multiparty_dataset,
    validate_multiparty_dataframe,
)

_TEST_DELIVERIES: dict[str, Path] = {}


@pytest.fixture(autouse=True)
def _sealed_crosswalk_contract_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _TEST_DELIVERIES.clear()
    inventory_path = tmp_path / "sealed_inventory.json"
    inventory_path.write_text('{"test":"sealed"}', encoding="utf-8")
    artifact_names = [
        "test_intake.py",
        "test_real_data_assembly.py",
        "test_cli_gates.py",
        "test_vap.py",
        "test_predictions.py",
    ]

    def fake_contract(
        scene: str,
        inventory_path_arg: str | Path,
        *,
        repository_root: str | Path,
    ) -> dict:
        del inventory_path_arg
        delivery = _TEST_DELIVERIES[scene]
        world_path = delivery / "world_persons.csv"
        hpe_authority = {
            "path": str((delivery / "mapping.json").resolve()),
            "sha256": _sha256(delivery / "mapping.json"),
        }
        root = Path(repository_root)
        files = {
            name: {
                "size_bytes": (root / "tests" / name).stat().st_size,
                "sha256": _sha256(root / "tests" / name),
            }
            for name in artifact_names
        }
        proposal = {"1": "R", "2": "A", "3": "B"}
        return {
            "artifact_kind": "vap_hpe_transcript_identity_downstream_contract",
            "format_version": 1,
            "inventory": {
                "path": str(inventory_path),
                "repository_path": "tests/fixtures/sealed_inventory.json",
                "sha256": _sha256(inventory_path),
                "artifact_kind": "vap_hpe_transcript_crosswalk_inventory",
                "format_version": 1,
            },
            "scene": scene,
            "accepted_window": {
                "scene": scene,
                "camera": "Cam_2",
                "local_start_inclusive": 0,
                "local_end_exclusive": 100,
                "logical_start_inclusive": 0,
                "logical_end_exclusive": 100,
                "filter_rule": "start <= frame_index < end",
            },
            "source_fps": 25.0,
            "hpe_logical_timing_authority": {
                "axis": "hpe_delivery_logical_frame_index",
                "source_fps": 25.0,
                "evidence_cameras": [
                    {
                        "camera": "Cam_2",
                        "source_fps": 25.0,
                        "run_metadata": hpe_authority,
                    }
                ],
                "validation_complete": hpe_authority,
            },
            "expected_hpe_persons": [1, 2, 3],
            "person_mapping": [
                {"hpe_person": 1, "participant": "R", "slot": "P1"},
                {"hpe_person": 2, "participant": "A", "slot": "P2"},
                {"hpe_person": 3, "participant": "B", "slot": "P3"},
            ],
            "artifact_dir": "tests",
            "artifact_files": files,
            "ledger_snapshot_sha256": "0" * 64,
            "upstream_proposal": proposal,
            "upstream_mismatch": {"differences": [], "acknowledgement": ""},
            "delivery_world_persons": {
                "path": str(world_path.resolve()),
                "sha256": _sha256(world_path),
            },
            "hpe_validation_authority": hpe_authority,
        }

    monkeypatch.setattr(
        "hri_turn_taking.data.intake.load_sealed_crosswalk_contract", fake_contract
    )


def _make_intake_base(
    directory: Path,
    *,
    session_id: str = "g01__s1",
    error_bound_ms: float = 33.0,
    offset_s: float = -1.2,
) -> tuple[Path, pd.DataFrame]:
    directory.mkdir(parents=True)
    if "__" not in session_id:
        raise ValueError("test real-intake session_id must use the canonical group__scenario form")
    group_id, scenario_id = session_id.split("__", 1)
    delivery = directory / "delivery" / session_id
    delivery.mkdir(parents=True)
    geometry = {
        "1": ((0.0, 0.0), (0.0, 1.0)),
        "2": ((0.0, 2.0), (0.0, -1.0)),
        "3": ((2.0, 0.0), (0.0, 1.0)),
    }
    rows: list[dict[str, object]] = []
    for person, ((x, z), (facing_x, facing_z)) in geometry.items():
        for frame_index in range(100):
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": frame_index / 25.0,
                    "world_person": int(person),
                    "n_cams": 3,
                    "cams": "Cam_2+Cam_3+Cam_4",
                    "x_w": x,
                    "y_w": 1.6,
                    "z_w": z,
                    "facing_x": facing_x,
                    "facing_y": 0.0,
                    "facing_z": facing_z,
                    "facing_src_cam": "Cam_2",
                    "facing_src_abs_yaw": 10.0,
                    "person": int(person),
                    "tracklet": 0,
                }
            )
    pd.DataFrame(rows).to_csv(delivery / "world_persons.csv", index=False)
    person_map = {"1": "R", "2": "A", "3": "B"}
    pd.DataFrame(
        {"person": list(person_map), "participant": list(person_map.values())}
    ).to_csv(delivery / "participant_map.csv", index=False)
    (delivery / "mapping.json").write_text(
        json.dumps(
            {
                "review_status": "confirmed",
                "delivery": session_id,
                "group": group_id,
                "scenario": scenario_id,
                "participant_map": person_map,
            }
        ),
        encoding="utf-8",
    )
    (delivery / "offset.json").write_text(
        json.dumps({"streams": {"6cam": offset_s}}), encoding="utf-8"
    )
    (delivery / "offset_quality.json").write_text(
        json.dumps(
            {
                "convention": "shared_time = stream_local_time + offset_s",
                "timebase_keys": {"6cam": "avi_frame_axis"},
                "quality": {
                    "6cam": {
                        "offset_s": offset_s,
                        "self_check_ok": True,
                        "status": "ok",
                        "error_bound_ms": error_bound_ms,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    _TEST_DELIVERIES[session_id] = delivery
    path = directory / f"{session_id}.csv"
    result = write_intake(delivery, path)
    return path, result.frame_df


def _feature_frame(base: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = base[["session_id", "timestamp_s"]].copy()
    frame_index = np.arange(len(base))
    for offset, column in enumerate(columns):
        if column.endswith("_vad"):
            result[column] = ((frame_index + offset) // 5 % 2).astype(int)
        elif column == "doa_angle_deg":
            result[column] = 0.0
        else:
            result[column] = np.linspace(0.1, 0.9, len(base))
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_feature(
    path: Path,
    base: pd.DataFrame,
    columns: list[str],
    *,
    participant_to_slot: dict[str, str] | None = None,
    column_bindings: dict[str, object] | None = None,
    causal: bool | None = None,
    window_alignment: str | None = None,
    lookahead_s: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_df = _feature_frame(base, columns)
    feature_df.to_csv(path, index=False)
    source_path = path.with_suffix(".source.csv")
    feature_df.to_csv(source_path, index=False)
    mapping = participant_to_slot or {"R": "P1", "A": "P2", "B": "P3"}
    participant_by_slot = {slot: participant for participant, slot in mapping.items()}
    if column_bindings is None:
        column_bindings = {}
        for column in columns:
            prefix = column.split("_", 1)[0]
            column_bindings[column] = (
                {"participant": participant_by_slot[prefix], "slot": prefix}
                if prefix in participant_by_slot
                else {"participant": None, "slot": None}
            )
    purpose = "gold_target" if all(column.endswith("_vad") for column in columns) else "model_input"
    if causal is None:
        causal = purpose == "model_input"
    if window_alignment is None:
        window_alignment = "trailing" if causal else "offline"
    if causal and lookahead_s is None:
        lookahead_s = 0.0
    column_sources = {
        column: {
            **column_bindings[column],
            "source_path": source_path.name,
            "source_sha256": _sha256(source_path),
            "source_column": column,
            "source_participant": column_bindings[column]["participant"],
            "source_keys": {"session_id": "session_id", "timestamp_s": "timestamp_s"},
            "value_transform": {"kind": "identity"},
        }
        for column in columns
    }
    manifest = {
        "format_version": 3,
        "artifact_kind": "vap_feature_source",
        "session_id": str(base["session_id"].iloc[0]),
        "participant_to_slot": mapping,
        "supplied_columns": columns,
        "feature_csv_sha256": _sha256(path),
        "source_files": [{"path": source_path.name, "sha256": _sha256(source_path)}],
        "purpose": purpose,
        "causality": {
            "causal": causal,
            "lookahead_s": lookahead_s,
            "window_alignment": window_alignment,
        },
        "column_sources": column_sources,
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return path


def _refresh_feature_hash(path: Path, *, sync_source: bool = True) -> None:
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sync_source:
        feature_df = pd.read_csv(path)
        for source in manifest["source_files"]:
            source_path = manifest_path.parent / source["path"]
            source_df = pd.read_csv(source_path)
            source_df[["session_id", "timestamp_s"]] = feature_df[
                ["session_id", "timestamp_s"]
            ]
            for column, record in manifest["column_sources"].items():
                if record["source_path"] == source["path"]:
                    source_df[record["source_column"]] = feature_df[column]
            source_df.to_csv(source_path, index=False)
            source["sha256"] = _sha256(source_path)
            for record in manifest["column_sources"].values():
                if record["source_path"] == source["path"]:
                    record["source_sha256"] = source["sha256"]
    manifest["feature_csv_sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _refresh_real_csv_binding(path: Path) -> None:
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    persisted = pd.read_csv(path)
    provenance.update(
        {
            "csv_sha256": _sha256(path),
            "columns": persisted.columns.tolist(),
            "dtypes": {column: str(persisted[column].dtype) for column in persisted},
            "n_rows": len(persisted),
            "n_frames": len(persisted),
        }
    )
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")


def _assemble_trainable_session(
    tmp_path: Path,
    session_id: str,
    *,
    error_bound_ms: float = 33.0,
) -> Path:
    base_path, base_df = _make_intake_base(
        tmp_path / "intake" / session_id,
        session_id=session_id,
        error_bound_ms=error_bound_ms,
    )
    feature_path = _write_feature(
        tmp_path / "features" / f"{session_id}_vad.csv",
        base_df,
        ["P1_vad", "P2_vad", "P3_vad"],
    )
    output = tmp_path / "assembled" / f"{session_id}.csv"
    assemble_session(base_path, [feature_path], output)
    return output


def test_assemble_session_exact_merge_and_provenance(tmp_path: Path) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    vad_path = _write_feature(
        tmp_path / "features" / "vad.csv",
        base_df,
        ["P1_vad", "P2_vad", "P3_vad"],
    )
    asd_path = _write_feature(
        tmp_path / "features" / "asd.csv",
        base_df,
        ["P1_asd_prob", "P2_asd_prob", "P3_asd_prob"],
    )
    output = tmp_path / "assembled" / "g01__s1.csv"

    result = assemble_session(base_path, [vad_path, asd_path], output)

    assert output.exists()
    assert result.provenance_path.exists()
    assert result.provenance["trainable"] is True
    assert result.provenance["trainability"]["missing_gold_vad_columns"] == []
    assert result.provenance["trainability"]["degenerate_gold_vad_columns"] == []
    assert "P1_vad" in result.provenance["real_columns"]
    assert "P1_vad" not in result.provenance["placeholder_columns"]
    assert "doa_angle_deg" in result.provenance["placeholder_columns"]
    assert result.provenance["inert_columns"] == ["doa_angle_deg"]
    assert len(result.provenance["feature_sources"]) == 2
    assert len(result.provenance["feature_sources"][0]["sha256"]) == 64
    loaded = load_vap_dataframe(output, data_kind="real")
    assert loaded.drop(columns="group_id").equals(
        pd.read_csv(output, float_precision="round_trip")
    )
    assert set(loaded["group_id"]) == {"g01"}


@pytest.mark.parametrize("artifact", ["intake", "assembled"])
def test_real_sidecar_binds_final_csv_bytes_and_schema(
    tmp_path: Path, artifact: str
) -> None:
    if artifact == "intake":
        path, _ = _make_intake_base(tmp_path / "intake")
    else:
        path = _assemble_trainable_session(tmp_path / "assembled", "g01__s1")
    persisted = pd.read_csv(path)
    provenance = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert provenance["csv_sha256"] == _sha256(path)
    assert provenance["columns"] == persisted.columns.tolist()
    assert provenance["dtypes"] == {
        column: str(persisted[column].dtype) for column in persisted
    }
    assert provenance["n_rows"] == len(persisted)
    tampered = pd.read_csv(path)
    tampered.loc[tampered.index[0], "P1_asd_prob"] = 0.123456
    tampered.to_csv(path, index=False)

    with pytest.raises(DataSourceGateError, match="csv_sha256 does not match"):
        load_vap_dataframe(path, data_kind="real")


@pytest.mark.parametrize("binding_field", ["columns", "dtypes", "n_rows"])
def test_real_loader_rejects_forged_csv_schema_binding(
    tmp_path: Path, binding_field: str
) -> None:
    path, _ = _make_intake_base(tmp_path / binding_field)
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    if binding_field == "columns":
        provenance["columns"] = list(reversed(provenance["columns"]))
    elif binding_field == "dtypes":
        current = provenance["dtypes"]["P1_vad"]
        provenance["dtypes"]["P1_vad"] = "int64" if current != "int64" else "float64"
    else:
        provenance["n_rows"] += 1
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match=rf"{binding_field} does not bind"):
        load_vap_dataframe(path, data_kind="real")


def test_real_artifact_kind_blocks_unmanifested_promotion(tmp_path: Path) -> None:
    intake, _ = _make_intake_base(tmp_path / "intake")
    sidecar = intake.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["placeholder_columns"].remove("P1_vad")
    provenance["real_columns"].append("P1_vad")
    world_source = provenance["intake_sources"]["world_persons"]
    provenance["column_sources"]["P1_vad"] = {
        "participant": "R",
        "slot": "P1",
        "purpose": "gold_target",
        **world_source,
    }
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="feature/gold promotion requires assembly"):
        load_vap_dataframe(intake, data_kind="real")

    assembled = _assemble_trainable_session(tmp_path / "assembled", "g02__s1")
    assembled_sidecar = assembled.with_suffix(".provenance.json")
    assembled_provenance = json.loads(assembled_sidecar.read_text(encoding="utf-8"))
    assembled_provenance["feature_sources"] = []
    assembled_sidecar.write_text(json.dumps(assembled_provenance), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="must retain feature_sources"):
        load_vap_dataframe(assembled, data_kind="real")


def test_assembled_artifact_rejects_duplicate_supplied_column_record(tmp_path: Path) -> None:
    assembled = _assemble_trainable_session(tmp_path, "g01__s1")
    sidecar = assembled.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["assembly"]["supplied_columns"].append(
        provenance["assembly"]["supplied_columns"][0]
    )
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="supplied_columns must match"):
        load_vap_dataframe(assembled, data_kind="real")


def test_assembly_rejects_any_non_exact_key_order(tmp_path: Path) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature = _feature_frame(base_df, ["P1_vad"])
    feature = feature.iloc[::-1].reset_index(drop=True)
    feature_path = tmp_path / "reversed.csv"
    feature.to_csv(feature_path, index=False)

    with pytest.raises(AssemblyGateError, match="keys/order"):
        assemble_session(base_path, [feature_path], tmp_path / "out.csv")


def test_assembly_rejects_duplicate_unknown_and_real_columns(tmp_path: Path) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    first = _write_feature(tmp_path / "first.csv", base_df, ["P1_vad"])
    duplicate = _write_feature(tmp_path / "duplicate.csv", base_df, ["P1_vad"])
    with pytest.raises(AssemblyGateError, match="more than one file"):
        assemble_session(base_path, [first, duplicate], tmp_path / "duplicate_out.csv")

    unknown = _write_feature(tmp_path / "unknown.csv", base_df, ["P1_vad"])
    unknown_df = pd.read_csv(unknown).rename(columns={"P1_vad": "P1_oracle_target"})
    unknown_df.to_csv(unknown, index=False)
    with pytest.raises(AssemblyGateError, match="non-canonical"):
        assemble_session(base_path, [unknown], tmp_path / "unknown_out.csv")

    real = _write_feature(tmp_path / "real.csv", base_df, ["P1_head_yaw_deg"])
    with pytest.raises(AssemblyGateError, match="non-placeholder"):
        assemble_session(base_path, [real], tmp_path / "real_out.csv")


def test_assembly_keeps_real_doa_inert(tmp_path: Path) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    doa = _feature_frame(base_df, ["doa_angle_deg"])
    doa["doa_angle_deg"] = 12.0
    doa_path = tmp_path / "doa.csv"
    doa.to_csv(doa_path, index=False)

    with pytest.raises(AssemblyGateError, match="inert zero"):
        assemble_session(base_path, [doa_path], tmp_path / "out.csv")

    inert = _write_feature(tmp_path / "inert_doa.csv", base_df, ["doa_angle_deg"])
    output = tmp_path / "inert_out.csv"
    result = assemble_session(base_path, [inert], output)
    assert "doa_angle_deg" in result.provenance["placeholder_columns"]
    assert "doa_angle_deg" not in result.provenance["column_sources"]
    load_vap_dataframe(output, data_kind="real")


@pytest.mark.parametrize("artifact", ["intake", "assembled"])
def test_real_loader_rejects_nonzero_doa_after_shallow_binding_refresh(
    tmp_path: Path, artifact: str
) -> None:
    if artifact == "intake":
        path, _ = _make_intake_base(tmp_path / "intake")
    else:
        path = _assemble_trainable_session(tmp_path / "assembled", "g01__s1")
    tampered = pd.read_csv(path)
    tampered["doa_angle_deg"] = 42.0
    tampered.to_csv(path, index=False)
    _refresh_real_csv_binding(path)

    with pytest.raises(DataSourceGateError, match="doa_angle_deg must remain strictly zero"):
        load_vap_dataframe(path, data_kind="real")


def test_trainability_requires_every_present_nondegenerate_vad(tmp_path: Path) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "missing" / "base")
    incomplete = _write_feature(
        tmp_path / "missing" / "vad.csv", base_df, ["P1_vad", "P2_vad"]
    )
    incomplete_output = tmp_path / "missing" / "assembled.csv"
    incomplete_result = assemble_session(base_path, [incomplete], incomplete_output)
    assert incomplete_result.provenance["trainable"] is False
    assert incomplete_result.provenance["trainability"]["missing_gold_vad_columns"] == [
        "P3_vad"
    ]
    assert len(load_vap_dataframe(incomplete_output, data_kind="real")) == len(base_df)
    incomplete_report = audit_vap_source(incomplete_output, data_kind="real")
    assert not incomplete_report.target_ready
    assert "P3_vad" in " ".join(incomplete_report.target_blockers)

    deg_base, deg_df = _make_intake_base(tmp_path / "degenerate" / "base")
    degenerate = _feature_frame(deg_df, ["P1_vad", "P2_vad", "P3_vad"])
    degenerate["P3_vad"] = 0
    deg_path = tmp_path / "degenerate" / "vad.csv"
    _write_feature(deg_path, deg_df, ["P1_vad", "P2_vad", "P3_vad"])
    degenerate.to_csv(deg_path, index=False)
    _refresh_feature_hash(deg_path)
    deg_result = assemble_session(
        deg_base, [deg_path], tmp_path / "degenerate" / "assembled.csv"
    )
    assert deg_result.provenance["trainable"] is False
    assert deg_result.provenance["trainability"]["degenerate_gold_vad_columns"] == [
        "P3_vad"
    ]
    assert not audit_vap_source(deg_result.output_csv, data_kind="real").target_ready


def test_chained_assembly_retains_all_unique_feature_manifests(tmp_path: Path) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    first_feature = _write_feature(tmp_path / "p1.csv", base_df, ["P1_vad"])
    first_output = tmp_path / "first.csv"
    assemble_session(base_path, [first_feature], first_output)

    remaining = _write_feature(
        tmp_path / "remaining.csv", base_df, ["P2_vad", "P3_vad"]
    )
    final_output = tmp_path / "final.csv"
    assemble_session(first_output, [remaining], final_output)

    loaded = load_vap_dataframe(final_output, data_kind="real")
    assert len(loaded) == len(base_df)
    provenance = json.loads(
        final_output.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    assert set(provenance["assembly"]["supplied_columns"]) == {
        "P1_vad",
        "P2_vad",
        "P3_vad",
    }


def test_chained_assembly_rejects_duplicate_inert_doa_before_writing(
    tmp_path: Path,
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    first_doa = _write_feature(tmp_path / "first_doa.csv", base_df, ["doa_angle_deg"])
    first_output = tmp_path / "first.csv"
    assemble_session(base_path, [first_doa], first_output)

    repeated_doa = _write_feature(
        tmp_path / "repeated_doa.csv", base_df, ["doa_angle_deg"]
    )
    rejected_output = tmp_path / "must_not_exist.csv"
    with pytest.raises(AssemblyGateError, match="repeat supplied columns"):
        assemble_session(first_output, [repeated_doa], rejected_output)

    assert not rejected_output.exists()
    assert not rejected_output.with_suffix(".provenance.json").exists()


def test_directory_loading_and_readiness_report(tmp_path: Path) -> None:
    first = _assemble_trainable_session(tmp_path, "g01__s1")
    second = _assemble_trainable_session(tmp_path, "g02__s1", error_bound_ms=55.0)
    assert first.parent == second.parent

    loaded = load_vap_dataframe(first.parent, data_kind="real")
    assert loaded["session_id"].drop_duplicates().tolist() == ["g01__s1", "g02__s1"]
    report = audit_vap_source(first.parent, data_kind="real")
    assert report.trainable
    assert report.n_sessions == report.session_count == 2
    assert report.n_groups == report.group_count == 2
    assert report.group_ids == ("g01", "g02")
    assert any(
        "at least three group-disjoint recording groups" in blocker
        for blocker in report.experiment_blockers
    )
    assert len(report.source_artifacts) == 2
    assert all(len(item["data_sha256"]) == 64 for item in report.source_artifacts)
    assert all(len(item["provenance_sha256"]) == 64 for item in report.source_artifacts)
    assert all(item["format_version"] == 2 for item in report.source_artifacts)
    assert all(
        item["identity_authority"] == "sealed_hpe_transcript_crosswalk"
        for item in report.source_artifacts
    )
    assert all(
        len(item["identity_inventory_sha256"]) == 64
        for item in report.source_artifacts
    )
    assert "doa_angle_deg" in report.placeholder_columns
    assert any(
        "55 ms" in warning and "scientific limitation" in warning
        for warning in report.warnings
    )

    _assemble_trainable_session(tmp_path, "g03__s1")
    three_group_report = audit_vap_source(first.parent, data_kind="real")
    assert three_group_report.n_groups == 3
    assert not three_group_report.experiment_blockers


def test_explicit_session_exclusion_is_present_exact_and_not_loaded(tmp_path: Path) -> None:
    first = _assemble_trainable_session(tmp_path, "g01__s1")
    _assemble_trainable_session(tmp_path, "g02__s1")

    loaded, report = load_and_audit_vap_source(
        first.parent,
        data_kind="real",
        excluded_session_ids=("g02__s1",),
    )
    assert loaded["session_id"].drop_duplicates().tolist() == ["g01__s1"]
    assert report.session_ids == ("g01__s1",)
    assert len(report.source_artifacts) == 1

    with pytest.raises(DataSourceGateError, match="absent from the source"):
        load_and_audit_vap_source(
            first.parent,
            data_kind="real",
            excluded_session_ids=("g99__missing",),
        )
    with pytest.raises(DataSourceGateError, match="cannot contain duplicates"):
        load_and_audit_vap_source(
            first.parent,
            data_kind="real",
            excluded_session_ids=("g02__s1", "g02__s1"),
        )
    with pytest.raises(DataSourceGateError, match="removed every source session"):
        load_and_audit_vap_source(
            first.parent,
            data_kind="real",
            excluded_session_ids=("g01__s1", "g02__s1"),
        )


def test_directory_loading_rejects_duplicate_sessions_and_missing_sidecars(
    tmp_path: Path,
) -> None:
    output = _assemble_trainable_session(tmp_path / "source", "g01__s1")
    duplicates = tmp_path / "duplicates"
    duplicates.mkdir()
    for name in ("a", "b"):
        copied_csv = duplicates / f"{name}.csv"
        copied_sidecar = duplicates / f"{name}.provenance.json"
        shutil.copyfile(output, copied_csv)
        provenance = json.loads(
            output.with_suffix(".provenance.json").read_text(encoding="utf-8")
        )
        provenance["assembly"]["output_csv"] = str(copied_csv.resolve())
        copied_sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="appears in both"):
        load_vap_dataframe(duplicates, data_kind="real")

    bare = tmp_path / "bare"
    bare.mkdir()
    shutil.copyfile(output, bare / "g01__s1.csv")
    with pytest.raises(DataSourceGateError, match="bare CSV files are never trusted"):
        load_vap_dataframe(bare, data_kind="real")


def test_source_loader_fails_closed_on_malformed_sidecar(tmp_path: Path) -> None:
    output = _assemble_trainable_session(tmp_path, "g01__s1")
    sidecar = output.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["trainable"] = "true"
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="JSON boolean"):
        load_vap_dataframe(output, data_kind="real")


def test_source_loader_replays_trainability_claims(tmp_path: Path) -> None:
    intake, _ = _make_intake_base(tmp_path / "intake")
    intake_sidecar = intake.with_suffix(".provenance.json")
    intake_provenance = json.loads(intake_sidecar.read_text(encoding="utf-8"))
    intake_provenance["trainable"] = True
    intake_sidecar.write_text(json.dumps(intake_provenance), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="replayed gold-VAD readiness"):
        load_vap_dataframe(intake, data_kind="real")

    assembled = _assemble_trainable_session(tmp_path / "assembled", "g01__s1")
    assembled_sidecar = assembled.with_suffix(".provenance.json")
    assembled_provenance = json.loads(
        assembled_sidecar.read_text(encoding="utf-8")
    )
    assembled_provenance["trainability"]["missing_gold_vad_columns"] = ["P1_vad"]
    assembled_sidecar.write_text(json.dumps(assembled_provenance), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="trainability does not match"):
        load_vap_dataframe(assembled, data_kind="real")


def test_real_data_cannot_bypass_provenance_by_renaming_session(tmp_path: Path) -> None:
    path, _ = _make_intake_base(tmp_path / "renamed", session_id="g98__innocent")
    path.with_suffix(".provenance.json").unlink()

    with pytest.raises(DataSourceGateError, match="bare CSV files are never trusted"):
        load_vap_dataframe(path, data_kind="real")


def test_synthetic_mode_rejects_real_provenance(tmp_path: Path) -> None:
    path, _ = _make_intake_base(tmp_path / "base", session_id="g98__innocent")

    with pytest.raises(DataSourceGateError, match="cannot be relabelled synthetic"):
        load_vap_dataframe(path, data_kind="synthetic")


def test_synthetic_mode_requires_hash_bound_provenance(tmp_path: Path) -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=4.0, n_participants=3, random_seed=7)
    )
    path = tmp_path / "synthetic.csv"
    frame_df.to_csv(path, index=False)

    with pytest.raises(DataSourceGateError, match="bare CSV files are never trusted"):
        load_vap_dataframe(path, data_kind="synthetic")

    sidecar = write_synthetic_provenance(
        path,
        generator_config=asdict(
            SyntheticConfig(
                n_sessions=1,
                duration_s=4.0,
                n_participants=3,
                random_seed=7,
            )
        ),
    )
    synthetic_provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    assert synthetic_provenance["format_version"] == 2
    assert synthetic_provenance["generator"]["id"] == (
        "hri_turn_taking.data.multiparty.generate_multiparty_dataset"
    )
    assert synthetic_provenance["generator"]["contract_version"] == 2
    assert "does not defend" in synthetic_provenance["trust_model"]
    loaded, report = load_and_audit_vap_source(path, data_kind="synthetic")
    assert loaded.equals(pd.read_csv(path, float_precision="round_trip"))
    assert report.source_artifacts[0]["data_sha256"] == _sha256(path)
    assert report.source_artifacts[0]["provenance_sha256"] == _sha256(sidecar)

    tampered = pd.read_csv(path)
    tampered.loc[0, "P1_asd_prob"] = 0.123456
    tampered.to_csv(path, index=False)
    with pytest.raises(DataSourceGateError, match="csv_sha256 does not match"):
        load_vap_dataframe(path, data_kind="synthetic")


def test_stale_synthetic_generator_v1_requires_regeneration(tmp_path: Path) -> None:
    config = SyntheticConfig(
        n_sessions=1, duration_s=1.0, n_participants=3, random_seed=17
    )
    frame_df = generate_multiparty_dataset(config)
    path = tmp_path / "synthetic.csv"
    frame_df.to_csv(path, index=False)
    sidecar = write_synthetic_provenance(path, generator_config=asdict(config))
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["generator"]["contract_version"] = 1
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="stale.*re-run vap-generate"):
        load_vap_dataframe(path, data_kind="synthetic")


def test_stale_real_intake_v1_requires_crosswalk_rebuild(tmp_path: Path) -> None:
    path, _ = _make_intake_base(tmp_path / "base")
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["format_version"] = 1
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="stale real provenance v1"):
        load_vap_dataframe(path, data_kind="real")


@pytest.mark.parametrize("bad_version", [2.0, True])
def test_real_provenance_version_requires_exact_integer(
    tmp_path: Path, bad_version: object
) -> None:
    path, _ = _make_intake_base(tmp_path / "base")
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["format_version"] = bad_version
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="format_version must be 2"):
        load_vap_dataframe(path, data_kind="real")


def test_real_csv_binding_rejects_float_row_count(tmp_path: Path) -> None:
    path, _ = _make_intake_base(tmp_path / "base")
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["n_rows"] = float(provenance["n_rows"])
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="n_rows does not bind exactly"):
        load_vap_dataframe(path, data_kind="real")


def test_pose_replay_rejects_bool_disguised_as_zero_window_bound(
    tmp_path: Path,
) -> None:
    path, _ = _make_intake_base(tmp_path / "base")
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    windows = (
        provenance["accepted_window"],
        provenance["identity_contract"]["accepted_window"],
    )
    for owner in windows:
        owner["local_start_inclusive"] = False
        owner["logical_start_inclusive"] = False
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(
        DataSourceGateError, match="(identity_contract|accepted_window) differs"
    ):
        load_vap_dataframe(path, data_kind="real")


@pytest.mark.parametrize("bad_version", [3.0, True])
def test_feature_manifest_version_requires_exact_integer(
    tmp_path: Path, bad_version: object
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    manifest_path = feature_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = bad_version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AssemblyGateError, match="format_version must be 3"):
        assemble_session(base_path, [feature_path], tmp_path / "out.csv")


@pytest.mark.parametrize(
    ("field", "bad_version", "message"),
    [
        ("format_version", 2.0, "format_version must be 2"),
        ("format_version", True, "format_version must be 2"),
        ("contract_version", 2.0, "generator contract is stale"),
        ("contract_version", True, "generator contract is stale"),
    ],
)
def test_synthetic_versions_require_exact_integers(
    tmp_path: Path, field: str, bad_version: object, message: str
) -> None:
    config = SyntheticConfig(
        n_sessions=1, duration_s=1.0, n_participants=3, random_seed=19
    )
    frame_df = generate_multiparty_dataset(config)
    path = tmp_path / "synthetic.csv"
    frame_df.to_csv(path, index=False)
    sidecar = write_synthetic_provenance(path, generator_config=asdict(config))
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    if field == "format_version":
        provenance[field] = bad_version
    else:
        provenance["generator"][field] = bad_version
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match=message):
        load_vap_dataframe(path, data_kind="synthetic")


def test_real_loader_replays_pose_after_shallow_csv_rebinding(tmp_path: Path) -> None:
    path, _ = _make_intake_base(tmp_path / "base")
    tampered = pd.read_csv(path)
    tampered.loc[tampered.index[0], "P2_head_yaw_deg"] = 123.0
    tampered.to_csv(path, index=False)
    _refresh_real_csv_binding(path)

    with pytest.raises(DataSourceGateError, match="differ from the sealed deterministic"):
        load_vap_dataframe(path, data_kind="real")


def test_real_csv_parser_preserves_shortest_binary64_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "round_trip.csv"
    value = 31.662214011816914
    # The default Pandas C parser only exhibits the neighbouring-ULP failure
    # for this value in a multi-row parse; keep the leading row so the test
    # fails if `_read_csv` silently drops `float_precision="round_trip"`.
    path.write_text(f"pose\n0.0\n{value!r}\n", encoding="utf-8")

    parsed = assembly_module._read_csv(path, error_type=DataSourceGateError)

    assert parsed.loc[1, "pose"] == value


def test_real_loader_replays_sealed_identity_contract(tmp_path: Path) -> None:
    path, _ = _make_intake_base(tmp_path / "base")
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["identity_contract"]["person_mapping"][1]["participant"] = "B"
    provenance["identity_contract"]["person_mapping"][1]["slot"] = "P3"
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="identity_contract differs from"):
        load_vap_dataframe(path, data_kind="real")


def test_synthetic_writer_cannot_resign_schema_valid_tampered_csv(tmp_path: Path) -> None:
    config = SyntheticConfig(
        n_sessions=1,
        duration_s=4.0,
        n_participants=3,
        random_seed=17,
    )
    frame_df = generate_multiparty_dataset(config)
    path = tmp_path / "synthetic.csv"
    frame_df.to_csv(path, index=False)
    sidecar = write_synthetic_provenance(path, generator_config=asdict(config))
    tampered = pd.read_csv(path)
    tampered.loc[tampered.index[0], "P1_asd_prob"] = 0.123456
    tampered.to_csv(path, index=False)
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    # Even refreshing all shallow CSV bindings cannot counterfeit the fixed
    # generator identity: loader replay must still compare actual values.
    reparsed = pd.read_csv(path)
    provenance.update(
        {
            "csv_sha256": _sha256(path),
            "columns": reparsed.columns.tolist(),
            "dtypes": {column: str(reparsed[column].dtype) for column in reparsed},
            "n_rows": len(reparsed),
        }
    )
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="deterministic replay"):
        load_vap_dataframe(path, data_kind="synthetic")

    with pytest.raises(DataSourceGateError, match="deterministic replay"):
        write_synthetic_provenance(
            path,
            generator_config=asdict(config),
            overwrite=True,
        )


def test_real_provenance_requires_complete_exclusive_column_partition(tmp_path: Path) -> None:
    path, _ = _make_intake_base(tmp_path / "base")
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["placeholder_columns"].remove("P2_asd_prob")
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="undeclared=.*P2_asd_prob"):
        load_vap_dataframe(path, data_kind="real")

    provenance["placeholder_columns"].append("P2_asd_prob")
    provenance["real_columns"].append("P2_asd_prob")
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="both real and placeholder"):
        load_vap_dataframe(path, data_kind="real")


def test_real_columns_require_exact_verifiable_column_sources(tmp_path: Path) -> None:
    path, _ = _make_intake_base(tmp_path / "base")
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["column_sources"].pop("P1_head_yaw_deg")
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="missing=.*P1_head_yaw_deg"):
        load_vap_dataframe(path, data_kind="real")


def test_real_loader_revalidates_confirmed_mapping_source(tmp_path: Path) -> None:
    path, _ = _make_intake_base(tmp_path / "base")
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    mapping_source = provenance["intake_sources"]["mapping"]
    mapping_path = Path(mapping_source["path"])
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["review_status"] = "proposed"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    # Even if an attacker edits the sidecar hash to match, semantic revalidation
    # must not let the sidecar self-certify an unconfirmed mapping.
    mapping_source["sha256"] = _sha256(mapping_path)
    mapping_source["size_bytes"] = mapping_path.stat().st_size
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="bound mapping.json review_status"):
        load_vap_dataframe(path, data_kind="real")


@pytest.mark.parametrize(
    ("extra_row", "message"),
    [
        ({"person": "2", "participant": "A"}, "duplicate person rows"),
        ({"person": "2", "participant": "B"}, "conflicting participants"),
    ],
)
def test_real_loader_replays_participant_map_duplicate_and_conflict_gates(
    tmp_path: Path,
    extra_row: dict[str, str],
    message: str,
) -> None:
    path, _ = _make_intake_base(tmp_path / message.replace(" ", "_"))
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    source = provenance["intake_sources"]["participant_map"]
    participant_map_path = Path(source["path"])
    map_df = pd.read_csv(participant_map_path)
    map_df = pd.concat([map_df, pd.DataFrame([extra_row])], ignore_index=True)
    map_df.to_csv(participant_map_path, index=False)
    source["sha256"] = _sha256(participant_map_path)
    source["size_bytes"] = participant_map_path.stat().st_size
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match=message):
        load_vap_dataframe(path, data_kind="real")


def test_feature_assembly_requires_verifiable_manifest(tmp_path: Path) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    feature_path.with_suffix(".manifest.json").unlink()

    with pytest.raises(AssemblyGateError, match="requires a verifiable source manifest"):
        assemble_session(base_path, [feature_path], tmp_path / "out.csv")


def test_feature_manifest_verifies_upstream_source_hash(tmp_path: Path) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    feature_path.with_suffix(".source.csv").write_text("tampered", encoding="utf-8")

    with pytest.raises(AssemblyGateError, match="source hash does not match"):
        assemble_session(base_path, [feature_path], tmp_path / "out.csv")


def test_feature_manifest_rejects_feature_csv_as_its_own_upstream(tmp_path: Path) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    manifest_path = feature_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    own_source = {"path": feature_path.name, "sha256": _sha256(feature_path)}
    manifest["source_files"] = [own_source]
    manifest["column_sources"]["P1_vad"].update(
        {"source_path": feature_path.name, "source_sha256": _sha256(feature_path)}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AssemblyGateError, match="not the generated feature/data artifact"):
        assemble_session(base_path, [feature_path], tmp_path / "out.csv")


def test_feature_manifest_rejects_swapped_participant_vad_binding(tmp_path: Path) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    swapped = {
        "P1_vad": {"participant": "A", "slot": "P2"},
        "P2_vad": {"participant": "R", "slot": "P1"},
        "P3_vad": {"participant": "B", "slot": "P3"},
    }
    feature_path = _write_feature(
        tmp_path / "vad.csv",
        base_df,
        ["P1_vad", "P2_vad", "P3_vad"],
        column_bindings=swapped,
    )

    with pytest.raises(AssemblyGateError, match="does not match base participant/slot"):
        assemble_session(base_path, [feature_path], tmp_path / "out.csv")


def test_feature_manifest_rejects_actual_swapped_participant_vad_values(
    tmp_path: Path,
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(
        tmp_path / "vad.csv",
        base_df,
        ["P1_vad", "P2_vad", "P3_vad"],
    )
    swapped = pd.read_csv(feature_path)
    p1 = swapped["P1_vad"].copy()
    swapped["P1_vad"] = swapped["P2_vad"]
    swapped["P2_vad"] = p1
    swapped.to_csv(feature_path, index=False)
    # Refresh only the generated feature hash. The independently bound upstream
    # participant columns remain unchanged, so exact value replay must catch this.
    _refresh_feature_hash(feature_path, sync_source=False)

    with pytest.raises(AssemblyGateError, match="feature values do not match source_column"):
        assemble_session(base_path, [feature_path], tmp_path / "out.csv")


def test_feature_manifest_rejects_coordinated_vad_value_and_source_column_swap(
    tmp_path: Path,
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(
        tmp_path / "vad.csv",
        base_df,
        ["P1_vad", "P2_vad", "P3_vad"],
    )
    swapped = pd.read_csv(feature_path)
    p1 = swapped["P1_vad"].copy()
    swapped["P1_vad"] = swapped["P2_vad"]
    swapped["P2_vad"] = p1
    swapped.to_csv(feature_path, index=False)

    manifest_path = feature_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_csv_sha256"] = _sha256(feature_path)
    manifest["column_sources"]["P1_vad"]["source_column"] = "P2_vad"
    manifest["column_sources"]["P2_vad"]["source_column"] = "P1_vad"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output = tmp_path / "out.csv"
    with pytest.raises(
        AssemblyGateError, match="source_column must exactly equal the canonical output"
    ):
        assemble_session(base_path, [feature_path], output)
    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()


def test_feature_manifest_rejects_cross_participant_pose_column_as_asd_source(
    tmp_path: Path,
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(
        tmp_path / "asd.csv", base_df, ["P1_asd_prob"]
    )
    manifest_path = feature_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = feature_path.with_suffix(".source.csv")
    source_df = pd.read_csv(source_path).rename(
        columns={"P1_asd_prob": "P2_pose_valid"}
    )
    source_df.to_csv(source_path, index=False)
    source_hash = _sha256(source_path)
    manifest["source_files"][0]["sha256"] = source_hash
    manifest["column_sources"]["P1_asd_prob"].update(
        {
            "source_column": "P2_pose_valid",
            "source_sha256": source_hash,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        AssemblyGateError, match="source_column must exactly equal the canonical output"
    ):
        assemble_session(base_path, [feature_path], tmp_path / "out.csv")


def test_feature_manifest_cannot_promote_a_base_placeholder_as_real_input(
    tmp_path: Path,
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(
        tmp_path / "asd.csv", base_df, ["P1_asd_prob"]
    )
    feature_df = pd.read_csv(feature_path)
    feature_df["P1_asd_prob"] = base_df["P1_asd_prob"].to_numpy()
    feature_df.to_csv(feature_path, index=False)
    manifest_path = feature_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_hash = _sha256(base_path)
    manifest["feature_csv_sha256"] = _sha256(feature_path)
    manifest["source_files"] = [
        {"path": str(base_path.resolve()), "sha256": base_hash}
    ]
    manifest["column_sources"]["P1_asd_prob"].update(
        {
            "source_path": str(base_path.resolve()),
            "source_sha256": base_hash,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "must_not_exist.csv"

    with pytest.raises(
        AssemblyGateError,
        match="generated feature/data artifact|generated VAP|forbidden",
    ):
        assemble_session(base_path, [feature_path], output)

    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()


def test_feature_manifest_cannot_promote_a_renamed_base_copy_without_sidecar(
    tmp_path: Path,
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "asd.csv", base_df, ["P1_asd_prob"])
    copied_base = tmp_path / "independent.csv"
    shutil.copyfile(base_path, copied_base)
    manifest_path = feature_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied_hash = _sha256(copied_base)
    manifest["source_files"] = [
        {"path": str(copied_base.resolve()), "sha256": copied_hash}
    ]
    manifest["column_sources"]["P1_asd_prob"].update(
        {
            "source_path": str(copied_base.resolve()),
            "source_sha256": copied_hash,
        }
    )
    feature_df = pd.read_csv(feature_path)
    feature_df["P1_asd_prob"] = base_df["P1_asd_prob"].to_numpy()
    feature_df.to_csv(feature_path, index=False)
    manifest["feature_csv_sha256"] = _sha256(feature_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "must_not_exist.csv"

    with pytest.raises(AssemblyGateError, match="copy|copied generated VAP"):
        assemble_session(base_path, [feature_path], output)

    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()


def test_chained_assembly_rejects_copied_historical_generated_base(
    tmp_path: Path,
) -> None:
    original_base, base_df = _make_intake_base(tmp_path / "base")
    first_feature = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    first_output = tmp_path / "first.csv"
    assemble_session(original_base, [first_feature], first_output)

    first_df = pd.read_csv(first_output)
    second_feature = _write_feature(
        tmp_path / "asd.csv", first_df, ["P1_asd_prob"]
    )
    copied_historical = tmp_path / "historical.csv"
    shutil.copyfile(original_base, copied_historical)
    copied_hash = _sha256(copied_historical)
    manifest_path = second_feature.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_files"] = [
        {"path": str(copied_historical.resolve()), "sha256": copied_hash}
    ]
    manifest["column_sources"]["P1_asd_prob"].update(
        {
            "source_path": str(copied_historical.resolve()),
            "source_sha256": copied_hash,
        }
    )
    second_df = pd.read_csv(second_feature)
    second_df["P1_asd_prob"] = first_df["P1_asd_prob"].to_numpy()
    second_df.to_csv(second_feature, index=False)
    manifest["feature_csv_sha256"] = _sha256(second_feature)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "must_not_exist.csv"

    with pytest.raises(AssemblyGateError, match="copied generated VAP wide table"):
        assemble_session(first_output, [second_feature], output)

    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()


def test_chained_assembly_rejects_cross_participant_upstream_reuse_before_writing(
    tmp_path: Path,
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    first_feature = _write_feature(tmp_path / "p1.csv", base_df, ["P1_vad"])
    first_output = tmp_path / "first.csv"
    assemble_session(base_path, [first_feature], first_output)

    second_feature = _write_feature(tmp_path / "p2.csv", base_df, ["P2_vad"])
    second_manifest_path = second_feature.with_suffix(".manifest.json")
    second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    first_manifest = json.loads(
        first_feature.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    first_source_path = Path(first_manifest["source_files"][0]["path"])
    if not first_source_path.is_absolute():
        first_source_path = first_feature.parent / first_source_path
    first_source_hash = _sha256(first_source_path)
    second_manifest["source_files"] = [
        {"path": str(first_source_path.resolve()), "sha256": first_source_hash}
    ]
    second_manifest["column_sources"]["P2_vad"].update(
        {
            "source_path": str(first_source_path.resolve()),
            "source_sha256": first_source_hash,
            "source_column": "P1_vad",
            # Keep the claimed participant internally consistent with the
            # canonical output; the canonical source-column check must still
            # reject the cross-participant ownership forgery.
            "source_participant": "A",
        }
    )
    p1_values = pd.read_csv(first_source_path)["P1_vad"]
    second_df = pd.read_csv(second_feature)
    second_df["P2_vad"] = p1_values
    second_df.to_csv(second_feature, index=False)
    second_manifest["feature_csv_sha256"] = _sha256(second_feature)
    second_manifest_path.write_text(json.dumps(second_manifest), encoding="utf-8")

    rejected_output = tmp_path / "must_not_exist.csv"
    with pytest.raises(
        AssemblyGateError, match="source_column must exactly equal the canonical output"
    ):
        assemble_session(first_output, [second_feature], rejected_output)
    assert not rejected_output.exists()
    assert not rejected_output.with_suffix(".provenance.json").exists()


def test_assembly_rolls_back_csv_when_sidecar_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    output = tmp_path / "assembled.csv"
    sidecar = output.with_suffix(".provenance.json")
    original_link = assembly_module.os.link

    def fail_sidecar_link(source: Path, target: Path) -> None:
        if Path(target) == sidecar:
            raise OSError("injected assembly sidecar failure")
        original_link(source, target)

    monkeypatch.setattr(assembly_module.os, "link", fail_sidecar_link)
    with pytest.raises(AssemblyGateError, match="Could not publish assembly output"):
        assemble_session(base_path, [feature_path], output)

    assert not output.exists()
    assert not sidecar.exists()
    assert not list(output.parent.glob(f".{output.name}.tmp.*.csv"))
    assert not list(sidecar.parent.glob(f".{sidecar.name}.*.json.tmp"))


@pytest.mark.parametrize("concurrent_target", ["csv", "sidecar"])
def test_assembly_never_clobbers_a_concurrent_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concurrent_target: str
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    output = tmp_path / "assembled.csv"
    sidecar = output.with_suffix(".provenance.json")
    target = output if concurrent_target == "csv" else sidecar
    original_link = assembly_module.os.link
    injected = False

    def create_concurrent_owner(source: Path, destination: Path) -> None:
        nonlocal injected
        if Path(destination) == target and not injected:
            target.write_bytes(b"CONCURRENT_OWNER")
            injected = True
        original_link(source, destination)

    monkeypatch.setattr(assembly_module.os, "link", create_concurrent_owner)
    with pytest.raises(AssemblyGateError, match="foreign replacement preserved"):
        assemble_session(base_path, [feature_path], output)

    assert target.read_bytes() == b"CONCURRENT_OWNER"
    other = sidecar if target == output else output
    assert not other.exists()


def test_assembly_rolls_back_pair_when_bound_delivery_changes_during_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    output = tmp_path / "assembled.csv"
    sidecar = output.with_suffix(".provenance.json")
    base_provenance = json.loads(
        base_path.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    world_path = Path(base_provenance["intake_sources"]["world_persons"]["path"])
    original_publish = assembly_module._publish_no_clobber
    changed = False

    def mutate_then_publish(staged: Path, destination: Path) -> None:
        nonlocal changed
        if not changed:
            world_path.write_text(
                world_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            changed = True
        original_publish(staged, destination)

    monkeypatch.setattr(assembly_module, "_publish_no_clobber", mutate_then_publish)
    with pytest.raises(AssemblyGateError, match="final source replay"):
        assemble_session(base_path, [feature_path], output)

    assert not output.exists()
    assert not sidecar.exists()


def test_assembly_rolls_back_csv_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    output = tmp_path / "assembled.csv"
    sidecar = output.with_suffix(".provenance.json")
    original_publish = assembly_module._publish_no_clobber
    calls = 0

    def interrupt_second_publish(staged: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        original_publish(staged, destination)

    monkeypatch.setattr(assembly_module, "_publish_no_clobber", interrupt_second_publish)
    with pytest.raises(KeyboardInterrupt):
        assemble_session(base_path, [feature_path], output)

    assert not output.exists()
    assert not sidecar.exists()


@pytest.mark.parametrize("publish_number", [1, 2])
def test_assembly_rolls_back_when_link_succeeds_then_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publish_number: int,
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    output = tmp_path / "assembled.csv"
    sidecar = output.with_suffix(".provenance.json")
    original_publish = assembly_module._publish_no_clobber
    calls = 0

    def interrupt_after_link(staged: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        original_publish(staged, destination)
        if calls == publish_number:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        assembly_module,
        "_publish_no_clobber",
        interrupt_after_link,
    )
    with pytest.raises(KeyboardInterrupt):
        assemble_session(base_path, [feature_path], output)
    assert not output.exists()
    assert not sidecar.exists()
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_assembly_cleans_first_temp_when_second_reservation_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    output = tmp_path / "assembled.csv"
    original_reserve = assembly_module._reserve_staged_path
    calls = 0

    def interrupt_second(directory: Path, *, prefix: str, suffix: str) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original_reserve(directory, prefix=prefix, suffix=suffix)

    monkeypatch.setattr(
        assembly_module,
        "_reserve_staged_path",
        interrupt_second,
    )
    with pytest.raises(KeyboardInterrupt):
        assemble_session(base_path, [feature_path], output)
    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()
    assert not list(tmp_path.glob(".*.tmp.*"))


@pytest.mark.parametrize("target_name", ["csv", "sidecar"])
@pytest.mark.parametrize("mutation", ["in_place", "replacement"])
def test_assembly_rechecks_owned_pair_after_final_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    mutation: str,
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    output = (tmp_path / "assembled.csv").resolve()
    sidecar = output.with_suffix(".provenance.json")
    original_load = assembly_module._load_dataset_file

    def mutate_after_output_load(path: Path, *, data_kind: str):
        loaded = original_load(path, data_kind=data_kind)
        if Path(path).resolve() == output:
            target = output if target_name == "csv" else sidecar
            if mutation == "replacement":
                target.unlink()
                target.write_bytes(b"FOREIGN_REPLACEMENT")
            else:
                target.write_bytes(target.read_bytes() + b"\n")
        return loaded

    monkeypatch.setattr(
        assembly_module,
        "_load_dataset_file",
        mutate_after_output_load,
    )
    with pytest.raises(AssemblyGateError, match="final source replay"):
        assemble_session(base_path, [feature_path], output)
    if mutation == "replacement" and os.name != "nt":
        target = output if target_name == "csv" else sidecar
        assert target.read_bytes() == b"FOREIGN_REPLACEMENT"
        other = sidecar if target == output else output
        assert not other.exists()
    else:
        assert not output.exists()
        assert not sidecar.exists()


@pytest.mark.parametrize("target_name", ["csv", "sidecar"])
def test_assembly_held_snapshot_rejects_same_bytes_new_inode_before_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    output = (tmp_path / "assembled.csv").resolve()
    sidecar = output.with_suffix(".provenance.json")
    target = output if target_name == "csv" else sidecar
    replacement = tmp_path / f"same-bytes-{target_name}.replacement"
    original_hash = assembly_module._hash_held_handle
    attempted = False
    replacement_denied = False
    replacement_identity: tuple[int, int] | None = None
    payload = b""

    def replace_after_fstat_before_hash(handle: object, path: Path) -> str:
        nonlocal attempted, replacement_denied, replacement_identity, payload
        if Path(path).resolve() == target and not attempted:
            attempted = True
            payload = target.read_bytes()
            replacement.write_bytes(payload)
            replacement_identity = assembly_module._file_identity(replacement)
            try:
                os.replace(replacement, target)
            except PermissionError as exc:
                replacement_denied = True
                replacement.unlink(missing_ok=True)
                raise assembly_module._HeldFileSnapshotError(
                    "held Windows handle denied same-byte inode replacement"
                ) from exc
        return original_hash(handle, path)

    monkeypatch.setattr(
        assembly_module,
        "_hash_held_handle",
        replace_after_fstat_before_hash,
    )
    with pytest.raises(AssemblyGateError, match="final source replay"):
        assemble_session(base_path, [feature_path], output)
    assert attempted
    if os.name == "nt":
        assert replacement_denied
        assert not output.exists()
        assert not sidecar.exists()
    else:
        assert not replacement_denied
        assert replacement_identity is not None
        assert target.read_bytes() == payload
        assert assembly_module._file_identity(target) == replacement_identity
        other = sidecar if target == output else output
        assert not other.exists()


def test_assembly_quarantine_delete_never_unlinks_post_identity_foreign_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    output = (tmp_path / "assembled.csv").resolve()
    original_load = assembly_module._load_dataset_file
    original_delete = assembly_module._delete_exact_quarantine
    recovery_path: Path | None = None

    def fail_output_load(path: Path, *, data_kind: str):
        if Path(path).resolve() == output:
            raise DataSourceGateError("injected postpublish failure")
        return original_load(path, data_kind=data_kind)

    def replace_after_quarantine_identity(
        recovery: Path,
        identity: tuple[int, int],
    ) -> bool:
        nonlocal recovery_path
        if recovery.name.startswith(f".{output.name}.recovery."):
            recovery_path = recovery
            recovery.unlink()
            recovery.write_bytes(b"FOREIGN_AFTER_IDENTITY")
        return original_delete(recovery, identity)

    monkeypatch.setattr(assembly_module, "_load_dataset_file", fail_output_load)
    monkeypatch.setattr(
        assembly_module,
        "_delete_exact_quarantine",
        replace_after_quarantine_identity,
    )
    with pytest.raises(AssemblyGateError, match="recovery="):
        assemble_session(base_path, [feature_path], output)
    assert recovery_path is not None
    assert recovery_path.read_bytes() == b"FOREIGN_AFTER_IDENTITY"
    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()


def test_assembly_preserves_foreign_replacement_of_staging_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    output = tmp_path / "assembled.csv"
    replacement: Path | None = None

    def replace_staging_then_interrupt(staged: Path, destination: Path) -> None:
        nonlocal replacement
        del destination
        staged.unlink()
        staged.write_bytes(b"FOREIGN_STAGING_OWNER")
        replacement = staged
        raise KeyboardInterrupt

    monkeypatch.setattr(
        assembly_module,
        "_publish_no_clobber",
        replace_staging_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        assemble_session(base_path, [feature_path], output)
    assert replacement is not None
    assert replacement.read_bytes() == b"FOREIGN_STAGING_OWNER"
    replacement.unlink()
    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()


def test_assembly_rejects_staged_csv_tamper_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(tmp_path / "vad.csv", base_df, ["P1_vad"])
    output = tmp_path / "assembled.csv"
    sidecar = output.with_suffix(".provenance.json")
    original_publish = assembly_module._publish_no_clobber
    tampered = False

    def tamper_then_publish(staged: Path, destination: Path) -> None:
        nonlocal tampered
        if Path(destination) == output and not tampered:
            staged.write_bytes(staged.read_bytes() + b"\n")
            tampered = True
        original_publish(staged, destination)

    monkeypatch.setattr(assembly_module, "_publish_no_clobber", tamper_then_publish)
    with pytest.raises(AssemblyGateError, match="final source replay"):
        assemble_session(base_path, [feature_path], output)

    assert not output.exists()
    assert not sidecar.exists()
    assert not list(output.parent.glob(f".{output.name}.tmp.*.csv"))


def test_assembly_staging_paths_are_writer_private(tmp_path: Path) -> None:
    first = assembly_module._reserve_staged_path(
        tmp_path, prefix=".assembly.tmp.", suffix=".csv"
    )
    second = assembly_module._reserve_staged_path(
        tmp_path, prefix=".assembly.tmp.", suffix=".csv"
    )
    try:
        assert first != second
        assert first.is_file()
        assert second.is_file()
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


def test_feature_manifest_rejects_noncausal_model_input(tmp_path: Path) -> None:
    base_path, base_df = _make_intake_base(tmp_path / "base")
    feature_path = _write_feature(
        tmp_path / "asd.csv",
        base_df,
        ["P1_asd_prob"],
        causal=False,
        window_alignment="offline",
        lookahead_s=None,
    )

    with pytest.raises(AssemblyGateError, match="model_input must declare causal=true"):
        assemble_session(base_path, [feature_path], tmp_path / "out.csv")


def test_real_loader_revalidates_assembled_feature_manifest(tmp_path: Path) -> None:
    output = _assemble_trainable_session(tmp_path, "g01__s1")
    sidecar = output.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["feature_sources"][0]["causality"] = {
        "causal": True,
        "lookahead_s": 0.1,
        "window_alignment": "trailing",
    }
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="causality does not match"):
        load_vap_dataframe(output, data_kind="real")

    other = _assemble_trainable_session(tmp_path / "manifest", "g02__s1")
    other_sidecar = other.with_suffix(".provenance.json")
    other_provenance = json.loads(other_sidecar.read_text(encoding="utf-8"))
    manifest_path = Path(other_provenance["feature_sources"][0]["manifest_path"])
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(DataSourceGateError, match="source hash does not match"):
        load_vap_dataframe(other, data_kind="real")


def test_real_loader_rejects_reanchored_assembly_output_path(tmp_path: Path) -> None:
    output = _assemble_trainable_session(tmp_path, "g01__s1")
    sidecar = output.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["assembly"]["output_csv"] = str(
        (tmp_path / "claimed_elsewhere.csv").resolve()
    )
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="assembly.output_csv"):
        load_vap_dataframe(output, data_kind="real")


def test_structural_mapping_and_d3_gates_are_never_readiness_overrides(
    tmp_path: Path,
) -> None:
    path, _ = _make_intake_base(tmp_path / "base")
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["review_status"] = "proposed"
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="review_status"):
        audit_vap_source(path, data_kind="real", feature_set="head_pose_only")

    provenance["review_status"] = "confirmed"
    provenance["offset_quality"]["offset_s"] = -9.0
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="does not match offset_applied_s"):
        audit_vap_source(path, data_kind="real", feature_set="head_pose_only")


def test_d3_gate_is_bound_to_original_files_and_timebase_keys(tmp_path: Path) -> None:
    path, _ = _make_intake_base(tmp_path / "tampered" / "base")
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    offset_path = Path(provenance["d3_sources"]["offset"]["path"])
    offset_path.write_text(json.dumps({"streams": {"6cam": 99.0}}), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="source hash does not match"):
        load_vap_dataframe(path, data_kind="real")

    other, _ = _make_intake_base(tmp_path / "timebase" / "base")
    other_sidecar = other.with_suffix(".provenance.json")
    other_provenance = json.loads(other_sidecar.read_text(encoding="utf-8"))
    other_provenance["offset_quality"]["timebase_keys"] = {"6cam": "invented_axis"}
    other_sidecar.write_text(json.dumps(other_provenance), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="timebase_keys does not match"):
        load_vap_dataframe(other, data_kind="real")


def test_real_loader_rejects_self_consistent_rgb_offset_substitution(
    tmp_path: Path,
) -> None:
    path, _ = _make_intake_base(tmp_path / "base")
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    offset_path = Path(provenance["d3_sources"]["offset"]["path"])
    quality_path = Path(provenance["d3_sources"]["offset_quality"]["path"])

    offset_path.write_text(
        json.dumps({"streams": {"participants_rgb": -1.2}}),
        encoding="utf-8",
    )
    quality_path.write_text(
        json.dumps(
            {
                "convention": "shared_time = stream_local_time + offset_s",
                "timebase_keys": {"participants_rgb": "rgb_frame_axis"},
                "quality": {
                    "participants_rgb": {
                        "offset_s": -1.2,
                        "self_check_ok": True,
                        "status": "ok",
                        "error_bound_ms": 33.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    provenance["stream_key"] = "participants_rgb"
    provenance["intake_config"]["stream_key"] = "participants_rgb"
    provenance["offset_quality"]["stream"] = "participants_rgb"
    provenance["offset_quality"]["timebase_keys"] = {
        "participants_rgb": "rgb_frame_axis"
    }
    provenance["d3_sources"]["offset"]["sha256"] = _sha256(offset_path)
    provenance["d3_sources"]["offset_quality"]["sha256"] = _sha256(quality_path)
    for source_name, source_path in (
        ("offset", offset_path),
        ("offset_quality", quality_path),
    ):
        provenance["d3_sources"][source_name]["size_bytes"] = source_path.stat().st_size
        provenance["intake_sources"][source_name]["sha256"] = _sha256(source_path)
        provenance["intake_sources"][source_name]["size_bytes"] = source_path.stat().st_size
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="stream_key must be '6cam'"):
        load_vap_dataframe(path, data_kind="real")


def test_group_id_rejects_trailing_whitespace_instead_of_forming_new_group(
    tmp_path: Path,
) -> None:
    path, _ = _make_intake_base(tmp_path / "base", session_id="g98__session-two")
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["group_id"] = "g01 "
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match="normalised identifier"):
        load_vap_dataframe(path, data_kind="real")


def test_audit_selected_feature_readiness_ignores_unused_placeholders(tmp_path: Path) -> None:
    path, _ = _make_intake_base(tmp_path / "base")

    head_only = audit_vap_source(path, data_kind="real", feature_set="head_pose_only")
    multimodal = audit_vap_source(path, data_kind="real", feature_set="multimodal")

    assert not head_only.target_ready
    assert head_only.selected_feature_ready
    assert head_only.selected_feature_blockers == ()
    assert not multimodal.selected_feature_ready
    assert "P1_asd_prob" in " ".join(multimodal.selected_feature_blockers)


def test_multiparty_schema_gates_ids_vad_and_25_hz_timestamps(tmp_path: Path) -> None:
    _, base = _make_intake_base(tmp_path / "base")

    empty_id = base.copy()
    empty_id.loc[empty_id.index[0], "session_id"] = "  "
    with pytest.raises(ValueError, match="non-empty"):
        validate_multiparty_dataframe(empty_id)

    fractional_vad = base.copy()
    fractional_vad.loc[fractional_vad.index[0], "P1_vad"] = 0.5
    with pytest.raises(ValueError, match="P1_vad must be binary"):
        validate_multiparty_dataframe(fractional_vad)

    absent_vad = base.copy()
    absent_vad.loc[absent_vad.index[0], "P4_vad"] = 1.0
    with pytest.raises(ValueError, match="absent slots"):
        validate_multiparty_dataframe(absent_vad)

    duplicate_time = base.copy()
    duplicate_time.loc[duplicate_time.index[1], "timestamp_s"] = duplicate_time.loc[
        duplicate_time.index[0], "timestamp_s"
    ]
    with pytest.raises(ValueError, match="strictly increasing and unique"):
        validate_multiparty_dataframe(duplicate_time)

    wrong_rate = base.copy()
    wrong_rate["timestamp_s"] *= 2.0
    with pytest.raises(ValueError, match="25 Hz"):
        validate_multiparty_dataframe(wrong_rate)


def _analysis_gold_report(
    parent: Path,
    *,
    scene: str,
    start_tick: int,
    end_tick: int,
    warning_speaker: str | None = None,
) -> dict[str, object]:
    sidecar = parent.with_suffix(".provenance.json")
    channels = {
        "A": {"kind": "identity_anchor", "correction_s": "0"},
        "B": {"kind": "identity", "correction_s": "0"},
        "C": {"kind": "constant_shift", "correction_s": "-0.022"},
        "R": {"kind": "identity", "correction_s": "0"},
    }
    decisions = {
        speaker: {
            "status": (
                "resolution_confirmed"
                if warning_speaker == speaker
                else "coarse_confirmed"
            ),
            "coarse_warning": warning_speaker == speaker,
        }
        for speaker in ("B", "C", "R")
    }
    return {
        "artifact_kind": analysis_base_module.audio_gold_alignment.ARTIFACT_KIND,
        "format_version": analysis_base_module.audio_gold_alignment.FORMAT_VERSION,
        "scene": scene,
        "counted_gold_ready": False,
        "sources": {
            "pose_csv": {
                "path": str(parent.resolve()),
                "sha256": _sha256(parent),
            },
            "pose_provenance": {
                "path": str(sidecar.resolve()),
                "sha256": _sha256(sidecar),
            },
        },
        "mapping_contract": {"channels": channels},
        "authorization": {
            "gold_vad_use_authorized": True,
            "authorized_tick_range": {
                "frame_rate_hz": 25,
                "start_inclusive": start_tick,
                "end_exclusive": end_tick,
                "n_ticks": end_tick - start_tick,
            },
        },
        "formal_window_replay": {"channel_decisions": decisions},
        "gates": {
            "alignment_prerequisite_pass": True,
            "counted_gold_ready": False,
        },
    }


def _analysis_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_id: str = "g01__s1",
    warning_speaker: str | None = "R",
    analysis_base_root: Path | None = None,
    descendant_output_paths: tuple[Path, ...] = (),
) -> tuple[
    analysis_base_module.AnalysisBaseAuthority,
    Path,
    pd.DataFrame,
    Path,
    dict[str, object],
]:
    parent, parent_df = _make_intake_base(
        tmp_path / "intake",
        session_id=session_id,
        offset_s=0.0,
    )
    parent_sidecar = json.loads(
        parent.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    parent_ticks = parent_sidecar["output_tick_range"]
    start = parent_ticks["start_inclusive"] + 5
    end = parent_ticks["end_exclusive"] - 7
    gold_root = tmp_path / "gold"
    gold_root.mkdir()
    gold_path = gold_root / f"{session_id}.audio_gold_alignment.json"
    gold_path.write_text('{"formal":"fixture"}\n', encoding="utf-8")
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    report = _analysis_gold_report(
        parent,
        scene=session_id,
        start_tick=start,
        end_tick=end,
        warning_speaker=warning_speaker,
    )
    authority = analysis_base_module.AnalysisBaseAuthority(
        pose_base_root=parent.parent,
        gold_alignment_root=gold_root,
        audio_manifest_root=audio_root,
        analysis_base_root=(analysis_base_root or (tmp_path / "analysis")),
        descendant_output_paths=descendant_output_paths,
        enforce_frozen_inventory=False,
    )
    monkeypatch.setattr(
        analysis_base_module,
        "_load_gold_alignment_strict",
        lambda path, *, authority: json.loads(json.dumps(report)),
    )
    monkeypatch.setattr(
        analysis_base_module,
        "_load_gold_alignment_sealed",
        lambda path, *, authority: json.loads(json.dumps(report)),
    )
    return authority, parent, parent_df, gold_path, report


def test_analysis_base_exact_integer_crop_and_warning_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, parent, parent_df, _, report = _analysis_fixture(
        tmp_path, monkeypatch
    )
    result = analysis_base_module.derive_analysis_base(
        "g01__s1",
        authority=authority,
    )
    ticks = report["authorization"]["authorized_tick_range"]
    parent_start = json.loads(
        parent.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )["output_tick_range"]["start_inclusive"]
    expected = parent_df.iloc[
        ticks["start_inclusive"] - parent_start : ticks["end_exclusive"] - parent_start
    ].reset_index(drop=True)

    assert result.frame_df.equals(expected)
    assert result.provenance["artifact_kind"] == "vap_real_analysis_base"
    assert result.provenance["trainable"] is False
    contract = result.provenance["analysis_window_contract"]
    assert contract["authorized_tick_range"] == ticks
    assert contract["row_slice"] == {
        "start_offset_inclusive": 5,
        "end_offset_exclusive": len(parent_df) - 7,
        "n_rows": len(parent_df) - 12,
    }
    assert contract["channel_alignment"]["R"] == {
        "mapping_kind": "identity",
        "correction_s": "0",
        "status": "resolution_confirmed",
        "coarse_warning": True,
    }
    assert contract["d3_quality"] == result.provenance["offset_quality"]
    assert contract["output_authority"]["canonical_analysis_base_csv"] == str(
        (authority.analysis_base_root / "g01__s1.csv").resolve()
    )


def test_analysis_base_requires_explicit_code_owned_output_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _, _, _, _ = _analysis_fixture(tmp_path, monkeypatch)
    canonical = authority.analysis_base_root / "g01__s1.csv"
    noncanonical = tmp_path / "attacker_chosen" / "g01__s1.csv"
    with pytest.raises(analysis_base_module.AnalysisBaseError, match="canonical path"):
        analysis_base_module.write_analysis_base(
            "g01__s1",
            noncanonical,
            authority=authority,
        )
    assert not noncanonical.parent.exists()

    analysis_base_module.write_analysis_base(
        "g01__s1",
        canonical,
        authority=authority,
    )
    with pytest.raises(DataSourceGateError, match="frozen ten-scene|canonical"):
        # No fixture monkeypatch of production authority is allowed.
        load_vap_dataframe(canonical, data_kind="real")


def test_default_analysis_descendant_authority_freezes_exact_first_matrix_paths() -> None:
    authority = analysis_base_module.default_analysis_base_authority()
    repository_root = Path(analysis_base_module.__file__).resolve().parents[3]
    formal_root = repository_root / "data" / "processed" / "formal_a_h_ah_v1"
    expected = tuple(
        (formal_root / f"{scene}.csv").resolve()
        for scene in analysis_base_module.EXPECTED_TICK_RANGES
    )

    assert len(expected) == 10
    assert len(set(expected)) == 10
    assert authority.descendant_output_paths == expected
    for scene, output in zip(
        analysis_base_module.EXPECTED_TICK_RANGES,
        expected,
        strict=True,
    ):
        analysis_base_module.validate_analysis_descendant_destination(scene, output)

    stale_real = repository_root / "data" / "processed" / "real" / "05_15__2.csv"
    arbitrary = formal_root / "05_15__2-copy.csv"
    for rejected in (stale_real, arbitrary):
        with pytest.raises(
            analysis_base_module.AnalysisBaseError,
            match="descendant_output_paths",
        ):
            analysis_base_module.validate_analysis_descendant_destination(
                "05_15__2",
                rejected,
            )


def test_analysis_descendant_requires_exact_live_authority_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorised = tmp_path / "descendant" / "g01__s1.csv"
    authority, _, _, _, _ = _analysis_fixture(
        tmp_path,
        monkeypatch,
        descendant_output_paths=(authorised,),
    )
    base = authority.analysis_base_root / "g01__s1.csv"
    result = analysis_base_module.write_analysis_base(
        "g01__s1", base, authority=authority
    )
    feature = _write_feature(
        tmp_path / "feature" / "vad.csv",
        result.frame_df,
        ["P1_vad"],
    )
    forbidden = tmp_path / "not_authorised" / "g01__s1.csv"
    with pytest.raises(AssemblyGateError, match="destination is not authorised"):
        assemble_session(
            base,
            [feature],
            forbidden,
            analysis_base_authority=authority,
        )
    assert not forbidden.parent.exists()

    assemble_session(
        base,
        [feature],
        authorised,
        analysis_base_authority=authority,
    )
    copied = tmp_path / "copied_descendant" / "g01__s1.csv"
    copied.parent.mkdir()
    shutil.copyfile(authorised, copied)
    copied_sidecar = copied.with_suffix(".provenance.json")
    provenance = json.loads(
        authorised.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    provenance["assembly"]["output_csv"] = str(copied.resolve())
    copied_sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(DataSourceGateError, match="descendant_output_paths"):
        load_vap_dataframe(
            copied,
            data_kind="real",
            analysis_base_authority=authority,
        )


def test_analysis_base_writer_load_and_assembled_lineage_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assembled = tmp_path / "assembled" / "g01__s1.csv"
    authority, _, _, _, _ = _analysis_fixture(
        tmp_path,
        monkeypatch,
        descendant_output_paths=(assembled,),
    )
    output = tmp_path / "analysis" / "g01__s1.csv"
    result = analysis_base_module.write_analysis_base(
        "g01__s1",
        output,
        authority=authority,
    )

    loaded = load_vap_dataframe(
        output,
        data_kind="real",
        analysis_base_authority=authority,
    )
    assert len(loaded) == len(result.frame_df)
    assert set(loaded["group_id"]) == {"g01"}
    report = audit_vap_source(
        output,
        data_kind="real",
        feature_set="head_pose_only",
        analysis_base_authority=authority,
    )
    assert report.structural_ready
    assert report.selected_feature_ready
    assert not report.target_ready

    feature = _write_feature(
        tmp_path / "feature" / "vad.csv",
        result.frame_df,
        ["P1_vad", "P2_vad", "P3_vad"],
    )
    assembled_result = assemble_session(
        output,
        [feature],
        assembled,
        analysis_base_authority=authority,
    )
    assert assembled_result.provenance["analysis_window_contract"] == (
        result.provenance["analysis_window_contract"]
    )
    load_vap_dataframe(
        assembled,
        data_kind="real",
        analysis_base_authority=authority,
    )


def test_analysis_base_separates_sealed_loads_from_explicit_raw_reaudit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _, _, _, report = _analysis_fixture(tmp_path, monkeypatch)
    strict_calls: list[Path] = []
    sealed_calls: list[Path] = []

    def strict(path: Path, *, authority: object) -> dict[str, object]:
        del authority
        strict_calls.append(Path(path))
        return json.loads(json.dumps(report))

    def sealed(path: Path, *, authority: object) -> dict[str, object]:
        del authority
        sealed_calls.append(Path(path))
        return json.loads(json.dumps(report))

    monkeypatch.setattr(analysis_base_module, "_load_gold_alignment_strict", strict)
    monkeypatch.setattr(analysis_base_module, "_load_gold_alignment_sealed", sealed)
    output = tmp_path / "analysis" / "g01__s1.csv"
    written = analysis_base_module.write_analysis_base(
        "g01__s1", output, authority=authority
    )
    modes = written.provenance["analysis_window_contract"]["validation_modes"]
    assert modes["strict_build_replay"]["performed"] is True
    assert modes["downstream_sealed_replay"]["raw_wav_decode_performed"] is False
    assert len(strict_calls) == 2
    assert len(sealed_calls) >= 2

    strict_calls.clear()
    sealed_calls.clear()
    load_vap_dataframe(
        output,
        data_kind="real",
        analysis_base_authority=authority,
    )
    assert strict_calls == []
    assert sealed_calls

    strict_calls.clear()
    sealed_calls.clear()
    analysis_base_module.reaudit_analysis_base(output, authority=authority)
    assert len(strict_calls) == 1
    assert sealed_calls


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("wrong_scene", "scene does not match"),
        ("bool_start", "exact JSON integer"),
        ("nan_end", "exact JSON integer"),
        ("end_inclusive_key", "must contain exactly"),
        ("out_of_bounds", "exceed the bound pose parent"),
    ],
)
def test_analysis_base_rejects_malformed_gold_window_before_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    authority, _, _, _, report = _analysis_fixture(tmp_path, monkeypatch)
    ticks = report["authorization"]["authorized_tick_range"]
    if mutation == "wrong_scene":
        report["scene"] = "g01__other"
    elif mutation == "bool_start":
        ticks["start_inclusive"] = True
    elif mutation == "nan_end":
        ticks["end_exclusive"] = float("nan")
    elif mutation == "end_inclusive_key":
        ticks["end_inclusive"] = ticks["end_exclusive"]
    else:
        ticks["end_exclusive"] = 10_000
        ticks["n_ticks"] = 10_000 - ticks["start_inclusive"]
    output = authority.analysis_base_root / "g01__s1.csv"
    with pytest.raises(analysis_base_module.AnalysisBaseError, match=match):
        analysis_base_module.write_analysis_base(
            "g01__s1", output, authority=authority
        )
    assert not output.parent.exists()


def test_analysis_base_rejects_gold_binding_to_copied_pose_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, parent, _, _, report = _analysis_fixture(tmp_path, monkeypatch)
    copied = tmp_path / "copied_parent.csv"
    shutil.copyfile(parent, copied)
    report["sources"]["pose_csv"] = {
        "path": str(copied.resolve()),
        "sha256": _sha256(copied),
    }
    with pytest.raises(analysis_base_module.AnalysisBaseError, match="exact pose parent"):
        analysis_base_module.derive_analysis_base("g01__s1", authority=authority)


def test_analysis_base_detects_parent_and_gold_changes_during_strict_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, parent, _, gold_path, report = _analysis_fixture(tmp_path, monkeypatch)

    def mutate_parent(path: Path, *, authority: object) -> dict[str, object]:
        del path, authority
        changed = pd.read_csv(parent)
        changed.loc[0, "P1_mar"] = 0.5
        changed.to_csv(parent, index=False)
        _refresh_real_csv_binding(parent)
        return json.loads(json.dumps(report))

    monkeypatch.setattr(
        analysis_base_module,
        "_load_gold_alignment_strict",
        mutate_parent,
    )
    with pytest.raises(analysis_base_module.AnalysisBaseError, match="parent.*changed"):
        analysis_base_module.derive_analysis_base("g01__s1", authority=authority)

    # Rebuild an isolated source pair for the manifest-side race.
    authority, _, _, gold_path, report = _analysis_fixture(
        tmp_path / "gold_race", monkeypatch
    )

    def mutate_gold(path: Path, *, authority: object) -> dict[str, object]:
        del authority
        path.write_text('{"changed":true}\n', encoding="utf-8")
        return json.loads(json.dumps(report))

    monkeypatch.setattr(
        analysis_base_module,
        "_load_gold_alignment_strict",
        mutate_gold,
    )
    with pytest.raises(analysis_base_module.AnalysisBaseError, match="changed while"):
        analysis_base_module.derive_analysis_base("g01__s1", authority=authority)
    assert gold_path.read_text(encoding="utf-8") == '{"changed":true}\n'


def test_analysis_base_hardlink_race_is_not_deleted_as_writer_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, parent, _, _, _ = _analysis_fixture(tmp_path, monkeypatch)
    output = tmp_path / "analysis" / "g01__s1.csv"
    original_snapshot = analysis_base_module._load_source_snapshot

    def race(
        scene: str,
        *,
        authority: object,
        strict_raw_replay: bool,
    ) -> object:
        snapshot = original_snapshot(
            scene,
            authority=authority,
            strict_raw_replay=strict_raw_replay,
        )
        if not strict_raw_replay and not output.exists():
            os.link(parent, output)
        return snapshot

    monkeypatch.setattr(analysis_base_module, "_load_source_snapshot", race)
    with pytest.raises(analysis_base_module.AnalysisBaseError, match="rollback"):
        analysis_base_module.write_analysis_base(
            "g01__s1", output, authority=authority
        )
    assert os.path.samefile(parent, output)
    assert not output.with_suffix(".provenance.json").exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("end_inclusive", "analysis_window_contract differs"),
        ("bool_tick", "analysis_window_contract differs"),
        ("correction", "analysis_window_contract differs"),
        ("warning", "analysis_window_contract differs"),
        ("d3", "analysis_window_contract differs"),
        ("scene", "does not bind exactly"),
    ],
)
def test_analysis_base_rejects_resigned_contract_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    authority, _, _, _, _ = _analysis_fixture(tmp_path, monkeypatch)
    output = tmp_path / "analysis" / "g01__s1.csv"
    analysis_base_module.write_analysis_base(
        "g01__s1",
        output,
        authority=authority,
    )
    sidecar = output.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    contract = provenance["analysis_window_contract"]
    if mutation == "end_inclusive":
        contract["authorized_tick_range"]["end_exclusive"] += 1
    elif mutation == "bool_tick":
        contract["authorized_tick_range"]["start_inclusive"] = True
    elif mutation == "correction":
        contract["channel_alignment"]["C"]["correction_s"] = "0"
    elif mutation == "warning":
        contract["channel_alignment"]["R"]["coarse_warning"] = False
    elif mutation == "d3":
        contract["d3_quality"]["error_bound_ms"] = 0.0
    else:
        provenance["session_id"] = "g01__other"
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(DataSourceGateError, match=match):
        load_vap_dataframe(
            output,
            data_kind="real",
            analysis_base_authority=authority,
        )


def test_analysis_base_rejects_resigned_csv_value_and_copied_parent_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _, _, _, _ = _analysis_fixture(tmp_path, monkeypatch)
    output = tmp_path / "analysis" / "g01__s1.csv"
    result = analysis_base_module.write_analysis_base(
        "g01__s1",
        output,
        authority=authority,
    )
    tampered = pd.read_csv(output)
    tampered.loc[0, "P2_head_yaw_deg"] = 123.0
    tampered.to_csv(output, index=False)
    _refresh_real_csv_binding(output)
    with pytest.raises(DataSourceGateError, match="exact deterministic parent row slice"):
        load_vap_dataframe(
            output,
            data_kind="real",
            analysis_base_authority=authority,
        )

    copy_case = tmp_path / "copy_source_case"
    rejected_output = copy_case / "must_not_exist.csv"
    copy_authority, _, _, _, _ = _analysis_fixture(
        copy_case,
        monkeypatch,
        descendant_output_paths=(rejected_output,),
    )
    clean = copy_authority.analysis_base_root / "g01__s1.csv"
    analysis_base_module.write_analysis_base(
        "g01__s1",
        clean,
        authority=copy_authority,
    )
    clean_df = pd.read_csv(clean, float_precision="round_trip")
    feature = _write_feature(
        tmp_path / "feature" / "asd.csv",
        clean_df,
        ["P1_asd_prob"],
    )
    copied = tmp_path / "copied_analysis.csv"
    shutil.copyfile(clean, copied)
    manifest_path = feature.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied_hash = _sha256(copied)
    manifest["source_files"] = [{"path": str(copied), "sha256": copied_hash}]
    manifest["column_sources"]["P1_asd_prob"].update(
        {"source_path": str(copied), "source_sha256": copied_hash}
    )
    feature_df = pd.read_csv(feature)
    feature_df["P1_asd_prob"] = clean_df["P1_asd_prob"]
    feature_df.to_csv(feature, index=False)
    manifest["feature_csv_sha256"] = _sha256(feature)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AssemblyGateError, match="copy|copied generated VAP"):
        assemble_session(
            clean,
            [feature],
            rejected_output,
            analysis_base_authority=copy_authority,
        )
    assert result.provenance["trainable"] is False


def test_analysis_base_rejects_copied_or_hardlinked_published_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _, _, _, _ = _analysis_fixture(tmp_path, monkeypatch)
    original = tmp_path / "analysis" / "g01__s1.csv"
    analysis_base_module.write_analysis_base(
        "g01__s1", original, authority=authority
    )
    copied = tmp_path / "copy" / "g01__s1.csv"
    copied.parent.mkdir()
    shutil.copyfile(original, copied)
    shutil.copyfile(
        original.with_suffix(".provenance.json"),
        copied.with_suffix(".provenance.json"),
    )
    copied_provenance = json.loads(
        copied.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    copied_provenance["analysis_base_csv"] = str(copied.resolve())
    copied.with_suffix(".provenance.json").write_text(
        json.dumps(copied_provenance), encoding="utf-8"
    )
    with pytest.raises(DataSourceGateError, match="live code-owned canonical"):
        load_vap_dataframe(
            copied,
            data_kind="real",
            analysis_base_authority=authority,
        )

    linked = tmp_path / "hardlink" / "g01__s1.csv"
    linked.parent.mkdir()
    os.link(original, linked)
    os.link(
        original.with_suffix(".provenance.json"),
        linked.with_suffix(".provenance.json"),
    )
    with pytest.raises(DataSourceGateError, match="canonical published path"):
        load_vap_dataframe(
            linked,
            data_kind="real",
            analysis_base_authority=authority,
        )


def test_analysis_base_no_clobber_partial_publish_and_baseexception_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _, _, _, _ = _analysis_fixture(tmp_path, monkeypatch)
    output = tmp_path / "analysis" / "g01__s1.csv"
    sidecar = output.with_suffix(".provenance.json")
    original_publish = analysis_base_module._publish_no_clobber
    calls = 0

    def interrupt_after_sidecar_link(staged: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        original_publish(staged, destination)
        if calls == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        analysis_base_module,
        "_publish_no_clobber",
        interrupt_after_sidecar_link,
    )
    with pytest.raises(KeyboardInterrupt):
        analysis_base_module.write_analysis_base(
            "g01__s1",
            output,
            authority=authority,
        )
    assert not output.exists()
    assert not sidecar.exists()
    assert not list(output.parent.glob(".*.tmp.*"))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("CONCURRENT_OWNER", encoding="utf-8")
    with pytest.raises(analysis_base_module.AnalysisBaseError, match="overwrite"):
        analysis_base_module.write_analysis_base(
            "g01__s1",
            output,
            authority=authority,
        )
    assert output.read_text(encoding="utf-8") == "CONCURRENT_OWNER"


def test_analysis_base_cleans_first_temp_if_second_reservation_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _, _, _, _ = _analysis_fixture(tmp_path, monkeypatch)
    output = tmp_path / "analysis" / "g01__s1.csv"
    original_reserve = analysis_base_module._reserve_staged_path
    calls = 0

    def interrupt_second(directory: Path, *, prefix: str, suffix: str) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original_reserve(directory, prefix=prefix, suffix=suffix)

    monkeypatch.setattr(
        analysis_base_module,
        "_reserve_staged_path",
        interrupt_second,
    )
    with pytest.raises(KeyboardInterrupt):
        analysis_base_module.write_analysis_base(
            "g01__s1", output, authority=authority
        )
    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()
    assert not list(output.parent.glob(".*.tmp.*"))


def test_analysis_base_rollback_never_deletes_replacement_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _, _, _, _ = _analysis_fixture(tmp_path, monkeypatch)
    output = tmp_path / "analysis" / "g01__s1.csv"
    original_rename = assembly_module.os.rename
    injected = False

    def fail_after_publish(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise analysis_base_module.AnalysisBaseError("injected postpublish failure")

    def quarantine_then_install_foreign(source: Path, destination: Path) -> None:
        nonlocal injected
        original_rename(source, destination)
        if Path(source).resolve() == output.resolve() and not injected:
            output.write_text("CONCURRENT_OWNER", encoding="utf-8")
            injected = True

    monkeypatch.setattr(
        analysis_base_module,
        "_load_owned_published_pair",
        fail_after_publish,
    )
    monkeypatch.setattr(assembly_module.os, "rename", quarantine_then_install_foreign)
    with pytest.raises(
        analysis_base_module.AnalysisBaseError,
        match="injected postpublish failure",
    ):
        analysis_base_module.write_analysis_base(
            "g01__s1",
            output,
            authority=authority,
        )
    assert output.read_text(encoding="utf-8") == "CONCURRENT_OWNER"
    assert not output.with_suffix(".provenance.json").exists()


@pytest.mark.parametrize("target_name", ["csv", "sidecar"])
@pytest.mark.parametrize("mutation", ["in_place", "replacement"])
def test_analysis_writer_rechecks_owned_pair_after_final_strict_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    mutation: str,
) -> None:
    authority, _, _, _, report = _analysis_fixture(tmp_path, monkeypatch)
    output = authority.analysis_base_root / "g01__s1.csv"
    sidecar = output.with_suffix(".provenance.json")
    strict_calls = 0

    def mutate_on_final_strict(
        path: Path,
        *,
        authority: object,
    ) -> dict[str, object]:
        nonlocal strict_calls
        del path, authority
        strict_calls += 1
        if strict_calls == 2:
            target = output if target_name == "csv" else sidecar
            if mutation == "replacement":
                target.unlink()
                target.write_bytes(b"FOREIGN_REPLACEMENT")
            else:
                target.write_bytes(target.read_bytes() + b"\n")
        return json.loads(json.dumps(report))

    monkeypatch.setattr(
        analysis_base_module,
        "_load_gold_alignment_strict",
        mutate_on_final_strict,
    )
    expected = "rollback preserved unowned" if mutation == "replacement" else (
        "identity/bytes changed"
    )
    with pytest.raises(analysis_base_module.AnalysisBaseError, match=expected):
        analysis_base_module.write_analysis_base(
            "g01__s1",
            output,
            authority=authority,
        )
    if mutation == "replacement":
        target = output if target_name == "csv" else sidecar
        assert target.read_bytes() == b"FOREIGN_REPLACEMENT"
        other = sidecar if target == output else output
        assert not other.exists()
    else:
        assert not output.exists()
        assert not sidecar.exists()


def test_analysis_writer_rechecks_pair_after_post_strict_pair_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _, _, _, _ = _analysis_fixture(tmp_path, monkeypatch)
    output = authority.analysis_base_root / "g01__s1.csv"
    original_load = assembly_module._load_dataset_file
    output_loads = 0

    def replace_after_second_output_load(
        path: Path,
        *,
        data_kind: str,
        analysis_base_authority: object | None = None,
    ):
        nonlocal output_loads
        loaded = original_load(
            path,
            data_kind=data_kind,
            analysis_base_authority=analysis_base_authority,
        )
        if Path(path).resolve() == output.resolve():
            output_loads += 1
            if output_loads == 2:
                output.unlink()
                output.write_bytes(b"FOREIGN_AFTER_FINAL_LOAD")
        return loaded

    monkeypatch.setattr(
        assembly_module,
        "_load_dataset_file",
        replace_after_second_output_load,
    )
    with pytest.raises(
        analysis_base_module.AnalysisBaseError,
        match="held-handle replay|rollback preserved unowned",
    ):
        analysis_base_module.write_analysis_base(
            "g01__s1",
            output,
            authority=authority,
        )
    if os.name == "nt":
        assert not output.exists()
        assert not output.with_suffix(".provenance.json").exists()
    else:
        assert output.read_bytes() == b"FOREIGN_AFTER_FINAL_LOAD"
        assert not output.with_suffix(".provenance.json").exists()


@pytest.mark.parametrize("target_name", ["csv", "sidecar"])
def test_analysis_writer_final_held_snapshot_rejects_same_bytes_new_inode_before_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    authority, _, _, _, _ = _analysis_fixture(tmp_path, monkeypatch)
    output = (authority.analysis_base_root / "g01__s1.csv").resolve()
    sidecar = output.with_suffix(".provenance.json")
    target = output if target_name == "csv" else sidecar
    replacement = tmp_path / f"analysis-same-bytes-{target_name}.replacement"
    original_hash = assembly_module._hash_held_handle
    target_hash_calls = 0
    attempted = False
    replacement_denied = False
    replacement_identity: tuple[int, int] | None = None
    payload = b""

    def replace_after_fstat_before_final_hash(handle: object, path: Path) -> str:
        nonlocal target_hash_calls, attempted, replacement_denied
        nonlocal replacement_identity, payload
        if Path(path).resolve() == target:
            target_hash_calls += 1
            # First pair replay hashes each target on entry and exit. Inject on
            # entry to the second replay, after the final strict source pass.
            if target_hash_calls == 3:
                attempted = True
                payload = target.read_bytes()
                replacement.write_bytes(payload)
                replacement_identity = assembly_module._file_identity(replacement)
                try:
                    os.replace(replacement, target)
                except PermissionError as exc:
                    replacement_denied = True
                    replacement.unlink(missing_ok=True)
                    raise assembly_module._HeldFileSnapshotError(
                        "held Windows handle denied final same-byte inode replacement"
                    ) from exc
        return original_hash(handle, path)

    monkeypatch.setattr(
        assembly_module,
        "_hash_held_handle",
        replace_after_fstat_before_final_hash,
    )
    with pytest.raises(
        analysis_base_module.AnalysisBaseError,
        match="held-handle|rollback preserved",
    ):
        analysis_base_module.write_analysis_base(
            "g01__s1",
            output,
            authority=authority,
        )
    assert attempted
    if os.name == "nt":
        assert replacement_denied
        assert not output.exists()
        assert not sidecar.exists()
    else:
        assert not replacement_denied
        assert replacement_identity is not None
        assert target.read_bytes() == payload
        assert assembly_module._file_identity(target) == replacement_identity
        other = sidecar if target == output else output
        assert not other.exists()
