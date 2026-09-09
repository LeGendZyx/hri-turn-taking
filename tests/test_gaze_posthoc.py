from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hri_turn_taking.data import gaze_posthoc
from hri_turn_taking.data.multiparty import (
    FEATURE_SCHEMA_V2,
    FEATURE_SCHEMA_V3_GAZE_VALIDITY,
    PARTICIPANT_FEATURE_NAMES,
    SyntheticConfig,
    generate_multiparty_dataset,
    input_feature_columns,
    participant_column,
    validate_multiparty_dataframe,
)
from hri_turn_taking.vap.dataset import (
    TRANSFORMED_GLOBAL_FEATURES,
    FeatureNormalizer,
    sessions_from_dataframe,
    transform_input_features,
    transformed_participant_features,
)
from hri_turn_taking.vap.feature_sets import (
    active_gaze_posthoc_feature_columns,
    apply_gaze_posthoc_feature_set,
    gaze_posthoc_feature_set_metadata,
)
from hri_turn_taking.vap.trainer import (
    CHECKPOINT_FORMAT_VERSION,
    VAPTrainerConfig,
    train_vap_model,
)


def _base(*, session_id: str = "fixture__s1", seed: int = 7) -> pd.DataFrame:
    frame = generate_multiparty_dataset(
        SyntheticConfig(
            n_sessions=1,
            duration_s=1.2,
            n_participants=3,
            random_seed=seed,
        )
    )
    frame.loc[:, "session_id"] = session_id
    return frame


def _evidence(
    base: pd.DataFrame,
    *,
    valid_slots: tuple[str, ...] = ("P2", "P3"),
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row_index, (session_id, timestamp) in enumerate(
        base[["session_id", "timestamp_s"]].itertuples(index=False, name=None)
    ):
        for slot in gaze_posthoc.SLOTS:
            valid = slot in valid_slots and row_index % 3 != 2
            value = float(row_index % 2) if valid else np.nan
            records.append(
                {
                    "session_id": session_id,
                    "timestamp_s": timestamp,
                    "slot": slot,
                    "gaze_robot_valid": float(valid),
                    "gaze_robot": value,
                }
            )
    return pd.DataFrame.from_records(records)


def _v3(base: pd.DataFrame | None = None) -> pd.DataFrame:
    source = _base() if base is None else base
    return gaze_posthoc.build_schema_v3_frame(source, _evidence(source))


def _write_bound(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _real_base_frame(scene: str, ticks: range, seed: int) -> pd.DataFrame:
    frame = generate_multiparty_dataset(
        SyntheticConfig(
            n_sessions=1,
            duration_s=1.2,
            n_participants=4,
            random_seed=seed,
        )
    ).iloc[: len(ticks)].copy()
    frame.loc[:, "session_id"] = scene
    frame.loc[:, "timestamp_s"] = np.asarray(list(ticks), dtype=float) / 25.0
    validate_multiparty_dataframe(frame, feature_schema=FEATURE_SCHEMA_V2)
    return frame


def _write_formal_base(
    root: Path,
    scene: str,
    group_id: str,
    ticks: range,
    seed: int,
) -> pd.DataFrame:
    formal_root = root / "data" / "processed" / "formal_a_h_ah_v1"
    formal_root.mkdir(parents=True, exist_ok=True)
    csv_path = formal_root / f"{scene}.csv"
    _real_base_frame(scene, ticks, seed).to_csv(csv_path, index=False)
    payload = csv_path.read_bytes()
    parsed = pd.read_csv(csv_path, float_precision="round_trip")
    provenance = {
        "artifact_kind": "vap_real_assembled_session",
        "format_version": 2,
        "data_kind": "real",
        "session_id": scene,
        "group_id": group_id,
        "trainable": True,
        "csv_sha256": hashlib.sha256(payload).hexdigest(),
        "columns": list(parsed.columns),
        "dtypes": {column: str(parsed[column].dtype) for column in parsed},
        "n_rows": len(parsed),
        "n_frames": len(parsed),
        "output_tick_range": {
            "start_inclusive": ticks.start,
            "end_exclusive": ticks.stop,
        },
    }
    (formal_root / f"{scene}.provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return parsed


def _real_evidence(scene: str, base: pd.DataFrame, ticks: range) -> pd.DataFrame:
    participant_by_slot = {"P1": "R", "P2": "A", "P3": "B", "P4": "C"}
    records: list[dict[str, object]] = []
    for offset, tick in enumerate(ticks):
        for slot_index, slot in enumerate(gaze_posthoc.SLOTS, start=1):
            valid = slot != "P1" and (offset + slot_index) % 3 != 0
            value = float((offset + slot_index) % 2) if valid else np.nan
            records.append(
                {
                    "session_id": scene,
                    "tick": tick,
                    "timestamp_s": tick / 25.0,
                    "source_frame_index": tick,
                    "source_timestamp_s": tick / 25.0,
                    "participant": participant_by_slot[slot],
                    "slot": slot,
                    "slot_present": bool(
                        base.loc[offset, f"{slot}_present"] == 1
                    ),
                    "source_observation_id": offset * 4 + slot_index if valid else pd.NA,
                    "gaze_target_valid": valid,
                    "gaze_target": "robot" if value == 1.0 else ("elsewhere" if valid else pd.NA),
                    "gaze_target_reason": "observed" if valid else "missing",
                    "gaze_robot_valid": valid,
                    "gaze_robot": value,
                    "gaze_robot_reason": "observed" if valid else "missing",
                }
            )
    evidence = pd.DataFrame.from_records(records)
    evidence["source_observation_id"] = evidence["source_observation_id"].astype(
        "Int64"
    )
    evidence["gaze_target_valid"] = evidence["gaze_target_valid"].astype(bool)
    evidence["gaze_robot_valid"] = evidence["gaze_robot_valid"].astype(bool)
    return evidence


def _fake_preflight_report(
    root: Path,
    scene: str,
    evidence: pd.DataFrame,
    ticks: range,
) -> tuple[dict[str, object], Path]:
    source_root = root / "sealed_preflight_fixture" / scene
    bindings = {
        name: _write_bound(source_root / f"{name}.txt", f"{scene}:{name}\n".encode())
        for name in (
            "analysis_base",
            "analysis_provenance",
            "inventory",
            "crosswalk",
            "identity",
            "tracklets",
            "source",
            "metadata",
        )
    }
    report: dict[str, object] = {
        "artifact_kind": gaze_posthoc.gaze_source_readiness.ARTIFACT_KIND,
        "format_version": gaze_posthoc.gaze_source_readiness.FORMAT_VERSION,
        "scene": scene,
        "assembly_eligible": False,
        "counted_input_ready": False,
        "writer_available": False,
        "artifact_or_feature_published": False,
        "analysis_base": {
            **bindings["analysis_base"],
            "provenance": bindings["analysis_provenance"],
            "tick_start_inclusive": ticks.start,
            "tick_end_exclusive": ticks.stop,
            "n_ticks": len(ticks),
            "slot_presence": {
                slot: bool(evidence.loc[evidence["slot"].eq(slot), "slot_present"].iloc[0])
                for slot in gaze_posthoc.SLOTS
            },
        },
        "sealed_identity": {
            "inventory": bindings["inventory"],
            "crosswalk_manifest": bindings["crosswalk"],
            "final_mapping": [],
            "identity_manifest": bindings["identity"],
            "tracklet_observations": bindings["tracklets"],
        },
        "source": {
            **bindings["source"],
            "metadata": bindings["metadata"],
        },
        "time_mapping": {
            "past_only": True,
            "nearest_frame": False,
            "interpolation": False,
            "forward_fill": False,
            "extrapolation": False,
        },
        "evidence": {
            "n_rows": len(evidence),
            "columns": list(evidence.columns),
            "parsed_dtypes": {
                column: str(evidence[column].dtype) for column in evidence
            },
        },
        "scientific_limits": {
            "robot_slot_gaze_observed": False,
            "model_input_promoted": False,
        },
    }
    return report, Path(str(bindings["source"]["path"]))


def _install_real_matrix_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[str], Path]:
    root = tmp_path / "repository"
    all_scenes = list(gaze_posthoc.analysis_base.EXPECTED_TICK_RANGES)
    all_ranges = {
        scene: (index * 10, index * 10 + 2)
        for index, scene in enumerate(all_scenes)
    }
    exact_eight_ranges = {
        scene: tick_range
        for scene, tick_range in all_ranges.items()
        if scene not in gaze_posthoc.EXCLUDED_OWNER_HOLD_SCENES
    }
    monkeypatch.setattr(
        gaze_posthoc.analysis_base,
        "EXPECTED_TICK_RANGES",
        all_ranges,
    )
    monkeypatch.setattr(gaze_posthoc.analysis_base, "EXPECTED_TOTAL_ROWS", 20)
    monkeypatch.setattr(
        gaze_posthoc,
        "EXACT_EIGHT_TICK_RANGES",
        exact_eight_ranges,
    )
    monkeypatch.setattr(gaze_posthoc, "EXACT_EIGHT_TOTAL_ROWS", 16)
    monkeypatch.setattr(gaze_posthoc, "_repository_root", lambda: root)
    readme = root / gaze_posthoc.GAZE_README_RELATIVE_PATH
    readme_binding = _write_bound(readme, b"fixture per-frame method evidence\n")
    monkeypatch.setattr(
        gaze_posthoc,
        "GAZE_README_SHA256",
        readme_binding["sha256"],
    )

    results: dict[str, gaze_posthoc.gaze_source_readiness.GazeSourcePreflight] = {}
    tamper_source: Path | None = None
    for index, scene in enumerate(all_scenes):
        start, end = all_ranges[scene]
        ticks = range(start, end)
        if scene in exact_eight_ranges:
            base = _write_formal_base(
                root,
                scene,
                gaze_posthoc.EXACT_EIGHT_GROUP_BY_SCENE[scene],
                ticks,
                seed=100 + index,
            )
        else:
            base = _real_base_frame(scene, ticks, seed=100 + index)
        evidence = _real_evidence(scene, base, ticks)
        report, source_path = _fake_preflight_report(
            root,
            scene,
            evidence,
            ticks,
        )
        if scene == "05_15__4":
            tamper_source = source_path
        results[scene] = gaze_posthoc.gaze_source_readiness.GazeSourcePreflight(
            scene=scene,
            evidence_df=evidence,
            report=report,
        )
    calls: list[str] = []

    def audit(scene: str) -> gaze_posthoc.gaze_source_readiness.GazeSourcePreflight:
        calls.append(scene)
        result = results[scene]
        return gaze_posthoc.gaze_source_readiness.GazeSourcePreflight(
            scene=scene,
            evidence_df=result.evidence_df.copy(deep=True),
            report=json.loads(json.dumps(result.report)),
        )

    typed_audit: Callable[
        [str],
        gaze_posthoc.gaze_source_readiness.GazeSourcePreflight,
    ] = audit
    monkeypatch.setattr(
        gaze_posthoc.gaze_source_readiness,
        "audit_gaze_source_preflight",
        typed_audit,
    )
    assert tamper_source is not None
    return root, calls, tamper_source


def test_parallel_schema_counts_and_historical_v2_defaults_are_unchanged() -> None:
    assert len(PARTICIPANT_FEATURE_NAMES) == 7
    assert len(input_feature_columns()) == 32
    assert len(input_feature_columns(feature_schema=FEATURE_SCHEMA_V2)) == 32
    assert len(input_feature_columns(feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY)) == 36

    v2 = transform_input_features(np.zeros((1, 32), dtype=np.float32))
    v3 = transform_input_features(
        np.zeros((1, 36), dtype=np.float32),
        feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY,
    )
    assert v2.shape == (1, 33)
    assert v3.shape == (1, 37)
    assert (
        len(transformed_participant_features(FEATURE_SCHEMA_V3_GAZE_VALIDITY))
        + len(TRANSFORMED_GLOBAL_FEATURES)
        == 13
    )
    assert CHECKPOINT_FORMAT_VERSION == 4


def test_schema_v3_join_preserves_nullable_semantics_and_p1_is_always_invalid() -> None:
    frame = _v3()

    validate_multiparty_dataframe(
        frame, feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY
    )
    assert frame["P1_gaze_robot_valid"].eq(0.0).all()
    assert frame["P1_gaze_robot"].eq(0.0).all()
    assert frame["P4_gaze_robot_valid"].eq(0.0).all()
    assert frame["P4_gaze_robot"].eq(0.0).all()
    for slot in gaze_posthoc.SLOTS:
        invalid = frame[f"{slot}_gaze_robot_valid"].eq(0.0)
        assert frame.loc[invalid, f"{slot}_gaze_robot"].eq(0.0).all()


def test_v3_table_requires_explicit_schema_in_sessions_and_v2_trainer(
    tmp_path: Path,
) -> None:
    frame = _v3()

    with pytest.raises(ValueError, match="schema-v3-only"):
        sessions_from_dataframe(frame)
    checkpoint = tmp_path / "must-not-exist.pt"
    with pytest.raises(ValueError, match="schema-v3-only"):
        train_vap_model(
            frame,
            checkpoint_path=checkpoint,
            trainer_config=VAPTrainerConfig(epochs=1),
        )
    assert not checkpoint.exists()
    assert not checkpoint.with_suffix(".metrics.json").exists()


def test_schema_v3_rejects_p1_observation_and_invalid_nonzero_value() -> None:
    base = _base()
    p1 = _evidence(base)
    first_p1 = p1.index[p1["slot"].eq("P1")][0]
    p1.loc[first_p1, ["gaze_robot_valid", "gaze_robot"]] = [1.0, 1.0]
    with pytest.raises(gaze_posthoc.GazePosthocGateError, match="P1/R"):
        gaze_posthoc.build_schema_v3_frame(base, p1)

    frame = _v3(base)
    frame.loc[0, ["P2_gaze_robot_valid", "P2_gaze_robot"]] = [0.0, 1.0]
    with pytest.raises(ValueError, match="exactly zero"):
        validate_multiparty_dataframe(
            frame, feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY
        )


@pytest.mark.parametrize("condition_id", ["C0", "CM", "CG"])
def test_c0_cm_cg_share_v3_width_and_zero_only_disabled_inputs(
    condition_id: str,
) -> None:
    frame = _v3()
    masked = apply_gaze_posthoc_feature_set(frame, condition_id)
    canonical = input_feature_columns(feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY)
    active = set(active_gaze_posthoc_feature_columns(condition_id))

    assert len(canonical) == 36
    assert masked[[column for column in canonical if column not in active]].eq(0.0).all().all()
    assert masked[list(active)].equals(frame[list(active)])
    assert masked.filter(regex=r"_vad$").equals(frame.filter(regex=r"_vad$"))
    metadata = gaze_posthoc_feature_set_metadata(condition_id)
    assert metadata["feature_schema"] == FEATURE_SCHEMA_V3_GAZE_VALIDITY
    assert metadata["posthoc"] is True
    assert metadata["production_eligible"] is False
    assert metadata["counted_eligible"] is False


def test_posthoc_condition_content_isolation_is_exact() -> None:
    frame = _v3()
    c0 = apply_gaze_posthoc_feature_set(frame, "C0")
    cm = apply_gaze_posthoc_feature_set(frame, "CM")
    cg = apply_gaze_posthoc_feature_set(frame, "CG")
    gaze_values = [participant_column(slot, "gaze_robot") for slot in range(1, 5)]
    gaze_validity = [
        participant_column(slot, "gaze_robot_valid") for slot in range(1, 5)
    ]

    assert c0[gaze_values + gaze_validity].eq(0.0).all().all()
    assert cm[gaze_values].eq(0.0).all().all()
    assert cm[gaze_validity].equals(cg[gaze_validity])
    non_gaze = [
        column
        for column in input_feature_columns(feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY)
        if column not in gaze_values + gaze_validity
    ]
    assert c0[non_gaze].equals(cm[non_gaze])
    assert cm[non_gaze].equals(cg[non_gaze])


def test_v3_normalizer_fits_only_valid_train_values_and_preserves_masks() -> None:
    train = _base(session_id="train__s1", seed=8)
    train_evidence = _evidence(train, valid_slots=("P2",))
    p2_rows = train_evidence["slot"].eq("P2")
    valid_indices = train_evidence.index[p2_rows][:2]
    train_evidence.loc[p2_rows, ["gaze_robot_valid", "gaze_robot"]] = [0.0, np.nan]
    train_evidence.loc[valid_indices[0], ["gaze_robot_valid", "gaze_robot"]] = [1.0, 0.0]
    train_evidence.loc[valid_indices[1], ["gaze_robot_valid", "gaze_robot"]] = [1.0, 1.0]
    train_v3 = gaze_posthoc.build_schema_v3_frame(train, train_evidence)
    train_session = sessions_from_dataframe(
        train_v3, feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY
    )[0]

    held_out = _base(session_id="heldout__s1", seed=9)
    held_evidence = _evidence(held_out, valid_slots=("P2",))
    held_evidence.loc[
        held_evidence["slot"].eq("P2"), ["gaze_robot_valid", "gaze_robot"]
    ] = [1.0, 1.0]
    held_v3 = gaze_posthoc.build_schema_v3_frame(held_out, held_evidence)
    held_session = sessions_from_dataframe(
        held_v3, feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY
    )[0]

    normalizer = FeatureNormalizer.fit([train_session])
    names = transformed_participant_features(FEATURE_SCHEMA_V3_GAZE_VALIDITY)
    value_index = names.index("gaze_robot")
    valid_index = names.index("gaze_robot_valid")
    assert normalizer.participant_mean[value_index] == pytest.approx(0.5)
    assert normalizer.participant_std[value_index] == pytest.approx(0.5)
    assert normalizer.participant_mean[valid_index] == 0.0
    assert normalizer.participant_std[valid_index] == 1.0

    normalized = normalizer.apply(held_session.features)
    per_slot = normalized[:, :-len(TRANSFORMED_GLOBAL_FEATURES)].reshape(
        len(held_v3), 4, len(names)
    )
    raw_per_slot = held_session.features[:, :-len(TRANSFORMED_GLOBAL_FEATURES)].reshape(
        len(held_v3), 4, len(names)
    )
    assert np.array_equal(per_slot[:, :, valid_index], raw_per_slot[:, :, valid_index])
    assert np.array_equal(
        per_slot[:, :, value_index][raw_per_slot[:, :, valid_index] == 0.0],
        np.zeros_like(per_slot[:, :, value_index][raw_per_slot[:, :, valid_index] == 0.0]),
    )
    # Held-out valid ones use train-only mean/std: (1 - 0.5) / 0.5 == 1.
    assert np.allclose(per_slot[:, 1, value_index], 1.0)


def test_c0_normalizer_uses_identity_gaze_stats_without_inventing_samples() -> None:
    c0 = apply_gaze_posthoc_feature_set(_v3(), "C0")
    session = sessions_from_dataframe(
        c0, feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY
    )[0]
    normalizer = FeatureNormalizer.fit([session])
    names = transformed_participant_features(FEATURE_SCHEMA_V3_GAZE_VALIDITY)

    assert normalizer.participant_mean[names.index("gaze_robot")] == 0.0
    assert normalizer.participant_std[names.index("gaze_robot")] == 1.0
    assert normalizer.apply(session.features).shape[1] == 37


def test_v3_normalizer_serialization_is_explicit_without_changing_v2_payloads() -> None:
    v3_session = sessions_from_dataframe(
        _v3(), feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY
    )[0]
    v3 = FeatureNormalizer.fit([v3_session])
    v3_payload = v3.to_dict()
    restored = FeatureNormalizer.from_dict(v3_payload)

    assert v3_payload["feature_schema"] == FEATURE_SCHEMA_V3_GAZE_VALIDITY
    assert restored.feature_schema == FEATURE_SCHEMA_V3_GAZE_VALIDITY
    assert np.array_equal(restored.apply(v3_session.features), v3.apply(v3_session.features))

    v2_session = sessions_from_dataframe(_base())[0]
    assert "feature_schema" not in FeatureNormalizer.fit([v2_session]).to_dict()


def test_fixture_publisher_is_hash_bound_replayable_and_never_trainable(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.csv"
    evidence_path = tmp_path / "gaze.csv"
    output_path = tmp_path / "v3.csv"
    base = _base()
    base.to_csv(base_path, index=False)
    _evidence(base).to_csv(evidence_path, index=False)

    artifact = gaze_posthoc.publish_fixture_artifact(
        base_path, evidence_path, output_path
    )
    audited = gaze_posthoc.audit_fixture_artifact(output_path)
    readiness = gaze_posthoc.fixture_readiness(output_path)

    assert artifact.manifest == audited.manifest
    assert audited.manifest["schema"]["raw_input_width"] == 36
    assert audited.manifest["schema"]["transformed_input_width"] == 37
    assert audited.manifest["schema"]["token_input_width"] == 13
    assert readiness["schema_fixture_ready"] is True
    assert readiness["hash_replay_ready"] is True
    assert readiness["real_training_ready"] is False
    assert readiness["production_eligible"] is False
    assert readiness["counted_eligible"] is False
    with pytest.raises(FileExistsError):
        gaze_posthoc.publish_fixture_artifact(base_path, evidence_path, output_path)

    evidence = pd.read_csv(evidence_path)
    valid_p2 = evidence.index[
        evidence["slot"].eq("P2") & evidence["gaze_robot_valid"].eq(1.0)
    ][0]
    evidence.loc[valid_p2, "gaze_robot"] = 1.0 - evidence.loc[
        valid_p2, "gaze_robot"
    ]
    evidence.to_csv(evidence_path, index=False)
    with pytest.raises(gaze_posthoc.GazePosthocGateError, match="binding"):
        gaze_posthoc.audit_fixture_artifact(output_path)


def test_current_real_preflight_cannot_authorize_publication() -> None:
    report = {
        "assembly_eligible": False,
        "counted_input_ready": False,
        "writer_available": False,
        "scientific_limits": {"model_input_promoted": False},
    }

    blockers = gaze_posthoc.current_real_publication_blockers(report)
    assert "gaze_extractor_model_input_causality_is_not_established" in blockers
    with pytest.raises(gaze_posthoc.GazePosthocGateError, match="real Gaze publication"):
        gaze_posthoc.require_current_real_publication_ready(report)


def test_exact8_exclusion_reasons_match_outer_authority() -> None:
    from hri_turn_taking.vap import outer_experiment

    assert (
        gaze_posthoc.EXCLUDED_OWNER_HOLD_SCENES
        == outer_experiment.DEADLINE_CORPUS_EXCLUSIONS
    )


def test_real_posthoc_publisher_is_exact8_create_once_and_boundary_preserving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, calls, _tamper_source = _install_real_matrix_fixture(tmp_path, monkeypatch)
    output_dir = tmp_path / "published_schema_v3"

    artifact = gaze_posthoc.publish_real_posthoc_matrix(output_dir)

    scenes = list(gaze_posthoc.EXACT_EIGHT_TICK_RANGES)
    all_scenes = list(gaze_posthoc.analysis_base.EXPECTED_TICK_RANGES)
    assert calls == all_scenes + all_scenes
    assert artifact.output_dir == output_dir.resolve()
    assert set(artifact.frames) == set(scenes)
    assert sum(len(frame) for frame in artifact.frames.values()) == 16
    assert {path.name for path in output_dir.iterdir()} == (
        {gaze_posthoc.REAL_MATRIX_MANIFEST_NAME}
        | {f"{scene}.csv" for scene in scenes}
        | {f"{scene}.provenance.json" for scene in scenes}
    )
    manifest = artifact.manifest
    assert manifest["exact_scene_inventory"] == scenes
    assert manifest["excluded_owner_hold_scenes"] == {
        "05_15__4": "no A speech in Transcript; pending owner manual review",
        "05_15__5": "A/R extremely sparse; pending owner manual review",
    }
    assert manifest["total_rows"] == 16
    assert manifest["group_ids"] == ["03_20", "03_26", "05_14", "05_15"]
    assert manifest["permitted_use"] == "posthoc_exploratory_training"
    assert manifest["posthoc_exploratory_training_ready"] is True
    assert manifest["formal_outer_runner_admitted"] is False
    assert manifest["production_eligible"] is False
    assert manifest["counted_eligible"] is False
    assert manifest["upstream_exact_regeneration_available"] is False
    assert manifest["implementation_counterfactual_causality_verified"] is False
    preflight_matrix = manifest["source_preflight_matrix"]
    assert preflight_matrix["frozen_matrix_replay_validated"] is True
    assert preflight_matrix["scene_inventory"] == all_scenes
    assert preflight_matrix["n_scenes"] == 10
    assert preflight_matrix["total_ticks"] == 20
    assert set(preflight_matrix["scenes"]) == set(all_scenes)
    for held_scene in gaze_posthoc.EXCLUDED_OWNER_HOLD_SCENES:
        assert len(
            preflight_matrix["scenes"][held_scene]["consumed_bindings"]
        ) == 8
    assert (
        manifest["method_evidence"]["gaze_lle_official_method_reference"]["commit"]
        == "2c4fedac7159a0ffd9e40d96d4f58f4dd85aec6a"
    )
    for scene, frame in artifact.frames.items():
        validate_multiparty_dataframe(
            frame,
            feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY,
        )
        assert frame["P1_gaze_robot_valid"].eq(0.0).all()
        assert frame["P1_gaze_robot"].eq(0.0).all()
        sidecar = json.loads(
            (output_dir / f"{scene}.provenance.json").read_text(encoding="utf-8")
        )
        assert sidecar["source_preflight"]["source_boundary"] == {
            "assembly_eligible": False,
            "counted_input_ready": False,
            "writer_available": False,
            "artifact_or_feature_published": False,
            "model_input_promoted": False,
        }
        assert sidecar["coverage"]["no_coverage_drift"] is True
        assert len(sidecar["source_preflight"]["consumed_bindings"]) == 8

    calls_before = list(calls)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gaze_posthoc.publish_real_posthoc_matrix(output_dir)
    assert calls == calls_before


def test_real_posthoc_audit_fails_on_output_source_readme_or_file_set_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _calls, tamper_source = _install_real_matrix_fixture(tmp_path, monkeypatch)
    output_dir = tmp_path / "published_schema_v3"
    gaze_posthoc.publish_real_posthoc_matrix(output_dir)
    scene = next(iter(gaze_posthoc.EXACT_EIGHT_TICK_RANGES))

    output_csv = output_dir / f"{scene}.csv"
    original_output = output_csv.read_bytes()
    output_csv.write_bytes(original_output + b"\n")
    with pytest.raises(gaze_posthoc.GazePosthocGateError, match="binding"):
        gaze_posthoc.audit_real_posthoc_matrix(output_dir)
    output_csv.write_bytes(original_output)
    gaze_posthoc.audit_real_posthoc_matrix(output_dir)

    original_source = tamper_source.read_bytes()
    tamper_source.write_bytes(original_source + b"tamper")
    with pytest.raises(gaze_posthoc.GazePosthocGateError, match="consumed file"):
        gaze_posthoc.audit_real_posthoc_matrix(output_dir)
    tamper_source.write_bytes(original_source)
    gaze_posthoc.audit_real_posthoc_matrix(output_dir)

    readme = root / gaze_posthoc.GAZE_README_RELATIVE_PATH
    original_readme = readme.read_bytes()
    readme.write_bytes(original_readme + b"tamper")
    with pytest.raises(gaze_posthoc.GazePosthocGateError, match="README"):
        gaze_posthoc.audit_real_posthoc_matrix(output_dir)
    readme.write_bytes(original_readme)

    extra = output_dir / "unexpected.txt"
    extra.write_text("unexpected", encoding="utf-8")
    with pytest.raises(gaze_posthoc.GazePosthocGateError, match="file set"):
        gaze_posthoc.audit_real_posthoc_matrix(output_dir)


def test_real_posthoc_publisher_rejects_preflight_coverage_drift_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, _calls, _tamper_source = _install_real_matrix_fixture(tmp_path, monkeypatch)
    original = gaze_posthoc.gaze_source_readiness.audit_gaze_source_preflight
    first_scene = next(iter(gaze_posthoc.EXACT_EIGHT_TICK_RANGES))

    def drifted(
        scene: str,
    ) -> gaze_posthoc.gaze_source_readiness.GazeSourcePreflight:
        result = original(scene)
        if scene != first_scene:
            return result
        evidence = result.evidence_df.iloc[:-1].copy()
        report = json.loads(json.dumps(result.report))
        report["evidence"]["n_rows"] = len(evidence)
        report["evidence"]["columns"] = list(evidence.columns)
        report["evidence"]["parsed_dtypes"] = {
            column: str(evidence[column].dtype) for column in evidence
        }
        return gaze_posthoc.gaze_source_readiness.GazeSourcePreflight(
            scene=scene,
            evidence_df=evidence,
            report=report,
        )

    monkeypatch.setattr(
        gaze_posthoc.gaze_source_readiness,
        "audit_gaze_source_preflight",
        drifted,
    )
    output_dir = tmp_path / "must_not_publish"
    with pytest.raises(gaze_posthoc.GazePosthocGateError, match="preflight"):
        gaze_posthoc.publish_real_posthoc_matrix(output_dir)
    assert not output_dir.exists()


def test_final_audit_failure_removes_only_unchanged_tree_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_real_matrix_fixture(tmp_path, monkeypatch)
    output_dir = tmp_path / "retry_after_safe_rollback"
    original_audit = gaze_posthoc.audit_real_posthoc_matrix
    state = {"forced": False}

    def fail_once(path: str | Path) -> gaze_posthoc.RealGazePosthocMatrixArtifact:
        if not state["forced"]:
            state["forced"] = True
            raise gaze_posthoc.GazePosthocGateError("forced final audit failure")
        return original_audit(path)

    monkeypatch.setattr(gaze_posthoc, "audit_real_posthoc_matrix", fail_once)
    with pytest.raises(gaze_posthoc.GazePosthocGateError, match="forced final audit"):
        gaze_posthoc.publish_real_posthoc_matrix(output_dir)
    assert not output_dir.exists()

    artifact = gaze_posthoc.publish_real_posthoc_matrix(output_dir)
    assert artifact.output_dir == output_dir.resolve()


def test_final_audit_failure_preserves_concurrently_mutated_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_real_matrix_fixture(tmp_path, monkeypatch)
    output_dir = tmp_path / "preserve_mutated_output"
    first_scene = next(iter(gaze_posthoc.EXACT_EIGHT_TICK_RANGES))

    def mutate_then_fail(
        path: str | Path,
    ) -> gaze_posthoc.RealGazePosthocMatrixArtifact:
        csv_path = Path(path) / f"{first_scene}.csv"
        csv_path.write_bytes(csv_path.read_bytes() + b"concurrent-mutation")
        raise gaze_posthoc.GazePosthocGateError("forced final audit failure")

    monkeypatch.setattr(
        gaze_posthoc,
        "audit_real_posthoc_matrix",
        mutate_then_fail,
    )
    with pytest.raises(
        gaze_posthoc.GazePosthocRecoveryError,
        match="preserved for owner recovery",
    ):
        gaze_posthoc.publish_real_posthoc_matrix(output_dir)
    assert output_dir.is_dir()
    assert (output_dir / f"{first_scene}.csv").read_bytes().endswith(
        b"concurrent-mutation"
    )
