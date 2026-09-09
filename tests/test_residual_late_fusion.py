from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hri_turn_taking.vap import residual_late_fusion as residual


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _file_binding(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _group_sessions(group_id: str) -> tuple[str, ...]:
    return tuple(
        session_id
        for session_id in residual.EXACT_EIGHT_SESSION_IDS
        if session_id.split("__", maxsplit=1)[0] == group_id
    )


def _prediction_payload(
    condition_id: str,
    outer_fold_id: str,
    group_id: str,
    run_manifest_sha256: str,
    *,
    inner_fold_id: str | None,
) -> dict[str, object]:
    group_index = residual.EXACT_FOUR_GROUP_IDS.index(group_id)
    sessions = []
    for session_index, session_id in enumerate(_group_sessions(group_id)):
        n_frames = 3
        labels = (
            np.arange(n_frames * 4 * 4, dtype=np.int64).reshape(n_frames, 4, 4)
            + group_index
            + session_index
        ) % 2
        vad = (
            np.arange(n_frames * 4, dtype=np.int64).reshape(n_frames, 4)
            + group_index
            + session_index
        ) % 2
        if condition_id == "A":
            per_frame = 0.54 + 0.01 * group_index + 0.005 * np.arange(n_frames)
            probabilities = np.broadcast_to(
                per_frame[:, None, None],
                (n_frames, 4, 4),
            ).copy()
        else:
            probabilities = np.empty((n_frames, 4, 4), dtype=np.float32)
            for frame in range(n_frames):
                for participant in range(4):
                    for bin_index in range(4):
                        probabilities[frame, participant, bin_index] = (
                            0.38
                            + 0.02 * group_index
                            + 0.015 * participant
                            + 0.006 * bin_index
                            + 0.004 * frame
                        )
        sessions.append(
            {
                "session_id": session_id,
                "group_id": group_id,
                "n_frames": n_frames,
                "tick_25hz": [0, 1, 2],
                "labels": labels.astype(np.float32).tolist(),
                "vad": vad.astype(np.float32).tolist(),
                "valid": np.ones((n_frames, 4, 4), dtype=np.int8).tolist(),
                "present": [1, 1, 1, 1],
                "probabilities": np.asarray(probabilities, dtype=np.float32).tolist(),
            }
        )
    payload: dict[str, object] = {
        "condition_id": condition_id,
        "outer_fold_id": outer_fold_id,
        "seed": 13,
        "run_manifest_sha256": run_manifest_sha256,
        "frame_rate_hz": 25,
        "label_config": {"bin_edges_s": [0.2, 0.6, 1.2, 2.0]},
        "threshold_policy": {"diagnostic_threshold": 0.5},
        "sessions": sessions,
    }
    if inner_fold_id is not None:
        payload["inner_fold_id"] = inner_fold_id
    return payload


@pytest.fixture
def source_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    protocol_root = (tmp_path / "protocol_15ep").resolve()
    protocol_root.mkdir()
    frozen_source = (tmp_path / "frozen_source").resolve()
    run_manifest = {
        "artifact_kind": residual.SOURCE_RUN_MANIFEST_KIND,
        "format_version": residual.SOURCE_RUN_MANIFEST_VERSION,
        "protocol_id": residual.SOURCE_PROTOCOL_ID,
        "result_role": "protocol_only_not_a_result",
        "counted_experiment_complete": False,
        "split_manifest_sha256": residual.SOURCE_SPLIT_MANIFEST_SHA256,
        "seeds": list(residual.SOURCE_SEEDS),
        "condition_ids": list(residual.MATRIX_CONDITIONS),
        "max_epochs": residual.SOURCE_MAX_EPOCHS,
        "aux_vad_weight": 0.0,
        "source": str(frozen_source),
    }
    run_manifest_path = protocol_root / "run_manifest.json"
    _write_json(run_manifest_path, run_manifest)
    run_manifest_sha256 = residual._canonical_json_sha256(run_manifest)
    monkeypatch.setattr(
        residual,
        "SOURCE_RUN_MANIFEST_SHA256",
        run_manifest_sha256,
    )

    source_root = (tmp_path / "matrix_seed13_15ep").resolve()
    source_root.mkdir()
    payloads: dict[Path, dict[str, object]] = {}
    prediction_paths: dict[tuple[str, str, str], Path] = {}
    validated_units: list[Path] = []
    validated_prepared_conditions: list[str] = []
    prepared_conditions: list[str] = []
    loaded_predictions: list[Path] = []
    units = []
    for condition_id in residual.MATRIX_CONDITIONS:
        for outer_index, outer_fold_id in enumerate(residual.OUTER_FOLD_IDS):
            unit_dir = source_root / residual.CONDITION_DIRECTORIES[condition_id] / outer_fold_id
            unit_dir.mkdir(parents=True)
            artifact_bindings: dict[str, object] = {}
            if condition_id in residual.SOURCE_CONDITIONS:
                inner_bindings = []
                inner_groups = [
                    group
                    for group in residual.EXACT_FOUR_GROUP_IDS
                    if group != residual.EXACT_FOUR_GROUP_IDS[outer_index]
                ]
                for inner_fold_id, group_id in zip(
                    residual.INNER_FOLD_IDS,
                    inner_groups,
                    strict=True,
                ):
                    prediction_path = unit_dir / f"{inner_fold_id}.npz"
                    prediction_path.write_bytes(
                        f"{condition_id}-{outer_fold_id}-{inner_fold_id}".encode()
                    )
                    payloads[prediction_path.resolve()] = _prediction_payload(
                        condition_id,
                        outer_fold_id,
                        group_id,
                        run_manifest_sha256,
                        inner_fold_id=inner_fold_id,
                    )
                    prediction_paths[(condition_id, outer_fold_id, inner_fold_id)] = (
                        prediction_path.resolve()
                    )
                    inner_bindings.append(
                        {
                            "inner_fold_id": inner_fold_id,
                            **_file_binding(prediction_path, unit_dir),
                        }
                    )
                outer_path = unit_dir / "outer.npz"
                outer_path.write_bytes(f"{condition_id}-{outer_fold_id}-outer".encode())
                outer_group = residual.EXACT_FOUR_GROUP_IDS[outer_index]
                payloads[outer_path.resolve()] = _prediction_payload(
                    condition_id,
                    outer_fold_id,
                    outer_group,
                    run_manifest_sha256,
                    inner_fold_id=None,
                )
                prediction_paths[(condition_id, outer_fold_id, "outer")] = outer_path.resolve()
                artifact_bindings = {
                    "inner_oof_predictions": inner_bindings,
                    "outer_predictions": _file_binding(outer_path, unit_dir),
                }
            result = {
                "condition_id": condition_id,
                "outer_fold_id": outer_fold_id,
                "seed": 13,
                "run_manifest_sha256": run_manifest_sha256,
                "artifact_bindings": artifact_bindings,
            }
            result_path = unit_dir / "result_manifest.json"
            _write_json(result_path, result)
            units.append(
                {
                    "condition_id": condition_id,
                    "outer_fold_id": outer_fold_id,
                    "seed": 13,
                    "result_manifest": str(result_path.resolve()),
                    "result_manifest_sha256": _sha256(result_path),
                    "elapsed_seconds": 1.0,
                }
            )
    index = {
        "artifact_kind": "vap_thesis_primary_matrix_index",
        "format_version": 1,
        "result_role": residual.thesis_backend.RESULT_ROLE,
        "data_kind": "real",
        "production_eligible": False,
        "counted_eligible": False,
        "analysis_variant": "primary",
        "seed": 13,
        "condition_ids": list(residual.MATRIX_CONDITIONS),
        "outer_fold_ids": list(residual.OUTER_FOLD_IDS),
        "expected_unit_count": 12,
        "completed_unit_count": 12,
        "protocol_directory": str(protocol_root),
        "run_manifest_sha256": run_manifest_sha256,
        "included_session_ids": list(residual.EXACT_EIGHT_SESSION_IDS),
        "group_ids": list(residual.EXACT_FOUR_GROUP_IDS),
        "corpus_exclusions": dict(residual.EXACT_CORPUS_EXCLUSIONS),
        "units": units,
    }
    index_path = source_root / "matrix_index.json"
    _write_json(index_path, index)

    def prepare_condition(
        protocol_directory: Path,
        *,
        source: Path,
        condition_id: str,
    ) -> SimpleNamespace:
        assert Path(protocol_directory).resolve() == protocol_root
        assert Path(source).resolve() == frozen_source
        prepared_conditions.append(condition_id)
        return SimpleNamespace(condition_id=condition_id)

    monkeypatch.setattr(
        residual.outer_experiment,
        "prepare_outer_condition",
        prepare_condition,
    )

    def validate_historical(path: Path, *, prepared: SimpleNamespace) -> None:
        validated_units.append(path.resolve())
        validated_prepared_conditions.append(prepared.condition_id)

    monkeypatch.setattr(residual, "_validate_historical_unit", validate_historical)

    def load_prediction(path: Path, inner_fold_id: str | None) -> dict[str, object]:
        loaded_predictions.append(path.resolve())
        payload = copy.deepcopy(payloads[path.resolve()])
        assert payload.get("inner_fold_id") == inner_fold_id
        return payload

    monkeypatch.setattr(residual, "_load_prediction", load_prediction)
    monkeypatch.setattr(
        residual.thesis_backend,
        "_sessions_from_prediction",
        lambda payload: tuple(payload["sessions"]),
    )

    def score(probabilities: tuple[np.ndarray, ...]) -> float:
        packed = np.concatenate(
            [np.asarray(value, dtype=np.float64).ravel() for value in probabilities]
        )
        return float(packed.mean())

    def frame_metrics(
        _sessions: tuple[dict[str, object], ...],
        probabilities: tuple[np.ndarray, ...],
    ) -> dict[str, object]:
        return {"participant_macro_mAP": score(probabilities)}

    def metrics(
        _sessions: tuple[dict[str, object], ...],
        probabilities: tuple[np.ndarray, ...],
    ) -> dict[str, object]:
        value = score(probabilities)
        return {
            "frame": {"participant_macro_mAP": value},
            "events": {
                "shift_prediction": {"average_precision": value},
                "overlap_resolution": {"average_precision": value},
            },
        }

    monkeypatch.setattr(residual.thesis_backend, "_frame_metrics", frame_metrics)
    monkeypatch.setattr(residual.thesis_backend, "_metrics", metrics)
    return SimpleNamespace(
        source_root=source_root,
        protocol_root=protocol_root,
        run_manifest_path=run_manifest_path,
        index_path=index_path,
        payloads=payloads,
        prediction_paths=prediction_paths,
        validated_units=validated_units,
        validated_prepared_conditions=validated_prepared_conditions,
        prepared_conditions=prepared_conditions,
        loaded_predictions=loaded_predictions,
    )


def test_source_accepts_exact_15_epoch_matrix(source_fixture: SimpleNamespace) -> None:
    source = residual._load_source_matrix(source_fixture.source_root)
    assert source.run_manifest["max_epochs"] == 15
    assert residual._canonical_json_sha256(source.run_manifest) == (
        residual.SOURCE_RUN_MANIFEST_SHA256
    )
    assert source.run_manifest_binding.sha256 == _sha256(source.run_manifest_path)
    assert source.run_manifest_binding.sha256 != residual.SOURCE_RUN_MANIFEST_SHA256
    assert len(source_fixture.validated_units) == 12
    assert source_fixture.prepared_conditions == list(residual.MATRIX_CONDITIONS)
    assert source_fixture.validated_prepared_conditions == [
        condition for condition in residual.MATRIX_CONDITIONS for _fold in residual.OUTER_FOLD_IDS
    ]
    assert any("A_plus_H" in path.parts for path in source_fixture.validated_units)
    assert source_fixture.loaded_predictions == []
    assert set(source.units) == {
        (condition, fold)
        for condition in residual.SOURCE_CONDITIONS
        for fold in residual.OUTER_FOLD_IDS
    }


def test_source_rejects_five_epoch_protocol(
    source_fixture: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_manifest = json.loads(source_fixture.run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["max_epochs"] = 5
    _write_json(source_fixture.run_manifest_path, run_manifest)
    changed_hash = residual._canonical_json_sha256(run_manifest)
    monkeypatch.setattr(residual, "SOURCE_RUN_MANIFEST_SHA256", changed_hash)
    index = json.loads(source_fixture.index_path.read_text(encoding="utf-8"))
    index["run_manifest_sha256"] = changed_hash
    _write_json(source_fixture.index_path, index)
    with pytest.raises(residual.ResidualLateFusionError, match="max_epochs"):
        residual._load_source_matrix(source_fixture.source_root)


def test_real_protocol_declares_canonical_json_not_raw_file_hash() -> None:
    index_path = residual.DEFAULT_SOURCE / "matrix_index.json"
    if not index_path.is_file():
        pytest.skip("repository-owned historical source is not installed")
    index = residual._read_json_strict(index_path)
    protocol_root = Path(index["protocol_directory"])
    run_manifest_path = protocol_root / "run_manifest.json"
    run_manifest = residual._read_json_strict(run_manifest_path)
    assert index["run_manifest_sha256"] == residual.SOURCE_RUN_MANIFEST_SHA256
    assert residual._canonical_json_sha256(run_manifest) == (residual.SOURCE_RUN_MANIFEST_SHA256)
    assert residual._sha256_file(run_manifest_path) != residual.SOURCE_RUN_MANIFEST_SHA256


def test_source_rejects_forged_exact_eight_index(source_fixture: SimpleNamespace) -> None:
    index = json.loads(source_fixture.index_path.read_text(encoding="utf-8"))
    index["included_session_ids"][-1] = "forged__scene"
    _write_json(source_fixture.index_path, index)
    with pytest.raises(residual.ResidualLateFusionError, match="exact-eight identity"):
        residual._load_source_matrix(source_fixture.source_root)


def test_lambda_selection_uses_only_inner_prediction_payloads(
    source_fixture: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = residual._load_source_matrix(source_fixture.source_root)
    paired = residual._paired_predictions
    select = residual._select_lambda
    selection_complete = False
    load_order: list[str | None] = []

    def guarded_pair(
        source_value: residual.SourceMatrix,
        outer_fold_id: str,
        inner_fold_id: str | None,
    ) -> residual.PairedPredictions:
        load_order.append(inner_fold_id)
        if inner_fold_id is None:
            assert selection_complete
        return paired(source_value, outer_fold_id, inner_fold_id)

    def guarded_select(
        pairs: tuple[residual.PairedPredictions, ...],
    ) -> tuple[float, list[dict[str, object]]]:
        nonlocal selection_complete
        result = select(pairs)
        selection_complete = True
        return result

    monkeypatch.setattr(residual, "_paired_predictions", guarded_pair)
    monkeypatch.setattr(residual, "_select_lambda", guarded_select)
    residual._recompute_expected_fold(source, "outer_01")
    assert load_order == [*residual.INNER_FOLD_IDS, None]


def test_lambda_ties_choose_smallest_and_candidates_are_group_equal(
    source_fixture: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = residual._load_source_matrix(source_fixture.source_root)
    inner_pairs = tuple(
        residual._paired_predictions(source, "outer_01", inner_fold_id)
        for inner_fold_id in residual.INNER_FOLD_IDS
    )
    monkeypatch.setattr(
        residual,
        "_participant_macro_strict_map",
        lambda _payload, _probabilities: 0.5,
    )
    selected, candidates = residual._select_lambda(inner_pairs)
    assert selected == min(residual.LAMBDA_GRID)
    assert len(candidates) == len(residual.LAMBDA_GRID)
    for candidate in candidates:
        inner_scores = [
            item["participant_macro_strict_mAP"] for item in candidate["inner_group_scores"]
        ]
        assert candidate["group_equal_participant_macro_strict_mAP"] == pytest.approx(
            sum(inner_scores) / 3
        )


def test_lambda_zero_is_exact_h_canonical_bytes() -> None:
    h_probability = np.linspace(0.1, 0.9, 48, dtype=np.float32).reshape(3, 4, 4)
    a_probability = np.full((3, 4, 4), 0.6, dtype=np.float32)
    control = residual.fuse_probability_arrays(h_probability, a_probability, 0.0)
    assert np.array_equal(control, h_probability)
    assert residual._canonical_probability_bytes_sha256((control,)) == (
        residual._canonical_probability_bytes_sha256((h_probability,))
    )


def test_a_symmetry_tolerance_is_enforced(
    source_fixture: SimpleNamespace,
) -> None:
    source = residual._load_source_matrix(source_fixture.source_root)
    path = source_fixture.prediction_paths[("A", "outer_01", "inner_01")]
    payload = source_fixture.payloads[path]
    probability = np.asarray(payload["sessions"][0]["probabilities"], dtype=np.float32)
    probability[:, 0, 0] += np.float32(5e-8)
    payload["sessions"][0]["probabilities"] = probability.tolist()
    pair = residual._paired_predictions(source, "outer_01", "inner_01")
    assert pair.a_symmetry_max_abs_diff <= residual.A_SYMMETRY_TOLERANCE
    probability[:, 0, 0] += np.float32(5e-7)
    payload["sessions"][0]["probabilities"] = probability.tolist()
    with pytest.raises(residual.ResidualLateFusionError, match="symmetry exceeds"):
        residual._paired_predictions(source, "outer_01", "inner_01")


def test_pairing_rejects_changed_labels(source_fixture: SimpleNamespace) -> None:
    source = residual._load_source_matrix(source_fixture.source_root)
    path = source_fixture.prediction_paths[("H", "outer_01", "inner_01")]
    payload = source_fixture.payloads[path]
    payload["sessions"][0]["labels"][0][0][0] = 1.0 - (payload["sessions"][0]["labels"][0][0][0])
    with pytest.raises(residual.ResidualLateFusionError, match="paired labels"):
        residual._paired_predictions(source, "outer_01", "inner_01")


def test_publish_audit_no_clobber_and_aggregate_replay(
    source_fixture: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = tmp_path / "published"
    aggregate_path = residual.publish_residual_late_fusion(
        source_fixture.source_root,
        output,
    )
    assert aggregate_path == output / residual.AGGREGATE_FILENAME
    audit = residual.audit_residual_late_fusion(output)
    assert audit["audit_passed"] is True
    aggregate_text = aggregate_path.read_text(encoding="utf-8")
    assert ".stage" not in aggregate_text
    assert '"output_root"' not in aggregate_text
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        residual.publish_residual_late_fusion(source_fixture.source_root, output)

    aggregate = json.loads(aggregate_text)
    aggregate["pooled"]["selected"]["frame"]["participant_macro_mAP"] += 0.1
    _write_json(aggregate_path, aggregate)
    with pytest.raises(residual.ResidualLateFusionError, match="aggregate differs"):
        residual.audit_residual_late_fusion(output)


def test_cli_run_validates_source_once_and_independent_audit_preserves_bytes(
    source_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cli_once"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_residual_late_fusion",
            "run",
            "--source",
            str(source_fixture.source_root),
            "--output",
            str(output),
        ],
    )
    assert residual._cli() == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["audit_passed"] is True
    assert Path(receipt["aggregate_path"]) == output / residual.AGGREGATE_FILENAME
    assert len(source_fixture.validated_units) == 12
    assert source_fixture.prepared_conditions == list(residual.MATRIX_CONDITIONS)

    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert residual.audit_residual_late_fusion(output)["audit_passed"] is True
    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert len(source_fixture.validated_units) == 24
    assert source_fixture.prepared_conditions == [
        *residual.MATRIX_CONDITIONS,
        *residual.MATRIX_CONDITIONS,
    ]


def test_cached_publish_audit_rejects_concurrent_source_mutation(
    source_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "source_mutation"
    recompute = residual._recompute_expected_fold
    calls = 0
    original_index_bytes = source_fixture.index_path.read_bytes()

    def mutate_after_cached_audit_entry(
        source: residual.SourceMatrix,
        outer_fold_id: str,
    ) -> object:
        nonlocal calls
        calls += 1
        # Four calls build the staged folds.  The fifth is the first pre-rename
        # replay fold, after its entry snapshot check but before its final check.
        if calls == 5:
            source_fixture.index_path.write_bytes(original_index_bytes + b"\n")
        return recompute(source, outer_fold_id)

    monkeypatch.setattr(residual, "_recompute_expected_fold", mutate_after_cached_audit_entry)
    with pytest.raises(residual.ResidualLateFusionError, match="snapshot changed"):
        residual.publish_residual_late_fusion(source_fixture.source_root, output)
    assert calls == 8
    assert not output.exists()
    assert not list(tmp_path.glob(".source_mutation.*.stage"))


def test_saved_probability_tamper_fails_formula_replay(
    source_fixture: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = tmp_path / "tamper"
    residual.publish_residual_late_fusion(source_fixture.source_root, output)
    fused_path = output / "outer_01" / residual.FUSED_PROBABILITIES_FILENAME
    with np.load(fused_path, allow_pickle=False) as saved:
        probabilities = np.asarray(saved["probabilities"]).copy()
        offsets = np.asarray(saved["session_offsets"]).copy()
    probabilities[0, 0, 0] = np.nextafter(probabilities[0, 0, 0], np.float32(1.0))
    with fused_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            probabilities=probabilities,
            session_offsets=offsets,
        )
    with pytest.raises(residual.ResidualLateFusionError, match="formula replay"):
        residual.audit_residual_late_fusion(output)


def test_unexpected_file_and_fold_manifest_tamper_fail_closed(
    source_fixture: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest_tamper"
    residual.publish_residual_late_fusion(source_fixture.source_root, output)
    unexpected = output / "unexpected.txt"
    unexpected.write_text("unexpected", encoding="utf-8")
    with pytest.raises(residual.ResidualLateFusionError, match="file set differs"):
        residual.audit_residual_late_fusion(output)
    unexpected.unlink()
    manifest_path = output / "outer_01" / residual.FOLD_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_lambda"] = 999.0
    _write_json(manifest_path, manifest)
    with pytest.raises(residual.ResidualLateFusionError, match="fold manifest replay differs"):
        residual.audit_residual_late_fusion(output)


def test_pre_rename_audit_failure_removes_stage_without_publishing(
    source_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pre_rename_failure"

    def fail_first(
        _path: str | Path,
        *,
        validated_source: residual.SourceMatrix | None,
    ) -> dict[str, object]:
        assert validated_source is not None
        raise residual.ResidualLateFusionError("injected pre-rename failure")

    monkeypatch.setattr(residual, "_audit_residual_late_fusion", fail_first)
    with pytest.raises(residual.ResidualLateFusionError, match="pre-rename failure"):
        residual.publish_residual_late_fusion(source_fixture.source_root, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".pre_rename_failure.*.stage"))


def test_post_rename_audit_failure_rolls_back_only_owned_output(
    source_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "rollback"
    audit = residual._audit_residual_late_fusion
    calls = 0

    def fail_second(
        path: str | Path,
        *,
        validated_source: residual.SourceMatrix | None,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise residual.ResidualLateFusionError("injected post-rename failure")
        return audit(path, validated_source=validated_source)

    monkeypatch.setattr(residual, "_audit_residual_late_fusion", fail_second)
    with pytest.raises(residual.ResidualLateFusionError, match="post-rename failure"):
        residual.publish_residual_late_fusion(source_fixture.source_root, output)
    assert calls == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".rollback.*.stage"))


def test_post_rename_replacement_is_never_deleted(
    source_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "replacement"
    moved_owned = tmp_path / "moved_owned"
    audit = residual._audit_residual_late_fusion
    calls = 0

    def replace_second(
        path: str | Path,
        *,
        validated_source: residual.SourceMatrix | None,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            Path(path).rename(moved_owned)
            Path(path).mkdir()
            (Path(path) / "foreign.txt").write_text("foreign", encoding="utf-8")
            raise residual.ResidualLateFusionError("injected concurrent replacement")
        return audit(path, validated_source=validated_source)

    monkeypatch.setattr(residual, "_audit_residual_late_fusion", replace_second)
    with pytest.raises(residual.ResidualLateFusionError, match="output identity changed"):
        residual.publish_residual_late_fusion(source_fixture.source_root, output)
    assert (output / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert moved_owned.is_dir()


def test_success_thresholds_are_descriptive_not_confirmatory() -> None:
    selected = {
        "frame": {"participant_macro_mAP": 0.55},
        "events": {
            "shift_prediction": {"average_precision": 0.60},
            "overlap_resolution": {"average_precision": 0.60},
        },
    }
    control = {
        "frame": {"participant_macro_mAP": 0.50},
        "events": {
            "shift_prediction": {"average_precision": 0.59},
            "overlap_resolution": {"average_precision": 0.59},
        },
    }
    result = residual._success_evaluation(
        selected,
        control,
        {fold: 0.01 for fold in residual.OUTER_FOLD_IDS},
    )
    assert result["descriptive_criteria_met"] is True
    assert result["verdict"] == "descriptive_criteria_met"
    assert result["confirmatory_hypothesis_verdict_eligible"] is False
    assert "success_supported" not in result
