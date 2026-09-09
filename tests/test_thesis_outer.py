from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from hri_turn_taking.vap import outer_experiment as outer
from hri_turn_taking.vap import thesis_backend as thesis
from hri_turn_taking.vap.dataset import FeatureNormalizer, SessionTensors
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG
from hri_turn_taking.vap.outer_experiment import (
    CalibrationRequest,
    FinalFitRequest,
    InnerFitRequest,
    OuterEvaluationRequest,
)
from hri_turn_taking.vap.thesis_backend import (
    CONDITION_FEATURE_SETS,
    COUNTED_BLOCKERS,
    THRESHOLD_POLICY,
    RealSessionAuthority,
    RealSessionBinding,
    ThesisBackendGateError,
    ThesisBackendSpec,
    ThesisRunBinding,
    ThesisTorchBackend,
    aggregate_outer_prediction_artifacts,
    build_real_session_authority,
    load_checkpoint_artifact,
    load_metrics_artifact,
    load_prediction_artifact,
    session_content_sha256,
    thesis_model_config,
    thesis_trainer_config,
)


def _load_runner_module() -> ModuleType:
    """Load the repository CLI independently of cwd and pytest entry point."""

    path = Path(__file__).resolve().parents[1] / "tools" / "run_thesis_outer.py"
    spec = importlib.util.spec_from_file_location("thesis_outer_runner_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load thesis outer runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner_module()


def _spec() -> ThesisBackendSpec:
    return ThesisBackendSpec(
        d_model=4,
        n_layers=1,
        n_heads=1,
        dropout=0.0,
        window_frames=4,
        stride_frames=3,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_fraction=0.0,
        gradient_clip=1.0,
        early_stopping_patience=2,
        inference_chunk_frames=16,
    )


def _normalizer() -> FeatureNormalizer:
    return FeatureNormalizer(
        participant_mean=np.zeros(7, dtype=np.float32),
        participant_std=np.ones(7, dtype=np.float32),
        global_mean=np.zeros(5, dtype=np.float32),
        global_std=np.ones(5, dtype=np.float32),
    )


def _session(
    session_id: str,
    group_id: str,
    index: int,
    *,
    n_frames: int = 9,
) -> SessionTensors:
    rng = np.random.default_rng(700 + index)
    features = rng.normal(size=(n_frames, 33)).astype(np.float32)
    frame = np.arange(n_frames)[:, None, None]
    participant = np.arange(4)[None, :, None]
    bin_index = np.arange(4)[None, None, :]
    labels = ((frame + participant + bin_index + index) % 2).astype(np.float32)
    vad = ((np.arange(n_frames)[:, None] + np.arange(4)[None, :] + index) % 2).astype(np.float32)
    return SessionTensors(
        session_id=session_id,
        group_id=group_id,
        features=features,
        labels=labels,
        vad=vad,
        valid=np.ones(n_frames, dtype=bool),
        present=np.ones(4, dtype=np.float32),
    )


def _sessions() -> tuple[SessionTensors, ...]:
    return (
        _session("fixture_01", "03_26", 1),
        _session("fixture_02", "05_14", 2),
        _session("fixture_03", "05_15", 3),
        _session("fixture_04", "03_20", 4),
    )


def _authority(
    sessions: tuple[SessionTensors, ...],
    *,
    condition_id: str = "A+H",
) -> RealSessionAuthority:
    bindings = []
    for index, session in enumerate(sorted(sessions, key=lambda item: item.session_id)):
        ticks = tuple(range(1000 * index + 17, 1000 * index + 17 + len(session.valid)))
        bindings.append(
            RealSessionBinding(
                session_id=session.session_id,
                group_id=str(session.group_id),
                tick_25hz=ticks,
                session_content_sha256=session_content_sha256(session, ticks),
            )
        )
    return RealSessionAuthority(
        condition_id=condition_id,
        feature_set=CONDITION_FEATURE_SETS[condition_id],
        source_manifest_sha256="a" * 64,
        bindings=tuple(bindings),
    )


def _run_manifest(spec: ThesisBackendSpec, source: Path | None = None) -> dict[str, object]:
    return {
        "source": str(source.resolve()) if source is not None else "fixture-source",
        "max_epochs": 1,
        "trainer_config": thesis_trainer_config(spec),
        "model_config": thesis_model_config(spec),
        "aux_vad_weight": 0.0,
        "label_config": json.loads(json.dumps(asdict(DEFAULT_LABEL_CONFIG))),
    }


def _inner_request(
    output_dir: Path,
    sessions: tuple[SessionTensors, ...],
    *,
    train_indices: tuple[int, ...],
    validation_index: int,
    inner_fold_id: str,
) -> InnerFitRequest:
    output_dir.mkdir()
    return InnerFitRequest(
        condition_id="A+H",
        feature_set="audio_plus_head_pose",
        encoder="transformer",
        outer_fold_id="outer_01",
        inner_fold_id=inner_fold_id,
        seed=13,
        max_epochs=1,
        train_sessions=tuple(sessions[index] for index in train_indices),
        validation_sessions=(sessions[validation_index],),
        normalizer=_normalizer(),
        pos_weight=np.ones(4, dtype=np.float32),
        output_dir=output_dir,
    )


def test_real_only_binding_exact_config_and_no_production_registration() -> None:
    spec = _spec()
    sessions = _sessions()
    authority = _authority(sessions)

    assert issubclass(ThesisTorchBackend, outer.BackendContractSmokeBackend)
    assert outer._PRODUCTION_BACKEND_REGISTRY == ()
    assert tuple(CONDITION_FEATURE_SETS) == ("A", "H", "A+H")
    assert authority.data_kind == "real"
    assert list(COUNTED_BLOCKERS) == thesis._base_payload("fixture")["counted_blockers"]

    with pytest.raises(ThesisBackendGateError, match="real authority only"):
        RealSessionAuthority(
            condition_id="A+H",
            feature_set="audio_plus_head_pose",
            source_manifest_sha256="a" * 64,
            bindings=authority.bindings,
            data_kind="synthetic",
        )
    with pytest.raises(ThesisBackendGateError, match="not audited real"):
        build_real_session_authority(
            object(),
            {"data_kind": "synthetic"},
            condition_id="A",
        )

    manifest = _run_manifest(spec)
    binding = ThesisRunBinding.from_run_manifest(manifest, spec)
    assert binding.binding_role == "frozen_outer_protocol"
    unicode_manifest = _run_manifest(spec)
    unicode_manifest["source"] = "E:/论文/真实数据"
    unicode_binding = ThesisRunBinding.from_run_manifest(unicode_manifest, spec)
    expected_unicode_hash = hashlib.sha256(
        json.dumps(
            unicode_manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert unicode_binding.run_manifest_sha256 == expected_unicode_hash
    drifted = json.loads(json.dumps(manifest))
    drifted["trainer_config"]["batch_size"] += 1
    with pytest.raises(ThesisBackendGateError, match="trainer config differs"):
        ThesisRunBinding.from_run_manifest(drifted, spec)
    wrong_aux_type = json.loads(json.dumps(manifest))
    wrong_aux_type["aux_vad_weight"] = 0
    with pytest.raises(ThesisBackendGateError, match="exactly 0.0"):
        ThesisRunBinding.from_run_manifest(wrong_aux_type, spec)


def test_fixture_lifecycle_uses_runner_statistics_and_one_outer_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    sessions = _sessions()
    authority = _authority(sessions)
    binding = ThesisRunBinding.benchmark(spec)
    backend = ThesisTorchBackend(spec, authority, binding)
    partitions = (
        ((1, 2), 0),
        ((0, 2), 1),
        ((0, 1), 2),
    )
    requests = [
        _inner_request(
            tmp_path / f"inner_{number:02d}",
            sessions,
            train_indices=train_indices,
            validation_index=validation_index,
            inner_fold_id=f"inner_{number:02d}",
        )
        for number, (train_indices, validation_index) in enumerate(partitions, start=1)
    ]
    normalizers_before = [request.normalizer.to_dict() for request in requests]
    weights_before = [request.pos_weight.copy() for request in requests]

    def forbidden_refit(*args, **kwargs):
        raise AssertionError("backend must not refit runner statistics")

    monkeypatch.setattr(FeatureNormalizer, "fit", forbidden_refit)
    receipts = [backend.fit_inner(request) for request in requests]
    for request, receipt, normalizer_before, weight_before in zip(
        requests,
        receipts,
        normalizers_before,
        weights_before,
        strict=True,
    ):
        assert receipt.best_epoch == 1
        assert request.normalizer.to_dict() == normalizer_before
        np.testing.assert_array_equal(request.pos_weight, weight_before)
        checkpoint = load_checkpoint_artifact(receipt.checkpoint_path)
        assert checkpoint["counted_eligible"] is False
        assert checkpoint["data_kind"] == "real"
        assert checkpoint["aux_vad_weight"] == 0.0
        oof = thesis._load_prediction_artifact(
            receipt.oof_predictions_path,
            expected_inner_fold_id=receipt.inner_fold_id,
        )
        assert oof["prediction_role"] == "inner_oof_not_outer_test"

    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    calibration = backend.calibrate_inner_oof(
        CalibrationRequest(
            condition_id="A+H",
            outer_fold_id="outer_01",
            seed=13,
            inner_fold_ids=tuple(receipt.inner_fold_id for receipt in receipts),
            oof_prediction_paths=tuple(receipt.oof_predictions_path for receipt in receipts),
            oof_session_ids=tuple(
                session_id for receipt in receipts for session_id in receipt.validation_session_ids
            ),
            output_dir=calibration_dir,
        )
    )
    calibration_payload = json.loads(calibration.calibration_path.read_text(encoding="utf-8"))
    assert calibration_payload["thresholds"] is None
    assert calibration_payload["threshold_policy"] == THRESHOLD_POLICY
    assert calibration_payload["counted_eligible"] is False

    final_dir = tmp_path / "final"
    final_dir.mkdir()
    final_request = FinalFitRequest(
        condition_id="A+H",
        feature_set="audio_plus_head_pose",
        encoder="transformer",
        outer_fold_id="outer_01",
        seed=13,
        fixed_epochs=1,
        train_sessions=sessions[:3],
        normalizer=_normalizer(),
        pos_weight=np.ones(4, dtype=np.float32),
        calibration_path=calibration.calibration_path,
        output_dir=final_dir,
    )
    final = backend.fit_final(final_request)
    final_checkpoint = load_checkpoint_artifact(final.checkpoint_path)
    assert final.initialized_from_scratch is True
    assert final_checkpoint["fit_role"] == "fixed_epoch_outer_train_from_scratch"
    assert final_checkpoint["train_session_ids"] == [item.session_id for item in sessions[:3]]

    outer_dir = tmp_path / "outer"
    outer_dir.mkdir()
    evaluation_request = OuterEvaluationRequest(
        condition_id="A+H",
        outer_fold_id="outer_01",
        seed=13,
        checkpoint_path=final.checkpoint_path,
        checkpoint_sha256=thesis._sha256_file(final.checkpoint_path),
        calibration_path=calibration.calibration_path,
        test_sessions=(sessions[3],),
        output_dir=outer_dir,
    )
    evaluation = backend.evaluate_outer(evaluation_request)
    predictions = load_prediction_artifact(evaluation.predictions_path)
    metrics = load_metrics_artifact(evaluation.metrics_path)
    assert evaluation.evaluation_count == 1
    assert predictions["prediction_pass_count"] == 1
    assert predictions["prediction_role"] == "outer_test_single_saved_pass"
    assert predictions["threshold_policy"] == THRESHOLD_POLICY
    assert metrics["counted_eligible"] is False
    assert set(metrics["events"]) == {
        "shift_hold",
        "shift_prediction",
        "overlap_resolution",
        "robot_entry",
        "next_speaker",
    }

    corrupted_metrics = json.loads(evaluation.metrics_path.read_text(encoding="utf-8"))
    corrupted_metrics["frame"] = {}
    corrupted_path = tmp_path / "corrupted_metrics.artifact"
    corrupted_path.write_text(json.dumps(corrupted_metrics), encoding="utf-8")
    with pytest.raises(ThesisBackendGateError, match="recomputation"):
        load_metrics_artifact(
            corrupted_path,
            prediction_path=evaluation.predictions_path,
        )

    def forbidden_second_prediction(*args, **kwargs):
        raise AssertionError("create-once gate must run before outer inference")

    fresh_backend = ThesisTorchBackend(spec, authority, binding)
    monkeypatch.setattr(thesis, "_predict_sessions", forbidden_second_prediction)
    with pytest.raises(FileExistsError, match="overwrite thesis artifact"):
        fresh_backend.evaluate_outer(evaluation_request)

    retry_dir = tmp_path / "outer_retry"
    retry_dir.mkdir()
    with pytest.raises(ThesisBackendGateError, match="already predicted"):
        backend.evaluate_outer(replace(evaluation_request, output_dir=retry_dir))


def test_outer_inference_is_bit_identical_when_gold_is_replaced() -> None:
    spec = _spec()
    original = _sessions()[3]
    changed_gold = replace(
        original,
        labels=1.0 - original.labels,
        vad=1.0 - original.vad,
        valid=~original.valid,
    )
    torch.manual_seed(91)
    model = thesis._model(spec)

    original_probability = thesis._predict_sessions(
        model,
        (original,),
        _normalizer(),
        spec,
    )[0]
    changed_probability = thesis._predict_sessions(
        model,
        (changed_gold,),
        _normalizer(),
        spec,
    )[0]

    np.testing.assert_array_equal(original_probability, changed_probability)


def test_thesis_metrics_pass_future_horizon_validity_to_every_event_family() -> None:
    n_frames = 160
    vad = np.zeros((n_frames, 4), dtype=np.float32)
    vad[0:30, 2] = 1.0
    vad[35:90, 0] = 1.0
    vad[80:120, 2] = 1.0
    vad[125:150, 0] = 1.0
    valid = np.arange(n_frames) + 50 <= n_frames
    session = SessionTensors(
        session_id="validity_fixture",
        group_id="05_15",
        features=np.zeros((n_frames, 33), dtype=np.float32),
        labels=np.zeros((n_frames, 4, 4), dtype=np.float32),
        vad=vad,
        valid=valid,
        present=np.ones(4, dtype=np.float32),
    )
    probabilities = np.full((n_frames, 4, 4), 0.5, dtype=np.float32)

    metrics = thesis._metrics((session,), (probabilities,))

    assert metrics["events"]["shift_hold"]["n_events"] == 1
    assert metrics["events"]["shift_prediction"]["n_events"] == 1
    assert metrics["events"]["overlap_resolution"]["n_events"] == 1
    assert metrics["events"]["robot_entry"]["n_events"] == 1
    assert metrics["events"]["next_speaker"]["n_events"] == 1


def test_saved_outer_artifacts_are_pooled_not_fold_averaged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    groups = ("03_20", "03_26", "05_14", "05_15")
    sessions = tuple(
        _session(f"pooled_{index:02d}", group, index, n_frames=5 + 2 * index)
        for index, group in enumerate(groups, start=1)
    )
    authority = _authority(sessions, condition_id="A")
    binding = ThesisRunBinding.benchmark(spec)
    paths = []
    for index, session in enumerate(sessions, start=1):
        if index == 2:
            probability = np.where(session.labels > 0, 0.1, 0.9).astype(np.float32)
        elif index == 3:
            probability = np.full(session.labels.shape, 0.5, dtype=np.float32)
        else:
            probability = np.where(session.labels > 0, 0.9, 0.1).astype(np.float32)
        payload = thesis._prediction_payload(
            (session,),
            (probability,),
            condition_id="A",
            feature_set="audio_only",
            outer_fold_id=f"outer_{index:02d}",
            seed=13,
            checkpoint_sha256="b" * 64,
            prediction_role="outer_test_single_saved_pass",
            authority=authority,
            run_binding=binding,
        )
        path = tmp_path / f"outer_{index:02d}.artifact"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    def forbidden_model(*args, **kwargs):
        raise AssertionError("saved-only aggregation must not instantiate a model")

    monkeypatch.setattr(thesis, "_model", forbidden_model)
    summary = aggregate_outer_prediction_artifacts(tuple(paths))
    pooled_map = summary["pooled"]["frame"]["mAP"]
    fold_maps = [
        summary["per_fold"][fold_id]["frame"]["mAP"]
        for fold_id in ("outer_01", "outer_02", "outer_03", "outer_04")
    ]
    assert pooled_map is not None
    assert all(value is not None for value in fold_maps)
    assert not np.isclose(pooled_map, float(np.mean(fold_maps)))
    assert summary["aggregation_source"] == "saved_outer_probabilities_only_no_model_inference"
    assert set(summary["fold_group_ids"].values()) == set(groups)
    assert "events" in summary["pooled"]

    tampered = json.loads(paths[0].read_text(encoding="utf-8"))
    tampered["counted_eligible"] = True
    forged = tmp_path / "forged.artifact"
    forged.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ThesisBackendGateError, match="base field differs"):
        load_prediction_artifact(forged)
    with pytest.raises(ThesisBackendGateError, match="exactly four"):
        aggregate_outer_prediction_artifacts(tuple(paths[:3]))


def test_matrix_preflight_is_all_or_nothing_and_plans_exactly_twelve_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _spec()
    source = tmp_path / "audited_source"
    manifest = _run_manifest(spec, source)
    monkeypatch.setattr(runner, "_load_protocol", lambda path: ({}, manifest))
    prepare_calls = []
    prepared_by_condition = {}
    frame_by_prepared = {}

    def fake_prepare(protocol, *, source, condition_id):
        prepared = object()
        prepare_calls.append(condition_id)
        prepared_by_condition[condition_id] = prepared
        frame_by_prepared[prepared] = object()
        return prepared

    def fake_inputs(prepared):
        return frame_by_prepared[prepared], SimpleNamespace(
            session_ids=runner.EXPECTED_PRIMARY_SESSIONS,
            group_ids=runner.EXPECTED_GROUP_IDS,
            n_rows=outer.DEADLINE_EXPECTED_N_ROWS,
            target_ready=True,
            selected_feature_ready=True,
            to_dict=lambda: {},
        )

    authority_calls = []

    def fake_authority(frame, report, *, condition_id):
        authority_calls.append((condition_id, frame, report))
        return object()

    monkeypatch.setattr(runner, "prepare_outer_condition", fake_prepare)
    monkeypatch.setattr(runner, "prepared_outer_authority_inputs", fake_inputs)
    monkeypatch.setattr(runner, "build_real_session_authority", fake_authority)
    monkeypatch.setattr(runner, "ThesisTorchBackend", lambda *args: object())
    calls = []

    def fake_outer(protocol, output, **kwargs):
        calls.append(
            (
                kwargs["condition_id"],
                kwargs["outer_fold_id"],
                kwargs["seed"],
                kwargs["prepared"],
            )
        )
        output = Path(output)
        output.mkdir(parents=True)
        result = output / "result_manifest.json"
        result.write_text("{}\n", encoding="utf-8")
        return result

    monkeypatch.setattr(runner, "run_outer_unit", fake_outer)
    root = tmp_path / "matrix"
    index_path = runner.run_matrix(
        tmp_path / "protocol",
        source,
        root,
        seed=13,
        spec=spec,
    )
    expected = [
        (condition, fold, 13, prepared_by_condition[condition])
        for condition in ("A", "H", "A+H")
        for fold in runner.OUTER_FOLD_IDS
    ]
    assert calls == expected
    assert prepare_calls == ["A", "H", "A+H"]
    assert [item[0] for item in authority_calls] == ["A", "H", "A+H"]
    for condition in ("A", "H", "A+H"):
        assert {item[3] for item in calls if item[0] == condition} == {
            prepared_by_condition[condition]
        }
    assert capsys.readouterr().out == ""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["completed_unit_count"] == 12
    assert index["counted_eligible"] is False

    def forbidden_prepare(*args, **kwargs):
        raise AssertionError("existing matrix must fail before source/audit access")

    monkeypatch.setattr(runner, "prepare_outer_condition", forbidden_prepare)
    with pytest.raises(FileExistsError, match="overwrite or resume"):
        runner.run_matrix(
            tmp_path / "protocol",
            source,
            root,
            seed=13,
            spec=spec,
        )


def test_benchmark_uses_predeclared_inner_scope_and_prints_only_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sessions = (
        _session("bench_outer", "03_20", 10),
        _session("bench_val", "03_26", 11),
        _session("bench_train_a", "05_14", 12),
        _session("bench_train_b", "05_15", 13),
    )
    captured = []

    class FakeBackend:
        def __init__(self, *args):
            pass

        def fit_inner(self, request):
            captured.append(request)

    monkeypatch.setattr(runner, "_load_authority", lambda *args: (object(), None, object()))
    monkeypatch.setattr(runner, "apply_feature_set", lambda frame, feature_set: frame)
    monkeypatch.setattr(runner, "sessions_from_dataframe", lambda frame: list(sessions))
    monkeypatch.setattr(runner, "ThesisTorchBackend", FakeBackend)
    payload = runner.benchmark_inner(
        tmp_path / "source",
        tmp_path / "benchmark",
        condition_id="A+H",
        seed=13,
        spec=_spec(),
    )
    assert len(captured) == 1
    request = captured[0]
    assert {session.group_id for session in request.train_sessions} == {"05_14", "05_15"}
    assert {session.group_id for session in request.validation_sessions} == {"03_26"}
    assert all(session.group_id != "03_20" for session in request.train_sessions)
    assert request.max_epochs == 1
    assert not any("map" in key.lower() or "metric" in key.lower() for key in payload)

    console = runner.benchmark_console_payload(payload)
    assert set(console) == {
        "elapsed_seconds",
        "python_peak_memory_mb",
        "process_rss_before_mb",
        "process_rss_after_mb",
    }
    monkeypatch.setattr(runner, "benchmark_inner", lambda *args, **kwargs: payload)
    assert (
        runner.main(
            [
                "benchmark-inner",
                "--data",
                str(tmp_path / "source"),
                "--output",
                str(tmp_path / "unused"),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed == console
