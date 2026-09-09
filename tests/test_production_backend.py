from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hri_turn_taking.vap import outer_experiment as outer
from hri_turn_taking.vap import production_backend as production
from hri_turn_taking.vap.dataset import FeatureNormalizer, SessionTensors
from hri_turn_taking.vap.outer_experiment import (
    CalibrationRequest,
    FinalFitRequest,
    InnerFitRequest,
    OuterEvaluationRequest,
)
from hri_turn_taking.vap.production_backend import (
    COUNTED_BLOCKERS,
    FrozenTorchSpec,
    ProductionBackendGateError,
    ProductionEvidenceGateV3,
    ProductionTorchBackendV3,
    SessionTickBinding,
    TickAuthorityV3,
    load_calibration_artifact_v3,
    load_metrics_artifact_v3,
    load_prediction_artifact_v3,
    load_production_checkpoint_v3,
    session_content_sha256,
)


def _spec() -> FrozenTorchSpec:
    return FrozenTorchSpec(
        d_model=4,
        n_layers=1,
        n_heads=1,
        dropout=0.0,
        attention_window=4,
        window_frames=4,
        stride_frames=3,
        inference_chunk_frames=16,
        learning_rate=1e-3,
    )


def _session(session_id: str, index: int, *, invert_vad: bool = False) -> SessionTensors:
    n_frames = 9
    rng = np.random.default_rng(100 + index)
    features = rng.normal(size=(n_frames, 33)).astype(np.float32)
    time = np.arange(n_frames)[:, None, None]
    participant = np.arange(4)[None, :, None]
    bin_index = np.arange(4)[None, None, :]
    labels = ((time + participant + bin_index + index) % 2).astype(np.float32)
    vad = ((np.arange(n_frames)[:, None] + np.arange(4)[None, :] + index) % 2).astype(
        np.float32
    )
    if invert_vad:
        vad = 1.0 - vad
    return SessionTensors(
        session_id=session_id,
        features=features,
        labels=labels,
        vad=vad,
        valid=np.ones(n_frames, dtype=bool),
        present=np.ones(4, dtype=np.float32),
        group_id=f"group_{index}",
    )


def _sessions(*, invert_vad: bool = False) -> tuple[SessionTensors, ...]:
    return tuple(
        _session(f"fixture_{index:02d}", index, invert_vad=invert_vad)
        for index in range(1, 7)
    )


def _authority(sessions: tuple[SessionTensors, ...]) -> TickAuthorityV3:
    bindings = []
    for index, session in enumerate(sessions):
        ticks = tuple(range(1000 * index + 17, 1000 * index + 17 + len(session.valid)))
        bindings.append(
            SessionTickBinding(
                session_id=session.session_id,
                tick_25hz=ticks,
                session_content_sha256=session_content_sha256(session, ticks),
            )
        )
    return TickAuthorityV3(tuple(bindings))


def _normalizer() -> FeatureNormalizer:
    return FeatureNormalizer(
        participant_mean=np.zeros(7, dtype=np.float32),
        participant_std=np.ones(7, dtype=np.float32),
        global_mean=np.zeros(5, dtype=np.float32),
        global_std=np.ones(5, dtype=np.float32),
    )


def _inner_request(
    output_dir: Path,
    sessions: tuple[SessionTensors, ...],
    *,
    inner_fold_id: str = "inner_01",
    validation_index: int = 2,
    train_indices: tuple[int, ...] = (0, 1),
    max_epochs: int = 1,
) -> InnerFitRequest:
    output_dir.mkdir()
    return InnerFitRequest(
        condition_id="A+H",
        feature_set="audio_plus_head_pose",
        encoder="transformer",
        outer_fold_id="outer_group_04",
        inner_fold_id=inner_fold_id,
        seed=13,
        max_epochs=max_epochs,
        train_sessions=tuple(sessions[index] for index in train_indices),
        validation_sessions=(sessions[validation_index],),
        normalizer=_normalizer(),
        pos_weight=np.ones(4, dtype=np.float32),
        output_dir=output_dir,
    )


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    return load_production_checkpoint_v3(path).model.state_dict()


def test_frozen_spec_has_exact_schema_and_hash() -> None:
    spec = _spec()
    payload = spec.to_payload()

    assert spec.input_dim == 33
    assert len(spec.sha256) == 64
    assert FrozenTorchSpec.from_payload(payload) == spec

    reordered = json.loads(json.dumps(payload))
    reordered["participant_features"][:2] = reversed(reordered["participant_features"][:2])
    with pytest.raises(ProductionBackendGateError, match="participant feature schema"):
        FrozenTorchSpec.from_payload(reordered)

    bad_hash = json.loads(json.dumps(payload))
    bad_hash["input_schema_sha256"] = "0" * 64
    with pytest.raises(ProductionBackendGateError, match="input schema hash"):
        FrozenTorchSpec.from_payload(bad_hash)

    wrong_numeric_type = json.loads(json.dumps(payload))
    wrong_numeric_type["dropout"] = 0
    with pytest.raises(ProductionBackendGateError, match="exact JSON float"):
        FrozenTorchSpec.from_payload(wrong_numeric_type)

    with pytest.raises(ProductionBackendGateError, match="cannot self-assert"):
        ProductionEvidenceGateV3(environment_capture_sha256="0" * 64)


def test_tick_authority_is_fixture_only_contiguous_and_content_bound(tmp_path: Path) -> None:
    sessions = _sessions()
    authority = _authority(sessions)
    backend = ProductionTorchBackendV3(_spec(), authority)
    sessions[0].features[0, 0] += 1.0

    with pytest.raises(ProductionBackendGateError, match="session bytes differ"):
        backend.fit_inner(_inner_request(tmp_path / "inner", sessions))

    with pytest.raises(ProductionBackendGateError, match="rejects non-fixture"):
        SessionTickBinding(
            "real_scene",
            (1, 2),
            "0" * 64,
            data_kind="real",
        )
    with pytest.raises(ProductionBackendGateError, match="strictly contiguous"):
        SessionTickBinding("fixture", (1, 3), "0" * 64)


def test_training_windows_cover_non_aligned_tail() -> None:
    sessions = (_session("fixture_01", 1),)
    windows = production._training_windows(sessions, window_frames=4, stride_frames=3)

    assert [(start, end) for _, start, end in windows] == [(0, 4), (3, 7), (5, 9)]
    assert windows[-1][2] == sessions[0].features.shape[0]


def test_inner_fit_uses_runner_statistics_and_strict_non_counted_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = _sessions()
    authority = _authority(sessions)
    backend = ProductionTorchBackendV3(_spec(), authority)
    request = _inner_request(tmp_path / "inner", sessions, max_epochs=2)
    normalizer_before = request.normalizer.to_dict()
    pos_weight_before = request.pos_weight.copy()

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("backend must not refit the normalizer")

    monkeypatch.setattr(FeatureNormalizer, "fit", forbidden_fit)
    receipt = backend.fit_inner(request)

    assert 1 <= receipt.best_epoch <= 2
    assert request.normalizer.to_dict() == normalizer_before
    np.testing.assert_array_equal(request.pos_weight, pos_weight_before)
    checkpoint = load_production_checkpoint_v3(receipt.checkpoint_path)
    predictions = load_prediction_artifact_v3(receipt.oof_predictions_path)
    assert checkpoint.payload["counted_eligible"] is False
    assert checkpoint.payload["counted_blockers"] == list(COUNTED_BLOCKERS)
    assert checkpoint.payload["aux_vad_weight"] == 0.0
    assert checkpoint.payload["history"]["entries"][0]["epoch"] == 1
    assert checkpoint.payload["history"]["entries"][-1]["epoch"] == 2
    assert predictions["counted_eligible"] is False
    record = predictions["sessions"][0]
    binding = authority.by_session()[record["session_id"]]
    assert record["tick_25hz"] == list(binding.tick_25hz)
    assert len(record["probabilities"]) == sessions[2].features.shape[0]
    assert record["tick_end_exclusive"] == binding.tick_25hz[-1] + 1


def test_auxiliary_vad_is_completely_isolated_from_fit(tmp_path: Path) -> None:
    original = _sessions()
    inverted = _sessions(invert_vad=True)
    backend_a = ProductionTorchBackendV3(_spec(), _authority(original))
    backend_b = ProductionTorchBackendV3(_spec(), _authority(inverted))

    receipt_a = backend_a.fit_inner(_inner_request(tmp_path / "a", original))
    receipt_b = backend_b.fit_inner(_inner_request(tmp_path / "b", inverted))

    state_a = _state_dict(receipt_a.checkpoint_path)
    state_b = _state_dict(receipt_b.checkpoint_path)
    assert state_a.keys() == state_b.keys()
    for name in state_a:
        torch.testing.assert_close(state_a[name], state_b[name], rtol=0.0, atol=0.0)


def test_complete_scaffold_lifecycle_is_full_session_but_never_counted(tmp_path: Path) -> None:
    sessions = _sessions()
    authority = _authority(sessions)
    backend = ProductionTorchBackendV3(_spec(), authority)
    receipts = []
    inner_partitions = (
        ((1, 2), 0),
        ((0, 2), 1),
        ((0, 1), 2),
    )
    for number, (train_indices, validation_index) in enumerate(inner_partitions, start=1):
        request = _inner_request(
            tmp_path / f"inner_{number}",
            sessions,
            inner_fold_id=f"inner_0{number}",
            validation_index=validation_index,
            train_indices=train_indices,
        )
        receipts.append(backend.fit_inner(request))

    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    calibration_request = CalibrationRequest(
        condition_id="A+H",
        outer_fold_id="outer_group_04",
        seed=13,
        inner_fold_ids=tuple(receipt.inner_fold_id for receipt in receipts),
        oof_prediction_paths=tuple(receipt.oof_predictions_path for receipt in receipts),
        oof_session_ids=tuple(
            session_id for receipt in receipts for session_id in receipt.validation_session_ids
        ),
        output_dir=calibration_dir,
    )
    calibration = backend.calibrate_inner_oof(calibration_request)
    calibration_payload = load_calibration_artifact_v3(calibration.calibration_path)
    assert calibration_payload["thresholds"] is None
    assert calibration_payload["counted_eligible"] is False

    wrong_final_dir = tmp_path / "wrong_final"
    wrong_final_dir.mkdir()
    wrong_final = FinalFitRequest(
        condition_id="A+H",
        feature_set="audio_plus_head_pose",
        encoder="transformer",
        outer_fold_id="outer_group_04",
        seed=13,
        fixed_epochs=1,
        train_sessions=sessions[:2],
        normalizer=_normalizer(),
        pos_weight=np.ones(4, dtype=np.float32),
        calibration_path=calibration.calibration_path,
        output_dir=wrong_final_dir,
    )
    with pytest.raises(ProductionBackendGateError, match="exactly cover final-train"):
        backend.fit_final(wrong_final)

    final_dir = tmp_path / "final"
    final_dir.mkdir()
    final_request = FinalFitRequest(
        condition_id="A+H",
        feature_set="audio_plus_head_pose",
        encoder="transformer",
        outer_fold_id="outer_group_04",
        seed=13,
        fixed_epochs=1,
        train_sessions=sessions[:3],
        normalizer=_normalizer(),
        pos_weight=np.ones(4, dtype=np.float32),
        calibration_path=calibration.calibration_path,
        output_dir=final_dir,
    )
    final = backend.fit_final(final_request)
    final_payload = load_production_checkpoint_v3(final.checkpoint_path).payload
    assert final_payload["selected_epoch"] == 1
    assert final_payload["validation_session_ids"] == []

    outer_dir = tmp_path / "outer"
    outer_dir.mkdir()
    evaluation_request = OuterEvaluationRequest(
        condition_id="A+H",
        outer_fold_id="outer_group_04",
        seed=13,
        checkpoint_path=final.checkpoint_path,
        checkpoint_sha256=production._sha256_file(final.checkpoint_path),
        calibration_path=calibration.calibration_path,
        test_sessions=(sessions[5],),
        output_dir=outer_dir,
    )
    evaluation = backend.evaluate_outer(evaluation_request)
    predictions = load_prediction_artifact_v3(evaluation.predictions_path)
    metrics = load_metrics_artifact_v3(evaluation.metrics_path)
    assert evaluation.evaluation_count == 1
    assert predictions["full_session_inference"] is True
    assert len(predictions["sessions"][0]["tick_25hz"]) == len(sessions[5].valid)
    assert metrics["counted_eligible"] is False
    assert metrics["threshold_status"] == "diagnostic_fixed_0.5_owner_policy_pending"
    assert set(metrics["events"]) == {
        "shift_hold",
        "shift_prediction",
        "overlap_resolution",
        "robot_entry",
        "next_speaker",
    }
    forged_metrics = json.loads(evaluation.metrics_path.read_text(encoding="utf-8"))
    forged_metrics["frame"]["mAP"] = "forged"
    forged_metrics_path = tmp_path / "forged_metrics.artifact"
    forged_metrics_path.write_text(json.dumps(forged_metrics), encoding="utf-8")
    with pytest.raises(ProductionBackendGateError, match="finite number"):
        load_metrics_artifact_v3(forged_metrics_path)

    overlap_dir = tmp_path / "overlap_outer"
    overlap_dir.mkdir()
    overlap_request = OuterEvaluationRequest(
        condition_id="A+H",
        outer_fold_id="outer_group_04",
        seed=13,
        checkpoint_path=final.checkpoint_path,
        checkpoint_sha256=production._sha256_file(final.checkpoint_path),
        calibration_path=calibration.calibration_path,
        test_sessions=(sessions[0],),
        output_dir=overlap_dir,
    )
    with pytest.raises(ProductionBackendGateError, match="overlap checkpoint training"):
        backend.evaluate_outer(overlap_request)
    with pytest.raises(ProductionBackendGateError, match="already evaluated"):
        backend.evaluate_outer(evaluation_request)


def test_strict_loaders_reject_schema_tampering(tmp_path: Path) -> None:
    sessions = _sessions()
    backend = ProductionTorchBackendV3(_spec(), _authority(sessions))
    receipt = backend.fit_inner(_inner_request(tmp_path / "inner", sessions))

    checkpoint_payload = torch.load(receipt.checkpoint_path, weights_only=True)
    checkpoint_payload["counted_eligible"] = True
    bad_checkpoint = tmp_path / "bad_checkpoint.artifact"
    torch.save(checkpoint_payload, bad_checkpoint)
    with pytest.raises(ProductionBackendGateError, match="cannot be counted"):
        load_production_checkpoint_v3(bad_checkpoint)

    prediction_payload = json.loads(receipt.oof_predictions_path.read_text(encoding="utf-8"))
    prediction_payload["unexpected"] = "field"
    bad_prediction = tmp_path / "bad_predictions.artifact"
    bad_prediction.write_text(json.dumps(prediction_payload), encoding="utf-8")
    with pytest.raises(ProductionBackendGateError, match="fields are not exact"):
        load_prediction_artifact_v3(bad_prediction)

    shifted = json.loads(receipt.oof_predictions_path.read_text(encoding="utf-8"))
    record = shifted["sessions"][0]
    record["tick_25hz"] = [tick + 100 for tick in record["tick_25hz"]]
    record["tick_start"] += 100
    record["tick_end_exclusive"] += 100
    shifted_path = tmp_path / "shifted_predictions.artifact"
    shifted_path.write_text(json.dumps(shifted), encoding="utf-8")
    with pytest.raises(ProductionBackendGateError, match="differ from tick authority"):
        load_prediction_artifact_v3(shifted_path)

    invalid_normalizer = _normalizer().to_dict()
    invalid_normalizer["participant_mean"] = [False] * 7
    with pytest.raises(ProductionBackendGateError, match="exact finite JSON floats"):
        production._normalizer_from_payload(invalid_normalizer, _spec())

    corrupted = tmp_path / "corrupted_checkpoint.artifact"
    corrupted.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ProductionBackendGateError, match="cannot load v3 checkpoint"):
        load_production_checkpoint_v3(corrupted)


def test_backend_is_deliberately_absent_from_outer_registry() -> None:
    backend = ProductionTorchBackendV3(_spec(), _authority(_sessions()))

    assert not isinstance(backend, outer.BackendContractSmokeBackend)
    assert outer._PRODUCTION_BACKEND_REGISTRY == ()
    assert backend.counted_blockers == COUNTED_BLOCKERS
    assert "backend_not_registered" in backend.counted_blockers
