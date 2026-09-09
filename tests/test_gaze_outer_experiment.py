from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from hri_turn_taking.data.gaze_posthoc import (
    EXACT_EIGHT_GROUP_BY_SCENE,
    EXCLUDED_OWNER_HOLD_SCENES,
    REAL_MATRIX_ARTIFACT_KIND,
    REAL_PERMITTED_USE,
    RealGazePosthocMatrixArtifact,
)
from hri_turn_taking.data.multiparty import (
    FEATURE_SCHEMA_V3_GAZE_VALIDITY,
    GLOBAL_FEATURE_COLUMNS,
    PARTICIPANT_FEATURE_NAMES_V3_GAZE_VALIDITY,
    participant_column,
    validate_multiparty_dataframe,
)
from hri_turn_taking.vap import gaze_outer_experiment as outer
from hri_turn_taking.vap import gaze_posthoc_backend as backend


def _small_frame(scene: str, group: str, start: int, end: int) -> pd.DataFrame:
    local = np.arange(end - start)
    ticks = np.arange(start, end, dtype=np.int64)
    data: dict[str, object] = {
        "session_id": np.repeat(scene, len(ticks)),
        "tick_25hz": ticks,
        "timestamp_s": ticks.astype(float) / 25.0,
    }
    for slot in range(1, 5):
        data[participant_column(slot, "present")] = np.ones(len(ticks))
        data[participant_column(slot, "vad")] = (
            (local + slot * 7) % (31 + slot * 3) < 10 + slot
        ).astype(float)
        for feature in PARTICIPANT_FEATURE_NAMES_V3_GAZE_VALIDITY:
            data[participant_column(slot, feature)] = np.zeros(len(ticks))
        data[participant_column(slot, "head_yaw_deg")] = np.sin(
            (local + slot) / 7.0
        )
        data[participant_column(slot, "pose_valid")] = np.ones(len(ticks))
        data[participant_column(slot, "pose_quality")] = np.ones(len(ticks))
        if slot > 1:
            valid = ((local + slot) % 4 != 0).astype(float)
            gaze = (((local + slot) % 3) == 0).astype(float) * valid
            data[participant_column(slot, "gaze_robot_valid")] = valid
            data[participant_column(slot, "gaze_robot")] = gaze
    voiced = (local % 5 != 0).astype(float)
    data["doa_angle_deg"] = np.zeros(len(ticks))
    data["array_energy"] = np.cos(local / 9.0)
    data["voiced"] = voiced
    data["f0_proxy"] = voiced * (100.0 + local % 20)
    frame = pd.DataFrame(data)
    validate_multiparty_dataframe(
        frame,
        feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY,
    )
    return frame


@pytest.fixture
def small_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    scenes = tuple(EXACT_EIGHT_GROUP_BY_SCENE)
    ranges = {
        scene: (1000 * index, 1000 * index + 80)
        for index, scene in enumerate(scenes, start=1)
    }
    total_rows = sum(end - start for start, end in ranges.values())
    monkeypatch.setattr(outer, "EXACT_EIGHT_TICK_RANGES", ranges)
    monkeypatch.setattr(outer, "EXACT_EIGHT_TOTAL_ROWS", total_rows)
    monkeypatch.setattr(backend, "EXACT_EIGHT_TICK_RANGES", ranges)
    monkeypatch.setattr(backend, "EXACT_EIGHT_TOTAL_ROWS", total_rows)
    frames = {
        scene: _small_frame(
            scene,
            EXACT_EIGHT_GROUP_BY_SCENE[scene],
            *ranges[scene],
        )
        for scene in scenes
    }
    source_dir = tmp_path / "source_matrix"
    source_dir.mkdir()
    manifest = {
        "artifact_kind": REAL_MATRIX_ARTIFACT_KIND,
        "format_version": 1,
        "permitted_use": REAL_PERMITTED_USE,
        "posthoc_exploratory_training_ready": True,
        "formal_outer_runner_admitted": False,
        "assembly_eligible": False,
        "production_eligible": False,
        "counted_eligible": False,
        "upstream_exact_regeneration_available": False,
        "implementation_counterfactual_causality_verified": False,
        "method_evidence": {"fixture": True},
        "source_corpus_scene_count": 10,
        "selected_scene_count": 8,
        "exact_scene_inventory": list(scenes),
        "excluded_owner_hold_scenes": dict(EXCLUDED_OWNER_HOLD_SCENES),
        "total_rows": total_rows,
        "group_ids": list(outer.EXACT_EIGHT_GROUPS),
        "n_groups": 4,
        "schema": {
            "feature_schema": FEATURE_SCHEMA_V3_GAZE_VALIDITY,
            "raw_input_columns": [
                participant_column(slot, feature)
                for slot in range(1, 5)
                for feature in PARTICIPANT_FEATURE_NAMES_V3_GAZE_VALIDITY
            ]
            + list(GLOBAL_FEATURE_COLUMNS),
            "raw_input_width": 36,
            "transformed_input_width": 37,
            "token_input_width": 13,
        },
        "output_directory": str(source_dir.resolve()),
        "scenes": {
            scene: {"group_id": EXACT_EIGHT_GROUP_BY_SCENE[scene]}
            for scene in scenes
        },
    }
    manifest_path = source_dir / "matrix.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact = RealGazePosthocMatrixArtifact(
        output_dir=source_dir.resolve(),
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        frames=frames,
    )
    calls: list[Path] = []

    def audit(path: str | Path) -> RealGazePosthocMatrixArtifact:
        calls.append(Path(path).resolve())
        return artifact

    monkeypatch.setattr(outer, "audit_real_posthoc_matrix", audit)
    return SimpleNamespace(
        artifact=artifact,
        calls=calls,
        ranges=ranges,
        total_rows=total_rows,
    )


def _history(epochs: int, *, validation: bool) -> list[dict[str, object]]:
    return [
        {
            "epoch": epoch,
            "train_loss": 1.0 / epoch,
            "validation_mAP": 0.4 + epoch * 0.01 if validation else None,
            "learning_rate": 3e-4,
            "n_training_windows": 2,
            "n_optimizer_steps": 1,
        }
        for epoch in range(1, epochs + 1)
    ]


def _predictions_for(
    sessions: tuple[backend.SessionTensors, ...],
    condition_id: str,
) -> tuple[np.ndarray, ...]:
    condition_bias = {"C0": 0.0, "CM": 0.02, "CG": 0.04}[condition_id]
    predictions = []
    for session in sessions:
        time_score = 0.45 + 0.20 * np.tanh(session.features[:, -3])
        probability = np.empty((len(time_score), 4, 4), dtype=np.float32)
        for slot in range(4):
            for bin_index in range(4):
                probability[:, slot, bin_index] = np.clip(
                    time_score
                    + condition_bias
                    + 0.01 * slot
                    - 0.005 * bin_index,
                    0.0,
                    1.0,
                )
        predictions.append(probability)
    return tuple(predictions)


def _checkpoint_payload(
    request: backend.GazeUnitRequest,
    *,
    selected_final_epochs: int,
    final_history: list[dict[str, object]],
) -> dict[str, object]:
    spec = backend.GazeBackendSpec()
    torch.manual_seed(request.seed)
    model = backend._model(spec)
    outer_train = tuple(
        session
        for session in request.sessions
        if session.group_id in request.outer_train_groups
    )
    outer_test = tuple(
        session
        for session in request.sessions
        if session.group_id == request.outer_test_group
    )
    return {
        "backend_spec": spec.to_dict(),
        "model_config": backend.gaze_model_config(spec),
        "trainer_spec": backend.gaze_trainer_spec(spec),
        "label_config": backend.gaze_label_config(),
        "model_state": model.state_dict(),
        "model_state_sha256": backend.model_state_sha256(model.state_dict()),
        "normalizer": backend.fit_gaze_training_normalizer(
            request.sessions,
            request.outer_train_groups,
        ).to_dict(),
        "outer_train_groups": list(request.outer_train_groups),
        "outer_train_session_ids": [session.session_id for session in outer_train],
        "outer_test_session_ids": [session.session_id for session in outer_test],
        "selected_final_epochs": selected_final_epochs,
        "final_history": final_history,
        "initial_state_sha256": request.expected_initial_state_sha256,
    }


def _inner_fits_for(
    request: backend.GazeUnitRequest,
    metrics: dict[str, object],
) -> list[dict[str, object]]:
    records = []
    for index, group in enumerate(request.inner_validation_groups, start=1):
        training_groups = tuple(
            value for value in request.outer_train_groups if value != group
        )
        records.append(
            {
                "inner_fold_id": f"inner_{index:02d}",
                "training_groups": list(training_groups),
                "validation_group": group,
                "training_sessions": [
                    session.session_id
                    for session in request.sessions
                    if session.group_id in training_groups
                ],
                "validation_sessions": [
                    session.session_id
                    for session in request.sessions
                    if session.group_id == group
                ],
                "selected_epoch": 2,
                "initial_state_sha256": request.expected_initial_state_sha256,
                "validation_metrics": metrics,
                "history": _history(2, validation=True),
            }
        )
    return records


class _FixtureBackend:
    def __init__(self) -> None:
        self.requests: list[backend.GazeUnitRequest] = []

    def run_unit(
        self,
        request: backend.GazeUnitRequest,
    ) -> backend.GazeUnitReceipt:
        self.requests.append(request)
        assert request.fixture_only is True
        outer_sessions = tuple(
            session
            for session in request.sessions
            if session.group_id == request.outer_test_group
        )
        predictions_tuple = _predictions_for(outer_sessions, request.condition_id)
        metrics = backend.gaze_metrics_from_predictions(
            outer_sessions,
            predictions_tuple,
        )
        final_history = _history(2, validation=False)
        inner_fits = _inner_fits_for(request, metrics)
        return backend.write_gaze_unit_bundle(
            request,
            checkpoint_payload=_checkpoint_payload(
                request,
                selected_final_epochs=2,
                final_history=final_history,
            ),
            outer_sessions=outer_sessions,
            predictions=predictions_tuple,
            metrics=metrics,
            inner_fits=inner_fits,
            selected_final_epochs=2,
            final_history=final_history,
        )


@pytest.fixture
def fast_formal_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fit(
        spec: backend.GazeBackendSpec,
        _train_sessions: tuple[backend.SessionTensors, ...],
        validation_sessions: tuple[backend.SessionTensors, ...],
        _normalizer: object,
        _pos_weight: np.ndarray,
        *,
        epochs: int,
        seed: int,
    ) -> backend._FitResult:
        torch.manual_seed(seed)
        model = backend._model(spec)
        selected = min(2, epochs)
        return backend._FitResult(
            model=model,
            history=_history(selected, validation=bool(validation_sessions)),
            selected_epoch=selected,
            initial_state_sha256=backend.model_state_sha256(model.state_dict()),
        )

    def fake_predict(
        _model: object,
        sessions: tuple[backend.SessionTensors, ...],
        _normalizer: object,
        _spec: backend.GazeBackendSpec,
    ) -> tuple[np.ndarray, ...]:
        return _predictions_for(sessions, "C0")

    monkeypatch.setattr(backend, "_fit", fake_fit)
    monkeypatch.setattr(backend, "_predict", fake_predict)


def test_protocol_is_exact8_v3_posthoc_and_create_once(
    tmp_path: Path,
    small_source: SimpleNamespace,
) -> None:
    protocol = outer.freeze_gaze_outer_protocol(
        small_source.artifact.output_dir,
        tmp_path / "protocol",
    )
    payload = protocol.protocol
    assert [item["condition_id"] for item in payload["conditions"]] == [
        "C0",
        "CM",
        "CG",
    ]
    assert payload["training"] == {
        "seed": 13,
        "max_epochs": 15,
        "aux_vad_weight": 0.0,
        "epoch_selection": "median_inner_logo_selected_epoch",
        "outer_test_access": "one_saved_probability_pass_per_condition_fold",
    }
    assert payload["schema_dimensions"] == {
        "feature_schema": FEATURE_SCHEMA_V3_GAZE_VALIDITY,
        "raw_input_columns": 36,
        "transformed_input_dimensions": 37,
        "participant_features_per_token": 8,
        "global_features_per_token": 5,
        "token_input_dimensions": 13,
    }
    assert len(payload["outer_folds"]) == 4
    assert all(len(fold["inner_folds"]) == 3 for fold in payload["outer_folds"])
    frozen_backend = payload["backend"]
    assert frozen_backend["backend_spec"] == backend.GazeBackendSpec().to_dict()
    assert frozen_backend["model_config"] == backend.gaze_model_config(
        backend.GazeBackendSpec()
    )
    assert frozen_backend["trainer_spec"] == backend.gaze_trainer_spec(
        backend.GazeBackendSpec()
    )
    assert frozen_backend["label_config"] == backend.gaze_label_config()
    initial_hash = frozen_backend["initial_state_strategy"]["sha256"]
    assert initial_hash == backend.initial_model_state_sha256(
        backend.GazeBackendSpec(), 13
    )
    assert all(
        set(fold["condition_initial_state_sha256"].values()) == {initial_hash}
        for fold in payload["outer_folds"]
    )
    assert payload["production_eligible"] is False
    assert payload["counted_eligible"] is False
    with pytest.raises(FileExistsError):
        outer.freeze_gaze_outer_protocol(
            small_source.artifact.output_dir,
            protocol.directory,
        )


def test_condition_sessions_retain_v3_width_and_exact_gaze_masks(
    small_source: SimpleNamespace,
) -> None:
    sessions = {
        condition: backend.build_gaze_condition_sessions(
            small_source.artifact.frames,
            condition,
            scene_to_group=outer._scene_to_group(small_source.artifact),
        )
        for condition in ("C0", "CM", "CG")
    }
    p2_start = 8
    gaze_value = p2_start + PARTICIPANT_FEATURE_NAMES_V3_GAZE_VALIDITY.index(
        "gaze_robot"
    )
    gaze_valid = p2_start + PARTICIPANT_FEATURE_NAMES_V3_GAZE_VALIDITY.index(
        "gaze_robot_valid"
    )
    c0 = sessions["C0"][0].features
    cm = sessions["CM"][0].features
    cg = sessions["CG"][0].features
    assert c0.shape[1] == cm.shape[1] == cg.shape[1] == 37
    assert np.count_nonzero(c0[:, gaze_value]) == 0
    assert np.count_nonzero(c0[:, gaze_valid]) == 0
    assert np.count_nonzero(cm[:, gaze_value]) == 0
    assert np.array_equal(cm[:, gaze_valid], cg[:, gaze_valid])
    assert np.count_nonzero(cg[:, gaze_value]) > 0

    training_groups = tuple(outer.EXACT_EIGHT_GROUPS[:-1])
    normalizer = backend.fit_gaze_training_normalizer(
        sessions["CG"],
        training_groups,
    )
    valid_values = []
    for session in sessions["CG"]:
        if session.group_id not in training_groups:
            continue
        for slot in range(4):
            start = slot * 8
            validity = session.features[:, start + 3] == 1.0
            valid_values.append(session.features[validity, start + 2])
    assert normalizer.participant_mean[2] == pytest.approx(
        np.concatenate(valid_values).mean()
    )
    held_out_changed = []
    for session in sessions["CG"]:
        if session.group_id != outer.EXACT_EIGHT_GROUPS[-1]:
            held_out_changed.append(session)
            continue
        features = session.features.copy()
        for slot in range(4):
            start = slot * 8
            validity = features[:, start + 3] == 1.0
            features[validity, start + 2] = 999.0
        held_out_changed.append(replace(session, features=features))
    repeated = backend.fit_gaze_training_normalizer(
        held_out_changed,
        training_groups,
    )
    assert repeated.to_dict() == normalizer.to_dict()


def test_publisher_style_manifest_groups_are_injected_in_memory_and_conflicts_fail(
    small_source: SimpleNamespace,
) -> None:
    mapping = outer._scene_to_group(small_source.artifact)
    assert all("group_id" not in frame for frame in small_source.artifact.frames.values())
    sessions = backend.build_gaze_condition_sessions(
        small_source.artifact.frames,
        "CG",
        scene_to_group=mapping,
    )
    assert {
        session.session_id: session.group_id for session in sessions
    } == mapping
    assert all("group_id" not in frame for frame in small_source.artifact.frames.values())

    conflicting = dict(small_source.artifact.frames)
    first_scene = next(iter(conflicting))
    conflicting_frame = conflicting[first_scene].copy()
    conflicting_frame["group_id"] = "not_the_manifest_group"
    conflicting[first_scene] = conflicting_frame
    with pytest.raises(backend.GazePosthocBackendError, match="conflicts"):
        backend.build_gaze_condition_sessions(
            conflicting,
            "CG",
            scene_to_group=mapping,
        )


def test_formal_matrix_is_12_units_and_aggregate_uses_saved_probabilities_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_source: SimpleNamespace,
    fast_formal_backend: None,
) -> None:
    protocol = outer.freeze_gaze_outer_protocol(
        small_source.artifact.output_dir,
        tmp_path / "protocol",
    )
    matrix = outer.run_gaze_outer_matrix(
        protocol.directory,
        small_source.artifact.output_dir,
        tmp_path / "matrix",
    )
    assert matrix.index["completed_units"] == 12
    assert matrix.index["artifact_kind"] == outer.FORMAL_MATRIX_KIND
    assert matrix.index["fixture_only"] is False
    assert len(matrix.units) == 12
    for fold_id in outer.OUTER_FOLD_IDS:
        assert len(
            {
                matrix.units[(condition, fold_id)].result["initial_state_sha256"]
                for condition in outer.CONDITIONS
            }
        ) == 1

    checkpoint = matrix.directory / "C0" / "outer_01" / backend.CHECKPOINT_FILENAME
    first_result = matrix.units[("C0", "outer_01")].result
    backend.load_gaze_checkpoint_artifact(
        checkpoint,
        condition_id="C0",
        fold_id="outer_01",
        outer_test_group=outer.EXACT_EIGHT_GROUPS[0],
        source_matrix_manifest_sha256=first_result[
            "source_matrix_manifest_sha256"
        ],
        protocol_sha256=first_result["protocol_sha256"],
        expected_initial_state_sha256=first_result["initial_state_sha256"],
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_payload["feature_schema"] = "multiparty_v2"
    tampered_checkpoint = tmp_path / "cross_schema.pt"
    torch.save(checkpoint_payload, tampered_checkpoint)
    with pytest.raises(backend.GazePosthocBackendError, match="feature_schema"):
        backend.load_gaze_checkpoint_artifact(
            tampered_checkpoint,
            condition_id="C0",
            fold_id="outer_01",
            outer_test_group=outer.EXACT_EIGHT_GROUPS[0],
            source_matrix_manifest_sha256=first_result[
                "source_matrix_manifest_sha256"
            ],
            protocol_sha256=first_result["protocol_sha256"],
            expected_initial_state_sha256=first_result["initial_state_sha256"],
        )

    with np.load(
        matrix.directory / "C0" / "outer_01" / backend.PREDICTIONS_FILENAME,
        allow_pickle=False,
    ) as saved:
        assert set(saved.files) == {"probabilities", "session_offsets"}
        assert saved["probabilities"].dtype == np.float32
        assert saved["session_offsets"].dtype == np.int64
        assert all(not saved[name].dtype.hasobject for name in saved.files)

    unit_directory = matrix.directory / "C0" / "outer_01"
    result_path = unit_directory / backend.RESULT_FILENAME
    original_result = result_path.read_bytes()
    outer_sessions = tuple(
        session
        for session in backend.build_gaze_condition_sessions(
            small_source.artifact.frames,
            "C0",
            scene_to_group=outer._scene_to_group(small_source.artifact),
        )
        if session.group_id == outer.EXACT_EIGHT_GROUPS[0]
    )
    load_kwargs = {
        "condition_id": "C0",
        "fold_id": "outer_01",
        "outer_test_group": outer.EXACT_EIGHT_GROUPS[0],
        "source_matrix_manifest_sha256": first_result[
            "source_matrix_manifest_sha256"
        ],
        "protocol_sha256": first_result["protocol_sha256"],
        "expected_initial_state_sha256": first_result["initial_state_sha256"],
    }
    invented = json.loads(original_result)
    invented["metrics"]["frame"]["mAP"] = 0.123456
    result_path.write_text(json.dumps(invented) + "\n", encoding="utf-8")
    with pytest.raises(backend.GazePosthocBackendError, match="recomputation"):
        backend.load_saved_gaze_unit(unit_directory, outer_sessions, **load_kwargs)
    result_path.write_bytes(original_result)

    wrong_median = json.loads(original_result)
    for inner in wrong_median["inner_fits"]:
        inner["selected_epoch"] = 1
        inner["history"][0]["validation_mAP"] = 0.9
        inner["history"][1]["validation_mAP"] = 0.8
    result_path.write_text(json.dumps(wrong_median) + "\n", encoding="utf-8")
    with pytest.raises(backend.GazePosthocBackendError, match="median"):
        backend.load_saved_gaze_unit(unit_directory, outer_sessions, **load_kwargs)
    result_path.write_bytes(original_result)

    def forbidden_inference(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("aggregate must not run model inference")

    monkeypatch.setattr(
        backend,
        "predict_session_probabilities",
        forbidden_inference,
    )
    monkeypatch.setattr(backend, "MultipartyVAP", forbidden_inference)
    monkeypatch.setattr(outer, "load_gaze_checkpoint_artifact", forbidden_inference)
    aggregate_path = outer.aggregate_gaze_outer_matrix(
        matrix.directory,
        small_source.artifact.output_dir,
        tmp_path / "aggregate.json",
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["aggregation_method"] == (
        "pooled_out_of_group_saved_probabilities_no_inference"
    )
    assert set(aggregate["metrics"]) == {"C0", "CM", "CG"}
    assert aggregate["comparisons"]["primary_CG_minus_CM"]["frame_mAP"] is not None
    assert aggregate["production_eligible"] is False
    assert aggregate["counted_eligible"] is False
    with pytest.raises(FileExistsError):
        outer.aggregate_gaze_outer_matrix(
            matrix.directory,
            small_source.artifact.output_dir,
            aggregate_path,
        )

    prediction = matrix.directory / "C0" / "outer_01" / backend.PREDICTIONS_FILENAME
    with prediction.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(
        outer.GazeOuterExperimentError,
        match="size|SHA-256|hash-bound",
    ):
        outer.aggregate_gaze_outer_matrix(
            matrix.directory,
            small_source.artifact.output_dir,
            tmp_path / "tampered.json",
        )


def test_fixture_harness_cannot_enter_the_formal_api(
    tmp_path: Path,
    small_source: SimpleNamespace,
) -> None:
    protocol = outer.freeze_gaze_outer_protocol(
        small_source.artifact.output_dir,
        tmp_path / "fixture_protocol",
    )
    fixture_backend = _FixtureBackend()
    fixture_matrix = outer._run_gaze_outer_fixture_matrix(
        protocol.directory,
        small_source.artifact.output_dir,
        tmp_path / "fixture_matrix",
        fixture_runner=fixture_backend,
    )
    assert len(fixture_backend.requests) == 12
    assert fixture_matrix.index["artifact_kind"] == outer.FIXTURE_MATRIX_KIND
    assert fixture_matrix.index["fixture_only"] is True
    with pytest.raises(outer.GazeOuterExperimentError, match="kind/version/protocol"):
        outer.audit_gaze_outer_matrix(
            fixture_matrix.directory,
            small_source.artifact.output_dir,
        )
    first_request = fixture_backend.requests[0]
    with pytest.raises(backend.GazePosthocBackendError, match="artifact_kind"):
        backend.load_gaze_checkpoint_artifact(
            fixture_matrix.directory / "C0" / "outer_01" / backend.CHECKPOINT_FILENAME,
            condition_id="C0",
            fold_id="outer_01",
            outer_test_group=first_request.outer_test_group,
            source_matrix_manifest_sha256=first_request.source_matrix_manifest_sha256,
            protocol_sha256=first_request.protocol_sha256,
            expected_initial_state_sha256=first_request.expected_initial_state_sha256,
        )
    with pytest.raises(TypeError, match="backend"):
        outer.run_gaze_outer_matrix(
            protocol.directory,
            small_source.artifact.output_dir,
            tmp_path / "forbidden_injection",
            backend=_FixtureBackend(),
        )
    with pytest.raises(outer.GazeOuterExperimentError, match="exact frozen"):
        outer._run_gaze_outer_matrix_with_runner(
            protocol.directory,
            small_source.artifact.output_dir,
            tmp_path / "forbidden_private_formal",
            runner=_FixtureBackend(),
            fixture_only=False,
        )


def test_outer_gold_is_separate_from_prediction_bytes_and_failure_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_source: SimpleNamespace,
    fast_formal_backend: None,
) -> None:
    sessions = backend.build_gaze_condition_sessions(
        small_source.artifact.frames,
        "CG",
        scene_to_group=outer._scene_to_group(small_source.artifact),
    )
    train_groups = tuple(outer.EXACT_EIGHT_GROUPS[1:])
    common = {
        "condition_id": "CG",
        "fold_id": "outer_01",
        "outer_test_group": outer.EXACT_EIGHT_GROUPS[0],
        "outer_train_groups": train_groups,
        "inner_validation_groups": train_groups,
        "source_matrix_manifest_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "expected_initial_state_sha256": backend.initial_model_state_sha256(
            backend.GazeBackendSpec(), 13
        ),
    }
    request_a = backend.GazeUnitRequest(
        sessions=sessions,
        output_directory=tmp_path / "gold_a",
        **common,
    )
    replaced_sessions = tuple(
        replace(
            session,
            labels=1.0 - session.labels,
            vad=1.0 - session.vad,
        )
        if session.group_id == request_a.outer_test_group
        else session
        for session in sessions
    )
    request_b = backend.GazeUnitRequest(
        sessions=replaced_sessions,
        output_directory=tmp_path / "gold_b",
        **common,
    )
    outer_a = tuple(
        session for session in sessions if session.group_id == request_a.outer_test_group
    )
    outer_b = tuple(
        session
        for session in replaced_sessions
        if session.group_id == request_b.outer_test_group
    )
    predictions = []
    for session in outer_a:
        value = 0.5 + 0.1 * np.tanh(session.features[:, -3])
        predictions.append(
            np.broadcast_to(value[:, None, None], (len(value), 4, 4))
            .copy()
            .astype(np.float32)
        )
    predictions_tuple = tuple(predictions)
    final_history = _history(2, validation=False)
    metrics_a = backend.gaze_metrics_from_predictions(outer_a, predictions_tuple)
    metrics_b = backend.gaze_metrics_from_predictions(outer_b, predictions_tuple)
    artifact_a = backend.write_gaze_unit_bundle(
        request_a,
        checkpoint_payload=_checkpoint_payload(
            request_a,
            selected_final_epochs=2,
            final_history=final_history,
        ),
        outer_sessions=outer_a,
        predictions=predictions_tuple,
        metrics=metrics_a,
        inner_fits=_inner_fits_for(request_a, metrics_a),
        selected_final_epochs=2,
        final_history=final_history,
    )
    artifact_b = backend.write_gaze_unit_bundle(
        request_b,
        checkpoint_payload=_checkpoint_payload(
            request_b,
            selected_final_epochs=2,
            final_history=final_history,
        ),
        outer_sessions=outer_b,
        predictions=predictions_tuple,
        metrics=metrics_b,
        inner_fits=_inner_fits_for(request_b, metrics_b),
        selected_final_epochs=2,
        final_history=final_history,
    )
    assert sha256(artifact_a.predictions_path.read_bytes()).hexdigest() == sha256(
        artifact_b.predictions_path.read_bytes()
    ).hexdigest()
    sidecar_a = json.loads(artifact_a.predictions_sidecar_path.read_text())
    sidecar_b = json.loads(artifact_b.predictions_sidecar_path.read_text())
    assert sidecar_a["sessions"][0]["prediction_input_sha256"] == (
        sidecar_b["sessions"][0]["prediction_input_sha256"]
    )
    assert sidecar_a["sessions"][0]["evaluation_target_sha256"] != (
        sidecar_b["sessions"][0]["evaluation_target_sha256"]
    )

    protocol = outer.freeze_gaze_outer_protocol(
        small_source.artifact.output_dir,
        tmp_path / "rollback_protocol",
    )

    class FailingBackend(_FixtureBackend):
        def run_unit(
            self,
            request: backend.GazeUnitRequest,
        ) -> backend.GazeUnitReceipt:
            if len(self.requests) == 3:
                raise RuntimeError("fixture backend failure")
            return super().run_unit(request)

    failed_output = tmp_path / "failed_matrix"
    with pytest.raises(RuntimeError, match="fixture backend failure"):
        outer._run_gaze_outer_fixture_matrix(
            protocol.directory,
            small_source.artifact.output_dir,
            failed_output,
            fixture_runner=FailingBackend(),
        )
    assert not failed_output.exists()
    assert not list(tmp_path.glob(".failed_matrix-stage-*"))

    def fail_prepublication_audit(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("staged matrix audit failure")

    monkeypatch.setattr(outer, "_audit_matrix", fail_prepublication_audit)
    failed_audit_output = tmp_path / "failed_audit_matrix"
    with pytest.raises(RuntimeError, match="staged matrix audit failure"):
        outer.run_gaze_outer_matrix(
            protocol.directory,
            small_source.artifact.output_dir,
            failed_audit_output,
        )
    assert not failed_audit_output.exists()
    assert not list(tmp_path.glob(".failed_audit_matrix-stage-*"))
