from __future__ import annotations

import json
import shutil
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from hri_turn_taking.data.assembly import ReadinessReport
from hri_turn_taking.data.multiparty import (
    GLOBAL_FEATURE_COLUMNS,
    PARTICIPANT_FEATURE_NAMES,
    participant_column,
    validate_multiparty_dataframe,
)
from hri_turn_taking.vap import oracle_prev_vad_backend as backend
from hri_turn_taking.vap import oracle_prev_vad_experiment as outer


def _small_frame(scene: str, group: str, start: int, end: int) -> pd.DataFrame:
    local = np.arange(end - start)
    ticks = np.arange(start, end, dtype=np.int64)
    data: dict[str, object] = {
        "session_id": np.repeat(scene, len(ticks)),
        "group_id": np.repeat(group, len(ticks)),
        "tick_25hz": ticks,
        "timestamp_s": ticks.astype(float) / 25.0,
    }
    for slot in range(1, 5):
        present = not (scene == next(iter(backend.EXACT_EIGHT_GROUP_BY_SCENE)) and slot == 4)
        data[participant_column(slot, "present")] = np.full(
            len(ticks), float(present)
        )
        vad = ((local + slot * 5) % (17 + slot * 2) < 7 + slot).astype(float)
        data[participant_column(slot, "vad")] = vad if present else np.zeros(len(ticks))
        for feature in PARTICIPANT_FEATURE_NAMES:
            data[participant_column(slot, feature)] = np.zeros(len(ticks))
        if present:
            data[participant_column(slot, "head_yaw_deg")] = 25.0 * np.sin(
                (local + slot) / 7.0
            )
            data[participant_column(slot, "pose_valid")] = np.ones(len(ticks))
            data[participant_column(slot, "pose_quality")] = 0.8 + 0.2 * np.cos(
                (local + slot) / 11.0
            )
    voiced = (local % 5 != 0).astype(float)
    data[GLOBAL_FEATURE_COLUMNS[0]] = np.zeros(len(ticks))
    data[GLOBAL_FEATURE_COLUMNS[1]] = np.cos(local / 9.0)
    data[GLOBAL_FEATURE_COLUMNS[2]] = voiced * (100.0 + local % 20)
    data[GLOBAL_FEATURE_COLUMNS[3]] = voiced
    frame = pd.DataFrame(data)
    validate_multiparty_dataframe(frame)
    return frame


@pytest.fixture
def small_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    scenes = tuple(backend.EXACT_EIGHT_GROUP_BY_SCENE)
    ranges = {
        scene: (1000 * index, 1000 * index + 80)
        for index, scene in enumerate(scenes, start=1)
    }
    total_rows = sum(end - start for start, end in ranges.values())
    monkeypatch.setattr(backend, "EXACT_EIGHT_TICK_RANGES", ranges)
    monkeypatch.setattr(backend, "EXACT_EIGHT_TOTAL_ROWS", total_rows)
    monkeypatch.setattr(outer, "EXACT_EIGHT_TICK_RANGES", ranges)
    monkeypatch.setattr(outer, "EXACT_EIGHT_TOTAL_ROWS", total_rows)
    frames = {
        scene: _small_frame(
            scene,
            backend.EXACT_EIGHT_GROUP_BY_SCENE[scene],
            *ranges[scene],
        )
        for scene in scenes
    }
    frame = pd.concat(frames.values(), ignore_index=True)
    source_frame = frame.drop(columns=["tick_25hz"])
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    readiness = ReadinessReport(
        source=str(source_dir.resolve()),
        data_kind="real",
        files=(),
        n_rows=total_rows,
        n_sessions=8,
        n_groups=4,
        session_ids=scenes,
        group_ids=backend.EXACT_EIGHT_GROUPS,
        target_ready=True,
        feature_set="head_pose_only",
        selected_feature_ready=True,
    )
    audit_calls: list[tuple[Path, str, str, tuple[str, ...]]] = []

    def load_source(
        path: str | Path,
        *,
        data_kind: str,
        feature_set: str,
        excluded_session_ids: tuple[str, ...],
    ) -> tuple[pd.DataFrame, ReadinessReport]:
        audit_calls.append(
            (Path(path).resolve(), data_kind, feature_set, excluded_session_ids)
        )
        return source_frame, readiness

    monkeypatch.setattr(outer, "load_and_audit_vap_source", load_source)
    monkeypatch.setattr(outer, "validate_outer_split_manifest", lambda _value: None)
    monkeypatch.setattr(
        outer,
        "validate_outer_run_manifest",
        lambda _run, _split: None,
    )
    canonical_dir = tmp_path / "canonical_15epoch"
    canonical_dir.mkdir()
    conditions = [
        {"condition_id": "A", "feature_set": "audio_only", "encoder": "transformer"},
        {
            "condition_id": "H",
            "feature_set": "head_pose_only",
            "encoder": "transformer",
        },
        {
            "condition_id": "A+H",
            "feature_set": "audio_plus_head_pose",
            "encoder": "transformer",
        },
    ]
    split = {
        "protocol_id": outer.DEADLINE_PROTOCOL_ID,
        "conditions": conditions,
        "source_session_to_group": dict(backend.EXACT_EIGHT_GROUP_BY_SCENE),
        "corpus_exclusions": dict(outer.EXCLUDED_OWNER_HOLD_SCENES),
    }
    spec = outer.ThesisBackendSpec()
    run = {
        "protocol_id": outer.DEADLINE_PROTOCOL_ID,
        "source": str(source_dir.resolve()),
        "seeds": [13, 42, 73],
        "condition_ids": ["A", "H", "A+H"],
        "max_epochs": 15,
        "aux_vad_weight": 0.0,
        "trainer_config": outer.thesis_trainer_config(spec),
        "model_config": outer.thesis_model_config(spec),
        "corpus_exclusions": dict(outer.EXCLUDED_OWNER_HOLD_SCENES),
        "readiness_by_condition": {
            "H": {
                "data_kind": "real",
                "feature_set": "head_pose_only",
                "n_rows": total_rows,
                "selected_feature_ready": True,
                "session_ids": list(scenes),
                "group_ids": list(backend.EXACT_EIGHT_GROUPS),
            }
        },
    }
    (canonical_dir / outer.CANONICAL_SPLIT_FILENAME).write_text(
        json.dumps(split, indent=2) + "\n",
        encoding="utf-8",
    )
    (canonical_dir / outer.CANONICAL_RUN_FILENAME).write_text(
        json.dumps(run, indent=2) + "\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        source_dir=source_dir,
        canonical_dir=canonical_dir,
        frames=frames,
        frame=frame,
        source_frame=source_frame,
        readiness=readiness,
        load_source=load_source,
        audit_calls=audit_calls,
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
) -> tuple[np.ndarray, ...]:
    predictions = []
    for session in sessions:
        block = session.features[:, :-backend.GLOBAL_WIDTH].reshape(
            len(session.features), 4, backend.PARTICIPANT_WIDTH
        )
        oracle_signal = block[:, :, backend.PREVIOUS_VAD_INDEX].mean(axis=1)
        head_signal = block[:, :, 3].mean(axis=1)
        time_score = 0.45 + 0.08 * np.tanh(head_signal / 20.0) + 0.08 * oracle_signal
        probability = np.empty((len(time_score), 4, 4), dtype=np.float32)
        for slot in range(4):
            for bin_index in range(4):
                probability[:, slot, bin_index] = np.clip(
                    time_score + 0.01 * slot - 0.005 * bin_index,
                    0.0,
                    1.0,
                )
        predictions.append(probability)
    return tuple(predictions)


def _checkpoint_payload(
    request: backend.OracleUnitRequest,
    *,
    selected_final_epochs: int,
    final_history: list[dict[str, object]],
) -> dict[str, object]:
    spec = backend.OracleBackendSpec()
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
        "model_config": backend.oracle_model_config(spec),
        "trainer_spec": backend.oracle_trainer_spec(spec),
        "label_config": backend.oracle_label_config(),
        "model_state": model.state_dict(),
        "model_state_sha256": backend.model_state_sha256(model.state_dict()),
        "normalizer": backend.fit_oracle_training_normalizer(
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
    request: backend.OracleUnitRequest,
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
        self.requests: list[backend.OracleUnitRequest] = []

    def run_unit(
        self,
        request: backend.OracleUnitRequest,
    ) -> backend.OracleUnitReceipt:
        self.requests.append(request)
        assert request.fixture_only is True
        outer_sessions = tuple(
            session
            for session in request.sessions
            if session.group_id == request.outer_test_group
        )
        predictions = _predictions_for(outer_sessions)
        metrics = backend.oracle_metrics_from_predictions(outer_sessions, predictions)
        final_history = _history(2, validation=False)
        return backend.write_oracle_unit_bundle(
            request,
            checkpoint_payload=_checkpoint_payload(
                request,
                selected_final_epochs=2,
                final_history=final_history,
            ),
            outer_sessions=outer_sessions,
            predictions=predictions,
            metrics=metrics,
            inner_fits=_inner_fits_for(request, metrics),
            selected_final_epochs=2,
            final_history=final_history,
        )


@pytest.fixture
def fast_formal_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fit(
        spec: backend.OracleBackendSpec,
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
        _spec: backend.OracleBackendSpec,
    ) -> tuple[np.ndarray, ...]:
        return _predictions_for(sessions)

    monkeypatch.setattr(backend, "_fit", fake_fit)
    monkeypatch.setattr(backend, "_predict", fake_predict)


def _freeze(small_source: SimpleNamespace, destination: Path) -> outer.OracleOuterProtocolArtifact:
    return outer.freeze_oracle_outer_protocol(
        small_source.source_dir,
        small_source.canonical_dir,
        destination,
    )


def test_protocol_binds_exact8_canonical_15epoch_and_is_create_once(
    tmp_path: Path,
    small_source: SimpleNamespace,
) -> None:
    protocol = _freeze(small_source, tmp_path / "protocol")
    payload = protocol.protocol
    assert [item["condition_id"] for item in payload["conditions"]] == ["O0", "O1"]
    assert payload["training"] == {
        "seed": 13,
        "max_epochs": 15,
        "aux_vad_weight": 0.0,
        "epoch_selection": "median_inner_logo_selected_epoch",
        "outer_test_access": "one_saved_probability_pass_per_condition_fold",
    }
    assert payload["schema_dimensions"] == {
        "feature_schema": backend.FEATURE_SCHEMA,
        "conceptual_raw_input_columns": 36,
        "transformed_input_dimensions": 37,
        "participant_features_per_token": 8,
        "global_features_per_token": 5,
        "token_input_dimensions": 13,
    }
    assert len(payload["outer_folds"]) == 4
    assert all(len(fold["inner_folds"]) == 3 for fold in payload["outer_folds"])
    initial_hash = payload["backend"]["initial_state_strategy"]["sha256"]
    assert initial_hash == backend.initial_model_state_sha256(
        backend.OracleBackendSpec(), 13
    )
    assert all(
        set(fold["condition_initial_state_sha256"].values()) == {initial_hash}
        for fold in payload["outer_folds"]
    )
    assert payload["oracle_gold_input"] is True
    assert payload["robot_observable"] is False
    assert payload["deployable"] is False
    with pytest.raises(FileExistsError):
        _freeze(small_source, protocol.directory)

    audited_source = outer._audit_source(small_source.source_dir)
    protocol_bytes = protocol.protocol_path.read_bytes()
    protocol.protocol_path.write_bytes(protocol_bytes.replace(b'"seed": 13', b'"seed": 42', 1))
    with pytest.raises(outer.OracleOuterExperimentError, match="SHA-256"):
        outer._load_protocol(protocol.directory, audited_source)
    protocol.protocol_path.write_bytes(protocol_bytes)

    canonical_run_path = small_source.canonical_dir / outer.CANONICAL_RUN_FILENAME
    canonical_bytes = canonical_run_path.read_bytes()
    canonical_run_path.write_bytes(canonical_bytes + b" \n")
    with pytest.raises(outer.OracleOuterExperimentError, match="frozen replay"):
        outer._load_protocol(protocol.directory, audited_source)
    canonical_run_path.write_bytes(canonical_bytes)

    run_path = small_source.canonical_dir / outer.CANONICAL_RUN_FILENAME
    original = run_path.read_bytes()
    run = json.loads(original)
    run["max_epochs"] = 5
    run_path.write_text(json.dumps(run) + "\n", encoding="utf-8")
    with pytest.raises(outer.OracleOuterExperimentError, match="15-epoch"):
        outer._audit_canonical_protocol(
            small_source.canonical_dir,
            outer._audit_source(small_source.source_dir),
        )
    run_path.write_bytes(original)


def test_real_source_without_tick_is_exactly_derived_and_bad_time_fails(
    monkeypatch: pytest.MonkeyPatch,
    small_source: SimpleNamespace,
) -> None:
    assert "tick_25hz" not in small_source.source_frame
    audited = outer._audit_source(small_source.source_dir)
    for scene, (start, end) in small_source.ranges.items():
        ticks = audited.frames[scene]["tick_25hz"].to_numpy()
        assert ticks.dtype == np.int64
        assert np.array_equal(ticks, np.arange(start, end, dtype=np.int64))

    bad = small_source.source_frame.copy(deep=True)
    bad.loc[bad.index[0], "timestamp_s"] += 1e-9
    monkeypatch.setattr(
        outer,
        "load_and_audit_vap_source",
        lambda *_args, **_kwargs: (bad, small_source.readiness),
    )
    with pytest.raises(
        outer.OracleOuterExperimentError,
        match="timestamp grid differs",
    ):
        outer._audit_source(small_source.source_dir)


def test_o0_o1_are_schema_matched_exact_lag_and_training_only_normalized(
    small_source: SimpleNamespace,
) -> None:
    mapping = dict(backend.EXACT_EIGHT_GROUP_BY_SCENE)
    o0 = backend.build_oracle_condition_sessions(
        small_source.frames,
        "O0",
        scene_to_group=mapping,
    )
    o1 = backend.build_oracle_condition_sessions(
        small_source.frames,
        "O1",
        scene_to_group=mapping,
    )
    for zero, lagged in zip(o0, o1, strict=True):
        zero_block = zero.features[:, :-backend.GLOBAL_WIDTH].reshape(
            len(zero.features), 4, backend.PARTICIPANT_WIDTH
        )
        lagged_block = lagged.features[:, :-backend.GLOBAL_WIDTH].reshape(
            len(lagged.features), 4, backend.PARTICIPANT_WIDTH
        )
        assert zero.features.shape[1] == lagged.features.shape[1] == 37
        assert np.array_equal(zero_block[:, :, :-1], lagged_block[:, :, :-1])
        assert np.array_equal(
            zero.features[:, -backend.GLOBAL_WIDTH :],
            lagged.features[:, -backend.GLOBAL_WIDTH :],
        )
        assert np.count_nonzero(zero_block[:, :, -1]) == 0
        assert np.count_nonzero(lagged_block[0, :, -1]) == 0
        assert np.array_equal(lagged_block[1:, :, -1], lagged.vad[:-1])
        assert np.count_nonzero(lagged_block[:, lagged.present <= 0, -1]) == 0

    training_groups = tuple(backend.EXACT_EIGHT_GROUPS[:-1])
    normalizer_o0 = backend.fit_oracle_training_normalizer(o0, training_groups)
    normalizer_o1 = backend.fit_oracle_training_normalizer(o1, training_groups)
    assert normalizer_o0.to_dict() == normalizer_o1.to_dict()
    assert normalizer_o1.participant_mean[backend.PREVIOUS_VAD_INDEX] == 0.0
    assert normalizer_o1.participant_std[backend.PREVIOUS_VAD_INDEX] == 1.0
    transformed_o0 = normalizer_o0.apply(o0[0].features)
    transformed_block = transformed_o0[:, :-backend.GLOBAL_WIDTH].reshape(
        len(transformed_o0), 4, backend.PARTICIPANT_WIDTH
    )
    assert np.count_nonzero(transformed_block[:, :, -1]) == 0

    held_out_changed = []
    for session in o1:
        if session.group_id != backend.EXACT_EIGHT_GROUPS[-1]:
            held_out_changed.append(session)
            continue
        features = session.features.copy()
        features[:, 3] = 999.0
        held_out_changed.append(replace(session, features=features))
    repeated = backend.fit_oracle_training_normalizer(
        held_out_changed,
        training_groups,
    )
    assert repeated.to_dict() == normalizer_o1.to_dict()


def test_current_and_future_gold_cannot_change_current_or_prefix_predictions(
    small_source: SimpleNamespace,
) -> None:
    mapping = dict(backend.EXACT_EIGHT_GROUP_BY_SCENE)
    original = backend.build_oracle_condition_sessions(
        small_source.frames,
        "O1",
        scene_to_group=mapping,
    )
    first_scene = next(iter(small_source.frames))
    changed_frames = dict(small_source.frames)
    changed = changed_frames[first_scene].copy(deep=True)
    cut = 20
    vad_column = participant_column(2, "vad")
    changed.loc[cut:, vad_column] = 1.0 - changed.loc[cut:, vad_column]
    changed_frames[first_scene] = changed
    perturbed = backend.build_oracle_condition_sessions(
        changed_frames,
        "O1",
        scene_to_group=mapping,
    )
    left = original[0]
    right = perturbed[0]
    assert np.array_equal(left.features[: cut + 1], right.features[: cut + 1])
    assert not np.array_equal(left.features[cut + 1 :], right.features[cut + 1 :])

    normalizer = backend.OracleFeatureNormalizer.fit((left,))
    torch.manual_seed(13)
    model = backend._model(backend.OracleBackendSpec()).eval()
    left_probability = backend._predict(
        model,
        (left,),
        normalizer,
        backend.OracleBackendSpec(),
    )[0]
    right_probability = backend._predict(
        model,
        (right,),
        normalizer,
        backend.OracleBackendSpec(),
    )[0]
    assert np.array_equal(left_probability[: cut + 1], right_probability[: cut + 1])
    assert not np.array_equal(left_probability[cut + 1 :], right_probability[cut + 1 :])

    injected = left.features.copy()
    block = injected[:, :-backend.GLOBAL_WIDTH].reshape(
        len(injected), 4, backend.PARTICIPANT_WIDTH
    )
    block[cut, 1, backend.PREVIOUS_VAD_INDEX] = left.vad[cut, 1]
    invalid = replace(left, features=injected)
    with pytest.raises(backend.OraclePosthocBackendError, match="lag-1"):
        backend.validate_oracle_condition_sessions((invalid,), "O1")


def test_gold_hashes_are_separate_and_o1_rejects_stale_lagged_features(
    small_source: SimpleNamespace,
) -> None:
    mapping = dict(backend.EXACT_EIGHT_GROUP_BY_SCENE)
    o0 = backend.build_oracle_condition_sessions(
        small_source.frames,
        "O0",
        scene_to_group=mapping,
    )[0]
    changed_o0 = replace(o0, labels=1.0 - o0.labels, vad=1.0 - o0.vad)
    assert backend.prediction_input_sha256(o0) == backend.prediction_input_sha256(
        changed_o0
    )
    assert backend.oracle_source_vad_sha256(o0) != backend.oracle_source_vad_sha256(
        changed_o0
    )
    assert backend.evaluation_target_sha256(o0) != backend.evaluation_target_sha256(
        changed_o0
    )
    backend.validate_oracle_condition_sessions((changed_o0,), "O0")

    o1 = backend.build_oracle_condition_sessions(
        small_source.frames,
        "O1",
        scene_to_group=mapping,
    )[0]
    stale = replace(o1, labels=1.0 - o1.labels, vad=1.0 - o1.vad)
    with pytest.raises(backend.OraclePosthocBackendError, match="lag-1"):
        backend.validate_oracle_condition_sessions((stale,), "O1")


def test_formal_matrix_has_eight_units_and_aggregate_is_saved_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_source: SimpleNamespace,
    fast_formal_backend: None,
) -> None:
    protocol = _freeze(small_source, tmp_path / "protocol")
    matrix = outer.run_oracle_outer_matrix(
        protocol.directory,
        small_source.source_dir,
        tmp_path / "matrix",
    )
    assert matrix.index["completed_units"] == 8
    assert matrix.index["artifact_kind"] == outer.FORMAL_MATRIX_KIND
    assert matrix.index["fixture_only"] is False
    assert len(matrix.units) == 8
    for fold_id in outer.OUTER_FOLD_IDS:
        units = [matrix.units[(condition, fold_id)] for condition in outer.CONDITIONS]
        assert len({unit.result["initial_state_sha256"] for unit in units}) == 1
        checkpoints = [
            torch.load(
                matrix.directory / condition / fold_id / backend.CHECKPOINT_FILENAME,
                map_location="cpu",
                weights_only=True,
            )
            for condition in outer.CONDITIONS
        ]
        assert checkpoints[0]["payload"]["normalizer"] == checkpoints[1]["payload"][
            "normalizer"
        ]

    first = matrix.units[("O0", "outer_01")].result
    checkpoint_path = (
        matrix.directory / "O0" / "outer_01" / backend.CHECKPOINT_FILENAME
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["payload"]["normalizer"]["participant_mean"][-1] = 0.5
    tampered_checkpoint = tmp_path / "tampered_normalizer.pt"
    torch.save(checkpoint, tampered_checkpoint)
    with pytest.raises(backend.OraclePosthocBackendError, match="identity"):
        backend.load_oracle_checkpoint_artifact(
            tampered_checkpoint,
            condition_id="O0",
            fold_id="outer_01",
            outer_test_group=backend.EXACT_EIGHT_GROUPS[0],
            source_snapshot_sha256=first["source_snapshot_sha256"],
            protocol_sha256=first["protocol_sha256"],
            expected_initial_state_sha256=first["initial_state_sha256"],
        )

    semantic_matrix = tmp_path / "semantic_normalizer_tamper"
    shutil.copytree(matrix.directory, semantic_matrix)
    semantic_checkpoint_path = (
        semantic_matrix / "O0" / "outer_01" / backend.CHECKPOINT_FILENAME
    )
    semantic_checkpoint = torch.load(
        semantic_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    semantic_checkpoint["payload"]["normalizer"]["participant_mean"][3] += 0.25
    torch.save(semantic_checkpoint, semantic_checkpoint_path)
    semantic_result_path = semantic_matrix / "O0" / "outer_01" / backend.RESULT_FILENAME
    semantic_result = json.loads(semantic_result_path.read_text(encoding="utf-8"))
    semantic_result["artifacts"]["checkpoint"] = outer._file_binding(
        semantic_checkpoint_path,
        relative_to=semantic_checkpoint_path.parent,
    )
    semantic_result_path.write_bytes(outer._json_bytes(semantic_result))
    audited_source = outer._audit_source(small_source.source_dir)
    semantic_index = outer._matrix_index(
        semantic_matrix,
        protocol,
        audited_source,
        fixture_only=False,
    )
    (semantic_matrix / outer.MATRIX_INDEX_FILENAME).write_bytes(
        outer._json_bytes(semantic_index)
    )
    with pytest.raises(
        outer.OracleOuterExperimentError,
        match="normalizer differs from outer-training replay",
    ):
        outer.audit_oracle_outer_matrix(
            semantic_matrix,
            small_source.source_dir,
        )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("aggregate must not construct or run a model")

    monkeypatch.setattr(backend, "predict_session_probabilities", forbidden)
    monkeypatch.setattr(backend, "MultipartyVAP", forbidden)
    monkeypatch.setattr(outer, "load_oracle_checkpoint_artifact", forbidden)
    aggregate_path = outer.aggregate_oracle_outer_matrix(
        matrix.directory,
        small_source.source_dir,
        tmp_path / "aggregate.json",
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["aggregation_method"] == (
        "pooled_out_of_group_saved_probabilities_no_inference"
    )
    assert set(aggregate["metrics"]) == {"O0", "O1"}
    assert set(aggregate["comparisons"]) == {"primary_O1_minus_O0"}
    assert aggregate["oracle_gold_input"] is True
    assert aggregate["deployable"] is False
    audited_aggregate = outer.audit_oracle_outer_aggregate(
        aggregate_path,
        matrix.directory,
        small_source.source_dir,
    )
    assert audited_aggregate.path == aggregate_path
    assert audited_aggregate.sha256 == sha256(aggregate_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        outer.aggregate_oracle_outer_matrix(
            matrix.directory,
            small_source.source_dir,
            aggregate_path,
        )

    aggregate_bytes = aggregate_path.read_bytes()
    invented = json.loads(aggregate_bytes)
    invented["metrics"]["O0"]["threshold_policy"] = "invented"
    aggregate_path.write_text(json.dumps(invented) + "\n", encoding="utf-8")
    with pytest.raises(
        outer.OracleOuterExperimentError,
        match="saved-probability replay",
    ):
        outer.audit_oracle_outer_aggregate(
            aggregate_path,
            matrix.directory,
            small_source.source_dir,
        )
    aggregate_path.write_bytes(aggregate_bytes)

    original_aggregate_audit = outer._audit_aggregate_file
    audit_calls = 0

    def fail_final_aggregate_audit(*args: object, **kwargs: object) -> object:
        nonlocal audit_calls
        audit_calls += 1
        if audit_calls == 2:
            raise RuntimeError("final aggregate audit failure")
        return original_aggregate_audit(*args, **kwargs)

    monkeypatch.setattr(outer, "_audit_aggregate_file", fail_final_aggregate_audit)
    failed_aggregate = tmp_path / "failed_aggregate.json"
    with pytest.raises(RuntimeError, match="final aggregate audit failure"):
        outer.aggregate_oracle_outer_matrix(
            matrix.directory,
            small_source.source_dir,
            failed_aggregate,
        )
    assert not failed_aggregate.exists()
    assert not list(tmp_path.glob(".failed_aggregate.json-stage-*"))
    monkeypatch.setattr(outer, "_audit_aggregate_file", original_aggregate_audit)

    original_write = outer._write_exclusive

    def partial_write(path: Path, payload: bytes) -> None:
        path.write_bytes(payload[:10])
        raise RuntimeError("partial aggregate write")

    monkeypatch.setattr(outer, "_write_exclusive", partial_write)
    partial_aggregate = tmp_path / "partial_aggregate.json"
    with pytest.raises(RuntimeError, match="partial aggregate write"):
        outer.aggregate_oracle_outer_matrix(
            matrix.directory,
            small_source.source_dir,
            partial_aggregate,
        )
    assert not partial_aggregate.exists()
    assert not list(tmp_path.glob(".partial_aggregate.json-stage-*"))
    monkeypatch.setattr(outer, "_write_exclusive", original_write)

    cleanup_audit_calls = 0

    def fail_final_before_cleanup(*args: object, **kwargs: object) -> object:
        nonlocal cleanup_audit_calls
        cleanup_audit_calls += 1
        if cleanup_audit_calls == 2:
            raise RuntimeError("final audit before cleanup failure")
        return original_aggregate_audit(*args, **kwargs)

    real_rmtree = outer.shutil.rmtree

    def fail_stage_cleanup(path: str | Path, *args: object, **kwargs: object) -> None:
        if Path(path).name.startswith(".cleanup_failure.json-stage-"):
            raise OSError("forced stage cleanup failure")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(outer, "_audit_aggregate_file", fail_final_before_cleanup)
    monkeypatch.setattr(outer.shutil, "rmtree", fail_stage_cleanup)
    cleanup_failure = tmp_path / "cleanup_failure.json"
    with pytest.raises(
        outer.OracleOuterExperimentError,
        match="destination was rolled back but stage cleanup failed",
    ):
        outer.aggregate_oracle_outer_matrix(
            matrix.directory,
            small_source.source_dir,
            cleanup_failure,
        )
    assert not cleanup_failure.exists()
    leftover_stages = list(tmp_path.glob(".cleanup_failure.json-stage-*"))
    assert len(leftover_stages) == 1
    monkeypatch.setattr(outer.shutil, "rmtree", real_rmtree)
    real_rmtree(leftover_stages[0])
    monkeypatch.setattr(outer, "_audit_aggregate_file", original_aggregate_audit)

    prediction_path = (
        matrix.directory / "O0" / "outer_01" / backend.PREDICTIONS_FILENAME
    )
    with prediction_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(
        outer.OracleOuterExperimentError,
        match="size|SHA-256|hash-bound",
    ):
        outer.aggregate_oracle_outer_matrix(
            matrix.directory,
            small_source.source_dir,
            tmp_path / "tampered.json",
        )


def test_fixture_cannot_enter_formal_api_and_failures_roll_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_source: SimpleNamespace,
) -> None:
    protocol = _freeze(small_source, tmp_path / "protocol")
    fixture_backend = _FixtureBackend()
    fixture_matrix = outer._run_oracle_outer_fixture_matrix(
        protocol.directory,
        small_source.source_dir,
        tmp_path / "fixture_matrix",
        fixture_runner=fixture_backend,
    )
    assert len(fixture_backend.requests) == 8
    assert fixture_matrix.index["artifact_kind"] == outer.FIXTURE_MATRIX_KIND
    assert fixture_matrix.index["fixture_only"] is True
    with pytest.raises(outer.OracleOuterExperimentError, match="kind/version/protocol"):
        outer.audit_oracle_outer_matrix(
            fixture_matrix.directory,
            small_source.source_dir,
        )
    first_request = fixture_backend.requests[0]
    with pytest.raises(backend.OraclePosthocBackendError, match="artifact_kind"):
        backend.load_oracle_checkpoint_artifact(
            fixture_matrix.directory
            / "O0"
            / "outer_01"
            / backend.CHECKPOINT_FILENAME,
            condition_id="O0",
            fold_id="outer_01",
            outer_test_group=first_request.outer_test_group,
            source_snapshot_sha256=first_request.source_snapshot_sha256,
            protocol_sha256=first_request.protocol_sha256,
            expected_initial_state_sha256=first_request.expected_initial_state_sha256,
        )
    with pytest.raises(TypeError, match="backend"):
        outer.run_oracle_outer_matrix(
            protocol.directory,
            small_source.source_dir,
            tmp_path / "forbidden_injection",
            backend=_FixtureBackend(),
        )
    with pytest.raises(outer.OracleOuterExperimentError, match="exact frozen"):
        outer._run_oracle_outer_matrix_with_runner(
            protocol.directory,
            small_source.source_dir,
            tmp_path / "forbidden_private_formal",
            runner=_FixtureBackend(),
            fixture_only=False,
        )

    class FailingBackend(_FixtureBackend):
        def run_unit(
            self,
            request: backend.OracleUnitRequest,
        ) -> backend.OracleUnitReceipt:
            if len(self.requests) == 2:
                raise RuntimeError("fixture backend failure")
            return super().run_unit(request)

    failed_output = tmp_path / "failed_matrix"
    with pytest.raises(RuntimeError, match="fixture backend failure"):
        outer._run_oracle_outer_fixture_matrix(
            protocol.directory,
            small_source.source_dir,
            failed_output,
            fixture_runner=FailingBackend(),
        )
    assert not failed_output.exists()
    assert not list(tmp_path.glob(".failed_matrix-stage-*"))

    original_audit = outer._audit_matrix
    calls = 0

    def fail_final_audit(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("final matrix audit failure")
        return original_audit(*args, **kwargs)

    monkeypatch.setattr(outer, "_audit_matrix", fail_final_audit)
    failed_final = tmp_path / "failed_final_matrix"
    with pytest.raises(RuntimeError, match="final matrix audit failure"):
        outer._run_oracle_outer_fixture_matrix(
            protocol.directory,
            small_source.source_dir,
            failed_final,
            fixture_runner=_FixtureBackend(),
        )
    assert not failed_final.exists()
    assert not list(tmp_path.glob(".failed_final_matrix-stage-*"))


def test_protocol_final_audit_failure_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_source: SimpleNamespace,
) -> None:
    original_load = outer._load_protocol
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("final protocol audit failure")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(outer, "_load_protocol", fail_second)
    destination = tmp_path / "failed_protocol"
    with pytest.raises(RuntimeError, match="final protocol audit failure"):
        _freeze(small_source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed_protocol-stage-*"))


def test_saved_probability_bytes_are_deterministic_npz(
    tmp_path: Path,
    small_source: SimpleNamespace,
) -> None:
    sessions = backend.build_oracle_condition_sessions(
        small_source.frames,
        "O0",
        scene_to_group=dict(backend.EXACT_EIGHT_GROUP_BY_SCENE),
    )
    train_groups = tuple(backend.EXACT_EIGHT_GROUPS[1:])
    common = {
        "condition_id": "O0",
        "fold_id": "outer_01",
        "outer_test_group": backend.EXACT_EIGHT_GROUPS[0],
        "outer_train_groups": train_groups,
        "inner_validation_groups": train_groups,
        "source_snapshot_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "expected_initial_state_sha256": backend.initial_model_state_sha256(
            backend.OracleBackendSpec(), 13
        ),
        "fixture_only": True,
    }
    request_a = backend.OracleUnitRequest(
        sessions=sessions,
        output_directory=tmp_path / "unit_a",
        **common,
    )
    request_b = replace(request_a, output_directory=tmp_path / "unit_b")
    fixture = _FixtureBackend()
    artifact_a = fixture.run_unit(request_a)
    artifact_b = fixture.run_unit(request_b)
    assert sha256(artifact_a.predictions_path.read_bytes()).hexdigest() == sha256(
        artifact_b.predictions_path.read_bytes()
    ).hexdigest()
    with np.load(artifact_a.predictions_path, allow_pickle=False) as saved:
        assert saved.files == ["probabilities", "session_offsets"]
        assert saved["probabilities"].dtype == np.float32
        assert saved["session_offsets"].dtype == np.int64
        assert all(not saved[name].dtype.hasobject for name in saved.files)
