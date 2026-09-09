from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hri_turn_taking.vap import h2_matched_replay as h2


def _prediction_payload(n_frames: int = 20) -> dict[str, object]:
    probabilities = np.linspace(0.05, 0.95, n_frames * 16).reshape(n_frames, 4, 4)
    return {
        "condition_id": "A+H",
        "outer_fold_id": "outer_01",
        "seed": 13,
        "frame_rate_hz": 25,
        "checkpoint_sha256": "a" * 64,
        "run_manifest_sha256": "b" * 64,
        "sessions": [
            {
                "session_id": "scene__1",
                "group_id": "g1",
                "n_frames": n_frames,
                "tick_25hz": list(range(100, 100 + n_frames)),
                "probabilities": probabilities.tolist(),
                "labels": np.zeros((n_frames, 4, 4)).tolist(),
                "vad": np.zeros((n_frames, 4)).tolist(),
                "valid": np.ones(n_frames).tolist(),
                "present": [1.0, 1.0, 1.0, 1.0],
            }
        ],
    }


def _detector() -> dict[str, object]:
    return {
        "scaler_mean": [0.0, 0.0, 0.0],
        "scaler_scale": [1.0, 1.0, 1.0],
        "coefficient": [0.2, 1.0, -0.1],
        "intercept": -0.25,
    }


def _calibration_session(group: str, index: int, n_frames: int = 30) -> h2.CalibrationSession:
    ticks = np.arange(index * 100, index * 100 + n_frames, dtype=np.int64)
    target = np.tile(np.asarray([0.0, 1.0]), n_frames // 2)
    vad = np.zeros((n_frames, 4), dtype=np.float64)
    vad[:, 1] = target
    acoustic = np.column_stack(
        (
            target * 2.0 + np.linspace(-0.2, 0.2, n_frames),
            target,
            target * 140.0,
        )
    )
    probabilities = np.full((n_frames, 4, 4), 0.2, dtype=np.float64)
    probabilities[:, 0, :] = np.linspace(0.1, 0.9, n_frames)[:, None]
    return h2.CalibrationSession(
        session_id=f"{group}__s1",
        group_id=group,
        ticks=ticks,
        probabilities=probabilities,
        vad=vad,
        valid=np.ones(n_frames, dtype=bool),
        present=np.ones(4, dtype=np.float64),
        acoustic_features=acoustic,
    )


def test_outer_gold_replacement_does_not_change_gold_free_view_or_trace() -> None:
    original = _prediction_payload()
    replaced = copy.deepcopy(original)
    session = replaced["sessions"][0]
    session["labels"] = np.ones((20, 4, 4)).tolist()
    session["vad"] = np.ones((20, 4)).tolist()
    session["valid"] = np.zeros(20).tolist()

    left = h2.gold_free_probability_view(original, budget_id="fixture", max_epochs=5)
    right = h2.gold_free_probability_view(replaced, budget_id="fixture", max_epochs=5)
    assert left == right
    assert set(left["sessions"][0]).isdisjoint({"labels", "vad", "valid", "floor_holder"})

    probabilities = np.asarray(left["sessions"][0]["probabilities"])
    acoustic = np.column_stack(
        (np.linspace(-1.0, 1.0, 20), np.zeros(20), np.zeros(20))
    )
    trace_left, _ = h2.replay_matched_policy(
        probabilities,
        acoustic,
        [1, 1, 1, 1],
        detector_model=_detector(),
        speech_threshold=0.5,
        reactive_silent_frames=5,
        now_threshold=0.5,
        future_threshold=0.5,
    )
    trace_right, _ = h2.replay_matched_policy(
        np.asarray(right["sessions"][0]["probabilities"]),
        acoustic,
        [1, 1, 1, 1],
        detector_model=_detector(),
        speech_threshold=0.5,
        reactive_silent_frames=5,
        now_threshold=0.5,
        future_threshold=0.5,
    )
    assert trace_left.keys() == trace_right.keys()
    for key in trace_left:
        np.testing.assert_array_equal(trace_left[key], trace_right[key])


def test_replay_is_prefix_causal_and_chunk_equivalent_without_floor_holder() -> None:
    assert "floor_holder" not in inspect.signature(h2.replay_matched_policy).parameters
    rng = np.random.default_rng(4)
    probabilities = rng.uniform(0.0, 1.0, size=(41, 4, 4))
    acoustic = rng.normal(size=(41, 3))
    acoustic[:, 1] = (acoustic[:, 1] > 0).astype(float)
    kwargs = {
        "detector_model": _detector(),
        "speech_threshold": 0.45,
        "reactive_silent_frames": 5,
        "now_threshold": 0.4,
        "future_threshold": 0.6,
    }
    full, final_state = h2.replay_matched_policy(
        probabilities, acoustic, [1, 1, 1, 1], **kwargs
    )
    prefix, _ = h2.replay_matched_policy(
        probabilities[:17], acoustic[:17], [1, 1, 1, 1], **kwargs
    )
    for key in full:
        np.testing.assert_allclose(prefix[key], full[key][:17])

    first, state = h2.replay_matched_policy(
        probabilities[:17], acoustic[:17], [1, 1, 1, 1], **kwargs
    )
    second, chunk_state = h2.replay_matched_policy(
        probabilities[17:], acoustic[17:], [1, 1, 1, 1], state=state, **kwargs
    )
    for key in full:
        np.testing.assert_allclose(np.concatenate((first[key], second[key])), full[key])
    assert chunk_state == final_state


def test_cross_fitted_detector_never_fits_validation_or_outer_group() -> None:
    sessions = tuple(
        _calibration_session(group, index)
        for index, group in enumerate(("g1", "g2", "g3"), start=1)
    )
    contract, probabilities = h2.fit_cross_fitted_speech_detector(sessions)
    assert set(probabilities) == {"g1__s1", "g2__s1", "g3__s1"}
    assert contract["outer_test_data_used"] is False
    assert contract["final_fit_group_ids"] == ["g1", "g2", "g3"]
    for fold in contract["folds"]:
        assert fold["validation_group_id"] not in fold["fit_group_ids"]
        assert "g4" not in fold["fit_group_ids"]


def test_pre_gap_primary_window_and_gap_only_sensitivity_are_distinct() -> None:
    n_frames = 60
    vad = np.zeros((n_frames, 4), dtype=np.float64)
    vad[:20, 1] = 1.0
    vad[30:50, 0] = 1.0
    session = h2.CalibrationSession(
        session_id="g1__s1",
        group_id="g1",
        ticks=np.arange(100, 100 + n_frames),
        probabilities=np.zeros((n_frames, 4, 4)),
        vad=vad,
        valid=np.ones(n_frames, dtype=bool),
        present=np.ones(4),
        acoustic_features=np.zeros((n_frames, 3)),
    )
    trigger = np.zeros(n_frames, dtype=np.int8)
    trigger[15] = 1
    primary = h2.score_trace_events(
        session,
        trigger,
        pre_gap_frames=10,
        policy_id="predictive_demonstration_P1",
    )
    sensitivity = h2.score_trace_events(
        session,
        trigger,
        pre_gap_frames=0,
        policy_id="predictive_demonstration_P1",
    )
    assert len(primary) == len(sensitivity) == 1
    assert primary[0]["triggered"] == 1
    assert primary[0]["latency_s"] == pytest.approx(-0.2)
    assert sensitivity[0]["triggered"] == 0


def test_exact_303_event_support_and_unique_ids_are_enforced() -> None:
    records = [
        {"event_id": f"event-{index}", "robot_entry_label": int(index < 111)}
        for index in range(303)
    ]
    h2.validate_exact_event_support(records)
    records[-1]["event_id"] = records[0]["event_id"]
    with pytest.raises(h2.H2MatchedReplayError, match="303/111/192"):
        h2.validate_exact_event_support(records)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _reviewed_acoustic_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    scene = "03_20__4"
    root = tmp_path / "acoustic"
    root.mkdir()
    frame = pd.DataFrame(
        {
            "session_id": [scene] * 4,
            "timestamp_s": [0.0, 0.04, 0.08, 0.12],
            "array_energy": [-2.0, -1.0, 0.0, 1.0],
            "f0_proxy": [0.0, 120.0, 0.0, 130.0],
            "voiced": [0.0, 1.0, 0.0, 1.0],
        }
    )
    csv_payload = frame.to_csv(index=False, lineterminator="\n")
    source_csv = root / f"{scene}.source.csv"
    feature_csv = root / f"{scene}.csv"
    source_csv.write_text(csv_payload, encoding="utf-8", newline="")
    feature_csv.write_text(csv_payload, encoding="utf-8", newline="")
    evidence = root / f"{scene}.evidence.json"
    pending = root / f"{scene}.pending.manifest.json"
    review = root / f"{scene}.review.json"
    manifest = root / f"{scene}.manifest.json"
    _write_json(evidence, {"artifact_kind": "fixture_evidence"})
    _write_json(pending, {"session_id": scene, "review_status": "review_pending"})
    reviewed_bundle = {
        "source_csv": h2._file_binding(source_csv),
        "evidence_json": h2._file_binding(evidence),
        "feature_csv": h2._file_binding(feature_csv),
        "pending_manifest": h2._file_binding(pending),
    }
    _write_json(
        review,
        {
            "artifact_kind": "vap_real_causal_acoustic_review_receipt",
            "format_version": 1,
            "scene": scene,
            "decision": "accepted",
            "independent_review": True,
            "reviewed_bundle": reviewed_bundle,
        },
    )
    source_files = [
        h2._file_binding(source_csv),
        h2._file_binding(evidence),
        h2._file_binding(pending),
        h2._file_binding(review),
    ]
    _write_json(
        manifest,
        {
            "artifact_kind": "vap_feature_source",
            "format_version": 3,
            "session_id": scene,
            "purpose": "model_input",
            "assembly_eligible": True,
            "counted_input_ready": True,
            "review_status": "accepted",
            "supplied_columns": ["array_energy", "f0_proxy", "voiced"],
            "causality": {
                "causal": True,
                "lookahead_s": 0.0,
                "window_alignment": "trailing",
            },
            "feature_csv_sha256": h2._sha256_file(feature_csv),
            "source_files": source_files,
        },
    )
    assembled_csv = tmp_path / f"{scene}.csv"
    assembled_csv.write_text("session_id\n03_20__4\n", encoding="utf-8")
    assembled_provenance = tmp_path / f"{scene}.provenance.json"
    _write_json(
        assembled_provenance,
        {
            "feature_sources": [
                {
                    "columns": ["array_energy", "f0_proxy", "voiced"],
                    "path": str(feature_csv.resolve()),
                    "sha256": h2._sha256_file(feature_csv),
                    "manifest_path": str(manifest.resolve()),
                    "manifest_sha256": h2._sha256_file(manifest),
                    "causality": {
                        "causal": True,
                        "lookahead_s": 0.0,
                        "window_alignment": "trailing",
                    },
                }
            ]
        },
    )
    source_artifact = {
        "artifact_kind": "vap_real_assembled_session",
        "data_path": str(assembled_csv.resolve()),
        "data_sha256": h2._sha256_file(assembled_csv),
        "format_version": 2,
        "identity_authority": "sealed_hpe_transcript_crosswalk",
        "identity_inventory_sha256": "c" * 64,
        "provenance_path": str(assembled_provenance.resolve()),
        "provenance_sha256": h2._sha256_file(assembled_provenance),
    }
    return root, source_artifact


def test_e_only_loader_replays_local_bindings_but_never_raw_audio(tmp_path: Path) -> None:
    root, source_artifact = _reviewed_acoustic_fixture(tmp_path)
    loaded = h2.load_e_only_acoustic_descendant(
        "03_20__4",
        assembled_source_artifact=source_artifact,
        acoustic_root=root,
    )
    assert loaded.ticks.tolist() == [0, 1, 2, 3]
    assert loaded.bindings["raw_source_replay"] == "skipped_sealed_descendant_only"
    assert loaded.bindings["raw_source_paths_dereferenced"] is False

    feature = root / "03_20__4.csv"
    feature.write_text(feature.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(h2.H2MatchedReplayError, match="differ|accepted"):
        h2.load_e_only_acoustic_descendant(
            "03_20__4",
            assembled_source_artifact=source_artifact,
            acoustic_root=root,
        )


def test_f_drive_paths_are_rejected_before_any_read() -> None:
    with pytest.raises(h2.H2MatchedReplayError, match="F-drive"):
        h2.load_e_only_acoustic_descendant(
            "03_20__4",
            assembled_source_artifact={},
            acoustic_root=Path("F:/must-not-open"),
        )


def test_hash_tamper_and_create_once_publication_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    binding = h2._file_binding(source)
    assert h2.verify_file_binding(binding, role="fixture") == source.resolve()
    source.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(h2.H2MatchedReplayError, match="differ"):
        h2.verify_file_binding(binding, role="fixture")

    destination = tmp_path / "published"
    h2._publish_directory_create_once(destination, {Path("artifact.txt"): b"sealed\n"})
    with pytest.raises(FileExistsError, match="overwrite"):
        h2._publish_directory_create_once(destination, {Path("artifact.txt"): b"other\n"})


def test_checkpoint_scope_requires_paired_inner_and_exact_outer_train() -> None:
    fold = {
        "outer_fold_id": "outer_01",
        "outer_train_group_ids": ["03_26", "05_14", "05_15"],
        "outer_test_group_ids": ["03_20"],
        "outer_train_session_ids": [
            "03_26__3", "05_14__2", "05_14__3", "05_14__4",
            "05_15__2", "05_15__3", "05_15__6",
        ],
        "outer_test_session_ids": ["03_20__4"],
    }
    inner = {
        "inner_fold_id": "inner_01",
        "train_group_ids": ["05_14", "05_15"],
        "validation_group_ids": ["03_26"],
        "train_session_ids": [
            "05_14__2", "05_14__3", "05_14__4",
            "05_15__2", "05_15__3", "05_15__6",
        ],
        "validation_session_ids": ["03_26__3"],
    }
    checkpoint_sha = "d" * 64
    inner_checkpoint = {
        "condition_id": "A+H",
        "outer_fold_id": "outer_01",
        "inner_fold_id": "inner_01",
        "fit_role": "inner_selection",
        "seed": 13,
        "train_session_ids": inner["train_session_ids"],
    }
    prediction = {"checkpoint_sha256": checkpoint_sha}
    h2._validate_inner_checkpoint_scope(
        inner_checkpoint,
        prediction,
        checkpoint_sha256=checkpoint_sha,
        fold=fold,
        inner=inner,
    )
    leaked = copy.deepcopy(inner_checkpoint)
    leaked["train_session_ids"] = [*inner["train_session_ids"], "03_26__3"]
    with pytest.raises(h2.H2MatchedReplayError, match="scope differs"):
        h2._validate_inner_checkpoint_scope(
            leaked,
            prediction,
            checkpoint_sha256=checkpoint_sha,
            fold=fold,
            inner=inner,
        )
    with pytest.raises(h2.H2MatchedReplayError, match="scope differs"):
        h2._validate_inner_checkpoint_scope(
            inner_checkpoint,
            {"checkpoint_sha256": "f" * 64},
            checkpoint_sha256=checkpoint_sha,
            fold=fold,
            inner=inner,
        )

    final_checkpoint = {
        "condition_id": "A+H",
        "outer_fold_id": "outer_01",
        "inner_fold_id": None,
        "fit_role": "fixed_epoch_outer_train_from_scratch",
        "seed": 13,
        "train_session_ids": fold["outer_train_session_ids"],
    }
    h2._validate_final_checkpoint_scope(
        final_checkpoint,
        prediction,
        checkpoint_sha256=checkpoint_sha,
        fold=fold,
    )
    wrong_final = copy.deepcopy(final_checkpoint)
    wrong_final["train_session_ids"] = fold["outer_train_session_ids"][:-1]
    with pytest.raises(h2.H2MatchedReplayError, match="scope differs"):
        h2._validate_final_checkpoint_scope(
            wrong_final,
            prediction,
            checkpoint_sha256=checkpoint_sha,
            fold=fold,
        )


def _opaque(path: Path, fill: str = "e") -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": fill * 64, "size_bytes": 1}


def test_full_freeze_trace_evaluate_audit_replay_and_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenes = list(h2.DEADLINE_SESSION_TO_GROUP)
    groups = list(dict.fromkeys(h2.DEADLINE_SESSION_TO_GROUP.values()))
    grouped = {
        group: [scene for scene in scenes if h2.DEADLINE_SESSION_TO_GROUP[scene] == group]
        for group in groups
    }
    prediction_payloads: dict[str, dict[str, object]] = {}
    outer_paths: set[Path] = set()
    source_root = tmp_path / "opaque_sources"
    source_root.mkdir()
    assembled = {scene: {"scene": scene} for scene in scenes}

    def fake_freeze_budget(
        matrix_root: Path,
        *,
        destination: Path,
        budget_id: str,
        max_epochs: int,
    ) -> tuple[dict[str, object], dict[Path, bytes]]:
        del matrix_root
        files: dict[Path, bytes] = {}
        folds: dict[str, object] = {}
        for fold_index, (fold_id, test_group) in enumerate(
            zip(h2.OUTER_FOLD_IDS, groups, strict=True), start=1
        ):
            test_sessions = grouped[test_group]
            train_groups = [group for group in groups if group != test_group]
            train_sessions = [scene for group in train_groups for scene in grouped[group]]
            inner_folds = []
            for inner_id, validation_group in zip(
                h2.INNER_FOLD_IDS, train_groups, strict=True
            ):
                validation_sessions = grouped[validation_group]
                inner_train_groups = [
                    group for group in train_groups if group != validation_group
                ]
                inner_train_sessions = [
                    scene for group in inner_train_groups for scene in grouped[group]
                ]
                inner_folds.append(
                    {
                        "inner_fold_id": inner_id,
                        "train_group_ids": inner_train_groups,
                        "train_session_ids": inner_train_sessions,
                        "validation_group_ids": [validation_group],
                        "validation_session_ids": validation_sessions,
                        "artifact": _opaque(
                            source_root / f"{budget_id}-{fold_id}-{inner_id}-oof.json"
                        ),
                        "checkpoint_provenance": _opaque(
                            source_root / f"{budget_id}-{fold_id}-{inner_id}-checkpoint"
                        ),
                    }
                )
            checkpoint_sha = f"{fold_index}" * 64
            source = source_root / f"{budget_id}-{fold_id}-outer.json"
            source.write_text("opaque outer source\n", encoding="utf-8")
            outer_paths.add(source.resolve())
            records = []
            for scene in test_sessions:
                n_frames = 4
                records.append(
                    {
                        "session_id": scene,
                        "group_id": test_group,
                        "n_frames": n_frames,
                        "tick_25hz": list(range(n_frames)),
                        "probabilities": np.full((n_frames, 4, 4), 0.6).tolist(),
                        "labels": np.zeros((n_frames, 4, 4)).tolist(),
                        "vad": np.zeros((n_frames, 4)).tolist(),
                        "valid": np.ones(n_frames).tolist(),
                        "present": [1.0, 1.0, 1.0, 1.0],
                    }
                )
            prediction = {
                "condition_id": "A+H",
                "outer_fold_id": fold_id,
                "seed": 13,
                "frame_rate_hz": 25,
                "checkpoint_sha256": checkpoint_sha,
                "run_manifest_sha256": "b" * 64,
                "sessions": records,
            }
            prediction_payloads[str(source.resolve())] = prediction
            view = h2.gold_free_probability_view(
                prediction, budget_id=budget_id, max_epochs=max_epochs
            )
            view_rel = Path("outer_probability_views") / budget_id / f"{fold_id}.json"
            view_bytes = h2._pretty_json_bytes(view)
            files[view_rel] = view_bytes
            folds[fold_id] = {
                "outer_train_group_ids": train_groups,
                "outer_test_group_ids": [test_group],
                "outer_train_session_ids": train_sessions,
                "outer_test_session_ids": test_sessions,
                "result_manifest": _opaque(source_root / f"{budget_id}-{fold_id}-result"),
                "inner_oof": inner_folds,
                "final_checkpoint_provenance": {
                    "path": str((source_root / f"{budget_id}-{fold_id}-final").resolve()),
                    "sha256": checkpoint_sha,
                    "size_bytes": 1,
                },
                "outer_prediction_provenance": h2._file_binding(source),
                "gold_free_outer_view": h2._planned_binding(
                    destination / view_rel, view_bytes
                ),
            }
        return (
            {
                "budget_id": budget_id,
                "max_epochs": max_epochs,
                "matrix_index": _opaque(source_root / f"{budget_id}-matrix"),
                "protocol_directory": str(source_root.resolve()),
                "split_manifest": _opaque(source_root / f"{budget_id}-split"),
                "run_manifest": _opaque(source_root / f"{budget_id}-run"),
                "run_manifest_payload_sha256": "b" * 64,
                "assembled_source_artifacts": assembled,
                "folds": folds,
            },
            files,
        )

    calibration = {
        "artifact_kind": "vap_h2_inner_only_calibration",
        "format_version": 1,
        "result_role": h2.RESULT_ROLE,
        "production_eligible": False,
        "counted_eligible": False,
        "calibration_scope": "three_inner_group_OOF_only_no_outer_test",
        "speech_detector": {"final_model": _detector()},
        "speech_threshold": {"selected": {"threshold": 0.5}},
        "reactive_policy": {"selected": {"silent_frames": 5}},
        "predictive_policy": {
            "selected": {"now_threshold": 0.5, "future_threshold": 0.5}
        },
    }

    def fake_acoustic(
        scene: str,
        *,
        assembled_source_artifact: dict[str, object],
        acoustic_root: str | Path,
    ) -> h2.EOnlyAcousticSession:
        assert assembled_source_artifact == {"scene": scene}
        assert Path(acoustic_root).resolve() == (tmp_path / "acoustic").resolve()
        return h2.EOnlyAcousticSession(
            session_id=scene,
            ticks=np.arange(4, dtype=np.int64),
            features=np.zeros((4, 3), dtype=np.float64),
            bindings={
                "scene": scene,
                "raw_source_replay": "skipped_sealed_descendant_only",
                "raw_source_paths_dereferenced": False,
            },
        )

    def fake_score(
        session: h2.CalibrationSession,
        trigger: object,
        *,
        pre_gap_frames: int,
        policy_id: str,
    ) -> list[dict[str, object]]:
        del trigger, pre_gap_frames
        scene_index = scenes.index(session.session_id)
        count = 38 if scene_index < 7 else 37
        positive = 14 if scene_index < 7 else 13
        return [
            {
                "event_id": f"{session.session_id}:event-{index}",
                "session_id": session.session_id,
                "group_id": session.group_id,
                "event_index": index,
                "gap_start_tick": index * 2,
                "gap_end_tick_exclusive": index * 2 + 1,
                "previous_slot": "P2",
                "next_slot": "P1" if index < positive else "P3",
                "robot_entry_label": int(index < positive),
                "policy_id": policy_id,
                "triggered": 0,
                "first_trigger_tick": None,
                "latency_s": None,
            }
            for index in range(count)
        ]

    monkeypatch.setattr(h2, "_freeze_budget", fake_freeze_budget)
    monkeypatch.setattr(
        h2,
        "_load_inner_calibration_sessions",
        lambda *args, **kwargs: ((), {}),
    )
    monkeypatch.setattr(h2, "_calibrate_fold", lambda sessions: copy.deepcopy(calibration))
    monkeypatch.setattr(h2, "load_e_only_acoustic_descendant", fake_acoustic)
    monkeypatch.setattr(h2, "score_trace_events", fake_score)
    monkeypatch.setattr(
        h2,
        "_legacy_crosscheck",
        lambda ids: {"role": "fixture", "event_count": len(ids)},
    )
    monkeypatch.setattr(
        h2.thesis_backend,
        "load_prediction_artifact",
        lambda path: prediction_payloads[str(Path(path).resolve())],
    )

    freeze_root = tmp_path / "freeze"
    trace_root = tmp_path / "traces"
    evaluation_root = tmp_path / "evaluation"
    h2.freeze_h2_matched_replay(
        freeze_root,
        primary_matrix=tmp_path / "primary",
        sensitivity_matrix=tmp_path / "sensitivity",
    )
    assert (freeze_root / h2.PROJECTION_SEAL_FILENAME).is_file()

    real_open = Path.open
    real_sha256 = h2._sha256_file
    phase = {"value": "trace"}

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if phase["value"] == "trace" and path.resolve() in outer_paths:
            raise AssertionError("trace phase opened original outer prediction")
        return real_open(path, *args, **kwargs)

    def guarded_sha256(path: Path) -> str:
        if phase["value"] == "trace" and path.resolve() in outer_paths:
            raise AssertionError("trace phase hashed original outer prediction")
        return real_sha256(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(h2, "_sha256_file", guarded_sha256)
    h2.calibrate_and_seal_traces(
        freeze_root,
        trace_root,
        acoustic_root=tmp_path / "acoustic",
    )

    phase["value"] = "evaluation"
    h2.evaluate_h2_matched_replay(
        freeze_root,
        trace_root,
        evaluation_root,
        acoustic_root=tmp_path / "acoustic",
    )
    audit = h2.audit_h2_matched_replay(
        freeze_root,
        trace_directory=trace_root,
        evaluation_directory=evaluation_root,
        acoustic_root=tmp_path / "acoustic",
    )
    assert audit["freeze"] == audit["trace_seal"] == audit["evaluation"] == "passed"

    trace_path = trace_root / "traces" / "primary_5ep" / "outer_01.csv"
    trace_seal_path = trace_root / h2.TRACE_SEAL_FILENAME
    trace_bytes = trace_path.read_bytes()
    trace_seal_bytes = trace_seal_path.read_bytes()
    trace_path.write_bytes(trace_bytes.replace(b"0.0", b"9.0", 1))
    trace_seal = json.loads(trace_seal_bytes)
    replacement = h2._file_binding(trace_path)
    trace_seal["budgets"]["primary_5ep"]["outer_01"]["trace"].update(replacement)
    trace_seal_path.write_bytes(h2._pretty_json_bytes(trace_seal))
    phase["value"] = "trace"
    with pytest.raises(h2.H2MatchedReplayError, match="replay"):
        h2.audit_h2_matched_replay(
            freeze_root,
            trace_directory=trace_root,
            acoustic_root=tmp_path / "acoustic",
        )

    trace_path.write_bytes(trace_bytes)
    trace_seal_path.write_bytes(trace_seal_bytes)
    phase["value"] = "evaluation"
    metrics_path = evaluation_root / h2.METRICS_FILENAME
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["claim_boundary"] = "tampered"
    metrics_path.write_bytes(h2._pretty_json_bytes(metrics))
    evaluation_path = evaluation_root / h2.EVALUATION_FILENAME
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["metrics"] = h2._file_binding(metrics_path)
    evaluation_path.write_bytes(h2._pretty_json_bytes(evaluation))
    with pytest.raises(h2.H2MatchedReplayError, match="metric bytes differ"):
        h2.audit_h2_matched_replay(
            freeze_root,
            trace_directory=trace_root,
            evaluation_directory=evaluation_root,
            acoustic_root=tmp_path / "acoustic",
        )
