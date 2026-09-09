from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hri_turn_taking.vap import h2_matched_replay as v1
from hri_turn_taking.vap import h2_multiseed_replay as h2


def _prediction_payload(seed: int, n_frames: int = 8) -> dict[str, object]:
    probabilities = np.linspace(0.05, 0.95, n_frames * 16).reshape(n_frames, 4, 4)
    return {
        "condition_id": "A+H",
        "outer_fold_id": "outer_01",
        "seed": seed,
        "frame_rate_hz": 25,
        "checkpoint_sha256": "a" * 64,
        "run_manifest_sha256": h2.CANONICAL_SEED_SOURCES[seed][
            "primary_5ep"
        ].run_manifest_payload_sha256,
        "sessions": [
            {
                "session_id": "03_20__4",
                "group_id": "03_20",
                "n_frames": n_frames,
                "tick_25hz": list(range(n_frames)),
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
        "estimator": {"random_state": 13},
    }


def test_v1_is_immutable_and_extension_seed_is_exact() -> None:
    assert hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest() == (
        h2.V1_IMPLEMENTATION_SHA256
    )
    assert h2._require_model_seed(42) == 42
    assert h2._require_model_seed(73) == 73
    for invalid in (13, 0, True, 42.0):
        with pytest.raises(h2.H2MultiSeedReplayError, match="model_seed"):
            h2._require_model_seed(invalid)


def test_canonical_source_hashes_and_aggregate_outer_bindings_match() -> None:
    required = [
        source
        for seed in h2.SUPPORTED_MODEL_SEEDS
        for budget_id in h2.ANALYSIS_ORDER
        for source in (
            h2.CANONICAL_SEED_SOURCES[seed][budget_id].matrix / "matrix_index.json",
            h2.CANONICAL_SEED_SOURCES[seed][budget_id].aggregate,
        )
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("requires private multi-seed experiment artifacts")
    for seed in h2.SUPPORTED_MODEL_SEEDS:
        for budget_id in h2.ANALYSIS_ORDER:
            spec = h2.CANONICAL_SEED_SOURCES[seed][budget_id]
            index_path = spec.matrix / "matrix_index.json"
            assert hashlib.sha256(index_path.read_bytes()).hexdigest() == (
                spec.matrix_index_sha256
            )
            assert hashlib.sha256(spec.aggregate.read_bytes()).hexdigest() == (
                spec.aggregate_sha256
            )
            index = json.loads(index_path.read_text(encoding="utf-8"))
            aggregate = json.loads(spec.aggregate.read_text(encoding="utf-8"))
            assert index["seed"] == aggregate["seed"] == seed
            assert index["run_manifest_sha256"] == spec.run_manifest_payload_sha256
            units = {
                unit["outer_fold_id"]: unit
                for unit in index["units"]
                if unit["condition_id"] == "A+H"
            }
            observed = []
            for fold_id in h2.OUTER_FOLD_IDS:
                result_path = Path(units[fold_id]["result_manifest"])
                result = json.loads(result_path.read_text(encoding="utf-8"))
                outer = result_path.parent / result["artifact_bindings"][
                    "outer_predictions"
                ]["relative_path"]
                observed.append(hashlib.sha256(outer.read_bytes()).hexdigest())
            assert observed == aggregate["conditions"]["A+H"][
                "prediction_artifact_sha256"
            ]


def test_gold_free_projection_is_seed_bound_and_gold_invariant(tmp_path: Path) -> None:
    original = _prediction_payload(42)
    replaced = copy.deepcopy(original)
    replaced["sessions"][0]["labels"] = np.ones((8, 4, 4)).tolist()
    replaced["sessions"][0]["vad"] = np.ones((8, 4)).tolist()
    replaced["sessions"][0]["valid"] = np.zeros(8).tolist()
    left = h2._gold_free_probability_view(
        original,
        budget_id="primary_5ep",
        max_epochs=5,
        model_seed=42,
    )
    right = h2._gold_free_probability_view(
        replaced,
        budget_id="primary_5ep",
        max_epochs=5,
        model_seed=42,
    )
    assert left == right
    assert set(left["sessions"][0]).isdisjoint(
        {"labels", "vad", "valid", "features", "floor_holder"}
    )
    path = tmp_path / "view.json"
    path.write_bytes(v1._pretty_json_bytes(left))
    views = h2.load_gold_free_probability_view(path, expected_model_seed=42)
    assert views[0].session_id == "03_20__4"
    with pytest.raises(h2.H2MultiSeedReplayError, match="identity"):
        h2.load_gold_free_probability_view(path, expected_model_seed=73)


def test_dynamic_checkpoint_scope_rejects_cross_seed() -> None:
    fold = {
        "outer_fold_id": "outer_01",
        "outer_train_group_ids": ["03_26", "05_14", "05_15"],
        "outer_test_group_ids": ["03_20"],
        "outer_train_session_ids": [
            "03_26__3",
            "05_14__2",
            "05_14__3",
            "05_14__4",
            "05_15__2",
            "05_15__3",
            "05_15__6",
        ],
        "outer_test_session_ids": ["03_20__4"],
    }
    inner = {
        "inner_fold_id": "inner_01",
        "train_group_ids": ["05_14", "05_15"],
        "validation_group_ids": ["03_26"],
        "train_session_ids": [
            "05_14__2",
            "05_14__3",
            "05_14__4",
            "05_15__2",
            "05_15__3",
            "05_15__6",
        ],
        "validation_session_ids": ["03_26__3"],
    }
    checkpoint_hash = "d" * 64
    inner_checkpoint = {
        "condition_id": "A+H",
        "outer_fold_id": "outer_01",
        "inner_fold_id": "inner_01",
        "fit_role": "inner_selection",
        "seed": 42,
        "train_session_ids": inner["train_session_ids"],
    }
    prediction = {"seed": 42, "checkpoint_sha256": checkpoint_hash}
    h2._validate_inner_checkpoint_scope(
        inner_checkpoint,
        prediction,
        checkpoint_sha256=checkpoint_hash,
        fold=fold,
        inner=inner,
        model_seed=42,
    )
    mixed = copy.deepcopy(inner_checkpoint)
    mixed["seed"] = 73
    with pytest.raises(h2.H2MultiSeedReplayError, match="model_seed"):
        h2._validate_inner_checkpoint_scope(
            mixed,
            prediction,
            checkpoint_sha256=checkpoint_hash,
            fold=fold,
            inner=inner,
            model_seed=42,
        )
    final_checkpoint = {
        "condition_id": "A+H",
        "outer_fold_id": "outer_01",
        "inner_fold_id": None,
        "fit_role": "fixed_epoch_outer_train_from_scratch",
        "seed": 42,
        "train_session_ids": fold["outer_train_session_ids"],
    }
    h2._validate_final_checkpoint_scope(
        final_checkpoint,
        prediction,
        checkpoint_sha256=checkpoint_hash,
        fold=fold,
        model_seed=42,
    )
    prediction["seed"] = 73
    with pytest.raises(h2.H2MultiSeedReplayError, match="model_seed"):
        h2._validate_final_checkpoint_scope(
            final_checkpoint,
            prediction,
            checkpoint_sha256=checkpoint_hash,
            fold=fold,
            model_seed=42,
        )


def test_owned_publication_rolls_back_but_never_deletes_changed_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "rollback"

    def fail(_: Path) -> None:
        raise h2.H2MultiSeedReplayError("audit failure")

    with pytest.raises(h2.H2MultiSeedReplayError, match="audit failure"):
        h2._publish_with_owned_rollback(
            destination, {Path("artifact.txt"): b"sealed\n"}, fail
        )
    assert not destination.exists()

    protected = tmp_path / "protected"
    monkeypatch.setattr(h2, "_owned_tree_unchanged", lambda *args: False)
    with pytest.raises(h2.H2MultiSeedReplayError, match="audit failure"):
        h2._publish_with_owned_rollback(
            protected, {Path("artifact.txt"): b"sealed\n"}, fail
        )
    assert protected.is_dir()
    assert (protected / "artifact.txt").read_bytes() == b"sealed\n"


def _event_frame(seed: int) -> pd.DataFrame:
    records = []
    for budget_id, max_epochs in (("primary_5ep", 5), ("sensitivity_15ep", 15)):
        for policy_id in ("reactive", "predictive_demonstration_P1"):
            for window_id in (
                "primary_pre_gap_0.4s_to_gap_end",
                "sensitivity_gap_only",
            ):
                for index in range(303):
                    label = int(index < 111)
                    if policy_id == "reactive":
                        triggered = int(index % 3 == 0)
                    else:
                        triggered = int(index % (3 + (seed % 3)) == 0)
                    latency = (
                        0.2
                        if triggered and policy_id == "reactive"
                        else -0.1
                        if triggered
                        else None
                    )
                    records.append(
                        {
                            "model_seed": seed,
                            "budget_id": budget_id,
                            "max_epochs": max_epochs,
                            "policy_id": policy_id,
                            "window_id": window_id,
                            "outer_fold_id": "outer_01",
                            "event_id": f"event-{index}",
                            "session_id": "03_20__4",
                            "group_id": "03_20",
                            "event_index": index,
                            "gap_start_tick": index * 2,
                            "gap_end_tick_exclusive": index * 2 + 1,
                            "previous_slot": "P2",
                            "next_slot": "P1" if label else "P3",
                            "robot_entry_label": label,
                            "triggered": triggered,
                            "first_trigger_tick": index * 2 if triggered else None,
                            "latency_s": latency,
                        }
                    )
    return pd.DataFrame(records, columns=h2.EVENT_COLUMNS)


def _metric_tree(seed: int) -> dict[str, object]:
    results = {}
    for budget_index, budget_id in enumerate(h2.ANALYSIS_ORDER):
        budget = {}
        for policy_index, policy in enumerate(
            ("reactive", "predictive_demonstration_P1")
        ):
            windows = {}
            for window_id in (
                "primary_pre_gap_0.4s_to_gap_end",
                "sensitivity_gap_only",
            ):
                if policy == "reactive":
                    recall, fpr, balanced = 0.55, 0.35, 0.60
                else:
                    recall = 0.40 + seed / 1000 + budget_index * 0.05
                    fpr = 0.50 + seed / 2000 + budget_index * 0.03
                    balanced = (recall + 1.0 - fpr) / 2.0
                windows[window_id] = {
                    "pooled": {
                        "n_opportunities": 303,
                        "n_positive": 111,
                        "n_negative": 192,
                        "tp": 1 + policy_index,
                        "fp": 2 + policy_index,
                        "fn": 3,
                        "tn": 4,
                        "positive_recall": recall,
                        "negative_false_entry_rate": fpr,
                        "balanced_accuracy": balanced,
                        "positive_hit_latency_mean_s": 0.1,
                    },
                    "outer_folds": {},
                }
            budget[policy] = windows
        results[budget_id] = budget
    return {"results": results}


def _summary_sources() -> dict[int, h2._EvaluationSource]:
    calibration = {
        budget: {fold: {"fixed": True} for fold in h2.OUTER_FOLD_IDS}
        for budget in h2.ANALYSIS_ORDER
    }
    traces = {
        budget: {fold: "f" * 64 for fold in h2.OUTER_FOLD_IDS}
        for budget in h2.ANALYSIS_ORDER
    }
    return {
        seed: h2._EvaluationSource(
            model_seed=seed,
            events=_event_frame(seed),
            metrics=_metric_tree(seed),
            reactive_calibrations=copy.deepcopy(calibration),
            reactive_traces=copy.deepcopy(traces),
            bindings={"seed": seed},
        )
        for seed in h2.ALL_REPLAY_MODEL_SEEDS
    }


def test_three_seed_summary_requires_full_opportunity_and_reactive_invariance() -> None:
    sources = _summary_sources()
    metrics = h2._build_multiseed_metrics(sources)
    primary = metrics["results"]["primary_5ep"][
        "primary_pre_gap_0.4s_to_gap_end"
    ]
    assert primary["reactive_exact_across_model_seeds"] is True
    assert primary["predictive_mean_sample_sd"]["positive_recall"]["sample_sd"] > 0
    assert metrics["primary_result_remains_seed13_5epoch"] is True

    changed_event = _summary_sources()
    changed_event[73].events.loc[
        changed_event[73].events["event_id"] == "event-1", "gap_start_tick"
    ] = 999
    with pytest.raises(h2.H2MultiSeedReplayError, match="opportunities|metadata"):
        h2._build_multiseed_metrics(changed_event)

    changed_trace = _summary_sources()
    changed_trace[42].reactive_traces["primary_5ep"]["outer_01"] = "0" * 64
    with pytest.raises(h2.H2MultiSeedReplayError, match="reactive"):
        h2._build_multiseed_metrics(changed_trace)


def _opaque(path: Path, fill: str = "e") -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": fill * 64, "size_bytes": 1}


def test_full_seed42_freeze_trace_evaluate_audit_without_outer_gold_in_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = 42
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
        spec: h2.FrozenSourceSpec,
        *,
        destination: Path,
        budget_id: str,
        model_seed: int,
    ) -> tuple[dict[str, object], dict[Path, bytes]]:
        assert model_seed == seed
        files: dict[Path, bytes] = {}
        folds: dict[str, object] = {}
        for fold_index, (fold_id, test_group) in enumerate(
            zip(h2.OUTER_FOLD_IDS, groups, strict=True), start=1
        ):
            test_sessions = grouped[test_group]
            train_groups = [group for group in groups if group != test_group]
            train_sessions = [scene for group in train_groups for scene in grouped[group]]
            inner_records = []
            for inner_id, validation_group in zip(
                h2.INNER_FOLD_IDS, train_groups, strict=True
            ):
                validation_sessions = grouped[validation_group]
                inner_train_groups = [
                    group for group in train_groups if group != validation_group
                ]
                inner_records.append(
                    {
                        "inner_fold_id": inner_id,
                        "train_group_ids": inner_train_groups,
                        "train_session_ids": [
                            scene
                            for group in inner_train_groups
                            for scene in grouped[group]
                        ],
                        "validation_group_ids": [validation_group],
                        "validation_session_ids": validation_sessions,
                        "artifact": _opaque(
                            source_root / f"{budget_id}-{fold_id}-{inner_id}-oof"
                        ),
                        "checkpoint_provenance": _opaque(
                            source_root / f"{budget_id}-{fold_id}-{inner_id}-checkpoint"
                        ),
                    }
                )
            checkpoint_sha = str(fold_index) * 64
            source = source_root / f"{budget_id}-{fold_id}-outer.json"
            source.write_text("opaque outer source\n", encoding="utf-8")
            outer_paths.add(source.resolve())
            records = []
            for scene in test_sessions:
                records.append(
                    {
                        "session_id": scene,
                        "group_id": test_group,
                        "n_frames": 4,
                        "tick_25hz": [0, 1, 2, 3],
                        "probabilities": np.full((4, 4, 4), 0.6).tolist(),
                        "labels": np.zeros((4, 4, 4)).tolist(),
                        "vad": np.zeros((4, 4)).tolist(),
                        "valid": np.ones(4).tolist(),
                        "present": [1.0, 1.0, 1.0, 1.0],
                    }
                )
            prediction = {
                "condition_id": "A+H",
                "outer_fold_id": fold_id,
                "seed": seed,
                "frame_rate_hz": 25,
                "checkpoint_sha256": checkpoint_sha,
                "run_manifest_sha256": spec.run_manifest_payload_sha256,
                "sessions": records,
            }
            prediction_payloads[str(source.resolve())] = prediction
            view = h2._gold_free_probability_view(
                prediction,
                budget_id=budget_id,
                max_epochs=spec.max_epochs,
                model_seed=seed,
            )
            relative = Path("outer_probability_views") / budget_id / f"{fold_id}.json"
            view_bytes = v1._pretty_json_bytes(view)
            files[relative] = view_bytes
            folds[fold_id] = {
                "outer_train_group_ids": train_groups,
                "outer_test_group_ids": [test_group],
                "outer_train_session_ids": train_sessions,
                "outer_test_session_ids": test_sessions,
                "result_manifest": _opaque(source_root / f"{budget_id}-{fold_id}-result"),
                "inner_oof": inner_records,
                "final_checkpoint_provenance": {
                    "path": str(
                        (source_root / f"{budget_id}-{fold_id}-final").resolve()
                    ),
                    "sha256": checkpoint_sha,
                    "size_bytes": 1,
                },
                "outer_prediction_provenance": v1._file_binding(source),
                "gold_free_outer_view": v1._planned_binding(
                    destination / relative, view_bytes
                ),
            }
        return (
            {
                "budget_id": budget_id,
                "max_epochs": spec.max_epochs,
                "model_seed": seed,
                "matrix_index": _opaque(source_root / f"{budget_id}-matrix"),
                "independently_audited_aggregate": _opaque(
                    source_root / f"{budget_id}-aggregate"
                ),
                "protocol_directory": str(source_root.resolve()),
                "split_manifest": _opaque(source_root / f"{budget_id}-split"),
                "run_manifest": _opaque(source_root / f"{budget_id}-run"),
                "run_manifest_payload_sha256": spec.run_manifest_payload_sha256,
                "assembled_source_artifacts": assembled,
                "outer_prediction_sha256_in_fold_order": ["f" * 64] * 4,
                "folds": folds,
            },
            files,
        )

    calibration = {
        "artifact_kind": "vap_h2_inner_only_calibration",
        "format_version": 1,
        "result_role": v1.RESULT_ROLE,
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
    ) -> v1.EOnlyAcousticSession:
        assert assembled_source_artifact == {"scene": scene}
        assert Path(acoustic_root).resolve() == (tmp_path / "acoustic").resolve()
        return v1.EOnlyAcousticSession(
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
        session: v1.CalibrationSession,
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
        h2, "_load_inner_calibration_sessions", lambda *args, **kwargs: ((), {})
    )
    monkeypatch.setattr(v1, "_calibrate_fold", lambda sessions: copy.deepcopy(calibration))
    monkeypatch.setattr(v1, "load_e_only_acoustic_descendant", fake_acoustic)
    monkeypatch.setattr(v1, "score_trace_events", fake_score)
    monkeypatch.setattr(
        v1, "_legacy_crosscheck", lambda ids: {"role": "fixture", "count": len(ids)}
    )
    monkeypatch.setattr(
        h2.thesis_backend,
        "load_prediction_artifact",
        lambda path: prediction_payloads[str(Path(path).resolve())],
    )

    freeze_root = tmp_path / "freeze"
    trace_root = tmp_path / "traces"
    evaluation_root = tmp_path / "evaluation"
    h2.freeze_h2_multiseed_seed(freeze_root, model_seed=seed)

    real_open = Path.open
    real_sha = v1._sha256_file
    phase = {"value": "trace"}

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if phase["value"] == "trace" and path.resolve() in outer_paths:
            raise AssertionError("trace phase opened original outer prediction")
        return real_open(path, *args, **kwargs)

    def guarded_sha(path: Path) -> str:
        if phase["value"] == "trace" and path.resolve() in outer_paths:
            raise AssertionError("trace phase hashed original outer prediction")
        return real_sha(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(v1, "_sha256_file", guarded_sha)
    h2.calibrate_and_seal_h2_multiseed_traces(
        freeze_root, trace_root, acoustic_root=tmp_path / "acoustic"
    )
    phase["value"] = "evaluation"
    h2.evaluate_h2_multiseed_seed(
        freeze_root,
        trace_root,
        evaluation_root,
        acoustic_root=tmp_path / "acoustic",
    )
    audit = h2.audit_h2_multiseed_seed(
        freeze_root,
        trace_directory=trace_root,
        evaluation_directory=evaluation_root,
        acoustic_root=tmp_path / "acoustic",
    )
    assert audit["freeze"] == audit["trace_seal"] == audit["evaluation"] == "passed"
    assert audit["model_seed"] == 42

    calibration_path = trace_root / "calibration" / "primary_5ep" / "outer_01.json"
    trace_seal_path = trace_root / h2.TRACE_SEAL_FILENAME
    original_calibration_bytes = calibration_path.read_bytes()
    original_trace_seal_bytes = trace_seal_path.read_bytes()
    forged_calibration = json.loads(original_calibration_bytes)
    forged_calibration["speech_threshold"]["selected"]["threshold"] = 0.25
    calibration_path.write_bytes(v1._pretty_json_bytes(forged_calibration))
    resealed_trace = json.loads(original_trace_seal_bytes)
    resealed_trace["budgets"]["primary_5ep"]["outer_01"]["calibration"] = (
        v1._file_binding(calibration_path)
    )
    trace_seal_path.write_bytes(v1._pretty_json_bytes(resealed_trace))
    with pytest.raises(h2.H2MultiSeedReplayError, match="canonical inner OOF/acoustic"):
        h2.audit_h2_multiseed_seed(
            freeze_root,
            trace_directory=trace_root,
            acoustic_root=tmp_path / "acoustic",
        )
    calibration_path.write_bytes(original_calibration_bytes)
    trace_seal_path.write_bytes(original_trace_seal_bytes)

    view_path = (
        freeze_root / "outer_probability_views" / "primary_5ep" / "outer_01.json"
    )
    forged_view = json.loads(view_path.read_bytes())
    forged_view["sessions"][0]["probabilities"][0][0][0] = 0.7
    view_path.write_bytes(v1._pretty_json_bytes(forged_view))
    resealed_view = v1._file_binding(view_path)

    projection_path = freeze_root / h2.PROJECTION_SEAL_FILENAME
    projection = json.loads(projection_path.read_bytes())
    projection["views"]["primary_5ep"]["outer_01"] = resealed_view
    projection_path.write_bytes(v1._pretty_json_bytes(projection))

    freeze_manifest_path = freeze_root / h2.FREEZE_FILENAME
    freeze_manifest = json.loads(freeze_manifest_path.read_bytes())
    freeze_manifest["budgets"]["primary_5ep"]["folds"]["outer_01"][
        "gold_free_outer_view"
    ] = resealed_view
    freeze_manifest["projection_seal"] = v1._file_binding(projection_path)
    freeze_manifest_path.write_bytes(v1._pretty_json_bytes(freeze_manifest))
    with pytest.raises(h2.H2MultiSeedReplayError, match="canonical source reconstruction"):
        h2.audit_h2_multiseed_seed(freeze_root)
