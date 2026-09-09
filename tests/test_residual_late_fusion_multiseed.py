from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hri_turn_taking.vap import residual_late_fusion as v1
from hri_turn_taking.vap import residual_late_fusion_multiseed as multi


def _binding(name: str, digest: str = "0" * 64) -> multi.ExternalBinding:
    return multi.ExternalBinding(
        absolute_path=str((Path.cwd() / name).resolve()),
        sha256=digest,
        size_bytes=1,
    )


def _session(probability: np.ndarray) -> dict[str, object]:
    frames = probability.shape[0]
    return {
        "session_id": "03_20__4",
        "group_id": "03_20",
        "n_frames": frames,
        "tick_25hz": list(range(frames)),
        "labels": np.zeros((frames, 4, 4), dtype=np.float32).tolist(),
        "vad": np.zeros((frames, 4), dtype=np.float32).tolist(),
        "valid": np.ones((frames, 4, 4), dtype=np.int8).tolist(),
        "present": [1, 1, 1, 1],
        "probabilities": probability.astype(np.float32).tolist(),
    }


def _prediction(condition: str, probability: np.ndarray, *, seed: int = 42) -> dict[str, object]:
    return {
        "condition_id": condition,
        "outer_fold_id": "outer_01",
        "inner_fold_id": "inner_01",
        "seed": seed,
        "run_manifest_sha256": v1.SOURCE_RUN_MANIFEST_SHA256,
        "frame_rate_hz": 25,
        "label_config": {"bin_edges_s": [0.2, 0.6, 1.2, 2.0]},
        "threshold_policy": {"diagnostic_threshold": 0.5},
        "sessions": [_session(probability)],
    }


def _advance_float32(value: float, ulps: int) -> np.float32:
    bits = np.asarray([value], dtype=np.float32).view(np.uint32)
    bits[0] += np.uint32(ulps)
    return bits.view(np.float32)[0]


def _metrics(base: float) -> dict[str, object]:
    return {
        "frame": {
            "mAP": base,
            "participant_macro_mAP": base + 0.01,
        },
        "events": {
            "shift_prediction": {"average_precision": base + 0.02},
            "overlap_resolution": {"average_precision": base + 0.03},
            "robot_entry": {"average_precision": base + 0.04},
            "next_speaker": {
                "participant_macro_ovr_ap": base + 0.05,
                "top1_accuracy": base + 0.06,
            },
        },
    }


def _seed_aggregate(seed: int, selected_base: float, control_base: float) -> dict[str, object]:
    return {
        "seed": seed,
        "pooled": {
            "selected": _metrics(selected_base),
            "matched_lambda0": _metrics(control_base),
        },
        "per_fold": {
            fold: {"selected_lambda": float(index) / 4.0}
            for index, fold in enumerate(v1.OUTER_FOLD_IDS)
        },
        "success_evaluation": {
            "descriptive_criteria_met": selected_base - control_base >= 0.02,
            "verdict": "descriptive_criteria_not_met",
        },
    }


def _frozen_inputs(seed13: dict[str, object]) -> multi.FrozenInputs:
    return multi.FrozenInputs(
        sources={},
        v1_implementation_binding=_binding("v1.py", multi.V1_IMPLEMENTATION_SHA256),
        v1_aggregate_root=(Path.cwd() / "v1").resolve(),
        v1_aggregate_binding=_binding("v1/aggregate.json", multi.V1_AGGREGATE_SHA256),
        v1_aggregate=seed13,
    )


def _group_sessions(group_id: str) -> tuple[str, ...]:
    return tuple(
        session_id
        for session_id in v1.EXACT_EIGHT_SESSION_IDS
        if session_id.split("__", maxsplit=1)[0] == group_id
    )


def _artifact_fixture_source(tmp_path: Path, seed: int) -> multi.SeedSource:
    root = (tmp_path / f"matrix_{seed}").resolve()
    protocol_root = (tmp_path / f"protocol_{seed}").resolve()
    root.mkdir()
    protocol_root.mkdir()
    index_path = root / "matrix_index.json"
    index_path.write_text("{}\n", encoding="utf-8")
    run_manifest_path = protocol_root / "run_manifest.json"
    run_manifest_path.write_text("{}\n", encoding="utf-8")
    aggregate_path = tmp_path / f"training_aggregate_{seed}.json"
    aggregate_path.write_text("{}\n", encoding="utf-8")
    units: dict[tuple[str, str], v1.SourceUnit] = {}
    result_bindings: dict[tuple[str, str], v1.FileBinding] = {}
    for condition in v1.SOURCE_CONDITIONS:
        for fold in v1.OUTER_FOLD_IDS:
            directory = root / condition / fold
            directory.mkdir(parents=True)
            result_path = directory / "result_manifest.json"
            result_path.write_text("{}\n", encoding="utf-8")
            prediction_bindings: dict[str, v1.FileBinding] = {}
            for key in (*v1.INNER_FOLD_IDS, "outer"):
                prediction_path = directory / f"{key}.npz"
                prediction_path.write_bytes(f"{seed}-{condition}-{fold}-{key}".encode())
                prediction_bindings[key] = v1._binding(prediction_path, directory)
            result_binding = v1._binding(result_path, root)
            result_bindings[(condition, fold)] = result_binding
            units[(condition, fold)] = v1.SourceUnit(
                condition_id=condition,
                outer_fold_id=fold,
                directory=directory,
                result_manifest_path=result_path,
                result_manifest_binding=result_binding,
                prediction_bindings=prediction_bindings,
            )
    matrix = v1.SourceMatrix(
        root=root,
        index_path=index_path,
        index_binding=v1._binding(index_path, root),
        index={},
        protocol_root=protocol_root,
        run_manifest_path=run_manifest_path,
        run_manifest_binding=v1._binding(run_manifest_path, protocol_root),
        run_manifest={},
        units=units,
        result_manifest_bindings=result_bindings,
        source_file_snapshots=(),
    )
    contract = multi.FrozenSeedContract(
        seed=seed,
        matrix_root=root,
        matrix_index_sha256=v1._sha256_file(index_path),
        training_aggregate=aggregate_path,
        training_aggregate_sha256=v1._sha256_file(aggregate_path),
        protocol_root=protocol_root,
        run_manifest_raw_sha256=v1._sha256_file(run_manifest_path),
    )
    return multi.SeedSource(
        contract=contract,
        matrix=matrix,
        training_aggregate_path=aggregate_path,
        training_aggregate_binding=multi._external_binding(aggregate_path),
        training_aggregate={},
    )


def _artifact_fixture_payload(
    condition: str,
    *,
    seed: int,
    outer_fold_id: str,
    group_id: str,
    inner_fold_id: str | None,
) -> dict[str, object]:
    group_index = v1.EXACT_FOUR_GROUP_IDS.index(group_id)
    sessions = []
    for session_index, session_id in enumerate(_group_sessions(group_id)):
        n_frames = 4
        if condition == "A":
            per_bin = np.asarray([0.53, 0.54, 0.55, 0.56], dtype=np.float32)
            probabilities = np.broadcast_to(
                per_bin[None, None, :],
                (n_frames, 4, 4),
            ).copy()
        else:
            probabilities = np.empty((n_frames, 4, 4), dtype=np.float32)
            for frame in range(n_frames):
                for participant in range(4):
                    for bin_index in range(4):
                        probabilities[frame, participant, bin_index] = (
                            0.31
                            + 0.015 * group_index
                            + 0.01 * session_index
                            + 0.012 * participant
                            + 0.006 * bin_index
                            + 0.003 * frame
                            + 0.0001 * (seed - 42)
                        )
        record = _session(probabilities)
        record["session_id"] = session_id
        record["group_id"] = group_id
        sessions.append(record)
    payload: dict[str, object] = {
        "condition_id": condition,
        "outer_fold_id": outer_fold_id,
        "seed": seed,
        "run_manifest_sha256": v1.SOURCE_RUN_MANIFEST_SHA256,
        "frame_rate_hz": 25,
        "label_config": {"bin_edges_s": [0.2, 0.6, 1.2, 2.0]},
        "threshold_policy": {"diagnostic_threshold": 0.5},
        "sessions": sessions,
    }
    if inner_fold_id is not None:
        payload["inner_fold_id"] = inner_fold_id
    return payload


def _artifact_fixture_inputs(
    tmp_path: Path,
) -> tuple[multi.FrozenInputs, dict[tuple[int, str, str | None], v1.PairedPredictions]]:
    sources = {
        seed: _artifact_fixture_source(tmp_path, seed) for seed in multi.ALLOWED_NEW_SEEDS
    }
    pairs: dict[tuple[int, str, str | None], v1.PairedPredictions] = {}
    for seed, source in sources.items():
        for outer_index, outer_fold_id in enumerate(v1.OUTER_FOLD_IDS):
            outer_group = v1.EXACT_FOUR_GROUP_IDS[outer_index]
            groups = [group for group in v1.EXACT_FOUR_GROUP_IDS if group != outer_group]
            identities = [*zip(v1.INNER_FOLD_IDS, groups, strict=True), (None, outer_group)]
            for inner_fold_id, group_id in identities:
                a_payload = _artifact_fixture_payload(
                    "A",
                    seed=seed,
                    outer_fold_id=outer_fold_id,
                    group_id=group_id,
                    inner_fold_id=inner_fold_id,
                )
                h_payload = _artifact_fixture_payload(
                    "H",
                    seed=seed,
                    outer_fold_id=outer_fold_id,
                    group_id=group_id,
                    inner_fold_id=inner_fold_id,
                )
                key = inner_fold_id if inner_fold_id is not None else "outer"
                a_unit = source.matrix.units[("A", outer_fold_id)]
                h_unit = source.matrix.units[("H", outer_fold_id)]
                a_path = v1._verify_binding(a_unit.prediction_bindings[key], a_unit.directory)
                h_path = v1._verify_binding(h_unit.prediction_bindings[key], h_unit.directory)
                symmetry = multi._validate_paired_payloads(
                    a_payload,
                    h_payload,
                    seed=seed,
                    outer_fold_id=outer_fold_id,
                    inner_fold_id=inner_fold_id,
                )
                pairs[(seed, outer_fold_id, inner_fold_id)] = v1.PairedPredictions(
                    a_payload=a_payload,
                    h_payload=h_payload,
                    a_path=a_path,
                    h_path=h_path,
                    a_binding=v1._binding(a_path, source.matrix.root),
                    h_binding=v1._binding(h_path, source.matrix.root),
                    a_symmetry_max_abs_diff=symmetry,
                )
    seed13 = _seed_aggregate(13, 0.38, 0.35)
    v1_root = (tmp_path / "seed13_v1").resolve()
    v1_root.mkdir()
    v1_path = v1_root / v1.AGGREGATE_FILENAME
    v1_path.write_text(json.dumps(seed13) + "\n", encoding="utf-8")
    inputs = multi.FrozenInputs(
        sources=sources,
        v1_implementation_binding=multi._external_binding(Path(v1.__file__).resolve()),
        v1_aggregate_root=v1_root,
        v1_aggregate_binding=multi._external_binding(v1_path),
        v1_aggregate=seed13,
    )
    return inputs, pairs


def test_v2_binds_repository_v1_implementation_without_modifying_it() -> None:
    implementation = Path(v1.__file__).resolve()
    digest = hashlib.sha256(implementation.read_bytes()).hexdigest()
    assert digest == multi.V1_IMPLEMENTATION_SHA256
    assert multi._verify_v1_core().sha256 == digest
    assert multi.ALLOWED_NEW_SEEDS == (42, 73)
    assert set(multi.SOURCE_CONTRACTS) == {42, 73}
    assert multi.A_SYMMETRY_ABS_TOLERANCE == 2e-7
    assert multi.A_SYMMETRY_ULP_TOLERANCE == 4


def test_production_source_and_result_hash_bindings_are_exact() -> None:
    v1_path = multi.V1_RESULT_ROOT / v1.AGGREGATE_FILENAME
    required = [v1_path]
    for seed in multi.ALLOWED_NEW_SEEDS:
        contract = multi.SOURCE_CONTRACTS[seed]
        required.extend(
            (
                contract.matrix_root / "matrix_index.json",
                contract.training_aggregate,
                contract.protocol_root / "run_manifest.json",
            )
        )
    if not all(path.is_file() for path in required):
        pytest.skip("requires private multi-seed experiment artifacts")
    assert v1._sha256_file(v1_path) == multi.V1_AGGREGATE_SHA256
    seed13 = v1._read_json_strict(v1_path)
    assert seed13["artifact_kind"] == v1.AGGREGATE_KIND
    assert seed13["seed"] == 13
    for seed in multi.ALLOWED_NEW_SEEDS:
        contract = multi.SOURCE_CONTRACTS[seed]
        index_path = contract.matrix_root / "matrix_index.json"
        assert v1._sha256_file(index_path) == contract.matrix_index_sha256
        index = v1._read_json_strict(index_path)
        assert index["seed"] == seed
        assert index["included_session_ids"] == list(v1.EXACT_EIGHT_SESSION_IDS)
        assert index["group_ids"] == list(v1.EXACT_FOUR_GROUP_IDS)
        assert index["run_manifest_sha256"] == v1.SOURCE_RUN_MANIFEST_SHA256
        assert v1._sha256_file(contract.training_aggregate) == (
            contract.training_aggregate_sha256
        )
        training_aggregate = v1._read_json_strict(contract.training_aggregate)
        assert training_aggregate["seed"] == seed
        assert training_aggregate["aggregation_source"] == (
            "saved_outer_probabilities_only_no_model_inference"
        )
        run_manifest_path = contract.protocol_root / "run_manifest.json"
        assert v1._sha256_file(run_manifest_path) == contract.run_manifest_raw_sha256
        run_manifest = v1._read_json_strict(run_manifest_path)
        assert v1._canonical_json_sha256(run_manifest) == v1.SOURCE_RUN_MANIFEST_SHA256
        assert run_manifest["max_epochs"] == 15
        assert run_manifest["seeds"] == list(v1.SOURCE_SEEDS)


def test_production_a_sources_pass_exact_repair_round1_symmetry_scan() -> None:
    required = [
        multi.SOURCE_CONTRACTS[seed].matrix_root / "matrix_index.json"
        for seed in multi.ALLOWED_NEW_SEEDS
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("requires private multi-seed experiment artifacts")
    expected = {
        42: (8.940696716308594e-08, 3),
        73: (1.1920928955078125e-07, 4),
    }
    for seed in multi.ALLOWED_NEW_SEEDS:
        root = multi.SOURCE_CONTRACTS[seed].matrix_root
        observed_abs = 0.0
        observed_ulp = 0
        artifact_count = 0
        frame_bin_count = 0
        for outer_fold_id in v1.OUTER_FOLD_IDS:
            unit = root / "A" / outer_fold_id
            result = v1._read_json_strict(unit / "result_manifest.json")
            bindings = result["artifact_bindings"]
            sources = [
                (item["inner_fold_id"], item)
                for item in bindings["inner_oof_predictions"]
            ]
            sources.append((None, bindings["outer_predictions"]))
            for inner_fold_id, binding in sources:
                path = unit / binding["relative_path"]
                payload = (
                    v1.thesis_backend.load_prediction_artifact(path)
                    if inner_fold_id is None
                    else v1.thesis_backend._load_prediction_artifact(
                        path,
                        expected_inner_fold_id=inner_fold_id,
                    )
                )
                current_abs, current_ulp = multi._participant_symmetry_diagnostics(payload)
                observed_abs = max(observed_abs, current_abs)
                observed_ulp = max(observed_ulp, current_ulp)
                artifact_count += 1
                frame_bin_count += sum(
                    int(np.asarray(record["probabilities"]).shape[0]) * 4
                    for record in payload["sessions"]
                )
        assert artifact_count == 16
        assert frame_bin_count == 1_028_784
        assert observed_abs == expected[seed][0]
        assert observed_ulp == expected[seed][1]
        assert observed_abs <= multi.A_SYMMETRY_ABS_TOLERANCE
        assert observed_ulp <= multi.A_SYMMETRY_ULP_TOLERANCE


def test_source_adapter_rejects_seed13_before_any_file_access() -> None:
    forbidden = multi.FrozenSeedContract(
        seed=13,
        matrix_root=Path("does-not-exist"),
        matrix_index_sha256="0" * 64,
        training_aggregate=Path("does-not-exist.json"),
        training_aggregate_sha256="0" * 64,
        protocol_root=Path("does-not-exist-protocol"),
        run_manifest_raw_sha256="0" * 64,
    )
    with pytest.raises(
        multi.ResidualLateFusionMultiseedError,
        match="only new seeds 42 and 73",
    ):
        multi._load_seed_source(forbidden)


def test_pairing_enforces_seed_labels_and_a_symmetry() -> None:
    a_probability = np.full((3, 4, 4), 0.55, dtype=np.float32)
    h_probability = np.linspace(0.2, 0.8, 48, dtype=np.float32).reshape(3, 4, 4)
    a_payload = _prediction("A", a_probability)
    h_payload = _prediction("H", h_probability)
    assert multi._validate_paired_payloads(
        a_payload,
        h_payload,
        seed=42,
        outer_fold_id="outer_01",
        inner_fold_id="inner_01",
    ) == 0.0

    wrong_seed = copy.deepcopy(h_payload)
    wrong_seed["seed"] = 13
    with pytest.raises(multi.ResidualLateFusionMultiseedError, match="fold/seed"):
        multi._validate_paired_payloads(
            a_payload,
            wrong_seed,
            seed=42,
            outer_fold_id="outer_01",
            inner_fold_id="inner_01",
        )

    wrong_label = copy.deepcopy(h_payload)
    wrong_label["sessions"][0]["labels"][0][0][0] = 1.0
    with pytest.raises(multi.ResidualLateFusionMultiseedError, match="paired labels"):
        multi._validate_paired_payloads(
            a_payload,
            wrong_label,
            seed=42,
            outer_fold_id="outer_01",
            inner_fold_id="inner_01",
        )

    asymmetric = copy.deepcopy(a_payload)
    asymmetric["sessions"][0]["probabilities"][0][0][0] += 1e-4
    with pytest.raises(multi.ResidualLateFusionMultiseedError, match="symmetry exceeds"):
        multi._validate_paired_payloads(
            asymmetric,
            h_payload,
            seed=42,
            outer_fold_id="outer_01",
            inner_fold_id="inner_01",
        )


def test_symmetry_gate_accepts_four_ulps_and_rejects_five_or_absolute_excess() -> None:
    base = np.float32(0.45)
    h_probability = np.full((3, 4, 4), 0.4, dtype=np.float32)

    four_ulp = np.full((3, 4, 4), base, dtype=np.float32)
    four_ulp[0, 3, 0] = _advance_float32(float(base), 4)
    four_payload = _prediction("A", four_ulp)
    h_payload = _prediction("H", h_probability)
    observed_abs, observed_ulp = multi._participant_symmetry_diagnostics(four_payload)
    assert observed_ulp == 4
    assert observed_abs <= multi.A_SYMMETRY_ABS_TOLERANCE
    assert multi._validate_paired_payloads(
        four_payload,
        h_payload,
        seed=42,
        outer_fold_id="outer_01",
        inner_fold_id="inner_01",
    ) == observed_abs

    five_ulp = np.full((3, 4, 4), base, dtype=np.float32)
    five_ulp[0, 3, 0] = _advance_float32(float(base), 5)
    five_payload = _prediction("A", five_ulp)
    five_abs, five_distance = multi._participant_symmetry_diagnostics(five_payload)
    assert five_abs <= multi.A_SYMMETRY_ABS_TOLERANCE
    assert five_distance == 5
    with pytest.raises(
        multi.ResidualLateFusionMultiseedError,
        match="numerical-invariance gate",
    ):
        multi._validate_paired_payloads(
            five_payload,
            h_payload,
            seed=42,
            outer_fold_id="outer_01",
            inner_fold_id="inner_01",
        )

    high_base = np.float32(0.9)
    absolute_excess = np.full((3, 4, 4), high_base, dtype=np.float32)
    absolute_excess[0, 3, 0] = _advance_float32(float(high_base), 4)
    absolute_payload = _prediction("A", absolute_excess)
    high_abs, high_distance = multi._participant_symmetry_diagnostics(absolute_payload)
    assert high_distance == 4
    assert high_abs > multi.A_SYMMETRY_ABS_TOLERANCE
    with pytest.raises(
        multi.ResidualLateFusionMultiseedError,
        match="numerical-invariance gate",
    ):
        multi._validate_paired_payloads(
            absolute_payload,
            h_payload,
            seed=42,
            outer_fold_id="outer_01",
            inner_fold_id="inner_01",
        )


def test_lambda_zero_is_exact_h_and_uses_frozen_formula() -> None:
    h_probability = np.linspace(0.1, 0.9, 48, dtype=np.float32).reshape(3, 4, 4)
    a_probability = np.full((3, 4, 4), 0.6, dtype=np.float32)
    control = v1.fuse_probability_arrays(h_probability, a_probability, 0.0)
    assert np.array_equal(control, h_probability)
    assert v1._canonical_probability_bytes_sha256((control,)) == (
        v1._canonical_probability_bytes_sha256((h_probability,))
    )
    assert v1.FORMULA == "sigmoid(logit(H_probability)+lambda*logit(A_probability))"


def test_fold_recompute_selects_lambda_before_loading_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SimpleNamespace(seed=42)
    load_order: list[str | None] = []
    selection_complete = False

    def pair(
        _source: object,
        _outer_fold_id: str,
        inner_fold_id: str | None,
    ) -> SimpleNamespace:
        load_order.append(inner_fold_id)
        if inner_fold_id is None:
            assert selection_complete
        return SimpleNamespace(marker=inner_fold_id)

    def select(pairs: tuple[SimpleNamespace, ...]) -> tuple[float, list[dict[str, object]]]:
        nonlocal selection_complete
        assert [pair.marker for pair in pairs] == list(v1.INNER_FOLD_IDS)
        selection_complete = True
        return 0.125, []

    monkeypatch.setattr(multi, "_paired_predictions", pair)
    monkeypatch.setattr(v1, "_select_lambda", select)
    monkeypatch.setattr(v1, "_fuse_pair", lambda _pair, value: (np.asarray([value]),))
    result = multi._recompute_expected_fold(source, "outer_01")
    assert result[0] == 0.125
    assert load_order == [*v1.INNER_FOLD_IDS, None]


def test_three_seed_aggregate_reports_each_seed_mean_and_sample_sd() -> None:
    aggregates = {
        13: _seed_aggregate(13, 0.30, 0.28),
        42: _seed_aggregate(42, 0.40, 0.35),
        73: _seed_aggregate(73, 0.50, 0.44),
    }
    inputs = _frozen_inputs(aggregates[13])
    seed_bindings = {
        seed: (aggregates[seed], v1.FileBinding(f"seed_{seed}/aggregate.json", "0" * 64, 1))
        for seed in multi.ALLOWED_NEW_SEEDS
    }
    payload = multi._multiseed_aggregate_payload(
        inputs=inputs,
        protocol_binding=v1.FileBinding("protocol.json", "0" * 64, 1),
        seed_aggregates=seed_bindings,
    )
    assert list(payload["per_seed"]) == ["13", "42", "73"]
    frame = payload["mean_and_sample_SD"]["selected"]["frame_mAP"]
    assert frame["mean"] == pytest.approx(0.4)
    assert frame["sample_SD"] == pytest.approx(0.1)
    delta = payload["mean_and_sample_SD"]["selected_minus_matched_lambda0"][
        "frame_mAP"
    ]
    assert delta["mean"] == pytest.approx((0.02 + 0.05 + 0.06) / 3)
    assert payload["significance_tests_run"] is False
    assert payload["significance_claims_permitted"] is False
    assert payload["hypothesis_boundary"] == "does_not_change_H1_or_test_H2"


def test_publish_is_create_once_and_returns_postrename_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "published"
    inputs = _frozen_inputs(_seed_aggregate(13, 0.3, 0.28))
    audit_calls: list[Path] = []

    monkeypatch.setattr(multi, "_load_frozen_inputs", lambda: inputs)
    monkeypatch.setattr(multi, "_protocol_payload", lambda _inputs: {"protocol": 2})

    def build(
        stage: Path,
        source: object,
        _protocol_binding: v1.FileBinding,
    ) -> tuple[dict[str, object], v1.FileBinding]:
        seed = source.seed
        path = stage / f"seed_{seed}" / multi.SEED_AGGREGATE_FILENAME
        v1._write_json_exclusive(path, {"seed": seed})
        return {"seed": seed}, v1._binding(path, stage)

    fake_sources = {seed: SimpleNamespace(seed=seed) for seed in multi.ALLOWED_NEW_SEEDS}
    object.__setattr__(inputs, "sources", fake_sources)
    monkeypatch.setattr(multi, "_build_seed", build)
    monkeypatch.setattr(
        multi,
        "_multiseed_aggregate_payload",
        lambda **_kwargs: {"artifact_kind": multi.MULTISEED_AGGREGATE_KIND},
    )

    def audit(path: str | Path, *, validated_inputs: object) -> dict[str, object]:
        assert validated_inputs is inputs
        root = Path(path).resolve()
        audit_calls.append(root)
        return {"audit_passed": True, "output_root": str(root)}

    monkeypatch.setattr(multi, "_audit_multiseed", audit)
    aggregate, receipt = multi._publish_with_audit(output)
    assert aggregate == output / multi.MULTISEED_AGGREGATE_FILENAME
    assert receipt["audit_passed"] is True
    assert len(audit_calls) == 2
    assert audit_calls[1] == output.resolve()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        multi._publish_with_audit(output)


def test_postrename_failure_rolls_back_only_owned_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "rollback"
    inputs = _frozen_inputs(_seed_aggregate(13, 0.3, 0.28))
    object.__setattr__(
        inputs,
        "sources",
        {seed: SimpleNamespace(seed=seed) for seed in multi.ALLOWED_NEW_SEEDS},
    )
    monkeypatch.setattr(multi, "_load_frozen_inputs", lambda: inputs)
    monkeypatch.setattr(multi, "_protocol_payload", lambda _inputs: {})

    def build(
        stage: Path,
        source: object,
        _protocol_binding: v1.FileBinding,
    ) -> tuple[dict[str, object], v1.FileBinding]:
        path = stage / f"seed_{source.seed}" / multi.SEED_AGGREGATE_FILENAME
        v1._write_json_exclusive(path, {})
        return {}, v1._binding(path, stage)

    monkeypatch.setattr(multi, "_build_seed", build)
    monkeypatch.setattr(multi, "_multiseed_aggregate_payload", lambda **_kwargs: {})
    calls = 0

    def fail_second(path: str | Path, *, validated_inputs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise multi.ResidualLateFusionMultiseedError("injected post-rename failure")
        return {"audit_passed": True}

    monkeypatch.setattr(multi, "_audit_multiseed", fail_second)
    with pytest.raises(multi.ResidualLateFusionMultiseedError, match="injected"):
        multi._publish_with_audit(output)
    assert not output.exists()
    assert not list(tmp_path.glob(".rollback.*.stage"))


def test_postrename_replacement_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "replacement"
    moved_owned = tmp_path / "moved_owned"
    inputs = _frozen_inputs(_seed_aggregate(13, 0.3, 0.28))
    object.__setattr__(
        inputs,
        "sources",
        {seed: SimpleNamespace(seed=seed) for seed in multi.ALLOWED_NEW_SEEDS},
    )
    monkeypatch.setattr(multi, "_load_frozen_inputs", lambda: inputs)
    monkeypatch.setattr(multi, "_protocol_payload", lambda _inputs: {})

    def build(
        stage: Path,
        source: object,
        _protocol_binding: v1.FileBinding,
    ) -> tuple[dict[str, object], v1.FileBinding]:
        path = stage / f"seed_{source.seed}" / multi.SEED_AGGREGATE_FILENAME
        v1._write_json_exclusive(path, {})
        return {}, v1._binding(path, stage)

    monkeypatch.setattr(multi, "_build_seed", build)
    monkeypatch.setattr(multi, "_multiseed_aggregate_payload", lambda **_kwargs: {})
    calls = 0

    def replace_second(path: str | Path, *, validated_inputs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            Path(path).rename(moved_owned)
            Path(path).mkdir()
            (Path(path) / "foreign.txt").write_text("foreign", encoding="utf-8")
            raise multi.ResidualLateFusionMultiseedError("injected replacement")
        return {"audit_passed": True}

    monkeypatch.setattr(multi, "_audit_multiseed", replace_second)
    with pytest.raises(multi.ResidualLateFusionMultiseedError, match="identity changed"):
        multi._publish_with_audit(output)
    assert (output / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert moved_owned.is_dir()


def test_training_aggregate_hash_and_seed_are_both_enforced(tmp_path: Path) -> None:
    aggregate = {
        "artifact_kind": "vap_thesis_primary_matrix_aggregate",
        "format_version": 1,
        "result_role": v1.thesis_backend.RESULT_ROLE,
        "data_kind": "real",
        "production_eligible": False,
        "counted_eligible": False,
        "analysis_variant": "primary",
        "included_session_ids": list(v1.EXACT_EIGHT_SESSION_IDS),
        "group_ids": list(v1.EXACT_FOUR_GROUP_IDS),
        "corpus_exclusions": dict(v1.EXACT_CORPUS_EXCLUSIONS),
        "seed": 42,
        "aggregation_source": "saved_outer_probabilities_only_no_model_inference",
        "run_manifest_sha256": v1.SOURCE_RUN_MANIFEST_SHA256,
        "conditions": {condition: {} for condition in v1.MATRIX_CONDITIONS},
    }
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(aggregate) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = multi.FrozenSeedContract(
        seed=42,
        matrix_root=tmp_path,
        matrix_index_sha256="0" * 64,
        training_aggregate=path,
        training_aggregate_sha256=digest,
        protocol_root=tmp_path,
        run_manifest_raw_sha256="0" * 64,
    )
    _, binding, observed = multi._validate_training_aggregate(contract)
    assert binding.sha256 == digest
    assert observed["seed"] == 42
    aggregate["seed"] = 73
    path.write_text(json.dumps(aggregate) + "\n", encoding="utf-8")
    changed = hashlib.sha256(path.read_bytes()).hexdigest()
    wrong_seed_contract = multi.FrozenSeedContract(
        **{**contract.__dict__, "training_aggregate_sha256": changed}
    )
    with pytest.raises(multi.ResidualLateFusionMultiseedError, match="differs at seed"):
        multi._validate_training_aggregate(wrong_seed_contract)


def test_real_fixture_saved_probability_to_three_seed_audit_and_tamper_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, pairs = _artifact_fixture_inputs(tmp_path)
    output = tmp_path / "audited_multiseed"
    monkeypatch.setattr(multi, "_load_frozen_inputs", lambda: inputs)
    monkeypatch.setattr(multi, "_revalidate_frozen_inputs", lambda _inputs: None)

    def paired(
        source: multi.SeedSource,
        outer_fold_id: str,
        inner_fold_id: str | None,
    ) -> v1.PairedPredictions:
        return pairs[(source.seed, outer_fold_id, inner_fold_id)]

    monkeypatch.setattr(multi, "_paired_predictions", paired)
    monkeypatch.setattr(
        v1.thesis_backend,
        "_sessions_from_prediction",
        lambda payload: tuple(payload["sessions"]),
    )

    def score(probabilities: tuple[np.ndarray, ...]) -> float:
        packed = np.concatenate(
            [np.asarray(value, dtype=np.float64).ravel() for value in probabilities]
        )
        return float(packed.mean())

    monkeypatch.setattr(
        v1,
        "_participant_macro_strict_map",
        lambda _payload, probabilities: score(probabilities),
    )
    monkeypatch.setattr(
        v1,
        "_metrics",
        lambda _payload, probabilities: _metrics(score(probabilities)),
    )
    monkeypatch.setattr(
        v1.thesis_backend,
        "_metrics",
        lambda _sessions, probabilities: _metrics(score(probabilities)),
    )

    aggregate_path, receipt = multi._publish_with_audit(output)
    assert aggregate_path == output / multi.MULTISEED_AGGREGATE_FILENAME
    assert receipt["audit_passed"] is True
    assert multi.audit_residual_late_fusion_multiseed(output)["audit_passed"] is True
    aggregate = v1._read_json_strict(aggregate_path)
    assert aggregate["seed_ids"] == [13, 42, 73]
    assert aggregate["significance_tests_run"] is False
    assert set(aggregate["per_seed"]) == {"13", "42", "73"}

    fused_path = output / "seed_42" / "outer_01" / multi.FUSED_PROBABILITIES_FILENAME
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
    with pytest.raises(
        multi.ResidualLateFusionMultiseedError,
        match="saved fused probabilities differ from formula replay",
    ):
        multi.audit_residual_late_fusion_multiseed(output)
