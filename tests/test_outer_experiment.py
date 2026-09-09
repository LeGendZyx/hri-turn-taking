from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hri_turn_taking.data.assembly import ReadinessReport
from hri_turn_taking.vap import outer_experiment as outer
from hri_turn_taking.vap.dataset import SessionTensors
from hri_turn_taking.vap.feature_sets import FEATURE_SET_NAMES, active_feature_columns
from hri_turn_taking.vap.outer_experiment import (
    BackendContractSmokeBackend,
    CalibrationReceipt,
    FinalFitReceipt,
    InnerFitReceipt,
    OuterCondition,
    OuterEvaluationReceipt,
    OuterExperimentGateError,
    PreparedOuterCondition,
    build_outer_run_manifest,
    build_outer_split_manifest,
    prepare_outer_condition,
    prepared_outer_authority_inputs,
    publish_outer_protocol,
    run_outer_unit,
    validate_outer_run_manifest,
    validate_outer_split_manifest,
)

SESSION_TO_GROUP = dict(outer.PRIMARY_SESSION_TO_GROUP)
CONDITIONS = (
    OuterCondition("A", "audio_only"),
    OuterCondition("H", "head_pose_only"),
    OuterCondition("A+H", "audio_plus_head_pose"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_source(
    tmp_path: Path,
) -> tuple[Path, list[dict[str, object]], tuple[dict[str, object], ...]]:
    source = tmp_path / "source"
    source.mkdir()
    bindings: list[dict[str, object]] = []
    artifacts = []
    for index, session_id in enumerate(outer.PRIMARY_D3_SESSION_IDS):
        data_path = source / f"{session_id}.csv"
        data_path.write_text(f"immutable-{session_id}\n", encoding="utf-8")
        frozen = outer.D3_FROZEN_VALUES[session_id]
        provenance_path = source / f"{session_id}.provenance.json"
        provenance_path.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "group_id": SESSION_TO_GROUP[session_id],
                    "offset_quality": {
                        "stream": "6cam",
                        "offset_s": frozen["offset_s"],
                        "status": frozen["status"],
                        "self_check_ok": frozen["self_check_ok"],
                        "error_bound_ms": frozen["error_bound_ms"],
                    },
                }
            ),
            encoding="utf-8",
        )
        bindings.extend(
            [
                {
                    "artifact_id": f"source_{index:03d}_data",
                    "role": "assembled_csv",
                    "path": str(data_path.resolve()),
                    "sha256": _sha256(data_path),
                    "size_bytes": data_path.stat().st_size,
                },
                {
                    "artifact_id": f"source_{index:03d}_provenance",
                    "role": "assembled_provenance",
                    "path": str(provenance_path.resolve()),
                    "sha256": _sha256(provenance_path),
                    "size_bytes": provenance_path.stat().st_size,
                },
            ]
        )
        artifacts.append(
            {
                "data_path": str(data_path.resolve()),
                "data_sha256": _sha256(data_path),
                "provenance_path": str(provenance_path.resolve()),
                "provenance_sha256": _sha256(provenance_path),
                "artifact_kind": "vap_real_assembled_session",
                "format_version": 2,
                "identity_authority": "sealed_hpe_transcript_crosswalk",
                "identity_inventory_sha256": f"{index:064x}",
            }
        )
    return source.resolve(), bindings, tuple(artifacts)


def _source_report(
    source: Path,
    feature_set: str,
    artifacts: tuple[dict[str, object], ...],
) -> ReadinessReport:
    return ReadinessReport(
        source=str(source),
        data_kind="real",
        files=tuple(str(item["data_path"]) for item in artifacts),
        n_rows=40,
        n_sessions=10,
        n_groups=4,
        session_ids=tuple(sorted(SESSION_TO_GROUP)),
        group_ids=tuple(sorted(set(SESSION_TO_GROUP.values()))),
        source_artifacts=artifacts,
        structural_ready=True,
        target_ready=True,
        feature_set=feature_set,
        selected_feature_ready=True,
    )


def _protocol_payloads(
    tmp_path: Path, *, conditions: tuple[OuterCondition, ...] = CONDITIONS
) -> tuple[
    dict[str, object],
    dict[str, object],
    Path,
    ReadinessReport,
]:
    source, bindings, artifacts = _create_source(tmp_path)
    first_report = _source_report(source, conditions[0].feature_set, artifacts)
    d3_records = outer._d3_records_from_report(first_report)
    split = build_outer_split_manifest(
        SESSION_TO_GROUP,
        source_bindings=bindings,
        conditions=conditions,
        d3_records=d3_records,
    )
    readiness = {
        condition.condition_id: outer._readiness_binding(
            _source_report(source, condition.feature_set, artifacts)
        )
        for condition in conditions
    }
    run = build_outer_run_manifest(
        split,
        source_path=source,
        max_epochs=9,
        trainer_config={"aux_vad_weight": 0.0, "batch_size": 2},
        model_config={"d_model": 8, "n_layers": 1},
        readiness_by_condition=readiness,
    )
    return split, run, source, first_report


def _sessions() -> list[SessionTensors]:
    result = []
    group_value = {"03_20": 1.0, "03_26": 2.0, "05_14": 3.0, "05_15": 1000.0}
    for index, (session_id, group_id) in enumerate(sorted(SESSION_TO_GROUP.items())):
        features = np.full((4, 33), group_value[group_id], dtype=np.float32)
        labels = np.zeros((4, 4, 4), dtype=np.float32)
        labels[:, :, index % 4] = 1.0
        result.append(
            SessionTensors(
                session_id=session_id,
                features=features,
                labels=labels,
                vad=np.zeros((4, 4), dtype=np.float32),
                valid=np.ones(4, dtype=bool),
                present=np.ones(4, dtype=np.float32),
                group_id=group_id,
            )
        )
    return result


class RecordingBackend(BackendContractSmokeBackend):
    def __init__(self) -> None:
        self.inner_requests = []
        self.calibration_requests = []
        self.final_requests = []
        self.evaluation_requests = []
        self.bad_receipt: str | None = None
        self.mutate_checkpoint = False
        self.mutate_source: Path | None = None
        self.mutate_oof = False
        self.mutate_calibration = False
        self.mutate_inputs = False
        self.observed_input_maxima: list[float] = []

    @staticmethod
    def _write(path: Path, value: str) -> Path:
        path.write_text(value, encoding="utf-8")
        return path

    def fit_inner(self, request):
        self.inner_requests.append(request)
        self.observed_input_maxima.append(
            max(float(item.features.max()) for item in request.train_sessions)
        )
        if self.mutate_inputs:
            request.train_sessions[0].features[:] = 999999.0
            request.validation_sessions[0].labels[:] = 1.0
            request.normalizer.global_mean[:] = 12345.0
            request.pos_weight[:] = 54321.0
        best_epoch = {"inner_01": 2, "inner_02": 7, "inner_03": 4}[request.inner_fold_id]
        train_ids = tuple(item.session_id for item in request.train_sessions)
        val_ids = tuple(item.session_id for item in request.validation_sessions)
        receipt = InnerFitReceipt(
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            inner_fold_id=request.inner_fold_id,
            seed=request.seed,
            best_epoch=best_epoch,
            train_session_ids=train_ids,
            validation_session_ids=val_ids,
            normalizer_fit_session_ids=train_ids,
            pos_weight_fit_session_ids=train_ids,
            checkpoint_path=self._write(
                request.output_dir / outer.INNER_CHECKPOINT_FILENAME,
                "inner-checkpoint",
            ),
            oof_predictions_path=self._write(
                request.output_dir / outer.INNER_OOF_FILENAME,
                "oof",
            ),
        )
        if self.bad_receipt == "inner_seed":
            return InnerFitReceipt(**{**receipt.__dict__, "seed": 999})
        if self.bad_receipt == "inner_outer_fold":
            return InnerFitReceipt(**{**receipt.__dict__, "outer_fold_id": "outer_99"})
        if self.bad_receipt == "bool_epoch":
            return InnerFitReceipt(**{**receipt.__dict__, "best_epoch": True})
        if self.bad_receipt == "inner_role_swap":
            return InnerFitReceipt(
                **{
                    **receipt.__dict__,
                    "checkpoint_path": receipt.oof_predictions_path,
                    "oof_predictions_path": receipt.checkpoint_path,
                }
            )
        if self.bad_receipt == "inner_outside_fold":
            shared = self._write(request.output_dir.parent / "shared.artifact", "shared")
            return InnerFitReceipt(**{**receipt.__dict__, "checkpoint_path": shared})
        return receipt

    def calibrate_inner_oof(self, request):
        self.calibration_requests.append(request)
        paths = request.oof_prediction_paths
        if self.bad_receipt == "drop_oof":
            paths = paths[:-1]
        if self.mutate_oof:
            request.oof_prediction_paths[0].write_text("changed-oof", encoding="utf-8")
        return CalibrationReceipt(
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            inner_fold_ids=request.inner_fold_ids,
            oof_prediction_paths=paths,
            calibration_path=self._write(
                request.output_dir / outer.CALIBRATION_FILENAME,
                "{}",
            ),
        )

    def fit_final(self, request):
        self.final_requests.append(request)
        if self.mutate_inputs:
            request.train_sessions[0].vad[:] = 1.0
            request.normalizer.participant_mean[:] = -999.0
            request.pos_weight[:] = -999.0
        train_ids = tuple(item.session_id for item in request.train_sessions)
        return FinalFitReceipt(
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            fixed_epochs=request.fixed_epochs,
            train_session_ids=train_ids,
            normalizer_fit_session_ids=train_ids,
            pos_weight_fit_session_ids=train_ids,
            initialized_from_scratch=self.bad_receipt != "warm_start",
            checkpoint_path=self._write(
                request.output_dir / outer.FINAL_CHECKPOINT_FILENAME,
                "final-checkpoint",
            ),
        )

    def evaluate_outer(self, request):
        self.evaluation_requests.append(request)
        if self.mutate_inputs:
            request.test_sessions[0].features[:] = -999.0
        if self.mutate_checkpoint:
            request.checkpoint_path.write_text("changed", encoding="utf-8")
        if self.mutate_calibration:
            request.calibration_path.write_text("changed-calibration", encoding="utf-8")
        if self.mutate_source is not None:
            target = next(self.mutate_source.glob("*.csv"))
            target.write_text("changed-source", encoding="utf-8")
        test_ids = tuple(item.session_id for item in request.test_sessions)
        count = True if self.bad_receipt == "bool_eval_count" else 1
        return OuterEvaluationReceipt(
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            checkpoint_sha256=request.checkpoint_sha256,
            evaluation_session_ids=test_ids,
            evaluation_count=count,
            predictions_path=self._write(
                request.output_dir / outer.OUTER_PREDICTIONS_FILENAME,
                "prediction",
            ),
            metrics_path=self._write(
                request.output_dir / outer.OUTER_METRICS_FILENAME,
                "{}",
            ),
        )


class SelfClaimedProductionBackend:
    backend_id = "production"
    backend_version = 999
    counted_eligible = True

    def __init__(self) -> None:
        self.delegate = RecordingBackend()

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class HardlinkInnerArtifactsBackend(RecordingBackend):
    def fit_inner(self, request):
        receipt = super().fit_inner(request)
        receipt.oof_predictions_path.unlink()
        os.link(receipt.checkpoint_path, receipt.oof_predictions_path)
        return receipt


def _published_protocol(tmp_path: Path):
    split, run, source, report = _protocol_payloads(tmp_path)
    bundle = publish_outer_protocol(tmp_path / "protocol", split, run)
    return bundle, source, report


def _patch_loader(monkeypatch, sessions, report):
    def load(_source, condition, _run):
        return sessions, ReadinessReport(
            **{
                **report.__dict__,
                "feature_set": condition.feature_set,
            }
        )

    monkeypatch.setattr(outer, "_load_condition_sessions", load)


def _run_smoke(
    bundle,
    source: Path,
    output: Path,
    backend,
    *,
    analysis_variant: str = "primary",
    outer_fold_id: str = "outer_04",
) -> Path:
    return run_outer_unit(
        bundle.directory,
        output,
        source=source,
        analysis_variant=analysis_variant,
        condition_id="A",
        outer_fold_id=outer_fold_id,
        seed=13,
        backend=backend,
    )


def test_split_has_both_executable_four_group_variants_and_frozen_d3(tmp_path: Path) -> None:
    split, _, _, _ = _protocol_payloads(tmp_path)
    assert set(split["analysis_variants"]) == {"primary", "d3_le80"}
    assert split["seeds"] == [13, 42, 73]
    assert split["aux_vad_weight"] == 0.0
    expected_membership = {
        "primary": set(outer.PRIMARY_D3_SESSION_IDS),
        "d3_le80": set(outer.PAIRED_D3_80MS_SESSION_IDS),
    }
    for name, expected_sessions in expected_membership.items():
        variant = split["analysis_variants"][name]
        assert set(variant["session_to_group"]) == expected_sessions
        assert len(variant["outer_folds"]) == 4
        assert {fold["outer_test_group_ids"][0] for fold in variant["outer_folds"]} == {
            "03_20",
            "03_26",
            "05_14",
            "05_15",
        }
        for fold in variant["outer_folds"]:
            assert len(fold["inner_folds"]) == 3
    assert {
        session_id: split["d3_records"][session_id]["offset_s"]
        for session_id in outer.PRIMARY_D3_SESSION_IDS
    } == {session_id: value["offset_s"] for session_id, value in outer.D3_FROZEN_VALUES.items()}


def test_deadline_eight_profile_is_exact_and_records_owner_pending_exclusions(
    tmp_path: Path,
) -> None:
    source, bindings, artifacts = _create_source(tmp_path)
    included = set(outer.DEADLINE_SESSION_TO_GROUP)
    selected_artifacts = tuple(
        artifact for artifact in artifacts if Path(str(artifact["data_path"])).stem in included
    )
    selected_paths = {
        str(artifact[key])
        for artifact in selected_artifacts
        for key in ("data_path", "provenance_path")
    }
    selected_bindings = [binding for binding in bindings if binding["path"] in selected_paths]
    report = ReadinessReport(
        source=str(source),
        data_kind="real",
        files=tuple(str(item["data_path"]) for item in selected_artifacts),
        n_rows=outer.DEADLINE_EXPECTED_N_ROWS,
        n_sessions=8,
        n_groups=4,
        session_ids=tuple(sorted(outer.DEADLINE_SESSION_TO_GROUP)),
        group_ids=tuple(sorted(set(outer.DEADLINE_SESSION_TO_GROUP.values()))),
        source_artifacts=selected_artifacts,
        structural_ready=True,
        target_ready=True,
        feature_set="audio_only",
        selected_feature_ready=True,
    )
    split = build_outer_split_manifest(
        dict(outer.DEADLINE_SESSION_TO_GROUP),
        source_bindings=selected_bindings,
        conditions=(CONDITIONS[0],),
        d3_records=outer._d3_records_from_report(report),
    )
    assert split["protocol_id"] == outer.DEADLINE_PROTOCOL_ID
    assert split["format_version"] == outer.DEADLINE_MANIFEST_FORMAT_VERSION
    assert split["corpus_exclusions"] == outer.DEADLINE_CORPUS_EXCLUSIONS
    assert set(split["analysis_variants"]["primary"]["session_to_group"]) == included
    assert set(split["d3_records"]) == included
    assert set(split["d3_sensitivity"]["primary"]["session_ids"]) == included
    assert set(split["analysis_variants"]["d3_le80"]["session_to_group"]) == (
        set(outer.PAIRED_D3_80MS_SESSION_IDS) & included
    )
    assert len(split["analysis_variants"]["primary"]["outer_folds"]) == 4
    fold_05_15 = next(
        fold
        for fold in split["analysis_variants"]["primary"]["outer_folds"]
        if fold["outer_test_group_ids"] == ["05_15"]
    )
    assert fold_05_15["outer_test_session_ids"] == [
        "05_15__2",
        "05_15__3",
        "05_15__6",
    ]

    readiness = {"A": outer._readiness_binding(report)}
    run = build_outer_run_manifest(
        split,
        source_path=source,
        max_epochs=1,
        trainer_config={"aux_vad_weight": 0.0},
        model_config={},
        readiness_by_condition=readiness,
    )
    assert run["corpus_exclusions"] == outer.DEADLINE_CORPUS_EXCLUSIONS
    changed = copy.deepcopy(split)
    changed["corpus_exclusions"]["05_15__4"] = "silent mutation"
    with pytest.raises(OuterExperimentGateError, match="corpus exclusions"):
        validate_outer_split_manifest(changed)


def test_deadline_profile_rejects_every_other_session_inventory(tmp_path: Path) -> None:
    split, _, _, _ = _protocol_payloads(tmp_path)
    arbitrary = dict(outer.PRIMARY_SESSION_TO_GROUP)
    arbitrary.pop("05_14__4")
    d3 = dict(split["d3_records"])
    d3.pop("05_14__4")
    with pytest.raises(OuterExperimentGateError, match="neither the frozen ten-scene"):
        build_outer_split_manifest(
            arbitrary,
            source_bindings=split["source_bindings"],
            conditions=CONDITIONS,
            d3_records=d3,
        )


def test_head_pose_sensitivity_is_derived_for_every_registered_feature_set(
    tmp_path: Path,
) -> None:
    conditions = tuple(
        OuterCondition(f"C{index}", feature_set)
        for index, feature_set in enumerate(FEATURE_SET_NAMES)
    )
    split, _, _, _ = _protocol_payloads(tmp_path, conditions=conditions)
    expected_ids = [
        condition.condition_id
        for condition in conditions
        if any(
            column.endswith("_head_yaw_deg")
            for column in active_feature_columns(condition.feature_set)
        )
    ]
    sensitivity = split["d3_sensitivity"]["pose_time_shift_metric_range"]
    assert sensitivity["condition_ids"] == expected_ids
    assert sensitivity["feature_sets"] == sorted(
        {
            condition.feature_set
            for condition in conditions
            if condition.condition_id in expected_ids
        }
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "outer_leak",
        "inner_leak",
        "bool_version",
        "seed_float",
        "variant_membership",
        "d3_offset",
        "d3_self_check",
        "extra_field",
    ],
)
def test_split_rejects_leakage_types_variant_and_d3_mutation(tmp_path: Path, mutation: str) -> None:
    split, _, _, _ = _protocol_payloads(tmp_path)
    changed = copy.deepcopy(split)
    primary = changed["analysis_variants"]["primary"]
    if mutation == "outer_leak":
        primary["outer_folds"][0]["outer_train_group_ids"].append(
            primary["outer_folds"][0]["outer_test_group_ids"][0]
        )
    elif mutation == "inner_leak":
        inner = primary["outer_folds"][0]["inner_folds"][0]
        inner["train_group_ids"].append(inner["validation_group_ids"][0])
    elif mutation == "bool_version":
        changed["format_version"] = True
    elif mutation == "seed_float":
        changed["seeds"] = [13.0, 42, 73]
    elif mutation == "variant_membership":
        changed["analysis_variants"]["d3_le80"]["session_to_group"]["05_15__4"] = "05_15"
    elif mutation == "d3_offset":
        changed["d3_records"]["05_15__2"]["offset_s"] = -999.0
    elif mutation == "d3_self_check":
        changed["d3_records"]["05_15__2"]["self_check_ok"] = 1
    else:
        changed["unexpected"] = "field"
    with pytest.raises(OuterExperimentGateError):
        validate_outer_split_manifest(changed)


def test_run_manifest_has_no_production_backend_and_rejects_self_registration(
    tmp_path: Path,
) -> None:
    split, run, source, _ = _protocol_payloads(tmp_path)
    assert run["backend_policy"]["production_backend"] is None
    assert run["backend_policy"]["production_status"] == "not_registered_blocker"
    assert run["implementation_status"]["environment_capture"] == "not_implemented_blocker"
    assert run["implementation_status"]["run_index"] == "not_implemented_blocker"

    with pytest.raises(OuterExperimentGateError, match="code-owned registry"):
        build_outer_run_manifest(
            split,
            source_path=source,
            max_epochs=9,
            trainer_config={"aux_vad_weight": 0.0},
            model_config={},
            production_backend={
                "backend_id": "claimed-production",
                "backend_version": 1,
                "result_role": "counted_outer_test",
                "counted_eligible": True,
            },
            readiness_by_condition=run["readiness_by_condition"],
        )

    changed = copy.deepcopy(run)
    changed["backend_policy"]["production_status"] = "registered"
    with pytest.raises(OuterExperimentGateError, match="code-owned registry"):
        validate_outer_run_manifest(changed, split)

    changed = copy.deepcopy(run)
    changed["readiness_by_condition"]["A"]["d3_records"]["05_15__2"]["error_bound_ms"] = 1.0
    with pytest.raises(OuterExperimentGateError, match="readiness D3"):
        validate_outer_run_manifest(changed, split)


def test_smoke_backend_can_never_claim_counted_result(tmp_path: Path, monkeypatch) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    _patch_loader(monkeypatch, _sessions(), report)
    result = _run_smoke(bundle, source, tmp_path / "result", RecordingBackend())
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["format_version"] == 2
    assert payload["result_role"] == "backend_contract_smoke"
    assert payload["counted_experiment_complete"] is False
    assert "backend_contract_smoke_not_counted" in payload["remaining_counted_gates"]
    assert "environment_capture" in payload["remaining_counted_gates"]
    assert "run_index" in payload["remaining_counted_gates"]


def test_default_outer_unit_retains_three_fresh_loader_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    sessions = _sessions()
    calls = []

    def load(_source, condition, _run):
        calls.append(condition.condition_id)
        return sessions, ReadinessReport(
            **{**report.__dict__, "feature_set": condition.feature_set}
        )

    monkeypatch.setattr(outer, "_load_condition_sessions", load)
    _run_smoke(bundle, source, tmp_path / "fresh", RecordingBackend())
    assert calls == ["A", "A", "A"]


def test_factory_prepared_condition_replays_once_and_reuses_both_validators(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    sessions = _sessions()
    frame = object()
    material_calls = []

    def material(_source, condition, _run):
        material_calls.append(condition.condition_id)
        return (
            frame,
            sessions,
            ReadinessReport(**{**report.__dict__, "feature_set": condition.feature_set}),
        )

    monkeypatch.setattr(outer, "_load_condition_material", material)
    prepared = prepare_outer_condition(
        bundle.directory,
        source=source,
        condition_id="A",
    )
    assert type(prepared) is PreparedOuterCondition
    assert material_calls == ["A"]

    monkeypatch.setattr(outer, "_sessions_from_condition_frame", lambda *args: sessions)
    authority_frame, authority_report = prepared_outer_authority_inputs(prepared)
    assert authority_frame is frame
    assert authority_report.feature_set == "audio_only"

    def forbidden_fresh_loader(*args):
        raise AssertionError("prepared unit must not start another full source replay")

    monkeypatch.setattr(outer, "_load_condition_sessions", forbidden_fresh_loader)
    real_validate = outer._validate_result_directory
    validation_calls = []

    def recording_validate(directory, **kwargs):
        validation_calls.append((Path(directory), kwargs.get("prepared")))
        return real_validate(directory, **kwargs)

    monkeypatch.setattr(outer, "_validate_result_directory", recording_validate)
    result = run_outer_unit(
        bundle.directory,
        tmp_path / "prepared",
        source=source,
        analysis_variant="primary",
        condition_id="A",
        outer_fold_id="outer_04",
        seed=13,
        backend=RecordingBackend(),
        prepared=prepared,
    )
    assert result.is_file()
    assert material_calls == ["A"]
    assert len(validation_calls) == 2
    assert all(item[1] is prepared for item in validation_calls)
    assert validation_calls[0][0].name.endswith(".stage")
    assert validation_calls[1][0] == result.parent


def test_prepared_condition_rejects_manual_copy_injection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    sessions = _sessions()
    monkeypatch.setattr(
        outer,
        "_load_condition_material",
        lambda *_args: (object(), sessions, report),
    )
    prepared = prepare_outer_condition(
        bundle.directory,
        source=source,
        condition_id="A",
    )
    forged = copy.copy(prepared)
    with pytest.raises(OuterExperimentGateError, match="factory-issued"):
        run_outer_unit(
            bundle.directory,
            tmp_path / "forged",
            source=source,
            analysis_variant="primary",
            condition_id="A",
            outer_fold_id="outer_04",
            seed=13,
            backend=RecordingBackend(),
            prepared=forged,
        )
    assert not (tmp_path / "forged").exists()


def test_prepared_unit_keeps_live_source_sha_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    sessions = _sessions()
    monkeypatch.setattr(
        outer,
        "_load_condition_material",
        lambda *_args: (object(), sessions, report),
    )
    prepared = prepare_outer_condition(
        bundle.directory,
        source=source,
        condition_id="A",
    )
    next(source.glob("*.csv")).write_text("changed-after-prepare", encoding="utf-8")
    with pytest.raises(OuterExperimentGateError, match="bound source bytes changed"):
        run_outer_unit(
            bundle.directory,
            tmp_path / "changed-source",
            source=source,
            analysis_variant="primary",
            condition_id="A",
            outer_fold_id="outer_04",
            seed=13,
            backend=RecordingBackend(),
            prepared=prepared,
        )
    assert not (tmp_path / "changed-source").exists()


def test_self_claimed_production_object_is_rejected_before_output(
    tmp_path: Path, monkeypatch
) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    _patch_loader(monkeypatch, _sessions(), report)
    output = tmp_path / "result"
    with pytest.raises(OuterExperimentGateError, match="code-owned registry"):
        _run_smoke(bundle, source, output, SelfClaimedProductionBackend())
    assert not output.exists()


def test_primary_runner_selection_stats_and_outer_scope(tmp_path: Path, monkeypatch) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    sessions = _sessions()
    original_hash = outer._sessions_sha256(sessions)
    _patch_loader(monkeypatch, sessions, report)
    backend = RecordingBackend()
    result = _run_smoke(bundle, source, tmp_path / "result", backend)

    assert len(backend.inner_requests) == 3
    assert len(backend.calibration_requests) == 1
    assert len(backend.final_requests) == 1
    assert len(backend.evaluation_requests) == 1
    outer_ids = {item.session_id for item in backend.evaluation_requests[0].test_sessions}
    assert outer_ids == {
        "05_15__2",
        "05_15__3",
        "05_15__4",
        "05_15__5",
        "05_15__6",
    }
    final = backend.final_requests[0]
    assert final.fixed_epochs == 4
    assert outer_ids.isdisjoint(item.session_id for item in final.train_sessions)
    np.testing.assert_allclose(final.normalizer.global_mean, np.full(5, 2.4))
    assert outer._sessions_sha256(sessions) == original_hash

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["analysis_variant"] == "primary"
    assert payload["selection"]["selected_fixed_epoch"] == 4
    assert payload["selection"]["calibration_oof_session_ids"] == [
        session_id
        for inner in bundle_split(bundle)["analysis_variants"]["primary"]["outer_folds"][3][
            "inner_folds"
        ]
        for session_id in inner["validation_session_ids"]
    ]


def bundle_split(bundle) -> dict[str, object]:
    return json.loads(bundle.split_manifest_path.read_text(encoding="utf-8"))


def test_d3_le80_runner_executes_six_scene_four_group_split(tmp_path: Path, monkeypatch) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    _patch_loader(monkeypatch, _sessions(), report)
    backend = RecordingBackend()
    result = _run_smoke(
        bundle,
        source,
        tmp_path / "result",
        backend,
        analysis_variant="d3_le80",
    )
    test_ids = {item.session_id for item in backend.evaluation_requests[0].test_sessions}
    assert test_ids == {"05_15__2", "05_15__3", "05_15__5"}
    all_seen = test_ids | {
        item.session_id for request in backend.final_requests for item in request.train_sessions
    }
    assert all_seen == set(outer.PAIRED_D3_80MS_SESSION_IDS)
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["analysis_variant"] == "d3_le80"


def test_backend_input_mutation_is_isolated_from_master_and_statistics(
    tmp_path: Path, monkeypatch
) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    sessions = _sessions()
    master_hash = outer._sessions_sha256(sessions)
    _patch_loader(monkeypatch, sessions, report)
    backend = RecordingBackend()
    backend.mutate_inputs = True
    result = _run_smoke(bundle, source, tmp_path / "result", backend)
    assert outer._sessions_sha256(sessions) == master_hash
    assert all(value < 999999.0 for value in backend.observed_input_maxima)
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["master_sessions_sha256_before"] == master_hash
    assert payload["master_sessions_sha256_after"] == master_hash
    np.testing.assert_allclose(
        payload["statistics"]["final"]["normalizer"]["global_mean"],
        [2.4] * 5,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_field",
        "statistics_scope",
        "statistics_shape",
        "statistics_value",
        "statistics_nan",
        "statistics_bool",
        "selection_epoch",
        "calibration_scope",
        "artifact_fold_swap",
        "artifact_role_reuse",
        "artifact_role_swap",
        "master_hash",
        "counted_claim",
    ],
)
def test_result_v2_rejects_manifest_scope_statistics_and_role_tampering(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    _patch_loader(monkeypatch, _sessions(), report)
    result = _run_smoke(bundle, source, tmp_path / "result", RecordingBackend())
    payload = json.loads(result.read_text(encoding="utf-8"))
    if mutation == "extra_field":
        payload["unexpected"] = True
    elif mutation == "statistics_scope":
        payload["statistics"]["final"]["fit_session_ids"] = ["05_15__2"]
    elif mutation == "statistics_shape":
        payload["statistics"]["final"]["normalizer"]["global_mean"] = [0.0]
    elif mutation == "statistics_value":
        payload["statistics"]["inner"][0]["pos_weight"][0] += 0.5
    elif mutation == "statistics_nan":
        payload["statistics"]["inner"][0]["normalizer"]["participant_std"][0] = float("nan")
    elif mutation == "statistics_bool":
        payload["statistics"]["final"]["pos_weight"][0] = True
    elif mutation == "selection_epoch":
        payload["selection"]["selected_fixed_epoch"] = 7
    elif mutation == "calibration_scope":
        payload["selection"]["calibration_oof_session_ids"] = ["05_15__2"]
    elif mutation == "artifact_fold_swap":
        payload["artifact_bindings"]["inner_oof_predictions"][0]["inner_fold_id"] = "inner_02"
    elif mutation == "artifact_role_reuse":
        source_binding = payload["artifact_bindings"]["inner_checkpoints"][0]
        target_binding = payload["artifact_bindings"]["inner_oof_predictions"][0]
        for key in ("relative_path", "sha256", "size_bytes"):
            target_binding[key] = source_binding[key]
    elif mutation == "artifact_role_swap":
        checkpoint = payload["artifact_bindings"]["inner_checkpoints"][0]
        predictions = payload["artifact_bindings"]["inner_oof_predictions"][0]
        for key in ("relative_path", "sha256", "size_bytes"):
            checkpoint[key], predictions[key] = predictions[key], checkpoint[key]
    elif mutation == "master_hash":
        payload["master_sessions_sha256_after"] = "0" * 64
    else:
        payload["result_role"] = "counted_outer_test"
        payload["counted_experiment_complete"] = True
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OuterExperimentGateError):
        outer._validate_result_directory(result.parent)


@pytest.mark.parametrize(
    "bad_receipt",
    [
        "inner_seed",
        "inner_outer_fold",
        "bool_epoch",
        "drop_oof",
        "warm_start",
        "bool_eval_count",
        "inner_role_swap",
        "inner_outside_fold",
    ],
)
def test_runner_rejects_backend_receipt_mismatches(
    tmp_path: Path, monkeypatch, bad_receipt: str
) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    _patch_loader(monkeypatch, _sessions(), report)
    backend = RecordingBackend()
    backend.bad_receipt = bad_receipt
    output = tmp_path / "result"
    with pytest.raises(OuterExperimentGateError):
        _run_smoke(bundle, source, output, backend)
    assert not output.exists()
    assert not list(tmp_path.glob(".result.*.stage"))


def test_runner_rejects_hardlinked_inner_checkpoint_and_oof(tmp_path: Path, monkeypatch) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    _patch_loader(monkeypatch, _sessions(), report)
    output = tmp_path / "result"
    with pytest.raises(OuterExperimentGateError, match="physical link"):
        _run_smoke(bundle, source, output, HardlinkInnerArtifactsBackend())
    assert not output.exists()
    assert not list(tmp_path.glob(".result.*.stage"))


def test_all_artifact_role_pairs_reject_physical_hardlink_reuse(tmp_path: Path) -> None:
    names = [
        *(f"inner_{index}_{outer.INNER_CHECKPOINT_FILENAME}" for index in range(3)),
        *(f"inner_{index}_{outer.INNER_OOF_FILENAME}" for index in range(3)),
        outer.CALIBRATION_FILENAME,
        outer.FINAL_CHECKPOINT_FILENAME,
        outer.OUTER_PREDICTIONS_FILENAME,
        outer.OUTER_METRICS_FILENAME,
    ]
    paths = tuple(tmp_path / name for name in names)
    for index, path in enumerate(paths):
        path.write_text(f"artifact-{index}", encoding="utf-8")
    for left_index, left in enumerate(paths):
        for right in paths[left_index + 1 :]:
            original = right.read_bytes()
            right.unlink()
            os.link(left, right)
            with pytest.raises(OuterExperimentGateError, match="physical link|physical file"):
                outer._validate_physical_artifact_uniqueness(paths)
            right.unlink()
            right.write_bytes(original)


def test_single_handle_snapshot_rejects_path_aba(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target.bin"
    alternate = tmp_path / "alternate.bin"
    parked = tmp_path / "parked.bin"
    target.write_bytes(b"original")
    alternate.write_bytes(b"alternate")
    real_hash = outer._hash_open_handle

    def aba_during_hash(fd, *, capture_bytes):
        moved_original = False
        moved_alternate = False
        try:
            os.replace(target, parked)
            moved_original = True
            os.replace(alternate, target)
            moved_alternate = True
            return real_hash(fd, capture_bytes=capture_bytes)
        finally:
            if moved_alternate and target.exists():
                os.replace(target, alternate)
            if moved_original and parked.exists():
                os.replace(parked, target)

    monkeypatch.setattr(outer, "_hash_open_handle", aba_during_hash)
    with pytest.raises(OuterExperimentGateError):
        outer._capture_file_snapshot(target)
    assert target.read_bytes() == b"original"
    assert alternate.read_bytes() == b"alternate"


def test_single_handle_snapshot_rejects_same_inode_temporary_rewrite(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "target.json"
    original = b'{"value":1}'
    corrupted = b'{"value":2}'
    assert len(original) == len(corrupted)
    target.write_bytes(original)
    original_stat = target.stat()
    real_hash = outer._hash_open_handle

    def rewrite_during_hash(fd, *, capture_bytes):
        modified = False
        try:
            with target.open("r+b") as writer:
                writer.write(corrupted)
                writer.flush()
                os.fsync(writer.fileno())
                modified = True
                os.utime(
                    target,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )
                result = real_hash(fd, capture_bytes=capture_bytes)
                writer.seek(0)
                writer.write(original)
                writer.flush()
                os.fsync(writer.fileno())
                os.utime(
                    target,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )
                modified = False
                return result
        finally:
            if modified:
                target.write_bytes(original)
                os.utime(
                    target,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )

    monkeypatch.setattr(outer, "_hash_open_handle", rewrite_during_hash)
    with pytest.raises(OuterExperimentGateError):
        outer._read_json_with_snapshot(target)
    assert target.read_bytes() == original


def test_snapshot_rejects_nonregular_handle_after_lstat_open_race(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"regular-before-open")
    read_fd, write_fd = os.pipe()
    os.close(write_fd)

    monkeypatch.setattr(outer, "_open_snapshot_fd", lambda _path: read_fd)
    with pytest.raises(OuterExperimentGateError, match="regular|open immutable"):
        outer._capture_file_snapshot(target)
    with pytest.raises(OSError):
        os.fstat(read_fd)


def test_snapshot_close_error_is_a_gate_failure(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"content")
    real_close = os.close

    def close_then_raise(fd):
        real_close(fd)
        raise OSError("simulated close failure")

    monkeypatch.setattr(outer.os, "close", close_then_raise)
    with pytest.raises(OuterExperimentGateError, match="close immutable-file"):
        outer._capture_file_snapshot(target)


def test_snapshot_rejects_reparse_ancestor(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"content")
    guarded_ancestor = target.parent
    real_lstat = Path.lstat

    class ReparseStat:
        def __init__(self, value):
            self.st_mode = value.st_mode
            self.st_dev = value.st_dev
            self.st_ino = value.st_ino
            self.st_mtime_ns = value.st_mtime_ns
            self.st_ctime_ns = value.st_ctime_ns
            self.st_file_attributes = (
                getattr(value, "st_file_attributes", 0) | outer._FILE_ATTRIBUTE_REPARSE_POINT
            )

    def lstat_with_reparse_ancestor(path):
        value = real_lstat(path)
        return ReparseStat(value) if path == guarded_ancestor else value

    monkeypatch.setattr(Path, "lstat", lstat_with_reparse_ancestor)
    with pytest.raises(OuterExperimentGateError, match="ancestor"):
        outer._capture_file_snapshot(target)


@pytest.mark.skipif(os.name != "nt", reason="Windows CreateFile share-mode contract")
@pytest.mark.parametrize("operation", ["write", "rename", "delete"])
def test_windows_snapshot_handle_denies_write_rename_delete(
    tmp_path: Path, monkeypatch, operation: str
) -> None:
    target = tmp_path / "locked.bin"
    moved = tmp_path / "moved.bin"
    target.write_bytes(b"locked")
    real_hash = outer._hash_open_handle
    denied = False

    def attack_during_hash(fd, *, capture_bytes):
        nonlocal denied
        try:
            if operation == "write":
                with target.open("r+b") as writer:
                    writer.write(b"broken")
            elif operation == "rename":
                target.rename(moved)
            else:
                target.unlink()
        except OSError:
            denied = True
        return real_hash(fd, capture_bytes=capture_bytes)

    monkeypatch.setattr(outer, "_hash_open_handle", attack_during_hash)
    outer._capture_file_snapshot(target)
    assert denied is True
    assert target.read_bytes() == b"locked"
    assert not moved.exists()


def test_runner_rejects_temporary_valid_manifest_rewrite_during_hash_and_rolls_back(
    tmp_path: Path, monkeypatch
) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    _patch_loader(monkeypatch, _sessions(), report)
    output = (tmp_path / "result").resolve()
    real_hash = outer._hash_open_handle
    attack_attempted = False

    def rewrite_result_during_hash(fd, *, capture_bytes):
        nonlocal attack_attempted
        candidates = list(tmp_path.glob(".result.*.stage/result_manifest.json"))
        target = candidates[0] if len(candidates) == 1 else None
        if target is not None and not attack_attempted:
            fd_stat = os.fstat(fd)
            target_stat = target.stat()
            if (fd_stat.st_dev, fd_stat.st_ino) == (
                target_stat.st_dev,
                target_stat.st_ino,
            ):
                attack_attempted = True
                original = target.read_bytes()
                changed = original.replace(b'"seed": 13', b'"seed": 42', 1)
                assert changed != original
                assert len(changed) == len(original)
                original_stat = target.stat()
                modified = False
                try:
                    # Windows must deny this writer while the snapshot handle is
                    # live.  POSIX permits it, in which case the handle ctime
                    # check must expose the rewrite even after bytes/mtime return.
                    with target.open("r+b") as writer:
                        writer.write(changed)
                        writer.flush()
                        os.fsync(writer.fileno())
                        modified = True
                        os.utime(
                            target,
                            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                        )
                        result = real_hash(fd, capture_bytes=capture_bytes)
                        writer.seek(0)
                        writer.write(original)
                        writer.flush()
                        os.fsync(writer.fileno())
                        os.utime(
                            target,
                            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                        )
                        modified = False
                        return result
                finally:
                    if modified:
                        target.write_bytes(original)
                        os.utime(
                            target,
                            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                        )
        return real_hash(fd, capture_bytes=capture_bytes)

    monkeypatch.setattr(outer, "_hash_open_handle", rewrite_result_during_hash)
    with pytest.raises(OuterExperimentGateError):
        _run_smoke(bundle, source, output, RecordingBackend())
    assert attack_attempted is True
    assert not output.exists()
    assert not list(tmp_path.glob(".result.*.stage"))


def test_runner_holds_complete_file_set_until_return_and_rolls_back_attack(
    tmp_path: Path, monkeypatch
) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    _patch_loader(monkeypatch, _sessions(), report)
    output = (tmp_path / "result").resolve()
    result_manifest = output / "result_manifest.json"
    real_snapshot = outer._snapshot_held_handle
    attack_attempted = False

    def snapshot_then_attack(held, *, capture_bytes):
        nonlocal attack_attempted
        snapshot = real_snapshot(held, capture_bytes=capture_bytes)
        if held.path == result_manifest and not attack_attempted:
            attack_attempted = True
            try:
                # This is the former collection-level hole: result_manifest was
                # already verified while the remaining files had not yet been.
                result_manifest.write_bytes(b"[")
            except OSError as exc:
                # Windows' held-set share mode is itself the required denial;
                # surface it as a gate failure so the invocation rolls back.
                raise OuterExperimentGateError(
                    "held result set denied a concurrent manifest rewrite"
                ) from exc
        return snapshot

    monkeypatch.setattr(outer, "_snapshot_held_handle", snapshot_then_attack)
    with pytest.raises(OuterExperimentGateError):
        _run_smoke(bundle, source, output, RecordingBackend())
    assert attack_attempted is True
    assert not output.exists()
    assert not list(tmp_path.glob(".result.*.stage"))


@pytest.mark.parametrize("mutation", ["source", "checkpoint", "oof", "calibration"])
def test_runner_rejects_bound_artifact_mutation_and_rolls_back(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    _patch_loader(monkeypatch, _sessions(), report)
    backend = RecordingBackend()
    backend.mutate_source = source if mutation == "source" else None
    backend.mutate_checkpoint = mutation == "checkpoint"
    backend.mutate_oof = mutation == "oof"
    backend.mutate_calibration = mutation == "calibration"
    output = tmp_path / "result"
    with pytest.raises(OuterExperimentGateError):
        _run_smoke(bundle, source, output, backend)
    assert not output.exists()


def test_fresh_d3_provenance_cannot_be_refreshed_to_changed_values(tmp_path: Path) -> None:
    split, _, _, report = _protocol_payloads(tmp_path)
    changed = copy.deepcopy(split["d3_records"])
    changed["05_15__2"]["status"] = "ok"
    with pytest.raises(OuterExperimentGateError, match="pre-outcome freeze"):
        build_outer_split_manifest(
            SESSION_TO_GROUP,
            source_bindings=split["source_bindings"],
            conditions=CONDITIONS,
            d3_records=changed,
        )
    assert outer._d3_records_from_report(report) == split["d3_records"]


def test_execution_rejects_changed_bound_d3_provenance(tmp_path: Path, monkeypatch) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    _patch_loader(monkeypatch, _sessions(), report)
    provenance = source / "05_15__2.provenance.json"
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["offset_quality"]["error_bound_ms"] = 1.0
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "result"
    with pytest.raises(OuterExperimentGateError, match="bound source bytes changed"):
        _run_smoke(bundle, source, output, RecordingBackend())
    assert not output.exists()


def test_protocol_and_result_are_no_clobber(tmp_path: Path, monkeypatch) -> None:
    split, run, source, report = _protocol_payloads(tmp_path)
    bundle = publish_outer_protocol(tmp_path / "protocol", split, run)
    marker = bundle.directory / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        publish_outer_protocol(bundle.directory, split, run)
    assert marker.read_text(encoding="utf-8") == "keep"

    _patch_loader(monkeypatch, _sessions(), report)
    output = tmp_path / "result"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run_smoke(bundle, source, output, RecordingBackend())
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("failure", ["mkdir", "link", "partial_marker"])
def test_stage_creation_baseexception_leaves_no_owned_stage(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    split, run, _, _ = _protocol_payloads(tmp_path)
    if failure == "mkdir":
        real_mkdir = Path.mkdir

        def mkdir_then_interrupt(path, *args, **kwargs):
            if path.name.endswith(".stage"):
                real_mkdir(path, *args, **kwargs)
                raise KeyboardInterrupt("after mkdir")
            return real_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", mkdir_then_interrupt)
        expected = KeyboardInterrupt
    elif failure == "link":
        real_link = os.link

        def link_then_exit(source, destination):
            real_link(source, destination)
            raise SystemExit("after link")

        monkeypatch.setattr(outer.os, "link", link_then_exit)
        expected = SystemExit
    else:

        def partial_marker(stage, token):
            (stage / f".ownership.{token}.tmp").write_text("partial", encoding="utf-8")
            raise KeyboardInterrupt("partial marker")

        monkeypatch.setattr(outer, "_write_ownership_marker", partial_marker)
        expected = KeyboardInterrupt
    with pytest.raises(expected):
        publish_outer_protocol(tmp_path / "protocol", split, run)
    assert not (tmp_path / "protocol").exists()
    assert not list(tmp_path.glob(".protocol.*.stage"))


def test_stage_creation_race_preserves_foreign_unowned_directory(
    tmp_path: Path, monkeypatch
) -> None:
    split, run, _, _ = _protocol_payloads(tmp_path)
    real_mkdir = Path.mkdir

    def competing_mkdir(path, *args, **kwargs):
        if path.name.endswith(".stage"):
            real_mkdir(path, *args, **kwargs)
            (path / "foreign.txt").write_text("keep", encoding="utf-8")
            raise FileExistsError("simulated stage race")
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", competing_mkdir)
    with pytest.raises(FileExistsError, match="stage race"):
        publish_outer_protocol(tmp_path / "protocol", split, run)
    foreign_stages = list(tmp_path.glob(".protocol.*.stage"))
    assert len(foreign_stages) == 1
    assert (foreign_stages[0] / "foreign.txt").read_text(encoding="utf-8") == "keep"


def test_interrupt_after_owned_stage_creation_is_cleaned(tmp_path: Path, monkeypatch) -> None:
    split, run, _, _ = _protocol_payloads(tmp_path)
    real_create = outer._create_owned_stage

    def create_then_interrupt(stage, token):
        real_create(stage, token)
        raise KeyboardInterrupt("after create returned")

    monkeypatch.setattr(outer, "_create_owned_stage", create_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="after create returned"):
        publish_outer_protocol(tmp_path / "protocol", split, run)
    assert not list(tmp_path.glob(".protocol.*.stage"))


def test_publish_race_preserves_foreign_destination(tmp_path: Path, monkeypatch) -> None:
    split, run, _, _ = _protocol_payloads(tmp_path)
    destination = tmp_path / "protocol"

    def collide(_source, target):
        target_path = Path(target)
        target_path.mkdir()
        (target_path / "ownership.json").write_text(
            json.dumps({"publication_token": "foreign"}), encoding="utf-8"
        )
        (target_path / "foreign.txt").write_text("keep", encoding="utf-8")
        raise FileExistsError("simulated concurrent publisher")

    monkeypatch.setattr(outer.os, "rename", collide)
    with pytest.raises(FileExistsError, match="concurrent"):
        publish_outer_protocol(destination, split, run)
    assert (destination / "foreign.txt").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".protocol.*.stage"))


def test_result_rename_then_raise_removes_only_own_result(tmp_path: Path, monkeypatch) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    _patch_loader(monkeypatch, _sessions(), report)
    real_rename = os.rename

    def rename_then_raise(staged, destination):
        real_rename(staged, destination)
        raise OSError("simulated result post-rename failure")

    monkeypatch.setattr(outer.os, "rename", rename_then_raise)
    destination = tmp_path / "result"
    with pytest.raises(OSError, match="result post-rename"):
        _run_smoke(bundle, source, destination, RecordingBackend())
    assert not destination.exists()
    assert not list(tmp_path.glob(".result.*.stage"))


@pytest.mark.parametrize("target", ["result_manifest", "bound_source"])
def test_final_validation_window_tamper_fails_and_rolls_back(
    tmp_path: Path, monkeypatch, target: str
) -> None:
    bundle, source, report = _published_protocol(tmp_path)
    _patch_loader(monkeypatch, _sessions(), report)
    destination = (tmp_path / "result").resolve()
    real_validate = outer._validate_result_directory

    def validate_then_tamper(directory):
        snapshot = real_validate(directory)
        if Path(directory).resolve() == destination:
            if target == "result_manifest":
                manifest = destination / "result_manifest.json"
                manifest.write_bytes(manifest.read_bytes() + b"\n")
            else:
                bound_source = next(source.glob("*.csv"))
                bound_source.write_text("changed-after-validation", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(outer, "_validate_result_directory", validate_then_tamper)
    with pytest.raises(OuterExperimentGateError, match="changed before final return"):
        _run_smoke(bundle, source, destination, RecordingBackend())
    assert not destination.exists()
    assert not list(tmp_path.glob(".result.*.stage"))
