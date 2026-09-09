from __future__ import annotations

import copy
import hashlib
import json

import pytest

from hri_turn_taking.vap import formal_run_spec as run_spec
from hri_turn_taking.vap.feature_sets import active_feature_columns
from hri_turn_taking.vap.production_backend import FrozenTorchSpec


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _torch_spec() -> FrozenTorchSpec:
    return FrozenTorchSpec(
        d_model=8,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        attention_window=8,
        window_frames=8,
        stride_frames=4,
        inference_chunk_frames=32,
        learning_rate=1e-3,
        weight_decay=2e-4,
        gradient_clip=0.75,
    )


def _candidate_inputs() -> dict[str, object]:
    torch_spec = _torch_spec()
    return {
        "condition": {
            "condition_id": "A+H",
            "feature_set": "audio_plus_head_pose",
            "encoder": "transformer",
            "active_feature_columns": active_feature_columns("audio_plus_head_pose"),
        },
        "frozen_torch_spec": torch_spec,
        "model": {
            "n_participant_features": 7,
            "n_global_features": 5,
            "n_participants": 4,
            "n_bins": 4,
            "input_dim": 33,
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "encoder": "transformer",
            "attention_window": 8,
            "device": "cpu",
            "deterministic_algorithms": True,
        },
        "label": {
            "frame_rate_hz": 25,
            "bin_edges_s": [0.2, 0.6, 1.2, 2.0],
            "active_ratio": 0.5,
            "now_bins": [0, 1],
            "future_bins": [2, 3],
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 2e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "parameter_group_rule": "ndim_ge_2_decay_else_zero_weight_decay",
            "gradient_clip": 0.75,
            "aux_vad_weight": 0.0,
        },
        "training_windows": {
            "window_frames": 8,
            "stride_frames": 4,
            "inference_chunk_frames": 32,
            "window_order": "seeded_epoch_permutation",
            "tail_policy": "append_final_overlapping_window",
            "short_session_policy": "one_full_short_window",
            "target_gate": "at_least_one_complete_future_horizon",
        },
        "max_epochs": 12,
        "batch_size": 3,
        "scheduler": {
            "name": "linear_warmup_cosine_decay",
            "warmup_fraction": 0.1,
            "warmup_steps_rule": "max_1_round_fraction_times_total_steps",
            "decay_rule": "cosine_after_warmup",
            "minimum_lr_scale": 0.1,
            "step_interval": "after_each_optimizer_step",
            "total_steps_rule": "max_1_max_epochs_times_ceil_windows_over_batch_size",
        },
    }


def _spec() -> dict[str, object]:
    return run_spec.build_formal_run_spec(**_candidate_inputs())


def _rehash_spec(payload: dict[str, object]) -> None:
    body = {key: payload[key] for key in payload if key != "spec_sha256"}
    payload["spec_sha256"] = _canonical_hash(body)


def _rehash_binding(payload: dict[str, object]) -> None:
    body = {key: payload[key] for key in payload if key != "binding_sha256"}
    payload["binding_sha256"] = _canonical_hash(body)


def test_spec_is_canonical_exact_and_permanently_fixture_only() -> None:
    spec = _spec()

    assert run_spec.validate_formal_run_spec(spec) == spec
    assert run_spec.formal_run_spec_sha256(spec) == spec["spec_sha256"]
    assert spec["fixture_only"] is True
    assert spec["counted_eligible"] is False
    assert spec["publisher_available"] is False
    assert spec["writer_available"] is False
    assert spec["hard_blockers"] == list(run_spec.HARD_BLOCKERS)
    assert spec["owner_pending"] == list(run_spec.OWNER_PENDING)
    assert spec["condition"]["active_feature_columns"] == active_feature_columns(
        "audio_plus_head_pose"
    )
    assert spec["max_epochs"] == 12
    assert spec["batch_size"] == 3
    assert "thresholds" not in spec

    encoded = run_spec.canonical_formal_run_spec_json(spec)
    assert encoded == json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert "\n" not in encoded


def test_caller_mapping_order_does_not_change_spec_identity() -> None:
    inputs = _candidate_inputs()
    inputs["condition"] = dict(reversed(list(inputs["condition"].items())))
    inputs["model"] = dict(reversed(list(inputs["model"].items())))
    inputs["scheduler"] = dict(reversed(list(inputs["scheduler"].items())))

    reordered = run_spec.build_formal_run_spec(**inputs)
    assert run_spec.canonical_formal_run_spec_json(reordered) == (
        run_spec.canonical_formal_run_spec_json(_spec())
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_epochs", True),
        ("max_epochs", 12.0),
        ("batch_size", False),
        ("batch_size", 3.0),
    ],
)
def test_budget_rejects_bool_and_float_as_int(field: str, value: object) -> None:
    inputs = _candidate_inputs()
    inputs[field] = value
    with pytest.raises(run_spec.FormalRunSpecGateError, match="exact positive integer"):
        run_spec.build_formal_run_spec(**inputs)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "dropout", 0),
        ("label", "active_ratio", 1),
        ("label", "now_bins", [0.0, 1]),
        ("optimizer", "learning_rate", 1),
        ("optimizer", "amsgrad", 0),
        ("training_windows", "window_frames", 8.0),
        ("scheduler", "warmup_fraction", 0),
        ("scheduler", "minimum_lr_scale", float("nan")),
    ],
)
def test_nested_configs_reject_equal_but_wrong_json_types_or_nonfinite(
    section: str,
    field: str,
    value: object,
) -> None:
    inputs = _candidate_inputs()
    inputs[section][field] = value
    with pytest.raises(run_spec.FormalRunSpecGateError):
        run_spec.build_formal_run_spec(**inputs)


@pytest.mark.parametrize(
    "section",
    ["condition", "model", "label", "optimizer", "training_windows", "scheduler"],
)
@pytest.mark.parametrize("change", ["extra", "missing"])
def test_every_nested_mapping_rejects_extra_or_missing_fields(
    section: str,
    change: str,
) -> None:
    inputs = _candidate_inputs()
    mapping = inputs[section]
    if change == "extra":
        mapping["unexpected"] = "forbidden"
    else:
        mapping.pop(next(iter(mapping)))
    with pytest.raises(run_spec.FormalRunSpecGateError, match="fields are not exact"):
        run_spec.build_formal_run_spec(**inputs)


def test_feature_set_columns_and_encoder_are_cross_bound() -> None:
    inputs = _candidate_inputs()
    inputs["condition"]["feature_set"] = "audio_only"
    with pytest.raises(run_spec.FormalRunSpecGateError, match="active_feature_columns"):
        run_spec.build_formal_run_spec(**inputs)

    inputs = _candidate_inputs()
    inputs["condition"]["active_feature_columns"] = list(
        reversed(inputs["condition"]["active_feature_columns"])
    )
    with pytest.raises(run_spec.FormalRunSpecGateError, match="active_feature_columns"):
        run_spec.build_formal_run_spec(**inputs)

    inputs = _candidate_inputs()
    inputs["condition"]["encoder"] = "gru"
    with pytest.raises(run_spec.FormalRunSpecGateError, match="encoder differs"):
        run_spec.build_formal_run_spec(**inputs)


def test_label_drift_and_recomputed_spec_hash_still_fail() -> None:
    inputs = _candidate_inputs()
    inputs["label"]["bin_edges_s"][1] = 0.7
    with pytest.raises(run_spec.FormalRunSpecGateError, match="label config drifted"):
        run_spec.build_formal_run_spec(**inputs)

    forged = copy.deepcopy(_spec())
    forged["label"]["bin_edges_s"][1] = 0.7
    _rehash_spec(forged)
    with pytest.raises(run_spec.FormalRunSpecGateError, match="label config drifted"):
        run_spec.validate_formal_run_spec(forged)


@pytest.mark.parametrize("mutation", ["hash", "body", "extra", "missing"])
def test_spec_rejects_hash_and_top_level_schema_tampering(mutation: str) -> None:
    forged = copy.deepcopy(_spec())
    if mutation == "hash":
        forged["spec_sha256"] = "0" * 64
    elif mutation == "body":
        forged["max_epochs"] += 1
    elif mutation == "extra":
        forged["unexpected"] = True
    else:
        forged.pop("batch_size")
    with pytest.raises(run_spec.FormalRunSpecGateError):
        run_spec.validate_formal_run_spec(forged)


def test_frozen_torch_spec_hash_and_model_cannot_drift_independently() -> None:
    forged = copy.deepcopy(_spec())
    forged["frozen_torch_spec_sha256"] = "0" * 64
    _rehash_spec(forged)
    with pytest.raises(run_spec.FormalRunSpecGateError, match="torch_spec_sha256 mismatch"):
        run_spec.validate_formal_run_spec(forged)

    forged = copy.deepcopy(_spec())
    forged["model"]["d_model"] = 16
    _rehash_spec(forged)
    with pytest.raises(run_spec.FormalRunSpecGateError, match="bound FrozenTorchSpec"):
        run_spec.validate_formal_run_spec(forged)


def test_manifest_like_binding_is_exact_canonical_and_non_counted() -> None:
    spec = _spec()
    manifest = run_spec.build_manifest_spec_binding(spec)

    assert run_spec.validate_manifest_spec_binding(manifest, spec) == manifest
    assert manifest["formal_run_spec_sha256"] == spec["spec_sha256"]
    assert manifest["fixture_only"] is True
    assert manifest["counted_eligible"] is False
    assert manifest["publisher_available"] is False
    assert manifest["writer_available"] is False
    assert manifest["condition"] == spec["condition"]
    assert manifest["model_config"] == spec["model"]
    assert manifest["label_config"] == spec["label"]
    assert manifest["optimizer_config"] == spec["optimizer"]
    assert manifest["training_windows"] == spec["training_windows"]
    assert manifest["scheduler_config"] == spec["scheduler"]
    encoded = run_spec.canonical_manifest_spec_binding_json(manifest, spec)
    assert json.loads(encoded) == manifest


@pytest.mark.parametrize(
    "mutation",
    [
        "condition",
        "features",
        "model",
        "label",
        "optimizer",
        "windows",
        "max_epochs",
        "batch_size",
        "scheduler",
        "spec_hash",
        "torch_hash",
    ],
)
def test_manifest_rejects_rehashed_configuration_drift(mutation: str) -> None:
    spec = _spec()
    forged = copy.deepcopy(run_spec.build_manifest_spec_binding(spec))
    if mutation == "condition":
        forged["condition"]["condition_id"] = "different"
    elif mutation == "features":
        forged["condition"]["active_feature_columns"].pop()
    elif mutation == "model":
        forged["model_config"]["d_model"] = 16
    elif mutation == "label":
        forged["label_config"]["bin_edges_s"][1] = 0.7
    elif mutation == "optimizer":
        forged["optimizer_config"]["gradient_clip"] = 0.5
    elif mutation == "windows":
        forged["training_windows"]["stride_frames"] = 2
    elif mutation == "max_epochs":
        forged["max_epochs"] += 1
    elif mutation == "batch_size":
        forged["batch_size"] += 1
    elif mutation == "scheduler":
        forged["scheduler_config"]["warmup_fraction"] = 0.2
    elif mutation == "spec_hash":
        forged["formal_run_spec_sha256"] = "0" * 64
    else:
        forged["frozen_torch_spec_sha256"] = "0" * 64
    _rehash_binding(forged)
    with pytest.raises(run_spec.FormalRunSpecGateError):
        run_spec.validate_manifest_spec_binding(forged, spec)


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("max_epochs", True),
        ("batch_size", 3.0),
        ("scheduler_warmup", 0),
        ("hash", "0" * 64),
        ("extra", True),
        ("missing", None),
    ],
)
def test_manifest_rejects_wrong_types_hash_and_schema(
    mutation: str,
    value: object,
) -> None:
    spec = _spec()
    forged = copy.deepcopy(run_spec.build_manifest_spec_binding(spec))
    if mutation in {"max_epochs", "batch_size"}:
        forged[mutation] = value
    elif mutation == "scheduler_warmup":
        forged["scheduler_config"]["warmup_fraction"] = value
    elif mutation == "hash":
        forged["binding_sha256"] = value
    elif mutation == "extra":
        forged["unexpected"] = value
    else:
        forged.pop("optimizer_config")
    with pytest.raises(run_spec.FormalRunSpecGateError):
        run_spec.validate_manifest_spec_binding(forged, spec)


def test_public_surface_has_no_writer_publisher_registry_or_real_data_entrypoint() -> None:
    assert run_spec.COUNTED_ELIGIBLE is False
    assert run_spec.FIXTURE_ONLY is True
    assert all(
        token not in name
        for name in run_spec.__all__
        for token in ("write", "publish", "register", "train", "real")
    )
