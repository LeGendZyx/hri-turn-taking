"""In-memory, fixture-only binding for a future formal VAP run specification.

This module deliberately does not integrate with the outer runner, register a
backend, inspect real data, train a model, or write/publish an artifact.  It
only makes the currently missing configuration boundary explicit and
adversarially testable.  Every returned payload is permanently fixture-only
and non-counted.

The caller must supply every candidate value, including ``max_epochs`` and
``batch_size``.  Building a payload records those values; it does not claim
that the owner approved them.  Calibration/policy thresholds are intentionally
absent from this schema and remain an explicit owner-pending item.
"""

from __future__ import annotations

import hashlib
import json
import math

from hri_turn_taking.vap.feature_sets import active_feature_columns
from hri_turn_taking.vap.production_backend import FrozenTorchSpec

FORMAT_VERSION = 1
SPEC_KIND = "vap_fixture_formal_run_spec_binding"
MANIFEST_BINDING_KIND = "vap_fixture_run_manifest_spec_binding"
COUNTED_ELIGIBLE = False
FIXTURE_ONLY = True

HARD_BLOCKERS = (
    "fixture_only_contract",
    "outer_run_manifest_integration_not_implemented",
    "production_backend_not_registered",
    "formal_batch_scheduler_execution_not_implemented",
    "formal_data_readiness_not_authorized",
    "atomic_run_index_not_implemented",
    "publisher_intentionally_absent",
)

OWNER_PENDING = (
    "formal_max_epochs_approval_pending",
    "calibration_threshold_policy_approval_pending",
    "filesystem_trust_boundary_decision_pending",
)

_SPEC_ROLE = "fixture_only_in_memory_formal_run_spec_not_counted"
_MANIFEST_ROLE = "fixture_only_in_memory_manifest_binding_not_counted"

_OPTIMIZER_KEYS = {
    "name",
    "learning_rate",
    "weight_decay",
    "betas",
    "eps",
    "amsgrad",
    "maximize",
    "foreach",
    "capturable",
    "differentiable",
    "fused",
    "parameter_group_rule",
    "gradient_clip",
    "aux_vad_weight",
}
_SCHEDULER_KEYS = {
    "name",
    "warmup_fraction",
    "warmup_steps_rule",
    "decay_rule",
    "minimum_lr_scale",
    "step_interval",
    "total_steps_rule",
}


class FormalRunSpecGateError(ValueError):
    """A fixture spec or manifest-binding contract failed closed."""


def _fail(message: str) -> None:
    raise FormalRunSpecGateError(message)


def _base_payload(kind: str, role: str) -> dict[str, object]:
    return {
        "artifact_kind": kind,
        "format_version": FORMAT_VERSION,
        "binding_role": role,
        "fixture_only": True,
        "counted_eligible": False,
        "publisher_available": False,
        "writer_available": False,
        "hard_blockers": list(HARD_BLOCKERS),
        "owner_pending": list(OWNER_PENDING),
    }


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FormalRunSpecGateError(f"value is not strict JSON: {exc}") from exc


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _json_clone(value: object) -> object:
    return json.loads(_canonical_json(value).decode("utf-8"))


def _type_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _type_exact_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _type_exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _normalised_string(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{name} must be a non-empty normalised string")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        _fail(f"{name} must be an exact positive integer")
    return value


def _exact_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        _fail(f"{name} must be an exact finite JSON float")
    if minimum is not None and value < minimum:
        _fail(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        _fail(f"{name} must be <= {maximum}")
    return value


def _exact_mapping(value: object, expected_keys: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected_keys:
        _fail(f"{name} fields are not exact")
    _canonical_json(value)
    return value


def _sha256_string(value: object, name: str) -> str:
    value = _normalised_string(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        _fail(f"{name} must be a lowercase SHA-256 string")
    return value


def _validate_base(payload: dict[str, object], kind: str, role: str) -> None:
    expected = _base_payload(kind, role)
    for key, value in expected.items():
        if key not in payload or not _type_exact_equal(payload[key], value):
            _fail(f"{kind} base field {key!r} differs from the code-owned fixture gate")


def _validate_condition(value: object) -> dict[str, object]:
    condition = _exact_mapping(
        value,
        {"condition_id", "feature_set", "encoder", "active_feature_columns"},
        "condition",
    )
    _normalised_string(condition["condition_id"], "condition_id")
    feature_set = _normalised_string(condition["feature_set"], "feature_set")
    encoder = _normalised_string(condition["encoder"], "condition encoder")
    if encoder not in {"transformer", "gru"}:
        _fail("condition encoder must be transformer or gru")
    columns = condition["active_feature_columns"]
    if type(columns) is not list or any(type(column) is not str for column in columns):
        _fail("active_feature_columns must be an exact JSON string list")
    try:
        expected_columns = active_feature_columns(feature_set)
    except ValueError as exc:
        raise FormalRunSpecGateError(str(exc)) from exc
    if columns != expected_columns:
        _fail("active_feature_columns differ from the code-owned feature set")
    return condition


def _validated_torch_spec(value: object) -> FrozenTorchSpec:
    if type(value) is not dict:
        _fail("frozen_torch_spec must be an exact JSON object")
    try:
        rebuilt = FrozenTorchSpec.from_payload(value)
    except (TypeError, ValueError) as exc:
        raise FormalRunSpecGateError(f"invalid FrozenTorchSpec: {exc}") from exc
    if not _type_exact_equal(value, rebuilt.to_payload()):
        _fail("frozen_torch_spec values/types are not canonical")
    return rebuilt


def _expected_model(spec: FrozenTorchSpec) -> dict[str, object]:
    return {
        "n_participant_features": len(spec.participant_features),
        "n_global_features": len(spec.global_features),
        "n_participants": spec.n_participants,
        "n_bins": spec.n_bins,
        "input_dim": spec.input_dim,
        "d_model": spec.d_model,
        "n_layers": spec.n_layers,
        "n_heads": spec.n_heads,
        "dropout": spec.dropout,
        "encoder": spec.encoder,
        "attention_window": spec.attention_window,
        "device": spec.device,
        "deterministic_algorithms": spec.deterministic_algorithms,
    }


def _validate_model(value: object, spec: FrozenTorchSpec) -> dict[str, object]:
    expected = _expected_model(spec)
    model = _exact_mapping(value, set(expected), "model")
    if not _type_exact_equal(model, expected):
        _fail("model values/types differ from the bound FrozenTorchSpec")
    return model


def _validate_label(value: object, spec: FrozenTorchSpec) -> dict[str, object]:
    label = _exact_mapping(
        value,
        {"frame_rate_hz", "bin_edges_s", "active_ratio", "now_bins", "future_bins"},
        "label",
    )
    _positive_int(label["frame_rate_hz"], "label.frame_rate_hz")
    edges = label["bin_edges_s"]
    if type(edges) is not list or not edges:
        _fail("label.bin_edges_s must be a non-empty exact JSON list")
    checked_edges = [
        _exact_float(edge, "label.bin_edges_s", minimum=0.0) for edge in edges
    ]
    if any(
        right <= left
        for left, right in zip(checked_edges, checked_edges[1:], strict=False)
    ):
        _fail("label.bin_edges_s must be strictly increasing")
    _exact_float(label["active_ratio"], "label.active_ratio", minimum=0.0, maximum=1.0)
    for key in ("now_bins", "future_bins"):
        indices = label[key]
        if type(indices) is not list or not indices:
            _fail(f"label.{key} must be a non-empty exact JSON list")
        if any(type(index) is not int or index < 0 for index in indices):
            _fail(f"label.{key} must contain exact non-negative integers")
        if len(indices) != len(set(indices)):
            _fail(f"label.{key} cannot contain duplicates")
    expected = spec.to_payload()["label_config"]
    if not _type_exact_equal(label, expected):
        _fail("label config drifted from the bound FrozenTorchSpec")
    return label


def _expected_optimizer(spec: FrozenTorchSpec) -> dict[str, object]:
    return {
        "name": "torch.optim.AdamW",
        "learning_rate": spec.learning_rate,
        "weight_decay": spec.weight_decay,
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "parameter_group_rule": "ndim_ge_2_decay_else_zero_weight_decay",
        "gradient_clip": spec.gradient_clip,
        "aux_vad_weight": spec.aux_vad_weight,
    }


def _validate_optimizer(value: object, spec: FrozenTorchSpec) -> dict[str, object]:
    optimizer = _exact_mapping(value, _OPTIMIZER_KEYS, "optimizer")
    expected = _expected_optimizer(spec)
    if not _type_exact_equal(optimizer, expected):
        _fail("optimizer values/types differ from the explicit backend contract")
    return optimizer


def _expected_training_windows(spec: FrozenTorchSpec) -> dict[str, object]:
    return {
        "window_frames": spec.window_frames,
        "stride_frames": spec.stride_frames,
        "inference_chunk_frames": spec.inference_chunk_frames,
        "window_order": spec.window_order,
        "tail_policy": "append_final_overlapping_window",
        "short_session_policy": "one_full_short_window",
        "target_gate": "at_least_one_complete_future_horizon",
    }


def _validate_training_windows(value: object, spec: FrozenTorchSpec) -> dict[str, object]:
    expected = _expected_training_windows(spec)
    windows = _exact_mapping(value, set(expected), "training_windows")
    if not _type_exact_equal(windows, expected):
        _fail("training-window values/types differ from the bound FrozenTorchSpec")
    return windows


def _validate_scheduler(value: object) -> dict[str, object]:
    scheduler = _exact_mapping(value, _SCHEDULER_KEYS, "scheduler")
    if scheduler["name"] != "linear_warmup_cosine_decay":
        _fail("scheduler.name must explicitly select linear warmup plus cosine decay")
    _exact_float(
        scheduler["warmup_fraction"],
        "scheduler.warmup_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    _exact_float(
        scheduler["minimum_lr_scale"],
        "scheduler.minimum_lr_scale",
        minimum=0.0,
        maximum=1.0,
    )
    expected_strings = {
        "warmup_steps_rule": "max_1_round_fraction_times_total_steps",
        "decay_rule": "cosine_after_warmup",
        "step_interval": "after_each_optimizer_step",
        "total_steps_rule": "max_1_max_epochs_times_ceil_windows_over_batch_size",
    }
    for key, expected in expected_strings.items():
        if scheduler[key] != expected or type(scheduler[key]) is not str:
            _fail(f"scheduler.{key} differs from the explicit scheduler contract")
    return scheduler


def build_formal_run_spec(
    *,
    condition: dict[str, object],
    frozen_torch_spec: FrozenTorchSpec,
    model: dict[str, object],
    label: dict[str, object],
    optimizer: dict[str, object],
    training_windows: dict[str, object],
    max_epochs: int,
    batch_size: int,
    scheduler: dict[str, object],
) -> dict[str, object]:
    """Build one strict candidate spec without approving or publishing it."""
    if type(frozen_torch_spec) is not FrozenTorchSpec:
        _fail("frozen_torch_spec must have the exact FrozenTorchSpec type")
    torch_spec = FrozenTorchSpec.from_payload(frozen_torch_spec.to_payload())
    checked_condition = _validate_condition(condition)
    if checked_condition["encoder"] != torch_spec.encoder:
        _fail("condition encoder differs from FrozenTorchSpec.encoder")
    body = {
        **_base_payload(SPEC_KIND, _SPEC_ROLE),
        "condition": _json_clone(checked_condition),
        "frozen_torch_spec": torch_spec.to_payload(),
        "frozen_torch_spec_sha256": torch_spec.sha256,
        "model": _json_clone(_validate_model(model, torch_spec)),
        "label": _json_clone(_validate_label(label, torch_spec)),
        "optimizer": _json_clone(_validate_optimizer(optimizer, torch_spec)),
        "training_windows": _json_clone(
            _validate_training_windows(training_windows, torch_spec)
        ),
        "max_epochs": _positive_int(max_epochs, "max_epochs"),
        "batch_size": _positive_int(batch_size, "batch_size"),
        "scheduler": _json_clone(_validate_scheduler(scheduler)),
    }
    payload = {**body, "spec_sha256": _json_sha256(body)}
    validate_formal_run_spec(payload)
    return payload


def validate_formal_run_spec(payload: object) -> dict[str, object]:
    """Validate a spec with exact JSON types and all redundant bindings."""
    expected_keys = {
        *set(_base_payload(SPEC_KIND, _SPEC_ROLE)),
        "condition",
        "frozen_torch_spec",
        "frozen_torch_spec_sha256",
        "model",
        "label",
        "optimizer",
        "training_windows",
        "max_epochs",
        "batch_size",
        "scheduler",
        "spec_sha256",
    }
    checked = _exact_mapping(payload, expected_keys, "formal run spec")
    _validate_base(checked, SPEC_KIND, _SPEC_ROLE)
    condition = _validate_condition(checked["condition"])
    torch_spec = _validated_torch_spec(checked["frozen_torch_spec"])
    if _sha256_string(
        checked["frozen_torch_spec_sha256"], "frozen_torch_spec_sha256"
    ) != torch_spec.sha256:
        _fail("frozen_torch_spec_sha256 mismatch")
    if condition["encoder"] != torch_spec.encoder:
        _fail("condition encoder differs from FrozenTorchSpec.encoder")
    _validate_model(checked["model"], torch_spec)
    _validate_label(checked["label"], torch_spec)
    _validate_optimizer(checked["optimizer"], torch_spec)
    _validate_training_windows(checked["training_windows"], torch_spec)
    _positive_int(checked["max_epochs"], "max_epochs")
    _positive_int(checked["batch_size"], "batch_size")
    _validate_scheduler(checked["scheduler"])
    claimed = _sha256_string(checked["spec_sha256"], "spec_sha256")
    body = {key: checked[key] for key in checked if key != "spec_sha256"}
    if claimed != _json_sha256(body):
        _fail("formal run spec hash mismatch")
    return checked


def formal_run_spec_sha256(payload: object) -> str:
    """Return the validated self-hash of one canonical spec body."""
    return validate_formal_run_spec(payload)["spec_sha256"]


def canonical_formal_run_spec_json(payload: object) -> str:
    """Return stable UTF-8 JSON text (without a trailing newline)."""
    return _canonical_json(validate_formal_run_spec(payload)).decode("utf-8")


def build_manifest_spec_binding(formal_run_spec: object) -> dict[str, object]:
    """Build a pure manifest-like view that binds every execution category."""
    spec = validate_formal_run_spec(formal_run_spec)
    body = {
        **_base_payload(MANIFEST_BINDING_KIND, _MANIFEST_ROLE),
        "formal_run_spec_sha256": spec["spec_sha256"],
        "frozen_torch_spec_sha256": spec["frozen_torch_spec_sha256"],
        "condition": _json_clone(spec["condition"]),
        "model_config": _json_clone(spec["model"]),
        "label_config": _json_clone(spec["label"]),
        "optimizer_config": _json_clone(spec["optimizer"]),
        "training_windows": _json_clone(spec["training_windows"]),
        "max_epochs": spec["max_epochs"],
        "batch_size": spec["batch_size"],
        "scheduler_config": _json_clone(spec["scheduler"]),
    }
    return {**body, "binding_sha256": _json_sha256(body)}


def validate_manifest_spec_binding(
    manifest_like: object,
    formal_run_spec: object,
) -> dict[str, object]:
    """Reject a manifest-like mapping unless it is exactly the supplied spec."""
    spec = validate_formal_run_spec(formal_run_spec)
    expected = build_manifest_spec_binding(spec)
    checked = _exact_mapping(manifest_like, set(expected), "manifest spec binding")
    _validate_base(checked, MANIFEST_BINDING_KIND, _MANIFEST_ROLE)
    _sha256_string(checked["formal_run_spec_sha256"], "formal_run_spec_sha256")
    _sha256_string(checked["frozen_torch_spec_sha256"], "frozen_torch_spec_sha256")
    _validate_condition(checked["condition"])
    torch_spec = _validated_torch_spec(spec["frozen_torch_spec"])
    _validate_model(checked["model_config"], torch_spec)
    _validate_label(checked["label_config"], torch_spec)
    _validate_optimizer(checked["optimizer_config"], torch_spec)
    _validate_training_windows(checked["training_windows"], torch_spec)
    _positive_int(checked["max_epochs"], "manifest max_epochs")
    _positive_int(checked["batch_size"], "manifest batch_size")
    _validate_scheduler(checked["scheduler_config"])
    claimed = _sha256_string(checked["binding_sha256"], "binding_sha256")
    body = {key: checked[key] for key in checked if key != "binding_sha256"}
    if claimed != _json_sha256(body):
        _fail("manifest spec binding hash mismatch")
    if not _type_exact_equal(checked, expected):
        _fail("manifest-like configuration differs from the supplied formal run spec")
    return checked


def canonical_manifest_spec_binding_json(
    manifest_like: object,
    formal_run_spec: object,
) -> str:
    """Return stable JSON for a validated fixture manifest binding."""
    checked = validate_manifest_spec_binding(manifest_like, formal_run_spec)
    return _canonical_json(checked).decode("utf-8")


__all__ = [
    "COUNTED_ELIGIBLE",
    "FIXTURE_ONLY",
    "HARD_BLOCKERS",
    "OWNER_PENDING",
    "FormalRunSpecGateError",
    "build_formal_run_spec",
    "build_manifest_spec_binding",
    "canonical_formal_run_spec_json",
    "canonical_manifest_spec_binding_json",
    "formal_run_spec_sha256",
    "validate_formal_run_spec",
    "validate_manifest_spec_binding",
]
