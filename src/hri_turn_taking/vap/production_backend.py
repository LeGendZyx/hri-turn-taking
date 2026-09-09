"""Unregistered, fixture-only PyTorch backend scaffold for the outer protocol.

This module exercises the intended production training boundary without opening
the counted-experiment gate.  It deliberately is not present in
``outer_experiment._PRODUCTION_BACKEND_REGISTRY`` and every artifact it emits
contains ``counted_eligible=false`` plus the remaining production blockers.

The scaffold is intentionally strict about the parts that are already knowable:

* CPU-only deterministic fitting from a hash-bound :class:`FrozenTorchSpec`;
* runner-supplied normalisation and class weights (never refitted here);
* explicit ``session_id``/``tick_25hz`` identity for every frame;
* one-based epochs, full-session OOF inference, and training-tail coverage;
* a primary VAP loss with ``aux_vad_weight`` fixed to exactly zero; and
* versioned, strictly parsed checkpoint, prediction, calibration, history, and
  diagnostic-metrics payloads.

Real-data execution is not accepted.  A future reviewed change must replace the
fixture authority with a provenance-backed real tick authority, implement the
environment/source/run-index and D3 evidence gates, and separately register the
backend before a counted result can exist.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hri_turn_taking.vap.dataset import (
    TRANSFORMED_GLOBAL_FEATURES,
    TRANSFORMED_PARTICIPANT_FEATURES,
    FeatureNormalizer,
    SessionTensors,
)
from hri_turn_taking.vap.events import (
    next_speaker_scores,
    overlap_resolution_scores,
    robot_entry_scores,
    score_events,
    score_next_speaker,
    shift_hold_scores,
    shift_prediction_scores,
)
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG
from hri_turn_taking.vap.model import MultipartyVAP, VAPModelConfig
from hri_turn_taking.vap.outer_experiment import (
    CALIBRATION_FILENAME,
    FINAL_CHECKPOINT_FILENAME,
    INNER_CHECKPOINT_FILENAME,
    INNER_OOF_FILENAME,
    OUTER_METRICS_FILENAME,
    OUTER_PREDICTIONS_FILENAME,
    CalibrationReceipt,
    CalibrationRequest,
    FinalFitReceipt,
    FinalFitRequest,
    InnerFitReceipt,
    InnerFitRequest,
    OuterEvaluationReceipt,
    OuterEvaluationRequest,
)
from hri_turn_taking.vap.trainer import compute_frame_metrics, predict_session_probabilities

BACKEND_ID = "torch_backend_v3_scaffold"
BACKEND_VERSION = 3
FORMAT_VERSION = 3
COUNTED_ELIGIBLE = False
FIXTURE_DATA_KIND = "synthetic_fixture"

SPEC_KIND = "vap_frozen_torch_spec"
TICK_AUTHORITY_KIND = "vap_fixture_tick_authority"
CHECKPOINT_KIND = "vap_torch_backend_checkpoint"
HISTORY_KIND = "vap_torch_backend_history"
PREDICTION_KIND = "vap_torch_backend_predictions"
CALIBRATION_KIND = "vap_torch_backend_calibration"
METRICS_KIND = "vap_torch_backend_metrics"

INNER_RESULT_ROLE = "production_backend_v3_inner_scaffold_not_counted"
FINAL_RESULT_ROLE = "production_backend_v3_final_scaffold_not_counted"
OOF_RESULT_ROLE = "production_backend_v3_inner_oof_scaffold_not_counted"
OUTER_RESULT_ROLE = "production_backend_v3_outer_scaffold_not_counted"
CALIBRATION_RESULT_ROLE = "production_backend_v3_calibration_scaffold_not_counted"
METRICS_RESULT_ROLE = "production_backend_v3_metrics_scaffold_not_counted"

COUNTED_BLOCKERS = (
    "backend_not_registered",
    "fixture_only_tick_authority",
    "synthetic_generator_replay_not_implemented",
    "outer_run_manifest_spec_binding_not_implemented",
    "formal_batch_scheduler_not_implemented",
    "environment_capture_not_implemented",
    "source_state_capture_not_implemented",
    "atomic_run_index_not_implemented",
    "d3_pose_shift_reassembly_not_implemented",
    "calibration_threshold_policy_owner_approval_pending",
)


class ProductionBackendGateError(ValueError):
    """A v3 scaffold schema, identity, or execution gate failed."""


@dataclass(frozen=True)
class ProductionEvidenceGateV3:
    """Code-owned interface for evidence that cannot yet unlock this scaffold."""

    environment_capture_sha256: None = None
    source_state_sha256: None = None
    atomic_run_index_sha256: None = None
    d3_pose_shift_reassembly_sha256: None = None
    verification_status: str = "interface_only_not_implemented"

    def __post_init__(self) -> None:
        bindings = (
            self.environment_capture_sha256,
            self.source_state_sha256,
            self.atomic_run_index_sha256,
            self.d3_pose_shift_reassembly_sha256,
        )
        if any(value is not None for value in bindings):
            raise ProductionBackendGateError(
                "v3 scaffold evidence bindings cannot self-assert implementation"
            )
        if self.verification_status != "interface_only_not_implemented":
            raise ProductionBackendGateError("production evidence verification is not implemented")

    def to_payload(self) -> dict[str, object]:
        return {
            "gate_contract_version": 1,
            "verification_status": self.verification_status,
            "environment_capture_sha256": self.environment_capture_sha256,
            "source_state_sha256": self.source_state_sha256,
            "atomic_run_index_sha256": self.atomic_run_index_sha256,
            "d3_pose_shift_reassembly_sha256": self.d3_pose_shift_reassembly_sha256,
        }


def _is_exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, name: str) -> int:
    if not _is_exact_int(value) or value < 1:
        raise ProductionBackendGateError(f"{name} must be an exact positive integer")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionBackendGateError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProductionBackendGateError(f"{name} must be a finite number")
    return result


def _normalised_string(value: object, name: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise ProductionBackendGateError(f"{name} must be a non-empty normalised string")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(payload: object) -> str:
    return _sha256_bytes(_canonical_json(payload))


def _sha256_string(value: object, name: str) -> str:
    value = _normalised_string(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ProductionBackendGateError(f"{name} must be a lowercase SHA-256 string")
    return value


def _backend_payload() -> dict[str, object]:
    return {
        "backend_id": BACKEND_ID,
        "backend_version": BACKEND_VERSION,
        "registry_status": "absent_by_design",
        "execution_scope": "synthetic_fixture_only",
    }


def _base_artifact_payload(kind: str, role: str) -> dict[str, object]:
    return {
        "artifact_kind": kind,
        "format_version": FORMAT_VERSION,
        "result_role": role,
        "counted_eligible": False,
        "backend": _backend_payload(),
        "counted_blockers": list(COUNTED_BLOCKERS),
        "production_evidence_gate": ProductionEvidenceGateV3().to_payload(),
    }


def _label_config_payload() -> dict[str, object]:
    config = DEFAULT_LABEL_CONFIG
    return {
        "frame_rate_hz": config.frame_rate_hz,
        "bin_edges_s": list(config.bin_edges_s),
        "active_ratio": config.active_ratio,
        "now_bins": list(config.now_bins),
        "future_bins": list(config.future_bins),
    }


@dataclass(frozen=True)
class FrozenTorchSpec:
    """Exact, hashable CPU training specification for the v3 scaffold."""

    frame_rate_hz: int = 25
    participant_features: tuple[str, ...] = tuple(TRANSFORMED_PARTICIPANT_FEATURES)
    global_features: tuple[str, ...] = tuple(TRANSFORMED_GLOBAL_FEATURES)
    n_participants: int = 4
    n_bins: int = 4
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    encoder: str = "transformer"
    attention_window: int = 250
    window_frames: int = 250
    stride_frames: int = 125
    inference_chunk_frames: int = 2000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    aux_vad_weight: float = 0.0
    device: str = "cpu"
    deterministic_algorithms: bool = True
    window_order: str = "seeded_epoch_permutation"

    def __post_init__(self) -> None:
        if not _is_exact_int(self.frame_rate_hz) or (
            self.frame_rate_hz != DEFAULT_LABEL_CONFIG.frame_rate_hz
        ):
            raise ProductionBackendGateError("frame_rate_hz must be exactly 25")
        if self.participant_features != tuple(TRANSFORMED_PARTICIPANT_FEATURES):
            raise ProductionBackendGateError("participant feature schema is not exact")
        if self.global_features != tuple(TRANSFORMED_GLOBAL_FEATURES):
            raise ProductionBackendGateError("global feature schema is not exact")
        if (
            not _is_exact_int(self.n_participants)
            or not _is_exact_int(self.n_bins)
            or self.n_participants != 4
            or self.n_bins != DEFAULT_LABEL_CONFIG.n_bins
        ):
            raise ProductionBackendGateError("participant/bin schema must be exactly 4 x 4")
        for name in (
            "d_model",
            "n_layers",
            "n_heads",
            "attention_window",
            "window_frames",
            "stride_frames",
            "inference_chunk_frames",
        ):
            _positive_int(getattr(self, name), name)
        if self.d_model % self.n_heads:
            raise ProductionBackendGateError("d_model must be divisible by n_heads")
        if self.encoder not in {"transformer", "gru"}:
            raise ProductionBackendGateError("encoder must be transformer or gru")
        if self.inference_chunk_frames <= self.attention_window * self.n_layers:
            raise ProductionBackendGateError(
                "inference_chunk_frames must exceed the stacked attention context"
            )
        if self.stride_frames > self.window_frames:
            raise ProductionBackendGateError("stride_frames cannot leave gaps between windows")
        for name in ("dropout", "learning_rate", "weight_decay", "gradient_clip"):
            if type(getattr(self, name)) is not float:
                raise ProductionBackendGateError(f"{name} must be an exact JSON float")
        dropout = _finite_float(self.dropout, "dropout")
        if not 0.0 <= dropout < 1.0:
            raise ProductionBackendGateError("dropout must be in [0, 1)")
        if _finite_float(self.learning_rate, "learning_rate") <= 0.0:
            raise ProductionBackendGateError("learning_rate must be > 0")
        if _finite_float(self.weight_decay, "weight_decay") < 0.0:
            raise ProductionBackendGateError("weight_decay must be >= 0")
        if _finite_float(self.gradient_clip, "gradient_clip") <= 0.0:
            raise ProductionBackendGateError("gradient_clip must be > 0")
        if type(self.aux_vad_weight) is not float or self.aux_vad_weight != 0.0:
            raise ProductionBackendGateError("aux_vad_weight must be exactly float 0.0")
        if self.device != "cpu" or self.deterministic_algorithms is not True:
            raise ProductionBackendGateError("the v3 scaffold is deterministic CPU-only")
        if self.window_order != "seeded_epoch_permutation":
            raise ProductionBackendGateError("window_order is frozen")

    @property
    def input_dim(self) -> int:
        return self.n_participants * len(self.participant_features) + len(self.global_features)

    def input_schema_payload(self) -> dict[str, object]:
        return {
            "frame_rate_hz": self.frame_rate_hz,
            "tick_column": "tick_25hz",
            "session_column": "session_id",
            "participant_features": list(self.participant_features),
            "global_features": list(self.global_features),
            "n_participants": self.n_participants,
            "n_bins": self.n_bins,
            "input_dim": self.input_dim,
        }

    @property
    def input_schema_sha256(self) -> str:
        return _json_sha256(self.input_schema_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": SPEC_KIND,
            "format_version": FORMAT_VERSION,
            **self.input_schema_payload(),
            "input_schema_sha256": self.input_schema_sha256,
            "label_config": _label_config_payload(),
            "label_config_sha256": _json_sha256(_label_config_payload()),
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "encoder": self.encoder,
            "attention_window": self.attention_window,
            "window_frames": self.window_frames,
            "stride_frames": self.stride_frames,
            "inference_chunk_frames": self.inference_chunk_frames,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "aux_vad_weight": self.aux_vad_weight,
            "device": self.device,
            "deterministic_algorithms": self.deterministic_algorithms,
            "window_order": self.window_order,
        }

    @property
    def sha256(self) -> str:
        return _json_sha256(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> FrozenTorchSpec:
        if not isinstance(payload, dict):
            raise ProductionBackendGateError("frozen torch spec must be a JSON object")
        expected = set(cls().to_payload())
        if set(payload) != expected:
            raise ProductionBackendGateError("frozen torch spec fields are not exact")
        if payload.get("artifact_kind") != SPEC_KIND or payload.get("format_version") != 3:
            raise ProductionBackendGateError("frozen torch spec kind/version mismatch")
        participant = payload.get("participant_features")
        global_features = payload.get("global_features")
        if not isinstance(participant, list) or not isinstance(global_features, list):
            raise ProductionBackendGateError("feature schemas must be JSON arrays")
        for name in ("dropout", "learning_rate", "weight_decay", "gradient_clip"):
            if type(payload.get(name)) is not float:
                raise ProductionBackendGateError(f"{name} must be an exact JSON float")
        spec = cls(
            frame_rate_hz=payload["frame_rate_hz"],
            participant_features=tuple(participant),
            global_features=tuple(global_features),
            n_participants=payload["n_participants"],
            n_bins=payload["n_bins"],
            d_model=payload["d_model"],
            n_layers=payload["n_layers"],
            n_heads=payload["n_heads"],
            dropout=_finite_float(payload["dropout"], "dropout"),
            encoder=payload["encoder"],
            attention_window=payload["attention_window"],
            window_frames=payload["window_frames"],
            stride_frames=payload["stride_frames"],
            inference_chunk_frames=payload["inference_chunk_frames"],
            learning_rate=_finite_float(payload["learning_rate"], "learning_rate"),
            weight_decay=_finite_float(payload["weight_decay"], "weight_decay"),
            gradient_clip=_finite_float(payload["gradient_clip"], "gradient_clip"),
            aux_vad_weight=payload["aux_vad_weight"],
            device=payload["device"],
            deterministic_algorithms=payload["deterministic_algorithms"],
            window_order=payload["window_order"],
        )
        if payload.get("tick_column") != "tick_25hz":
            raise ProductionBackendGateError("tick column must be tick_25hz")
        if payload.get("session_column") != "session_id":
            raise ProductionBackendGateError("session column must be session_id")
        if not _is_exact_int(payload.get("input_dim")) or payload.get(
            "input_dim"
        ) != spec.input_dim:
            raise ProductionBackendGateError("input_dim does not match the exact schema")
        if payload.get("input_schema_sha256") != spec.input_schema_sha256:
            raise ProductionBackendGateError("input schema hash mismatch")
        if _canonical_json(payload.get("label_config")) != _canonical_json(
            _label_config_payload()
        ):
            raise ProductionBackendGateError("label config differs from the frozen v3 contract")
        if payload.get("label_config_sha256") != _json_sha256(_label_config_payload()):
            raise ProductionBackendGateError("label config hash mismatch")
        return spec


@dataclass(frozen=True)
class SessionTickBinding:
    """Hash-bound frame identity for one fixture session."""

    session_id: str
    tick_25hz: tuple[int, ...]
    session_content_sha256: str
    data_kind: str = FIXTURE_DATA_KIND

    def __post_init__(self) -> None:
        _normalised_string(self.session_id, "session_id")
        if self.data_kind != FIXTURE_DATA_KIND:
            raise ProductionBackendGateError("v3 scaffold rejects non-fixture data")
        _sha256_string(self.session_content_sha256, "session_content_sha256")
        if not self.tick_25hz:
            raise ProductionBackendGateError("tick_25hz cannot be empty")
        if any(not _is_exact_int(tick) or tick < 0 for tick in self.tick_25hz):
            raise ProductionBackendGateError("tick_25hz values must be non-negative integers")
        if any(
            right != left + 1
            for left, right in zip(self.tick_25hz, self.tick_25hz[1:], strict=False)
        ):
            raise ProductionBackendGateError("tick_25hz must be strictly contiguous")

    def to_payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "data_kind": self.data_kind,
            "n_frames": len(self.tick_25hz),
            "tick_start": self.tick_25hz[0],
            "tick_end_exclusive": self.tick_25hz[-1] + 1,
            "tick_25hz": list(self.tick_25hz),
            "session_content_sha256": self.session_content_sha256,
        }


@dataclass(frozen=True)
class TickAuthorityV3:
    """Complete fixture-only session/tick authority supplied outside the backend."""

    bindings: tuple[SessionTickBinding, ...]

    def __post_init__(self) -> None:
        if not self.bindings:
            raise ProductionBackendGateError("tick authority cannot be empty")
        session_ids = [binding.session_id for binding in self.bindings]
        if session_ids != sorted(session_ids) or len(session_ids) != len(set(session_ids)):
            raise ProductionBackendGateError(
                "tick authority bindings must be unique and sorted by session_id"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": TICK_AUTHORITY_KIND,
            "format_version": FORMAT_VERSION,
            "data_kind": FIXTURE_DATA_KIND,
            "tick_rate_hz": 25,
            "bindings": [binding.to_payload() for binding in self.bindings],
        }

    @property
    def sha256(self) -> str:
        return _json_sha256(self.to_payload())

    def by_session(self) -> dict[str, SessionTickBinding]:
        return {binding.session_id: binding for binding in self.bindings}


def session_content_sha256(session: SessionTensors, tick_25hz: tuple[int, ...]) -> str:
    """Bind all tensor bytes and exact frame identities for a fixture session."""
    digest = hashlib.sha256()
    identity = {
        "session_id": session.session_id,
        "group_id": session.group_id,
        "tick_25hz": list(tick_25hz),
    }
    digest.update(_canonical_json(identity))
    for name in ("features", "labels", "vad", "valid", "present"):
        value = np.ascontiguousarray(getattr(session, name))
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(value.shape)))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedProductionCheckpointV3:
    model: MultipartyVAP
    normalizer: FeatureNormalizer
    pos_weight: np.ndarray
    payload: dict[str, object]
    spec: FrozenTorchSpec


class ProductionTorchBackendV3:
    """Fixture-only implementation of the future production backend protocol."""

    def __init__(self, spec: FrozenTorchSpec, tick_authority: TickAuthorityV3) -> None:
        self.spec = FrozenTorchSpec.from_payload(spec.to_payload())
        self.tick_authority = tick_authority
        self._evaluated_units: set[tuple[str, str, int, str]] = set()

    @property
    def counted_blockers(self) -> tuple[str, ...]:
        return COUNTED_BLOCKERS

    def fit_inner(self, request: InnerFitRequest) -> InnerFitReceipt:
        self._validate_request_spec(request.encoder)
        train = self._validate_sessions(request.train_sessions, "inner train")
        validation = self._validate_sessions(request.validation_sessions, "inner validation")
        self._require_disjoint(train, validation)
        max_epochs = _positive_int(request.max_epochs, "max_epochs")
        output_dir = self._output_directory(request.output_dir)
        normalizer = _validated_normalizer_copy(request.normalizer, self.spec)
        pos_weight = _validated_pos_weight_copy(request.pos_weight, self.spec.n_bins)
        normalizer_hash = _normalizer_sha256(normalizer)
        pos_weight_hash = _json_sha256(pos_weight.tolist())
        input_hashes = _session_hashes((*train, *validation), self.tick_authority)

        with _deterministic_cpu(request.seed):
            fit = _fit_model(
                self.spec,
                train,
                normalizer,
                pos_weight,
                epochs=max_epochs,
                seed=request.seed,
                validation_sessions=validation,
            )
        _assert_supplied_statistics_unchanged(
            request.normalizer,
            request.pos_weight,
            normalizer_hash,
            pos_weight_hash,
            self.spec,
        )
        _assert_session_hashes(input_hashes, (*train, *validation), self.tick_authority)

        checkpoint_path = output_dir / INNER_CHECKPOINT_FILENAME
        checkpoint_payload = _checkpoint_payload(
            spec=self.spec,
            authority=self.tick_authority,
            model=fit.model,
            normalizer=normalizer,
            pos_weight=pos_weight,
            role=INNER_RESULT_ROLE,
            condition_id=request.condition_id,
            feature_set=request.feature_set,
            outer_fold_id=request.outer_fold_id,
            inner_fold_id=request.inner_fold_id,
            seed=request.seed,
            epochs_completed=max_epochs,
            selected_epoch=fit.selected_epoch,
            train_session_ids=_session_ids(train),
            validation_session_ids=_session_ids(validation),
            history=fit.history,
        )
        _write_torch_exclusive(checkpoint_path, checkpoint_payload)
        loaded = load_production_checkpoint_v3(checkpoint_path)
        if loaded.payload["selected_epoch"] != fit.selected_epoch:
            raise ProductionBackendGateError("checkpoint round-trip changed selected_epoch")

        predictions = _predict_sessions(loaded.model, validation, normalizer, self.spec)
        oof_path = output_dir / INNER_OOF_FILENAME
        oof_payload = _prediction_payload(
            spec=self.spec,
            authority=self.tick_authority,
            sessions=validation,
            predictions=predictions,
            role=OOF_RESULT_ROLE,
            condition_id=request.condition_id,
            feature_set=request.feature_set,
            outer_fold_id=request.outer_fold_id,
            inner_fold_id=request.inner_fold_id,
            seed=request.seed,
            checkpoint_sha256=_sha256_file(checkpoint_path),
        )
        _write_json_exclusive(oof_path, oof_payload)
        load_prediction_artifact_v3(oof_path)
        _assert_session_hashes(input_hashes, (*train, *validation), self.tick_authority)

        train_ids = _session_ids(train)
        return InnerFitReceipt(
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            inner_fold_id=request.inner_fold_id,
            seed=request.seed,
            best_epoch=fit.selected_epoch,
            train_session_ids=train_ids,
            validation_session_ids=_session_ids(validation),
            normalizer_fit_session_ids=train_ids,
            pos_weight_fit_session_ids=train_ids,
            checkpoint_path=checkpoint_path,
            oof_predictions_path=oof_path,
        )

    def calibrate_inner_oof(self, request: CalibrationRequest) -> CalibrationReceipt:
        output_dir = self._output_directory(request.output_dir)
        if len(request.inner_fold_ids) != 3 or len(request.oof_prediction_paths) != 3:
            raise ProductionBackendGateError("calibration requires exactly three inner folds")
        if len(set(request.inner_fold_ids)) != 3:
            raise ProductionBackendGateError("inner fold identifiers must be unique")
        expected_sessions = tuple(request.oof_session_ids)
        if len(expected_sessions) != len(set(expected_sessions)):
            raise ProductionBackendGateError("pooled OOF session identities must be unique")

        observed_sessions: list[str] = []
        sources = []
        observed_feature_set: str | None = None
        for inner_fold_id, path_value in zip(
            request.inner_fold_ids, request.oof_prediction_paths, strict=True
        ):
            path = Path(path_value).resolve()
            payload = load_prediction_artifact_v3(path)
            _require_artifact_identity(
                payload,
                condition_id=request.condition_id,
                outer_fold_id=request.outer_fold_id,
                seed=request.seed,
                inner_fold_id=inner_fold_id,
            )
            if payload["result_role"] != OOF_RESULT_ROLE:
                raise ProductionBackendGateError("calibration accepts inner OOF artifacts only")
            if payload["spec_sha256"] != self.spec.sha256:
                raise ProductionBackendGateError("OOF artifact spec differs from backend spec")
            if payload["tick_authority_sha256"] != self.tick_authority.sha256:
                raise ProductionBackendGateError("OOF tick authority differs from backend")
            if observed_feature_set is None:
                observed_feature_set = payload["feature_set"]
            elif payload["feature_set"] != observed_feature_set:
                raise ProductionBackendGateError("OOF feature sets differ across inner folds")
            observed_sessions.extend(payload["session_ids"])
            sources.append(
                {
                    "inner_fold_id": inner_fold_id,
                    "relative_path": os.path.relpath(path, output_dir).replace("\\", "/"),
                    "sha256": _sha256_file(path),
                }
            )
        if tuple(observed_sessions) != expected_sessions:
            raise ProductionBackendGateError(
                "pooled OOF artifacts do not match the runner-supplied session order"
            )

        payload = {
            **_base_artifact_payload(CALIBRATION_KIND, CALIBRATION_RESULT_ROLE),
            "condition_id": _normalised_string(request.condition_id, "condition_id"),
            "feature_set": _normalised_string(observed_feature_set, "feature_set"),
            "outer_fold_id": _normalised_string(request.outer_fold_id, "outer_fold_id"),
            "seed": _positive_int(request.seed, "seed"),
            "spec_sha256": self.spec.sha256,
            "tick_authority_sha256": self.tick_authority.sha256,
            "inner_fold_ids": list(request.inner_fold_ids),
            "oof_session_ids": list(expected_sessions),
            "oof_sources": sources,
            "threshold_policy": "not_estimated_owner_approval_pending",
            "thresholds": None,
        }
        calibration_path = output_dir / CALIBRATION_FILENAME
        _write_json_exclusive(calibration_path, payload)
        load_calibration_artifact_v3(calibration_path)
        return CalibrationReceipt(
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            inner_fold_ids=request.inner_fold_ids,
            oof_prediction_paths=request.oof_prediction_paths,
            calibration_path=calibration_path,
        )

    def fit_final(self, request: FinalFitRequest) -> FinalFitReceipt:
        self._validate_request_spec(request.encoder)
        train = self._validate_sessions(request.train_sessions, "outer train")
        fixed_epochs = _positive_int(request.fixed_epochs, "fixed_epochs")
        calibration = load_calibration_artifact_v3(request.calibration_path)
        _require_artifact_identity(
            calibration,
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            inner_fold_id=None,
        )
        if calibration["spec_sha256"] != self.spec.sha256:
            raise ProductionBackendGateError("calibration spec differs from backend spec")
        if calibration["tick_authority_sha256"] != self.tick_authority.sha256:
            raise ProductionBackendGateError("calibration tick authority differs from backend")
        if calibration["feature_set"] != request.feature_set:
            raise ProductionBackendGateError("calibration feature_set differs from final request")
        if set(calibration["oof_session_ids"]) != set(_session_ids(train)):
            raise ProductionBackendGateError(
                "calibration OOF sessions must exactly cover final-train sessions"
            )
        output_dir = self._output_directory(request.output_dir)
        normalizer = _validated_normalizer_copy(request.normalizer, self.spec)
        pos_weight = _validated_pos_weight_copy(request.pos_weight, self.spec.n_bins)
        normalizer_hash = _normalizer_sha256(normalizer)
        pos_weight_hash = _json_sha256(pos_weight.tolist())
        input_hashes = _session_hashes(train, self.tick_authority)

        with _deterministic_cpu(request.seed):
            fit = _fit_model(
                self.spec,
                train,
                normalizer,
                pos_weight,
                epochs=fixed_epochs,
                seed=request.seed,
                validation_sessions=(),
            )
        _assert_supplied_statistics_unchanged(
            request.normalizer,
            request.pos_weight,
            normalizer_hash,
            pos_weight_hash,
            self.spec,
        )
        _assert_session_hashes(input_hashes, train, self.tick_authority)

        checkpoint_path = output_dir / FINAL_CHECKPOINT_FILENAME
        payload = _checkpoint_payload(
            spec=self.spec,
            authority=self.tick_authority,
            model=fit.model,
            normalizer=normalizer,
            pos_weight=pos_weight,
            role=FINAL_RESULT_ROLE,
            condition_id=request.condition_id,
            feature_set=request.feature_set,
            outer_fold_id=request.outer_fold_id,
            inner_fold_id=None,
            seed=request.seed,
            epochs_completed=fixed_epochs,
            selected_epoch=fixed_epochs,
            train_session_ids=_session_ids(train),
            validation_session_ids=(),
            history=fit.history,
        )
        _write_torch_exclusive(checkpoint_path, payload)
        load_production_checkpoint_v3(checkpoint_path)
        train_ids = _session_ids(train)
        return FinalFitReceipt(
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            fixed_epochs=fixed_epochs,
            train_session_ids=train_ids,
            normalizer_fit_session_ids=train_ids,
            pos_weight_fit_session_ids=train_ids,
            initialized_from_scratch=True,
            checkpoint_path=checkpoint_path,
        )

    def evaluate_outer(self, request: OuterEvaluationRequest) -> OuterEvaluationReceipt:
        test = self._validate_sessions(request.test_sessions, "outer test")
        checkpoint_path = Path(request.checkpoint_path).resolve()
        checkpoint_sha256 = _sha256_string(request.checkpoint_sha256, "checkpoint_sha256")
        if _sha256_file(checkpoint_path) != checkpoint_sha256:
            raise ProductionBackendGateError("sealed checkpoint hash mismatch")
        loaded = load_production_checkpoint_v3(checkpoint_path)
        _require_artifact_identity(
            loaded.payload,
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            inner_fold_id=None,
        )
        if loaded.payload["result_role"] != FINAL_RESULT_ROLE:
            raise ProductionBackendGateError("outer evaluation requires a final-fit checkpoint")
        if loaded.payload["tick_authority_sha256"] != self.tick_authority.sha256:
            raise ProductionBackendGateError("checkpoint tick authority differs from backend")
        calibration_path = Path(request.calibration_path).resolve()
        calibration = load_calibration_artifact_v3(calibration_path)
        _require_artifact_identity(
            calibration,
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            inner_fold_id=None,
        )
        if calibration["spec_sha256"] != loaded.spec.sha256:
            raise ProductionBackendGateError("calibration/checkpoint spec mismatch")
        if calibration["tick_authority_sha256"] != self.tick_authority.sha256:
            raise ProductionBackendGateError("calibration tick authority differs from backend")
        if calibration["feature_set"] != loaded.payload["feature_set"]:
            raise ProductionBackendGateError("calibration/checkpoint feature_set mismatch")
        if set(_session_ids(test)) & set(loaded.payload["train_session_ids"]):
            raise ProductionBackendGateError("outer-test sessions overlap checkpoint training")
        unit_key = (request.condition_id, request.outer_fold_id, request.seed, checkpoint_sha256)
        if unit_key in self._evaluated_units:
            raise ProductionBackendGateError(
                "outer test was already evaluated by this backend instance"
            )
        self._evaluated_units.add(unit_key)
        input_hashes = _session_hashes(test, self.tick_authority)
        output_dir = self._output_directory(request.output_dir)

        with _deterministic_cpu(request.seed):
            predictions = _predict_sessions(
                loaded.model, test, loaded.normalizer, loaded.spec
            )
        prediction_path = output_dir / OUTER_PREDICTIONS_FILENAME
        prediction_payload = _prediction_payload(
            spec=loaded.spec,
            authority=self.tick_authority,
            sessions=test,
            predictions=predictions,
            role=OUTER_RESULT_ROLE,
            condition_id=request.condition_id,
            feature_set=loaded.payload["feature_set"],
            outer_fold_id=request.outer_fold_id,
            inner_fold_id=None,
            seed=request.seed,
            checkpoint_sha256=checkpoint_sha256,
        )
        _write_json_exclusive(prediction_path, prediction_payload)
        checked_predictions = load_prediction_artifact_v3(prediction_path)
        metrics = _metrics_payload(
            checked_predictions,
            test,
            calibration_sha256=_sha256_file(calibration_path),
            prediction_sha256=_sha256_file(prediction_path),
        )
        metrics_path = output_dir / OUTER_METRICS_FILENAME
        _write_json_exclusive(metrics_path, metrics)
        load_metrics_artifact_v3(metrics_path)
        _assert_session_hashes(input_hashes, test, self.tick_authority)
        if _sha256_file(checkpoint_path) != checkpoint_sha256:
            raise ProductionBackendGateError("checkpoint changed during outer evaluation")

        return OuterEvaluationReceipt(
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            checkpoint_sha256=checkpoint_sha256,
            evaluation_session_ids=_session_ids(test),
            evaluation_count=1,
            predictions_path=prediction_path,
            metrics_path=metrics_path,
        )

    def _validate_request_spec(self, encoder: object) -> None:
        if encoder != self.spec.encoder:
            raise ProductionBackendGateError("request encoder differs from the frozen spec")

    def _validate_sessions(
        self, sessions: tuple[SessionTensors, ...], role: str
    ) -> tuple[SessionTensors, ...]:
        if not isinstance(sessions, tuple) or not sessions:
            raise ProductionBackendGateError(f"{role} sessions must be a non-empty tuple")
        _validate_sessions(sessions, self.spec, self.tick_authority)
        return sessions

    @staticmethod
    def _require_disjoint(
        left: tuple[SessionTensors, ...], right: tuple[SessionTensors, ...]
    ) -> None:
        if set(_session_ids(left)) & set(_session_ids(right)):
            raise ProductionBackendGateError("train and validation sessions overlap")

    @staticmethod
    def _output_directory(path_value: Path) -> Path:
        path = Path(path_value).resolve()
        if not path.exists() or not path.is_dir() or path.is_symlink():
            raise ProductionBackendGateError("backend output_dir must be an existing directory")
        return path


@dataclass
class _FitResult:
    model: MultipartyVAP
    history: dict[str, object]
    selected_epoch: int


def _model_config(spec: FrozenTorchSpec) -> VAPModelConfig:
    return VAPModelConfig(
        n_participant_features=len(spec.participant_features),
        n_global_features=len(spec.global_features),
        n_participants=spec.n_participants,
        n_bins=spec.n_bins,
        d_model=spec.d_model,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        dropout=spec.dropout,
        encoder=spec.encoder,
        attention_window=spec.attention_window,
    )


def _fit_model(
    spec: FrozenTorchSpec,
    train_sessions: tuple[SessionTensors, ...],
    normalizer: FeatureNormalizer,
    pos_weight: np.ndarray,
    *,
    epochs: int,
    seed: int,
    validation_sessions: tuple[SessionTensors, ...],
) -> _FitResult:
    """Deterministic CPU fit core; it never calls a statistics fitter."""
    torch.manual_seed(seed)
    model = MultipartyVAP(_model_config(spec)).cpu()
    decay = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    ]
    no_decay = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim < 2
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": spec.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=spec.learning_rate,
    )
    windows = _training_windows(train_sessions, spec.window_frames, spec.stride_frames)
    if not windows:
        raise ProductionBackendGateError("no training windows contain a valid VAP target")
    normalised = [normalizer.apply(session.features) for session in train_sessions]
    pos_tensor = torch.from_numpy(pos_weight.astype(np.float32, copy=True))
    entries: list[dict[str, object]] = []
    best_epoch: int | None = None
    best_map = -math.inf
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        order = generator.permutation(len(windows)).tolist()
        losses = []
        optimizer_steps = 0
        for window_index in order:
            session_index, start, end = windows[window_index]
            session = train_sessions[session_index]
            features = torch.from_numpy(normalised[session_index][start:end]).unsqueeze(0)
            labels = torch.from_numpy(session.labels[start:end]).unsqueeze(0)
            valid = torch.from_numpy(session.valid[start:end].astype(np.float32)).unsqueeze(0)
            present = torch.from_numpy(session.present.copy()).unsqueeze(0)
            mask = valid.unsqueeze(-1) * present.unsqueeze(1)
            if float(mask.sum()) == 0.0:
                continue
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features, present=present)
            loss = _primary_vap_loss(outputs["bin_logits"], labels, mask, pos_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            optimizer_steps += 1
        if not losses:
            raise ProductionBackendGateError("epoch has no future-horizon-valid optimizer step")

        validation_loss: float | None = None
        validation_map: float | None = None
        validation_frames = 0
        if validation_sessions:
            predictions = _predict_sessions(model, validation_sessions, normalizer, spec)
            frame_metrics = _frame_metrics_from_arrays(validation_sessions, predictions)
            validation_map = frame_metrics["mAP"]
            validation_loss = _probability_loss(
                validation_sessions, predictions, pos_weight
            )
            validation_frames = sum(session.features.shape[0] for session in validation_sessions)
            if validation_map is not None and validation_map > best_map:
                best_map = float(validation_map)
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
        entries.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
                "validation_mAP": validation_map,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "n_training_windows": len(windows),
                "n_optimizer_steps": optimizer_steps,
                "validation_frame_count": validation_frames,
            }
        )

    if validation_sessions:
        if best_epoch is None or best_state is None:
            raise ProductionBackendGateError(
                "strict validation mAP is unavailable; epoch selection is blocked"
            )
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
        history_role = "inner_selection"
    else:
        selected_epoch = epochs
        history_role = "fixed_epoch_final_fit"
    model.eval()
    history = {
        **_base_artifact_payload(HISTORY_KIND, "production_backend_v3_history_not_counted"),
        "history_role": history_role,
        "epoch_indexing": "one_based",
        "selected_epoch": selected_epoch,
        "entries": entries,
    }
    _validate_history_payload(history, expected_epochs=epochs)
    return _FitResult(model=model, history=history, selected_epoch=selected_epoch)


def _training_windows(
    sessions: tuple[SessionTensors, ...], window_frames: int, stride_frames: int
) -> list[tuple[int, int, int]]:
    """Return fixed/short spans with a final overlapping span that covers every tail."""
    result: list[tuple[int, int, int]] = []
    for session_index, session in enumerate(sessions):
        n_frames = session.features.shape[0]
        if n_frames <= window_frames:
            spans = [(0, n_frames)]
        else:
            starts = list(range(0, n_frames - window_frames + 1, stride_frames))
            tail_start = n_frames - window_frames
            if starts[-1] != tail_start:
                starts.append(tail_start)
            spans = [(start, start + window_frames) for start in starts]
        covered = np.zeros(n_frames, dtype=bool)
        for start, end in spans:
            covered[start:end] = True
            if session.valid[start:end].any() and (session.present > 0).any():
                result.append((session_index, start, end))
        if not covered.all():
            raise ProductionBackendGateError(
                f"training windows do not cover the tail of {session.session_id!r}"
            )
    return result


def _primary_vap_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight, reduction="none"
    )
    expanded = mask.unsqueeze(-1).expand_as(losses)
    return (losses * expanded).sum() / expanded.sum().clamp(min=1.0)


def _predict_sessions(
    model: MultipartyVAP,
    sessions: tuple[SessionTensors, ...],
    normalizer: FeatureNormalizer,
    spec: FrozenTorchSpec,
) -> tuple[np.ndarray, ...]:
    model.cpu().eval()
    outputs = []
    for session in sessions:
        probabilities = predict_session_probabilities(
            model,
            session,
            normalizer,
            device="cpu",
            chunk_frames=spec.inference_chunk_frames,
        )
        expected = (session.features.shape[0], spec.n_participants, spec.n_bins)
        if probabilities.shape != expected or not np.isfinite(probabilities).all():
            raise ProductionBackendGateError("full-session prediction shape/value failure")
        outputs.append(probabilities.astype(np.float32, copy=False))
    return tuple(outputs)


def _frame_metrics_from_arrays(
    sessions: tuple[SessionTensors, ...], predictions: tuple[np.ndarray, ...]
) -> dict[str, object]:
    labels = []
    scores = []
    participants = []
    for session, probability in zip(sessions, predictions, strict=True):
        present = session.present > 0
        labels.append(session.labels[session.valid][:, present, :].reshape(-1, 4))
        scores.append(probability[session.valid][:, present, :].reshape(-1, 4))
        participants.append(np.tile(np.flatnonzero(present), int(session.valid.sum())))
    if not labels or not any(value.shape[0] for value in labels):
        raise ProductionBackendGateError("evaluation partition has no valid labels")
    return compute_frame_metrics(
        np.concatenate(labels),
        np.concatenate(scores),
        participant_indices=np.concatenate(participants),
        max_participants=4,
    )


def _probability_loss(
    sessions: tuple[SessionTensors, ...],
    predictions: tuple[np.ndarray, ...],
    pos_weight: np.ndarray,
) -> float:
    total = 0.0
    count = 0
    epsilon = np.finfo(np.float32).eps
    for session, probability in zip(sessions, predictions, strict=True):
        mask = session.valid[:, None, None] & (session.present[None, :, None] > 0)
        mask = np.broadcast_to(mask, session.labels.shape)
        clipped = np.clip(probability, epsilon, 1.0 - epsilon)
        losses = -(
            session.labels * pos_weight.reshape(1, 1, -1) * np.log(clipped)
            + (1.0 - session.labels) * np.log1p(-clipped)
        )
        total += float(losses[mask].sum())
        count += int(mask.sum())
    if count == 0:
        raise ProductionBackendGateError("evaluation partition has no valid loss entries")
    return total / count


def _checkpoint_payload(
    *,
    spec: FrozenTorchSpec,
    authority: TickAuthorityV3,
    model: MultipartyVAP,
    normalizer: FeatureNormalizer,
    pos_weight: np.ndarray,
    role: str,
    condition_id: str,
    feature_set: str,
    outer_fold_id: str,
    inner_fold_id: str | None,
    seed: int,
    epochs_completed: int,
    selected_epoch: int,
    train_session_ids: tuple[str, ...],
    validation_session_ids: tuple[str, ...],
    history: dict[str, object],
) -> dict[str, object]:
    normalizer_payload = _normalizer_payload(normalizer)
    return {
        **_base_artifact_payload(CHECKPOINT_KIND, role),
        "spec": spec.to_payload(),
        "spec_sha256": spec.sha256,
        "input_schema_sha256": spec.input_schema_sha256,
        "tick_authority": authority.to_payload(),
        "tick_authority_sha256": authority.sha256,
        "condition_id": _normalised_string(condition_id, "condition_id"),
        "feature_set": _normalised_string(feature_set, "feature_set"),
        "outer_fold_id": _normalised_string(outer_fold_id, "outer_fold_id"),
        "inner_fold_id": inner_fold_id,
        "seed": _positive_int(seed, "seed"),
        "epochs_completed": _positive_int(epochs_completed, "epochs_completed"),
        "selected_epoch": _positive_int(selected_epoch, "selected_epoch"),
        "epoch_indexing": "one_based",
        "initialized_from_scratch": True,
        "train_session_ids": list(train_session_ids),
        "validation_session_ids": list(validation_session_ids),
        "normalizer": normalizer_payload,
        "normalizer_sha256": _json_sha256(normalizer_payload),
        "pos_weight": pos_weight.tolist(),
        "pos_weight_sha256": _json_sha256(pos_weight.tolist()),
        "aux_vad_weight": 0.0,
        "history": history,
        "history_sha256": _json_sha256(history),
        "model_state": {
            name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
        },
    }


def load_production_checkpoint_v3(path_value: str | Path) -> LoadedProductionCheckpointV3:
    """Strictly load a v3 checkpoint with PyTorch's weights-only unpickler."""
    path = Path(path_value).resolve()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ProductionBackendGateError(f"cannot load v3 checkpoint: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProductionBackendGateError("checkpoint must contain a mapping")
    expected = {
        *set(_base_artifact_payload(CHECKPOINT_KIND, INNER_RESULT_ROLE)),
        "spec",
        "spec_sha256",
        "input_schema_sha256",
        "tick_authority",
        "tick_authority_sha256",
        "condition_id",
        "feature_set",
        "outer_fold_id",
        "inner_fold_id",
        "seed",
        "epochs_completed",
        "selected_epoch",
        "epoch_indexing",
        "initialized_from_scratch",
        "train_session_ids",
        "validation_session_ids",
        "normalizer",
        "normalizer_sha256",
        "pos_weight",
        "pos_weight_sha256",
        "aux_vad_weight",
        "history",
        "history_sha256",
        "model_state",
    }
    if set(payload) != expected:
        raise ProductionBackendGateError("checkpoint fields are not exact")
    _validate_base_artifact(payload, CHECKPOINT_KIND, {INNER_RESULT_ROLE, FINAL_RESULT_ROLE})
    spec = FrozenTorchSpec.from_payload(payload["spec"])
    if payload["spec_sha256"] != spec.sha256:
        raise ProductionBackendGateError("checkpoint spec hash mismatch")
    if payload["input_schema_sha256"] != spec.input_schema_sha256:
        raise ProductionBackendGateError("checkpoint input schema hash mismatch")
    _validate_tick_authority_payload(payload["tick_authority"], payload["tick_authority_sha256"])
    _normalised_string(payload["condition_id"], "condition_id")
    _normalised_string(payload["feature_set"], "feature_set")
    _normalised_string(payload["outer_fold_id"], "outer_fold_id")
    inner_fold_id = payload["inner_fold_id"]
    if inner_fold_id is not None:
        _normalised_string(inner_fold_id, "inner_fold_id")
    for key in ("seed", "epochs_completed", "selected_epoch"):
        _positive_int(payload[key], key)
    if payload["selected_epoch"] > payload["epochs_completed"]:
        raise ProductionBackendGateError("selected_epoch exceeds epochs_completed")
    if payload["epoch_indexing"] != "one_based":
        raise ProductionBackendGateError("checkpoint epoch indexing mismatch")
    if payload["initialized_from_scratch"] is not True:
        raise ProductionBackendGateError("checkpoint must record from-scratch initialization")
    _string_list(payload["train_session_ids"], "train_session_ids", allow_empty=False)
    _string_list(payload["validation_session_ids"], "validation_session_ids", allow_empty=True)
    normalizer = _normalizer_from_payload(payload["normalizer"], spec)
    if payload["normalizer_sha256"] != _normalizer_sha256(normalizer):
        raise ProductionBackendGateError("checkpoint normalizer hash mismatch")
    pos_weight = _validated_pos_weight_copy(payload["pos_weight"], spec.n_bins)
    if payload["pos_weight_sha256"] != _json_sha256(pos_weight.tolist()):
        raise ProductionBackendGateError("checkpoint pos_weight hash mismatch")
    if type(payload["aux_vad_weight"]) is not float or payload["aux_vad_weight"] != 0.0:
        raise ProductionBackendGateError("checkpoint auxiliary loss is not isolated")
    _validate_history_payload(payload["history"], expected_epochs=payload["epochs_completed"])
    if payload["history_sha256"] != _json_sha256(payload["history"]):
        raise ProductionBackendGateError("checkpoint history hash mismatch")
    if payload["history"]["selected_epoch"] != payload["selected_epoch"]:
        raise ProductionBackendGateError("checkpoint/history selected epoch mismatch")
    if payload["result_role"] == INNER_RESULT_ROLE:
        if inner_fold_id is None or not payload["validation_session_ids"]:
            raise ProductionBackendGateError("inner checkpoint lacks inner/validation identity")
        if payload["history"]["history_role"] != "inner_selection":
            raise ProductionBackendGateError("inner checkpoint history role mismatch")
    else:
        if inner_fold_id is not None or payload["validation_session_ids"]:
            raise ProductionBackendGateError("final checkpoint contains inner-validation identity")
        if payload["history"]["history_role"] != "fixed_epoch_final_fit":
            raise ProductionBackendGateError("final checkpoint history role mismatch")
        if payload["selected_epoch"] != payload["epochs_completed"]:
            raise ProductionBackendGateError("final checkpoint must use every fixed epoch")

    model = MultipartyVAP(_model_config(spec)).cpu()
    model_state = payload["model_state"]
    if not isinstance(model_state, dict) or set(model_state) != set(model.state_dict()):
        raise ProductionBackendGateError("checkpoint model state keys are not exact")
    expected_state = model.state_dict()
    for name, tensor in model_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ProductionBackendGateError("checkpoint model state must contain tensors")
        if tensor.device.type != "cpu" or tensor.shape != expected_state[name].shape:
            raise ProductionBackendGateError(f"checkpoint tensor shape/device mismatch: {name}")
        if tensor.dtype != expected_state[name].dtype or not torch.isfinite(tensor).all():
            raise ProductionBackendGateError(f"checkpoint tensor dtype/value mismatch: {name}")
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return LoadedProductionCheckpointV3(model, normalizer, pos_weight, payload, spec)


def _prediction_payload(
    *,
    spec: FrozenTorchSpec,
    authority: TickAuthorityV3,
    sessions: tuple[SessionTensors, ...],
    predictions: tuple[np.ndarray, ...],
    role: str,
    condition_id: str,
    feature_set: str,
    outer_fold_id: str,
    inner_fold_id: str | None,
    seed: int,
    checkpoint_sha256: str,
) -> dict[str, object]:
    bindings = authority.by_session()
    records = []
    for session, probability in zip(sessions, predictions, strict=True):
        binding = bindings[session.session_id]
        records.append(
            {
                "session_id": session.session_id,
                "group_id": session.group_id,
                "n_frames": session.features.shape[0],
                "tick_start": binding.tick_25hz[0],
                "tick_end_exclusive": binding.tick_25hz[-1] + 1,
                "tick_25hz": list(binding.tick_25hz),
                "present": session.present.astype(float).tolist(),
                "target_available": session.valid.astype(bool).tolist(),
                "probabilities": probability.astype(float).tolist(),
                "targets": session.labels.astype(float).tolist(),
                "session_content_sha256": binding.session_content_sha256,
            }
        )
    return {
        **_base_artifact_payload(PREDICTION_KIND, role),
        "spec": spec.to_payload(),
        "spec_sha256": spec.sha256,
        "input_schema_sha256": spec.input_schema_sha256,
        "tick_authority": authority.to_payload(),
        "tick_authority_sha256": authority.sha256,
        "condition_id": _normalised_string(condition_id, "condition_id"),
        "feature_set": _normalised_string(feature_set, "feature_set"),
        "outer_fold_id": _normalised_string(outer_fold_id, "outer_fold_id"),
        "inner_fold_id": inner_fold_id,
        "seed": _positive_int(seed, "seed"),
        "checkpoint_sha256": _sha256_string(checkpoint_sha256, "checkpoint_sha256"),
        "frame_identity": "session_id_plus_tick_25hz",
        "full_session_inference": True,
        "session_ids": list(_session_ids(sessions)),
        "sessions": records,
    }


def load_prediction_artifact_v3(path_value: str | Path) -> dict[str, object]:
    payload = _read_json_strict(Path(path_value).resolve())
    expected = {
        *set(_base_artifact_payload(PREDICTION_KIND, OOF_RESULT_ROLE)),
        "spec",
        "spec_sha256",
        "input_schema_sha256",
        "tick_authority",
        "tick_authority_sha256",
        "condition_id",
        "feature_set",
        "outer_fold_id",
        "inner_fold_id",
        "seed",
        "checkpoint_sha256",
        "frame_identity",
        "full_session_inference",
        "session_ids",
        "sessions",
    }
    if set(payload) != expected:
        raise ProductionBackendGateError("prediction artifact fields are not exact")
    _validate_base_artifact(payload, PREDICTION_KIND, {OOF_RESULT_ROLE, OUTER_RESULT_ROLE})
    spec = FrozenTorchSpec.from_payload(payload["spec"])
    if payload["spec_sha256"] != spec.sha256:
        raise ProductionBackendGateError("prediction spec hash mismatch")
    if payload["input_schema_sha256"] != spec.input_schema_sha256:
        raise ProductionBackendGateError("prediction input schema hash mismatch")
    authority = _validate_tick_authority_payload(
        payload["tick_authority"], payload["tick_authority_sha256"]
    )
    _sha256_string(payload["checkpoint_sha256"], "checkpoint_sha256")
    _normalised_string(payload["condition_id"], "condition_id")
    _normalised_string(payload["feature_set"], "feature_set")
    _normalised_string(payload["outer_fold_id"], "outer_fold_id")
    if payload["inner_fold_id"] is not None:
        _normalised_string(payload["inner_fold_id"], "inner_fold_id")
    _positive_int(payload["seed"], "seed")
    if payload["frame_identity"] != "session_id_plus_tick_25hz":
        raise ProductionBackendGateError("prediction frame identity is not exact")
    if payload["full_session_inference"] is not True:
        raise ProductionBackendGateError("prediction must cover full sessions")
    session_ids = _string_list(payload["session_ids"], "session_ids", allow_empty=False)
    records = payload["sessions"]
    if not isinstance(records, list) or len(records) != len(session_ids):
        raise ProductionBackendGateError("prediction session records do not match session_ids")
    if [record.get("session_id") for record in records if isinstance(record, dict)] != session_ids:
        raise ProductionBackendGateError("prediction session order/identity mismatch")
    authority_bindings = authority.by_session()
    if any(session_id not in authority_bindings for session_id in session_ids):
        raise ProductionBackendGateError("prediction session is absent from tick authority")
    for record in records:
        _validate_prediction_record(record, spec)
        binding = authority_bindings[record["session_id"]]
        if record["tick_25hz"] != list(binding.tick_25hz):
            raise ProductionBackendGateError("prediction ticks differ from tick authority")
        if record["session_content_sha256"] != binding.session_content_sha256:
            raise ProductionBackendGateError("prediction content hash differs from tick authority")
    if payload["result_role"] == OOF_RESULT_ROLE and payload["inner_fold_id"] is None:
        raise ProductionBackendGateError("inner OOF artifact lacks inner_fold_id")
    if payload["result_role"] == OUTER_RESULT_ROLE and payload["inner_fold_id"] is not None:
        raise ProductionBackendGateError("outer prediction artifact contains inner_fold_id")
    return payload


def _validate_prediction_record(record: object, spec: FrozenTorchSpec) -> None:
    expected = {
        "session_id",
        "group_id",
        "n_frames",
        "tick_start",
        "tick_end_exclusive",
        "tick_25hz",
        "present",
        "target_available",
        "probabilities",
        "targets",
        "session_content_sha256",
    }
    if not isinstance(record, dict) or set(record) != expected:
        raise ProductionBackendGateError("prediction session fields are not exact")
    _normalised_string(record["session_id"], "prediction session_id")
    if record["group_id"] is not None:
        _normalised_string(record["group_id"], "prediction group_id")
    n_frames = _positive_int(record["n_frames"], "prediction n_frames")
    ticks = record["tick_25hz"]
    if not isinstance(ticks, list) or len(ticks) != n_frames:
        raise ProductionBackendGateError("prediction tick length mismatch")
    if any(not _is_exact_int(value) or value < 0 for value in ticks):
        raise ProductionBackendGateError("prediction ticks must be non-negative integers")
    if any(right != left + 1 for left, right in zip(ticks, ticks[1:], strict=False)):
        raise ProductionBackendGateError("prediction ticks must be contiguous")
    if (
        not _is_exact_int(record["tick_start"])
        or not _is_exact_int(record["tick_end_exclusive"])
        or record["tick_start"] != ticks[0]
        or record["tick_end_exclusive"] != ticks[-1] + 1
    ):
        raise ProductionBackendGateError("prediction tick bounds mismatch")
    present = _finite_array(record["present"], "prediction present", (spec.n_participants,))
    if not np.isin(present, (0.0, 1.0)).all():
        raise ProductionBackendGateError("prediction present mask must be binary")
    available = record["target_available"]
    if not isinstance(available, list) or len(available) != n_frames or any(
        type(value) is not bool for value in available
    ):
        raise ProductionBackendGateError("prediction target availability must be boolean")
    probabilities = _finite_array(
        record["probabilities"],
        "prediction probabilities",
        (n_frames, spec.n_participants, spec.n_bins),
    )
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ProductionBackendGateError("prediction probabilities must be in [0,1]")
    targets = _finite_array(
        record["targets"],
        "prediction targets",
        (n_frames, spec.n_participants, spec.n_bins),
    )
    if not np.isin(targets, (0.0, 1.0)).all():
        raise ProductionBackendGateError("prediction targets must be binary")
    _sha256_string(record["session_content_sha256"], "session_content_sha256")


def load_calibration_artifact_v3(path_value: str | Path) -> dict[str, object]:
    payload = _read_json_strict(Path(path_value).resolve())
    expected = {
        *set(_base_artifact_payload(CALIBRATION_KIND, CALIBRATION_RESULT_ROLE)),
        "condition_id",
        "feature_set",
        "outer_fold_id",
        "seed",
        "spec_sha256",
        "tick_authority_sha256",
        "inner_fold_ids",
        "oof_session_ids",
        "oof_sources",
        "threshold_policy",
        "thresholds",
    }
    if set(payload) != expected:
        raise ProductionBackendGateError("calibration fields are not exact")
    _validate_base_artifact(payload, CALIBRATION_KIND, {CALIBRATION_RESULT_ROLE})
    _normalised_string(payload["condition_id"], "condition_id")
    _normalised_string(payload["feature_set"], "feature_set")
    _normalised_string(payload["outer_fold_id"], "outer_fold_id")
    _positive_int(payload["seed"], "seed")
    _sha256_string(payload["spec_sha256"], "spec_sha256")
    _sha256_string(payload["tick_authority_sha256"], "tick_authority_sha256")
    folds = _string_list(payload["inner_fold_ids"], "inner_fold_ids", allow_empty=False)
    sessions = _string_list(payload["oof_session_ids"], "oof_session_ids", allow_empty=False)
    if len(folds) != 3 or len(set(folds)) != 3 or len(set(sessions)) != len(sessions):
        raise ProductionBackendGateError("calibration OOF identity is malformed")
    sources = payload["oof_sources"]
    if not isinstance(sources, list) or len(sources) != 3:
        raise ProductionBackendGateError("calibration must bind three OOF sources")
    for expected_fold, source in zip(folds, sources, strict=True):
        if not isinstance(source, dict) or set(source) != {
            "inner_fold_id",
            "relative_path",
            "sha256",
        }:
            raise ProductionBackendGateError("calibration OOF source fields are not exact")
        if source["inner_fold_id"] != expected_fold:
            raise ProductionBackendGateError("calibration OOF fold order mismatch")
        _normalised_string(source["relative_path"], "OOF relative_path")
        _sha256_string(source["sha256"], "OOF sha256")
    if payload["threshold_policy"] != "not_estimated_owner_approval_pending":
        raise ProductionBackendGateError("calibration threshold policy is not the blocker")
    if payload["thresholds"] is not None:
        raise ProductionBackendGateError("scaffold calibration cannot contain thresholds")
    return payload


def _metrics_payload(
    prediction_payload: dict[str, object],
    sessions: tuple[SessionTensors, ...],
    *,
    calibration_sha256: str,
    prediction_sha256: str,
) -> dict[str, object]:
    probabilities = tuple(
        np.asarray(record["probabilities"], dtype=np.float32)
        for record in prediction_payload["sessions"]
    )
    frame = _frame_metrics_from_arrays(sessions, probabilities)
    pooled = {
        "shift_hold": ([], []),
        "shift_prediction": ([], []),
        "overlap_resolution": ([], []),
        "robot_entry": ([], []),
    }
    next_labels: list[int] = []
    next_scores: list[np.ndarray] = []
    functions = {
        "shift_hold": shift_hold_scores,
        "shift_prediction": shift_prediction_scores,
        "overlap_resolution": overlap_resolution_scores,
    }
    for session, probability in zip(sessions, probabilities, strict=True):
        for name, function in functions.items():
            labels, scores = function(
                probability,
                session.vad,
                frame_rate_hz=25,
                label_config=DEFAULT_LABEL_CONFIG,
                present=session.present,
            )
            pooled[name][0].extend(labels)
            pooled[name][1].extend(scores)
        labels, scores = robot_entry_scores(
            probability,
            session.vad,
            robot_slot=0,
            frame_rate_hz=25,
            label_config=DEFAULT_LABEL_CONFIG,
            present=session.present,
        )
        pooled["robot_entry"][0].extend(labels)
        pooled["robot_entry"][1].extend(scores)
        labels, scores = next_speaker_scores(
            probability,
            session.vad,
            frame_rate_hz=25,
            label_config=DEFAULT_LABEL_CONFIG,
            present=session.present,
        )
        next_labels.extend(labels)
        next_scores.extend(scores)
    events = {
        name: asdict(score_events(np.asarray(labels), np.asarray(scores), 0.5))
        for name, (labels, scores) in pooled.items()
    }
    next_array = (
        np.asarray(next_scores, dtype=float)
        if next_scores
        else np.empty((0, 4), dtype=float)
    )
    events["next_speaker"] = asdict(
        score_next_speaker(np.asarray(next_labels), next_array)
    )
    return {
        **_base_artifact_payload(METRICS_KIND, METRICS_RESULT_ROLE),
        "condition_id": prediction_payload["condition_id"],
        "feature_set": prediction_payload["feature_set"],
        "outer_fold_id": prediction_payload["outer_fold_id"],
        "seed": prediction_payload["seed"],
        "prediction_sha256": _sha256_string(prediction_sha256, "prediction_sha256"),
        "calibration_sha256": _sha256_string(calibration_sha256, "calibration_sha256"),
        "threshold_status": "diagnostic_fixed_0.5_owner_policy_pending",
        "frame": _json_safe(frame),
        "events": _json_safe(events),
    }


def load_metrics_artifact_v3(path_value: str | Path) -> dict[str, object]:
    payload = _read_json_strict(Path(path_value).resolve())
    expected = {
        *set(_base_artifact_payload(METRICS_KIND, METRICS_RESULT_ROLE)),
        "condition_id",
        "feature_set",
        "outer_fold_id",
        "seed",
        "prediction_sha256",
        "calibration_sha256",
        "threshold_status",
        "frame",
        "events",
    }
    if set(payload) != expected:
        raise ProductionBackendGateError("metrics artifact fields are not exact")
    _validate_base_artifact(payload, METRICS_KIND, {METRICS_RESULT_ROLE})
    _normalised_string(payload["condition_id"], "condition_id")
    _normalised_string(payload["feature_set"], "feature_set")
    _normalised_string(payload["outer_fold_id"], "outer_fold_id")
    _positive_int(payload["seed"], "seed")
    _sha256_string(payload["prediction_sha256"], "prediction_sha256")
    _sha256_string(payload["calibration_sha256"], "calibration_sha256")
    if payload["threshold_status"] != "diagnostic_fixed_0.5_owner_policy_pending":
        raise ProductionBackendGateError("metrics threshold status mismatch")
    if not isinstance(payload["frame"], dict) or not isinstance(payload["events"], dict):
        raise ProductionBackendGateError("metrics frame/events must be JSON objects")
    _validate_frame_metrics_schema(payload["frame"])
    _validate_event_metrics_schema(payload["events"])
    return payload


def _validate_frame_metrics_schema(frame: dict[str, object]) -> None:
    expected = {
        "mAP",
        "mAP_na_reason",
        "mAP_evaluable_bins",
        "n_bins",
        "n_bins_evaluable_ap",
        "micro_AP",
        "micro_AP_na_reason",
        "macro_f1_bins",
        "balanced_accuracy",
        "balanced_accuracy_na_reason",
        "f1",
        "micro_f1",
        "positive_rate",
        "n_positive",
        "n_negative",
        "bins",
        "participants",
        "participant_support_available",
        "participant_macro_mAP",
        "participant_macro_mAP_na_reason",
        *(f"AP_bin{index}" for index in range(4)),
        *(f"F1_bin{index}" for index in range(4)),
    }
    if set(frame) != expected:
        raise ProductionBackendGateError("frame metrics fields are not exact")
    if frame["n_bins"] != 4 or not _is_exact_int(frame["n_bins"]):
        raise ProductionBackendGateError("frame metrics n_bins must be exactly four")
    for key in ("n_bins_evaluable_ap", "n_positive", "n_negative"):
        _nonnegative_int(frame[key], f"frame metrics {key}")
    if frame["participant_support_available"] is not True:
        raise ProductionBackendGateError("participant support must be available")
    for key in (
        "mAP",
        "mAP_evaluable_bins",
        "micro_AP",
        "macro_f1_bins",
        "balanced_accuracy",
        "f1",
        "micro_f1",
        "positive_rate",
        "participant_macro_mAP",
        *(f"AP_bin{index}" for index in range(4)),
        *(f"F1_bin{index}" for index in range(4)),
    ):
        _nullable_unit_metric(frame[key], f"frame metrics {key}")
    for key in (
        "mAP_na_reason",
        "micro_AP_na_reason",
        "balanced_accuracy_na_reason",
        "participant_macro_mAP_na_reason",
    ):
        _nullable_reason(frame[key], f"frame metrics {key}")
    bins = frame["bins"]
    if not isinstance(bins, dict) or set(bins) != {f"bin{index}" for index in range(4)}:
        raise ProductionBackendGateError("frame bin metrics fields are not exact")
    for value in bins.values():
        _validate_binary_summary(value, include_f1=True)
    participants = frame["participants"]
    if not isinstance(participants, dict) or set(participants) != {
        f"P{index}" for index in range(1, 5)
    }:
        raise ProductionBackendGateError("participant metric fields are not exact")
    participant_expected = {"in_scope", "n_samples", "mAP", "mAP_na_reason", "bins"}
    for participant in participants.values():
        if not isinstance(participant, dict) or set(participant) != participant_expected:
            raise ProductionBackendGateError("participant metric schema is not exact")
        if type(participant["in_scope"]) is not bool:
            raise ProductionBackendGateError("participant in_scope must be boolean")
        _nonnegative_int(participant["n_samples"], "participant n_samples")
        _nullable_unit_metric(participant["mAP"], "participant mAP")
        _nullable_reason(participant["mAP_na_reason"], "participant mAP_na_reason")
        participant_bins = participant["bins"]
        if not isinstance(participant_bins, dict) or set(participant_bins) != {
            f"bin{index}" for index in range(4)
        }:
            raise ProductionBackendGateError("participant bin schema is not exact")
        for value in participant_bins.values():
            _validate_binary_summary(value, include_f1=False)


def _validate_event_metrics_schema(events: dict[str, object]) -> None:
    binary_names = {"shift_hold", "shift_prediction", "overlap_resolution", "robot_entry"}
    if set(events) != {*binary_names, "next_speaker"}:
        raise ProductionBackendGateError("event metrics fields are not exact")
    expected = {
        "n_events",
        "n_positive",
        "n_negative",
        "balanced_accuracy",
        "balanced_accuracy_na_reason",
        "f1",
        "f1_na_reason",
        "average_precision",
        "average_precision_na_reason",
    }
    for name in binary_names:
        value = events[name]
        if not isinstance(value, dict) or set(value) != expected:
            raise ProductionBackendGateError(f"{name} metric schema is not exact")
        for key in ("n_events", "n_positive", "n_negative"):
            _nonnegative_int(value[key], f"{name} {key}")
        for key in ("balanced_accuracy", "f1", "average_precision"):
            _nullable_unit_metric(value[key], f"{name} {key}")
        for key in (
            "balanced_accuracy_na_reason",
            "f1_na_reason",
            "average_precision_na_reason",
        ):
            _nullable_reason(value[key], f"{name} {key}")
    next_speaker = events["next_speaker"]
    expected_next = {
        "n_events",
        "n_participants_in_scope",
        "n_participants_evaluable_ap",
        "participant_macro_ovr_ap",
        "participant_macro_ovr_ap_na_reason",
        "balanced_recall",
        "mrr",
        "top1_accuracy",
        "tie_policy",
        "tie_rate",
        "per_participant",
        "na_reason",
    }
    if not isinstance(next_speaker, dict) or set(next_speaker) != expected_next:
        raise ProductionBackendGateError("next-speaker metric schema is not exact")
    for key in ("n_events", "n_participants_in_scope", "n_participants_evaluable_ap"):
        _nonnegative_int(next_speaker[key], f"next-speaker {key}")
    for key in (
        "participant_macro_ovr_ap",
        "balanced_recall",
        "mrr",
        "top1_accuracy",
        "tie_rate",
    ):
        _nullable_unit_metric(next_speaker[key], f"next-speaker {key}")
    for key in ("participant_macro_ovr_ap_na_reason", "na_reason"):
        _nullable_reason(next_speaker[key], f"next-speaker {key}")
    if next_speaker["tie_policy"] != "uniform_credit_among_max_scores":
        raise ProductionBackendGateError("next-speaker tie policy mismatch")
    per_participant = next_speaker["per_participant"]
    if not isinstance(per_participant, dict) or set(per_participant) != {
        f"P{index}" for index in range(1, 5)
    }:
        raise ProductionBackendGateError("next-speaker participant schema is not exact")
    participant_expected = {
        "in_scope",
        "n_positive",
        "n_negative",
        "ovr_average_precision",
        "tie_aware_recall",
        "ap_na_reason",
        "recall_na_reason",
    }
    for value in per_participant.values():
        if not isinstance(value, dict) or set(value) != participant_expected:
            raise ProductionBackendGateError("next-speaker participant fields are not exact")
        if type(value["in_scope"]) is not bool:
            raise ProductionBackendGateError("next-speaker in_scope must be boolean")
        for key in ("n_positive", "n_negative"):
            _nonnegative_int(value[key], f"next-speaker {key}")
        for key in ("ovr_average_precision", "tie_aware_recall"):
            _nullable_unit_metric(value[key], f"next-speaker {key}")
        for key in ("ap_na_reason", "recall_na_reason"):
            _nullable_reason(value[key], f"next-speaker {key}")


def _validate_binary_summary(value: object, *, include_f1: bool) -> None:
    expected = {"n_positive", "n_negative", "average_precision", "ap_na_reason"}
    if include_f1:
        expected.add("f1")
    if not isinstance(value, dict) or set(value) != expected:
        raise ProductionBackendGateError("binary metric summary fields are not exact")
    _nonnegative_int(value["n_positive"], "binary n_positive")
    _nonnegative_int(value["n_negative"], "binary n_negative")
    _nullable_unit_metric(value["average_precision"], "binary average_precision")
    _nullable_reason(value["ap_na_reason"], "binary ap_na_reason")
    if include_f1:
        _nullable_unit_metric(value["f1"], "binary f1")


def _nonnegative_int(value: object, name: str) -> int:
    if not _is_exact_int(value) or value < 0:
        raise ProductionBackendGateError(f"{name} must be a non-negative integer")
    return value


def _nullable_unit_metric(value: object, name: str) -> None:
    if value is None:
        return
    metric = _finite_float(value, name)
    if not 0.0 <= metric <= 1.0:
        raise ProductionBackendGateError(f"{name} must be in [0,1] or null")


def _nullable_reason(value: object, name: str) -> None:
    if value is not None:
        _normalised_string(value, name)


def _validate_base_artifact(
    payload: dict[str, object], kind: str, allowed_roles: set[str]
) -> None:
    if payload.get("artifact_kind") != kind or payload.get("format_version") != FORMAT_VERSION:
        raise ProductionBackendGateError("artifact kind/version mismatch")
    if payload.get("result_role") not in allowed_roles:
        raise ProductionBackendGateError("artifact result role is not allowed")
    if payload.get("counted_eligible") is not False:
        raise ProductionBackendGateError("scaffold artifact cannot be counted eligible")
    if payload.get("backend") != _backend_payload():
        raise ProductionBackendGateError("artifact backend identity mismatch")
    if payload.get("counted_blockers") != list(COUNTED_BLOCKERS):
        raise ProductionBackendGateError("artifact counted blockers mismatch")
    if payload.get("production_evidence_gate") != ProductionEvidenceGateV3().to_payload():
        raise ProductionBackendGateError("artifact production evidence gate mismatch")


def _validate_tick_authority_payload(
    payload: object, expected_sha256: object
) -> TickAuthorityV3:
    if not isinstance(payload, dict) or set(payload) != {
        "artifact_kind",
        "format_version",
        "data_kind",
        "tick_rate_hz",
        "bindings",
    }:
        raise ProductionBackendGateError("tick authority payload fields are not exact")
    if payload["artifact_kind"] != TICK_AUTHORITY_KIND or payload["format_version"] != 3:
        raise ProductionBackendGateError("tick authority kind/version mismatch")
    if (
        payload["data_kind"] != FIXTURE_DATA_KIND
        or not _is_exact_int(payload["tick_rate_hz"])
        or payload["tick_rate_hz"] != 25
    ):
        raise ProductionBackendGateError("tick authority is not a 25 Hz fixture authority")
    bindings = payload["bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise ProductionBackendGateError("tick authority bindings are missing")
    rebuilt = []
    for value in bindings:
        expected = {
            "session_id",
            "data_kind",
            "n_frames",
            "tick_start",
            "tick_end_exclusive",
            "tick_25hz",
            "session_content_sha256",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ProductionBackendGateError("tick binding fields are not exact")
        ticks = value["tick_25hz"]
        if not isinstance(ticks, list):
            raise ProductionBackendGateError("tick binding ticks must be an array")
        binding = SessionTickBinding(
            session_id=value["session_id"],
            data_kind=value["data_kind"],
            tick_25hz=tuple(ticks),
            session_content_sha256=value["session_content_sha256"],
        )
        if value != binding.to_payload():
            raise ProductionBackendGateError("tick binding redundant fields mismatch")
        rebuilt.append(binding)
    authority = TickAuthorityV3(tuple(rebuilt))
    if _sha256_string(expected_sha256, "tick_authority_sha256") != authority.sha256:
        raise ProductionBackendGateError("tick authority hash mismatch")
    return authority


def _validate_history_payload(payload: object, *, expected_epochs: int) -> None:
    expected = {
        *set(_base_artifact_payload(HISTORY_KIND, "production_backend_v3_history_not_counted")),
        "history_role",
        "epoch_indexing",
        "selected_epoch",
        "entries",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ProductionBackendGateError("history fields are not exact")
    _validate_base_artifact(
        payload, HISTORY_KIND, {"production_backend_v3_history_not_counted"}
    )
    if payload["history_role"] not in {"inner_selection", "fixed_epoch_final_fit"}:
        raise ProductionBackendGateError("history role mismatch")
    if payload["epoch_indexing"] != "one_based":
        raise ProductionBackendGateError("history must use one-based epochs")
    selected = _positive_int(payload["selected_epoch"], "history selected_epoch")
    if selected > expected_epochs:
        raise ProductionBackendGateError("history selected_epoch exceeds epoch count")
    entries = payload["entries"]
    if not isinstance(entries, list) or len(entries) != expected_epochs:
        raise ProductionBackendGateError("history epoch count mismatch")
    expected_entry = {
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_mAP",
        "learning_rate",
        "n_training_windows",
        "n_optimizer_steps",
        "validation_frame_count",
    }
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or set(entry) != expected_entry:
            raise ProductionBackendGateError("history entry fields are not exact")
        if entry["epoch"] != index or isinstance(entry["epoch"], bool):
            raise ProductionBackendGateError("history epochs must be contiguous and one-based")
        for key in ("train_loss", "learning_rate"):
            _finite_float(entry[key], f"history {key}")
        for key in ("validation_loss", "validation_mAP"):
            if entry[key] is not None:
                _finite_float(entry[key], f"history {key}")
        for key in ("n_training_windows", "n_optimizer_steps"):
            _positive_int(entry[key], f"history {key}")
        if not _is_exact_int(entry["validation_frame_count"]) or entry[
            "validation_frame_count"
        ] < 0:
            raise ProductionBackendGateError("validation_frame_count must be non-negative")
    if payload["history_role"] == "inner_selection":
        if any(
            entry["validation_loss"] is None
            or entry["validation_mAP"] is None
            or entry["validation_frame_count"] < 1
            for entry in entries
        ):
            raise ProductionBackendGateError("inner history requires evaluable full-session OOF")
        best_epoch = max(entries, key=lambda entry: entry["validation_mAP"])["epoch"]
        if selected != best_epoch:
            raise ProductionBackendGateError("inner selected epoch is not the first best mAP")
    elif any(
        entry["validation_loss"] is not None
        or entry["validation_mAP"] is not None
        or entry["validation_frame_count"] != 0
        for entry in entries
    ):
        raise ProductionBackendGateError("final-fit history cannot contain validation outcomes")


def _validated_normalizer_copy(
    normalizer: FeatureNormalizer, spec: FrozenTorchSpec
) -> FeatureNormalizer:
    if not isinstance(normalizer, FeatureNormalizer):
        raise ProductionBackendGateError("normalizer must be runner-supplied FeatureNormalizer")
    arrays = {
        "participant_mean": (normalizer.participant_mean, len(spec.participant_features), False),
        "participant_std": (normalizer.participant_std, len(spec.participant_features), True),
        "global_mean": (normalizer.global_mean, len(spec.global_features), False),
        "global_std": (normalizer.global_std, len(spec.global_features), True),
    }
    copied = {}
    for name, (value, length, require_positive) in arrays.items():
        array = np.asarray(value)
        if (
            array.shape != (length,)
            or array.dtype != np.dtype(np.float32)
            or not np.isfinite(array).all()
        ):
            raise ProductionBackendGateError(f"normalizer {name} shape/dtype/value failure")
        if require_positive and (array <= 0.0).any():
            raise ProductionBackendGateError(f"normalizer {name} must be positive")
        copied[name] = array.copy()
    return FeatureNormalizer(**copied)


def _normalizer_payload(normalizer: FeatureNormalizer) -> dict[str, object]:
    return {
        "participant_mean": normalizer.participant_mean.tolist(),
        "participant_std": normalizer.participant_std.tolist(),
        "global_mean": normalizer.global_mean.tolist(),
        "global_std": normalizer.global_std.tolist(),
    }


def _normalizer_sha256(normalizer: FeatureNormalizer) -> str:
    return _json_sha256(_normalizer_payload(normalizer))


def _normalizer_from_payload(
    payload: object, spec: FrozenTorchSpec
) -> FeatureNormalizer:
    if not isinstance(payload, dict) or set(payload) != {
        "participant_mean",
        "participant_std",
        "global_mean",
        "global_std",
    }:
        raise ProductionBackendGateError("normalizer payload fields are not exact")
    vectors = {}
    lengths = {
        "participant_mean": len(spec.participant_features),
        "participant_std": len(spec.participant_features),
        "global_mean": len(spec.global_features),
        "global_std": len(spec.global_features),
    }
    for name, length in lengths.items():
        value = payload[name]
        if (
            not isinstance(value, list)
            or len(value) != length
            or any(type(item) is not float or not math.isfinite(item) for item in value)
        ):
            raise ProductionBackendGateError(
                f"normalizer {name} must contain exact finite JSON floats"
            )
        vectors[name] = np.asarray(value, dtype=np.float32)
    normalizer = FeatureNormalizer(**vectors)
    return _validated_normalizer_copy(normalizer, spec)


def _validated_pos_weight_copy(value: object, n_bins: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (n_bins,) or array.dtype not in {
        np.dtype(np.float32),
        np.dtype(np.float64),
    }:
        raise ProductionBackendGateError("pos_weight must be a finite vector with one value/bin")
    if not np.isfinite(array).all() or (array <= 0.0).any():
        raise ProductionBackendGateError("pos_weight values must be finite and positive")
    return array.astype(np.float64, copy=True)


def _assert_supplied_statistics_unchanged(
    normalizer: FeatureNormalizer,
    pos_weight: np.ndarray,
    normalizer_sha256: str,
    pos_weight_sha256: str,
    spec: FrozenTorchSpec,
) -> None:
    if _normalizer_sha256(_validated_normalizer_copy(normalizer, spec)) != normalizer_sha256:
        raise ProductionBackendGateError("runner-supplied normalizer changed during fit")
    current_pos_weight = _validated_pos_weight_copy(pos_weight, spec.n_bins)
    if _json_sha256(current_pos_weight.tolist()) != pos_weight_sha256:
        raise ProductionBackendGateError("runner-supplied pos_weight changed during fit")


def _validate_sessions(
    sessions: tuple[SessionTensors, ...],
    spec: FrozenTorchSpec,
    authority: TickAuthorityV3,
) -> None:
    ids = _session_ids(sessions)
    if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
        raise ProductionBackendGateError("session tuples must be unique and sorted")
    bindings = authority.by_session()
    missing = [session_id for session_id in ids if session_id not in bindings]
    if missing:
        raise ProductionBackendGateError(f"tick authority is missing sessions: {missing}")
    for session in sessions:
        _normalised_string(session.session_id, "session_id")
        n_frames = session.features.shape[0] if session.features.ndim == 2 else -1
        expected_shapes = {
            "features": (n_frames, spec.input_dim),
            "labels": (n_frames, spec.n_participants, spec.n_bins),
            "vad": (n_frames, spec.n_participants),
            "valid": (n_frames,),
            "present": (spec.n_participants,),
        }
        for name, expected in expected_shapes.items():
            value = np.asarray(getattr(session, name))
            if value.shape != expected:
                raise ProductionBackendGateError(
                    f"session {session.session_id!r} {name} shape mismatch"
                )
            if name != "valid" and value.dtype != np.dtype(np.float32):
                raise ProductionBackendGateError(
                    f"session {session.session_id!r} {name} must be float32"
                )
            if value.dtype.kind in "f" and not np.isfinite(value).all():
                raise ProductionBackendGateError(
                    f"session {session.session_id!r} {name} contains NaN/Inf"
                )
        if n_frames < 1:
            raise ProductionBackendGateError("sessions cannot be empty")
        if not np.isin(session.labels, (0.0, 1.0)).all():
            raise ProductionBackendGateError("session labels must be binary")
        if not np.isin(session.vad, (0.0, 1.0)).all():
            raise ProductionBackendGateError("session VAD must be binary")
        if not np.isin(session.present, (0.0, 1.0)).all() or not session.present.any():
            raise ProductionBackendGateError("session present mask must be binary and non-empty")
        if session.valid.dtype != np.dtype(bool):
            raise ProductionBackendGateError("session valid mask must have boolean dtype")
        binding = bindings[session.session_id]
        if len(binding.tick_25hz) != n_frames:
            raise ProductionBackendGateError("session length differs from tick authority")
        if session_content_sha256(session, binding.tick_25hz) != binding.session_content_sha256:
            raise ProductionBackendGateError("session bytes differ from tick authority")


def _session_ids(sessions: tuple[SessionTensors, ...]) -> tuple[str, ...]:
    return tuple(session.session_id for session in sessions)


def _session_hashes(
    sessions: tuple[SessionTensors, ...], authority: TickAuthorityV3
) -> dict[str, str]:
    bindings = authority.by_session()
    return {
        session.session_id: session_content_sha256(
            session, bindings[session.session_id].tick_25hz
        )
        for session in sessions
    }


def _assert_session_hashes(
    expected: dict[str, str],
    sessions: tuple[SessionTensors, ...],
    authority: TickAuthorityV3,
) -> None:
    if _session_hashes(sessions, authority) != expected:
        raise ProductionBackendGateError("backend mutated runner-supplied session tensors")


def _finite_array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ProductionBackendGateError(f"{name} must be numeric") from exc
    if raw.dtype.kind not in "fiu":
        raise ProductionBackendGateError(f"{name} must contain JSON numbers, not booleans/text")
    array = raw.astype(float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ProductionBackendGateError(f"{name} shape/value failure")
    return array


def _string_list(value: object, name: str, *, allow_empty: bool) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ProductionBackendGateError(f"{name} must be a JSON string array")
    result = [_normalised_string(item, name) for item in value]
    if len(result) != len(set(result)):
        raise ProductionBackendGateError(f"{name} must contain unique values")
    return result


def _require_artifact_identity(
    payload: dict[str, object],
    *,
    condition_id: str,
    outer_fold_id: str,
    seed: int,
    inner_fold_id: str | None,
) -> None:
    expected = {
        "condition_id": condition_id,
        "outer_fold_id": outer_fold_id,
        "seed": seed,
        "inner_fold_id": inner_fold_id,
    }
    for key, value in expected.items():
        if key in payload and payload[key] != value:
            raise ProductionBackendGateError(f"artifact {key} identity mismatch")


@contextmanager
def _deterministic_cpu(seed: int) -> Iterator[None]:
    seed = _positive_int(seed, "seed")
    old_rng = torch.random.get_rng_state()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_threads = torch.get_num_threads()
    old_mkldnn = torch.backends.mkldnn.enabled
    try:
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(old_rng)
        torch.use_deterministic_algorithms(old_deterministic)
        torch.backends.mkldnn.enabled = old_mkldnn
        torch.set_num_threads(old_threads)


def _write_torch_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    raw = _canonical_json(_json_safe(payload)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionBackendGateError("artifact payload contains NaN/Inf")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise ProductionBackendGateError(f"artifact payload is not strict JSON: {type(value)!r}")


def _read_json_strict(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ProductionBackendGateError(f"non-finite JSON constant is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProductionBackendGateError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionBackendGateError(f"cannot parse strict JSON artifact: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProductionBackendGateError("artifact root must be a JSON object")
    return payload


__all__ = [
    "BACKEND_ID",
    "BACKEND_VERSION",
    "COUNTED_BLOCKERS",
    "COUNTED_ELIGIBLE",
    "FrozenTorchSpec",
    "LoadedProductionCheckpointV3",
    "ProductionBackendGateError",
    "ProductionEvidenceGateV3",
    "ProductionTorchBackendV3",
    "SessionTickBinding",
    "TickAuthorityV3",
    "load_calibration_artifact_v3",
    "load_metrics_artifact_v3",
    "load_prediction_artifact_v3",
    "load_production_checkpoint_v3",
    "session_content_sha256",
]
