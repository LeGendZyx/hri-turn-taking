"""Non-production real-data backend for the deadline MSc outer evaluation.

This module deliberately does not register a counted backend.  It implements
the training/evaluation protocol needed to obtain honest group-disjoint thesis
results while every artifact remains explicitly non-production and non-counted.
The :mod:`outer_experiment` runner remains the authority for audited real-data
loading, nested LOGO partitions, runner-fitted statistics, and one-pass outer
test access.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from hri_turn_taking.vap.dataset import (
    FeatureNormalizer,
    SessionTensors,
    sessions_from_dataframe,
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
from hri_turn_taking.vap.feature_sets import apply_feature_set
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG
from hri_turn_taking.vap.model import MultipartyVAP, VAPModelConfig
from hri_turn_taking.vap.outer_experiment import (
    CALIBRATION_FILENAME,
    FINAL_CHECKPOINT_FILENAME,
    INNER_CHECKPOINT_FILENAME,
    INNER_OOF_FILENAME,
    OUTER_METRICS_FILENAME,
    OUTER_PREDICTIONS_FILENAME,
    BackendContractSmokeBackend,
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

FORMAT_VERSION = 1
BACKEND_ID = "thesis_torch_backend_v1_nonproduction"
RESULT_ROLE = "thesis_real_outer_evaluation_nonproduction_not_counted"
COUNTED_ELIGIBLE = False
DATA_KIND = "real"
THRESHOLD_POLICY = "diagnostic_fixed_0.5_not_calibrated"
COUNTED_BLOCKERS = (
    "deadline_thesis_backend_not_registered",
    "production_evidence_not_integrated",
    "atomic_run_index_not_implemented",
    "d3_sensitivity_not_executed",
)
CONDITION_FEATURE_SETS = {
    "A": "audio_only",
    "H": "head_pose_only",
    "A+H": "audio_plus_head_pose",
}


class ThesisBackendGateError(ValueError):
    """A deadline-backend contract or artifact validation failed."""


@dataclass(frozen=True)
class ThesisBackendSpec:
    """Small, explicit CPU training specification used by every paired condition."""

    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    encoder: str = "transformer"
    window_frames: int = 250
    stride_frames: int = 125
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.1
    gradient_clip: float = 1.0
    early_stopping_patience: int = 4
    inference_chunk_frames: int = 2000

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "n_layers",
            "n_heads",
            "window_frames",
            "stride_frames",
            "batch_size",
            "early_stopping_patience",
            "inference_chunk_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ThesisBackendGateError(f"{name} must be a positive integer")
        for name in (
            "dropout",
            "learning_rate",
            "weight_decay",
            "warmup_fraction",
            "gradient_clip",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ThesisBackendGateError(f"{name} must be a finite float")
        if self.d_model % self.n_heads:
            raise ThesisBackendGateError("d_model must be divisible by n_heads")
        if self.encoder not in {"transformer", "gru"}:
            raise ThesisBackendGateError("encoder must be transformer or gru")
        if not 0.0 <= self.dropout < 1.0:
            raise ThesisBackendGateError("dropout must be in [0, 1)")
        if self.learning_rate <= 0.0 or self.gradient_clip <= 0.0:
            raise ThesisBackendGateError("learning_rate and gradient_clip must be > 0")
        if self.weight_decay < 0.0:
            raise ThesisBackendGateError("weight_decay must be >= 0")
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise ThesisBackendGateError("warmup_fraction must be in [0, 1]")
        if self.stride_frames > self.window_frames:
            raise ThesisBackendGateError("stride_frames cannot leave gaps")
        if self.inference_chunk_frames <= self.window_frames * self.n_layers:
            raise ThesisBackendGateError("inference chunk must exceed stacked temporal context")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: object) -> ThesisBackendSpec:
        if not isinstance(payload, dict) or set(payload) != set(asdict(cls())):
            raise ThesisBackendGateError("backend spec fields are not exact")
        return cls(**payload)


def thesis_trainer_config(spec: ThesisBackendSpec) -> dict[str, object]:
    """Exact runner configuration that the backend actually executes."""

    return {
        "aux_vad_weight": 0.0,
        "batch_size": spec.batch_size,
        "learning_rate": spec.learning_rate,
        "weight_decay": spec.weight_decay,
        "warmup_fraction": spec.warmup_fraction,
        "gradient_clip": spec.gradient_clip,
        "window_frames": spec.window_frames,
        "stride_frames": spec.stride_frames,
        "early_stopping_patience": spec.early_stopping_patience,
        "inference_chunk_frames": spec.inference_chunk_frames,
    }


def thesis_model_config(spec: ThesisBackendSpec) -> dict[str, object]:
    """Exact model configuration frozen by the outer run manifest."""

    return {
        "d_model": spec.d_model,
        "n_layers": spec.n_layers,
        "n_heads": spec.n_heads,
        "dropout": spec.dropout,
        "encoder": spec.encoder,
    }


@dataclass(frozen=True)
class ThesisRunBinding:
    """Hash binding between the executable backend and one frozen run manifest."""

    binding_role: str
    max_epochs: int
    spec: ThesisBackendSpec
    run_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.binding_role not in {"frozen_outer_protocol", "timing_benchmark_only"}:
            raise ThesisBackendGateError("unknown run-binding role")
        if (
            isinstance(self.max_epochs, bool)
            or not isinstance(self.max_epochs, int)
            or self.max_epochs <= 0
        ):
            raise ThesisBackendGateError("max_epochs must be a positive integer")
        if (
            not isinstance(self.run_manifest_sha256, str)
            or len(self.run_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.run_manifest_sha256)
        ):
            raise ThesisBackendGateError("run_manifest_sha256 must be lowercase SHA-256")

    def to_dict(self) -> dict[str, object]:
        return {
            "binding_role": self.binding_role,
            "max_epochs": self.max_epochs,
            "spec": self.spec.to_dict(),
            "run_manifest_sha256": self.run_manifest_sha256,
        }

    @classmethod
    def from_run_manifest(
        cls, manifest: object, spec: ThesisBackendSpec
    ) -> ThesisRunBinding:
        if not isinstance(manifest, dict):
            raise ThesisBackendGateError("run manifest must be a JSON object")
        if manifest.get("trainer_config") != thesis_trainer_config(spec):
            raise ThesisBackendGateError("frozen trainer config differs from executable backend")
        if manifest.get("model_config") != thesis_model_config(spec):
            raise ThesisBackendGateError("frozen model config differs from executable backend")
        if (
            type(manifest.get("aux_vad_weight")) is not float
            or manifest.get("aux_vad_weight") != 0.0
        ):
            raise ThesisBackendGateError("frozen auxiliary VAD weight must be exactly 0.0")
        expected_label = json.loads(_canonical_json(asdict(DEFAULT_LABEL_CONFIG)))
        if manifest.get("label_config") != expected_label:
            raise ThesisBackendGateError("frozen label config differs from executable backend")
        max_epochs = manifest.get("max_epochs")
        digest = hashlib.sha256(
            _outer_protocol_canonical_json(manifest).encode("utf-8")
        ).hexdigest()
        return cls("frozen_outer_protocol", max_epochs, spec, digest)

    @classmethod
    def benchmark(cls, spec: ThesisBackendSpec) -> ThesisRunBinding:
        payload = {
            "binding_role": "timing_benchmark_only",
            "max_epochs": 1,
            "spec": spec.to_dict(),
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return cls("timing_benchmark_only", 1, spec, digest)


@dataclass(frozen=True)
class RealSessionBinding:
    session_id: str
    group_id: str
    tick_25hz: tuple[int, ...]
    session_content_sha256: str

    def __post_init__(self) -> None:
        if not self.session_id or not self.group_id:
            raise ThesisBackendGateError("real session binding requires session/group identity")
        if not self.tick_25hz:
            raise ThesisBackendGateError("real session tick binding cannot be empty")
        if any(
            isinstance(tick, bool) or not isinstance(tick, int) or tick < 0
            for tick in self.tick_25hz
        ):
            raise ThesisBackendGateError("real session ticks must be non-negative integers")
        if any(
            right != left + 1
            for left, right in zip(self.tick_25hz, self.tick_25hz[1:], strict=False)
        ):
            raise ThesisBackendGateError("real session ticks must be contiguous")
        if (
            len(self.session_content_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.session_content_sha256
            )
        ):
            raise ThesisBackendGateError("real session content hash is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "group_id": self.group_id,
            "tick_25hz": list(self.tick_25hz),
            "session_content_sha256": self.session_content_sha256,
        }


@dataclass(frozen=True)
class RealSessionAuthority:
    """Condition-specific identity for audited real tensors and their exact ticks."""

    condition_id: str
    feature_set: str
    source_manifest_sha256: str
    bindings: tuple[RealSessionBinding, ...]
    data_kind: str = DATA_KIND

    def __post_init__(self) -> None:
        if self.data_kind != DATA_KIND:
            raise ThesisBackendGateError("thesis backend accepts real authority only")
        if CONDITION_FEATURE_SETS.get(self.condition_id) != self.feature_set:
            raise ThesisBackendGateError("authority condition/feature-set mismatch")
        if (
            len(self.source_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_manifest_sha256
            )
        ):
            raise ThesisBackendGateError("authority source-manifest hash is invalid")
        ids = [binding.session_id for binding in self.bindings]
        if not ids or ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ThesisBackendGateError("authority bindings must be non-empty, sorted, unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": "vap_thesis_real_session_authority",
            "format_version": FORMAT_VERSION,
            "data_kind": self.data_kind,
            "condition_id": self.condition_id,
            "feature_set": self.feature_set,
            "source_manifest_sha256": self.source_manifest_sha256,
            "bindings": [binding.to_dict() for binding in self.bindings],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def by_session(self) -> dict[str, RealSessionBinding]:
        return {binding.session_id: binding for binding in self.bindings}

    def validate_subset(self, sessions: tuple[SessionTensors, ...], role: str) -> None:
        authority = self.by_session()
        for session in sessions:
            binding = authority.get(session.session_id)
            if binding is None or binding.group_id != session.group_id:
                raise ThesisBackendGateError(f"{role} differs from real session authority")
            if len(binding.tick_25hz) != session.features.shape[0]:
                raise ThesisBackendGateError(f"{role} frame count differs from tick authority")
            digest = session_content_sha256(session, binding.tick_25hz)
            if digest != binding.session_content_sha256:
                raise ThesisBackendGateError(f"{role} tensor content differs from authority")


def session_content_sha256(session: SessionTensors, ticks: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "session_id": session.session_id,
                "group_id": session.group_id,
                "tick_25hz": list(ticks),
            }
        ).encode("utf-8")
    )
    for name in ("features", "labels", "vad", "valid", "present"):
        value = np.ascontiguousarray(getattr(session, name))
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_real_session_authority(
    frame_df: object,
    source_manifest: object,
    *,
    condition_id: str,
) -> RealSessionAuthority:
    """Bind one audited real dataframe to condition-masked tensors and exact ticks."""

    feature_set = CONDITION_FEATURE_SETS.get(condition_id)
    if feature_set is None:
        raise ThesisBackendGateError("real authority supports only A, H, and A+H")
    if not isinstance(source_manifest, dict):
        raise ThesisBackendGateError("source manifest must be a JSON object")
    if source_manifest.get("data_kind") != DATA_KIND:
        raise ThesisBackendGateError("source manifest is not audited real data")
    if source_manifest.get("target_ready") is not True:
        raise ThesisBackendGateError("real authority requires target-ready data")
    if source_manifest.get("feature_set") != feature_set:
        raise ThesisBackendGateError("source manifest feature set differs from condition")
    if source_manifest.get("selected_feature_ready") is not True:
        raise ThesisBackendGateError("real authority requires selected-feature readiness")
    masked = apply_feature_set(frame_df, feature_set)
    sessions = sessions_from_dataframe(masked)
    bindings = []
    for session in sessions:
        rows = masked.loc[masked["session_id"].astype(str) == session.session_id]
        rows = rows.sort_values("timestamp_s")
        if "tick_25hz" in rows.columns:
            raw_ticks = rows["tick_25hz"].to_numpy()
            if not np.issubdtype(raw_ticks.dtype, np.integer):
                raise ThesisBackendGateError("tick_25hz must retain an integer dtype")
            ticks = tuple(int(value) for value in raw_ticks.tolist())
        else:
            scaled = rows["timestamp_s"].to_numpy(dtype=float) * 25.0
            rounded = np.rint(scaled)
            if not np.allclose(scaled, rounded, rtol=0.0, atol=1e-9):
                raise ThesisBackendGateError("timestamps do not map exactly to 25 Hz ticks")
            ticks = tuple(int(value) for value in rounded.tolist())
        bindings.append(
            RealSessionBinding(
                session_id=session.session_id,
                group_id=str(session.group_id),
                tick_25hz=ticks,
                session_content_sha256=session_content_sha256(session, ticks),
            )
        )
    manifest_hash = hashlib.sha256(
        _canonical_json(source_manifest).encode("utf-8")
    ).hexdigest()
    return RealSessionAuthority(
        condition_id=condition_id,
        feature_set=feature_set,
        source_manifest_sha256=manifest_hash,
        bindings=tuple(bindings),
    )


@dataclass
class _FitResult:
    model: MultipartyVAP
    history: list[dict[str, object]]
    selected_epoch: int


def _base_payload(kind: str) -> dict[str, object]:
    return {
        "artifact_kind": kind,
        "format_version": FORMAT_VERSION,
        "backend_id": BACKEND_ID,
        "result_role": RESULT_ROLE,
        "data_kind": DATA_KIND,
        "production_eligible": False,
        "counted_eligible": False,
        "counted_blockers": list(COUNTED_BLOCKERS),
    }


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _outer_protocol_canonical_json(payload: object) -> str:
    """Match ``outer_experiment._canonical_json`` including ASCII escaping."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ThesisBackendGateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_strict(path_value: str | Path) -> dict[str, object]:
    path = Path(path_value).resolve()
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ThesisBackendGateError(f"non-standard JSON constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ThesisBackendGateError(f"cannot read strict JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ThesisBackendGateError("JSON artifact root must be an object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_base_payload(payload: object, artifact_kind: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ThesisBackendGateError("artifact root must be a JSON object")
    expected = _base_payload(artifact_kind)
    for field, value in expected.items():
        if payload.get(field) != value or type(payload.get(field)) is not type(value):
            raise ThesisBackendGateError(f"artifact base field differs: {field}")
    return payload


def _require_outputs_absent(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite thesis artifact: {existing}")


def _session_ids(sessions: tuple[SessionTensors, ...]) -> tuple[str, ...]:
    return tuple(session.session_id for session in sessions)


def _session_state_sha256(sessions: tuple[SessionTensors, ...]) -> str:
    digest = hashlib.sha256()
    for session in sessions:
        digest.update(
            _canonical_json(
                {"session_id": session.session_id, "group_id": session.group_id}
            ).encode("utf-8")
        )
        for name in ("features", "labels", "vad", "valid", "present"):
            value = np.ascontiguousarray(getattr(session, name))
            digest.update(name.encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(_canonical_json(list(value.shape)).encode("ascii"))
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _normalizer_payload(normalizer: FeatureNormalizer) -> dict[str, list[float]]:
    return normalizer.to_dict()


def _statistics_sha256(
    normalizer: FeatureNormalizer, pos_weight: np.ndarray
) -> tuple[str, str]:
    normalizer_hash = hashlib.sha256(
        _canonical_json(_normalizer_payload(normalizer)).encode("utf-8")
    ).hexdigest()
    pos_hash = hashlib.sha256(_canonical_json(pos_weight.tolist()).encode("utf-8")).hexdigest()
    return normalizer_hash, pos_hash


def _validated_statistics(
    normalizer: FeatureNormalizer, pos_weight: np.ndarray
) -> tuple[FeatureNormalizer, np.ndarray]:
    copied = FeatureNormalizer.from_dict(normalizer.to_dict())
    weight = np.asarray(pos_weight, dtype=np.float32).copy()
    expected_shapes = ((7,), (7,), (5,), (5,))
    arrays = (
        copied.participant_mean,
        copied.participant_std,
        copied.global_mean,
        copied.global_std,
    )
    if tuple(array.shape for array in arrays) != expected_shapes:
        raise ThesisBackendGateError("runner-supplied normalizer shapes are invalid")
    if weight.shape != (DEFAULT_LABEL_CONFIG.n_bins,):
        raise ThesisBackendGateError("runner-supplied pos_weight must have four bins")
    if not all(np.isfinite(array).all() for array in (*arrays, weight)):
        raise ThesisBackendGateError("runner-supplied statistics must be finite")
    if (copied.participant_std <= 0).any() or (copied.global_std <= 0).any():
        raise ThesisBackendGateError("runner-supplied normalizer std must be positive")
    if (weight <= 0).any():
        raise ThesisBackendGateError("runner-supplied pos_weight must be positive")
    return copied, weight


def _validate_sessions(sessions: tuple[SessionTensors, ...], role: str) -> None:
    if not isinstance(sessions, tuple) or not sessions:
        raise ThesisBackendGateError(f"{role} sessions must be a non-empty tuple")
    seen: set[str] = set()
    for session in sessions:
        if not isinstance(session, SessionTensors):
            raise ThesisBackendGateError(f"{role} contains a non-SessionTensors value")
        if not session.session_id or session.session_id in seen:
            raise ThesisBackendGateError(f"{role} session ids must be unique and non-empty")
        if not session.group_id:
            raise ThesisBackendGateError(f"{role} requires provenance group_id")
        if session.features.ndim != 2 or session.features.shape[1] != 33:
            raise ThesisBackendGateError(f"{role} uses an unexpected feature schema")
        n_frames = session.features.shape[0]
        if session.labels.shape != (n_frames, 4, 4):
            raise ThesisBackendGateError(f"{role} labels have an unexpected shape")
        if session.vad.shape != (n_frames, 4) or session.valid.shape != (n_frames,):
            raise ThesisBackendGateError(f"{role} VAD/valid shape mismatch")
        if session.present.shape != (4,):
            raise ThesisBackendGateError(f"{role} present mask shape mismatch")
        if not all(
            np.isfinite(value).all()
            for value in (session.features, session.labels, session.vad, session.present)
        ):
            raise ThesisBackendGateError(f"{role} contains non-finite values")
        seen.add(session.session_id)


def _model(spec: ThesisBackendSpec) -> MultipartyVAP:
    return MultipartyVAP(
        VAPModelConfig(
            n_participant_features=7,
            n_global_features=5,
            n_participants=4,
            n_bins=4,
            d_model=spec.d_model,
            n_layers=spec.n_layers,
            n_heads=spec.n_heads,
            dropout=spec.dropout,
            encoder=spec.encoder,
            attention_window=spec.window_frames,
        )
    )


def _training_spans(
    sessions: tuple[SessionTensors, ...], spec: ThesisBackendSpec
) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    for session_index, session in enumerate(sessions):
        n_frames = session.features.shape[0]
        if n_frames <= spec.window_frames:
            starts = [0]
        else:
            starts = list(range(0, n_frames - spec.window_frames + 1, spec.stride_frames))
            tail = n_frames - spec.window_frames
            if starts[-1] != tail:
                starts.append(tail)
        for start in starts:
            end = min(n_frames, start + spec.window_frames)
            if session.valid[start:end].any() and (session.present > 0).any():
                spans.append((session_index, start, end))
    if not spans:
        raise ThesisBackendGateError("training partition contains no valid VAP windows")
    return spans


def _batch_indices(
    order: list[int], spans: list[tuple[int, int, int]], batch_size: int
) -> Iterator[list[int]]:
    pending: dict[int, list[int]] = {}
    for index in order:
        _, start, end = spans[index]
        length = end - start
        bucket = pending.setdefault(length, [])
        bucket.append(index)
        if len(bucket) == batch_size:
            yield bucket.copy()
            bucket.clear()
    for length in sorted(pending):
        if pending[length]:
            yield pending[length].copy()


def _batch_loss(
    model: MultipartyVAP,
    sessions: tuple[SessionTensors, ...],
    normalized: list[np.ndarray],
    spans: list[tuple[int, int, int]],
    indices: list[int],
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    features = []
    labels = []
    valid = []
    present = []
    for index in indices:
        session_index, start, end = spans[index]
        session = sessions[session_index]
        features.append(torch.from_numpy(normalized[session_index][start:end]))
        labels.append(torch.from_numpy(session.labels[start:end]))
        valid.append(torch.from_numpy(session.valid[start:end].astype(np.float32)))
        present.append(torch.from_numpy(session.present.copy()))
    feature_batch = torch.stack(features)
    label_batch = torch.stack(labels)
    valid_batch = torch.stack(valid)
    present_batch = torch.stack(present)
    output = model(feature_batch, present=present_batch)
    mask = valid_batch.unsqueeze(-1) * present_batch.unsqueeze(1)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        output["bin_logits"], label_batch, pos_weight=pos_weight, reduction="none"
    )
    expanded = mask.unsqueeze(-1).expand_as(losses)
    return (losses * expanded).sum() / expanded.sum().clamp(min=1.0)


def _predict_sessions(
    model: MultipartyVAP,
    sessions: tuple[SessionTensors, ...],
    normalizer: FeatureNormalizer,
    spec: ThesisBackendSpec,
) -> tuple[np.ndarray, ...]:
    model.cpu().eval()
    result = []
    for session in sessions:
        probability = predict_session_probabilities(
            model,
            session,
            normalizer,
            device="cpu",
            chunk_frames=spec.inference_chunk_frames,
        )
        if probability.shape != (session.features.shape[0], 4, 4):
            raise ThesisBackendGateError("prediction shape mismatch")
        if not np.isfinite(probability).all():
            raise ThesisBackendGateError("prediction contains non-finite values")
        result.append(probability.astype(np.float32, copy=False))
    return tuple(result)


def _frame_metrics(
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
    return compute_frame_metrics(
        np.concatenate(labels),
        np.concatenate(scores),
        participant_indices=np.concatenate(participants),
        max_participants=4,
    )


def _metrics(
    sessions: tuple[SessionTensors, ...], predictions: tuple[np.ndarray, ...]
) -> dict[str, object]:
    pooled: dict[str, tuple[list[int], list[float]]] = {
        "shift_hold": ([], []),
        "shift_prediction": ([], []),
        "overlap_resolution": ([], []),
        "robot_entry": ([], []),
    }
    next_labels: list[int] = []
    next_scores: list[np.ndarray] = []
    event_functions = {
        "shift_hold": shift_hold_scores,
        "shift_prediction": shift_prediction_scores,
        "overlap_resolution": overlap_resolution_scores,
    }
    for session, probability in zip(sessions, predictions, strict=True):
        for name, function in event_functions.items():
            labels, scores = function(
                probability,
                session.vad,
                frame_rate_hz=25,
                label_config=DEFAULT_LABEL_CONFIG,
                present=session.present,
                valid=session.valid,
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
            valid=session.valid,
        )
        pooled["robot_entry"][0].extend(labels)
        pooled["robot_entry"][1].extend(scores)
        labels, scores = next_speaker_scores(
            probability,
            session.vad,
            frame_rate_hz=25,
            label_config=DEFAULT_LABEL_CONFIG,
            present=session.present,
            valid=session.valid,
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
        "threshold_policy": THRESHOLD_POLICY,
        "frame": _json_safe(_frame_metrics(sessions, predictions)),
        "events": _json_safe(events),
    }


def _fit(
    spec: ThesisBackendSpec,
    train_sessions: tuple[SessionTensors, ...],
    validation_sessions: tuple[SessionTensors, ...],
    normalizer: FeatureNormalizer,
    pos_weight: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> _FitResult:
    torch.manual_seed(seed)
    model = _model(spec).cpu()
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": spec.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=spec.learning_rate,
    )
    spans = _training_spans(train_sessions, spec)
    steps_per_epoch = math.ceil(len(spans) / spec.batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, round(spec.warmup_fraction * total_steps))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.45 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    normalized = [normalizer.apply(session.features) for session in train_sessions]
    pos_tensor = torch.from_numpy(pos_weight.astype(np.float32, copy=True))
    history: list[dict[str, object]] = []
    best_epoch: int | None = None
    best_map = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        order = generator.permutation(len(spans)).tolist()
        losses = []
        for indices in _batch_indices(order, spans, spec.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, train_sessions, normalized, spans, indices, pos_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        if not losses:
            raise ThesisBackendGateError("epoch contains no optimizer step")
        validation_map: float | None = None
        if validation_sessions:
            predictions = _predict_sessions(model, validation_sessions, normalizer, spec)
            validation_map = _frame_metrics(validation_sessions, predictions)["mAP"]
            if (
                validation_map is not None
                and math.isfinite(validation_map)
                and validation_map > best_map + 1e-4
            ):
                best_map = float(validation_map)
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_mAP": validation_map,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "n_training_windows": len(spans),
                "n_optimizer_steps": len(losses),
            }
        )
        if validation_sessions and epochs_without_improvement >= spec.early_stopping_patience:
            break

    if validation_sessions:
        if best_state is None or best_epoch is None:
            raise ThesisBackendGateError(
                "strict inner-validation mAP is unavailable; epoch selection is blocked"
            )
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
    else:
        selected_epoch = epochs
    model.eval()
    return _FitResult(model=model, history=history, selected_epoch=selected_epoch)


@contextmanager
def _deterministic_cpu(seed: int) -> Iterator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_threads = torch.get_num_threads()
    try:
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        yield
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        handle.write("\n")


def _write_torch_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _prediction_payload(
    sessions: tuple[SessionTensors, ...],
    predictions: tuple[np.ndarray, ...],
    *,
    condition_id: str,
    feature_set: str,
    outer_fold_id: str,
    seed: int,
    checkpoint_sha256: str,
    prediction_role: str,
    authority: RealSessionAuthority,
    run_binding: ThesisRunBinding,
) -> dict[str, object]:
    authority_by_session = authority.by_session()
    return {
        **_base_payload("vap_thesis_saved_probabilities"),
        "prediction_role": prediction_role,
        "condition_id": condition_id,
        "feature_set": feature_set,
        "outer_fold_id": outer_fold_id,
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "real_session_authority_sha256": authority.sha256,
        "source_manifest_sha256": authority.source_manifest_sha256,
        "run_manifest_sha256": run_binding.run_manifest_sha256,
        "frame_rate_hz": 25,
        "label_config": json.loads(_canonical_json(asdict(DEFAULT_LABEL_CONFIG))),
        "threshold_policy": THRESHOLD_POLICY,
        "prediction_pass_count": 1,
        "sessions": [
            {
                "session_id": session.session_id,
                "group_id": session.group_id,
                "n_frames": int(session.features.shape[0]),
                "tick_25hz": list(authority_by_session[session.session_id].tick_25hz),
                "probabilities": probability.tolist(),
                "labels": session.labels.tolist(),
                "vad": session.vad.tolist(),
                "valid": session.valid.astype(int).tolist(),
                "present": session.present.astype(int).tolist(),
            }
            for session, probability in zip(sessions, predictions, strict=True)
        ],
    }


def _load_prediction_artifact(
    path_value: str | Path,
    *,
    expected_inner_fold_id: str | None,
) -> dict[str, object]:
    payload = _read_json_strict(path_value)
    required = {
        "artifact_kind",
        "format_version",
        "backend_id",
        "result_role",
        "data_kind",
        "production_eligible",
        "counted_eligible",
        "counted_blockers",
        "prediction_role",
        "condition_id",
        "feature_set",
        "outer_fold_id",
        "seed",
        "checkpoint_sha256",
        "real_session_authority_sha256",
        "source_manifest_sha256",
        "run_manifest_sha256",
        "frame_rate_hz",
        "label_config",
        "threshold_policy",
        "prediction_pass_count",
        "sessions",
    }
    if expected_inner_fold_id is not None:
        required.add("inner_fold_id")
    if not isinstance(payload, dict) or set(payload) != required:
        raise ThesisBackendGateError("prediction artifact fields are not exact")
    _validate_base_payload(payload, "vap_thesis_saved_probabilities")
    expected_role = (
        "outer_test_single_saved_pass"
        if expected_inner_fold_id is None
        else "inner_oof_not_outer_test"
    )
    if payload["prediction_role"] != expected_role:
        raise ThesisBackendGateError("prediction role differs from requested artifact role")
    if expected_inner_fold_id is not None and payload["inner_fold_id"] != expected_inner_fold_id:
        raise ThesisBackendGateError("inner OOF fold identity mismatch")
    if payload["prediction_pass_count"] != 1 or isinstance(
        payload["prediction_pass_count"], bool
    ):
        raise ThesisBackendGateError("outer prediction must record exactly one pass")
    if payload["frame_rate_hz"] != 25 or isinstance(payload["frame_rate_hz"], bool):
        raise ThesisBackendGateError("prediction frame rate must be exactly 25 Hz")
    if payload["label_config"] != json.loads(_canonical_json(asdict(DEFAULT_LABEL_CONFIG))):
        raise ThesisBackendGateError("prediction label config mismatch")
    if payload["threshold_policy"] != THRESHOLD_POLICY:
        raise ThesisBackendGateError("prediction threshold policy is not diagnostic 0.5")
    expected_feature = CONDITION_FEATURE_SETS.get(payload["condition_id"])
    if expected_feature != payload["feature_set"]:
        raise ThesisBackendGateError("prediction condition/feature-set mismatch")
    if (
        not isinstance(payload["outer_fold_id"], str)
        or not payload["outer_fold_id"].startswith("outer_")
    ):
        raise ThesisBackendGateError("prediction outer fold id is invalid")
    if (
        isinstance(payload["seed"], bool)
        or not isinstance(payload["seed"], int)
        or payload["seed"] <= 0
    ):
        raise ThesisBackendGateError("prediction seed is invalid")
    for field in (
        "checkpoint_sha256",
        "real_session_authority_sha256",
        "source_manifest_sha256",
        "run_manifest_sha256",
    ):
        if not _is_sha256(payload[field]):
            raise ThesisBackendGateError(f"prediction hash field is invalid: {field}")
    sessions = payload["sessions"]
    if not isinstance(sessions, list) or not sessions:
        raise ThesisBackendGateError("prediction artifact has no sessions")
    seen: set[str] = set()
    observed_groups: set[str] = set()
    for record in sessions:
        required_record = {
            "session_id",
            "group_id",
            "n_frames",
            "tick_25hz",
            "probabilities",
            "labels",
            "vad",
            "valid",
            "present",
        }
        if not isinstance(record, dict) or set(record) != required_record:
            raise ThesisBackendGateError("prediction session fields are not exact")
        session_id = record["session_id"]
        if not isinstance(session_id, str) or not session_id or session_id in seen:
            raise ThesisBackendGateError("prediction session ids are invalid or duplicated")
        group_id = record["group_id"]
        if not isinstance(group_id, str) or not group_id:
            raise ThesisBackendGateError("prediction group id is invalid")
        n_frames = record["n_frames"]
        if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
            raise ThesisBackendGateError("prediction n_frames is invalid")
        try:
            probability = np.asarray(record["probabilities"], dtype=np.float32)
            labels = np.asarray(record["labels"], dtype=np.float32)
            vad = np.asarray(record["vad"], dtype=np.float32)
            valid = np.asarray(record["valid"], dtype=np.float32)
            present = np.asarray(record["present"], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ThesisBackendGateError("prediction arrays are not numeric") from exc
        if probability.shape != (n_frames, 4, 4) or labels.shape != probability.shape:
            raise ThesisBackendGateError("prediction probability/label shape mismatch")
        ticks = record["tick_25hz"]
        if (
            not isinstance(ticks, list)
            or len(ticks) != n_frames
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in ticks
            )
            or any(
                right != left + 1
                for left, right in zip(ticks, ticks[1:], strict=False)
            )
        ):
            raise ThesisBackendGateError("prediction ticks are not exact contiguous 25 Hz ticks")
        if vad.shape != (n_frames, 4) or valid.shape != (n_frames,) or present.shape != (4,):
            raise ThesisBackendGateError("prediction VAD/mask shape mismatch")
        if not all(
            np.isfinite(value).all()
            for value in (probability, labels, vad, valid, present)
        ):
            raise ThesisBackendGateError("prediction arrays must be finite")
        if ((probability < 0) | (probability > 1)).any():
            raise ThesisBackendGateError("prediction probabilities must be finite in [0,1]")
        if not np.isin(labels, (0.0, 1.0)).all() or not np.isin(vad, (0.0, 1.0)).all():
            raise ThesisBackendGateError("prediction targets/VAD must be binary")
        if not np.isin(valid, (0, 1)).all() or not np.isin(present, (0.0, 1.0)).all():
            raise ThesisBackendGateError("prediction masks must be binary")
        seen.add(session_id)
        observed_groups.add(group_id)
    if len(observed_groups) != 1:
        raise ThesisBackendGateError("one fold prediction must contain exactly one held-out group")
    return payload


def load_prediction_artifact(path_value: str | Path) -> dict[str, object]:
    """Strictly load one outer-test saved-probability artifact."""

    return _load_prediction_artifact(path_value, expected_inner_fold_id=None)


def load_checkpoint_artifact(path_value: str | Path) -> dict[str, object]:
    """Strictly load a thesis checkpoint without granting it counted status."""

    path = Path(path_value).resolve()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ThesisBackendGateError(f"cannot load thesis checkpoint {path}: {exc}") from exc
    required = {
        *_base_payload("vap_thesis_checkpoint"),
        "spec",
        "run_binding",
        "real_session_authority_sha256",
        "source_manifest_sha256",
        "label_config",
        "condition_id",
        "feature_set",
        "outer_fold_id",
        "inner_fold_id",
        "seed",
        "fit_role",
        "selected_epoch",
        "initialized_from_scratch",
        "train_session_ids",
        "normalizer",
        "pos_weight",
        "aux_vad_weight",
        "history",
        "model_state",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ThesisBackendGateError("checkpoint fields are not exact")
    _validate_base_payload(payload, "vap_thesis_checkpoint")
    spec = ThesisBackendSpec.from_dict(payload["spec"])
    run_binding = payload["run_binding"]
    if not isinstance(run_binding, dict) or set(run_binding) != {
        "binding_role",
        "max_epochs",
        "spec",
        "run_manifest_sha256",
    }:
        raise ThesisBackendGateError("checkpoint run binding fields are not exact")
    parsed_binding = ThesisRunBinding(
        binding_role=run_binding["binding_role"],
        max_epochs=run_binding["max_epochs"],
        spec=ThesisBackendSpec.from_dict(run_binding["spec"]),
        run_manifest_sha256=run_binding["run_manifest_sha256"],
    )
    if parsed_binding.spec != spec:
        raise ThesisBackendGateError("checkpoint spec differs from run binding")
    if not _is_sha256(payload["real_session_authority_sha256"]) or not _is_sha256(
        payload["source_manifest_sha256"]
    ):
        raise ThesisBackendGateError("checkpoint real-data hashes are invalid")
    if payload["label_config"] != json.loads(_canonical_json(asdict(DEFAULT_LABEL_CONFIG))):
        raise ThesisBackendGateError("checkpoint label config mismatch")
    expected_feature = CONDITION_FEATURE_SETS.get(payload["condition_id"])
    if expected_feature != payload["feature_set"]:
        raise ThesisBackendGateError("checkpoint condition/feature-set mismatch")
    if (
        not isinstance(payload["outer_fold_id"], str)
        or not payload["outer_fold_id"].startswith("outer_")
    ):
        raise ThesisBackendGateError("checkpoint outer fold id is invalid")
    if (
        isinstance(payload["seed"], bool)
        or not isinstance(payload["seed"], int)
        or payload["seed"] <= 0
    ):
        raise ThesisBackendGateError("checkpoint seed is invalid")
    if (
        isinstance(payload["selected_epoch"], bool)
        or not isinstance(payload["selected_epoch"], int)
        or payload["selected_epoch"] <= 0
    ):
        raise ThesisBackendGateError("checkpoint selected epoch is invalid")
    if payload["initialized_from_scratch"] is not True:
        raise ThesisBackendGateError("checkpoint was not initialized from scratch")
    if type(payload["aux_vad_weight"]) is not float or payload["aux_vad_weight"] != 0.0:
        raise ThesisBackendGateError("checkpoint auxiliary VAD weight is not exactly 0.0")
    if payload["fit_role"] not in {
        "inner_selection",
        "fixed_epoch_outer_train_from_scratch",
    }:
        raise ThesisBackendGateError("checkpoint fit role is invalid")
    if payload["fit_role"] == "inner_selection":
        if not isinstance(payload["inner_fold_id"], str) or not payload["inner_fold_id"]:
            raise ThesisBackendGateError("inner checkpoint lacks an inner fold id")
    elif payload["inner_fold_id"] is not None:
        raise ThesisBackendGateError("final checkpoint must not claim an inner fold")
    train_ids = payload["train_session_ids"]
    if (
        not isinstance(train_ids, list)
        or not train_ids
        or any(not isinstance(value, str) or not value for value in train_ids)
        or len(train_ids) != len(set(train_ids))
    ):
        raise ThesisBackendGateError("checkpoint training session ids are invalid")
    history = payload["history"]
    if (
        not isinstance(history, list)
        or len(history) < payload["selected_epoch"]
        or any(not isinstance(record, dict) for record in history)
    ):
        raise ThesisBackendGateError("checkpoint training history is invalid")
    try:
        normalizer = FeatureNormalizer.from_dict(payload["normalizer"])
        _validated_statistics(normalizer, np.asarray(payload["pos_weight"], dtype=np.float32))
    except (KeyError, TypeError, ValueError) as exc:
        raise ThesisBackendGateError("checkpoint statistics are invalid") from exc
    model_state = payload["model_state"]
    if (
        not isinstance(model_state, dict)
        or not model_state
        or any(
            not isinstance(name, str) or not torch.is_tensor(value)
            for name, value in model_state.items()
        )
    ):
        raise ThesisBackendGateError("checkpoint model state is invalid")
    return payload


def load_metrics_artifact(
    path_value: str | Path,
    *,
    prediction_path: str | Path | None = None,
) -> dict[str, object]:
    """Load outer metrics and optionally recompute them from saved predictions."""

    payload = _read_json_strict(path_value)
    required = {
        *_base_payload("vap_thesis_outer_metrics"),
        "condition_id",
        "feature_set",
        "outer_fold_id",
        "seed",
        "prediction_sha256",
        "checkpoint_sha256",
        "real_session_authority_sha256",
        "source_manifest_sha256",
        "run_manifest_sha256",
        "threshold_policy",
        "frame",
        "events",
    }
    if set(payload) != required:
        raise ThesisBackendGateError("outer metrics fields are not exact")
    _validate_base_payload(payload, "vap_thesis_outer_metrics")
    if CONDITION_FEATURE_SETS.get(payload["condition_id"]) != payload["feature_set"]:
        raise ThesisBackendGateError("outer metrics condition/feature-set mismatch")
    if (
        not isinstance(payload["outer_fold_id"], str)
        or not payload["outer_fold_id"].startswith("outer_")
        or isinstance(payload["seed"], bool)
        or not isinstance(payload["seed"], int)
        or payload["seed"] <= 0
    ):
        raise ThesisBackendGateError("outer metrics fold/seed is invalid")
    for field in (
        "prediction_sha256",
        "checkpoint_sha256",
        "real_session_authority_sha256",
        "source_manifest_sha256",
        "run_manifest_sha256",
    ):
        if not _is_sha256(payload[field]):
            raise ThesisBackendGateError(f"outer metrics hash field is invalid: {field}")
    if payload["threshold_policy"] != THRESHOLD_POLICY:
        raise ThesisBackendGateError("outer metrics threshold policy is not diagnostic 0.5")
    if not isinstance(payload["frame"], dict) or not isinstance(payload["events"], dict):
        raise ThesisBackendGateError("outer metric families are malformed")
    if prediction_path is not None:
        prediction = load_prediction_artifact(prediction_path)
        if (
            payload["prediction_sha256"]
            != _sha256_file(Path(prediction_path).resolve())
            or payload["condition_id"] != prediction["condition_id"]
            or payload["feature_set"] != prediction["feature_set"]
            or payload["outer_fold_id"] != prediction["outer_fold_id"]
            or payload["seed"] != prediction["seed"]
            or payload["checkpoint_sha256"] != prediction["checkpoint_sha256"]
            or payload["real_session_authority_sha256"]
            != prediction["real_session_authority_sha256"]
            or payload["source_manifest_sha256"]
            != prediction["source_manifest_sha256"]
            or payload["run_manifest_sha256"] != prediction["run_manifest_sha256"]
        ):
            raise ThesisBackendGateError("outer metrics identity differs from saved prediction")
        sessions = _sessions_from_prediction(prediction)
        probabilities = tuple(
            np.asarray(record["probabilities"], dtype=np.float32)
            for record in prediction["sessions"]
        )
        recomputed = _metrics(sessions, probabilities)
        observed = {
            "threshold_policy": payload["threshold_policy"],
            "frame": payload["frame"],
            "events": payload["events"],
        }
        if _canonical_json(observed) != _canonical_json(recomputed):
            raise ThesisBackendGateError(
                "outer metrics content differs from saved-prediction recomputation"
            )
    return payload


def _sessions_from_prediction(payload: dict[str, object]) -> tuple[SessionTensors, ...]:
    result = []
    for record in payload["sessions"]:
        n_frames = record["n_frames"]
        result.append(
            SessionTensors(
                session_id=record["session_id"],
                group_id=record["group_id"],
                features=np.zeros((n_frames, 33), dtype=np.float32),
                labels=np.asarray(record["labels"], dtype=np.float32),
                vad=np.asarray(record["vad"], dtype=np.float32),
                valid=np.asarray(record["valid"], dtype=bool),
                present=np.asarray(record["present"], dtype=np.float32),
            )
        )
    return tuple(result)


def aggregate_outer_prediction_artifacts(
    paths: tuple[str | Path, ...],
) -> dict[str, object]:
    """Pool one saved outer prediction per fold without running a model again."""

    if not isinstance(paths, tuple) or len(paths) != 4:
        raise ThesisBackendGateError("aggregation requires exactly four outer-fold artifacts")
    payloads = [load_prediction_artifact(path) for path in paths]
    identities = {
        (
            payload["condition_id"],
            payload["feature_set"],
            payload["seed"],
            payload["real_session_authority_sha256"],
            payload["source_manifest_sha256"],
            payload["run_manifest_sha256"],
        )
        for payload in payloads
    }
    if len(identities) != 1:
        raise ThesisBackendGateError(
            "aggregated folds must share condition, seed, and real/run authority"
        )
    folds = [payload["outer_fold_id"] for payload in payloads]
    if sorted(folds) != ["outer_01", "outer_02", "outer_03", "outer_04"]:
        raise ThesisBackendGateError("aggregation requires each outer fold exactly once")
    session_ids = [
        record["session_id"] for payload in payloads for record in payload["sessions"]
    ]
    if len(session_ids) != len(set(session_ids)):
        raise ThesisBackendGateError("a session appears in more than one outer fold")
    fold_groups = [
        next(iter({record["group_id"] for record in payload["sessions"]}))
        for payload in payloads
    ]
    if len(set(fold_groups)) != 4:
        raise ThesisBackendGateError("the four outer artifacts must hold out four groups")
    sessions = tuple(
        session
        for payload in payloads
        for session in _sessions_from_prediction(payload)
    )
    predictions = tuple(
        np.asarray(record["probabilities"], dtype=np.float32)
        for payload in payloads
        for record in payload["sessions"]
    )
    per_fold = {
        payload["outer_fold_id"]: _metrics(
            _sessions_from_prediction(payload),
            tuple(
                np.asarray(record["probabilities"], dtype=np.float32)
                for record in payload["sessions"]
            ),
        )
        for payload in payloads
    }
    (
        condition_id,
        feature_set,
        seed,
        authority_sha256,
        source_manifest_sha256,
        run_manifest_sha256,
    ) = next(iter(identities))
    return {
        **_base_payload("vap_thesis_outer_aggregate"),
        "condition_id": condition_id,
        "feature_set": feature_set,
        "seed": seed,
        "real_session_authority_sha256": authority_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "outer_fold_ids": sorted(folds),
        "session_ids": sorted(session_ids),
        "fold_session_ids": {
            payload["outer_fold_id"]: [
                record["session_id"] for record in payload["sessions"]
            ]
            for payload in payloads
        },
        "fold_group_ids": {
            payload["outer_fold_id"]: next(
                iter({record["group_id"] for record in payload["sessions"]})
            )
            for payload in payloads
        },
        "prediction_artifact_sha256": [
            _sha256_file(Path(path).resolve()) for path in paths
        ],
        "aggregation_source": "saved_outer_probabilities_only_no_model_inference",
        "threshold_policy": THRESHOLD_POLICY,
        "pooled": _metrics(sessions, predictions),
        "per_fold": per_fold,
    }


class ThesisTorchBackend(BackendContractSmokeBackend):
    """Real-data scientific backend accepted only through the non-counted smoke slot."""

    def __init__(
        self,
        spec: ThesisBackendSpec,
        authority: RealSessionAuthority,
        run_binding: ThesisRunBinding,
    ) -> None:
        self.spec = ThesisBackendSpec.from_dict(spec.to_dict())
        if run_binding.spec != self.spec:
            raise ThesisBackendGateError("run binding spec differs from backend spec")
        self.authority = authority
        self.run_binding = run_binding
        self._evaluated_units: set[tuple[str, str, int, str]] = set()

    def _validate_condition(self, condition_id: str, feature_set: str | None) -> str:
        expected = CONDITION_FEATURE_SETS.get(condition_id)
        if expected is None:
            raise ThesisBackendGateError("only A, H, and A+H are supported")
        if feature_set is not None and feature_set != expected:
            raise ThesisBackendGateError("condition and feature set do not match")
        if (
            self.authority.condition_id != condition_id
            or self.authority.feature_set != expected
        ):
            raise ThesisBackendGateError("backend real authority differs from condition")
        return expected

    def fit_inner(self, request: InnerFitRequest) -> InnerFitReceipt:
        feature_set = self._validate_condition(request.condition_id, request.feature_set)
        checkpoint_path = request.output_dir / INNER_CHECKPOINT_FILENAME
        oof_path = request.output_dir / INNER_OOF_FILENAME
        _require_outputs_absent(checkpoint_path, oof_path)
        if request.encoder != self.spec.encoder:
            raise ThesisBackendGateError("request encoder differs from backend spec")
        _validate_sessions(request.train_sessions, "inner train")
        _validate_sessions(request.validation_sessions, "inner validation")
        self.authority.validate_subset(request.train_sessions, "inner train")
        self.authority.validate_subset(request.validation_sessions, "inner validation")
        if request.max_epochs != self.run_binding.max_epochs:
            raise ThesisBackendGateError("inner max_epochs differs from frozen run binding")
        if set(_session_ids(request.train_sessions)) & set(
            _session_ids(request.validation_sessions)
        ):
            raise ThesisBackendGateError("inner train and validation sessions overlap")
        normalizer, pos_weight = _validated_statistics(request.normalizer, request.pos_weight)
        stats_before = _statistics_sha256(request.normalizer, np.asarray(request.pos_weight))
        state_before = _session_state_sha256(
            (*request.train_sessions, *request.validation_sessions)
        )
        with _deterministic_cpu(request.seed):
            fit = _fit(
                self.spec,
                request.train_sessions,
                request.validation_sessions,
                normalizer,
                pos_weight,
                epochs=request.max_epochs,
                seed=request.seed,
            )
            predictions = _predict_sessions(
                fit.model, request.validation_sessions, normalizer, self.spec
            )
        if stats_before != _statistics_sha256(
            request.normalizer, np.asarray(request.pos_weight)
        ):
            raise ThesisBackendGateError("runner-supplied statistics changed during inner fit")
        if state_before != _session_state_sha256(
            (*request.train_sessions, *request.validation_sessions)
        ):
            raise ThesisBackendGateError("runner-supplied sessions changed during inner fit")
        payload = self._checkpoint_payload(
            fit,
            normalizer,
            pos_weight,
            request.condition_id,
            feature_set,
            request.outer_fold_id,
            request.seed,
            _session_ids(request.train_sessions),
            "inner_selection",
            request.inner_fold_id,
        )
        _write_torch_exclusive(checkpoint_path, payload)
        load_checkpoint_artifact(checkpoint_path)
        oof_payload = _prediction_payload(
            request.validation_sessions,
            predictions,
            condition_id=request.condition_id,
            feature_set=feature_set,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            checkpoint_sha256=_sha256_file(checkpoint_path),
            prediction_role="inner_oof_not_outer_test",
            authority=self.authority,
            run_binding=self.run_binding,
        )
        oof_payload["inner_fold_id"] = request.inner_fold_id
        _write_json_exclusive(oof_path, oof_payload)
        _load_prediction_artifact(
            oof_path,
            expected_inner_fold_id=request.inner_fold_id,
        )
        train_ids = _session_ids(request.train_sessions)
        return InnerFitReceipt(
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            inner_fold_id=request.inner_fold_id,
            seed=request.seed,
            best_epoch=fit.selected_epoch,
            train_session_ids=train_ids,
            validation_session_ids=_session_ids(request.validation_sessions),
            normalizer_fit_session_ids=train_ids,
            pos_weight_fit_session_ids=train_ids,
            checkpoint_path=checkpoint_path,
            oof_predictions_path=oof_path,
        )

    def calibrate_inner_oof(self, request: CalibrationRequest) -> CalibrationReceipt:
        if len(request.inner_fold_ids) != 3 or len(request.oof_prediction_paths) != 3:
            raise ThesisBackendGateError("calibration requires all three inner folds")
        if len(set(request.inner_fold_ids)) != 3 or len(set(request.oof_prediction_paths)) != 3:
            raise ThesisBackendGateError("calibration inner folds and artifacts must be unique")
        calibration_path = request.output_dir / CALIBRATION_FILENAME
        _require_outputs_absent(calibration_path)
        observed_sessions = []
        source_hashes = []
        feature_set = self._validate_condition(request.condition_id, None)
        for inner_fold_id, path_value in zip(
            request.inner_fold_ids, request.oof_prediction_paths, strict=True
        ):
            payload = _load_prediction_artifact(
                path_value,
                expected_inner_fold_id=inner_fold_id,
            )
            if payload["condition_id"] != request.condition_id:
                raise ThesisBackendGateError("inner OOF condition mismatch")
            if payload["feature_set"] != feature_set:
                raise ThesisBackendGateError("inner OOF feature set differs from condition")
            if payload["outer_fold_id"] != request.outer_fold_id:
                raise ThesisBackendGateError("inner OOF outer-fold identity mismatch")
            if payload["seed"] != request.seed:
                raise ThesisBackendGateError("inner OOF seed mismatch")
            if payload["real_session_authority_sha256"] != self.authority.sha256:
                raise ThesisBackendGateError("inner OOF real authority mismatch")
            if payload["source_manifest_sha256"] != self.authority.source_manifest_sha256:
                raise ThesisBackendGateError("inner OOF source manifest mismatch")
            if payload["run_manifest_sha256"] != self.run_binding.run_manifest_sha256:
                raise ThesisBackendGateError("inner OOF run manifest mismatch")
            observed_sessions.extend(record["session_id"] for record in payload["sessions"])
            source_hashes.append(_sha256_file(Path(path_value)))
        if len(observed_sessions) != len(set(observed_sessions)):
            raise ThesisBackendGateError("inner OOF contains repeated validation sessions")
        if tuple(observed_sessions) != request.oof_session_ids:
            raise ThesisBackendGateError("inner OOF sessions do not exactly cover outer train")
        _write_json_exclusive(
            calibration_path,
            {
                **_base_payload("vap_thesis_inner_oof_calibration"),
                "condition_id": request.condition_id,
                "feature_set": feature_set,
                "outer_fold_id": request.outer_fold_id,
                "seed": request.seed,
                "inner_fold_ids": list(request.inner_fold_ids),
                "oof_session_ids": list(request.oof_session_ids),
                "oof_artifact_sha256": source_hashes,
                "real_session_authority_sha256": self.authority.sha256,
                "source_manifest_sha256": self.authority.source_manifest_sha256,
                "run_manifest_sha256": self.run_binding.run_manifest_sha256,
                "threshold_policy": THRESHOLD_POLICY,
                "thresholds": None,
            },
        )
        return CalibrationReceipt(
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            inner_fold_ids=request.inner_fold_ids,
            oof_prediction_paths=request.oof_prediction_paths,
            calibration_path=calibration_path,
        )

    def fit_final(self, request: FinalFitRequest) -> FinalFitReceipt:
        feature_set = self._validate_condition(request.condition_id, request.feature_set)
        checkpoint_path = request.output_dir / FINAL_CHECKPOINT_FILENAME
        _require_outputs_absent(checkpoint_path)
        if request.encoder != self.spec.encoder:
            raise ThesisBackendGateError("request encoder differs from backend spec")
        _validate_sessions(request.train_sessions, "outer train")
        self.authority.validate_subset(request.train_sessions, "outer train")
        if request.fixed_epochs > self.run_binding.max_epochs:
            raise ThesisBackendGateError("final fixed epoch exceeds frozen maximum")
        calibration = _read_json_strict(request.calibration_path)
        calibration_required = {
            *_base_payload("vap_thesis_inner_oof_calibration"),
            "condition_id",
            "feature_set",
            "outer_fold_id",
            "seed",
            "inner_fold_ids",
            "oof_session_ids",
            "oof_artifact_sha256",
            "real_session_authority_sha256",
            "source_manifest_sha256",
            "run_manifest_sha256",
            "threshold_policy",
            "thresholds",
        }
        if set(calibration) != calibration_required:
            raise ThesisBackendGateError("final fit calibration fields are not exact")
        _validate_base_payload(calibration, "vap_thesis_inner_oof_calibration")
        if (
            calibration["threshold_policy"] != THRESHOLD_POLICY
            or calibration["thresholds"] is not None
        ):
            raise ThesisBackendGateError("final fit calibration policy mismatch")
        if (
            calibration["condition_id"] != request.condition_id
            or calibration["feature_set"] != feature_set
        ):
            raise ThesisBackendGateError("final fit calibration condition mismatch")
        if calibration["outer_fold_id"] != request.outer_fold_id:
            raise ThesisBackendGateError("final fit calibration outer fold mismatch")
        if calibration["seed"] != request.seed:
            raise ThesisBackendGateError("final fit calibration seed mismatch")
        if calibration["real_session_authority_sha256"] != self.authority.sha256:
            raise ThesisBackendGateError("final fit calibration real authority mismatch")
        if calibration["source_manifest_sha256"] != self.authority.source_manifest_sha256:
            raise ThesisBackendGateError("final fit calibration source mismatch")
        if calibration["run_manifest_sha256"] != self.run_binding.run_manifest_sha256:
            raise ThesisBackendGateError("final fit calibration run binding mismatch")
        inner_ids = calibration["inner_fold_ids"]
        hashes = calibration["oof_artifact_sha256"]
        if (
            not isinstance(inner_ids, list)
            or len(inner_ids) != 3
            or len(set(inner_ids)) != 3
            or not isinstance(hashes, list)
            or len(hashes) != 3
            or any(not _is_sha256(value) for value in hashes)
        ):
            raise ThesisBackendGateError("final fit calibration OOF bindings are invalid")
        oof_ids = calibration["oof_session_ids"]
        if (
            not isinstance(oof_ids, list)
            or len(oof_ids) != len(set(oof_ids))
            or set(oof_ids) != set(_session_ids(request.train_sessions))
        ):
            raise ThesisBackendGateError("inner OOF coverage differs from outer train")
        normalizer, pos_weight = _validated_statistics(request.normalizer, request.pos_weight)
        stats_before = _statistics_sha256(request.normalizer, np.asarray(request.pos_weight))
        state_before = _session_state_sha256(request.train_sessions)
        with _deterministic_cpu(request.seed):
            fit = _fit(
                self.spec,
                request.train_sessions,
                (),
                normalizer,
                pos_weight,
                epochs=request.fixed_epochs,
                seed=request.seed,
            )
        if stats_before != _statistics_sha256(
            request.normalizer, np.asarray(request.pos_weight)
        ):
            raise ThesisBackendGateError("runner-supplied statistics changed during final fit")
        if state_before != _session_state_sha256(request.train_sessions):
            raise ThesisBackendGateError("runner-supplied sessions changed during final fit")
        _write_torch_exclusive(
            checkpoint_path,
            self._checkpoint_payload(
                fit,
                normalizer,
                pos_weight,
                request.condition_id,
                feature_set,
                request.outer_fold_id,
                request.seed,
                _session_ids(request.train_sessions),
                "fixed_epoch_outer_train_from_scratch",
                None,
            ),
        )
        load_checkpoint_artifact(checkpoint_path)
        train_ids = _session_ids(request.train_sessions)
        return FinalFitReceipt(
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            fixed_epochs=request.fixed_epochs,
            train_session_ids=train_ids,
            normalizer_fit_session_ids=train_ids,
            pos_weight_fit_session_ids=train_ids,
            initialized_from_scratch=True,
            checkpoint_path=checkpoint_path,
        )

    def evaluate_outer(self, request: OuterEvaluationRequest) -> OuterEvaluationReceipt:
        prediction_path = request.output_dir / OUTER_PREDICTIONS_FILENAME
        metric_path = request.output_dir / OUTER_METRICS_FILENAME
        _require_outputs_absent(prediction_path, metric_path)
        _validate_sessions(request.test_sessions, "outer test")
        self.authority.validate_subset(request.test_sessions, "outer test")
        checkpoint_path = Path(request.checkpoint_path).resolve()
        if _sha256_file(checkpoint_path) != request.checkpoint_sha256:
            raise ThesisBackendGateError("outer checkpoint hash mismatch")
        unit = (
            request.condition_id,
            request.outer_fold_id,
            request.seed,
            request.checkpoint_sha256,
        )
        if unit in self._evaluated_units:
            raise ThesisBackendGateError("this outer unit was already predicted")
        checkpoint = load_checkpoint_artifact(checkpoint_path)
        feature_set = self._validate_condition(
            request.condition_id, checkpoint.get("feature_set")
        )
        if ThesisBackendSpec.from_dict(checkpoint["spec"]) != self.spec:
            raise ThesisBackendGateError("outer checkpoint spec differs from backend")
        if checkpoint.get("fit_role") != "fixed_epoch_outer_train_from_scratch":
            raise ThesisBackendGateError("outer evaluation requires a final-fit checkpoint")
        if checkpoint.get("outer_fold_id") != request.outer_fold_id:
            raise ThesisBackendGateError("outer checkpoint fold mismatch")
        if checkpoint.get("seed") != request.seed:
            raise ThesisBackendGateError("outer checkpoint seed mismatch")
        if checkpoint.get("run_binding") != self.run_binding.to_dict():
            raise ThesisBackendGateError("outer checkpoint run binding mismatch")
        if checkpoint.get("real_session_authority_sha256") != self.authority.sha256:
            raise ThesisBackendGateError("outer checkpoint real authority mismatch")
        if checkpoint.get("source_manifest_sha256") != self.authority.source_manifest_sha256:
            raise ThesisBackendGateError("outer checkpoint source manifest mismatch")
        if set(checkpoint.get("train_session_ids", [])) & set(
            _session_ids(request.test_sessions)
        ):
            raise ThesisBackendGateError("outer test overlaps final training sessions")
        spec = ThesisBackendSpec.from_dict(checkpoint["spec"])
        model = _model(spec)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
        self._evaluated_units.add(unit)
        with _deterministic_cpu(request.seed):
            predictions = _predict_sessions(model, request.test_sessions, normalizer, spec)
        _write_json_exclusive(
            prediction_path,
            _prediction_payload(
                request.test_sessions,
                predictions,
                condition_id=request.condition_id,
                feature_set=feature_set,
                outer_fold_id=request.outer_fold_id,
                seed=request.seed,
                checkpoint_sha256=request.checkpoint_sha256,
                prediction_role="outer_test_single_saved_pass",
                authority=self.authority,
                run_binding=self.run_binding,
            ),
        )
        saved_prediction = load_prediction_artifact(prediction_path)
        _write_json_exclusive(
            metric_path,
            {
                **_base_payload("vap_thesis_outer_metrics"),
                "condition_id": request.condition_id,
                "feature_set": feature_set,
                "outer_fold_id": request.outer_fold_id,
                "seed": request.seed,
                "prediction_sha256": _sha256_file(prediction_path),
                "checkpoint_sha256": request.checkpoint_sha256,
                "real_session_authority_sha256": self.authority.sha256,
                "source_manifest_sha256": self.authority.source_manifest_sha256,
                "run_manifest_sha256": self.run_binding.run_manifest_sha256,
                **_metrics(request.test_sessions, predictions),
            },
        )
        saved_metrics = load_metrics_artifact(
            metric_path,
            prediction_path=prediction_path,
        )
        if saved_metrics["prediction_sha256"] != _sha256_file(prediction_path):
            raise ThesisBackendGateError("saved metrics do not bind saved outer predictions")
        if saved_prediction["checkpoint_sha256"] != request.checkpoint_sha256:
            raise ThesisBackendGateError("saved predictions do not bind final checkpoint")
        return OuterEvaluationReceipt(
            condition_id=request.condition_id,
            outer_fold_id=request.outer_fold_id,
            seed=request.seed,
            checkpoint_sha256=request.checkpoint_sha256,
            evaluation_session_ids=_session_ids(request.test_sessions),
            evaluation_count=1,
            predictions_path=prediction_path,
            metrics_path=metric_path,
        )

    def _checkpoint_payload(
        self,
        fit: _FitResult,
        normalizer: FeatureNormalizer,
        pos_weight: np.ndarray,
        condition_id: str,
        feature_set: str,
        outer_fold_id: str,
        seed: int,
        train_session_ids: tuple[str, ...],
        fit_role: str,
        inner_fold_id: str | None,
    ) -> dict[str, object]:
        return {
            **_base_payload("vap_thesis_checkpoint"),
            "spec": self.spec.to_dict(),
            "run_binding": self.run_binding.to_dict(),
            "real_session_authority_sha256": self.authority.sha256,
            "source_manifest_sha256": self.authority.source_manifest_sha256,
            "label_config": json.loads(_canonical_json(asdict(DEFAULT_LABEL_CONFIG))),
            "condition_id": condition_id,
            "feature_set": feature_set,
            "outer_fold_id": outer_fold_id,
            "inner_fold_id": inner_fold_id,
            "seed": seed,
            "fit_role": fit_role,
            "selected_epoch": fit.selected_epoch,
            "initialized_from_scratch": True,
            "train_session_ids": list(train_session_ids),
            "normalizer": normalizer.to_dict(),
            "pos_weight": pos_weight.tolist(),
            "aux_vad_weight": 0.0,
            "history": fit.history,
            "model_state": {
                name: tensor.detach().cpu().clone()
                for name, tensor in fit.model.state_dict().items()
            },
        }


__all__ = [
    "BACKEND_ID",
    "CONDITION_FEATURE_SETS",
    "COUNTED_ELIGIBLE",
    "RESULT_ROLE",
    "THRESHOLD_POLICY",
    "ThesisBackendGateError",
    "ThesisBackendSpec",
    "ThesisTorchBackend",
    "aggregate_outer_prediction_artifacts",
    "load_prediction_artifact",
]
