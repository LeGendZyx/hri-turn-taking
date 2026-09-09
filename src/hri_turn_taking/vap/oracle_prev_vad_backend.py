"""Isolated backend for the one-tick previous-VAD oracle diagnostic.

The core project correctly forbids gold VAD as a deployable model input.  This
module deliberately lives outside that core schema: it creates a matched
``O0``/``O1`` upper-bound diagnostic where both conditions use head pose and an
extra participant channel, while only ``O1`` receives the same-session gold
VAD from exactly one 25 Hz tick earlier.  Every artifact is post-hoc,
non-deployable, non-production and non-counted.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import statistics
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from hri_turn_taking.data.analysis_base import EXPECTED_TICK_RANGES
from hri_turn_taking.data.multiparty import PARTICIPANT_FEATURE_NAMES
from hri_turn_taking.vap.dataset import (
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
    DEADLINE_EXPECTED_N_ROWS,
    DEADLINE_SESSION_TO_GROUP,
)
from hri_turn_taking.vap.trainer import (
    compute_frame_metrics,
    predict_session_probabilities,
)

FORMAT_VERSION = 1
BACKEND_ID = "previous_vad_oracle_torch_backend_v1"
RESULT_ROLE = "posthoc_previous_vad_oracle_outer_evaluation"
FIXTURE_RESULT_ROLE = "previous_vad_oracle_fixture_never_real"
FEATURE_SCHEMA = "multiparty_oracle_prev_vad_v1"
CONDITION_IDS = ("O0", "O1")
BASE_PARTICIPANT_FEATURES = tuple(PARTICIPANT_FEATURE_NAMES)
ORACLE_PARTICIPANT_FEATURES = (*BASE_PARTICIPANT_FEATURES, "previous_vad_oracle")
PREVIOUS_VAD_INDEX = len(ORACLE_PARTICIPANT_FEATURES) - 1
PARTICIPANT_WIDTH = len(ORACLE_PARTICIPANT_FEATURES)
GLOBAL_WIDTH = 5
TRANSFORMED_WIDTH = 4 * PARTICIPANT_WIDTH + GLOBAL_WIDTH
EXACT_EIGHT_GROUP_BY_SCENE = dict(DEADLINE_SESSION_TO_GROUP)
EXACT_EIGHT_GROUPS = tuple(sorted(set(EXACT_EIGHT_GROUP_BY_SCENE.values())))
EXACT_EIGHT_TICK_RANGES = {
    scene: EXPECTED_TICK_RANGES[scene] for scene in EXACT_EIGHT_GROUP_BY_SCENE
}
EXACT_EIGHT_TOTAL_ROWS = DEADLINE_EXPECTED_N_ROWS
if (
    PARTICIPANT_WIDTH != 8
    or TRANSFORMED_WIDTH != 37
    or sum(end - start for start, end in EXACT_EIGHT_TICK_RANGES.values())
    != EXACT_EIGHT_TOTAL_ROWS
):
    raise RuntimeError("previous-VAD oracle schema/corpus constants are inconsistent")
CHECKPOINT_FILENAME = "checkpoint.pt"
PREDICTIONS_FILENAME = "outer_predictions.npz"
PREDICTIONS_SIDECAR_FILENAME = "outer_predictions.json"
RESULT_FILENAME = "result.json"
UNIT_FILENAMES = frozenset(
    {
        CHECKPOINT_FILENAME,
        PREDICTIONS_FILENAME,
        PREDICTIONS_SIDECAR_FILENAME,
        RESULT_FILENAME,
    }
)


class OraclePosthocBackendError(ValueError):
    """A schema, split, training, or saved-artifact gate failed closed."""


def _boundary() -> dict[str, object]:
    return {
        "posthoc_exploratory": True,
        "oracle_gold_input": True,
        "robot_observable": False,
        "deployable": False,
        "production_eligible": False,
        "counted_eligible": False,
        "formal_result_eligible": False,
    }


def _artifact_contract(fixture_only: bool) -> dict[str, str]:
    if type(fixture_only) is not bool:
        _fail("fixture_only must be an exact boolean")
    if fixture_only:
        return {
            "checkpoint": "vap_previous_vad_oracle_fixture_checkpoint_never_real",
            "predictions": (
                "vap_previous_vad_oracle_fixture_saved_probabilities_never_real"
            ),
            "result": "vap_previous_vad_oracle_fixture_result_never_real",
            "result_role": FIXTURE_RESULT_ROLE,
        }
    return {
        "checkpoint": "vap_previous_vad_oracle_checkpoint",
        "predictions": "vap_previous_vad_oracle_saved_outer_probabilities",
        "result": "vap_previous_vad_oracle_outer_result",
        "result_role": RESULT_ROLE,
    }


@dataclass(frozen=True)
class OracleBackendSpec:
    """Frozen specification shared by every paired O0/O1 unit."""

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
        integer_fields = (
            "d_model",
            "n_layers",
            "n_heads",
            "window_frames",
            "stride_frames",
            "batch_size",
            "early_stopping_patience",
            "inference_chunk_frames",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise OraclePosthocBackendError(f"{name} must be a positive integer")
        float_fields = (
            "dropout",
            "learning_rate",
            "weight_decay",
            "warmup_fraction",
            "gradient_clip",
        )
        for name in float_fields:
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise OraclePosthocBackendError(f"{name} must be a finite float")
        if self.d_model % self.n_heads:
            raise OraclePosthocBackendError("d_model must be divisible by n_heads")
        if self.encoder not in {"transformer", "gru"}:
            raise OraclePosthocBackendError("encoder must be transformer or gru")
        if not 0.0 <= self.dropout < 1.0:
            raise OraclePosthocBackendError("dropout must be in [0, 1)")
        if self.learning_rate <= 0.0 or self.gradient_clip <= 0.0:
            raise OraclePosthocBackendError("learning_rate and gradient_clip must be > 0")
        if self.weight_decay < 0.0:
            raise OraclePosthocBackendError("weight_decay must be >= 0")
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise OraclePosthocBackendError("warmup_fraction must be in [0, 1]")
        if self.stride_frames > self.window_frames:
            raise OraclePosthocBackendError("stride_frames cannot leave gaps")
        if self.inference_chunk_frames <= self.window_frames * self.n_layers:
            raise OraclePosthocBackendError(
                "inference_chunk_frames must exceed stacked temporal context"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> OracleBackendSpec:
        if not isinstance(value, dict) or set(value) != set(asdict(cls())):
            _fail("backend spec fields are not exact")
        return cls(**value)


def oracle_model_config(spec: OracleBackendSpec) -> dict[str, object]:
    return {
        "n_participant_features": PARTICIPANT_WIDTH,
        "n_global_features": GLOBAL_WIDTH,
        "n_participants": 4,
        "n_bins": 4,
        "d_model": spec.d_model,
        "n_layers": spec.n_layers,
        "n_heads": spec.n_heads,
        "dropout": spec.dropout,
        "encoder": spec.encoder,
        "attention_window": spec.window_frames,
    }


def oracle_trainer_spec(spec: OracleBackendSpec) -> dict[str, object]:
    return {
        "optimizer": "AdamW",
        "loss": "masked_per_bin_pos_weighted_BCE",
        "scheduler": "linear_warmup_cosine_to_0.1",
        "device": "cpu_deterministic_single_thread",
        "epoch_selection": "median_of_three_inner_LOGO_best_mAP_epochs",
        "max_epochs": 15,
        "seed": 13,
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


def oracle_label_config() -> dict[str, object]:
    return {
        "frame_rate_hz": DEFAULT_LABEL_CONFIG.frame_rate_hz,
        "bin_edges_s": list(DEFAULT_LABEL_CONFIG.bin_edges_s),
        "active_ratio": DEFAULT_LABEL_CONFIG.active_ratio,
        "now_bins": list(DEFAULT_LABEL_CONFIG.now_bins),
        "future_bins": list(DEFAULT_LABEL_CONFIG.future_bins),
    }


def oracle_condition_metadata(condition_id: str) -> dict[str, object]:
    """Return the exact matched O0/O1 condition semantics."""

    if condition_id not in CONDITION_IDS:
        _fail(f"unknown previous-VAD oracle condition: {condition_id!r}")
    enabled = condition_id == "O1"
    return {
        "condition_id": condition_id,
        "feature_schema": FEATURE_SCHEMA,
        "base_feature_set": "head_pose_only",
        "previous_vad_channel_enabled": enabled,
        "previous_vad_delay_ticks": 1,
        "previous_vad_delay_seconds": 0.04,
        "previous_vad_source": "human-reviewed-Transcript-derived VAD",
        "first_session_tick_value": 0.0,
        "normalization": "identity_mean_0_std_1_for_oracle_channel",
        "description": (
            "H-only plus an all-zero schema-matched oracle channel"
            if not enabled
            else "H-only plus same-session one-tick-lagged gold VAD"
        ),
        **_boundary(),
    }


@dataclass(frozen=True)
class OracleFeatureNormalizer:
    """Slot-shared normalizer for the isolated 8+5 oracle layout.

    The previous-VAD channel deliberately keeps the identity transform in both
    O0 and O1.  This prevents condition-specific prevalence statistics from
    becoming a second experimental difference and keeps O0 exactly zero at the
    model input.
    """

    participant_mean: np.ndarray
    participant_std: np.ndarray
    global_mean: np.ndarray
    global_std: np.ndarray
    feature_schema: str = FEATURE_SCHEMA

    @classmethod
    def fit(cls, sessions: Sequence[SessionTensors]) -> OracleFeatureNormalizer:
        _validate_sessions(sessions, "normalizer fit")
        participant_rows: list[np.ndarray] = []
        global_rows: list[np.ndarray] = []
        for session in sessions:
            n_frames = session.features.shape[0]
            per_slot = session.features[:, :-GLOBAL_WIDTH].reshape(
                n_frames, 4, PARTICIPANT_WIDTH
            )
            for slot in np.flatnonzero(session.present > 0):
                participant_rows.append(per_slot[:, slot, :])
            global_rows.append(session.features[:, -GLOBAL_WIDTH:])
        if not participant_rows or not global_rows:
            _fail("normalizer found no present training-slot observations")

        def stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mean = values.mean(axis=0).astype(np.float32)
            std = values.std(axis=0).astype(np.float32)
            std[std < 1e-6] = 1.0
            return mean, std

        participant_mean, participant_std = stats(
            np.concatenate(participant_rows, axis=0)
        )
        global_mean, global_std = stats(np.concatenate(global_rows, axis=0))
        participant_mean[PREVIOUS_VAD_INDEX] = 0.0
        participant_std[PREVIOUS_VAD_INDEX] = 1.0
        return cls(participant_mean, participant_std, global_mean, global_std)

    def __post_init__(self) -> None:
        if self.feature_schema != FEATURE_SCHEMA:
            _fail("oracle normalizer feature schema differs")
        arrays = (
            np.asarray(self.participant_mean),
            np.asarray(self.participant_std),
            np.asarray(self.global_mean),
            np.asarray(self.global_std),
        )
        if tuple(array.shape for array in arrays) != (
            (PARTICIPANT_WIDTH,),
            (PARTICIPANT_WIDTH,),
            (GLOBAL_WIDTH,),
            (GLOBAL_WIDTH,),
        ):
            _fail("oracle normalizer statistic shapes differ")
        if not all(np.isfinite(array).all() for array in arrays):
            _fail("oracle normalizer contains non-finite statistics")
        if (arrays[1] <= 0).any() or (arrays[3] <= 0).any():
            _fail("oracle normalizer standard deviations must be positive")
        if (
            float(arrays[0][PREVIOUS_VAD_INDEX]) != 0.0
            or float(arrays[1][PREVIOUS_VAD_INDEX]) != 1.0
        ):
            _fail("previous-VAD channel must retain identity normalization")

    def apply(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != TRANSFORMED_WIDTH:
            _fail("features do not match the isolated oracle schema")
        n_frames = values.shape[0]
        per_slot = values[:, :-GLOBAL_WIDTH].reshape(
            n_frames, 4, PARTICIPANT_WIDTH
        )
        previous = per_slot[:, :, PREVIOUS_VAD_INDEX]
        if not np.isin(previous, (0.0, 1.0)).all():
            _fail("previous-VAD channel must remain binary before normalization")
        normalized = (per_slot - self.participant_mean) / self.participant_std
        normalized[:, :, PREVIOUS_VAD_INDEX] = previous
        global_block = (
            values[:, -GLOBAL_WIDTH:] - self.global_mean
        ) / self.global_std
        return np.concatenate(
            [normalized.reshape(n_frames, -1), global_block], axis=1
        ).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_schema": self.feature_schema,
            "participant_mean": np.asarray(
                self.participant_mean, dtype=np.float32
            ).tolist(),
            "participant_std": np.asarray(
                self.participant_std, dtype=np.float32
            ).tolist(),
            "global_mean": np.asarray(self.global_mean, dtype=np.float32).tolist(),
            "global_std": np.asarray(self.global_std, dtype=np.float32).tolist(),
            "previous_vad_identity_normalized": True,
        }

    @classmethod
    def from_dict(cls, payload: object) -> OracleFeatureNormalizer:
        expected = {
            "feature_schema",
            "participant_mean",
            "participant_std",
            "global_mean",
            "global_std",
            "previous_vad_identity_normalized",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            _fail("oracle normalizer payload fields are not exact")
        if payload["previous_vad_identity_normalized"] is not True:
            _fail("oracle normalizer identity-channel declaration differs")
        return cls(
            participant_mean=np.asarray(payload["participant_mean"], dtype=np.float32),
            participant_std=np.asarray(payload["participant_std"], dtype=np.float32),
            global_mean=np.asarray(payload["global_mean"], dtype=np.float32),
            global_std=np.asarray(payload["global_std"], dtype=np.float32),
            feature_schema=str(payload["feature_schema"]),
        )


@dataclass(frozen=True)
class OracleUnitRequest:
    condition_id: str
    fold_id: str
    outer_test_group: str
    outer_train_groups: tuple[str, ...]
    inner_validation_groups: tuple[str, ...]
    sessions: tuple[SessionTensors, ...]
    output_directory: Path
    source_snapshot_sha256: str
    protocol_sha256: str
    expected_initial_state_sha256: str
    seed: int = 13
    max_epochs: int = 15
    aux_vad_weight: float = 0.0
    fixture_only: bool = False


@dataclass(frozen=True)
class OracleUnitReceipt:
    output_directory: Path
    checkpoint_path: Path
    predictions_path: Path
    predictions_sidecar_path: Path
    result_path: Path
    result: dict[str, object]


@dataclass(frozen=True)
class LoadedOracleUnit:
    result: dict[str, object]
    sidecar: dict[str, object]
    predictions: tuple[np.ndarray, ...]


@dataclass
class _FitResult:
    model: MultipartyVAP
    history: list[dict[str, object]]
    selected_epoch: int
    initial_state_sha256: str


def _fail(message: str) -> None:
    raise OraclePosthocBackendError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_bytes(value: object) -> bytes:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return text.encode("utf-8")


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OraclePosthocBackendError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                OraclePosthocBackendError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OraclePosthocBackendError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"JSON artifact must contain an object: {path}")
    return value


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


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_binding(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    stat = path.stat()
    if not path.is_file() or path.is_symlink():
        _fail(f"artifact must be a regular non-symlink file: {path}")
    recorded_path = path.relative_to(relative_to).as_posix() if relative_to else str(path)
    return {
        "path": recorded_path,
        "sha256": _sha256_file(path),
        "size_bytes": stat.st_size,
    }


def _validate_file_binding(
    value: object,
    root: Path,
    expected_name: str,
) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        _fail(f"{expected_name} binding fields are not exact")
    if value["path"] != expected_name:
        _fail(f"{expected_name} binding path is invalid")
    path = root / expected_name
    if not path.is_file() or path.is_symlink():
        _fail(f"{expected_name} is missing or not a regular file")
    if type(value["size_bytes"]) is not int or value["size_bytes"] != path.stat().st_size:
        _fail(f"{expected_name} size binding differs")
    if value["sha256"] != _sha256_file(path):
        _fail(f"{expected_name} SHA-256 binding differs")
    return path


def _validate_boundary(value: Mapping[str, object]) -> None:
    for key, expected in _boundary().items():
        if value.get(key) is not expected:
            _fail(f"eligibility boundary {key!r} must be {expected!r}")


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _hash_session_parts(
    session: SessionTensors,
    names: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "session_id": session.session_id,
                "group_id": session.group_id,
                "feature_schema": session.feature_schema,
            }
        ).encode("utf-8")
    )
    for name in names:
        value = np.ascontiguousarray(getattr(session, name))
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def prediction_input_sha256(session: SessionTensors) -> str:
    """Bind the actual inference input, including lagged gold in O1."""

    return _hash_session_parts(session, ("features", "present"))


def evaluation_target_sha256(session: SessionTensors) -> str:
    """Bind held-out labels separately from inference inputs."""

    return _hash_session_parts(session, ("labels", "vad", "valid"))


def oracle_source_vad_sha256(session: SessionTensors) -> str:
    """Bind the gold VAD source separately from its lagged input derivative."""

    return _hash_session_parts(session, ("vad",))


def _session_sha256(session: SessionTensors) -> str:
    return _hash_session_parts(
        session,
        ("features", "labels", "vad", "valid", "present"),
    )


def _validate_sessions(sessions: Sequence[SessionTensors], role: str) -> None:
    if not sessions:
        _fail(f"{role} sessions must be non-empty")
    seen: set[str] = set()
    for session in sessions:
        if not isinstance(session, SessionTensors):
            _fail(f"{role} contains a non-SessionTensors value")
        if session.session_id in seen or not session.session_id:
            _fail(f"{role} session identifiers must be unique and non-empty")
        if session.feature_schema != FEATURE_SCHEMA:
            _fail(f"{role} must use {FEATURE_SCHEMA!r}")
        if session.group_id != EXACT_EIGHT_GROUP_BY_SCENE.get(session.session_id):
            _fail(f"{role} group provenance differs for {session.session_id!r}")
        n_frames = session.features.shape[0]
        if session.features.shape != (n_frames, 37):
            _fail(f"{role} transformed feature width must be 37")
        if session.labels.shape != (n_frames, 4, 4):
            _fail(f"{role} future-label shape is invalid")
        if session.vad.shape != (n_frames, 4) or session.valid.shape != (n_frames,):
            _fail(f"{role} VAD/future-valid shape is invalid")
        if session.present.shape != (4,):
            _fail(f"{role} participant-presence shape is invalid")
        if not all(
            np.isfinite(value).all()
            for value in (session.features, session.labels, session.vad, session.present)
        ):
            _fail(f"{role} contains non-finite tensor values")
        previous = session.features[:, :-GLOBAL_WIDTH].reshape(
            n_frames, 4, PARTICIPANT_WIDTH
        )[:, :, PREVIOUS_VAD_INDEX]
        if not np.isin(previous, (0.0, 1.0)).all():
            _fail(f"{role} previous-VAD channel must be binary")
        if (previous[:, session.present <= 0] != 0.0).any():
            _fail(f"{role} absent slots carry previous-VAD input")
        seen.add(session.session_id)


def validate_oracle_condition_sessions(
    sessions: Sequence[SessionTensors],
    condition_id: str,
    *,
    role: str = "Oracle condition",
) -> None:
    """Prove that the added channel has the declared O0/O1 temporal semantics."""

    if condition_id not in CONDITION_IDS:
        _fail(f"unknown Oracle condition: {condition_id!r}")
    _validate_sessions(sessions, role)
    for session in sessions:
        previous = session.features[:, :-GLOBAL_WIDTH].reshape(
            session.features.shape[0], 4, PARTICIPANT_WIDTH
        )[:, :, PREVIOUS_VAD_INDEX]
        if condition_id == "O0":
            if np.count_nonzero(previous):
                _fail(f"{role} O0 channel is not all zero for {session.session_id}")
            continue
        expected = np.zeros_like(previous)
        if len(expected) > 1:
            expected[1:] = session.vad[:-1]
        expected[:, session.present <= 0] = 0.0
        if not np.array_equal(previous, expected):
            _fail(
                f"{role} O1 channel is not exact same-session lag-1 VAD for "
                f"{session.session_id}"
            )


def build_oracle_condition_sessions(
    frames: Mapping[str, pd.DataFrame],
    condition_id: str,
    *,
    scene_to_group: Mapping[str, str],
) -> tuple[SessionTensors, ...]:
    """Build H-only O0/O1 sessions and shift VAD within each session only."""

    if condition_id not in CONDITION_IDS:
        _fail(f"unknown Oracle condition: {condition_id!r}")
    expected = tuple(EXACT_EIGHT_TICK_RANGES)
    if tuple(frames) != expected:
        _fail("audited Oracle frame inventory/order differs from exact-eight")
    if not isinstance(scene_to_group, Mapping) or tuple(scene_to_group) != expected:
        _fail("audited scene-to-group mapping inventory/order differs from exact-eight")
    expected_mapping = dict(EXACT_EIGHT_GROUP_BY_SCENE)
    if dict(scene_to_group) != expected_mapping:
        _fail("audited scene-to-group mapping differs from the frozen mapping")
    sessions: list[SessionTensors] = []
    total_rows = 0
    for scene in expected:
        frame = frames[scene]
        if not isinstance(frame, pd.DataFrame):
            _fail(f"audited frame for {scene} is not a DataFrame")
        start, end = EXACT_EIGHT_TICK_RANGES[scene]
        if len(frame) != end - start:
            _fail(f"audited frame count differs for {scene}")
        if "tick_25hz" in frame:
            ticks = frame["tick_25hz"].to_numpy()
            if not np.issubdtype(ticks.dtype, np.integer):
                _fail(f"{scene} tick_25hz must retain an integer dtype")
            if not np.array_equal(ticks, np.arange(start, end, dtype=ticks.dtype)):
                _fail(f"{scene} tick_25hz range differs from the frozen window")
        declared_sessions = frame["session_id"].astype(str).unique().tolist()
        if declared_sessions != [scene]:
            _fail(f"{scene} frame contains a different session identity")
        expected_group = scene_to_group[scene]
        prepared = frame
        if "group_id" in frame:
            groups = frame["group_id"].astype(str).unique().tolist()
            if groups != [expected_group]:
                _fail(f"{scene} frame group_id conflicts with audited manifest")
        else:
            prepared = frame.copy(deep=False)
            prepared.insert(1, "group_id", expected_group)
        masked = apply_feature_set(prepared, "head_pose_only")
        built = sessions_from_dataframe(masked)
        if len(built) != 1 or built[0].session_id != scene:
            _fail(f"{scene} did not build exactly one tensor session")
        base = built[0]
        n_frames = base.features.shape[0]
        if base.features.shape != (n_frames, 33):
            _fail(f"{scene} base H-only transformed width differs")
        base_participant = base.features[:, :-GLOBAL_WIDTH].reshape(
            n_frames, 4, len(BASE_PARTICIPANT_FEATURES)
        )
        previous = np.zeros((n_frames, 4), dtype=np.float32)
        if condition_id == "O1" and n_frames > 1:
            previous[1:] = np.asarray(base.vad[:-1], dtype=np.float32)
        previous[:, base.present <= 0] = 0.0
        participant = np.concatenate(
            [base_participant, previous[:, :, None]], axis=2
        )
        features = np.concatenate(
            [participant.reshape(n_frames, -1), base.features[:, -GLOBAL_WIDTH:]],
            axis=1,
        ).astype(np.float32)
        session = replace(base, features=features, feature_schema=FEATURE_SCHEMA)
        observed = session.features[:, :-GLOBAL_WIDTH].reshape(
            n_frames, 4, PARTICIPANT_WIDTH
        )[:, :, PREVIOUS_VAD_INDEX]
        if condition_id == "O0" and np.count_nonzero(observed):
            _fail(f"{scene} O0 previous-VAD channel is not all zero")
        if condition_id == "O1" and (
            np.count_nonzero(observed[0])
            or not np.array_equal(observed[1:], base.vad[:-1])
        ):
            _fail(f"{scene} O1 is not an exact same-session one-tick VAD lag")
        sessions.append(session)
        total_rows += len(frame)
    if total_rows != EXACT_EIGHT_TOTAL_ROWS:
        _fail("Oracle condition tensor total differs from 64,299 rows")
    validate_oracle_condition_sessions(sessions, condition_id, role=f"condition {condition_id}")
    return tuple(sessions)


def _select_groups(
    sessions: Sequence[SessionTensors], groups: Sequence[str]
) -> tuple[SessionTensors, ...]:
    group_set = set(groups)
    return tuple(session for session in sessions if session.group_id in group_set)


def fit_oracle_training_normalizer(
    sessions: Sequence[SessionTensors],
    training_groups: Sequence[str],
) -> OracleFeatureNormalizer:
    """Fit H statistics on training groups; keep the oracle channel identity."""

    _validate_sessions(sessions, "normalizer source")
    groups = tuple(training_groups)
    if not groups or len(groups) != len(set(groups)):
        _fail("normalizer training groups must be non-empty and unique")
    selected = _select_groups(sessions, groups)
    if {str(session.group_id) for session in selected} != set(groups):
        _fail("normalizer training-group selection is incomplete")
    return OracleFeatureNormalizer.fit(selected)


def _fit_pos_weight(sessions: Sequence[SessionTensors]) -> np.ndarray:
    positives = np.zeros(4, dtype=np.float64)
    negatives = np.zeros(4, dtype=np.float64)
    for session in sessions:
        mask = session.valid[:, None] & (session.present[None, :] > 0)
        selected = session.labels[mask]
        positives += selected.sum(axis=0)
        negatives += selected.shape[0] - selected.sum(axis=0)
    if (positives <= 0).any() or (negatives <= 0).any():
        _fail("training partition has a single-class future bin")
    return np.clip(negatives / positives, 1.0, 10.0).astype(np.float32)


def _model(spec: OracleBackendSpec) -> MultipartyVAP:
    return MultipartyVAP(VAPModelConfig(**oracle_model_config(spec)))


def model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            _fail("model state must map string names to tensors")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def initial_model_state_sha256(spec: OracleBackendSpec, seed: int = 13) -> str:
    """Hash the exact CPU initial weights recreated before every paired fit."""

    if seed != 13:
        _fail("Oracle initial-state strategy is frozen to seed 13")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _model(spec).cpu()
    return model_state_sha256(model.state_dict())


def _training_spans(
    sessions: Sequence[SessionTensors], spec: OracleBackendSpec
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
        _fail("training partition contains no valid VAP window")
    return spans


def _batches(
    order: Sequence[int],
    spans: Sequence[tuple[int, int, int]],
    batch_size: int,
) -> Iterator[list[int]]:
    pending: dict[int, list[int]] = {}
    for index in order:
        _, start, end = spans[index]
        bucket = pending.setdefault(end - start, [])
        bucket.append(index)
        if len(bucket) == batch_size:
            yield bucket.copy()
            bucket.clear()
    for length in sorted(pending):
        if pending[length]:
            yield pending[length].copy()


def _batch_loss(
    model: MultipartyVAP,
    sessions: Sequence[SessionTensors],
    normalized: Sequence[np.ndarray],
    spans: Sequence[tuple[int, int, int]],
    indices: Sequence[int],
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
        output["bin_logits"],
        label_batch,
        pos_weight=pos_weight,
        reduction="none",
    )
    expanded = mask.unsqueeze(-1).expand_as(losses)
    return (losses * expanded).sum() / expanded.sum().clamp(min=1.0)


def _predict(
    model: MultipartyVAP,
    sessions: Sequence[SessionTensors],
    normalizer: OracleFeatureNormalizer,
    spec: OracleBackendSpec,
) -> tuple[np.ndarray, ...]:
    model.cpu().eval()
    predictions = []
    for session in sessions:
        probability = predict_session_probabilities(
            model,
            session,
            normalizer,
            device="cpu",
            chunk_frames=spec.inference_chunk_frames,
        )
        if probability.shape != (session.features.shape[0], 4, 4):
            _fail("prediction shape differs from (time, participant, bin)")
        if not np.isfinite(probability).all():
            _fail("prediction contains non-finite values")
        predictions.append(probability.astype(np.float32, copy=False))
    return tuple(predictions)


def _frame_metrics(
    sessions: Sequence[SessionTensors], predictions: Sequence[np.ndarray]
) -> dict[str, object]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    participants: list[np.ndarray] = []
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


def oracle_metrics_from_predictions(
    sessions: Sequence[SessionTensors], predictions: Sequence[np.ndarray]
) -> dict[str, object]:
    """Compute pooled frame/event metrics from saved probabilities only."""

    _validate_sessions(sessions, "evaluation")
    if len(sessions) != len(predictions):
        _fail("prediction/session counts differ")
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
    checked_predictions: list[np.ndarray] = []
    for session, raw_probability in zip(sessions, predictions, strict=True):
        probability = np.asarray(raw_probability, dtype=np.float32)
        if probability.shape != (session.features.shape[0], 4, 4):
            _fail(f"saved prediction shape differs for {session.session_id}")
        if not np.isfinite(probability).all() or not (
            (probability >= 0.0).all() and (probability <= 1.0).all()
        ):
            _fail(f"saved predictions are not finite probabilities for {session.session_id}")
        checked_predictions.append(probability)
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
    score_array = (
        np.asarray(next_scores, dtype=float)
        if next_scores
        else np.empty((0, 4), dtype=float)
    )
    events["next_speaker"] = asdict(
        score_next_speaker(np.asarray(next_labels), score_array)
    )
    return {
        "threshold_policy": "diagnostic_fixed_0.5_not_calibrated",
        "frame": _json_safe(_frame_metrics(sessions, checked_predictions)),
        "events": _json_safe(events),
    }


def _fit(
    spec: OracleBackendSpec,
    train_sessions: tuple[SessionTensors, ...],
    validation_sessions: tuple[SessionTensors, ...],
    normalizer: OracleFeatureNormalizer,
    pos_weight: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> _FitResult:
    torch.manual_seed(seed)
    model = _model(spec).cpu()
    initial_state = model_state_sha256(model.state_dict())
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
        losses: list[float] = []
        for indices in _batches(order, spans, spec.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(
                model,
                train_sessions,
                normalized,
                spans,
                indices,
                pos_tensor,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        if not losses:
            _fail("epoch contains no optimizer step")
        validation_map: float | None = None
        if validation_sessions:
            predictions = _predict(model, validation_sessions, normalizer, spec)
            validation_map = _frame_metrics(validation_sessions, predictions)["mAP"]
            if (
                validation_map is not None
                and math.isfinite(float(validation_map))
                and float(validation_map) > best_map + 1e-4
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
            _fail("strict inner-validation mAP is unavailable")
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
    else:
        selected_epoch = epochs
    model.eval()
    return _FitResult(
        model=model,
        history=history,
        selected_epoch=selected_epoch,
        initial_state_sha256=initial_state,
    )


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


def _write_torch_exclusive(path: Path, value: object) -> None:
    with path.open("xb") as handle:
        torch.save(value, handle)


def _npy_bytes(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    if array.dtype.hasobject:
        _fail("saved arrays must never use object dtype")
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _write_npz_exclusive(path: Path, probabilities: np.ndarray, offsets: np.ndarray) -> None:
    with path.open("xb") as handle:
        with zipfile.ZipFile(
            handle,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, value in (
                ("probabilities.npy", probabilities),
                ("session_offsets.npy", offsets),
            ):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(value), compresslevel=9)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def _validate_request(request: OracleUnitRequest) -> None:
    if request.condition_id not in CONDITION_IDS:
        _fail("unit condition must be O0 or O1")
    if request.fold_id not in {f"outer_{index:02d}" for index in range(1, 5)}:
        _fail("unit fold id is invalid")
    expected_test_group = EXACT_EIGHT_GROUPS[int(request.fold_id[-2:]) - 1]
    if request.outer_test_group != expected_test_group:
        _fail("unit outer test group differs from its frozen fold id")
    if (
        type(request.seed) is not int
        or request.seed != 13
        or type(request.max_epochs) is not int
        or request.max_epochs != 15
        or type(request.aux_vad_weight) is not float
        or request.aux_vad_weight != 0.0
    ):
        _fail("unit must retain seed=13, max_epochs=15 and aux_vad_weight=0.0")
    if request.outer_test_group in request.outer_train_groups:
        _fail("outer test group leaks into training groups")
    if tuple(request.inner_validation_groups) != tuple(request.outer_train_groups):
        _fail("inner LOGO validation groups must equal the three outer-training groups")
    if len(request.outer_train_groups) != 3 or len(set(request.outer_train_groups)) != 3:
        _fail("each outer unit must train on exactly three groups")
    if not _is_sha256(request.source_snapshot_sha256) or not _is_sha256(
        request.protocol_sha256
    ):
        _fail("unit is missing source/protocol SHA-256 bindings")
    if not _is_sha256(request.expected_initial_state_sha256):
        _fail("unit is missing the frozen initial-state SHA-256")
    if type(request.fixture_only) is not bool:
        _fail("unit fixture_only flag must be an exact boolean")
    _validate_sessions(request.sessions, "unit")
    validate_oracle_condition_sessions(
        request.sessions,
        request.condition_id,
        role="unit",
    )
    if tuple(session.session_id for session in request.sessions) != tuple(
        EXACT_EIGHT_TICK_RANGES
    ):
        _fail("unit session inventory/order differs from exact-eight")
    if sum(session.features.shape[0] for session in request.sessions) != (
        EXACT_EIGHT_TOTAL_ROWS
    ):
        _fail("unit tensor total differs from 64,299 rows")
    groups = {str(session.group_id) for session in request.sessions}
    if groups != {*request.outer_train_groups, request.outer_test_group}:
        _fail("unit sessions do not cover the exact four frozen groups")


_HISTORY_KEYS = {
    "epoch",
    "train_loss",
    "validation_mAP",
    "learning_rate",
    "n_training_windows",
    "n_optimizer_steps",
}
_CHECKPOINT_CORE_KEYS = {
    "backend_spec",
    "model_config",
    "trainer_spec",
    "label_config",
    "model_state",
    "model_state_sha256",
    "normalizer",
    "outer_train_groups",
    "outer_train_session_ids",
    "outer_test_session_ids",
    "selected_final_epochs",
    "final_history",
    "initial_state_sha256",
}
_INNER_FIT_KEYS = {
    "inner_fold_id",
    "training_groups",
    "validation_group",
    "training_sessions",
    "validation_sessions",
    "selected_epoch",
    "initial_state_sha256",
    "validation_metrics",
    "history",
}


def _scene_ids_for_groups(groups: Sequence[str]) -> list[str]:
    selected = set(groups)
    return [
        scene
        for scene, group in EXACT_EIGHT_GROUP_BY_SCENE.items()
        if group in selected
    ]


def _validate_history(
    value: object,
    *,
    selected_epoch: int,
    final_fit: bool,
) -> None:
    if not isinstance(value, list) or not value or len(value) > 15:
        _fail("training history length is invalid")
    replay_best_epoch: int | None = None
    replay_best_map = -math.inf
    for epoch, record in enumerate(value, start=1):
        if not isinstance(record, dict) or set(record) != _HISTORY_KEYS:
            _fail("training history fields are not exact")
        if record["epoch"] != epoch:
            _fail("training history epochs are not consecutive")
        for name in ("train_loss", "learning_rate"):
            number = record[name]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                _fail(f"training history {name} is not numeric")
            if not math.isfinite(float(number)):
                _fail(f"training history {name} is not finite")
        validation_map = record["validation_mAP"]
        if validation_map is not None and (
            isinstance(validation_map, bool)
            or not isinstance(validation_map, (int, float))
            or not math.isfinite(float(validation_map))
        ):
            _fail("training history validation_mAP is invalid")
        if final_fit and validation_map is not None:
            _fail("final-fit history must not contain validation metrics")
        if (
            not final_fit
            and validation_map is not None
            and float(validation_map) > replay_best_map + 1e-4
        ):
            replay_best_map = float(validation_map)
            replay_best_epoch = epoch
        for name in ("n_training_windows", "n_optimizer_steps"):
            count = record[name]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                _fail(f"training history {name} is invalid")
    if not 1 <= selected_epoch <= len(value):
        _fail("selected epoch is outside its training history")
    if final_fit and len(value) != selected_epoch:
        _fail("final-fit history length must equal the selected epoch budget")
    if not final_fit and replay_best_epoch != selected_epoch:
        _fail("inner selected epoch does not replay from validation history")


def _validate_metrics_shape(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "threshold_policy",
        "frame",
        "events",
    }:
        _fail("metrics fields are not exact")
    if value["threshold_policy"] != "diagnostic_fixed_0.5_not_calibrated":
        _fail("metrics threshold policy differs")
    if not isinstance(value["frame"], dict) or not isinstance(value["events"], dict):
        _fail("metrics frame/events payloads are invalid")


def _validate_inner_fits(
    value: object,
    *,
    outer_train_groups: Sequence[str],
    expected_initial_state_sha256: str,
    selected_final_epochs: int,
) -> None:
    groups = tuple(outer_train_groups)
    if not isinstance(value, list) or len(value) != 3:
        _fail("unit must record exactly three inner LOGO fits")
    selected_epochs: list[int] = []
    for index, (record, validation_group) in enumerate(
        zip(value, groups, strict=True), start=1
    ):
        if not isinstance(record, dict) or set(record) != _INNER_FIT_KEYS:
            _fail("inner-fit fields are not exact")
        training_groups = tuple(group for group in groups if group != validation_group)
        expected = {
            "inner_fold_id": f"inner_{index:02d}",
            "training_groups": list(training_groups),
            "validation_group": validation_group,
            "training_sessions": _scene_ids_for_groups(training_groups),
            "validation_sessions": _scene_ids_for_groups((validation_group,)),
            "initial_state_sha256": expected_initial_state_sha256,
        }
        for key, expected_value in expected.items():
            if record[key] != expected_value:
                _fail(f"inner-fit field {key!r} differs")
        selected = record["selected_epoch"]
        if isinstance(selected, bool) or not isinstance(selected, int):
            _fail("inner-fit selected_epoch is invalid")
        _validate_history(record["history"], selected_epoch=selected, final_fit=False)
        _validate_metrics_shape(record["validation_metrics"])
        selected_epochs.append(selected)
    if selected_final_epochs != int(statistics.median(selected_epochs)):
        _fail("selected final epoch is not the median inner-LOGO epoch")


def _validate_checkpoint_core_payload(
    value: Mapping[str, object],
    *,
    outer_test_group: str,
    selected_final_epochs: int,
    expected_initial_state_sha256: str,
) -> None:
    if set(value) != _CHECKPOINT_CORE_KEYS:
        _fail("checkpoint core payload fields are not exact")
    spec = OracleBackendSpec.from_dict(value["backend_spec"])
    if spec != OracleBackendSpec():
        _fail("checkpoint backend spec differs from the frozen backend")
    model_config = value["model_config"]
    if not isinstance(model_config, dict) or _canonical_json(
        model_config
    ) != _canonical_json(oracle_model_config(spec)):
        _fail("checkpoint model config differs from the frozen Oracle model")
    trainer_spec = value["trainer_spec"]
    if not isinstance(trainer_spec, dict) or _canonical_json(
        trainer_spec
    ) != _canonical_json(oracle_trainer_spec(spec)):
        _fail("checkpoint trainer spec differs from the frozen trainer")
    label_config = value["label_config"]
    if not isinstance(label_config, dict) or _canonical_json(
        label_config
    ) != _canonical_json(oracle_label_config()):
        _fail("checkpoint label config differs from the frozen labels")
    if value["selected_final_epochs"] != selected_final_epochs:
        _fail("checkpoint selected epoch differs from the unit result")
    if value["initial_state_sha256"] != expected_initial_state_sha256:
        _fail("checkpoint initial-state hash differs from the frozen protocol")
    if expected_initial_state_sha256 != initial_model_state_sha256(spec, 13):
        _fail("frozen initial-state hash cannot be replayed")
    outer_train_groups = [
        group for group in EXACT_EIGHT_GROUPS if group != outer_test_group
    ]
    expected_membership = {
        "outer_train_groups": outer_train_groups,
        "outer_train_session_ids": _scene_ids_for_groups(outer_train_groups),
        "outer_test_session_ids": _scene_ids_for_groups((outer_test_group,)),
    }
    for key, expected in expected_membership.items():
        if value[key] != expected:
            _fail(f"checkpoint membership field {key!r} differs")
    _validate_history(
        value["final_history"],
        selected_epoch=selected_final_epochs,
        final_fit=True,
    )
    normalizer = value["normalizer"]
    expected_normalizer_keys = {
        "participant_mean",
        "participant_std",
        "global_mean",
        "global_std",
        "feature_schema",
        "previous_vad_identity_normalized",
    }
    if not isinstance(normalizer, dict) or set(normalizer) != expected_normalizer_keys:
        _fail("checkpoint normalizer fields are not exact")
    try:
        loaded = OracleFeatureNormalizer.from_dict(normalizer)
    except (KeyError, TypeError, ValueError) as exc:
        raise OraclePosthocBackendError(f"checkpoint normalizer is invalid: {exc}") from exc
    if loaded.feature_schema != FEATURE_SCHEMA:
        _fail("checkpoint normalizer is not the isolated oracle schema")
    expected_shapes = {
        "participant_mean": (8,),
        "participant_std": (8,),
        "global_mean": (5,),
        "global_std": (5,),
    }
    for name, shape in expected_shapes.items():
        array = getattr(loaded, name)
        if array.shape != shape or not np.isfinite(array).all():
            _fail(f"checkpoint normalizer {name} shape/values are invalid")
        if name.endswith("_std") and (array <= 0.0).any():
            _fail(f"checkpoint normalizer {name} contains a non-positive value")
    state = value["model_state"]
    if not isinstance(state, Mapping) or not state:
        _fail("checkpoint model_state is invalid")
    try:
        model = _model(spec).cpu()
        expected_state = model.state_dict()
        if not all(isinstance(name, str) for name in state) or set(state) != set(
            expected_state
        ):
            _fail("checkpoint model_state keys differ from the frozen model")
        if value["model_state_sha256"] != model_state_sha256(state):
            _fail("checkpoint model-state SHA-256 differs")
        model.load_state_dict(dict(state), strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise OraclePosthocBackendError(
            f"checkpoint model_state cannot be loaded strictly: {exc}"
        ) from exc
    for tensor in model.state_dict().values():
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(
            tensor
        ).all():
            _fail("checkpoint model_state contains non-finite values")


def write_oracle_unit_bundle(
    request: OracleUnitRequest,
    *,
    checkpoint_payload: Mapping[str, object],
    outer_sessions: Sequence[SessionTensors],
    predictions: Sequence[np.ndarray],
    metrics: Mapping[str, object],
    inner_fits: Sequence[Mapping[str, object]],
    selected_final_epochs: int,
    final_history: Sequence[Mapping[str, object]],
) -> OracleUnitReceipt:
    """Publish one unit atomically; also supports fast fixture backends in tests."""

    _validate_request(request)
    if (
        isinstance(selected_final_epochs, bool)
        or not isinstance(selected_final_epochs, int)
        or not 1 <= selected_final_epochs <= request.max_epochs
    ):
        _fail("selected final epoch count is outside the frozen budget")
    _validate_sessions(outer_sessions, "outer test")
    if {session.group_id for session in outer_sessions} != {request.outer_test_group}:
        _fail("outer artifact contains a non-test-group session")
    if [session.session_id for session in outer_sessions] != _scene_ids_for_groups(
        (request.outer_test_group,)
    ):
        _fail("outer artifact session inventory/order differs")
    if len(outer_sessions) != len(predictions):
        _fail("outer prediction/session counts differ")
    checked: list[np.ndarray] = []
    for session, value in zip(outer_sessions, predictions, strict=True):
        probability = np.asarray(value, dtype=np.float32)
        if probability.shape != (session.features.shape[0], 4, 4):
            _fail(f"prediction shape differs for {session.session_id}")
        if not np.isfinite(probability).all() or not (
            (probability >= 0.0).all() and (probability <= 1.0).all()
        ):
            _fail(f"prediction values are invalid for {session.session_id}")
        checked.append(probability)
    safe_metrics = _json_safe(oracle_metrics_from_predictions(outer_sessions, checked))
    if _canonical_json(_json_safe(dict(metrics))) != _canonical_json(safe_metrics):
        _fail("provided outer metrics differ from saved-probability recomputation")
    _validate_metrics_shape(safe_metrics)
    safe_inner_fits = _json_safe(list(inner_fits))
    if not isinstance(safe_inner_fits, list):
        _fail("inner-fit payload is invalid")
    _validate_inner_fits(
        safe_inner_fits,
        outer_train_groups=request.outer_train_groups,
        expected_initial_state_sha256=request.expected_initial_state_sha256,
        selected_final_epochs=selected_final_epochs,
    )
    safe_final_history = _json_safe(list(final_history))
    if not isinstance(safe_final_history, list):
        _fail("final-fit history is invalid")
    _validate_history(
        safe_final_history,
        selected_epoch=selected_final_epochs,
        final_fit=True,
    )
    if checkpoint_payload.get("final_history") != safe_final_history:
        _fail("checkpoint and result final histories differ")
    _validate_checkpoint_core_payload(
        checkpoint_payload,
        outer_test_group=request.outer_test_group,
        selected_final_epochs=selected_final_epochs,
        expected_initial_state_sha256=request.expected_initial_state_sha256,
    )
    contract = _artifact_contract(request.fixture_only)
    destination = Path(request.output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Oracle unit: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-stage-", dir=destination.parent)
    ).resolve()
    try:
        checkpoint = {
            "artifact_kind": contract["checkpoint"],
            "format_version": FORMAT_VERSION,
            "backend_id": BACKEND_ID,
            "result_role": contract["result_role"],
            "fixture_only": request.fixture_only,
            "condition_id": request.condition_id,
            "oracle_input_contract": oracle_condition_metadata(request.condition_id),
            "fold_id": request.fold_id,
            "outer_test_group": request.outer_test_group,
            "feature_schema": FEATURE_SCHEMA,
            "source_snapshot_sha256": request.source_snapshot_sha256,
            "protocol_sha256": request.protocol_sha256,
            "seed": request.seed,
            "max_epochs": request.max_epochs,
            "aux_vad_weight": request.aux_vad_weight,
            "payload": dict(checkpoint_payload),
            **_boundary(),
        }
        checkpoint_path = stage / CHECKPOINT_FILENAME
        _write_torch_exclusive(checkpoint_path, checkpoint)
        concatenated = np.concatenate(checked, axis=0).astype(np.float32, copy=False)
        lengths = [0, *[session.features.shape[0] for session in outer_sessions]]
        offsets = np.cumsum(np.asarray(lengths, dtype=np.int64))
        predictions_path = stage / PREDICTIONS_FILENAME
        _write_npz_exclusive(predictions_path, concatenated, offsets)
        session_records = []
        for index, (session, probability) in enumerate(
            zip(outer_sessions, checked, strict=True)
        ):
            session_records.append(
                {
                    "session_id": session.session_id,
                    "group_id": session.group_id,
                    "n_frames": session.features.shape[0],
                    "start_index": int(offsets[index]),
                    "end_index": int(offsets[index + 1]),
                    "session_tensor_sha256": _session_sha256(session),
                    "prediction_input_sha256": prediction_input_sha256(session),
                    "oracle_source_vad_sha256": oracle_source_vad_sha256(session),
                    "evaluation_target_sha256": evaluation_target_sha256(session),
                    "probabilities_sha256": _sha256_array(probability),
                }
            )
        sidecar = {
            "artifact_kind": contract["predictions"],
            "format_version": FORMAT_VERSION,
            "backend_id": BACKEND_ID,
            "result_role": contract["result_role"],
            "fixture_only": request.fixture_only,
            "condition_id": request.condition_id,
            "oracle_input_contract": oracle_condition_metadata(request.condition_id),
            "fold_id": request.fold_id,
            "outer_test_group": request.outer_test_group,
            "feature_schema": FEATURE_SCHEMA,
            "source_snapshot_sha256": request.source_snapshot_sha256,
            "protocol_sha256": request.protocol_sha256,
            "array_contract": {
                "probabilities_shape": list(concatenated.shape),
                "probabilities_dtype": "float32",
                "session_offsets_shape": list(offsets.shape),
                "session_offsets_dtype": "int64",
            },
            "npz": _file_binding(predictions_path, relative_to=stage),
            "sessions": session_records,
            **_boundary(),
        }
        sidecar_path = stage / PREDICTIONS_SIDECAR_FILENAME
        _write_bytes_exclusive(sidecar_path, _json_bytes(sidecar))
        result = {
            "artifact_kind": contract["result"],
            "format_version": FORMAT_VERSION,
            "backend_id": BACKEND_ID,
            "result_role": contract["result_role"],
            "fixture_only": request.fixture_only,
            "condition_id": request.condition_id,
            "oracle_input_contract": oracle_condition_metadata(request.condition_id),
            "fold_id": request.fold_id,
            "outer_test_group": request.outer_test_group,
            "outer_train_groups": list(request.outer_train_groups),
            "inner_validation_groups": list(request.inner_validation_groups),
            "feature_schema": FEATURE_SCHEMA,
            "schema_dimensions": {
                "conceptual_raw_input_columns": 36,
                "transformed_input_dimensions": TRANSFORMED_WIDTH,
                "participant_features_per_token": PARTICIPANT_WIDTH,
                "global_features_per_token": GLOBAL_WIDTH,
                "token_input_dimensions": PARTICIPANT_WIDTH + GLOBAL_WIDTH,
            },
            "source_snapshot_sha256": request.source_snapshot_sha256,
            "protocol_sha256": request.protocol_sha256,
            "seed": request.seed,
            "max_epochs": request.max_epochs,
            "aux_vad_weight": request.aux_vad_weight,
            "initial_state_sha256": request.expected_initial_state_sha256,
            "selected_final_epochs": selected_final_epochs,
            "inner_fits": safe_inner_fits,
            "final_history": safe_final_history,
            "metrics": safe_metrics,
            "artifacts": {
                "checkpoint": _file_binding(checkpoint_path, relative_to=stage),
                "predictions": _file_binding(predictions_path, relative_to=stage),
                "predictions_sidecar": _file_binding(sidecar_path, relative_to=stage),
            },
            **_boundary(),
        }
        result_path = stage / RESULT_FILENAME
        _write_bytes_exclusive(result_path, _json_bytes(result))
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite Oracle unit: {destination}")
        stage.rename(destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return OracleUnitReceipt(
        output_directory=destination,
        checkpoint_path=destination / CHECKPOINT_FILENAME,
        predictions_path=destination / PREDICTIONS_FILENAME,
        predictions_sidecar_path=destination / PREDICTIONS_SIDECAR_FILENAME,
        result_path=destination / RESULT_FILENAME,
        result=result,
    )


def load_saved_oracle_unit(
    unit_directory: str | Path,
    expected_sessions: Sequence[SessionTensors],
    *,
    condition_id: str,
    fold_id: str,
    outer_test_group: str,
    source_snapshot_sha256: str,
    protocol_sha256: str,
    expected_initial_state_sha256: str,
    fixture_only: bool = False,
) -> LoadedOracleUnit:
    """Audit one unit and load only its saved probabilities, never its model."""

    _validate_sessions(expected_sessions, "saved outer test")
    validate_oracle_condition_sessions(
        expected_sessions,
        condition_id,
        role="saved outer test",
    )
    if [session.session_id for session in expected_sessions] != _scene_ids_for_groups(
        (outer_test_group,)
    ):
        _fail("saved outer-test session inventory/order differs")
    if {str(session.group_id) for session in expected_sessions} != {outer_test_group}:
        _fail("saved outer-test sessions contain a non-test group")
    root = Path(unit_directory).resolve()
    if not root.is_dir() or root.is_symlink():
        _fail(f"unit directory is missing or is a symlink: {root}")
    if {path.name for path in root.iterdir()} != UNIT_FILENAMES:
        _fail("Oracle unit file set is not exact")
    result = _strict_json(root / RESULT_FILENAME)
    sidecar = _strict_json(root / PREDICTIONS_SIDECAR_FILENAME)
    contract = _artifact_contract(fixture_only)
    result_keys = {
        "artifact_kind",
        "format_version",
        "backend_id",
        "result_role",
        "fixture_only",
        "condition_id",
        "oracle_input_contract",
        "fold_id",
        "outer_test_group",
        "outer_train_groups",
        "inner_validation_groups",
        "feature_schema",
        "schema_dimensions",
        "source_snapshot_sha256",
        "protocol_sha256",
        "seed",
        "max_epochs",
        "aux_vad_weight",
        "initial_state_sha256",
        "selected_final_epochs",
        "inner_fits",
        "final_history",
        "metrics",
        "artifacts",
        *_boundary(),
    }
    sidecar_keys = {
        "artifact_kind",
        "format_version",
        "backend_id",
        "result_role",
        "fixture_only",
        "condition_id",
        "oracle_input_contract",
        "fold_id",
        "outer_test_group",
        "feature_schema",
        "source_snapshot_sha256",
        "protocol_sha256",
        "array_contract",
        "npz",
        "sessions",
        *_boundary(),
    }
    if set(result) != result_keys or set(sidecar) != sidecar_keys:
        _fail("Oracle result/sidecar fields are not exact")
    for value, kind in (
        (result, contract["result"]),
        (sidecar, contract["predictions"]),
    ):
        if value.get("artifact_kind") != kind or value.get("format_version") != FORMAT_VERSION:
            _fail(f"Oracle unit {kind} kind/version is invalid")
        if (
            value.get("backend_id") != BACKEND_ID
            or value.get("result_role") != contract["result_role"]
            or value.get("fixture_only") is not fixture_only
        ):
            _fail(f"Oracle unit {kind} backend/result role is invalid")
        _validate_boundary(value)
        expected = {
            "condition_id": condition_id,
            "fold_id": fold_id,
            "outer_test_group": outer_test_group,
            "feature_schema": FEATURE_SCHEMA,
            "source_snapshot_sha256": source_snapshot_sha256,
            "protocol_sha256": protocol_sha256,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                _fail(f"Oracle unit {kind} field {key!r} differs")
        if value.get("oracle_input_contract") != oracle_condition_metadata(condition_id):
            _fail(f"Oracle unit {kind} input contract differs")
    expected_train_groups = [
        group for group in EXACT_EIGHT_GROUPS if group != outer_test_group
    ]
    result_contract = {
        "outer_train_groups": expected_train_groups,
        "inner_validation_groups": expected_train_groups,
        "seed": 13,
        "max_epochs": 15,
        "aux_vad_weight": 0.0,
        "schema_dimensions": {
            "conceptual_raw_input_columns": 36,
            "transformed_input_dimensions": TRANSFORMED_WIDTH,
            "participant_features_per_token": PARTICIPANT_WIDTH,
            "global_features_per_token": GLOBAL_WIDTH,
            "token_input_dimensions": PARTICIPANT_WIDTH + GLOBAL_WIDTH,
        },
    }
    for key, expected in result_contract.items():
        if _canonical_json(result.get(key)) != _canonical_json(expected):
            _fail(f"Oracle result contract field {key!r} differs")
    if result.get("initial_state_sha256") != expected_initial_state_sha256:
        _fail("Oracle result initial-state hash differs from the frozen protocol")
    selected_epoch = result.get("selected_final_epochs")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or not 1 <= selected_epoch <= 15
    ):
        _fail("Oracle result selected_final_epochs is invalid")
    _validate_inner_fits(
        result.get("inner_fits"),
        outer_train_groups=expected_train_groups,
        expected_initial_state_sha256=expected_initial_state_sha256,
        selected_final_epochs=selected_epoch,
    )
    _validate_history(
        result.get("final_history"),
        selected_epoch=selected_epoch,
        final_fit=True,
    )
    _validate_metrics_shape(result.get("metrics"))
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "checkpoint",
        "predictions",
        "predictions_sidecar",
    }:
        _fail("Oracle result artifact bindings are not exact")
    _validate_file_binding(artifacts["checkpoint"], root, CHECKPOINT_FILENAME)
    predictions_path = _validate_file_binding(
        artifacts["predictions"], root, PREDICTIONS_FILENAME
    )
    _validate_file_binding(
        artifacts["predictions_sidecar"], root, PREDICTIONS_SIDECAR_FILENAME
    )
    _validate_file_binding(sidecar.get("npz"), root, PREDICTIONS_FILENAME)
    try:
        with np.load(predictions_path, allow_pickle=False) as payload:
            if payload.files != ["probabilities", "session_offsets"]:
                _fail("saved probability NPZ keys are not exact")
            probabilities = np.array(payload["probabilities"], copy=True)
            offsets = np.array(payload["session_offsets"], copy=True)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise OraclePosthocBackendError(
            f"saved probability NPZ is invalid or requires pickle: {exc}"
        ) from exc
    if probabilities.dtype != np.float32 or probabilities.ndim != 3:
        _fail("saved probability array dtype/shape is invalid")
    if probabilities.shape[1:] != (4, 4):
        _fail("saved probability participant/bin dimensions are invalid")
    if offsets.dtype != np.int64 or offsets.shape != (len(expected_sessions) + 1,):
        _fail("saved session offsets dtype/shape is invalid")
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0.0).all() and (probabilities <= 1.0).all()
    ):
        _fail("saved probability array contains invalid values")
    expected_offsets = np.cumsum(
        np.asarray(
            [0, *[session.features.shape[0] for session in expected_sessions]],
            dtype=np.int64,
        )
    )
    if not np.array_equal(offsets, expected_offsets):
        _fail("saved probability offsets differ from source sessions")
    expected_array_contract = {
        "probabilities_shape": list(probabilities.shape),
        "probabilities_dtype": "float32",
        "session_offsets_shape": list(offsets.shape),
        "session_offsets_dtype": "int64",
    }
    if sidecar.get("array_contract") != expected_array_contract:
        _fail("saved probability array contract differs")
    records = sidecar.get("sessions")
    if not isinstance(records, list) or len(records) != len(expected_sessions):
        _fail("saved probability session records are invalid")
    split_predictions = []
    for index, (record, session) in enumerate(zip(records, expected_sessions, strict=True)):
        if not isinstance(record, dict):
            _fail("saved probability session record is not an object")
        probability = probabilities[offsets[index] : offsets[index + 1]]
        expected_record = {
            "session_id": session.session_id,
            "group_id": session.group_id,
            "n_frames": session.features.shape[0],
            "start_index": int(offsets[index]),
            "end_index": int(offsets[index + 1]),
            "session_tensor_sha256": _session_sha256(session),
            "prediction_input_sha256": prediction_input_sha256(session),
            "oracle_source_vad_sha256": oracle_source_vad_sha256(session),
            "evaluation_target_sha256": evaluation_target_sha256(session),
            "probabilities_sha256": _sha256_array(probability),
        }
        if record != expected_record:
            _fail(f"saved probability binding differs for {session.session_id}")
        split_predictions.append(probability.copy())
    recomputed_metrics = _json_safe(
        oracle_metrics_from_predictions(expected_sessions, split_predictions)
    )
    if _canonical_json(result["metrics"]) != _canonical_json(recomputed_metrics):
        _fail("Oracle result metrics differ from saved-probability recomputation")
    return LoadedOracleUnit(
        result=result,
        sidecar=sidecar,
        predictions=tuple(split_predictions),
    )


def load_oracle_checkpoint_artifact(
    checkpoint_path: str | Path,
    *,
    condition_id: str,
    fold_id: str,
    outer_test_group: str,
    source_snapshot_sha256: str,
    protocol_sha256: str,
    expected_initial_state_sha256: str,
    fixture_only: bool = False,
) -> dict[str, object]:
    """Load one isolated Oracle checkpoint and strictly restore its state on CPU."""

    path = Path(checkpoint_path).resolve()
    if not path.is_file() or path.is_symlink():
        _fail(f"Oracle checkpoint is missing or invalid: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise OraclePosthocBackendError(f"cannot load safe Oracle checkpoint: {exc}") from exc
    if not isinstance(value, dict):
        _fail("Oracle checkpoint payload is not an object")
    contract = _artifact_contract(fixture_only)
    expected_keys = {
        "artifact_kind",
        "format_version",
        "backend_id",
        "result_role",
        "fixture_only",
        "condition_id",
        "oracle_input_contract",
        "fold_id",
        "outer_test_group",
        "feature_schema",
        "source_snapshot_sha256",
        "protocol_sha256",
        "seed",
        "max_epochs",
        "aux_vad_weight",
        "payload",
        *_boundary(),
    }
    if set(value) != expected_keys:
        _fail("Oracle checkpoint fields are not exact")
    required = {
        "artifact_kind": contract["checkpoint"],
        "format_version": FORMAT_VERSION,
        "backend_id": BACKEND_ID,
        "result_role": contract["result_role"],
        "fixture_only": fixture_only,
        "condition_id": condition_id,
        "oracle_input_contract": oracle_condition_metadata(condition_id),
        "fold_id": fold_id,
        "outer_test_group": outer_test_group,
        "feature_schema": FEATURE_SCHEMA,
        "source_snapshot_sha256": source_snapshot_sha256,
        "protocol_sha256": protocol_sha256,
        "seed": 13,
        "max_epochs": 15,
        "aux_vad_weight": 0.0,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            _fail(f"Oracle checkpoint field {key!r} differs")
    if (
        type(value["seed"]) is not int
        or type(value["max_epochs"]) is not int
        or type(value["aux_vad_weight"]) is not float
    ):
        _fail("Oracle checkpoint frozen numeric field types differ")
    _validate_boundary(value)
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        _fail("Oracle checkpoint core payload is invalid")
    selected_final_epochs = payload.get("selected_final_epochs")
    if (
        isinstance(selected_final_epochs, bool)
        or not isinstance(selected_final_epochs, int)
        or not 1 <= selected_final_epochs <= 15
    ):
        _fail("checkpoint selected final epoch is invalid")
    _validate_checkpoint_core_payload(
        payload,
        outer_test_group=outer_test_group,
        selected_final_epochs=selected_final_epochs,
        expected_initial_state_sha256=expected_initial_state_sha256,
    )
    return value


class OraclePosthocTorchBackend:
    """Nested-LOGO CPU backend used by the isolated post-hoc matrix runner."""

    def __init__(self, spec: OracleBackendSpec | None = None) -> None:
        self.spec = spec or OracleBackendSpec()
        if self.spec != OracleBackendSpec():
            _fail("formal Oracle backend spec is frozen and cannot be overridden")

    def run_unit(self, request: OracleUnitRequest) -> OracleUnitReceipt:
        _validate_request(request)
        outer_train = _select_groups(request.sessions, request.outer_train_groups)
        outer_test = _select_groups(request.sessions, (request.outer_test_group,))
        _validate_sessions(outer_train, "outer train")
        _validate_sessions(outer_test, "outer test")
        inner_fits: list[dict[str, object]] = []
        selected_epochs: list[int] = []
        with _deterministic_cpu(request.seed):
            for inner_index, validation_group in enumerate(
                request.inner_validation_groups, start=1
            ):
                inner_train_groups = tuple(
                    group
                    for group in request.outer_train_groups
                    if group != validation_group
                )
                inner_train = _select_groups(outer_train, inner_train_groups)
                inner_validation = _select_groups(outer_train, (validation_group,))
                normalizer = fit_oracle_training_normalizer(
                    outer_train,
                    inner_train_groups,
                )
                fit = _fit(
                    self.spec,
                    inner_train,
                    inner_validation,
                    normalizer,
                    _fit_pos_weight(inner_train),
                    epochs=request.max_epochs,
                    seed=request.seed,
                )
                if fit.initial_state_sha256 != request.expected_initial_state_sha256:
                    _fail("inner fit did not start from the frozen initial state")
                predictions = _predict(
                    fit.model,
                    inner_validation,
                    normalizer,
                    self.spec,
                )
                inner_fits.append(
                    {
                        "inner_fold_id": f"inner_{inner_index:02d}",
                        "training_groups": list(inner_train_groups),
                        "validation_group": validation_group,
                        "training_sessions": [
                            session.session_id for session in inner_train
                        ],
                        "validation_sessions": [
                            session.session_id for session in inner_validation
                        ],
                        "selected_epoch": fit.selected_epoch,
                        "initial_state_sha256": fit.initial_state_sha256,
                        "validation_metrics": oracle_metrics_from_predictions(
                            inner_validation, predictions
                        ),
                        "history": fit.history,
                    }
                )
                selected_epochs.append(fit.selected_epoch)
            selected_final_epochs = int(statistics.median(selected_epochs))
            final_normalizer = fit_oracle_training_normalizer(
                request.sessions,
                request.outer_train_groups,
            )
            final_fit = _fit(
                self.spec,
                outer_train,
                (),
                final_normalizer,
                _fit_pos_weight(outer_train),
                epochs=selected_final_epochs,
                seed=request.seed,
            )
            if final_fit.initial_state_sha256 != request.expected_initial_state_sha256:
                _fail("final fit did not start from the frozen initial state")
            outer_predictions = _predict(
                final_fit.model,
                outer_test,
                final_normalizer,
                self.spec,
            )
        metrics = oracle_metrics_from_predictions(outer_test, outer_predictions)
        checkpoint_payload = {
            "backend_spec": self.spec.to_dict(),
            "model_config": oracle_model_config(self.spec),
            "trainer_spec": oracle_trainer_spec(self.spec),
            "label_config": oracle_label_config(),
            "model_state": final_fit.model.state_dict(),
            "model_state_sha256": model_state_sha256(final_fit.model.state_dict()),
            "normalizer": final_normalizer.to_dict(),
            "outer_train_groups": list(request.outer_train_groups),
            "outer_train_session_ids": [session.session_id for session in outer_train],
            "outer_test_session_ids": [session.session_id for session in outer_test],
            "selected_final_epochs": selected_final_epochs,
            "final_history": final_fit.history,
            "initial_state_sha256": final_fit.initial_state_sha256,
        }
        return write_oracle_unit_bundle(
            request,
            checkpoint_payload=checkpoint_payload,
            outer_sessions=outer_test,
            predictions=outer_predictions,
            metrics=metrics,
            inner_fits=inner_fits,
            selected_final_epochs=selected_final_epochs,
            final_history=final_fit.history,
        )


__all__ = [
    "BACKEND_ID",
    "BASE_PARTICIPANT_FEATURES",
    "CHECKPOINT_FILENAME",
    "CONDITION_IDS",
    "EXACT_EIGHT_GROUP_BY_SCENE",
    "EXACT_EIGHT_GROUPS",
    "EXACT_EIGHT_TICK_RANGES",
    "EXACT_EIGHT_TOTAL_ROWS",
    "FEATURE_SCHEMA",
    "FIXTURE_RESULT_ROLE",
    "FORMAT_VERSION",
    "GLOBAL_WIDTH",
    "LoadedOracleUnit",
    "OracleBackendSpec",
    "OracleFeatureNormalizer",
    "OraclePosthocBackendError",
    "OraclePosthocTorchBackend",
    "OracleUnitReceipt",
    "OracleUnitRequest",
    "ORACLE_PARTICIPANT_FEATURES",
    "PARTICIPANT_WIDTH",
    "PREDICTIONS_FILENAME",
    "PREDICTIONS_SIDECAR_FILENAME",
    "PREVIOUS_VAD_INDEX",
    "RESULT_FILENAME",
    "RESULT_ROLE",
    "TRANSFORMED_WIDTH",
    "UNIT_FILENAMES",
    "build_oracle_condition_sessions",
    "evaluation_target_sha256",
    "fit_oracle_training_normalizer",
    "initial_model_state_sha256",
    "load_oracle_checkpoint_artifact",
    "load_saved_oracle_unit",
    "model_state_sha256",
    "oracle_condition_metadata",
    "oracle_label_config",
    "oracle_metrics_from_predictions",
    "oracle_model_config",
    "oracle_source_vad_sha256",
    "oracle_trainer_spec",
    "prediction_input_sha256",
    "validate_oracle_condition_sessions",
    "write_oracle_unit_bundle",
]
