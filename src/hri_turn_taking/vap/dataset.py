"""Tensor datasets for multiparty VAP training.

Converts the wide frame-level table (``P{i}_<feature>`` columns at 25 fps)
into per-session tensors, derives future-bin labels from gold VAD, and yields
fixed-length training windows. Sessions are the grouping unit for splits so
that no window ever straddles train/validation boundaries.

Two dataset-level guarantees back the model's participant structure:

- ``transform_input_features`` encodes the circular DoA angle as sin/cos —
  a wrapped angle (ReSpeaker DoA wraps at 0/360 deg) must not be z-scored
  across its wrap point.
- ``FeatureNormalizer`` keeps one statistic per participant *feature*, pooled
  over present slots only, so every slot is normalised identically. Per-slot
  statistics would break slot permutation equivariance and be degenerate for
  the always-padded slot 4 of 3-party corpora.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hri_turn_taking.data.multiparty import (
    DEFAULT_FEATURE_SCHEMA,
    FEATURE_SCHEMA_V2,
    FEATURE_SCHEMA_V3_GAZE_VALIDITY,
    GLOBAL_FEATURE_COLUMNS,
    MAX_PARTICIPANTS,
    PARTICIPANT_FEATURE_NAMES,
    PARTICIPANT_FEATURE_NAMES_V3_GAZE_VALIDITY,
    input_feature_columns,
    participant_feature_names,
    present_columns,
    vad_columns,
    validate_multiparty_dataframe,
)
from hri_turn_taking.vap.labels import (
    DEFAULT_LABEL_CONFIG,
    VAPLabelConfig,
    future_activity_labels,
)

TRANSFORMED_PARTICIPANT_FEATURES: list[str] = list(PARTICIPANT_FEATURE_NAMES)
TRANSFORMED_PARTICIPANT_FEATURES_V3_GAZE_VALIDITY: list[str] = list(
    PARTICIPANT_FEATURE_NAMES_V3_GAZE_VALIDITY
)
TRANSFORMED_GLOBAL_FEATURES: list[str] = [
    "doa_sin",
    "doa_cos",
    "array_energy",
    "f0_proxy",
    "voiced",
]

_DOA_INDEX = GLOBAL_FEATURE_COLUMNS.index("doa_angle_deg")


def transformed_participant_features(
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
) -> list[str]:
    """Return the transformed per-slot order for one explicit schema."""

    if feature_schema == FEATURE_SCHEMA_V2:
        return list(TRANSFORMED_PARTICIPANT_FEATURES)
    if feature_schema == FEATURE_SCHEMA_V3_GAZE_VALIDITY:
        return list(TRANSFORMED_PARTICIPANT_FEATURES_V3_GAZE_VALIDITY)
    # Reuse the central error text and reject unknown schemas before tensor work.
    participant_feature_names(feature_schema)
    raise AssertionError("unreachable")


def transform_input_features(
    raw: np.ndarray,
    max_participants: int = MAX_PARTICIPANTS,
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
) -> np.ndarray:
    """Map raw wide-table input columns to the model feature layout.

    Identity for all columns except ``doa_angle_deg``, which becomes a
    (sin, cos) pair. Output layout: ``[P1..Pn participant features |
    TRANSFORMED_GLOBAL_FEATURES]``.
    """
    raw = np.asarray(raw, dtype=np.float32)
    n_participant = max_participants * len(participant_feature_names(feature_schema))
    if raw.ndim != 2 or raw.shape[1] != n_participant + len(GLOBAL_FEATURE_COLUMNS):
        raise ValueError("raw features do not match the input_feature_columns layout")
    doa_rad = np.deg2rad(raw[:, n_participant + _DOA_INDEX])
    global_block = np.stack(
        [
            np.sin(doa_rad),
            np.cos(doa_rad),
            raw[:, n_participant + GLOBAL_FEATURE_COLUMNS.index("array_energy")],
            raw[:, n_participant + GLOBAL_FEATURE_COLUMNS.index("f0_proxy")],
            raw[:, n_participant + GLOBAL_FEATURE_COLUMNS.index("voiced")],
        ],
        axis=1,
    )
    return np.concatenate([raw[:, :n_participant], global_block], axis=1).astype(np.float32)


@dataclass(frozen=True)
class SessionTensors:
    session_id: str
    features: np.ndarray  # (T, F) float32, transformed robot-observable tier only
    labels: np.ndarray  # (T, N, K) float32, derived from gold VAD
    vad: np.ndarray  # (T, N) float32, gold tier (aux task + event eval)
    valid: np.ndarray  # (T,) bool, False where future horizon is truncated
    present: np.ndarray  # (N,) float32 participant slot mask
    # Real-data group identity is injected only after provenance validation.
    # Synthetic callers may leave it unset and request delimiter-based grouping.
    group_id: str | None = None
    # Absent from historical constructors/artifacts, therefore defaulted to v2.
    feature_schema: str = FEATURE_SCHEMA_V2


def sessions_from_dataframe(
    frame_df: pd.DataFrame,
    *,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    max_participants: int = MAX_PARTICIPANTS,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
) -> list[SessionTensors]:
    validate_multiparty_dataframe(
        frame_df, max_participants, feature_schema=feature_schema
    )
    feature_cols = input_feature_columns(
        max_participants, feature_schema=feature_schema
    )
    vad_cols = vad_columns(max_participants)
    present_cols = present_columns(max_participants)

    sessions: list[SessionTensors] = []
    for session_id, group in frame_df.groupby("session_id", sort=True):
        group = group.sort_values("timestamp_s")
        group_id: str | None = None
        if "group_id" in group.columns:
            declared = group["group_id"].dropna().astype(str).unique().tolist()
            if len(declared) != 1 or not declared[0].strip():
                raise ValueError(
                    f"session {session_id!r} must have exactly one non-empty provenance group_id"
                )
            group_id = declared[0]
        raw = group[feature_cols].to_numpy(dtype=np.float32)
        features = transform_input_features(
            raw, max_participants, feature_schema=feature_schema
        )
        # Pandas may expose read-only zero-copy views; downstream
        # ``torch.from_numpy`` requires writable buffers even for read-only
        # training inputs, otherwise PyTorch warns about undefined writes.
        vad = group[vad_cols].to_numpy(dtype=np.float32).copy()
        # validate_multiparty_dataframe guarantees present is static per session
        # and binary, so the first row IS the slot mask (no .max() that would
        # silently crush a dynamic column — Codex R1.2-1)
        present = group[present_cols].to_numpy(dtype=np.float32)[0].copy()
        labels, valid = future_activity_labels(vad, label_config)
        sessions.append(
            SessionTensors(
                session_id=str(session_id),
                features=features,
                labels=labels.astype(np.float32),
                vad=vad,
                valid=valid,
                present=present,
                group_id=group_id,
                feature_schema=feature_schema,
            )
        )
    return sessions


@dataclass
class FeatureNormalizer:
    """Slot-shared z-score statistics (see module docstring)."""

    participant_mean: np.ndarray  # (n_participant_features,)
    participant_std: np.ndarray
    global_mean: np.ndarray  # (n_global_features,)
    global_std: np.ndarray
    feature_schema: str = FEATURE_SCHEMA_V2

    @classmethod
    def fit(cls, sessions: list[SessionTensors]) -> FeatureNormalizer:
        if not sessions:
            raise ValueError("FeatureNormalizer.fit requires at least one training session")
        schemas = {session.feature_schema for session in sessions}
        if len(schemas) != 1:
            raise ValueError("FeatureNormalizer cannot mix feature schemas")
        feature_schema = schemas.pop()
        part_names = transformed_participant_features(feature_schema)
        n_part = len(part_names)
        n_glob = len(TRANSFORMED_GLOBAL_FEATURES)
        participant_rows, global_rows = [], []
        valid_gaze_values: list[np.ndarray] = []
        for session in sessions:
            n_frames = session.features.shape[0]
            expected_width = len(session.present) * n_part + n_glob
            if session.features.ndim != 2 or session.features.shape[1] != expected_width:
                raise ValueError(
                    f"session {session.session_id!r} feature width does not match "
                    f"{feature_schema!r}"
                )
            per_slot = session.features[:, :-n_glob].reshape(n_frames, -1, n_part)
            for slot in np.flatnonzero(session.present > 0):
                participant_rows.append(per_slot[:, slot, :])
                if feature_schema == FEATURE_SCHEMA_V3_GAZE_VALIDITY:
                    valid_index = part_names.index("gaze_robot_valid")
                    value_index = part_names.index("gaze_robot")
                    validity = per_slot[:, slot, valid_index]
                    values = per_slot[:, slot, value_index]
                    if not np.isin(validity, (0.0, 1.0)).all():
                        raise ValueError("gaze_robot_valid must stay binary before normalisation")
                    if (values[validity == 0.0] != 0.0).any():
                        raise ValueError(
                            "invalid gaze_robot values must be exactly zero before normalisation"
                        )
                    valid_gaze_values.append(values[validity == 1.0])
            global_rows.append(session.features[:, -n_glob:])

        def stats(stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            std[std < 1e-6] = 1.0
            return mean, std

        if not participant_rows or not global_rows:
            raise ValueError("FeatureNormalizer found no present training-slot observations")
        participant_mean, participant_std = stats(np.concatenate(participant_rows, axis=0))
        global_mean, global_std = stats(np.concatenate(global_rows, axis=0))
        if feature_schema == FEATURE_SCHEMA_V3_GAZE_VALIDITY:
            gaze_values = np.concatenate(valid_gaze_values)
            value_index = part_names.index("gaze_robot")
            valid_index = part_names.index("gaze_robot_valid")
            if gaze_values.size:
                participant_mean[value_index] = gaze_values.mean()
                gaze_std = float(gaze_values.std())
                participant_std[value_index] = 1.0 if gaze_std < 1e-6 else gaze_std
            else:
                # C0 disables both Gaze channels by design. Its schema still
                # needs deterministic identity statistics, not a fake sample.
                participant_mean[value_index] = 0.0
                participant_std[value_index] = 1.0
            # The observation-validity bit is model input, not a continuous
            # measurement: retaining exact 0/1 values is part of the contract.
            participant_mean[valid_index] = 0.0
            participant_std[valid_index] = 1.0
        return cls(
            participant_mean,
            participant_std,
            global_mean,
            global_std,
            feature_schema=feature_schema,
        )

    def apply(self, features: np.ndarray) -> np.ndarray:
        part_names = transformed_participant_features(self.feature_schema)
        n_part = len(part_names)
        n_glob = len(TRANSFORMED_GLOBAL_FEATURES)
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or (features.shape[1] - n_glob) % n_part != 0:
            raise ValueError(
                f"features do not match normalizer schema {self.feature_schema!r}"
            )
        n_frames = features.shape[0]
        per_slot = features[:, :-n_glob].reshape(n_frames, -1, n_part)
        normalized = (per_slot - self.participant_mean) / self.participant_std
        if self.feature_schema == FEATURE_SCHEMA_V3_GAZE_VALIDITY:
            value_index = part_names.index("gaze_robot")
            valid_index = part_names.index("gaze_robot_valid")
            validity = per_slot[:, :, valid_index]
            values = per_slot[:, :, value_index]
            if not np.isin(validity, (0.0, 1.0)).all():
                raise ValueError("gaze_robot_valid must stay binary before normalisation")
            if (values[validity == 0.0] != 0.0).any():
                raise ValueError(
                    "invalid gaze_robot values must be exactly zero before normalisation"
                )
            normalized[:, :, value_index][validity == 0.0] = 0.0
            normalized[:, :, valid_index] = validity
        global_block = (features[:, -n_glob:] - self.global_mean) / self.global_std
        return np.concatenate(
            [normalized.reshape(n_frames, -1), global_block], axis=1
        ).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "participant_mean": self.participant_mean.tolist(),
            "participant_std": self.participant_std.tolist(),
            "global_mean": self.global_mean.tolist(),
            "global_std": self.global_std.tolist(),
        }
        # Preserve the exact historical v4 normalizer payload. A v3 payload is
        # self-identifying and belongs to the separate post-hoc artifact route.
        if self.feature_schema != FEATURE_SCHEMA_V2:
            payload["feature_schema"] = self.feature_schema
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FeatureNormalizer:
        feature_schema = payload.get("feature_schema", FEATURE_SCHEMA_V2)
        if not isinstance(feature_schema, str):
            raise ValueError("normalizer feature_schema must be a string")
        transformed_participant_features(feature_schema)
        return cls(
            participant_mean=np.asarray(payload["participant_mean"], dtype=np.float32),
            participant_std=np.asarray(payload["participant_std"], dtype=np.float32),
            global_mean=np.asarray(payload["global_mean"], dtype=np.float32),
            global_std=np.asarray(payload["global_std"], dtype=np.float32),
            feature_schema=feature_schema,
        )


class WindowDataset(Dataset):
    """Fixed-length windows over normalised session tensors."""

    def __init__(
        self,
        sessions: list[SessionTensors],
        normalizer: FeatureNormalizer,
        *,
        window_frames: int = 250,
        stride_frames: int = 125,
    ) -> None:
        if window_frames < 1 or stride_frames < 1:
            raise ValueError("window_frames and stride_frames must be >= 1")
        mismatched = [
            session.session_id
            for session in sessions
            if session.feature_schema != normalizer.feature_schema
        ]
        if mismatched:
            raise ValueError(
                "WindowDataset session/normalizer feature schemas differ for "
                f"{mismatched[:5]}"
            )
        self.window_frames = window_frames
        self._items: list[tuple[int, int]] = []
        self._sessions = sessions
        self._normalized = [normalizer.apply(session.features) for session in sessions]
        for session_idx, session in enumerate(sessions):
            n_frames = session.features.shape[0]
            if n_frames < window_frames:
                continue
            for start in range(0, n_frames - window_frames + 1, stride_frames):
                self._items.append((session_idx, start))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        session_idx, start = self._items[index]
        session = self._sessions[session_idx]
        end = start + self.window_frames
        return {
            "features": torch.from_numpy(self._normalized[session_idx][start:end]),
            "labels": torch.from_numpy(session.labels[start:end]),
            "vad": torch.from_numpy(session.vad[start:end]),
            "valid": torch.from_numpy(session.valid[start:end].astype(np.float32)),
            "present": torch.from_numpy(session.present),
        }


def split_sessions(
    sessions: list[SessionTensors],
    *,
    validation_fraction: float = 0.25,
    random_seed: int = 42,
    group_delimiter: str | None = None,
) -> tuple[list[SessionTensors], list[SessionTensors]]:
    """Session-level split (no frame ever appears in both partitions).

    Real tables carry a provenance-derived ``SessionTensors.group_id`` and that
    value always takes precedence over names. ``group_delimiter`` remains an
    explicit convenience for synthetic experiments. Punctuation in a session
    identifier is never used to infer whether data are real.
    """
    if len(sessions) < 2:
        raise ValueError("Need at least two sessions for a train/validation split")
    explicit_group_ids = [getattr(session, "group_id", None) for session in sessions]
    has_explicit_groups = any(group_id is not None for group_id in explicit_group_ids)
    if has_explicit_groups and any(group_id is None for group_id in explicit_group_ids):
        raise ValueError(
            "group provenance is incomplete: every session in a grouped dataset must "
            "declare group_id"
        )
    if not has_explicit_groups and group_delimiter is not None:
        without = [s.session_id for s in sessions if group_delimiter not in s.session_id]
        if without:
            raise ValueError(
                f"group_delimiter {group_delimiter!r} does not occur in synthetic "
                f"session ids {without[:5]}"
            )
    rng = np.random.default_rng(random_seed)
    if not has_explicit_groups and group_delimiter is None:
        order = rng.permutation(len(sessions))
        n_val = max(1, round(len(sessions) * validation_fraction))
        val_idx = set(order[:n_val].tolist())
    else:
        if has_explicit_groups:
            groups = [str(group_id) for group_id in explicit_group_ids]
        else:
            assert group_delimiter is not None
            groups = [s.session_id.split(group_delimiter)[0] for s in sessions]
        unique = sorted(set(groups))
        if len(unique) < 2:
            raise ValueError(
                "Need at least two groups for a group-level split; "
                "check group_delimiter against the session ids"
            )
        target = max(1, round(len(sessions) * validation_fraction))
        val_groups: set[str] = set()
        n_val_sessions = 0
        for group in rng.permutation(unique):
            if len(val_groups) == len(unique) - 1:
                break  # always keep at least one group in train
            val_groups.add(str(group))
            n_val_sessions += groups.count(str(group))
            if n_val_sessions >= target:
                break
        val_idx = {i for i, g in enumerate(groups) if g in val_groups}
    train = [s for i, s in enumerate(sessions) if i not in val_idx]
    val = [s for i, s in enumerate(sessions) if i in val_idx]
    return train, val
