"""Tensor datasets for multiparty VAP training.

Converts the wide frame-level table (``P{i}_<feature>`` columns at 25 fps)
into per-session tensors, derives future-bin labels from gold VAD, and yields
fixed-length training windows. Sessions are the grouping unit for splits so
that no window ever straddles train/validation boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hri_turn_taking.data.multiparty import (
    MAX_PARTICIPANTS,
    input_feature_columns,
    present_columns,
    vad_columns,
    validate_multiparty_dataframe,
)
from hri_turn_taking.vap.labels import (
    DEFAULT_LABEL_CONFIG,
    VAPLabelConfig,
    future_activity_labels,
)


@dataclass(frozen=True)
class SessionTensors:
    session_id: str
    features: np.ndarray  # (T, F) float32, robot-observable tier only
    labels: np.ndarray  # (T, N, K) float32, derived from gold VAD
    vad: np.ndarray  # (T, N) float32, gold tier (aux task + event eval)
    valid: np.ndarray  # (T,) bool, False where future horizon is truncated
    present: np.ndarray  # (N,) float32 participant slot mask


def sessions_from_dataframe(
    frame_df: pd.DataFrame,
    *,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    max_participants: int = MAX_PARTICIPANTS,
) -> list[SessionTensors]:
    validate_multiparty_dataframe(frame_df, max_participants)
    feature_cols = input_feature_columns(max_participants)
    vad_cols = vad_columns(max_participants)
    present_cols = present_columns(max_participants)

    sessions: list[SessionTensors] = []
    for session_id, group in frame_df.groupby("session_id", sort=True):
        group = group.sort_values("timestamp_s")
        features = group[feature_cols].to_numpy(dtype=np.float32)
        vad = group[vad_cols].to_numpy(dtype=np.float32)
        present = group[present_cols].to_numpy(dtype=np.float32).max(axis=0)
        labels, valid = future_activity_labels(vad, label_config)
        sessions.append(
            SessionTensors(
                session_id=str(session_id),
                features=features,
                labels=labels.astype(np.float32),
                vad=vad,
                valid=valid,
                present=present,
            )
        )
    return sessions


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, sessions: list[SessionTensors]) -> FeatureNormalizer:
        stacked = np.concatenate([session.features for session in sessions], axis=0)
        std = stacked.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean=stacked.mean(axis=0), std=std)

    def apply(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> FeatureNormalizer:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
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
) -> tuple[list[SessionTensors], list[SessionTensors]]:
    """Session-level split (no frame ever appears in both partitions)."""
    if len(sessions) < 2:
        raise ValueError("Need at least two sessions for a train/validation split")
    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(sessions))
    n_val = max(1, round(len(sessions) * validation_fraction))
    val_idx = set(order[:n_val].tolist())
    train = [s for i, s in enumerate(sessions) if i not in val_idx]
    val = [s for i, s in enumerate(sessions) if i in val_idx]
    return train, val
