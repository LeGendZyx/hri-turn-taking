from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hri_turn_taking.legacy.data.schema import (
    DEFAULT_FEATURE_COLUMNS,
    TARGET_COLUMN,
    coerce_feature_types,
    validate_frame_dataframe,
)


@dataclass(frozen=True)
class WindowedDataset:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    timestamps_s: np.ndarray
    feature_names: list[str]
    source_frame_count: int


def load_frame_dataframe(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() == ".parquet":
        frame_df = pd.read_parquet(dataset_path)
    else:
        frame_df = pd.read_csv(dataset_path)
    return coerce_feature_types(frame_df)


def build_window_matrix(
    frame_df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    window_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Flatten recent frame history into examples suitable for baseline classifiers."""
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    features = feature_columns or DEFAULT_FEATURE_COLUMNS
    validate_frame_dataframe(frame_df, feature_columns=features, target_column=None)
    sorted_df = frame_df.sort_values(["session_id", "timestamp_s"]).reset_index(drop=True)

    windows: list[np.ndarray] = []
    groups: list[str] = []
    timestamps: list[float] = []

    for session_id, group in sorted_df.groupby("session_id", sort=False):
        values = group[features].to_numpy(dtype=float)
        session_timestamps = group["timestamp_s"].to_numpy(dtype=float)
        for row_idx in range(values.shape[0]):
            start = max(0, row_idx - window_size + 1)
            history = values[start : row_idx + 1]
            if history.shape[0] < window_size:
                pad = np.repeat(history[:1], window_size - history.shape[0], axis=0)
                history = np.vstack([pad, history])
            windows.append(history.reshape(-1))
            groups.append(str(session_id))
            timestamps.append(float(session_timestamps[row_idx]))

    feature_names = _window_feature_names(features, window_size)
    return np.vstack(windows), np.asarray(groups), np.asarray(timestamps), feature_names


def load_windowed_dataset(
    path: str | Path,
    *,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
    window_size: int = 1,
) -> WindowedDataset:
    frame_df = load_frame_dataframe(path)
    features = feature_columns or DEFAULT_FEATURE_COLUMNS
    validate_frame_dataframe(frame_df, feature_columns=features, target_column=target_column)
    x, groups, timestamps_s, feature_names = build_window_matrix(
        frame_df,
        feature_columns=features,
        window_size=window_size,
    )
    y = frame_df.sort_values(["session_id", "timestamp_s"])[target_column].to_numpy(dtype=int)
    return WindowedDataset(
        x=x,
        y=y,
        groups=groups,
        timestamps_s=timestamps_s,
        feature_names=feature_names,
        source_frame_count=len(frame_df),
    )


def _window_feature_names(feature_columns: list[str], window_size: int) -> list[str]:
    names: list[str] = []
    for lag in range(window_size - 1, -1, -1):
        suffix = "t" if lag == 0 else f"t-{lag}"
        names.extend([f"{column}_{suffix}" for column in feature_columns])
    return names
