from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET_COLUMN = "trp_label"

METADATA_COLUMNS = [
    "session_id",
    "timestamp_s",
]

DEFAULT_FEATURE_COLUMNS = [
    "audio_vad",
    "active_speaker_count",
    "overlap_active",
    "dominant_doa_angle_deg",
    "mouth_aspect_ratio_max",
    "visual_vad_max",
    "gaze_to_robot_score",
    "head_yaw_abs_min_deg",
    "speaking_towards_robot",
    "silence_duration_s",
    "current_turn_duration_s",
    "recent_speaker_change",
]

BOOLEAN_FEATURE_COLUMNS = [
    "audio_vad",
    "overlap_active",
    "visual_vad_max",
    "speaking_towards_robot",
    "recent_speaker_change",
]


@dataclass(frozen=True)
class FeatureSchema:
    feature_columns: list[str]
    target_column: str = TARGET_COLUMN

    @classmethod
    def default(cls) -> FeatureSchema:
        return cls(feature_columns=DEFAULT_FEATURE_COLUMNS.copy())


def validate_frame_dataframe(
    frame_df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    target_column: str | None = TARGET_COLUMN,
) -> None:
    """Validate the frame-level table used for TRP model training."""
    features = feature_columns or DEFAULT_FEATURE_COLUMNS
    required = [*METADATA_COLUMNS, *features]
    if target_column is not None:
        required.append(target_column)

    missing = [column for column in required if column not in frame_df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame_df.empty:
        raise ValueError("Dataset must contain at least one row")


def coerce_feature_types(frame_df: pd.DataFrame) -> pd.DataFrame:
    """Convert known boolean feature columns to 0/1 numeric values."""
    df = frame_df.copy()
    for column in BOOLEAN_FEATURE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype(int)
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df
