"""Auditable robot-observable feature sets for controlled VAP ablations.

Every feature set preserves the canonical wide-table schema and model input
width.  Disabled input columns are zero-filled in a copy of the dataframe;
gold VAD, presence masks, timestamps, and all non-input columns are untouched.
This makes checkpoints comparable while recording exactly which raw columns
were active.

``audio_only`` deliberately removes every participant-specific input.  Since
the remaining global acoustic block is broadcast to every participant token,
all present tokens receive identical inputs and a slot-equivariant model must
produce identical participant predictions.  It is therefore an honest
single-channel timing baseline, not a next-speaker-identification model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hri_turn_taking.data.multiparty import (
    FEATURE_SCHEMA_V3_GAZE_VALIDITY,
    GLOBAL_FEATURE_COLUMNS,
    MAX_PARTICIPANTS,
    PARTICIPANT_FEATURE_NAMES,
    input_feature_columns,
    participant_column,
)


@dataclass(frozen=True)
class FeatureSet:
    """One fixed ablation definition in canonical raw-column terms."""

    participant_features: tuple[str, ...]
    global_features: tuple[str, ...]
    description: str
    participant_tokens_symmetric: bool = False

    def active_columns(self, max_participants: int = MAX_PARTICIPANTS) -> list[str]:
        columns = [
            participant_column(slot, feature)
            for slot in range(1, max_participants + 1)
            for feature in self.participant_features
        ]
        columns.extend(self.global_features)
        return columns


_ALL_PARTICIPANT = tuple(PARTICIPANT_FEATURE_NAMES)
_ALL_GLOBAL = tuple(GLOBAL_FEATURE_COLUMNS)


FEATURE_SETS: dict[str, FeatureSet] = {
    "multimodal": FeatureSet(
        participant_features=_ALL_PARTICIPANT,
        global_features=_ALL_GLOBAL,
        description="All canonical robot-observable participant and global features.",
    ),
    "audio_only": FeatureSet(
        participant_features=(),
        global_features=_ALL_GLOBAL,
        description="Global single-channel acoustic features only; no slot identity evidence.",
        participant_tokens_symmetric=True,
    ),
    "head_pose_only": FeatureSet(
        participant_features=("head_yaw_deg", "pose_valid", "pose_quality"),
        global_features=(),
        description="Per-participant head yaw and its observation validity/quality only.",
    ),
    "audio_plus_head_pose": FeatureSet(
        participant_features=("head_yaw_deg", "pose_valid", "pose_quality"),
        global_features=_ALL_GLOBAL,
        description="R0-Core: global causal acoustics plus per-participant head pose.",
    ),
    "visual_only_strict": FeatureSet(
        participant_features=tuple(
            feature for feature in _ALL_PARTICIPANT if feature != "asd_prob"
        ),
        global_features=(),
        description="Strict visual inputs only; excludes ASD because ASD uses audio evidence.",
    ),
    "visual_plus_asd": FeatureSet(
        participant_features=_ALL_PARTICIPANT,
        global_features=(),
        description="All participant visual features plus pretrained audio-visual ASD.",
    ),
    "audio_plus_visual_strict": FeatureSet(
        participant_features=tuple(
            feature for feature in _ALL_PARTICIPANT if feature != "asd_prob"
        ),
        global_features=_ALL_GLOBAL,
        description="Global causal acoustics plus strict visual inputs, excluding ASD.",
    ),
    "no_head_pose": FeatureSet(
        participant_features=tuple(
            feature
            for feature in _ALL_PARTICIPANT
            if feature not in {"head_yaw_deg", "pose_valid", "pose_quality"}
        ),
        global_features=_ALL_GLOBAL,
        description="Multimodal minus head pose and its observation validity/quality channels.",
    ),
    "no_gaze": FeatureSet(
        participant_features=tuple(
            feature for feature in _ALL_PARTICIPANT if feature not in {"gaze_robot", "gazed_at"}
        ),
        global_features=_ALL_GLOBAL,
        description="Multimodal minus gaze-to-robot and gaze-at-participant evidence.",
    ),
    "no_asd": FeatureSet(
        participant_features=tuple(
            feature for feature in _ALL_PARTICIPANT if feature != "asd_prob"
        ),
        global_features=_ALL_GLOBAL,
        description="Multimodal minus pretrained ASD probability.",
    ),
    "no_mar": FeatureSet(
        participant_features=tuple(feature for feature in _ALL_PARTICIPANT if feature != "mar"),
        global_features=_ALL_GLOBAL,
        description="Multimodal minus the interpretable MAR ablation feature.",
    ),
}

FEATURE_SET_NAMES: tuple[str, ...] = tuple(FEATURE_SETS)


# Deliberately separate from FEATURE_SETS: adding these names to the v2
# registry would alter historical CLI choices and checkpoint metadata. All
# three post-hoc conditions retain the same 36-column v3 input schema; only
# their active-value masks differ.
GAZE_POSTHOC_FEATURE_SETS: dict[str, FeatureSet] = {
    "C0": FeatureSet(
        participant_features=("head_yaw_deg", "pose_valid", "pose_quality"),
        global_features=_ALL_GLOBAL,
        description=(
            "Schema-v3 A+H anchor; Gaze value and Gaze validity are both disabled."
        ),
    ),
    "CM": FeatureSet(
        participant_features=(
            "gaze_robot_valid",
            "head_yaw_deg",
            "pose_valid",
            "pose_quality",
        ),
        global_features=_ALL_GLOBAL,
        description=(
            "Schema-v3 A+H plus the real Gaze-validity mask; Gaze value disabled."
        ),
    ),
    "CG": FeatureSet(
        participant_features=(
            "gaze_robot",
            "gaze_robot_valid",
            "head_yaw_deg",
            "pose_valid",
            "pose_quality",
        ),
        global_features=_ALL_GLOBAL,
        description=(
            "Schema-v3 A+H plus binary Gaze-to-robot content and its validity mask."
        ),
    ),
}
GAZE_POSTHOC_CONDITION_IDS: tuple[str, ...] = tuple(GAZE_POSTHOC_FEATURE_SETS)


def get_feature_set(name: str) -> FeatureSet:
    """Return a fixed feature-set definition with a useful validation error."""
    try:
        return FEATURE_SETS[name]
    except KeyError as exc:
        choices = ", ".join(FEATURE_SET_NAMES)
        raise ValueError(f"Unknown feature_set {name!r}; choose one of: {choices}") from exc


def active_feature_columns(name: str) -> list[str]:
    """Canonical raw input columns enabled by ``name`` (stable schema order)."""
    definition = get_feature_set(name)
    active = set(definition.active_columns())
    return [column for column in input_feature_columns() if column in active]


def feature_set_metadata(name: str) -> dict[str, object]:
    """JSON-serialisable audit record for checkpoints and experiment tables."""
    definition = get_feature_set(name)
    return {
        "name": name,
        "active_columns": active_feature_columns(name),
        "participant_tokens_symmetric": definition.participant_tokens_symmetric,
        "description": definition.description,
    }


def apply_feature_set(frame_df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Copy ``frame_df`` and zero every canonical input disabled by ``name``.

    The function never mutates its caller's dataframe and never touches gold
    ``P{i}_vad`` columns.  Keeping the full schema means all ablations use the
    same model dimensions, normaliser layout, and checkpoint loader.
    """
    canonical = input_feature_columns()
    missing = [column for column in canonical if column not in frame_df.columns]
    if missing:
        raise ValueError(f"Cannot apply feature_set {name!r}; missing input columns: {missing}")

    active = set(active_feature_columns(name))
    disabled = [column for column in canonical if column not in active]
    masked = frame_df.copy(deep=True)
    masked.loc[:, disabled] = 0.0
    return masked


def get_gaze_posthoc_feature_set(condition_id: str) -> FeatureSet:
    """Return one C0/CM/CG definition without exposing it to the v2 registry."""

    try:
        return GAZE_POSTHOC_FEATURE_SETS[condition_id]
    except KeyError as exc:
        choices = ", ".join(GAZE_POSTHOC_CONDITION_IDS)
        raise ValueError(
            f"Unknown Gaze post-hoc condition {condition_id!r}; choose one of: {choices}"
        ) from exc


def active_gaze_posthoc_feature_columns(condition_id: str) -> list[str]:
    """Stable v3 raw-column order enabled by one post-hoc condition."""

    definition = get_gaze_posthoc_feature_set(condition_id)
    active = set(definition.active_columns())
    canonical = input_feature_columns(feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY)
    return [column for column in canonical if column in active]


def gaze_posthoc_feature_set_metadata(condition_id: str) -> dict[str, object]:
    """Hashable, explicit non-production metadata for a C0/CM/CG artifact."""

    definition = get_gaze_posthoc_feature_set(condition_id)
    return {
        "condition_id": condition_id,
        "feature_schema": FEATURE_SCHEMA_V3_GAZE_VALIDITY,
        "active_columns": active_gaze_posthoc_feature_columns(condition_id),
        "description": definition.description,
        "posthoc": True,
        "production_eligible": False,
        "counted_eligible": False,
    }


def apply_gaze_posthoc_feature_set(
    frame_df: pd.DataFrame, condition_id: str
) -> pd.DataFrame:
    """Zero disabled v3 inputs without mutating labels, keys, or the caller."""

    canonical = input_feature_columns(feature_schema=FEATURE_SCHEMA_V3_GAZE_VALIDITY)
    missing = [column for column in canonical if column not in frame_df.columns]
    if missing:
        raise ValueError(
            f"Cannot apply Gaze post-hoc condition {condition_id!r}; "
            f"missing input columns: {missing}"
        )
    active = set(active_gaze_posthoc_feature_columns(condition_id))
    masked = frame_df.copy(deep=True)
    masked.loc[:, [column for column in canonical if column not in active]] = 0.0
    return masked
