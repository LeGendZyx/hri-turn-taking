"""Multiparty frame-level schema and naive synthetic data generator.

Wide-format table at a fixed frame rate (default 25 fps), one row per frame,
columns named ``P{i}_<feature>`` per the project convention.

Feature tiers (CLAUDE.md design decision #2):
- Gold tier (labels / offline eval only): ``P{i}_vad`` from lavalier mics.
- Robot-observable tier (model input): per-face ASD probability, MAR,
  gaze-to-robot score, head yaw, observation validity/quality, plus global
  acoustic features (combined-mix-derived on real data; the DoA column only
  carries real values in the synthetic / mic-array-ablation pipeline).

The synthetic generator simulates 3-4 party conversations with turn shifts,
holds, overlaps and backchannels. Robot-observable features are *noisy,
probabilistic* correlates of the underlying speech state, including genuinely
anticipatory cues found in real conversation (turn-final prosodic lowering,
pre-offset gaze shifts). Labels are never copied into features, so models must
learn real temporal structure rather than exploit leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MAX_PARTICIPANTS = 4

# The deadline A/H/A+H matrix and every v4 development checkpoint use v2.
# Keep it as the default forever: the nullable Gaze experiment is a parallel,
# explicitly selected schema rather than a migration of historical artifacts.
FEATURE_SCHEMA_V2 = "multiparty_v2"
FEATURE_SCHEMA_V3_GAZE_VALIDITY = "multiparty_v3_gaze_validity"
DEFAULT_FEATURE_SCHEMA = FEATURE_SCHEMA_V2

GLOBAL_FEATURE_COLUMNS = ["doa_angle_deg", "array_energy", "f0_proxy", "voiced"]
# pose_valid: per-frame 0/1 flag — 1 when the head-pose observation for this
# slot is real, 0 when the tracker lost the face (features are zero-filled
# there). Robot-observable at runtime (Furhat knows when it cannot see a face);
# lets the model distinguish "missing observation" from a genuine 0° yaw.
# P{i}_present stays STATIC per session ("this participant exists") — tracking
# dropouts must not fake participants leaving the conversation (Codex R1.1-1).
# pose_quality: per-frame continuous [0, 1] observation quality (Codex R1.2-5
# ruling): for R = valid-target-member fraction backing the centroid bearing,
# for non-R = 1 iff its single target (R) is validly observed. Kept separate
# from the BINARY pose_valid flag — continuous quality must not be overloaded
# into the validity bit.
PARTICIPANT_FEATURE_NAMES = ["asd_prob", "mar", "gaze_robot", "head_yaw_deg", "gazed_at",
                             "pose_valid", "pose_quality"]
PARTICIPANT_FEATURE_NAMES_V3_GAZE_VALIDITY = [
    "asd_prob",
    "mar",
    "gaze_robot",
    "gaze_robot_valid",
    "head_yaw_deg",
    "gazed_at",
    "pose_valid",
    "pose_quality",
]
METADATA_COLUMNS = ["session_id", "timestamp_s"]


def participant_feature_names(feature_schema: str = DEFAULT_FEATURE_SCHEMA) -> list[str]:
    """Return an isolated schema's participant feature order.

    A copy is returned so callers cannot mutate the module-level contract.
    """

    if feature_schema == FEATURE_SCHEMA_V2:
        return list(PARTICIPANT_FEATURE_NAMES)
    if feature_schema == FEATURE_SCHEMA_V3_GAZE_VALIDITY:
        return list(PARTICIPANT_FEATURE_NAMES_V3_GAZE_VALIDITY)
    raise ValueError(
        f"Unknown feature_schema {feature_schema!r}; choose {FEATURE_SCHEMA_V2!r} "
        f"or {FEATURE_SCHEMA_V3_GAZE_VALIDITY!r}"
    )


def participant_column(index: int, feature: str) -> str:
    """1-based participant feature column name, e.g. ``P2_asd_prob``."""
    return f"P{index}_{feature}"


def vad_columns(max_participants: int = MAX_PARTICIPANTS) -> list[str]:
    return [participant_column(i + 1, "vad") for i in range(max_participants)]


def present_columns(max_participants: int = MAX_PARTICIPANTS) -> list[str]:
    return [participant_column(i + 1, "present") for i in range(max_participants)]


def input_feature_columns(
    max_participants: int = MAX_PARTICIPANTS,
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
) -> list[str]:
    """Robot-observable model input columns (never includes gold VAD)."""
    feature_names = participant_feature_names(feature_schema)
    columns = []
    for i in range(max_participants):
        columns.extend(participant_column(i + 1, name) for name in feature_names)
    columns.extend(GLOBAL_FEATURE_COLUMNS)
    return columns


def validate_multiparty_dataframe(
    frame_df: pd.DataFrame,
    max_participants: int = MAX_PARTICIPANTS,
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
) -> None:
    if feature_schema == FEATURE_SCHEMA_V2:
        v3_only = [
            participant_column(index, "gaze_robot_valid")
            for index in range(1, max_participants + 1)
            if participant_column(index, "gaze_robot_valid") in frame_df.columns
        ]
        if v3_only:
            raise ValueError(
                "Multiparty schema v2 must not contain schema-v3-only columns: "
                f"{v3_only}; select feature_schema={FEATURE_SCHEMA_V3_GAZE_VALIDITY!r} "
                "explicitly so Gaze missingness cannot be ignored"
            )
    required = [
        *METADATA_COLUMNS,
        *vad_columns(max_participants),
        *present_columns(max_participants),
        *input_feature_columns(max_participants, feature_schema=feature_schema),
    ]
    missing = [column for column in required if column not in frame_df.columns]
    if missing:
        raise ValueError(f"Multiparty dataset is missing required columns: {missing}")
    if frame_df.empty:
        raise ValueError("Multiparty dataset must contain at least one row")

    session_ids = frame_df["session_id"]
    if session_ids.isna().any() or session_ids.map(
        lambda value: not isinstance(value, str) or not value.strip()
    ).any():
        raise ValueError("Multiparty dataset session_id values must be non-empty strings")

    # value-level guarantees (Codex R1.2-1): downstream takes presence per
    # session as a single static row, and attention/loss assume binary masks —
    # a dynamic or fractional presence column must fail here, not be silently
    # crushed later
    # every model input, gold VAD and timestamp must be finite — NaN/Inf must
    # fail the final gate, not surface as non-finite tensors (Codex R1.3-4)
    numeric_required = [
        "timestamp_s",
        *vad_columns(max_participants),
        *input_feature_columns(max_participants, feature_schema=feature_schema),
    ]
    non_finite = [
        column
        for column in numeric_required
        if not np.isfinite(frame_df[column].to_numpy(dtype=float)).all()
    ]
    if non_finite:
        raise ValueError(f"Multiparty dataset has NaN/Inf in columns: {non_finite}")
    voiced = frame_df["voiced"].to_numpy(dtype=float)
    if not np.isin(voiced, (0.0, 1.0)).all():
        raise ValueError("voiced must be binary 0/1 (global F0-validity flag)")
    f0_proxy = frame_df["f0_proxy"].to_numpy(dtype=float)
    if (f0_proxy[voiced == 0.0] != 0.0).any():
        raise ValueError(
            "f0_proxy must be exactly zero where voiced=0; unvoiced frames cannot "
            "carry a stale or ambiguous pitch value"
        )
    for i in range(max_participants):
        present = frame_df[participant_column(i + 1, "present")].to_numpy(dtype=float)
        if not np.isin(present, (0.0, 1.0)).all():
            raise ValueError(f"P{i + 1}_present must be binary 0/1 (slot-exists mask)")
        vad = frame_df[participant_column(i + 1, "vad")].to_numpy(dtype=float)
        if not np.isin(vad, (0.0, 1.0)).all():
            raise ValueError(f"P{i + 1}_vad must be binary 0/1 (gold voice activity)")
        if (vad[present == 0.0] != 0.0).any():
            raise ValueError(
                f"P{i + 1}_vad must be zero on frames where present=0 — "
                "absent slots cannot carry gold voice activity"
            )
        pose_valid = frame_df[participant_column(i + 1, "pose_valid")].to_numpy(dtype=float)
        if not np.isin(pose_valid, (0.0, 1.0)).all():
            raise ValueError(f"P{i + 1}_pose_valid must be binary 0/1 (per-frame validity)")
        pose_quality = frame_df[participant_column(i + 1, "pose_quality")].to_numpy(dtype=float)
        if not (np.isfinite(pose_quality).all() and
                (pose_quality >= 0.0).all() and (pose_quality <= 1.0).all()):
            raise ValueError(f"P{i + 1}_pose_quality must be finite within [0, 1]")
        if (pose_valid > present).any() or (pose_quality > present).any():
            raise ValueError(
                f"P{i + 1} has pose_valid/pose_quality on frames where present=0 — "
                "absent slots cannot carry valid observations"
            )
        yaw = frame_df[participant_column(i + 1, "head_yaw_deg")].to_numpy(dtype=float)
        if (yaw[pose_valid == 0.0] != 0.0).any():
            raise ValueError(
                f"P{i + 1}_head_yaw_deg is non-zero on frames with pose_valid=0 — "
                "invalid observations must be zero-filled, not carry stale values"
            )
        if feature_schema == FEATURE_SCHEMA_V3_GAZE_VALIDITY:
            gaze_valid = frame_df[
                participant_column(i + 1, "gaze_robot_valid")
            ].to_numpy(dtype=float)
            gaze_value = frame_df[participant_column(i + 1, "gaze_robot")].to_numpy(
                dtype=float
            )
            if not np.isin(gaze_valid, (0.0, 1.0)).all():
                raise ValueError(
                    f"P{i + 1}_gaze_robot_valid must be binary 0/1 "
                    "(nullable Gaze observation-validity bit)"
                )
            if not np.isin(gaze_value, (0.0, 1.0)).all():
                raise ValueError(
                    f"P{i + 1}_gaze_robot must be binary 0/1 in the Gaze-validity schema"
                )
            if (gaze_valid > present).any():
                raise ValueError(
                    f"P{i + 1}_gaze_robot_valid is 1 where present=0 — absent slots "
                    "cannot carry valid Gaze observations"
                )
            if (gaze_value[gaze_valid == 0.0] != 0.0).any():
                raise ValueError(
                    f"P{i + 1}_gaze_robot must be exactly zero where "
                    "gaze_robot_valid=0; the validity bit, not the filled value, "
                    "distinguishes missingness"
                )
            if i == 0 and (gaze_valid != 0.0).any():
                raise ValueError(
                    "P1/R gaze_robot_valid must remain 0: the delivered Gaze export "
                    "does not observe the robot-player's gaze"
                )
    for session_id, group in frame_df.groupby("session_id", sort=False):
        timestamps = group["timestamp_s"].to_numpy(dtype=float)
        differences = np.diff(timestamps)
        if (differences <= 0.0).any():
            raise ValueError(
                f"timestamp_s must be strictly increasing and unique within session "
                f"{session_id!r}"
            )
        if differences.size and not np.allclose(
            differences, 1.0 / 25.0, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                f"timestamp_s must use the canonical 25 Hz grid (0.04 s steps) within "
                f"session {session_id!r}"
            )
        for i in range(max_participants):
            column = participant_column(i + 1, "present")
            if group[column].nunique() > 1:
                raise ValueError(
                    f"{column} varies within session {session_id!r} — present is the "
                    "STATIC slot-exists mask; per-frame dropouts belong in P{i}_pose_valid"
                )


@dataclass(frozen=True)
class SyntheticConfig:
    n_sessions: int = 12
    duration_s: float = 180.0
    frame_rate_hz: int = 25
    n_participants: int = 3
    max_participants: int = MAX_PARTICIPANTS
    shift_probability: float = 0.65
    overlap_probability: float = 0.22
    backchannel_rate_hz: float = 0.06
    cue_reliability: float = 0.7
    random_seed: int = 42


@dataclass(frozen=True)
class _Turn:
    speaker: int
    start_f: int
    end_f: int
    is_backchannel: bool = False


DEFAULT_SYNTHETIC_CONFIG = SyntheticConfig()


def generate_multiparty_dataset(
    config: SyntheticConfig = DEFAULT_SYNTHETIC_CONFIG,
) -> pd.DataFrame:
    if not 2 <= config.n_participants <= config.max_participants:
        raise ValueError("n_participants must be within [2, max_participants]")
    rng = np.random.default_rng(config.random_seed)
    sessions = [
        _generate_session(f"session_{idx:03d}", config, rng) for idx in range(config.n_sessions)
    ]
    return pd.concat(sessions, ignore_index=True)


def save_multiparty_dataset(
    path: str | Path, config: SyntheticConfig = DEFAULT_SYNTHETIC_CONFIG
) -> pd.DataFrame:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_df = generate_multiparty_dataset(config)
    if output_path.suffix.lower() == ".parquet":
        frame_df.to_parquet(output_path, index=False)
    else:
        frame_df.to_csv(output_path, index=False)
    return frame_df


def load_multiparty_dataset(
    path: str | Path, *, feature_schema: str = DEFAULT_FEATURE_SCHEMA
) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() == ".parquet":
        frame_df = pd.read_parquet(dataset_path)
    else:
        frame_df = pd.read_csv(dataset_path)
    validate_multiparty_dataframe(frame_df, feature_schema=feature_schema)
    return frame_df


def _generate_session(
    session_id: str, config: SyntheticConfig, rng: np.random.Generator
) -> pd.DataFrame:
    fps = config.frame_rate_hz
    n_frames = round(config.duration_s * fps)
    n_active = config.n_participants

    turns = _sample_turns(n_frames, config, rng)
    vad = _vad_from_turns(turns, n_frames, n_active)
    features = _features_from_state(turns, vad, config, rng)

    data: dict[str, np.ndarray] = {
        "timestamp_s": np.arange(n_frames) / fps,
    }
    for i in range(config.max_participants):
        present = i < n_active
        data[participant_column(i + 1, "present")] = np.full(n_frames, int(present))
        data[participant_column(i + 1, "vad")] = (
            vad[:, i].astype(int) if present else np.zeros(n_frames, dtype=int)
        )
        for name in PARTICIPANT_FEATURE_NAMES:
            key = participant_column(i + 1, name)
            data[key] = features[key] if present else np.zeros(n_frames)
    for name in GLOBAL_FEATURE_COLUMNS:
        data[name] = features[name]

    frame_df = pd.DataFrame(data)
    frame_df.insert(0, "session_id", session_id)
    return frame_df


def _sample_turns(
    n_frames: int, config: SyntheticConfig, rng: np.random.Generator
) -> list[_Turn]:
    fps = config.frame_rate_hz
    turns: list[_Turn] = []
    cursor = round(rng.uniform(0.2, 1.0) * fps)
    speaker = int(rng.integers(0, config.n_participants))

    while cursor < n_frames:
        duration = round(rng.uniform(1.2, 6.0) * fps)
        end = min(n_frames, cursor + duration)
        turns.append(_Turn(speaker=speaker, start_f=cursor, end_f=end))
        if end >= n_frames:
            break

        if rng.random() < config.shift_probability:
            candidates = [p for p in range(config.n_participants) if p != speaker]
            next_speaker = int(rng.choice(candidates))
        else:
            next_speaker = speaker

        if next_speaker != speaker and rng.random() < config.overlap_probability:
            gap = -round(rng.uniform(0.1, 0.5) * fps)  # competitive overlap entry
        else:
            gap = round(rng.uniform(0.15, 0.9) * fps)
        cursor = max(0, end + gap)
        speaker = next_speaker

    main_turns = list(turns)
    for turn in main_turns:
        turn_len = turn.end_f - turn.start_f
        expected = config.backchannel_rate_hz * turn_len / fps * (config.n_participants - 1)
        for _ in range(rng.poisson(expected)):
            listener_pool = [p for p in range(config.n_participants) if p != turn.speaker]
            listener = int(rng.choice(listener_pool))
            bc_len = round(rng.uniform(0.2, 0.5) * fps)
            bc_start = int(rng.integers(turn.start_f, max(turn.start_f + 1, turn.end_f - bc_len)))
            turns.append(
                _Turn(
                    speaker=listener,
                    start_f=bc_start,
                    end_f=min(n_frames, bc_start + bc_len),
                    is_backchannel=True,
                )
            )
    return turns


def _vad_from_turns(turns: list[_Turn], n_frames: int, n_participants: int) -> np.ndarray:
    vad = np.zeros((n_frames, n_participants), dtype=float)
    for turn in turns:
        vad[turn.start_f : turn.end_f, turn.speaker] = 1.0
    return vad


def _features_from_state(
    turns: list[_Turn],
    vad: np.ndarray,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    fps = config.frame_rate_hz
    n_frames, _ = vad.shape
    n_active = config.n_participants
    angles = np.linspace(-55.0, 55.0, n_active)
    features: dict[str, np.ndarray] = {}

    # --- anticipatory cue envelopes, only for genuine turn endings ---------
    final_lowering = np.zeros(n_frames)  # prosodic declination before offsets
    gaze_yield = np.zeros((n_frames, n_active))  # speaker looks up before yielding
    main_turns = [t for t in turns if not t.is_backchannel]
    for turn in main_turns:
        if rng.random() > config.cue_reliability:
            continue
        ramp_len = min(round(0.6 * fps), turn.end_f - turn.start_f)
        if ramp_len <= 0:
            continue
        ramp = np.linspace(0.0, 1.0, ramp_len)
        final_lowering[turn.end_f - ramp_len : turn.end_f] = np.maximum(
            final_lowering[turn.end_f - ramp_len : turn.end_f], ramp
        )
        gaze_yield[turn.end_f - ramp_len : turn.end_f, turn.speaker] = np.maximum(
            gaze_yield[turn.end_f - ramp_len : turn.end_f, turn.speaker], ramp
        )

    # Pre-onset cues of the *incoming* speaker (pre-turn in-breath, mouth
    # opening, head orientation) — the signal that lets a model distinguish
    # who takes the floor next, mirroring real conversational behaviour.
    pre_onset = np.zeros((n_frames, n_active))
    for turn in main_turns:
        if rng.random() > config.cue_reliability:
            continue
        lead = round(0.4 * fps)
        start = max(0, turn.start_f - lead)
        if turn.start_f <= start:
            continue
        ramp = np.linspace(0.0, 1.0, turn.start_f - start)
        pre_onset[start : turn.start_f, turn.speaker] = np.maximum(
            pre_onset[start : turn.start_f, turn.speaker], ramp
        )

    # Gaze-target simulation (MCGaze-style "who is being looked at"):
    # listeners watch the current speaker; when yielding to a *different*
    # speaker, the current speaker redirects gaze to the addressee shortly
    # before the offset and across the gap — the primary next-speaker cue.
    gaze_target = np.full((n_frames, n_active), -1, dtype=int)
    ordered = sorted(main_turns, key=lambda t: t.start_f)
    for turn_idx, turn in enumerate(ordered):
        watch_until = ordered[turn_idx + 1].start_f if turn_idx + 1 < len(ordered) else n_frames
        watch_until = max(watch_until, turn.end_f)
        for j in range(n_active):
            if j != turn.speaker:
                gaze_target[turn.start_f : watch_until, j] = turn.speaker
    for prev, nxt in zip(ordered[:-1], ordered[1:], strict=False):
        if nxt.speaker == prev.speaker or rng.random() > config.cue_reliability:
            continue
        redirect_start = max(prev.start_f, prev.end_f - round(0.6 * fps))
        redirect_end = min(n_frames, max(prev.end_f, nxt.start_f))
        gaze_target[redirect_start:redirect_end, prev.speaker] = nxt.speaker
    # observers occasionally look elsewhere (gaze-estimation noise)
    distracted = rng.random((n_frames, n_active)) < 0.12
    gaze_target[distracted] = -1

    # --- per-participant robot-observable features -------------------------
    for i in range(n_active):
        speaking = vad[:, i]

        asd = _smooth(speaking, attack=0.55, release=0.88)
        asd = np.clip(asd + rng.normal(0.0, 0.08, n_frames), 0.0, 1.0)
        # cross-talk false positives when anyone else speaks
        others_active = (vad.sum(axis=1) - speaking) > 0
        asd = np.where(
            others_active & (speaking == 0),
            np.clip(asd + rng.uniform(0.0, 0.25, n_frames), 0.0, 1.0),
            asd,
        )
        asd = np.clip(asd + 0.30 * pre_onset[:, i], 0.0, 1.0)
        features[participant_column(i + 1, "asd_prob")] = asd

        articulation = np.abs(np.sin(np.arange(n_frames) * rng.uniform(0.9, 1.4)))
        mar = np.where(speaking > 0, 0.12 + 0.28 * articulation, 0.08)
        mar = mar + 0.08 * pre_onset[:, i]
        features[participant_column(i + 1, "mar")] = np.clip(
            mar + rng.normal(0.0, 0.03, n_frames), 0.0, 0.8
        )

        gaze = 0.15 + 0.45 * gaze_yield[:, i] + rng.normal(0.0, 0.10, n_frames)
        features[participant_column(i + 1, "gaze_robot")] = np.clip(gaze, 0.0, 1.0)

        yaw = np.where(
            speaking > 0, rng.normal(8.0, 6.0, n_frames), rng.normal(22.0, 10.0, n_frames)
        )
        yaw = yaw * (1.0 - 0.6 * gaze_yield[:, i])  # head turns forward when yielding
        yaw = yaw * (1.0 - 0.5 * pre_onset[:, i])  # incoming speaker orients early
        features[participant_column(i + 1, "head_yaw_deg")] = np.abs(yaw)

        gazed_at = (gaze_target == i).sum(axis=1) / max(1, n_active - 1)
        features[participant_column(i + 1, "gazed_at")] = np.clip(
            gazed_at + rng.normal(0.0, 0.08, n_frames), 0.0, 1.0
        )

        # synthetic cameras never lose a face; real intake writes 0 on dropouts
        features[participant_column(i + 1, "pose_valid")] = np.ones(n_frames)
        features[participant_column(i + 1, "pose_quality")] = np.ones(n_frames)

    # --- global acoustic features (mic-array only in this synthetic pipeline;
    # real data sources energy/f0 from the combined mix, DoA stays inert) ----
    any_speech = vad.sum(axis=1) > 0
    dominant = np.argmax(vad + rng.normal(0.0, 0.05, vad.shape), axis=1)
    doa = np.where(any_speech, angles[dominant] + rng.normal(0.0, 5.0, n_frames), 0.0)
    doa[~any_speech] = rng.uniform(-60.0, 60.0, int((~any_speech).sum()))
    features["doa_angle_deg"] = doa

    energy = np.where(any_speech, rng.normal(0.8, 0.1, n_frames), rng.normal(0.08, 0.04, n_frames))
    features["array_energy"] = np.clip(energy * (1.0 - 0.35 * final_lowering), 0.0, 1.2)

    # A separate validity flag removes the old collision between an unvoiced
    # frame and a genuinely voiced 100 Hz frame (both mapped to f0_proxy=0).
    # False positives/negatives keep this robot-observable detector output a
    # noisy correlate rather than a deterministic copy of gold VAD.
    voiced_probability = np.where(any_speech, 0.93, 0.04)
    voiced = rng.random(n_frames) < voiced_probability
    features["voiced"] = voiced.astype(float)
    f0 = rng.normal(0.65, 0.30, n_frames) * (1.0 - 0.45 * final_lowering)
    features["f0_proxy"] = np.where(voiced, np.clip(f0, -1.5, 2.0), 0.0)

    return features


def _smooth(signal: np.ndarray, *, attack: float, release: float) -> np.ndarray:
    """Asymmetric exponential smoothing emulating detector attack/release lag."""
    output = np.zeros_like(signal, dtype=float)
    state = 0.0
    for idx, value in enumerate(signal):
        coefficient = attack if value > state else release
        state = coefficient * state + (1.0 - coefficient) * value
        output[idx] = state
    return output
