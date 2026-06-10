"""Multiparty frame-level schema and naive synthetic data generator.

Wide-format table at a fixed frame rate (default 25 fps), one row per frame,
columns named ``P{i}_<feature>`` per the project convention.

Feature tiers (CLAUDE.md design decision #2):
- Gold tier (labels / offline eval only): ``P{i}_vad`` from lavalier mics.
- Robot-observable tier (model input): per-face ASD probability, MAR,
  gaze-to-robot score, head yaw, plus global mic-array features.

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

GLOBAL_FEATURE_COLUMNS = ["doa_angle_deg", "array_energy", "f0_proxy"]
PARTICIPANT_FEATURE_NAMES = ["asd_prob", "mar", "gaze_robot", "head_yaw_deg", "gazed_at"]
METADATA_COLUMNS = ["session_id", "timestamp_s"]


def participant_column(index: int, feature: str) -> str:
    """1-based participant feature column name, e.g. ``P2_asd_prob``."""
    return f"P{index}_{feature}"


def vad_columns(max_participants: int = MAX_PARTICIPANTS) -> list[str]:
    return [participant_column(i + 1, "vad") for i in range(max_participants)]


def present_columns(max_participants: int = MAX_PARTICIPANTS) -> list[str]:
    return [participant_column(i + 1, "present") for i in range(max_participants)]


def input_feature_columns(max_participants: int = MAX_PARTICIPANTS) -> list[str]:
    """Robot-observable model input columns (never includes gold VAD)."""
    columns = []
    for i in range(max_participants):
        columns.extend(participant_column(i + 1, name) for name in PARTICIPANT_FEATURE_NAMES)
    columns.extend(GLOBAL_FEATURE_COLUMNS)
    return columns


def validate_multiparty_dataframe(
    frame_df: pd.DataFrame, max_participants: int = MAX_PARTICIPANTS
) -> None:
    required = [
        *METADATA_COLUMNS,
        *vad_columns(max_participants),
        *present_columns(max_participants),
        *input_feature_columns(max_participants),
    ]
    missing = [column for column in required if column not in frame_df.columns]
    if missing:
        raise ValueError(f"Multiparty dataset is missing required columns: {missing}")
    if frame_df.empty:
        raise ValueError("Multiparty dataset must contain at least one row")


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


def load_multiparty_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() == ".parquet":
        frame_df = pd.read_parquet(dataset_path)
    else:
        frame_df = pd.read_csv(dataset_path)
    validate_multiparty_dataframe(frame_df)
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

    # --- global mic-array features -----------------------------------------
    any_speech = vad.sum(axis=1) > 0
    dominant = np.argmax(vad + rng.normal(0.0, 0.05, vad.shape), axis=1)
    doa = np.where(any_speech, angles[dominant] + rng.normal(0.0, 5.0, n_frames), 0.0)
    doa[~any_speech] = rng.uniform(-60.0, 60.0, int((~any_speech).sum()))
    features["doa_angle_deg"] = doa

    energy = np.where(any_speech, rng.normal(0.8, 0.1, n_frames), rng.normal(0.08, 0.04, n_frames))
    features["array_energy"] = np.clip(energy * (1.0 - 0.35 * final_lowering), 0.0, 1.2)

    f0 = np.where(any_speech, rng.normal(1.0, 0.12, n_frames), 0.0)
    features["f0_proxy"] = np.clip(f0 * (1.0 - 0.45 * final_lowering), 0.0, 1.5)

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
