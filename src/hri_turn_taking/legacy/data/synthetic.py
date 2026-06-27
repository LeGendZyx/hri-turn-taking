from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_turn_taking_data(
    *,
    n_sessions: int = 8,
    duration_s: float = 120.0,
    frame_rate_hz: int = 25,
    n_participants: int = 3,
    trp_horizon_s: float = 0.50,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a small but useful stand-in dataset before real recordings exist."""
    if n_sessions < 1:
        raise ValueError("n_sessions must be >= 1")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if frame_rate_hz <= 0:
        raise ValueError("frame_rate_hz must be positive")
    if n_participants < 2:
        raise ValueError("n_participants must be >= 2")

    rng = np.random.default_rng(random_seed)
    frames: list[pd.DataFrame] = []
    participant_angles = np.linspace(-55.0, 55.0, n_participants)

    for session_idx in range(n_sessions):
        timeline = _sample_conversation_timeline(duration_s, n_participants, rng)
        timestamps = np.arange(0.0, duration_s, 1.0 / frame_rate_hz)
        rows = [
            _frame_from_timeline(
                timestamp_s=float(timestamp),
                session_id=f"session_{session_idx:03d}",
                timeline=timeline,
                participant_angles=participant_angles,
                trp_horizon_s=trp_horizon_s,
                rng=rng,
            )
            for timestamp in timestamps
        ]
        frames.append(pd.DataFrame(rows))

    return pd.concat(frames, ignore_index=True)


def save_synthetic_turn_taking_data(path: str | Path, **kwargs: object) -> pd.DataFrame:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_df = generate_synthetic_turn_taking_data(**kwargs)
    frame_df.to_csv(output_path, index=False)
    return frame_df


def _sample_conversation_timeline(
    duration_s: float,
    n_participants: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int | bool]]:
    timeline: list[dict[str, float | int | bool]] = []
    current_time = 0.0
    previous_speaker = -1

    while current_time < duration_s:
        gap_s = float(rng.uniform(0.05, 0.65))
        current_time += gap_s
        if current_time >= duration_s:
            break

        speaker = int(rng.integers(0, n_participants))
        if speaker == previous_speaker and n_participants > 1:
            speaker = (speaker + 1) % n_participants

        turn_duration_s = float(rng.uniform(1.2, 5.0))
        start_s = current_time
        end_s = min(duration_s, start_s + turn_duration_s)
        has_overlap = bool(rng.random() < 0.32)
        overlap_start_s = end_s - float(rng.uniform(0.15, 0.65)) if has_overlap else duration_s + 1
        overlap_speaker = int((speaker + int(rng.integers(1, n_participants))) % n_participants)

        timeline.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "speaker": speaker,
                "has_overlap": has_overlap,
                "overlap_start_s": max(start_s, overlap_start_s),
                "overlap_speaker": overlap_speaker,
            }
        )
        current_time = end_s
        previous_speaker = speaker

    return timeline


def _frame_from_timeline(
    *,
    timestamp_s: float,
    session_id: str,
    timeline: list[dict[str, float | int | bool]],
    participant_angles: np.ndarray,
    trp_horizon_s: float,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    active_turn = None
    last_end = 0.0
    recent_change = 0
    for turn in timeline:
        start_s = float(turn["start_s"])
        end_s = float(turn["end_s"])
        if end_s <= timestamp_s:
            last_end = end_s
        if start_s <= timestamp_s <= end_s:
            active_turn = turn
            recent_change = int(timestamp_s - start_s <= 0.40)
            break

    if active_turn is None:
        silence_duration_s = max(0.0, timestamp_s - last_end)
        return {
            "session_id": session_id,
            "timestamp_s": timestamp_s,
            "audio_vad": 0,
            "active_speaker_count": 0,
            "overlap_active": 0,
            "dominant_doa_angle_deg": float(rng.normal(0.0, 6.0)),
            "mouth_aspect_ratio_max": float(np.clip(rng.normal(0.09, 0.025), 0.02, 0.20)),
            "visual_vad_max": 0,
            "gaze_to_robot_score": float(np.clip(rng.normal(0.22, 0.12), 0.0, 1.0)),
            "head_yaw_abs_min_deg": float(np.clip(rng.normal(28.0, 8.0), 0.0, 70.0)),
            "speaking_towards_robot": 0,
            "silence_duration_s": silence_duration_s,
            "current_turn_duration_s": 0.0,
            "recent_speaker_change": 0,
            "trp_label": int(0.15 <= silence_duration_s <= 0.55),
        }

    speaker = int(active_turn["speaker"])
    start_s = float(active_turn["start_s"])
    end_s = float(active_turn["end_s"])
    time_to_turn_end_s = max(0.0, end_s - timestamp_s)
    near_trp = time_to_turn_end_s <= trp_horizon_s
    overlap_active = int(
        bool(active_turn["has_overlap"]) and timestamp_s >= float(active_turn["overlap_start_s"])
    )
    active_speaker_count = 2 if overlap_active else 1

    gaze_base = 0.72 if near_trp else 0.38
    head_base = 9.0 if near_trp else 25.0
    speaking_towards_robot = int(gaze_base + rng.normal(0.0, 0.12) > 0.60)

    return {
        "session_id": session_id,
        "timestamp_s": timestamp_s,
        "audio_vad": 1,
        "active_speaker_count": active_speaker_count,
        "overlap_active": overlap_active,
        "dominant_doa_angle_deg": float(participant_angles[speaker] + rng.normal(0.0, 4.0)),
        "mouth_aspect_ratio_max": float(np.clip(rng.normal(0.35, 0.08), 0.05, 0.75)),
        "visual_vad_max": 1,
        "gaze_to_robot_score": float(np.clip(rng.normal(gaze_base, 0.16), 0.0, 1.0)),
        "head_yaw_abs_min_deg": float(np.clip(rng.normal(head_base, 6.0), 0.0, 70.0)),
        "speaking_towards_robot": speaking_towards_robot,
        "silence_duration_s": 0.0,
        "current_turn_duration_s": timestamp_s - start_s,
        "recent_speaker_change": recent_change,
        "trp_label": int(near_trp and speaking_towards_robot),
    }
