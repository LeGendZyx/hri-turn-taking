"""Turn-taking event extraction and event-level evaluation.

Derives shift/hold events from gold VAD at evaluation time (CLAUDE.md
decision #1) following the conventions of Ekstedt & Skantze's VAP test suite:

- SHIFT/HOLD: at a mutual-silence gap bounded by single-speaker speech on both
  sides, did the floor change hands? The model is scored on the probability it
  assigns to "someone other than the previous speaker speaks next", read out
  at the gap onset.
- SHIFT-PREDICTION: in the window *before* the offset (while the speaker is
  still talking), does the model already anticipate the upcoming shift?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG, VAPLabelConfig, p_now_p_future


@dataclass(frozen=True)
class TurnEvent:
    gap_start_f: int
    gap_end_f: int
    previous_speaker: int
    next_speaker: int

    @property
    def is_shift(self) -> bool:
        return self.previous_speaker != self.next_speaker


def extract_shift_hold_events(
    vad: np.ndarray,
    *,
    frame_rate_hz: int = 25,
    min_gap_s: float = 0.2,
    max_gap_s: float = 2.0,
    context_s: float = 0.4,
) -> list[TurnEvent]:
    """Find mutual-silence gaps with a single clean speaker on each side."""
    activity = np.asarray(vad, dtype=float)
    if activity.ndim != 2:
        raise ValueError("vad must have shape (n_frames, n_participants)")
    n_frames = activity.shape[0]
    silent = activity.sum(axis=1) == 0
    min_gap = max(1, round(min_gap_s * frame_rate_hz))
    max_gap = round(max_gap_s * frame_rate_hz)
    context = max(1, round(context_s * frame_rate_hz))

    events: list[TurnEvent] = []
    idx = 0
    while idx < n_frames:
        if not silent[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx < n_frames and silent[idx]:
            idx += 1
        gap_end = idx  # first non-silent frame after the gap
        gap_len = gap_end - gap_start
        if gap_len < min_gap or gap_len > max_gap:
            continue
        if gap_start < context or gap_end + context > n_frames:
            continue

        before = activity[gap_start - context : gap_start]
        after = activity[gap_end : gap_end + context]
        before_speakers = np.flatnonzero(before.sum(axis=0) > 0)
        after_speakers = np.flatnonzero(after.sum(axis=0) > 0)
        if before_speakers.size != 1 or after_speakers.size != 1:
            continue
        events.append(
            TurnEvent(
                gap_start_f=gap_start,
                gap_end_f=gap_end,
                previous_speaker=int(before_speakers[0]),
                next_speaker=int(after_speakers[0]),
            )
        )
    return events


def shift_score_from_bins(
    bin_probabilities: np.ndarray,
    previous_speaker: int,
    *,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    present: np.ndarray | None = None,
    eps: float = 1e-9,
) -> float:
    """P(shift) readout: future speech of the strongest challenger vs. holder.

    bin_probabilities: (n_participants, n_bins) at a single frame.
    """
    probs = np.asarray(bin_probabilities, dtype=float)
    _, p_future = p_now_p_future(probs, label_config)
    if present is not None:
        p_future = np.where(np.asarray(present) > 0, p_future, -np.inf)
    challenger = np.max(np.delete(p_future, previous_speaker))
    holder = p_future[previous_speaker]
    if not np.isfinite(challenger):
        return 0.0
    return float(challenger / (challenger + max(holder, 0.0) + eps))


@dataclass(frozen=True)
class EventEvaluation:
    n_events: int
    n_shifts: int
    balanced_accuracy: float
    f1: float
    average_precision: float


def evaluate_shift_hold(
    bin_probabilities: np.ndarray,
    vad: np.ndarray,
    *,
    frame_rate_hz: int = 25,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    present: np.ndarray | None = None,
    threshold: float = 0.5,
) -> EventEvaluation | None:
    """Score shift-vs-hold discrimination over mutual-silence gaps.

    Following the VAP evaluation convention, the model's shift score is
    averaged over the frames inside the gap (the model is causal, so every
    frame only sees past context).

    bin_probabilities: (T, n_participants, n_bins) causal model outputs.
    Returns None when the session contains no scoreable events.
    """
    events = extract_shift_hold_events(vad, frame_rate_hz=frame_rate_hz)
    if not events:
        return None
    scores, labels = [], []
    for event in events:
        gap_scores = [
            shift_score_from_bins(
                frame_probs,
                event.previous_speaker,
                label_config=label_config,
                present=present,
            )
            for frame_probs in bin_probabilities[event.gap_start_f : event.gap_end_f]
        ]
        scores.append(float(np.mean(gap_scores)))
        labels.append(int(event.is_shift))
    return _score_events(np.asarray(labels), np.asarray(scores), threshold)


def evaluate_shift_prediction(
    bin_probabilities: np.ndarray,
    vad: np.ndarray,
    *,
    frame_rate_hz: int = 25,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    present: np.ndarray | None = None,
    pre_offset_s: float = 0.5,
    threshold: float = 0.5,
) -> EventEvaluation | None:
    """Score anticipation of shifts while the previous speaker still talks."""
    events = extract_shift_hold_events(vad, frame_rate_hz=frame_rate_hz)
    if not events:
        return None
    region = max(1, round(pre_offset_s * frame_rate_hz))
    scores, labels = [], []
    for event in events:
        start = max(0, event.gap_start_f - region)
        window = bin_probabilities[start : event.gap_start_f]
        if window.shape[0] == 0:
            continue
        window_scores = [
            shift_score_from_bins(
                frame_probs,
                event.previous_speaker,
                label_config=label_config,
                present=present,
            )
            for frame_probs in window
        ]
        scores.append(float(np.mean(window_scores)))
        labels.append(int(event.is_shift))
    if not scores:
        return None
    return _score_events(np.asarray(labels), np.asarray(scores), threshold)


def _score_events(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> EventEvaluation:
    predictions = (scores >= threshold).astype(int)
    single_class = np.unique(labels).size < 2
    return EventEvaluation(
        n_events=int(labels.size),
        n_shifts=int(labels.sum()),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        average_precision=float("nan")
        if single_class
        else float(average_precision_score(labels, scores)),
    )
