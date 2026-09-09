"""Turn-taking event extraction and event-level evaluation.

Derives events from gold VAD at evaluation time (CLAUDE.md decision #1)
following the conventions of Ekstedt & Skantze's VAP test suite:

- SHIFT/HOLD: at a mutual-silence gap bounded by single-speaker speech on both
  sides, did the floor change hands? The model is scored on the probability it
  assigns to "someone other than the previous speaker speaks next", averaged
  over the frames inside the gap.
- SHIFT-PREDICTION: in the window *before* the offset (while the speaker is
  still talking), does the model already anticipate the upcoming shift?
- NEXT-SPEAKER: over the same pre-offset window (or, optionally, the silence
  gap), which participant has the largest duration-weighted ``p_future``?
- OVERLAP-RESOLUTION (RQ3): when a challenger starts speaking over a stable
  floor holder, does the model predict who keeps the floor — discriminating
  competitive barge-ins (holder yields) from backchannels / failed barge-ins
  (holder continues)?
- ROBOT-ENTRY (demonstration agreement): at gaps following a *human* turn,
  does the model anticipate whether the robot slot — occupied in the real
  corpus by a human role-playing the robot — takes the floor?

Each test exposes a ``*_scores`` function returning per-event ``(labels,
scores)`` so callers can pool events across sessions before computing metrics
(per-session macro-averaging is noisy with tens of events per session).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
)

from hri_turn_taking.vap.labels import (
    DEFAULT_LABEL_CONFIG,
    VAPLabelConfig,
    p_now_p_future,
)

_SHIFT_HOLD_CONTEXT_S = 0.4


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
    context_s: float = _SHIFT_HOLD_CONTEXT_S,
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


@dataclass(frozen=True)
class OverlapEvent:
    """A challenger onset over a stable floor holder, with its resolution."""

    onset_f: int
    end_f: int  # first frame after the joint-activity stretch
    holder: int
    challenger: int
    holder_yields: bool  # True -> challenger takes the floor (competitive win)


def extract_overlap_events(
    vad: np.ndarray,
    *,
    frame_rate_hz: int = 25,
    pre_context_s: float = 0.4,
    min_overlap_s: float = 0.12,
    max_overlap_s: float = 2.0,
    post_context_s: float = 0.4,
) -> list[OverlapEvent]:
    """Find two-party overlaps with an unambiguous resolution.

    Requires the holder to have been the sole speaker for ``pre_context_s``
    before the challenger onset, no third participant during the overlap, and
    exactly one of the two active in the ``post_context_s`` after it ends.
    """
    activity = np.asarray(vad, dtype=float)
    if activity.ndim != 2:
        raise ValueError("vad must have shape (n_frames, n_participants)")
    n_frames, n_participants = activity.shape
    pre = max(1, round(pre_context_s * frame_rate_hz))
    min_overlap = max(1, round(min_overlap_s * frame_rate_hz))
    max_overlap = round(max_overlap_s * frame_rate_hz)
    post = max(1, round(post_context_s * frame_rate_hz))

    events: list[OverlapEvent] = []
    for onset in range(pre, n_frames - 1):
        for challenger in range(n_participants):
            if not (activity[onset, challenger] > 0 and activity[onset - 1, challenger] == 0):
                continue
            others = [i for i in np.flatnonzero(activity[onset] > 0) if i != challenger]
            if len(others) != 1:
                continue
            holder = int(others[0])
            pre_window = activity[onset - pre : onset]
            holder_solo = pre_window[:, holder].all() and (
                pre_window.sum(axis=1) == pre_window[:, holder]
            ).all()
            if not holder_solo:
                continue

            end = onset
            clean = True
            while (
                end < n_frames
                and activity[end, holder] > 0
                and activity[end, challenger] > 0
            ):
                if activity[end].sum() > 2:
                    clean = False
                    break
                end += 1
            overlap_len = end - onset
            if not clean or overlap_len < min_overlap or overlap_len > max_overlap:
                continue
            if end + post > n_frames:
                continue

            post_window = activity[end : end + post]
            holder_share = float(post_window[:, holder].mean())
            challenger_share = float(post_window[:, challenger].mean())
            if (holder_share >= 0.5) == (challenger_share >= 0.5):
                continue  # ambiguous resolution
            events.append(
                OverlapEvent(
                    onset_f=onset,
                    end_f=end,
                    holder=holder,
                    challenger=int(challenger),
                    holder_yields=bool(challenger_share >= 0.5),
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


def pairwise_shift_score_from_bins(
    bin_probabilities: np.ndarray,
    holder: int,
    challenger: int,
    *,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    present: np.ndarray | None = None,
    eps: float = 1e-9,
) -> float:
    """Compare one overlap event's challenger directly with its holder.

    Unlike :func:`shift_score_from_bins`, this readout must not let an
    unrelated, silent participant stand in for the challenger identified by
    :class:`OverlapEvent`.
    """
    probs = np.asarray(bin_probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("bin_probabilities must have shape (participants, bins)")
    n_participants = probs.shape[0]
    if not 0 <= holder < n_participants or not 0 <= challenger < n_participants:
        raise ValueError("holder and challenger must be valid participant slots")
    if holder == challenger:
        raise ValueError("holder and challenger must be different participant slots")
    if present is not None:
        present_mask = np.asarray(present, dtype=float) > 0
        if present_mask.shape != (n_participants,):
            raise ValueError("present must have shape (n_participants,)")
        if not present_mask[holder] or not present_mask[challenger]:
            raise ValueError("overlap holder and challenger must both be present")

    _, p_future = p_now_p_future(probs, label_config)
    holder_score = max(float(p_future[holder]), 0.0)
    challenger_score = max(float(p_future[challenger]), 0.0)
    if not np.isfinite(holder_score) or not np.isfinite(challenger_score):
        raise ValueError("holder and challenger scores must be finite")
    return float(challenger_score / (challenger_score + holder_score + eps))


def _event_validity_mask(valid: np.ndarray | None, vad: np.ndarray) -> np.ndarray | None:
    """Validate the full-future-horizon mask used by event score windows."""
    if valid is None:
        return None
    activity = np.asarray(vad)
    mask = np.asarray(valid)
    if activity.ndim != 2 or mask.shape != (activity.shape[0],):
        raise ValueError("valid must have shape (n_frames,)")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("valid must be binary")
    return mask.astype(bool, copy=False)


def _event_window_has_full_horizon(valid: np.ndarray | None, window: slice) -> bool:
    return valid is None or bool(valid[window].all())


def shift_hold_scores(
    bin_probabilities: np.ndarray,
    vad: np.ndarray,
    *,
    frame_rate_hz: int = 25,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    present: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> tuple[list[int], list[float]]:
    """Per-gap (label, score) pairs; the score is averaged over the gap frames
    (the model is causal, so every frame only sees past context).

    When ``valid`` is supplied, an event is retained only if every model output
    in its score window has the complete configured future horizon available.
    """
    validity = _event_validity_mask(valid, vad)
    labels, scores = [], []
    for event in extract_shift_hold_events(vad, frame_rate_hz=frame_rate_hz):
        window_slice = slice(event.gap_start_f, event.gap_end_f)
        if not _event_window_has_full_horizon(validity, window_slice):
            continue
        gap_scores = [
            shift_score_from_bins(
                frame_probs, event.previous_speaker, label_config=label_config, present=present
            )
            for frame_probs in bin_probabilities[window_slice]
        ]
        labels.append(int(event.is_shift))
        scores.append(float(np.mean(gap_scores)))
    return labels, scores


def _pre_offset_solo_window(
    vad: np.ndarray,
    event: TurnEvent,
    *,
    frame_rate_hz: int,
    pre_offset_s: float,
) -> slice:
    """Bound a predictive window to the event extractor's verified context.

    Starting at the gap boundary, walk backwards only over the contiguous tail
    where the event's previous speaker is the sole active participant.  The
    window is additionally capped at the 0.4 s context used by
    :func:`extract_shift_hold_events`, even when callers request a longer
    ``pre_offset_s``.
    """
    activity = np.asarray(vad, dtype=float) > 0
    requested = max(1, round(pre_offset_s * frame_rate_hz))
    verified = max(1, round(_SHIFT_HOLD_CONTEXT_S * frame_rate_hz))
    lower_bound = max(0, event.gap_start_f - min(requested, verified))
    start = event.gap_start_f
    while start > lower_bound:
        frame = activity[start - 1]
        if int(frame.sum()) != 1 or not bool(frame[event.previous_speaker]):
            break
        start -= 1
    return slice(start, event.gap_start_f)


def shift_prediction_scores(
    bin_probabilities: np.ndarray,
    vad: np.ndarray,
    *,
    frame_rate_hz: int = 25,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    present: np.ndarray | None = None,
    pre_offset_s: float = 0.5,
    valid: np.ndarray | None = None,
) -> tuple[list[int], list[float]]:
    """Per-gap (label, score) pairs read out *before* the offset, while the
    previous speaker is still talking.

    The window contains only the contiguous previous-speaker-solo tail and can
    never extend beyond the context verified by event extraction.
    """
    validity = _event_validity_mask(valid, vad)
    labels, scores = [], []
    for event in extract_shift_hold_events(vad, frame_rate_hz=frame_rate_hz):
        window_slice = _pre_offset_solo_window(
            vad,
            event,
            frame_rate_hz=frame_rate_hz,
            pre_offset_s=pre_offset_s,
        )
        window = bin_probabilities[window_slice]
        if window.shape[0] == 0:
            continue
        if not _event_window_has_full_horizon(validity, window_slice):
            continue
        window_scores = [
            shift_score_from_bins(
                frame_probs, event.previous_speaker, label_config=label_config, present=present
            )
            for frame_probs in window
        ]
        labels.append(int(event.is_shift))
        scores.append(float(np.mean(window_scores)))
    return labels, scores


def next_speaker_scores(
    bin_probabilities: np.ndarray,
    vad: np.ndarray,
    *,
    frame_rate_hz: int = 25,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    present: np.ndarray | None = None,
    pre_offset_s: float = 0.5,
    prediction_region: str = "pre_offset",
    valid: np.ndarray | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    """Return true next-speaker slots and participant ``p_future`` vectors.

    ``prediction_region="pre_offset"`` is the genuinely predictive test: it
    averages causal model outputs over the final ``pre_offset_s`` while the
    current holder is still speaking.  ``"gap"`` is also supported for the
    conventional silence-gap readout.  HOLD events remain in the multiclass
    test, with the previous and next speaker being the same slot.
    """
    probabilities = np.asarray(bin_probabilities, dtype=float)
    activity = np.asarray(vad, dtype=float)
    if probabilities.ndim != 3:
        raise ValueError("bin_probabilities must have shape (time, participants, bins)")
    if activity.shape != probabilities.shape[:2]:
        raise ValueError("vad must match the time and participant dimensions of probabilities")
    if prediction_region not in {"pre_offset", "gap"}:
        raise ValueError("prediction_region must be 'pre_offset' or 'gap'")
    validity = _event_validity_mask(valid, activity)

    present_mask = (
        np.ones(probabilities.shape[1], dtype=bool)
        if present is None
        else np.asarray(present, dtype=float) > 0
    )
    if present_mask.shape != (probabilities.shape[1],):
        raise ValueError("present must have shape (n_participants,)")

    labels: list[int] = []
    scores: list[np.ndarray] = []
    for event in extract_shift_hold_events(activity, frame_rate_hz=frame_rate_hz):
        if prediction_region == "pre_offset":
            window_slice = _pre_offset_solo_window(
                activity,
                event,
                frame_rate_hz=frame_rate_hz,
                pre_offset_s=pre_offset_s,
            )
            window = probabilities[window_slice]
        else:
            window_slice = slice(event.gap_start_f, event.gap_end_f)
            window = probabilities[window_slice]
        if window.shape[0] == 0:
            continue
        if not _event_window_has_full_horizon(validity, window_slice):
            continue
        _, p_future = p_now_p_future(window, label_config)
        participant_scores = np.mean(p_future, axis=0)
        participant_scores = np.where(present_mask, participant_scores, -np.inf)
        labels.append(event.next_speaker)
        scores.append(participant_scores.astype(float))
    return labels, scores


@dataclass(frozen=True)
class NextSpeakerParticipantEvaluation:
    """One-vs-rest support and tie-aware recall for one participant slot."""

    in_scope: bool
    n_positive: int
    n_negative: int
    ovr_average_precision: float | None
    tie_aware_recall: float | None
    ap_na_reason: str | None
    recall_na_reason: str | None


@dataclass(frozen=True)
class NextSpeakerEvaluation:
    """Pooled Next Speaker metrics with an explicit, slot-neutral tie policy."""

    n_events: int
    n_participants_in_scope: int
    n_participants_evaluable_ap: int
    participant_macro_ovr_ap: float | None
    participant_macro_ovr_ap_na_reason: str | None
    balanced_recall: float | None
    mrr: float | None
    top1_accuracy: float | None
    tie_policy: str
    tie_rate: float | None
    per_participant: dict[str, NextSpeakerParticipantEvaluation]
    na_reason: str | None


_NEXT_SPEAKER_TIE_POLICY = "uniform_credit_among_max_scores"


def _empty_next_speaker_evaluation(n_participants: int) -> NextSpeakerEvaluation:
    per_participant = {
        f"P{participant + 1}": NextSpeakerParticipantEvaluation(
            in_scope=False,
            n_positive=0,
            n_negative=0,
            ovr_average_precision=None,
            tie_aware_recall=None,
            ap_na_reason="not_in_scope",
            recall_na_reason="no_positive_events",
        )
        for participant in range(n_participants)
    }
    return NextSpeakerEvaluation(
        n_events=0,
        n_participants_in_scope=0,
        n_participants_evaluable_ap=0,
        participant_macro_ovr_ap=None,
        participant_macro_ovr_ap_na_reason="no_events",
        balanced_recall=None,
        mrr=None,
        top1_accuracy=None,
        tie_policy=_NEXT_SPEAKER_TIE_POLICY,
        tie_rate=None,
        per_participant=per_participant,
        na_reason="no_events",
    )


def score_next_speaker(
    labels: np.ndarray,
    score_vectors: np.ndarray,
) -> NextSpeakerEvaluation:
    """Score pooled Next Speaker predictions without slot-index tie bias.

    Top-1 ties use uniform fractional credit over all maximum-scoring present
    participants.  Mean reciprocal rank uses the corresponding average rank.
    Per-participant AP is computed one-vs-rest only on events where that slot
    has a finite score (``-inf`` denotes an absent slot).  The primary macro AP
    is ``None`` if any in-scope participant has single-class support.
    """
    targets = np.asarray(labels, dtype=int)
    scores = np.asarray(score_vectors, dtype=float)
    if targets.ndim != 1:
        raise ValueError("labels must have shape (events,)")
    if targets.size == 0 and scores.ndim == 1 and scores.size == 0:
        scores = np.empty((0, 0), dtype=float)
    if scores.ndim != 2 or scores.shape[0] != targets.size:
        raise ValueError("labels/scores must have shapes (events,) and (events, participants)")
    if targets.size == 0:
        return _empty_next_speaker_evaluation(scores.shape[1])
    if (targets < 0).any() or (targets >= scores.shape[1]).any():
        raise ValueError("next-speaker label is outside the score-vector participant range")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("scores may be finite or -inf for absent slots, but not NaN/+inf")
    true_scores = scores[np.arange(targets.size), targets]
    if not np.isfinite(true_scores).all():
        raise ValueError("the true next-speaker slot must have a finite score")

    top1_credits: list[float] = []
    reciprocal_ranks: list[float] = []
    tied_events = 0
    for row, target, true_score in zip(scores, targets, true_scores, strict=True):
        finite = np.isfinite(row)
        maximum = float(np.max(row[finite]))
        maximum_ties = finite & np.isclose(row, maximum, rtol=1e-7, atol=1e-12)
        n_maximum_ties = int(maximum_ties.sum())
        if n_maximum_ties > 1:
            tied_events += 1
        top1_credits.append(
            1.0 / n_maximum_ties if bool(maximum_ties[int(target)]) else 0.0
        )

        tied_with_true = finite & np.isclose(row, true_score, rtol=1e-7, atol=1e-12)
        strictly_higher = finite & (row > true_score) & ~tied_with_true
        n_strictly_higher = int(strictly_higher.sum())
        n_tied_other = int(tied_with_true.sum()) - 1
        average_rank = 1.0 + n_strictly_higher + 0.5 * n_tied_other
        reciprocal_ranks.append(1.0 / average_rank)

    credits = np.asarray(top1_credits, dtype=float)
    per_participant: dict[str, NextSpeakerParticipantEvaluation] = {}
    ap_values: list[float] = []
    recall_values: list[float] = []
    all_in_scope_ap_evaluable = True
    n_participants_in_scope = 0
    for participant in range(scores.shape[1]):
        in_scope_mask = np.isfinite(scores[:, participant])
        in_scope = bool(in_scope_mask.any())
        binary_targets = targets[in_scope_mask] == participant
        n_positive = int(binary_targets.sum())
        n_negative = int(binary_targets.size - n_positive)

        ap: float | None = None
        ap_na_reason: str | None = None
        if not in_scope:
            ap_na_reason = "not_in_scope"
        elif n_positive == 0:
            ap_na_reason = "no_positive_events"
        elif n_negative == 0:
            ap_na_reason = "no_negative_events"
        else:
            ap = float(
                average_precision_score(binary_targets, scores[in_scope_mask, participant])
            )
            ap_values.append(ap)

        recall: float | None = None
        recall_na_reason: str | None = None
        participant_positive = targets == participant
        if n_positive == 0:
            recall_na_reason = "no_positive_events"
        else:
            recall = float(np.mean(credits[participant_positive]))
            recall_values.append(recall)

        if in_scope:
            n_participants_in_scope += 1
            if ap is None:
                all_in_scope_ap_evaluable = False
        per_participant[f"P{participant + 1}"] = NextSpeakerParticipantEvaluation(
            in_scope=in_scope,
            n_positive=n_positive,
            n_negative=n_negative,
            ovr_average_precision=ap,
            tie_aware_recall=recall,
            ap_na_reason=ap_na_reason,
            recall_na_reason=recall_na_reason,
        )

    if n_participants_in_scope == 0:
        macro_ap = None
        macro_ap_na_reason = "no_in_scope_participants"
    elif not all_in_scope_ap_evaluable:
        macro_ap = None
        macro_ap_na_reason = "one_or_more_in_scope_participants_single_class"
    else:
        macro_ap = float(np.mean(ap_values))
        macro_ap_na_reason = None

    return NextSpeakerEvaluation(
        n_events=int(targets.size),
        n_participants_in_scope=n_participants_in_scope,
        n_participants_evaluable_ap=len(ap_values),
        participant_macro_ovr_ap=macro_ap,
        participant_macro_ovr_ap_na_reason=macro_ap_na_reason,
        balanced_recall=float(np.mean(recall_values)) if recall_values else None,
        mrr=float(np.mean(reciprocal_ranks)),
        top1_accuracy=float(np.mean(credits)),
        tie_policy=_NEXT_SPEAKER_TIE_POLICY,
        tie_rate=float(tied_events / targets.size),
        per_participant=per_participant,
        na_reason=None,
    )


def evaluate_next_speaker(
    bin_probabilities: np.ndarray,
    vad: np.ndarray,
    *,
    frame_rate_hz: int = 25,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    present: np.ndarray | None = None,
    pre_offset_s: float = 0.5,
    prediction_region: str = "pre_offset",
) -> NextSpeakerEvaluation:
    """Single-session convenience wrapper with an explicit empty result."""
    labels, scores = next_speaker_scores(
        bin_probabilities,
        vad,
        frame_rate_hz=frame_rate_hz,
        label_config=label_config,
        present=present,
        pre_offset_s=pre_offset_s,
        prediction_region=prediction_region,
    )
    score_array = (
        np.asarray(scores)
        if scores
        else np.empty((0, np.asarray(bin_probabilities).shape[1]), dtype=float)
    )
    return score_next_speaker(np.asarray(labels), score_array)


def overlap_resolution_scores(
    bin_probabilities: np.ndarray,
    vad: np.ndarray,
    *,
    frame_rate_hz: int = 25,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    present: np.ndarray | None = None,
    score_window_s: float = 0.3,
    valid: np.ndarray | None = None,
) -> tuple[list[int], list[float]]:
    """Per-overlap (label, score) pairs; label 1 = holder yields the floor.

    The shift score w.r.t. the holder is averaged over the first
    ``score_window_s`` of the overlap, so the readout happens while the fight
    for the floor is still undecided.
    """
    window = max(1, round(score_window_s * frame_rate_hz))
    validity = _event_validity_mask(valid, vad)
    labels, scores = [], []
    for event in extract_overlap_events(vad, frame_rate_hz=frame_rate_hz):
        stop = min(event.end_f, event.onset_f + window)
        window_slice = slice(event.onset_f, stop)
        if not _event_window_has_full_horizon(validity, window_slice):
            continue
        frame_scores = [
            pairwise_shift_score_from_bins(
                frame_probs,
                event.holder,
                event.challenger,
                label_config=label_config,
                present=present,
            )
            for frame_probs in bin_probabilities[window_slice]
        ]
        if not frame_scores:
            continue
        labels.append(int(event.holder_yields))
        scores.append(float(np.mean(frame_scores)))
    return labels, scores


def robot_entry_scores(
    bin_probabilities: np.ndarray,
    vad: np.ndarray,
    *,
    robot_slot: int,
    frame_rate_hz: int = 25,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    present: np.ndarray | None = None,
    max_gap_s: float = 2.0,
    valid: np.ndarray | None = None,
) -> tuple[list[int], list[float]]:
    """Per-gap (label, score) pairs for the ROBOT-ENTRY test.

    Restricts the shift/hold gaps to those following a *human* turn
    (``previous_speaker != robot_slot``) — the deployment-relevant decision
    "should the robot take this gap". The label is 1 when the demonstrator
    (the robot slot) actually took the floor; the score is the robot slot's
    duration-weighted p_future averaged over the gap frames.
    """
    if present is not None and float(np.asarray(present)[robot_slot]) <= 0:
        return [], []
    validity = _event_validity_mask(valid, vad)
    labels, scores = [], []
    for event in extract_shift_hold_events(
        vad, frame_rate_hz=frame_rate_hz, max_gap_s=max_gap_s
    ):
        if event.previous_speaker == robot_slot:
            continue
        window_slice = slice(event.gap_start_f, event.gap_end_f)
        if not _event_window_has_full_horizon(validity, window_slice):
            continue
        window = np.asarray(bin_probabilities[window_slice], dtype=float)
        _, p_future = p_now_p_future(window, label_config)
        labels.append(int(event.next_speaker == robot_slot))
        scores.append(float(np.mean(p_future[:, robot_slot])))
    return labels, scores


@dataclass(frozen=True)
class EventEvaluation:
    n_events: int
    n_positive: int  # shifts (shift tests) / holder-yields (overlap test)
    n_negative: int
    balanced_accuracy: float | None
    balanced_accuracy_na_reason: str | None
    f1: float | None
    f1_na_reason: str | None
    average_precision: float | None
    average_precision_na_reason: str | None


def score_events(labels: np.ndarray, scores: np.ndarray, threshold: float) -> EventEvaluation:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("labels and scores must be one-dimensional arrays of equal shape")
    if not np.isfinite(scores).all():
        raise ValueError("event scores must be finite")
    n_events = int(labels.size)
    n_positive = int(labels.sum())
    n_negative = n_events - n_positive
    if n_events == 0:
        return EventEvaluation(
            n_events=0,
            n_positive=0,
            n_negative=0,
            balanced_accuracy=None,
            balanced_accuracy_na_reason="no_events",
            f1=None,
            f1_na_reason="no_events",
            average_precision=None,
            average_precision_na_reason="no_events",
        )
    predictions = (scores >= threshold).astype(int)
    single_class = np.unique(labels).size < 2
    single_class_reason = "no_positive_events" if n_positive == 0 else "no_negative_events"
    return EventEvaluation(
        n_events=n_events,
        n_positive=n_positive,
        n_negative=n_negative,
        balanced_accuracy=None
        if single_class
        else float(balanced_accuracy_score(labels, predictions)),
        balanced_accuracy_na_reason=single_class_reason if single_class else None,
        f1=float(f1_score(labels, predictions, zero_division=0)),
        f1_na_reason=None,
        average_precision=None if single_class else float(average_precision_score(labels, scores)),
        average_precision_na_reason=single_class_reason if single_class else None,
    )


def evaluate_shift_hold(
    bin_probabilities: np.ndarray,
    vad: np.ndarray,
    *,
    frame_rate_hz: int = 25,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    present: np.ndarray | None = None,
    threshold: float = 0.5,
) -> EventEvaluation:
    """Single-session convenience wrapper with an explicit empty result."""
    labels, scores = shift_hold_scores(
        bin_probabilities,
        vad,
        frame_rate_hz=frame_rate_hz,
        label_config=label_config,
        present=present,
    )
    return score_events(np.asarray(labels), np.asarray(scores), threshold)


def evaluate_shift_prediction(
    bin_probabilities: np.ndarray,
    vad: np.ndarray,
    *,
    frame_rate_hz: int = 25,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    present: np.ndarray | None = None,
    pre_offset_s: float = 0.5,
    threshold: float = 0.5,
) -> EventEvaluation:
    """Single-session convenience wrapper with an explicit empty result."""
    labels, scores = shift_prediction_scores(
        bin_probabilities,
        vad,
        frame_rate_hz=frame_rate_hz,
        label_config=label_config,
        present=present,
        pre_offset_s=pre_offset_s,
    )
    return score_events(np.asarray(labels), np.asarray(scores), threshold)
