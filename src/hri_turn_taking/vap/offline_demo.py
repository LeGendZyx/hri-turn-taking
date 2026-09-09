"""Offline policy comparisons on recorded sessions.

Two comparisons, both replayed on gold VAD:

- ``compare_policies_on_session``: floor-holder dual-threshold policy vs. the
  reactive silence baseline — entry rate and latency at every silence gap
  (the primary objective KPI of the online evaluation, RQ3).
- ``compare_policies_with_demonstrator``: scores each policy against the
  demonstrator's own behaviour. The real corpus contains a human role-playing
  the robot, so the gaps that person did or did not take are ground truth for
  "should the robot enter" — a reactive baseline cannot avoid entering gaps
  the demonstrator left alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hri_turn_taking.vap.dataset import FeatureNormalizer, SessionTensors
from hri_turn_taking.vap.events import extract_shift_hold_events, robot_entry_scores
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG, VAPLabelConfig
from hri_turn_taking.vap.model import MultipartyVAP
from hri_turn_taking.vap.policy import (
    DEFAULT_DEMONSTRATION_CONFIG,
    DEFAULT_POLICY_CONFIG,
    DemonstrationPolicy,
    DualThresholdPolicy,
    PolicyConfig,
)
from hri_turn_taking.vap.reactive_baseline import SilenceBaseline
from hri_turn_taking.vap.trainer import predict_session_probabilities


@dataclass(frozen=True)
class PolicyComparison:
    n_gaps: int
    predictive_entries: int
    predictive_mean_latency_s: float
    reactive_entries: int
    reactive_mean_latency_s: float


def compare_policies_on_session(
    session: SessionTensors,
    model: MultipartyVAP,
    normalizer: FeatureNormalizer,
    *,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    policy_config: PolicyConfig = DEFAULT_POLICY_CONFIG,
    silence_baseline_s: float = 1.5,
    device: str = "cpu",
) -> PolicyComparison | None:
    fps = label_config.frame_rate_hz
    probs = predict_session_probabilities(model, session, normalizer, device=device)

    baseline = SilenceBaseline(silence_seconds=silence_baseline_s, frame_rate_hz=fps)
    any_speech = session.vad.sum(axis=1) > 0
    reactive_triggers = np.array([baseline.update(bool(speech)) for speech in any_speech])

    events = extract_shift_hold_events(session.vad, frame_rate_hz=fps, max_gap_s=10.0)
    if not events:
        return None

    predictive_latencies, reactive_latencies = [], []
    for event in events:
        # Replay the gap as a response opportunity: the previous speaker is
        # the floor holder; the robot may respond once they have yielded.
        policy = DualThresholdPolicy(policy_config, label_config)
        entry: float | None = None
        for t in range(event.gap_start_f, event.gap_end_f):
            decision = policy.update(
                probs[t],
                robot_has_intent=True,
                present=session.present,
                floor_holder=event.previous_speaker,
            )
            if decision.should_speak:
                entry = float((t - event.gap_start_f) / fps)
                break
        predictive_latencies.append(entry)
        reactive_latencies.append(
            _entry_latency(reactive_triggers, event.gap_start_f, event.gap_end_f, fps)
        )

    predictive_hits = [lat for lat in predictive_latencies if lat is not None]
    reactive_hits = [lat for lat in reactive_latencies if lat is not None]
    nan = float("nan")
    return PolicyComparison(
        n_gaps=len(events),
        predictive_entries=len(predictive_hits),
        predictive_mean_latency_s=float(np.mean(predictive_hits)) if predictive_hits else nan,
        reactive_entries=len(reactive_hits),
        reactive_mean_latency_s=float(np.mean(reactive_hits)) if reactive_hits else nan,
    )


def _entry_latency(
    triggers: np.ndarray, gap_start_f: int, gap_end_f: int, fps: int
) -> float | None:
    """Seconds from gap onset to the first trigger inside the gap, else None."""
    window = triggers[gap_start_f:gap_end_f]
    hits = np.flatnonzero(window)
    if hits.size == 0:
        return None
    return float(hits[0] / fps)


@dataclass(frozen=True)
class PolicyDemonstrationStats:
    entries_on_robot_gaps: int  # gaps the demonstrator took AND the policy entered
    false_entries: int  # entries into gaps the demonstrator left alone
    mean_lead_vs_demonstrator_s: float  # >0 = earlier than the human; nan without entries


@dataclass(frozen=True)
class DemonstrationComparison:
    robot_slot: int
    n_opportunities: int  # gaps following a human turn
    n_robot_gaps: int  # of those, gaps the demonstrator actually took
    demonstration: PolicyDemonstrationStats
    floor_holder: PolicyDemonstrationStats
    reactive: PolicyDemonstrationStats
    entry_labels: list[int]  # per-opportunity ground truth (for pooled AP)
    entry_scores: list[float]  # per-opportunity robot-slot p_future readout


def compare_policies_with_demonstrator(
    session: SessionTensors,
    model: MultipartyVAP,
    normalizer: FeatureNormalizer,
    *,
    robot_slot: int,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    policy_config: PolicyConfig = DEFAULT_POLICY_CONFIG,
    demonstration_config: PolicyConfig = DEFAULT_DEMONSTRATION_CONFIG,
    silence_baseline_s: float = 1.5,
    max_gap_s: float = 10.0,
    device: str = "cpu",
) -> DemonstrationComparison | None:
    """Score every policy against the demonstrator's own entries.

    Opportunities are the mutual-silence gaps following a *human* turn
    (``previous_speaker != robot_slot``). Ground truth: the gaps the
    demonstrator — the human role-playing the robot — actually took. Per
    policy: entries on demonstrator-taken gaps (recall), entries on the
    remaining gaps (false entries), and mean lead time relative to the
    demonstrator's own onset (the human ceiling).
    """
    if float(session.present[robot_slot]) <= 0:
        return None
    fps = label_config.frame_rate_hz
    probs = predict_session_probabilities(model, session, normalizer, device=device)
    events = [
        event
        for event in extract_shift_hold_events(
            session.vad, frame_rate_hz=fps, max_gap_s=max_gap_s
        )
        if event.previous_speaker != robot_slot
    ]
    if not events:
        return None
    entry_labels, entry_scores = robot_entry_scores(
        probs,
        session.vad,
        robot_slot=robot_slot,
        frame_rate_hz=fps,
        label_config=label_config,
        present=session.present,
        max_gap_s=max_gap_s,
    )

    baseline = SilenceBaseline(silence_seconds=silence_baseline_s, frame_rate_hz=fps)
    any_speech = session.vad.sum(axis=1) > 0
    reactive_triggers = np.array([baseline.update(bool(speech)) for speech in any_speech])

    holder_entries: list[float | None] = []
    demonstration_entries: list[float | None] = []
    reactive_entries: list[float | None] = []
    for event in events:
        holder_policy = DualThresholdPolicy(policy_config, label_config)
        demo_policy = DemonstrationPolicy(demonstration_config, label_config)
        holder_entry: float | None = None
        demo_entry: float | None = None
        for t in range(event.gap_start_f, event.gap_end_f):
            offset = float((t - event.gap_start_f) / fps)
            if holder_entry is None:
                decision = holder_policy.update(
                    probs[t],
                    robot_has_intent=True,
                    present=session.present,
                    floor_holder=event.previous_speaker,
                )
                if decision.should_speak:
                    holder_entry = offset
            if demo_entry is None:
                decision = demo_policy.update(
                    probs[t],
                    robot_slot=robot_slot,
                    robot_has_intent=True,
                    present=session.present,
                )
                if decision.should_speak:
                    demo_entry = offset
            if holder_entry is not None and demo_entry is not None:
                break
        holder_entries.append(holder_entry)
        demonstration_entries.append(demo_entry)
        reactive_entries.append(
            _entry_latency(reactive_triggers, event.gap_start_f, event.gap_end_f, fps)
        )

    def stats(entries: list[float | None]) -> PolicyDemonstrationStats:
        on_robot = false_entries = 0
        leads: list[float] = []
        for event, label, entry in zip(events, entry_labels, entries, strict=True):
            if entry is None:
                continue
            if label:
                on_robot += 1
                leads.append((event.gap_end_f - event.gap_start_f) / fps - entry)
            else:
                false_entries += 1
        return PolicyDemonstrationStats(
            entries_on_robot_gaps=on_robot,
            false_entries=false_entries,
            mean_lead_vs_demonstrator_s=float(np.mean(leads)) if leads else float("nan"),
        )

    return DemonstrationComparison(
        robot_slot=robot_slot,
        n_opportunities=len(events),
        n_robot_gaps=int(sum(entry_labels)),
        demonstration=stats(demonstration_entries),
        floor_holder=stats(holder_entries),
        reactive=stats(reactive_entries),
        entry_labels=entry_labels,
        entry_scores=entry_scores,
    )
