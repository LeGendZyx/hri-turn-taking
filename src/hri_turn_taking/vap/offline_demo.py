"""Offline predictive-vs-reactive policy comparison on recorded sessions.

Replays gold VAD through (a) the VAP model + dual-threshold policy and
(b) the reactive silence baseline, then measures floor-entry latency at every
silence gap — the primary objective KPI of the online evaluation (RQ3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hri_turn_taking.vap.dataset import FeatureNormalizer, SessionTensors
from hri_turn_taking.vap.events import extract_shift_hold_events
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG, VAPLabelConfig
from hri_turn_taking.vap.model import MultipartyVAP
from hri_turn_taking.vap.policy import (
    DEFAULT_POLICY_CONFIG,
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
