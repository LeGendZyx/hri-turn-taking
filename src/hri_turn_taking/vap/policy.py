"""Robot floor-taking policies on top of VAP outputs.

Two dual-threshold p_now / p_future entry modes (Skantze & Irfan, HRI 2025;
CLAUDE.md decision #6) instead of a single P(TRP) threshold:

- ``DualThresholdPolicy`` (floor-holder gated): enter once the current
  holder's p_now / p_future both stay *below* their thresholds — the holder
  has yielded.
- ``DemonstrationPolicy`` (demonstration driven): enter once the *robot
  slot's own* p_now / p_future both rise *above* their thresholds — the model
  predicts that a human in the robot's position (the corpus' role-player)
  would be speaking now and would sustain the turn.

Both debounce over consecutive frames and apply a refractory period after a
take-turn decision. Which mode drives the live robot is an open choice; the
offline demo compares them against the demonstrator's own entries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG, VAPLabelConfig, p_now_p_future


@dataclass(frozen=True)
class PolicyConfig:
    now_threshold: float = 0.35
    future_threshold: float = 0.45
    min_consecutive_frames: int = 3
    refractory_frames: int = 25  # 1 s at 25 fps

    def __post_init__(self) -> None:
        for name in ("now_threshold", "future_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.min_consecutive_frames < 1:
            raise ValueError("min_consecutive_frames must be >= 1")
        if self.refractory_frames < 0:
            raise ValueError("refractory_frames must be >= 0")


DEFAULT_POLICY_CONFIG = PolicyConfig()


@dataclass(frozen=True)
class PolicyDecision:
    should_speak: bool
    p_now_max: float
    p_future_max: float
    reason: str


class DualThresholdPolicy:
    """Stateful frame-by-frame floor-entry decision."""

    def __init__(
        self,
        config: PolicyConfig = DEFAULT_POLICY_CONFIG,
        label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    ) -> None:
        self.config = config
        self.label_config = label_config
        self._consecutive_clear = 0
        self._refractory_remaining = 0

    def reset(self) -> None:
        self._consecutive_clear = 0
        self._refractory_remaining = 0

    def update(
        self,
        bin_probabilities: np.ndarray,
        *,
        robot_has_intent: bool,
        present: np.ndarray | None = None,
        floor_holder: int | None = None,
    ) -> PolicyDecision:
        """bin_probabilities: (n_participants, n_bins) for the current frame.

        When ``floor_holder`` is given (the participant who last held the
        floor, e.g. from the ASD stream), the thresholds gate on *that*
        participant continuing — the original Skantze & Irfan semantics: the
        robot may respond once the holder has yielded. Without it, the policy
        falls back to the conservative max over all present participants.
        """
        probs = np.asarray(bin_probabilities, dtype=float)
        p_now, p_future = p_now_p_future(probs, self.label_config)
        if present is not None:
            mask = np.asarray(present, dtype=float) > 0
            p_now = np.where(mask, p_now, 0.0)
            p_future = np.where(mask, p_future, 0.0)
        if floor_holder is not None:
            p_now_max = float(p_now[floor_holder])
            p_future_max = float(p_future[floor_holder])
        else:
            p_now_max = float(p_now.max())
            p_future_max = float(p_future.max())

        if self._refractory_remaining > 0:
            self._refractory_remaining -= 1
            return PolicyDecision(False, p_now_max, p_future_max, "refractory")

        floor_open = (
            p_now_max < self.config.now_threshold
            and p_future_max < self.config.future_threshold
        )
        self._consecutive_clear = self._consecutive_clear + 1 if floor_open else 0

        if not robot_has_intent:
            return PolicyDecision(False, p_now_max, p_future_max, "no_robot_intent")
        if not floor_open:
            return PolicyDecision(False, p_now_max, p_future_max, "humans_hold_floor")
        if self._consecutive_clear < self.config.min_consecutive_frames:
            return PolicyDecision(False, p_now_max, p_future_max, "debouncing")

        self._consecutive_clear = 0
        self._refractory_remaining = self.config.refractory_frames
        return PolicyDecision(True, p_now_max, p_future_max, "floor_open")


DEFAULT_DEMONSTRATION_CONFIG = PolicyConfig(now_threshold=0.5, future_threshold=0.5)


class DemonstrationPolicy:
    """Demonstration-driven floor entry: act when the model predicts that the
    *robot slot itself* would be speaking.

    The real corpus contains a human role-playing the robot, so the robot
    slot's future-activity prediction reads as "would a human in the robot's
    position speak now". Entry requires *both* readouts to exceed their
    thresholds — an imminent (p_now) and sustained (p_future) turn — which
    naturally skips predicted backchannels. The reported decision carries the
    robot slot's p_now / p_future in the ``p_now_max`` / ``p_future_max``
    fields.
    """

    def __init__(
        self,
        config: PolicyConfig = DEFAULT_DEMONSTRATION_CONFIG,
        label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    ) -> None:
        self.config = config
        self.label_config = label_config
        self._consecutive_active = 0
        self._refractory_remaining = 0

    def reset(self) -> None:
        self._consecutive_active = 0
        self._refractory_remaining = 0

    def update(
        self,
        bin_probabilities: np.ndarray,
        *,
        robot_slot: int,
        robot_has_intent: bool,
        present: np.ndarray | None = None,
    ) -> PolicyDecision:
        """bin_probabilities: (n_participants, n_bins) for the current frame."""
        probs = np.asarray(bin_probabilities, dtype=float)
        p_now, p_future = p_now_p_future(probs, self.label_config)
        p_now_robot = float(p_now[robot_slot])
        p_future_robot = float(p_future[robot_slot])
        if present is not None and float(np.asarray(present)[robot_slot]) <= 0:
            return PolicyDecision(False, p_now_robot, p_future_robot, "robot_slot_absent")

        if self._refractory_remaining > 0:
            self._refractory_remaining -= 1
            return PolicyDecision(False, p_now_robot, p_future_robot, "refractory")

        wants_floor = (
            p_now_robot > self.config.now_threshold
            and p_future_robot > self.config.future_threshold
        )
        self._consecutive_active = self._consecutive_active + 1 if wants_floor else 0

        if not robot_has_intent:
            return PolicyDecision(False, p_now_robot, p_future_robot, "no_robot_intent")
        if not wants_floor:
            return PolicyDecision(False, p_now_robot, p_future_robot, "awaiting_demonstration")
        if self._consecutive_active < self.config.min_consecutive_frames:
            return PolicyDecision(False, p_now_robot, p_future_robot, "debouncing")

        self._consecutive_active = 0
        self._refractory_remaining = self.config.refractory_frames
        return PolicyDecision(True, p_now_robot, p_future_robot, "demonstration_entry")
