from dataclasses import dataclass

from hri_turn_taking.legacy.features.fusion import FrameFeatures
from hri_turn_taking.legacy.models.baseline import HeuristicTRPModel


@dataclass(frozen=True)
class TurnTakingDecision:
    trp_probability: float
    should_speak: bool
    reason: str


class PredictiveTurnTakingPipeline:
    """Minimal online decision layer that can be replaced by a trained model."""

    def __init__(self, trp_threshold: float = 0.65, mar_threshold: float = 0.20) -> None:
        if not 0.0 <= trp_threshold <= 1.0:
            raise ValueError("trp_threshold must be within [0, 1]")
        self.trp_threshold = trp_threshold
        self.model = HeuristicTRPModel(mar_threshold=mar_threshold)

    def step(self, frame: FrameFeatures, *, robot_has_intent: bool) -> TurnTakingDecision:
        probability = self.model.predict_proba(frame)
        should_speak = robot_has_intent and probability >= self.trp_threshold
        reason = "threshold_met" if should_speak else "waiting"
        if not robot_has_intent:
            reason = "no_robot_intent"
        return TurnTakingDecision(
            trp_probability=probability,
            should_speak=should_speak,
            reason=reason,
        )
