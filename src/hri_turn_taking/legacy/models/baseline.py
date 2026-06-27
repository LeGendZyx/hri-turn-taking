from hri_turn_taking.legacy.features.fusion import FrameFeatures
from hri_turn_taking.legacy.vision.mouth import is_visual_speaking


class HeuristicTRPModel:
    """Small deterministic model used before supervised training data exists."""

    def __init__(self, mar_threshold: float = 0.20) -> None:
        self.mar_threshold = mar_threshold

    def predict_proba(self, frame: FrameFeatures) -> float:
        visual_speaking = (
            frame.mouth_aspect_ratio is not None
            and is_visual_speaking(frame.mouth_aspect_ratio, self.mar_threshold)
        )

        if frame.audio_vad and visual_speaking:
            return 0.25
        if frame.audio_vad and not visual_speaking:
            return 0.55
        if not frame.audio_vad and visual_speaking:
            return 0.35
        return 0.80
