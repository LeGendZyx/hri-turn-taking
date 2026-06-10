from collections import deque

from hri_turn_taking.features.fusion import FrameFeatures
from hri_turn_taking.vision.mouth import is_visual_speaking


class SilenceBaseline:
    """Reactive baseline that fires after a fixed duration of silence."""

    def __init__(self, silence_seconds: float, frame_rate_hz: int) -> None:
        if silence_seconds <= 0:
            raise ValueError("silence_seconds must be positive")
        if frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")
        self.required_silent_frames = max(1, round(silence_seconds * frame_rate_hz))
        self._recent_silence: deque[bool] = deque(maxlen=self.required_silent_frames)

    def update(self, audio_vad: bool) -> bool:
        self._recent_silence.append(not audio_vad)
        return (
            len(self._recent_silence) == self.required_silent_frames
            and all(self._recent_silence)
        )


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
