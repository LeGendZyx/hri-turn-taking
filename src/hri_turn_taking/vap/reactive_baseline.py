"""Reactive silence baseline for the predictive-vs-reactive policy comparison.

This is core evaluation infrastructure (the baseline the VAP policy is measured
against), not part of the frozen legacy TRP ablation. Relocated here from
``legacy/models/baseline.py`` so the ``vap`` package has no legacy dependency.
"""

from collections import deque


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
