from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class FrameFeatures:
    timestamp_s: float
    doa_angle_deg: float | None
    gaze_vector: list[float] | None
    mouth_aspect_ratio: float | None
    audio_vad: bool
    speaker_id: str | None = None

    def to_vector(self) -> np.ndarray:
        doa = 0.0 if self.doa_angle_deg is None else float(self.doa_angle_deg)
        gaze = _normalise_gaze(self.gaze_vector)
        mar = 0.0 if self.mouth_aspect_ratio is None else float(self.mouth_aspect_ratio)
        vad = 1.0 if self.audio_vad else 0.0
        return np.array([doa, *gaze, mar, vad], dtype=float)


def stack_feature_window(frames: list[FrameFeatures]) -> np.ndarray:
    if not frames:
        raise ValueError("frames must not be empty")
    return np.vstack([frame.to_vector() for frame in frames])


def _normalise_gaze(gaze_vector: ArrayLike | None) -> np.ndarray:
    if gaze_vector is None:
        return np.zeros(3, dtype=float)
    gaze = np.asarray(gaze_vector, dtype=float)
    if gaze.shape != (3,):
        raise ValueError("gaze_vector must contain exactly three values")
    norm = np.linalg.norm(gaze)
    if norm == 0:
        return np.zeros(3, dtype=float)
    return gaze / norm
