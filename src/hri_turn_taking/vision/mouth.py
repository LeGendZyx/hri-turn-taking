from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class MouthLandmarkIndices:
    upper_lip: int = 13
    lower_lip: int = 14
    left_inner_upper: int = 37
    left_inner_lower: int = 84
    left_corner: int = 78
    right_corner: int = 308


DEFAULT_MOUTH_INDICES = MouthLandmarkIndices()


def mouth_aspect_ratio(
    landmarks: ArrayLike,
    indices: MouthLandmarkIndices = DEFAULT_MOUTH_INDICES,
    eps: float = 1e-12,
) -> float:
    """Compute Mouth Aspect Ratio from MediaPipe-style face landmarks."""
    points = np.asarray(landmarks, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("landmarks must have shape (n_points, n_dimensions)")

    required = max(
        indices.upper_lip,
        indices.lower_lip,
        indices.left_inner_upper,
        indices.left_inner_lower,
        indices.left_corner,
        indices.right_corner,
    )
    if points.shape[0] <= required:
        raise ValueError(f"landmarks must contain index {required}")

    vertical_main = np.linalg.norm(points[indices.upper_lip] - points[indices.lower_lip])
    vertical_inner = np.linalg.norm(
        points[indices.left_inner_upper] - points[indices.left_inner_lower]
    )
    horizontal = np.linalg.norm(points[indices.left_corner] - points[indices.right_corner])
    return float((vertical_main + vertical_inner) / (2.0 * horizontal + eps))


def is_visual_speaking(mar: float, threshold: float = 0.20) -> bool:
    """Return True when mouth motion is above the visual speech threshold."""
    return mar >= threshold
