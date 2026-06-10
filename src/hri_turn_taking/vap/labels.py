"""Future voice-activity bin labels (VAP objective).

Following Ekstedt & Skantze, the prediction target at frame t is, for every
participant, whether they will be speaking within each of a set of future
horizon bins. The default bin edges (0.2, 0.6, 1.2, 2.0 s) match the original
VAP formulation. A bin is positive when the participant is active for at least
``active_ratio`` of the frames inside that bin.

Labels are derived purely from gold-tier per-participant VAD; they must never
be fed to the model as input features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

DEFAULT_BIN_EDGES_S = (0.2, 0.6, 1.2, 2.0)


@dataclass(frozen=True)
class VAPLabelConfig:
    frame_rate_hz: int = 25
    bin_edges_s: tuple[float, ...] = DEFAULT_BIN_EDGES_S
    active_ratio: float = 0.5
    # Bins whose mean probability forms p_now / p_future (Skantze & Irfan 2025).
    now_bins: tuple[int, ...] = (0, 1)
    future_bins: tuple[int, ...] = (2, 3)

    @property
    def n_bins(self) -> int:
        return len(self.bin_edges_s)

    @property
    def horizon_frames(self) -> int:
        return round(self.bin_edges_s[-1] * self.frame_rate_hz)

    def bin_frame_ranges(self) -> list[tuple[int, int]]:
        """Half-open frame ranges (start, end) for each future bin."""
        edges = [0.0, *self.bin_edges_s]
        ranges = []
        for left, right in zip(edges[:-1], edges[1:], strict=False):
            ranges.append((round(left * self.frame_rate_hz), round(right * self.frame_rate_hz)))
        return ranges


DEFAULT_LABEL_CONFIG = VAPLabelConfig()


def future_activity_labels(
    vad: ArrayLike,
    config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute VAP bin labels from a (T, N) per-participant VAD matrix.

    Returns
    -------
    labels: (T, N, n_bins) float array of {0, 1}
    valid:  (T,) bool array — False for trailing frames whose full future
            horizon extends past the end of the session.
    """
    activity = np.asarray(vad, dtype=float)
    if activity.ndim != 2:
        raise ValueError("vad must have shape (n_frames, n_participants)")
    n_frames, n_participants = activity.shape

    ranges = config.bin_frame_ranges()
    labels = np.zeros((n_frames, n_participants, config.n_bins), dtype=float)

    # Prefix sums let every (frame, bin) query run in O(1).
    cumulative = np.concatenate(
        [np.zeros((1, n_participants)), np.cumsum(activity, axis=0)], axis=0
    )
    for bin_idx, (start, end) in enumerate(ranges):
        width = max(1, end - start)
        for t in range(n_frames):
            lo = min(n_frames, t + start)
            hi = min(n_frames, t + end)
            if hi <= lo:
                continue
            mean_active = (cumulative[hi] - cumulative[lo]) / width
            labels[t, :, bin_idx] = (mean_active >= config.active_ratio).astype(float)

    valid = np.ones(n_frames, dtype=bool)
    if config.horizon_frames > 0:
        valid[max(0, n_frames - config.horizon_frames) :] = False
    return labels, valid


def p_now_p_future(
    bin_probabilities: ArrayLike,
    config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-bin probabilities (..., n_bins) into p_now and p_future."""
    probs = np.asarray(bin_probabilities, dtype=float)
    if probs.shape[-1] != config.n_bins:
        raise ValueError(f"expected last dimension of size {config.n_bins}")
    p_now = probs[..., list(config.now_bins)].mean(axis=-1)
    p_future = probs[..., list(config.future_bins)].mean(axis=-1)
    return p_now, p_future
