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

import math
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

    def __post_init__(self) -> None:
        """Reject malformed label semantics before training or artifact loading."""
        if isinstance(self.frame_rate_hz, bool) or self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be a positive integer")
        if not self.bin_edges_s:
            raise ValueError("bin_edges_s must contain at least one future-bin edge")
        if any(not math.isfinite(edge) or edge <= 0 for edge in self.bin_edges_s):
            raise ValueError("bin_edges_s must contain finite positive values")
        if any(
            right <= left
            for left, right in zip(self.bin_edges_s, self.bin_edges_s[1:], strict=False)
        ):
            raise ValueError("bin_edges_s must be strictly increasing")
        frame_endpoints = tuple(
            round(edge * self.frame_rate_hz) for edge in self.bin_edges_s
        )
        if any(
            right <= left
            for left, right in zip((0, *frame_endpoints[:-1]), frame_endpoints, strict=True)
        ):
            raise ValueError(
                "bin_edges_s must map to strictly increasing positive frame endpoints; "
                "the configured frame rate creates a zero-width or repeated future bin"
            )
        if not math.isfinite(self.active_ratio) or not 0 < self.active_ratio <= 1:
            raise ValueError("active_ratio must be finite and in (0, 1]")
        for name, indices in (("now_bins", self.now_bins), ("future_bins", self.future_bins)):
            if not indices:
                raise ValueError(f"{name} must contain at least one bin index")
            if len(set(indices)) != len(indices):
                raise ValueError(f"{name} must not contain duplicate bin indices")
            if any(
                isinstance(index, bool) or index < 0 or index >= len(self.bin_edges_s)
                for index in indices
            ):
                raise ValueError(
                    f"{name} indices must be integers in [0, {len(self.bin_edges_s) - 1}]"
                )

    @property
    def n_bins(self) -> int:
        return len(self.bin_edges_s)

    @property
    def horizon_frames(self) -> int:
        return round(self.bin_edges_s[-1] * self.frame_rate_hz)

    @property
    def bin_widths_s(self) -> tuple[float, ...]:
        edges = (0.0, *self.bin_edges_s)
        return tuple(right - left for left, right in zip(edges[:-1], edges[1:], strict=False))

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

    # A half-open future interval [t, t + horizon) is fully contained when its
    # right endpoint equals the recording length; equality must remain valid.
    valid = np.arange(n_frames) + config.horizon_frames <= n_frames
    return labels, valid


def p_now_p_future(
    bin_probabilities: ArrayLike,
    config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-bin probabilities (..., n_bins) into p_now and p_future.

    Bins are averaged weighted by their duration, so each readout is the
    expected fraction of its interval (~0-0.6 s / ~0.6-2 s) that contains
    speech; a plain mean would over-weight the short near bins.
    """
    probs = np.asarray(bin_probabilities, dtype=float)
    if probs.shape[-1] != config.n_bins:
        raise ValueError(f"expected last dimension of size {config.n_bins}")
    widths = np.asarray(config.bin_widths_s, dtype=float)

    def weighted(bin_indices: tuple[int, ...]) -> np.ndarray:
        idx = list(bin_indices)
        return (probs[..., idx] * widths[idx]).sum(axis=-1) / widths[idx].sum()

    return weighted(config.now_bins), weighted(config.future_bins)
