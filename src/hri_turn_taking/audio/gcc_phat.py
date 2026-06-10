import math

import numpy as np
from numpy.typing import ArrayLike


def estimate_tdoa_gcc_phat(
    signal: ArrayLike,
    reference: ArrayLike,
    *,
    sample_rate: int,
    max_tau: float | None = None,
    interp: int = 16,
    eps: float = 1e-12,
) -> float:
    """Estimate time delay between two signals using GCC-PHAT."""
    sig = np.asarray(signal, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if sig.ndim != 1 or ref.ndim != 1:
        raise ValueError("signal and reference must be one-dimensional arrays")
    if sig.size == 0 or ref.size == 0:
        raise ValueError("signal and reference must not be empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if interp < 1:
        raise ValueError("interp must be >= 1")

    n = sig.size + ref.size
    sig_fft = np.fft.rfft(sig, n=n)
    ref_fft = np.fft.rfft(ref, n=n)
    cross_power = sig_fft * np.conj(ref_fft)
    weighted = cross_power / (np.abs(cross_power) + eps)
    cc = np.fft.irfft(weighted, n=interp * n)

    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(max_shift, int(interp * sample_rate * max_tau))

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc)) - max_shift)
    return shift / float(interp * sample_rate)


def doa_from_tdoa(
    tdoa_s: float,
    *,
    mic_distance_m: float,
    speed_of_sound_mps: float = 343.0,
) -> float:
    """Convert TDOA to a horizontal Direction of Arrival angle in degrees."""
    if mic_distance_m <= 0:
        raise ValueError("mic_distance_m must be positive")
    if speed_of_sound_mps <= 0:
        raise ValueError("speed_of_sound_mps must be positive")

    ratio = (tdoa_s * speed_of_sound_mps) / mic_distance_m
    clipped = float(np.clip(ratio, -1.0, 1.0))
    return math.degrees(math.asin(clipped))
