import numpy as np

from hri_turn_taking.audio.gcc_phat import doa_from_tdoa, estimate_tdoa_gcc_phat


def test_estimate_tdoa_gcc_phat_for_shifted_impulse() -> None:
    sample_rate = 16000
    delay_samples = 5
    reference = np.zeros(512)
    signal = np.zeros(512)
    reference[128] = 1.0
    signal[128 + delay_samples] = 1.0

    tdoa = estimate_tdoa_gcc_phat(signal, reference, sample_rate=sample_rate, interp=1)

    assert abs(tdoa - delay_samples / sample_rate) < 1e-9


def test_doa_from_tdoa_clips_to_valid_angle() -> None:
    angle = doa_from_tdoa(1.0, mic_distance_m=0.06)

    assert angle == 90.0
