import numpy as np
import pytest

from hri_turn_taking.features.fusion import FrameFeatures, stack_feature_window
from hri_turn_taking.realtime.pipeline import PredictiveTurnTakingPipeline
from hri_turn_taking.vision.mouth import mouth_aspect_ratio


def test_mouth_aspect_ratio_from_landmarks() -> None:
    landmarks = np.zeros((309, 3), dtype=float)
    landmarks[13] = [0.0, 0.04, 0.0]
    landmarks[14] = [0.0, -0.04, 0.0]
    landmarks[37] = [-0.02, 0.02, 0.0]
    landmarks[84] = [-0.02, -0.02, 0.0]
    landmarks[78] = [-0.08, 0.0, 0.0]
    landmarks[308] = [0.08, 0.0, 0.0]

    assert mouth_aspect_ratio(landmarks) == pytest.approx(0.375)


def test_frame_features_stack_to_expected_shape() -> None:
    frames = [
        FrameFeatures(0.0, 10.0, [0.0, 0.0, 2.0], 0.1, False),
        FrameFeatures(0.1, None, None, None, True),
    ]

    matrix = stack_feature_window(frames)

    assert matrix.shape == (2, 6)
    assert np.allclose(matrix[0, 1:4], [0.0, 0.0, 1.0])


def test_pipeline_speaks_when_trp_probability_crosses_threshold() -> None:
    frame = FrameFeatures(
        timestamp_s=0.0,
        doa_angle_deg=0.0,
        gaze_vector=[0.0, 0.0, 1.0],
        mouth_aspect_ratio=0.0,
        audio_vad=False,
    )
    pipeline = PredictiveTurnTakingPipeline(trp_threshold=0.65)

    decision = pipeline.step(frame, robot_has_intent=True)

    assert decision.should_speak
    assert decision.reason == "threshold_met"
