import numpy as np
import pytest
import torch

from hri_turn_taking.data.multiparty import (
    SyntheticConfig,
    generate_multiparty_dataset,
    input_feature_columns,
    validate_multiparty_dataframe,
)
from hri_turn_taking.vap.dataset import (
    FeatureNormalizer,
    WindowDataset,
    sessions_from_dataframe,
    split_sessions,
)
from hri_turn_taking.vap.events import extract_shift_hold_events, shift_score_from_bins
from hri_turn_taking.vap.labels import VAPLabelConfig, future_activity_labels, p_now_p_future
from hri_turn_taking.vap.model import MultipartyVAP, VAPModelConfig
from hri_turn_taking.vap.policy import DualThresholdPolicy, PolicyConfig
from hri_turn_taking.vap.trainer import VAPTrainerConfig, train_vap_model


def small_config(**overrides) -> SyntheticConfig:
    defaults = dict(n_sessions=3, duration_s=40.0, n_participants=3, random_seed=7)
    defaults.update(overrides)
    return SyntheticConfig(**defaults)


def test_future_activity_labels_mark_upcoming_speech() -> None:
    config = VAPLabelConfig(frame_rate_hz=25)
    vad = np.zeros((200, 2))
    vad[100:150, 0] = 1.0  # P1 speaks from frame 100 (t=4.0s) to 150

    labels, valid = future_activity_labels(vad, config)

    # At frame 95 (0.2s before onset), P1 active throughout bins 2/3 horizons.
    assert labels[95, 0, 2] == 1.0
    # At frame 40, P1 onset is 2.4s away — beyond every bin.
    assert labels[40, 0].sum() == 0.0
    # While P1 speaks mid-turn, near bins are positive.
    assert labels[110, 0, 0] == 1.0
    # P2 never speaks.
    assert labels[:, 1, :].sum() == 0.0
    # Trailing horizon frames are flagged invalid.
    assert not valid[-1] and valid[0]


def test_p_now_p_future_split_bins() -> None:
    config = VAPLabelConfig()
    probs = np.array([0.8, 0.6, 0.2, 0.0])
    p_now, p_future = p_now_p_future(probs, config)
    assert p_now == pytest.approx(0.7)
    assert p_future == pytest.approx(0.1)


def test_extract_shift_hold_events_detects_shift_and_hold() -> None:
    fps = 25
    vad = np.zeros((400, 2))
    vad[0:100, 0] = 1.0  # P1 turn
    vad[110:200, 1] = 1.0  # shift to P2 after 0.4s gap
    vad[215:300, 1] = 1.0  # P2 holds across a 0.6s pause

    events = extract_shift_hold_events(vad, frame_rate_hz=fps)

    assert len(events) == 2
    assert events[0].is_shift
    assert (events[0].previous_speaker, events[0].next_speaker) == (0, 1)
    assert not events[1].is_shift


def test_shift_score_prefers_challenger() -> None:
    bins = np.array(
        [
            [0.1, 0.1, 0.05, 0.05],  # previous speaker fading out
            [0.2, 0.4, 0.8, 0.9],  # challenger about to speak
        ]
    )
    score = shift_score_from_bins(bins, previous_speaker=0)
    assert score > 0.8


def test_synthetic_dataset_schema_and_no_gold_leakage() -> None:
    frame_df = generate_multiparty_dataset(small_config())
    validate_multiparty_dataframe(frame_df)
    inputs = input_feature_columns()
    assert not any(column.endswith("_vad") for column in inputs)
    assert frame_df["P4_present"].eq(0).all()  # 3-party session pads slot 4
    assert frame_df[[c for c in frame_df.columns if c.endswith("_vad")]].isin([0, 1]).all().all()


def test_model_forward_shapes() -> None:
    config = VAPModelConfig(input_dim=19, n_participants=4, n_bins=4, d_model=32, n_layers=1)
    model = MultipartyVAP(config)
    features = torch.randn(2, 50, 19)

    outputs = model(features)

    assert outputs["bin_logits"].shape == (2, 50, 4, 4)
    assert outputs["vad_logits"].shape == (2, 50, 4)


def test_model_is_causal() -> None:
    torch.manual_seed(0)
    config = VAPModelConfig(input_dim=8, n_participants=2, n_bins=4, d_model=32, n_layers=1)
    model = MultipartyVAP(config).eval()
    base = torch.randn(1, 40, 8)
    perturbed = base.clone()
    perturbed[0, 30:] += 5.0  # only future frames change

    with torch.no_grad():
        out_base = model(base)["bin_logits"][0, :30]
        out_pert = model(perturbed)["bin_logits"][0, :30]

    assert torch.allclose(out_base, out_pert, atol=1e-5)


def test_policy_debounce_and_refractory() -> None:
    label_config = VAPLabelConfig()
    policy = DualThresholdPolicy(
        PolicyConfig(min_consecutive_frames=2, refractory_frames=3), label_config
    )
    quiet = np.full((2, 4), 0.05)

    first = policy.update(quiet, robot_has_intent=True)
    assert not first.should_speak and first.reason == "debouncing"
    second = policy.update(quiet, robot_has_intent=True)
    assert second.should_speak and second.reason == "floor_open"
    third = policy.update(quiet, robot_has_intent=True)
    assert not third.should_speak and third.reason == "refractory"


def test_policy_blocks_when_humans_hold_floor() -> None:
    policy = DualThresholdPolicy(PolicyConfig(min_consecutive_frames=1))
    busy = np.full((2, 4), 0.9)
    decision = policy.update(busy, robot_has_intent=True)
    assert not decision.should_speak and decision.reason == "humans_hold_floor"


def test_window_dataset_and_split() -> None:
    frame_df = generate_multiparty_dataset(small_config())
    sessions = sessions_from_dataframe(frame_df)
    train, val = split_sessions(sessions, validation_fraction=0.34, random_seed=0)
    assert len(train) + len(val) == len(sessions) and len(val) >= 1

    normalizer = FeatureNormalizer.fit(train)
    dataset = WindowDataset(train, normalizer, window_frames=100, stride_frames=100)
    sample = dataset[0]
    assert sample["features"].shape == (100, len(input_feature_columns()))
    assert sample["labels"].shape == (100, 4, 4)
    assert sample["present"].shape == (4,)


def test_train_vap_model_smoke(tmp_path) -> None:
    frame_df = generate_multiparty_dataset(small_config(n_sessions=4, duration_s=60.0))
    checkpoint = tmp_path / "vap.pt"

    result = train_vap_model(
        frame_df,
        checkpoint_path=checkpoint,
        trainer_config=VAPTrainerConfig(
            epochs=2,
            batch_size=8,
            window_frames=100,
            stride_frames=100,
            validation_fraction=0.25,
            random_seed=0,
        ),
        encoder="gru",
        d_model=16,
        n_layers=1,
    )

    assert checkpoint.exists()
    assert 0.0 <= result.frame_metrics["mAP"] <= 1.0
    assert result.n_train_sessions == 3 and result.n_val_sessions == 1
