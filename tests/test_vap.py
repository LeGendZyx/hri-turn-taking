import json

import numpy as np
import pytest
import torch

from hri_turn_taking.data.multiparty import (
    FEATURE_SCHEMA_V3_GAZE_VALIDITY,
    GLOBAL_FEATURE_COLUMNS,
    PARTICIPANT_FEATURE_NAMES,
    SyntheticConfig,
    generate_multiparty_dataset,
    input_feature_columns,
    validate_multiparty_dataframe,
)
from hri_turn_taking.vap.dataset import (
    TRANSFORMED_GLOBAL_FEATURES,
    TRANSFORMED_PARTICIPANT_FEATURES,
    FeatureNormalizer,
    SessionTensors,
    WindowDataset,
    sessions_from_dataframe,
    split_sessions,
    transform_input_features,
)
from hri_turn_taking.vap.events import (
    extract_overlap_events,
    extract_shift_hold_events,
    robot_entry_scores,
    shift_hold_scores,
    shift_score_from_bins,
)
from hri_turn_taking.vap.labels import VAPLabelConfig, future_activity_labels, p_now_p_future
from hri_turn_taking.vap.model import MultipartyVAP, VAPModelConfig
from hri_turn_taking.vap.policy import DemonstrationPolicy, DualThresholdPolicy, PolicyConfig
from hri_turn_taking.vap.trainer import (
    VAPTrainerConfig,
    predict_session_probabilities,
    train_vap_model,
)


def small_config(**overrides) -> SyntheticConfig:
    defaults = dict(n_sessions=3, duration_s=40.0, n_participants=3, random_seed=7)
    defaults.update(overrides)
    return SyntheticConfig(**defaults)


def small_model_config(**overrides) -> VAPModelConfig:
    defaults = dict(
        n_participant_features=len(TRANSFORMED_PARTICIPANT_FEATURES),
        n_global_features=len(TRANSFORMED_GLOBAL_FEATURES),
        n_participants=4,
        n_bins=4,
        d_model=32,
        n_layers=1,
    )
    defaults.update(overrides)
    return VAPModelConfig(**defaults)


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


def test_future_activity_valid_includes_exact_right_endpoint() -> None:
    config = VAPLabelConfig(
        frame_rate_hz=10,
        bin_edges_s=(0.1, 0.2),
        now_bins=(0,),
        future_bins=(1,),
    )
    vad = np.zeros((5, 2))

    _, valid = future_activity_labels(vad, config)

    assert valid.tolist() == [True, True, True, True, False]


@pytest.mark.parametrize("bin_edges_s", [(0.01, 0.08), (0.04, 0.05)])
def test_label_config_rejects_zero_width_quantised_bins(
    bin_edges_s: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="zero-width or repeated future bin"):
        VAPLabelConfig(
            frame_rate_hz=25,
            bin_edges_s=bin_edges_s,
            now_bins=(0,),
            future_bins=(1,),
        )


def test_p_now_p_future_are_duration_weighted() -> None:
    config = VAPLabelConfig()
    probs = np.array([0.8, 0.6, 0.2, 0.0])
    # Bin widths are 0.2/0.4/0.6/0.8 s; the readout is the expected fraction
    # of each interval that contains speech, not a plain mean over bins.
    p_now, p_future = p_now_p_future(probs, config)
    assert p_now == pytest.approx((0.2 * 0.8 + 0.4 * 0.6) / 0.6)
    assert p_future == pytest.approx((0.6 * 0.2 + 0.8 * 0.0) / 1.4)


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


def test_extract_overlap_events_labels_backchannel_vs_competitive() -> None:
    fps = 25
    vad = np.zeros((300, 3))
    vad[0:100, 0] = 1.0  # P1 long turn
    vad[50:60, 1] = 1.0  # P2 backchannel inside the turn -> holder keeps floor
    vad[90:200, 1] = 1.0  # P2 barges in at 90, P1 stops at 100 -> holder yields

    events = sorted(extract_overlap_events(vad, frame_rate_hz=fps), key=lambda e: e.onset_f)

    assert len(events) == 2
    assert (events[0].holder, events[0].challenger, events[0].holder_yields) == (0, 1, False)
    assert (events[1].holder, events[1].challenger, events[1].holder_yields) == (0, 1, True)


def test_synthetic_dataset_schema_and_no_gold_leakage() -> None:
    frame_df = generate_multiparty_dataset(small_config())
    validate_multiparty_dataframe(frame_df)
    inputs = input_feature_columns()
    assert not any(column.endswith("_vad") for column in inputs)
    assert frame_df["P4_present"].eq(0).all()  # 3-party session pads slot 4
    assert frame_df[[c for c in frame_df.columns if c.endswith("_vad")]].isin([0, 1]).all().all()


def test_transform_encodes_doa_wraparound() -> None:
    n_raw = len(input_feature_columns())
    a = np.zeros((1, n_raw), dtype=np.float32)
    b = np.zeros((1, n_raw), dtype=np.float32)
    doa_index = 4 * len(PARTICIPANT_FEATURE_NAMES) + GLOBAL_FEATURE_COLUMNS.index(
        "doa_angle_deg"
    )
    a[0, doa_index] = 179.5
    b[0, doa_index] = -179.5  # physically adjacent direction across the wrap

    ta, tb = transform_input_features(a), transform_input_features(b)

    expected_dim = 4 * len(TRANSFORMED_PARTICIPANT_FEATURES) + len(TRANSFORMED_GLOBAL_FEATURES)
    assert ta.shape == (1, expected_dim)
    global_start = 4 * len(TRANSFORMED_PARTICIPANT_FEATURES)
    assert np.allclose(
        ta[0, global_start : global_start + 2],
        tb[0, global_start : global_start + 2],
        atol=0.02,
    )  # sin/cos nearly identical


def test_model_forward_shapes() -> None:
    model = MultipartyVAP(small_model_config())
    features = torch.randn(2, 50, model.config.input_dim)
    present = torch.ones(2, 4)

    outputs = model(features, present=present)

    assert outputs["bin_logits"].shape == (2, 50, 4, 4)
    assert outputs["vad_logits"].shape == (2, 50, 4)


def test_model_is_causal() -> None:
    torch.manual_seed(0)
    config = small_model_config(n_participant_features=3, n_global_features=2, n_participants=2)
    model = MultipartyVAP(config).eval()
    base = torch.randn(1, 40, config.input_dim)
    perturbed = base.clone()
    perturbed[0, 30:] += 5.0  # only future frames change

    with torch.no_grad():
        out_base = model(base)["bin_logits"][0, :30]
        out_pert = model(perturbed)["bin_logits"][0, :30]

    assert torch.allclose(out_base, out_pert, atol=1e-5)


def test_model_is_permutation_equivariant() -> None:
    torch.manual_seed(0)
    fp, fg, n = 5, 4, 3
    config = small_model_config(
        n_participant_features=fp, n_global_features=fg, n_participants=n, n_layers=2
    )
    model = MultipartyVAP(config).eval()
    features = torch.randn(1, 30, config.input_dim)
    present = torch.ones(1, n)
    perm = [2, 0, 1]

    part = features[..., : n * fp].view(1, 30, n, fp)
    permuted = torch.cat(
        [part[:, :, perm, :].reshape(1, 30, n * fp), features[..., n * fp :]], dim=-1
    )
    with torch.no_grad():
        base = model(features, present=present)["bin_logits"]
        out = model(permuted, present=present)["bin_logits"]

    assert torch.allclose(out, base[:, :, perm, :], atol=1e-5)


def test_absent_slot_cannot_influence_present_participants() -> None:
    torch.manual_seed(0)
    fp, fg, n = 2, 1, 3
    config = small_model_config(
        n_participant_features=fp, n_global_features=fg, n_participants=n, n_layers=2
    )
    model = MultipartyVAP(config).eval()
    present = torch.tensor([[1.0, 1.0, 0.0]])
    base = torch.randn(1, 20, config.input_dim)
    tampered = base.clone()
    tampered[..., 2 * fp : 3 * fp] += 7.0  # only the absent slot's features change

    with torch.no_grad():
        out_base = model(base, present=present)["bin_logits"][:, :, :2]
        out_tamp = model(tampered, present=present)["bin_logits"][:, :, :2]

    assert torch.allclose(out_base, out_tamp, atol=1e-6)


def test_chunked_inference_matches_full_pass() -> None:
    torch.manual_seed(0)
    frame_df = generate_multiparty_dataset(small_config(n_sessions=1, duration_s=60.0))
    session = sessions_from_dataframe(frame_df)[0]
    normalizer = FeatureNormalizer.fit([session])
    model = MultipartyVAP(small_model_config(n_layers=2, attention_window=100)).eval()

    full = predict_session_probabilities(model, session, normalizer, chunk_frames=5000)
    chunked = predict_session_probabilities(model, session, normalizer, chunk_frames=400)

    assert full.shape == (1500, 4, 4)
    assert np.allclose(full, chunked, atol=1e-5)


def test_gru_long_inference_forces_full_pass_and_matches_full() -> None:
    # A GRU carries unbounded recurrent state; a finite overlap cannot make
    # independently reset chunks exact. Until model.forward exposes h_n, the
    # inference helper must ignore chunking for GRU and run one full pass.
    torch.manual_seed(3)
    frame_df = generate_multiparty_dataset(
        small_config(n_sessions=1, duration_s=8.0, random_seed=13)
    )
    session = sessions_from_dataframe(frame_df)[0]
    normalizer = FeatureNormalizer.fit([session])
    model = MultipartyVAP(
        small_model_config(
            encoder="gru",
            d_model=8,
            n_layers=2,
            attention_window=1,
        )
    ).eval()

    full = predict_session_probabilities(model, session, normalizer, chunk_frames=5000)
    requested_chunk = predict_session_probabilities(
        model, session, normalizer, chunk_frames=40
    )

    assert full.shape == (200, 4, 4)
    assert np.array_equal(full, requested_chunk)


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
    expected_dim = 4 * len(TRANSFORMED_PARTICIPANT_FEATURES) + len(TRANSFORMED_GLOBAL_FEATURES)
    assert sample["features"].shape == (100, expected_dim)
    assert sample["labels"].shape == (100, 4, 4)
    assert sample["present"].shape == (4,)


def test_voiced_disambiguates_pitch_and_is_transformed_without_leakage() -> None:
    frame_df = generate_multiparty_dataset(small_config())
    assert "voiced" in GLOBAL_FEATURE_COLUMNS
    assert set(frame_df["voiced"].unique()) <= {0.0, 1.0}
    assert (frame_df.loc[frame_df["voiced"] == 0.0, "f0_proxy"] == 0.0).all()

    # The synthetic detector deliberately has both false positives and false
    # negatives, so it is not a deterministic encoding of gold any-speech.
    any_gold = frame_df[[f"P{i}_vad" for i in range(1, 5)]].max(axis=1).to_numpy()
    voiced = frame_df["voiced"].to_numpy()
    assert ((any_gold == 0.0) & (voiced == 1.0)).any()
    assert ((any_gold == 1.0) & (voiced == 0.0)).any()

    raw = frame_df[input_feature_columns()].to_numpy(dtype=np.float32)
    transformed = transform_input_features(raw)
    assert transformed.shape[1] == 4 * len(PARTICIPANT_FEATURE_NAMES) + 5
    assert np.array_equal(transformed[:, -1], voiced.astype(np.float32))


@pytest.mark.parametrize(
    ("voiced_value", "f0_value", "message"),
    [(0.5, 0.0, "voiced must be binary"), (0.0, 0.1, "f0_proxy must be exactly zero")],
)
def test_schema_rejects_invalid_voiced_pitch_pair(
    voiced_value: float, f0_value: float, message: str
) -> None:
    frame_df = generate_multiparty_dataset(small_config())
    frame_df.loc[0, "voiced"] = voiced_value
    frame_df.loc[0, "f0_proxy"] = f0_value
    with pytest.raises(ValueError, match=message):
        validate_multiparty_dataframe(frame_df)


def test_normalizer_shares_stats_across_slots() -> None:
    frame_df = generate_multiparty_dataset(small_config())
    sessions = sessions_from_dataframe(frame_df)
    normalizer = FeatureNormalizer.fit(sessions)

    # One set of statistics per participant *feature*, shared by every slot —
    # required for slot permutation equivariance and 3-/4-party transfer.
    assert normalizer.participant_mean.shape == (len(TRANSFORMED_PARTICIPANT_FEATURES),)
    assert normalizer.global_mean.shape == (len(TRANSFORMED_GLOBAL_FEATURES),)
    applied = normalizer.apply(sessions[0].features)
    assert applied.shape == sessions[0].features.shape


def test_split_sessions_group_level_keeps_groups_together() -> None:
    def stub(session_id: str) -> SessionTensors:
        return SessionTensors(
            session_id=session_id,
            features=np.zeros((10, 24), dtype=np.float32),
            labels=np.zeros((10, 4, 4), dtype=np.float32),
            vad=np.zeros((10, 4), dtype=np.float32),
            valid=np.ones(10, dtype=bool),
            present=np.ones(4, dtype=np.float32),
        )

    # 3 recording groups x 2 scenario videos each: a per-video split could put
    # the same speakers on both sides; the group-level split must not.
    sessions = [stub(f"g{g}__s{s}") for g in range(3) for s in range(2)]
    for seed in range(5):
        train, val = split_sessions(
            sessions, validation_fraction=0.34, random_seed=seed, group_delimiter="__"
        )
        assert len(train) + len(val) == len(sessions) and train and val
        train_groups = {s.session_id.split("__")[0] for s in train}
        val_groups = {s.session_id.split("__")[0] for s in val}
        assert not (train_groups & val_groups)


def test_robot_entry_scores_label_gaps_the_demonstrator_takes() -> None:
    fps = 25
    robot = 2
    vad = np.zeros((320, 3))
    vad[0:50, 0] = 1.0  # P1 turn
    vad[63:113, 1] = 1.0  # gap, then P2: human takes the floor -> label 0
    vad[126:176, 2] = 1.0  # gap, then the demonstrator enters -> label 1
    vad[189:239, 2] = 1.0  # demonstrator resumes after own pause -> excluded
    vad[252:302, 0] = 1.0  # gap following the robot's turn -> excluded

    probs = np.full((320, 3, 4), 0.05)
    probs[113:126, robot, :] = 0.9  # the model foresees the robot entry in gap 2

    labels, scores = robot_entry_scores(probs, vad, robot_slot=robot, frame_rate_hz=fps)

    assert labels == [0, 1]
    assert scores[1] > scores[0]


def test_gap_scores_require_complete_future_horizon_when_valid_is_supplied() -> None:
    n_frames = 160
    vad = np.zeros((n_frames, 3), dtype=float)
    vad[0:30, 2] = 1.0
    vad[35:90, 0] = 1.0
    vad[80:120, 2] = 1.0
    vad[125:150, 0] = 1.0
    probabilities = np.full((n_frames, 3, 4), 0.5, dtype=float)
    valid = np.arange(n_frames) + 50 <= n_frames

    unfiltered_shift, _ = shift_hold_scores(probabilities, vad)
    filtered_shift, _ = shift_hold_scores(probabilities, vad, valid=valid)
    unfiltered_robot, _ = robot_entry_scores(probabilities, vad, robot_slot=0)
    filtered_robot, _ = robot_entry_scores(
        probabilities,
        vad,
        robot_slot=0,
        valid=valid,
    )

    assert unfiltered_shift == [1, 1]
    assert filtered_shift == [1]
    assert unfiltered_robot == [1, 1]
    assert filtered_robot == [1]
    with pytest.raises(ValueError, match=r"valid must have shape \(n_frames,\)"):
        shift_hold_scores(probabilities, vad, valid=valid[:-1])


def test_demonstration_policy_triggers_on_robot_slot_prediction() -> None:
    label_config = VAPLabelConfig()
    policy = DemonstrationPolicy(
        PolicyConfig(
            now_threshold=0.5, future_threshold=0.5, min_consecutive_frames=2, refractory_frames=3
        ),
        label_config,
    )
    robot_speaks = np.full((3, 4), 0.05)
    robot_speaks[2] = 0.9
    other_speaks = np.full((3, 4), 0.05)
    other_speaks[0] = 0.9

    blocked = policy.update(other_speaks, robot_slot=2, robot_has_intent=True)
    assert not blocked.should_speak and blocked.reason == "awaiting_demonstration"
    first = policy.update(robot_speaks, robot_slot=2, robot_has_intent=True)
    assert not first.should_speak and first.reason == "debouncing"
    second = policy.update(robot_speaks, robot_slot=2, robot_has_intent=True)
    assert second.should_speak and second.reason == "demonstration_entry"
    third = policy.update(robot_speaks, robot_slot=2, robot_has_intent=True)
    assert not third.should_speak and third.reason == "refractory"

    fresh = DemonstrationPolicy(PolicyConfig(min_consecutive_frames=1), label_config)
    absent = fresh.update(
        robot_speaks, robot_slot=2, robot_has_intent=True, present=np.array([1.0, 1.0, 0.0])
    )
    assert not absent.should_speak and absent.reason == "robot_slot_absent"


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
            aux_vad_weight=0.0,
            feature_set="no_mar",
        ),
        encoder="gru",
        d_model=16,
        n_layers=1,
        run_metadata={
            "requested_overrides": ["allow_untrainable"],
            "effective_smoke_reasons": ["test fixture uses a readiness override"],
        },
        data_manifest={"data_kind": "synthetic", "source_artifacts": []},
    )

    assert checkpoint.exists()
    assert 0.0 <= result.frame_metrics["mAP"] <= 1.0
    assert result.n_train_sessions == 3 and result.n_val_sessions == 1
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train_vap_model(frame_df, checkpoint_path=checkpoint)

    # Checkpoint format v4 round-trip: load must reproduce a usable model and
    # retain development/smoke/source status in both paired artifacts.
    from hri_turn_taking.vap.trainer import load_vap_checkpoint

    loaded = load_vap_checkpoint(checkpoint)
    assert loaded.model.config.n_participant_features == len(PARTICIPANT_FEATURE_NAMES)

    # a checkpoint from the previous feature schema must fail LOADING with a
    # clear retrain message, not later with a normalizer broadcast error
    # (Codex R1.2-1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["trainer_config"]["feature_set"] == "no_mar"
    assert payload["trainer_config"]["aux_vad_weight"] == 0.0
    assert payload["feature_meta"]["feature_set"] == "no_mar"
    assert "P1_mar" not in payload["feature_meta"]["active_columns"]
    assert payload["run_metadata"]["artifact_status"] == "smoke_only"
    assert payload["run_metadata"]["counted_experiment_eligible"] is False
    assert payload["data_manifest"]["data_kind"] == "synthetic"
    assert payload["data_split"]["reported_partition"] == "inner_validation"
    assert payload["data_split"]["outer_test_session_ids"] == []
    metrics = json.loads(checkpoint.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_metadata"] == payload["run_metadata"]
    assert metrics["counted_experiment_eligible"] is False
    assert "NaN" not in checkpoint.with_suffix(".metrics.json").read_text(encoding="utf-8")
    assert set(payload["data_split"]["train_session_ids"]).isdisjoint(
        payload["data_split"]["val_session_ids"]
    )
    assert loaded.data_split == payload["data_split"]
    stale = dict(payload)
    stale["format_version"] = 2
    old_path = tmp_path / "vap_v2.pt"
    torch.save(stale, old_path)
    with pytest.raises(ValueError, match="retrain"):
        load_vap_checkpoint(old_path)

    wrong_meta = dict(payload)
    wrong_meta["feature_meta"] = {
        "participant_features": ["asd_prob", "mar", "gaze_robot", "head_yaw_deg", "gazed_at"],
        "global_features": payload["feature_meta"]["global_features"],
    }
    meta_path = tmp_path / "vap_stale_meta.pt"
    torch.save(wrong_meta, meta_path)
    with pytest.raises(ValueError, match="retrain"):
        load_vap_checkpoint(meta_path)

    # Disguised current-version payloads: correct feature names wrapping the
    # wrong raw columns, model feature counts or normalizer statistics must
    # fail at load, not later with a broadcast error / non-finite output
    import copy

    disguised_v3_normalizer = copy.deepcopy(payload)
    # Keep the historical seven/five statistic lengths so this specifically
    # tests the schema declaration rather than falling through to shape checks.
    disguised_v3_normalizer["normalizer"]["feature_schema"] = (
        FEATURE_SCHEMA_V3_GAZE_VALIDITY
    )
    torch.save(disguised_v3_normalizer, tmp_path / "vap_v3_normalizer.pt")
    with pytest.raises(ValueError, match="historical.*multiparty_v2"):
        load_vap_checkpoint(tmp_path / "vap_v3_normalizer.pt")

    bad_columns = copy.deepcopy(payload)
    bad_columns["feature_meta"]["raw_columns"] = ["wrong"]
    torch.save(bad_columns, tmp_path / "vap_cols.pt")
    with pytest.raises(ValueError, match="retrain"):
        load_vap_checkpoint(tmp_path / "vap_cols.pt")

    bad_counts = copy.deepcopy(payload)
    bad_counts["model_config"]["n_participant_features"] = 5
    torch.save(bad_counts, tmp_path / "vap_counts.pt")
    with pytest.raises(ValueError, match="retrain"):
        load_vap_checkpoint(tmp_path / "vap_counts.pt")

    bad_norm = copy.deepcopy(payload)
    bad_norm["normalizer"]["global_mean"] = [0.0, 0.0, 0.0]
    torch.save(bad_norm, tmp_path / "vap_norm.pt")
    with pytest.raises(ValueError, match="retrain"):
        load_vap_checkpoint(tmp_path / "vap_norm.pt")

    bad_std = copy.deepcopy(payload)
    bad_std["normalizer"]["participant_std"][0] = 0.0
    torch.save(bad_std, tmp_path / "vap_std.pt")
    with pytest.raises(ValueError, match="retrain"):
        load_vap_checkpoint(tmp_path / "vap_std.pt")

    # cross-field disguises (Codex R1.4-4): a self-consistent 3-slot model or a
    # model/label bin-count mismatch must fail at load, not at inference
    bad_slots = copy.deepcopy(payload)
    bad_slots["model_config"]["n_participants"] = 3
    torch.save(bad_slots, tmp_path / "vap_slots.pt")
    with pytest.raises(ValueError, match="retrain"):
        load_vap_checkpoint(tmp_path / "vap_slots.pt")

    bad_bins = copy.deepcopy(payload)
    bad_bins["model_config"]["n_bins"] = 3
    torch.save(bad_bins, tmp_path / "vap_bins.pt")
    with pytest.raises(ValueError, match="retrain"):
        load_vap_checkpoint(tmp_path / "vap_bins.pt")
