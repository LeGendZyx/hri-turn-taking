import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from hri_turn_taking.data.multiparty import (
    GLOBAL_FEATURE_COLUMNS,
    SyntheticConfig,
    generate_multiparty_dataset,
    input_feature_columns,
    participant_column,
)
from hri_turn_taking.vap.baselines import (
    ASDPersistenceBaseline,
    LastSpeakerBaseline,
    asd_persistence_probabilities,
    last_speaker_probabilities,
)
from hri_turn_taking.vap.dataset import (
    TRANSFORMED_GLOBAL_FEATURES,
    TRANSFORMED_PARTICIPANT_FEATURES,
    SessionTensors,
    sessions_from_dataframe,
)
from hri_turn_taking.vap.events import (
    evaluate_shift_hold,
    evaluate_shift_prediction,
    next_speaker_scores,
    overlap_resolution_scores,
    score_next_speaker,
    shift_prediction_scores,
)
from hri_turn_taking.vap.feature_sets import (
    FEATURE_SETS,
    active_feature_columns,
    apply_feature_set,
    feature_set_metadata,
)
from hri_turn_taking.vap.labels import VAPLabelConfig, future_activity_labels
from hri_turn_taking.vap.trainer import compute_frame_metrics, evaluate_model


def test_feature_sets_zero_only_disabled_canonical_inputs() -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=5.0, n_participants=3, random_seed=4)
    )
    original = frame_df.copy(deep=True)
    canonical = input_feature_columns()

    for name in FEATURE_SETS:
        masked = apply_feature_set(frame_df, name)
        active = set(active_feature_columns(name))
        disabled = [column for column in canonical if column not in active]
        assert masked[disabled].eq(0.0).all().all()
        assert masked[list(active)].equals(frame_df[list(active)])
        assert masked.filter(regex=r"_vad$").equals(frame_df.filter(regex=r"_vad$"))

    assert frame_df.equals(original), "apply_feature_set must not mutate its caller"
    assert not any(
        column.endswith("asd_prob")
        for column in active_feature_columns("visual_only_strict")
    )


def test_audio_only_produces_identical_participant_token_inputs() -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=5.0, n_participants=3, random_seed=5)
    )
    session = sessions_from_dataframe(apply_feature_set(frame_df, "audio_only"))[0]
    n_participant_features = len(TRANSFORMED_PARTICIPANT_FEATURES)
    participant_block = session.features[
        :, : -len(TRANSFORMED_GLOBAL_FEATURES)
    ].reshape(session.features.shape[0], 4, n_participant_features)

    assert np.allclose(participant_block, 0.0)
    assert FEATURE_SETS["audio_only"].participant_tokens_symmetric


@pytest.mark.parametrize(
    ("name", "participant_features", "include_audio"),
    [
        (
            "head_pose_only",
            ("head_yaw_deg", "pose_valid", "pose_quality"),
            False,
        ),
        (
            "audio_plus_head_pose",
            ("head_yaw_deg", "pose_valid", "pose_quality"),
            True,
        ),
        (
            "audio_plus_visual_strict",
            tuple(
                feature
                for feature in TRANSFORMED_PARTICIPANT_FEATURES
                if feature != "asd_prob"
            ),
            True,
        ),
    ],
)
def test_scoped_feature_set_metadata_is_exact(
    name: str,
    participant_features: tuple[str, ...],
    include_audio: bool,
) -> None:
    expected = [
        participant_column(slot, feature)
        for slot in range(1, 5)
        for feature in participant_features
    ]
    if include_audio:
        expected.extend(GLOBAL_FEATURE_COLUMNS)

    metadata = feature_set_metadata(name)
    assert active_feature_columns(name) == expected
    assert metadata["name"] == name
    assert metadata["active_columns"] == expected
    assert metadata["participant_tokens_symmetric"] is False


def test_next_speaker_uses_pre_offset_future_vectors() -> None:
    fps = 25
    vad = np.zeros((200, 3), dtype=float)
    vad[0:50, 0] = 1.0
    vad[63:113, 1] = 1.0
    vad[126:176, 2] = 1.0
    probabilities = np.full((200, 3, 4), 0.05, dtype=float)
    probabilities[38:50, 1, 2:] = 0.95
    probabilities[101:113, 2, 2:] = 0.95

    labels, scores = next_speaker_scores(probabilities, vad, frame_rate_hz=fps)
    evaluation = score_next_speaker(np.asarray(labels), np.asarray(scores))

    assert labels == [1, 2]
    assert evaluation.n_events == 2
    assert evaluation.top1_accuracy == pytest.approx(1.0)
    assert evaluation.balanced_recall == pytest.approx(1.0)
    assert evaluation.mrr == pytest.approx(1.0)
    # P1 was present for both events but was never the true next speaker, so
    # its one-vs-rest AP is undefined and the strict primary macro stays NA.
    assert evaluation.participant_macro_ovr_ap is None
    assert evaluation.per_participant["P1"].ap_na_reason == "no_positive_events"


def test_predictive_windows_exclude_speaker_before_verified_solo_context() -> None:
    fps = 10
    vad = np.zeros((24, 3), dtype=float)
    vad[5, 2] = 1.0  # P3 spoke 0.5 s before the offset, outside verified context.
    vad[6:10, 0] = 1.0  # P1 is the sole speaker in the verified 0.4 s context.
    vad[12:20, 1] = 1.0  # Gap starts at frame 10, then P2 takes the floor.

    probabilities = np.full((24, 3, 4), 0.1, dtype=float)
    high_p3 = probabilities.copy()
    high_p3[5, 2, 2:] = 0.99

    shift_labels, shift_scores = shift_prediction_scores(
        high_p3,
        vad,
        frame_rate_hz=fps,
        pre_offset_s=0.5,
    )
    _, reference_shift_scores = shift_prediction_scores(
        probabilities,
        vad,
        frame_rate_hz=fps,
        pre_offset_s=0.5,
    )
    next_labels, next_scores = next_speaker_scores(
        high_p3,
        vad,
        frame_rate_hz=fps,
        pre_offset_s=0.5,
    )
    _, reference_next_scores = next_speaker_scores(
        probabilities,
        vad,
        frame_rate_hz=fps,
        pre_offset_s=0.5,
    )

    assert shift_labels == [1]
    assert next_labels == [1]
    assert shift_scores == pytest.approx(reference_shift_scores)
    assert np.asarray(next_scores) == pytest.approx(np.asarray(reference_next_scores))


def test_predictive_and_overlap_scores_filter_incomplete_horizon_windows() -> None:
    n_frames = 160
    valid = np.arange(n_frames) + 50 <= n_frames
    probabilities = np.full((n_frames, 3, 4), 0.5, dtype=float)

    gap_vad = np.zeros((n_frames, 3), dtype=float)
    gap_vad[0:30, 2] = 1.0
    gap_vad[35:90, 0] = 1.0
    gap_vad[80:120, 2] = 1.0
    gap_vad[125:150, 0] = 1.0
    unfiltered_shift, _ = shift_prediction_scores(probabilities, gap_vad)
    filtered_shift, _ = shift_prediction_scores(probabilities, gap_vad, valid=valid)
    unfiltered_next, _ = next_speaker_scores(probabilities, gap_vad)
    filtered_next, _ = next_speaker_scores(probabilities, gap_vad, valid=valid)

    overlap_vad = np.zeros((n_frames, 3), dtype=float)
    overlap_vad[0:60, 0] = 1.0
    overlap_vad[20:70, 1] = 1.0
    overlap_vad[80:140, 0] = 1.0
    overlap_vad[120:150, 1] = 1.0
    unfiltered_overlap, _ = overlap_resolution_scores(probabilities, overlap_vad)
    filtered_overlap, _ = overlap_resolution_scores(
        probabilities,
        overlap_vad,
        valid=valid,
    )

    assert unfiltered_shift == [1, 1]
    assert filtered_shift == [1]
    assert unfiltered_next == [0, 0]
    assert filtered_next == [0]
    assert unfiltered_overlap == [1, 1]
    assert filtered_overlap == [1]


def test_shift_wrappers_return_explicit_empty_event_evaluations() -> None:
    probabilities = np.full((20, 3, 4), 0.1, dtype=float)
    vad = np.zeros((20, 3), dtype=float)

    for evaluation in (
        evaluate_shift_hold(probabilities, vad, frame_rate_hz=10),
        evaluate_shift_prediction(probabilities, vad, frame_rate_hz=10),
    ):
        assert evaluation.n_events == 0
        assert evaluation.n_positive == 0
        assert evaluation.n_negative == 0
        assert evaluation.average_precision is None
        assert evaluation.average_precision_na_reason == "no_events"


def test_next_speaker_ties_use_uniform_credit_without_slot_index_bias() -> None:
    labels = np.array([1, 2], dtype=int)
    scores = np.full((2, 3), 0.5, dtype=float)

    evaluation = score_next_speaker(labels, scores)

    assert evaluation.tie_policy == "uniform_credit_among_max_scores"
    assert evaluation.tie_rate == pytest.approx(1.0)
    assert evaluation.top1_accuracy == pytest.approx(1.0 / 3.0)
    assert evaluation.balanced_recall == pytest.approx(1.0 / 3.0)
    assert evaluation.mrr == pytest.approx(0.5)
    assert evaluation.per_participant["P2"].tie_aware_recall == pytest.approx(1.0 / 3.0)
    assert evaluation.per_participant["P3"].tie_aware_recall == pytest.approx(1.0 / 3.0)
    assert evaluation.participant_macro_ovr_ap is None
    assert (
        evaluation.participant_macro_ovr_ap_na_reason
        == "one_or_more_in_scope_participants_single_class"
    )


def test_next_speaker_primary_macro_ap_uses_all_in_scope_participants() -> None:
    labels = np.array([0, 1, 0, 1], dtype=int)
    scores = np.array(
        [
            [0.9, 0.1, -np.inf],
            [0.1, 0.9, -np.inf],
            [0.8, 0.2, -np.inf],
            [0.2, 0.8, -np.inf],
        ],
        dtype=float,
    )

    evaluation = score_next_speaker(labels, scores)

    assert evaluation.n_participants_in_scope == 2
    assert evaluation.n_participants_evaluable_ap == 2
    assert evaluation.participant_macro_ovr_ap == pytest.approx(1.0)
    assert evaluation.participant_macro_ovr_ap_na_reason is None
    assert evaluation.per_participant["P1"].n_positive == 2
    assert evaluation.per_participant["P1"].n_negative == 2
    assert evaluation.per_participant["P3"].ap_na_reason == "not_in_scope"


def test_next_speaker_empty_result_is_explicit_and_strict_json_safe() -> None:
    evaluation = score_next_speaker(
        np.empty(0, dtype=int),
        np.empty((0, 4), dtype=float),
    )

    assert evaluation.n_events == 0
    assert evaluation.participant_macro_ovr_ap is None
    assert evaluation.balanced_recall is None
    assert evaluation.mrr is None
    assert evaluation.top1_accuracy is None
    assert evaluation.tie_rate is None
    assert evaluation.na_reason == "no_events"
    assert all(not item.in_scope for item in evaluation.per_participant.values())
    encoded = json.dumps(asdict(evaluation), allow_nan=False)
    assert "null" in encoded
    assert "NaN" not in encoded


def test_overlap_resolution_ignores_high_scoring_silent_non_challengers() -> None:
    fps = 10
    vad = np.zeros((80, 4), dtype=float)
    vad[0:40, 0] = 1.0  # P1 holds the floor.
    vad[20:50, 1] = 1.0  # P2 challenges, then retains the floor after P1 yields.

    probabilities = np.full((80, 4, 4), 0.01, dtype=float)
    probabilities[:, 0, :] = 0.6
    probabilities[:, 1, :] = 0.2
    low_unrelated = probabilities.copy()
    probabilities[:, 2:, :] = 0.99  # Silent P3/P4 must not stand in for P2.

    labels, high_scores = overlap_resolution_scores(
        probabilities,
        vad,
        frame_rate_hz=fps,
        present=np.ones(4, dtype=float),
    )
    _, low_scores = overlap_resolution_scores(
        low_unrelated,
        vad,
        frame_rate_hz=fps,
        present=np.ones(4, dtype=float),
    )

    assert labels == [1]
    assert high_scores == pytest.approx([0.25])
    assert high_scores == pytest.approx(low_scores)


def test_frame_map_is_macro_mean_of_per_bin_ap_not_micro_ap() -> None:
    labels = np.array(
        [
            [1, 0],
            [0, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=float,
    )
    scores = np.array(
        [
            [0.9, 0.9],
            [0.8, 0.8],
            [0.7, 0.7],
            [0.1, 0.6],
        ],
        dtype=float,
    )

    metrics = compute_frame_metrics(labels, scores)

    assert metrics["mAP"] == pytest.approx(
        np.mean([metrics["AP_bin0"], metrics["AP_bin1"]])
    )
    assert metrics["micro_AP"] != pytest.approx(metrics["mAP"])
    assert "macro_f1_bins" in metrics
    assert {"F1_bin0", "F1_bin1"}.issubset(metrics)


def test_frame_map_is_na_when_any_bin_has_single_class_and_records_support() -> None:
    labels = np.array(
        [
            [1, 0],
            [0, 0],
            [1, 0],
            [0, 0],
        ],
        dtype=float,
    )
    scores = np.array(
        [
            [0.9, 0.4],
            [0.2, 0.3],
            [0.8, 0.2],
            [0.1, 0.1],
        ],
        dtype=float,
    )

    metrics = compute_frame_metrics(
        labels,
        scores,
        participant_indices=np.array([0, 0, 1, 1]),
    )

    assert metrics["mAP"] is None
    assert metrics["mAP_na_reason"] == "one_or_more_bins_single_class"
    assert metrics["mAP_evaluable_bins"] == pytest.approx(metrics["AP_bin0"])
    assert metrics["bins"]["bin1"] == {
        "n_positive": 0,
        "n_negative": 4,
        "average_precision": None,
        "ap_na_reason": "no_positive",
        "f1": 0.0,
    }
    assert metrics["participants"]["P1"]["bins"]["bin1"]["n_negative"] == 2
    encoded = json.dumps(metrics, allow_nan=False)
    assert "NaN" not in encoded


def test_evaluate_model_keeps_all_empty_event_sections_with_null_metrics() -> None:
    config = VAPLabelConfig()
    n_frames = 100
    vad = np.zeros((n_frames, 4), dtype=np.float32)
    labels, valid = future_activity_labels(vad, config)
    width = 4 * len(TRANSFORMED_PARTICIPANT_FEATURES) + len(TRANSFORMED_GLOBAL_FEATURES)
    session = SessionTensors(
        session_id="quiet",
        features=np.zeros((n_frames, width), dtype=np.float32),
        labels=labels.astype(np.float32),
        vad=vad,
        valid=valid,
        present=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )

    _, event_metrics = evaluate_model(LastSpeakerBaseline(), [session], None)

    assert set(event_metrics) == {
        "shift_hold",
        "shift_prediction",
        "overlap_resolution",
        "robot_entry",
        "next_speaker",
    }
    assert all(section["n_events"] == 0 for section in event_metrics.values())
    assert event_metrics["shift_hold"]["average_precision"] is None
    assert event_metrics["next_speaker"]["participant_macro_ovr_ap"] is None
    encoded = json.dumps(event_metrics, allow_nan=False)
    assert "NaN" not in encoded


def _baseline_session() -> SessionTensors:
    config = VAPLabelConfig(frame_rate_hz=25)
    n_frames = 320
    vad = np.zeros((n_frames, 4), dtype=np.float32)
    vad[0:50, 0] = 1.0
    vad[63:113, 1] = 1.0
    vad[126:176, 2] = 1.0
    vad[189:239, 2] = 1.0
    vad[252:302, 0] = 1.0
    labels, valid = future_activity_labels(vad, config)

    n_participant_features = len(TRANSFORMED_PARTICIPANT_FEATURES)
    width = 4 * n_participant_features + len(TRANSFORMED_GLOBAL_FEATURES)
    features = np.zeros((n_frames, width), dtype=np.float32)
    # Robot-observable ASD evidence. These are the only channels the baseline
    # reads; the separate gold VAD matrix remains evaluation-only.
    features[0:50, 0 * n_participant_features] = 0.9
    features[63:113, 1 * n_participant_features] = 0.9
    features[126:239, 2 * n_participant_features] = 0.9
    features[252:302, 0 * n_participant_features] = 0.9
    return SessionTensors(
        session_id="baseline",
        features=features,
        labels=labels.astype(np.float32),
        vad=vad,
        valid=valid,
        present=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def test_baselines_are_vad_independent_and_use_unified_event_evaluation() -> None:
    session = _baseline_session()
    persistence = asd_persistence_probabilities(session)
    last_speaker = last_speaker_probabilities(session)
    without_gold = replace(session, vad=np.ones_like(session.vad))

    assert persistence.shape == session.labels.shape
    assert last_speaker.shape == session.labels.shape
    assert np.array_equal(persistence, asd_persistence_probabilities(without_gold))
    assert np.array_equal(last_speaker, last_speaker_probabilities(without_gold))
    assert np.all(last_speaker[55, 0] == 1.0)

    for baseline in (ASDPersistenceBaseline(), LastSpeakerBaseline()):
        frame_metrics, event_metrics = evaluate_model(
            baseline,
            [session],
            None,
            robot_slot=2,
        )
        assert "mAP" in frame_metrics
        assert event_metrics["next_speaker"]["n_events"] >= 2
        assert event_metrics["robot_entry"]["n_events"] == 2


def test_last_speaker_baseline_does_not_break_asd_ties_by_slot_index() -> None:
    session = _baseline_session()
    features = np.zeros_like(session.features)
    stride = len(TRANSFORMED_PARTICIPANT_FEATURES)
    features[0, 0 * stride] = 0.9
    features[0, 1 * stride] = 0.9  # Initial ambiguity: no state should be invented.
    features[1, 1 * stride] = 0.9  # Unique evidence establishes P2.
    features[2, 0 * stride] = 0.9
    features[2, 1 * stride] = 0.9  # Tie includes established P2, so P2 persists.

    output = last_speaker_probabilities(replace(session, features=features))

    assert np.all(output[0] == 0.0)
    assert np.all(output[1, 1] == 1.0)
    assert np.all(output[1, [0, 2, 3]] == 0.0)
    assert np.all(output[2, 1] == 1.0)
    assert np.all(output[2, [0, 2, 3]] == 0.0)
