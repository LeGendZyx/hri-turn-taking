from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pytest

from hri_turn_taking.data.multiparty import SyntheticConfig, generate_multiparty_dataset
from hri_turn_taking.vap import predictions as prediction_module
from hri_turn_taking.vap.dataset import (
    TRANSFORMED_GLOBAL_FEATURES,
    TRANSFORMED_PARTICIPANT_FEATURES,
    FeatureNormalizer,
    sessions_from_dataframe,
)
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG, VAPLabelConfig
from hri_turn_taking.vap.model import MultipartyVAP, VAPModelConfig
from hri_turn_taking.vap.predictions import (
    load_prediction_checkpoint_snapshot,
    prediction_dataframe,
    write_prediction_artifact,
)
from hri_turn_taking.vap.trainer import LoadedVAPCheckpoint


def _checkpoint(
    frame_df,
    *,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    trainer_config: dict | None = None,
    run_metadata: dict | None = None,
) -> LoadedVAPCheckpoint:
    sessions = sessions_from_dataframe(frame_df, label_config=label_config)
    return LoadedVAPCheckpoint(
        model=MultipartyVAP(
            VAPModelConfig(
                n_participant_features=len(TRANSFORMED_PARTICIPANT_FEATURES),
                n_global_features=len(TRANSFORMED_GLOBAL_FEATURES),
                n_bins=label_config.n_bins,
                d_model=8,
                n_layers=1,
                attention_window=50,
            )
        ).eval(),
        normalizer=FeatureNormalizer.fit(sessions),
        label_config=label_config,
        trainer_config=trainer_config or {"feature_set": "multimodal"},
        run_metadata=run_metadata,
    )


def test_prediction_artifact_contains_only_primary_participantwise_output() -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=3.0, n_participants=3, random_seed=7)
    )
    checkpoint = _checkpoint(frame_df)

    table, metadata = prediction_dataframe(frame_df, checkpoint)

    assert len(table) == len(frame_df)
    assert metadata["format_version"] == 3
    assert metadata["session_tensor_shapes"] == {"session_000": [75, 4, 4]}
    assert metadata["auxiliary_current_vad_exported"] is False
    assert not any("vad_logit" in column or column.endswith("_vad") for column in table.columns)
    assert all(
        f"P{slot}_vap_bin{bin_index}" in table
        for slot in range(1, 5)
        for bin_index in range(4)
    )
    assert np.isfinite(table.filter(like="_vap_bin").to_numpy()).all()
    assert (table["P4_present"] == 0).all()
    assert (table.filter(regex=r"^P4_(vap_bin|p_now|p_future)") == 0.0).all().all()


def test_prediction_is_independent_of_gold_vad_values() -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=3.0, n_participants=3, random_seed=17)
    )
    checkpoint = _checkpoint(frame_df)
    feature_only = frame_df.copy()
    vad_columns = [column for column in feature_only if column.endswith("_vad")]
    feature_only.loc[:, vad_columns] = 0.0

    original, metadata = prediction_dataframe(frame_df, checkpoint)
    without_gold, feature_only_metadata = prediction_dataframe(feature_only, checkpoint)

    tensor_columns = metadata["tensor_columns"]
    assert original[tensor_columns].equals(without_gold[tensor_columns])
    assert metadata["model_input_view_sha256"] == feature_only_metadata[
        "model_input_view_sha256"
    ]
    assert metadata["input_dataframe_sha256"] != feature_only_metadata[
        "input_dataframe_sha256"
    ]


def test_prediction_exports_custom_label_validity_per_bin_and_participant() -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=3.0, n_participants=3, random_seed=8)
    )
    label_config = VAPLabelConfig(
        frame_rate_hz=25,
        bin_edges_s=(0.08, 0.2),
        active_ratio=0.6,
        now_bins=(0,),
        future_bins=(1,),
    )

    table, metadata = prediction_dataframe(
        frame_df,
        _checkpoint(frame_df, label_config=label_config),
        source_manifest={"target_ready": True},
    )

    assert table["full_future_horizon_temporally_available"].sum() == len(table) - 5 + 1
    assert table["P1_vap_bin0_horizon_temporally_available"].sum() == len(table) - 2 + 1
    assert table["P1_vap_bin1_gold_target_available"].sum() == len(table) - 5 + 1
    last_complete = len(table) - 5
    assert table.loc[last_complete, "P1_vap_bin1_horizon_temporally_available"] == 1
    assert table.loc[last_complete + 1, "P1_vap_bin1_horizon_temporally_available"] == 0
    assert table["P4_vap_bin0_gold_target_available"].eq(0).all()
    assert metadata["label_config"] == {
        "frame_rate_hz": 25,
        "bin_edges_s": [0.08, 0.2],
        "active_ratio": 0.6,
        "now_bins": [0],
        "future_bins": [1],
        "horizon_frames": 5,
        "bin_frame_ranges": [[0, 2], [2, 5]],
    }
    assert len(metadata["horizon_temporal_availability_columns"]) == 8
    assert len(metadata["gold_target_availability_columns"]) == 8
    assert metadata["source_target_ready"] is True


def test_feature_only_prediction_never_claims_gold_target_availability() -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=3.0, n_participants=3, random_seed=18)
    )
    vad_columns = [column for column in frame_df if column.endswith("_vad")]
    frame_df.loc[:, vad_columns] = 0.0

    table, metadata = prediction_dataframe(
        frame_df,
        _checkpoint(frame_df),
        source_manifest={"target_ready": False, "target_blockers": ["no gold VAD"]},
    )

    temporal_columns = metadata["horizon_temporal_availability_columns"]
    target_columns = metadata["gold_target_availability_columns"]
    assert table[temporal_columns].to_numpy().sum() > 0
    assert table[target_columns].eq(0).all().all()
    assert metadata["source_target_ready"] is False


def test_dataframe_hash_binds_column_order_and_dtype() -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=3.0, n_participants=3, random_seed=19)
    )
    checkpoint = _checkpoint(frame_df)
    _, original = prediction_dataframe(frame_df, checkpoint)

    reordered = frame_df.loc[:, list(reversed(frame_df.columns))]
    _, reordered_metadata = prediction_dataframe(reordered, checkpoint)
    assert original["input_dataframe_sha256"] != reordered_metadata[
        "input_dataframe_sha256"
    ]
    assert original["model_input_view_sha256"] == reordered_metadata[
        "model_input_view_sha256"
    ]

    dtype_changed = frame_df.copy()
    dtype_changed["P1_asd_prob"] = dtype_changed["P1_asd_prob"].astype("float32")
    _, dtype_metadata = prediction_dataframe(dtype_changed, checkpoint)
    assert original["input_dataframe_sha256"] != dtype_metadata["input_dataframe_sha256"]
    assert original["model_input_view_sha256"] != dtype_metadata[
        "model_input_view_sha256"
    ]


@pytest.mark.parametrize("existing_kind", ["csv", "metadata"])
def test_prediction_writer_refuses_either_existing_artifact_file_by_default(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    output = tmp_path / existing_kind / "predictions.csv"
    output.parent.mkdir(parents=True)
    metadata_path = output.with_suffix(".metadata.json")
    existing_path = output if existing_kind == "csv" else metadata_path
    existing_path.write_bytes(b"do-not-overwrite")
    checkpoint_path = tmp_path / "unused.pt"

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_prediction_artifact(
            generate_multiparty_dataset(
                SyntheticConfig(n_sessions=1, duration_s=3.0, n_participants=3)
            ),
            checkpoint_path=checkpoint_path,
            output_path=output,
        )
    assert existing_path.read_bytes() == b"do-not-overwrite"


def test_prediction_writer_hashes_standard_json_and_merges_smoke_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=3.0, n_participants=3, random_seed=9)
    )
    checkpoint = _checkpoint(
        frame_df,
        run_metadata={
            "artifact_status": "smoke_only",
            "result_role": "development_inner_validation",
            "counted_experiment_eligible": False,
            "effective_smoke_reasons": ["training override"],
        },
    )
    checkpoint_path = tmp_path / "vap.pt"
    checkpoint_path.write_bytes(b"checkpoint bytes")
    load_calls = 0

    def load_once(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return checkpoint

    monkeypatch.setattr(prediction_module, "load_vap_checkpoint", load_once)
    source_manifest = {"files": [{"session_id": "session_000", "csv_sha256": "a" * 64}]}
    output = tmp_path / "predictions.csv"

    artifact = write_prediction_artifact(
        frame_df,
        checkpoint_path=checkpoint_path,
        output_path=output,
        source_manifest=source_manifest,
        override_status={"requested_overrides": {"allow_placeholder_inputs": True}},
    )

    raw_metadata = artifact.metadata_path.read_text(encoding="utf-8")
    stored = json.loads(raw_metadata)
    assert artifact.metadata == stored
    assert "NaN" not in raw_metadata
    assert stored["prediction_csv_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert stored["checkpoint_sha256"] == hashlib.sha256(b"checkpoint bytes").hexdigest()
    assert len(stored["source_manifest_sha256"]) == 64
    assert stored["artifact_status"] == "smoke_only"
    assert stored["counted_experiment_eligible"] is False
    assert stored["override_status"]["checkpoint"]["run_metadata"][
        "artifact_status"
    ] == "smoke_only"
    assert stored["override_status"]["prediction_input"]["requested_overrides"][
        "allow_placeholder_inputs"
    ] is True
    assert load_calls == 1
    first_artifact_id = uuid.UUID(stored["artifact_id"])

    replacement = write_prediction_artifact(
        frame_df,
        checkpoint_path=checkpoint_path,
        output_path=output,
        overwrite=True,
        source_manifest=source_manifest,
    )
    assert uuid.UUID(replacement.metadata["artifact_id"]) != first_artifact_id
    assert replacement.metadata["prediction_csv_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert load_calls == 2


def test_preloaded_checkpoint_snapshot_binds_inference_to_original_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=3.0, n_participants=3, random_seed=20)
    )
    checkpoint = _checkpoint(frame_df)
    checkpoint_path = tmp_path / "vap.pt"
    original_bytes = b"first checkpoint revision"
    checkpoint_path.write_bytes(original_bytes)
    loads: list[bytes] = []

    def capture_loaded_bytes(handle, *, device):
        loads.append(handle.read())
        return checkpoint

    monkeypatch.setattr(prediction_module, "load_vap_checkpoint", capture_loaded_bytes)
    snapshot = load_prediction_checkpoint_snapshot(checkpoint_path)
    checkpoint_path.write_bytes(b"replacement checkpoint revision")

    artifact = write_prediction_artifact(
        frame_df,
        checkpoint_snapshot=snapshot,
        output_path=tmp_path / "predictions.csv",
    )

    assert loads == [original_bytes]
    assert artifact.metadata["checkpoint_sha256"] == hashlib.sha256(
        original_bytes
    ).hexdigest()


def test_prediction_rechecks_audited_source_hashes_after_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=3.0, n_participants=3, random_seed=21)
    )
    checkpoint = _checkpoint(frame_df)
    checkpoint_path = tmp_path / "vap.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        prediction_module,
        "load_vap_checkpoint",
        lambda *args, **kwargs: checkpoint,
    )
    data_path = tmp_path / "source.csv"
    provenance_path = tmp_path / "source.provenance.json"
    data_path.write_bytes(b"audited data")
    provenance_path.write_bytes(b'{"data_kind":"synthetic"}')
    manifest = {
        "target_ready": True,
        "source_artifacts": [
            {
                "data_path": str(data_path),
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "provenance_path": str(provenance_path),
                "provenance_sha256": hashlib.sha256(
                    provenance_path.read_bytes()
                ).hexdigest(),
            }
        ],
    }
    real_prediction_dataframe = prediction_module.prediction_dataframe

    def mutate_after_inference(*args, **kwargs):
        result = real_prediction_dataframe(*args, **kwargs)
        data_path.write_bytes(b"changed after readiness")
        return result

    monkeypatch.setattr(
        prediction_module,
        "prediction_dataframe",
        mutate_after_inference,
    )

    with pytest.raises(ValueError, match="changed after readiness audit"):
        write_prediction_artifact(
            frame_df,
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "predictions.csv",
            source_manifest=manifest,
        )
    assert not (tmp_path / "predictions.csv").exists()


def test_prediction_writer_cleans_csv_when_metadata_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=3.0, n_participants=3, random_seed=10)
    )
    checkpoint = _checkpoint(frame_df)
    checkpoint_path = tmp_path / "vap.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        prediction_module,
        "load_vap_checkpoint",
        lambda *args, **kwargs: checkpoint,
    )
    output = tmp_path / "predictions.csv"
    metadata_path = output.with_suffix(".metadata.json")
    real_replace = os.replace

    def fail_metadata_commit(source, destination) -> None:
        if Path(destination) == metadata_path:
            raise OSError("injected metadata commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(prediction_module.os, "replace", fail_metadata_commit)
    with pytest.raises(OSError, match="injected"):
        write_prediction_artifact(
            frame_df,
            checkpoint_path=checkpoint_path,
            output_path=output,
        )

    assert not output.exists()
    assert not metadata_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_prediction_writer_restores_prior_pair_when_overwrite_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_df = generate_multiparty_dataset(
        SyntheticConfig(n_sessions=1, duration_s=3.0, n_participants=3, random_seed=11)
    )
    checkpoint = _checkpoint(frame_df)
    checkpoint_path = tmp_path / "vap.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        prediction_module,
        "load_vap_checkpoint",
        lambda *args, **kwargs: checkpoint,
    )
    output = tmp_path / "predictions.csv"
    metadata_path = output.with_suffix(".metadata.json")
    write_prediction_artifact(
        frame_df,
        checkpoint_path=checkpoint_path,
        output_path=output,
    )
    prior_csv = output.read_bytes()
    prior_metadata = metadata_path.read_bytes()
    real_replace = os.replace

    def fail_metadata_commit(source, destination) -> None:
        if Path(destination) == metadata_path:
            raise OSError("injected metadata commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(prediction_module.os, "replace", fail_metadata_commit)
    with pytest.raises(OSError, match="injected"):
        write_prediction_artifact(
            frame_df,
            checkpoint_path=checkpoint_path,
            output_path=output,
            overwrite=True,
        )

    assert output.read_bytes() == prior_csv
    assert metadata_path.read_bytes() == prior_metadata
    assert not list(tmp_path.glob(".*.tmp"))
