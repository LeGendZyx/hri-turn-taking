"""Export the single delivered participant-wise VAP output.

The training model may carry an auxiliary current-VAD head, but the project
scope defines exactly one delivered model output: future voice-activity
probabilities with shape ``time x participant x future_bin``.  This module
turns that tensor into a stable, auditable wide table and a JSON sidecar.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hri_turn_taking.data.multiparty import (
    MAX_PARTICIPANTS,
    input_feature_columns,
    present_columns,
)
from hri_turn_taking.vap.dataset import sessions_from_dataframe
from hri_turn_taking.vap.labels import p_now_p_future
from hri_turn_taking.vap.trainer import (
    LoadedVAPCheckpoint,
    load_vap_checkpoint,
    predict_session_probabilities,
)

PREDICTION_FORMAT_VERSION = 3

_SMOKE_OVERRIDE_KEYS = {
    "allow_untrainable",
    "allow_placeholder_inputs",
}


@dataclass(frozen=True)
class PredictionArtifact:
    table: pd.DataFrame
    metadata: dict[str, object]
    output_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class PredictionCheckpointSnapshot:
    """A checkpoint deserialised from exactly the bytes named by its digest."""

    checkpoint: LoadedVAPCheckpoint
    path: Path
    sha256: str


def load_prediction_checkpoint_snapshot(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> PredictionCheckpointSnapshot:
    """Read, hash, and deserialize one immutable checkpoint byte snapshot.

    Reading the file into memory before calling the checkpoint loader prevents
    the loaded model and the recorded digest from referring to different file
    revisions if the path is replaced concurrently.
    """
    path = Path(checkpoint_path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = load_vap_checkpoint(io.BytesIO(payload), device=device)
    return PredictionCheckpointSnapshot(checkpoint=checkpoint, path=path, sha256=digest)


def _standard_json_object(value: Mapping[str, object] | None, name: str) -> dict[str, object]:
    """Copy a mapping through strict standard JSON (no NaN/custom scalar types)."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite standard-JSON values") from exc
    return json.loads(encoded)


def _canonical_json_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_sha256(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    normalised = value.lower()
    invalid_character = any(
        character not in "0123456789abcdef" for character in normalised
    )
    if len(normalised) != 64 or invalid_character:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 digest")
    return normalised


def _source_binding(
    source_manifest: Mapping[str, object] | None,
    source_manifest_sha256: str | None,
) -> tuple[dict[str, object], str | None]:
    manifest = _standard_json_object(source_manifest, "source_manifest")
    declared_hash = _validated_sha256(source_manifest_sha256, "source_manifest_sha256")
    computed_hash = _canonical_json_sha256(manifest) if source_manifest is not None else None
    if declared_hash is not None and computed_hash is not None and declared_hash != computed_hash:
        raise ValueError(
            "source_manifest_sha256 does not match the canonical source_manifest payload"
        )
    return manifest, declared_hash or computed_hash


def _dataframe_sha256(frame_df: pd.DataFrame) -> str:
    """Hash ordered schema (column name + dtype) and ordered cell values."""
    schema = [
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in zip(frame_df.columns, frame_df.dtypes, strict=True)
    ]
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"n_rows": len(frame_df), "columns": schema},
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    value_hashes = pd.util.hash_pandas_object(
        frame_df,
        index=False,
        categorize=False,
    ).to_numpy(dtype="<u8", copy=False)
    digest.update(value_hashes.tobytes(order="C"))
    return digest.hexdigest()


def _verify_source_artifact_hashes(source_manifest: Mapping[str, object]) -> None:
    """Fail closed if audited source files no longer match their recorded hashes."""
    artifacts = source_manifest.get("source_artifacts")
    if artifacts is None:
        return
    if not isinstance(artifacts, list):
        raise ValueError("source_manifest.source_artifacts must be a JSON array")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise ValueError(f"source_artifacts[{index}] must be a JSON object")
        for path_key, hash_key in (
            ("data_path", "data_sha256"),
            ("provenance_path", "provenance_sha256"),
        ):
            path_value = artifact.get(path_key)
            hash_value = artifact.get(hash_key)
            if path_value is None and hash_value is None:
                continue
            if not isinstance(path_value, str):
                raise ValueError(f"source_artifacts[{index}].{path_key} must be a path")
            expected = _validated_sha256(
                hash_value if isinstance(hash_value, str) else None,
                f"source_artifacts[{index}].{hash_key}",
            )
            if expected is None:
                raise ValueError(
                    f"source_artifacts[{index}].{hash_key} is required when {path_key} is set"
                )
            path = Path(path_value)
            try:
                actual = _sha256_file(path)
            except OSError as exc:
                raise ValueError(f"Audited source artifact is unavailable: {path}") from exc
            if actual != expected:
                raise ValueError(
                    f"Audited source artifact changed after readiness audit: {path}"
                )


def _checkpoint_override_status(checkpoint: LoadedVAPCheckpoint) -> dict[str, object]:
    """Recover current/future checkpoint run-context layouts without coupling them."""
    status: dict[str, object] = {}
    for attribute in ("run_metadata", "override_status", "run_context", "artifact_context"):
        value = getattr(checkpoint, attribute, None)
        if isinstance(value, Mapping):
            status[attribute] = dict(value)
    for attribute in ("artifact_status", "scientific_status"):
        value = getattr(checkpoint, attribute, None)
        if isinstance(value, str):
            status[attribute] = value
    trainer_config = checkpoint.trainer_config or {}
    for key in ("override_status", "run_context", "artifact_context"):
        value = trainer_config.get(key)
        if isinstance(value, Mapping):
            status[key] = dict(value)
    for key in ("artifact_status", "scientific_status", "effective_smoke_reasons"):
        if key in trainer_config:
            status[key] = trainer_config[key]
    requested = {
        key: trainer_config[key]
        for key in _SMOKE_OVERRIDE_KEYS
        if key in trainer_config
    }
    if requested:
        status["requested_overrides"] = requested
    return _standard_json_object(status, "checkpoint override status")


def _is_smoke_status(status: object) -> bool:
    if not isinstance(status, Mapping):
        return False
    for key in ("artifact_status", "scientific_status", "status", "effective_status"):
        value = status.get(key)
        if isinstance(value, str) and value.lower() in {"smoke", "smoke_only"}:
            return True
    for key in _SMOKE_OVERRIDE_KEYS | {"smoke_only", "is_smoke"}:
        if status.get(key) is True:
            return True
    reasons = status.get("effective_smoke_reasons")
    if isinstance(reasons, list) and reasons:
        return True
    return any(
        _is_smoke_status(value)
        for key, value in status.items()
        if key
        in {
            "run_metadata",
            "requested_overrides",
            "override_status",
            "run_context",
            "artifact_context",
        }
    )


def _smoke_reasons(status: Mapping[str, object], origin: str) -> list[str]:
    reasons: list[str] = []
    explicit = status.get("effective_smoke_reasons")
    if isinstance(explicit, list):
        reasons.extend(f"{origin}: {reason}" for reason in explicit if isinstance(reason, str))
    for key in _SMOKE_OVERRIDE_KEYS:
        if status.get(key) is True:
            reasons.append(f"{origin}: {key}=true")
    requested = status.get("requested_overrides")
    if isinstance(requested, Mapping):
        for key in _SMOKE_OVERRIDE_KEYS:
            if requested.get(key) is True:
                reasons.append(f"{origin}: {key}=true")
    for key in ("run_metadata", "override_status", "run_context", "artifact_context"):
        nested = status.get(key)
        if isinstance(nested, Mapping):
            reasons.extend(_smoke_reasons(nested, f"{origin}.{key}"))
    if _is_smoke_status(status) and not reasons:
        reasons.append(f"{origin}: marked smoke_only")
    return reasons


def _merged_override_status(
    checkpoint: LoadedVAPCheckpoint,
    prediction_status: Mapping[str, object] | None,
) -> dict[str, object]:
    checkpoint_status = _checkpoint_override_status(checkpoint)
    input_status = _standard_json_object(prediction_status, "override_status")
    reasons = [
        *_smoke_reasons(checkpoint_status, "checkpoint"),
        *_smoke_reasons(input_status, "prediction_input"),
    ]
    reasons = list(dict.fromkeys(reasons))
    smoke_only = _is_smoke_status(checkpoint_status) or _is_smoke_status(input_status)
    return {
        "artifact_status": "smoke_only" if smoke_only else "development",
        "result_role": "development_inference",
        "counted_experiment_eligible": False,
        "checkpoint": checkpoint_status,
        "prediction_input": input_status,
        "effective_smoke_reasons": reasons,
    }


def _label_config_metadata(checkpoint: LoadedVAPCheckpoint) -> dict[str, object]:
    config = checkpoint.label_config
    return {
        "frame_rate_hz": config.frame_rate_hz,
        "bin_edges_s": list(config.bin_edges_s),
        "active_ratio": config.active_ratio,
        "now_bins": list(config.now_bins),
        "future_bins": list(config.future_bins),
        "horizon_frames": config.horizon_frames,
        "bin_frame_ranges": [list(bounds) for bounds in config.bin_frame_ranges()],
    }


def prediction_dataframe(
    frame_df: pd.DataFrame,
    checkpoint: LoadedVAPCheckpoint,
    *,
    device: str = "cpu",
    session_ids: set[str] | None = None,
    source_manifest: Mapping[str, object] | None = None,
    source_manifest_sha256: str | None = None,
    override_status: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run causal inference and return the participant-wise VAP artifact.

    Feature-set masking is recovered from the checkpoint, so an ablation is
    evaluated with exactly the same declared inputs that were used to train it.
    Absent padded slots remain in the fixed-width tensor but are explicitly
    zeroed and accompanied by their ``P{i}_present`` mask.
    """
    if session_ids is not None:
        available = set(frame_df["session_id"].astype(str).unique())
        missing = sorted(session_ids - available)
        if missing:
            raise ValueError(f"Prediction data is missing requested sessions: {missing}")
        frame_df = frame_df.loc[frame_df["session_id"].astype(str).isin(session_ids)].copy()

    manifest, manifest_hash = _source_binding(source_manifest, source_manifest_sha256)
    target_ready = manifest.get("target_ready") is True

    # Bind both the selected source rows and the model-visible feature view.
    # Feature-set masking below can intentionally zero columns, so hashing only
    # that view would fail to distinguish two different source datasets.
    original_frame_ref = frame_df
    source_frame_df = frame_df.copy(deep=True)
    source_data_hash = _dataframe_sha256(source_frame_df)

    trainer_config = checkpoint.trainer_config or {}
    feature_set = str(trainer_config.get("feature_set", "multimodal"))
    if feature_set != "multimodal":
        from hri_turn_taking.vap.feature_sets import apply_feature_set

        frame_df = apply_feature_set(frame_df, feature_set)

    sessions = sessions_from_dataframe(frame_df, label_config=checkpoint.label_config)
    rows: list[pd.DataFrame] = []
    session_shapes: dict[str, list[int]] = {}
    tensor_columns: list[str] = []
    temporal_availability_columns: list[str] = []
    gold_target_availability_columns: list[str] = []

    for session in sessions:
        probabilities = predict_session_probabilities(
            checkpoint.model,
            session,
            checkpoint.normalizer,
            device=device,
        )
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Non-finite VAP probabilities for session {session.session_id!r}")
        present = session.present > 0
        probabilities[:, ~present, :] = 0.0
        p_now, p_future = p_now_p_future(probabilities, checkpoint.label_config)

        source_rows = (
            frame_df.loc[frame_df["session_id"].astype(str) == session.session_id]
            .sort_values("timestamp_s")
            .reset_index(drop=True)
        )
        if len(source_rows) != probabilities.shape[0]:
            raise ValueError(f"Prediction/source row mismatch for session {session.session_id!r}")
        output: dict[str, object] = {
            "session_id": np.full(len(source_rows), session.session_id, dtype=object),
            "timestamp_s": source_rows["timestamp_s"].to_numpy(dtype=float),
            "full_future_horizon_temporally_available": session.valid.astype(int),
        }
        frame_indices = np.arange(len(source_rows))
        bin_available = [
            frame_indices + end_frame <= len(source_rows)
            for _, end_frame in checkpoint.label_config.bin_frame_ranges()
        ]
        for participant in range(MAX_PARTICIPANTS):
            slot = participant + 1
            output[f"P{slot}_present"] = np.full(len(source_rows), int(present[participant]))
            for bin_index in range(checkpoint.label_config.n_bins):
                column = f"P{slot}_vap_bin{bin_index}"
                output[column] = probabilities[:, participant, bin_index]
                if column not in tensor_columns:
                    tensor_columns.append(column)
                temporal_column = f"{column}_horizon_temporally_available"
                temporal_availability = (
                    bin_available[bin_index] & present[participant]
                )
                output[temporal_column] = temporal_availability.astype(int)
                if temporal_column not in temporal_availability_columns:
                    temporal_availability_columns.append(temporal_column)
                target_column = f"{column}_gold_target_available"
                output[target_column] = (temporal_availability & target_ready).astype(int)
                if target_column not in gold_target_availability_columns:
                    gold_target_availability_columns.append(target_column)
            output[f"P{slot}_p_now"] = p_now[:, participant]
            output[f"P{slot}_p_future"] = p_future[:, participant]
        rows.append(pd.DataFrame(output))
        session_shapes[session.session_id] = list(probabilities.shape)

    table = pd.concat(rows, ignore_index=True)
    if _dataframe_sha256(original_frame_ref) != source_data_hash:
        raise RuntimeError("Prediction input dataframe changed during inference")
    model_visible_columns = [*input_feature_columns(), *present_columns()]
    missing_model_columns = [
        column for column in model_visible_columns if column not in frame_df.columns
    ]
    if missing_model_columns:
        raise ValueError(
            f"Prediction data is missing model-visible columns: {missing_model_columns}"
        )
    model_view_hash = _dataframe_sha256(frame_df.loc[:, model_visible_columns])
    merged_override = _merged_override_status(checkpoint, override_status)
    metadata: dict[str, object] = {
        "format_version": PREDICTION_FORMAT_VERSION,
        "primary_output": "participant_wise_future_voice_activity",
        "tensor_axes": ["time", "participant", "future_bin"],
        "tensor_columns": tensor_columns,
        "full_future_horizon_temporally_available_column": (
            "full_future_horizon_temporally_available"
        ),
        "horizon_temporal_availability_columns": temporal_availability_columns,
        "horizon_temporal_availability_semantics": (
            "1 iff the participant slot exists and the complete future-bin interval "
            "is contained in the recorded session; this does not assert that a gold "
            "target source exists"
        ),
        "gold_target_availability_columns": gold_target_availability_columns,
        "gold_target_availability_semantics": (
            "1 iff horizon temporal availability is 1 and source_manifest.target_ready "
            "is exactly true"
        ),
        "source_target_ready": target_ready,
        "session_tensor_shapes": session_shapes,
        "participant_slots": [f"P{i}" for i in range(1, MAX_PARTICIPANTS + 1)],
        "bin_edges_s": list(checkpoint.label_config.bin_edges_s),
        "label_config": _label_config_metadata(checkpoint),
        "feature_set": feature_set,
        "exported_session_ids": [session.session_id for session in sessions],
        "input_dataframe_sha256": source_data_hash,
        "model_input_view_sha256": model_view_hash,
        "source_manifest": manifest,
        "source_manifest_sha256": manifest_hash,
        "artifact_status": merged_override["artifact_status"],
        "result_role": merged_override["result_role"],
        "counted_experiment_eligible": False,
        "override_status": merged_override,
        "auxiliary_current_vad_exported": False,
    }
    return table, metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(target: Path, artifact_id: str, purpose: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.{artifact_id}.{purpose}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _commit_prediction_pair(
    csv_temp: Path,
    metadata_temp: Path,
    output_file: Path,
    metadata_file: Path,
    *,
    artifact_id: str,
    overwrite: bool,
) -> None:
    """Two-phase, fail-closed commit with metadata written as the commit marker.

    Two independent ``os.replace`` calls cannot make the pair filesystem-atomic.
    Instead, the metadata is committed last and binds the CSV digest. A crash
    after the CSV replace therefore leaves no valid sidecar for the new CSV.
    """
    csv_existed = output_file.exists()
    metadata_existed = metadata_file.exists()
    if not overwrite and (csv_existed or metadata_existed):
        existing = [str(path) for path in (output_file, metadata_file) if path.exists()]
        raise FileExistsError(
            f"Refusing to overwrite existing prediction artifact files: {existing}"
        )

    csv_backup: Path | None = None
    csv_committed = False
    restore_failed = False
    try:
        if overwrite and csv_existed:
            csv_backup = _temporary_path(output_file, artifact_id, "backup")
            shutil.copy2(output_file, csv_backup)
        os.replace(csv_temp, output_file)
        csv_committed = True
        # This sidecar is the commit marker. Its prediction_csv_sha256 binds it
        # to the CSV, so a process crash between replaces fails closed rather
        # than presenting a silently mixed pair as valid.
        os.replace(metadata_temp, metadata_file)
    except Exception:
        try:
            if csv_committed:
                if csv_backup is not None and csv_backup.exists():
                    os.replace(csv_backup, output_file)
                else:
                    output_file.unlink(missing_ok=True)
        except OSError as restore_error:
            restore_failed = True
            raise RuntimeError(
                "Prediction artifact commit failed and the prior CSV could not be restored; "
                f"recovery copy retained at {csv_backup}"
            ) from restore_error
        raise
    finally:
        csv_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
        if csv_backup is not None and not restore_failed:
            csv_backup.unlink(missing_ok=True)


def write_prediction_artifact(
    frame_df: pd.DataFrame,
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_snapshot: PredictionCheckpointSnapshot | None = None,
    output_path: str | Path,
    device: str = "cpu",
    validation_only: bool = False,
    overwrite: bool = False,
    source_manifest: Mapping[str, object] | None = None,
    source_manifest_sha256: str | None = None,
    override_status: Mapping[str, object] | None = None,
) -> PredictionArtifact:
    """Stage and two-phase commit a fail-closed CSV/JSON prediction pair.

    Pass a preloaded ``checkpoint_snapshot`` when the caller also needs the
    checkpoint for readiness decisions. This guarantees readiness selection,
    inference, and the recorded checkpoint digest all refer to one byte snapshot.
    ``checkpoint_path`` remains as a convenience and is loaded exactly once here.
    """
    if (checkpoint_path is None) == (checkpoint_snapshot is None):
        raise ValueError("Provide exactly one of checkpoint_path or checkpoint_snapshot")
    output_file = Path(output_path)
    if output_file.suffix.lower() != ".csv":
        raise ValueError("Participant-wise VAP prediction output must be a .csv file")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    metadata_file = output_file.with_suffix(".metadata.json")
    if not overwrite and (output_file.exists() or metadata_file.exists()):
        existing = [str(path) for path in (output_file, metadata_file) if path.exists()]
        raise FileExistsError(
            f"Refusing to overwrite existing prediction artifact files: {existing}"
        )
    if checkpoint_snapshot is None:
        assert checkpoint_path is not None
        snapshot = load_prediction_checkpoint_snapshot(checkpoint_path, device=device)
    else:
        snapshot = checkpoint_snapshot
    checkpoint_file = snapshot.path
    checkpoint = snapshot.checkpoint
    session_ids = None
    if validation_only:
        split = checkpoint.data_split or {}
        validation_sessions = split.get("inner_validation_session_ids") or split.get(
            "val_session_ids"
        )
        if not validation_sessions:
            raise ValueError("Checkpoint does not record inner-validation session IDs")
        session_ids = {str(session_id) for session_id in validation_sessions}
    manifest, _ = _source_binding(source_manifest, source_manifest_sha256)
    _verify_source_artifact_hashes(manifest)
    frame_snapshot = frame_df.copy(deep=True)
    frame_snapshot_hash = _dataframe_sha256(frame_snapshot)
    table, metadata = prediction_dataframe(
        frame_snapshot,
        checkpoint,
        device=device,
        session_ids=session_ids,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        override_status=override_status,
    )
    if _dataframe_sha256(frame_snapshot) != frame_snapshot_hash:
        raise RuntimeError("Prediction input dataframe changed during inference")
    _verify_source_artifact_hashes(manifest)
    metadata["validation_only"] = validation_only
    metadata["checkpoint_name"] = checkpoint_file.name
    metadata["checkpoint_sha256"] = snapshot.sha256
    metadata["overwrite_requested"] = overwrite
    artifact_id = str(uuid.uuid4())
    metadata["artifact_id"] = artifact_id

    output_file.parent.mkdir(parents=True, exist_ok=True)
    csv_temp = _temporary_path(output_file, artifact_id, "csv")
    metadata_temp: Path | None = None
    try:
        metadata_temp = _temporary_path(metadata_file, artifact_id, "metadata")
        table.to_csv(csv_temp, index=False)
        metadata["prediction_csv_sha256"] = _sha256_file(csv_temp)
        metadata = _standard_json_object(metadata, "prediction metadata")
        metadata_temp.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        _commit_prediction_pair(
            csv_temp,
            metadata_temp,
            output_file,
            metadata_file,
            artifact_id=artifact_id,
            overwrite=overwrite,
        )
    finally:
        csv_temp.unlink(missing_ok=True)
        if metadata_temp is not None:
            metadata_temp.unlink(missing_ok=True)
    return PredictionArtifact(
        table=table,
        metadata=metadata,
        output_path=output_file,
        metadata_path=metadata_file,
    )
