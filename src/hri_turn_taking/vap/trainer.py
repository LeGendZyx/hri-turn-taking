"""Training loop for the multiparty VAP model.

Design points required by the project conventions:
- session-level train/validation split (never split within a session),
- masked BCE objective: frames whose future horizon is truncated and absent
  participant slots contribute no gradient,
- per-bin pos_weight to counter the heavy class imbalance of future-activity
  labels,
- AdamW (no weight decay on norms/biases) with linear warmup + cosine decay,
  early stopping on validation macro per-bin mAP,
- evaluation with balanced metrics (macro mAP, micro AP and per-bin metrics,
  accuracy) plus the VAP event tests (shift/hold, shift-prediction,
  overlap-resolution), pooled across validation sessions — never raw
  accuracy alone.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from hri_turn_taking.data.multiparty import (
    FEATURE_SCHEMA_V2,
    MAX_PARTICIPANTS,
    input_feature_columns,
)
from hri_turn_taking.vap.dataset import (
    TRANSFORMED_GLOBAL_FEATURES,
    TRANSFORMED_PARTICIPANT_FEATURES,
    FeatureNormalizer,
    SessionTensors,
    WindowDataset,
    sessions_from_dataframe,
    split_sessions,
)
from hri_turn_taking.vap.events import (
    next_speaker_scores,
    overlap_resolution_scores,
    robot_entry_scores,
    score_events,
    score_next_speaker,
    shift_hold_scores,
    shift_prediction_scores,
)
from hri_turn_taking.vap.feature_sets import (
    active_feature_columns,
    apply_feature_set,
    feature_set_metadata,
    get_feature_set,
)
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG, VAPLabelConfig
from hri_turn_taking.vap.model import MultipartyVAP, VAPModelConfig

# v4 adds explicit development/smoke status, source-manifest provenance and
# train/inner-validation/outer-test split roles. Older artifacts must be
# retrained because they cannot establish whether an override affected a run.
CHECKPOINT_FORMAT_VERSION = 4


@dataclass(frozen=True)
class VAPTrainerConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.1
    window_frames: int = 250  # 10 s at 25 fps; also the model's attention window
    stride_frames: int = 125
    validation_fraction: float = 0.25
    # Group-level (person-disjoint) splits: sessions named `<group><delim><scenario>`
    # stay on one side. None = plain session-level split (synthetic default).
    group_delimiter: str | None = None
    early_stopping_patience: int = 4
    aux_vad_weight: float = 0.3
    feature_set: str = "multimodal"
    robot_slot: int = 0  # zero-based; the real-data demonstrator is P1
    gradient_clip: float = 1.0
    random_seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"


DEFAULT_TRAINER_CONFIG = VAPTrainerConfig()


@dataclass
class VAPTrainingResult:
    checkpoint_path: Path
    frame_metrics: dict[str, object]
    event_metrics: dict[str, dict[str, object]]
    history: list[dict[str, object]] = field(default_factory=list)
    n_train_sessions: int = 0
    n_val_sessions: int = 0
    result_role: str = "development_inner_validation"
    counted_experiment_eligible: bool = False


@dataclass
class LoadedVAPCheckpoint:
    model: MultipartyVAP
    normalizer: FeatureNormalizer
    label_config: VAPLabelConfig
    trainer_config: dict | None
    data_split: dict | None = None
    run_metadata: dict | None = None
    data_manifest: dict | None = None


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_vap_model(
    frame_df: pd.DataFrame,
    *,
    checkpoint_path: str | Path,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    trainer_config: VAPTrainerConfig = DEFAULT_TRAINER_CONFIG,
    encoder: str = "transformer",
    d_model: int = 64,
    n_layers: int = 2,
    max_participants: int = MAX_PARTICIPANTS,
    run_metadata: dict[str, object] | None = None,
    data_manifest: dict[str, object] | None = None,
    overwrite: bool = False,
) -> VAPTrainingResult:
    output_path = Path(checkpoint_path)
    metrics_path = output_path.with_suffix(".metrics.json")
    existing = [path for path in (output_path, metrics_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing training artifact(s): "
            + ", ".join(str(path) for path in existing)
        )
    effective_run_metadata = _development_run_metadata(run_metadata)
    effective_data_manifest = _strict_json_data(data_manifest or {}, name="data_manifest")
    get_feature_set(trainer_config.feature_set)
    if not 0 <= trainer_config.robot_slot < max_participants:
        raise ValueError(
            f"robot_slot must be a zero-based slot in [0, {max_participants - 1}]"
        )
    if trainer_config.aux_vad_weight < 0:
        raise ValueError("aux_vad_weight must be >= 0 (use 0 for the clean VAP-only run)")
    torch.manual_seed(trainer_config.random_seed)
    device = resolve_device(trainer_config.device)

    experiment_df = apply_feature_set(frame_df, trainer_config.feature_set)
    sessions = sessions_from_dataframe(
        experiment_df, label_config=label_config, max_participants=max_participants
    )
    train_sessions, val_sessions = split_sessions(
        sessions,
        validation_fraction=trainer_config.validation_fraction,
        random_seed=trainer_config.random_seed,
        group_delimiter=trainer_config.group_delimiter,
    )
    data_split = _split_metadata(
        train_sessions,
        val_sessions,
        group_delimiter=trainer_config.group_delimiter,
    )
    normalizer = FeatureNormalizer.fit(train_sessions)

    train_loader = DataLoader(
        WindowDataset(
            train_sessions,
            normalizer,
            window_frames=trainer_config.window_frames,
            stride_frames=trainer_config.stride_frames,
        ),
        batch_size=trainer_config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        WindowDataset(
            val_sessions,
            normalizer,
            window_frames=trainer_config.window_frames,
            stride_frames=trainer_config.window_frames,
        ),
        batch_size=trainer_config.batch_size,
        shuffle=False,
    )
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError(
            "Not enough frames for the configured window size; "
            "reduce window_frames or provide longer sessions"
        )

    model_config = VAPModelConfig(
        n_participant_features=len(TRANSFORMED_PARTICIPANT_FEATURES),
        n_global_features=len(TRANSFORMED_GLOBAL_FEATURES),
        n_participants=max_participants,
        n_bins=label_config.n_bins,
        d_model=d_model,
        n_layers=n_layers,
        encoder=encoder,
        attention_window=trainer_config.window_frames,
    )
    model = MultipartyVAP(model_config).to(device)

    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": trainer_config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=trainer_config.learning_rate,
    )
    total_steps = max(1, trainer_config.epochs * len(train_loader))
    warmup_steps = max(1, round(trainer_config.warmup_fraction * total_steps))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.45 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    pos_weight = _bin_pos_weight(train_sessions).to(device)

    history: list[dict[str, object]] = []
    best_val_map = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0

    for epoch in range(trainer_config.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _, _, _ = _batch_forward(
                model, batch, device, pos_weight, trainer_config.aux_vad_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_config.gradient_clip)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                loss, bin_logits, labels, mask = _batch_forward(
                    model, batch, device, pos_weight, trainer_config.aux_vad_weight
                )
                val_loss += loss.item()
                # Index the (B,T,N) mask while retaining the final bin axis;
                # flattening all four bins here would turn mAP into micro-AP.
                keep = (mask > 0).cpu().numpy()
                val_probs.append(torch.sigmoid(bin_logits).cpu().numpy()[keep])
                val_labels.append(labels.cpu().numpy()[keep])
        val_loss /= max(1, len(val_loader))
        val_frame_metrics = compute_frame_metrics(
            np.concatenate(val_labels), np.concatenate(val_probs)
        )
        val_map = val_frame_metrics["mAP"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mAP": val_map,
                "val_micro_AP": val_frame_metrics["micro_AP"],
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        if val_map is not None and math.isfinite(val_map) and val_map > best_val_map + 1e-4:
            best_val_map = float(val_map)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= trainer_config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()

    effective_run_metadata = {
        **effective_run_metadata,
        "selection_metric": "inner_validation_mAP",
        "selection_metric_available": best_state is not None,
        "selected_epoch": best_epoch,
        "selection_fallback": None
        if best_state is not None
        else "no_strictly_evaluable_inner_validation_mAP; final_epoch_retained",
    }

    frame_metrics, event_metrics = evaluate_model(
        model,
        val_sessions,
        normalizer,
        label_config=label_config,
        device=device,
        robot_slot=trainer_config.robot_slot,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": model.state_dict(),
        "model_config": model_config.to_dict(),
        "label_config": asdict(label_config),
        "trainer_config": asdict(trainer_config),
        "normalizer": normalizer.to_dict(),
        "feature_meta": {
            "raw_columns": input_feature_columns(max_participants),
            "participant_features": TRANSFORMED_PARTICIPANT_FEATURES,
            "global_features": TRANSFORMED_GLOBAL_FEATURES,
            **feature_set_metadata(trainer_config.feature_set),
            "feature_set": trainer_config.feature_set,
        },
        "data_split": data_split,
        "data_manifest": effective_data_manifest,
        "run_metadata": effective_run_metadata,
        "frame_metrics": frame_metrics,
        "event_metrics": event_metrics,
        "history": history,
    }
    metrics_payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "result_role": "development_inner_validation",
        "counted_experiment_eligible": False,
        "run_metadata": effective_run_metadata,
        "data_manifest": effective_data_manifest,
        "frame": frame_metrics,
        "events": event_metrics,
        "history": history,
        "feature_set": feature_set_metadata(trainer_config.feature_set),
        "data_split": data_split,
    }
    _atomic_torch_save(checkpoint_payload, output_path)
    _atomic_json_write(metrics_payload, metrics_path)

    return VAPTrainingResult(
        checkpoint_path=output_path,
        frame_metrics=frame_metrics,
        event_metrics=event_metrics,
        history=history,
        n_train_sessions=len(train_sessions),
        n_val_sessions=len(val_sessions),
        result_role="development_inner_validation",
        counted_experiment_eligible=False,
    )


def load_vap_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> LoadedVAPCheckpoint:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format v{version} does not match the current artifact contract "
            f"(v{CHECKPOINT_FORMAT_VERSION}: explicit run status and source provenance); "
            "retrain with `hri-turn-taking vap-train`."
        )
    run_metadata = payload.get("run_metadata")
    if (
        not isinstance(run_metadata, dict)
        or run_metadata.get("result_role") != "development_inner_validation"
        or run_metadata.get("counted_experiment_eligible") is not False
        or not isinstance(run_metadata.get("requested_overrides"), list)
        or not isinstance(run_metadata.get("effective_smoke_reasons"), list)
    ):
        raise ValueError(
            "Checkpoint has missing or inconsistent development/smoke metadata; retrain with "
            "`hri-turn-taking vap-train`."
        )
    data_manifest = payload.get("data_manifest")
    if not isinstance(data_manifest, dict):
        raise ValueError(
            "Checkpoint data_manifest is missing or malformed; retrain with "
            "`hri-turn-taking vap-train`."
        )
    # exact schema assertion (Codex R1.3-2): names, raw column order, model
    # feature counts and all four normalizer statistics must match the current
    # schema — a "v3" payload wrapping an old model must fail HERE with a
    # retrain message, not later with a broadcast error or non-finite output
    feature_meta = payload.get("feature_meta", {})
    max_participants = MAX_PARTICIPANTS
    if (
        feature_meta.get("participant_features") != TRANSFORMED_PARTICIPANT_FEATURES
        or feature_meta.get("global_features") != TRANSFORMED_GLOBAL_FEATURES
        or feature_meta.get("raw_columns") != input_feature_columns(max_participants)
    ):
        raise ValueError(
            f"Checkpoint feature_meta {feature_meta.get('participant_features')!r} / "
            f"{feature_meta.get('raw_columns')!r} does not match the current schema; "
            "retrain with `hri-turn-taking vap-train`."
        )
    normalizer_payload = payload.get("normalizer")
    if not isinstance(normalizer_payload, dict):
        raise ValueError(
            "Checkpoint normalizer is missing or malformed; retrain with "
            "`hri-turn-taking vap-train`."
        )
    try:
        normalizer = FeatureNormalizer.from_dict(normalizer_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint normalizer payload is invalid; retrain with "
            "`hri-turn-taking vap-train`."
        ) from exc
    if normalizer.feature_schema != FEATURE_SCHEMA_V2:
        raise ValueError(
            f"Checkpoint v{CHECKPOINT_FORMAT_VERSION} requires the historical "
            f"{FEATURE_SCHEMA_V2!r} normalizer schema, not "
            f"{normalizer.feature_schema!r}; retrain through the separately gated "
            "schema-v3 post-hoc route."
        )
    trainer_payload = payload.get("trainer_config") or {}
    feature_set_name = feature_meta.get(
        "feature_set", trainer_payload.get("feature_set", "multimodal")
    )
    try:
        expected_active_columns = active_feature_columns(feature_set_name)
    except ValueError as exc:
        raise ValueError(
            f"Checkpoint declares unknown feature_set {feature_set_name!r}; retrain with "
            "`hri-turn-taking vap-train`."
        ) from exc
    if (
        ("active_columns" in feature_meta
         and feature_meta["active_columns"] != expected_active_columns)
        or ("name" in feature_meta and feature_meta["name"] != feature_set_name)
        or ("feature_set" in trainer_payload
            and trainer_payload["feature_set"] != feature_set_name)
    ):
        raise ValueError(
            "Checkpoint feature-set metadata is internally inconsistent; retrain with "
            "`hri-turn-taking vap-train`."
        )
    model_config = payload["model_config"]
    if (
        model_config.get("n_participant_features") != len(TRANSFORMED_PARTICIPANT_FEATURES)
        or model_config.get("n_global_features") != len(TRANSFORMED_GLOBAL_FEATURES)
    ):
        raise ValueError(
            "Checkpoint model_config feature counts do not match the current schema; "
            "retrain with `hri-turn-taking vap-train`."
        )
    # cross-field exactness (Codex R1.4-4): a self-consistent 3-slot model or a
    # bin-count that disagrees with the label config would load fine and crash
    # (or mis-read probabilities) only at inference
    if model_config.get("n_participants") != MAX_PARTICIPANTS:
        raise ValueError(
            f"Checkpoint model has n_participants={model_config.get('n_participants')!r} "
            f"but the wide schema is {MAX_PARTICIPANTS}-slot; retrain with "
            "`hri-turn-taking vap-train`."
        )
    label_bins = len(payload["label_config"]["bin_edges_s"])
    if model_config.get("n_bins") != label_bins:
        raise ValueError(
            f"Checkpoint model predicts {model_config.get('n_bins')!r} bins but its "
            f"label_config defines {label_bins}; retrain with `hri-turn-taking vap-train`."
        )
    model = MultipartyVAP(VAPModelConfig(**model_config))
    model.load_state_dict(payload["model_state"])
    model.eval()
    expected_shapes = {
        "participant_mean": len(TRANSFORMED_PARTICIPANT_FEATURES),
        "participant_std": len(TRANSFORMED_PARTICIPANT_FEATURES),
        "global_mean": len(TRANSFORMED_GLOBAL_FEATURES),
        "global_std": len(TRANSFORMED_GLOBAL_FEATURES),
    }
    for name, width in expected_shapes.items():
        stats = getattr(normalizer, name)
        if stats.shape != (width,) or not np.isfinite(stats).all():
            raise ValueError(
                f"Checkpoint normalizer {name} has shape {stats.shape} or non-finite "
                f"values (expected finite ({width},)); retrain with `hri-turn-taking "
                "vap-train`."
            )
        if name.endswith("_std") and (stats <= 0).any():
            raise ValueError(
                f"Checkpoint normalizer {name} contains non-positive std; retrain with "
                "`hri-turn-taking vap-train`."
            )
    label_payload = dict(payload["label_config"])
    for key in ("bin_edges_s", "now_bins", "future_bins"):
        label_payload[key] = tuple(label_payload[key])
    return LoadedVAPCheckpoint(
        model=model,
        normalizer=normalizer,
        label_config=VAPLabelConfig(**label_payload),
        trainer_config=payload.get("trainer_config"),
        data_split=payload.get("data_split"),
        run_metadata=run_metadata,
        data_manifest=data_manifest,
    )


def predict_session_probabilities(
    model: MultipartyVAP,
    session: SessionTensors,
    normalizer: FeatureNormalizer,
    *,
    device: torch.device | str = "cpu",
    chunk_frames: int = 2000,
) -> np.ndarray:
    """Full-session causal forward pass -> (T, n_participants, n_bins).

    Long sessions are processed in chunks with discarded left context of
    ``attention_window * n_layers`` frames — the receptive field of the
    stacked banded attention — so chunked inference is *identical* to a
    full-session pass for the transformer. A GRU has an unbounded recurrent
    state, and the current model forward API does not expose that state across
    chunks; GRU sessions therefore deliberately use one full forward pass
    rather than silently resetting hidden state at chunk boundaries.
    """
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be >= 1")
    features = torch.from_numpy(normalizer.apply(session.features)).unsqueeze(0).to(device)
    # Some dataframe/serialization paths expose a read-only NumPy view;
    # PyTorch tensors must not alias it because that produces undefined-write
    # warnings even though inference itself does not mutate the mask.
    present = torch.from_numpy(np.array(session.present, copy=True)).unsqueeze(0).to(device)
    n_frames = features.shape[1]
    if model.config.encoder == "gru" or n_frames <= chunk_frames:
        return model.predict_probabilities(features, present).squeeze(0).cpu().numpy()

    context = model.config.attention_window * max(1, model.config.n_layers)
    if context >= chunk_frames:
        raise ValueError("chunk_frames must exceed the stacked attention receptive field")
    outputs = []
    emit_from = 0  # next frame index whose output has not been produced yet
    while emit_from < n_frames:
        chunk_start = max(0, emit_from - context)
        chunk_end = min(n_frames, chunk_start + chunk_frames)
        probs = model.predict_probabilities(features[:, chunk_start:chunk_end], present).squeeze(0)
        outputs.append(probs[emit_from - chunk_start :])
        emit_from = chunk_end
    return torch.cat(outputs, dim=0).cpu().numpy()


def compute_frame_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    participant_indices: np.ndarray | None = None,
    max_participants: int = MAX_PARTICIPANTS,
) -> dict[str, object]:
    """Compute support-aware bin metrics without silently dropping single classes.

    The strict ``mAP`` is available only when every future bin has both positive
    and negative examples. ``mAP_evaluable_bins`` is retained as an explicitly
    named diagnostic, never as a substitute for the strict primary value.
    """
    targets = np.asarray(labels, dtype=float)
    probabilities = np.asarray(scores, dtype=float)
    if targets.ndim != 2 or probabilities.shape != targets.shape or targets.shape[0] == 0:
        raise ValueError("labels and scores must be non-empty arrays of shape (samples, bins)")
    if not np.isfinite(targets).all() or not np.isfinite(probabilities).all():
        raise ValueError("labels and scores must be finite")
    if not np.isin(targets, (0.0, 1.0)).all():
        raise ValueError("labels must be binary")

    participant_ids: np.ndarray | None = None
    if participant_indices is not None:
        participant_ids = np.asarray(participant_indices, dtype=int)
        if participant_ids.shape != (targets.shape[0],):
            raise ValueError("participant_indices must have shape (samples,)")
        if (participant_ids < 0).any() or (participant_ids >= max_participants).any():
            raise ValueError("participant_indices contains a slot outside the schema")

    flat_targets = targets.reshape(-1)
    flat_scores = probabilities.reshape(-1)
    flat_predictions = (flat_scores >= 0.5).astype(int)
    bins: dict[str, dict[str, object]] = {}
    per_bin_ap: list[float | None] = []
    per_bin_f1: list[float] = []
    for bin_idx in range(targets.shape[1]):
        summary = _average_precision_summary(targets[:, bin_idx], probabilities[:, bin_idx])
        bin_f1 = float(
            f1_score(targets[:, bin_idx], probabilities[:, bin_idx] >= 0.5, zero_division=0)
        )
        summary["f1"] = bin_f1
        bins[f"bin{bin_idx}"] = summary
        per_bin_ap.append(summary["average_precision"])
        per_bin_f1.append(bin_f1)

    evaluable_ap = [value for value in per_bin_ap if value is not None]
    all_bins_evaluable = len(evaluable_ap) == targets.shape[1]
    micro_summary = _average_precision_summary(flat_targets, flat_scores)
    flat_has_both_classes = np.unique(flat_targets).size == 2
    metrics: dict[str, object] = {
        "mAP": float(np.mean(evaluable_ap)) if all_bins_evaluable else None,
        "mAP_na_reason": None
        if all_bins_evaluable
        else "one_or_more_bins_single_class",
        "mAP_evaluable_bins": float(np.mean(evaluable_ap)) if evaluable_ap else None,
        "n_bins": int(targets.shape[1]),
        "n_bins_evaluable_ap": len(evaluable_ap),
        "micro_AP": micro_summary["average_precision"],
        "micro_AP_na_reason": micro_summary["ap_na_reason"],
        "macro_f1_bins": float(np.mean(per_bin_f1)),
        "balanced_accuracy": float(balanced_accuracy_score(flat_targets, flat_predictions))
        if flat_has_both_classes
        else None,
        "balanced_accuracy_na_reason": None
        if flat_has_both_classes
        else micro_summary["ap_na_reason"],
        "f1": float(f1_score(flat_targets, flat_predictions, zero_division=0)),
        "micro_f1": float(f1_score(flat_targets, flat_predictions, zero_division=0)),
        "positive_rate": float(flat_targets.mean()),
        "n_positive": int(flat_targets.sum()),
        "n_negative": int(flat_targets.size - flat_targets.sum()),
        "bins": bins,
        "participants": {},
        "participant_support_available": participant_ids is not None,
    }
    for bin_idx, (ap, bin_f1) in enumerate(zip(per_bin_ap, per_bin_f1, strict=True)):
        metrics[f"AP_bin{bin_idx}"] = ap
        metrics[f"F1_bin{bin_idx}"] = bin_f1

    if participant_ids is not None:
        participants: dict[str, dict[str, object]] = {}
        participant_maps: list[float] = []
        all_in_scope_evaluable = True
        for participant in range(max_participants):
            selected = participant_ids == participant
            participant_bins: dict[str, dict[str, object]] = {}
            participant_bin_ap: list[float | None] = []
            for bin_idx in range(targets.shape[1]):
                if selected.any():
                    summary = _average_precision_summary(
                        targets[selected, bin_idx], probabilities[selected, bin_idx]
                    )
                else:
                    summary = {
                        "n_positive": 0,
                        "n_negative": 0,
                        "average_precision": None,
                        "ap_na_reason": "not_in_scope",
                    }
                participant_bins[f"bin{bin_idx}"] = summary
                participant_bin_ap.append(summary["average_precision"])
            in_scope = bool(selected.any())
            participant_evaluable = in_scope and all(
                value is not None for value in participant_bin_ap
            )
            participant_map = (
                float(np.mean([value for value in participant_bin_ap if value is not None]))
                if participant_evaluable
                else None
            )
            if in_scope and participant_map is None:
                all_in_scope_evaluable = False
            if participant_map is not None:
                participant_maps.append(participant_map)
            participants[f"P{participant + 1}"] = {
                "in_scope": in_scope,
                "n_samples": int(selected.sum()),
                "mAP": participant_map,
                "mAP_na_reason": None
                if participant_evaluable
                else ("not_in_scope" if not in_scope else "one_or_more_bins_single_class"),
                "bins": participant_bins,
            }
        metrics["participants"] = participants
        metrics["participant_macro_mAP"] = (
            float(np.mean(participant_maps))
            if participant_maps and all_in_scope_evaluable
            else None
        )
        metrics["participant_macro_mAP_na_reason"] = (
            None
            if participant_maps and all_in_scope_evaluable
            else "one_or_more_in_scope_participants_or_bins_not_evaluable"
        )
    return metrics


def evaluate_model(
    model: object,
    sessions: list[SessionTensors],
    normalizer: FeatureNormalizer | None,
    *,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    device: torch.device | str = "cpu",
    robot_slot: int = 0,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Evaluate a learned VAP model or a robot-observable baseline.

    Binary and multiclass event examples are pooled across sessions before
    scoring.  Baselines opt into this same path by exposing
    ``predict_session_probabilities(session, label_config=...)``.
    """
    if not sessions:
        raise ValueError("At least one evaluation session is required")
    if not 0 <= robot_slot < sessions[0].present.size:
        raise ValueError("robot_slot is outside the participant-slot range")

    score_fns = {
        "shift_hold": shift_hold_scores,
        "shift_prediction": shift_prediction_scores,
        "overlap_resolution": overlap_resolution_scores,
    }
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_participant_indices: list[np.ndarray] = []
    pooled: dict[str, tuple[list[int], list[float]]] = {
        name: ([], []) for name in (*score_fns, "robot_entry")
    }
    next_labels: list[int] = []
    next_scores: list[np.ndarray] = []

    for session in sessions:
        probs = _predict_for_evaluation(
            model,
            session,
            normalizer,
            label_config=label_config,
            device=device,
        )
        expected_shape = (
            session.features.shape[0],
            session.present.size,
            label_config.n_bins,
        )
        if probs.shape != expected_shape or not np.isfinite(probs).all():
            raise ValueError(
                f"predictor returned shape {probs.shape}; expected finite {expected_shape}"
            )
        valid = session.valid
        present_mask = session.present > 0
        all_scores.append(probs[valid][:, present_mask, :].reshape(-1, label_config.n_bins))
        all_labels.append(
            session.labels[valid][:, present_mask, :].reshape(-1, label_config.n_bins)
        )
        present_slots = np.flatnonzero(present_mask)
        all_participant_indices.append(np.tile(present_slots, int(valid.sum())))

        for name, score_fn in score_fns.items():
            event_labels, event_scores = score_fn(
                probs,
                session.vad,
                frame_rate_hz=label_config.frame_rate_hz,
                label_config=label_config,
                present=session.present,
            )
            pooled[name][0].extend(event_labels)
            pooled[name][1].extend(event_scores)

        entry_labels, entry_scores = robot_entry_scores(
            probs,
            session.vad,
            robot_slot=robot_slot,
            frame_rate_hz=label_config.frame_rate_hz,
            label_config=label_config,
            present=session.present,
        )
        pooled["robot_entry"][0].extend(entry_labels)
        pooled["robot_entry"][1].extend(entry_scores)

        session_next_labels, session_next_scores = next_speaker_scores(
            probs,
            session.vad,
            frame_rate_hz=label_config.frame_rate_hz,
            label_config=label_config,
            present=session.present,
        )
        next_labels.extend(session_next_labels)
        next_scores.extend(session_next_scores)

    frame_metrics = compute_frame_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_scores),
        participant_indices=np.concatenate(all_participant_indices),
        max_participants=sessions[0].present.size,
    )

    event_metrics: dict[str, dict[str, object]] = {}
    for name, (event_labels, event_scores) in pooled.items():
        evaluation = score_events(np.asarray(event_labels), np.asarray(event_scores), 0.5)
        event_metrics[name] = asdict(evaluation)
    next_score_array = (
        np.asarray(next_scores, dtype=float)
        if next_scores
        else np.empty((0, sessions[0].present.size), dtype=float)
    )
    next_evaluation = score_next_speaker(np.asarray(next_labels), next_score_array)
    event_metrics["next_speaker"] = asdict(next_evaluation)
    return frame_metrics, event_metrics


def _predict_for_evaluation(
    model: object,
    session: SessionTensors,
    normalizer: FeatureNormalizer | None,
    *,
    label_config: VAPLabelConfig,
    device: torch.device | str,
) -> np.ndarray:
    baseline_predict = getattr(model, "predict_session_probabilities", None)
    if callable(baseline_predict):
        return np.asarray(
            baseline_predict(session, label_config=label_config), dtype=np.float32
        )
    if not isinstance(model, MultipartyVAP):
        raise TypeError(
            "model must be MultipartyVAP or expose predict_session_probabilities"
        )
    if normalizer is None:
        raise ValueError("normalizer is required when evaluating MultipartyVAP")
    return predict_session_probabilities(model, session, normalizer, device=device)


def _batch_forward(
    model: MultipartyVAP,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    pos_weight: torch.Tensor,
    aux_vad_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked loss plus the tensors needed for epoch-level validation mAP."""
    features = batch["features"].to(device)
    labels = batch["labels"].to(device)
    vad = batch["vad"].to(device)
    valid = batch["valid"].to(device)  # (B, T)
    present = batch["present"].to(device)  # (B, N)

    outputs = model(features, present=present)
    mask = valid.unsqueeze(-1) * present.unsqueeze(1)  # (B, T, N)

    bin_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["bin_logits"], labels, pos_weight=pos_weight, reduction="none"
    )
    bin_mask = mask.unsqueeze(-1).expand_as(bin_loss)
    bin_loss = (bin_loss * bin_mask).sum() / bin_mask.sum().clamp(min=1.0)

    vad_mask = present.unsqueeze(1).expand_as(vad)
    vad_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["vad_logits"], vad, reduction="none"
    )
    vad_loss = (vad_loss * vad_mask).sum() / vad_mask.sum().clamp(min=1.0)

    loss = bin_loss + aux_vad_weight * vad_loss
    return loss, outputs["bin_logits"], labels, mask


def _average_precision_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    targets = np.asarray(labels, dtype=float)
    n_positive = int(targets.sum())
    n_negative = int(targets.size - n_positive)
    if n_positive == 0:
        value = None
        reason = "no_positive"
    elif n_negative == 0:
        value = None
        reason = "no_negative"
    else:
        value = float(average_precision_score(targets, scores))
        reason = None
    return {
        "n_positive": n_positive,
        "n_negative": n_negative,
        "average_precision": value,
        "ap_na_reason": reason,
    }


def _split_metadata(
    train_sessions: list[SessionTensors],
    val_sessions: list[SessionTensors],
    *,
    group_delimiter: str | None,
) -> dict[str, object]:
    """Exact, serialisable split provenance stored in every new checkpoint."""

    def session_ids(items: list[SessionTensors]) -> list[str]:
        return [session.session_id for session in items]

    def group_ids(items: list[SessionTensors]) -> list[str]:
        explicit = [getattr(session, "group_id", None) for session in items]
        if any(group_id is not None for group_id in explicit):
            if any(group_id is None for group_id in explicit):
                raise ValueError("split metadata found incomplete provenance group_id values")
            return sorted({str(group_id) for group_id in explicit})
        identifiers = session_ids(items)
        if group_delimiter is not None:
            return sorted(
                {identifier.split(group_delimiter, maxsplit=1)[0] for identifier in identifiers}
            )
        return sorted(set(identifiers))

    return {
        "group_delimiter": group_delimiter,
        "train_session_ids": session_ids(train_sessions),
        "inner_validation_session_ids": session_ids(val_sessions),
        "outer_test_session_ids": [],
        # Compatibility alias; its role is explicit above and in result_role.
        "val_session_ids": session_ids(val_sessions),
        "train_group_ids": group_ids(train_sessions),
        "inner_validation_group_ids": group_ids(val_sessions),
        "outer_test_group_ids": [],
        "val_group_ids": group_ids(val_sessions),
        "inner_validation_used_for_model_selection": True,
        "outer_test_used_for_model_selection": False,
        "outer_test_evaluated": False,
        "reported_partition": "inner_validation",
        "result_role": "development_inner_validation",
        "counted_experiment_eligible": False,
    }


def _development_run_metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    supplied = _strict_json_data(metadata or {}, name="run_metadata")
    requested = supplied.get("requested_overrides", [])
    reasons = supplied.get("effective_smoke_reasons", [])
    if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
        raise ValueError("run_metadata.requested_overrides must be a list of strings")
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise ValueError("run_metadata.effective_smoke_reasons must be a list of strings")
    if supplied.get("smoke_only") is True and not reasons:
        reasons = ["caller_marked_smoke_only"]
    smoke_only = bool(reasons)
    return {
        **supplied,
        "requested_overrides": requested,
        "effective_smoke_reasons": reasons,
        "smoke_only": smoke_only,
        "artifact_status": "smoke_only" if smoke_only else "development",
        "result_role": "development_inner_validation",
        "counted_experiment_eligible": False,
        "uses_outer_test": False,
    }


def _strict_json_data(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")

    def normalize(item: object) -> object:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{name} contains a non-finite number")
            return item
        if isinstance(item, np.generic):
            return normalize(item.item())
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError(f"{name} object keys must be strings")
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        raise ValueError(f"{name} contains unsupported value type {type(item).__name__}")

    normalized = normalize(value)
    assert isinstance(normalized, dict)
    return normalized


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_write(payload: dict[str, object], path: Path) -> None:
    normalized = _strict_json_data(payload, name="metrics")
    encoded = json.dumps(normalized, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _bin_pos_weight(sessions: list[SessionTensors], max_weight: float = 10.0) -> torch.Tensor:
    """Per-bin negative/positive ratio over valid frames of present slots."""
    positives = np.zeros(sessions[0].labels.shape[-1])
    totals = np.zeros_like(positives)
    for session in sessions:
        present_mask = session.present > 0
        labels = session.labels[session.valid][:, present_mask, :]
        positives += labels.sum(axis=(0, 1))
        totals += labels.shape[0] * labels.shape[1]
    positives = np.clip(positives, 1.0, None)
    weight = np.clip((totals - positives) / positives, 1.0, max_weight)
    return torch.tensor(weight, dtype=torch.float32)
