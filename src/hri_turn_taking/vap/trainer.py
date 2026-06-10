"""Training loop for the multiparty VAP model.

Design points required by the project conventions:
- session-level train/validation split (never split within a session),
- masked BCE objective: frames whose future horizon is truncated and absent
  participant slots contribute no gradient,
- per-bin pos_weight to counter the heavy class imbalance of future-activity
  labels,
- evaluation with balanced metrics (mAP, balanced accuracy) plus the standard
  VAP event tests (shift/hold, shift-prediction) — never raw accuracy alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from hri_turn_taking.data.multiparty import MAX_PARTICIPANTS, input_feature_columns
from hri_turn_taking.vap.dataset import (
    FeatureNormalizer,
    SessionTensors,
    WindowDataset,
    sessions_from_dataframe,
    split_sessions,
)
from hri_turn_taking.vap.events import evaluate_shift_hold, evaluate_shift_prediction
from hri_turn_taking.vap.labels import DEFAULT_LABEL_CONFIG, VAPLabelConfig
from hri_turn_taking.vap.model import MultipartyVAP, VAPModelConfig


@dataclass(frozen=True)
class VAPTrainerConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    window_frames: int = 250  # 10 s at 25 fps
    stride_frames: int = 125
    validation_fraction: float = 0.25
    early_stopping_patience: int = 4
    aux_vad_weight: float = 0.3
    gradient_clip: float = 1.0
    random_seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"


DEFAULT_TRAINER_CONFIG = VAPTrainerConfig()


@dataclass
class VAPTrainingResult:
    checkpoint_path: Path
    frame_metrics: dict[str, float]
    event_metrics: dict[str, dict[str, float]]
    history: list[dict[str, float]] = field(default_factory=list)
    n_train_sessions: int = 0
    n_val_sessions: int = 0


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
) -> VAPTrainingResult:
    torch.manual_seed(trainer_config.random_seed)
    device = resolve_device(trainer_config.device)

    sessions = sessions_from_dataframe(
        frame_df, label_config=label_config, max_participants=max_participants
    )
    train_sessions, val_sessions = split_sessions(
        sessions,
        validation_fraction=trainer_config.validation_fraction,
        random_seed=trainer_config.random_seed,
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
        input_dim=len(input_feature_columns(max_participants)),
        n_participants=max_participants,
        n_bins=label_config.n_bins,
        d_model=d_model,
        n_layers=n_layers,
        encoder=encoder,
    )
    model = MultipartyVAP(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
    )
    pos_weight = _bin_pos_weight(train_sessions).to(device)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(trainer_config.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = _batch_loss(model, batch, device, pos_weight, trainer_config.aux_vad_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_config.gradient_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += _batch_loss(
                    model, batch, device, pos_weight, trainer_config.aux_vad_weight
                ).item()
        val_loss /= max(1, len(val_loader))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= trainer_config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()

    frame_metrics, event_metrics = evaluate_model(
        model, val_sessions, normalizer, label_config=label_config, device=device
    )

    output_path = Path(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "label_config": asdict(label_config),
            "normalizer": normalizer.to_dict(),
            "feature_columns": input_feature_columns(max_participants),
            "frame_metrics": frame_metrics,
            "event_metrics": event_metrics,
            "history": history,
        },
        output_path,
    )
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {"frame": frame_metrics, "events": event_metrics, "history": history}, indent=2
        ),
        encoding="utf-8",
    )

    return VAPTrainingResult(
        checkpoint_path=output_path,
        frame_metrics=frame_metrics,
        event_metrics=event_metrics,
        history=history,
        n_train_sessions=len(train_sessions),
        n_val_sessions=len(val_sessions),
    )


def load_vap_checkpoint(
    checkpoint_path: str | Path, device: str = "cpu"
) -> tuple[MultipartyVAP, FeatureNormalizer, VAPLabelConfig]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MultipartyVAP(VAPModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    normalizer = FeatureNormalizer.from_dict(payload["normalizer"])
    label_payload = dict(payload["label_config"])
    for key in ("bin_edges_s", "now_bins", "future_bins"):
        label_payload[key] = tuple(label_payload[key])
    return model, normalizer, VAPLabelConfig(**label_payload)


def predict_session_probabilities(
    model: MultipartyVAP,
    session: SessionTensors,
    normalizer: FeatureNormalizer,
    *,
    device: torch.device | str = "cpu",
    context_frames: int = 500,
) -> np.ndarray:
    """Full-session causal forward pass -> (T, n_participants, n_bins).

    Sessions longer than the model's positional table are processed in chunks
    with ``context_frames`` of left context, whose outputs are discarded so
    every retained frame has seen at least that much history.
    """
    features = torch.from_numpy(normalizer.apply(session.features)).unsqueeze(0).to(device)
    n_frames = features.shape[1]
    max_len = model.config.max_len
    if n_frames <= max_len:
        return model.predict_probabilities(features).squeeze(0).cpu().numpy()

    if context_frames >= max_len:
        raise ValueError("context_frames must be smaller than the model max_len")
    outputs = []
    emit_from = 0  # next frame index whose output has not been produced yet
    while emit_from < n_frames:
        chunk_start = max(0, emit_from - context_frames)
        chunk_end = min(n_frames, chunk_start + max_len)
        probs = model.predict_probabilities(features[:, chunk_start:chunk_end]).squeeze(0)
        outputs.append(probs[emit_from - chunk_start :])
        emit_from = chunk_end
    return torch.cat(outputs, dim=0).cpu().numpy()


def evaluate_model(
    model: MultipartyVAP,
    sessions: list[SessionTensors],
    normalizer: FeatureNormalizer,
    *,
    label_config: VAPLabelConfig = DEFAULT_LABEL_CONFIG,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    all_scores, all_labels = [], []
    shift_hold_results, shift_pred_results = [], []

    for session in sessions:
        probs = predict_session_probabilities(model, session, normalizer, device=device)
        valid = session.valid
        present_mask = session.present > 0
        all_scores.append(probs[valid][:, present_mask, :].reshape(-1))
        all_labels.append(session.labels[valid][:, present_mask, :].reshape(-1))

        active = probs[:, present_mask, :]
        vad_active = session.vad[:, present_mask]
        shift_hold = evaluate_shift_hold(
            active, vad_active, frame_rate_hz=label_config.frame_rate_hz, label_config=label_config
        )
        if shift_hold is not None:
            shift_hold_results.append(shift_hold)
        shift_pred = evaluate_shift_prediction(
            active, vad_active, frame_rate_hz=label_config.frame_rate_hz, label_config=label_config
        )
        if shift_pred is not None:
            shift_pred_results.append(shift_pred)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    predictions = (scores >= 0.5).astype(int)
    frame_metrics = {
        "mAP": float(average_precision_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "positive_rate": float(labels.mean()),
    }

    event_metrics: dict[str, dict[str, float]] = {}
    for name, results in (
        ("shift_hold", shift_hold_results),
        ("shift_prediction", shift_pred_results),
    ):
        if results:
            event_metrics[name] = {
                "n_events": int(sum(r.n_events for r in results)),
                "n_shifts": int(sum(r.n_shifts for r in results)),
                "balanced_accuracy": float(np.mean([r.balanced_accuracy for r in results])),
                "f1": float(np.mean([r.f1 for r in results])),
                "average_precision": float(
                    np.nanmean([r.average_precision for r in results])
                ),
            }
    return frame_metrics, event_metrics


def _batch_loss(
    model: MultipartyVAP,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    pos_weight: torch.Tensor,
    aux_vad_weight: float,
) -> torch.Tensor:
    features = batch["features"].to(device)
    labels = batch["labels"].to(device)
    vad = batch["vad"].to(device)
    valid = batch["valid"].to(device)  # (B, T)
    present = batch["present"].to(device)  # (B, N)

    outputs = model(features)
    mask = valid.unsqueeze(-1) * present.unsqueeze(1)  # (B, T, N)

    bin_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["bin_logits"], labels, pos_weight=pos_weight, reduction="none"
    )
    bin_loss = (bin_loss * mask.unsqueeze(-1)).sum() / mask.unsqueeze(-1).sum().clamp(min=1.0)

    vad_mask = present.unsqueeze(1).expand_as(vad)
    vad_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["vad_logits"], vad, reduction="none"
    )
    vad_loss = (vad_loss * vad_mask).sum() / vad_mask.sum().clamp(min=1.0)

    return bin_loss + aux_vad_weight * vad_loss


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
