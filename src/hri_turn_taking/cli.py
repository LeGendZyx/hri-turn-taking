from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from hri_turn_taking.audio.gcc_phat import doa_from_tdoa, estimate_tdoa_gcc_phat
from hri_turn_taking.config import ProjectConfig, load_config
from hri_turn_taking.data.synthetic import save_synthetic_turn_taking_data
from hri_turn_taking.features.fusion import FrameFeatures
from hri_turn_taking.models.inference import predict_trp_from_csv
from hri_turn_taking.models.trainer import train_trp_classifier
from hri_turn_taking.realtime.pipeline import PredictiveTurnTakingPipeline
from hri_turn_taking.vision.mouth import mouth_aspect_ratio

app = typer.Typer(help="Predictive turn-taking HRI utilities.")
console = Console()


@app.command()
def demo(config: Path = Path("configs/default.yaml")) -> None:
    """Run a tiny synthetic end-to-end pipeline demo."""
    cfg: ProjectConfig = load_config(config)

    sample_rate = cfg.audio.sample_rate
    delay_samples = 4
    reference = np.zeros(512)
    signal = np.zeros(512)
    reference[128] = 1.0
    signal[128 + delay_samples] = 1.0

    tdoa = estimate_tdoa_gcc_phat(signal, reference, sample_rate=sample_rate)
    doa = doa_from_tdoa(
        tdoa,
        mic_distance_m=cfg.audio.mic_distance_m,
        speed_of_sound_mps=cfg.audio.speed_of_sound_mps,
    )

    landmarks = np.zeros((309, 3), dtype=float)
    landmarks[13] = [0.0, 0.04, 0.0]
    landmarks[14] = [0.0, -0.04, 0.0]
    landmarks[37] = [-0.02, 0.02, 0.0]
    landmarks[84] = [-0.02, -0.02, 0.0]
    landmarks[78] = [-0.08, 0.0, 0.0]
    landmarks[308] = [0.08, 0.0, 0.0]
    mar = mouth_aspect_ratio(landmarks)

    frame = FrameFeatures(
        timestamp_s=0.0,
        doa_angle_deg=float(doa),
        gaze_vector=[0.0, 0.0, 1.0],
        mouth_aspect_ratio=float(mar),
        audio_vad=True,
        speaker_id="participant_1",
    )

    pipeline = PredictiveTurnTakingPipeline(
        trp_threshold=cfg.turn_taking.trp_threshold,
        mar_threshold=cfg.vision.mouth_aspect_ratio_threshold,
    )
    decision = pipeline.step(frame, robot_has_intent=True)

    console.print("[bold green]Demo complete[/bold green]")
    console.print(f"TDOA: {tdoa:.6f}s")
    console.print(f"DoA: {doa:.2f} degrees")
    console.print(f"MAR: {mar:.3f}")
    console.print(f"TRP probability: {decision.trp_probability:.2f}")
    console.print(f"Robot should speak: {decision.should_speak}")


@app.command()
def show_config(config: Path = Path("configs/default.yaml")) -> None:
    """Print the loaded project configuration."""
    cfg = load_config(config)
    console.print(cfg.model_dump())


@app.command()
def generate_synthetic(
    output: Path = Path("data/processed/synthetic_turn_taking.csv"),
    config: Path = Path("configs/default.yaml"),
    sessions: int = 8,
    duration_s: float = 120.0,
    participants: int = 3,
) -> None:
    """Generate a synthetic frame-level dataset for early model development."""
    cfg = load_config(config)
    frame_df = save_synthetic_turn_taking_data(
        output,
        n_sessions=sessions,
        duration_s=duration_s,
        frame_rate_hz=cfg.fusion.frame_rate_hz,
        n_participants=participants,
        random_seed=cfg.project.random_seed,
    )
    positive_rate = frame_df[cfg.training.target_column].mean()
    console.print("[bold green]Synthetic dataset generated[/bold green]")
    console.print(f"Path: {output}")
    console.print(f"Rows: {len(frame_df)}")
    console.print(f"TRP positive rate: {positive_rate:.3f}")


@app.command()
def train(
    data: Path = Path("data/processed/synthetic_turn_taking.csv"),
    output: Path = Path("models/trp_baseline.joblib"),
    config: Path = Path("configs/default.yaml"),
    model_type: str | None = None,
    window_seconds: float | None = None,
) -> None:
    """Train a baseline TRP classifier from frame-level multimodal features."""
    cfg = load_config(config)
    selected_model_type = model_type or cfg.training.model_type
    selected_window_seconds = window_seconds or cfg.training.window_seconds
    window_size = max(1, round(selected_window_seconds * cfg.fusion.frame_rate_hz))

    result = train_trp_classifier(
        dataset_path=data,
        model_path=output,
        feature_columns=cfg.training.feature_columns,
        target_column=cfg.training.target_column,
        model_type=selected_model_type,
        window_size=window_size,
        test_size=cfg.training.test_size,
        random_seed=cfg.project.random_seed,
    )

    console.print("[bold green]Training complete[/bold green]")
    console.print(f"Model: {result.model_type}")
    console.print(f"Saved to: {result.model_path}")
    console.print(f"Train examples: {result.n_train}")
    console.print(f"Test examples: {result.n_test}")
    console.print(f"Dataset positive rate: {result.positive_rate:.3f}")
    console.print("Metrics:")
    for name, value in result.metrics.items():
        console.print(f"  {name}: {value:.3f}")


@app.command()
def predict(
    data: Path,
    model: Path = Path("models/trp_baseline.joblib"),
    output: Path = Path("outputs/trp_predictions.csv"),
) -> None:
    """Predict TRP probabilities for a frame-level feature CSV."""
    result = predict_trp_from_csv(dataset_path=data, model_path=model, output_path=output)
    console.print("[bold green]Prediction complete[/bold green]")
    console.print(f"Rows: {len(result)}")
    console.print(f"Saved to: {output}")


@app.command()
def vap_generate(
    output: Path = Path("data/processed/multiparty_synthetic.csv"),
    sessions: int = 12,
    duration_s: float = 180.0,
    participants: int = 3,
    seed: int = 42,
) -> None:
    """Generate a naive multiparty dataset in the per-participant VAP schema."""
    from hri_turn_taking.data.multiparty import SyntheticConfig, save_multiparty_dataset

    config = SyntheticConfig(
        n_sessions=sessions,
        duration_s=duration_s,
        n_participants=participants,
        random_seed=seed,
    )
    frame_df = save_multiparty_dataset(output, config)
    vad_cols = [c for c in frame_df.columns if c.endswith("_vad")]
    console.print("[bold green]Multiparty synthetic dataset generated[/bold green]")
    console.print(f"Path: {output}")
    console.print(f"Rows: {len(frame_df)}  Sessions: {sessions}  Participants: {participants}")
    activity = frame_df[vad_cols].mean().round(3).to_dict()
    console.print(f"Speech activity rate per participant: {activity}")


@app.command()
def vap_train(
    data: Path = Path("data/processed/multiparty_synthetic.csv"),
    output: Path = Path("models/vap_multiparty.pt"),
    encoder: str = "transformer",
    epochs: int = 20,
    d_model: int = 64,
    n_layers: int = 2,
    window_seconds: float = 10.0,
    seed: int = 42,
) -> None:
    """Train the multiparty VAP model (future voice-activity bin prediction)."""
    from hri_turn_taking.data.multiparty import load_multiparty_dataset
    from hri_turn_taking.vap.labels import VAPLabelConfig
    from hri_turn_taking.vap.trainer import VAPTrainerConfig, train_vap_model

    label_config = VAPLabelConfig()
    frame_df = load_multiparty_dataset(data)
    trainer_config = VAPTrainerConfig(
        epochs=epochs,
        window_frames=max(1, round(window_seconds * label_config.frame_rate_hz)),
        stride_frames=max(1, round(window_seconds * label_config.frame_rate_hz / 2)),
        random_seed=seed,
    )
    result = train_vap_model(
        frame_df,
        checkpoint_path=output,
        label_config=label_config,
        trainer_config=trainer_config,
        encoder=encoder,
        d_model=d_model,
        n_layers=n_layers,
    )

    console.print("[bold green]VAP training complete[/bold green]")
    console.print(f"Checkpoint: {result.checkpoint_path}")
    console.print(
        f"Sessions: {result.n_train_sessions} train / {result.n_val_sessions} validation"
    )
    console.print(f"Frame metrics: {result.frame_metrics}")
    for event_name, metrics in result.event_metrics.items():
        console.print(f"{event_name}: {metrics}")


@app.command()
def vap_demo(
    data: Path = Path("data/processed/multiparty_synthetic.csv"),
    model: Path = Path("models/vap_multiparty.pt"),
    silence_baseline_s: float = 1.5,
) -> None:
    """Compare predictive dual-threshold policy vs. reactive silence baseline."""
    from hri_turn_taking.data.multiparty import load_multiparty_dataset
    from hri_turn_taking.vap.dataset import sessions_from_dataframe
    from hri_turn_taking.vap.offline_demo import compare_policies_on_session
    from hri_turn_taking.vap.trainer import load_vap_checkpoint

    vap_model, normalizer, label_config = load_vap_checkpoint(model)
    frame_df = load_multiparty_dataset(data)
    sessions = sessions_from_dataframe(frame_df, label_config=label_config)

    console.print("[bold green]Predictive vs. reactive floor-entry comparison[/bold green]")
    for session in sessions[:3]:
        comparison = compare_policies_on_session(
            session,
            vap_model,
            normalizer,
            label_config=label_config,
            silence_baseline_s=silence_baseline_s,
        )
        if comparison is None:
            console.print(f"{session.session_id}: no scoreable silence gaps")
            continue
        console.print(
            f"{session.session_id}: gaps={comparison.n_gaps} | "
            f"predictive entries={comparison.predictive_entries} "
            f"latency={comparison.predictive_mean_latency_s:.2f}s | "
            f"reactive entries={comparison.reactive_entries} "
            f"latency={comparison.reactive_mean_latency_s:.2f}s"
        )
