import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console

from hri_turn_taking.config import ProjectConfig, load_config
from hri_turn_taking.legacy.audio.gcc_phat import doa_from_tdoa, estimate_tdoa_gcc_phat
from hri_turn_taking.legacy.data.synthetic import save_synthetic_turn_taking_data
from hri_turn_taking.legacy.features.fusion import FrameFeatures
from hri_turn_taking.legacy.models.inference import predict_trp_from_csv
from hri_turn_taking.legacy.models.trainer import train_trp_classifier
from hri_turn_taking.legacy.realtime.pipeline import PredictiveTurnTakingPipeline
from hri_turn_taking.legacy.vision.mouth import mouth_aspect_ratio

app = typer.Typer(help="Predictive turn-taking HRI utilities.")
console = Console()


def _json_safe(value: object) -> object:
    """Replace non-finite metric values with JSON ``null`` recursively."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


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
    from dataclasses import asdict

    from hri_turn_taking.data.assembly import (
        DataSourceGateError,
        write_synthetic_provenance,
    )
    from hri_turn_taking.data.multiparty import SyntheticConfig, save_multiparty_dataset

    if output.suffix.lower() != ".csv":
        console.print(
            "[bold red]Synthetic generation failed[/bold red]: "
            "the gated VAP source must be a .csv file"
        )
        raise typer.Exit(code=1)
    config = SyntheticConfig(
        n_sessions=sessions,
        duration_s=duration_s,
        n_participants=participants,
        random_seed=seed,
    )
    try:
        frame_df = save_multiparty_dataset(output, config)
        provenance_path = write_synthetic_provenance(
            output,
            generator_config=asdict(config),
            overwrite=True,
        )
    except (DataSourceGateError, OSError, TypeError, ValueError) as error:
        console.print(f"[bold red]Synthetic generation failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error
    vad_cols = [c for c in frame_df.columns if c.endswith("_vad")]
    console.print("[bold green]Multiparty synthetic dataset generated[/bold green]")
    console.print(f"Path: {output}")
    console.print(f"Provenance: {provenance_path}")
    console.print(f"Rows: {len(frame_df)}  Sessions: {sessions}  Participants: {participants}")
    activity = frame_df[vad_cols].mean().round(3).to_dict()
    console.print(f"Speech activity rate per participant: {activity}")


@app.command()
def vap_intake(
    delivery_dir: Path,
    output: Path | None = None,
    crosswalk_inventory: Path = Path(
        "configs/hpe_transcript_crosswalk_inventory.json"
    ),
) -> None:
    """Convert one HPE delivery directory into a validated 25 fps wide table.

    Applies the sealed human HPE→Transcript crosswalk (the upstream HPE map is
    retained only as a proposal), the accepted half-open frame window, and the
    fixed physical ``6cam`` D3 offset. It places ``world_persons.csv`` on the
    shared mic A axis, resamples to the complete 25 Hz output grid, maps participants to slots
    (P1=R, P2=A, P3=B, P4=C) and writes the wide table plus a
    ``.provenance.json`` report listing real vs. placeholder columns.
    """
    from hri_turn_taking.data.intake import DeliveryGateError, write_intake

    if output is None:
        output = Path("data/processed/intake_v2") / f"{delivery_dir.name}.csv"
    try:
        result = write_intake(
            delivery_dir,
            output,
            crosswalk_inventory=crosswalk_inventory,
        )
    except DeliveryGateError as error:
        console.print(f"[bold red]Intake gate failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error

    provenance = result.provenance
    console.print("[bold green]Intake complete[/bold green]")
    console.print(f"Session: {provenance['session_id']}  Frames: {provenance['n_frames']}")
    console.print(f"Wide table: {output}")
    console.print(f"Provenance: {output.with_suffix('.provenance.json')}")
    console.print(
        f"Identity authority: {provenance['identity_authority']}  "
        f"Window: {provenance['accepted_window']['logical_start_inclusive']}.."
        f"{provenance['accepted_window']['logical_end_exclusive']} (end-exclusive)"
    )
    console.print(
        f"Offset applied ({provenance['stream_key']}): "
        f"{provenance['offset_applied_s']:+.3f} s"
    )
    console.print(f"Real columns: {provenance['real_columns']}")
    console.print(
        f"[yellow]Placeholder columns (zero-filled, NOT trainable yet):[/yellow] "
        f"{provenance['placeholder_columns']}"
    )


@app.command()
def vap_assemble(
    base: Path,
    output: Path,
    feature: Annotated[
        list[Path],
        typer.Option(
            "--feature",
            help="Exact-key feature/VAD CSV; repeat this option for each source file.",
        ),
    ],
) -> None:
    """Merge extracted real features into one intake table without temporal joins."""
    from hri_turn_taking.data.assembly import AssemblyGateError, assemble_session

    try:
        result = assemble_session(base, feature, output)
    except AssemblyGateError as error:
        console.print(f"[bold red]Assembly gate failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error

    console.print("[bold green]Real session assembly complete[/bold green]")
    console.print(f"Wide table: {result.output_csv}")
    console.print(f"Provenance: {result.provenance_path}")
    console.print(f"Trainable target tier: {result.provenance['trainable']}")
    console.print(f"Remaining placeholders: {result.provenance['placeholder_columns']}")


@app.command()
def vap_audit(
    data_kind: Annotated[
        str,
        typer.Option(
            "--data-kind",
            help="Required explicit source kind: real or synthetic.",
        ),
    ],
    data: Path = Path("data/processed/real"),
    feature_set: str | None = None,
    group_delimiter: str | None = None,
    require_ready: bool = False,
) -> None:
    """Report structural, gold-target, and selected-feature readiness separately."""
    from hri_turn_taking.data.assembly import DataSourceGateError, audit_vap_source

    try:
        report = audit_vap_source(
            data,
            data_kind=data_kind,
            feature_set=feature_set,
            group_delimiter=group_delimiter,
        )
    except (DataSourceGateError, ValueError) as error:
        console.print(f"[bold red]Audit gate failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error

    console.print_json(json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False))
    if require_ready and feature_set is None:
        console.print(
            "[bold red]Audit gate failed[/bold red]: --require-ready needs an explicit "
            "--feature-set"
        )
        raise typer.Exit(code=1)
    if require_ready and (
        not report.target_ready
        or report.selected_feature_ready is not True
    ):
        console.print(
            "[bold red]Dataset target/selected-feature readiness failed.[/bold red] "
            "This flag does not certify a formal outer-test protocol."
        )
        raise typer.Exit(code=1)


@app.command()
def vap_train(
    data_kind: Annotated[
        str,
        typer.Option(
            "--data-kind",
            help="Required explicit source kind: real or synthetic.",
        ),
    ],
    data: Path = Path("data/processed/multiparty_synthetic.csv"),
    output: Path = Path("models/vap_multiparty.pt"),
    encoder: str = "transformer",
    epochs: int = 20,
    d_model: int = 64,
    n_layers: int = 2,
    window_seconds: float = 10.0,
    seed: int = 42,
    group_delimiter: str | None = None,
    feature_set: str = "multimodal",
    aux_vad_weight: float = 0.3,
    robot_slot: int = 1,
    allow_untrainable: bool = False,
    overwrite: bool = False,
) -> None:
    """Train the multiparty VAP model (future voice-activity bin prediction).

    This runner performs train/inner-validation checkpoint selection and reports
    that same development split. It does not produce a formal outer-test result.
    Real group identity comes only from gated provenance, never session names.
    """
    from hri_turn_taking.data.assembly import (
        DataSourceGateError,
        load_and_audit_vap_source,
    )
    from hri_turn_taking.vap.feature_sets import get_feature_set
    from hri_turn_taking.vap.labels import VAPLabelConfig
    from hri_turn_taking.vap.trainer import VAPTrainerConfig, train_vap_model

    if not 1 <= robot_slot <= 4:
        console.print("[bold red]Training gate failed[/bold red]: --robot-slot must be 1..4")
        raise typer.Exit(code=1)
    try:
        get_feature_set(feature_set)
        frame_df, readiness = load_and_audit_vap_source(
            data,
            data_kind=data_kind,
            feature_set=feature_set,
            group_delimiter=group_delimiter,
        )
    except (DataSourceGateError, ValueError) as error:
        console.print(f"[bold red]Training gate failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error

    readiness_blockers = [
        *readiness.target_blockers,
        *readiness.selected_feature_blockers,
    ]
    if readiness_blockers and not allow_untrainable:
        console.print(
            "[bold red]Training gate failed[/bold red]: target/selected-feature readiness "
            f"failed: {readiness_blockers}. Supply the required target/features or use "
            "--allow-untrainable only for an explicitly labelled smoke test."
        )
        raise typer.Exit(code=1)

    label_config = VAPLabelConfig()
    trainer_config = VAPTrainerConfig(
        epochs=epochs,
        window_frames=max(1, round(window_seconds * label_config.frame_rate_hz)),
        stride_frames=max(1, round(window_seconds * label_config.frame_rate_hz / 2)),
        random_seed=seed,
        group_delimiter=group_delimiter,
        feature_set=feature_set,
        aux_vad_weight=aux_vad_weight,
        robot_slot=robot_slot - 1,
    )
    requested_overrides = [
        name
        for name, enabled in (
            ("allow_untrainable", allow_untrainable),
        )
        if enabled
    ]
    effective_smoke_reasons = (
        [f"readiness override: {blocker}" for blocker in readiness_blockers]
        if allow_untrainable
        else []
    )
    try:
        result = train_vap_model(
            frame_df,
            checkpoint_path=output,
            label_config=label_config,
            trainer_config=trainer_config,
            encoder=encoder,
            d_model=d_model,
            n_layers=n_layers,
            run_metadata={
                "data_kind": data_kind,
                "requested_overrides": requested_overrides,
                "effective_smoke_reasons": effective_smoke_reasons,
                "experiment_blockers": list(readiness.experiment_blockers),
            },
            data_manifest=readiness.to_dict(),
            overwrite=overwrite,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        console.print(f"[bold red]Training failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error

    console.print("[bold green]VAP development training complete[/bold green]")
    console.print(
        "[bold yellow]Development result only:[/bold yellow] inner validation was used "
        "for checkpoint selection and reporting; no outer test was evaluated."
    )
    console.print(f"Checkpoint: {result.checkpoint_path}")
    console.print(
        f"Feature set: {feature_set}  Auxiliary current-VAD weight: {aux_vad_weight:g}  "
        f"Robot slot: P{robot_slot}"
    )
    console.print(
        f"Sessions: {result.n_train_sessions} train / {result.n_val_sessions} validation"
    )
    console.print(f"Frame metrics: {result.frame_metrics}")
    for event_name, metrics in result.event_metrics.items():
        console.print(f"{event_name}: {metrics}")


@app.command()
def vap_predict(
    data: Path,
    data_kind: Annotated[
        str,
        typer.Option(
            "--data-kind",
            help="Required explicit source kind: real or synthetic.",
        ),
    ],
    model: Path = Path("models/vap_multiparty.pt"),
    output: Path = Path("outputs/vap_predictions.csv"),
    device: str = "cpu",
    validation_only: bool = False,
    allow_placeholder_inputs: bool = False,
    overwrite: bool = False,
) -> None:
    """Export the sole participant-wise ``time x participant x bin`` VAP output."""
    from hri_turn_taking.data.assembly import (
        DataSourceGateError,
        load_and_audit_vap_source,
    )
    from hri_turn_taking.vap.predictions import (
        load_prediction_checkpoint_snapshot,
        write_prediction_artifact,
    )

    try:
        checkpoint_snapshot = load_prediction_checkpoint_snapshot(model, device=device)
        checkpoint = checkpoint_snapshot.checkpoint
        feature_set = str(
            (checkpoint.trainer_config or {}).get("feature_set", "multimodal")
        )
        frame_df, readiness = load_and_audit_vap_source(
            data,
            data_kind=data_kind,
            feature_set=feature_set,
        )
        selected_blockers = list(readiness.selected_feature_blockers)
        if selected_blockers and not allow_placeholder_inputs:
            raise DataSourceGateError(
                f"checkpoint feature set {feature_set!r} is not input-ready: "
                f"{selected_blockers}"
            )
        # Gold VAD target readiness is deliberately not required for prediction.
        # Canonical VAD columns may be provenance-declared placeholders.
        artifact = write_prediction_artifact(
            frame_df,
            checkpoint_snapshot=checkpoint_snapshot,
            output_path=output,
            device=device,
            validation_only=validation_only,
            overwrite=overwrite,
            source_manifest=readiness.to_dict(),
            override_status={
                "requested_overrides": {
                    "allow_placeholder_inputs": allow_placeholder_inputs,
                },
                "effective_smoke_reasons": [
                    f"selected-feature override: {blocker}" for blocker in selected_blockers
                ]
                if allow_placeholder_inputs
                else [],
                "smoke_only": bool(selected_blockers and allow_placeholder_inputs),
                "data_kind": data_kind,
                "target_readiness_required": False,
            },
        )
    except (DataSourceGateError, FileExistsError, OSError, ValueError, RuntimeError) as error:
        console.print(f"[bold red]Prediction gate failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error

    if artifact.metadata["artifact_status"] == "smoke_only":
        console.print(
            "[bold yellow]Smoke-only:[/bold yellow] untrainable/placeholder data were "
            "explicitly allowed; these predictions are not a scientific result."
        )
    console.print("[bold green]Participant-wise VAP prediction complete[/bold green]")
    console.print(f"Rows: {len(artifact.table)}")
    console.print(f"Tensor shapes: {artifact.metadata['session_tensor_shapes']}")
    console.print(f"Predictions: {artifact.output_path}")
    console.print(f"Metadata: {artifact.metadata_path}")


@app.command()
def vap_baselines(
    data: Path,
    data_kind: Annotated[
        str,
        typer.Option(
            "--data-kind",
            help="Required explicit source kind: real or synthetic.",
        ),
    ],
    group_delimiter: str | None = None,
    validation_fraction: float = 0.25,
    seed: int = 42,
    robot_slot: int = 1,
    asd_threshold: float = 0.5,
    output: Path | None = None,
) -> None:
    """Evaluate robot-observable ASD persistence/last-speaker VAP baselines."""
    from hri_turn_taking.data.assembly import (
        DataSourceGateError,
        load_and_audit_vap_source,
    )
    from hri_turn_taking.vap.baselines import ASDPersistenceBaseline, LastSpeakerBaseline
    from hri_turn_taking.vap.dataset import sessions_from_dataframe, split_sessions
    from hri_turn_taking.vap.trainer import evaluate_model

    if not 1 <= robot_slot <= 4:
        console.print("[bold red]Baseline gate failed[/bold red]: --robot-slot must be 1..4")
        raise typer.Exit(code=1)
    try:
        frame_df, readiness = load_and_audit_vap_source(
            data,
            data_kind=data_kind,
            group_delimiter=group_delimiter,
        )
        if not readiness.target_ready:
            raise DataSourceGateError(
                f"baseline evaluation requires gold targets: {list(readiness.target_blockers)}"
            )
        present_slots = {
            slot
            for slot in range(1, 5)
            if frame_df[f"P{slot}_present"].eq(1.0).any()
        }
        asd_placeholders = [
            column
            for column in readiness.placeholder_columns
            if column.endswith("_asd_prob")
            and int(column.split("_", 1)[0][1:]) in present_slots
        ]
        if asd_placeholders:
            raise DataSourceGateError(
                f"ASD baselines require real ASD inputs, still placeholders: {asd_placeholders}"
            )
        sessions = sessions_from_dataframe(frame_df)
        _, validation_sessions = split_sessions(
            sessions,
            validation_fraction=validation_fraction,
            random_seed=seed,
            group_delimiter=group_delimiter,
        )
    except (DataSourceGateError, ValueError) as error:
        console.print(f"[bold red]Baseline gate failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error

    results: dict[str, object] = {
        "result_role": "development_inner_validation",
        "counted_experiment_eligible": False,
        "inner_validation_used_for_selection_and_reporting": True,
        "inner_validation_session_ids": [
            session.session_id for session in validation_sessions
        ],
        "robot_slot": f"P{robot_slot}",
        "seed": seed,
        "baselines": {},
    }
    baseline_results = results["baselines"]
    assert isinstance(baseline_results, dict)
    for name, baseline in (
        ("asd_persistence", ASDPersistenceBaseline()),
        ("last_speaker", LastSpeakerBaseline(asd_threshold=asd_threshold)),
    ):
        frame_metrics, event_metrics = evaluate_model(
            baseline,
            validation_sessions,
            None,
            robot_slot=robot_slot - 1,
        )
        baseline_results[name] = {
            "frame_metrics": frame_metrics,
            "event_metrics": event_metrics,
        }

    payload = json.dumps(
        _json_safe(results),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    console.print(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        console.print(f"Saved to: {output}")


@app.command()
def vap_demo(
    data_kind: Annotated[
        str,
        typer.Option(
            "--data-kind",
            help="Required explicit source kind: real or synthetic.",
        ),
    ],
    data: Path = Path("data/processed/multiparty_synthetic.csv"),
    model: Path = Path("models/vap_multiparty.pt"),
    silence_baseline_s: float = 1.5,
    robot_slot: int = 1,
) -> None:
    """Compare predictive policies vs. the reactive silence baseline.

    ``--robot-slot`` designates the demonstrator slot for the
    demonstration-agreement comparison: on the real corpus, the participant
    who role-plays the robot; on synthetic data, an arbitrary
    pseudo-demonstrator (default P1). CLI slots are one-based.
    """
    from hri_turn_taking.data.assembly import (
        DataSourceGateError,
        load_and_audit_vap_source,
    )
    from hri_turn_taking.vap.dataset import sessions_from_dataframe, split_sessions
    from hri_turn_taking.vap.events import score_events
    from hri_turn_taking.vap.feature_sets import apply_feature_set
    from hri_turn_taking.vap.offline_demo import (
        compare_policies_on_session,
        compare_policies_with_demonstrator,
    )
    from hri_turn_taking.vap.trainer import load_vap_checkpoint

    if not 1 <= robot_slot <= 4:
        console.print("[bold red]Demo gate failed[/bold red]: --robot-slot must be 1..4")
        raise typer.Exit(code=1)
    try:
        loaded = load_vap_checkpoint(model)
        feature_set = str((loaded.trainer_config or {}).get("feature_set", "multimodal"))
        frame_df, readiness = load_and_audit_vap_source(
            data,
            data_kind=data_kind,
            feature_set=feature_set,
        )
        if not readiness.target_ready:
            raise DataSourceGateError(
                f"demo evaluation requires gold targets: {list(readiness.target_blockers)}"
            )
        if readiness.selected_feature_ready is not True:
            raise DataSourceGateError(
                f"checkpoint feature set is not ready: "
                f"{list(readiness.selected_feature_blockers)}"
            )
    except (DataSourceGateError, ValueError, RuntimeError) as error:
        console.print(f"[bold red]Demo gate failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error
    frame_df = apply_feature_set(frame_df, feature_set)
    sessions = sessions_from_dataframe(frame_df, label_config=loaded.label_config)

    scope = "all sessions"
    validation_ids = set(
        (loaded.data_split or {}).get("inner_validation_session_ids")
        or (loaded.data_split or {}).get("val_session_ids", [])
    )
    if validation_ids:
        missing = sorted(validation_ids - {session.session_id for session in sessions})
        if missing:
            console.print(
                f"[bold red]Demo gate failed[/bold red]: checkpoint validation sessions "
                f"are missing from data: {missing}"
            )
            raise typer.Exit(code=1)
        sessions = [session for session in sessions if session.session_id in validation_ids]
        scope = "checkpoint-recorded inner-validation sessions (development only)"
    elif loaded.trainer_config is not None and len(sessions) >= 2:
        _, sessions = split_sessions(
            sessions,
            validation_fraction=loaded.trainer_config["validation_fraction"],
            random_seed=loaded.trainer_config["random_seed"],
            group_delimiter=loaded.trainer_config.get("group_delimiter"),
        )
        scope = "reconstructed inner-validation sessions (development only)"

    console.print(
        f"[bold green]Predictive vs. reactive floor-entry comparison[/bold green] ({scope})"
    )
    for session in sessions:
        comparison = compare_policies_on_session(
            session,
            loaded.model,
            loaded.normalizer,
            label_config=loaded.label_config,
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

    console.print(
        f"[bold green]Demonstration-agreement comparison[/bold green] "
        f"(robot slot = P{robot_slot}; ground truth = the demonstrator's own entries)"
    )
    pooled_labels: list[int] = []
    pooled_scores: list[float] = []
    for session in sessions:
        agreement = compare_policies_with_demonstrator(
            session,
            loaded.model,
            loaded.normalizer,
            robot_slot=robot_slot - 1,
            label_config=loaded.label_config,
            silence_baseline_s=silence_baseline_s,
        )
        if agreement is None:
            console.print(f"{session.session_id}: no entry opportunities")
            continue
        pooled_labels.extend(agreement.entry_labels)
        pooled_scores.extend(agreement.entry_scores)
        n_other = agreement.n_opportunities - agreement.n_robot_gaps
        console.print(
            f"{session.session_id}: opportunities={agreement.n_opportunities} "
            f"(demonstrator entered {agreement.n_robot_gaps})"
        )
        for name, stats in (
            ("demonstration", agreement.demonstration),
            ("floor-holder ", agreement.floor_holder),
            ("reactive     ", agreement.reactive),
        ):
            console.print(
                f"  {name}: recall {stats.entries_on_robot_gaps}/{agreement.n_robot_gaps} | "
                f"false entries {stats.false_entries}/{n_other} | "
                f"lead vs demonstrator {stats.mean_lead_vs_demonstrator_s:.2f}s"
            )
    if len(set(pooled_labels)) == 2:
        pooled = score_events(np.asarray(pooled_labels), np.asarray(pooled_scores), 0.5)
        console.print(
            f"Pooled robot-entry test: AP={pooled.average_precision:.3f} "
            f"({pooled.n_positive}/{pooled.n_events} demonstrator entries)"
        )
