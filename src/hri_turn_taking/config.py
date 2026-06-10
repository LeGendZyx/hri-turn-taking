from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectSection(BaseModel):
    name: str = "predictive_turn_taking_hri"
    random_seed: int = 42


class DataSection(BaseModel):
    root_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    annotations_dir: Path = Path("data/annotations")


class AudioSection(BaseModel):
    sample_rate: int = 16000
    mic_distance_m: float = 0.06
    speed_of_sound_mps: float = 343.0
    gcc_phat_interp: int = 16


class VisionSection(BaseModel):
    mouth_aspect_ratio_threshold: float = 0.20
    gaze_confidence_threshold: float = 0.50


class FusionSection(BaseModel):
    frame_rate_hz: int = 25
    history_seconds: float = 2.0


class TurnTakingSection(BaseModel):
    trp_threshold: float = 0.65
    silence_baseline_seconds: float = 1.5


class ModelSection(BaseModel):
    type: str = "random_forest"
    checkpoint_path: Path = Path("models/trp_baseline.joblib")


class TrainingSection(BaseModel):
    dataset_path: Path = Path("data/processed/frame_features.csv")
    synthetic_dataset_path: Path = Path("data/processed/synthetic_turn_taking.csv")
    target_column: str = "trp_label"
    model_type: str = "random_forest"
    window_seconds: float = 2.0
    test_size: float = 0.2
    feature_columns: list[str] = Field(
        default_factory=lambda: [
            "audio_vad",
            "active_speaker_count",
            "overlap_active",
            "dominant_doa_angle_deg",
            "mouth_aspect_ratio_max",
            "visual_vad_max",
            "gaze_to_robot_score",
            "head_yaw_abs_min_deg",
            "speaking_towards_robot",
            "silence_duration_s",
            "current_turn_duration_s",
            "recent_speaker_change",
        ]
    )


class RuntimeSection(BaseModel):
    furhat_host: str = "127.0.0.1"
    furhat_port: int = 54321


class ProjectConfig(BaseModel):
    project: ProjectSection = Field(default_factory=ProjectSection)
    data: DataSection = Field(default_factory=DataSection)
    audio: AudioSection = Field(default_factory=AudioSection)
    vision: VisionSection = Field(default_factory=VisionSection)
    fusion: FusionSection = Field(default_factory=FusionSection)
    turn_taking: TurnTakingSection = Field(default_factory=TurnTakingSection)
    model: ModelSection = Field(default_factory=ModelSection)
    training: TrainingSection = Field(default_factory=TrainingSection)
    runtime: RuntimeSection = Field(default_factory=RuntimeSection)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream) or {}
    return ProjectConfig.model_validate(payload)
