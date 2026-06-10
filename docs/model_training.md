# Model Training Framework

## Purpose

This framework lets you start model development before the full audio-video dataset is ready. It trains a baseline TRP classifier from frame-level multimodal features, then saves a reusable model artifact.

The first baseline is intentionally simple and inspectable:

- Input: fixed-rate frame features, usually 25 fps.
- Temporal context: the last `window_seconds` of features are flattened into one training example.
- Model choices: `random_forest`, `hist_gradient_boosting`, or `logistic_regression`.
- Target: `trp_label`, meaning the robot should consider entering the floor at this moment.

Later, the same dataset schema can feed an LSTM, Conformer, or audio-visual Transformer.

## Commands

Create a synthetic dataset:

```powershell
hri-turn-taking generate-synthetic
```

Train the baseline:

```powershell
hri-turn-taking train
```

Predict on a feature CSV:

```powershell
hri-turn-taking predict data/processed/synthetic_turn_taking.csv
```

## Expected Feature Table

The current default CSV schema is one row per timestamp, aggregated at group level.

Required metadata:

```text
session_id,timestamp_s
```

Default feature columns:

```text
audio_vad
active_speaker_count
overlap_active
dominant_doa_angle_deg
mouth_aspect_ratio_max
visual_vad_max
gaze_to_robot_score
head_yaw_abs_min_deg
speaking_towards_robot
silence_duration_s
current_turn_duration_s
recent_speaker_change
```

Default target:

```text
trp_label
```

## How This Maps to Your Project

- `audio_vad`: whether anyone is currently speaking.
- `active_speaker_count`: number of active speakers, useful for overlap.
- `overlap_active`: whether overlapping speech is happening.
- `dominant_doa_angle_deg`: audio Direction of Arrival from GCC-PHAT or microphone array.
- `mouth_aspect_ratio_max`: maximum participant MAR in the current frame.
- `visual_vad_max`: visual speech activity from lip motion.
- `gaze_to_robot_score`: gaze/addressee confidence, from MCGaze or a later gaze model.
- `head_yaw_abs_min_deg`: closest participant head orientation toward the robot.
- `speaking_towards_robot`: binary addressee signal.
- `silence_duration_s`: duration since speech stopped.
- `current_turn_duration_s`: duration of the current speaker turn.
- `recent_speaker_change`: whether the current speaker changed recently.

If your later feature extractor produces per-participant rows instead of group-level rows, keep this baseline as an aggregation target first. After that is stable, the next model can move to participant-level sequence tensors.
