# Predictive Turn-Taking for Multi-Party HRI

This repository contains the starter framework for the MSc project:

`Learning from Humans: A Hybrid Multimodal Framework for Predictive Turn-Taking in Multi-Party HRI`

The project is organised as a research Python package for collecting multimodal data, extracting audio-visual features, training a predictive turn-taking model, and later integrating the decision layer with Furhat or ROS2.

## Project Layout

```text
configs/                 Experiment and runtime configuration
data/                    Local datasets and annotations, ignored by git (see data/README.md)
docs/                    Notes, proposal summary, experiment protocol drafts
outputs/                 Experiment outputs, ignored by git
src/hri_turn_taking/     Main package: vap/ (core method) + legacy/ (frozen TRP ablation)
tests/                   Lightweight tests for core feature logic
```

## Quick Start

Create and activate the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the demo:

```powershell
hri-turn-taking demo
```

### Multiparty VAP pipeline (core method)

Generate a naive multiparty dataset, train the VAP model, and compare the
predictive dual-threshold policy against the reactive silence baseline:

```powershell
hri-turn-taking vap-generate
hri-turn-taking vap-train
hri-turn-taking vap-demo
```

The dataset uses the per-participant wide schema (`P{i}_<feature>` at 25 fps);
gold `P{i}_vad` columns are used only to derive future voice-activity bin
labels and for event-level evaluation, never as model input.

### Legacy frame-level TRP baseline (ablation)

```powershell
hri-turn-taking generate-synthetic
hri-turn-taking train
hri-turn-taking predict data/processed/synthetic_turn_taking.csv
```

Run tests:

```powershell
pytest
```

## Current Scope

The framework includes:

- **Multiparty VAP model** (`src/hri_turn_taking/vap/`): causal Transformer/GRU
  predicting each participant's future voice activity over discretised bins
  (0.2/0.6/1.2/2.0 s), trained with masked, class-weighted BCE and an
  auxiliary current-VAD head. Session-level splits, balanced metrics
  (mAP/balanced accuracy) and the standard VAP event tests (shift/hold,
  shift-prediction).
- **Dual-threshold floor policy** (`vap/policy.py`): p_now / p_future gating
  on the floor holder (Skantze & Irfan, HRI 2025) with debouncing and a
  refractory period, plus an offline predictive-vs-reactive latency demo.
- GCC-PHAT utilities for Time Difference of Arrival and Direction of Arrival estimation.
- Mouth Aspect Ratio utilities kept as an interpretable V-VAD ablation baseline.
- A legacy frame-level TRP classifier (scikit-learn) kept as a low-cost comparison.
- Configuration, CLI, tests, and placeholder folders for datasets, annotations, models, and experiment outputs.

## Optional Future Dependencies

The core environment is intentionally light. Install optional extras only when that part of the project is needed:

```powershell
python -m pip install -e ".[vision]"   # MediaPipe/OpenCV feature extraction
python -m pip install -e ".[audio]"    # live audio capture helpers
python -m pip install -e ".[ml]"       # torchaudio (PyTorch itself is a core dependency)
```

ROS2 and Furhat SDK setup are platform-specific and should be added once the local hardware/runtime path is confirmed.
