# CLAUDE.md — Project Context

> This file gives Claude Code persistent context about this project.
> Keep it updated when key decisions change.

## Project Overview

**MSc Robotics dissertation (KCL, supervised by Dr. Oya Celiktutan)**
Title: *Learning from Humans: A Hybrid Multimodal Framework for Predictive Turn-Taking in Multi-Party HRI*

Goal: Replace the Furhat robot's reactive, silence-based turn-taking (wait >1.5s)
with a **predictive multimodal model** that estimates upcoming speaker transitions
(VAP-style future voice activity prediction) in 3–4 person group conversations.

**Core novelty claim** (positioning vs. 2025 SOTA):
multimodal (audio + visual) × multi-party (3–4 people) × **deployed on a physical
robot (Furhat) with closed-loop online evaluation**.
- NOT claiming "first multi-party turn prediction" (Elmers et al., Interspeech 2025
  did audio-only triadic VAP) nor "first multimodal VAP" (Onishi et al., IEICE 2025,
  dyadic). The physical HRI deployment + multimodal multi-party combination is the gap.

## Key Design Decisions (agreed, do not silently change)

1. **Training objective**: VAP-style **future voice activity prediction**
   (predict each participant's voice activity over the next 2s, discretized bins),
   NOT a hand-labeled 3-class Speaking/Listening/TRP classifier.
   - Labels are generated **automatically** from per-participant lavalier mic VAD.
   - Turn events (shift/hold, shift-prediction, backchannel, overlap resolution)
     are *derived* at evaluation time, following Skantze/Inoue/Ekstedt conventions.

2. **Two feature tiers — never mix them in the deployed model**:
   - **Gold tier** (training labels + offline eval ONLY): per-speaker lavalier VAD,
     multi-view 3D reconstruction from the 6 calibrated mono cameras.
   - **Robot-observable tier** (model INPUT): robot-view camera features
     (gaze, head pose, lip/face), mic-array audio. The deployed model must only
     consume features Furhat can actually obtain at runtime.
   - Optionally train a "full-feature oracle" model as an upper bound for comparison.

3. **V-VAD**: use a pretrained lightweight ASD model (Light-ASD / LR-ASD,
   AVA-pretrained, open weights) to produce per-face per-frame speaking probability.
   - MAR (Mouth Aspect Ratio from facial landmarks) is kept only as an
     interpretable baseline / ablation, NOT the core method.

4. **Audio encoding**: prefer pretrained CPC encoder (from Ekstedt's
   VoiceActivityProjection repo) over hand-crafted features. GCC-PHAT DoA kept
   as auxiliary spatial feature (CPC does not encode direction).
   - If prosody is used explicitly: openSMILE/Praat (F0, energy, speech rate).

5. **Training strategy**: pretrain/freeze encoders, fine-tune only light fusion +
   prediction head on our own data. Our dataset (~few hours) is too small to train
   a transformer from scratch. Consider AMI corpus for pretraining experiments.
   A lightweight LSTM remains as a low-cost comparison architecture.

6. **Decision policy on robot**: dual-threshold p_now / p_future strategy
   (following Skantze & Irfan, HRI 2025), not a single P(TRP) > δ threshold.

7. **Evaluation**:
   - Offline: balanced accuracy / F1 / mAP (NOT raw accuracy — heavy class imbalance).
     Standard VAP event tests: shift/hold prediction, shift-prediction before pause,
     overlap handling. Main comparison = our own audio-only ablation
     (not the 18% figure from Abbo et al., which is a different measurement).
   - Online: user study on Furhat, predictive vs. reactive silence baseline.
     Objective KPIs (floor-entry latency, successful floor acquisition in overlap)
     + Godspeed (Animacy) + perceived latency Likert items.
   - Latency budget for deployment: end-to-end < 500ms; account for camera (~33ms),
     feature extraction, inference, network bridge, Furhat TTS startup (200–400ms).

## Implementation Status (updated 2026-06-11)

Repo: github.com/LeGendZyx/hri-turn-taking (private), branch `main`.
The multiparty VAP pipeline is implemented and verified end-to-end on
synthetic data; real recordings are not yet available.
Architecture diagram: `ARCHITECTURE.md` (repo root, Mermaid) — keep it in
sync when `vap/model.py`, the policy thresholds, or the feature schema change.

### Core pipeline — `src/hri_turn_taking/vap/`
- `labels.py` — VAP bin labels from gold per-participant VAD
  (bins 0.2/0.6/1.2/2.0 s, active-ratio 0.5); p_now/p_future readout.
- `model.py` — `MultipartyVAP`: Linear(input→64) projection → causal encoder
  (2-layer Transformer default, GRU as low-cost comparison) → per-participant
  × per-bin logits + auxiliary current-VAD head. ~100k params. Independent
  per-bin BCE objective (multi-party simplification of the 256-class joint
  VAP state space).
- `dataset.py` — wide table → tensors, feature normalisation, sliding
  windows; session-level splits only.
- `trainer.py` — masked BCE (absent participant slots + truncated-horizon
  frames excluded) with per-bin pos_weight, AdamW, early stopping; checkpoint
  stores model config + normaliser + label config for reproduction. Chunked
  full-session inference for sequences beyond the positional table.
- `events.py` — shift/hold and shift-prediction event tests; shift score is
  averaged over the silence gap (Ekstedt & Skantze convention — gap-onset
  single-frame readout is near chance because addressee cues live in
  gaze-target features and late-gap frames).
- `policy.py` — dual-threshold p_now/p_future gated on the *floor holder*
  (original Skantze & Irfan semantics; gating on max over all humans never
  fires), with debouncing + refractory period.
- `offline_demo.py` — predictive vs. reactive floor-entry latency comparison.

### Data schema — `src/hri_turn_taking/data/multiparty.py`
- Wide frame table at 25 fps, columns `P{i}_<feature>`, up to 4 slots with
  `P{i}_present` masks (3-party sessions pad slot 4).
- Gold tier: `P{i}_vad` — used ONLY for labels and event evaluation.
- Robot-observable input tier: `P{i}_asd_prob`, `P{i}_mar`, `P{i}_gaze_robot`,
  `P{i}_head_yaw_deg`, `P{i}_gazed_at` + global `doa_angle_deg`,
  `array_energy`, `f0_proxy` (23 input dims).
- Synthetic generator: features are *probabilistic noisy correlates* of the
  underlying speech state with genuine anticipatory cues (turn-final prosodic
  lowering, speaker gaze redirection to next speaker, pre-onset articulation),
  gated by cue_reliability=0.7. Features must never be deterministic
  functions of labels (an earlier scaffold had that leakage — do not refit).
- Real data drop-in: produce this wide table from extracted features and feed
  `vap-train` directly; only `input_dim` changes if features are added.

### CLI (`hri-turn-taking` / `python -m hri_turn_taking`)
- `vap-generate` / `vap-train` / `vap-demo` — core pipeline.
- `generate-synthetic` / `train` / `predict` — legacy frame-level sklearn TRP
  baseline, kept solely as an ablation comparison; do not extend it.

Reference numbers on synthetic data (12×180 s sessions, 20 epochs, CPU):
frame mAP ≈ 0.88, shift/hold AP ≈ 0.68, shift-prediction AP ≈ 0.72
(ceiling limited by cue_reliability); predictive policy enters gaps in
0.16–0.58 s while the 1.5 s silence baseline never enters.

PyTorch is a core dependency (CPU build installed in `.venv`).
Tests: `pytest` (18 tests, includes a causality test and a training smoke).

## Data Collection (in progress)

- 3–4 human participants per session, open-ended collaborative discussion tasks.
- **Sensors**: 6 calibrated monochrome cameras (with calibration data),
  2 RGB-D cameras, per-participant lavalier mics (gold-standard audio),
  mic array (robot's-ear spatial audio).
- Feature frame rate: 25 fps. Sync via audiovisual beacon (clap/beep);
  verify drift with audio-video lip sync; target alignment within ±1 frame (40ms).

### Team feature-extraction split
- **Michelle**: audio sync, video/skeleton sync, audio-video lip sync.
- **Tomas**: emotion, facial keypoints, gaze direction/interaction (MCGaze).
- **Zobaid**: transcript (audio-to-text), whole-body 3D via calibration, hand gesture
  (CMU ssp toolkit).
- **Yixuan (me)**: speaking-towards / addressee (MMSI format),
  **speaker detect** (ELAN annotation with start/end timestamps), head pose.

### My annotation pipeline (speaker detect)
- Semi-automatic: run automatic VAD (Silero VAD or pyannote) on each lavalier track
  → import segments into ELAN → manual correction only. (~5x faster than manual.)
- ELAN tiers must include an **overlap-type tier**: distinguish
  `backchannel` (uh-huh, yeah, laughter) vs `competitive` (barge-in).
  This is required for the overlap-scenario analysis (RQ3).

### Feature gaps to cover
- **Prosody** (F0/energy/rate) — unassigned; or rely on CPC embeddings implicitly.
- **Pretrained ASD speaking probability** per face (Light-ASD) — also useful to
  pre-label/accelerate my ELAN annotation.
- **Gaze target (discrete "looking at A/B/C/robot")** — derive by intersecting
  MCGaze vectors with 3D head positions from calibrated multi-view; falls between
  Tomas's and Zobaid's outputs, I may need to own this derivation step.
- **Silence/pause/overlap statistics** auto-derived from VAD (for evaluation).

## Tech Stack & Integration

- Furhat SDK (Kotlin / remote API) ↔ ROS2 node bridging sensors and model.
- MediaPipe Face Mesh for landmarks; MCGaze for gaze; CMU ssp for body/gesture.
- Hardware note: Azure Kinect DK is discontinued — use ZED 2 (or Orbbec Femto Bolt).
- Python for ML pipeline; PyTorch.

## Scope / Priority (if time runs short)

1. **Core deliverable**: offline multimodal prediction model (H1 + part of H2).
2. Second: Furhat integration demo.
3. Stretch: formal user study (can shrink to 2–3 group pilot).

## Key References

- Ekstedt & Skantze — VAP (code: github.com/ErikEkstedt/VoiceActivityProjection)
- Inoue et al. 2024 — Real-time continuous VAP (arXiv:2401.04868)
- Skantze & Irfan, HRI 2025 — VAP on Furhat, dual-threshold policy (arXiv:2501.08946)
- Elmers et al., Interspeech 2025 — Triadic audio-only VAP (arXiv:2507.07518)
- Onishi et al., IEICE 2025 — Multimodal dyadic VAP (gaze/AU/head pose/gesture)
- Heo et al., Interspeech 2025 — Gaze-enhanced triadic turn prediction (arXiv:2505.13688)
- Liao et al. — Light-ASD (CVPR 2023) / LR-ASD (IJCV 2025), github.com/Junhua-Liao/Light-ASD
- Abbo et al. 2025 — Reactive baseline system (arXiv:2503.15496)

## Conventions for Claude Code

- Ask before changing any decision in "Key Design Decisions".
- Keep all timestamps in seconds (float) relative to the session sync beacon.
- Feature files: one parquet/npz per session per modality, aligned to 25 fps grid;
  column naming `<participant_id>_<feature>` (e.g. `P2_gaze_target`).
- Ethics: raw audio/video must never leave the local machine or be committed to git.
  Data directories are gitignored (`data/*`, `models/*`, `outputs/*`, `logs/*`).
- New modelling work goes in `src/hri_turn_taking/vap/`; the legacy sklearn TRP
  path (`models/trainer.py`, `data/schema.py`) is frozen as an ablation baseline.
- Evaluation must report balanced metrics + event-level tests, never raw
  accuracy alone (heavy class imbalance).
- `gh` CLI is installed at `C:\Program Files\GitHub CLI\gh.exe` and
  authenticated as LeGendZyx; remote is `origin` (HTTPS).
