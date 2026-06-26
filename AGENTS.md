# AGENTS.md — Agent Guide for the Predictive Turn-Taking Project

> Canonical instructions for AI coding agents (Codex, Claude Code, and any
> `agents.md`-aware tool) working in this repository. `CLAUDE.md` imports this
> file, so **this is the single source of truth — edit shared guidance here.**
> Last verified against the codebase: 2026-06-26.

## Communication

- **Language:** respond to the user in **Simplified Chinese (简体中文)** by default;
  keep code, commands, and error messages in English.

## 1. Project Overview

**MSc Robotics dissertation (King's College London, supervised by Dr. Oya
Celiktutan).**
Title: *Learning from Humans: A Hybrid Multimodal Framework for Predictive
Turn-Taking in Multi-Party HRI.*

Goal: replace the Furhat robot's reactive, silence-based turn-taking (wait
>1.5 s) with a **predictive multimodal model** that estimates upcoming speaker
transitions (VAP-style future voice-activity prediction) in 3–4 person group
conversations, then deploy it on the physical robot for closed-loop evaluation.

**Novelty claim** (vs. 2025 SOTA): multimodal (audio + visual) × multi-party
(3–4 people) × **deployed on a physical robot (Furhat) with online evaluation**.
Not claiming "first multi-party VAP" (Elmers et al., Interspeech 2025, audio-only
triadic) nor "first multimodal VAP" (Onishi et al., IEICE 2025, dyadic) — the
physical-HRI + multimodal + multi-party *combination* is the gap.

## 2. Research Framing (from the proposal, Feb 2026)

- **Problem**: reactive silence heuristics cause unnatural latency and are
  "blind to overlap"; audio-only speaker ID degrades badly in overlapping speech.
- **Hypotheses**: **H1** — fusing visual voice-activity cues with audio improves
  active-speaker detection during overlap; **H2** — a predictive model lets the
  robot identify turn-transition points *during* speech (prediction) instead of
  *after* silence (reaction), cutting turn-entry latency and missed turns.
- **Research questions**: RQ1 multimodal perception (fuse gaze + lip/ASD with
  audio DoA for robust speaker ID); RQ2 hybrid architecture (drop the predictive
  model into the modular Furhat stack, replacing heuristics); RQ3 evaluation
  (floor acquisition in overlap + latency vs. the reactive baseline).
- **Evaluation targets**: offline balanced metrics + VAP event tests; online
  KPIs (floor-entry latency, successful floor acquisition) + Godspeed (Animacy)
  + perceived-latency Likert. End-to-end latency budget **< 500 ms** (camera
  ~33 ms, feature extraction, inference, bridge, Furhat TTS startup 200–400 ms).

> **The implemented method has evolved past the proposal — follow the current
> design in §4, not the proposal's older framing.** The proposal described a
> 3-class LSTM (Speaking/Listening/TRP), MAR-based V-VAD as the *core* innovation,
> and a single `P(TRP) > δ` threshold. The project has since moved to VAP
> future-voice-activity prediction, **pretrained ASD as the core V-VAD** (MAR
> demoted to an ablation), and a **dual-threshold** policy. Do not re-introduce
> the older framing.

## 3. Supervisor-Agreed Direction (email thread, 2026)

- Supervisor (Oya) and the PhD (Michelle) **agreed to focus on VAP**. The
  reference Oya shared, **arXiv:2205.09812** (Ekstedt & Skantze, *Voice Activity
  Projection*, Interspeech 2022), is the original VAP paper — it reinforces the
  existing key reference, it does not change scope.
- Liaise with **Michelle** on the precise problem formulation (she owns audio
  sync and VAP framing on the team).
- **Open question — not yet decided, do not silently resolve:** single-channel
  vs. multi-channel audio as the VAP input. Hardware is a 4-channel ReSpeaker
  array; the current code uses a single-channel audio path + GCC-PHAT DoA as an
  auxiliary spatial feature. Surface this when relevant; ask before committing.

## 4. Key Design Decisions — DO NOT silently change

Ask the user before altering any of these.

1. **Training objective = VAP future voice-activity prediction**, not a
   hand-labeled 3-class Speaking/Listening/TRP classifier. Labels are generated
   automatically from per-participant lavalier-mic VAD; turn events (shift/hold,
   shift-prediction, backchannel, overlap) are *derived* at evaluation time.
2. **Two feature tiers — never mix them in the deployed model:**
   - *Gold tier* (training labels + offline eval ONLY): per-speaker lavalier VAD,
     multi-view 3D reconstruction.
   - *Robot-observable tier* (model INPUT): robot-view camera features (gaze,
     head pose, lip/ASD), mic-array audio. The deployed model must only consume
     features Furhat can obtain at runtime. A "full-feature oracle" may be trained
     as an upper bound for comparison.
3. **V-VAD = a pretrained lightweight ASD** (Light-ASD / LR-ASD, AVA-pretrained,
   open weights) producing per-face per-frame speaking probability. MAR (Mouth
   Aspect Ratio) is kept only as an interpretable ablation baseline, NOT the core.
4. **Audio encoding** prefers a pretrained CPC encoder (Ekstedt's VAP repo) over
   hand-crafted features; GCC-PHAT DoA is an auxiliary spatial feature (CPC does
   not encode direction). openSMILE/Praat only if explicit prosody (F0/energy/rate)
   is used.
5. **Training strategy**: pretrain/freeze encoders, fine-tune only a light fusion
   + prediction head — our dataset (~a few hours) is too small to train a
   transformer from scratch. A lightweight LSTM/GRU stays as a low-cost comparison;
   AMI corpus considered for pretraining experiments.
6. **Robot decision policy = dual-threshold p_now / p_future** (Skantze & Irfan,
   HRI 2025), not a single `P(TRP) > δ`.
7. **Evaluation reports balanced metrics + event-level tests, never raw accuracy
   alone** (heavy class imbalance): balanced accuracy / F1 / mAP, plus shift/hold,
   shift-prediction, and overlap tests. Main offline comparison = our own
   audio-only ablation (NOT the ≈18% figure from Abbo et al., a different
   measurement).

## 5. Repository Map

**Core method — `src/hri_turn_taking/vap/`:**
- `labels.py` — VAP bin labels from gold VAD (bins 0.2 / 0.6 / 1.2 / 2.0 s,
  active-ratio 0.5); p_now / p_future readout.
- `model.py` — `MultipartyVAP`: `Linear(input→64)` → causal encoder (2-layer
  Transformer default; GRU as low-cost comparison) → per-participant × per-bin
  logits + auxiliary current-VAD head (~100k params, independent per-bin BCE —
  a multi-party simplification of the 256-class joint VAP state space).
- `dataset.py` — wide table → tensors, feature normalisation, sliding windows;
  session-level splits only.
- `trainer.py` — masked, per-bin class-weighted BCE (absent slots + truncated
  horizon excluded), AdamW, early stopping; checkpoint stores model config +
  normaliser + label config; chunked full-session inference.
- `events.py` — shift/hold + shift-prediction event tests (shift score averaged
  over the silence gap — Ekstedt & Skantze convention).
- `policy.py` — dual-threshold policy gated on the **floor holder** (original
  Skantze & Irfan semantics) with debouncing + refractory period.
- `offline_demo.py` — predictive vs. reactive floor-entry latency comparison.

**Data schema — `src/hri_turn_taking/data/multiparty.py`:** wide frame table at
25 fps, columns `P{i}_<feature>`, up to 4 slots with `P{i}_present` masks
(3-party sessions pad slot 4). Input tier = `P{i}_asd_prob`, `P{i}_mar`,
`P{i}_gaze_robot`, `P{i}_head_yaw_deg`, `P{i}_gazed_at` + global `doa_angle_deg`,
`array_energy`, `f0_proxy` (**23 input dims**). `P{i}_vad` is gold tier — labels
and event eval ONLY, never model input.

**Frozen legacy ablation — do NOT extend.** The scikit-learn frame-level TRP path:
`data/{schema,synthetic,dataset}.py`, `models/{baseline,trainer,inference}.py`,
`audio/gcc_phat.py`, `vision/mouth.py`, `features/fusion.py`,
`evaluation/metrics.py`, `realtime/pipeline.py`. Kept solely as a low-cost
comparison baseline.

**Shared / other:** `cli.py`, `config.py`, `configs/default.yaml`,
`scripts/run_demo.py`, `tests/`, `docs/`. Architecture diagram: **`ARCHITECTURE.md`**
(Mermaid) — keep it in sync with `vap/model.py`, the policy thresholds, and the
feature schema.

## 6. Build / Test / Run

Environment: Windows, PowerShell primary. Python ≥ 3.10. PyTorch (CPU build) is a
core dependency, already installed in `.venv`. Do not assume a GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"      # core + pytest/ruff
```

Core VAP pipeline (the method that matters):

```powershell
hri-turn-taking vap-generate           # synthetic multiparty dataset (P{i}_ schema)
hri-turn-taking vap-train              # train MultipartyVAP; prints frame + event metrics
hri-turn-taking vap-demo               # predictive dual-threshold vs. reactive silence
```

Frozen legacy ablation (do not extend):
`hri-turn-taking generate-synthetic | train | predict <csv>`.

Run **both** before declaring work done:

```powershell
pytest                                 # 18 tests, incl. a causality test + a training smoke
ruff check .                           # line-length 100; rules E, F, I, UP, B
```

Reference numbers on synthetic data (12×180 s, 20 epochs, CPU): frame mAP ≈ 0.88,
shift/hold AP ≈ 0.68, shift-prediction AP ≈ 0.72 (ceiling-limited by
`cue_reliability`); the predictive policy enters gaps in 0.16–0.58 s while the
1.5 s silence baseline never enters.

## 7. Conventions

- All timestamps in seconds (float) relative to the session sync beacon.
- Feature files: one parquet/npz per session per modality, aligned to a 25 fps
  grid; column naming `P{i}_<feature>` (e.g. `P2_gaze_target`).
- **Real-data drop-in:** produce the same wide table from extracted features and
  feed `vap-train` directly; only `input_dim` changes if features are added.
- New modelling work goes in `src/hri_turn_taking/vap/`; the legacy sklearn TRP
  path is frozen.
- Synthetic features must be *probabilistic noisy correlates* of the speech state
  with genuine anticipatory cues (turn-final prosodic lowering, speaker gaze
  redirection, pre-onset articulation), gated by `cue_reliability`. They must
  **never** be deterministic functions of the labels (an earlier scaffold had that
  leakage — do not reintroduce it).

## 8. Project Status, Data & Team

- **Status (2026-06-26):** the multiparty VAP pipeline is implemented and verified
  end-to-end on synthetic data; real recordings are not yet available.
- **Sensors:** 6 calibrated monochrome cameras (+ calibration data), 2 RGB-D
  cameras, per-participant lavalier mics (gold audio), a 4-channel ReSpeaker mic
  array (robot's-ear spatial audio). Sync via audiovisual beacon (clap/beep);
  target alignment within ±1 frame (40 ms). Azure Kinect DK is EOL — use ZED 2 /
  Orbbec Femto Bolt.
- **Team feature-extraction split:** Michelle — audio / video-skeleton sync,
  lip-sync; Tomas — emotion, facial keypoints, gaze (MCGaze); Zobaid — transcript,
  whole-body 3D via calibration, hand gesture; **Yixuan (this repo's owner)** —
  speaking-towards / addressee (MMSI), speaker detection (ELAN start/end
  timestamps), head pose.
- **Yixuan's annotation pipeline:** semi-automatic — run VAD (Silero / pyannote)
  per lavalier track → import to ELAN → manual correction. ELAN must include an
  **overlap-type tier** distinguishing `backchannel` vs `competitive` (barge-in),
  required for the RQ3 overlap analysis.
- **Feature gaps still to cover:** prosody (F0/energy/rate) — unassigned, or rely
  on CPC embeddings implicitly; pretrained ASD speaking probability per face
  (Light-ASD — also useful to pre-label the ELAN annotation); discrete gaze target
  ("looking at A/B/C/robot") — derived by intersecting MCGaze vectors with 3D head
  positions, a derivation Yixuan may need to own; silence/pause/overlap statistics
  auto-derived from VAD for evaluation.

## 9. Guardrails / Do-Not

- **Ethics**: raw audio/video must never leave the local machine or be committed
  to git. Per `.gitignore`, everything under `data/`, `outputs/`, `models/`, and
  `logs/` is ignored — only `.gitkeep` placeholders and `data/README.md` are
  tracked — and `.claude/` (local agent settings & worktrees) is ignored too.
  Keep it that way.
- Do not extend the frozen legacy sklearn TRP path; treat it as a fixed baseline.
- Do not change any item in §4 (Key Design Decisions) without asking the user.
- Report balanced metrics + event tests, never raw accuracy alone.
- torch stays CPU-only here.

## 10. Verify Your Work

- Run `pytest` (must stay green, incl. the causality test that perturbs future
  frames) and `ruff check .` before reporting completion.
- If you touch `vap/model.py`, the policy thresholds, or the feature schema,
  update `ARCHITECTURE.md` to match.
- State explicitly what you ran and its result; if a step was skipped, say so.

## 11. Environment Notes

- `gh` CLI is at `C:\Program Files\GitHub CLI\gh.exe`, authenticated as
  `LeGendZyx`; git remote `origin` (HTTPS), default branch `main`. Repo:
  `github.com/LeGendZyx/hri-turn-taking` (private).
- Furhat SDK (Kotlin / remote API) ↔ ROS2 bridge is the eventual deployment path;
  add it only once the local hardware/runtime path is confirmed.
- Perception toolchain (team-wide): MediaPipe Face Mesh (landmarks), MCGaze
  (gaze), CMU `ssp` toolkit (body/gesture). ML pipeline is Python + PyTorch.

## 12. Scope / Priority (if time runs short)

1. **Core deliverable**: the offline multimodal prediction model (H1 + part of H2).
2. Second: the Furhat integration demo.
3. Stretch: the formal user study (can shrink to a 2–3 group pilot).

## Key References

- Ekstedt & Skantze 2022 — VAP (arXiv:2205.09812; code:
  github.com/ErikEkstedt/VoiceActivityProjection)
- Inoue et al. 2024 — Real-time continuous VAP (arXiv:2401.04868)
- Skantze & Irfan, HRI 2025 — VAP on Furhat, dual-threshold policy (arXiv:2501.08946)
- Elmers et al., Interspeech 2025 — Triadic audio-only VAP (arXiv:2507.07518)
- Onishi et al., IEICE 2025 — Multimodal dyadic VAP (gaze/AU/head pose/gesture)
- Heo et al., Interspeech 2025 — Gaze-enhanced triadic turn prediction (arXiv:2505.13688)
- Liao et al. — Light-ASD (CVPR 2023) / LR-ASD (IJCV 2025),
  github.com/Junhua-Liao/Light-ASD
- Abbo et al. 2025 — Reactive baseline system (arXiv:2503.15496)
