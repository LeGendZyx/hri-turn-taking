# Model Architecture — MultipartyVAP

Causal sequence model predicting each participant's **future voice activity**
over discretised bins (0.2 / 0.6 / 1.2 / 2.0 s), following the VAP literature
(Ekstedt & Skantze) scaled down to this project's data budget (~100k params).

Source: `src/hri_turn_taking/vap/model.py` · defaults from `VAPModelConfig`.

```mermaid
flowchart TB
    subgraph INPUT["Robot-observable input features — 25 fps, 23 dims"]
        direction TB
        PF["Per-participant ×4 slots:<br/>P{i}_asd_prob · P{i}_mar · P{i}_gaze_robot<br/>P{i}_head_yaw_deg · P{i}_gazed_at"]
        GF["Global:<br/>doa_angle_deg · array_energy · f0_proxy"]
    end

    PF --> CAT["(B, T, 23) feature tensor<br/><i>z-score FeatureNormalizer</i>"]
    GF --> CAT

    CAT --> PROJ["Input projection<br/>Linear 23 → 64 · LayerNorm · GELU · Dropout 0.1"]

    PROJ --> CHOICE{{"encoder ="}}

    subgraph TF["Transformer (default)"]
        direction TB
        PE["+ Sinusoidal positional encoding<br/>(max_len 8192; longer sessions →<br/>chunked inference w/ 500-frame left context)"]
        TE["TransformerEncoder × 2 layers<br/>d_model 64 · 4 heads · FFN 256<br/>pre-norm · GELU · <b>causal mask</b>"]
        PE --> TE
    end

    subgraph GRU["GRU (low-cost comparison)"]
        GR["GRU × 2 layers · hidden 64<br/>(causal by construction)"]
    end

    CHOICE -->|"transformer"| PE
    CHOICE -->|"gru"| GR

    TE --> H["Hidden states (B, T, 64)"]
    GR --> H

    H --> BH["<b>bin_head</b> Linear 64 → 16<br/>reshape (B, T, 4 participants, 4 bins)"]
    H --> VH["<b>vad_head</b> Linear 64 → 4<br/>auxiliary current-VAD task (weight 0.3)"]

    BH --> SIG["sigmoid → per-participant per-bin<br/>P(speaking in future bin)"]

    SIG --> READ["p_now = mean(bins 0–1) ≈ 0–0.6 s<br/>p_future = mean(bins 2–3) ≈ 0.6–2 s"]

    READ --> POL["DualThresholdPolicy (vap/policy.py)<br/>gate on <b>floor holder</b>: p_now &lt; 0.35 ∧ p_future &lt; 0.45<br/>debounce 3 frames · refractory 25 frames"]

    POL --> ACT(["Robot floor-entry decision"])

    style INPUT fill:#eef6ff,stroke:#4a90d9
    style TF fill:#f5f0ff,stroke:#8a6dc9
    style GRU fill:#f7f7f7,stroke:#999,stroke-dasharray: 5 5
    style POL fill:#fff4e6,stroke:#d9904a
    style ACT fill:#e8f5e9,stroke:#4a9d5b
```

## Training (vap/trainer.py)

```mermaid
flowchart LR
    GOLD["Gold tier: P{i}_vad<br/>(lavalier VAD — labels only,<br/><b>never</b> model input)"] --> LBL["future_activity_labels<br/>(T, 4, 4) binary bins<br/>active-ratio 0.5"]
    LBL --> LOSS["Masked BCE<br/>mask = valid frames × present slots<br/>per-bin pos_weight clipped [1, 10]"]
    PRED["bin_logits + vad_logits"] --> LOSS
    LOSS --> OPT["AdamW 3e-4 · grad clip 1.0<br/>early stopping (patience 4)<br/>session-level train/val split"]
```

## Key design notes

| Choice | Rationale |
|---|---|
| Independent per-bin BCE | Multi-party simplification of the 256-class joint VAP state space (joint space explodes to 2^12+ for 3–4 people) |
| Causal mask / GRU | No future leakage; verified by a unit test perturbing future frames |
| Two heads | Auxiliary current-VAD head regularises the shared encoder |
| Feature-agnostic input | Model only sees a `(B, T, features)` tensor — real extracted features drop in by changing `input_dim` |
| Floor-holder gating in policy | Max-over-all-humans gating never fires; original Skantze & Irfan (HRI 2025) semantics gate on the previous speaker yielding |

Reference numbers on synthetic data (12×180 s, 20 epochs, CPU): frame mAP ≈ 0.88,
shift/hold AP ≈ 0.68, shift-prediction AP ≈ 0.72; predictive policy enters gaps
in 0.16–0.58 s while the 1.5 s silence baseline never enters.
