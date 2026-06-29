# Research Note — Predictive Multi-Party Turn-Taking (VAP)

> 简短学术笔记：项目整体架构、待训练模型架构、数学依据。
> 工程细节见 [`ARCHITECTURE.md`](../ARCHITECTURE.md) 与 [`AGENTS.md`](../AGENTS.md)。
> Last verified against code: 2026-06-29.

---

## 中文

### 1. 项目整体架构

目标：用**预测式多模态模型**替换 Furhat 机器人基于静音阈值（>1.5 s）的反应式轮替策略，
在 3–4 人多方对话中**提前**估计说话权转移点，并部署到实体机器人做闭环评估。

数据分**两层**（部署模型严禁混用）：

- **Gold 层（仅用于训练标签与离线评估）**：每人佩戴的领夹麦 VAD、多视角 3D 重建。
- **Robot-observable 层（模型输入）**：机器人视角相机特征（gaze / head pose / 唇动 ASD）+
  麦克风阵列音频（DoA / 能量 / f0 代理）。部署模型只能消费 Furhat 运行时能拿到的特征。

数据流：

```
感知（相机+麦阵） → 25 fps 宽表特征 (23 维) → MultipartyVAP（因果序列模型）
   → 每人未来语音活动概率 → p_now / p_future 读出 → 双阈值策略 → 机器人抢话决策
```

特征宽表：每帧一行，每人 ×4 槽位，列名 `P{i}_<feature>`；3 人会话填充第 4 槽并用
`P{i}_present` 掩码。`P{i}_vad`（领夹麦 VAD）属 Gold 层，**只生成标签，绝不作模型输入**。

### 2. 待训练模型架构（MultipartyVAP，约 100k 参数）

```
输入 x_t ∈ R^23  →  Linear(23→64) + LayerNorm + GELU + Dropout
                 →  因果编码器（默认 2 层 Transformer，d=64，4 头，FFN 256，因果掩码；
                    GRU 2 层作为低成本对照）
                 →  隐状态 h_t ∈ R^64
                 →  bin_head:  Linear(64→16) → reshape (4 人 × 4 时间桶) 的 logits
                 →  vad_head:  Linear(64→4)  当前 VAD 辅助任务（权重 0.3）
```

时间桶边界 0.2 / 0.6 / 1.2 / 2.0 s（共 4 桶）。Transformer 用正弦位置编码 + 因果掩码保证
不泄露未来；长会话用分块推理（500 帧左上下文）。

### 3. 数学依据

**(a) VAP 标签生成。** 对参与者 $p$、未来第 $b$ 个时间桶 $[t+s_{b-1},\,t+s_b]$，
按 Gold VAD 在该窗内的活跃比例阈值 0.5 二值化：

$$ y_{p,b}(t) = \mathbb{1}\!\left[\tfrac{1}{|W_b|}\sum_{\tau\in W_b}\text{VAD}_p(\tau)\ \ge\ 0.5\right] $$

**(b) 预测。** 模型输出每人每桶的 logit $\ell_{p,b}(t)$，预测概率
$\hat y_{p,b}=\sigma(\ell_{p,b})$。这里把原始 VAP 在 2 人上的 256 类**联合**状态空间，
简化为多方场景下**逐桶独立的 Bernoulli**——3–4 人的联合空间会爆炸到 $2^{16}$，
故按 (人, 桶) 因子分解。

**(c) 损失：带掩码、逐桶类别加权的 BCE。**

$$ \mathcal{L}=\frac{1}{|\mathcal{M}|}\sum_{(t,p)\in\mathcal{M}}\sum_{b} w_b\cdot
\text{BCE}\!\big(\sigma(\ell_{p,b}),\,y_{p,b}\big)\;+\;\lambda\,\mathcal{L}_{\text{vad}} $$

掩码 $\mathcal{M}$ = 有效帧（剔除被截断的预测视界）× 在场槽位；
逐桶正样本权重 $w_b=\mathrm{clip}(N^{-}_b/N^{+}_b,\,1,\,10)$ 抵消类别不均衡；
辅助 VAD 项权重 $\lambda=0.3$。优化器 AdamW（lr 3e-4，梯度裁剪 1.0，早停），
**仅按会话切分** train/val。

**(d) 读出与决策。** 对每人取
$p_{\text{now}}=\text{mean}(\hat y_{p,0},\hat y_{p,1})$（≈0–0.6 s）、
$p_{\text{future}}=\text{mean}(\hat y_{p,2},\hat y_{p,3})$（≈0.6–2 s）。
**双阈值策略**（Skantze & Irfan, HRI 2025）：仅对**当前话权持有者**判定让出——
当 $p_{\text{now}}<0.35\ \wedge\ p_{\text{future}}<0.45$ 时允许机器人抢话，
配合去抖（3 帧）与不应期（25 帧）。这避免了"对所有人取 max"恒不触发的退化。

**评估**：报告平衡指标 + 事件级测试（frame mAP、shift/hold AP、shift-prediction AP），
绝不只看原始准确率（重度类别不均衡）。

---

## English

### 1. System architecture

Replace Furhat's reactive silence threshold (>1.5 s) with a **predictive multimodal
model** that anticipates speaker transitions in 3–4-party conversation, deployed on the
physical robot for closed-loop evaluation. Two strictly separated feature tiers:

- **Gold tier (training labels + offline eval only):** per-speaker lavalier VAD,
  multi-view 3D reconstruction.
- **Robot-observable tier (model input):** robot-view camera features (gaze, head pose,
  lip/ASD) + mic-array audio (DoA, energy, f0 proxy).

Flow: `perception → 25 fps wide table (23 dims) → MultipartyVAP → per-participant future
voice-activity → p_now / p_future → dual-threshold policy → robot floor-entry decision`.
`P{i}_vad` is gold-tier: it generates labels and is **never** a model input.

### 2. Model (MultipartyVAP, ≈100k params)

`x_t∈R^23 → Linear(23→64)+LN+GELU+Dropout → causal encoder (2-layer Transformer, d=64,
4 heads, FFN 256, causal mask; GRU as low-cost baseline) → h_t∈R^64 →` two heads:
`bin_head Linear(64→16)→(4 participants × 4 bins)` logits, and an auxiliary
`vad_head Linear(64→4)` current-VAD task (weight 0.3). Bin edges 0.2/0.6/1.2/2.0 s.
Causal mask / GRU prevents future leakage (unit-tested); long sessions use chunked
inference with 500-frame left context.

### 3. Mathematical basis

**(a) Labels.** For participant $p$, future bin $b$ over $[t+s_{b-1},t+s_b]$:
$y_{p,b}(t)=\mathbb{1}[\frac{1}{|W_b|}\sum_{\tau\in W_b}\text{VAD}_p(\tau)\ge 0.5]$.

**(b) Prediction.** $\hat y_{p,b}=\sigma(\ell_{p,b}(t))$. The original VAP joint 256-state
space (2 speakers) is factorised into **independent per-bin Bernoulli** because the joint
space explodes to $2^{16}$ for 3–4 people.

**(c) Loss — masked, per-bin class-weighted BCE.**
$\mathcal{L}=\frac{1}{|\mathcal{M}|}\sum_{(t,p)\in\mathcal{M}}\sum_b w_b\,
\text{BCE}(\sigma(\ell_{p,b}),y_{p,b})+\lambda\mathcal{L}_{\text{vad}}$, with mask
$\mathcal{M}$ = valid frames × present slots, $w_b=\mathrm{clip}(N^-_b/N^+_b,1,10)$,
$\lambda=0.3$. AdamW (3e-4), grad clip 1.0, early stopping, session-level splits.

**(d) Readout & policy.** $p_{\text{now}}=\text{mean}(\hat y_{p,0},\hat y_{p,1})$,
$p_{\text{future}}=\text{mean}(\hat y_{p,2},\hat y_{p,3})$. Dual-threshold policy
(Skantze & Irfan, HRI 2025) gates on the **floor holder**: the robot may take the floor
when $p_{\text{now}}<0.35 \wedge p_{\text{future}}<0.45$, with debounce (3 frames) and
refractory (25 frames). Evaluation reports balanced + event-level metrics (frame mAP,
shift/hold AP, shift-prediction AP), never raw accuracy alone.

---

### Key references

- Ekstedt & Skantze 2022 — Voice Activity Projection (arXiv:2205.09812)
- Skantze & Irfan, HRI 2025 — VAP on Furhat, dual-threshold policy (arXiv:2501.08946)
- Elmers et al., Interspeech 2025 — Triadic audio-only VAP (arXiv:2507.07518)
- Onishi et al., IEICE 2025 — Multimodal dyadic VAP
