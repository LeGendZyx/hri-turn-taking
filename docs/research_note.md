# Research Note — Predictive Multi-Party Turn-Taking (VAP)

> 论文写作底稿：方法、数学基础、评估协议与合成数据试点结果。
> Working note for the dissertation: method, mathematical basis, evaluation
> protocol, and pilot results on synthetic data.
> 工程细节见 [`ARCHITECTURE.md`](../ARCHITECTURE.md) 与 [`AGENTS.md`](../AGENTS.md)。
> 最新监督范围与扩展门槛见 [`PROJECT_SCOPE.md`](PROJECT_SCOPE.md)；如有冲突，以该文件为准。
> Last status sync against code and readiness ledgers: 2026-08-16.
> 修订 / Revision: 2026-07-03 — 模型由扁平编码器（v1）升级为**参与者结构化**架构（v2）；
> v1 保留为消融基线。部署模型使用单通道音频；真实数据的 DoA 列为惰性常数，多通道仅作消融。
> 修订 / Revision: 2026-07-05 — 补充语料设计："人扮机器人"示范范式（§1 末段）；新增
> robot-entry 事件测试、示范驱动策略（DemonstrationPolicy）与按组切分（§4、试点结果）。
> 修订 / Revision: 2026-08-03 — owner 定案：核心声学表示 = 因果 energy+F0+全局 voiced 列
> （CPC 降为 gated 可选消融；schema v2 已实现 32 原始列→33 模型维）；首个正式 run
> `aux_vad_weight=0.0`；speaking_towards 数据整体退出论文范围（无输入、无消融、无
> addressee 指标）。汇总见 `docs/final_model_plan.md`。
> 状态同步 / Status sync: 2026-08-16 — 首轮标签为仅处理四个不可能区间后的
> **human-reviewed-Transcript-derived VAD**；声学阈值 `0.45` 已冻结，首轮矩阵为
> `A / H / A+H`，ASD/Gaze/Facial 延后。owner 已接受本地单用户/单写者发布边界，残余并发
> 替换风险如实披露；同一阻塞最多三轮修复与复测。

---

## 中文

### 1. 问题与系统概览

现有多方人机对话系统（含 Furhat 默认行为）采用**反应式**轮替：等待 >1.5 s 静音再应答，
延迟不自然，且对重叠语音“失明”。本项目的**必需核心交付**是参与者级 VAP：以 VAP
（Voice Activity Projection, Ekstedt & Skantze 2022）为训练目标，在 3–4 人对话中逐帧输出
一个 `time × participant × 4 future bins` 概率张量，并在按组互斥的真实数据外层测试上完成
首轮 `A / H / A+H` 对比。p_now/p_future、Shift/Hold、Next Speaker、重叠解析和机器人进场均
由这一张量确定性读出或评估，不是额外学习输出。

实体 Furhat 部署与 GVAP 均不是完成核心论文结果的条件：只有参与者级 VAP 在真实数据上
跑通并完成核心消融后，才按 `PROJECT_SCOPE.md` 的 decision gate 选择 optional stretch。
GVAP 若开展，只能作为不替换单输出基线的后续消融；Furhat 若开展，则是 replay/control 或
完整 live-loop 部署扩展。核心新颖性不依赖二者必然完成，也不声称“首个多方 VAP”或“首个
多模态 VAP”。

数据分两层，部署模型严禁混用：

- **Gold 层**（仅训练标签 + 离线评估）：首轮使用人审 Transcript 派生 VAD；逐人领夹麦
  仅作全语料一致性诊断或明确命名的 oracle，多视角 3D 仅作离线基础设施；
- **Robot-observable 层**（模型输入）：机器人视角相机特征（ASD 说话概率、MAR、注视、
  头姿、逐帧观测有效位 pose_valid 与连续观测质量 pose_quality）+ 全局声学表示。真实声学
  表示只能来自混合总轨；**2026-08-03 owner 定案：因果 energy+F0+全局 voiced 列为核心
  声学表示**（CPC 降为 gated 可选消融；voiced 列属已实现的 schema v2）。DoA 列仅在合成/
  麦阵消融管线有真实取值，在真实数据上为惰性常数。
- **音频通道**（2026-07-03 决策）：部署输入为**单通道**（机器人耳）音频；多通道与
  逐人领夹麦音频仅作离线消融 / oracle 上界。GCC-PHAT DoA 辅助特征仅存在于合成/
  麦阵消融管线（真实语料未录阵列，见下）。

当前 Gaze 交付仅提供 nullable 2D `robot / participant / elsewhere` 证据，没有已完成的
3D `gazed_at` 派生。owner 已在结果检查前冻结声学阈值 `0.45`。四天首轮只比较 `A`（因果
混合声学）、`H`（头姿+有效性/质量）与 `A+H`；ASD、Gaze 与 Facial 不进入首轮矩阵。

训练表中的 `P{i}_head_yaw_deg` 当前由多视角 `world_persons.csv` 的 facing vector 投影得到。
head yaw 这种特征在运行时可由机器人视角估计，因此仍属 robot-observable feature kind；但
训练来源使用 multi-view gold-tier infrastructure，形成需要报告的 train/deploy domain shift，
而不是把 gold signal 作为模型输入的 tier violation。

数据流：

```
感知（相机+音频）→ 25 fps 宽表（32 原始列 → 33 模型维；DoA 编码为 sin/cos）
   → MultipartyVAP（参与者结构化因果模型）→ 每人×每桶未来语音活动概率
   → 离线帧级/事件级评估与 p_now / p_future 读出
   → [真实核心结果后的可选门禁] 双阈值控制与 Furhat 部署
```

**语料与"人扮机器人"示范范式（2026-07-05 确认）。** 训练语料为已完成录制的 3–4 人
情景对话（接待/服务机器人场景，每段情景视频约 3–7 分钟）。每组一名参与者**扮演机器人**：站立于规划中的可选 Furhat 部署
的位置，并被要求**以自然人的方式回应**；其余 2–3 人与其面对面。6 路相机约 3 路朝向
扮演者（记录示范行为）、3 路朝向其余参与者（近似机器人视角）；全体参与者（含扮演者）
佩戴领夹麦，因此机器人槽位也有逐人音频一致性证据；首轮 gold-tier 标签来自人审
Transcript 派生 VAD，而非 sample-perfect 领夹麦 VAD。音频清单：逐人 DJI 无线领夹麦
（一致性诊断/oracle）与一条混合总轨（全局声学特征来源）；**未录制麦克风阵列**，DoA 特征
**已决定在真实数据上删除**（2026-07-10；不做几何模拟以避免 oracle 泄漏——列保留为惰性
常数，常数经 z-score 归零，模型布局不变；合成管线与麦阵消融仍可使用）。该设计
使机器人槽位成为**受监督的行为示范**：模型对该槽位的预测即"处于机器人位置的人类此刻是否会开口"——从人类示范中
蒸馏的接话/让出策略，与题名 *Learning from Humans* 字面对应（先例：Skantze & Irfan 在
纯人-人对话上训练、部署时将 TTS 作为机器人声道输入）。部署时该槽位特征由 Furhat 自身
状态替换（TTS 活动、指令注视/头姿、口型参数），替换引起的分布偏移可先以离线特征替换
实验量化。评估上，示范者实际的进场/未进场为"应否进场"提供真值，事件测试可按机器人
槽位切片，在用户研究之前即离线评估机器人行为（已实现：robot-entry 事件测试与示范
一致性策略对比）。须写明的局限：其一，人扮
机器人与真实机器人之间存在迁移间隙（人类韵律与时机；面对真实机器人时参与者行为会
改变；TTS 启动延迟 200–400 ms 叠加于策略触发之上，二者应分开报告，示范者自身进场延迟
即人类上限）；其二，习得策略属接待场景域，与部署场景匹配，但不宜外推为通用闲聊策略。

### 2. 模型：参与者结构化 MultipartyVAP（≈130k 参数）

下列宽度、层数、头数与 250 帧窗口只是**当前开发脚手架默认值/候选值**，用于描述现有
实现，不是 owner 已批准的正式 counted-experiment 配置；正式配置须在真实数据 readiness
满足后另行裁定并版本化。

每位参与者是一个 token：$x^{(i)}_t = [\text{本人特征(7)} \,\|\, \text{广播全局特征(5)}]$，
过共享嵌入 Linear(12→64)+LN+GELU，乘以在场掩码。堆叠 2 个 pre-norm 残差块，每块三个子层：

1. **时间混合**：各参与者沿时间轴做因果多头注意力（4 头，ALiBi 相对偏置，250 帧带状
   窗口），权重沿参与者共享；单层 GRU 为低成本替代；
2. **参与者混合**：同帧 ≤4 个 token 间的多头注意力，缺席槽位作为 key 被掩蔽；
3. **前馈**：64→256→64。

最终 LayerNorm 后接每 token 共享的双头：`bin_head` Linear(64→4) 输出每人 4 个未来时间桶
logits；`vad_head` Linear(64→1) 为当前 VAD 辅助任务。其权重当前脚手架默认是 0.3；
**2026-08-03 owner 定案：首个正式 run 取 0.0**（0.0 vs 0.3 配对比较仍待另行批准）。

**设计依据**（每条均有单元测试）：

- **槽位排列等变**：预测随参与者置换而置换——任意的槽位分配不改变行为；参与者处理权重
  全共享（小数据下每参数约 4× 监督），同一套权重原生覆盖 3 人与 4 人会话；
- **关系归纳偏置**：轮替线索本质是关系性的（谁被注视、谁在竞争话权），参与者轴注意力
  显式建模跨人交互；缺席（填充）槽位被掩蔽，不可能影响在场者；
- **相对位置（ALiBi）**：训练窗口恒从位置 0 开始而整段推理达位置 ~4500，绝对位置编码存在
  训练/推理失配；ALiBi 只依赖相对距离，且各 head 斜率几何递减、覆盖长短记忆；
- **流式一致性**：带状因果注意力使分块流式推理与整段前向**逐位一致**（左上下文 =
  n_layers × 窗口，即堆叠感受野），满足机器人在线运行的增量推理需求。

### 3. 数学基础

**(a) VAP 标签。** 对参与者 $p$，四个未来桶明确为半开区间
$W_0=[t,t+0.2)$、$W_1=[t+0.2,t+0.6)$、$W_2=[t+0.6,t+1.2)$ 和
$W_3=[t+1.2,t+2.0)$；按 Gold VAD 活跃比例二值化：

$$ y_{p,b}(t) = \mathbb{1}\!\left[\tfrac{1}{|W_b|}\sum_{\tau\in W_b}\text{VAD}_p(\tau)\ \ge\ 0.5\right] $$

**(b) 因子分解预测。** $\hat y_{p,b}=\sigma(\ell_{p,b}(t))$。原始双人 VAP 的 256 类
**联合**状态空间在 3–4 人下会爆炸至 $2^{16}$，故按（人, 桶）因子分解为独立 Bernoulli。

**(c) 损失：带掩码、逐桶类别加权 BCE。**

令 $z_{t,p,b}$ 为未来桶 logit，$a_{t,p}$ 为辅助 VAD logit，
$\mathcal{M}=\{(t,p,b):\text{完整未来视界且槽位在场}\}$，
$\mathcal{A}=\{(t,p):\text{槽位在场}\}$。代码实际计算

$$
\mathcal{L}_{\mathrm{bin}}=
\frac{\sum_{(t,p,b)\in\mathcal{M}}\left[
(1-y_{t,p,b})\,\operatorname{softplus}(z_{t,p,b})
+q_b y_{t,p,b}\,\operatorname{softplus}(-z_{t,p,b})\right]}
{|\mathcal{M}|},
$$

$$
\mathcal{L}_{\mathrm{vad}}=
\frac{\sum_{(t,p)\in\mathcal{A}}\left[
(1-v_{t,p})\,\operatorname{softplus}(a_{t,p})
+v_{t,p}\,\operatorname{softplus}(-a_{t,p})\right]}
{|\mathcal{A}|},
\qquad
\mathcal{L}=\mathcal{L}_{\mathrm{bin}}+\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{vad}}.
$$

因此 $q_b$（PyTorch `pos_weight`）**只乘正类项**，负类项不加权；分母分别是有效
$(t,p,b)$ 三元组数与在场 $(t,p)$ 对数。当前脚手架以截断到 $[1,10]$ 的训练集
$N_b^-/N_b^+$ 作为 $q_b$，并默认 $\lambda_{\mathrm{aux}}=0.3$、AdamW 学习率
$3\times10^{-4}$、10% warmup、余弦调度和梯度裁剪 1.0；除 $\lambda_{\mathrm{aux}}$
（2026-08-03 定案：首个正式 run 取 0）外，这些数值均是
**candidate scaffold defaults**，不是 frozen decision。当前 runner 的单次
train/inner-validation 结果只能标为 development；正式协议中的 train、inner-validation 与
未参与选模的 outer-test 三个边界都必须 group-disjoint。

**(d) 读出与决策。** 按桶时长 $d_b$=0.2/0.4/0.6/0.8 s 加权：

$$ p_{\text{now}}=\frac{\sum_{b\in\{0,1\}} d_b\,\hat y_{p,b}}{\sum_{b\in\{0,1\}} d_b},\qquad
p_{\text{future}}=\frac{\sum_{b\in\{2,3\}} d_b\,\hat y_{p,b}}{\sum_{b\in\{2,3\}} d_b} $$

即"区间内含语音的期望时间占比"（简单平均会高估短的近桶）。双阈值策略仅对**当前话权
持有者**判定让出：$p_{\text{now}}<0.35 \wedge p_{\text{future}}<0.45$ 时允许抢话，配合
去抖（3 帧）与不应期（25 帧）；对所有人取 max 的门控会退化为恒不触发。
这些阈值与帧数描述当前离线脚手架，不是 owner 已冻结的正式部署参数。

### 4. 评估协议

- **帧级**：mAP（总体 + 逐桶）、balanced accuracy、F1——重度类别不均衡，不报原始准确率；
- **事件级**（由 gold VAD 在评估时导出；事件跨验证会话**汇总**后计分，逐会话宏平均在
  几十个事件下噪声过大）：
  - *shift/hold*：双侧为单一说话人的互静音间隙，话权是否易主（评分 = 间隙内逐帧
    challenger/holder 未来活动比值的均值）；
  - *shift-prediction*：只读 offset 前 previous speaker 连续独占说话的尾段；请求
    0.5 s 时仍以上游事件提取已验证的 0.4 s context 为上限；
  - *overlap-resolution*（服务 RQ3）：challenger 压话后话权花落谁家——区分 backchannel /
    失败抢话（holder 保持）与竞争性抢话成功（holder 让出），评分读出于重叠前 0.3 s;
  - *robot-entry*（示范一致性）：仅取人类话轮之后的间隙，标签 = 扮演机器人的示范者是否
    实际进场，得分 = 机器人槽位在间隙内的 p_future 均值——"机器人该不该接话"的直接
    离线检验；
- **策略级**：与 1.5 s 静音反应式基线对比留出会话的 gap 进入率与进入时延；并以示范者
  实际进/未进为真值，对比两种双阈值模式（floor-holder 门控 vs 机器人槽位示范驱动）的
  召回、误进率与相对示范者的提前量。
- **切分纪律**：真实数据的 train、inner-validation、outer-test 须全部按**组**（人员不相交）
  划分——同组录制的多个情景视频不得跨任何边界，防止说话人身份泄漏虚增指标。

---

## English

### 1. Problem & system overview

Deployed multi-party HRI systems (including Furhat's default) take turns
*reactively* — respond after >1.5 s of silence — causing unnatural latency and
blindness to overlap. The **mandatory core deliverable** is one
participant-wise VAP tensor: trained with the VAP objective (Ekstedt & Skantze
2022), the model continuously predicts each participant's voice activity over
four future bins in 3–4-party conversation, with shape
`time × participant × 4 future bins`. The first core real-data result is the
group-disjoint `A / H / A+H` comparison; ASD, Gaze, and Facial inputs are deferred.
p_now/p_future, Shift/Hold, Next Speaker, overlap resolution, and robot entry
are deterministic readouts or evaluations of that same tensor, not additional
learned outputs.

Physical Furhat deployment and GVAP are both decision-gated optional stretches
after the participant-wise VAP works on real data and the core ablations are
complete. A later GVAP head may only be evaluated as an ablation against the
unchanged single-output baseline; Furhat may be attempted as replay/control or
a full live loop if its external dependencies are ready. Neither is required
for the core novelty, which does not claim the first multi-party VAP or the
first multimodal VAP.

Two strictly separated feature tiers: **gold** (human-reviewed-Transcript-derived
VAD for the first run; per-speaker lavaliers as consistency/oracle evidence;
multi-view 3D — labels/offline eval only) vs. **robot-observable** (robot-view
ASD probability, MAR, gaze, head pose, per-frame pose_valid flag + continuous
pose_quality, plus a global acoustic representation from the combined mix).
**Owner decision 2026-08-03: causal energy + F0 plus a global `voiced` column is the core
acoustic representation** (CPC demoted to a gated optional ablation; the `voiced` column is
implemented in schema v2). The DoA
column only carries varying values in the synthetic / mic-array-ablation
pipeline and is inert on real data. **Audio channels (decided 2026-07-03):** the
deployed input is single-channel (robot's-ear) audio;
multi-channel and per-speaker lavalier audio serve only as offline ablations /
oracle upper bounds, and the GCC-PHAT DoA auxiliary exists only there (no
array was recorded, see below).

Current Gaze evidence is nullable 2D `robot / participant / elsewhere`, not an
existing 3D-derived `gazed_at` input. The owner froze the acoustic voicing
threshold at `0.45` before outcome inspection. The four-day first counted
matrix is `A / H / A+H`; ASD, Gaze, and Facial inputs are deferred.

Training-time `P{i}_head_yaw_deg` is currently projected from facing vectors in
multi-view `world_persons.csv`. Head yaw remains a robot-observable feature
kind because it can be estimated from the robot view at runtime, but the
gold-tier multi-view training source creates a documented train/deploy domain
shift rather than a tier violation.

Flow: `perception → 25 fps wide table (32 raw columns → 33 model dims; DoA as
sin/cos) → MultipartyVAP → per-participant future-bin probabilities → offline
frame/event evaluation and duration-weighted p_now / p_future → [optional gate
after the core real-data result] policy/Furhat deployment`.

**Corpus and the human-plays-robot demonstration paradigm (confirmed
2026-07-05).** The fully recorded training corpus consists of 3–4-person
scenario conversations (receptionist-style service-robot scenarios; each
scenario video runs ≈3–7 minutes). In each
group one participant **role-plays the robot**: standing where Furhat would be
placed in an optional deployment and instructed to respond **naturally, as a human**, facing the other
2–3 participants. Of the six cameras, ≈3 face the robot-player (capturing the
demonstrator's behaviour) and ≈3 face the others (≈ the robot's-eye view);
everyone, including the robot-player, wore a lavalier microphone, so the robot
slot has the same per-participant consistency evidence as every other slot.
First-run supervision is human-reviewed-Transcript-derived VAD, not
sample-perfect lavalier VAD. Audio inventory: four per-participant DJI wireless
lavaliers (consistency/oracle evidence) and their derived
four-channel aligned master; there is no independent robot-ear/room mix. The
global acoustic features use its sample-replayed equal-weight mono average as
an explicitly reported close-talk surrogate. **No microphone array was recorded**,
so the DoA feature is **dropped on real data** (decided 2026-07-10; no geometry
simulation — oracle-leak risk; the column remains as an inert constant that
z-scores to zero, so the model layout is unchanged and the synthetic pipeline /
mic-array ablation keep using it). The robot slot thereby becomes a
**supervised behavioural demonstration**: the model's prediction for that slot
reads as "would a human in the robot's position speak now" — a
floor-taking/yielding policy distilled from human demonstration, the literal
*Learning from Humans* reading (precedent: Skantze & Irfan trained on
human-human dialogue and fed TTS into the robot channel at deployment). At
deployment the slot's features are substituted by Furhat's own state (TTS
activity, commanded gaze/head pose, lip-sync parameter); the induced shift can
be quantified offline by feature substitution on validation sessions. For
evaluation, the demonstrator's actual entries/non-entries supply ground truth
for "should the robot enter", and event tests can be sliced by the robot
slot — an offline assessment of robot behaviour before any user study
(implemented: the robot-entry event test and the demonstration-agreement
policy comparison). Two limitations to state: the human-plays-robot transfer gap (human
prosody/timing; people behave differently toward real robots; the 200–400 ms
TTS startup adds to any policy trigger and should be reported separately, with
the demonstrator's own entry latency as the human ceiling), and the domain of
the learned policy (receptionist-style, matched to deployment; not general
open conversation).

### 2. Model: participant-structured MultipartyVAP (≈130k params)

The widths, layer/head counts, and 250-frame window below are **current
development-scaffold defaults/candidates**. They describe the implemented
runner but are not owner-approved counted-experiment settings; the formal
configuration must be versioned after real-data readiness is established.

Each participant is a token $x^{(i)}_t = [\text{own feats (7)} \| \text{broadcast
global feats (5)}]$ through a shared embedding Linear(12→64), multiplied by a
presence mask. Two pre-norm residual blocks, each with three sublayers:
(i) **temporal mixing** — causal MHA over time per participant with shared
weights (4 heads, ALiBi relative bias, banded 250-frame window; a 1-layer GRU
is the low-cost alternative); (ii) **participant mixing** — MHA across the ≤4
same-frame tokens, absent slots masked out as keys; (iii) FFN 64→256→64.
A final LayerNorm feeds two shared per-token heads: `bin_head` Linear(64→4)
(future-bin logits per participant) and auxiliary `vad_head` Linear(64→1)
(current VAD). Its current scaffold default is 0.3; owner decision
2026-08-03: the first formal run uses 0.0 (the 0.0-vs-0.3 paired comparison still needs
separate approval).

**Design rationale** (each property unit-tested): slot **permutation
equivariance** (arbitrary slot assignment cannot change behaviour; fully
shared participant weights ≈ 4× supervision per parameter on a few-hours
dataset; one set of weights natively serves 3- and 4-party sessions);
**relational inductive bias** (turn-taking cues are relational — who is gazed
at, who competes for the floor — modelled explicitly by participant-axis
attention; padded slots cannot influence present participants); **relative
position via ALiBi** (removes the absolute-position train/inference mismatch
between windows starting at position 0 and sessions reaching ~4500; head
slopes span short/long memory); **streaming consistency** (banded causal
attention makes chunked inference *exactly* equal to a full pass with left
context = n_layers × window — the stacked receptive field — as required for
incremental on-robot inference).

### 3. Mathematical basis

**(a) Labels.** For participant $p$, the four future bins are the half-open
intervals $W_0=[t,t+0.2)$, $W_1=[t+0.2,t+0.6)$,
$W_2=[t+0.6,t+1.2)$, and $W_3=[t+1.2,t+2.0)$:
$y_{p,b}(t)=\mathbb{1}[\frac{1}{|W_b|}\sum_{\tau\in W_b}\text{VAD}_p(\tau)\ge 0.5]$.

**(b) Factorised prediction.** $\hat y_{p,b}=\sigma(\ell_{p,b}(t))$; the
dyadic VAP's 256-state joint space explodes to $2^{16}$ for 3–4 people, so we
factorise over (participant, bin) as independent Bernoullis.

**(c) Loss — masked, per-bin class-weighted BCE.**
Let $z_{t,p,b}$ be a future-bin logit, $a_{t,p}$ an auxiliary-VAD logit,
$\mathcal{M}$ the set of valid $(t,p,b)$ triples (the full configured future
horizon exists and slot $p$ is present), and $\mathcal{A}$ the set of present
$(t,p)$ pairs. The implementation computes

$$
\mathcal{L}_{\mathrm{bin}}=
\frac{\sum_{\mathcal{M}}\left[(1-y)\operatorname{softplus}(z)
+q_b y\operatorname{softplus}(-z)\right]}{|\mathcal{M}|},
\quad
\mathcal{L}_{\mathrm{vad}}=
\frac{\sum_{\mathcal{A}}\left[(1-v)\operatorname{softplus}(a)
+v\operatorname{softplus}(-a)\right]}{|\mathcal{A}|},
\quad
\mathcal{L}=\mathcal{L}_{\mathrm{bin}}+
\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{vad}}.
$$

Thus PyTorch `pos_weight` $q_b$ multiplies only the positive term;
negatives are unweighted. The denominators are respectively the number of
valid $(t,p,b)$ triples and present $(t,p)$ pairs. The current scaffold uses
clipped training-set $N_b^-/N_b^+$ for $q_b$, $\lambda_{\mathrm{aux}}=0.3$,
AdamW at $3\times10^{-4}$, 10% warmup, cosine scheduling, and gradient
clipping at 1.0. Apart from $\lambda_{\mathrm{aux}}$ (fixed 2026-08-03 at 0 for the
first formal run), these are **candidate scaffold defaults**, not frozen
decisions. The single train/inner-validation runner reports development
results only; formal train, inner-validation, and untouched outer-test
boundaries must all be group-disjoint.

**(d) Readout & policy.** Duration-weighted over bin widths
$d_b$ = 0.2/0.4/0.6/0.8 s:
$p_{\text{now}}=\sum_{b\le 1} d_b\hat y_{p,b}/\sum_{b\le 1} d_b$,
$p_{\text{future}}=\sum_{b\ge 2} d_b\hat y_{p,b}/\sum_{b\ge 2} d_b$ — the
expected fraction of each interval containing speech. The dual-threshold
policy gates on the **floor holder**: enter when
$p_{\text{now}}<0.35 \wedge p_{\text{future}}<0.45$, with debounce (3 frames)
and refractory (25 frames); max-over-all gating degenerates to never firing.
These thresholds and frame counts describe the current offline scaffold and
are not owner-frozen deployment settings.

### 4. Evaluation protocol

Frame level: mAP (overall + per bin), balanced accuracy, F1 — never raw
accuracy under heavy class imbalance. Event level (derived from gold VAD at
eval time; events **pooled across validation sessions** before scoring):
*shift/hold* (does the floor change hands at a clean mutual-silence gap;
score = mean over gap frames of a challenger-vs-holder future-activity
ratio), *shift-prediction* (anticipation over the previous speaker's contiguous
sole-speaker tail, capped at the event extractor's verified 0.4 s context, while the
speaker still talks), and *overlap-resolution* (RQ3: when a challenger speaks
over a stable holder, who keeps the floor — separating backchannels / failed
barge-ins from competitive wins; scored over the first 0.3 s of overlap), and
*robot-entry* (demonstration agreement: at gaps following a human turn,
label = whether the robot-role demonstrator actually took the floor, score =
the robot slot's mean p_future over the gap — a direct offline test of
"should the robot enter"). Policy level: gap-entry rate and latency on
held-out sessions vs. the 1.5 s reactive silence baseline, plus a
demonstration-agreement comparison of the two dual-threshold modes
(floor-holder gated vs. robot-slot demonstration driven) reporting recall,
false entries and lead time against the demonstrator's own onsets. On real
data, train, inner-validation, and outer-test boundaries must all be
**group-level (person-disjoint)**: all scenario videos recorded by the same
group of people stay on one side of every boundary, so speaker-identity
leakage cannot inflate the metrics.

---

## 试点结果 / Pilot results (synthetic, 12×180 s, 3-party, 20 epochs, CPU)

Validation = 3 held-out sessions (107 shift/hold events, 59 shifts; 47
overlap events, 11 holder-yields). "Flat" = the v1 non-structured encoder
(single Linear over concatenated slots), kept as an ablation.

| Metric (validation) | Flat (v1, ablation) | Structured Transformer (v2) | Structured GRU (v2) |
|---|---|---|---|
| Frame mAP | 0.881 | **0.941** | 0.942 |
| — per-bin AP (0.2→2.0 s) | — | 0.98 / 0.95 / 0.91 / 0.82 | 0.98 / 0.95 / 0.91 / 0.82 |
| Shift/hold AP | 0.683 † | **0.991** | 0.987 |
| Shift-prediction AP | 0.718 † | **0.909** | 0.912 |
| Overlap-resolution AP | n/a | **0.992** | 1.000 |
| Policy: gap-entry rate | ~18% ‡ | **40–46%** | — |
| Policy: entry latency | 0.37 s ‡ | **0.08–0.09 s** | — |
| Reactive 1.5 s baseline | 0 entries | 0 entries | — |

† v1 event APs were macro-averaged per session (older protocol) — indicative,
not strictly comparable; the balanced-accuracy gap (0.56 → 0.89 shift/hold) is
protocol-independent.
‡ On the one held-out session shared by both demo runs (session_000: 6/33 →
14/33 entries); v1's remaining demo sessions were training sessions.

**Demonstration agreement** (pseudo-demonstrator = slot P1, pooled over the 3
validation sessions): the demonstration-driven policy enters 15/15 gaps the
demonstrator took, with 0/57 false entries, leading the demonstrator's own
onset by ≈0.4 s; the floor-holder policy reaches 13/15 recall but 20/57 false
entries — it answers "is the floor free", not "is it my turn"; the 1.5 s
reactive baseline never enters. Pooled robot-entry AP = 1.00 (synthetic
ceiling). This is the offline form of the "correct non-response" argument:
floor availability alone over-generates entries; the demonstrated policy is
selective.

**Notes for the paper.** (i) Per-bin AP degrades gracefully with horizon
(0.98 → 0.82), as expected for longer-range prediction. (ii) The GRU variant
matches the transformer on synthetic data — the gain comes from the
**participant structure**, not the temporal mixer; this is a ready-made
ablation argument. (iii) Synthetic cues are capped by `cue_reliability = 0.7`;
these numbers verify software properties and pipeline wiring only; they do not
test or support H1/H2. Real recordings are the scientific test. (iv) All inputs
are robot-observable only; gold VAD never enters the model.

---

### Key references

- Ekstedt & Skantze 2022 — Voice Activity Projection (arXiv:2205.09812)
- Inoue et al. 2024 — Real-time continuous VAP (arXiv:2401.04868)
- Skantze & Irfan, HRI 2025 — VAP on Furhat, dual-threshold policy (arXiv:2501.08946)
- Elmers et al., Interspeech 2025 — Triadic audio-only VAP (arXiv:2507.07518)
- Onishi et al., IEICE 2025 — Multimodal dyadic VAP
- Press et al., ICLR 2022 — ALiBi attention bias (arXiv:2108.12409)
- Liao et al. — Light-ASD (CVPR 2023) / LR-ASD (IJCV 2025)
- Gurvich et al. 2023 — causal real-time ASD with microphone-array spatial querying
  (arXiv:2309.08295); relevant proof of feasibility, but not a drop-in source for this
  no-array corpus.

### ASD implementation audit (2026-08-15)

The official Light-ASD repository was bound at commit
[`ed38c232de5efe0261dbd68627c0ade7cdfe14eb`](https://github.com/Junhua-Liao/Light-ASD/commit/ed38c232de5efe0261dbd68627c0ade7cdfe14eb),
and the official LR-ASD repository at
[`1b6dcd2d8fc2895683de6508ec6294ec47d388ca`](https://github.com/Junhua-Liao/LR-ASD/commit/1b6dcd2d8fc2895683de6508ec6294ec47d388ca).
In both codebases the encoder applies symmetric temporal padding and the classifier runs a
backward/reversed-sequence GRU. Validation loads the complete candidate face track. A frame's
stock score therefore changes when later audio/video in that track changes; the future horizon
extends to the chosen track/chunk end. The words “lightweight” and “real-time” describe
throughput, not zero-lookahead causality. Stock weights may support an offline diagnostic or
annotation proposal, but cannot populate a formal causal `P{i}_asd_prob` column.

The read-only source/identity/time-axis preflight is implemented but formal acceptance remains
**BLOCKED at R3/3**. Fixture provenance can masquerade as formal provenance, and ancestors
above the configured authority root are not reparse-checked. The three-round repair limit is
exhausted pending owner review. No ASD model or weights were downloaded, no model was executed,
no probability or feature was generated, and insufficient Cam6 evidence for R fails closed.
Any later counted route still needs a genuinely causal pretrained detector or separately
approved causal-head training plus future-pixel, future-audio, prefix-extension and
chunk-boundary invariance tests.
