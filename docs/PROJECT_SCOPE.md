# Current Project Scope and Methodology Handoff

> Canonical handoff for the supervisor-directed scope of the MSc final project.
> New sessions working on methodology, experiments, or Furhat integration must
> read this file before proposing changes. Cross-repository review decisions
> remain in `../../COORDINATION.md`.
>
> Last updated: 2026-08-03.

## 1. Latest supervisor direction (2026-07-14)

The supervisor asked the project owner to narrow the work because time is
limited:

- Treat extracted features such as head pose and gaze as model inputs.
- Treat Voice Activity Projection (VAP) as the model output/learning target.
- Specify the methodology as soon as possible.
- If time remains after the core model works, deploy the system on Furhat.

The immediate task is therefore not to invent a robot action classifier or a
joint addressee policy. It is to train and evaluate a causal model that projects
future participant voice activity from past multimodal features.

## 2. Frozen core formulation

### Input

At frame `t`, the model may use only robot-observable information available at
or before `t`:

- per participant: ASD probability, head pose, gaze/gaze target,
  MAR where available, and observation validity/quality (speaking-towards
  evidence was removed from the dissertation scope on 2026-08-03 and is not
  consumed anywhere);
- global: causal features from the single-channel combined audio
  (owner-frozen 2026-08-03: causal log-energy + F0 with a separate global
  `voiced` flag; CPC is deferred to a gated optional ablation);
- presence masks for three- versus four-person sessions.

Gold per-lavalier VAD is a target/evaluation source only. It must never be fed
to the model as an input.

### One primary output

The core model has one participant-wise VAP output tensor:

```text
P(participant i is voice-active in future bin k | all inputs up to t)
shape: time x participant x 4 future bins
```

The four bins cover `0-0.2`, `0.2-0.6`, `0.6-1.2`, and `1.2-2.0` seconds.
Targets are generated automatically from gold VAD; a bin is active when at
least half of its frames contain speech. A four-person frame therefore produces
16 probabilities in one output head, not two cooperating decision modules.

`p_now`, `p_future`, Shift/Hold, Next Speaker, overlap resolution, and robot
entry are deterministic readouts/evaluations derived from this same VAP tensor.
They are not additional learned outputs. The existing current-VAD head is an
optional auxiliary training loss, not a delivered model output; it is given
zero weight for the first formal run (owner decision 2026-08-03; the
0.0-versus-0.3 paired comparison remains an unapproved candidate).

### Core model and loss

The current `MultipartyVAP` is the starting point:

1. shared per-participant feature embedding;
2. causal temporal modelling (small Transformer as the main implementation,
   GRU as a low-cost baseline);
3. cross-participant attention with absent slots masked;
4. a shared four-bin participant-wise output head;
5. masked, class-weighted binary cross-entropy over valid participants and
   complete future horizons.

This should be described as a **lightweight factorised multimodal multi-party
VAP**. It is a practical multi-party approximation of the original joint VAP
codebook: retaining four bins for four people would otherwise create 65,536
joint classes.

## 3. Required experiments before optional extensions

The minimum dissertation result is an offline real-data comparison using
group-disjoint splits:

1. simple persistence/last-speaker and reactive-silence baselines;
2. audio-only VAP;
3. visual-only VAP;
4. audio + visual VAP;
5. targeted removal of head pose, gaze, and ASD where data completeness
   permits (the speaking-towards removal was retired with the 2026-08-03
   scope decision).

Report frame/bin mAP, macro or balanced F1, and event-level Shift/Hold,
Shift Prediction, Next Speaker, overlap, and robot-entry results. A well
controlled finding that a modality does not help is still a valid result.

Critical dependencies are synchronized real features and reliable
per-participant gold VAD. Do not let optional architecture work delay these.

## 4. Optional GVAP + participant-wise VAP extension

Recent MuVAP work separates:

- a role-relative GlobalVAP (GVAP) for the current-versus-next floor-holder
  pattern; and
- a participant-specific VAP for identifying the likely next speaker.

Adding a role-relative global head to the existing participant-wise VAP is
normally simpler than a complete live Furhat deployment: it is mainly new label
generation, a small prediction head, losses, tests, and offline experiments. It
does not require robot access, live identity tracking, camera/microphone
streaming, or latency engineering.

However, **GVAP + participant-wise VAP is not part of the first supervisor-
mandated run**. The base participant-wise VAP must work on real data first. If
attempted later, it must be an ablation against the unchanged single-output
baseline, not a silent replacement. Possible additions are:

- a 136-state Role-Relative Projection auxiliary head;
- hierarchical decoding: global Hold/Shift first, participant identity second;
- a light consistency loss between global and participant-wise predictions.

Do not copy the large MuVAP or MM-VAP pipelines wholesale. MuVAP was trained
with substantially more data and compute, and its official repository currently
says that code is forthcoming. MM-VAP's TalkNet/WhisperFlamingo plus LoRA
pipeline changes the project from extracted-feature modelling to end-to-end
audio-visual adaptation.

## 5. Optional Furhat deployment

Furhat is a supervisor-endorsed stretch goal after the offline model and core
ablations. A deployed system does not need to install all PyTorch perception
models inside the robot. The practical architecture is:

```text
live camera + mono microphone
    -> external Python feature extraction and identity tracking
    -> 25 fps causal feature buffer
    -> participant-wise VAP inference
    -> p_now/p_future policy and target selection
    -> Furhat Realtime API (attention/speech actions)
```

Use the current WebSocket Realtime API, not the deprecated Remote API. Separate
two deployment claims:

1. **Replay/control integration:** replay recorded causal features in real time
   and let the model control a physical/virtual Furhat. This validates the
   model-policy-robot control path but not live perception.
2. **Full live loop:** capture live audio/video, extract all required features,
   track participant identities, infer online, and actuate Furhat. This is the
   stronger but riskier stretch goal.

Full live deployment is usually harder than the GVAP extension because it adds
hardware availability, sensor/API behaviour, train-deploy domain shifts,
real-time feature latency, failure handling, and dialogue integration. Coding
support cannot remove those external dependencies.

### Decision gate after the core result

- If physical Furhat access and a causal live feature pipeline are ready,
  prioritize a Furhat demo because the supervisor explicitly suggested it.
- If hardware or live perception is not ready, implement the GVAP extension as
  the lower-risk offline enhancement and, if useful, still complete the replay
  Furhat integration.
- If both are ready and time genuinely remains, do both in that order without
  weakening the core ablations.

## 6. Scope boundaries

- VAP answers future voice activity, timing, and likely next-speaker questions.
- The person the robot should address is not identical to the next speaker.
  Addressee selection stays outside the core; since the 2026-08-03 owner
  decision the dissertation does not consume speaking_towards data at all,
  so the addressee-accuracy evaluation is deferred out of scope rather than
  kept as a downstream module.
- Semantic transcript context, large end-to-end encoders, and a formal user
  study remain optional after the required offline result.
- Do not claim the first multimodal multi-party robotic VAP. Recent MuVAP and
  MM-VAP work makes that claim unsafe.

The defensible contribution is adaptation and evaluation on the project's own
real three-/four-person receptionist-style HRI corpus: explicit social cues,
small-data participant-structured modelling, strict modality ablations,
human-demonstrator robot-slot supervision, and deployment feasibility.

## 7. Questions the methodology session must resolve

1. ~~Freeze the exact causal acoustic input representation available from the
   recorded combined mix and at Furhat runtime.~~ Resolved 2026-08-03 (see
   §8): energy + F0 + `voiced`; the extractor parameters must still be
   versioned in the first counted-run manifest.
2. Audit the availability and quality of gold VAD and every proposed input
   feature before finalising the ablation matrix.
3. Define model-selection and group-disjoint evaluation protocol based on the
   actual number of recording groups.
4. After the first real-data VAP run, apply the decision gate in section 5 for
   GVAP versus Furhat work.
5. Keep this file updated with dated owner/supervisor decisions so future
   sessions do not reconstruct scope from chat history.

## 8. Decision log

- **2026-07-14, supervisor:** extracted multimodal features in; VAP as output;
  define methodology quickly; Furhat deployment if time remains.
- **2026-07-14, project owner/Codex clarification:** one participant-wise VAP
  tensor is the mandatory first output. GVAP is an optional later ablation, not
  a second mandatory core module. Full live Furhat is harder but has explicit
  supervisor value, so choose between them using the readiness gate above.
- **2026-07-14, local readiness audit:** the only processed real table
  (`05_15__2`) remains `trainable: false`, has no gold VAD or real acoustic
  inputs, and supplies only one recording group. Its provenance records a
  `6cam` offset that the current delivery no longer publishes; the current
  bridge diagnostic is suppressed because no finite error bound is available.
  The table is smoke-only and must be regenerated after synchronization is
  republished. The GVAP/Furhat gate therefore remains **closed**.
- **2026-07-27, HPE final handoff:** head-pose D1 and participant-mapping D2
  are complete and consumable for 21 indexed scenes across five recording
  groups; all mappings are human-confirmed. The upstream
  `outputs/delivery/README.md` is the formal contract, mirrored locally by
  `configs/hpe_delivery_inventory.json`. Only `03_20__5` publishes a usable
  `6cam` D3 offset. The other 20 scenes remain on local six-camera time and are
  explicitly ineligible for shared-timeline assembly until their own offsets
  are published (14 are `absent`, 6 are `suppressed`). This completes the HPE
  D1/D2 dependency but does not open the GVAP/Furhat gate because synchronized
  multi-group inputs, gold VAD, and acoustic features are still missing.
- **2026-08-03, project owner:** three decisions, consolidated in
  `docs/final_model_plan.md`. (i) **speaking_towards is out of the
  dissertation scope** — never a model input, the `-speaking-towards`
  ablation is retired, and the addressee-accuracy evaluation is deferred out
  of scope; `gazed_at` (MCGaze gaze vectors intersected with 3D head
  positions) remains the sole gaze-target evidence. The team-level
  speaking-towards deliverable is unaffected; only its consumption here is
  removed. (ii) **Core acoustic representation frozen: causal log-energy +
  F0 from the combined mix plus a separate global `voiced` 0/1 column**
  (resolves the voiced-100 Hz encoding collision; schema v2,
  32 raw columns → 33 model dims, implemented 2026-08-15). CPC is deferred
  to a gated optional ablation after the core result. (iii)
  **`aux_vad_weight = 0.0` for the first formal run**; the 0.0-versus-0.3
  paired comparison remains a candidate requiring separate approval.
- **2026-08-03, upstream status update (HPE 2026-07-28 snapshot received):**
  D3 close-out — **14 of the 21 head-pose scenes now publish a usable,
  self-checked `6cam` offset, covering all five recording groups**
  (`03_20__{4,5}`, `03_24_1__{2,3}`, `03_26__3`, `05_14__{2..5}` (mic_A+6cam
  only), `05_15__{2..6}`). 9 are `degraded` and 8 exceed the ±40 ms target
  (max ±104.4 ms), so the synchronization exclusion/sensitivity decision is
  now live. 7 scenes have no physical bridge (`03_20__{2,3,6}`,
  `03_26__{4,6,7}`, `05_14__6`). `05_15__2`'s republished offset
  (`-3.7621 s`) supersedes the stale table's recorded `-6.3225 s` — that
  table must be regenerated. `configs/hpe_delivery_inventory.json` was
  regenerated from the actual delivery files. The GVAP/Furhat gate stays
  closed (gold VAD, acoustics, and visual inputs still missing), but the
  synchronization side of the group-count blocker is resolved.
- **2026-08-18, post-hoc evidence amendment (does not rewrite the earlier
  owner decisions):** the isolated schema-v3 Gaze experiment completed
  C0/CM/CG × four outer folds on the frozen exact-eight view (64,299 rows,
  four groups), seed 13 and `max_epochs=15`. The independently audited
  saved-probability aggregate has SHA-256
  `0b4f2c5e3deafafa0d23cdb4822f328d95af5c0788ba662cd9c04d7414955cea`.
  Pooled frame mAP was 0.2717/0.2461/0.2843 for C0/CM/CG. The primary
  content contrast `CG-CM=+0.0381` was positive in every outer fold, while
  `CM-C0=-0.0256` records a material observability/missingness effect.
  P1/R has no direct Gaze observation; the actual upstream extractor
  variant/weights/configuration are not fully bound, exact upstream
  regeneration is unavailable, and implementation counterfactual causality
  is unverified. This single-seed post-hoc result is non-production,
  non-counted and does not reclassify H1 as supported. Separately, the
  exact-eight matched predictive-versus-reactive policy replay is complete
  and independently audited but does not support H2; it contains no
  physical-robot, TTS or live-latency evidence. Facial remains dropped from
  dissertation modelling and ASD remains deferred at its R3/3 gate.

## 9. Primary references for the next session

- Ekstedt & Skantze (2022), original VAP:
  <https://arxiv.org/abs/2205.09812>
- Elmers et al. (2025), triadic VAP:
  <https://arxiv.org/abs/2507.07518>
- Qi & Skantze (2026), MuVAP and Role-Relative Projection:
  <https://arxiv.org/abs/2606.16731>
- Cano et al. (2026), multimodal VAP for social robots:
  <https://arxiv.org/abs/2607.07294>
- Furhat Realtime API:
  <https://docs.furhat.io/realtime-api/intro>
