# Post-hoc 15-epoch training-budget sensitivity

## Status and claim boundary

This is a post-hoc, single-seed training-budget sensitivity analysis performed
after inspection of the frozen five-epoch deadline result. It is not a
replacement primary experiment and is not production-eligible or
repository-counted evidence.

- Corpus: exact eight-scene, four-group deadline view, 64,299 ticks.
- Conditions: A, H and A+H, paired over the same four outer groups.
- Seed: 13 only.
- Primary maximum budget: 5 epochs.
- Sensitivity maximum budget: 15 epochs.
- D3: delivered central offsets only; no D3 sensitivity run.
- Exclusions: `05_15__4` and `05_15__5` remain preserved for owner review.
- Policy: no calibrated real policy or reactive-baseline latency replay.

The split manifests are identical. Recursive comparison of the run manifests
finds exactly one changed field: `max_epochs`, from 5 to 15. The warm-up/cosine
scheduler derives its total number of steps from that field, so the sensitivity
changes the learning-rate trajectory from epoch 1 as well as the available
number of epochs. It must not be described as simply continuing the original
checkpoints for ten more epochs.

## Authority artifacts

- Five-epoch aggregate:
  `outputs/thesis_deadline_20260816/aggregate_seed13_v1.json`
  - SHA-256: `76373fd75f0685a324f05eb1adba2785d646d575c4d0b75dcde46ed6ef648995`
- Fifteen-epoch aggregate:
  `outputs/thesis_posthoc_15epoch_20260817/aggregate_seed13_v1.json`
  - SHA-256: `151a64ff23e6b82aa5750750de7b30b76c9885c9c7a774dd5cf1aa1a3e2f2ac4`
- Fifteen-epoch matrix index:
  `outputs/thesis_posthoc_15epoch_20260817/matrix_seed13_v1/matrix_index.json`
  - SHA-256: `781fab62795f414d4593712d89d694e36602e8dc42723b25610836e0b71f3d68`

Independent read-only replay accepted the full chain: 12/12 outer units,
120/120 bound artifacts, 36 inner checkpoints, 36 inner OOF predictions,
12 calibration artifacts, 12 final checkpoints, 12 outer predictions and
12 recomputed outer metric files. All 16 source CSV/provenance bindings still
match their frozen sizes and SHA-256 values. Re-aggregation from saved outer
probabilities is exactly equal to the published aggregate. No staging directory
remains.

## Identical evaluation support

The five- and fifteen-epoch runs have identical support and NA structure in all
three conditions:

- frame labels: 217,146 positive / 805,366 negative;
- shift/hold: 478 events (312 positive / 166 negative);
- shift prediction: 479 (312 / 167);
- overlap resolution: 69 (35 / 34);
- robot entry: 303 (111 / 192);
- next speaker: 479 events.

The only fold-level NA is overlap resolution for `outer_01`, which has three
positive and no negative overlap events. Pooled overlap remains evaluable.

## Results

Each cell reports five epochs to fifteen epochs.

| Metric | A | H | A+H |
|---|---:|---:|---:|
| Pooled frame mAP | .232468 → .231699 | .308666 → .399998 | .262867 → .264446 |
| Participant-macro frame mAP | .246991 → .241187 | .238788 → .290042 | .208313 → .212326 |
| Shift/hold AP | .582990 → .622963 | .664425 → .656493 | .690001 → .689712 |
| Shift-prediction AP | .604155 → .640331 | .661872 → .645186 | .693749 → .694882 |
| Overlap-resolution AP | .706247 → .691236 | .575906 → .568187 | .577822 → .524749 |
| Robot-entry AP | .291436 → .303124 | .359463 → .531790 | .305533 → .399298 |
| Next-speaker macro OVR AP | .264027 → .268163 | .271407 → .284251 | .274940 → .283041 |

Under fifteen epochs, A+H minus A is +.032746 pooled frame mAP, but
-.028861 participant-macro mAP and -.166486 overlap-resolution AP. A+H minus H
is -.135552 pooled frame mAP, -.077717 participant-macro mAP and -.043438
overlap AP.

The H gain is group-heterogeneous and is largest in the held-out `05_14` group.
A+H exceeds A on pooled frame mAP in only three of four groups and on
participant-macro mAP in only two of four; A+H remains below H on both frame
aggregations in all four groups.

## Training-budget evidence

At five epochs, 16/36 inner fits selected the ceiling. At fifteen epochs, 9/36
inner fits still select epoch 15: 5/12 for H and 4/12 for A+H. Three of twelve
outer refits use the 15-epoch ceiling. Therefore the longer run demonstrates
budget sensitivity, particularly for H, but does not establish complete
convergence.

## Scientific interpretation

- H1 remains **not supported under the limited head-pose operationalisation**.
  The longer budget improves H-only frame ranking, but fusion still does not
  outperform H and has substantially worse overlap AP than A.
- H2 remains **not established**. Shift-prediction and robot-entry AP are
  offline ranking metrics, not a calibrated predictive policy, and no reactive
  latency or missed-turn comparison was run.
- The result is descriptive: four independent groups, one seed, no confidence
  interval or significance claim, no three-participant evaluation, no D3
  robustness run, and no ASD/Gaze/Facial inference.

Safe dissertation wording:

> A post-hoc single-seed training-budget sensitivity increased the maximum
> budget from five to fifteen epochs while preserving the corpus, group-disjoint
> folds, features, labels and model configuration. Because the warm-up/cosine
> schedule depends on the maximum budget, the learning-rate trajectory also
> changed. Head-pose-only pooled frame mAP increased from 0.309 to 0.400,
> whereas audio-only and fused results remained approximately 0.232 and 0.264.
> Fusion still reduced participant-macro mAP and overlap-resolution AP relative
> to audio-only, so the H1/H2 conclusions did not change.
