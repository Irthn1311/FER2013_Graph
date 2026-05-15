# D12A Rare-Class Rescue 6-Run Analysis

Date: 2026-05-15

Scope: read-only analysis of six completed D12A Rare-Class Rescue runs. No training, no model rerun, no code changes.

Analyzed runs:

1. `outputs/d12_experiments/d12a_rare_hardneg_margin`
2. `outputs/d12_experiments/d12a_speed_control_ce_first`
3. `outputs/d12_experiments/d12a_rare_aux_logit_tau05`
4. `outputs/d12_experiments/d12a_repeat_disgust_rare_aux`
5. `outputs/d12_experiments/d12a_target_repeat_disgust`
6. `outputs/d12_experiments/d12a_rare_aux_bce`

Reference baseline from previous stable CE-first artifact:

| Baseline | Eval Acc | Eval Macro F1 | Eval Weighted F1 | Eval Pred Count | Angry F1 | Disgust F1 |
|---|---:|---:|---:|---|---:|---:|
| `d12_stable_ce_first_outputs` | 0.3876 | 0.2715 | 0.3350 | `[0, 0, 355, 1369, 525, 623, 717]` | 0.0000 | 0.0000 |

## 1. Executive Summary

The Rare-Class Rescue round did not rescue Disgust. All six runs still have Disgust prediction count = 0 on evaluation/test, Disgust recall = 0, and Disgust F1 = 0.

Only one intervention produced a meaningful Angry signal: `d12a_rare_aux_logit_tau05`.

- It predicts Angry 403 times on eval/test.
- Angry F1 rises to 0.1477, clearly above the previous logit-only run from the earlier round.
- However, Disgust remains 0, and Neutral collapses to 0 predictions.

The strongest result in this round by eval macro F1 is still weak:

| Best in this round | Eval Macro F1 | Baseline Stable Macro F1 | Gap |
|---|---:|---:|---:|
| `d12a_rare_aux_logit_tau05` | 0.1677 | 0.2715 | -0.1038 |

None of the six runs beats or matches the stable CE-first baseline. Three runs collapse to all-Happy on eval/test:

- `d12a_speed_control_ce_first`
- `d12a_target_repeat_disgust`
- `d12a_rare_hardneg_margin`

This is important because `d12a_speed_control_ce_first` was supposed to be the control under the speed runtime. It does not reproduce the old stable baseline. That makes the speed runtime / DDP / fixed-shape sampler path a serious quality suspect, although the artifacts here cannot isolate which runtime component caused the drop.

Root-cause verdict from these artifacts:

1. Disgust is not primarily solved by exposure: target-repeat increased epoch length as expected, but Disgust remained zero.
2. Disgust is not solved by rare auxiliary BCE: rare aux loss exists but barely improves, and no Disgust prediction appears.
3. Angry is partly a classifier-bias / logit-prior problem: rare aux + logit adjustment produces clear Angry predictions.
4. Disgust is more likely a representation / micro-expression problem, possibly worsened by runtime sampling and CE dynamics.
5. Current artifacts are not enough to prove scale2 erases micro-detail, but motif/attention diagnostics are consistent with weak or diffuse rare-class representation.

## 2. Artifact Inventory

Files found consistently:

- `training_history.json`
- `resolved_config.yaml`
- `evaluation/metrics.json`
- `evaluation/classification_report.json`
- `evaluation/classification_report.txt`
- `evaluation/predictions.csv`
- `evaluation/d6b_diagnostics.json`
- `evaluation/confusion_matrix.png`
- `evaluation/correct_examples.png`
- `evaluation/wrong_examples.png`
- `checkpoints/best.pth`
- `checkpoints/last.pth`
- W&B local logs and summaries

Files not found as separate structured artifacts:

- `history.json`
- `predictions.json`
- `confusion_matrix.npy`
- `confusion_matrix.json`
- `confusion_matrix.csv`
- `diagnostics.json`
- `diagnostics_history.json`
- root-level `config.yaml`
- `test_metrics.json`
- `val_metrics.json`

Confusion rows in this report are reconstructed from `evaluation/predictions.csv`, not from the PNG image.

Checkpoint metadata was inspected from `best.pth`. Each best checkpoint contains `epoch`, `best_epoch`, `best_metric`, `best_metric_name`, optimizer/scheduler/scaler states, config, and model state.

## 3. Main Summary Table

| Run | Intervention | Best Ep | Best Val Macro | Eval Acc | Eval Macro | Eval Weighted | Local Best Macro | Eval Pred Count | Zero-Predict Classes | Angry F1 | Disgust F1 | Fear F1 | Happy F1 | Sad F1 | Surprise F1 | Neutral F1 | Mean Train sec/batch |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `d12a_rare_hardneg_margin` | hard-negative margin | 2 | 0.0749 | 0.2449 | 0.0562 | 0.0964 | 0.0746 | `[0, 0, 0, 3589, 0, 0, 0]` | Angry, Disgust, Fear, Sad, Surprise, Neutral | 0.0000 | 0.0000 | 0.0000 | 0.3935 | 0.0000 | 0.0000 | 0.0000 | 0.3746 |
| `d12a_speed_control_ce_first` | speed control | 1 | 0.1300 | 0.2449 | 0.0562 | 0.0964 | 0.0779 | `[0, 0, 0, 3589, 0, 0, 0]` | Angry, Disgust, Fear, Sad, Surprise, Neutral | 0.0000 | 0.0000 | 0.0000 | 0.3935 | 0.0000 | 0.0000 | 0.0000 | 0.3839 |
| `d12a_rare_aux_logit_tau05` | rare aux + logit | 29 | 0.1795 | 0.2809 | 0.1677 | 0.2090 | 0.1374 | `[403, 0, 73, 2195, 572, 346, 0]` | Disgust, Neutral | 0.1477 | 0.0000 | 0.0566 | 0.4411 | 0.2264 | 0.3018 | 0.0000 | 0.3494 |
| `d12a_repeat_disgust_rare_aux` | repeat + rare aux | 23 | 0.1363 | 0.2639 | 0.1332 | 0.1739 | 0.0951 | `[11, 0, 7, 2362, 694, 515, 0]` | Disgust, Neutral | 0.0239 | 0.0000 | 0.0037 | 0.4116 | 0.2438 | 0.2492 | 0.0000 | 0.3807 |
| `d12a_target_repeat_disgust` | target repeat | 1 | 0.1035 | 0.2449 | 0.0562 | 0.0964 | 0.0357 | `[0, 0, 0, 3589, 0, 0, 0]` | Angry, Disgust, Fear, Sad, Surprise, Neutral | 0.0000 | 0.0000 | 0.0000 | 0.3935 | 0.0000 | 0.0000 | 0.0000 | 0.3438 |
| `d12a_rare_aux_bce` | rare aux | 30 | 0.1619 | 0.2923 | 0.1581 | 0.1983 | 0.0920 | `[5, 0, 50, 2245, 623, 666, 0]` | Disgust, Neutral | 0.0040 | 0.0000 | 0.0346 | 0.4334 | 0.2597 | 0.3752 | 0.0000 | 0.3439 |

Comparison against stable CE-first baseline:

- No run exceeds baseline macro F1.
- No run is near baseline macro F1.
- No run predicts Disgust.
- All six runs kill Neutral on eval/test.
- The only clear Angry rescue is `d12a_rare_aux_logit_tau05`.

## 4. Stable Baseline Comparison

The old stable CE-first baseline was still bad for Angry/Disgust, but it learned a broader class distribution:

| Run | Eval Macro | Eval Acc | Eval Pred Count | Active Classes |
|---|---:|---:|---|---:|
| Stable CE-first old | 0.2715 | 0.3876 | `[0, 0, 355, 1369, 525, 623, 717]` | 5 |
| Speed control new | 0.0562 | 0.2449 | `[0, 0, 0, 3589, 0, 0, 0]` | 1 |
| Best rare-rescue new | 0.1677 | 0.2809 | `[403, 0, 73, 2195, 572, 346, 0]` | 5 |

This means the new speed-control run did not reproduce the old stable quality. The gap is too large to treat as noise.

Important caveat: the old stable run and this new round are compared from artifacts, not from a newly rerun controlled experiment. The evidence says the current speed-runtime control is much worse; it does not by itself prove whether the cause is DDP, fixed-shape sampler, compile, AMP, config resolution, checkpoint selection, or their interaction.

## 5. Epoch-Level Rare-Class Collapse

| Run | Val Pred 0/1 Epoch 1 | First Epoch Val 0/1 Become 0 | Best Epoch | Best Val Macro | Best Pred 0/1 | Final Pred 0/1 | Eval/Test Pred 0/1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d12a_rare_hardneg_margin` | 1909 / 783 | 3 / 2 | 2 | 0.0749 | 2024 / 0 | 0 / 0 | 0 / 0 |
| `d12a_speed_control_ce_first` | 976 / 249 | 2 / 2 | 1 | 0.1300 | 976 / 249 | 2 / 0 | 0 / 0 |
| `d12a_rare_aux_logit_tau05` | 1329 / 406 | 2 / 2 | 29 | 0.1795 | 439 / 0 | 443 / 0 | 403 / 0 |
| `d12a_repeat_disgust_rare_aux` | 2368 / 528 | 3 / 3 | 23 | 0.1363 | 12 / 0 | 12 / 0 | 11 / 0 |
| `d12a_target_repeat_disgust` | 1104 / 972 | 3 / 3 | 1 | 0.1035 | 1104 / 972 | 0 / 0 | 0 / 0 |
| `d12a_rare_aux_bce` | 722 / 398 | 2 / 2 | 30 | 0.1619 | 1 / 0 | 1 / 0 | 5 / 0 |

### Epoch 1

All six runs predict class 0/1 heavily in epoch 1. This is not reliable rare-class learning.

Epoch-1 diagnostics are almost identical across runs:

| Run | Epoch-1 Logits Std | Epoch-1 Slot Entropy | Epoch-1 Class Top1 | Epoch-1 Disgust-Angry Similarity |
|---|---:|---:|---:|---:|
| `rare_hardneg_margin` | 0.0102 | 2.0794 | 0.1250 | 1.0000 |
| `speed_control_ce_first` | 0.0143 | 2.0794 | 0.1251 | 1.0000 |
| `rare_aux_logit_tau05` | 0.0035 | 2.0794 | 0.1251 | 1.0000 |
| `repeat_disgust_rare_aux` | 0.0097 | 2.0794 | 0.1250 | 1.0000 |
| `target_repeat_disgust` | 0.0165 | 2.0794 | 0.1250 | 1.0000 |
| `rare_aux_bce` | 0.0033 | 2.0794 | 0.1251 | 1.0000 |

`log(8) = 2.0794`, so slot entropy is effectively uniform over 8 slots. `class_top1` near 0.125 also means class-part attention is uniform. Therefore epoch-1 rare predictions are mostly unlearned low-logit argmax behavior, not stable rare-class recognition.

### Epoch 2-5

The collapse is early, not late overfitting:

- `speed_control_ce_first`: validation 0/1 become 0 at epoch 2.
- `rare_aux_bce`: validation 0/1 become 0 at epoch 2.
- `rare_aux_logit_tau05`: validation 0/1 become 0 at epoch 2, but class 0 later recovers.
- `target_repeat_disgust`: validation 0/1 survive into epoch 2 but both die by epoch 3.
- `repeat_disgust_rare_aux`: validation 0/1 survive into epoch 2 but Disgust dies by epoch 3.
- `hardneg_margin`: Disgust dies at epoch 2; Angry dies by epoch 3, except the best checkpoint still has many low-confidence Angry predictions.

This is the same pattern as the previous six-run analysis: rare classes appear during near-random early logits and disappear once the model begins learning a real decision surface.

### Best Epoch

Three best checkpoints are essentially early/unlearned:

| Run | Best Epoch | Best Diagnostics |
|---|---:|---|
| `speed_control_ce_first` | 1 | uniform slots, uniform class attention, tiny logits |
| `target_repeat_disgust` | 1 | uniform slots, uniform class attention, tiny logits |
| `rare_hardneg_margin` | 2 | almost uniform slots, tiny logits |

These best checkpoints do not represent useful trained rare-class models.

The only runs with learned best epochs are:

- `rare_aux_logit_tau05`
- `rare_aux_bce`
- `repeat_disgust_rare_aux`

Among those, only `rare_aux_logit_tau05` has meaningful Angry prediction count.

### Final Epoch

Final-epoch behavior:

- `rare_aux_logit_tau05`: keeps Angry, never gets Disgust.
- `rare_aux_bce`: almost no Angry, no Disgust.
- `repeat_disgust_rare_aux`: tiny Angry, no Disgust.
- `speed_control_ce_first`: mostly Happy/Surprise, no Disgust.
- `target_repeat_disgust`: Happy/Surprise only, no rare.
- `hardneg_margin`: all Happy.

No run shows late recovery of Disgust.

## 6. Class 0 Angry Analysis

### Eval/Test Angry Results

| Run | Eval Angry Pred Count | Angry Precision | Angry Recall | Angry F1 |
|---|---:|---:|---:|---:|
| `rare_aux_logit_tau05` | 403 | 0.1638 | 0.1344 | 0.1477 |
| `repeat_disgust_rare_aux` | 11 | 0.5455 | 0.0122 | 0.0239 |
| `rare_aux_bce` | 5 | 0.2000 | 0.0020 | 0.0040 |
| `speed_control_ce_first` | 0 | 0.0000 | 0.0000 | 0.0000 |
| `target_repeat_disgust` | 0 | 0.0000 | 0.0000 | 0.0000 |
| `rare_hardneg_margin` | 0 | 0.0000 | 0.0000 | 0.0000 |

`rare_aux_logit_tau05` is the only useful Angry rescue in this round. It improves over the previous logit-only run, whose Angry F1 was about 0.0709. Here Angry F1 is 0.1477.

### Angry Confusion Rows

Rows reconstructed from `predictions.csv`:

| Run | True Angry Predictions |
|---|---|
| `rare_hardneg_margin` | Happy: 491 |
| `speed_control_ce_first` | Happy: 491 |
| `rare_aux_logit_tau05` | Angry: 66, Fear: 12, Happy: 265, Sad: 92, Surprise: 56 |
| `repeat_disgust_rare_aux` | Angry: 6, Fear: 1, Happy: 287, Sad: 119, Surprise: 78 |
| `target_repeat_disgust` | Happy: 491 |
| `rare_aux_bce` | Angry: 1, Fear: 11, Happy: 277, Sad: 108, Surprise: 94 |

Angry is mostly swallowed by Happy, then Sad and Surprise in the learned rare-aux runs. The all-Happy runs provide no meaningful boundary.

Mean Angry true-vs-best-other logit margin:

| Run | Angry Margin Mean |
|---|---:|
| `rare_aux_logit_tau05` | -0.3245 |
| `repeat_disgust_rare_aux` | -1.0472 |
| `rare_aux_bce` | -1.1440 |
| all-Happy runs | near 0 due near-tied logits |

The least negative meaningful margin is `rare_aux_logit_tau05`, matching its best Angry F1.

### Angry Interpretation

Angry is not completely unrecoverable. The fact that logit adjustment plus rare aux creates 403 Angry predictions and 66 correct Angry predictions means Angry has at least a weak representation. The problem is partly classifier bias / class-prior bias, but not only that: recall is still only 0.1344 and most true Angry is still predicted Happy/Sad/Surprise.

Rare aux alone is insufficient. Hard-negative margin does not help Angry on eval/test; it instead collapses to all-Happy.

## 7. Class 1 Disgust Analysis

### Eval/Test Disgust Results

| Run | Eval Disgust Pred Count | Disgust Precision | Disgust Recall | Disgust F1 |
|---|---:|---:|---:|---:|
| all six runs | 0 | 0.0000 | 0.0000 | 0.0000 |

No intervention produced any Disgust prediction on evaluation/test.

### Disgust Confusion Rows

Rows reconstructed from `predictions.csv`:

| Run | True Disgust Predictions |
|---|---|
| `rare_hardneg_margin` | Happy: 55 |
| `speed_control_ce_first` | Happy: 55 |
| `rare_aux_logit_tau05` | Angry: 10, Fear: 1, Happy: 36, Sad: 6, Surprise: 2 |
| `repeat_disgust_rare_aux` | Fear: 1, Happy: 34, Sad: 12, Surprise: 8 |
| `target_repeat_disgust` | Happy: 55 |
| `rare_aux_bce` | Fear: 1, Happy: 40, Sad: 9, Surprise: 5 |

Disgust is mostly swallowed by Happy. In the more learned rare-aux runs, some Disgust samples go to Sad and Surprise, and in `rare_aux_logit_tau05` some go to Angry. None go to Disgust.

Mean Disgust true-vs-best-other logit margin:

| Run | Disgust Margin Mean |
|---|---:|
| `repeat_disgust_rare_aux` | -0.6155 |
| `rare_aux_bce` | -2.6500 |
| `rare_aux_logit_tau05` | -2.8822 |
| all-Happy runs | near 0 due near-tied logits |

`repeat_disgust_rare_aux` makes the Disgust margin much less negative than rare-aux alone or rare-aux+logit, but still not enough to create a single Disgust prediction.

### Target Repeat Exposure

Resolved config confirms:

```yaml
data:
  target_class_repeat_factors:
    1: 8.0
```

The sampler did not persist `repeated_num_indices_total` or per-rank label histogram in the final artifacts, so exact class exposure by rank/chunk cannot be verified from logs.

However, train batch counts strongly indicate target repeat was applied:

| Run | Final Train Batches | Change vs 888-Batch Control |
|---|---:|---:|
| control / no repeat | 888 | baseline |
| `target_repeat_disgust` | 988 | +100 batches |
| `repeat_disgust_rare_aux` | 982 | +94 batches |

With per-rank batch size 32, this is about +3008 to +3200 extra per-rank samples, close to the expected duplicate volume for class 1 repeat factor 8.0. So exposure likely increased, but it did not create Disgust predictions.

### Rare Aux BCE Signal

Rare aux loss exists in the three rare-aux runs:

| Run | Val Rare Aux Loss Epoch 1 | Best Val Rare Aux Loss | Final Val Rare Aux Loss |
|---|---:|---:|---:|
| `rare_aux_bce` | 0.5006 | 0.4966 | 0.4966 |
| `rare_aux_logit_tau05` | 0.5007 | 0.4997 | 0.4997 |
| `repeat_disgust_rare_aux` | 0.7790 | 0.7693 | 0.7584 |

The loss decreases only slightly. There are no persisted binary metrics for rare aux heads, so we cannot say whether the auxiliary head specifically learned Disgust. From main-branch predictions, it did not transfer into Disgust classification.

### Disgust Interpretation

Disgust is not solved by:

- target repeat alone,
- rare auxiliary BCE alone,
- rare auxiliary + logit adjustment,
- repeat + rare auxiliary,
- hard-negative margin.

This pushes the diagnosis away from pure exposure and pure objective weighting. The strongest current interpretation is representation/micro-expression failure: D12A is not forming a usable Disgust-specific feature region.

## 8. Local vs Full Branch

Best-epoch local macro F1:

| Run | Main Best Val Macro | Local Best Val Macro | Best Main Pred 0/1 | Best Local Pred 0/1 |
|---|---:|---:|---:|---:|
| `rare_hardneg_margin` | 0.0749 | 0.0746 | 2024 / 0 | 2555 / 290 |
| `speed_control_ce_first` | 0.1300 | 0.0779 | 976 / 249 | 1557 / 1271 |
| `rare_aux_logit_tau05` | 0.1795 | 0.1374 | 439 / 0 | 711 / 0 |
| `repeat_disgust_rare_aux` | 0.1363 | 0.0951 | 12 / 0 | 0 / 0 |
| `target_repeat_disgust` | 0.1035 | 0.0357 | 1104 / 972 | 3017 / 565 |
| `rare_aux_bce` | 0.1619 | 0.0920 | 1 / 0 | 0 / 0 |

The local branch does not provide stable Disgust signal. In early best checkpoints for speed/target/hardneg, local rare predictions are also from uniform/unlearned attention, not reliable learning.

For `rare_aux_logit_tau05`, local predicts Angry but no Disgust. Full/main is better than local overall and preserves Angry better, but neither branch learns Disgust.

Conclusion:

- If local had strong Disgust but full lost it, the main suspect would be global fusion/classifier bias.
- Instead, local also lacks Disgust.
- The problem is likely before or inside local motif representation, not only global fusion.

## 9. Motif / Attention / Micro-Expression Diagnostics

Best-epoch diagnostics:

| Run | Best Logits Std | Best Slot Entropy | Best Effective Slots | Best Class Top1 | Best Class Similarity Mean | Best Disgust-Angry Similarity |
|---|---:|---:|---:|---:|---:|---:|
| `rare_hardneg_margin` | 0.0055 | 2.0794 | 7.9999 | 0.1254 | 1.0000 | 1.0000 |
| `speed_control_ce_first` | 0.0143 | 2.0794 | 7.9999 | 0.1251 | 1.0000 | 1.0000 |
| `rare_aux_logit_tau05` | 1.0057 | 1.9407 | 6.9891 | 0.2929 | 0.6923 | 0.1129 |
| `repeat_disgust_rare_aux` | 0.4940 | 1.8575 | 6.4869 | 0.2006 | 0.8791 | 0.9424 |
| `target_repeat_disgust` | 0.0165 | 2.0794 | 7.9999 | 0.1250 | 1.0000 | 1.0000 |
| `rare_aux_bce` | 0.8889 | 1.9107 | 6.7860 | 0.2528 | 0.8227 | 0.7958 |

Interpretation:

- Runs whose best checkpoint is epoch 1/2 are basically uniform in slot and class attention.
- Learned rare-aux runs still have high effective slot count, around 6.5 to 7.0.
- The old stable baseline had effective slots around 3.44 at best epoch and much better macro F1.

High effective slot count here does not mean better motif coverage. In these artifacts it is more consistent with diffuse slot use.

Disgust-Angry attention similarity:

- `rare_aux_logit_tau05`: 0.1129 at best, very separated.
- `rare_aux_bce`: 0.7958, still fairly similar.
- `repeat_disgust_rare_aux`: 0.9424, almost the same.

This repeats the earlier pattern: logit adjustment is the only mechanism that strongly separates Angry/Disgust attention and is also the only mechanism that gives strong Angry prediction. But even when attention separates, Disgust does not become a class decision.

Evidence for micro-expression dilution:

- Disgust has no stable local or full signal.
- Repeat exposure does not create Disgust predictions.
- Rare aux does not create Disgust predictions.
- Slots remain diffuse relative to the older stable baseline.
- Disgust is mostly swallowed by Happy/Sad/Surprise.

What is not proven:

- These artifacts do not directly show scale1 vs scale2 feature degradation.
- There is no per-layer `h_pixel_std`, no after-scale1/after-scale2 cosine similarity, no slot visualization by class.
- Therefore, "scale2 erases micro-expression" remains a plausible hypothesis, not a proven fact.

## 10. Encoder / Scale2 / Global / DDP Runtime

All six configs use:

- global branch on,
- scale2 on,
- DDP enabled,
- AMP enabled,
- compile before DDP,
- fixed batch shape,
- global batch 64 through `training.global_batch_size=64`,
- per-rank batch size 32 with world size 2.

Resolved config uses `data.batch_size=32` because it is per rank; `training.global_batch_size=64` and `training.per_rank_batch_size=32` confirm the intended global batch.

### Speed Control Result

`d12a_speed_control_ce_first` fails to reproduce the stable CE-first baseline.

| Run | Eval Macro | Eval Acc | Pred Count |
|---|---:|---:|---|
| old stable CE-first | 0.2715 | 0.3876 | `[0, 0, 355, 1369, 525, 623, 717]` |
| new speed control | 0.0562 | 0.2449 | `[0, 0, 0, 3589, 0, 0, 0]` |

This is a major warning. The quality drop is not specific to rare aux or repeat; it appears already in the control.

Possible explanations not separable from current artifacts:

- DDP fixed-shape chunk-aware sampler changed training distribution.
- Compile/AMP/DDP interaction changed optimization.
- Checkpoint selection selected an epoch-1 low-confidence model because later validation never recovered enough.
- Config resolution differs from the earlier stable artifact.

### Sampler Repeat

Target repeat appears to increase epoch length:

- no-repeat control: 888 train batches,
- target repeat: 988 train batches,
- repeat + rare aux: 982 train batches.

This supports that repeat was applied. But no persisted per-rank label histogram is available, so we cannot prove rank/chunk balance from the saved artifacts.

### Encoder/Gate Diagnostics

Learned runs show gate specialization:

- gate mean drops from about 0.50 to about 0.40-0.44,
- gate std rises to about 0.23-0.27.

This means the gate is not frozen. But without per-layer or per-scale diagnostics, this does not prove whether it preserves or smooths micro-detail.

## 11. Intervention-by-Intervention Evaluation

### 11.1 `d12a_speed_control_ce_first`

Verdict: failed control.

- Best epoch is 1.
- Eval/test collapses to all Happy.
- Eval macro F1 is 0.0562, far below stable baseline 0.2715.
- Epoch-1 rare predictions are uniform/unlearned.

This run cannot be used as a healthy baseline for rare-class conclusions. It instead raises a runtime/config parity alarm.

### 11.2 `d12a_target_repeat_disgust`

Verdict: exposure increased, Disgust not rescued.

- Train batches increase from 888 to about 988.
- Epoch 1 predicts Disgust heavily, but with uniform attention/logits.
- By epoch 3, validation Disgust is 0.
- Eval/test collapses to all Happy.

Conclusion: exposure alone does not solve Disgust, and under this runtime it may destabilize the already weak control.

### 11.3 `d12a_rare_aux_bce`

Verdict: modest macro recovery, no Disgust rescue, almost no Angry.

- Eval macro F1 = 0.1581.
- Angry F1 = 0.0040.
- Disgust F1 = 0.
- Neutral F1 = 0.
- Rare aux loss barely changes: val 0.5006 -> 0.4966.

Conclusion: rare aux BCE as implemented does not create useful rare-class decision boundaries.

### 11.4 `d12a_rare_aux_logit_tau05`

Verdict: best run in this round; rescues Angry, still fails Disgust.

- Eval macro F1 = 0.1677.
- Angry F1 = 0.1477.
- Disgust F1 = 0.
- Neutral F1 = 0.
- Disgust-Angry attention similarity drops to 0.1129 at best epoch.

This is the only intervention that clearly improves Angry. It suggests that Angry is partly classifier-bias/prior-bias constrained. But Disgust remains deeply unsolved.

### 11.5 `d12a_repeat_disgust_rare_aux`

Verdict: strongest Disgust-oriented intervention, still no Disgust prediction.

- Repeat appears active from train batch count.
- Rare aux is active.
- Eval Disgust pred count = 0.
- Disgust margin improves compared with rare aux alone, but remains negative.
- Eval macro F1 = 0.1332, below rare aux alone and far below baseline.

This is the most important negative result: exposure + binary rare aux is still not enough. That strongly weakens the "Disgust only lacks exposure" hypothesis.

### 11.6 `d12a_rare_hardneg_margin`

Verdict: failed; collapses to all Happy.

- Best epoch is 2 with near-uniform logits/attention.
- Final and eval/test are all Happy.
- Margin loss exists, but does not open useful boundary.
- Train final still predicts many Angry, but validation/test do not; this indicates poor generalization or optimization mismatch.

Conclusion: hard-negative margin in this form is not usable.

## 12. Root Cause Conclusion

### 12.1 Data / Exposure

Evidence:

- Target repeat likely increased exposure because train batches increased by about 100.
- Disgust remained 0 on eval/test.
- Repeat + rare aux still Disgust = 0.

Conclusion: exposure may contribute, but it is not the main sufficient explanation.

### 12.2 Objective

Evidence:

- Rare aux BCE does not rescue Disgust.
- Hard-negative margin collapses.
- Logit adjustment helps Angry only.

Conclusion: objective-level rescue helps Angry but does not solve Disgust. Continuing to stack loss tricks is unlikely to be enough.

### 12.3 Classifier Bias

Evidence:

- `rare_aux_logit_tau05` rescues Angry more than all other runs.
- Disgust remains strongly negative-margin and never appears.

Conclusion: Angry has a classifier-bias component. Disgust is not only classifier bias.

### 12.4 Representation / Micro-Expression

Evidence:

- Disgust remains 0 under repeat, aux, aux+logit, repeat+aux, and margin.
- Local branch does not learn Disgust either.
- Slots are diffuse in learned runs relative to stable baseline.
- Disgust is mostly swallowed by Happy/Sad/Surprise.
- Rare aux loss barely improves and does not translate into main predictions.

Conclusion: the strongest diagnosis is representation/micro-expression failure, possibly made worse by speed runtime and CE dynamics.

### 12.5 Runtime / DDP

Evidence:

- Speed control collapses far below old stable baseline.
- Target-repeat and hard-margin also collapse to all-Happy.
- Runtime fields confirm DDP/AMP/compile/fixed-shape path.

Conclusion: runtime/DDP/fixed-shape path is a serious suspect because the control itself failed. But the current artifacts cannot isolate runtime as the only cause. A direct parity run would be needed.

## 13. Answers to the Seven Main Questions

1. Target repeat Disgust có làm Disgust xuất hiện lại không?  
   No. It likely increased exposure, but eval/test Disgust pred count remains 0.

2. Rare auxiliary BCE có tạo tín hiệu riêng cho Angry/Disgust không?  
   Only very weakly for Angry in rare-aux alone. It does not create Disgust predictions. Rare aux loss barely decreases.

3. Rare auxiliary + logit adjustment có cứu Angry tốt hơn và có chạm Disgust không?  
   Yes for Angry, no for Disgust. Angry F1 reaches 0.1477; Disgust remains 0.

4. Repeat Disgust + rare auxiliary có phải hướng mạnh nhất cho Disgust không?  
   It is the strongest Disgust-targeted test, but it still fails. It improves Disgust margin relative to rare-aux alone, but not enough to produce class-1 predictions.

5. Hard-negative margin có mở boundary cho Angry/Disgust không?  
   No. The run collapses to all-Happy on eval/test.

6. Speed control có tái hiện stable CE-first cũ không?  
   No. It is far worse and collapses to all-Happy. This is a major warning about runtime/config parity.

7. Nếu vẫn chết class 0/1, lỗi nghiêng về sampling/exposure, objective, hay representation/micro-expression?  
   Angry: classifier bias + weak representation.  
   Disgust: representation/micro-expression failure is most likely, with runtime/sampler still a serious confound because the control failed.

## 14. Why D10/D11 Did Not Die as Badly as D12A

This report did not re-read D10/D11 artifacts, so this conclusion is inferential from the current D12A artifacts and previous D12 context.

The current D12A rare-rescue runs show that CE-only/full-global/scale2 under speed runtime does not preserve rare-class decision boundaries. D10/D11 likely had stronger or more stable motif/representation training, while this D12A path is relying on weak CE plus auxiliary attempts after the local motif representation is already insufficient.

The key difference is not simply class imbalance. If it were mostly exposure, `target_repeat_disgust` or `repeat_disgust_rare_aux` should have moved Disgust. They did not.

## 15. Decision

Do not continue this exact rare-rescue family as the main path. It did not rescue Disgust and it damaged Neutral across all six runs.

Do not jump to D12B/window attention yet. The current evidence first requires:

1. speed-runtime parity check,
2. D12A micro-detail diagnostics,
3. only then a targeted architecture change.

Do not return to stronger SupCon immediately. The issue now is not lack of contrastive pressure alone; Disgust has no stable main/local decision signal even with exposure and rare aux.

## 16. Recommended Next Steps

Maximum three next directions:

### Step 1: Runtime / Sampler Parity Audit

Because `speed_control_ce_first` failed badly, run a direct parity check next, not a new rare loss:

- same stable CE-first config,
- compare old runtime vs DDP speed runtime,
- log per-rank/per-batch label histograms,
- persist sampler diagnostics:
  - `target_class_repeat_factors`,
  - repeated count,
  - expanded sample count,
  - per-rank label histogram,
  - per-epoch class exposure.

Goal: determine whether speed runtime itself changed optimization quality.

### Step 2: D12A-Micro Diagnostics Before Architecture Change

Add diagnostics before changing the encoder:

- `h_pixel_std` after scale1,
- `h_pixel_std` after scale2,
- node embedding cosine similarity before/after scale2,
- slot attention visualization for true Angry and Disgust,
- per-class slot area and slot centers,
- UMAP/t-SNE for local context or motif embeddings by class,
- binary rare-aux metrics: AUROC, AP, precision/recall for class 0 and 1 heads.

Goal: prove whether micro-expression is being smoothed away.

### Step 3: If Diagnostics Confirm Micro-Detail Loss, Build D12A-Micro

Proposed design:

```text
h_fine = scale1 local encoder
h_context = scale2/context encoder
h_pixel = h_fine + alpha * gate * h_context
```

With:

- small `alpha`, e.g. 0.2-0.3,
- scale2 as residual context, not overwrite path,
- optional micro-slot area/entropy diagnostic or very light loss,
- no D12B window attention yet,
- no node_dim=12,
- no batch96.

## 17. Appendix A: Read Artifacts

For each run, read:

- `training_history.json`
- `resolved_config.yaml`
- `evaluation/metrics.json`
- `evaluation/classification_report.json`
- `evaluation/classification_report.txt`
- `evaluation/predictions.csv`
- `evaluation/d6b_diagnostics.json`
- checkpoint metadata from `checkpoints/best.pth`
- file presence of `checkpoints/last.pth`
- W&B local summary/logs where useful

## 18. Appendix B: Missing / Limited Artifacts

Missing or not persisted as structured files:

- numeric confusion matrix as `.npy/.json/.csv`
- `diagnostics_history.json`
- `val_metrics.json`
- `test_metrics.json`
- rare auxiliary binary metrics
- sampler repeat diagnostics beyond resolved config and train batch count
- per-rank/chunk label histograms
- local-branch predictions on eval/test
- scale1 vs scale2 per-layer diagnostics
- slot visualization by class

Reliability warnings:

- Early rare predictions are unreliable because epoch-1 logits and class attention are nearly uniform.
- Best checkpoints for speed control, target repeat, and hard-negative margin occur at epoch 1/2 and are not mature learned states.
- Confusion rows are exact from `predictions.csv`; explanations about micro-expression are evidence-supported but still inferential until visual/embedding diagnostics are added.

