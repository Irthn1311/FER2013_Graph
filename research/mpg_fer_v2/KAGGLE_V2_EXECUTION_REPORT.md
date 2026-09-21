# MPG-FER v2 Official Kaggle T4 Execution Report

## 1. Final status

`COMPLETED`

The official source-locked MPG-FER v2 run completed normally in one Kaggle segment. Training stopped at epoch 82 by the preregistered early-stopping rule (`min_epochs=50`, `patience=20`); the globally selected checkpoint was epoch 62. PrivateTest was evaluated once, only after that EMA checkpoint was frozen and hashed.

## 2. Provenance

| Field | Recorded value |
|---|---|
| GitHub Issue | `#92` |
| Branch | `research/mpg-fer-v2-issue92` |
| Reproducibility commit | `96e5aec27cde315038965ee525b14043cc5ab29a` |
| Reviewed source SHA-256 | `f9a06cd4f7482c6a6e37d58844c1a30022602e9fc825eff73240f71740b0af1c` |
| Reviewed notebook SHA-256 | `3dc96c5d45030ae09b1138bf37720cfc6d8049be3e09ee440ccea478d86f9956` |
| Submitted notebook SHA-256 | `fcea88ac8f3bea1872fa357d3a0d14d60b9b6597a2d2d1905e5e916b5a9ce03a` |
| Kaggle account | `irthn1311` |
| Actual Kaggle kernel | `irthn1311/mpg-fer-v2-official-t4-segment-01`, version 1, private |
| Dataset | `doduyquynii/fer13-split` |
| Run ID | `08b511fd-317e-414c-b854-e5633311e8b1` |
| Trainable parameters | `2,238,609` |
| Scientific config SHA-256 | `d30792dd6e66572bb4aff885ef98154c7b8e5e5fa5df984c0c2c80c757b3d624` |
| Best EMA checkpoint SHA-256 | `f3cda72fc4d791e7017e2e0374f83ef22e9f22f03e8172b7389e53f8cbc6dc1c` |
| Final resume checkpoint SHA-256 | `fb8a846708da0ca19942e769d924eed3fb1e90dda97ae90e98c0d2c009f33fa2` |
| Downloaded artifact ZIP SHA-256 | `b6f9f3baf38740d7e69507ce4550cce98950400273f82593b128c6549ed2465b` |
| Pre-execution test suite | `35 passed in 53.19s` |
| Notebook static validation | Six code cells compiled successfully |

The submitted notebook differs from the reviewed notebook only by staging-time execution fields such as commit, kernel reference, segment number, and resume mode. The embedded scientific source hash remained exactly equal to the reviewed source hash.

## 3. Environment and preflight

| Check | Result |
|---|---|
| GPU | `Tesla T4` |
| CUDA | `12.8` |
| PyTorch | `2.10.0+cu128` |
| Dataset rows | Train `28,709`; PublicTest `3,589`; PrivateTest `3,589` |
| Dataset content gate | Train and Public validated; Private path/role isolated and not content-read during the preflight/training gate |
| Parameter gate | PASS: `2,238,609` |
| Bounded full-model AMP gate | PASS, batch 16, accumulation 2, effective batch 32 |
| Worst-case consistency forward | PASS |
| Finite activation/loss audits | PASS for all registered tensors |
| Gradient audits | PASS for all registered parameter groups |
| EMA update gate | PASS |
| Real FER micro-overfit | PASS: `15/16 = 93.75%` after 65 steps; target `>=87.5%` |
| Fresh reset after preflight | Confirmed by `fresh_reset_ready=true` |
| Preflight peak allocated | `2,823.321 MiB` |
| Preflight peak reserved | `2,990.000 MiB` |

## 4. Segments and training lifecycle

| Segment | Kernel/version | GPU | Epochs | Wall-clock | Best at segment end | Status | Resume verification |
|---:|---|---|---:|---:|---:|---|---|
| 1 | `irthn1311/mpg-fer-v2-official-t4-segment-01`, v1 | Tesla T4 | 1-82 | `26,564.109 s` (`7:22:44.1`) | 62 | `TRAINING_COMPLETED` | Not applicable; fresh run and no boundary |

- Number of segments: **1**.
- Epochs completed: **82**.
- Sum of measured epoch durations: `26,527.936 s` (`7:22:07.9`).
- Training wall-clock from the segment manifest: `26,564.109 s` (`7:22:44.1`).
- Approximate whole-kernel elapsed time from the final Kaggle log timestamp: `26,666 s` (`7:24:26`).
- Termination: legitimate early stopping at epoch 82, exactly 20 non-improving epochs after best epoch 62.
- Best epoch: **62**.
- LR at best epoch: `0.00015254197172800965`.
- Maximum measured training allocation: `2,828.850 MiB`.
- Maximum measured training reservation: `3,172.000 MiB`.
- No resume boundary occurred; therefore no resume dataset or Segment 2 was created. The final `resume_latest.pt` and its metadata nevertheless passed SHA-256 verification.

## 5. Selected PublicTest results

All values below come from the frozen best EMA checkpoint at epoch 62.

| View | Accuracy | Macro-F1 | Loss |
|---|---:|---:|---:|
| EMA raw | **67.4283%** | **64.5323%** | `1.038632` |
| EMA horizontal-flip TTA | **68.6821%** | **66.3030%** | `0.973634` |

### Public per-class metrics

Class order is the FER2013 integer-label order: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

| Class | Raw precision | Raw recall | Raw F1 | TTA precision | TTA recall | TTA F1 |
|---|---:|---:|---:|---:|---:|---:|
| Angry (0) | 57.7253% | 57.6017% | 57.6635% | 60.7456% | 59.3148% | 60.0217% |
| Disgust (1) | 55.5556% | 53.5714% | 54.5455% | 62.7451% | 57.1429% | 59.8131% |
| Fear (2) | 56.5217% | 49.7984% | 52.9475% | 58.1176% | 49.7984% | 53.6374% |
| Happy (3) | 86.0068% | 84.4693% | 85.2311% | 85.5693% | 84.8045% | 85.1852% |
| Sad (4) | 58.3463% | 57.2741% | 57.8053% | 59.6947% | 59.8775% | 59.7859% |
| Surprise (5) | 79.9539% | 83.6145% | 81.7432% | 79.5455% | 84.3373% | 81.8713% |
| Neutral (6) | 58.5546% | 65.4036% | 61.7899% | 60.5926% | 67.3806% | 63.8066% |

### Public confusion matrices

Rows are true classes and columns are predicted classes in the class order above.

Raw:

```text
269 12 51 17 60 16 42
 12 30  2  2  7  0  3
 47  3 247 14 84 38 63
 21  3 16 756 16 17 66
 71  6 69 23 374 11 99
 11  0 27 15  7 347  8
 35  0 25 52 93  5 397
```

TTA:

```text
277  9 50 17 60 16 38
 10 32  2  2  8  0  2
 43  3 247 17 93 39 54
 19  1 18 759 14 20 64
 66  5 56 23 391 12 100
  8  0 27 14  8 350  8
 33  1 25 55 81  3 409
```

## 6. One-shot PrivateTest results

The best checkpoint was frozen and its SHA-256 recorded before this evaluation. `PRIVATE_EVALUATED=true` and `private_evaluated_only_after_freeze=true` are recorded in the execution manifest.

| View | Accuracy | Macro-F1 | Loss |
|---|---:|---:|---:|
| EMA raw | **68.0412%** | **67.0094%** | `0.990854` |
| EMA horizontal-flip TTA | **69.8245%** | **69.3816%** | `0.917704` |

### Private per-class metrics

| Class | Raw precision | Raw recall | Raw F1 | TTA precision | TTA recall | TTA F1 |
|---|---:|---:|---:|---:|---:|---:|
| Angry (0) | 59.5789% | 57.6375% | 58.5921% | 63.1470% | 62.1181% | 62.6283% |
| Disgust (1) | 65.1515% | 78.1818% | 71.0744% | 74.1379% | 78.1818% | 76.1062% |
| Fear (2) | 52.0879% | 44.8864% | 48.2197% | 56.7506% | 46.9697% | 51.3990% |
| Happy (3) | 88.8367% | 86.0068% | 87.3988% | 89.1101% | 86.5757% | 87.8246% |
| Sad (4) | 52.7301% | 56.9024% | 54.7368% | 54.9536% | 59.7643% | 57.2581% |
| Surprise (5) | 79.5294% | 81.2500% | 80.3805% | 81.2796% | 82.4519% | 81.8616% |
| Neutral (6) | 66.1243% | 71.4058% | 68.6636% | 65.4572% | 72.0447% | 68.5932% |

### Private confusion matrices

Raw:

```text
283  9 59 12 71 12 45
  7 43  1  1  1  1  1
 67  3 237  9 121 43 48
 20  0 19 756 25 21 38
 55  6 76 28 338  3 88
 12  1 31 16  9 338  9
 31  4 32 29 76  7 447
```

TTA:

```text
305  7 46 13 70  8 42
  8 43  1  1  0  1  1
 59  4 248  8 110 43 56
 20  0 19 761 28 17 34
 44  1 67 26 355  5 96
 13  0 26 16  9 343  9
 34  3 30 29 74  5 451
```

## 7. Frozen v1 to v2 comparison

Absolute deltas are percentage points (`v2 - v1`). Regressions are retained explicitly.

| Private metric | Frozen v1 | MPG-FER v2 | Delta |
|---|---:|---:|---:|
| Raw accuracy | 68.0691% | 68.0412% | **-0.0279 pp** |
| TTA accuracy | 69.6016% | 69.8245% | **+0.2229 pp** |
| Raw macro-F1 | 66.9523% | 67.0094% | **+0.0571 pp** |
| TTA macro-F1 | 68.7379% | 69.3816% | **+0.6437 pp** |

The primary practical result is a small TTA accuracy gain and a larger TTA macro-F1 gain, while raw accuracy is essentially flat with a `-0.0279 pp` regression. No rerun or tuning was initiated after observing these results.

## 8. Motif diagnostics

| Diagnostic | v1 approximate final | v2 best epoch 62 | v2 final epoch 82 |
|---|---:|---:|---:|
| Temperature `tau` | Not supplied | 0.150031 | 0.150007 |
| Assignment/local entropy | ~3.85855 | 2.544694 | 2.540968 |
| Normalized local entropy | Not supplied | 0.657340 | 0.656377 |
| Global entropy | Not supplied | 3.865198 | 3.862309 |
| Normalized global entropy | Not supplied | 0.998449 | 0.997703 |
| Effective motif count | ~47.4 / 48 | 12.7394 / 48 | 12.6921 / 48 |
| Utilization minimum | ~0.02009 | 0.017563 | 0.017426 |
| Utilization maximum | ~0.02136 | 0.024894 | 0.026249 |
| Utilization standard deviation | Not supplied | 0.002183 | 0.002700 |
| Mean top-1 probability | Not supplied | 0.132060 | 0.132598 |
| Mean top-2 probability | Not supplied | 0.127617 | 0.127826 |
| Mean top1-top2 margin | Not supplied | 0.004442 | 0.004772 |
| Mean off-diagonal prototype cosine | Not supplied | -0.020821 | -0.020809 |
| Scale weight 8 | Not supplied | 0.328471 | 0.317708 |
| Scale weight 12 | Not supplied | 0.360191 | 0.361824 |
| Scale weight 16 | Not supplied | 0.311338 | 0.320468 |

Measured behavior: v2 learned much sharper local assignments than the v1 approximation, reducing assignment entropy by about `1.3176` nats and reducing the entropy-equivalent active set from about `47.4` to `12.69` motifs. At the same time, global entropy remained near its 48-way maximum (`3.8712` nats) and utilization remained distributed across all prototypes, so the evidence does **not** show pathological global collapse. The very small top1-top2 margin shows that local specialization is stronger but not hard or decisively separated. `tau` converged essentially to its configured lower bound, which should be recorded as a saturation observation rather than interpreted automatically as success. Scale 12 received the largest average weight, with scales 8 and 16 still materially active.

## 9. Artifact verification

- `best_val_acc.pt` SHA matches `best_val_acc.json` and the execution/final-selection manifests.
- `resume_latest.pt` SHA matches `resume_latest.json` and the segment manifest.
- Source SHA, config SHA, run ID, best epoch, and checkpoint identity agree across manifests.
- The downloaded ZIP contains 21 members and Python `ZipFile.testzip()` returned no bad member.
- The `final_run` copy contains the same 21 run files and all source-to-copy SHA comparisons passed.
- Both confusion-matrix PNGs and `training_curves.png` are present.
- The canonical final-state file is `resume_latest.pt`; no fabricated or renamed `last.pt` was introduced.

Local artifact locations:

- Segment archive and raw Kaggle output: `research/mpg_fer_v2/outputs/kaggle_v2_final/segments/segment_01/`
- Canonical final run: `research/mpg_fer_v2/outputs/kaggle_v2_final/final_run/`
- Kaggle ZIP: `research/mpg_fer_v2/outputs/kaggle_v2_final/segments/segment_01/mpg_fer_v2_artifacts.zip`
- Kernel log: `research/mpg_fer_v2/outputs/kaggle_v2_final/segments/segment_01/mpg-fer-v2-official-t4-segment-01.log`

## 10. Deviations and limitations

1. Kaggle normalized the staged suggested ID `mpg-fer-v2-final-t4-2026-09-21` to the actual notebook slug `mpg-fer-v2-official-t4-segment-01`. The actual slug above is the authoritative retrieval reference; the embedded manifest retains the originally requested kernel reference.
2. The run completed by early stopping in Segment 1, so the expected multi-segment resume procedure was not exercised in this performance run. This is not a protocol failure.
3. The notebook did not print a separate “saved best checkpoint” line during epochs. Saving nevertheless occurred atomically before the epoch log, and the downloaded checkpoint passed its recorded SHA-256 check.
4. Kaggle emitted notebook-format and nbconvert warnings after execution. They did not fail the kernel or alter the recorded training status.
5. The v1 motif figures supplied for comparison are approximate and incomplete. Claims about motif change are limited to the reported fields and do not establish semantic quality by themselves.

## 11. Completion decision

All required runtime gates passed, the official source and parameter identities matched, training completed under the registered early-stopping rule, the global best EMA checkpoint was frozen before the one-shot PrivateTest evaluation, required metrics and diagnostics were measured, and downloaded checkpoint/artifact integrity checks passed.

**Final status: `COMPLETED`**
