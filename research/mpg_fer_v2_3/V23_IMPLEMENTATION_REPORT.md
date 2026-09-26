# MPG-FER v2.3 implementation report

## Outcome

`V23_IMPLEMENTATION_READY_FOR_OFFICIAL_TRAINING`

The implementation and local validation are complete. Official FER2013 training and all v2.3 Public/Private performance, classwise, model-resolvable, calibration, and probe results remain **UNKNOWN**.

## Provenance and source lock

- Preregistered Issue: [#97](https://github.com/Irthn1311/FER2013_Graph/issues/97)
- Branch: `research/mpg-fer-v2-3-early-depth-generalization`
- Reviewed v2.2 base: `0a258fd43cc4d8f45afa54ea1328f068c52cbee0`
- Frozen v2.2 source SHA-256: `a8dc77db29e997c4c3ab69bb862704c8a948f940a4636e1c01e0d96bab40de65`
- v2.3 implementation commit: `08faea291ef425c10cc11ab0bc880e6cef302e97`
- v2.3 source SHA-256: `1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87`
- v2.3 scientific config SHA-256: `8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2`

## A6-R2 source-lock cleanup

The cleanup did not rerun any experiment. Derived hypothesis and final-target JSON are now deterministic projections of `a6r2_master_results.json`; the consistency checker compares complete derived documents, all standalone numerical sections, and all 55 formatted routing values in the source-lock report. The corrected routing rescue-association range is `-0.0535335...` to `+0.0402906...`, with all p-values at least `0.192981...`. The result is `CONSISTENCY_PASS (9/9)`.

Historical A6 and A6-R artifacts were not changed.

## Single scientific delta

A6-R2 shows the Train/held-out probe gap expanding most rapidly from PRE through Motif Layers 1-2. v2.3 therefore scales both existing residual branches in each early block:

```text
h <- h + scale * DropPath(attention_update)
h <- h + scale * DropPath(ffn_update)
scale = [0.5, 0.5, 1.0, 1.0, 1.0]
```

This fixed half-step preserves more of the incoming contextual representation while retaining learned relational updates. It introduces no parameter, learned gate, schedule, auxiliary objective, or sweep. The falsifiable hypothesis and rejected alternatives are recorded in `V23_DESIGN_RATIONALE.md`.

All other scientific components remain v2.2-equivalent, including Pixel Graph, Composer, WHAT/TYPE/WHERE, 49 motif nodes, geometry features, five motif layers, Top-K `[8,16,16,16,24]`, readout, fusion, classifier, losses, optimizer/scheduler, augmentation, EMA, split roles, checkpoint selection, and TTA.

## Compatibility and measured validation

| Gate | Measured result |
|---|---|
| Parameters | `2,304,528`; delta from v2.2 `0` |
| v2.3 full suite | `90 passed in 328.73s` |
| Frozen v2.2 full suite | `73 passed in 283.66s` |
| Sparse/intervention focused suite | `37 passed in 55.25s` |
| v2.2-equivalent OFF behavior | scale-one v2.3 strict-loads v2.2 state; logits exactly equal |
| Non-target source lock | checkpoint/data/EMA/evaluate/features/graph/losses/motif/utils byte-identical to v2.2 |
| Real-Train bounded micro-overfit | 4/4 correct at step 20; loss `0.4638562` |
| AMP bounded audit | batch 8 PASS; optimizer + EMA step; every audited motif Q/K/V/geometry gradient finite and nonzero |
| AMP local peak | `1451.46 MiB` allocated; `1560 MiB` reserved on RTX 3050 Ti |

The official 16-example FP32 micro-overfit could not run on the local 4 GiB GPU: CUDA allocation failed before a result. This is recorded as a resource limitation, not a model failure. The source-locked Kaggle notebook reruns the unchanged 16-example gate on Tesla T4 and stops on failure. The local four-example check is implementation validation only and provides no generalization evidence.

## Historical reference metrics

These are existing official artifacts, not measurements produced by v2.3 work.

| Model | Params | Public raw Acc / F1 | Private raw Acc / F1 | Public TTA Acc / F1 | Private TTA Acc / F1 |
|---|---:|---:|---:|---:|---:|
| v2.1 dense | 2,304,528 | 0.674283 / 0.648979 | 0.682084 / 0.673331 | 0.692672 / 0.670585 | 0.699359 / 0.689995 |
| v2.2 sparse | 2,304,528 | 0.676790 / 0.659180 | 0.684313 / 0.675953 | 0.696573 / 0.676789 | 0.695458 / 0.687141 |
| v2.3 | 2,304,528 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

The later A6 canonical FP32 raw values use a separate locked evaluation artifact and must not be mixed silently with this official-run table.

## Official Kaggle handoff

Regenerate and stage the exact implementation commit:

```powershell
cd D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\research\mpg_fer_v2_3
C:\Users\ADMIN\anaconda3\envs\fer-graph\python.exe tools\sync_notebook.py
C:\Users\ADMIN\anaconda3\envs\fer-graph\python.exe tools\stage_kaggle_kernel.py `
  --output-dir staged\segment_01 `
  --kernel-ref irthn1311/mpg-fer-v2-3-official-t4-segment-01 `
  --git-commit 08faea291ef425c10cc11ab0bc880e6cef302e97 `
  --segment 1 `
  --resume-mode fresh `
  --dataset doduyquynii/fer13-split
```

Staged notebook: `staged/segment_01/MPG_FER_v2_3_Kaggle_T4.ipynb`.

- Attached input: Kaggle dataset `doduyquynii/fer13-split`.
- Expected resolved split mount: `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/{train,val,test}.csv`; resolution and row counts fail closed.
- Internet: disabled. Source is embedded in the notebook; no clone is performed.
- Working output: `/kaggle/working/mpg_fer_v2_3_run/`.
- Required archive: `/kaggle/working/mpg_fer_v2_3_artifacts.zip`.

For continuation, attach only the complete prior v2.3 output dataset and stage with `--resume-mode required --segment N`. The loader only discovers `mpg-fer-v2-3-resume*`, and verifies checkpoint SHA, run ID, source hash, config hash, epoch boundary, optimizer/EMA/scaler/RNG state, best EMA checkpoint, and protocol identity. A v2.2 checkpoint is incompatible by both config and source hash.

## Expected official artifacts

During each segment, `/kaggle/working/mpg_fer_v2_3_run/` must contain the current `preflight_report.json`, `v23_model_summary.json`, `config.json`, `history.json`, `history.csv`, `routing_diagnostics.json`, `execution_manifest.json`, `segment_manifest.json`, `resume_latest.pt`, `resume_latest.json`, retained periodic resume snapshots, and the global `best_val_acc.pt` plus metadata.

Only after `TRAINING_COMPLETED` and checkpoint freeze, it must also contain `public_metrics.json`, `private_metrics.json`, `motif_diagnostics.json`, `final_selection_manifest.json`, confusion matrices, and training curves. The downloaded review bundle is then copied under `research/mpg_fer_v2_3/official_runs/<segment>/mpg_fer_v2_3_run/` without overwriting earlier segments.

The frozen post-training analysis must produce:

- `v23_training_history.json`
- `v23_best_checkpoint_metadata.json`
- `v23_public_results.json`
- `v23_private_results.json`
- `v23_probe_analysis.json` for PRE, L1, L2, L5, Motif Readout, and Fusion using Train-fit-only `StandardScaler` and multinomial L2 logistic regression (`C=1`, `lbfgs`, `max_iter=5000`, `tol=1e-6`)
- `v23_classwise_metrics.json` including Angry, Fear, Sad, Neutral, Happy, Surprise and requested confusion directions
- `v23_comparison_v21_v22_v23.json` including raw/TTA, model-resolvable counts, parameters, calibration/overfit tracking, and the separate A6-canonical comparison

PrivateTest is opened once by the notebook only after `TRAINING_COMPLETED` freezes the global best EMA checkpoint. Probe and error-set interpretation occurs after that freeze and cannot feed back into checkpoint selection.

## Scientific decision rule

Primary architecture evidence is raw single-view. A positive decision requires improved early-depth held-out probe behavior without simple representation collapse, meaningful raw held-out benefit, no Private reversal or catastrophic hard-class degradation, and no TTA-only gain. Until the official run and frozen analyses exist, the v2.3 hypothesis is neither supported nor rejected.

## Integrity statement

- scientific source changed: **YES, only for the intended v2.3 delta**
- v2.2 scientific source changed: **NO**
- FER labels changed: **NO**
- dataset split changed: **NO**
- Private used for tuning: **NO**
- extra training data used: **NO**
- pretrained CNN/ViT used: **NO**
- ensemble used for primary result: **NO**
- TTA used as primary architecture metric: **NO**
- A6/A6-R historical artifacts overwritten: **NO**
