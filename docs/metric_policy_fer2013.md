# FER-2013 Metric Policy

## Classification

- Best checkpoint: `val_macro_f1`.
- Early stopping: `val_macro_f1`.
- Accuracy is a reporting metric, not the primary checkpoint selector.
- `val_loss` is logged and may be used by `ReduceLROnPlateau`.
- Cosine-style schedulers step by epoch and do not monitor validation metrics.
- If a configured monitor metric is missing, training should fail clearly instead of falling back to accuracy or loss.

## Motif Discovery Stage 1

- Best checkpoint: motif quality, currently `motif_quality_score` in `scripts/train_motif_discovery_stage1.py`.
- Do not use emotion accuracy or macro F1 for Stage 1 motif discovery checkpointing.
- Judge Stage 1 with motif-quality logs plus visualization, separability, and stability audits after training.

## Deprecated

- Best checkpoint by `val_accuracy` for FER-2013 classification.
- Early stopping by `val_loss` for classification when the reported target is macro F1.
- Treating `schedloss` as a reason to also select checkpoints or early-stop by loss.
- Historical configs with names such as `bestacc`, `earlyloss`, or `schedloss` should remain for provenance, but should not be used as mainline configs.
