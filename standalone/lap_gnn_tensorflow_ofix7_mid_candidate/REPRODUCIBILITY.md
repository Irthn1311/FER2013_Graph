# Reproducibility

- Locked seed: 42.
- Locked split counts: 28,709 / 3,589 / 3,589.
- Resume: disabled for the first TensorFlow run.
- Primary checkpoint: validation macro-F1.
- Scheduler and early stopping monitor validation loss.
- Parity mode: CPU float32, dropout off, mixed precision off, XLA off.
- Kaggle mode: GPU required by default, batch size 16, bounded prefetch, XLA off.

Converted weights come from `validation_assets/golden/model_state.npz`; normal
TensorFlow runtime never requires a `.pt` file.

