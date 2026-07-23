# Reproducibility

The source of truth is the two lock hashes recorded in `package_manifest.json`.
The runtime was mechanically extracted from parent commit
`241a8872027cd284fe679533a0be95cb48e7d253`.

Install with `python -m pip install -e .`, validate data first, and always pass
FER CSV, prior root, optional cache root and output root explicitly. Resume is
disabled. The primary checkpoint is validation macro-F1; validation loss drives
the scheduler and early stopping.

The historical replicated environment was Python 3.12.12, PyTorch 2.10.0+cu128,
CUDA 12.8 and cuDNN 91002. Local extraction validation also records its tested
environment in the machine-readable reports.
