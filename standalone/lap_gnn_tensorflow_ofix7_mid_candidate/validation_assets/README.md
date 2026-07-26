# Validation Assets

These are bounded, portable NumPy/JSON fixtures exported from the exact
historical parent implementation at commit
`241a8872027cd284fe679533a0be95cb48e7d253`.

The eight-sample golden batch covers all seven FER2013 classes and includes a
fallback landmark sample. `model_state.npz` is a framework-neutral array export,
not a historical PyTorch checkpoint. The assets exist only for parity,
isolated-copy smoke validation and a future TensorFlow implementation contract.

No fixture metric is a research result. See `manifest.json` for sample IDs,
checkpoint hash, tolerances and per-file hashes.
