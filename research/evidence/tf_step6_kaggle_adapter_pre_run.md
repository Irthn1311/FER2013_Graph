# TensorFlow Step 6 Kaggle adapter pre-run evidence

## Scope and provenance

- Issue: #11, `[TF Research Step 6] Run fixed-checkpoint fixed-topology validation prior-sensitivity experiment`.
- Exact base: `main` at `69f4571c5069da9a7f8558ef3c01101635ee904a`.
- Adapter notebook: `notebooks/kaggle-issue11-fixed-topology-prior-probe.ipynb`.
- Notebook SHA-256: `26011e492b7684c5a62b38c5013dbaaee5550b1eda3544dd63f9bc94c55bc838`.
- Deterministic builder SHA-256: `b6c86769fea6ce754f940dfd34051d4ca65d83b357ab73ad5a5e4e58a7c7a050`.
- Reviewed Step 5 tool SHA-256: `564eab26b7cf683bd531fec08bf6539a1384d9ef370961b9484335726c7c2351`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- This PR prepares the adapter only. The notebook was not executed, the real Issue #7 checkpoint was not loaded, and no C0/C1/C2 outcome was observed.

## Locked Issue #7 artifacts

The adapter accepts exactly one matching file for each expected basename outside the public FER/prior/cache inputs and verifies it by SHA-256 before use:

- `best_val_accuracy.keras`: `9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16`;
- `best_val_accuracy.metadata.json`: `e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37`;
- `resolved_config.json`: `3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32`.

It additionally locks epoch `31`, seed `42`, config hash `a4038682bf09c03786e86119001cf5f81ac5fec25d09062e6a0866484c32a3cf`, the execution/graph/feature/prior/split signatures from Issue #11, and the exact persisted C0 reference metrics. Artifact hashes are checked again after the probe. The actual Kaggle mount is intentionally not trusted and will be resolved only during the approved Kaggle session.

## Kaggle inputs and access boundary

- FER dataset input: `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`; resolved sample artifact is `val.csv` only.
- MediaPipe prior input: `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`; resolved sample artifacts are `val/*.npz` only, plus shared root schema/name metadata required by the frozen loader.
- Clean graph cache input: `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`; resolved sample artifacts are `val/index.json` and its referenced validation shards. Root `CACHE_COMPLETE.json` is read only as required non-sample aggregate loader metadata and may summarize other splits.
- Issue #7 artifact input: a separate read-only Kaggle Input mounted anywhere beneath `/kaggle/input`; the three files above are located by expected basename plus exact SHA-256, with public sample inputs excluded from the search.
- Internet: required only to clone exact commit `69f4571c5069da9a7f8558ef3c01101635ee904a` and install registered dependencies if the Kaggle image is not already TensorFlow `2.18.1` / Keras `3.15.0` compatible.

The registered command invokes `evaluate_fixed_checkpoint_prior_probe.py` once with the checkpoint, metadata, config, official prior root, official clean-cache root, and a fresh output root. It contains no `--limit-val-batches`, training command, optimizer path, test split path, or raw-prior corruption path.

## Compact outputs after approved execution

- Report: `/kaggle/working/tf_step6_fixed_topology_prior_sensitivity.md`.
- Archive: `/kaggle/working/tf_step6_fixed_topology_prior_sensitivity_kaggle_t4.zip`.

The archive is limited to the Step 5 probe outputs, adapter metadata/evidence, and compact report. It rejects `.keras`, train/test CSVs or directories, and test-metric artifacts.

## Verification performed without experiment execution

- `python -m pytest -q tests/test_issue11_kaggle_adapter.py` — PASS, `10 passed`.
- Relevant Step 5, payload, isolation, and existing Kaggle-notebook tests — PASS, `31 passed`.
- `python tools/verify_checksums.py` from the TensorFlow package — PASS, `checked=261 failures=0`.
- `python tools/verify_no_parent_imports.py` — PASS, zero violations.
- `python tools/verify_no_torch_runtime.py` — PASS, zero violations.
- `nbformat.validate(...)` — PASS, notebook format `4.5`, `19` cells.
- Every notebook code cell compiles; all execution counts are null and all outputs are empty.
- `git diff --check` — PASS.

## Pre-run limitations and review gate

- No live Kaggle T4 execution, checkpoint discovery, TensorFlow environment probe, validation asset count, C0 reproduction gate, or output archive has been verified yet. Those remain runtime evidence for the approved run.
- C0 failure is locked to `INVALID_REFERENCE_REPRODUCTION`; in that case the adapter preserves raw files and suppresses C1/C2 derived reporting and scientific interpretation.
- C2 remains explicit semantic-prior/direct-part sensitivity conditional on the official MediaPipe-derived scaffold. It is not MediaPipe removal and is not a prior-free graph.
- The notebook must remain unexecuted until research-lead pre-run approval. No outcome-driven protocol change or rerun is authorized.
