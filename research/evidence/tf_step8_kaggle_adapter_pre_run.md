# TensorFlow Step 8 Kaggle adapter pre-run evidence

## Scope and immutable provenance

- Issue: #15, `[TF Research Step 8] Run fixed-checkpoint direct-part pathway decomposition experiment`.
- Exact scientific base and execution commit: `d14e7e1e3eec2ffbd5339b5a3bd0d5db5ab3de8b`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- Reviewed Step 7 harness SHA-256: `e611b74ac143c50149326c9761b35177183a09b3cf44b52ab018b01ed3d87ffd`.
- Reviewed Step 6 support-tool SHA-256: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`.
- Frozen execution-contract SHA-256: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`.

The notebook clones the repository without checkout, detaches the exact registered commit, requires a clean worktree, runs the package checksum verifier, and independently checks both reviewed tool hashes before execution. No file under frozen `src/lap_gnn_tf`, package `src/lap_gnn_tf`, `contracts`, or `validation_assets` changed.

## Adapter artifacts

- Notebook: `notebooks/kaggle-issue15-direct-part-decomposition.ipynb`, SHA-256 `67b9c54676d65490019ca69a1fa38c467882d7be74f41b7ff1a4fbd8488e6e58`.
- Deterministic builder: `tools/build_issue15_kaggle_adapter.py`, SHA-256 `92a40b54491088043673d973061a7f1a96a40f6176395a689827d974a79df713`.
- Adapter tests: `tests/test_issue15_kaggle_adapter.py`, SHA-256 `5c9d8d25b35bb823aa1e7b6cbbf5094aac55805a670e8d7eaa8725a3cc27143b`.
- Every code cell is unexecuted and has empty outputs; rebuilding is byte-deterministic.

## Locked Issue #7 artifacts

The adapter searches outside the public sample inputs, accepts exactly one basename-plus-content-hash match under `/kaggle/input`, and rejects missing or ambiguous matches:

- `best_val_accuracy.keras`: `9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16`.
- `best_val_accuracy.metadata.json`: `e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37`.
- `resolved_config.json`: `3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32`.

Metadata is fail-closed on epoch `31`, seed `42`, config hash `a4038682bf09c03786e86119001cf5f81ac5fec25d09062e6a0866484c32a3cf`, execution contract, graph/feature/prior signatures, and dataset split signature. All three artifact hashes are captured before and after the single harness invocation and must remain identical.

## Kaggle inputs and runtime boundary

- FER input mount: `/kaggle/input/datasets/doduyquynii/fer13-split`; resolved sample path `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/val.csv` only.
- Prior input mount: `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue`; resolved sample path `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue/val/*.npz` plus the frozen loader's shared root schema/name metadata.
- Cache input mount/root: `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`; resolved sample paths are `val/index.json` and its referenced validation shards. Root `CACHE_COMPLETE.json` is read only as disclosed non-sample aggregate loader metadata and may summarize other splits.
- Issue #7 artifact input: a separate read-only Kaggle Input; its mount name is deliberately not trusted, and the exact resolved paths are recorded at runtime after SHA-only discovery.
- Internet: required only to clone the exact registered repository commit and, if the Kaggle image versions differ, install the registered dependencies. Dataset, cache, checkpoint, metadata, and config inputs are offline.
- Runtime is fail-closed on Kaggle GPU T4, TensorFlow `2.18.1`, Keras `3.15.0`, and at least one visible TensorFlow GPU.

No train/test sample path is constructed or opened. No training, optimizer, gradient, checkpoint selection, graph rebuild, raw-prior corruption, or scientific intervention logic exists in the adapter.

## Registered invocation and evidence gates

The notebook constructs one command for `evaluate_fixed_checkpoint_direct_part_decomposition_probe.py`, passes the SHA-verified checkpoint/metadata/config and only the registered prior/cache roots, and calls it exactly once. Resource-only inference arguments are present. No `--limit-val-batches` argument is present.

Before producing a scientific report, the adapter requires and preserves:

- Gate A `PASS`, exact prediction agreement `1.0`, and the locked native/manual logit/probability tolerances;
- Gate B `PASS`, exactly `3589` validation samples, and the locked D0 references/tolerances;
- Gate C `PASS` and the locked D5 references/tolerances;
- the exact D0-D5 condition order and output inventory;
- all D0-D5 metrics and per-class F1, D1-D5 paired diagnostics, D1-D4 deltas/labels, overall decision, native/manual batch evidence, and intervention checks per condition;
- unchanged checkpoint file and model-weight hashes, unchanged graph/node/edge inputs and topology, paired original-batch evaluation, and explicit training/test isolation.

The compact report consumes the reviewed harness's `registered_gates_and_diagnostics` verbatim. It does not reclassify outcomes. It retains the preregistered non-additivity, functional-sensitivity, non-causal, non-model-selection, and fixed-MediaPipe-scaffold boundaries.

## Compact future outputs

- Runtime report: `/kaggle/working/tf_step8_direct_part_decomposition.md`.
- Compact archive: `/kaggle/working/tf_step8_direct_part_decomposition_kaggle_t4.zip`.
- Future reviewed repository report path: `research/evidence/tf_step8_direct_part_decomposition.md`.

The archive contains the report, harness outputs, and adapter metadata only. It rejects `.keras`, train/test CSV paths, train/test directories, and test-metric artifacts.

## Verification performed

- Issue #15 and Issue #11 adapter suites: PASS, `21 passed`.
- Existing Step 7 direct-part harness suite in the dedicated TensorFlow environment: PASS, `14 passed`.
- Package checksum verification: PASS, `checked=263 failures=0`.
- Parent-import isolation verification: PASS, zero violations.
- PyTorch-runtime isolation verification: PASS, zero violations.
- `git diff --check`: PASS.

## Pre-run stop

The notebook was not executed. The real Issue #7 checkpoint was not located or loaded locally, Kaggle was not run, full validation was not run, and no D1-D4 metric, label, decision, or scientific outcome was produced. This PR must remain draft and stop at research-lead pre-run review before any Kaggle execution.
