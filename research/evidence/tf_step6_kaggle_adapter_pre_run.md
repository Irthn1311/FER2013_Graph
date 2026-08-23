# TensorFlow Step 6 Kaggle adapter pre-run evidence

## Scope and provenance

- Issue: #11, `[TF Research Step 6] Run fixed-checkpoint fixed-topology validation prior-sensitivity experiment`.
- Frozen scientific base: `main` at `69f4571c5069da9a7f8558ef3c01101635ee904a`.
- Technical execution source: config-identity harness hotfix commit `9d7f7ef9b9f821e66d7f671e7ec860c1fe8aa81f`.
- Adapter notebook: `notebooks/kaggle-issue11-fixed-topology-prior-probe.ipynb`.
- Notebook SHA-256: `ebb3cf56d0724e61561b3447dd09c1e823735abe71fcc85377c79443058163b7`.
- Deterministic builder SHA-256: `fd8ed0f5f4a9c954455682542bcfc4924ee2b5f1ce4bf8129d6919cc384661a3`.
- Reviewed Step 5 tool SHA-256: `b3a668bb16d4daf70b9f32b03bd35281b3791925dff97da35d4a245bcf75c4d4`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- The checked-in notebook remains unexecuted. The first approved Kaggle attempt failed before model evaluation and before any C0/C1/C2 metric was produced; it is classified `PRE-METRIC TECHNICAL HARNESS FAILURE`. No Kaggle rerun was performed for this hotfix.

## Runtime-blocker closure

The failed attempt found all three exact registered artifacts at their read-only Kaggle mount and verified their SHA-256 values:

- `/kaggle/input/datasets/irthn1311/kaggle-issue7-validation-only/best_val_accuracy.keras`;
- `/kaggle/input/datasets/irthn1311/kaggle-issue7-validation-only/best_val_accuracy.metadata.json`;
- `/kaggle/input/datasets/irthn1311/kaggle-issue7-validation-only/resolved_config.json`.

It then stopped at the config identity guard: the persisted JSON identity was `a4038682bf09c03786e86119001cf5f81ac5fec25d09062e6a0866484c32a3cf`, while loading that JSON through the YAML loader reinterpreted scientific-notation floats as strings and produced `916472d6e813b15dca6e7cd5016e87e95fad5d608bc1e341117675797cce8e54`. Validation assets were correct and no test access occurred.

The minimal harness hotfix now loads the persisted resolved config as UTF-8 JSON, checks metadata identity against that raw mapping before any mutation, and creates a deep copy for runtime overrides only after the identity gate passes. No identity value is hardcoded into the tool, and every existing fail-closed guard remains active. A regression test locks the exact `a403...` raw identity, reproduces the obsolete `9164...` YAML reinterpretation, and proves runtime mutation cannot alter the raw identity object.

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
- Internet: required only to clone technical execution commit `9d7f7ef9b9f821e66d7f671e7ec860c1fe8aa81f` (whose scientific base remains `69f4571c5069da9a7f8558ef3c01101635ee904a`) and install registered dependencies if the Kaggle image is not already TensorFlow `2.18.1` / Keras `3.15.0` compatible.

The registered command invokes `evaluate_fixed_checkpoint_prior_probe.py` once with the checkpoint, metadata, config, official prior root, official clean-cache root, and a fresh output root. It contains no `--limit-val-batches`, training command, optimizer path, test split path, or raw-prior corruption path.

## Compact outputs after approved execution

- Report: `/kaggle/working/tf_step6_fixed_topology_prior_sensitivity.md`.
- Archive: `/kaggle/working/tf_step6_fixed_topology_prior_sensitivity_kaggle_t4.zip`.

The archive is limited to the Step 5 probe outputs, adapter metadata/evidence, and compact report. It rejects `.keras`, train/test CSVs or directories, and test-metric artifacts.

## Verification performed without experiment execution

- `python -m pytest -q tests/test_issue11_kaggle_adapter.py` — PASS, `10 passed`.
- Relevant Step 5, payload, and isolation tests — PASS, `31 passed`.
- `python tools/verify_checksums.py` from the TensorFlow package — PASS, `checked=261 failures=0`.
- `python tools/verify_no_parent_imports.py` — PASS, zero violations.
- `python tools/verify_no_torch_runtime.py` — PASS, zero violations.
- `nbformat.validate(...)` — PASS, notebook format `4.5`, `19` cells.
- Every notebook code cell compiles; all execution counts are null and all outputs are empty.
- `git diff --check` — PASS.

## Pre-run limitations and review gate

- The first approved live attempt established artifact discovery and hashes, then failed at the pre-metric config identity gate. It produced no validation asset count, C0 reproduction result, C1/C2 outcome, or output archive.
- C0 failure is locked to `INVALID_REFERENCE_REPRODUCTION`; in that case the adapter preserves raw files and suppresses C1/C2 derived reporting and scientific interpretation.
- C2 remains explicit semantic-prior/direct-part sensitivity conditional on the official MediaPipe-derived scaffold. It is not MediaPipe removal and is not a prior-free graph.
- One identical registered rerun is allowed only after research-lead review of this hotfix. No outcome-driven protocol change or rerun has been performed here.
