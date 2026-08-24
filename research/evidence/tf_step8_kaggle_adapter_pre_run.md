# TensorFlow Step 8 Kaggle adapter pre-run evidence

## Scope and immutable provenance

- Issue: #15, `[TF Research Step 8] Run fixed-checkpoint direct-part pathway decomposition experiment`.
- Preregistered scientific base commit: `d14e7e1e3eec2ffbd5339b5a3bd0d5db5ab3de8b`.
- Reviewed execution commit: `a1b1d279bb9ec388f1d93ad86196e423dc750ad1`, the merged Gate-A hotfix commit whose parent is the scientific base.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- Reviewed Step 7 harness SHA-256: `fc60ece71caea14927c4840edfcd527d005737106f60d0bb475b9b1ba79eadd3`.
- Reviewed Step 6 support-tool SHA-256: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`.
- Frozen execution-contract SHA-256: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`.

The notebook clones the repository without checkout, detaches the exact reviewed execution commit, requires a clean worktree, runs the package checksum verifier, and independently checks both reviewed tool hashes before execution. The preregistered scientific base remains separately recorded. The merged hotfix changes only the Step 7 technical harness boundary handling and its tests/evidence; the frozen scientific payload remains unchanged.

## Failed-attempt preservation

The first registered Step 8 attempt remains classified exactly as **PRE-INTERVENTION TECHNICAL HARNESS FAILURE / INVALID_MANUAL_FORWARD_EQUIVALENCE**. It stopped at Gate A before any D1-D4 intervention outcome. It produced no valid Step 8 scientific result and is not overwritten or reinterpreted by this adapter relock. No post-hotfix Kaggle rerun was performed.

## Adapter artifacts

- Notebook: `notebooks/kaggle-issue15-direct-part-decomposition.ipynb`, SHA-256 `e97a6a0482fbd09b16e196a5f777babfc6cc8f412785de3f39ece0e09c914a11`.
- Deterministic builder: `tools/build_issue15_kaggle_adapter.py`, SHA-256 `d34a2ec1252a4c897b12a99fe26ba4debd117dfd604a5ddf63e6f90bff243eff`.
- Adapter tests: `tests/test_issue15_kaggle_adapter.py`, SHA-256 `72b55d28b3b47d9e76376ca5e6f6cc1f880f4d88a059c52434a82e19e6302410`.
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
- Internet: required only to clone the exact reviewed execution commit and, if the Kaggle image versions differ, install the registered dependencies. Dataset, cache, checkpoint, metadata, and config inputs are offline.
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
- Existing Step 7 direct-part harness suite in the dedicated TensorFlow environment: PASS, `16 passed`.
- Package checksum verification: PASS, `checked=263 failures=0`.
- Parent-import isolation verification: PASS, zero violations.
- PyTorch-runtime isolation verification: PASS, zero violations.
- `git diff --check`: PASS.

## Pre-run stop

Every code cell in the relocked notebook remains unexecuted with empty outputs. The first registered attempt's Gate-A failure is preserved as stated above; Kaggle was not rerun after the hotfix. The real Issue #7 checkpoint was not located or loaded locally, full validation was not run, and no valid D1-D4 metric, label, decision, or scientific outcome was produced. This PR must remain draft and stop at renewed research-lead pre-run review before any Kaggle execution.
