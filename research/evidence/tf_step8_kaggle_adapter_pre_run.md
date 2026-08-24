# TensorFlow Step 8 Kaggle adapter pre-run evidence

## Scope and immutable provenance

- Issue: #15, `[TF Research Step 8] Run fixed-checkpoint direct-part pathway decomposition experiment`.
- Preregistered scientific base commit: `d14e7e1e3eec2ffbd5339b5a3bd0d5db5ab3de8b`.
- Reviewed execution commit/current main: `27c366a955648764386fe48e489a6a1e94a479a1`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- Reviewed Step 7 harness SHA-256: `c0b1df778e469665dd6437c58831d29dcc34fbde44231db75894c5469a1ade78`.
- Reviewed Step 6 support-tool SHA-256: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`.
- Frozen execution-contract SHA-256: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`.

The notebook clones the repository without checkout, detaches the exact reviewed execution commit, requires a clean worktree, runs the package checksum verifier, and independently checks both reviewed tool hashes before execution. The preregistered scientific base remains separately recorded. The execution lineage contains the reviewed model-boundary autocast hotfix and the Issue #19 technical Gate-A calibration; the frozen scientific payload remains unchanged.

## Failed-attempt preservation

The registered execution history remains classified exactly as follows:

1. **PRE-INTERVENTION TECHNICAL HARNESS FAILURE / INVALID_MANUAL_FORWARD_EQUIVALENCE**.
2. **POST-HOTFIX PRE-INTERVENTION TECHNICAL HARNESS FAILURE / INVALID_MANUAL_FORWARD_EQUIVALENCE**.

Both attempts stopped at Gate A before any D1-D4 intervention outcome. Neither produced a valid Step 8 scientific result, and neither is overwritten or reinterpreted by this adapter relock.

The separate validation-only Gate-A forensic covered all `113` batches / `3589` samples. It established that the original `1e-6` probability tolerance was below the measured registered Kaggle T4 same-path repeatability envelope. The reviewed forensic archive SHA-256 is `bf693500078f170ceea094fad319f513c2a64d8610fa41c65ebac088ec954c8d`. This technical result motivated the preregistered Issue #19 calibration; it is not a D1-D4 scientific result and makes no claim that GPU nondeterminism affects D1-D4 outcomes.

## Adapter artifacts

- Notebook: `notebooks/kaggle-issue15-direct-part-decomposition.ipynb`, SHA-256 `e0640c763c4f4ed49b3c86898968a29ef4d51037b783e5fe60d54091370b3283`.
- Deterministic builder: `tools/build_issue15_kaggle_adapter.py`, SHA-256 `aa0c300c767f114543009e9db333037489f582942fd2442906b561072e904fcc`.
- Adapter tests: `tests/test_issue15_kaggle_adapter.py`, SHA-256 `b32a648a52787f479fc82c6dce32f5714ee0760374bc245c35b47c98250d1812`.
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
- Registered resource arguments remain evaluation batch size `32`, graph workers `2`, and graph cache size `64`.
- The registered normal-runtime policy is unchanged; this relock does not enable operation determinism.

No train/test sample path is constructed or opened. No training, optimizer, gradient, checkpoint selection, graph rebuild, raw-prior corruption, or scientific intervention logic exists in the adapter.

## Registered invocation and evidence gates

The notebook constructs one command for `evaluate_fixed_checkpoint_direct_part_decomposition_probe.py`, passes the SHA-verified checkpoint/metadata/config and only the registered prior/cache roots, and calls it exactly once. It executes exactly the preregistered D0-D5 condition set, with no extra intervention condition. Resource-only inference arguments are present. No `--limit-val-batches` argument is present.

Before producing a scientific report, the adapter requires and preserves:

- Gate A `PASS`: prediction agreement exactly `1.0`, max absolute logit difference `<= 1e-5`, and max absolute probability difference `<= 3e-6`;
- Gate B `PASS`: D0 accuracy `0.63137364168292`, macro-F1 `0.5932591901893336`, loss `1.1537981724317095`, tolerances `0.001 / 0.001 / 0.005`, and exactly `3589` samples;
- Gate C `PASS`: D5 accuracy `0.27751462803009197`, macro-F1 `0.19745892656222366`, loss `1.757720434560185`, the same tolerances, and exactly `3589` samples;
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
- Existing Step 7 direct-part harness suite in the dedicated TensorFlow environment: PASS, `17 passed`.
- Package checksum verification: PASS, `checked=265 failures=0`.
- Parent-import isolation verification: PASS, zero violations.
- PyTorch-runtime isolation verification: PASS, zero violations.
- `git diff --check`: PASS.

## Pre-run stop

Every code cell in the relocked notebook remains unexecuted with empty outputs. Both registered Gate-A failures and the reviewed forensic conclusion are preserved as stated above. Kaggle was not rerun for this relock. The real Issue #7 checkpoint was not located or loaded locally, full validation was not run, and no valid D1-D4 metric, label, decision, or scientific outcome was produced. This PR must remain draft and stop at renewed research-lead pre-run review before any Kaggle execution.
