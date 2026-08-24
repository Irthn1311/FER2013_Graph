# TensorFlow Step 8 Gate-A forensic pre-run evidence

## Scope and status

This is a separate validation-only technical diagnostic for Issue #15. It does not modify or rerun the Step 8 D0-D5 scientific decomposition and cannot produce a D1-D4 scientific outcome. Both registered attempts remain `PRE-INTERVENTION TECHNICAL HARNESS FAILURE / INVALID_MANUAL_FORWARD_EQUIVALENCE`; the second is the post-hotfix attempt. Kaggle has not been run for this forensic path.

## Exact provenance

- Preregistered scientific base: `d14e7e1e3eec2ffbd5339b5a3bd0d5db5ab3de8b`.
- Required merged hotfix ancestor: `a1b1d279bb9ec388f1d93ad86196e423dc750ad1`.
- Exact forensic execution commit: `3cae1f6c78048cd6cd518d87cd0a5429d72f01e1`; both the scientific base and hotfix are verified ancestors.
- Forensic tool SHA-256: `30c00fd6985810533cc09be05f66b64f7da5a794903aef493b9839b461eac7c0`.
- Reviewed Step-7 harness SHA-256: `fc60ece71caea14927c4840edfcd527d005737106f60d0bb475b9b1ba79eadd3`.
- Reviewed Step-6 support-tool SHA-256: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- Frozen execution-contract SHA-256: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`.

The exact Issue #7 artifacts remain:

- checkpoint `best_val_accuracy.keras`: `9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16`;
- metadata `best_val_accuracy.metadata.json`: `e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37`;
- config `resolved_config.json`: `3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32`.

Epoch `31`, seed `42`, config hash, execution contract, graph/feature/prior signatures, and dataset split signature are fail-closed. No file under `src/lap_gnn_tf`, `contracts`, or `validation_assets` changed. Package finalization reproduced the frozen scientific payload exactly.

## Forensic tool contract

For each original validation batch, in order, the tool executes only:

1. `model(batch, training=False)` as `native_1`;
2. `model(batch, training=False)` as `native_2`;
3. the reviewed Step-7 `manual_forward(..., D0)` as `manual_1`;
4. the same reviewed manual D0 as `manual_2`.

It records batch index, sample count, sample-ID SHA-256, node/edge counts, source dtypes, observed model-boundary dtypes, and prediction/logit/probability comparisons for native/native, manual/manual, and both paired native/manual calls. Source tensors are checked after every forward. The loaded model-weight hash is checked after every completed batch; checkpoint and model hashes are checked again on exit.

Each batch is written atomically to `forensic/batches/batch_NNNNN.json`, followed by an atomic `progress.json` update. The unchanged Gate-A tolerances are recorded only as reference flags; exceedance does not stop collection. TensorFlow op determinism is not enabled.

The dtype manifest covers the outer LapGNN, encoder, all GNN layers, PartGlobalContext, readout, classifier, and tracked nested layers, recording dtype policy, compute dtype, variable dtype, input dtype, and autocast where available.

D1-D5 interventions, training/optimizer/gradient operations, test access, raw-prior corruption, and graph rebuild are absent and fail-closed.

## Kaggle adapter and failure preservation

- Notebook SHA-256: `f4a30d5c4a9892c12fab6bd121e52ebf13ba2da4e9962e5bdd2729fc80ae0d75`.
- Deterministic builder SHA-256: `f8656519d43b338158a9802b9bd5c2fed891c48ad15443e6239a0c316da829dd`.
- Adapter test SHA-256: `fccc473acbb58df1b261e57c4b33297f1a608de55b6ca1966492bb12b182e374`.
- Forensic-tool test SHA-256: `d499e0ba3251046f59aa5d2b52f6a7084c143845cc7d894d0e1094c147510251`.
- All 9 notebook code cells have `execution_count: null` and empty outputs.

The wrapper captures combined subprocess output in `tf_step8_gate_a_forensic/forensic_subprocess.log`. On a non-zero return or wrapper exception it records `TECHNICAL_FORENSIC_FAILURE`, writes `wrapper_execution.json` and an explicit failure report, preserves all existing incremental batch/progress/manifest/failure JSON, and creates and verifies the compact ZIP. It then completes normally so Kaggle can publish `/kaggle/working` outputs while remaining scientifically fail-closed: `scientific_interpretation` is null, `scientific_decomposition_run` is false, interventions remain empty, and no success-only `final_evidence.json` is fabricated. A focused regression executes a synthetic non-zero subprocess and verifies this full failure-publication contract without expecting `RuntimeError`.

## Kaggle inputs and output contract

- FER mount `/kaggle/input/datasets/doduyquynii/fer13-split`; resolved sample file `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/val.csv` only.
- Prior mount `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue`; resolved sample records `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue/val/*.npz` plus required shared schema/name metadata.
- Cache mount/root `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`; resolved `val/index.json` and its validation shards, plus disclosed shared aggregate `CACHE_COMPLETE.json`.
- Issue #7 artifacts: one separate read-only Kaggle Input. The mount name is deliberately not fixed; exact resolved artifact paths are discovered and recorded at runtime only after unique basename-plus-SHA matching outside the three sample inputs.
- Internet: required only for cloning the exact execution commit and conditional installation of registered dependencies. All model/data/cache assets are offline inputs.
- Report: `/kaggle/working/tf_step8_gate_a_forensic.md`.
- Always-created compact archive: `/kaggle/working/tf_step8_gate_a_forensic_kaggle_t4.zip`.

## Verification performed

- Gate-A forensic tool plus reviewed Step-7 regression suites: PASS, `23 passed`.
- Gate-A adapter plus Issue #11 adapter suites: PASS, `18 passed`.
- Package checksum verification: PASS, `checked=265 failures=0`.
- Parent-import isolation: PASS, zero violations.
- PyTorch-runtime isolation: PASS, zero violations.
- Notebook JSON parse, compilation, deterministic rebuild, and unexecuted-cell checks: PASS.
- `git diff --check`: PASS.

## Pre-run stop

No Kaggle forensic execution, registered checkpoint load, full validation pass, D1-D5 intervention, or scientific result occurred. The diagnostic hypotheses remain unresolved until the research lead reviews a future Kaggle forensic archive. Stop for research-lead pre-run review before execution.
