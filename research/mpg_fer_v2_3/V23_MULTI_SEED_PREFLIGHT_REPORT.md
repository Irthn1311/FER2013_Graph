# MPG-FER v2.3 Remaining-Seeds Pre-flight Report

Date: 2026-09-27 (Asia/Saigon)  
Registered experiment: GitHub Issue #99  
Frozen architecture commit: `08faea291ef425c10cc11ab0bc880e6cef302e97`  
Prepared from commit: `75e192d0e81ef53bb71e0853530d6a48a7c13582` (the exact execution-tooling commit is injected when each kernel is staged)  
Frozen scientific source SHA-256: `1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87`

## Required launch gates

- [PASS] Code version. The v2.3 scientific source tree is byte-identical to the frozen tree at the registered architecture commit. Later repository changes do not alter `research/mpg_fer_v2_3/src`. The instantiated model has exactly 2,304,528 trainable parameters, residual scales `[0.5, 0.5, 1.0, 1.0, 1.0]`, and Top-K `[8, 16, 16, 16, 24]`.
- [PASS] v2.3 config parity. The canonical multi-seed template constructs the existing `MPGConfig` and changes only `seed`; account alias, kernel slug, and output path are operational metadata. Seed 42 reproduces config SHA-256 `8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2`.
- [PASS] Seed injection. One generated canonical notebook is used for all runs. The staging tool validates the fixed account-to-seed assignment and replaces a single seed control field; tests verify distinct seed-specific config hashes and a seed-only scientific-config delta.
- [PASS] Dataset split. Local preflight verified 28,709 Train rows, 3,589 PublicTest rows, and 3,589 PrivateTest rows in distinct `train.csv`, `val.csv`, and `test.csv` files. The frozen loader validates split role, basename, row count, and distinct resolved paths.
- [PASS] No PrivateTest during training. Training constructs only Train and PublicTest loaders. The notebook constructs the PrivateTest loader only after training has completed, the selected checkpoint has been frozen, and its SHA-256 has been recorded.
- [PASS] Checkpoint rule. The frozen rule remains EMA Public horizontal-flip TTA accuracy, then macro-F1 and loss tie-breaks. PrivateTest is not part of checkpoint selection.
- [PASS] Output artifacts. The canonical finalizer creates the required per-seed directory, manifest, resolved config, environment, command, logs/history, real checkpoint mapping, original runtime evaluation, canonical FP32 evaluation, raw predictions, confusion matrix, per-class metrics, SHA256SUMS, and a per-seed ZIP. Partial segments emit a resumable manifest/archive without fabricating final metrics.
- [PASS] Kaggle credentials mapped safely. All three supplied JSON files passed schema/authentication checks and resolve to three distinct authorized accounts. Credentials are used only through process-local environment variables, are not copied into the repository/notebook/dataset, and their usernames/keys are not logged.

## Reproducibility and execution checks

- [PASS] Python test suite: `55 passed` across multi-seed protocol, notebook protocol, v2.3 contract, and sparse-routing tests.
- [PASS] Notebook validation: generated notebook parses as JSON and every code cell compiles.
- [PASS] CUDA smoke test on actual FER rows: model instantiation, one training batch, backward pass, AdamW optimizer step, FP32 evaluation batch, and checkpoint save/load all completed; 2,304,528 parameters reconciled. Smoke metrics were not retained as experimental results.
- [PASS] Canonical precision: official AMP runtime evaluation remains preserved. Additional FP32 raw/TTA evaluation disables autocast and TF32 and runs only against the already frozen checkpoint. Existing seed-42 frozen-checkpoint FP32 evidence is retained for six-seed comparability.
- [PASS] Determinism: Python, NumPy, PyTorch CPU/CUDA, DataLoader generator, epoch sampler, and sample-local augmentation are seeded. The frozen cuDNN policy (`deterministic=True`, `benchmark=False`) is logged.

## Planned assignment

- Account A: seed 0 and seed 1
- Account B: seed 43 and seed 123
- Account C: seed 3047

At most two active jobs will be requested per account. If Kaggle enforces a lower limit, jobs remain on their assigned account and are queued sequentially; no quota bypass is permitted.

## Seed-specific config identities

| Seed | Config SHA-256 |
|---:|---|
| 0 | `fc57b7c057df4226ac2eadddb3b902ad157bd426710c6dc413ab80feaf920565` |
| 1 | `5086767014b1938f7783597498a3ea1fdb99fbd6178e8b42fb890d88d009d26e` |
| 42 | `8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2` |
| 43 | `97f594b19295822946e5632cff9535d630a3b505a2a0375cf04fc9a36dbece4e` |
| 123 | `e85608767dc1b4c57897c6273b6014089b7df9f80005345c4da9b595a21823f8` |
| 3047 | `895abb86818e7ac77c31186bac12efc2353ac79265715e14387fbc0ab135f779` |

## Discrepancies and scope boundary

No scientific discrepancy was found. The only added code is execution/evaluation/provenance tooling and its tests; the frozen training/model implementation is unchanged. A passing preflight establishes technical readiness only and is not evidence for a scientific hypothesis.
