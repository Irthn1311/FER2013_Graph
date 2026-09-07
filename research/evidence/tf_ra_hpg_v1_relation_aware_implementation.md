# RA-HPG v1.0 relation-aware bridge implementation evidence

Status: `RA_HPG_V1_RELATION_AWARE_IMPLEMENTATION_ONLY`

## Provenance and scope

- Issue: `#55 [Bridge RA-HPG v1.0 Implementation]`
- Exact parent: `8468dfe40ae2e6229c835b3356bb61eac1f22c77`
- Branch: `codex/issue-55-ra-hpg-v1-relation-aware`
- Implementation source/test commit: `56e4fca84804f5012cb90581731064b203e0d5e0`
- Evidence-bearing review head: the commit containing this file; its exact SHA is recorded in the draft PR body.
- Isolated package: `research/candidates/tf_ra_hpg_v1_relation_aware/`
- No FER2013 train/validation data was opened. Kaggle was not launched. The test split was not accessed. No scientific result or RA-HPG follow-up was produced.

Changed implementation files at the source/test commit:

- `research/candidates/tf_ra_hpg_v1_relation_aware/__init__.py`
- `research/candidates/tf_ra_hpg_v1_relation_aware/data.py`
- `research/candidates/tf_ra_hpg_v1_relation_aware/graph.py`
- `research/candidates/tf_ra_hpg_v1_relation_aware/model.py`
- `research/candidates/tf_ra_hpg_v1_relation_aware/train_validation_only.py`
- `tests/test_tf_ra_hpg_v1_relation_aware.py`

## Locked identities

| Artifact | SHA-256 |
|---|---|
| `model.py` | `a312e7bc9fc00df45dc11e897ba99a09eb24484ecd7a9c03748a3bb4e7ba6500` |
| `graph.py` | `fd425766f8db4d53f87edca0bebb88ff476c43938fff6a507c0687a0efb2d8ec` |
| `data.py` | `e8b586aafc6fbad74d913cb08d25a61432c0e55173a60621597f4c5a8d091776` |
| `train_validation_only.py` | `ef24467f132df342333435c48251fc115b99bec622329bb725b7d58ab55f787b` |

`graph.py` and `data.py` are byte-identical to accepted CF-HPG v1.2. The built model identity is exactly:

- parameters: `626987`
- trainable variables: `74`
- Keras variables: `84`

## Architecture and equations

Observed synthetic shapes:

- input `[B,48,48,1]`
- row-major patches `[B,144,16]`
- fine block inputs `[B,144,96]`
- fine adjacency per block `[B,144,144]`
- hierarchy output `[B,36,96]`
- coarse block inputs `[B,36,128]`
- coarse adjacency per block `[B,36,36]`
- final coarse nodes `[B,36,128]`
- coarse-only mean/max readout `[B,256]`
- readout hidden `[B,128]`
- logits `[B,7]`

Each of the two fine and two coarse blocks implements the registered pre-norm equations:

1. `z_i = LayerNorm(x_i)`
2. `r_ij = z_j - z_i`
3. `a_ij = sigmoid(Dense_D_to_1(r_ij))`
4. `m_i = sum_j 1[A_ij] * a_ij * r_ij / max(sum_j 1[A_ij], 1)`
5. graph update `concat[z_i || m_i] -> Dense(2D,D) -> GELU -> Dense(D,D) -> Dropout(0.15)`, then residual add
6. FFN `LayerNorm(D) -> Dense(D,4D) -> GELU -> Dense(4D,D) -> Dropout(0.15)`, then residual add

The relation gate is scalar, biased, sigmoid-bounded, and has shape `[B,N,N,1]`. The golden test zeroes gate kernel and bias, obtains gate `0.5`, and verifies exact `0.5 * mean_neighbor(z_j-z_i)` degree-normalized messages. Inactive and self edges contribute zero.

The frozen `spatial_8 UNION cosine_top4` builder is called exactly twice at each scale, immediately before the corresponding block and on that block's current unnormalized input. Tests record the order `builder, block, builder, block` and prove that no adjacency tensor is intentionally reused.

The tokenizer, parameter-free `12x12 -> 6x6` arithmetic-mean hierarchy, coarse-only v1.2 readout, augmentation, optimizer, learning-rate schedule, checkpoint policy, early stopping, comparator, diagnostic thresholds, and lifecycle remain frozen.

## Verification

Environment: `C:\Users\ADMIN\anaconda3\envs\lap-gnn-tf\python.exe`, TensorFlow `2.18.1`.

- Focused suite:
  - command: `python -m pytest -q tests/test_tf_ra_hpg_v1_relation_aware.py`
  - result: `51 passed`, `14` third-party deprecation warnings, `70.87s`
- Combined same-process CF-HPG v1.0-v1.3 plus RA-HPG suite:
  - command: `python -m pytest -q tests/test_tf_cf_hpg_v1.py tests/test_tf_cf_hpg_v1_1_resolution.py tests/test_tf_cf_hpg_v1_2_tokenizer.py tests/test_tf_cf_hpg_v1_3_multiscale_readout.py tests/test_tf_ra_hpg_v1_relation_aware.py`
  - result: `307 passed`, `14` third-party deprecation warnings, `119.17s`
  - an earlier invocation reached the shell's `120s` orchestration timeout; the unchanged suite was rerun with a longer wrapper timeout and passed.
- Synthetic `tf.function` forward/backward: finite logits, finite loss, all `74` gradients finite, AdamW changed parameters: `PASS`.
- Synthetic `.keras` save/load: restored logits reproduce original logits within `1e-6`: `PASS`.
- Lexical test-path rejection before file open, CLI rejection before loader/output creation, and explicit tiny-CSV-only open tracking: `PASS`.
- Fresh import from outside the repository with explicit repository root and `PYTHONPATH` cleared: identity `626987 / 74 / 84`: `PASS`.
- PyTorch runtime isolation: `torch` absent from imports and `sys.modules`: `PASS`.
- Frozen TensorFlow checksum verifier: `PASS checked=267 failures=0`.
- `git diff --check`: `PASS`.

## Frozen-source verification

`git diff --quiet 8468dfe40ae2e6229c835b3356bb61eac1f22c77 -- <path>` returned `0` for every protected path:

- `research/candidates/tf_cf_hpg/`
- `research/candidates/tf_cf_hpg_v1_1_resolution/`
- `research/candidates/tf_cf_hpg_v1_2_tokenizer/`
- `research/candidates/tf_cf_hpg_v1_3_multiscale_readout/`
- `research/candidates/tf_learned_local_residual_slots/`
- `tools/run_issue40_step13_execution.py`
- `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/`

These checks establish implementation conformance only. They do not establish FER2013 performance or support any scientific interpretation.
