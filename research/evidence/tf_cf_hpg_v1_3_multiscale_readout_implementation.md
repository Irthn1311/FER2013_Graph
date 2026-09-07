# CF-HPG v1.3 multiscale-readout implementation evidence

Status: `CF_HPG_V1_3_MULTISCALE_READOUT_IMPLEMENTATION_ONLY`

## Scope and provenance

- Issue: `#51 [Gen2 CF-HPG v1.3 Implementation] Add fine+coarse multiscale readout only`
- Authoritative identity amendment: issue comment `#5565175531`
- Exact parent: `032c38d49d189000b364d24af7e248eba200acf7`
- Implementation commit: `919deaf5bcbae93a912cae561db41a030f73addc`
- Branch: `codex/issue-51-cf-hpg-v1-3-multiscale-readout`
- Isolated candidate: `research/candidates/tf_cf_hpg_v1_3_multiscale_readout/`

This is implementation evidence only. No FER2013 training or validation,
Kaggle execution, test-split access, or scientific interpretation occurred.

## Exact architecture change

The accepted v1.2 model remains unchanged through the second fine graph block.
At that point the candidate computes the following parameter-free readout:

`fine_readout = concat([fine_mean, fine_max])`, shape `[B,192]`.

The existing hierarchy and coarse graph path then run unchanged. The existing
coarse readout remains `concat([coarse_mean, coarse_max])`, shape `[B,256]`.
The classifier input is exactly:

`concat([fine_mean, fine_max, coarse_mean, coarse_max])`, shape `[B,448]`
`-> LayerNorm(448) -> Dense(448,128) -> GELU -> Dropout(0.20) -> Dense(128,7)`.

There is no learned scale weighting, attention, slot, query, auxiliary head,
or additional graph operation.

| Tensor | Synthetic shape |
|---|---|
| Raw patches | `[B,144,16]` |
| Fine adjacency | `[B,144,144]` |
| Fine nodes | `[B,144,96]` |
| Fine readout | `[B,192]` |
| Mean-hierarchy output | `[B,36,96]` |
| Coarse adjacency | `[B,36,36]` |
| Coarse nodes | `[B,36,128]` |
| Coarse readout | `[B,256]` |
| Fused readout | `[B,448]` |
| Logits | `[B,7]` |

## Exact amended identity

- Total/trainable parameters: `445799`
- Trainable variables: `66`
- Total Keras variables: `76`
- Delta from v1.2: `+24960` parameters
  - `readout_dense`: `+24576`
  - `readout_norm`: `+384`

Layer-by-layer shape comparison proves that only `readout_norm` and
`readout_dense` parameter shapes differ from v1.2. The future lifecycle fails
closed unless all three amended identity fields match.

## Frozen components

`data.py` and `graph.py` are byte-identical to accepted v1.2:

- `data.py`: `e8b586aafc6fbad74d913cb08d25a61432c0e55173a60621597f4c5a8d091776`
- `graph.py`: `fd425766f8db4d53f87edca0bebb88ff476c43938fff6a507c0687a0efb2d8ec`

The tokenizer, patch resolution, graph construction/reuse, Max-Relative
blocks, arithmetic-mean hierarchy, augmentation, optimizer, LR schedule,
loss, checkpoint, early stopping, seed, and test-path guards remain frozen.
The v1.0, v1.1, v1.2, Generation-1, and Step-13 paths have empty diffs from
the exact parent.

## Source SHA-256

- `__init__.py`: `86ebab1643f03ae75260c4ed160d152b8c2cb8b5b52b6760780ceb296d7f67bd`
- `data.py`: `e8b586aafc6fbad74d913cb08d25a61432c0e55173a60621597f4c5a8d091776`
- `graph.py`: `fd425766f8db4d53f87edca0bebb88ff476c43938fff6a507c0687a0efb2d8ec`
- `model.py`: `2001fa00a2c21400817dfcd0857e1f3dfe1e0f7979039eee2efa7b7123160495`
- `train_validation_only.py`: `2739575b77664046e97ac3fb128b50b7beb100a676f32bd1bbe977a2ebebd932`
- focused test: `33707ed1a40d7be1ae9d6b0a421e71d5ef119cc0bb1dd28b70a8645bd4492ac2`

The evidence-file SHA is reported separately because a file cannot contain
its own stable digest.

## Synthetic verification

- Final focused v1.3 suite: `73 passed`.
- Combined v1.0/v1.1/v1.2/v1.3 same-process suite: `257 passed`; no Keras
  registry/name collision.
- Synthetic `tf.function` forward/backward: finite logits, loss, and all
  gradients; AdamW updated parameters.
- Keras save/load round trip: exact synthetic inference logits.
- Explicit forbidden test-path selection: `15 passed`, `58 deselected`.
- Fresh import with `PYTHONPATH` removed: PASS; exact identity reproduced.
- PyTorch runtime isolation: PASS; `torch` was not imported.
- Frozen checksum verifier: `PASS checked=267 failures=0`.
- Frozen source diff from the exact parent: PASS, empty.
- Staged `git diff --check`: PASS.

Only synthetic tensors and one-row temporary CSV fixtures were used. These
checks establish implementation integrity and do not constitute scientific
outcome evidence.
