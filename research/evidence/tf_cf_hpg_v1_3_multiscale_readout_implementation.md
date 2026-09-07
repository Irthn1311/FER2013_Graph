# CF-HPG v1.3 multiscale-readout implementation evidence

Status: `CF_HPG_V1_3_MULTISCALE_READOUT_IMPLEMENTATION_ONLY`

## Scope and provenance

- Issue: `#51 [Gen2 CF-HPG v1.3 Implementation] Add fine+coarse multiscale readout only`
- Authoritative identity amendment: issue comment `#5565175531`
- Authoritative lifecycle clarification: issue comment `#5565353891`
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

## Frozen v1.3 lifecycle comparator and decisions

The future v1.3 lifecycle compares only against the accepted Issue #50 v1.2
result:

- clean-train accuracy: `0.614894284022432`
- clean-train macro-F1: `0.5614369765915708`
- validation accuracy: `0.5806631373641683`
- validation macro-F1: `0.5238975323290902`

All four percentage-point deltas subtract these exact v1.2 values. The result
artifact records them under `deltas_vs_v1_2_pp`. Exact decision precedence is
`CF_HPG_V1_3_STRETCH_PASS`, `CF_HPG_V1_3_PASS`,
`READOUT_OVERFIT_SHIFT`, `READOUT_STRONG_SIGNAL`,
`READOUT_PARTIAL_SIGNAL`, `READOUT_UNDERFIT_REMAINS`, then
`READOUT_INCONCLUSIVE`. Focused overlap tests prove overfit precedes strong and
partial, while PASS and STRETCH_PASS remain higher priority. Source regression
also proves the superseded v1.1 comparator and v1.2/tokenizer labels are absent
from the v1.3 lifecycle and tests.

## Source SHA-256

- `__init__.py`: `86ebab1643f03ae75260c4ed160d152b8c2cb8b5b52b6760780ceb296d7f67bd`
- `data.py`: `e8b586aafc6fbad74d913cb08d25a61432c0e55173a60621597f4c5a8d091776`
- `graph.py`: `fd425766f8db4d53f87edca0bebb88ff476c43938fff6a507c0687a0efb2d8ec`
- `model.py`: `2001fa00a2c21400817dfcd0857e1f3dfe1e0f7979039eee2efa7b7123160495`
- `train_validation_only.py`: `332586ad60646f48fb26d72a78571fde9e46d8ff5b06c830a2ba68605c66fc16`
- focused test: `74edb7cbaefcd877f3c28e10d7588da055550bc4eb58cbfaed8fe14a20bf3480`

The evidence-file SHA is reported separately because a file cannot contain
its own stable digest.

## Synthetic verification

- Final focused v1.3 suite: `72 passed`.
- Comparator, delta, decision, result-field, and stale-identifier selection:
  `17 passed`, `55 deselected`.
- Combined v1.0/v1.1/v1.2/v1.3 same-process suite: `256 passed`; no Keras
  registry/name collision.
- Synthetic `tf.function` forward/backward: finite logits, loss, and all
  gradients; AdamW updated parameters.
- Keras save/load round trip: exact synthetic inference logits.
- Explicit forward/backward and serialization selection: `2 passed`.
- Explicit forbidden test-path selection: `15 passed`, `57 deselected`.
- Fresh import with `PYTHONPATH` removed: PASS; exact identity and v1.2
  comparator reproduced.
- PyTorch runtime isolation: PASS; `torch` was not imported.
- Frozen checksum verifier: `PASS checked=267 failures=0`.
- Frozen source diff from the exact parent: PASS, empty.
- Staged `git diff --check`: PASS.

Only synthetic tensors and one-row temporary CSV fixtures were used. These
checks establish implementation integrity and do not constitute scientific
outcome evidence.
