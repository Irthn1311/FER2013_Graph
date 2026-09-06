# CF-HPG v1.2 Tokenizer implementation evidence

Status: `CF_HPG_V1_2_TOKENIZER_IMPLEMENTATION_ONLY`

## Scope and provenance

- Issue: `#48 [Gen2 CF-HPG v1.2] Test deeper convolution-free raw-patch tokenizer after v1.1 partial resolution signal`
- Exact parent: `934057f60a6cb3bc7ec247e000573a40c390a426`
- Branch: `codex/issue-48-cf-hpg-v1-2-tokenizer`
- Isolated candidate: `research/candidates/tf_cf_hpg_v1_2_tokenizer/`
- The reviewed v1.0 and v1.1 candidates, Generation-1 package, Step-13 probe,
  and Step-13 execution adapter remain unchanged from the exact parent.

This implementation adds one shared raw-local refinement to the reviewed v1.1
model. It does not contain a full FER2013 run, a Kaggle execution, test-split
access, or scientific outcome evidence.

## Single registered architecture change

The raw patch path is exactly:

`[B,144,16] -> Dense(16,96) -> GELU -> Dense(96,96) -> add positional Dense(2,96) -> LayerNorm -> GELU -> Dropout(0.10)`

The added `raw_local_refinement` Dense has `96*96 + 96 = 9312`
parameters. It is the only new parameterized layer relative to v1.1. There is
no tokenizer residual branch, extra normalization, convolution, attention,
overlap, multiscale branch, pretrained feature, anatomy, handcrafted feature,
or learned slot.

All remaining tensors and semantics are frozen from v1.1:

| Tensor | Synthetic shape |
|---|---|
| Input | `[B,48,48,1]` |
| Raw patches | `[B,144,16]` |
| Fine adjacency | `[B,144,144]` |
| Mean-hierarchy output | `[B,36,96]` |
| Coarse adjacency | `[B,36,36]` |
| Final coarse nodes | `[B,36,128]` |
| Logits | `[B,7]` |

The width-96 fine and width-128 coarse stages each retain exactly two
Max-Relative blocks. Each spatial-8-neighbor union cosine-top-4 graph is built
once and reused twice. The exact non-overlapping 2x2 arithmetic mean and global
mean/max readout remain unchanged.

`data.py` and `graph.py` are byte-identical to v1.1. Their SHA-256 values are,
respectively, `e8b586aafc6fbad74d913cb08d25a61432c0e55173a60621597f4c5a8d091776`
and `fd425766f8db4d53f87edca0bebb88ff476c43938fff6a507c0687a0efb2d8ec`.

## Exact model identity

- Total/trainable parameters: `420839`
- Trainable variables: `66`
- Total Keras variables: `76`
- Difference from v1.1: exactly `+9312` parameters and one Dense layer

The future lifecycle fails closed unless all three identities match exactly.

## Frozen future lifecycle and diagnostics

Training configuration remains seed 42, batch 64, AdamW, learning rate `3e-4`,
weight decay `5e-4`, global clip norm 1.0, five-epoch warmup, cosine decay to
`1e-6`, 100 maximum epochs, label smoothing 0.05, earliest strict maximum
validation-accuracy checkpointing, and validation-loss early stopping with
patience 15 and minimum delta 0. Train-only augmentation, clean evaluation,
and fail-closed test-path guards are byte-identical to v1.1.

The accepted v1.1 clean-train and validation references are locked exactly.
Future percentage-point deltas are computed from those values. Decision
precedence is STRETCH_PASS, PASS, TOKENIZER_OVERFIT_SHIFT,
TOKENIZER_STRONG_SIGNAL, TOKENIZER_PARTIAL_SIGNAL,
TOKENIZER_UNDERFIT_REMAINS, then TOKENIZER_INCONCLUSIVE. Tests lock exact
boundaries and overlaps. Twelve-decimal comparison normalization prevents
binary floating-point representation from moving a mathematically exact 3.0
or 5.0 pp boundary; threshold values are unchanged.

## Source SHA-256

- `__init__.py`: `18349cbb55052faf26b879a03c67e3cfb4cc399eaea5ac3206f756dd95a99de4`
- `data.py`: `e8b586aafc6fbad74d913cb08d25a61432c0e55173a60621597f4c5a8d091776`
- `graph.py`: `fd425766f8db4d53f87edca0bebb88ff476c43938fff6a507c0687a0efb2d8ec`
- `model.py`: `36a9a3e9e7b3ca84581a46654b592e26f71ed1cd4a156acd387417c4320986b0`
- `train_validation_only.py`: `33c70c15e04ff5c381a0c6bb19365adb4cf3066d404f0a76f8940a2d270beac5`
- `tests/test_tf_cf_hpg_v1_2_tokenizer.py`: `2bbbd0d4b080c1b02711d4392f09eddb1056b65afdf7271456d219d0186caa2f`

The evidence-file hash is reported separately because a file cannot contain
its own stable SHA-256.

## Synthetic verification

- Focused Issue #48 suite: `70 passed`.
- Combined v1.0/v1.1/v1.2 same-process regression: `184 passed`; Keras
  serialization registries coexist without conflict.
- Synthetic `tf.function` forward/backward: finite logits, loss, and all
  gradients; AdamW changes at least one parameter.
- Keras save/load round trip: exact synthetic logits.
- Explicit forward/backward and serialization selection: `2 passed`.
- Explicit forbidden test-path selection: `15 passed`, `55 deselected`.
- Exact params and variable inventories: PASS.
- Graph construction/reuse, Max-Relative golden tensor, hierarchy, readout,
  augmentation, lifecycle, decision boundaries, and path isolation: PASS.
- Fresh candidate/CLI import with `PYTHONPATH` removed: PASS; PyTorch remained
  absent from the runtime.
- Frozen Generation-1 checksum verification: `PASS checked=267 failures=0`.
- v1.0/v1.1 candidate diff and Generation-1/Step-13 diff from the exact parent:
  empty.
- `git diff --check`: PASS after staging the complete patch.
- Only synthetic tensors and one-row temporary CSV files are used.

No full FER training or validation was run, no Kaggle job was launched, and no
test split was accessed.
