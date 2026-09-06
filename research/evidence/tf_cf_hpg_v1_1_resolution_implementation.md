# CF-HPG v1.1 Resolution implementation evidence

Status: `CF_HPG_V1_1_RESOLUTION_IMPLEMENTATION_ONLY`

## Scope and provenance

- Issue: `#45 [Gen2 CF-HPG v1.1] Test higher-resolution 4x4 patch graph after v1.0 underfit`
- Exact parent: `562c80fe675d8527416df1b5e441df76f1f62620`
- Branch: `codex/issue-45-cf-hpg-v1-1-resolution`
- Isolated candidate: `research/candidates/tf_cf_hpg_v1_1_resolution/`
- The reviewed v1.0 candidate, Generation-1 package, Step-13 probe, and Step-13 execution adapter are unchanged from the exact parent.

This PR changes only generic spatial token resolution. It does not contain a
full FER2013 run, a Kaggle execution, a test-split evaluation, or scientific
outcome evidence.

## Registered resolution-only architecture

The candidate scales raw `48x48x1` grayscale pixels by `x / 127.5 - 1.0` and
uses reshape/transpose-only, non-overlapping `4x4` patchification. This produces
`[B,144,16]` row-major raw patches on a `12x12` generic grid. The raw projection
is therefore the only parameterized architecture component whose input width
changes, from `36 -> 96` to `16 -> 96`.

All graph and block semantics remain frozen: normalized `(x,y)` centers; one
spatial-8-neighbor union cosine-top-4 graph reused across exactly two width-96
fine Max-Relative blocks; exact non-overlapping `2x2` arithmetic-mean hierarchy
from `12x12` to `6x6`; one corresponding hybrid graph reused across exactly two
width-128 coarse blocks; and global mean/max readout to seven logits.

Measured synthetic shapes:

| Tensor | Shape |
|---|---|
| Raw patches | `[B,144,16]` |
| Fine adjacency | `[B,144,144]` |
| Hierarchy output | `[B,36,96]` |
| Coarse adjacency | `[B,36,36]` |
| Final coarse nodes | `[B,36,128]` |
| Logits | `[B,7]` |

`data.py` and `graph.py` are byte-identical copies of their reviewed v1.0
counterparts. Tests lock the same top-4 self exclusion, boolean-union
deduplication, graph reuse, Max-Relative golden tensor, augmentation sequence,
and validation/test-path isolation behavior.

## Exact model inventory

- Total/trainable parameters: `411527`
- Trainable parameter variables: `64`
- Keras variables: `74`
- Raw patch projection `16 -> 96`: `1632` parameters
- Difference from v1.0: exactly `-1920` parameters
- No convolution, overlap, attention, learned slot, pretrained feature,
  anatomical feature, extra graph block, width increase, or top-k change.

The future lifecycle fails closed unless the built model has exactly `411527`
parameters.

## Frozen future lifecycle

The future validation-only harness keeps seed `42`, AdamW, LR `3e-4`, weight
decay `5e-4`, global clip norm `1.0`, batch `64`, 5-epoch linear warmup, cosine
decay to `1e-6`, maximum 100 epochs, label smoothing `0.05`, earliest strict
maximum validation-accuracy checkpointing, and validation-loss early stopping
with patience `15` and minimum delta `0`.

Train-only augmentation and clean validation/evaluation behavior are identical
to v1.0. The CLI exposes only `--train-csv`, `--val-csv`, and `--output-root`.
Final-split lexical guards remain fail closed, including calls with
`expected_samples=3589`.

Future PASS/STRETCH thresholds and the preregistered resolution diagnostic
labels are locked in the focused tests. The result schema records the four exact
percentage-point deltas from the accepted v1.0 metrics. No values have been
observed for v1.1.

## Source SHA-256

- `__init__.py`: `fd5540dd67ab5f5178357560cc3862044b04e31121b40980199424c8be1eb693`
- `data.py`: `e8b586aafc6fbad74d913cb08d25a61432c0e55173a60621597f4c5a8d091776`
- `graph.py`: `fd425766f8db4d53f87edca0bebb88ff476c43938fff6a507c0687a0efb2d8ec`
- `model.py`: `87723409fd6f22d333b9bded2b8707d4e4a3bd5bd3bac19e48022f12f6df0fea`
- `train_validation_only.py`: `7671d596c1050e4d55c8d3de8d4b1600191c1d72a8a6b9d87344eaeb0ce088bd`
- `tests/test_tf_cf_hpg_v1_1_resolution.py`: `79023b4736475cf7c55bb39c7cde351e1db111a1c6c344f67365feba701fa28c`

The evidence-file hash is reported separately in the PR because a file cannot
contain its own stable SHA-256.

## Synthetic verification

- Focused Issue #45 suite: `55 passed`.
- Reviewed v1.0 suite: `53 passed`.
- Combined v1.0/v1.1 process regression: `108 passed`; serialization registry
  coexistence produced no conflict.
- Synthetic `tf.function` forward/backward: finite logits, loss, and every
  gradient; AdamW updates at least one parameter.
- Keras save/load round trip: exact synthetic logits.
- Fresh candidate/CLI import with `PYTHONPATH` removed: PASS; exact parameter
  identity and train/validation/output-only CLI confirmed.
- PyTorch runtime isolation: PASS (`torch` absent after fresh candidate import).
- Frozen Generation-1 checksum verification: `PASS checked=267 failures=0`.
- v1.0 candidate diff from exact parent: empty.
- Generation-1 and Step-13 source diff from exact parent: empty.
- `git diff --check`: recorded after staging all new files in the PR.

Only synthetic images and a one-row temporary CSV were used by tests. No real
FER train/validation data was loaded, no Kaggle job was launched, and no final
split was accessed.
