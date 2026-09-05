# CF-HPG v1.0 implementation evidence

Status: `CF_HPG_V1_IMPLEMENTATION_ONLY`

## Scope and provenance

- Issue: `#42 [Gen2 CF-HPG v1.0] Build minimal convolution-free hierarchical patch graph for FER2013`
- Exact parent: `56a22a3200f1e0dc0100093d43f0db296c11487d`
- Candidate: `research/candidates/tf_cf_hpg/`
- Dataset signature reserved for the future run: `fer2013_train28709_val3589_test3589`
- This change is additive and isolated. Generation-1, the reviewed Step-13 probe, and the Step-13 execution adapter are unchanged.

## Registered architecture

Raw `[B,48,48,1]` grayscale pixels are scaled with `x / 127.5 - 1.0`, rearranged without convolution into row-major `[B,64,36]` patches, and fused additively with generic `(x,y)` grid-center projections. The fine stage uses two width-96 pre-normalized Max-Relative blocks over one spatial-8-neighbor union cosine-top-4 graph. Both blocks receive the same graph tensor within a forward pass.

The parameter-free hierarchy reshapes the fine nodes to `8x8`, arithmetic-means each non-overlapping `2x2` cell, and produces 16 nodes. A `96 -> 128` projection precedes two width-128 Max-Relative blocks over one newly constructed `4x4` hybrid graph, again reused by both blocks. Readout concatenates global mean and maximum, then applies LayerNorm, `256 -> 128` GELU, dropout 0.20, and `128 -> 7` logits. There is no softmax or attention operation.

The graph is stored as a boolean adjacency mask, but its active edge set is exactly the sparse registered neighbor union. Boolean union provides exact deduplication. No graph is rebuilt between the two blocks at a scale.

## Test-split path isolation

Every CSV request passes through the candidate data-boundary guard before `is_file`, `open`, parsing, hashing, or loading. The guard rejects the exact basename `test.csv`, the precise `test_*.csv` and `test-*.csv` patterns, and explicit directory components `test`, `testing`, `test_split`, and `test-split`, case-insensitively. It does not use substring matching: `train.csv`, `val.csv`, `validation.csv`, `tiny.csv`, and `contest_data.csv` remain valid. The production CLI lexically preflights both supplied CSV paths before creating its output directory or loading either source; `load_fer_csv()` independently applies the same guard so non-CLI callers cannot bypass it.

## Model inventory

- Total parameters: `413447`
- Trainable parameters: `413447`
- Non-trainable parameters: `0`
- Keras variables: `74` (`64` trainable parameter variables plus `10` zero-parameter RNG-state variables owned by dropout layers)
- Hard budget: PASS (`413447 <= 600000`)
- Preferred budget: PASS (`413447 <= 500000`)

| Layer/group | Parameters |
|---|---:|
| Raw patch projection `36 -> 96` | 3,552 |
| Position projection `2 -> 96` | 288 |
| Patch LayerNorm | 192 |
| Fine Max-Relative block 1 | 65,376 |
| Fine Max-Relative block 2 | 65,376 |
| Coarse projection `96 -> 128` | 12,416 |
| Coarse LayerNorm | 256 |
| Coarse Max-Relative block 1 | 115,840 |
| Coarse Max-Relative block 2 | 115,840 |
| Readout LayerNorm | 512 |
| Readout Dense `256 -> 128` | 32,896 |
| Classifier `128 -> 7` | 903 |
| Total | 413,447 |

## Train-only augmentation contract

Augmentation acts on the image before patchification. Each random draw is statelessly derived from `(registered seed 42, enumerated training sample index, operation salt)`: flip `0`, rotation `1`, translation `2`, contrast `3`, brightness `4`, erasing decision `5`, area `6`, aspect `7`, top `8`, and left `9`. Geometric rotation within `+/-10` degrees and translation within `+/-4` pixels are applied together with bilinear interpolation and image-safe `REFLECT` fill. Contrast is `[0.85,1.15]`; normalized brightness delta is `[-0.10,+0.10]`. Random erasing uses probability `0.25`, area `[0.02,0.10]`, aspect `[0.5,2.0]`, and normalized fill value `0`.

Validation and clean-train evaluation call the non-augmentation dataset path. Horizontal flip changes only raw image pixels; there is no semantic identifier or label remapping.

## Frozen future training contract

- Seed `42`; train `28709`; validation `3589`; batch `64`; at most `100` epochs.
- AdamW, learning rate `3e-4`, weight decay `5e-4`, global-norm clip `1.0`.
- Linear warmup for 5 epochs, then cosine decay to `1e-6`.
- Categorical cross-entropy from logits with sparse labels converted only inside the loss to 7-way one-hot and label smoothing `0.05`.
- Checkpoint is the earliest strict maximum `val_accuracy`.
- Early stopping monitors `val_loss`, patience `15`, minimum delta `0`, without outcome-dependent adjustment.
- The CLI accepts exactly `--train-csv`, `--val-csv`, and `--output-root`. It has no alternate split or final-evaluation input.

## Source SHA-256

- `__init__.py`: `427150bf919257d221f0e10d7fbbc154c8cd6e1004150edfc193867bd849a7c2`
- `graph.py`: `fd425766f8db4d53f87edca0bebb88ff476c43938fff6a507c0687a0efb2d8ec`
- `model.py`: `74c29b114b37ab8b48e45eed187978213fdecb858554f4d37468042a3db4050d`
- `data.py`: `e8b586aafc6fbad74d913cb08d25a61432c0e55173a60621597f4c5a8d091776`
- `train_validation_only.py`: `045eb7762f8dda7a057201f7c4b599d76778ba1aa18b6bac3c797e8c620b4180`
- `tests/test_tf_cf_hpg_v1.py`: `0e00b2ea819d23582d734f3438896721cb5b9c7a4ec3cae722997e15f21e1e25`

These hashes are the implementation files before adding this evidence document. The evidence-file hash is reported separately in the PR.

## Synthetic verification

- Focused Issue #42 suite: `53 passed`.
- Fourteen forbidden-path cases cover all registered basename/directory forms with both unrestricted and exact-`3589` sample-count arguments; tracked `Path.open` calls remain empty in every case.
- The synthetic CLI regression rejects `--val-csv test.csv` before either CSV loader is called and before the output directory is created.
- Five allowed-name cases prove exact matching does not reject ordinary train/validation/tiny sources or `contest_data.csv`.
- Candidate plus reviewed Step-7/Step-6 regression suites: `54 passed`.
- The synthetic `tf.function` train step produced finite logits, finite smoothed loss, finite gradients for every trainable variable, and changed model parameters through AdamW.
- Dynamic learned-relation graph construction executed inside that traced training step.
- Synthetic Keras save/load round-trip reproduced logits exactly.
- Fresh candidate/CLI import with `PYTHONPATH` removed and PyTorch isolation: PASS.
- Frozen Generation-1 checksums: `PASS checked=267 failures=0`.
- Frozen Generation-1 diff and reviewed Step-13 source diff from the exact parent: empty.
- `git diff --check`: PASS.
- No full FER2013 training or validation was executed. No final split was read or inspected.

## Future Kaggle command — documentation only, not executed

```bash
python -m research.candidates.tf_cf_hpg.train_validation_only \
  --train-csv /kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/train.csv \
  --val-csv /kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/val.csv \
  --output-root /kaggle/working/tf_cf_hpg_v1_seed42
```

This implementation does not establish FER performance, generalization, or any scientific decision label. A future full seed-42 run requires separate authorization after review.
