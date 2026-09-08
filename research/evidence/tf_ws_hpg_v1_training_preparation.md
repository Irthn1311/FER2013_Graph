# WS-HPG v1.0 training preparation evidence

Status: `WS_HPG_V1_TRAINING_PREPARATION_ONLY`

## Provenance and scope

- Issue: #67.
- Exact implementation parent: `157c8a87f84e0aa53b4432393b99a3a560c0974e`.
- Isolated implementation commit: `7d13f69d68be55e0be45c02b3e6d14915edaa19f`.
- Branch: `codex/issue-67-ws-hpg-v1-training-prep`.
- Accepted `model.py`: `177a782cd8d5c2178303c44d120dcdd22b0a2108a0b720c5091a19f3d7cbffe3`.
- Accepted `support.py`: `b6ed2ddd20a4e82824208929709ff2d6bcb1c5557144d778ed0157fd4768aeee`.
- Measured accepted identity: 707,213 parameters, 118 trainable variable
  objects, 138 total Keras variables.

This change adds only an isolated future training/data/gate adapter. The
accepted raw-3x3-patch WS architecture remains byte-identical and its 64/16
nodes remain unnamed coarsened structural/compositional units. No CNN, ROI,
attention, anatomy, semantic-region discovery, or support after Pool 1 exists.

## Data and detector-output cache contract

Only explicit train and validation CSV/cache paths are accepted. Lexical guards
reject `test`, `testing`, `test_split`, `test-split`, and test-like CSV basenames
before I/O. Cache split names are restricted to `train`, `val`, or `validation`;
indexed paths are constructed directly and no glob, walk, or split discovery is
used.

The minimal NPZ reader requests only:

`sample_index`, `label`, `image_48`, `detected`, `landmark_xy_48`.

It never requests `face_mask`, `part_soft_masks`, `micro_anchor_maps`,
`distance_maps`, `valid_part_mask`, `valid_anchor_mask`, or `quality_score`, and
does not import `PixelPriorDataset`. Sample index, label, and the exact clean
48x48 image must match before support is accepted. Detector coordinates must be
finite `[N,2]` FER pixel coordinates in `[0,47]`; they are immediately reduced
through the accepted extent-only support helper. Missing, invalid, out-of-range,
or degenerate output yields all-ones support. A golden test proves cached and
direct support construction are bit-identical from the same coordinate array.

## Frozen augmentation and training configuration

Sampling is stateless from seed 42 plus sample index/salt. The one affine vector
contains flip p=0.5, rotation uniform [-10,+10] degrees, and x/y translation
uniform [-4,+4] pixels, and is passed identically to image and support. Support
uses bilinear interpolation and one-valued fill. Contrast [0.85,1.15], normalized
brightness [-0.10,+0.10], and random erase p=0.25 / area [0.02,0.10] / aspect
[0.5,2.0] affect the image only. No MediaPipe rerun or support dropout exists.

The one registered lifecycle locks seed 42, AdamW, LR `3e-4`, weight decay
`5e-4`, global clip norm 1.0, batch 64, max 100 epochs, five-epoch linear
warmup, cosine decay to `1e-6`, training label smoothing 0.05, validation each
epoch, earliest strict maximum validation-accuracy checkpoint, and val-loss
early stopping with patience 15/min_delta 0.0. Mixed precision, XLA,
MirroredStrategy, sweeps and alternate protocol values are explicitly false or
absent. Evaluation uses hard CE.

After completion the selected `.keras` is loaded with `compile=False`. The same
loaded object is evaluated exactly once on clean train with normal support,
exactly once on validation with normal support, and exactly once on validation
with all-ones support. A before/after all-variable SHA prevents diagnostic
weight drift. Accuracy/macro-F1/loss, train-validation gaps, support dependency
deltas, and descriptive coverage/failure statistics are persisted. Diagnostics
cannot affect training, selection, early stopping, or schedule.

The exact six-level Issue #67 decision precedence and every inclusive/exclusive
threshold are locked by boundary tests. Only the first two outcomes are marked
replacement candidates; no automatic paper replacement is implemented.

## Source identities

| File | SHA-256 |
|---|---|
| `__init__.py` | `b211b99534a7c3e9ff57ea5ac0b2a7f059307cb56d49f9f1e1214189e6402f43` |
| `augmentation.py` | `766182e07f3fa0b53b49f0b7bb82401bcd8fd1ec62f9fb9fa94f2a68c9cb0244` |
| `data.py` | `a8fe9757ab3faa6895cac5e09c5484fc86fd7ed1a961a9ddc90285b42810e524` |
| `train_validation_only.py` | `4b7736f7b4e5a04ac6ecd791792c0ec96cb233a11d636df155c849dc9b9d44e2` |
| focused test | `7a5f394e9413bc5814403d40fc7c13c1ced39525d8e6ca0d401b453b041f4737` |

## Verification

- Focused Issue #67: **31 passed, 14 warnings**.
- Combined accepted CF-HPG v1.0-v1.3, RA-HPG, WS-HPG architecture, and
  Issue #67 adapter: **366 passed, 14 warnings**.
- `tf.function` synthetic training loss and all 118 gradients: finite.
- One synthetic optimizer step changed weights.
- `.keras` compile-false round trip: fixed-input logits identical.
- Lifecycle synthetic mock: selected checkpoint loaded with `compile=False`;
  clean/normal/ones evaluation inventory exact; same selected object/weights.
- Frozen checksum verifier: **PASS checked=267 failures=0**.
- Git diff against the parent under accepted WS/LAP/CF/RA packages: empty.
- `git diff --check`: PASS.

Synthetic TensorFlow 2.18.1 CPU benchmark, one post-warmup repeat (no accelerator
visible, so peak accelerator memory is unavailable):

| Batch | Forward s | Forward+backward s | Forward ex/s | F+B ex/s | Fine edges |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.0282202 | 0.0965717 | 283.48 | 82.84 | 14,880 |
| 16 | 0.0420342 | 0.1437407 | 380.64 | 111.31 | 29,760 |
| 32 | 0.0633847 | 0.2834271 | 504.85 | 112.90 | 59,520 |
| 64 | 0.1346123 | 1.0284025 | 475.44 | 62.23 | 119,040 |

Batch 64 completed; fine-edge growth remains exactly `1,860 * batch`.

No FER2013 full train/validation execution, Kaggle execution, validation tuning,
test split access, or scientific outcome occurred.
