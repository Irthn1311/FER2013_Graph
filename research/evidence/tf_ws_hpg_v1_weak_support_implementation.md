# WS-HPG v1.0 implementation evidence

Status: `WS_HPG_V1_WEAK_SUPPORT_IMPLEMENTATION_ONLY`

## Provenance and boundary

- Contract: GitHub Issue #65, registered before implementation.
- Exact parent: `6d89d17b2d3c39b7bf57084f9de4b707a92eb73e`.
- Verified implementation-source commit: `0d4eeaf3ed2801342f27c8298fb4ddbb7db5ee6a`.
- Branch: `codex/gen3-ws-hpg-v1-weak-support`.
- TensorFlow used for verification: 2.18.1 (CPU; no accelerator was visible).
- No FER2013 train or validation data was opened, no Kaggle job was launched, no
  test split was accessed, and no scientific execution or tuning occurred.

The implementation tests only the architecture hypothesis that a weak external
facial-support prior can attenuate early nuisance/background communication while
learned sparse relations and hierarchy discover expression structure bottom-up.
It does not provide a scientific outcome for that hypothesis.

## Frozen architecture

The tokenizer is `3x3 raw patch -> Dense(9,64) -> GELU -> Dense(64,64)`, added
to a generic normalized-grid `Dense(2,64)` projection, followed by
`LayerNorm -> GELU -> Dropout(0.10)`. Shapes are:

`[B,48,48,1] -> [B,256,9] -> [B,256,64]`.

Every sparse edge `j -> i` uses only generic current features and grid geometry:

- `z = LayerNorm(h)`;
- gate: `[abs(z_j-z_i), dx, dy, distance] -> Dense(D/2) -> GELU -> Dense(1) -> sigmoid`;
- content: `[z_j, z_j-z_i] -> Dense(D) -> GELU`;
- fine-only support: `c_ij = 0.25 + 0.75 sqrt(s_i s_j)`;
- message: `m_ij = q_ij c_ij v_ij` (later stages use `c_ij = 1`);
- sparse incoming PNA-lite: mean, max and
  `sqrt(max(mean(m^2)-mean(m)^2,0)+1e-6)`;
- mixer: `[z,mean,max,std] -> Dense(D)` plus residual Dropout(0.10);
- FFN: pre-LN, `Dense(D,2D) -> GELU -> Dropout(0.10) -> Dense(2D,D)
  -> Dropout(0.10)` plus residual.

The hierarchy is exactly:

- fine: 16x16, 256 nodes, D=64, fixed directed spatial-8 edge list, two blocks;
- Pool 1: deterministic 2x2 mean+max, Dense(128,96), LN, GELU, yielding `[B,64,96]`;
- mid: 8x8, 64 nodes, D=96, spatial-8 union current-feature cosine top-4,
  deduplicated and rebuilt before each of two blocks;
- Pool 2: deterministic 2x2 mean+max, Dense(192,128), LN, GELU, yielding `[B,16,128]`;
- coarse: 4x4, 16 nodes, D=128, the same deduplicated hybrid graph rebuilt
  before each of two blocks;
- readout: global mean+max `[B,256]`, LN, Dense(256,128), GELU,
  Dropout(0.20), Dense(128,7) logits (no in-model softmax).

Fine messages are gathered from a flat sparse edge list and aggregated by
unsorted segment operations. There is no dense `[B,256,256,D]` message path.
The fixed fine graph has 1,860 directed edges per example and is reused by both
fine blocks. Dynamic graphs are constructed only at 64 and 16 nodes.

## Support construction and hard boundary

The detector is called once on the clean image. Its anonymous points are reduced
immediately to global `xmin/xmax/ymin/ymax`; individual identity and coordinates
are not returned or retained. The bounds expand by 10 percent on every side.
For elliptical radius `r`, support is 1 inside the ellipse and
`exp(-0.5*((r-1)/0.25)^2)` outside, clamped to `[0,1]`.

Invalid, missing, non-finite, or degenerate detector output yields an all-ones
field (no prior). Image and support receive the exact same flip/rotation/
translation projective vector; support uses bilinear interpolation and an
all-ones fill so detector failure remains no-prior. Brightness, contrast, and
erasing affect only the image. Fine scores are 3x3 support means.

Only `FineWeakSupportRelationBlock.call` accepts support. Immediately after
Pool 1, only the pooled node tensor continues. `SupportFreeRelationBlock.call`
has no support argument, and the mid/coarse graph builders receive node features
only. A source-structure regression verifies that no support name or value exists
in the post-Pool-1 forward segment. The normal/all-ones inference override changes
neither architecture nor weights.

## Exact identity and independent arithmetic

All Dense layers use bias and all LayerNorm layers have trainable gamma/beta.

| Component | Arithmetic | Parameters |
|---|---:|---:|
| Tokenizer and position | `(9*64+64) + (64*64+64) + (2*64+64) + 2*64` | 5,120 |
| One D=64 relation block | `10.5D^2 + 11.5D + 1` | 43,745 |
| Two fine blocks | `2 * 43,745` | 87,490 |
| Pool 1 | `128*96+96 + 2*96` | 12,576 |
| Two D=96 blocks | `2 * 97,873` | 195,746 |
| Pool 2 | `192*128+128 + 2*128` | 24,960 |
| Two D=128 blocks | `2 * 173,505` | 347,010 |
| Readout | `2*256 + 256*128+128 + 128*7+7` | 34,311 |
| **Total** | | **707,213** |

Measured identity: 707,213 parameters, 118 trainable variable objects, and 138
total Keras variables. The extra 20 non-trainable variables are Keras Dropout
SeedGenerator states.

Source SHA-256:

| File | SHA-256 |
|---|---|
| `__init__.py` | `1d82e7ed8e069e5a01db5033ed73cc44b4edd2d9c3732545a86a9f55bb53bba1` |
| `graph.py` | `8289e2548a13f7948b614d0a09dbd52869f43b09cc2cf53e45b0ed5c2649b757` |
| `layers.py` | `f9418516f26f278e48ab32374f15ca182627fa9293ab69bb7c1d47bd3d851a77` |
| `model.py` | `177a782cd8d5c2178303c44d120dcdd22b0a2108a0b720c5091a19f3d7cbffe3` |
| `support.py` | `39738a8e6a6a324c5387e4f8ce26f7d9abf0d137b6ba8d41fa61366635babc3d` |
| `synthetic_benchmark.py` | `499eda9f99ece928321f4fb8d4cb051264415df447cacf4bf9bdc0eb5f65483f` |
| focused test | `2780f2141e2a3ce1a3e0eb423a10263f76c487c2ace4451e1f4816b7bbbf7101` |

## Golden and regression verification

Focused command used the `lap-gnn-tf` Python environment:

`python -m pytest -q tests/test_tf_ws_hpg_v1_weak_support.py`

Result: **25 passed**. The assertions cover all registered golden requirements:
patch/token shapes; fine sparse topology/no self/no kNN; extent-only support and
failure fallback; range/floor/no deletion; absence from node/gate inputs; generic
geometry and exact message inputs; PNA mean/max/std/degree-1; sparse-vs-dense
numerical equivalence; absence of dense fine messages; both pool shapes and Pool-1
golden statistics; post-Pool-1 support inaccessibility; mid/coarse union,
deduplication and current-representation construction; exactly two rebuilds at
each dynamic level; logits and identity; finite `tf.function` forward/loss/all
gradients; parameter change after an optimizer step; bit-identical `.keras`
round trip; geometric parameter coupling; photometric/erase isolation; all-ones
gate identity; absence of data/training lifecycle and test paths; PyTorch source
isolation; and fresh absolute CLI import with `PYTHONPATH` removed from outside
the repository.

Combined accepted Generation-2 plus WS-HPG command covered CF-HPG v1.0-v1.3,
RA-HPG v1.0, and this suite on the final source: **332 passed, 14 warnings**.

Frozen package checksum verifier:

`python -B tools/verify_checksums.py` from the frozen package: **PASS,
checked=267, failures=0**. Git diff against the exact parent contains zero changes
under the frozen LAP package, accepted CF/RA candidates, Generation-1, or Step-13.

## Synthetic runtime benchmark

Synthetic inputs only, TensorFlow 2.18.1 CPU, one measured repeat after traced
warm-up. Accelerator peak memory is unavailable because no accelerator was
visible. Batch 64 completed without memory failure.

| Batch | Forward s | Forward+backward s | Forward ex/s | F+B ex/s | Fine edges | Mid edges block 1/2 | Coarse edges block 1/2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.0220294 | 0.0753824 | 363.15 | 106.13 | 14,880 | 3,739 / 3,801 | 735 / 755 |
| 16 | 0.0337171 | 0.1258001 | 474.54 | 127.19 | 29,760 | 7,468 / 7,556 | 1,466 / 1,504 |
| 32 | 0.0582498 | 0.2785493 | 549.36 | 114.88 | 59,520 | 14,926 / 15,107 | 2,940 / 3,001 |
| 64 | 0.1186553 | 0.5493199 | 539.38 | 116.51 | 119,040 | 29,925 / 30,256 | 5,865 / 6,015 |

The exact linear fine-edge relation (`1,860 * batch`) and successful batch-64
run provide implementation evidence against an accidental quadratic fine-message
allocation. Timing is local synthetic CPU evidence, not a scientific result or a
Kaggle performance claim.
