# MPG-FER v2 — Final Code Audit Before Kaggle

Date: 2026-09-21  
Repository branch: `research/mpg-fer-v2-issue92`  
Issue: [#92](https://github.com/Irthn1311/FER2013_Graph/issues/92)  
Scope: aggressive code/protocol audit, defect fixes, deterministic tests, notebook regeneration, bounded GPU audit, and resume serialization benchmark. No full FER2013 or Kaggle training was run.

## 1. Executive verdict

The audited implementation is ready for a source-locked Kaggle T4 segmented training attempt. Architecture identity is preserved, parameter count remains 2,238,609, all 35 tests pass, the embedded notebook is synchronized, multi-worker augmentation/order continuation passes, the strengthened actual-v2 stochastic trajectory resumes correctly, and the worst-case local batch-16 AMP audit fits a 4 GiB GPU.

This verdict establishes implementation and lifecycle readiness. It does not establish v2 scientific performance, Kaggle T4 runtime, or motif improvement.

## 2. Architecture alignment

The locked chain remains:

```text
48x48 grayscale
-> 2304 explicit pixel nodes
-> 32D continuous relational features
-> explicit 8-neighbor Pixel Graph
-> Pixel GNN x4, D=96, 4 heads
-> soft assignment to 48 learned prototypes
-> aligned 8x8 / 12x12 / 16x16 occurrence composition
-> exactly 49 fused motif nodes
-> complete learned-center geometry graph
-> Motif Transformer x5, D=192, 6 heads
-> 128D pixel + 384D motif readout
-> 512D fusion -> 256D -> 7 classes
```

No convolutional backbone, ViT/Swin, KMeans, hard motif ID, K128/K512 CRS, Q_M21, TopK node removal, or connected-component extraction exists in v2. Multi-scale paths share TYPE and occurrence projection machinery; only three small saliency heads and a scale gate are scale-specific.

Measured trainable parameters:

```text
v1: 2,219,788
v2: 2,238,609
increase: 18,821 (0.848%)
```

## 3. Motif assignment / MI audit

Actual implementation computes cosine similarity between learned assignment-query pixel embeddings and learned prototype-key embeddings, divides by bounded temperature, and applies softmax. Assignment shape remains `[B,2304,48]`; there is no argmax.

The implemented objective is:

```text
H_local = mean_i H(A_i) / log(M)
H_global = H(mean_i A_i) / log(M)
L_MI = H_local - beta * H_global
beta = 1.0
lambda_mi = 0.05
```

The sign is correct: minimizing rewards locally sharp assignment and globally diverse use. Synthetic results:

| Case | H_local | H_global | L_MI |
|---|---:|---:|---:|
| Uniform local assignment | 1 | 1 | 0 |
| All nodes collapsed to one prototype | 0 | 0 | 0 |
| Locally sharp, globally balanced | 0 | 1 | -1 |

The balanced-sharp case is strictly preferred. The old utilization KL is absent; prototype diversity remains separately weighted by 0.01.

The bounded GPU step measured the following relative magnitudes:

```text
CE_final:             1.96555877
weighted CE_motif:    0.39308593
weighted CE_pixel:    0.09914350
weighted MI:         -0.00007712
weighted diversity:   0.00005561
weighted consistency: 0.00076597
total:                2.45853281
```

MI does not dominate FER supervision. No coefficient was changed.

Temperature is implemented as `0.15 + 1.35*sigmoid(raw_tau)`, initializes to `0.70000005`, is bounded for arbitrary raw values, receives finite nonzero gradient, lives in model/optimizer state, and is included in EMA. EMA inference therefore uses the EMA copy of learned temperature, as intended.

## 4. Multi-scale motif audit

The original 12x12 starts `0,6,12,18,24,30,36` define half-pixel centers `5.5,11.5,...,41.5`. Explicit logical-start tests verify, before reflection:

```text
8x8  start = s + 2
12x12 start = s
16x16 start = s - 2
```

for top-left, center, and bottom-right anchors. Support tensors remain `[49,64]`, `[49,144]`, and `[49,256]`.

Coarse reflection is reflection without edge repetition, not negative indexing, wraparound, or clamp. Audited border sequences are:

```text
top/left:     2,1,0,1,2,...,13
bottom/right: 34,35,...,47,46,45
```

Reflection changes sampled indices only; conceptual anchor centers do not move.

Scale weights are positive per-image/per-occurrence softmax values with shape `[B,49,3]` and sum exactly to one. The same weights fuse representations and learned `(cx,cy)` centers. Motif graph geometry is built from those fused centers without detach. A dedicated test detaches motif content and proves a loss passing only through motif-graph geometry produces nonzero gradients in the scale gate and all three scale-local occurrence saliency heads.

## 5. Regularization / EMA / consistency audit

Dropout defaults remain 0.15 pixel, 0.15 motif, and 0.30 classifier. Residual DropPath schedules are:

```text
Pixel: 0.000000, 0.016667, 0.033333, 0.050000
Motif: 0.000000, 0.025000, 0.050000, 0.075000, 0.100000
```

DropPath masks are per sample/residual branch, retained paths are divided by keep probability, graph nodes are not removed, and evaluation is deterministic.

EMA updates only after a successful optimizer step. A four-microbatch test with accumulation 2 produced exactly two optimizer steps and two EMA updates. EMA covers all model state, including `raw_tau`, prototypes, Q/K projections, scale saliency/gates, and floating buffers; non-floating buffers are copied rather than nonsensically averaged. EMA update count survives resume.

Validation selection remains lexicographic EMA Public flip-TTA accuracy, higher EMA TTA macro-F1, then lower EMA TTA loss. `best_val_acc.pt` now explicitly records:

```text
checkpoint_type = EMA_INFERENCE_ONLY
weights_type = EMA
```

and its inference state is tested equal to the selected EMA model.

Consistency is a deterministic decision per optimizer accumulation group using `(seed, epoch, group_index)` with probability 0.20. It applies `horizontal_flip` to the current augmented image before relational feature extraction, compares both final distributions with differentiable symmetric JS divergence, and transforms no labels or intermediate coordinates. The same decision sequence is reproduced across resume.

## 6. Training scheduler audit

The 120-epoch schedule is five-epoch linear warmup followed by cosine decay to `1e-6`. Representative values measured from the implementation:

| Epoch | LR |
|---:|---:|
| 1 | 0.000060000000 |
| 2 | 0.000120000000 |
| 3 | 0.000180000000 |
| 4 | 0.000240000000 |
| 5 | 0.000300000000 |
| 6 | 0.000299944219 |
| 40 | 0.000236713708 |
| 80 | 0.000081720277 |
| 100 | 0.000022764299 |
| 120 | 0.000001000000 |

Restoring scheduler state at epoch 57 yields exactly the same epoch 58–120 LR sequence. Early stopping remains `min_epochs=50`, `patience=20`; comparator, best epoch, best metrics, and non-improvement counter are included in the resume bundle and tested across continuation.

## 7. Exact resume audit

Schema v2 contains:

```text
run_id, source_hash, scientific config/hash, schema
completed_epoch, next_epoch, global_optimizer_step
online model, EMA model/update count
AdamW, scheduler, GradScaler
best comparator, best epoch/metrics, patience
history
Python, NumPy, torch CPU, torch CUDA RNG
DataLoader generator
epoch sampler state
sample-local augmentation state
consistency scheduling state
```

The strengthened trajectory test uses the actual `SpatialMotifComposer`, learnable bounded temperature, MI objective, DropPath, deterministic consistency schedule, gradient accumulation, AdamW, EMA, scheduler, comparator, patience, sample order, and history. Continuous four epochs and two epochs + atomic save/load + two epochs produce:

- online model: bit-exact;
- EMA model and update count: bit-exact;
- temperature: bit-exact;
- scheduler/LR, global step, comparator, best epoch, patience, history, order, and consistency decisions: exact;
- AdamW integer state: exact; floating moments: equal within `atol=1e-9`, `rtol=1e-7` after process-style serialize/reload.

The smaller baseline resume test remains bit-exact for model, EMA, and complete AdamW state. A real CUDA GradScaler step/state round-trip also passes locally. Full CUDA training can still contain backend-level nondeterminism; the contract is greatest practical determinism, not an unsupported bitwise guarantee across arbitrary CUDA versions/hardware.

Runtime-only config fields excluded from the scientific hash are limited to workers, output/resume paths, segment number, soft-limit/safety-margin controls, and optional run-ID config field. Run identity is separately checked when supplied. Changing loss coefficients, motif scales, model dimensions, schedule, seed, augmentation semantics, EMA decay, or batch/accumulation changes the hash and fails closed.

## 8. DataLoader RNG / augmentation resume audit

This audit found and removed a remaining dependency on sequential sampler-generator history. Official sample permutation is now a pure function of `(base_seed, epoch)` through `EpochRandomSampler`. Augmentation is a pure function of `(base_seed, epoch, sample_index)` and uses a local `torch.Generator`; `__getitem__` never mutates or consumes main/worker global RNG.

The dataset and sampler expose `set_epoch`, `state_dict`, and compatibility-checked `load_state_dict`. Training sets both before every epoch. `persistent_workers` is false, but correctness no longer depends on worker lifetime or scheduling.

A real multi-process test with `num_workers=2` compared epochs 1–4 continuously against epochs 1–2, recreated dataset/sampler/workers/generator, restored state, then epochs 3–4. Both sample-index order and SHA-256 of every augmented image were exactly equal. A separate test perturbed global torch/NumPy RNG between repeated sample reads and obtained identical augmentation.

## 9. Kaggle segmentation audit

The guard is evaluated after a completed epoch and now checks:

```text
elapsed + conservative next-epoch estimate + 15-minute safety margin
>= 10.5-hour soft limit
```

On stop it already has an atomic full resume bundle and history, rewrites resume metadata with `NEEDS_RESUME`, atomically writes/fsyncs the segment manifest, returns normally, skips PrivateTest, creates the working ZIP, and lets the notebook finish successfully. It does not intentionally crash or assume `/kaggle/working` survives.

`resume_latest.pt`, its sidecar, history, segment manifest, and best EMA checkpoint use temp-file + flush/fsync + `os.replace` semantics. `.tmp` files are ignored by resolver patterns. Every 10 epochs an immutable snapshot and independent SHA sidecar are written; at least two are retained. If latest SHA fails, the resolver reports the newest identity-compatible, independently hash-valid fallback but refuses to load it automatically.

Concrete continuation is documented in `README.md`: download/persist Segment N outputs, upload unzipped resume files to a private dataset named `mpg-fer-v2-resume-<run-id>-segNN`, attach it to Segment N+1, keep reviewed sources unchanged, set `RESUME_MODE=auto`, and increment segment number. Auto mode scans only paths with an `mpg-fer-v2-resume*` component, requires one `resume_latest.pt`, verifies sidecar SHA, then validates schema/run/source/scientific config. It cannot discover v1, `best_val_acc.pt`, `last.pt`, or unrelated files.

Measured full-state CPU serialization benchmark:

```text
resume_latest.pt:       37,381,625 bytes (35.6499 MiB)
atomic save + SHA-256:  0.2295 s
load + SHA-256:         0.2731 s
```

This local filesystem measurement shows serialization is small relative to the 15-minute margin; Kaggle storage latency remains unmeasured.

## 10. Tests executed

Final command:

```text
C:\Users\ADMIN\anaconda3\envs\fer-graph\python.exe -m pytest -q research\mpg_fer_v2\tests
```

Result:

```text
35 passed in 56.38s
```

Coverage includes all requested MI cases, temperature bounds/gradient/EMA, logical support alignment and border reflection, scale simplex and geometry-only gradient path, DropPath schedules and behavior, linear-before-gather K/V/logit/message/output equivalence, EMA accumulation/checkpoint/resume, real CUDA GradScaler state, full resume/fallback atomicity, multi-worker augmentation/order continuation, actual-v2 stochastic trajectory continuation, LR continuation, config compatibility, source-notebook synchronization, and restricted resume routing.

The regenerated notebook contains six code cells; all compile, and its embedded source manifest matches reviewed module bytes.

## 11. Bounded GPU audit

Measured on available local hardware, not T4:

```text
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GiB
batch: 16
AMP: enabled
all three motif scales: enabled
DropPath training mode: enabled
consistency second forward: forced on (worst case)
backward + optimizer step: successful
EMA updates: 1
peak allocated: 2,821.9463 MiB
peak reserved:  3,008.0000 MiB
```

Gradient audits pass for `raw_tau`, assignment query, prototypes, scale gate, 8x8 scale saliency, and classifier. Initial diagnostics were:

```text
tau: 0.70000005
H_local normalized:  0.99748898
H_global normalized: 0.99903131
L_MI: -0.00154233
effective motifs: 47.5357
mean alphas (8,12,16): (0.32445, 0.34536, 0.33019)
```

This is bounded synthetic evidence. No FER2013 data and no full experiment were used. T4 memory/runtime remain unmeasured.

## 12. Problems found and fixes

1. **Sampler continuation depended on sequential generator state.** It was restorable but weaker than necessary and lacked multi-worker proof. Replaced with epoch-derived deterministic sampler; added explicit sampler/augmentation state and a two-worker exact continuation test.
2. **Wall-clock guard had no serialization/completion margin.** Added configurable 15-minute safety margin and a boundary test.
3. **Immutable snapshots lacked independent SHA metadata and fallback reporting.** Added atomic `.json` sidecars, identity-aware validation, retention of two snapshots, and explicit no-auto-fallback reporting.
4. **Some segment/history/best-checkpoint writes were atomic without full fsync discipline or were direct writes.** Converted critical history, manifest, and EMA checkpoint writes to temp + flush/fsync + replace.
5. **Runtime-only identity/segment controls could participate in config hash.** Tightened the explicit runtime-safe allowlist while keeping all scientific fields fail-closed.
6. **Best checkpoint metadata implied EMA but lacked a direct field.** Added `weights_type=EMA` and a state-equality test.
7. **Round 1 resume evidence did not exercise actual motif/MI/DropPath components or worker processes.** Added actual-v2 stochastic resume, real multi-worker augmentation/order, real CUDA GradScaler, EMA accumulation, and fallback tests.
8. **Audit coverage gaps** existed for MI collapse ordering, raw-temperature gradient/EMA, exact logical starts/reflection, geometry-only gradient route, intermediate linear-before-gather equivalence, and representative LR values. Added mandatory tests; no scientific coefficients or architecture were changed.

## 13. Remaining risks

- Kaggle T4 execution, memory, epoch runtime, output persistence, and cross-kernel dataset attachment have not been measured.
- Full 120-epoch scientific performance, generalization, motif specialization, best epoch, and early-stop behavior are UNKNOWN.
- Local multi-worker testing ran on Windows with two workers; the design is worker-history-independent, but the exact Kaggle Linux environment remains unexecuted.
- CUDA kernels may not be bitwise deterministic across different GPU architectures, driver/library versions, or kernels despite complete state restoration and deterministic data/control decisions.
- Resume benchmark used local CPU tensors/filesystem; Kaggle upload and dataset-version latency are outside this code audit.

## 14. Source hash

Reviewed source-tree SHA-256 after all fixes:

```text
f9a06cd4f7482c6a6e37d58844c1a30022602e9fc825eff73240f71740b0af1c
```

Trainable parameter count remains exactly `2,238,609`.

## 15. FINAL STATUS

`READY_FOR_KAGGLE_V2`
