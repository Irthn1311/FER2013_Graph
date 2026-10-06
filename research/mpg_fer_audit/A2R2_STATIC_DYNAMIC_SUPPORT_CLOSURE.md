# MPG-FER A2-R2 ? Static-vs-Dynamic Sparse Support Final Closure Report

**Audit Date**: September 2026  
**Auditor**: Opencode CLI Diagnostic Agent  
**Repository Working Directory**: `D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5`  
**Current Git Branch**: `research/mpg-fer-v2-1-issue93`  
**Current Git HEAD**: `4967cc5dac3495be2300210215f72422f6f97aa4`  
**Scope**: Finalize the exact empirical closure distinguishing fixed complete support, static random sparse support, static Train-derived frequency support, and sample-dynamic sparse support across 20 random seeds, full-train K verification, and cross-version replication on v2.

---

## 1. Correct Support Semantics Nomenclature

Per strict audit conventions, the experimental conditions are unambiguously defined:

1. **`FULL`**:
   `FIXED_COMPLETE_SUPPORT + DYNAMIC_WEIGHTS` (Degree 48 complete graph; dynamic softmax attention weights).
2. **`DYNAMIC_TOPK`**:
   `SAMPLE_DYNAMIC_SPARSE_SUPPORT + DYNAMIC_WEIGHTS` (Per-sample top-K edge selection from dynamic scores; dynamic softmax over selected edges).
3. **`STATIC_RANDOM`**:
   `FIXED_RANDOM_SPARSE_SUPPORT + DYNAMIC_WEIGHTS` (Fixed random binary support with degree K per query node; dynamic softmax over fixed edges).
4. **`STATIC_TRAIN_FREQ`**:
   `TRAIN_DERIVED_FIXED_SPARSE_SUPPORT + DYNAMIC_WEIGHTS` (Fixed support choosing top-K most frequent edges on Train; dynamic softmax over fixed edges).
5. **`STATIC_FULL_STATIC_WEIGHT`**:
   `FIXED_COMPLETE_SUPPORT + STATIC_WEIGHTS` (Degree 48 complete graph; fixed Train-mean attention weights).

---

## 2. Verification of `STATIC_RANDOM` Implementation

To rule out any accidental dynamic or data-dependent leakage in `STATIC_RANDOM`:
- **Generation**: Generated strictly once per seed via `np.random.RandomState(seed)` before any image is forwarded.
- **Invariance**: Unchanged across every image, split, and TTA forward pass.
- **Degrees**: Every query node has **exactly K allowed non-self neighbors**; self-connections are strictly zero on the diagonal.
- **Integrity**: Binary support tensors for all 20 seeds were serialized and SHA-256 hashed (recorded in `a2r2_random_seed_controls.csv`).
- **Weighting**: Once the static binary support is applied to mask disallowed scores to `-inf`, softmax attention weights are computed dynamically per sample from $QK^T/\sqrt{d} + \text{geometry}$.

---

## 3. 20-Seed Static Random Support Distribution ($K=24$)

Evaluating 20 deterministic seeds of fixed random support at 50% sparsity ($K=24$ non-self neighbors per query node) on v2.1:

| Metric | `FULL` (K=48) | `DYNAMIC_TOPK` (K=24) | `STATIC_RANDOM` (20 Seeds) Mean $\pm$ Std | `STATIC_RANDOM` Median | `STATIC_RANDOM` Range [Min, Max] |
|---|---|---|---|---|---|
| **Public TTA Acc** | 0.6927 | 0.6916 | **0.6848 $\pm$ 0.0035** | 0.6853 | [0.6773, 0.6907] |
| **Public TTA F1** | 0.6706 | 0.6696 | **0.6627 $\pm$ 0.0039** | 0.6628 | [0.6558, 0.6702] |
| **Private TTA Acc** | 0.6994 | 0.6991 | **0.6968 $\pm$ 0.0036** | 0.6967 | [0.6907, 0.7033] |
| **Private TTA F1** | 0.6900 | 0.6898 | **0.6881 $\pm$ 0.0041** | 0.6879 | [0.6806, 0.6946] |
| **Motif Readout Cosine**| 1.0000 | 0.9997 | **0.9702 $\pm$ 0.0021** | 0.9703 | [0.9664, 0.9739] |
| **Pred Change Rate** | 0.0000 | 0.0084 | **0.0894 $\pm$ 0.0076** | 0.0886 | [0.0763, 0.1039] |

### Comparison:
- `DYNAMIC_TOPK` ($K=24$) reliably achieves **69.91%** Private TTA Acc with **0.9997** cosine similarity to `FULL`.
- `STATIC_RANDOM` ($K=24$) averages **69.68%** Private TTA Acc with high fidelity (cosine 0.9702), trailing `FULL` by only -0.26 pp on average. Across all 20 seeds, random sparse graphs consistently maintain ~69.1?70.3% accuracy.

---

## 4. Random Control for Layerwise Degree Schedule (`[8, 16, 16, 16, 24]`)

When the graph is constrained to the tighter Train-derived layerwise schedule (Layer 1: $K=8$, Layers 2?4: $K=16$, Layer 5: $K=24$):

| Condition | Degree | Public TTA Acc | $\Delta$ Pub Acc | Private TTA Acc | $\Delta$ Priv Acc | Readout Cosine |
|---|---|---|---|---|---|---|
| **DYNAMIC_LAYERWISE** | `[8,16,16,16,24]` | **0.6927** | **+0.0000** | **0.6988** | **-0.0006** | **0.9956** |
| **STATIC_RANDOM_LW** (20 Seeds Mean $\pm$ Std) | `[8,16,16,16,24]` | **0.6468 $\pm$ 0.0046** | -0.0459 | **0.6536 $\pm$ 0.0053** | -0.0458 | **0.8601** |
| **STATIC_RANDOM_LW** Range [Min, Max] | `[8,16,16,16,24]` | [0.6358, 0.6556] | | [0.6459, 0.6634] | | [0.8432, 0.8720] |

### Key Distinction:
At aggressive layerwise sparsity ($K=8$ in early layers), fixed random connectivity collapses by **-4.6 percentage points** (65.36% vs 69.88%), whereas **`DYNAMIC_LAYERWISE` completely preserves full-model performance (69.88%)**. Dynamic support selection is decisive when operating in high-sparsity regimes ($K \le 16$).

---

## 5. Train-Frequency Static Support Evaluation

Constructing a static support by accumulating the frequency with which each edge is selected in sample-specific Top-K across 5,000 non-augmented Train images:

| Condition | Public TTA Acc | $\Delta$ Pub Acc | Private TTA Acc | $\Delta$ Priv Acc | Readout Cosine |
|---|---|---|---|---|---|
| **`FULL`** | 0.6927 | 0.0000 | 0.6994 | 0.0000 | 1.0000 |
| **`DYNAMIC_TOPK_K24`** | 0.6916 | -0.0011 | 0.6991 | -0.0003 | 0.9997 |
| **`STATIC_TRAIN_FREQ_K24`** | **0.6888** | **-0.0039** | **0.6991** | **-0.0003** | **0.9850** |
| **`STATIC_TRAIN_FREQ_LW`** `[8,16,16,16,24]` | **0.6581** | **-0.0346** | **0.6824** | **-0.0170** | **0.9379** |

### Finding:
`STATIC_TRAIN_FREQ` at $K=24$ achieves **69.91%** Private TTA Acc with 0.9853 cosine fidelity. This demonstrates that the catastrophic -12.3 pp failure observed previously in `TRAIN_MEAN_TOPK` was an artifact of picking geometrically localized edges, **not a failure of static sparse support in general**. At layerwise degree, however, dynamic support selection again outperforms Train-frequency support by **+1.6 pp** on Private and **+3.5 pp** on Public.

---

## 6. Support Structure Diagnostic: Why Did Train-Mean Collapse?

Comparing the structural network metrics across the 4 support constructions:

| Metric | `TRAIN_MEAN_TOPK` (Collapsed: 57.65%) | `TRAIN_FREQ_TOPK` (Successful: 69.91%) | `STATIC_RANDOM` (Successful: 69.68%) |
|---|---|---|---|
| **Local Edge Share** ($d=1$) | **39.39%** (Heavy bias) | **14.66%** | **13.10%** |
| **Meso Edge Share** ($d \in [2,3]$) | **60.61%** | **46.96%** | **43.06%** |
| **Far Edge Share** ($d \ge 4$) | **0.00% (Zero far edges!)** | **38.38%** | **43.84%** |
| **In-Degree Gini** | 0.1707 | 0.4334 (Hub-and-spoke) | 0.0785 |
| **In-Degree Range** | [8, 24] | [0, 48] (High centralization) | [13, 35] |
| **Reciprocity** | 100.0% (Symmetric grid) | 53.40% | 49.79% |
| **Total Edge Coverage** | 792 / 2352 (33.7%) | **2352 / 2352 (100.0%)** | 2352 / 2352 (100.0%) |

### Diagnostic Explanation:
`TRAIN_MEAN_TOPK` collapsed because ranking mean attention over the training set biased the support exclusively toward spatial neighbors within distance $d \le 2$ (local and meso), resulting in **exactly 0% far edges**. As proven in A2, expression classification strictly requires cross-facial communication (Layer 4 enriches far edges to 51.4%). Both `STATIC_TRAIN_FREQ` and `STATIC_RANDOM` succeed because they preserve 38?44% far edges and cover the full edge universe.

---

## 7. Full Train K-Selection Recheck (ALL 28,709 Images)

The preregistered K-selection rule ($	ext{Mean Mass} \ge 0.90$ and $p_{10} \ge 0.75$) was recomputed across all 8,440,446 query instances on the entire 28,709 non-augmented Train dataset:
- **Layer 1**: Mean 0.9052, $p_{10}$ 0.7552 $	o$ **$K=8$**
- **Layer 2**: Mean 0.9388, $p_{10}$ 0.8300 $	o$ **$K=16$**
- **Layer 3**: Mean 0.9124, $p_{10}$ 0.7667 $	o$ **$K=16$**
- **Layer 4**: Mean 0.9156, $p_{10}$ 0.7553 $	o$ **$K=16$**
- **Layer 5**: Mean 0.9426, $p_{10}$ 0.8434 $	o$ **$K=24$**
- **Result**: `K_schedule = [8, 16, 16, 16, 24]` and `single_K = 24` **REMAIN EXACTLY UNCHANGED**.

---

## 8. Cross-Version Compact Replication on v2 Checkpoint

Replicating on the official v2 EMA checkpoint (`f3cda72f...`, Epoch 62):

| Condition | Support Type | v2 Public TTA Acc | v2 Private TTA Acc |
|---|---|---|---|
| **`FULL`** | Complete (48) | 0.6868 | 0.6982 |
| **`DYNAMIC_TOPK_K24`** | Dynamic Top-24 | **0.6882 (+0.14 pp)** | **0.6960 (-0.22 pp)** |
| **`DYNAMIC_LAYERWISE`** | Dynamic `[8,16,16,16,24]` | **0.6851 (-0.17 pp)** | **0.6974 (-0.08 pp)** |
| **`STATIC_RANDOM_K24`** (20 Seeds Mean) | Random Static (24) | **0.6805 (-0.63 pp)** | **0.6945 (-0.37 pp)** |
| **`STATIC_RANDOM_K24`** (20 Seeds Range) | Random Static (24) | [0.6768, 0.6843] | [0.6871, 0.7027] |

### Finding:
Replication on v2 confirms identical mechanics: dynamic Top-K and dynamic layerwise schedules preserve full-model performance, while static random graphs average slightly lower accuracy.

---

## 9. Final Scientific Answers & Statuses

1. **`DYNAMIC_WEIGHTS`**:
   - **`STRONGLY_SUPPORTED`** (Frozen static weights collapse performance by >9.3 pp; sample-specific attention weighting is essential).
2. **`FIXED_SPARSE_SUPPORT_SUFFICIENT` vs `DYNAMIC_SPARSE_SUPPORT`**:
   - **Status**: **`SPARSITY_SUPPORTED_SUPPORT_ADAPTIVITY_UNRESOLVED`**
   - **Reasoning**: At moderate sparsity ($K=24$, 50% edges removed), fixed sparse supports with proper distance coverage (`STATIC_RANDOM` and `STATIC_TRAIN_FREQ`) preserve FULL performance (69.68?69.91% vs 69.94%), performing closely to `DYNAMIC_TOPK` (69.91%).
   - However, at aggressive sparsity ($K \le 16$, such as the layerwise schedule `[8, 16, 16, 16, 24]`), `DYNAMIC_LAYERWISE` substantially outperforms static random graphs (+4.6 pp on Private, +4.6 pp on Public) and static frequency graphs (+1.6 pp on Private, +3.5 pp on Public).
   - Therefore, while sparsity is definitively supported, the necessity of sample-dynamic support is **sparsity-dependent** (unresolved at $K=24$, but clearly superior under tight layerwise budgets).
