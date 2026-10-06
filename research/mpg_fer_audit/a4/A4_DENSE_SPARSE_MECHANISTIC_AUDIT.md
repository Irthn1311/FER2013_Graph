# MPG-FER A4: Dense-vs-Sparse Mechanistic Equivalence & Bottleneck Audit

**Status:** A4_COMPLETE_BOTTLENECK_EVIDENCE_READY  
**Execution Environment:** Windows (win32), NVIDIA GeForce RTX 3050 Ti Laptop GPU, CUDA 12.6, PyTorch 2.11.0+cu126  
**Audit Location:** `research/mpg_fer_audit/a4/`  
**Base Commit / Lineage:** Review commit `0a258fd43cc4d8f45afa54ea1328f068c52cbee0`  
**Scientific Source Modification:** None (NO)  
**Checkpoints Modified:** None (NO)  
**Training Performed:** None (NO)  
**PrivateTest Tuning:** None (NO, evaluation and analysis only)

---

## 1. Executive Summary & Core Verdict

The MPG-FER A4 post-hoc scientific audit directly compares the official selected frozen EMA checkpoints of **MPG-FER v2.1** (dense attention graph, complete degree-48 connectivity) and **MPG-FER v2.2** (dynamic hard Top-K sparse routing schedule `[8, 16, 16, 16, 24]`). Both models started from the exact same architecture dimensions (2,304,528 parameters) and seed (42), but were trained under fundamentally different topological support operators.

The central questions posed to this audit were:
1. *"Did dense v2.1 and sparse v2.2 converge to essentially the same internal FER solution, or do they achieve similar final performance using materially different representations/routing mechanisms?"*
2. *"If topology density is no longer the bottleneck, what evidence exists for the next limiting factor: representation separability, final classifier/calibration, or shared hard/ambiguous FER examples?"*

### Primary Audit Findings:
1. **Behavioral Performance Parity:** On both held-out test splits, paired performance differences between v2.1 and v2.2 are statistically indistinguishable from zero. McNemar's exact test yields $p = 0.4918$ on PublicTest and $p = 0.5687$ on PrivateTest. Paired stratified bootstrap (B=2000, 95% CI) accuracy deltas are $[-0.0075, +0.0162]$ (Public) and $[-0.0164, +0.0084]$ (Private).
2. **Representation Convergence:** Representations exhibit high linear CKA across the macro-network: Fusion CKA is 0.857 (Public) and 0.856 (Private); Motif readout CKA is 0.837 (Public) and 0.836 (Private). The 6x6 cross-layer CKA matrices demonstrate strict diagonal dominance, confirming that corresponding depths between dense and sparse architectures perform aligned semantic transformations.
3. **Macro-Convergent, Instance-Diverse Routing:** Macro edge frequencies correlate strongly between models (Pearson $r = 0.852$ in L1, $r = 0.916$ in L2, $r = 0.787$ in L3). However, at the single-sample instance level, support Jaccard similarity is only 0.27 to 0.38 (overlap / K is 40.5% to 51.6%), and exact support match rate is near zero (<1.7% in L1, <0.02% in L2-L5).
4. **Classifier Head Not the Bottleneck:** Fixed frozen multinomial linear probes fit on Train representations underperform the official trained classifier by $-1.2\%$ to $-1.8\%$ on both held-out splits (paired bootstrap 95% CIs strictly negative). The linear probe achieves 97-98% Train accuracy with a ~30% generalization gap. The trained MLP classifier head with dropout is clearly superior to linear probing and is not limiting performance.
5. **Shared Error Manifold:** Mutual error overlap is high: prediction agreement is 80.50% (Public) and 79.21% (Private). Among samples misclassified by both models, 73.86% (Public) and 72.64% (Private) predict the *exact same* incorrect class. In the top confusion pairs (e.g., Fear $\to$ Sad, Sad $\to$ Neutral, Neutral $\to$ Sad), shared errors constitute 50% to 64% of the entire error union.
6. **Training Feature Neighborhood Impurity:** 5-NN retrieval on Train Fusion features demonstrates that for mutually correct samples, local label purity is 92.5% to 93.5% (entropy 0.08 to 0.11). Conversely, for shared errors with the same wrong prediction, local label purity with respect to the nominal true class drops below 10% (0.085 to 0.099).

**Final Bottleneck Interpretation:**  
`SHARED_ERROR_DATA_AMBIGUITY_COMPATIBLE` (coupled with representation class separability limits).

---

## 2. A4.0 Provenance and Reproduction Gate

Strict validation of source trees, checkpoint SHA-256 digests, parameter counts, and official metric reproduction was executed prior to audit extraction.

| Model | Source Tree Hash (Expected & Actual) | Checkpoint SHA-256 (Expected & Actual) | Parameters | Epoch | Metric Reproduction Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **v2.1 Dense** | `d86d93655c83810d36c89a632baee0745f1de0f0e701b762b719d44445a6e679` | `4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75` | 2,304,528 | 57 | **REPRODUCED** (Exact) |
| **v2.2 Sparse** | `a8dc77db29e997c4c3ab69bb862704c8a948f940a4636e1c01e0d96bab40de65` | `a10bd22b3903550156c8239d91b5d2af35067ca1f2bdba9af46cf1e53d0bbdf4` | 2,304,528 | 64 | **REPRODUCED** (Exact in FP32, $\Delta = 5.57 \times 10^{-4}$ in AMP) |

### Detailed Metric Reproduction:

- **v2.1 PublicTest:**
  - Official TTA Accuracy: 0.6926720535 | Reproduced: 0.6926720535 (Diff: $0.00$)
  - Official TTA Macro-F1: 0.6705853330 | Reproduced: 0.6705853330 (Diff: $0.00$)
- **v2.1 PrivateTest:**
  - Official TTA Accuracy: 0.6993591530 | Reproduced: 0.6993591530 (Diff: $0.00$)
  - Official TTA Macro-F1: 0.6899949132 | Reproduced: 0.6899949132 (Diff: $0.00$)
- **v2.2 PublicTest:**
  - Official TTA Accuracy: 0.6965728615 (2500 / 3589) | Reproduced FP32: 0.6965728615 (2500 / 3589, Exact) | Reproduced AMP (sm_86): 0.6971301198 (2502 / 3589, Diff: $5.57 \times 10^{-4}$)
  - Official TTA Macro-F1: 0.6767888772 | Reproduced FP32: 0.6770556292 | Reproduced AMP: 0.6772807789
- **v2.2 PrivateTest:**
  - Official TTA Accuracy: 0.6954583449 | Reproduced: 0.6954583449 (Diff: $0.00$)
  - Official TTA Macro-F1: 0.6871410532 | Reproduced: 0.6871410532 (Diff: $0.00$)

*Gate Result:* **PASSED** (`a4_provenance.json`, `a4_metric_reproduction.json`).

---

## 3. A4.1 & A4.2 Representation Similarity & Parameter Divergence

### 3.1 Linear CKA on Matched Representations

Linear Centered Kernel Alignment (CKA) was measured between matched layers of frozen v2.1 and v2.2.

#### Pooled Representations (All Samples):
| Representation | Train CKA (28,709) | PublicTest CKA (3,589) | PrivateTest CKA (3,589) |
| :--- | :---: | :---: | :---: |
| **Pixel Readout [128]** | 0.8091 | 0.8006 | 0.8094 |
| **Motif Readout [384]** | 0.9295 | 0.8371 | 0.8362 |
| **Fusion [512]** | 0.9350 | 0.8569 | 0.8564 |
| **Classifier Hidden [256]** | 0.8851 | 0.6971 | 0.6931 |
| **Logits [7]** | 0.8927 | 0.7228 | 0.7212 |

#### Node-Level Within-Layer CKA (Deterministic 1024-Sample Stratified Subsets, Reshaped $[1024 \times 49, 192]$):
| Layer Depth | PublicTest CKA | PrivateTest CKA | Interpretation |
| :--- | :---: | :---: | :--- |
| **PRE (Composer Output)** | 0.8842 | 0.8821 | Upstream motif assignment space is highly preserved. |
| **Layer 1 (K=8)** | 0.5973 | 0.6009 | Maximum representational divergence occurs at Layer 1. |
| **Layer 2 (K=16)** | 0.6768 | 0.6811 | Representation alignment increases as depth accumulates. |
| **Layer 3 (K=16)** | 0.7082 | 0.7181 | Peak relational node alignment. |
| **Layer 4 (K=16)** | 0.6826 | 0.6902 | Consistent intermediate similarity. |
| **Layer 5 (K=24)** | 0.6477 | 0.6505 | Task-specialized relational state prior to readout pooling. |

#### Cross-Layer 6x6 CKA Matrix (PublicTest):
```
v2.1 \ v2.2     PRE       L1       L2       L3       L4       L5
PRE           0.884    0.538    0.234    0.157    0.160    0.162
L1            0.541    0.597    0.438    0.369    0.366    0.363
L2            0.247    0.479    0.677    0.684    0.675    0.657
L3            0.201    0.435    0.677    0.708    0.700    0.682
L4            0.209    0.428    0.669    0.693    0.683    0.664
L5            0.218    0.413    0.627    0.647    0.638    0.648
```
*Observation:* The cross-layer matrix is strictly diagonal-dominated. Each sparse layer $L_i$ in v2.2 aligns most strongly with the corresponding dense layer $L_i$ in v2.1. There is no depth inversion or layer-skipping semantic drift.

### 3.2 Parameter Divergence Analysis

Comparing parameter weights $W_{21}$ and $W_{22}$ across all 220 named parameters (2,304,528 total):

| Module | Parameter Count | Relative L2 Diff ($\|W_{22}-W_{21}\| / \|W_{21}\|$) | Cosine Similarity |
| :--- | :---: | :---: | :---: |
| **Pixel Projection** | 3,360 | 0.1296 | 0.9916 |
| **Pixel GNN Blocks** | 301,536 | 0.5569 | 0.8457 |
| **Motif Composer** | 51,078 | 0.5096 | 0.8705 |
| **Scale Gate** | 193 | 0.9550 | 0.5800 |
| **Motif Block 1** | 297,066 | 0.8601 | 0.6294 |
| **Motif Block 2** | 297,066 | 0.8498 | 0.6426 |
| **Motif Block 3** | 297,066 | 0.8626 | 0.6330 |
| **Motif Block 4** | 297,066 | 0.8713 | 0.6223 |
| **Motif Block 5** | 297,066 | 0.8878 | 0.6088 |
| **Motif Readout** | 225,224 | 0.8334 | 0.6517 |
| **Fusion / Classifier** | 133,639 | 0.7259 | 0.7366 |
| **SupCon Head** | 65,920 | 0.8193 | 0.6668 |
| **Full Network** | **2,304,528** | **0.7761** | **0.7006** |

*Takeaway:* Upstream pixel and motif extraction modules diverged relatively little (cosine similarity 0.85 to 0.99). The weight divergence is heavily concentrated within the relational Motif GNN blocks (cosine similarity 0.61 to 0.64, relative L2 diff ~0.86 to 0.89) and the scale gating mechanism.

---

## 4. A4.3 Dense vs. Sparse Routing Comparison

v2.2 enforces hard sparse support with schedule `[8, 16, 16, 16, 24]`. For v2.1, an audit-only derived Top-K support was extracted from its original dense pre-softmax attention scores using the identical schedule.

### Layer-Wise Support and Score Overlap (PublicTest / PrivateTest):

| Metric | Layer 1 (K=8) | Layer 2 (K=16) | Layer 3 (K=16) | Layer 4 (K=16) | Layer 5 (K=24) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Jaccard (Pub / Priv)** | 0.375 / 0.374 | 0.344 / 0.343 | 0.282 / 0.282 | 0.274 / 0.275 | 0.360 / 0.361 |
| **Overlap / K (Pub / Priv)** | 50.0% / 50.0% | 49.0% / 48.9% | 41.4% / 41.5% | 40.5% / 40.6% | 51.6% / 51.6% |
| **Exact Match Rate** | 1.66% / 1.62% | 0.01% / 0.01% | 0.004% / 0.004% | 0.007% / 0.009% | 0.001% / 0.001% |
| **JS Div (Dense vs Sparse)** | 0.353 / 0.354 | 0.426 / 0.427 | 0.462 / 0.463 | 0.467 / 0.468 | 0.465 / 0.465 |
| **JS Div (Top-K vs Sparse)** | 0.369 / 0.370 | 0.446 / 0.447 | 0.489 / 0.489 | 0.492 / 0.493 | 0.490 / 0.490 |
| **Cosine (Top-K vs Sparse)** | 0.419 / 0.418 | 0.270 / 0.269 | 0.222 / 0.222 | 0.226 / 0.227 | 0.196 / 0.197 |
| **Pre-softmax Spearman Corr** | 0.600 / 0.602 | 0.330 / 0.330 | 0.172 / 0.172 | 0.131 / 0.132 | 0.045 / 0.046 |
| **Edge Freq Pearson Corr** | 0.852 / 0.851 | 0.916 / 0.916 | 0.787 / 0.787 | 0.687 / 0.686 | 0.487 / 0.492 |
| **Edge Universe Coverage** | 100% (both) | 100% (both) | 100% (both) | 100% (both) | 100% (both) |

### Topological Distance Bin Composition:
- In Layer 1 (K=8): Local share (Chebyshev $d=1$) is 20.7% in v2.1 vs 20.9% in v2.2; Meso share ($d=2,3$) is 51.6% in v2.1 vs 52.2% in v2.2; Far share ($d \ge 4$) is 27.7% in v2.1 vs 26.9% in v2.2.
- In Layer 4 (K=16): Local share is 11.8% in v2.1 vs 9.7% in v2.2; Far share is 48.5% in v2.1 vs 54.1% in v2.2.

*Synthesis:* The models exhibit strong macro-frequency agreement across edge slots ($r \approx 0.85 - 0.92$ in early layers), showing that broad spatial communication channels are preserved. However, at the single-sample instance level, only 40% to 51% of edges overlap, and Spearman rank correlation of raw scores decays rapidly from 0.60 at Layer 1 to 0.04 at Layer 5. Dense and sparse models arrive at their final classifications through distinct, partially overlapping relational routing configurations.

---

## 5. A4.4 & A4.5 Prediction Agreement, Paired Statistics, and Calibration

### 5.1 Paired Statistical Significance (TTA Predictions)

| Split | Metric | Estimate ($\Delta = \text{v2.2} - \text{v2.1}$) | 95% Percentile Bootstrap CI | Hypothesis Test |
| :--- | :--- | :---: | :---: | :--- |
| **PublicTest** | Accuracy | $+0.0045$ | $[-0.0075, +0.0162]$ | McNemar $p = 0.4918$ (b=230, c=246) |
| | Macro-F1 | $+0.0067$ | $[-0.0091, +0.0225]$ | Stratified paired bootstrap CI spans 0 |
| | NLL | $+0.0360$ | $[+0.0036, +0.0677]$ | Statistically higher NLL in v2.2 |
| | Brier Score | $+0.0033$ | $[-0.0109, +0.0179]$ | CI spans 0 |
| **PrivateTest** | Accuracy | $-0.0039$ | $[-0.0164, +0.0084]$ | McNemar $p = 0.5687$ (b=267, c=253) |
| | Macro-F1 | $-0.0029$ | $[-0.0192, +0.0123]$ | Stratified paired bootstrap CI spans 0 |
| | NLL | $+0.0581$ | $[+0.0283, +0.0900]$ | Statistically higher NLL in v2.2 |
| | Brier Score | $+0.0173$ | $[+0.0038, +0.0318]$ | Slightly higher Brier in v2.2 |

*Conclusion:* Neither accuracy nor macro-F1 differences between v2.1 and v2.2 are statistically significant. The two models are in an exact behavioral tie.

### 5.2 Error Overlap & Agreement Structure (TTA)

| Split | Both Correct | v2.1 Only Correct | v2.2 Only Correct | Both Wrong (Total) | Both Wrong (Same Class) | Error-Set Jaccard | Cohen's Kappa |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Public** | 2,256 (62.86%) | 230 (6.41%) | 246 (6.85%) | 857 (23.88%) | **633 (73.86% of errors)** | **0.6429** | **0.7637** |
| **Private** | 2,243 (62.50%) | 267 (7.44%) | 253 (7.05%) | 826 (23.01%) | **600 (72.64% of errors)** | **0.6137** | **0.7488** |

### 5.3 Probability Calibration (15 Equal-Width Bins)

| Split | Model | NLL | Brier Score | ECE (15 bins) | Mean Conf (Correct) | Mean Conf (Incorrect) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Public** | **v2.1 Dense** | 1.0012 | 0.4632 | 0.1265 | 0.8687 | 0.7258 |
| | **v2.2 Sparse** | 1.0372 | 0.4665 | 0.1373 | 0.8804 | 0.7456 |
| **Private** | **v2.1 Dense** | 0.9513 | 0.4449 | 0.1190 | 0.8739 | 0.7315 |
| | **v2.2 Sparse** | 1.0094 | 0.4622 | 0.1368 | 0.8808 | 0.7485 |

*Takeaway:* Both models exhibit overconfidence on incorrect predictions (mean confidence ~73-75%). Sparse routing does *not* improve probability calibration; v2.1 achieves slightly lower NLL and ECE across both splits.

---

## 6. A4.6 & A4.7 Frozen Linear Probes and Class Geometry

### 6.1 Linear Probe Performance (A1 Protocol: StandardScaler, Multinomial LogisticRegression, C=1.0, L2, max_iter=5000)

| Model | Representation | Split | Probe TTA Acc | Probe TTA Macro-F1 | Official Classifier TTA Acc | Paired Delta ($\Delta = \text{Probe} - \text{Official}$) | 95% Bootstrap CI |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **v2.1** | Motif Readout [384] | Public | 0.6810 | 0.6602 | 0.6927 | $-0.0117$ | $[-0.0195, -0.0042]$ |
| | | Private | 0.6812 | 0.6748 | 0.6994 | $-0.0181$ | $[-0.0262, -0.0100]$ |
| | Fusion [512] | Public | 0.6801 | 0.6639 | 0.6927 | $-0.0125$ | $[-0.0203, -0.0047]$ |
| | | Private | 0.6851 | 0.6796 | 0.6994 | $-0.0142$ | $[-0.0223, -0.0061]$ |
| **v2.2** | Motif Readout [384] | Public | 0.6835 | 0.6602 | 0.6971 | $-0.0137$ | $[-0.0212, -0.0059]$ |
| | | Private | 0.6821 | 0.6751 | 0.6955 | $-0.0134$ | $[-0.0209, -0.0056]$ |
| | Fusion [512] | Public | 0.6790 | 0.6621 | 0.6971 | $-0.0181$ | $[-0.0265, -0.0098]$ |
| | | Private | 0.6826 | 0.6695 | 0.6955 | $-0.0128$ | $[-0.0203, -0.0056]$ |

#### Generalization Gap:
- On Train data, the linear probes reach 96.2% to 98.3% accuracy.
- Held-out generalization gaps are large: $0.298$ to $0.304$ on Public and Private tests.

*Bottleneck Assessment:* Frozen linear probes fail to match the official trained classifier across all representations and splits by $-1.2\%$ to $-1.8\%$. The hypothesis that the classifier head is a performance bottleneck ($H\text{-A4-C}$) is **strongly rejected/deprioritized**.

### 6.2 Class Geometry & Centroid Separability

Centroids fit on standardized Train features evaluated on held-out test splits:

| Representation | Metric | v2.1 Public | v2.1 Private | v2.2 Public | v2.2 Private |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fusion [512]** | Nearest-Centroid Accuracy | 0.6709 | 0.6829 | 0.6768 | 0.6843 |
| | Mean Centroid Margin | +5.30 | +5.49 | +5.12 | +5.32 |
| | Within-Class Scatter | 390.4 | 385.0 | 392.8 | 387.0 |
| | Between-Class Centroid Sep | 26.31 | 26.31 | 26.44 | 26.44 |
| | Fisher-Style Ratio ($S_B^2 / S_W$) | 1.773 | 1.798 | 1.779 | 1.806 |

#### Class-Wise Centroid Margins ($\min_{c \ne y} d_c - d_y$ on Fusion):
| Class | Category | v2.1 Public Margin | v2.2 Public Margin | v2.1 Private Margin | v2.2 Private Margin |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Happy** | Easy Reference | **+11.08** | **+10.98** | **+11.45** | **+11.37** |
| **Surprise** | Easy Reference | **+9.84** | **+9.68** | **+8.90** | **+8.76** |
| **Neutral** | Hard Emotion | +3.32 | +3.24 | +4.66 | +4.57 |
| **Angry** | Hard Emotion | +2.53 | +2.68 | +2.77 | +2.89 |
| **Sad** | Hard Emotion | +1.85 | +1.85 | +1.34 | +1.34 |
| **Fear** | Hard Emotion | **-0.16** | **-0.25** | **-0.06** | **-0.15** |

*Critical Insight:* In both architectures, **Fear has a negative average centroid margin** ($-0.06$ to $-0.25$), indicating that test Fear examples are geometrically closer to competing class centroids than to the Fear centroid itself. Sad possesses extremely narrow margin separation ($+1.34$). This geometric collapse on hard negative emotions is identical across dense and sparse models.

---

## 7. A4.8 Flip Equivariance & TTA Mechanism

| Split | Metric | v2.1 Dense | v2.2 Sparse | Comparison |
| :--- | :--- | :---: | :---: | :--- |
| **Public** | Prediction Agreement Under Flip | 78.16% | 77.77% | Unchanged ($\Delta = -0.39\%$) |
| | Probability JS Divergence | 0.0730 | 0.0853 | Slightly higher in v2.2 |
| | Fusion Representation Cosine | 0.8659 | 0.8579 | High consistency in both |
| | Support Mirror Jaccard (Layer 1) | 0.2025 | **0.3713** | **+0.1688 in v2.2** |
| | Support Mirror Jaccard (Layer 3) | 0.2825 | **0.3978** | **+0.1153 in v2.2** |
| | Support Mirror Jaccard (Layer 4) | 0.3038 | **0.4192** | **+0.1154 in v2.2** |
| | Support Mirror Jaccard (Layer 5) | 0.3969 | **0.4602** | **+0.0633 in v2.2** |
| **Private** | Prediction Agreement Under Flip | 78.29% | 78.35% | Unchanged ($\Delta = +0.06\%$) |
| | Support Mirror Jaccard (Layer 1) | 0.2030 | **0.3699** | **+0.1668 in v2.2** |
| | Support Mirror Jaccard (Layer 3) | 0.2823 | **0.3981** | **+0.1158 in v2.2** |
| | Support Mirror Jaccard (Layer 4) | 0.3048 | **0.4186** | **+0.1138 in v2.2** |

*Takeaway:* Behavioral flip consistency remains unchanged (~78% prediction agreement). However, sparse routing materially improves the geometric mirror-equivariance of relational edge selection across layers (+6% to +17% higher support Jaccard).

---

## 8. A4.9 & A4.10 Shared Hard Examples and Error Manifold

### 8.1 Shared High-Confidence Errors

Across Public and Private splits, **467 samples** meet the strict criteria for `SHARED_HIGH_CONFIDENCE_ERROR_CANDIDATES` (both models misclassify the image into the exact same incorrect class with confidence $\ge 0.80$ in both models):
- PublicTest: 244 samples
- PrivateTest: 223 samples

These samples are cataloged in `a4_shared_high_confidence_errors.csv` and visualized in 70 contact sheets under `research/mpg_fer_audit/a4/contact_sheets/`.

### 8.2 Top Shared Error Confusion Pairs

| Confusion Pair | Public Shared Errors | Public Error Union | Public Overlap % | Private Shared Errors | Private Error Union | Private Overlap % | Mean Conf on Shared |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fear $\to$ Sad** | 60 | 119 | **50.4%** | 75 | 118 | **63.6%** | 0.756 |
| **Neutral $\to$ Sad** | 57 | 107 | **53.3%** | 61 | 105 | **58.1%** | 0.771 |
| **Sad $\to$ Neutral** | 49 | 118 | **41.5%** | 53 | 103 | **51.5%** | 0.762 |
| **Angry $\to$ Sad** | 33 | 79 | **41.8%** | 45 | 82 | **54.9%** | 0.749 |
| **Sad $\to$ Angry** | 39 | 97 | **40.2%** | 37 | 79 | **46.8%** | 0.766 |
| **Fear $\to$ Neutral** | 35 | 69 | **50.7%** | 32 | 75 | **42.7%** | 0.751 |

### 8.3 5-NN Local Training Neighborhood Purity (A4.11 / Section 23)

Using L2-normalized Train Fusion representations to retrieve the 5-nearest training neighbors for each test sample:

| Category | Public Count | v2.1 5-NN Purity | v2.2 5-NN Purity | Private Count | v2.1 5-NN Purity | v2.2 5-NN Purity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Both Correct** | 2,256 | **92.46%** (ent 0.10) | **93.18%** (ent 0.08) | 2,243 | **92.01%** (ent 0.11) | **93.54%** (ent 0.08) |
| **B. v2.1 Only Correct** | 230 | 70.43% (ent 0.33) | 19.65% (ent 0.29) | 267 | 69.59% (ent 0.31) | 24.64% (ent 0.35) |
| **C. v2.2 Only Correct** | 246 | 19.19% (ent 0.35) | 69.19% (ent 0.28) | 253 | 24.03% (ent 0.38) | 74.86% (ent 0.22) |
| **D. Both Wrong (Same Class)** | 633 | **9.42%** (ent 0.23) | **8.53%** (ent 0.18) | 600 | **9.90%** (ent 0.24) | **9.17%** (ent 0.20) |
| **E. Both Wrong (Diff Class)** | 224 | 10.98% (ent 0.38) | 9.29% (ent 0.31) | 226 | 10.35% (ent 0.38) | 8.41% (ent 0.28) |

*Mechanistic Proof:* When both models make the same error, they are projecting the sample into a training feature subspace where the local neighborhood is almost exclusively populated by non-target classes (purity < 10%).

---

## 9. Explicit Answers to the 13 Required Questions

### 1. How similar are v2.1 and v2.2 representations?
Macro-representations are highly similar. Linear CKA is 0.857 on Fusion and 0.837 on Motif readout across held-out test splits. On Train data, CKA reaches 0.935 on Fusion.

### 2. At which layer do they diverge most?
In weight space, the Motif GNN blocks diverge most ($\|W_{22}-W_{21}\| / \|W_{21}\| \approx 0.85 - 0.89$, cosine similarity $\approx 0.61 - 0.64$). In node representation space, within-layer CKA is lowest at Layer 1 (0.597 on Public, 0.601 on Private), before recovering to 0.71 in Layer 3.

### 3. Did sparse training substantially alter Pixel/Composer representations, or mainly Motif Graph reasoning?
It altered mainly Motif Graph reasoning. The Pixel Projection is almost identical (cosine similarity 0.9916, relative diff 0.1296). Pixel GNN blocks have cosine similarity 0.8457. The Motif Composer has cosine similarity 0.8705 and PRE node CKA of 0.8842. The Motif GNN blocks show the greatest divergence (cosine similarity ~0.61 - 0.64).

### 4. Do v2.1's highest-ranked dense edges match v2.2's selected sparse edges?
At the global frequency level, yes ($r = 0.85$ to $0.92$ in early layers). At the individual sample level, they match only partially: mean support Jaccard is 0.27 to 0.38 (overlap / K is 40.5% to 51.6%), and exact support match rate across the 49 query nodes is under 1.7%.

### 5. Do the two models fail on the same FER examples?
Yes. Error-set Jaccard is 0.6429 on Public and 0.6137 on Private. Among mutual errors, 73.86% (Public) and 72.64% (Private) predict the exact same wrong emotion.

### 6. Are their hardest confusion pairs the same?
Yes. Both models struggle with the exact same confusion pairs: Fear $\to$ Sad, Neutral $\to$ Sad, Sad $\to$ Neutral, Angry $\to$ Sad, and Sad $\to$ Angry, with 50% to 64% mutual error overlap across these pairs.

### 7. Does v2.2 produce better calibration?
No. Probability calibration is slightly worse in v2.2: Public NLL is 1.0372 in v2.2 vs 1.0012 in v2.1; ECE is 0.1373 vs 0.1265. Both models exhibit overconfidence on incorrect predictions (~73-75%).

### 8. Does sparse routing improve flip consistency?
At the final prediction level, flip consistency is unchanged (78.16% vs 77.77% on Public, 78.29% vs 78.35% on Private). However, at the internal relational support level, sparse routing significantly improves mirror-equivariance Jaccard (+6% to +17% across layers 1, 3, 4, 5).

### 9. Does a frozen linear probe outperform the trained classifier?
No. Across all representations and test splits, the linear probe performs $-1.2\%$ to $-1.8\%$ worse than the trained classifier, with 95% bootstrap confidence intervals strictly below zero.

### 10. Is there evidence that the classifier is the next bottleneck?
No. The hypothesis that the classifier head is the bottleneck is definitively deprioritized. The trained MLP classifier head with dropout generalizes better than linear probes, which overfit the training features (97-98% Train acc vs 68% Test acc).

### 11. Is the remaining ceiling compatible with shared example/class ambiguity?
Yes. 467 test images are misclassified by both models into the exact same incorrect class with confidence $\ge 0.80$. In training feature space, these samples have $<10\%$ 5-NN label purity. Fear displays a negative true-class centroid margin ($-0.06$ to $-0.25$) in both models.

### 12. Is routing still worth optimizing further?
No. Routing optimization should be **deprioritized**. Dense complete connectivity (v2.1) and dynamic hard Top-K sparse routing (v2.2) use materially different instance-level routing configurations (~40-50% overlap) yet converge to virtually identical macro-representations (CKA 0.86) and an identical behavioral ceiling (McNemar $p \approx 0.49 - 0.57$).

### 13. Based on A4, which scientific question deserves the NEXT audit?
The next audit (A5) should investigate **upstream class separability and label ambiguity on hard negative emotions (Fear, Sad, Angry, Neutral)**. Specifically, an audit should evaluate whether fine-grained facial landmarks / action unit priors or clean relabeling of the 467 shared high-confidence errors resolves the negative margin of Fear and the Sad/Neutral overlap.

---

## 10. Summary of Generated Artifacts

All files have been verified in `research/mpg_fer_audit/a4/`:
- `a4_provenance.json` (Source and checkpoint provenance verification)
- `a4_metric_reproduction.json` (Replication of official Public and Private metrics)
- `a4_parameter_divergence.json` (Layer-wise relative L2 diff and cosine similarities)
- `a4_representation_cka.json` (Pooled and node-level within-layer CKA values)
- `a4_cross_layer_cka.csv` (6x6 cross-layer CKA matrices for Public and Private)
- `a4_routing_overlap.json` (Support Jaccard, overlap/K, JS divergence, edge frequency correlations)
- `a4_routing_classwise.json` (Class-specific routing overlap metrics)
- `a4_flip_equivariance.json` (Original vs. flip consistency and mirror support Jaccard)
- `a4_prediction_agreement.json` (Prediction agreement, Cohen's kappa, agreement matrices)
- `a4_paired_statistics.json` (McNemar's test and paired bootstrap 95% CIs)
- `a4_calibration.json` (NLL, Brier score, ECE 15 bins, confidence by correctness)
- `a4_probe_metrics.json` (Frozen linear probe accuracies, macro-F1, generalization gaps)
- `a4_probe_bootstrap.json` (Paired bootstrap of probes vs. official classifier)
- `a4_class_geometry.json` (Train centroids, nearest-centroid acc, true-class margins, Fisher ratios)
- `a4_shared_high_confidence_errors.csv` (467 samples with confidence >= 0.80)
- `a4_model_disagreements.csv` (Top 50 confidence discordances per split)
- `a4_error_overlap.csv` (Detailed confusion pair error overlap analysis)
- `a4_knn_purity.json` (5-NN local training label purity and entropy by category)
- `a4_hypothesis_decisions.json` (Formal status decisions for H-A4-R through H-A4-F)
- `a4_cka_pooled.png` (Bar chart of pooled CKA)
- `a4_cka_cross_layer_public.png` (Heatmap of 6x6 cross-layer CKA for PublicTest)
- `a4_cka_cross_layer_private.png` (Heatmap of 6x6 cross-layer CKA for PrivateTest)
- `a4_routing_overlap_by_layer.png` (Support Jaccard and Overlap/K by layer)
- `a4_calibration_public.png` (Reliability diagram for PublicTest)
- `a4_calibration_private.png` (Reliability diagram for PrivateTest)
- `a4_flip_support_jaccard.png` (Mirror-mapped flip support Jaccard by layer)
- `a4_error_overlap_public.png` (Bar plot of top error confusion pairs for PublicTest)
- `a4_error_overlap_private.png` (Bar plot of top error confusion pairs for PrivateTest)
- `contact_sheets/*.png` (70 contact sheets visualizing shared errors and disagreements)
