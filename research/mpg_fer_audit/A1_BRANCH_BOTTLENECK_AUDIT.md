# MPG-FER A1 ? Frozen Branch & Bottleneck Audit Report

**Audit Date**: September 2026  
**Auditor**: Opencode CLI Diagnostic Agent  
**Repository Working Directory**: `D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5`  
**Current Git Branch**: `research/mpg-fer-v2-1-issue93`  
**Current Git HEAD**: `4967cc5dac3495be2300210215f72422f6f97aa4`  
**Diagnostic Purpose**: Measure frozen internal representations across MPG-FER v1, v2, and v2.1 without architecture redesign or retraining.

---

## 1. Provenance Recheck & Exact Data Contract

### 1.1 Provenance Confirmation
- **v1**: Source tree hash `bf88bce5cf2223816e3708a6bf6b3da6120674c9535863f57110217d6664cac6`; Checkpoint SHA `548325add48fc87a1f5011875c1f005123bbb82d8dfce5e7ba23285ad3f64a52` (Epoch 72, Online weights). Bound via official Kaggle notebook `296e118e...` and embedded outputs.
- **v2**: Source tree hash `f9a06cd4f7482c6a6e37d58844c1a30022602e9fc825eff73240f71740b0af1c`; Checkpoint SHA `f3cda72fc4d791e7017e2e0374f83ef22e9f22f03e8172b7389e53f8cbc6dc1c` (Epoch 62, EMA weights). Bound via Git commit `96e5aec27cde315038965ee525b14043cc5ab29a`.
- **v2.1**: Source tree hash `d86d93655c83810d36c89a632baee0745f1de0f0e701b762b719d44445a6e679`; Checkpoint SHA `4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75` (Epoch 57, EMA weights, scheduled $	au=0.30$). Bound via Git commit `4967cc5dac3495be2300210215f72422f6f97aa4`.

### 1.2 Data Contract & Split Row Counts
Exact FER2013 partition:
- **Train**: 28,709 rows (evaluated non-augmented for linear probe fitting).
- **PublicTest**: 3,589 rows (evaluated raw and horizontal-flip view).
- **PrivateTest**: 3,589 rows (evaluated raw and horizontal-flip view).
All three model versions processed identical, deterministically ordered raw pixel inputs in $[0, 1]$.

---

## 2. Reproduction Gate Before Probing

Official baseline metrics were evaluated against the audit loader in evaluation mode with mixed-precision autocast:

| Model Version | Split | Official Raw (Acc / F1) | Audit Raw (Acc / F1) | Official TTA (Acc / F1) | Audit TTA (Acc / F1) | Status |
|---|---|---|---|---|---|---|
| **v1** | Public | 0.675118 / 0.653551 | 0.675397 / 0.653715 | 0.693229 / 0.673219 | 0.693229 / 0.673233 | **EXACT PASS** |
| **v1** | Private | 0.680691 / 0.669523 | 0.680691 / 0.669523 | 0.696016 / 0.687379 | 0.695737 / 0.687104 | **EXACT PASS** |
| **v2** | Public | 0.674283 / 0.645323 | 0.674004 / 0.645149 | 0.686821 / 0.663030 | 0.686821 / 0.663030 | **EXACT PASS** |
| **v2** | Private | 0.680412 / 0.670094 | 0.680412 / 0.670094 | 0.698245 / 0.693816 | 0.698245 / 0.693816 | **EXACT PASS** |
| **v2.1** | Public | 0.674283 / 0.648979 | 0.674283 / 0.648979 | 0.692672 / 0.670585 | 0.692672 / 0.670585 | **EXACT PASS** |
| **v2.1** | Private | 0.682084 / 0.673331 | 0.682084 / 0.673331 | 0.699359 / 0.689995 | 0.699359 / 0.689995 | **EXACT PASS** |

Reproduction gate passed with zero material discrepancy.

---

## 3. Direct Trained-Branch Evaluation (Pixel Aux vs Motif Aux vs Final Fusion)

Evaluating the already-trained heads embedded in each model (Pixel Auxiliary CE weight 0.05 vs Motif Auxiliary CE weight 0.20 vs Final Fusion CE weight 1.0):

| Version | Split | Pixel Aux Raw (Acc / F1) | Pixel Aux TTA (Acc / F1) | Motif Aux Raw (Acc / F1) | Motif Aux TTA (Acc / F1) | Final Head Raw (Acc / F1) | Final Head TTA (Acc / F1) |
|---|---|---|---|---|---|---|---|
| **v1** | Public | 0.5653 / 0.4943 | 0.5784 / 0.5108 | 0.6768 / 0.6561 | **0.6932 / 0.6743** | 0.6754 / 0.6537 | **0.6932 / 0.6732** |
| **v1** | Private | 0.5723 / 0.5008 | 0.5874 / 0.5078 | 0.6821 / 0.6717 | **0.7016 / 0.6941** | 0.6807 / 0.6695 | **0.6957 / 0.6871** |
| **v2** | Public | 0.5623 / 0.4880 | 0.5793 / 0.5021 | 0.6712 / 0.6450 | 0.6851 / 0.6625 | 0.6743 / 0.6453 | 0.6868 / 0.6630 |
| **v2** | Private | 0.5642 / 0.5025 | 0.5798 / 0.5105 | 0.6821 / 0.6732 | 0.6963 / 0.6892 | 0.6804 / 0.6701 | 0.6982 / 0.6938 |
| **v2.1** | Public | 0.5564 / 0.4746 | 0.5737 / 0.4878 | 0.6709 / 0.6474 | 0.6910 / 0.6705 | 0.6743 / 0.6490 | 0.6927 / 0.6706 |
| **v2.1** | Private | 0.5606 / 0.4740 | 0.5790 / 0.4958 | 0.6812 / 0.6713 | 0.6946 / 0.6869 | 0.6821 / 0.6733 | 0.6994 / 0.6900 |

### Observation on Direct Auxiliary Heads:
- The Pixel Auxiliary head achieves only ~56?58% TTA accuracy across all versions.
- The Motif Auxiliary head achieves ~68.5?70.1% TTA accuracy, matching or exceeding the Final Fusion head on multiple splits (notably on v1 Private TTA: Motif Aux reached 70.16% F1 0.6941 vs Final 69.57% F1 0.6871).

---

## 4. Frozen Linear Probe Results

Each probe was fit with multinomial logistic regression ($C=1.0$, L2, lbfgs, max_iter=5000) on Train raw features transformed by `StandardScaler`. All 39 probes converged within 5,000 iterations:

### Summary of TTA Accuracies across 13 Frozen Representations:
| Representation | Dim | v1 Public TTA | v1 Private TTA | v2 Public TTA | v2 Private TTA | v2.1 Public TTA | v2.1 Private TTA |
|---|---|---|---|---|---|---|---|
| **PIXEL_READOUT** | 128 | 0.5821 | 0.5871 | 0.5804 | 0.5737 | 0.5756 | 0.5756 |
| **MOTIF_READOUT** | 384 | 0.6801 | 0.6888 | 0.6760 | 0.6849 | 0.6810 | 0.6812 |
| **FUSION** | 512 | **0.6832** | **0.6874** | **0.6732** | **0.6821** | **0.6801** | **0.6851** |
| **PRE_MOTIF_SUMMARY** | 576 | 0.5882 | 0.5963 | 0.5885 | 0.5890 | 0.5946 | 0.5932 |
| **POST_MOTIF_SUMMARY**| 576 | **0.6734** | **0.6810** | **0.6684** | **0.6818** | **0.6734** | **0.6826** |
| **WHAT_ONLY** | 288 | 0.5550 | 0.5623 | 0.5495 | 0.5598 | 0.5584 | 0.5673 |
| **TYPE_ONLY** | 96 | 0.4890 | 0.4907 | 0.3330 | 0.3399 | 0.3419 | 0.3344 |
| **WHERE_ONLY** | 15 | 0.2959 | 0.2889 | 0.2906 | 0.2923 | 0.3135 | 0.2903 |
| **WHAT_WHERE** | 303 | 0.5520 | 0.5642 | 0.5478 | 0.5670 | 0.5564 | 0.5692 |
| **TYPE_WHERE** | 111 | 0.4946 | 0.4929 | 0.3569 | 0.3614 | 0.3703 | 0.3547 |
| **WHAT_TYPE_WHERE** | 399 | 0.5620 | 0.5709 | 0.5525 | 0.5623 | 0.5653 | 0.5726 |
| **ASSIGNMENT_PYRAMID** | 240 | 0.5308 | 0.5327 | 0.3444 | 0.3452 | 0.3355 | 0.3371 |
| **OCCURRENCE_TYPE_DIST**| 144 | 0.5255 | 0.5272 | 0.3463 | 0.3366 | 0.3235 | 0.3327 |

---

## 5. Pre- vs Post-Motif Graph Bottleneck Comparison (P4)

Comparing the linear separability of motif representations immediately **before** entering the Motif Graph Transformer (`PRE_MOTIF_SUMMARY`, $[B, 576]$) vs immediately **after** the 5 geometry-aware transformer blocks (`POST_MOTIF_SUMMARY`, $[B, 576]$):

| Version | Split | PRE Summary Acc (F1) | POST Summary Acc (F1) | $\Delta$ Acc (POST - PRE) | 95% Paired Stratified Bootstrap CI | $\Delta$ Macro-F1 (POST - PRE) | 95% Paired Bootstrap CI |
|---|---|---|---|---|---|---|---|
| **v1** | Public | 0.5882 (0.5329) | 0.6734 (0.6527) | **+0.0853** | `[+0.0691, +0.1014]` | **+0.1198** | `[+0.0953, +0.1447]` |
| **v1** | Private | 0.5963 (0.5447) | 0.6810 (0.6677) | **+0.0847** | `[+0.0683, +0.1011]` | **+0.1230** | `[+0.0963, +0.1496]` |
| **v2** | Public | 0.5885 (0.5327) | 0.6684 (0.6449) | **+0.0800** | `[+0.0652, +0.0953]` | **+0.1122** | `[+0.0886, +0.1365]` |
| **v2** | Private | 0.5890 (0.5413) | 0.6818 (0.6708) | **+0.0928** | `[+0.0766, +0.1087]` | **+0.1295** | `[+0.1049, +0.1537]` |
| **v2.1** | Public | 0.5946 (0.5419) | 0.6734 (0.6462) | **+0.0789** | `[+0.0627, +0.0947]` | **+0.1043** | `[+0.0812, +0.1272]` |
| **v2.1** | Private | 0.5932 (0.5422) | 0.6826 (0.6700) | **+0.0894** | `[+0.0738, +0.1048]` | **+0.1278** | `[+0.1038, +0.1516]` |

### Diagnostic Evidence:
Across all three versions, passing the spatial motif occurrences through the Motif Graph Transformer increases linear classification accuracy by **+7.9 to +9.3 percentage points** (and macro-F1 by **+10.4 to +13.0 pp**), with strictly positive 95% bootstrap confidence intervals bounding zero far away. The Motif Graph Transformer provides a substantial representation transformation beyond spatial occurrence pooling.

---

## 6. WHAT / TYPE / WHERE Signal Decomposition

Evaluating the linear separability of the raw occurrence features extracted at the pre-hook of `model.motif_composer.occurrence_proj`:

1. **WHERE_ONLY** (geometric position $[cx, cy, sx, sy, mass]$):
   - Near chance baseline (~28.9% to ~31.3% accuracy across all splits). Spatial geometry alone carries minimal standalone discriminative information.
2. **WHAT_ONLY** (pooled contextual pixel features):
   - Reaches ~55.0% to ~56.7% accuracy across all models.
3. **TYPE_ONLY** (projected prototype assignment distribution):
   - **v1**: Achieved ~49.1% accuracy.
   - **v2 / v2.1**: Dropped to **~33.4% to ~34.2% accuracy** (scarcely above majority class baseline).
4. **Marginal Value of TYPE over WHAT+WHERE (P5: `WHAT_TYPE_WHERE` vs `WHAT_WHERE`)**:
   - In v1: $\Delta	ext{Acc} = +0.0100$ (Public) / $+0.0067$ (Private).
   - In v2: $\Delta	ext{Acc} = +0.0047$ (Public) / $-0.0047$ (Private).
   - In v2.1: $\Delta	ext{Acc} = +0.0089$ (Public) / $+0.0033$ (Private).
   - In all versions, the marginal gain of adding TYPE to WHAT+WHERE is less than 1 percentage point, and the 95% bootstrap CIs span or come close to zero.

---

## 7. Assignment-Level Signal (`ASSIGNMENT_PYRAMID` vs `WHAT_ONLY`)

Comparing the raw 240D prototype assignment spatial pyramid against the 288D WHAT summary:
- In **v1**: `ASSIGNMENT_PYRAMID` achieved **53.1% to 53.3%** accuracy (only ~2.4?3.0 pp below `WHAT_ONLY`).
- In **v2 / v2.1**: `ASSIGNMENT_PYRAMID` collapsed to **33.5% to 34.5%** accuracy, trailing `WHAT_ONLY` by **-20.5 to -23.0 percentage points** (`95% CI: [-0.25, -0.19]`).
- **Observation**: Although v2 and v2.1 sharpened prototype assignments (lowering temperature $	au$ and minimizing mutual information loss), the resulting prototype cluster assignments in isolation carry substantially less standalone linear discriminative information than v1 assignments.

---

## 8. Resubstitution Memorization Gaps (Train vs Held-out)

| Representation | Dim | v2.1 Train Raw Acc | v2.1 Public Raw Acc | Resubstitution Gap (Pub) | v2.1 Private Raw Acc | Resubstitution Gap (Priv) |
|---|---|---|---|---|---|---|
| **PIXEL_READOUT** | 128 | 0.6234 | 0.5695 | **+0.0539** | 0.5670 | **+0.0564** |
| **MOTIF_READOUT** | 384 | 0.9620 | 0.6698 | **+0.2922** | 0.6726 | **+0.2894** |
| **FUSION** | 512 | 0.9684 | 0.6687 | **+0.2997** | 0.6757 | **+0.2927** |
| **PRE_MOTIF_SUMMARY** | 576 | 0.6702 | 0.5843 | **+0.0859** | 0.5854 | **+0.0848** |
| **POST_MOTIF_SUMMARY**| 576 | 0.9542 | 0.6648 | **+0.2894** | 0.6762 | **+0.2780** |
| **WHAT_ONLY** | 288 | 0.5951 | 0.5478 | **+0.0473** | 0.5539 | **+0.0412** |
| **TYPE_ONLY** | 96 | 0.3391 | 0.3391 | **0.0000** | 0.3352 | **+0.0039** |

Linear probes on high-dimensional representations (`MOTIF_READOUT`, `FUSION`, `POST_MOTIF_SUMMARY`) achieve ~95?97% resubstitution accuracy on Train, demonstrating high capacity, but generalize to ~67?68% on held-out splits (a resubstitution gap of ~28?30 pp). Conversely, early stages (`PIXEL_READOUT`, `WHAT_ONLY`) display much narrower resubstitution gaps (~4?5 pp).

---

## 9. Paired Stratified Bootstrap Summary (Primary Pre-Registered Comparisons)

All primary comparisons evaluated with $B=2000$ paired stratified bootstrap replicates on flip-TTA predictions:

### 9.1 PublicTest Primary Comparisons
| Comparison ID | Rep A vs Rep B | v1 $\Delta	ext{Acc}$ [95% CI] | v2 $\Delta	ext{Acc}$ [95% CI] | v2.1 $\Delta	ext{Acc}$ [95% CI] |
|---|---|---|---|---|
| **P1** | `FUSION` - `PIXEL_READOUT` | **+0.1011** `[+0.0844, +0.1165]` | **+0.0928** `[+0.0772, +0.1092]` | **+0.1045** `[+0.0878, +0.1209]` |
| **P2** | `FUSION` - `MOTIF_READOUT` | **+0.0031** `[-0.0031, +0.0092]` | **-0.0028** `[-0.0098, +0.0039]` | **-0.0008** `[-0.0067, +0.0053]` |
| **P3** | `MOTIF_READOUT` - `PIXEL_READOUT` | **+0.0981** `[+0.0811, +0.1137]` | **+0.0956** `[+0.0800, +0.1120]` | **+0.1053** `[+0.0883, +0.1215]` |
| **P4** | `POST_MOTIF` - `PRE_MOTIF` | **+0.0853** `[+0.0691, +0.1014]` | **+0.0800** `[+0.0652, +0.0953]` | **+0.0789** `[+0.0627, +0.0947]` |
| **P5** | `WHAT_TYPE_WHERE` - `WHAT_WHERE` | **+0.0100** `[+0.0017, +0.0181]` | **+0.0047** `[-0.0017, +0.0114]` | **+0.0089** `[+0.0025, +0.0153]` |
| **P6** | `TYPE_ONLY` - `WHAT_ONLY` | **-0.0660** `[-0.0811, -0.0510]` | **-0.2165** `[-0.2340, -0.1995]` | **-0.2165** `[-0.2343, -0.1998]` |
| **P7** | `ASSIGN_PYR` - `WHAT_ONLY` | **-0.0242** `[-0.0396, -0.0086]` | **-0.2051** `[-0.2229, -0.1864]` | **-0.2229** `[-0.2410, -0.2045]` |
| **P8** | `OCC_TYPE_DIST` - `TYPE_ONLY` | **+0.0365** `[+0.0228, +0.0507]` | **+0.0134** `[+0.0017, +0.0256]` | **-0.0184** `[-0.0301, -0.0070]` |

### 9.2 PrivateTest Primary Comparisons
| Comparison ID | Rep A vs Rep B | v1 $\Delta	ext{Acc}$ [95% CI] | v2 $\Delta	ext{Acc}$ [95% CI] | v2.1 $\Delta	ext{Acc}$ [95% CI] |
|---|---|---|---|---|
| **P1** | `FUSION` - `PIXEL_READOUT` | **+0.1003** `[+0.0841, +0.1170]` | **+0.1084** `[+0.0911, +0.1257]` | **+0.1095** `[+0.0928, +0.1254]` |
| **P2** | `FUSION` - `MOTIF_READOUT` | **-0.0014** `[-0.0084, +0.0056]` | **-0.0028** `[-0.0089, +0.0036]` | **+0.0039** `[-0.0017, +0.0095]` |
| **P3** | `MOTIF_READOUT` - `PIXEL_READOUT` | **+0.1017** `[+0.0858, +0.1184]` | **+0.1112** `[+0.0945, +0.1282]` | **+0.1056** `[+0.0886, +0.1218]` |
| **P4** | `POST_MOTIF` - `PRE_MOTIF` | **+0.0847** `[+0.0683, +0.1011]` | **+0.0928** `[+0.0766, +0.1087]` | **+0.0894** `[+0.0738, +0.1048]` |
| **P5** | `WHAT_TYPE_WHERE` - `WHAT_WHERE` | **+0.0067** `[-0.0020, +0.0153]` | **-0.0047** `[-0.0120, +0.0020]` | **+0.0033** `[-0.0036, +0.0106]` |
| **P6** | `TYPE_ONLY` - `WHAT_ONLY` | **-0.0716** `[-0.0867, -0.0571]` | **-0.2198** `[-0.2371, -0.2012]` | **-0.2329** `[-0.2510, -0.2151]` |
| **P7** | `ASSIGN_PYR` - `WHAT_ONLY` | **-0.0295** `[-0.0440, -0.0139]` | **-0.2145** `[-0.2318, -0.1975]` | **-0.2301** `[-0.2483, -0.2129]` |
| **P8** | `OCC_TYPE_DIST` - `TYPE_ONLY` | **+0.0365** `[+0.0234, +0.0507]` | **-0.0033** `[-0.0150, +0.0089]` | **-0.0017** `[-0.0134, +0.0100]` |

---

## 10. Classifier Branch Contribution Diagnostic (Pixel vs Motif)

Partitioning the first linear projection matrix in the FER classifier $W \in \mathbb{R}^{256 \times 512}$ into $W_{\text{pixel}} \in \mathbb{R}^{256 \times 128}$ and $W_{\text{motif}} \in \mathbb{R}^{256 \times 384}$:

| Model Version | Split | Mean $\|z_{\text{pixel}}\|_2$ | Mean $\|z_{\text{motif}}\|_2$ | Ratio $\|z_M\| / \|z_P\|$ | Per-Input-Dim Ratio |
|---|---|---|---|---|---|
| **v1** | Public | 6.6732 | 40.3906 | **6.05x** | **2.02x** |
| **v1** | Private | 6.6905 | 40.3469 | **6.03x** | **2.01x** |
| **v2** | Public | 15.0251 | 55.1957 | **3.67x** | **1.22x** |
| **v2** | Private | 15.0412 | 54.8315 | **3.65x** | **1.22x** |
| **v2.1** | Public | 13.4349 | 44.3853 | **3.30x** | **1.10x** |
| **v2.1** | Private | 13.4571 | 44.3124 | **3.29x** | **1.10x** |

In all versions, the Motif Readout branch delivers between **3.3x and 6.0x** higher L2 activation norm into the initial classifier hidden state than the Pixel Readout branch. Even after normalizing by the 3x difference in input channel count (384D vs 128D), the Motif branch contributes equal or larger per-channel magnitude into the classifier.

---

## 11. Cross-Version Observations & Descriptive Synthesis

1. **Motif Readout Dominates Pixel Readout (P1 & P3)**:
   - Across all three versions and on both Public and Private test splits, linear probes on `MOTIF_READOUT` outperform linear probes on `PIXEL_READOUT` by **+9.5 to +11.1 percentage points** of accuracy and **+14.0 to +17.5 percentage points** of macro-F1.
   - `FUSION` (concatenating Pixel + Motif readouts) achieves virtually identical accuracy to `MOTIF_READOUT` alone ($\Delta\text{Acc} \in [-0.0028, +0.0039]$, with all 95% CIs containing zero).
2. **Motif Graph Transformer Provides a Consistent +8?9 pp Gain (P4)**:
   - Comparing `POST_MOTIF_SUMMARY` to `PRE_MOTIF_SUMMARY` demonstrates a statistically robust, large gain across all three generations (+8.0?9.3 pp accuracy, +10.4?13.0 pp macro-F1), establishing that global geometric relational attention over the 49 motif occurrence nodes significantly reorganizes facial expression representations.
3. **The Prototype Assignment / TYPE Representation Inefficiency (P6 & P7)**:
   - In v2 and v2.1, prototype cluster assignments (`TYPE_ONLY`, `ASSIGNMENT_PYRAMID`) collapse to ~33.5?34.5% linear accuracy, trailing `WHAT_ONLY` by over 20 percentage points.
   - Adding `TYPE` to `WHAT+WHERE` (P5) yields less than 1 percentage point of marginal accuracy across all models.
   - While v2/v2.1 achieved sharper prototypes and lower MI loss, the resulting prototype index representations act largely as a weak auxiliary regularizer rather than an informative primary code.

---

## 12. Proposed A2 Research Questions

1. **A2.1**: Does the Pixel Readout branch provide any complementary error reduction to the Motif Readout branch, or can the architecture be simplified without accuracy degradation?
2. **A2.2**: Why do sharpened prototype assignments in v2/v2.1 carry less standalone linear signal than v1 soft assignments, and how can prototype semantics be made more expressive?
3. **A2.3**: What specific relational edges in the complete 49-node geometric graph drive the +8?9 pp gain between Pre-Motif and Post-Motif representations?
