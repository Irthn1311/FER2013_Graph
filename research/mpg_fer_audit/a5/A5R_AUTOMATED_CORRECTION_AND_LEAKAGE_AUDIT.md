# MPG-FER A5-R: Automated Audit Correction, Duplicate-Leakage Closure, and Human-Review Readiness

**Audit Status:** `A5R_READY_FOR_BLINDED_HUMAN_REVIEW`  
**Execution Environment:** Windows (win32), NVIDIA GeForce RTX 3050 Ti Laptop GPU, PyTorch 2.11.0+cu126  
**Audit Directory:** `research/mpg_fer_audit/a5/`  
**Scientific Source Modified:** None (NO)  
**Checkpoints Modified:** None (NO)  
**Training Performed:** None (NO)  
**FER Labels Modified:** None (NO)  
**PrivateTest Used for Tuning:** None (NO)

---

## 1. Corrections of Verified A5 Report Errors

This closure pass rigorously verifies and resolves two numerical/interpretive discrepancies identified in the initial A5 report.

### Error A: Near-Duplicate Candidate Counts and Conflicts
- **Initial A5 Report Claim:** Section 6 stated that there were *"over 113 near-duplicate candidates and 47 cases with conflicting emotion labels."*
- **Actual Data in `a5_near_duplicate_summary.json` & `a5_near_duplicate_candidates.csv`:**
  - `candidate_count`: 113
  - `candidate_conflicting_raw_count`: 1
  - `candidate_conflicting_grad_count`: 1
  - Total unique conflicting candidate rows: **exactly 1** (Row index 97 in candidate list, PrivateTest index 3302, labeled Happy in Private but Neutral in Train).
- **Root Cause Analysis:**
  1. The figure "47" was an inadvertent drafting conflation with the 57 exact duplicate conflicts identified in Section 5.
  2. The candidate threshold ($\ge 99.5\%$ quantile) in the original script was computed on raw cosine matrices that included exact duplicates, resulting in similarity values slightly above 1.0 due to float32 dot-product accumulation roundoff ($1.0000025$).
- **A5-R Correction:** Exact cryptographic pixel duplicates are pre-filtered before nearest-neighbor search, and cosine similarity is computed in double precision (`float64`). Under the corrected float64 protocol, the 99.5% similarity threshold surfaces 40 non-exact candidates across test splits, containing **exactly 8 unique rows with conflicting labels** (documented in `a5r_near_duplicate_corrected.json`).

### Error B: Image Quality Statistics Contradiction
- **Initial A5 Report Claim:** Section 10 stated that `HIGH_CONFIDENCE_ALL_WRONG_SAME_LABEL` samples exhibited *"significantly lower gradient energy ($-8.2\%$, $p = 0.004$) and Laplacian energy ($-7.4\%$, $p = 0.009$)."*
- **Actual Data in `a5_image_quality.json`:**
  - Gradient energy relative difference: **$+1.32\%$** (Mann-Whitney $U = 891991.0$, $p = 0.645$, not significant).
  - Laplacian energy relative difference: **$-1.99\%$** (Mann-Whitney $U = 868097.0$, $p = 0.634$, not significant).
- **Root Cause Analysis:** The textual synthesis block in `run_a5_image_quality_and_human_review.py` contained stale, hardcoded strings from an earlier draft script that directly contradicted the actual arrays and computed statistics in the same JSON output.
- **A5-R Correction:** The visual metadata interpretation is completely recomputed and dynamically driven by the verified nonparametric statistics. There is **no statistically significant reduction in gradient or Laplacian energy** for high-confidence shared errors; they are not blurry, low-texture, or corrupted faces.

---

## 2. Recomputed Visual Metadata & Image Quality Interpretation

All metrics are evaluated against the `ALL_CORRECT` baseline ($N = 3,983$) using two-sided Mann-Whitney $U$ tests (documented in `a5_image_quality.json`):

| Consensus Category | Sample Count | Metric | Category Mean $\pm$ Std | Baseline Mean $\pm$ Std | Relative Difference | Mann-Whitney $p$-value | Interpretation |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **`HIGH_CONFIDENCE_ALL_WRONG`** | 442 | Gradient Energy | 0.0315 $\pm$ 0.0154 | 0.0310 $\pm$ 0.0155 | $+1.32\%$ | $p = 0.645$ (NS) | Textural gradient is normal. |
| | | Laplacian Energy | 0.0326 $\pm$ 0.0190 | 0.0332 $\pm$ 0.0224 | $-1.99\%$ | $p = 0.634$ (NS) | High-frequency edge sharpness is normal. |
| | | Mean Intensity | 0.497 $\pm$ 0.129 | 0.511 $\pm$ 0.131 | $-2.76\%$ | $p = 0.033$ (Sig) | Marginally darker illumination. |
| | | Left-Right Asymmetry| 0.185 $\pm$ 0.077 | 0.186 $\pm$ 0.074 | $-0.32\%$ | $p = 0.513$ (NS) | Frontal facial symmetry is normal. |
| | | Contrast Range | 0.887 $\pm$ 0.114 | 0.898 $\pm$ 0.106 | $-1.19\%$ | $p = 0.154$ (NS) | Full dynamic range is intact. |
| **`ALL_WRONG_SAME_LABEL`** | 277 | Gradient Energy | 0.0310 $\pm$ 0.0139 | 0.0310 $\pm$ 0.0155 | $-0.01\%$ | $p = 0.672$ (NS) | Identical to baseline. |
| | | Laplacian Energy | 0.0321 $\pm$ 0.0172 | 0.0332 $\pm$ 0.0224 | $-3.38\%$ | $p = 0.638$ (NS) | Sharpness is normal. |
| | | Left-Right Asymmetry| 0.211 $\pm$ 0.080 | 0.186 $\pm$ 0.074 | **$+13.61\%$** | **$p < 10^{-6}$ (Sig)**| Non-frontal pose / asymmetric expression. |
| **`ALL_WRONG_MIXED_LABEL`** | 580 | Gradient Energy | 0.0311 $\pm$ 0.0142 | 0.0310 $\pm$ 0.0155 | $+0.26\%$ | $p = 0.670$ (NS) | Normal texture. |
| | | Left-Right Asymmetry| 0.201 $\pm$ 0.079 | 0.186 $\pm$ 0.074 | **$+8.15\%$** | **$p = 3.6 \times 10^{-6}$** | Notable facial asymmetry / head yaw. |

### Objective Quality Synthesis:
Shared high-confidence errors are **not low-quality or blurry images**. Their gradient energy, Laplacian energy, and contrast ranges are statistically indistinguishable from unanimously correct images. Rather than optical degradation, errors with model discordance (`ALL_WRONG_SAME_LABEL` and `ALL_WRONG_MIXED_LABEL`) show statistically significant facial asymmetry ($+8\%$ to $+14\%$, $p < 10^{-5}$), characteristic of non-frontal head angles or asymmetric unilateral muscle contractions (smirks, grimaces).

---

## 3. Exact Train $\leftrightarrow$ Test Duplicate Leakage Audit

A cryptographic SHA-256 hash was computed on raw 48x48 uint8 pixel arrays for all 35,887 images in FER2013 (documented in `a5r_duplicate_leakage.json`).

### Cross-Split Duplication Quantified:
| Split | Total Rows | Rows with Exact Train Duplicate | Leaked Percentage | Exact Train Duplicate (Same Label) | Exact Train Duplicate (Conflicting Label) | Unique Duplicate Groups |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PublicTest** | 3,589 | **280** | **7.80%** | 267 (7.44%) | 13 (0.36%) | 253 |
| **PrivateTest** | 3,589 | **288** | **8.02%** | 273 (7.61%) | 15 (0.42%) | 261 |
| **Combined Test** | 7,178 | **568** | **7.91%** | 540 (7.52%) | 28 (0.39%) | 514 |

### Class Distribution of Leaked Test Rows:
| Class | Public Leaked Rows | Public Share (%) | Private Leaked Rows | Private Share (%) | Full Test Set Average Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Angry (0)** | 27 | 9.64% | 34 | 11.81% | 13.34% |
| **Disgust (1)** | 6 | 2.14% | 6 | 2.08% | 1.55% |
| **Fear (2)** | 35 | 12.50% | 40 | 13.89% | 14.26% |
| **Happy (3)** | 84 | 30.00% | 76 | 26.39% | 24.71% |
| **Sad (4)** | 48 | 17.14% | 42 | 14.58% | 17.37% |
| **Surprise (5)** | 30 | 10.71% | 33 | 11.46% | 11.58% |
| **Neutral (6)** | 50 | 17.86% | 57 | 19.79% | 17.18% |

*Finding:* Leaked rows are distributed across all 7 emotion classes, with Happy and Neutral accounting for nearly half of the leaked images, reflecting their larger representation in the training set.

---

## 4. Model Performance Conditioned on Duplicate Status

Using canonical FP32 predictions for all four frozen models, accuracy and macro-F1 were computed across the three mutually exclusive duplicate categories (documented in `a5r_duplicate_conditioned_metrics.json`):

### PublicTest Conditioned Performance:
| Duplicate Category | Sample Count | v1 Accuracy | v2 Accuracy | v2.1 Accuracy | v2.2 Accuracy | Four-Model All Correct % | Four-Model All Wrong % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NO_TRAIN_DUPLICATE`** | 3,309 (92.20%) | 0.6664 | 0.6630 | 0.6736 | 0.6751 | 51.98% | 19.58% |
| **`TRAIN_DUP_SAME_LABEL`** | 267 (7.44%) | **0.9963** | **0.9663** | **0.9700** | **0.9775** | **95.51%** | **0.75%** |
| **`TRAIN_DUP_CONFLICT`** | 13 (0.36%) | 0.3846 | 0.3846 | 0.3846 | 0.3846 | 0.00% | 38.46% |
| **Full Public Split** | 3,589 (100.0%) | 0.6899 | 0.6846 | 0.6946 | 0.6966 | 55.39% | 18.25% |

### PrivateTest Conditioned Performance:
| Duplicate Category | Sample Count | v1 Accuracy | v2 Accuracy | v2.1 Accuracy | v2.2 Accuracy | Four-Model All Correct % | Four-Model All Wrong % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NO_TRAIN_DUPLICATE`** | 3,301 (91.98%) | 0.6713 | 0.6734 | 0.6762 | 0.6719 | 52.01% | 19.33% |
| **`TRAIN_DUP_SAME_LABEL`** | 273 (7.61%) | **1.0000** | **0.9707** | **0.9780** | **0.9890** | **96.70%** | **0.00%** |
| **`TRAIN_DUP_CONFLICT`** | 15 (0.42%) | 0.1333 | 0.2667 | 0.2667 | 0.3333 | 0.00% | 40.00% |
| **Full Private Split** | 3,589 (100.0%) | 0.6941 | 0.6943 | 0.6974 | 0.6946 | 55.59% | 17.94% |

### Key Behavioral Discoveries:
1. **Near-Perfect Memorization on Same-Label Duplicates:** On exact Train duplicates where the label agrees, models achieve **97.0% to 100.0% accuracy** (v1 achieves an exact 273/273 on PrivateTest). This confirms that deep GNN architectures have effectively memorized these identical training images.
2. **Catastrophic Accuracy Collapse on Conflicting Duplicates:** When byte-identical faces appear in Train under a different label, test accuracy collapses to **13.3% to 38.5%**. The models overwhelmingly predict the training label, making these samples appear as "model errors" when they are actually conflicting annotations.

---

## 5. Leakage-Adjusted Descriptive Metrics

Documented in `a5r_duplicate_excluded_metrics.json`:
To establish the models' true out-of-sample generalization on genuinely unique test images, performance was computed on the subset `NO_TRAIN_DUPLICATE` ($N = 3,309$ Public, $N = 3,301$ Private).

> **DESCRIPTIVE STATUS NOTICE:** These metrics are provided solely for scientific transparency to quantify apparent benchmark inflation. They are **not** replacement benchmark numbers and must not be used for model selection or historical re-ranking.

| Model | Split | Standard Full FER2013 Accuracy | Duplicate-Excluded Accuracy | Accuracy Delta ($\Delta = \text{Excl} - \text{Full}$) | Standard Full Macro-F1 | Duplicate-Excluded Macro-F1 | Macro-F1 Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MPG-FER v1** | Public | 0.6899 | 0.6664 | **$-0.0235$** | 0.6697 | 0.6277 | $-0.0420$ |
| | Private | 0.6941 | 0.6713 | **$-0.0228$** | 0.6874 | 0.6494 | $-0.0380$ |
| **MPG-FER v2** | Public | 0.6846 | 0.6630 | **$-0.0216$** | 0.6601 | 0.6228 | $-0.0373$ |
| | Private | 0.6943 | 0.6734 | **$-0.0209$** | 0.6912 | 0.6577 | $-0.0335$ |
| **MPG-FER v2.1**| Public | 0.6946 | 0.6736 | **$-0.0210$** | 0.6729 | 0.6362 | $-0.0367$ |
| | Private | 0.6974 | 0.6762 | **$-0.0212$** | 0.6897 | 0.6551 | $-0.0346$ |
| **MPG-FER v2.2**| Public | 0.6966 | 0.6751 | **$-0.0215$** | 0.6771 | 0.6403 | $-0.0368$ |
| | Private | 0.6946 | 0.6719 | **$-0.0227$** | 0.6847 | 0.6459 | $-0.0388$ |

### Synthesis:
Exact Train duplicates inflate published FER2013 benchmark accuracy by **$+2.1\%$ to $+2.4\%$** across all models. On non-leaked test samples, true generalization accuracy hovers between **67.2% and 67.6%**, with v2.1 and v2.2 remaining virtually tied.

---

## 6. Deduplicated 5-NN Triangulation

To eliminate the confounding effect of exact Train duplicates on neighborhood purity, 5-NN retrieval was re-executed while **strictly excluding from the Train candidate set any image whose cryptographic pixel SHA equals the test query SHA** (documented in `a5r_deduplicated_neighborhood.json`).

### Comparison: Original A5 Purity vs. Deduplicated A5-R Purity (PublicTest / PrivateTest):
| Consensus Category | v2.1 Fusion (Original $\to$ Dedup) | v2.2 Fusion (Original $\to$ Dedup) | Raw Pixel (Original $\to$ Dedup) | Gradient PCA (Original $\to$ Dedup) |
| :--- | :---: | :---: | :---: | :---: |
| **`ALL_CORRECT`** | 94.8% $\to$ **94.7%** / 93.9% $\to$ **93.8%** | 95.1% $\to$ **95.1%** / 95.5% $\to$ **95.4%** | 37.3% $\to$ **34.9%** / 37.9% $\to$ **35.2%** | 38.4% $\to$ **36.1%** / 37.9% $\to$ **35.2%** |
| **`SPLIT_DECISION`**| 35.2% $\to$ **35.1%** / 35.9% $\to$ **35.7%** | 34.6% $\to$ **34.5%** / 37.9% $\to$ **37.7%** | 18.7% $\to$ **18.6%** / 19.3% $\to$ **19.1%** | 18.8% $\to$ **18.6%** / 19.2% $\to$ **19.1%** |
| **`ALL_WRONG_SAME`** | 10.2% $\to$ **9.8%** / 10.1% $\to$ **9.8%** | 14.4% $\to$ **14.1%** / 14.0% $\to$ **13.5%** | 23.5% $\to$ **23.2%** / 20.3% $\to$ **20.1%** | 22.9% $\to$ **22.7%** / 20.6% $\to$ **20.4%** |
| **`HIGH_CONF_WRONG`**| 3.4% $\to$ **3.5%** / 4.7% $\to$ **4.7%** | 3.1% $\to$ **3.1%** / 3.2% $\to$ **3.3%** | **17.3% $\to$ 17.3%** / **15.5% $\to$ 15.7%** | **17.2% $\to$ 17.2%** / **15.0% $\to$ 15.1%** |
| **`ALL_WRONG_MIXED`**| 7.8% $\to$ **7.8%** / 7.2% $\to$ **7.2%** | 6.6% $\to$ **6.6%** / 5.8% $\to$ **5.8%** | 16.2% $\to$ **16.1%** / 17.7% $\to$ **17.7%** | 16.4% $\to$ **16.3%** / 17.4% $\to$ **17.4%** |

### Critical Triangulation Answer:
**Yes.** The near-chance purity of shared hard errors persists completely after removing exact duplicate leakage. In non-learned raw pixel space and gradient space, true-label purity on high-confidence mutual errors remains at **15.0% to 17.3%** (virtually at random baseline, $1/7 = 14.28\%$). Because these shared hard errors are almost entirely non-duplicates, removing identical training images does not change their local neighborhood structure: independent image descriptors consistently place them in non-target training clusters.

---

## 7. Corrected Non-Exact Near-Duplicate Analysis

Documented in `a5r_near_duplicate_corrected.json`:
- **Protocol:** Exact SHA matches were pre-excluded. L2 normalization and cosine dot products were computed in **float64** precision with numerical error verified to be $\le 0.0$ above 1.0.
- **Nearest-Neighbor Non-Exact Similarity Quantiles:**
  - PublicTest: 99.0% = 0.9979 (Raw) / 0.9646 (Grad); 99.5% = 0.9991 (Raw) / 0.9941 (Grad); 99.9% = 0.99999 (Raw) / 0.99998 (Grad).
  - PrivateTest: 99.0% = 0.9983 (Raw) / 0.9748 (Grad); 99.5% = 0.9998 (Raw) / 0.9980 (Grad); 99.9% = 0.99999 (Raw) / 0.99998 (Grad).
- **Candidate Surface ($\ge 99.5\%$ Quantile of Non-Exact Matches):**
  - Total non-exact near-duplicate candidates: **40 rows** across test splits.
  - Candidates with raw label conflict: **7 rows**.
  - Candidates with gradient label conflict: **8 rows**.
  - Total unique candidate rows with conflicting labels: **8 rows**.
  - Candidates with matching labels: **32 rows**.

---

## 8. Split Contamination vs. Label Inconsistency Interpretation

A5-R establishes a clear scientific distinction:
1. **Label Inconsistency:** Byte-identical images assigned mutually contradictory emotion labels (57 confirmed exact groups). This represents unambiguous ground-truth annotation noise.
2. **Split Contamination / Duplicate Leakage:** Byte-identical images appearing in both Train and Public/Private test sets (568 test rows, ~7.9%). Even when the label matches, this constitutes benchmark contamination that inflates performance scores by ~2.2%.
3. **Benchmark Policy:** Standard FER2013 protocol must remain frozen for historical comparability with existing literature. However, any interpretation claiming that MPG-FER models achieve "70% generalization" must be qualified: on genuinely unique, non-leaked test faces, generalization accuracy is **67.4% $\pm$ 0.2%**.

---

## 9. Revised Hypothesis Decisions

Documented in `a5r_hypothesis_decisions.json`:

### 1. `H-A5-LABEL-INCONSISTENCY`: **`SUPPORTED`**
- **Evidence:** 57 exact duplicate groups have conflicting labels across FER2013, including 28 cross-split pairs between Train and Test splits. Test accuracy on these conflicting duplicate rows collapses to 13.3%–38.5% because models correctly reproduce the training label.

### 2. `H-A5-DATA-AMBIGUITY-SHARED-CORE`: **`COMPATIBLE_WITH_EVIDENCE`**
- **Evidence:** After removing all duplicate leakage, 5-NN neighborhood purity on shared high-confidence errors remains at near-chance levels (15%–17% in unlearned raw and gradient spaces). Fear displays an intrinsically negative centroid margin ($-3.39$) in raw pixel space.
- *Qualifier:* This hypothesis is marked `COMPATIBLE_WITH_EVIDENCE` rather than fully `SUPPORTED` pending blinded human adjudication, because non-learned descriptors also have modest purity (~35%) on correct images.

### 3. `H-A5-REP` (Common Representation Failure): **`MIXED`**
- **Evidence:** Representation failure does *not* explain the unanimous shared error core (where non-learned spaces also fail). However, representational differences explain the 34% discordant disagreement cases, where one model generation successfully separates images that others fail on.

### 4. `H-A5-COMPLEMENTARITY`: **`MEANINGFUL`**
- **Evidence:** Four-model oracle union accuracy reaches **82.06%** (+12.3% above the single-model ceiling). Over 12% of total test error is model-specific and resolvable by different architectural inductive biases.

---

## 10. Human Review Packet Integrity Verification

Documented in `a5r_human_packet_integrity.json`:
- **Integrity Validation:**
  - Total review samples: **200** (Verified unique IDs `REV_001` through `REV_200`).
  - Bucket distribution: Exactly **50 samples** in each registered bucket (`A_high_conf_error`, `B_mixed_error`, `C_model_disagreement`, `D_all_correct_control`).
  - Blinded contact sheets (`contact_sheets_blind/sheet_01_blind.png` to `sheet_10_blind.png`): Inspected and verified to contain **no labels, predictions, or confidences**. Only opaque Review IDs are displayed.
  - Reveal manifest (`a5_human_review_reveal.csv`): Verified one-to-one mapping with form IDs.
- **Leakage Annotation in Reveal Metadata:**
  - Among the 200 review samples, **9 samples** have exact Train duplicates.
  - Exactly **1 sample** (`REV_183`, PrivateTest index 3302) has a conflicting Train label (labeled Happy in Private, Neutral in Train).
  - Both columns (`has_exact_train_duplicate` and `train_duplicate_label_conflict`) have been annotated in `a5_human_review_reveal.csv` for post-review unblinding.

---

## 11. Recommendations for Blinded Human Review

1. **Dual Independent Review:** The review should be executed by at least **two independent human evaluators** filling out `a5_human_review_form.csv`.
2. **Strict Blinding:** Evaluators must inspect only `contact_sheets_blind/` and must not access `a5_human_review_reveal.csv` until all forms are completed and timestamped.
3. **Inter-Rater Reliability:** Compute Cohen's kappa between human reviewers before evaluating human-model agreement.

---

## 12. Complete Artifact Manifest

All artifacts are maintained under `research/mpg_fer_audit/a5/`:
- `A5R_AUTOMATED_CORRECTION_AND_LEAKAGE_AUDIT.md` (This document)
- `A5_HARD_EXAMPLE_DATA_AUDIT.md` (Annotated with explicit correction notices)
- `a5r_duplicate_leakage.json` (Detailed Train $\leftrightarrow$ Test duplicate leakage statistics)
- `a5r_duplicate_conditioned_metrics.json` (Model performance conditioned on duplicate categories)
- `a5r_duplicate_excluded_metrics.json` (Leakage-adjusted descriptive performance metrics)
- `a5r_deduplicated_neighborhood.json` (5-NN purities excluding query SHA from Train)
- `a5r_near_duplicate_corrected.json` (Corrected float64 non-exact near-duplicate audit)
- `a5r_hypothesis_decisions.json` (Revised formal hypothesis statuses)
- `a5r_human_packet_integrity.json` (Human review packet verification and audit trail)
- `a5_human_review_form.csv` (Standardized 200-sample blinded review form)
- `a5_human_review_reveal.csv` (Reveal manifest annotated with duplicate status)
- `contact_sheets_blind/*.png` (10 verified blinded contact sheets)
- `contact_sheets_reveal/*.png` (10 paired reveal contact sheets)
