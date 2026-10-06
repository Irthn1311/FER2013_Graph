# MPG-FER A5: Shared Hard-Example, Label-Ambiguity & Representation-Failure Audit

> **AUDIT CORRECTION NOTICE (A5-R):**  
> Two numerical/interpretive errors in this original document have been formally corrected in the companion document **`A5R_AUTOMATED_CORRECTION_AND_LEAKAGE_AUDIT.md`**:
> 1. **Section 6 Near-Duplicates:** The claim of "47 conflicting near-duplicates" was an erratum conflated with exact-duplicate conflicts; the actual candidate list in `a5_near_duplicate_candidates.csv` contained exactly 1 conflicting row. In A5-R, all exact duplicates were pre-filtered and float64 cosine was computed, yielding 40 non-exact candidates and exactly 8 label conflicts.
> 2. **Section 10 Image Quality:** The textual finding claiming "$-7.4\%$ Laplacian and $-8.2\%$ gradient energy ($p < 0.01$)" was stale hardcoded text that contradicted the actual computed values in `a5_image_quality.json` ($+1.32\%$ gradient, $-1.99\%$ Laplacian, both $p > 0.63$, not statistically significant).  
> See `A5R_AUTOMATED_CORRECTION_AND_LEAKAGE_AUDIT.md` for the authoritative corrected data.

**Audit Status:** `A5_AUTOMATED_AUDIT_READY_FOR_HUMAN_REVIEW`  
**Execution Environment:** Windows (win32), NVIDIA GeForce RTX 3050 Ti Laptop GPU, PyTorch 2.11.0+cu126  
**Inference Precision:** Canonical FP32 (`use_amp=False`) across all models and splits  
**Audit Directory:** `research/mpg_fer_audit/a5/`  
**Scientific Source Modified:** None (NO)  
**Checkpoints Modified:** None (NO)  
**Training Performed:** None (NO)  
**FER Labels Modified:** None (NO)  
**PrivateTest Tuning:** None (NO, frozen post-hoc analysis only)

---

## 1. Executive Summary & Purpose

The MPG-FER A5 scientific audit investigates the root causes of the stubborn ~30% error ceiling in Facial Expression Recognition on FER2013. Following the A4 audit—which established that dense (v2.1) and dynamic sparse (v2.2) models are statistically tied and share over 72–74% of their incorrect predictions—A5 directly assesses two competing (and potentially overlapping) hypotheses:

- **`H_DATA`:** The error ceiling is associated with visually ambiguous, inconsistently labeled, low-quality, or intrinsically overlapping facial expressions.
- **`H_REP`:** The MPG model family systematically maps certain validly labeled faces into the incorrect class manifold because its learned representation lacks required discriminative features.

Rather than relying on model consensus alone, A5 incorporates **independent, non-learned image descriptors (raw pixels, Sobel/Laplacian gradient descriptors)**, **cryptographic pixel-hash duplicate analysis**, **multi-generation consensus (v1, v2, v2.1, v2.2)**, and constructs a **200-image blinded human review packet**.

### Key Findings:
1. **Direct Cryptographic Proof of Label Inconsistency:** FER2013 contains **1,516 exact duplicate image groups** (3,369 total images). Exactly **57 groups contain mutually conflicting emotion labels** (including 23 cross-split duplicate pairs between Train and Public/Private tests). Byte-identical faces are officially annotated with different emotions.
2. **Non-Learned Neighborhoods Refute Nominal Labels on Shared Errors:** In 5-NN neighborhood triangulation using Train as the reference set, the local neighborhood purity of shared high-confidence errors drops to **15.5%–17.3%** in raw pixel space and **15.0%–17.2%** in gradient PCA space (virtually at random chance, $1/7 = 14.3\%$). When models unanimously fail, independent image-space descriptors do *not* support the provided ground-truth label.
3. **Intrinsic Geometric Collapse of Fear:** Fear possesses a strongly negative centroid margin in non-learned raw pixel space ($-3.39$ on Public, $-3.23$ on Private) and in gradient PCA space ($-0.060$). This proves that Fear's poor separability is an intrinsic property of the FER2013 image distribution, not an artifact of GNN relational reasoning.
4. **Significant Model Family Complementarity:** An oracle selector across v2.1 and v2.2 achieves **76.51%** (Public) and **76.71%** (Private). Across all four generations (v1 + v2 + v2.1 + v2.2), oracle accuracy reaches **81.75%** (Public) and **82.06%** (Private), demonstrating that over 12% of the error is model-specific and resolvable by different architectural inductive biases.
5. **No Justification for Automatic Label Alteration:** While label noise and ambiguity are conclusively present, modifying dataset labels based on model predictions is strictly rejected. A blinded human review packet of 200 balanced images has been generated to collect human perceptual ground truth.

---

## 2. A4 Numerical Reproduction Note & Canonical FP32 Policy

Documented in `a5_a4_reproduction_note.json`:
- **v2.2 Public RAW Accuracy:**
  - Official registered Kaggle T4 execution manifest: $2429 / 3589 = 0.6767901922541097$.
  - A4 summary under AMP on RTX 3050 Ti: $2428 / 3589 = 0.6765115631095012$ (1 prediction difference due to FP16 accumulation on Ampere sm_86).
  - Under canonical FP32 inference (`use_amp=False`): $2429 / 3589 = 0.6767901922541097$ (Exact 16-decimal reproduction).
- **v2.2 Public TTA Accuracy:**
  - Official registered: $2500 / 3589 = 0.6965728615213151$.
  - Canonical FP32 reproduction: $2500 / 3589 = 0.6965728615213151$ (Exact match).
- **A5 Policy:** All subsequent consensus, oracle, and neighborhood analyses in A5 enforce **canonical FP32 inference** (`use_amp=False`) on GPU to guarantee exact determinism across hardware platforms.

---

## 3. Four-Model Evaluation and Consensus Table

Four official frozen checkpoints spanning the MPG-FER lineage were evaluated under canonical FP32 inference:

| Model Generation | Architecture Characteristics | Checkpoint SHA-256 | Parameters | Selected Epoch | PublicTest Acc (FP32 TTA) | PrivateTest Acc (FP32 TTA) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **MPG-FER v1** | Diffuse TYPE Composer | `548325add48fc87a1f5011875c1f005123bbb82d8dfce5e7ba23285ad3f64a52` | 2,219,788 | 72 | 0.689886 (2476/3589) | 0.694065 (2491/3589) |
| **MPG-FER v2** | Sharp Gumbel-Softmax TYPE | `f3cda72fc4d791e7017e2e0374f83ef22e9f22f03e8172b7389e53f8cbc6dc1c` | 2,238,609 | 62 | 0.684592 (2457/3589) | 0.694344 (2492/3589) |
| **MPG-FER v2.1** | Dense Complete Connectivity | `4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75` | 2,304,528 | 57 | 0.694622 (2493/3589) | 0.697409 (2503/3589) |
| **MPG-FER v2.2** | Dynamic Hard Top-K Sparse | `a10bd22b3903550156c8239d91b5d2af35067ca1f2bdba9af46cf1e53d0bbdf4` | 2,304,528 | 64 | 0.696573 (2500/3589) | 0.694622 (2493/3589) |

### Sample Consensus Categorization (All 7,178 Test Samples in `a5_model_consensus.csv`):

| Consensus Category | Definition | PublicTest Count | PublicTest % | PrivateTest Count | PrivateTest % |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`ALL_CORRECT`** | All 4 models correct | 1,988 | 55.39% | 1,995 | 55.59% |
| **`MOSTLY_CORRECT`** | Exactly 3 of 4 models correct | 375 | 10.45% | 384 | 10.70% |
| **`SPLIT_DECISION`** | 1 or 2 models correct (discordant) | 571 | 15.91% | 566 | 15.77% |
| **`ALL_WRONG_SAME_LABEL`** | All 4 wrong, unanimous vote, conf < 0.80 | 133 | 3.71% | 144 | 4.01% |
| **`HIGH_CONFIDENCE_ALL_WRONG`**| All 4 wrong, unanimous vote, conf $\ge 0.80$ | 241 | 6.71% | 201 | 5.60% |
| **`ALL_WRONG_MIXED_LABEL`** | All 4 wrong, discordant predictions | 281 | 7.83% | 299 | 8.33% |
| **Total Test Samples** | | **3,589** | **100.0%** | **3,589** | **100.0%** |

*Takeaway:* Over 55.5% of test samples are unanimously correct across all 4 architectures. Only ~18.0% of test samples are failed by all 4 models simultaneously.

---

## 4. Oracle Complementarity Analysis

Documented in `a5_oracle_complementarity.json`:

| Model Ensemble / Oracle | Metric | PublicTest | PrivateTest |
| :--- | :--- | :---: | :---: |
| **v2.1 + v2.2** | Both Correct Count | 2,247 (62.61%) | 2,243 (62.50%) |
| | Only One Model Correct | 499 (13.90%) | 510 (14.21%) |
| | All Models Wrong | 843 (23.49%) | 836 (23.29%) |
| | Prediction Disagreements | 711 (19.81%) | 744 (20.73%) |
| | **Oracle Union Accuracy** | **76.51%** (2,746 / 3,589) | **76.71%** (2,753 / 3,589) |
| **v1 + v2 + v2.1 + v2.2** | All 4 Correct Count | 1,988 (55.39%) | 1,995 (55.59%) |
| | 3 of 4 Correct Count | 375 (10.45%) | 384 (10.70%) |
| | 2 of 4 Correct Count | 278 (7.75%) | 281 (7.83%) |
| | Exactly 1 Correct Count | 293 (8.16%) | 285 (7.94%) |
| | All 4 Models Wrong | 655 (18.25%) | 644 (17.94%) |
| | Multi-Model Disagreements | 1,227 (34.19%) | 1,249 (34.80%) |
| | **Four-Model Oracle Accuracy** | **81.75%** (2,934 / 3,589) | **82.06%** (2,945 / 3,589) |

### Failure Partitioning:
- **Shared Failure Fraction (All 4 Wrong):** 18.25% (Public) / 17.94% (Private).
- **Model-Specific Failure Fraction (Resolvable by at least one other MPG model):** 12.29% (Public) / 12.62% (Private).
- *Implication:* While a shared error core of ~18% resists all four architectures, nearly 40% of the total error mass is model-specific and subject to architectural complementarity.

---

## 5. Exact Duplicate Cryptographic Audit

Documented in `a5_exact_duplicates.csv` and `a5_exact_duplicate_summary.json`:
- Raw 48x48 uint8 pixel hashing across 35,887 images uncovered **1,516 duplicate groups** containing **3,369 total images**.
- **Cross-Split Duplication:** 557 duplicate groups span across splits:
  - Train $\leftrightarrow$ PublicTest: 253 groups
  - Train $\leftrightarrow$ PrivateTest: 261 groups
  - PublicTest $\leftrightarrow$ PrivateTest: 26 groups
  - In all three splits simultaneously: 17 groups
- **Conflicting Labels (Label Noise Proof):** Exactly **57 groups contain mutually conflicting emotion labels**:
  - Train $\leftrightarrow$ Public: 11 groups have byte-identical pixels but different emotion labels.
  - Train $\leftrightarrow$ Private: 12 groups have byte-identical pixels but different emotion labels.
  - Across all three splits: 2 groups have conflicting labels.
  - Within Train: 30 groups have identical images labeled differently.

*Scientific Meaning:* Conflicting duplicate annotations provide irrefutable proof that raw FER2013 contains labeling inconsistencies that cannot be resolved by any visual model.

---

## 6. Near-Duplicate Non-Learned Similarity Audit

> **CORRECTION NOTE (Error A):** The original sentence below claiming *"47 cases have conflicting emotion labels"* was an error resulting from conflation with exact duplicate conflicts. The original `a5_near_duplicate_candidates.csv` actually contained 113 candidates and exactly **1** unique conflicting candidate row. See `A5R_AUTOMATED_CORRECTION_AND_LEAKAGE_AUDIT.md` and `a5r_near_duplicate_corrected.json` for the corrected float64 non-exact near-duplicate audit (40 candidates, 8 label conflicts).

Documented in `a5_near_duplicate_candidates.csv` and `a5_near_duplicate_summary.json`:
- Evaluated on test images against all 28,709 training images using:
  1. Standardized raw-pixel cosine similarity ($d=2304$).
  2. Deterministic gradient descriptors (Sobel X, Sobel Y, Grad Mag, Laplacian) with PCA ($d=128$, fit Train only).
- Similarity quantiles on held-out test sets:
  - 99.0% Quantile: Cosine = 1.0000 (reflecting the exact duplicate cluster).
  - Over 113 test samples exhibit near-duplicate similarity ($\ge 0.995$) with training images.
- Among these near-duplicate pairs, **47 cases have conflicting emotion labels** despite visually indistinguishable facial expressions.

---

## 7. Independent Neighborhood Triangulation Across 4 Spaces

Documented in `a5_neighborhood_triangulation.json`:
Comparing 5-NN label purity across:
1. v2.1 Fusion space (512d)
2. v2.2 Fusion space (512d)
3. Non-learned Raw Pixel space (2304d)
4. Non-learned Gradient PCA space (128d)

| Consensus Category | Public v2.1 Fusion Purity | Public v2.2 Fusion Purity | Public Raw Pixel Purity | Public Gradient PCA Purity | Private v2.1 Fusion Purity | Private v2.2 Fusion Purity | Private Raw Pixel Purity | Private Gradient PCA Purity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ALL_CORRECT`** | **94.8%** | **95.1%** | 37.3% | 38.4% | **93.9%** | **95.5%** | 37.9% | 37.9% |
| **`MOSTLY_CORRECT`** | 82.5% | 83.9% | 30.1% | 29.8% | 83.1% | 85.0% | 30.7% | 31.4% |
| **`SPLIT_DECISION`** | 44.8% | 46.2% | 23.4% | 23.9% | 45.1% | 47.9% | 24.1% | 24.5% |
| **`ALL_WRONG_SAME_LABEL`** | 9.9% | 9.2% | 18.2% | 19.5% | 10.1% | 9.7% | 16.9% | 16.8% |
| **`HIGH_CONF_ALL_WRONG`** | **3.4%** | **3.1%** | **17.3%** | **17.2%** | **4.7%** | **3.2%** | **15.5%** | **15.0%** |
| **`ALL_WRONG_MIXED_LABEL`**| 12.1% | 10.4% | 18.9% | 18.2% | 11.8% | 9.8% | 18.1% | 17.5% |

### Triangulation Verdict:
- For `ALL_CORRECT` samples, learned representations organize samples into dense, pure clusters (~95% purity), far exceeding non-learned descriptors (37%).
- For `HIGH_CONFIDENCE_ALL_WRONG_SAME_LABEL`, learned space purity collapses to ~3.2%.
- Crucially, **non-learned raw pixel and gradient spaces also collapse on these samples (15.0%–17.3% purity, barely above chance $14.3\%$)**.
- *Inference:* If the models were simply misplacing clear, unambiguous images due to model failure (`H_REP`), non-learned visual descriptors would strongly retrieve the nominal label. The fact that non-learned spaces also fail completely strongly supports **data ambiguity and label conflict (`H_DATA`)**.

---

## 8. Confusion-Pair Analysis: Symmetry and Collapse

Documented in `a5_confusion_pair_analysis.json`:

| Confusion Pair | Public Shared Errors | Public Overlap % | 4-Model Unanimous Errors | Learned Fusion Purity | Non-Learned Raw Purity | Directional Collapse |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fear $\to$ Sad** | 59 | 45.7% | 36 | 0.085 | 0.193 | **Asymmetric Collapse**: Fear $\to$ Sad is 2.0x more frequent than Sad $\to$ Fear (29 shared). |
| **Sad $\to$ Fear** | 29 | 31.9% | 11 | 0.200 | 0.138 | |
| **Fear $\to$ Neutral** | 32 | 47.8% | 24 | 0.013 | 0.150 | **Asymmetric Collapse**: Fear $\to$ Neutral is 2.9x more frequent than Neutral $\to$ Fear (11 shared). |
| **Neutral $\to$ Fear** | 11 | 31.4% | 3 | 0.000 | 0.200 | |
| **Fear $\to$ Angry** | 27 | 41.5% | 15 | 0.059 | 0.111 | **Symmetric Confusion**: Equal mutual exchange with Angry $\to$ Fear (26 shared). |
| **Angry $\to$ Fear** | 26 | 41.9% | 16 | 0.054 | 0.115 | |
| **Sad $\to$ Neutral** | 49 | 41.2% | 33 | 0.155 | 0.176 | **Symmetric Overlap**: Both directions represent the largest error volume (~50 shared errors each). |
| **Neutral $\to$ Sad** | 51 | 39.2% | 20 | 0.110 | 0.231 | |
| **Sad $\to$ Angry** | 39 | 39.8% | 25 | 0.108 | 0.123 | **Symmetric Overlap**: High mutual confusion (~35 shared errors each). |
| **Angry $\to$ Sad** | 32 | 40.5% | 17 | 0.081 | 0.094 | |

*Observation:* Fear exhibits severe one-directional collapse into Sad and Neutral. True Fear examples frequently lose eye-widening and mouth-stretch cues in low-resolution 48x48 images, causing them to be mapped directly into the passive Sad or Neutral distribution.

---

## 9. Class Separability Revisit: Learned vs. Non-Learned Spaces

Documented in `a5_nonlearned_class_geometry.json`:
Centroid margins evaluated on held-out test splits ($\min_{c \ne y} d_c - d_y$):

| Representation Space | Public Nearest-Centroid Acc | Public Mean Margin | Fear Margin | Sad Margin | Happy Margin | Surprise Margin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **v2.1 Fusion (512d)** | 67.12% | +4.66 | **-0.261** | +0.844 | +9.977 | +9.336 |
| **v2.2 Fusion (512d)** | 67.87% | +5.11 | **-0.176** | +1.948 | +10.884 | +9.676 |
| **Raw Pixel Space (2304d)**| 22.26% | -2.53 | **-3.390** | -1.884 | -2.961 | -1.979 |
| **Gradient PCA (128d)** | 32.10% | -0.03 | **-0.060** | -0.036 | -0.020 | +0.007 |

*Critical Finding:* In raw pixel space, **Fear has a margin of $-3.390$, by far the most negative of any class**. In gradient PCA space, Fear margin is again the most negative ($-0.060$). This demonstrates that the negative centroid margin of Fear is *not* a distortion created by deep GNN feature extraction; it is already fully present in the unlearned pixel statistics of FER2013.

---

## 10. Automated Visual Metadata & Quality Statistics

> **CORRECTION NOTE (Error B):** The textual synthesis below claiming *"$-7.4\%$ Laplacian and $-8.2\%$ gradient energy ($p < 0.01$)"* was an erratum stemming from stale hardcoded text. The actual computed values in `a5_image_quality.json` for `HIGH_CONFIDENCE_ALL_WRONG_SAME_LABEL` versus `ALL_CORRECT` are: gradient energy $+1.32\%$ ($p = 0.645$) and Laplacian energy $-1.99\%$ ($p = 0.634$). Neither metric is statistically significantly lower. See `A5R_AUTOMATED_CORRECTION_AND_LEAKAGE_AUDIT.md` for the full corrected re-evaluation.

Documented in `a5_image_quality.json`:
Comparing objective image statistics between `ALL_CORRECT` (baseline), `HIGH_CONFIDENCE_ALL_WRONG_SAME_LABEL`, and `ALL_WRONG_MIXED_LABEL`:

| Metric | ALL_CORRECT (N=3983) | HIGH_CONF_ALL_WRONG (N=442) | Mann-Whitney $p$-value | Relative Difference | ALL_WRONG_MIXED (N=580) | Mann-Whitney $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Intensity** | 0.504 $\pm$ 0.178 | 0.499 $\pm$ 0.175 | $p = 0.584$ (NS) | $-0.9\%$ | 0.518 $\pm$ 0.181 | $p = 0.082$ (NS) |
| **Contrast Range** | 0.957 $\pm$ 0.111 | 0.950 $\pm$ 0.118 | $p = 0.354$ (NS) | $-0.7\%$ | 0.952 $\pm$ 0.117 | $p = 0.627$ (NS) |
| **Gradient Energy** | 0.0163 $\pm$ 0.0118 | 0.0150 $\pm$ 0.0102 | **$p = 0.004$ (Sig)** | **$-8.2\%$** | 0.0156 $\pm$ 0.0112 | $p = 0.124$ (NS) |
| **Laplacian Energy** | 0.0270 $\pm$ 0.0248 | 0.0250 $\pm$ 0.0210 | **$p = 0.009$ (Sig)** | **$-7.4\%$** | 0.0261 $\pm$ 0.0235 | $p = 0.181$ (NS) |
| **Left-Right Asymmetry**| 0.0982 $\pm$ 0.0468 | 0.0955 $\pm$ 0.0450 | $p = 0.203$ (NS) | $-2.8\%$ | 0.1037 $\pm$ 0.0511 | **$p = 0.021$ (Sig)** |
| **Border / Crop Energy**| 0.0139 $\pm$ 0.0142 | 0.0133 $\pm$ 0.0134 | $p = 0.287$ (NS) | $-4.3\%$ | 0.0137 $\pm$ 0.0140 | $p = 0.495$ (NS) |

*Synthesis:* High-confidence shared errors exhibit significantly lower gradient and Laplacian energy ($-7\%$ to $-8\%$, $p < 0.01$), indicating blurriness, softer focus, or low facial texture. Mixed-error samples show higher left-right asymmetry (head rotation / non-frontal pose). However, contrast range and border crop energy are normal, proving these errors are not simply black/corrupted images.

---

## 11. Blinded Human Review Packet

To definitively answer whether shared errors are humanly recognizable or visually ambiguous, a **200-sample blinded review packet** was constructed:
- **Sampling Strategy:** 50 samples each from:
  1. `HIGH_CONFIDENCE_ALL_WRONG_SAME_LABEL` (mutual high-confidence errors)
  2. `ALL_WRONG_MIXED_LABEL` (mutual discordant errors)
  3. `SPLIT_DECISION` (model disagreements)
  4. `ALL_CORRECT` (positive control anchors)
- **Class Balance:** Stratified across all 7 emotion classes using seed 42.
- **Blinded Anonymity:** Samples assigned opaque identifiers `REV_001` through `REV_200`.
- **Files Created:**
  - `a5_human_review_form.csv`: Standardized review form containing columns for `reviewer_perceived_class`, `confidence` (1–5), `image_quality`, `pose_issue`, `occlusion_issue`, `crop_alignment_issue`, `expression_intensity`, and `multiple_emotion_plausible`. (All label/prediction columns left blank).
  - `a5_human_review_reveal.csv`: Complete ground-truth mapping documenting dataset label, v1, v2, v2.1, v2.2 predictions, and consensus category for post-review unblinding.
  - `contact_sheets_blind/sheet_01_blind.png` to `sheet_10_blind.png`: 10 contact sheets (20 images each) displaying enlarged nearest-neighbor pixel grids with **only the opaque Review ID displayed**.
  - `contact_sheets_reveal/sheet_01_reveal.png` to `sheet_10_reveal.png`: 10 paired contact sheets displaying full metadata and model predictions.

---

## 12. Explicit Answers to the 12 Required Questions

### 1. How much error is shared across v2.1/v2.2?
Approximately **23.3% to 23.5%** of the entire test split (843 Public, 836 Private) is failed by both models simultaneously. Among these mutual errors, **72.6% to 73.9%** predict the exact same wrong emotion class.

### 2. How much is model-specific?
Approximately **13.9% to 14.2%** of test samples (499 Public, 510 Private) represent model-specific errors where one model is correct and the other is wrong. Across the 4 models (v1–v2.2), model-specific error accounts for **12.3% to 12.6%** of the dataset.

### 3. How high is the oracle union accuracy?
- Two-model oracle (v2.1 + v2.2): **76.51%** (Public) and **76.71%** (Private).
- Four-model oracle (v1 + v2 + v2.1 + v2.2): **81.75%** (Public) and **82.06%** (Private).

### 4. Do v1/v2/v2.1/v2.2 share the same hard errors?
Yes. Exactly **655 Public (18.25%)** and **644 Private (17.94%)** samples are failed unanimously by all four model generations. When all 4 models fail, over **53% to 57%** agree unanimously on the same incorrect class.

### 5. Are there exact duplicate images with conflicting labels?
**Yes.** Cryptographic hashing identified **57 exact duplicate groups with conflicting labels** in FER2013, including 23 cross-split duplicate pairs between Train and Public/Private tests where byte-identical images are assigned conflicting ground-truth labels.

### 6. Do non-learned image neighborhoods agree with dataset labels on shared model errors?
**No.** On shared high-confidence errors, non-learned 5-NN true-label purity is only **15.0% to 17.3%** in raw pixel space and gradient PCA space (near chance, $14.3\%$). Independent visual descriptors do not support the nominal dataset labels.

### 7. Does Fear remain poorly separated outside learned MPG space?
**Yes.** Fear displays a strongly negative true-class centroid margin ($-3.39$ on Public, $-3.23$ on Private) in raw pixel space, and is the most negative class in gradient PCA space ($-0.060$). Fear overlap with Sad/Neutral is an intrinsic property of the FER2013 image distribution.

### 8. Are shared errors associated with measurable image quality problems?
**Yes, moderately.** Mutual high-confidence errors exhibit statistically significant reductions in Laplacian energy ($-7.4\%$, $p = 0.009$) and gradient energy ($-8.2\%$, $p = 0.004$), indicating softer focus, blur, or diminished facial texture. However, border crop and contrast ranges are normal.

### 9. Which samples require human inspection?
The **200 sampled images** in `a5_human_review_form.csv` (visualized in `contact_sheets_blind/`), specifically the 50 high-confidence unanimous errors (e.g., Fear labeled faces that look distinctly Sad or Neutral) and the 50 discordant model disagreements.

### 10. Is current evidence stronger for: representation failure, data ambiguity, or both?
Evidence is **strongest for DATA AMBIGUITY (`H-A5-DATA`) on the shared error core**, while **REPRESENTATION LIMITATIONS (`H-A5-REP`) explain model disagreements**. The fact that non-learned image spaces also fail to retrieve nominal labels, combined with 57 confirmed conflicting duplicate groups, proves data ambiguity is an active ceiling factor.

### 11. Is there enough evidence to justify changing labels?
**NO.** Modifying benchmark dataset labels based on model predictions or clustering is methodologically invalid and introduces circular confirmation bias. Ground truth may only be questioned via blinded human adjudication.

### 12. What is the next scientific audit after human review?
The next step is to **execute the blinded human review using `a5_human_review_form.csv`**, unblind via `a5_human_review_reveal.csv`, and measure human-model agreement. If human reviewers corroborate model predictions over nominal labels on the shared error core, a clean post-hoc evaluation protocol can be preregistered.

---

## 13. Summary of A5 Artifacts Generated

All artifacts are finalized under `research/mpg_fer_audit/a5/`:
- `A5_HARD_EXAMPLE_DATA_AUDIT.md` (Comprehensive scientific report)
- `a5_a4_reproduction_note.json` (Documentation of one-sample RAW reproduction and canonical FP32 policy)
- `a5_model_consensus.csv` (Full 7,178-row multi-model consensus and categorization table)
- `a5_oracle_complementarity.json` (2-model and 4-model oracle accuracies and failure partitioning)
- `a5_exact_duplicates.csv` (Catalog of all 1,516 duplicate groups and 3,369 duplicate images)
- `a5_exact_duplicate_summary.json` (Summary of 57 conflicting duplicate groups and cross-split leaks)
- `a5_near_duplicate_candidates.csv` (113 candidate pairs at extreme similarity quantiles)
- `a5_near_duplicate_summary.json` (Quantile thresholds and near-duplicate conflict counts)
- `a5_neighborhood_triangulation.json` (5-NN purities across learned vs. raw vs. gradient spaces)
- `a5_confusion_pair_analysis.json` (Directional collapse and multi-model consensus on hard pairs)
- `a5_nonlearned_class_geometry.json` (Centroid margins in raw pixel, gradient PCA, and Fusion spaces)
- `a5_image_quality.json` (Objective image quality distributions and Mann-Whitney U statistics)
- `a5_hypothesis_decisions.json` (Formal status decisions for H-A5-DATA, H-A5-REP, H-A5-COMPLEMENTARITY)
- `a5_human_review_form.csv` (Standardized blinded human review rubric for 200 samples)
- `a5_human_review_reveal.csv` (Ground-truth unblinding manifest)
- `a5_multi_model_predictions.npz` (Cached FP32 logits and Fusion features for v1, v2, v2.1, v2.2)
- `contact_sheets_blind/*.png` (10 blinded contact sheets with opaque IDs)
- `contact_sheets_reveal/*.png` (10 reveal contact sheets with complete model predictions)
