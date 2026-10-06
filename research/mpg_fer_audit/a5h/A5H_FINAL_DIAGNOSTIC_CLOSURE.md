# MPG-FER A5-H: Blinded AI Adjudication & Final Diagnostic Closure

**Final Audit Verdict:** `A5H_COMPLETE_DIAGNOSTIC_CLOSURE_READY`  
**Execution Environment:** Windows (win32), NVIDIA GeForce RTX 3050 Ti Laptop GPU, PyTorch 2.11.0+cu126  
**Adjudication Scope:** Final post-hoc diagnostic synthesis integrating automated audits, cryptographic duplicate evidence, multi-model consensus, and independent blinded AI visual reviews.  
**Scientific Source Modified:** None (NO)  
**Checkpoints Modified:** None (NO)  
**Training Performed:** None (NO)  
**FER Labels Modified:** None (NO)  
**PrivateTest Used for Tuning:** None (NO)  
**Automatic Relabeling Performed:** None (NO)

---

## 1. Executive Summary & Reviewer Roles

This audit delivers the final diagnostic closure for the MPG-FER post-hoc audit series. It integrates:
1. Automated model consensus across four official frozen checkpoints (`v1`, `v2`, `v2.1`, `v2.2`);
2. Cryptographic exact duplicate and cross-split leakage audits (A5/A5-R);
3. Independent non-learned visual neighborhood triangulations;
4. Two independent, blinded AI visual reviewers evaluating 200 balanced test samples.

### Clarification of Reviewer Identities:
- **Reviewer C:** Claude
- **Reviewer G:** Gemini R2
- Both evaluators are **independent blinded AI visual reviewers** (blinded AI adjudicators). They are *not* human annotators, expert human adjudicators, or replacement ground truth.
- The initial Gemini review was rejected during blinded semantic QC and was completely excluded.
- Blinded evaluations were conducted prior to unblinding dataset labels or model predictions.

---

## 2. Reviewer Integrity and Alignment Gate

Documented in `a5h_reviewer_agreement.json`:
- **Sample Counts:** Exactly 200 rows in `a5_reviewer_claude.csv`, `a5_reviewer_gemini_r2.csv`, and `a5_human_review_reveal.csv`.
- **Review IDs:** Unique identifiers `REV_001` through `REV_200` match identically across all files without missing or duplicated entries.
- **Vocabulary Compliance:** All perceived classes belong to the valid 9-state vocabulary (`Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`, `Ambiguous`, `CannotJudge`).
- **Confidence Range:** Reviewer C: 1 to 5 (mean 3.26); Reviewer G: 2 to 5 (mean 3.59).
- **Metadata Compliance:** All metadata fields (`image_quality`, `pose_issue`, `occlusion_issue`, `crop_alignment_issue`, `expression_intensity`, `multiple_emotion_plausible`) adhere to registered vocabularies.
- *Gate Result:* **PASSED** (`a5h_reviewer_agreement.json`).

---

## 3. Primary Blind Inter-Rater Agreement (Before Unblinding)

Inter-rater reliability between Claude and Gemini R2 was evaluated on raw blinded perceptions before exposing dataset or model labels:

| Reliability Metric | Value | Observations |
| :--- | :---: | :--- |
| **Raw Exact-Class Agreement (All 9 States)** | **47.00%** (94 / 200) | Moderate overall agreement across full review space. |
| **Cohen's Kappa (All 9 States)** | **0.3811** | Fair to moderate agreement when accounting for chance. |
| **Both Select One of 7 FER Classes** | **87.50%** (175 / 200) | In 175 samples, neither reviewer opted for Ambiguous/CannotJudge. |
| **Exact Agreement on 7-Class Subset** | **53.71%** (94 / 175) | Substantial agreement when both choose standard emotion classes. |
| **Cohen's Kappa on 7-Class Subset** | **0.4432** | Moderate inter-rater concordance on decidable faces. |
| **Both Choose Same FER Class** | **94 samples** | Core strict AI consensus pool. |
| **One or Both Flagged Ambiguous** | **6 samples** (Claude: 4, Gemini: 2) | Explicit visual ambiguity abstentions. |
| **One or Both Flagged CannotJudge** | **19 samples** (Claude: 19, Gemini: 0) | Severe occlusion / crop / blur abstentions. |
| **Both Abstained Simultaneously** | **0 samples** | Reviewers rarely abstained on the exact same sample. |

### Reviewer Confusion Matrix (Claude Rows vs. Gemini Cols, `a5h_reviewer_confusion.csv`):
```
Claude \ Gemini   Angry  Disgust   Fear  Happy    Sad  Surprise  Neutral  Ambiguous  CannotJudge
Angry                16        0      3      0      3         0        3          0            0
Disgust               6        4      1      0      0         0        2          0            0
Fear                  4        0     13      1      4         6        3          0            0
Happy                 0        0      0     20      0         0        1          0            0
Sad                   2        0      4      0     16         1       11          0            0
Surprise              1        0      2      0      1        12        1          0            0
Neutral               7        0      0      0     10         3       13          0            0
Ambiguous             0        0      0      0      1         1        2          0            0
CannotJudge           2        0      3      2      2         3        7          0            0
```

### Reviewer Style & Calibration Differences:
- **Abstention Behavior:** Reviewer C exhibited higher caution, abstaining in 23 cases (11.5% total: 19 `CannotJudge`, 4 `Ambiguous`). Reviewer G abstained in only 2 cases (1.0% total: 2 `Ambiguous`, 0 `CannotJudge`).
- **Multiple Emotion Plausibility:** Reviewer G flagged multiple plausible emotions in 28.5% of samples (57/200), whereas Reviewer C flagged 16.5% (33/200).
- **Class Usage:** Both reviewers assigned highest frequency to `Neutral` (C: 43, G: 49) and lowest frequency to `Disgust` (C: 13, G: 11).

---

## 4. Unblinded Bucket Analysis Across 4 Registered Buckets

The review packet comprises exactly 50 samples from each registered bucket (`a5h_bucket_analysis.json`):

| Metric / Outcome | Bucket A: High-Conf Error (N=50) | Bucket B: Mixed Error (N=50) | Bucket C: Split Decision (N=50) | Bucket D: All Correct Control (N=50) |
| :--- | :---: | :---: | :---: | :---: |
| **Reviewer C: Dataset Match** | 11 (22.0%) | 13 (26.0%) | 14 (28.0%) | 24 (48.0%) |
| **Reviewer C: Model Consensus Match** | 21 (42.0%) | 14 (28.0%) | 12 (24.0%) | 24 (48.0%)* |
| **Reviewer G: Dataset Match** | 11 (22.0%) | 14 (28.0%) | 25 (50.0%) | 34 (68.0%) |
| **Reviewer G: Model Consensus Match** | 28 (56.0%) | 16 (32.0%) | 17 (34.0%) | 34 (68.0%)* |
| **Strict Two-Reviewer Consensus Rate**| **25 (50.0%)** | **19 (38.0%)** | **21 (42.0%)** | **29 (58.0%)** |
| - *Consensus Supports Dataset* | **6 (12.0%)** | **5 (10.0%)** | **9 (18.0%)** | **20 (40.0%)** |
| - *Consensus Supports Model Error* | **17 (34.0%)** | **7 (14.0%)** | **5 (10.0%)** | **0 (0.0%)** |
| - *Consensus Supports Third Class* | 2 (4.0%) | 7 (14.0%) | 7 (14.0%) | 9 (18.0%) |
| - *No Reviewer Consensus / Abstain* | **25 (50.0%)** | **31 (62.0%)** | **29 (58.0%)** | **21 (42.0%)** |

*\*Note for Bucket D: Model consensus equals the dataset label by definition.*

---

## 5. Primary Statistical Adjudication Tests

### 5.1 High-Confidence Shared Error Test (`HIGH_CONFIDENCE_ALL_WRONG_SAME_LABEL`, N=50)
Samples where all four MPG models unanimously predict the same incorrect emotion with high confidence ($\ge 0.80$, documented in `a5h_high_conf_shared_error_analysis.json`):

| Adjudication Consensus Outcome | Count | Point Proportion | Percentile Bootstrap 95% CI (B=2000, seed=42) |
| :--- | :---: | :---: | :---: |
| **Consensus Supports Model Wrong Prediction** | **17** | **34.0%** | **[22.0%, 48.0%]** |
| **No Consensus / Reviewer Disagreement** | **25** | **50.0%** | **[36.0%, 64.0%]** |
| **Consensus Supports Nominal Dataset Label** | **6** | **12.0%** | **[4.0%, 22.0%]** |
| **Consensus Supports Third FER Class** | **2** | **4.0%** | **[0.0%, 10.0%]** |

#### Exploratory Binomial Comparison:
Among the 23 decided consensus cases supporting either the model prediction or the nominal dataset label:
- Supported Model Prediction: 17
- Supported Dataset Label: 6
- Two-sided exact binomial test against null $p=0.5$: **$p = 0.0347$** (exploratory).
- *Scientific Statement:* In samples where blind AI reviewers achieved strict consensus, their visual judgments favored the class predicted by the four-model consensus nearly 3x more often than the nominal dataset label ($17$ vs. $6$). In half of the bucket (25/50), visual ambiguity or inter-rater disagreement prevented consensus.

### 5.2 Positive Control Anchor (`ALL_CORRECT`, N=50)
Samples where all four models agree with the nominal dataset label (`a5h_control_analysis.json`):
- Reviewer C matched dataset label in 24/50 cases (48.0%).
- Reviewer G matched dataset label in 34/50 cases (68.0%).
- Strict Two-Reviewer Consensus supported dataset label in **20/50 cases (40.0%)**.
- Disagreement or abstention occurred in **21/50 cases (42.0%)**.
- *Control Meaning:* Even on faces where models and ground truth perfectly align, blind reviewers fail to reach consensus 42% of the time, demonstrating the high baseline ambiguity inherent in 48x48 web crops.

---

## 6. Four Strict Candidate Lists

### 6.1 `LABEL_MISMATCH_CANDIDATES` (17 Samples, `a5h_label_mismatch_candidates.csv`)
*Definition:* All 4 models predict the same wrong class; both blind reviewers independently select that exact same model class; reviewer consensus $\ne$ dataset label.
- **Tier A (High Confidence & Clean Quality, 10 samples):** Both reviewers confidence $\ge 4$, no severe crop/occlusion flags.
  - Examples: `REV_058` (labeled Fear $\to$ perceived/predicted Sad), `REV_080` (labeled Fear $\to$ perceived/predicted Neutral), `REV_064` (labeled Disgust $\to$ perceived/predicted Angry), `REV_017` (labeled Sad $\to$ perceived/predicted Neutral).
- **Tier B (Moderate Confidence or Quality Flags, 7 samples):** One/both reviewer confidence $\le 3$ or minor alignment concerns.

### 6.2 `COMMON_REPRESENTATION_FAILURE_CANDIDATES` (11 Samples, `a5h_common_representation_failure_candidates.csv`)
*Definition:* All 4 models fail; both blind reviewers agree with the nominal dataset label.
- **Tier A (High Confidence Clean Quality, 4 samples):**
  - `REV_024`: True Happy | Models predicted Surprise | Both reviewers perceived Happy (Conf 4/4)
  - `REV_031`: True Happy | Models predicted Neutral | Both reviewers perceived Happy (Conf 4/4)
  - `REV_140`: True Happy | Models predicted Neutral | Both reviewers perceived Happy (Conf 4/4)
  - `REV_188`: True Happy | Models predicted Surprise | Both reviewers perceived Happy (Conf 4/4)
- **Tier B (Moderate Confidence, 7 samples):**
  - Examples: `REV_014` (True Disgust $\to$ models predicted Neutral/Sad), `REV_028` (True Sad $\to$ models predicted Fear/Neutral), `REV_159` (True Sad $\to$ models predicted Angry).
- *Scientific Implication:* These 11 samples represent verified representational blind spots in the MPG architecture: facial cues are visibly clear to external vision systems, but are systematically misclassified by all MPG variants.

### 6.3 `VISUAL_AMBIGUITY_CANDIDATES` (123 Samples, `a5h_visual_ambiguity_candidates.csv`)
*Definition:* Reviewer disagreement, abstention (`Ambiguous` / `CannotJudge`), or both flagged multiple plausible emotions.
- 94 discordant classifications
- 19 CannotJudge abstentions
- 6 Ambiguous abstentions
- 4 multiple-emotion concordances

### 6.4 `BENCHMARK_CONFLICT_CANDIDATES` (16 Unique Groups, `a5h_benchmark_conflict_candidates.csv`)
*Definition:* Cryptographically verified byte-identical raw 48x48 pixel images present in Train and held-out test splits with conflicting emotion annotations (e.g. Train labeled Neutral, Private labeled Happy).

---

## 7. Confusion-Pair Triangulation

Connecting automated error manifold statistics (`a5_confusion_pair_analysis.json`) with blinded review packet adjudications:

| Confusion Pair | Public Shared Errors | Private Shared Errors | Reviewed in Packet | Blind Reviewers Support Dataset | Blind Reviewers Support Model Error | Reviewer Disagreement / Abstention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fear $\to$ Sad** | 59 | 66 | 2 | 0 | **1** | 1 |
| **Sad $\to$ Fear** | 29 | 29 | 3 | 0 | **1** | 2 |
| **Fear $\to$ Neutral** | 32 | 27 | 2 | 0 | **1** | 1 |
| **Sad $\to$ Neutral** | 49 | 62 | 5 | 1 | **3** | 1 |
| **Neutral $\to$ Sad** | 51 | 51 | 5 | **3** | 1 | 1 |
| **Angry $\to$ Sad** | 32 | 40 | 6 | 0 | 0 | **6** |
| **Sad $\to$ Angry** | 39 | 31 | 1 | 0 | 0 | 1 |

*Insight:* On `Sad -> Neutral` and `Fear -> Sad`, blind adjudicators favored the model error over the nominal label in 4 of 7 reviewed cases. In `Angry -> Sad`, reviewers disagreed in 100% of reviewed cases (6/6), demonstrating extreme visual ambiguity between subdued anger and sadness in 48x48 resolution.

---

## 8. Duplicate-Conditioned Review Results

Evaluated across the 200 reviewed samples (`a5r_human_packet_integrity.json`):
- **`NO_TRAIN_DUPLICATE` (191 samples):** Both reviewers agreed on an FER class in 92 cases; supported dataset label in 39 cases (20.4%).
- **`TRAIN_DUPLICATE_SAME_LABEL` (8 samples):** Both reviewers agreed on an FER class in 2 cases; supported dataset label in 1 case.
- **`TRAIN_DUPLICATE_CONFLICTING_LABEL` (1 sample, `REV_089`):**
  - Dataset label: `Angry` (Private row 1499).
  - Train duplicate label: `Sad`.
  - v2.1 and v2.2 predictions: `Sad` (memorized Train label).
  - Reviewer C perceived: `Angry` (Conf 3).
  - Reviewer G perceived: `Disgust` (Conf 3).
  - Result: Reviewers did not reach consensus, reflecting mixed negative valence.

---

## 9. AI-Reviewer Limitations & Methodological Boundaries

1. **Non-Human Nature:** Claude and Gemini R2 are large vision-language AI models, not human psychophysicists or certified FACS annotators.
2. **Shared Internet Priors:** Both foundation models may share overlapping pretraining biases regarding prototypical facial expressions.
3. **Moderate Inter-Rater Reliability:** The baseline agreement of 47.0% (kappa = 0.38) reflects significant uncertainty in 48x48 grayscale crops.
4. **No Relabeling Authority:** AI reviewer agreement provides valuable external triangulation against the MPG model family, but **must never be used to modify benchmark dataset labels**.

---

## 10. Qualitative Ceiling Decomposition

The observed ~69–70% FER2013 benchmark ceiling arises from five overlapping, non-additive factors:
1. **Benchmark Duplication Benefit (~2.2%):** 7.9% of test images are exact duplicates of training images with matching labels, artificially inflating standard scores from ~67.4% to ~69.6%.
2. **Direct Annotation Inconsistencies:** 57 exact duplicate groups with contradictory labels, plus conflicting cross-split leakage rows.
3. **Severe Visual Ambiguity:** 53% of reviewed difficult samples show reviewer discordance or explicit ambiguity flags, driven by low resolution (48x48), head yaw, and subtle facial dynamics.
4. **Systematic Label Mismatches:** A verified subset (34% of high-confidence shared errors) where visual cues align with the model prediction rather than the nominal label.
5. **Common Representation Failures & Model Disagreements:** 11 verified clean representation failure candidates, alongside ~12% model-specific resolvable errors where architectural complementarity exists.

---

## 11. Final Hypothesis Decisions

Documented in `a5h_hypothesis_decisions.json`:

1. **`H-A5H-LABEL-INCONSISTENCY`:** **`SUPPORTED`**  
   *Evidence:* 57 verified exact duplicate groups with contradictory labels across FER2013; model accuracy drops to 13–38% on conflicting duplicates.
2. **`H-A5H-VISUAL-AMBIGUITY`:** **`SUPPORTED_AS_MATERIAL_FACTOR`**  
   *Evidence:* Blind inter-rater agreement is only 47.0%; 123 of 200 reviewed samples exhibit reviewer discordance, abstention, or multiple plausible emotions; reviewers fail to reach consensus on 42% of control images.
3. **`H-A5H-SYSTEMATIC-LABEL-MISMATCH`:** **`SUPPORTED_FOR_SUBSET`**  
   *Evidence:* In the high-confidence shared error bucket, strict consensus supported the model's prediction nearly 3x more often than the dataset label (34.0% vs. 12.0%, exact binomial $p = 0.0347$). 17 strict candidates cataloged.
4. **`H-A5H-COMMON-REPRESENTATION-FAILURE`:** **`SUPPORTED_FOR_SUBSET`**  
   *Evidence:* 11 samples across mutual error buckets have strict two-reviewer consensus supporting the nominal dataset label while all four MPG models unanimously failed.
5. **`H-A5H-COMPLEMENTARITY`:** **`MEANINGFUL`**  
   *Evidence:* Four-model oracle union accuracy reaches **82.06%** (+12.3% gain over individual models). Over 12% of total test error is model-specific.

---

## 12. Final Bottleneck Statement

> **Project-Level Bottleneck Statement:**  
> MPG-FER remains bounded near ~69–70% on standard FER2013 due to five overlapping, non-additive factors: (1) ~2.2% benchmark score inflation from exact cross-split Train duplicates; (2) 57 verified exact duplicate groups with conflicting ground-truth labels that induce unavoidable errors; (3) severe intrinsic data ambiguity and low resolution that cause blind independent reviewers to disagree on 53% of challenging faces; (4) systematic label mismatches in a distinct subset of high-confidence shared errors where both blind reviewers visually agree with the model prediction rather than the nominal label; and (5) common representation failures on a separate subset of valid expressions, alongside ~12% model-specific resolvable errors. Graph topology optimization alone cannot surpass this ceiling because topology density is not the primary limiting factor.

---

## 13. Final Research Recommendations

1. **Graph Topology Optimization:** **DEPRIORITIZE**. Dense v2.1 and sparse v2.2 achieve statistical parity; further tuning of Top-K or graph edges will not overcome the shared error ceiling.
2. **Automatic Label Cleaning:** **NO**. Automatic relabeling is rejected due to circularity.
3. **Next Scientific Investigation:** Focus on **representation quality on model-resolvable cases** (the ~12% complementarity pool) and evaluate **multi-scale facial action unit / landmark priors** to resolve the 11 verified common representation failure candidates.
