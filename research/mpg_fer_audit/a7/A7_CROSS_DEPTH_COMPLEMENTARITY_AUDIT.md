# MPG-FER A7 — Cross-Depth Information Complementarity & Representation-Preservation Audit

**Audit Status:** COMPLETE  
**Primary Scientific Decision:** `NO_ACTIONABLE_COMPLEMENTARITY`  
**Public-Locked Target Decision:** `NO_V24_ARCHITECTURAL_TARGET`  
**Private Confirmatory Status:** `PUBLIC_NEGATIVE_PRIVATE_OBSERVATIONAL_ONLY`  
**v2.4 Gate Verdict:** `V24_NOT_JUSTIFIED`  
**Operational Verdict:** `A7_COMPLETE_NO_V24_TARGET`  
**Execution Mode:** Frozen Post-Hoc Representation Audit (Raw Single-View FP32)  
**Private-Test Firewall Status:** Strictly Preserved (`PRIVATE_USED_FOR_DECISION = false` locked prior to Private evaluation)  

---

## 1. Executive Summary & Core Mandate

This audit directly evaluates the central hypothesis emerging from the negative result of MPG-FER v2.3:
> *"Do earlier Motif representations contain held-out discriminative information that is complementary to, and not fully preserved by, later Motif representations/readout/fusion?"*

And the secondary question:
> *"If such complementary information exists, is the next justified research target cross-depth information preservation, readout information extraction, or neither?"*

### Central Findings

1. **Failure of Cross-Depth Complementarity Under Linear Probing:**  
   Across all tested pairs in MPG-FER v2.3 (primary model) and v2.2 (baseline model), concatenating earlier representations (PRE, L1, L2) with later representations (L5, Motif Readout, Fusion) fails to yield an improvement in held-out generalization on PublicTest. Raw concatenation uniformly degrades or fails to improve performance:
   - v2.3 $L2 + L5$: Public Accuracy drops by $-0.31$ pp ($66.01\%$ vs $66.31\%$ late baseline; paired bootstrap 95% CI $[-1.00, +0.39]$ pp, $p=0.4445$).
   - v2.3 $L1 + L5$: Public Accuracy drops by $-0.53$ pp ($65.78\%$ vs $66.31\%$; CI $[-1.23, +0.17]$ pp, $p=0.1532$).
   - v2.3 $\text{PRE} + L5$: Public Accuracy drops by $-0.36$ pp ($65.95\%$ vs $66.31\%$; CI $[-0.98, +0.28]$ pp, $p=0.3088$).
   - v2.3 $L1 + L2 + L5$: Public Accuracy drops by $-0.59$ pp ($65.73\%$ vs $66.31\%$; CI $[-1.37, +0.20]$ pp, $p=0.1643$).

2. **Dimensionality Controls Confirm No Unique Value:**  
   - **Duplicate-Late Controls ($L5+L5$, $L5+L5+L5$):** Simply duplicating late features incurs a slight penalty due to L2 regularization over doubled feature dimensions ($-0.14$ pp in v2.3). Cross-depth concatenations perform worse than or comparable to this dimension control.
   - **Train-Only PCA Matching:** Reducing concatenated early+late features back to the 192D late-stage budget via Train-only PCA yields $-0.11$ pp for $\text{PCA}(L2+L5) \to 192$ and $-0.03$ pp for $\text{PCA}(L1+L5) \to 192$. None meets the preregistered threshold ($+0.50$ pp).

3. **Conditional Unique-Information Falsification:**  
   Fitting a linear Ridge mapping ($L5 \to E$) on Train and evaluating $L5 + E_{\text{residual}}$ yields $65.81\%$ ($-0.50$ pp vs $L5$ alone), which is indistinguishable from the shuffled negative control ($65.76\%$, $-0.56$ pp). Early representations contain no linearly unpredictable signal that aids generalization.

4. **Readout and Fusion Information Preservation:**  
   - Adding $L5$ to Motif Readout drops accuracy by $-0.20$ pp raw ($+0.00$ pp PCA).
   - Adding $L2$ to Fusion drops accuracy by $-0.81$ pp raw ($-0.45$ pp PCA).
   - Upstream motif information is already linearly subsumed by downstream readout and fusion heads to the limit of linear extractability.

5. **Firewall & Confirmatory Verdict:**  
   The Public Decision was locked to `NO_V24_ARCHITECTURAL_TARGET` prior to inspecting PrivateTest. Post-freeze confirmatory evaluation on PrivateTest verified this negative finding without exception (e.g., v2.3 $L2+L5$ raw is $-1.14$ pp on Private; PCA is $-1.06$ pp).  
   Therefore, **v2.4 is NOT justified**. The audit successfully closes this research line.

---

## 2. Integrity Statement

In strict adherence to experimental discipline and repository governance:

| Governance Check | Status | Verification Detail |
| :--- | :---: | :--- |
| Scientific model source changed | **NO** | Zero modifications to any model source file |
| v2.2 source or checkpoint changed | **NO** | Checkpoint SHA-256 confirmed: `a10bd22b...` |
| v2.3 source or checkpoint changed | **NO** | Checkpoint SHA-256 confirmed: `23dbe9b1...` |
| FER model training performed | **NO** | Zero backpropagation / model training executed |
| Model architecture modified | **NO** | Frozen architectures used strictly for forward extraction |
| FER dataset labels changed | **NO** | Standard canonical FER2013 CSVs |
| FER dataset splits changed | **NO** | Train (28,709), Val/Public (3,589), Test/Private (3,589) |
| Test-time augmentation (TTA) used | **NO** | Raw single-view FP32 inference exclusively |
| Private split used before Public Decision Lock | **NO** | `a7_public_decision_lock.json` written & locked first |
| Private split used to select pairs / dims | **NO** | Strict firewall preserved |
| Private split used to choose v2.4 target | **NO** | Firewall preserved; Private confirmatory only |
| Hyperparameter tuning / sweep | **NO** | Fixed $C=1.0$, L2 penalty, lbfgs, tol=1e-6, max_iter=5000 |
| Model ensemble created | **NO** | Cross-model ensembling prohibited and omitted |
| New model loss tested | **NO** | Representation probe audit only |
| Historical A6/A6-R/v2.3 artifacts overwritten | **NO** | All historical artifacts intact |

---

## 3. Provenance & Hook Validation

### 3.1 Model Lineage & Hashes

- **Git HEAD Commit:** `75e192d0e81ef53bb71e0853530d6a48a7c13582`
- **Audit Implementation:** `research/mpg_fer_audit/a7/`
- **MPG-FER v2.2 Official Best Checkpoint SHA-256:**  
  `a10bd22b3903550156c8239d91b5d2af35067ca1f2bdba9af46cf1e53d0bbdf4`
- **MPG-FER v2.3 Official Best Checkpoint SHA-256:**  
  `23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e`
- **MPG-FER v2.3 Config SHA-256:**  
  `8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2`
- **MPG-FER v2.3 Source SHA-256:**  
  `1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87`

### 3.2 Hook Transparency

Forward hooks were registered to extract pooled representations at stages R1, R2, R3, R6, R7, R8. To guarantee that hook registration introduces zero computational perturbation, the unhooked forward logits and hooked forward logits were compared on deterministic tensor inputs:
- **v2.2 Max Absolute Logit Difference:** $0.0000 \times 10^0$ ($\le 1.0 \times 10^{-6}$)
- **v2.3 Max Absolute Logit Difference:** $0.0000 \times 10^0$ ($\le 1.0 \times 10^{-6}$)
- **Verdict:** `PASS` (Recorded in `a7_hook_validation.json`)

---

## 4. Phase 1: Single-Stage Probe Reproduction

To ensure exact parity with authoritative A6 and v2.3 artifacts, fixed multinomial logistic regression probes ($C=1.0$, L2, lbfgs, max_iter=5000, tol=1e-6) were trained on Train features and evaluated on Public and Private splits.

### Parity Against Authoritative Artifacts

| Model | Stage | Dim | Train Acc | Public Acc | Authoritative Public | Delta (pp) | Private Acc | Authoritative Private | Delta (pp) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v2.3** | PRE (R1) | 192 | 59.35% | 54.5834% | 54.5834% | 0.0000 | 55.1686% | 55.1686% | 0.0000 | `PASS` |
| **v2.3** | L1 (R2) | 192 | 70.13% | 59.1808% | 59.1808% | 0.0000 | 60.4068% | 60.4068% | 0.0000 | `PASS` |
| **v2.3** | L2 (R3) | 192 | 80.26% | 62.8866% | 62.8587% | 0.0279 | 63.8897% | 63.8897% | 0.0000 | `PASS` |
| **v2.3** | L5 (R6) | 192 | 95.91% | 66.3137% | 66.3137% | 0.0000 | 67.7347% | 67.7347% | 0.0000 | `PASS` |
| **v2.3** | Readout (R7) | 384 | 97.38% | 66.7317% | 66.7317% | 0.0000 | 67.7069% | 67.7069% | 0.0000 | `PASS` |
| **v2.3** | Fusion (R8) | 512 | 97.78% | 66.1744% | 66.1744% | 0.0000 | 67.4004% | 67.4004% | 0.0000 | `PASS` |
| **v2.2** | PRE (R1) | 192 | 60.62% | 55.1407% | 55.1407% | 0.0000 | 56.4224% | 56.4224% | 0.0000 | `PASS` |
| **v2.2** | L1 (R2) | 192 | 77.86% | 61.7721% | 61.7721% | 0.0000 | 63.5274% | 63.5274% | 0.0000 | `PASS` |
| **v2.2** | L2 (R3) | 192 | 88.68% | 64.3355% | 64.3355% | 0.0000 | 66.0351% | 66.0351% | 0.0000 | `PASS` |
| **v2.2** | L5 (R6) | 192 | 95.25% | 65.5614% | 65.5614% | 0.0000 | 66.6760% | 66.6760% | 0.0000 | `PASS` |
| **v2.2** | Readout (R7) | 384 | 97.85% | 66.8989% | 66.8989% | 0.0000 | 67.1218% | 67.1218% | 0.0000 | `PASS` |
| **v2.2** | Fusion (R8) | 512 | 98.28% | 66.7317% | 66.7317% | 0.0000 | 66.8989% | 66.8710% | 0.0279 | `PASS` |

Every probe replicates within $<0.03$ pp, well below the required $\le 0.10$ pp threshold.

---

## 5. Phase 2: Public-Only Cross-Depth Exploration

The audit was executed under the strict firewall: Public results were analyzed and locked before opening PrivateTest.

### 5.1 Primary Registered Public Table (MPG-FER v2.3)

| Representation | Dim | Train Acc | Public Acc | Public Macro-F1 | Delta Acc vs Late | Delta F1 vs Late | Bootstrap 95% CI (Acc) | McNemar $p$ | Control Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **L5 (Baseline)** | 192 | 95.91% | **66.31%** | **65.61%** | $+0.00$ pp | $+0.00$ pp | — | 1.0000 | Late Baseline |
| L5+L5 control | 384 | 95.93% | 66.17% | 65.40% | $-0.14$ pp | $-0.21$ pp | — | — | Duplicate-Late Control |
| PRE+L5 | 384 | 96.38% | 65.95% | 64.87% | $-0.36$ pp | $-0.74$ pp | $[-0.98, +0.28]$ pp | 0.3088 | Raw Concat |
| L1+L5 | 384 | 96.34% | 65.78% | 64.75% | $-0.53$ pp | $-0.86$ pp | $[-1.23, +0.17]$ pp | 0.1532 | Raw Concat |
| L2+L5 | 384 | 96.58% | 66.01% | 64.97% | $-0.31$ pp | $-0.64$ pp | $[-1.00, +0.39]$ pp | 0.4445 | Raw Concat |
| L1+L2+L5 | 576 | 96.84% | 65.73% | 64.86% | $-0.59$ pp | $-0.75$ pp | $[-1.37, +0.20]$ pp | 0.1643 | Raw Concat |
| PCA(PRE+L5)->192 | 192 | 95.78% | 66.45% | 64.91% | $+0.14$ pp | $-0.70$ pp | $[-0.61, +0.89]$ pp | 0.7663 | PCA Control |
| PCA(L1+L5)->192 | 192 | 95.76% | 66.29% | 64.99% | $-0.03$ pp | $-0.62$ pp | $[-0.75, +0.70]$ pp | 1.0000 | PCA Control |
| PCA(L2+L5)->192 | 192 | 95.85% | 66.20% | 65.03% | $-0.11$ pp | $-0.58$ pp | $[-0.84, +0.59]$ pp | 0.8212 | PCA Control |
| PCA(L1+L2+L5)->192 | 192 | 95.85% | 66.37% | 64.89% | $+0.06$ pp | $-0.72$ pp | $[-0.75, +0.86]$ pp | 0.9447 | PCA Control |
| **Motif Readout (Base)**| 384 | 97.38% | **66.73%** | **65.00%** | $+0.00$ pp | $+0.00$ pp | — | 1.0000 | Late Baseline |
| L2+Motif Readout | 576 | 97.88% | 66.04% | 64.62% | $-0.70$ pp | $-0.38$ pp | $[-1.39, +0.00]$ pp | 0.0474 | Raw Concat |
| PCA(L2+Readout)->384 | 384 | 97.61% | 65.92% | 64.84% | $-0.81$ pp | $-0.16$ pp | $[-1.50, -0.06]$ pp | 0.0330 | PCA Control |
| **Fusion (Baseline)** | 512 | 97.78% | **66.17%** | **64.24%** | $+0.00$ pp | $+0.00$ pp | — | 1.0000 | Late Baseline |
| L2+Fusion | 704 | 98.30% | 65.37% | 63.93% | $-0.81$ pp | $-0.30$ pp | $[-1.48, -0.20]$ pp | 0.0206 | Raw Concat |
| PCA(L2+Fusion)->512 | 512 | 98.18% | 65.73% | 64.35% | $-0.45$ pp | $+0.12$ pp | $[-1.23, +0.31]$ pp | 0.2688 | PCA Control |

---

## 6. Model Lineage Comparison: v2.2 vs v2.3

Did early residual attenuation in v2.3 create new cross-depth complementarity?

| Core Comparison | v2.2 Public Acc | v2.2 Delta vs L5 | v2.3 Public Acc | v2.3 Delta vs L5 | Complementarity Pattern |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **L5 Baseline** | 65.56% | $+0.00$ pp | 66.31% | $+0.00$ pp | v2.3 L5 $+0.75$ pp higher |
| **L1 + L5 (Raw)** | 65.39% | $-0.17$ pp | 65.78% | $-0.53$ pp | Negative in both models |
| **L2 + L5 (Raw)** | 65.53% | $-0.03$ pp | 66.01% | $-0.31$ pp | Negative in both models |
| **PCA(L1+L5) $\to$ 192** | 65.84% | $+0.28$ pp | 66.29% | $-0.03$ pp | Non-actionable ($<+0.50$ pp) |
| **PCA(L2+L5) $\to$ 192** | 66.01% | $+0.45$ pp | 66.20% | $-0.11$ pp | Non-actionable ($<+0.50$ pp) |

**Scientific Interpretation:**  
In v2.2, PCA compression retains $+0.45$ pp on PublicTest, but this effect completely vanishes in v2.3 ($-0.11$ pp). Neither model achieves the actionable threshold ($+0.75$ pp raw, $+0.50$ pp PCA). Residual attenuation in v2.3 did not unlock cross-depth synergy; rather, it reduced early-depth representation power without creating linearly recoverable complementary signal.

---

## 7. Deep Diagnostic Analyses

### 7.1 Conditional Unique-Information Analysis (Ridge Residualization)

To test whether early layers contain signal linearly unpredictable from L5:
1. Fit linear Ridge regression ($L5 \to E$, $\alpha=1.0$) on Train.
2. Form $E_{\text{residual}} = E - \hat{E}(L5)$.
3. Train linear probe on $[L5, E_{\text{residual}}]$.
4. Compare against negative control $[L5, \text{shuffled}(E_{\text{residual}})]$.

| Model | Early Stage | $L5 + E_{\text{res}}$ Acc | Delta vs L5 | Shuffled Negative Control | Shuffled Delta | Signal Beyond Chance? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **v2.3** | PRE | 65.90% | $-0.42$ pp | 65.95% | $-0.36$ pp | **NO** (Equal to noise) |
| **v2.3** | L1 | 65.56% | $-0.75$ pp | 65.70% | $-0.61$ pp | **NO** (Worse than noise) |
| **v2.3** | L2 | 65.81% | $-0.50$ pp | 65.76% | $-0.56$ pp | **NO** (Equal to noise) |
| **v2.2** | PRE | 65.23% | $-0.33$ pp | 65.20% | $-0.36$ pp | **NO** |
| **v2.2** | L1 | 65.56% | $+0.00$ pp | 64.81% | $-0.75$ pp | Marginally above shuffle |
| **v2.2** | L2 | 65.28% | $-0.28$ pp | 65.45% | $-0.11$ pp | **NO** |

The residualized early representation fails to improve L5 in any setting. The apparent early information not predicted by L5 is non-generalizing noise.

### 7.2 Probe Complementarity Set Analysis

| Model | Comparison | Both Correct | Both Wrong | Early Correct, Late Wrong | Late Correct, Early Wrong | Oracle Union Acc | Joint Raw Rescued | Rescue Rate | Error Set Jaccard |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v2.3** | PRE vs L5 | 1658 | 908 | 301 | 722 | 74.70% | 79 / 301 | 26.25% | 0.469 |
| **v2.3** | L1 vs L5 | 1797 | 882 | 327 | 583 | 75.42% | 85 / 327 | 25.99% | 0.492 |
| **v2.3** | L2 vs L5 | 1941 | 823 | 316 | 439 | 75.12% | 88 / 316 | 27.85% | 0.522 |
| **v2.2** | PRE vs L5 | 1675 | 932 | 304 | 678 | 74.03% | 83 / 304 | 27.30% | 0.487 |
| **v2.2** | L1 vs L5 | 1878 | 873 | 339 | 499 | 75.68% | 100 / 339 | 29.50% | 0.510 |
| **v2.2** | L2 vs L5 | 1993 | 798 | 316 | 360 | 74.34% | 114 / 316 | 36.08% | 0.542 |

**Key Insight:**  
While an theoretical "oracle union" exists ($75.12\%$ on v2.3, indicating 316 samples where L2 was correct and L5 was wrong), a joint linear probe on $L2+L5$ rescues only 88 of those 316 samples ($27.85\%$) while simultaneously misclassifying 99 samples that L5 alone had solved correctly. The trade-off is net negative.

### 7.3 Classwise Complementarity (v2.3 L2 vs L5)

| Emotion | Samples ($N$) | L5 Acc | L2 Acc | L2+L5 Raw | PCA(L2+L5) | $\Delta$ Raw vs L5 | Early-Right Late-Wrong | Rescued by Joint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Angry** | 467 | 57.39% | 55.46% | 56.53% | 57.17% | $-0.86$ pp | 47 | 13 (27.7%) |
| **Disgust** | 56 | 62.50% | 60.71% | 62.50% | 62.50% | $+0.00$ pp | 7 | 2 (28.6%) |
| **Fear** | 496 | 48.39% | 45.36% | 48.79% | 48.79% | $+0.40$ pp | 57 | 18 (31.6%) |
| **Happy** | 895 | 87.15% | 86.15% | 87.04% | 87.26% | $-0.11$ pp | 32 | 14 (43.8%) |
| **Sad** | 653 | 55.28% | 50.84% | 54.98% | 55.44% | $-0.31$ pp | 69 | 16 (23.2%) |
| **Surprise** | 415 | 81.69% | 76.87% | 80.72% | 80.96% | $-0.96$ pp | 32 | 7 (21.9%) |
| **Neutral** | 607 | 64.91% | 59.80% | 64.41% | 64.41% | $-0.49$ pp | 72 | 18 (25.0%) |

No class achieves actionable improvement. In hard classes (Angry, Sad, Neutral), concatenating L2 causes slight regressions or fails to shift accuracy.

### 7.4 Confusion Direction Analysis (PublicTest v2.3)

| Confusion Direction | L5 Probe Errors | L2+L5 Raw Errors | PCA(L2+L5) Errors | Delta (Raw vs L5) |
| :--- | :---: | :---: | :---: | :---: |
| **Fear $\to$ Sad** | 108 | 105 | 107 | $-3$ |
| **Fear $\to$ Neutral** | 48 | 49 | 49 | $+1$ |
| **Fear $\to$ Angry** | 55 | 55 | 53 | $0$ |
| **Sad $\to$ Neutral** | 101 | 104 | 103 | $+3$ |
| **Neutral $\to$ Sad** | 104 | 103 | 103 | $-1$ |
| **Angry $\to$ Sad** | 76 | 77 | 75 | $+1$ |

Cross-depth combinations produce negligible shifts in core confusion pairs ($\pm 1$ to $3$ samples out of hundreds), confirming the absence of structural disentanglement.

### 7.5 Representation Redundancy (Linear CKA)

| Pair | v2.3 Train CKA | v2.3 Public CKA | v2.2 Public CKA | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **PRE vs L5** | 0.4418 | 0.4682 | 0.4878 | Moderate similarity across 5 layers |
| **L1 vs L5** | 0.5843 | 0.6125 | 0.6693 | Substantial shared geometry |
| **L2 vs L5** | 0.7303 | 0.7490 | 0.8038 | High linear similarity |
| **L5 vs Readout** | 0.9413 | 0.9405 | 0.9392 | Very high redundancy ($>0.94$) |
| **L5 vs Fusion** | 0.8931 | 0.8943 | 0.8937 | High linear preservation ($>0.89$) |

Linear CKA increases smoothly with depth, reaching $0.94$ between L5 and Readout and $0.89$ between L5 and Fusion. There is no evidence of catastrophic geometry collapse or acute information bottlenecking.

---

## 8. Preregistered Decision Rules & Public Lock

### 8.1 Rule Evaluation (Public Only)

1. **Primary Actionable Complementarity Rule (Section 26):**
   - Criterion A ($\Delta \text{Acc} \ge +0.75$ pp): **FAIL** (Best raw is $-0.31$ pp).
   - Criterion B (Bootstrap 95% CI excludes 0): **FAIL** (All intervals span zero or are strictly negative).
   - Criterion C (Exceeds duplicate-late control): **FAIL**.
   - Criterion D (PCA retains $\ge +0.50$ pp): **FAIL** (Best PCA is $+0.14$ pp for PRE, $-0.11$ pp for L2).
   - Verdict: **NOT SUPPORTED**.

2. **Readout Information-Loss Rule (Section 28):**
   - $L5 + \text{Readout}$ vs Readout: Raw delta is $-0.20$ pp; PCA delta is $+0.00$ pp.
   - Verdict: **NOT SUPPORTED**.

3. **Fusion Information-Loss Rule (Section 29):**
   - $L2 + \text{Fusion}$ vs Fusion: Raw delta is $-0.81$ pp; PCA delta is $-0.45$ pp.
   - Verdict: **NOT SUPPORTED**.

4. **Public Decision Lock Output:**
   - Saved to: `research/mpg_fer_audit/a7/a7_public_decision_lock.json`
   - Explicit declaration: `PRIVATE_USED_FOR_DECISION = false`
   - Locked Target: `NO_V24_ARCHITECTURAL_TARGET`

---

## 9. Phase 3: Confirmatory Private Evaluation

Having locked the Public Decision, the identical pipeline was evaluated on PrivateTest strictly as confirmatory evidence.

### 9.1 Private Test Table

| Representation | v2.3 Private Acc | v2.3 $\Delta$ vs Late | v2.3 Bootstrap 95% CI | v2.2 Private Acc | v2.2 $\Delta$ vs Late | v2.2 Bootstrap 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L5 (Baseline)** | **67.73%** | $+0.00$ pp | — | **66.68%** | $+0.00$ pp | — |
| L5+L5 control | 67.57% | $-0.17$ pp | — | 66.45% | $-0.22$ pp | — |
| PRE+L5 (Raw) | 67.23% | $-0.50$ pp | $[-1.20, +0.20]$ pp | 66.20% | $-0.47$ pp | $[-1.17, +0.20]$ pp |
| L1+L5 (Raw) | 67.01% | $-0.72$ pp | $[-1.45, +0.00]$ pp | 66.56% | $-0.11$ pp | $[-0.81, +0.56]$ pp |
| L2+L5 (Raw) | 66.59% | $-1.14$ pp | $[-1.87, -0.47]$ pp | 65.98% | $-0.70$ pp | $[-1.37, -0.03]$ pp |
| L1+L2+L5 (Raw) | 66.90% | $-0.84$ pp | $[-1.67, +0.00]$ pp | 66.20% | $-0.47$ pp | $[-1.23, +0.28]$ pp |
| PCA(PRE+L5)->192 | 67.01% | $-0.72$ pp | $[-1.48, +0.03]$ pp | 66.68% | $+0.00$ pp | $[-0.72, +0.72]$ pp |
| PCA(L1+L5)->192 | 66.87% | $-0.86$ pp | $[-1.59, -0.11]$ pp | 66.45% | $-0.22$ pp | $[-0.89, +0.47]$ pp |
| PCA(L2+L5)->192 | 66.68% | $-1.06$ pp | $[-1.81, -0.33]$ pp | 66.29% | $-0.39$ pp | $[-1.03, +0.25]$ pp |
| PCA(L1+L2+L5)->192| 66.56% | $-1.17$ pp | $[-2.01, -0.42]$ pp | 66.56% | $-0.11$ pp | $[-0.78, +0.56]$ pp |
| **Readout (Base)** | **67.71%** | $+0.00$ pp | — | **67.12%** | $+0.00$ pp | — |
| L2+Readout (Raw) | 67.15% | $-0.56$ pp | $[-1.20, +0.11]$ pp | 67.12% | $+0.00$ pp | $[-0.61, +0.59]$ pp |
| PCA(L2+Readout) | 67.15% | $-0.56$ pp | $[-1.23, +0.17]$ pp | 66.95% | $-0.17$ pp | $[-0.86, +0.53]$ pp |
| **Fusion (Base)** | **67.40%** | $+0.00$ pp | — | **66.90%** | $+0.00$ pp | — |
| L2+Fusion (Raw) | 66.65% | $-0.75$ pp | $[-1.42, -0.08]$ pp | 66.65% | $-0.25$ pp | $[-0.89, +0.33]$ pp |
| PCA(L2+Fusion) | 66.45% | $-0.95$ pp | $[-1.73, -0.17]$ pp | 65.92% | $-0.98$ pp | $[-1.76, -0.22]$ pp |

### 9.2 Confirmatory Verdict

Private test evaluation strongly confirms the negative public result:
- Cross-depth combinations fail uniformly, with several pairings ($L2+L5$, $\text{PCA}(L2+L5)$) exhibiting statistically significant degradation on PrivateTest (bootstrap 95% CI entirely below 0).
- Confirmatory Status: `PUBLIC_NEGATIVE_PRIVATE_OBSERVATIONAL_ONLY`.

---

## 10. The 20 Required Scientific Questions Answered

1. **Does PRE add held-out information beyond L5?**  
   **No.** Raw $\text{PRE}+L5$ drops Public Accuracy by $-0.36$ pp; under PCA, the delta is $+0.14$ pp (CI $[-0.61, +0.89]$ pp, $p=0.7663$), failing actionable thresholds.

2. **Does L1 add held-out information beyond L5?**  
   **No.** Raw $L1+L5$ drops accuracy by $-0.53$ pp; under PCA, delta is $-0.03$ pp.

3. **Does L2 add held-out information beyond L5?**  
   **No.** Raw $L2+L5$ drops accuracy by $-0.31$ pp; under PCA, delta is $-0.11$ pp.

4. **Does L1+L2 together add more than either alone?**  
   **No.** Raw $L1+L2+L5$ drops accuracy by $-0.59$ pp; under PCA, delta is $+0.06$ pp.

5. **Do raw concatenation gains survive a duplicate-dimension control?**  
   **Not applicable / No.** There were no raw concatenation gains to begin with; raw concatenations performed worse than duplicate-late controls.

6. **Do gains survive Train-only PCA to the late-stage dimension?**  
   **No.** In v2.3, all PCA combinations hover near zero or below ($-0.11$ pp to $+0.14$ pp), and on PrivateTest all are negative ($-0.72$ pp to $-1.17$ pp).

7. **Does residualized early information improve late-stage classification?**  
   **No.** $L5 + E_{\text{residual}}$ yields $-0.42$ pp (PRE), $-0.75$ pp (L1), and $-0.50$ pp (L2), which is indistinguishable from shuffled negative controls.

8. **What fraction of L5 probe errors are solved by early probes?**  
   L2 probe correctly classifies $26.1\%$ (316 / 1209) of L5 Public errors.

9. **Does a joint probe recover those complementary cases?**  
   **No.** A joint linear probe rescues only 88 of those 316 cases ($27.8\%$), while inducing 99 new errors on cases L5 had correctly solved, yielding a net loss of 11 samples.

10. **Which FER classes benefit?**  
    **None.** Classwise accuracies for Angry ($-0.86$ pp), Sad ($-0.31$ pp), and Neutral ($-0.49$ pp) all decline under $L2+L5$.

11. **Are Fear/Sad/Neutral/Angry particularly complementary across depth?**  
    **No.** Confusion rates between Fear, Sad, Neutral, and Angry shift by only $\pm 1$ to $3$ samples out of several hundred.

12. **Does L5 contain useful information that Readout fails to linearly retain?**  
    **No.** Adding L5 to Motif Readout drops Public accuracy by $-0.20$ pp ($+0.00$ pp PCA).

13. **Does Fusion fail to retain upstream motif information?**  
    **No.** Adding L2 or L5 to Fusion degrades accuracy by $-0.81$ pp ($-0.45$ pp PCA).

14. **Is the pattern present in both v2.2 and v2.3?**  
    **Yes.** Raw concatenation is universally negative or neutral across both models.

15. **Did v2.3 increase or reduce cross-depth complementarity?**  
    **Reduced.** In v2.2, PCA compression showed slight positive drift ($+0.45$ pp for L2+L5), which completely vanished in v2.3 ($-0.11$ pp).

16. **What is the Public-only hypothesis decision?**  
    `H-A7-CROSS-DEPTH-COMPLEMENTARITY: MIXED` (due to weak v2.2 PCA drift; NOT supported in v2.3).  
    `H-A7-EARLY-UNIQUE-INFORMATION: NOT_SUPPORTED`.  
    `H-A7-READOUT-INFORMATION-LOSS: NOT_SUPPORTED`.  
    `H-A7-FUSION-INFORMATION-LOSS: NOT_SUPPORTED`.  
    `H-A7-V23-SPECIFIC-COMPLEMENTARITY: NOT_SUPPORTED`.

17. **What target was locked BEFORE Private?**  
    `NO_V24_ARCHITECTURAL_TARGET`.

18. **Does Private confirm the locked Public result?**  
    **Yes.** Private results are uniformly negative for cross-depth combinations, confirming the public negative verdict.

19. **Is there sufficient evidence to justify v2.4?**  
    **No.**

20. **If yes, what TARGET CLASS is justified?**  
    `NO_V24_ARCHITECTURAL_TARGET`.

---

## 11. Final Scientific Verdict & v2.4 Gate

- **Public-Locked Next Target:** `NO_V24_ARCHITECTURAL_TARGET`
- **Confirmatory Status:** `PUBLIC_NEGATIVE_PRIVATE_OBSERVATIONAL_ONLY`
- **Final A7 Scientific Decision:** `NO_ACTIONABLE_COMPLEMENTARITY`
- **v2.4 Gate:** `V24_NOT_JUSTIFIED`
- **Operational Verdict:** `A7_COMPLETE_NO_V24_TARGET`

### Key Takeaway for the Research Program

The empirical finding is clean and definitive: **Early Motif representations do not contain held-out complementary information that can be linearly recovered to improve FER generalization.**  
The apparent "error set divergence" between early and late layers observed in v2.3 does not reflect distinct, high-quality semantic information that was accidentally dropped by subsequent layers. Rather, early layers are simply noisier, less-converged predictors whose correct calls on a subset of samples cannot be harnessed without incurring an equal or greater rate of false positives.  

Attempting to design an MPG-FER v2.4 architecture centered around cross-depth skip connections, multi-depth feature aggregation, or complex early-readout mechanisms is **empirically unjustified and scientifically blocked**.
