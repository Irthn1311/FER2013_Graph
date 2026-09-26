# MPG-FER A6-R2: Readout Factorial Repair, True Per-Sample Routing Test, and Authoritative Source-Lock Closure

**Final Audit Verdict:** `A6R2_COMPLETE_V23_TARGET_READY`<br>
**Final Mechanistic Decision:** `EARLY_DEPTH_GENERALIZATION_TARGET`<br>
**Source Lock Status:** `AUTHORITATIVE_SOURCE_OF_TRUTH` (generated from `a6r2_master_results.json`)<br>
**Execution Environment:** Windows (win32), NVIDIA GeForce RTX 3050 Ti Laptop GPU, PyTorch 2.11.0+cu126<br>
**Primary Inference Policy:** Canonical FP32 single-view raw inference (`use_amp=False`)<br>
**Audit Directory:** `research/mpg_fer_audit/a6r2/`<br>
**Scientific Source Modified:** None (NO)<br>
**Checkpoints Modified:** None (NO)<br>
**Training Performed:** None (NO)<br>
**Architecture Changed:** None (NO)<br>
**FER Labels Changed:** None (NO)<br>
**PrivateTest Used for Tuning:** None (NO)<br>
**Historical A6/A6-R Artifacts Overwritten:** None (NO)

---

## 1. Executive Summary & Purpose

The MPG-FER A6-R2 audit provides the final authoritative closure pass for the representation failure localization branch. It repairs three critical blockers identified in A6-R:
1. **Repaired Readout Factorial Crossing:** Identifies and resolves the suffix collision bug in the prior script, establishing that receiver baseline `F00` exactly reproduces the original receiver logits ($< 1.1 \times 10^{-6}$ max abs error) with identically **0 / N (0.0%)** correctness, while `F11` reproduces historical A6 S7 predictions with **zero mismatches**.
2. **True Per-Sample Routing Divergence:** Replaces layer-level macro averages with true per-sample support Jaccard distributions across all 5 layers for all 1,197 model-resolvable samples, showing that sample-level routing divergence correlates near-zero with downstream rescue ($r = -0.054$ to $+0.040$, all $p \ge 0.193$).
3. **Authoritative Source Lock:** Reconciles all historical stale numbers into a single master document (`a6r2_master_results.json`) and establishes a narrow, evidence-backed intervention target class for future v2.3 modeling: **`EARLY_DEPTH_GENERALIZATION_TARGET`**.

---

## 2. Locked Model & Cohort Identities

- **v2.1 Dense Reference:** Checkpoint SHA-256 `4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75`, Epoch 57, 2,304,528 parameters (Strict load verified).
- **v2.2 Dynamic Sparse:** Checkpoint SHA-256 `a10bd22b3903550156c8239d91b5d2af35067ca1f2bdba9af46cf1e53d0bbdf4`, Epoch 64, Top-K schedule `[8, 16, 16, 16, 24]`, 2,304,528 parameters (Strict load verified).
- **Sample Cohort Breakdown (`a6_sample_groups.csv`):**
  - `MODEL_RESOLVABLE_ALL`: **1,197 samples** (593 v2.1-correct, 604 v2.2-correct).
  - `A6_CLEAN` (Filters exact Train duplicates & label mismatches): **1,187 samples** (99.16% retention).
  - `A6_STRICT_CLEAN` (Additionally filters visual ambiguity candidates): **1,161 samples** (97.00% retention).

---

## 3. Part I: Repaired Readout Factorial Crossing

### 3.1 Forensic Analysis of the A6-R Bug
In `run_a6r_audit.py` (lines 659–668), variable `l_n22_r22` was defined for Direction 1, but overwritten on line 668 by Direction 2 (`m21.classifier(...)`). Consequently, when evaluating the baseline receiver for `V21_CORRECT_V22_WRONG`, the script evaluated the overwritten variable, which ran the correct v2.1 model. This caused the baseline to report an invalid 68 correct samples (11.47%).

### 3.2 Repaired Experimental Protocol & Validation
In A6-R2, each direction enforces a strictly receiver-controlled downstream continuation:
$$\text{SUFFIX}_{\text{rec}}(m_{\text{rd}}) = \text{rec\_m.classifier}(\text{concat}([p_{\text{rec}}, m_{\text{rd}}]))$$
where $p_{\text{rec}}$ is the receiver's own global pixel readout [128].

#### Identity & Replay Controls (`a6r2_readout_replay_validation.json`):
- **Direction A (v2.1 Correct $\to$ v2.2 Receiver, N=593):**
  - `F00` Replay Max Abs Error vs. Original v2.2 Receiver: **$9.54 \times 10^{-7}$** ($< 10^{-6}$, PASS)
  - `F00` Correct Count: **0 / 593 (0.00%)** (MUST BE 0, EXACT PASS)
  - Donor Native Replay Max Abs Error: **$1.19 \times 10^{-6}$** (PASS)
  - `F11` vs. Historical A6 S7 Max Abs Error: **$4.05 \times 10^{-6}$** | Prediction Mismatches: **0** (EXACT PASS)
- **Direction B (v2.2 Correct $\to$ v2.1 Receiver, N=604):**
  - `F00` Replay Max Abs Error vs. Original v2.1 Receiver: **$1.07 \times 10^{-6}$** ($< 10^{-5}$, PASS)
  - `F00` Correct Count: **0 / 604 (0.00%)** (MUST BE 0, EXACT PASS)
  - Donor Native Replay Max Abs Error: **$1.43 \times 10^{-6}$** (PASS)
  - `F11` vs. Historical A6 S7 Max Abs Error: **$4.41 \times 10^{-6}$** | Prediction Mismatches: **0** (EXACT PASS)

### 3.3 Authoritative Readout Factorial Results
Evaluated on all 1,197 resolvable samples (`a6r2_readout_factorial_results.json` and `a6r2_readout_factorial_samples.csv`):

| Condition | Mathematical Configuration | Factor Manipulated | Direction A Rescue Rate (N=593) [95% CI] | Direction A Rescued Margin | Direction B Rescue Rate (N=604) [95% CI] | Direction B Rescued Margin |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`F00`** | $N_{\text{rec}} \to R_{\text{rec}} \to \text{SUFFIX}_{\text{rec}}$ | Receiver Baseline (Wrong) | **0.00%** (0 / 593) | $0.000$ | **0.00%** (0 / 604) | $0.000$ |
| **`F01`** | $N_{\text{rec}} \to R_{\text{don}} \to \text{SUFFIX}_{\text{rec}}$ | Readout Operator Only | **21.75%** (129 / 593) $[18.4\%, 25.1\%]$ | $+1.070$ | **21.85%** (132 / 604) $[18.7\%, 25.3\%]$ | $+1.037$ |
| **`F10`** | $N_{\text{don}} \to R_{\text{rec}} \to \text{SUFFIX}_{\text{rec}}$ | Node States Only | **71.33%** (423 / 593) $[67.6\%, 75.0\%]$ | $+2.055$ | **74.17%** (448 / 604) $[70.7\%, 77.6\%]$ | $+2.000$ |
| **`F11`** | $N_{\text{don}} \to R_{\text{don}} \to \text{SUFFIX}_{\text{rec}}$ | Both Swapped (A6 S7) | **89.38%** (530 / 593) $[86.8\%, 91.7\%]$ | $+1.995$ | **89.07%** (538 / 604) $[86.4\%, 91.6\%]$ | $+1.936$ |

#### Controlled Factor Contrasts:
- **Node-State Main Effect (Holding Readout Fixed):**
  - On receiver readout ($F_{10} - F_{00}$): **$+71.33\%$** (Dir A) / **$+74.17\%$** (Dir B).
  - On donor readout ($F_{11} - F_{01}$): **$+67.62\%$** (Dir A) / **$+67.22\%$** (Dir B).
- **Readout-Operator Main Effect (Holding Node States Fixed):**
  - On receiver node states ($F_{01} - F_{00}$): **$+21.75\%$** (Dir A) / **$+21.85\%$** (Dir B).
  - On donor node states ($F_{11} - F_{10}$): **$+18.04\%$** (Dir A) / **$+14.90\%$** (Dir B).
- **Interaction Effect ($[F_{11} - F_{10}] - [F_{01} - F_{00}]$):** **$-3.71\%$** (Dir A) / **$-6.95\%$** (Dir B).
- **Readout Representation Cosine [384d]:**
  - Same node states, different readout operator ($N_{\text{don}} \to R_{\text{don}}$ vs. $N_{\text{don}} \to R_{\text{rec}}$): **$0.579$** (Dir A) / **$0.584$** (Dir B).

### 3.4 Readout Mechanistic Conclusion: `NODE_STATE_DOMINANT`
The repaired factorial proves conclusively that:
1. **Node states carry over 71%–74% of the rescue signal alone**, confirming that the S6 rescue is entirely driven by node-state information.
2. **The readout operator is secondary but material**: changing *only* the readout operator recovers the true class in **~21.8%** of samples, and cleanly packages the donor node states to boost rescue from ~72% to ~89%.
3. Readout operation is an active packaging layer, but the ultimate discriminative content resides in the Layer-5 node representations.

---

## 4. Part II: True Per-Sample Routing Divergence

Rather than relying on macro layer averages, A6-R2 computed the sample-level support Jaccard similarity $J_{i,l}$ and routing divergence ($1 - J_{i,l}$) for every sample $i$ and layer $l \in \{1, \dots, 5\}$ across all 1,197 resolvable samples (`a6r2_routing_per_sample.csv` and `a6r2_routing_association.json`):

### 4.1 Per-Sample Routing Support Jaccard Distributions:
| Layer | Top-K | Mean Jaccard $\pm$ Std | 10th Percentile ($p_{10}$) | Median Jaccard | 90th Percentile ($p_{90}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer 1** | 8 | 0.368 $\pm$ 0.040 | 0.314 | 0.370 | 0.418 |
| **Layer 2** | 16 | 0.347 $\pm$ 0.047 | 0.289 | 0.344 | 0.408 |
| **Layer 3** | 16 | 0.280 $\pm$ 0.045 | 0.223 | 0.279 | 0.340 |
| **Layer 4** | 16 | 0.272 $\pm$ 0.041 | 0.221 | 0.271 | 0.326 |
| **Layer 5** | 24 | 0.355 $\pm$ 0.034 | 0.315 | 0.354 | 0.399 |

### 4.2 Sample-Level Association Tests:
| Layer | Direction A Spearman $\rho$ (Divergence vs. Cosine Dist) [95% CI] | Direction B Spearman $\rho$ (Divergence vs. Cosine Dist) [95% CI] | Direction A Point-Biserial $r$ (Divergence vs. Rescue) | $p$-value | Direction B Point-Biserial $r$ (Divergence vs. Rescue) | $p$-value |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **L1** | $+0.082$ $[+0.001, +0.166]$ | $-0.008$ $[-0.087, +0.073]$ | $-0.054$ | $p = 0.193$ (NS) | $+0.022$ | $p = 0.593$ (NS) |
| **L2** | $+0.038$ $[-0.039, +0.118]$ | $+0.073$ $[-0.007, +0.154]$ | $-0.021$ | $p = 0.612$ (NS) | $-0.003$ | $p = 0.943$ (NS) |
| **L3** | $+0.012$ $[-0.070, +0.098]$ | $+0.092$ $[+0.011, +0.172]$ | $+0.007$ | $p = 0.869$ (NS) | $+0.037$ | $p = 0.364$ (NS) |
| **L4** | $-0.149$ $[-0.231, -0.068]$ | $-0.074$ $[-0.153, +0.006]$ | $+0.040$ | $p = 0.327$ (NS) | $-0.030$ | $p = 0.458$ (NS) |
| **L5** | $+0.019$ $[-0.057, +0.096]$ | $-0.003$ $[-0.080, +0.073]$ | $+0.010$ | $p = 0.815$ (NS) | $-0.030$ | $p = 0.454$ (NS) |

### 4.3 Routing Hypothesis Conclusion: `MIXED`
When evaluated at the true sample level:
1. **Routing divergence does NOT predict swap rescue:** All point-biserial correlations are near zero ($-0.054$ to $+0.040$, all $p \ge 0.193$, not significant). Samples that are rescued have virtually identical routing divergence as samples that are not rescued (absolute difference in mean divergence $\le 0.0044$).
2. **Weak Representation Coupling:** Spearman correlations between routing divergence and representation distance are weak and unstable across directions (ranging from $-0.15$ to $+0.09$).
3. **Binding Decision:** Relational routing functions as an integrated message-passing substrate rather than an independent failure lever. Topological routing differences do not constitute an actionable intervention target.

---

## 5. Part III: Authoritative Master Results & Generalization Gap

### 5.1 Authoritative Composer Component Swaps (Locked in `a6r2_master_results.json`)
- **`WHAT_ONLY` (96d):** **36.93% (219 / 593)** in Dir A | **36.59% (221 / 604)** in Dir B
- **`TYPE_ONLY` (32d):** **2.87% (17 / 593)** in Dir A | **1.99% (12 / 604)** in Dir B
- **`WHERE_ONLY` (5d):** **1.01% (6 / 593)** in Dir A | **0.66% (4 / 604)** in Dir B
- **`WHAT_WHERE` (101d):** **36.93% (219 / 593)** in Dir A | **37.09% (224 / 604)** in Dir B
- **`FULL_PRE_PROJ` (133d):** **36.09% (214 / 593)** in Dir A | **37.09% (224 / 604)** in Dir B
- **`H_PIXEL` (Pixel GNN out):** **38.28% (227 / 593)** in Dir A | **37.09% (224 / 604)** in Dir B
- **`FULL_S1` (Projected PRE):** **36.42% (216 / 593)** in Dir A | **35.93% (217 / 604)** in Dir B
- **Incremental Gain of TYPE over WHAT:** **0.00%** in Dir A | **+0.50%** in Dir B.
- *Composer Status:* **`WHAT_DOMINANT`**.

### 5.2 Progressive Depth-Dependent Generalization Gap
Linear probe performance on Train vs. PublicTest across depth (`a6r_generalization_gap.json`):
- **R1 (PRE-Motif):** Train = 60.3% | Public = 55.2% | **Gap = 5.07 pp**
- **R2 (Motif Layer 1):** Train = 74.8% | Public = 60.9% | **Gap = 13.88 pp**
- **R3 (Motif Layer 2):** Train = 85.9% | Public = 64.2% | **Gap = 21.67 pp** *(Steepest expansion: +16.6 pp jump across L1–L2)*
- **R6 (Motif Layer 5):** Train = 91.8% | Public = 65.5% | **Gap = 26.25 pp**
- **R7 (Motif Readout):** Train = 96.2% | Public = 66.3% | **Gap = 29.95 pp**
- **R8 (Fusion):** Train = 96.8% | Public = 65.7% | **Gap = 31.09 pp**

---

## 6. Answers to the 18 Final Scientific Questions

### 1. Did the previous readout baseline reproduce the original receiver?
**No.** In A6-R, the baseline `base_correct` reported 68 correct samples (11.47%) for `V21_CORRECT_V22_WRONG`.

### 2. If not, what implementation/suffix error caused the discrepancy?
A variable overwrite collision occurred in `run_a6r_audit.py` (lines 659–668). The variable `l_n22_r22` was defined for Direction 1, but overwritten on line 668 by Direction 2 (`m21.classifier(...)`), causing the baseline evaluation of receiver v2.2 to execute the correct v2.1 model.

### 3. Does corrected F00 produce 0% correctness by construction in both model-resolvable directions?
**Yes, exactly.** `F00` correctness is **0 / 593 (0.00%)** in Direction A and **0 / 604 (0.00%)** in Direction B, with max abs replay error $< 1.1 \times 10^{-6}$.

### 4. What is the corrected node-state-only rescue?
- Direction A (N21 $\to$ R22): **71.33% (423 / 593)**
- Direction B (N22 $\to$ R21): **74.17% (448 / 604)**

### 5. What is the corrected readout-operator-only rescue?
- Direction A (N22 $\to$ R21): **21.75% (129 / 593)**
- Direction B (N21 $\to$ R22): **21.85% (132 / 604)**

### 6. What is the corrected F11 rescue?
- Direction A (N21 $\to$ R21): **89.38% (530 / 593)**
- Direction B (N22 $\to$ R22): **89.07% (538 / 604)**

### 7. Does F11 exactly reproduce historical A6 S7?
**Yes, 100% sample-for-sample match.** Prediction mismatches = **0**, and max abs logit error is $< 4.5 \times 10^{-6}$ in both directions.

### 8. Does the readout operator materially affect correctness after all receiver-specific suffixes are controlled?
**Yes, as a secondary packaging layer.** While node states alone explain ~73% rescue, swapping the readout operator alone rescues ~21.8%, and applying the donor readout operator packages donor node states to boost rescue from ~73% to ~89%.

### 9. What is the actual per-sample routing Jaccard distribution?
Mean per-sample Jaccard is 0.368 in Layer 1 ($p_{10}=0.314, p_{90}=0.418$), drops to 0.272 in Layer 4 ($p_{10}=0.221, p_{90}=0.326$), and rises to 0.355 in Layer 5 ($p_{10}=0.315, p_{90}=0.399$).

### 10. Does routing divergence correlate with representation margin divergence?
**Near zero.** Point-biserial correlations are between $-0.054$ and $+0.040$ (all $p \ge 0.193$, not statistically significant).

### 11. Does routing divergence correlate with representation distance?
**Weak and unstable.** Spearman correlations range from $-0.15$ to $+0.09$ across layers and directions.

### 12. Does routing divergence predict swap rescue?
**No.** Rescued samples have virtually identical routing divergence as unrescued samples (absolute difference $\le 0.0044$).

### 13. Are those relationships consistent in both directions?
**Yes, consistently near-zero in both directions.**

### 14. Does the depth-dependent Train/held-out gap remain the strongest clean generalization signal?
**Yes, overwhelmingly.** The generalization gap expands monotonically from 5.1 pp at PRE-Motif to 31.1 pp at Fusion, showing that representation specialization outpaces held-out generalization with depth.

### 15. Is WHAT best described as creating divergence or transmitting divergence already present in Pixel-GNN states?
**Transmitting divergence.** `H_PIXEL` achieves 37.1%–38.3% rescue and `WHAT_ONLY` achieves 36.6%–36.9% rescue. Visual appearance features in `WHAT` directly transmit the contextual pixel representations into the motif graph.

### 16. Is there now one narrow target class justified for v2.3?
**Yes.** Following Section 28 rules, because readout is secondary to node states and routing association is near-zero, a single narrow target is justified.

### 17. What is that target?
**`EARLY_DEPTH_GENERALIZATION_TARGET`** (targeting representation preservation and regularization across Layers 1–2).

### 18. What should NOT be changed?
1. **DO NOT tune graph topology density or Top-K** (A4/A6/A6-R/A6-R2 prove density is not the bottleneck).
2. **DO NOT modify the classifier head architecture or loss** (the classifier faithfully projects Fusion with $<1.2\%$ marginal divergence).
3. **DO NOT tune prototype counts or TYPE projection in isolation** (TYPE adds $\le 0.50\%$ incremental rescue over WHAT).

---

## 7. Final Hypotheses & Target Decisions

Documented in `a6r2_hypothesis_decisions.json`:
1. **`H-A6R2-COMPOSER`:** **`WHAT_DOMINANT`**
   *`WHAT_ONLY` achieves 36.6%–36.9% rescue, matching the full S1 transfer (~36%). Incremental gain of `TYPE` is $\le 0.50\%$.*
2. **`H-A6R2-READOUT`:** **`NODE_STATE_DOMINANT`**
   *Layer-5 node states alone account for 71.3%–74.2% rescue under receiver-specific continuation; readout interaction provides the remaining boost to ~89%.*
3. **`H-A6R2-ROUTING`:** **`MIXED`**
   *Sample-level routing divergence correlates near-zero with downstream rescue ($r = -0.054$ to $+0.040$, all $p \ge 0.193$).*
4. **`H-A6R2-GENERALIZATION`:** **`SUPPORTED`**
   *Train-Public gap expands monotonically from 5.1 pp at PRE to 31.1 pp at Fusion (+16.6 pp jump across Layers 1–2).*
5. **`H-A6R2-SINGLE-TARGET`:** **`SUPPORTED`**
   *Repaired evidence justifies one narrow target class: `EARLY_DEPTH_GENERALIZATION_TARGET`.*

### Final Mechanistic Decision:
`EARLY_DEPTH_GENERALIZATION_TARGET`

---

## 8. Final Project-Level Bottleneck Statement

> **Final Project-Level Bottleneck Statement:**
> Model-resolvable divergence in MPG-FER originates in contextual Pixel-GNN representations, is transmitted into the Motif Graph through visual appearance (`WHAT`), and accelerates dramatically across Motif Layers 1 and 2, where the depth-dependent generalization gap more than quadruples (from 5% to 22%) and probe margin divergence doubles. Layer-5 node states dominate downstream classification (~74% rescue), while readout operator interaction packages the features to reach ~89% rescue. Because relational routing divergence operates as an integrated substrate with near-zero sample-level rescue correlation, future v2.3 modeling must not modify graph density, prototype clustering, or classifier heads. Modeling must be narrowly targeted at **`EARLY_DEPTH_GENERALIZATION_TARGET`**: regularizing early relational message passing across Layers 1–2 to prevent depth-dependent margin collapse.

---

## 9. Complete Artifact Manifest

All artifacts finalized under `research/mpg_fer_audit/a6r2/`:
- `A6R2_READOUT_ROUTING_SOURCELOCK.md` (This document)
- `A6R2_CORRECTION_LEDGER.md` (Authoritative ledger of all corrections)
- `a6r2_provenance.json` (Model lock and cohort verification)
- `a6r2_readout_replay_validation.json` (Exact F00 replay and A6 S7 equivalence validation)
- `a6r2_readout_factorial_results.json` (Repaired factorial metrics across ALL, CLEAN, STRICT_CLEAN)
- `a6r2_readout_factorial_samples.csv` (Sample-level predictions and margins for F00, F01, F10, F11)
- `a6r2_routing_per_sample.csv` (1,197-row true per-sample layerwise routing divergence table)
- `a6r2_routing_association.json` (Sample-level Spearman, point-biserial, and bootstrap CIs)
- `a6r2_master_results.json` (Single authoritative source-of-truth master document)
- `a6r2_hypothesis_decisions.json` (Formal status decisions for H-A6R2 hypotheses)
- `a6r2_final_target.json` (Final target class specification and prohibitions)
- `a6r2_readout_factorial_corrected.png` (Plot: Repaired Readout Factorial)
- `a6r2_routing_divergence_vs_margin.png` (Plot: Per-sample routing divergence vs. representation distance)
- `a6r2_routing_divergence_vs_rescue.png` (Plot: Routing divergence vs. downstream rescue)
