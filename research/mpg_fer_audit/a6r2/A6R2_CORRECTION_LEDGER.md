# MPG-FER A6-R2: Authoritative Correction Ledger & Audit Trail

**Document Identity:** `A6R2_CORRECTION_LEDGER.md`<br>
**Purpose:** Exhaustive forensic ledger cataloging every identified inconsistency, root cause, and authoritative correction across the A6 and A6-R audit lineage. Historical artifacts under `a6/` and `a6r/` remain preserved intact; this document serves as the binding reference for all downstream research.

---

## 1. Comprehensive Correction Ledger

| Ledger ID | Faulty Artifact / Statement | Root Cause Analysis | Corrected State in A6-R2 | Authoritative Reference |
| :--- | :--- | :--- | :--- | :--- |
| **LEDGER-01** | `a6r_readout_factorial_samples.csv` had `base_correct = 68` (11.47%) for `V21_CORRECT_V22_WRONG`. | **Variable Overwrite Collision:** In `run_a6r_audit.py` (lines 659–668), variable `l_n22_r22` was defined for Direction 1, but overwritten on line 668 by Direction 2 (`m21.classifier(...)`). Downstream evaluation of baseline receiver v2.2 evaluated this shadowed variable, accidentally running the correct v2.1 model. | **Repaired receiver suffix:** `F00` reproduces original receiver logits with max abs error $< 1.1 \times 10^{-6}$, and correctness is **identically 0 / 593 (0.00%)** and **0 / 604 (0.00%)** in both directions. | `a6r2_readout_replay_validation.json`, `a6r2_readout_factorial_results.json` |
| **LEDGER-02** | `a6r_readout_factorial_results.json` reported `F11 = 100.0%` for Direction A and `89.07%` for Direction B. | Suffix variable collision caused Direction A F11 to evaluate without the receiver's global pixel readout. | **Exact A6 S7 Equivalence:** Under the strictly receiver-controlled suffix (`concat([p_rec, m_rd]) -> rec_m.classifier`), F11 achieves **89.38% (530 / 593)** in Direction A and **89.07% (538 / 604)** in Direction B, reproducing historical A6 S7 predictions with **zero mismatches**. | `a6r2_readout_replay_validation.json` |
| **LEDGER-03** | `a6r_hypothesis_decisions.json` reported stale Composer numbers: `what_only = 15.01% / 13.91%` and `what_where = 34.74% / 34.27%`. | These figures were generated from an early prototype script before full 3-scale composition was hooked, and were never updated after the final `a6r_composer_swap_results.json` run. | **Authoritative Composer Swaps:** `WHAT_ONLY` achieves **36.93% (219 / 593)** in Dir A and **36.59% (221 / 604)** in Dir B. `WHAT+WHERE` achieves **36.93%** (Dir A) and **37.09%** (Dir B). Incremental gain of `TYPE` over `WHAT` is **$\le 0.50\%$**. | `a6r2_master_results.json`, `a6r_composer_swap_results.json` |
| **LEDGER-04** | `a6r_routing_sample_association.json` claimed sample-level routing correlation, but stored only macro layer-average Jaccards from A4. | The script correlated true-class margin deltas with swap rescue, but substituted a single scalar macro Jaccard per layer instead of evaluating per-sample routing divergence. | **True Per-Sample Routing Divergence:** Extracted per-sample Jaccard and divergence ($1 - \text{Jaccard}$) across all 5 layers for all 1,197 samples. Evaluated true per-sample Spearman and point-biserial correlations. | `a6r2_routing_per_sample.csv`, `a6r2_routing_association.json` |
| **LEDGER-05** | Claim that routing divergence strongly predicts rescue. | Conflation of representation margin delta with routing divergence. When tested at the true sample level, correlation between routing divergence and swap rescue is near zero ($r = -0.054$ to $+0.040$, all $p \ge 0.193$). | **Routing Status = `MIXED`:** Routing divergence does *not* provide an isolated causal threshold for rescue; it acts as an integrated GNN propagation substrate. | `a6r2_routing_association.json` |
| **LEDGER-06** | In `a6_swap_results.json`, `mean_rescue_margin` was negative (e.g. $-1.06$ at S1) for supposedly rescued samples. | Semantic ambiguity: the metric averaged margins over the *entire resolvable cohort* (including unrescued samples with large negative margins), rather than rescued samples only. | **Semantic Separation:** Defined `mean_swap_margin_all` (cohort mean) and `mean_margin_rescued_only` (strictly positive: $+2.1$ to $+2.6$ on rescued samples). | `a6r2_master_results.json` |
| **LEDGER-07** | Ambiguous usage of "clean" vs "strict clean" sample cohorts across A6 and A6-R. | Lack of formal hierarchy in sample filtering documentation. | **Explicit Cohort Hierarchy:** `MODEL_RESOLVABLE_ALL` ($N=1,197$); `A6_CLEAN` ($N=1,187$, 99.16% retention); `A6_STRICT_CLEAN` ($N=1,161$, 97.00% retention). | `a6r2_provenance.json`, `a6r2_strict_clean_sensitivity.json` |
| **LEDGER-08** | Target class over-expansion: A6-R proposed multi-module `DISTRIBUTED_INFORMATION_PRESERVATION_TARGET`. | A6-R did not have the repaired Readout Factorial and true per-sample routing results to narrow the target scope. | **Narrow Target Justified:** With readout operator confirmed as secondary to node states (74% vs 21%), routing association confirmed as near-zero, and the generalization gap expanding sharply at Layers 1–2 (+16% gap jump), Section 28 rules mandate **`EARLY_DEPTH_GENERALIZATION_TARGET`**. | `a6r2_final_target.json` |

---

## 2. Stale Number Sweep & Audit Trail

| String / Number | Historical Location | Historical Status | A6-R2 Authoritative Value | Authoritative Context |
| :---: | :---: | :---: | :---: | :--- |
| `15.01%` / `13.91%` | `a6r_hypothesis_decisions.json` | **STALE_DERIVED** | **36.93% / 36.59%** | What-only functional rescue rate |
| `34.74%` / `34.27%` | `a6r_hypothesis_decisions.json` | **STALE_DERIVED** | **36.93% / 37.09%** | What+Where functional rescue rate |
| `20.07%` / `20.70%` | `a6r_hypothesis_decisions.json` | **STALE_DERIVED** | **21.75% / 21.85%** | Readout operator only rescue rate (repaired suffix) |
| `71.13%` / `72.85%` | `a6r_hypothesis_decisions.json` | **STALE_DERIVED** | **71.33% / 74.17%** | Node states only rescue rate (repaired suffix) |
| `100.0%` (F11 Dir A) | `a6r_readout_factorial_results.json` | **STALE_DERIVED** | **89.38%** | F11 both swapped (repaired suffix, matches S7 exactly) |
| `base_correct = 68` | `a6r_readout_factorial_samples.csv` | **STALE_DERIVED** | **0 / 593 (0.00%)** | F00 receiver baseline correct count |
| `r = +0.22 to +0.34` | `a6r_routing_sample_association.json` | **STALE_DERIVED** | **r = -0.054 to +0.040** | True per-sample routing divergence vs. rescue correlation |

---

## 3. Verification Protocol

All headline figures in `A6R2_READOUT_ROUTING_SOURCELOCK.md`, `a6r2_hypothesis_decisions.json`, and `a6r2_final_target.json` are programmatically verified against `a6r2_master_results.json` via `verify_a6r2_source_consistency.py`. The two derived JSON documents are regenerated by `regenerate_a6r2_derived.py`; zero manual number copying is permitted.
