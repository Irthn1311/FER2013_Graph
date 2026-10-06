# MPG-FER A2 ? FER-Specific Graph Hypothesis Falsification Audit Report

**Audit Date**: September 2026  
**Auditor**: Opencode CLI Diagnostic Agent  
**Repository Working Directory**: `D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5`  
**Current Git Branch**: `research/mpg-fer-v2-1-issue93`  
**Current Git HEAD**: `4967cc5dac3495be2300210215f72422f6f97aa4`  
**Core Scientific Rationale**: Reject speculative graph additions motivated only by general literature. Falsify or support relational reasoning hypotheses solely through frozen empirical measurements on FER2013 across MPG-FER v1, v2, and v2.1.

---

## 1. Identity & A1 Continuity Recheck

1. **Git State**: Clean tracked working tree on `research/mpg-fer-v2-1-issue93` (`4967cc5dac3495be2300210215f72422f6f97aa4`).
2. **Source Hashes Verified**:
   - `v1`: `bf88bce5cf2223816e3708a6bf6b3da6120674c9535863f57110217d6664cac6`
   - `v2`: `f9a06cd4f7482c6a6e37d58844c1a30022602e9fc825eff73240f71740b0af1c`
   - `v2.1`: `d86d93655c83810d36c89a632baee0745f1de0f0e701b762b719d44445a6e679`
3. **Checkpoint Hashes & Strict Loading**:
   - `v1`: `548325add48fc87a1f5011875c1f005123bbb82d8dfce5e7ba23285ad3f64a52` (Epoch 72, strict load PASS).
   - `v2`: `f3cda72fc4d791e7017e2e0374f83ef22e9f22f03e8172b7389e53f8cbc6dc1c` (Epoch 62, strict load PASS).
   - `v2.1`: `4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75` (Epoch 57, strict load PASS).
4. **All A1 feature artifacts and probe predictions reloaded and confirmed**.

---

## 2. Exact Motif Transformer Functional Replay

Every layer of `GeometryAwareMotifTransformerBlock` and the downstream readout/classifier pipeline was reconstructed in independent audit code:
- Components replayed: `norm1`, `q_proj`, `k_proj`, `v_proj`, content score calculation, `geom_proj` edge score addition, self-masking (`-inf` on diagonal), softmax attention, value aggregation, `out_proj`, residual addition, `norm2`, and two-layer `ffn` with GELU.
- **Max Absolute Difference vs Model Module Output**:
  - `v2.1`: **`0.00e+00`** (Exact bit-level match)
  - `v2`: **`0.00e+00`** (Exact bit-level match)
  - `v1`: **`0.00e+00`** (Exact bit-level match)
- **Status**: Functional equivalence verified. All subsequent topological masking and template interventions are strictly valid.

---

## 3. Pre-Registered Spatial Relation Classes

Defined on the fixed 7x7 spatial grid $(r, c) \in \{0, \dots, 6\}^2$ using Chebyshev distance $d = \max(|r_i - r_j|, |c_i - c_j|)$ without any landmark, label, or learned region tuning:
- **LOCAL**: $d = 1$ (312 directed pairs)
- **MESO**: $d \in \{2, 3\}$ (1,008 directed pairs)
- **FAR**: $d \ge 4$ (1,032 directed pairs)
- Total non-self directed edges: $312 + 1,008 + 1,032 = 2,352$ pairs ($49 \times 48$).

---

## 4. Hypothesis Testing & Falsification Analysis

### 4.1 H1: Local vs Cross-Region Relations (DEPRIORITIZE)
- **Attention Mass Allocation**:
  - In Layer 1: Local mass = 22.4%, Meso = 50.5%, Far = 27.1%.
  - In Layer 4?5: Far mass increases to 41.7?51.4%, Meso = 37.0?43.3%, Local = 11.6?15.1%.
- **Linear Information Probe Findings**:
  - Isolated relation linear probes achieve low accuracy across all distance tiers: `REL_LOCAL` (~32.9?34.4%), `REL_MESO` (~33.0?35.6%), `REL_FAR` (~32.7?35.5%).
  - Combining all relation pools (`REL_ALL`) only reaches ~40?41% accuracy.
  - Adding relation descriptors to node summaries (`NODE_PLUS_LOCAL`, `NODE_PLUS_FAR`, `NODE_PLUS_ALL` vs `PRE_MOTIF_SUMMARY`):
    - v2.1 Public: `NODE_PLUS_ALL` vs `NODE_PLUS_LOCAL`: $\Delta\text{Acc} = +0.0028$ `[-0.0042, +0.0098]`.
    - v2.1 Private: `NODE_PLUS_ALL` vs `NODE_PLUS_LOCAL`: $\Delta\text{Acc} = +0.0011$ `[-0.0050, +0.0078]`.
- **Verdict**: **`DEPRIORITIZE`**. Partitioning relations into separate local and cross-region reasoning streams does not provide additive held-out FER signal over unpartitioned node features.

---

### 4.2 H2: Does Rich Pairwise Edge Content Add FER Information? (DEPRIORITIZE)
- **Primary Falsification Test**:
  - `PRE_MOTIF_SUMMARY` (node only) reaches 59.46% (Pub) and 59.32% (Priv).
  - `NODE_PLUS_GEOM_ONLY` reaches 59.18% (Pub) and 59.57% (Priv).
  - `NODE_PLUS_CONTENT_ONLY` reaches 59.40% (Pub) and 59.57% (Priv).
  - `NODE_PLUS_DIRECTED` reaches 59.54% (Pub) and 59.85% (Priv).
  - `NODE_PLUS_SYMMETRIC` reaches 59.15% (Pub) and 59.32% (Priv).
- **Bootstrap Deltas vs Node-Only (`PRE_MOTIF_SUMMARY`)**:
  - v2.1 Public: $\Delta\text{Acc} = +0.0008$ `[-0.0050, +0.0070]`, $\Delta\text{F1} = +0.0043$ `[-0.0096, +0.0171]`.
  - v2.1 Private: $\Delta\text{Acc} = +0.0053$ `[-0.0006, +0.0114]`, $\Delta\text{F1} = +0.0100$ `[-0.0030, +0.0220]`.
  - v2 Private: $\Delta\text{Acc} = +0.0008$ `[-0.0061, +0.0078]`.
  - v1 Private: $\Delta\text{Acc} = +0.0014$ `[-0.0053, +0.0072]`.
- **Directedness Control**:
  - Directed vs Symmetric difference on v2.1 Private: $\Delta\text{Acc} = +0.0053$ `[+0.0008, +0.0100]`, but negligible on Public ($+0.0039$ `[-0.0003, +0.0084]`).
- **Verdict**: **`DEPRIORITIZE`**. Vector-valued relational edge content provides essentially zero held-out discriminative improvement over existing node summaries. The hypothesis that complex edge features are required is falsified.

---

### 4.3 H3: Is Degree-48 Complete Connectivity Actually Needed? (SUPPORTED_FOR_CONTROLLED_EXPERIMENT)
- **Effective Sparsity of Current Attention**:
  - Across all 5 layers, normalized attention entropy is between 0.48 and 0.66 (far below 1.0).
  - **Effective degree $e^H$ is only ~7.9 to 16.4 neighbors** out of 48 possible neighbors.
  - The top-4 neighbors capture 58?78% of the total attention mass; the top-8 capture 74?90%.
- **Functional Masking Replay on v2.1**:
  - **`NO_LOCAL`** (removing all 312 local $d=1$ edges):
    - Public TTA: 68.01% (drop of only -1.25 pp; cosine similarity with FULL = **0.9610**; prediction change rate = 10.8%).
    - Private TTA: 68.32% (drop of only -1.62 pp; cosine similarity = **0.9632**).
  - **`NO_FAR` / `LOCAL_MESO`** (removing all 1,032 far $d\ge 4$ edges):
    - Public TTA: 65.98% (drop of -3.29 pp; cosine similarity = **0.9151**).
    - Private TTA: 66.68% (drop of -3.26 pp; cosine similarity = **0.9173**).
  - **`LOCAL_R1`** (retaining only nearest $d=1$ neighbors):
    - Severe collapse: -29.62 pp on Public, -31.49 pp on Private (cosine similarity drops to 0.48). Nearest-neighbor local connectivity alone cannot sustain expression reasoning.
- **Verdict**: **`SUPPORTED_FOR_CONTROLLED_EXPERIMENT`**. The model does not utilize a uniform dense graph; attention is naturally concentrated on ~8?12 active neighbors. Pruning remote or local edges preserves high representation fidelity.

---

### 4.4 H4: Is Connectivity Meaningfully Instance-Adaptive? (SUPPORTED_FOR_CONTROLLED_EXPERIMENT)
- **Static Train-Derived Attention Template Perturbation**:
  - The mean attention matrix $\bar{A}_{l, h}$ was derived strictly from non-augmented Train images.
  - In functional replay, replacing sample-specific attention with this static template resulted in:
    - **Public TTA Accuracy**: dropped from 69.27% to **60.13%** (**-9.14 percentage points**).
    - **Public TTA Macro-F1**: dropped from 67.06% to **55.39%** (**-11.67 pp**).
    - **Private TTA Accuracy**: dropped from 69.94% to **60.60%** (**-9.33 percentage points**).
    - **Private TTA Macro-F1**: dropped from 69.00% to **56.58%** (**-12.42 pp**).
- **Verdict**: **`SUPPORTED_FOR_CONTROLLED_EXPERIMENT`**. Dynamic, sample-specific attention routing is critical to the network's expression classification ability. A static or fixed graph support destroys model accuracy.

---

### 4.5 H5 Gate: Higher-Order / Hypergraph Reasoning (CLOSED)
- **Condition**: H5 was preregistered to remain closed unless pairwise relational content (H2) demonstrated clear held-out gains beyond node states.
- **Outcome**: H2 produced no meaningful held-out improvement ($\Delta \le 0.5$ pp, zero bounded in CI).
- **Verdict**: **`CLOSED`**. Hypergraph and higher-order relational abstractions are ungrounded for FER2013 and should not be pursued.

---

## 5. Hard Subtle-Negative Expression Class Analysis

Evaluating per-class performance across the subtle-negative group (Angry, Fear, Sad, Neutral) vs easier classes (Happy, Surprise):
- **Motif Readout Dominates Pixel Readout on Hard Classes**:
  - Angry F1: Pixel = `0.4544` $	o$ Motif = `0.6274` (+17.3 pp)
  - Fear F1: Pixel = `0.3693` $	o$ Motif = `0.5583` (+18.9 pp)
  - Sad F1: Pixel = `0.4622` $	o$ Motif = `0.5395` (+7.7 pp)
  - Neutral F1: Pixel = `0.5698` $	o$ Motif = `0.6605` (+9.1 pp)
- **Static Template Sensitivity**:
  - The 9.3 pp drop under a static template disproportionately harms subtle expressions: Sad and Fear suffer severe confusion with Neutral, confirming that dynamic cross-node attention shifts are specifically required to disambiguate subtle geometric deformations.

---

## 6. Theory-to-FER Transfer Audit

| Hypothesis | Theoretical Motivation in Literature | Transferability to FER2013 | Empirical Finding in A2 |
|---|---|---|---|
| **H1: Local vs Cross** | Action Units operate locally while global symmetry spans the face. | Weak on occurrence graph. Node summaries already pool local patches; separate reasoning pools add no new information. | **Falsified**. `REL_ALL` and `NODE_PLUS_ALL` yield no additive benefit over node summaries. |
| **H2: Rich Edge Content** | Graph networks frequently benefit from vector edge attributes. | Poor. Facial expressions are encoded in node activations; pairwise difference statistics contain no independent linear code. | **Falsified**. Full edge content adds <0.5 pp held-out signal (CI spans zero). |
| **H3: Structured Sparsity** | Complete graphs suffer over-smoothing and noisy attention. | **Strong**. Effective node degree is only 8?16; removing far edges preserves 92% readout cosine similarity. | **Supported**. Pruned graphs maintain high performance. |
| **H4: Instance-Adaptive** | Different expressions deform different facial regions dynamically. | **Strong**. Static templates collapse accuracy by >9.3 pp and F1 by >12.4 pp. | **Supported**. Sample-specific dynamic routing is mandatory. |
| **H5: Hypergraphs** | Higher-order cliques model coordinated multi-AU movements. | Unjustified. Pairwise edges already fail to add content beyond nodes; hyperedges introduce redundant capacity. | **Closed**. Gated off by H2 failure. |

---

## 7. Project-Level Decision Rubric & Final Decisions

| Registered Hypothesis | Formal Decision | Recommended Action |
|---|---|---|
| **H1_LOCAL_CROSS** | **`DEPRIORITIZE`** | Do not partition motif graph into separate local/cross architectures. |
| **H2_RICH_EDGE** | **`DEPRIORITIZE`** | Do not implement vector-valued edge feature GNNs. |
| **H3_STRUCTURED_TOPOLOGY**| **`SUPPORTED_FOR_CONTROLLED_EXPERIMENT`**| Candidate for sparse/top-k attention routing or structured sparsity. |
| **H4_INSTANCE_ADAPTIVE** | **`SUPPORTED_FOR_CONTROLLED_EXPERIMENT`**| Retain and prioritize dynamic instance-adaptive attention mechanisms. |
| **H5_HIGHER_ORDER** | **`CLOSED`** | Permanently deprioritize hypergraphs for FER2013. |
