# MPG-FER A6-R: Formal A6 Corrections & Semantic Clarifications

**Document Purpose:** Authoritative closure document listing all semantic, numerical, and interpretive corrections to the original A6 report before A6/A6-R findings are utilized for future modeling. Historical A6 files are preserved intact; this document establishes the binding corrected definitions.

---

## 1. Correction Summary Table

| Issue | Original A6 Phrasing / Semantic | Binding Corrected A6-R Definition | Root Cause / Justification |
| :--- | :--- | :--- | :--- |
| **Correction A: S0 Swap Scope** | *"Pixel branch is irrelevant / Pixel GNN contributes <2.2% rescue."* | **"The direct global Pixel Readout [128] contributes little to cross-model correctness transfer (<2.2% rescue)."** | S0 swaps only the global pooled pixel vector entering Fusion. It does *not* replace the 2,304-node Pixel GNN representations consumed by the Spatial Motif Composer. |
| **Correction B: S1 Swap Attribution** | *"Spatial prototype assignment establishes 36% rescue / Composer prototypes are foundational."* | **"S1 swaps the complete 49-node PRE-Motif representation [49, 192], which is dominated by visual appearance (WHAT: 36.9% rescue), while prototype categorical assignment (TYPE) contributes <3%."** | PRE-Motif combines WHAT, TYPE, WHERE, and scale gating. A6-R functional subcomponent swaps prove WHAT is the active transfer component; TYPE alone transfers almost no correctness. |
| **Correction C: Cumulative Swap Shares** | *"Composer contributes 36%, Early Motif contributes 21%, Mid/Late Motif contributes 14%, Readout contributes 18%."* | **"Cumulative rescue rises monotonically from ~36% at PRE-Motif to ~58% after Layer 2, ~71% after Layer 5, and ~89% after Motif Readout."** | Stage swaps are nested/cumulative interventions. Intermediate percentage increments cannot be interpreted as independent additive causal shares. |
| **Correction D: Cross-Direction Symmetry** | *"Perfect bidirectional symmetry across all stages."* | **"Qualitatively symmetric / strongly mirrored across directions."** | While rescue rates track closely (within 1.5%), exact mathematical equality does not hold across distinct models. |
| **Correction E: Routing Association** | *"H-A6-ROUTING-LINK = SUPPORTED (based on layerwise Jaccard co-occurrence)."* | **"H-A6R-ROUTING-LINK = MIXED (based on true sample-level point-biserial correlations: r = +0.12 to +0.22)."** | Layer-level co-occurrence is insufficient to establish support. Sample-level correlations confirm an integrated relational mechanism, but not an isolated topology bottleneck. |
| **Correction F: Rescue Margin Semantics** | `mean_rescue_margin` in `a6_swap_results.json` showed negative values (e.g. $-1.06$ at S1) for rescued samples. | **Separated into `mean_swap_margin_all` (mean over full cohort) and `mean_margin_rescued_only` (strictly positive margin among rescued samples: +2.1 to +2.6).** | The A6 code averaged margins over the entire 1,197 resolvable cohort rather than rescued samples only. A correctly classified sample cannot have a negative true margin. |
| **Correction G: Clean vs Strict Clean** | Confusing references between "clean model-resolvable" and "ambiguity-filtered". | **`MODEL_RESOLVABLE_ALL` (N=1,197); `A6_CLEAN` (N=1,187, filters duplicates/mismatches); `A6_STRICT_CLEAN` (N=1,161, additionally filters visual ambiguity).** | Clarifies the exact hierarchical filtering thresholds. |

---

## 2. Detailed Technical Reconciliations

### 2.1 Correction A: Pixel Branch vs. Global Pixel Readout
In the MPG-FER architecture:
1. Raw image $x \in \mathbb{R}^{48 \times 48}$ is processed by `pixel_extractor`, `pixel_proj`, and a 4-layer `EdgeAwarePixelGNNLayer` to produce contextual pixel node states $h \in \mathbb{R}^{B \times 2304 \times 96}$.
2. Global pixel readout $p_{\text{readout}} \in \mathbb{R}^{B \times 128}$ is pooled from $h$ via mean, max, and attention pooling, and fed *only* to the final `fusion = concat([p_readout, m_readout])`.
3. Boundary S0 in A6 replaced *only* $p_{\text{readout}}$. Because $p_{\text{readout}}$ only acts as a skip-connection to the final linear layer, replacing it resulted in only 1.5%–2.2% rescue.
4. Crucially, the Spatial Motif Composer directly ingests the unpooled node states $h$. When donor $h$ is passed into receiver composer (`H_PIXEL` swap in A6-R), rescue reaches **37.1% to 38.3%**.
5. *Binding Conclusion:* The Pixel GNN representations *do* transfer substantial correctness via the composer. Only the direct global pixel readout bypass contributes little.

### 2.2 Correction B: Localization Inside the Composer (WHAT vs. TYPE vs. WHERE)
In A6, the 36.4% rescue at S1 (PRE-Motif Graph) was loosely attributed to "motif prototype assignment." A6-R directly dismantled the internal components of `SpatialMotifComposer` across all three scales (8, 12, 16):
- `WHAT` (visual appearance pooled from pixel states): **36.9% rescue**
- `TYPE` (prototype categorical assignment projected to 32d): **2.0% to 2.9% rescue**
- `WHERE` (geometric moments cx, cy, sx, sy, mass): **0.7% to 1.0% rescue**
- `WHAT + WHERE`: **36.9% to 37.1% rescue**
- `H_PIXEL` (entering pixel GNN states): **37.1% to 38.3% rescue**
- *Binding Conclusion:* S1 transfer is driven almost entirely by `WHAT` (visual features). `TYPE` (prototype categorical clustering) contributes $<3\%$ functional rescue. This directly validates and reinforces the earlier A1 finding that prototype assignments provide little conditional discriminative information over WHAT+WHERE.

### 2.3 Correction F: Margin Semantics Reconciliation
In `a6_swap_results.json`:
- S1 reported `mean_rescue_margin = -1.0634`.
- S2 reported `mean_rescue_margin = -0.2559`.
This occurred because the metric averaged true-class margins across all 593 or 604 resolvable samples (including unrescued samples that had large negative margins such as $-4.0$).
In A6-R:
- `mean_swap_margin_all` reflects the full cohort average ($-1.06$ at S1, $-0.26$ at S2, $+1.73$ at S7).
- `mean_margin_rescued_only` evaluates only samples where the receiver model flipped from incorrect to correct:
  - S1 rescued samples: **$+2.607$**
  - S2 rescued samples: **$+2.601$**
  - S3 rescued samples: **$+2.244$**
  - S7 rescued samples: **$+1.996$**
  - S8 rescued samples: **$+1.905$**
All rescued samples have strongly positive true-class margins, as logically required.
