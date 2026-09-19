# H1 Train-Only Sparse Information Retention and Relational Necessity Preregistration

## 0. Objective and Research Questions

Following the accepted, frozen Train finalization of Issue #86 (`|Q_M| = 21`, `|Q_C| = 23`, `Q_star = 21`), this study conducts an out-of-fold diagnostic entirely within official FER2013 Train to determine where discriminative expression information is lost between dense CRS representation, qualification filtering, connected-component sparsification, and omission of spatial pairwise relations.

This diagnostic addresses exactly four registered scientific questions:

1. **Q1 (Vocabulary Filtering Loss)**: How much predictive information is lost when the frozen dense M vocabulary is restricted from all 512 CRS types to the 21 qualified M types?
   - Comparison: Arm B minus Arm A (`Δ_vocab = B - A`)
2. **Q2 (Component Sparsification Loss)**: Given the same fixed qualified vocabulary $Q_M$, how much predictive information is lost by replacing dense qualified-type occupancy with the frozen one-representative-per-connected-component sparse occurrence representation?
   - Comparison: Arm C minus Arm B (`Δ_sparse = C - B`)
3. **Q3 (Pairwise Relational Necessity)**: Does deterministic pairwise type-and-direction geometry among the frozen sparse candidate occurrences add out-of-fold FER signal beyond the unary sparse occurrence histogram?
   - Comparison: Arm D minus Arm C (`Δ_rel = D - C`)
4. **Q4 (Geometry-Specific Control)**: Is any predictive gain from the pairwise relation feature block specifically due to registered directional spatial geometry rather than merely adding a larger type-pair count feature block?
   - Comparison: Arm D minus Arm E (`Δ_geom = D - E`)

Graph construction remains locked. No PublicTest or PrivateTest data is accessed.

---

## 1. Frozen Prior Artifacts and Invariants

This study strictly consumes upstream outputs without modification:

- **Official Train Rows**: `28,709` images
- **Train CSV SHA256**: `deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8`
- **Canonical Substrate Identity SHA256**: `ee5d8262f7ec439bd6e2cd8ad01a17df17cd152da10b474c49ea93872fbfc46c`
- **Accepted CRS Train Model SHA256**: `77b8a41d4a7de79b2216b4b9c7ad2d19e326cac25a46ec2a7a3dcaaa1f95607a`
- **Upstream Finalizer Artifacts**:
  - `motif_stability_merged.npz` (`aa7b3c14e2c544eff673969366c3e1801abe33ba0de2611159946b027a0b1f2b`)
  - `motif_train_occurrence_diagnostics.npz` (`218838aeaefd575b96a15037e84f52d966ddae66b86850fea7f394febe146658`)
  - `motif_train_s_m_all_features.npz` (`900e0fd07aa7bb9ef0eec947246df8d6d66d92603508994e1c09fcbca71092ad`)
  - `motif_train_summary.json` (`ea191bd389c6cad44c4eb6f1ff600559b38c649eec904e4d886a3f9003963d18`)
  - `motif_train_execution_manifest.json` (`3ad2d559fe33ce4d852156a06ae2a778e3cff1ea460983cb2b7cb24495e29786`)

### Frozen Qualified Vocabulary ($Q_M$)
$Q_M$ is loaded directly from `motif_stability_merged.npz` and verified to equal the exact 21 types:
```text
Q_M = [31, 81, 116, 118, 123, 125, 167, 169, 171, 174, 179, 195, 223, 305, 311, 344, 381, 440, 451, 479, 491]
```
No re-qualification, threshold search, or vocabulary expansion is permitted.

---

## 2. Evaluation Protocol: Frozen Train Out-Of-Fold (OOF)

To prevent same-fit overestimation, all arms are evaluated via 5-fold stratified cross-validation on official Train:

- **Splitter**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Shared Folds**: Exactly the same 5 folds are shared across all Arms A, B, C, D, and E.
- **Coverage**: Every Train row appears in exactly one validation fold. Concatenated OOF predictions form a complete 28,709-row vector for primary evaluation.
- **Classifier**: `LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=5000, tol=1e-4)`.
  - No hyperparameter tuning.
  - No label-guided feature selection, PCA, or learned embeddings.
  - Every fold must converge (`converged == True`).

---

## 3. Representation Arms (A through E)

| Arm | Code Identifier | Description | Feature Dimension | Normalization |
| :--- | :--- | :--- | :---: | :--- |
| **Arm A** | `M_DENSE_ALL` | Dense 36×36 CRS assignment map using all 512 M types | `2560` (5×512) | Region L1, global L2 |
| **Arm B** | `M_DENSE_Q` | Dense 36×36 CRS assignment map masked to $Q_M$ (non-$Q_M$ bins zeroed, 512 coordinate space kept) | `2560` (5×512) | Region L1, global L2 |
| **Arm C** | `M_SPARSE_Q` | Frozen official $S_{M,all}$ sparse occurrence spatial pyramid (`motif_train_s_m_all_features.npz`) | `2560` (5×512) | Frozen official |
| **Arm D** | `M_SPARSE_Q_PLUS_REL` | Concatenation of unary Arm C ($U$) and pairwise directional sector relation block ($R$) | `6088` (2560 + 3528) | $U, R$ L2-normed, concat, global L2 |
| **Arm E** | `M_SPARSE_Q_PLUS_REL_CONTROL` | Concatenation of unary Arm C ($U$) and image-specific sector-permuted control block ($R_{control}$) | `6088` (2560 + 3528) | $U, R_{ctrl}$ L2-normed, concat, global L2 |

### Construction of Pairwise Relation Feature $R$ (Arm D)
1. **Nodes**: Reconstructed deterministically from Issue #86 occurrence logic ($S_{M,all}$ occurrences). Verified to reproduce the 48 zero-node images and occurrence counts identically.
2. **Type Mapping**: $Q_M$ types mapped to indices $0..20$ in ascending numerical order.
3. **Direction Sectors**: For each ordered pair of nodes $i \ne j$, $(\Delta r, \Delta c) = (row_j - row_i, col_j - col_i)$. Direction is assigned to 1 of 8 sectors:
   - 0: NW $(-1, -1)$
   - 1: N  $(-1, 0)$
   - 2: NE $(-1, +1)$
   - 3: W  $(0, -1)$
   - 4: E  $(0, +1)$
   - 5: SW $(+1, -1)$
   - 6: S  $(+1, 0)$
   - 7: SE $(+1, +1)$
4. **Dimension**: $21 \times 21 \times 8 = 3,528$ dimensions. Count vector is globally L2-normalized when non-zero. Images with $< 2$ nodes receive the all-zero vector.

### Construction of Relation Control $R_{control}$ (Arm E)
- For each image independently, permute the 8 sector labels deterministically using seed derived as:
  $$\text{seed} = \text{uint64\_be}(\text{SHA256}(\text{"42\|relation\_control\|"} + \text{canonical\_image\_id})[:8])$$
- Apply this image-specific sector permutation to all ordered pairs in that image, destroying consistent spatial directional meaning across images while preserving type counts, ordered pair counts, and norm.

---

## 4. Primary Metrics and Paired Bootstrap Evaluation

- **Primary Metrics**: Out-Of-Fold Accuracy and Out-Of-Fold Macro-F1 across all 28,709 images.
- **Paired Bootstrap**:
  - $B = 2,000$ bootstrap resamples with replacement of size 28,709.
  - RNG seeded with `Generator(PCG64(42))`.
  - Shared bootstrap resample indices across all arms and comparisons.
  - 95% two-sided confidence intervals computed via empirical percentiles (2.5% and 97.5%).

---

## 5. Registered Scientific Decision and Interpretation Rules

All comparisons are fixed diagnostic evaluations; they do not trigger model selection or pipeline modifications.

### 1. Vocabulary Filtering Loss (`B - A`)
- **Supported** iff:
  $$\text{Upper 95\% CI}(\Delta_{\text{vocab, Acc}}) < 0 \quad \text{AND} \quad \text{Upper 95\% CI}(\Delta_{\text{vocab, Macro-F1}}) < 0$$
- **Verdict**:
  - If true: `QUALIFIED-VOCABULARY FILTERING LOSS SUPPORTED`
  - Otherwise: `QUALIFIED-VOCABULARY FILTERING LOSS NOT ESTABLISHED`

### 2. Component Sparsification Loss (`C - B`)
- **Supported** iff:
  $$\text{Upper 95\% CI}(\Delta_{\text{sparse, Acc}}) < 0 \quad \text{AND} \quad \text{Upper 95\% CI}(\Delta_{\text{sparse, Macro-F1}}) < 0$$
- **Verdict**:
  - If true: `COMPONENT SPARSIFICATION LOSS SUPPORTED`
  - Otherwise: `COMPONENT SPARSIFICATION LOSS NOT ESTABLISHED`

### 3. Pairwise Relational Necessity (`D - C` and `D - E`)
- **Supported** iff:
  $$\text{Lower 95\% CI}(D_{\text{Acc}} - C_{\text{Acc}}) > 0 \quad \text{AND} \quad \text{Lower 95\% CI}(D_{\text{Macro-F1}} - C_{\text{Macro-F1}}) > 0$$
  $$\text{AND}$$
  $$\text{Lower 95\% CI}(D_{\text{Acc}} - E_{\text{Acc}}) > 0 \quad \text{AND} \quad \text{Lower 95\% CI}(D_{\text{Macro-F1}} - E_{\text{Macro-F1}}) > 0$$
- **Verdict**:
  - If both hold: `SPARSE PAIRWISE RELATIONAL SIGNAL SUPPORTED`
  - Otherwise: `SPARSE PAIRWISE RELATIONAL SIGNAL NOT ESTABLISHED`

---

## 6. Zero-Node Diagnostic Investigation

Mechanically inspect whether the 48 zero-occurrence images in Arm C ($S_{M,all}$) originate from zero dense qualified type occurrences in Arm B:
- If the zero-sum indicator sets of Arm B and Arm C are identical:
  `ZERO-NODE COVERAGE FAILURE ORIGINATES AT QUALIFIED-VOCABULARY COVERAGE, NOT COMPONENT COLLAPSE`
- Report FER label distribution of the 48 zero-node images relative to official Train.
- No dummy nodes or threshold adjustments may be introduced.

---

## 7. Explicit Prohibitions

This study must NOT:
1. Modify thresholds ($0.75$ Jaccard, $288$ support).
2. Expand or modify $Q_M$.
3. Tune $K$, distance bins, or logistic regression hyperparameters.
4. Introduce graph neural networks, attention mechanisms, or learned edge embeddings.
5. Access or evaluate PublicTest or PrivateTest.
