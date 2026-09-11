# Pure-GNN v3.1 Implementation Contract

## 1. Scientific Status and Purpose

Pure-GNN v3.1 is an empirical **falsification platform** designed to test foundational hypotheses regarding graph-only representation learning on raw pixel lattices for Facial Expression Recognition (FER2013). It is not claimed as a final novel paper architecture.

### Research Questions
- **RQ-A**: Can graph-only local processing learn useful visual representations directly from raw FER2013 pixel intensities without Conv2D or a pretrained visual backbone?
- **RQ-B**: Does uniform receiver-relative non-local context improve over purely local graph processing?
- **RQ-C**: Does learned pair-specific non-local weighting improve over the SAME receiver-relative non-local computation with uniform weights? (Primary causal comparison: **G1 vs G0.5**).
- **RQ-D**: Does content-conditioned local communication improve over geometry-only local filtering? (**G1 vs G2**).
- **RQ-E**: Does fixed anti-aliased coarsening improve over simple deterministic mean/downsample coarsening? (**G1 vs G3**).

---

## 2. Hard Architectural Exclusions

The `pure_gnn_v31` package and all experimental conditions strictly exclude:
- `Conv2D` / `Conv1D` spatial convolutions
- CNN backbones or pretrained visual backbones
- MediaPipe priors or facial landmarks
- Semantic ROIs / RoIAlign
- Superpixel segmentations
- Transformer self-attention blocks / multi-head attention modules
- Supervised contrastive loss (SupCon)
- Feature-kNN or dynamic learned graph topologies
- Learned pooling layers
- Semantic region discovery
- Absolute $(x, y)$ node coordinate injection into node features
- Intermediate trainable classifier/readout bypasses

**Allowed primitives**:
- Pointwise Dense / MLP projections
- Fixed spatial graph indices and relative coordinate deltas $(\Delta x, \Delta y)$
- Fixed deterministic signal-processing coarsening (binomial anti-aliasing)
- Exact degree normalization and receiver-relative differencing

---

## 3. Architecture Specification

### 3.1 Input and Initial Graph
- **Input**: Raw $48 \times 48$ grayscale image ($N_1 = 2304$ nodes, 1 scalar intensity per node).
- **No absolute position**: Coordinates $(x, y)$ are not node features.
- **Initial Node Projection**: Pointwise Dense $1 \to 32$.
- **Graph Topology**: Fixed 8-neighbor grid graph on $48 \times 48$.
  - Interior nodes: degree 8.
  - Border nodes: degree 5.
  - Corner nodes: degree 3.
  - No synthetic, wrapped, or zero-padded virtual neighbors.
  - Neighbor aggregation degree-normalizes by the exact node degree $d_i$.

### 3.2 Local Adaptive Relation Block
For receiver node $i$ and neighbor sender $j \in \mathcal{N}(i)$:
- Raw difference vector: $r_{ij} = h_j - h_i$
- Directional coordinate deltas: $\Delta x \in \{-1, 0, 1\}$, $\Delta y \in \{-1, 0, 1\}$, direction ID $d \in \{0 \dots 7\}$.
- Local gate logit:
  $$s_{ij} = \text{MLP}_{\text{local\_gate}}\left([\text{LN}(h_i), \text{LN}(r_{ij}), \Delta x, \Delta y]\right)$$
- Scalar gate:
  $$g_{ij} = 1 + \tanh(s_{ij}) \in (0, 2)$$
- **Zero-Initialization**: The final linear projection of $\text{MLP}_{\text{local\_gate}}$ is initialized with weights $= 0$ and bias $= 0$, ensuring $s_{ij} = 0 \implies g_{ij} = 1.0$ at initialization.
- Directional message (the value path uses raw $h_j$ and raw $r_{ij}$; only
  gate copies are normalized):
  $$m_{ij} = g_{ij} \cdot \left( W_{\text{direction}}[d] \, h_j + W_{\text{relation}} \, (h_j - h_i) \right)$$
- Real-degree aggregation:
  $$m_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} m_{ij}$$
- Pre-norm residual update:
  $$h_i \leftarrow h_i + m_i$$
  $$h_i \leftarrow h_i + \text{FFN}(\text{LN}(h_i))$$
  where $\text{FFN}$ is a pointwise two-layer MLP with GELU activation and expansion factor 4.

### 3.3 Graph Pyramid Schedule
1. **Stage 1**: $48 \times 48$, $N=2304$, $C=32$, 2 Local Relation Blocks.
2. **Coarsening 1**: Fixed $3 \times 3$ AA filter $\to$ Stride 2 $\to$ Pointwise Dense $32 \to 64$.
3. **Stage 2**: $24 \times 24$, $N=576$, $C=64$, 2 Local Relation Blocks.
4. **Coarsening 2**: Fixed $3 \times 3$ AA filter $\to$ Stride 2 $\to$ Pointwise Dense $64 \to 96$.
5. **Stage 3**: $12 \times 12$, $N=144$, $C=96$, 2 Local Relation Blocks.
6. **Coarsening 3**: Fixed $3 \times 3$ AA filter $\to$ Stride 2 $\to$ Pointwise Dense $96 \to 128$.
7. **Stage 4 (Coarse Stage)**: $6 \times 6$, $N=36$, $C=128$, 2 Coarse Blocks (governed by experimental condition).

### 3.4 Fixed Anti-Aliased Graph Coarsening
- Filter: Non-trainable separable binomial kernel:
  $$B = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$
- **Boundary Renormalization**: For border and corner nodes, only valid grid neighbors are summed, and weights are renormalized by the sum of available binomial coefficients:
  - Interior sum: 16 (weights divide by 16)
  - Edge sum: 12 (weights divide by 12)
  - Corner sum: 9 (weights divide by 9)
  - Strictly no zero-padding, no reflection, and no virtual nodes.
- Followed by deterministic stride-2 sampling and pointwise channel projection.

---

## 4. Experimental Conditions Specification

| Condition | Description | Coarse Graph Mechanism | Local Gate Input | Coarsening Type |
|---|---|---|---|---|
| **G0** | Local only | 2 node-wise residual FFN coarse blocks (no non-local edges) | $[\text{LN}(h_i), \text{LN}(r_{ij}), \Delta x, \Delta y]$ | Fixed AA (Binomial) |
| **G0.5** | Uniform receiver-relative non-local | Complete graph (1260 edges), $w_{ij} = 1/35$, $q_{ij} = V(h_j - h_i)$ | $[\text{LN}(h_i), \text{LN}(r_{ij}), \Delta x, \Delta y]$ | Fixed AA (Binomial) |
| **G1** | Learned relational graph | Complete graph (1260 edges), learned normalized $a_{ij}$, $q_{ij} = V(h_j - h_i)$ | $[\text{LN}(h_i), \text{LN}(r_{ij}), \Delta x, \Delta y]$ | Fixed AA (Binomial) |
| **G2** | Geometry-only local control | Identical to G1 coarse graph | $[\mathbf{0}, \mathbf{0}, \Delta x, \Delta y]$ (content masked to zero) | Fixed AA (Binomial) |
| **G3** | Coarsening control | Identical to G1 coarse graph | $[\text{LN}(h_i), \text{LN}(r_{ij}), \Delta x, \Delta y]$ | Local $2 \times 2$ mean + Stride 2 |

### 4.1 G0.5 / G1 Functional Matching Contract
Between G0.5 and G1:
- Value function $V(h_j - h_i)$ uses the raw receiver-relative difference and
  is identical in architecture and parameter count. Normalized copies are used
  only by the G1 gate.
- Update function $U$ and pointwise FFN are identical in architecture and parameter count.
- Edge inventory is identical: complete directed graph without self-loops ($36 \times 35 = 1260$ directed edges).
- Aggregation formula:
  - **G0.5**: $m_i = \frac{1}{35} \sum_{j \ne i} q_{ij}$
  - **G1**: $m_i = \sum_{j \ne i} \bar{a}_{ij} \, q_{ij}$, where $\bar{a}_{ij} = \frac{\sigma(s_{ij})}{\epsilon + \sum_{k \ne i} \sigma(s_{ik})}$
- Zero-initialized G1 coarse gate guarantees $\bar{a}_{ij} \approx 1/35$ at initialization.

### 4.2 Gate Parameter Share Constraint
$$\frac{\text{Params}(\text{G1 coarse gate MLP})}{\text{Params}(\text{Total G1 Model})} \le 0.5\% \quad (0.005)$$
Gate hidden dimension is fixed at $4$ and must satisfy this strict inequality.

---

## 5. Readout and Classifier

- Only final $6 \times 6 = 36$ node states enter the readout.
- Permutation-invariant pooling:
  $$z = [\text{mean}_{i}(h_i) \mathbin{\Vert} \text{max}_{i}(h_i)] \in \mathbb{R}^{256}$$
- Pointwise classification head:
  $$\text{Dense}(128, \text{relu}) \to \text{Dropout}(0.1) \to \text{Dense}(7)$$
- Strictly no multi-scale or intermediate bypass to logits.

---

## 6. Data Governance Rules

1. **Test Isolation**: Official `test.csv` must NEVER be read, opened, evaluated, or hashed during development, testing, or technical preflight.
2. **Validation Isolation**: Official `val.csv` must not be accessed during technical preflight.
3. **Research Split**: No split is registered in this snapshot. Any future
   `ResearchTrain`/`ResearchDev` split requires an explicit seed and dev ratio
   in a later scientific registration.
4. **Notebook Default**: `RUN_RESEARCH_SCREEN` in orchestration notebooks must default to `False`.

---

## 7. Opt-in Technical Diagnostics

Diagnostics are disabled in the ordinary forward call. When explicitly
requested, every one of the six local blocks and two coarse blocks exposes
feature variance, feature norm, and effective rank. A separate technical
GradientTape helper records a gradient norm for every graph-block boundary.
Local blocks additionally report gate mean, standard deviation, and fractions
near the low/high bounds. Coarse blocks report normalized-weight entropy,
effective neighbor count, and minimum/maximum weight. These quantities are
technical observations only and are not model-selection criteria.

The coarsening shift probe operates directly on
`FixedAntiAliasedCoarsening`: a two-pixel input translation is aligned by one
coarse output cell and measured on the interior, while a one-pixel shift is
reported only as sensitivity. It does not claim perfect shift invariance.
