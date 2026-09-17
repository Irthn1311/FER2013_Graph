# CRS Stage Preregistration — Pixel–Motif–Graph Hướng 1

Status: **FROZEN BEFORE IMPLEMENTATION**

Base commit: `49e65e1032f1af13fd615a2e6f7a64f20760684b`

Implementation branch: `research/pixel-relational-composition-crs`

## 1. Scientific question

This stage tests whether a label-free local composition of frozen relational primitives yields a useful higher-level structural representation before any graph is introduced.

Historical verdicts remain immutable:

- E0.1: `NEGATIVE — MOTIF EXISTENCE NOT SUPPORTED` for the ontology `single GMM component = validated motif`.
- E0.2: strong positive evidence that ordered directional local relations carry FER-relevant information.
- E0.R1: sparse relational occurrences supported.
- E0.R2: relative spatial geometry strongly supported.
- M0: `MIXED — NOT SUPPORTED BY REGISTERED JOINT CRITERION` for minimal graph processing on the primitive-node substrate.

The K=128 GMM units are called **relational primitives/components**, never validated motifs.

## 2. Data protocol

- FER2013 Training: 28,709 images. Used for fitting the CRS dictionaries and downstream linear probes.
- FER2013 PublicTest: 3,589 images. Development-stage validation only.
- FER2013 PrivateTest: sealed. This stage must not open, hash, count, inspect, or infer it.

PublicTest has historical project exposure and is therefore not treated as pristine final confirmation. No registered design choice may be changed after viewing the PublicTest result of this stage, except a documented software bug fix that restores compliance with this preregistration.

## 3. Frozen primitive substrate

Use the historical E0.1-v533 artifact exactly:

`e01_dictionary.npz`

SHA256:

`68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b`

Source scientific commit:

`5fda000413c4dcfe810917f07e37bcafb1c8d394`

The artifact contains the frozen PCA/log-sigma transform and the canonical diagonal GMM with K=128.

Primitive assignment is:

`primitive_id = argmax posterior over ALL 128 canonical GMM components`.

Do **not** filter by `canonical_stable_components`. E0.1's empty validated motif set does not remove the 128 components as a frozen primitive vocabulary.

## 4. Coordinate and shape contract

Raw image: `48×48` grayscale.

Historical 5×5 valid descriptor extraction gives:

- primitive map: `44×44`;
- primitive-map indices: `0..43`;
- raw descriptor centers: `(y,x)=2..45`.

The CRS local support is a centered `9×9` window on the primitive map, radius 4.

Valid composition-center indices are:

- primitive-grid `y,x = 4..39`;
- shape `36×36`;
- total valid centers per image: `1296`;
- equivalent raw centers: `y,x = 6..41`.

P, M, and C must use exactly the same `36×36` valid-center domain.

## 5. Local composition descriptor

For every valid center:

1. Extract a centered `9×9` primitive-ID support (81 primitive IDs).
2. Divide it into a row-major `3×3` grid of cells.
3. Each cell is `3×3` primitive positions (9 primitive IDs).
4. Build a 128-bin primitive histogram per cell.
5. Concatenate the nine cell histograms in row-major cell order.

Descriptor shape:

`3×3×128 = 1152D`.

Pre-normalization invariants:

- each cell histogram sums to 9;
- full descriptor sums to 81.

Then apply L2 normalization to the 1152D descriptor.

No alternative support size, cell layout, descriptor family, or multi-scale search is allowed in this stage.

## 6. Dictionary sampling

Dictionary fitting uses Training only.

Sample exactly 24 distinct valid composition centers per Training image, uniformly without replacement from the 1296-center domain.

Use deterministic per-image sampling derived from master seed 42 and image identity so results do not depend on iteration order.

M and C use the exact same sampled center coordinates.

The 24 samples/image are **only** for dictionary fitting. They are not used to build the final per-image representation.

## 7. M — actual compositional CRS

For sampled Training patches, use the ordered 1152D descriptor above.

Fit a label-free spherical k-means dictionary with:

- `K_CRS = 512`;
- `n_init = 3`;
- master seed `42`;
- `max_iter = 50`;
- assignment by maximum cosine similarity;
- centroids L2-normalized after every update;
- choose the initialization with the highest final cosine objective;
- convergence if assignments are unchanged OR relative objective improvement `< 1e-6`;
- deterministic empty-cluster reseeding with the currently worst-represented descriptor (lowest maximum cosine), tie-broken by global descriptor index.

Initialization uses 512 distinct sampled Training descriptors, chosen deterministically from the corresponding initialization seed.

No K-grid or clustering algorithm search is allowed.

## 8. C — keyed cell-arrangement destruction control

C starts from the same nine 128-bin cell histograms as M for each patch.

For each patch, apply a deterministic keyed permutation of the nine **cell blocks** only.

The key includes:

`(master_seed, split_id, image_id, y, x)`.

The implementation must use a version-independent, well-mixed cryptographic-hash-based construction. Python's built-in `hash()` is forbidden.

C preserves exactly:

- all 81 primitive IDs;
- the same nine within-cell histograms;
- descriptor dimension and sparsity;
- patch center;
- final L2 norm.

C destroys only coherent coarse cell identity across patches.

C fits its own spherical k-means dictionary with the exact same K, optimizer, initialization policy, convergence rule, and sampled center coordinates as M.

Do not shuffle the 81 primitive positions and re-bin them.

## 9. Dense final assignment

After fitting the M and C dictionaries, every image is represented densely.

For M and C:

- compute descriptors at all `36×36 = 1296` valid centers;
- hard-assign every center to its nearest spherical centroid by cosine similarity;
- produce a dense `36×36` ID map in `{0,...,511}`.

No threshold, confidence cutoff, NMS, cap, top-K selection, fallback, or sparsification is allowed.

## 10. P — dense primitive baseline

P uses the primitive ID at the exact same `36×36` valid-center coordinates used by M/C.

P therefore also has exactly 1296 positions per image.

No use of the full `44×44` primitive map is allowed for the P image representation.

## 11. Spatial pyramid and normalization

Use a fixed spatial pyramid:

- one whole-image region (`1×1`);
- four quadrants (`2×2`).

On the `36×36` map, quadrant boundaries are exactly at index 18:

- rows `0:18`, `18:36`;
- columns `0:18`, `18:36`.

Counts:

- whole region: 1296 positions;
- each quadrant: 324 positions.

For every region:

1. build the corresponding ID histogram;
2. L1-normalize the regional histogram.

Then concatenate the five regional histograms and apply global L2 normalization.

Dimensions:

- P: `5×128 = 640D`;
- M: `5×512 = 2560D`;
- C: `5×512 = 2560D`.

No TF-IDF, power normalization, learned whitening, pyramid weighting, or normalization search is allowed.

## 12. Fixed downstream classifier

Use the same fixed multinomial logistic-regression configuration for P/M/C:

- `C = 1.0`;
- `solver = "lbfgs"`;
- `class_weight = "balanced"`;
- `max_iter = 5000`.

No per-arm tuning and no hyperparameter search.

Convergence must be checked before opening PublicTest results. If the only problem is non-convergence, `max_iter` may be increased identically for all arms, documented as a technical amendment, and all affected Training fits rerun before Public evaluation. C, solver, weighting, representation, and normalization must not be changed for this reason.

Runtime package versions must be recorded in the experiment manifest.

## 13. Metrics and paired inference

Primary metrics:

- Accuracy;
- Macro-F1 over explicit labels `[0,1,2,3,4,5,6]`, with `zero_division=0`.

Use paired image bootstrap on the 3,589 PublicTest images:

- `B = 2000`;
- bootstrap seed `42`;
- one shared bootstrap-index matrix reused for all paired arm comparisons.

Report per-class F1 as diagnostic only.

### Gate A — Structural Arrangement Support

PASS iff both hold:

- `LB95[ΔAccuracy(M,C)] > 0`;
- `LB95[ΔMacroF1(M,C)] > 0`.

### Gate B — Abstraction Utility

PASS iff both hold:

- `LB95[ΔAccuracy(M,P)] > 0`;
- `LB95[ΔMacroF1(M,P)] > 0`.

Overall decision table:

| Gate A | Gate B | Verdict |
|---|---|---|
| FAIL | FAIL | `COMPOSITIONAL FORMULATION NOT SUPPORTED` |
| FAIL | PASS | `UTILITY WITHOUT ARRANGEMENT EVIDENCE — NOT COMPOSITIONAL SUPPORT` |
| PASS | FAIL | `ARRANGEMENT SUPPORTED — ABSTRACTION UTILITY NOT SUPPORTED` |
| PASS | PASS | `COMPOSITIONAL STAGE SUPPORTED` |

Only `PASS + PASS` allows work to proceed to a separate motif-qualification/sparsification stage. It does not automatically unlock a graph and does not validate CRS clusters as motifs.

## 14. R comparator

A reproduced/current `[O||G]` route may be evaluated as a secondary practical comparator if reconstructed under the current protocol.

Historical E0.R2 numbers remain historical evidence and must not be treated as a matched current comparison unless the protocol is identical.

If reconstructed with the current classifier/protocol, call it `R_current`, not an exact historical R2 reproduction.

R is not part of Gate A or Gate B.

## 15. Diagnostics

Allowed diagnostics include:

- spherical-k-means cosine objective/distortion;
- cluster occupancy and entropy;
- tiny/empty cluster counts;
- per-class F1;
- CRS exemplar patches;
- association with local intensity and contrast.

Diagnostics do not alter K, support, cells, assignment, normalization, classifier, or PASS criteria.

## 16. Forbidden changes in this stage

No:

- graph/GNN/GAT/Transformer/hypergraph;
- CNN/ResNet/ConvNeXt/pretrained backbone;
- learned attention/ROI before motif formation;
- FER-label gradient redefining primitive or CRS discovery;
- soft assignment;
- K-grid;
- support-size search;
- multi-scale descriptor search;
- classifier search;
- threshold/NMS/cap/sparsification;
- PrivateTest access.

## 17. Required pre-Public implementation contracts

Before any registered PublicTest evaluation, tests must establish at least:

1. Frozen dictionary SHA and K=128; assignment over all 128 components.
2. `48×48 → 44×44` primitive map and historical descriptor shape contract.
3. `36×36 = 1296` exact common P/M/C valid-center domain.
4. 1152D composition shape and `9-per-cell / 81-per-patch` count invariants.
5. C preserves the nine M cell histograms exactly up to permutation and is deterministic across processes.
6. 24/image sampling is Training dictionary-fit only; final P/M/C representations all use 1296 positions.
7. Spherical centroids are unit norm and assignments are cosine argmax.
8. Spatial-pyramid count/dimension/normalization invariants.
9. Identical logistic configuration and convergence checks across arms.
10. Paired bootstrap uses one shared resample matrix and the runner cannot access PrivateTest.

A result produced by code that violates this preregistration is `INVALID — IMPLEMENTATION DEVIATION`, not a scientific negative or positive result.

## 18. Technical Amendment A1 — Train-only optimizer cap

Date: `2026-09-16`

The first official Train-only execution used scientific source lock
`194876ccb5a723255f6e6ffaf3105f8cc3f1fc9e`. It completed the exact
28,709-image, 24-centers-per-image sampled pool with sample-identity SHA256
`96a6468f2a7d27d81d6ed87249cc381a62b12b59eef7f70ff232b626b28b0662`,
then the M spherical k-means fit reached the registered `max_iter=50` with
objective `371651.925665`, selected initialization `0`, zero empty-cluster
reseeds, and `converged=false`. The runner stopped fail-closed before fitting
C, building dense features, or fitting the probes. PublicTest and PrivateTest
were not accessed. This was a technical Train-stage non-convergence, not a CRS
scientific result.

A1 authorizes exactly one optimizer-cap change:

- spherical k-means `max_iter: 50 -> 500`;
- the same cap applies identically to M and C.

This is a Train-only technical optimization amendment. All other spherical
k-means settings and semantics remain frozen: `K=512`, `n_init=3`, seed 42,
`tol=1e-6`, cosine assignment, unit-normalized centroids, deterministic
initialization and empty-cluster reseeding, highest-final-objective
initialization selection, exact CSR cosine computation, and convergence only
when assignments are unchanged or relative objective improvement is below
`1e-6` on a no-reseed iteration.

All representation, sampling, control, classifier, data-role, bootstrap, and
Gate A/B contracts above remain frozen. The rerun must start from scratch and
must reproduce the v1 sample-identity SHA256 before clustering. If either M or
C reaches 500 iterations without convergence, execution stops fail-closed and
PublicTest remains locked.
