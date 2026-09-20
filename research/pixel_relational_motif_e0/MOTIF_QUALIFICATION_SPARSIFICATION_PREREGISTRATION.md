# Motif Qualification & Sparsification — Preregistration

Status: **FROZEN BEFORE IMPLEMENTATION**

Issue: #86

Base scientific state: accepted CRS Public result from Issue #84 / PR #85 with registered verdict `COMPOSITIONAL STAGE SUPPORTED`.

This preregistration defines the next label-free qualification/sparsification experiment. It does **not** reopen the CRS result, validate semantic motifs, or authorize graph construction.

## 1. Scientific question

Can the accepted dense ordered CRS representation be converted into a sparse, reproducible set of candidate occurrences that:

1. retains structure-specific information relative to a matched destroyed-arrangement control;
2. retains utility beyond the frozen primitive baseline; and
3. yields a non-empty bounded node set for every image, sufficient to justify a later separate graph experiment?

## 2. Ontology

- Historical K=128 GMM components: **relational primitives**, never motifs.
- Frozen ordered K=512 M clusters: **CRS types**, not motifs.
- CRS types passing Train-only recurrence and stability criteria: **qualified CRS types**.
- Deterministically selected sparse representatives: **candidate motif occurrences**.
- Only if the registered sparse gates and graph-readiness coverage rule pass may the occurrences be called **motif-candidate graph nodes**.
- No semantic facial-motif claim is made in this stage.

## 3. Data governance

### Train

Official FER2013 Train: 28,709 images.

Train is the only split used for:

- recurrence qualification;
- resampling stability fitting/evaluation;
- vocabulary selection;
- sparse feature construction for probe fitting;
- fitting the new sparse logistic probes.

FER labels must not affect recurrence, stability, type selection, component extraction, occurrence ranking, node capping, or nuisance diagnostics.

### PublicTest

Official PublicTest: 3,589 images.

Public has already been researcher-exposed in earlier registered stages. This experiment is therefore a **sequential registered follow-up under prior PublicTest exposure**, not a pristine first-look confirmatory experiment.

No new Public predictions may be produced until this preregistration and implementation are source-locked and reviewed.

### PrivateTest

PrivateTest remains sealed through this stage and all later graph development.

Only after the entire final H1 representation, architecture, and classifier are frozen may one non-iterative bundled Private evaluation occur. If the motif-candidate Gate A/B pattern does not replicate on Private, report non-replication; do not redesign after Private.

## 4. Frozen upstream state

Reuse the accepted CRS Train model and exact historical primitive substrate from Issue #84.

No refit of the accepted full-Train M/C dictionaries is allowed for the primary sparse representation.

The accepted Public P and M_dense predictions are frozen historical comparators and must be reused exactly in the Public stage.

## 5. Recurrence qualification

For each frozen CRS type independently in M and C:

- compute its dense 36x36 assignment field on every Train image;
- count the number of **distinct Train images** containing at least one occurrence of that type;
- do not use the 24 dictionary-sampling descriptors for recurrence.

A type is recurrence-eligible iff:

`distinct_image_support >= 288`

where 288 = `ceil(0.01 * 28709)`.

## 6. Resampling stability

Run the same stability protocol independently for ordered M and destroyed-arrangement C.

### 6.1 Replicates

- replicates: 20
- split unit: Train image
- fit fraction: 80%
- held-out fraction: 20%
- deterministic seeds under namespace `(42, "motif_stability", arm, replicate_id)`

### 6.2 Replicate fitting

For each replicate and arm:

1. select the deterministic 80% Train fit-image subset;
2. use the exact already-registered 24 dictionary descriptors/image for that arm on fit images only;
3. fit spherical K-means K=512 **from scratch**;
4. no warm start from accepted full-Train centers;
5. no reuse of accepted full-Train memberships.

Replicate optimizer semantics are frozen to:

- K=512
- n_init=3
- max_iter=500
- tol=1e-6
- exact cosine assignment
- unit-normalized centroids
- existing deterministic initialization semantics
- existing deterministic empty-cluster semantics
- select the initialization with the highest final objective.

Every replicate must converge. A non-converged replicate is a technical blocker; do not inspect or alter Public because of it.

### 6.3 Replicate-to-original type matching

Match each replicate's 512 centroids to the accepted full-Train arm's 512 centroids using maximum-weight one-to-one Hungarian assignment on the 512x512 **centroid cosine similarity matrix**.

Held-out descriptors, held-out assignments, FER labels, and Public information must not affect this matching.

### 6.4 Held-out Jaccard evaluation

This point is explicit and mandatory:

**Held-out stability evaluation uses the dense 36x36 assignment field over all 1296 valid centers/image in the excluded 20% Train images. It does NOT use only the 24 sampled dictionary-fitting descriptors.**

For each held-out dense descriptor:

- assign once using the accepted full-Train arm centers;
- assign once using the replicate centers;
- map replicate type IDs to original type IDs using the already-fixed centroid Hungarian matching.

For each original type k:

- `A_k` = held-out dense descriptors assigned to k by the accepted original centers;
- `B_k` = the same held-out dense descriptors assigned to the replicate cluster matched to k.

Compute:

`J_k = |A_k intersect B_k| / |A_k union B_k|`

If the union is empty, set `J_k = 0` conservatively.

For every type report across the 20 replicates:

- median Jaccard;
- Q1;
- Q3;
- min;
- max.

A type is stability-eligible iff:

`median_jaccard >= 0.75`.

## 7. Final qualification rule

Apply the identical rule independently to M and C:

`Q_M = {M types: support >= 288 AND median Jaccard >= 0.75}`

`Q_C = {C types: support >= 288 AND median Jaccard >= 0.75}`

No unqualified type may be promoted for count matching.

## 8. Count-matched structural vocabulary

For Gate A define:

`Q_star = min(|Q_M|, |Q_C|)`.

Within each already-qualified set, rank types by the exact same deterministic order:

1. median stability descending;
2. distinct-image support descending;
3. type ID ascending.

Take the top `Q_star` from M and C to form:

- `S_M_match` vocabulary;
- `S_C_match` vocabulary.

If `Q_star == 0`, Gate A cannot establish structural support and graph remains locked.

For Gate B and future graph-candidate ontology use **all** qualified M types:

`S_M_all = Q_M`.

Mandatory diagnostic: report `Q_star / |Q_M|` (equivalently `|S_M_match| / |S_M_all|`) so the distinction between the two gate vocabularies remains explicit.

## 9. Candidate occurrence extraction

For each image and selected arm vocabulary:

1. start from the dense 36x36 frozen arm assignment map;
2. mask to the selected qualified types;
3. for each selected type independently, find 8-connected components;
4. convert each component to one occurrence;
5. choose the representative position with the highest assignment margin:
   `margin = cosine(best center) - cosine(second-best center)`;
6. deterministic tie-breaking: higher margin, then lower type ID, then lower row-major coordinate;
7. pool occurrences across types;
8. if more than 64 candidates remain, retain the global top 64 by the same deterministic order; otherwise retain all.

No threshold grid, alternate connectivity, NMS-radius search, soft assignment, label-guided selection, or post-result rescue is allowed in this experiment.

## 10. Sparse features

Each retained occurrence contributes exactly one count to its selected CRS type.

Use the same spatial pyramid as the accepted dense CRS representation:

- 1x1 + 2x2 regions;
- 512 bins/region;
- regional L1 normalization;
- concatenation;
- global L2 normalization.

Final sparse dimensionality: 2560D.

Arms:

- `P`: exact frozen primitive baseline prediction from accepted CRS Public result;
- `M_dense`: exact frozen ordered dense CRS prediction from accepted CRS Public result;
- `S_M_all`: all qualified M types;
- `S_M_match`: count-matched qualified M subset;
- `S_C_match`: count-matched qualified C subset.

## 11. Sparse Train probes

Only after Train-only qualification and sparse features are frozen may new probes be fit.

For `S_M_all`, `S_M_match`, and `S_C_match`, use identical multinomial logistic probes:

- C=1.0
- solver=lbfgs
- class_weight=balanced
- max_iter=5000
- tol=1e-4

No tuning, classifier search, threshold search, or feature selection.

All probes must converge before any new Public access.

## 12. Public bootstrap

Reuse the exact accepted Public bootstrap index matrix from the CRS stage:

- B=2000
- N=3589
- PCG64
- seed=42

Do not create a new resampling universe for this stage.

## 13. Registered gates

### Gate A — matched structural support

Compare:

`S_M_match - S_C_match`

PASS iff both:

- lower 95% paired-bootstrap bound for delta Accuracy > 0;
- lower 95% paired-bootstrap bound for delta Macro-F1 > 0.

This is the primary matched structural comparison.

### Gate B — useful sparse node ontology

Compare:

`S_M_all - P`

PASS iff both:

- lower 95% paired-bootstrap bound for delta Accuracy > 0;
- lower 95% paired-bootstrap bound for delta Macro-F1 > 0.

### Dense retention

Compare:

`S_M_all - M_dense`

Report point deltas and paired-bootstrap intervals, but this is **report-only**. No 2%, 5%, 80%, or other non-inferiority threshold is registered.

Any `S_M_all - S_C_all` comparison, if reported, is diagnostic only and is not a gate.

## 14. Graph unlock rule

A later graph experiment may be designed only if all are true:

1. Gate A PASS;
2. Gate B PASS;
3. zero Train images with zero `S_M_all` candidate occurrences;
4. zero Public images with zero `S_M_all` candidate occurrences.

The occurrence construction guarantees at most 64 candidate nodes/image.

Passing this rule means only:

`candidate motif occurrence ontology supported for a later incremental graph test`.

It does not establish graph utility or semantic motif validity.

## 15. Mandatory diagnostics reported with the scientific verdict

The same report containing Gate A/B must also include:

- |Q_M|, |Q_C|, Q_star;
- `|S_M_match| / |S_M_all|`;
- recurrence support distributions;
- per-type Jaccard median/Q1/Q3/min/max;
- component-size distributions;
- fraction of components of size 1;
- median/P90/P95/max component size;
- pre-cap nodes/image distribution;
- post-cap nodes/image distribution;
- fraction of images where top-64 binds;
- Train and Public zero-node counts;
- per-qualified-type local intensity association;
- per-qualified-type local contrast association;
- spatial/border concentration;
- representative Train patches/spatial maps;
- per-class F1;
- dense-vs-sparse descriptive retention.

Diagnostics are interpretation aids only and cannot trigger same-stage reselection or rescue.

## 16. Failure policy

If either sparse gate fails, or graph-readiness coverage fails, report the exact result.

Do not automatically try:

- alternate support thresholds;
- alternate Jaccard thresholds;
- more/fewer nodes;
- alternate connectivity/NMS;
- soft assignment;
- label-guided type ranking;
- alternate classifier;
- graph rescue.

A scientifically different follow-up requires a new preregistration and must explicitly acknowledge prior Public exposure.

## 17. Distributed execution policy

The 40 stability fits (20 M + 20 C) are deterministic and replicate-independent. After implementation is source-locked, they may be sharded across multiple compute sessions/accounts for engineering efficiency **only if**:

- every shard runs the exact same scientific source SHA;
- every replicate is keyed by `(arm, replicate_id)`;
- no replicate is repeated and selected by quality;
- merging is a deterministic union by replicate ID;
- no Public data are involved in stability shards;
- environment/source/artifact hashes are recorded per shard;
- the merged result is independently validated for complete replicate coverage 0..19 for both arms.

Multiple sessions must never be used to choose a best scientific run.

## 18. Interpretation guards

A sparse-stage PASS would establish that a deterministic subset of recurrent/stable ordered CRS occurrences remains useful under the registered controls. It would not establish:

- semantic facial motifs;
- causal expression parts;
- that every qualified type is meaningful;
- that node-to-node relations add information;
- that a GNN will improve performance.

A later graph experiment must separately test whether relational interaction among these already-supported sparse occurrences adds information beyond a matched motif-only baseline.

## 19. Review provenance

Adversarial review of v1 identified two blockers:

1. stability based on recomputing centroids from frozen original memberships was artificially optimistic;
2. C top-Q count forcing used a different selection mechanism from M.

v2 fixed them by:

- from-scratch K=512 reclustering on deterministic 80% Train subsets;
- centroid-only Hungarian matching;
- held-out 20% dense 36x36 Jaccard evaluation;
- identical independent M/C qualification;
- symmetric count matching only *within* already-qualified sets.

Final blocker-closure review found one remaining ambiguity only: whether held-out Jaccard used dense assignments or the 24-sample pool. This preregistration explicitly fixes it to the dense 36x36 / 1296-position held-out field.

No further reviewer-driven design changes are authorized before implementation unless an implementation inconsistency with this frozen specification is discovered.