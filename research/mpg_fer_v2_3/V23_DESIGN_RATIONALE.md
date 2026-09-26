# MPG-FER v2.3 Design Rationale

## Registered question

Can MPG-FER preserve the useful discriminative gain of Motif Layers 1-2 while reducing the rapid depth-dependent specialization/generalization gap?

Issue #97 preregisters the experiment on exact v2.2 base commit `0a258fd43cc4d8f45afa54ea1328f068c52cbee0`. The frozen v2.2 scientific source SHA-256 is `a8dc77db29e997c4c3ab69bb862704c8a948f940a4636e1c01e0d96bab40de65`.

## A6-R2 evidence and target localization

A6-R2 found that contextual Pixel-GNN information is largely transmitted through Composer WHAT features, Layer-5 motif node states dominate readout transfer, and sample-level routing divergence is not an actionable isolated rescue lever. The cleanest generalization signal is depth-dependent specialization: the Train-Public linear-probe gap rises from roughly 5 pp at PRE to 14-16 pp after L1 and 22-24 pp after L2. This does not prove memorization or implicate topology. It localizes the intervention target to the transformations in Motif Layers 1-2.

## Candidate single-delta interventions

### 1. Fixed early residual-branch scaling — selected

- Mechanism: multiply both existing residual updates in Motif Layers 1-2 by `0.5`; use scale `1.0` in Layers 3-5.
- Modules touched: v2.3 configuration and `GeometryAwareMotifTransformerBlock` construction/forward only.
- Evidence link: directly preserves more of each incoming early-depth representation while retaining attention and FFN relational updates.
- Main risk: held-out quality may not improve, or the model may compensate by increasing branch weights.
- Parameters: unchanged.
- Inference: changed only by the two fixed early residual scales.
- Selection reason: narrow, deterministic, zero-parameter, easy to falsify, and leaves routing, losses, data, readout, and classifier untouched.

The preregistered scale `0.5` is the single midpoint step for the four residual transformations across L1-L2. It is not selected by a PublicTest or PrivateTest sweep.

### 2. Flip-aligned L1-L2 representation consistency — rejected

- Mechanism: penalize disagreement between spatially aligned original/flip motif states at L1-L2.
- Modules touched: model outputs, training loop, loss accounting, and resume/provenance.
- Evidence link: directly regularizes early representations.
- Main risk: adds a new objective, coefficient, alignment assumptions, and extra activation retention on the existing second consistency forward.
- Parameters: unchanged.
- Inference: unchanged.
- Rejection reason: larger loss/compute confound than necessary for the first early-depth experiment.

### 3. Shared transformation parameters across L1-L2 — rejected

- Mechanism: recurrent early motif transformation with shared Q/K/V/geometry/FFN parameters.
- Modules touched: block construction, state dictionaries, checkpoint compatibility, and optimizer parameter groups.
- Evidence link: constrains rapid early specialization.
- Main risk: L1 and L2 have different locked K values (`8` and `16`), and sharing creates a large parameter-count/capacity confound.
- Parameters: materially reduced.
- Inference: changed.
- Rejection reason: less direct and less comparable to v2.2.

### 4. Increased DropPath only in L1-L2 — rejected

- Mechanism: apply stronger stochastic residual deletion in the two early motif blocks.
- Modules touched: DropPath schedule/configuration.
- Evidence link: localized regularization at the observed gap expansion.
- Main risk: may merely weaken early layers; it is a generic stochastic regularizer whose result is less mechanistically specific.
- Parameters: unchanged.
- Inference: unchanged.
- Rejection reason: weaker representation-preservation interpretation than fixed residual scaling.

## One precise hypothesis

Compared with v2.2, fixed half-step residual updates in Motif Layers 1-2 will preserve more incoming contextual Pixel-GNN information while retaining relational discriminative updates, reducing early train/held-out specialization and improving raw held-out FER performance.

## Falsification criteria

The hypothesis is not supported if any central condition holds:

- L1/L2 Train-Public and Train-Private probe behavior does not improve;
- raw Public accuracy does not improve;
- frozen-checkpoint Private raw performance materially reverses any Public gain;
- any apparent gain exists only under TTA;
- hard-class behavior collapses;
- the gap shrinks only because both Train and held-out separability collapse.

Implementation tests establish correctness only and cannot support this hypothesis.

## Frozen non-target behavior

The following remain v2.2-identical: pixel feature extractor, Pixel GNN, SpatialMotifComposer, prototype/TYPE/WHAT/WHERE definitions, multiscale anchors, 49 motif nodes, geometry edges, five motif layers, K schedule `[8,16,16,16,24]`, motif readout, fusion, classifier, loss stack, optimizer and scheduler, augmentations, EMA, checkpoint selection, early stopping, FER split roles, and TTA. v2.2 source is not edited. No CNN, ViT, pretrained backbone, external detector, extra data, relabeling, or ensemble is introduced.
