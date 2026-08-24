# TensorFlow Step 8 Gate-A model-boundary autocast hotfix

## Failed-attempt preservation

- Experiment: Issue #15 registered Step 8 fixed-checkpoint direct-part pathway decomposition.
- Attempt: first registered Kaggle T4 execution of the reviewed PR #16 adapter at commit `0d283ee64d1f71c5b60b198aa80a964023bda6e0`.
- Original Step 7 harness SHA-256: `e611b74ac143c50149326c9761b35177183a09b3cf44b52ab018b01ed3d87ffd`.
- Failure label: `INVALID_MANUAL_FORWARD_EQUIVALENCE`.
- Classification: **PRE-INTERVENTION TECHNICAL HARNESS FAILURE / INVALID_MANUAL_FORWARD_EQUIVALENCE**.
- Scientific boundary: Gate A failed before any valid D1-D4 intervention outcome. No D1-D4 metric, sensitivity label, overall decision, or scientific interpretation is retained from this attempt.
- The supplied runtime evidence contains the failure label but not the failed batch's numeric differences; those Kaggle-specific numeric values remain `UNKNOWN`. The failed run is not reclassified as a scientific result.

No Kaggle rerun was performed for this hotfix.

## Root cause reproduced

Keras `Model.__call__` recursively autocasts floating tensors in the structured model input to the outer model's `input_dtype` before entering `LapGNN.call`. The persisted-checkpoint boundary is represented by:

- process global policy `float32`;
- outer `LapGNN` policy `mixed_float16` and compute/input dtype `float16`;
- internal encoder/GNN/readout/classifier policies `float32` after subclass-model reconstruction;
- original graph batch floating tensors `float32`.

The original Step 7 `manual_forward` entered the internal call graph directly with the original float32 batch. It therefore skipped the outer Keras boundary conversion that native `model(batch, training=False)` applies.

Regression coverage recreates this boundary with the frozen golden graph/model state. When the new boundary-normalization helper is replaced by the old identity behavior, Gate A fails with:

- prediction agreement `1.0`;
- maximum absolute logit difference `0.002503812313079834` (> `1e-5`);
- maximum absolute probability difference `0.00017446279525756836` (> `1e-6`).

This is a local controlled regression reproduction, not a substituted Kaggle result.

## Minimal technical fix

Only the external Step 7 diagnostic harness is changed. Before decomposing the internal call graph, `manual_forward` now deterministically mirrors Keras model-boundary semantics:

1. read the loaded model's exact `input_dtype` and `autocast` setting;
2. cast only floating source tensors whose dtype differs from the boundary dtype when autocast is enabled;
3. leave integer tensors unchanged;
4. perform the already-registered internal float32 casts and D0-D5 pathway operations on that effective boundary batch;
5. retain the original batch unchanged and record source/effective dtypes in integrity evidence.

The frozen `LapGNN` model, checkpoint, data loader, graph construction, topology, features, intervention definitions, Gate A tolerances, Gate B/C references, thresholds, labels, and decision rules are unchanged.

## Regression evidence after the fix

Under the reproduced outer-mixed/internal-float32 boundary:

- native D0 versus manual D0 prediction agreement: `1.0`;
- maximum absolute D0 logit difference: `0.0`;
- maximum absolute D0 probability difference: `0.0`;
- Gate A under the existing tolerances: `PASS`;
- manual D5 versus native Step 6 C1 maximum logit difference: `0.0`;
- manual D5 versus native Step 6 C1 maximum probability difference: `0.0`;
- source batch: byte/value unchanged;
- D0-D5 condition order and every registered `changed_pathway_arguments` list: unchanged and revalidated;
- checkpoint/model state: not mutated by the regression.

The exact equality is local regression evidence. The real checkpoint/full-validation Gate A remains unverified until a separately approved Kaggle rerun.

## Immutable provenance and hashes

- Scientific base: `d14e7e1e3eec2ffbd5339b5a3bd0d5db5ab3de8b`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e` — unchanged.
- Frozen execution-contract SHA-256: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22` — unchanged.
- Step 6 support-tool SHA-256: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3` — unchanged.
- Hotfixed Step 7 harness SHA-256: `fc60ece71caea14927c4840edfcd527d005737106f60d0bb475b9b1ba79eadd3`.
- Updated Step 7 test SHA-256: `1e5d2e068e3df5d32234e207ad28ee911b93fa6b5948599595b792fd18b3b4b5`.
- Updated package manifest SHA-256: `c26dac14eb171c20ae43f8cc8594c2569bab9b947a37f7bedaa3a4ffb18c58a1`.
- Updated package checksums SHA-256: `2bc1508fd7f22121dd087abdeb6efc3a66ff383741cec11baf20d64663dad5fe`.

No file under package `src/lap_gnn_tf`, `contracts`, or `validation_assets` changed.

## Verification performed

- Combined Step 6/Step 7 fixed-checkpoint, mixed-boundary, payload, reference-fixture, and Keras clean-roundtrip suite: PASS, `45 passed`.
- Mixed-boundary regression alone: PASS, `2 passed`; the test explicitly proves the old identity boundary fails and the normalized boundary gives exact D0/D5 equality.
- Package checksum verification: PASS, `checked=263 failures=0`.
- Parent-import isolation verification: PASS, zero violations.
- PyTorch-runtime isolation verification: PASS, zero violations.
- `git diff --check`: PASS.

## Review and execution gate

This hotfix must be independently reviewed and merged before PR #16 is updated to lock a later execution commit and the new Step 7 tool SHA. PR #16 must not be used unchanged, and no real experiment rerun is authorized by this implementation evidence. Stop for research-lead review.
