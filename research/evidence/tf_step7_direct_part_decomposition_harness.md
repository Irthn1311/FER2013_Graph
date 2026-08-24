# TensorFlow Step 7 direct-part decomposition harness evidence

## Scope and immutable provenance

- Issue: #13, `[TF Research Step 7] Build fixed-checkpoint direct-part pathway decomposition harness`.
- Exact base: `main` at `8675c839004c18322da28c95770ee6e126e0e22f`.
- New external tool: `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/tools/evaluate_fixed_checkpoint_direct_part_decomposition_probe.py`.
- Tool SHA-256: `e611b74ac143c50149326c9761b35177183a09b3cf44b52ab018b01ed3d87ffd`.
- Test SHA-256: `0aabdab8676bdace43706e9f96afc96f533678b6f631f38b1bdd321237d87b5b`.
- Package manifest SHA-256: `15891b2ec729a1592c5bec96184d5997966b911e79cc377677c70ab610b996ee`.
- Package checksums SHA-256: `61ac4ed2f707e9b92313a24100fbace50bb30274b5b2399a5bcb9846408e7ce7`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e` — unchanged.
- Frozen execution-contract SHA-256: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22` — unchanged.

No file under `src/lap_gnn_tf`, `contracts`, or `validation_assets` changed.

## Fixed registered conditions

| Condition | Exact manual-forward intervention |
| --- | --- |
| D0 `official_manual_forward` | Identity manual forward; every context, pool, and readout pathway receives official values. Native `model(batch, training=False)` is evaluated on the same batch for Gate A. |
| D1 `context_local_prior_neutralized` | Only `PartGlobalContext.part_soft` is zero. The context block and built-in global token/path remain; pooling, pooled embeddings/validity, and readout prior stay official. |
| D2 `readout_local_prior_neutralized` | Only `MicroMotifSupportReadout.part_soft` is zero, making the clipped local log-prior bias spatially constant. Attention, global-prior queries, pooled embeddings, and validity flags remain. |
| D3 `local_part_residual_zero` | Only pooled `mouth`, `eye`, `brow`, and `nose_cheek` embeddings passed to readout are zero. Pooled `global`, every validity flag, and readout prior remain official. |
| D4 `local_motif_validity_off` | Only the four local readout validity flags are false; `global` is true. All pooled embeddings and readout priors remain official. |
| D5 `full_direct_part_zero_anchor` | Exact Step 6 C1 composition: zero context `part_soft`, pool `part_soft`, pool `valid_part_mask`, and readout `part_soft`; node/edge features and topology remain unchanged. |

The tool has no public split, condition, or intervention selector. Condition order is fixed to D0, D1, D2, D3, D4, D5 and all six conditions are evaluated from each original batch before advancing.

## Mandatory gates and interpretation boundary

- Gate A, `INVALID_MANUAL_FORWARD_EQUIVALENCE`: every native/manual D0 batch requires prediction agreement exactly `1.0`, maximum absolute logit difference `<=1e-5`, maximum absolute probability difference `<=1e-6`, and the same sample order. Failure evidence is written without condition metrics or scientific interpretation.
- Gate B, `INVALID_D0_REFERENCE_REPRODUCTION`: an unbounded run requires exactly `3589` samples and D0 accuracy `0.63137364168292`, macro-F1 `0.5932591901893336`, loss `1.1537981724317095` within locked tolerances `0.001`, `0.001`, and `0.005`.
- Gate C, `INVALID_C1_ANCHOR_REPRODUCTION`: D5 must reproduce Step 6 C1 accuracy `0.27751462803009197`, macro-F1 `0.19745892656222366`, loss `1.757720434560185` within the same locked tolerances.
- A bounded `--limit-val-batches` run is always labeled `BOUNDED_SMOKE_NO_SCIENTIFIC_INTERPRETATION`; Gates B/C, per-path labels, and the overall decision remain unset.
- Only after all unbounded gates pass are the preregistered D1-D4 delta thresholds and overall `SINGLE_HIGH_DIRECT_PATH`, `MULTIPLE_HIGH_DIRECT_PATHS`, or `INTERACTION_DOMINATED_DIRECT_DEPENDENCY` rule applied.
- Negative deltas remain exact, are labeled LOW, and receive the required improvement note.
- D1-D4 effects are explicitly non-additive nonlinear sensitivity diagnostics and must not be summed or treated as causal contributions.

## Runtime and integrity controls

- One `.keras` model load with `compile=False`; exact parameter count required; restored optimizer state is rejected.
- `GraphBatchGenerator(split="val", shuffle=False)` only; no train/test split or public selector.
- One model instance and immutable checkpoint/model-weight hashes before and after inference.
- Source graph batch snapshots are checked after every D0-D5 condition.
- Node features, edge features, edge index, graph indices, coordinates, topology, labels, and sample IDs remain unchanged.
- Each condition records the exact changed intermediate arguments; all unregistered pathway arguments are equality-checked.
- D0 native/manual identity and D5/Step-6-C1 identity are directly regression-tested on the frozen golden fixture.
- No optimizer, gradient, fit, training, graph rebuild, raw-prior corruption, or test lifecycle exists in the tool.
- Fresh output root is mandatory.

## Deterministic future outputs

An approved future run will create only:

- six `validation_metrics_<condition>.json` files;
- one `paired_validation_predictions.csv` covering D0-D5;
- `native_manual_d0_equivalence.json`;
- `intervention_integrity.json`;
- `probe_manifest.json` with gates, paired diagnostics, hashes, resources, environment, and explicit training/test isolation.

The required runtime inputs are an exact `.keras` checkpoint, its metadata, the resolved JSON config, the official validation prior root, and the official clean validation graph cache. This Issue does not create a Kaggle execution adapter or run those inputs.

## Verification performed

- Focused Step 7 suite — PASS, `14 passed`.
- Combined Step 6/Step 7 probe, payload, isolation, checkpoint, Keras round-trip, and reference-fixture suite — PASS, `46 passed`.
- Golden D0 native/manual comparison — PASS with exact logits and probabilities (`0.0` maximum differences; prediction agreement `1.0`).
- Golden D5 versus native Step 6 C1 comparison — PASS with exact logits and probabilities (`0.0` maximum differences).
- `python tools/verify_checksums.py` — PASS, `checked=263 failures=0`.
- `python tools/verify_no_parent_imports.py` — PASS, zero violations.
- `python tools/verify_no_torch_runtime.py` — PASS, zero violations.
- `git diff --check` — PASS.

## Pre-run limitation and review gate

Only controlled/golden fixtures and mocked bounded control paths were exercised. The real Issue #7 checkpoint was not loaded, full validation was not run, and no D1-D4 scientific metric or outcome was produced. This implementation establishes harness correctness only and does not answer the Step 7 research question.

Stop at draft-PR research-lead review. A real checkpoint/full-validation run requires separate approval and must not use `--limit-val-batches`.
