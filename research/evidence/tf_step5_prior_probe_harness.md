# TensorFlow Step 5 fixed-topology prior-probe harness

## Provenance and scope

- Issue: #9, `[TF Research Step 5] Build fixed-topology validation-only prior-intervention probe`
- Base branch and commit: `main` at `caa4fb7d13314fae56689abaeca091886102095e`
- Implementation branch: `codex/issue-9-fixed-topology-prior-probe`
- Tool SHA-256: `564eab26b7cf683bd531fec08bf6539a1384d9ef370961b9484335726c7c2351`
- Frozen scientific payload SHA-256 before: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`
- Frozen scientific payload SHA-256 after: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`
- The real Issue #7 checkpoint was not loaded or evaluated. No scientific prior-dependence metric was produced.
- No FER2013 test split, test label, test prediction, test metric, or test artifact was accessed.

## Changed files

- `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/tools/evaluate_fixed_checkpoint_prior_probe.py`
- `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/tests/test_fixed_checkpoint_prior_probe.py`
- `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/README.md`
- `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/package_manifest.json`
- `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/CHECKSUMS.sha256`
- `research/evidence/tf_step5_prior_probe_harness.md`

No file under `src/lap_gnn_tf/`, `contracts/`, or `validation_assets/` changed.

## Registered conditions

All conditions are applied to copies at the post-graph tensor boundary. They use the same original validation batch, single loaded model, node/edge order, topology, graph assignments, coordinates, anchor mask, labels, sample IDs, and image tensors.

- C0 `official`: identity; every tensor is unchanged.
- C1 `direct_part_path_zero_fixed_graph`: zero only `part_soft` and `valid_part_mask`.
- C2 `semantic_prior_zero_fixed_graph`: C1 plus zero node-feature columns `5..31` (`face_mask`, `part_soft_0..12`, `distance_map_0..11`, `landmark_missing_flag`) and edge-feature columns `6..7` (`part_similarity`, `same_dominant_part`). Node columns `0..4` and `32..36`, and edge columns `0..5`, remain unchanged.

C2 is semantic-prior-content removal conditional on the official MediaPipe-derived topology. It is not a prior-free graph.

The CLI exposes no split or arbitrary-intervention argument, constructs only `GraphBatchGenerator(split="val", shuffle=False)`, loads one `.keras` model with `compile=False`, creates no optimizer or training step, and verifies checkpoint-file and in-memory weight hashes before/after inference. Raw-prior corruption and graph reconstruction paths are absent.

## Verification evidence

Run from `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/` in the `lap-gnn-tf` environment unless noted.

- `python -m pytest -q tests/test_fixed_checkpoint_prior_probe.py tests/test_scientific_payload_checksum_portable.py tests/test_feature_schema.py tests/test_edge_schema.py tests/test_graph_batch_layout.py tests/test_no_parent_imports.py tests/test_no_torch_runtime.py tests/test_checkpoint_contains_execution_contract.py tests/test_keras_clean_roundtrip.py tests/test_reference_fixture_integrity.py` — PASS, `33 passed`.
- `python tools/verify_checksums.py` — PASS, `checked=261 failures=0`.
- `python tools/verify_no_parent_imports.py` — PASS, zero violations.
- `python tools/verify_no_torch_runtime.py` — PASS, zero violations.
- `python -m lap_gnn_tf.cli.compare_golden --package-root . --output <temporary-path>` — PASS; prediction agreement `1.0`, max logit difference `3.0994415283203125e-06`, parameter count `1,061,192`.
- `python tools/evaluate_fixed_checkpoint_prior_probe.py --help` — PASS; narrow validation-only CLI displayed.
- `git diff --check` — PASS.

The new tests use synthetic tensors and the existing golden graph fixture. They prove C0/C1/C2 field and column invariance, topology/sample identity, deterministic non-mutation, `37/8/13` schema compatibility, paired condition ordering, validation-only construction, single `compile=False` load, checkpoint/config fail-closed checks, output schema, and absence of training/test/raw-prior paths.

## Step 6 limitations

- This is implementation evidence only. The complete validation set and real Issue #7 checkpoint remain untested by design.
- Step 6 must provide the reviewed Issue #7 `.keras` checkpoint, its matching metadata and resolved config, the official prior root, and a clean graph cache directory.
- The registered Step 6 run must use a fresh output directory and omit `--limit-val-batches`; that option is only for bounded implementation smoke tests.
- Scientific interpretation and decision rules belong to the separately preregistered Step 6 Issue.
