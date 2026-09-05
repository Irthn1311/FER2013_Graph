# TF Step 13 remaining-prior probe harness

Status: `PROBE_HARNESS_IMPLEMENTATION_ONLY`

## Scope and provenance

- Issue: `#38`
- Exact implementation base: `e9b4deec2d4986b4a94fce32f3c1586cdb301047`
- Probe: `research/candidates/tf_learned_local_residual_slots/evaluate_remaining_prior_probe.py`
- Probe SHA-256: `407cfed62c5a1dc7e2c381282082c7b7335f5ab400a37e0feaf38be0ce740809`
- Candidate model SHA-256: `0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca`
- Reviewed Step-12E archive SHA-256: `f436b0a7a20c751b2fd2f47738469fb409ecf9a1a40628e05d20974639927451`
- Epoch-42 checkpoint SHA-256: `e0d633cb6200e963f31a28750e28c7febdaae40344c90ba9d94b826a09e4b78c`
- Epoch-42 weights SHA-256: `a18a372f70ce56868ae43257e9b7fa5e20517499c2c1e35c48dba4d65eaaaa74`
- Epoch-42 metadata SHA-256: `a5ee759bc6fbef587e025199d0dcfe6ebd3a1764cffa567f793c53e972eb47cf`
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`
- Frozen execution contract SHA-256: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`

The production entrypoint is validation-only and inference-only. It verifies the reviewed archive, checkpoint, weights, metadata, candidate source, frozen contract, candidate class, `1,061,576` parameters, `128` trainable variables, and Q at trainable-variable index `127` with shape `[4, 96]`, dtype `float32`, and flat-float32 SHA-256 `54b368aa183c65d5843d8b8e340d3020412d1a2dfeaabbe8b2c0166684ab3ff9`. The same model/Q/weight and source-artifact identities are checked again after evaluation.

## Registered conditions

The condition inventory and order are fixed; there is no public condition selector.

| ID | Condition | Only registered intervention |
| --- | --- | --- |
| P0 | `official_candidate_manual_forward` | Identity manual forward |
| P1 | `node_face_mask_zero_fixed_graph` | Node column `5` |
| P2 | `node_part_soft_channels_zero_fixed_graph` | Node columns `6:19`; downstream direct `part_soft` remains official |
| P3 | `node_distance_map_channels_zero_fixed_graph` | Node columns `19:31` |
| P4 | `node_landmark_missing_flag_zero_fixed_graph` | Node column `31` |
| P5 | `edge_semantic_channels_zero_fixed_graph` | Edge columns `6:8` |
| P6 | `context_direct_part_soft_neutralized` | Zero `part_soft` only at `gnn.encode`; downstream pooling/readout use the official prior and learned slots are recomputed from post-GNN state |
| P7 | `readout_direct_part_soft_neutralized` | Zero `part_soft` only at readout |
| P8 | `readout_validity_off` | Disable only four local validity gates at readout; global remains valid |
| P9 | `all_explicit_semantic_prior_zero_fixed_topology_anchor` | Joint registered node, edge, context, readout-prior, and readout-validity interventions |

Every condition consumes the same already-constructed graph batch. Source tensors are snapshotted and checked after each condition. Node/edge indices, graph memberships and counts, coordinates, labels, sample IDs, retained visual/geometric channels, weights, and Q remain unchanged. P9 retains the official topology and is not a MediaPipe-free graph.

## Gates and reporting contract

- Gate A: native/P0 prediction agreement exactly `1.0`, maximum absolute logit difference `<= 1e-5`, maximum absolute probability difference `<= 3e-6`.
- Gate B: exactly `3,589` samples and P0 reference accuracy `0.6232933964892727`, macro-F1 `0.596090717851928`, loss `1.1486882999934982`, with tolerances `0.001`, `0.001`, and `0.005` respectively.
- Gate C: all checkpoint/source/contract/model/Q identities pass before evaluation and remain unchanged afterward.
- P1-P8 thresholds: `>=10` pp high, `>=5` and `<10` pp moderate, `<5` pp low; negative deltas remain low.
- Overall decision: exact registered zero/one/two-or-more HIGH branches, using P9 only for the zero-HIGH joint-dependency branch.

The machine-readable output supports accuracy, macro-F1, loss, per-class F1, prediction disagreement count/rate, four correctness transitions, and P0-minus-condition per-class F1 deltas. It explicitly forbids additive or percentage-contribution interpretation.

## Verification performed

- Focused Issue #38 suite: `38 passed`.
- Candidate model plus reviewed Step-7 and Step-6 regressions: `54 passed`.
- Frozen package checksum verification: `PASS checked=267 failures=0`.
- Frozen package diff from the exact base: empty.
- `git diff --check`: passed.

Only synthetic/golden bounded fixtures were evaluated. Kaggle was not run. Full FER validation P0-P9 was not run. No training, fine-tuning, optimizer update, gradient path, graph rebuild, or test access occurred. No scientific P0-P9 metric, sensitivity label, overall decision, or conclusion exists yet. The frozen package under `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/` is unchanged.
