# TensorFlow Step 8 Gate-A probability calibration

## Scope and status

This implements the preregistered Issue #19 technical amendment only. It does not run the Step 8 experiment, alter `manual_forward`, change any D0-D5 intervention, or produce a D1-D4 scientific outcome. Kaggle was not run and PR #16 was not updated or relocked.

## Provenance and evidence basis

- Exact implementation base: `4b0053dfa54c88e1c3e2551f078e02e06e00c918`.
- Reviewed forensic archive: `tf_step8_gate_a_forensic_kaggle_t4.zip`.
- Reviewed forensic archive SHA-256: `bf693500078f170ceea094fad319f513c2a64d8610fa41c65ebac088ec954c8d`.
- Measured same-path maximum probability difference: `2.205371856689453e-06` (`manual_1_vs_manual_2`).
- Pre-amendment Step-7 tool SHA-256: `fc60ece71caea14927c4840edfcd527d005737106f60d0bb475b9b1ba79eadd3`.
- Amended Step-7 tool SHA-256: `c0b1df778e469665dd6437c58831d29dcc34fbde44231db75894c5469a1ade78`.
- Unchanged Step-6 support-tool SHA-256: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.

## Registered Gate-A amendment

- Prediction agreement: unchanged, exactly `1.0`.
- Maximum absolute logit difference: unchanged, `<= 1e-5`.
- Maximum absolute probability difference: changed only from `<= 1e-6` to `<= 3e-6`.

The `3e-6` ceiling is above the measured same-path T4 envelope `2.205371856689453e-06`. The controlled pre-hotfix probability mismatch around `1.744628e-4` remains more than 58 times the amended tolerance and still fails.

## Regression evidence

The Step-7 regression suite proves:

- prediction agreement below `1.0` fails while the logit and probability guards remain within tolerance;
- maximum logit difference above `1e-5` fails while prediction/probability guards pass;
- probability difference `2.205371856689453e-06` passes;
- a representable just-below-`3e-6` value passes and a `3.1e-6` value fails;
- controlled pre-hotfix probability mismatch `1.744628e-4` fails decisively;
- mixed-precision native/manual D0 equivalence still passes;
- D5 remains exactly equal to native Step-6 C1;
- source batches remain unchanged and D1-D5 registered pathway semantics remain exact;
- the scientific payload remains exact.

## Frozen invariants

- D0-D5 condition order and intervention definitions: unchanged.
- Gate B D0 reference: accuracy `0.63137364168292`, macro-F1 `0.5932591901893336`, loss `1.1537981724317095`, sample count `3589`; tolerances `0.001`, `0.001`, `0.005` unchanged.
- Gate C D5/C1 reference: accuracy `0.27751462803009197`, macro-F1 `0.19745892656222366`, loss `1.757720434560185`; the same tolerances and sample count remain unchanged.
- D1-D4 sensitivity thresholds and overall decision rules: unchanged.
- Issue #7 checkpoint SHA-256: `9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16`.
- Issue #7 metadata SHA-256: `e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37`.
- Issue #7 resolved-config SHA-256: `3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32`.
- Validation-only/test-isolation/no-training boundary: unchanged.
- Architecture, features, graph topology, and frozen scientific source: unchanged.

## Packaging and verification

- Step-7 test SHA-256: `b601fddd4da4773afa02ec7b339ae9477a1589ca40757757b874559d16347758`.
- Package-manifest SHA-256: `b11095500533bf4653fefe3d3330df5a4a1779bad0b3cf5a28386653afbab0cb`.
- `CHECKSUMS.sha256` SHA-256: `564d455d60bb21d868644bd3ee741f964fecf4ecf2e2c8fb2552cabb6e26b569`.
- Step-7 harness regression suite: PASS, `17 passed`.
- Step-6 support regression suite: PASS, `26 passed`.
- Package checksum verification: PASS, `checked=265 failures=0`.
- Parent-import isolation: PASS, zero violations.
- PyTorch-runtime isolation: PASS, zero violations.

Stop for research-lead review before merge, any PR #16 relock, or any Kaggle execution.
