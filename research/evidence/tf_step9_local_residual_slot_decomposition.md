# TensorFlow Step 9 local residual-slot decomposition

## Archive

- Archive: `tf_step9_local_residual_slot_decomposition_kaggle_t4(1).zip`.
- Archive SHA-256: `52abd5fda11049787e33bb1849e6cbae5ff18a984afd2611edc6387db2b62a90`.
- Archive size: `1521640` bytes.
- Archive members: `15`.

## Scientific lineage

- Issue: #23.
- Preregistered scientific Step-9 base: `753ae1a27b9e4467d11c5d68cb416df63de29ff5`.
- Reviewed technical execution commit: `73a5bd6fe1210b379287ca9e0048526ff682e7a9`.
- Step-9 harness SHA-256: `50a310f622cdf9dccf13eff4edf6394f1d39b8ccf315dce5ede07d0a45bdd77a`.
- Step-7 harness SHA-256: `c0b1df778e469665dd6437c58831d29dcc34fbde44231db75894c5469a1ade78`.
- Step-6 support SHA-256: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- Frozen execution contract SHA-256: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`.

## Completion, runtime, and integrity

- Wrapper status: `COMPLETE`.
- Step-9 subprocess return code: `0`.
- Scientific result valid: `true`.
- Scientific interpretation: `MULTIPLE_HIGH_LOCAL_SLOTS`.
- Validation samples/batches: `3589` / `113`.
- Runtime: TensorFlow `2.18.1`, Keras `3.15.0`, two Tesla T4 GPUs.
- Evaluation resources: batch size `32`, graph workers `2`, graph cache size `64`, shuffle `false`.
- Training: `false`.
- Test access: `false`.
- Checkpoint SHA-256 before/after: `9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16`.
- Checkpoint metadata SHA-256: `e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37`.
- Resolved config SHA-256: `3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32`.
- Loaded model-weight SHA-256 before/after: `112aab8827b2414d4bc9b4abfd7c77f1bafe20bfb1b478747ee46401d50a00d3`.
- Checkpoint/config artifacts and loaded model weights unchanged: `true`.
- Graph, node, edge, feature, coordinate, and topology inputs unchanged: `true`.

## Mandatory gates

### Gate A — native/manual S0 equivalence: `PASS`

- Prediction agreement: `1.0`; registered requirement: exactly `1.0`.
- Maximum absolute logit difference: `3.337860107421875e-06`; registered tolerance: `<= 1e-5`.
- Maximum absolute probability difference: `1.0728836059570312e-06`; registered tolerance: `<= 3e-6`.

### Gate B — S0 reproduction: `PASS`

- Accuracy: `0.63137364168292`.
- Macro-F1: `0.5932591901893336`.
- Loss: `1.1537981724317095`.

### Gate C — S5 / Step-8 D3 reproduction: `PASS`

- Accuracy: `0.22596823627751464`.
- Macro-F1: `0.1958426679087715`.
- Loss: `1.8832219533160723`.

## S0-S5 validation results

| Condition | Accuracy | Macro-F1 | Loss | S0-minus-condition macro-F1 | Registered label |
| --- | ---: | ---: | ---: | ---: | --- |
| S0 `official_manual_forward` | `0.63137364168292` | `0.5932591901893336` | `1.1537981724317095` | — | Reference |
| S1 `mouth_local_residual_zero` | `0.45528002229033154` | `0.4306587375753964` | `1.7606334501663141` | `16.260045261393717` pp | `HIGH_SLOT_SENSITIVITY` |
| S2 `eye_local_residual_zero` | `0.5466703817219282` | `0.4885782825564391` | `1.3974468623642373` | `10.46809076328945` pp | `HIGH_SLOT_SENSITIVITY` |
| S3 `brow_local_residual_zero` | `0.567288938422959` | `0.538957345134899` | `1.336691303063283` | `5.430184505443458` pp | `MODERATE_SLOT_SENSITIVITY` |
| S4 `nose_cheek_local_residual_zero` | `0.5775982167734746` | `0.5207253360259078` | `1.354720703795948` | `7.253385416342583` pp | `MODERATE_SLOT_SENSITIVITY` |
| S5 `all_local_residuals_zero_anchor` | `0.22596823627751464` | `0.1958426679087715` | `1.8832219533160723` | — | Step-8 D3 anchor |

## Paired diagnostics versus S0

| Condition | Prediction disagreement rate | S0 correct to intervention incorrect | S0 incorrect to intervention correct | Unchanged correct | Unchanged incorrect |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 mouth | `0.395653385344107` | `816` | `184` | `1450` | `1139` |
| S2 eye | `0.26971301198105324` | `479` | `175` | `1787` | `1148` |
| S3 brow | `0.25633881303984396` | `423` | `193` | `1843` | `1130` |
| S4 nose/cheek | `0.22736138200055725` | `366` | `173` | `1900` | `1150` |
| S5 all local residuals | `0.7258289217052104` | `1663` | `208` | `603` | `1115` |

Independent research-lead recomputation from the complete `3589`-row paired CSV reproduced all accuracies, macro-F1 values, disagreement counts and rates, and correctness-transition counts exactly.

## Preregistered decision

`MULTIPLE_HIGH_LOCAL_SLOTS`

## Bounded conclusion

The large local pooled-residual dependency identified in Step 8 is not concentrated in one individual slot. Under the fixed official MediaPipe-derived scaffold, the mouth and eye local residual pathways each have `HIGH_SLOT_SENSITIVITY`; the brow and nose/cheek local residual pathways each have `MODERATE_SLOT_SENSITIVITY`.

S1-S4 are nonlinear functional-sensitivity interventions on learned pooled local residual pathways. Their effects are not additive causal attributions and must not be summed or divided by the S5 effect. The deltas are not percentage contributions by anatomical region. This result does not establish that MediaPipe-defined regions are inherently necessary, that MediaPipe caused the Issue #7 generalization gap, or that the evaluated graph is MediaPipe-free or prior-free.

## Preserved first attempt

- Classification: `PRE-INTERVENTION TECHNICAL HARNESS FAILURE`.
- Failed archive SHA-256: `ff19925fc4ad6f6d8144512979dd2f725355cacc31303a848bd77037d4a41b17`.
- Scientific result valid: `false`.
- Scientific interpretation: `null`.
- S0-S5 scientific outcome: none.

The failed attempt remains separate technical provenance and is not erased, replaced, or reclassified by the successful registered run.
