# TensorFlow Step 8 Gate-A forensic result

## Archive

- Archive: `tf_step8_gate_a_forensic_kaggle_t4.zip`.
- Archive SHA-256: `bf693500078f170ceea094fad319f513c2a64d8610fa41c65ebac088ec954c8d`.
- Archive members: `124`.

## Execution

- Forensic execution commit: `3cae1f6c78048cd6cd518d87cd0a5429d72f01e1`.
- Forensic tool SHA-256: `30c00fd6985810533cc09be05f66b64f7da5a794903aef493b9839b461eac7c0`.
- Step-7 tool SHA-256: `fc60ece71caea14927c4840edfcd527d005737106f60d0bb475b9b1ba79eadd3`.
- Step-6 support-tool SHA-256: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- Runtime: TensorFlow `2.18.1`, Keras `3.15.0`, two Tesla T4 GPUs.

## Completion and boundaries

- Validation batches: `113`.
- Validation samples: `3589`.
- Diagnostic subprocess return code: `0`.
- Wrapper status: `COMPLETE`.
- D1-D5 intervention conditions executed: `[]`.
- Training: `false`.
- Test access: `false`.
- Scientific interpretation: `null`.

## Numerical envelopes

### `native_1_vs_native_2`

- Maximum absolute logit difference: `4.976987838745117e-06`.
- Maximum absolute probability difference: `1.1324882507324219e-06`.
- Minimum prediction agreement: `1.0`.
- Batches exceeding the old Gate-A threshold: `1`.

### `manual_1_vs_manual_2`

- Maximum absolute logit difference: `5.364418029785156e-06`.
- Maximum absolute probability difference: `2.205371856689453e-06`.
- Minimum prediction agreement: `1.0`.
- Batches exceeding the old Gate-A threshold: `1`.

### `native_1_vs_manual_1`

- Maximum absolute logit difference: `4.5299530029296875e-06`.
- Maximum absolute probability difference: `1.0132789611816406e-06`.
- Minimum prediction agreement: `1.0`.
- Batches exceeding the old Gate-A threshold: `1`.

### `native_2_vs_manual_2`

- Maximum absolute logit difference: `3.7550926208496094e-06`.
- Maximum absolute probability difference: `8.642673492431641e-07`.
- Minimum prediction agreement: `1.0`.
- Batches exceeding the old Gate-A threshold: `0`.

## Important per-batch observations

- Batch `2`, `native_1_vs_manual_1` maximum absolute probability difference: `1.0132789611816406e-06`.
- Batch `80`, `native_1_vs_native_2` maximum absolute probability difference: `1.1324882507324219e-06`.
- Batch `80`, `manual_1_vs_manual_2` maximum absolute probability difference: `2.205371856689453e-06`.

## Immutability

- Checkpoint SHA-256 before/after: `9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16` / `9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16`.
- Model-weight SHA-256 before/after: `112aab8827b2414d4bc9b4abfd7c77f1bafe20bfb1b478747ee46401d50a00d3` / `112aab8827b2414d4bc9b4abfd7c77f1bafe20bfb1b478747ee46401d50a00d3`.
- Source batches unchanged: `true`.

## Dtype conclusion

- Restored outer LapGNN: policy `mixed_float16`, compute dtype `float16`, variable dtype `float32`, input dtype `float16`, autocast `true`.
- Encoder, GNN, PartGlobalContext, readout, and classifier internals: `float32`.

## Bounded research-lead conclusion

The registered `1e-6` probability Gate-A threshold is below the measured same-path numerical repeatability envelope of the registered Kaggle T4 runtime. This explains the remaining Gate-A blocker without evidence of a material residual native/manual semantic mismatch.

This technical diagnosis does not establish that GPU nondeterminism affects scientific D1-D4 outcomes. Gate A is not modified in PR #18. The separate amendment is preregistered in Issue #19.
