# TensorFlow Step 8 direct-part pathway decomposition

## Archive

- Archive: `tf_step8_direct_part_decomposition_kaggle_t4.zip`.
- Archive SHA-256: `d050c607d71d45c597e192436566e97cbf51a7709e771fbe66166f1654e782ce`.
- Archive members: `15`.

## Provenance

- Issue: #15.
- Preregistered scientific base commit: `d14e7e1e3eec2ffbd5339b5a3bd0d5db5ab3de8b`.
- Reviewed execution commit: `27c366a955648764386fe48e489a6a1e94a479a1`.
- Step 7 probe-tool SHA-256: `c0b1df778e469665dd6437c58831d29dcc34fbde44231db75894c5469a1ade78`.
- Step 6 support-tool SHA-256: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- Runtime: TensorFlow `2.18.1`, Keras `3.15.0`, two Tesla T4 GPUs.

## Completion and integrity

- Wrapper status: `COMPLETE`.
- Step 7 subprocess return code: `0`.
- Scientific result valid: `true`.
- Validation samples/batches: `3589` / `113`.
- Training, optimizer, and gradient execution: `false`.
- Test access: `false`.
- Checkpoint/config artifacts unchanged: `true`.
- Model weights unchanged: `true`.
- Graph, node, edge, and topology inputs unchanged: `true`.

## Mandatory gates

### Gate A — native/manual D0 equivalence: `PASS`

- Prediction agreement: `1.0`; registered requirement: exactly `1.0`.
- Maximum absolute logit difference: `4.76837158203125e-06`; registered tolerance: `<= 1e-5`.
- Maximum absolute probability difference: `1.519918441772461e-06`; registered tolerance: `<= 3e-6`.

### Gate B — D0 reproduction: `PASS`

- Accuracy: `0.63137364168292`.
- Macro-F1: `0.5932591901893336`.
- Loss: `1.1537981840361535`.

### Gate C — D5/C1 anchor reproduction: `PASS`

- Accuracy: `0.27751462803009197`.
- Macro-F1: `0.19745892656222366`.
- Loss: `1.7577204408898819`.

## D0-D5 validation results

| Condition | Accuracy | Macro-F1 | Loss | D0-minus-condition macro-F1 | Registered label |
| --- | ---: | ---: | ---: | ---: | --- |
| D0 `official_manual_forward` | `0.63137364168292` | `0.5932591901893336` | `1.1537981840361535` | — | Reference |
| D1 `context_local_prior_neutralized` | `0.6241292839230984` | `0.5903344983698285` | `1.1655407532126503` | `0.2924691819505054` pp | `LOW_PATH_SENSITIVITY` |
| D2 `readout_local_prior_neutralized` | `0.6177208135971023` | `0.5779821585567638` | `1.1573156239712132` | `1.527703163256977` pp | `LOW_PATH_SENSITIVITY` |
| D3 `local_part_residual_zero` | `0.22596823627751464` | `0.1958426679087715` | `1.883221954371022` | `39.74165222805621` pp | `HIGH_PATH_SENSITIVITY` |
| D4 `local_motif_validity_off` | `0.6138200055725829` | `0.5736200475460801` | `1.1431544808159888` | `1.9639142643253504` pp | `LOW_PATH_SENSITIVITY` |
| D5 `full_direct_part_zero_anchor` | `0.27751462803009197` | `0.19745892656222366` | `1.7577204408898819` | — | Step 6 C1 anchor |

## Paired diagnostics versus D0

- D1 prediction disagreement rate: `0.11033714126497632`.
- D2 prediction disagreement rate: `0.08581777653942603`.
- D3 prediction disagreement rate: `0.7258289217052104`.
- D4 prediction disagreement rate: `0.11284480356645304`.
- D5 prediction disagreement rate: `0.6857063248815826`.

D3 correctness transitions:

- D0 correct to D3 incorrect: `1663`.
- D0 incorrect to D3 correct: `208`.
- Unchanged correct: `603`.
- Unchanged incorrect: `1115`.

## Preregistered decision

`SINGLE_HIGH_DIRECT_PATH`

## Bounded conclusion

Under the fixed official MediaPipe-derived scaffold, the large direct-part dependency identified by Step 6 is functionally concentrated in the local pooled-part residual pathway. D3 alone reduces macro-F1 from `59.3259%` to `19.5843%`, close to the full D5/C1 anchor at `19.7459%`. D1, D2, and D4 are `LOW_PATH_SENSITIVITY` under their registered individual interventions.

These nonlinear intervention results are not additive causal attribution. The `39.74165222805621` pp D3 change is not a percentage contribution of MediaPipe. The result does not establish that MediaPipe caused the Issue #7 generalization gap, that MediaPipe-specific pooling is necessary, or that the evaluated graph is prior-free.

D3 zeros the learned local pooled residual embeddings themselves. Step 8 therefore identifies the critical pathway but does not distinguish dependence on MediaPipe-specific pooling or semantic assignment from dependence on having strong learned local visual embeddings.
