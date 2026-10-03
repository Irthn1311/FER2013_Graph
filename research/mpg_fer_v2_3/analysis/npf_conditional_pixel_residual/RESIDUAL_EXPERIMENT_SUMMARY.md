# NPF conditional pixel-residual experiment

Status: **PASS**
Decision: **NO_GO_STOP_STAGE_1**

## Stage 1 selection

| Variant | Selected epoch | Public TTA Acc. | Private raw Acc. | Private TTA Acc. | Private TTA Macro-F1 |
|---|---:|---:|---:|---:|---:|
| R1 | 2 | 69.4344% | 68.5149% | 70.4374% | 69.7995% |
| R2 | 2 | 69.6294% | 68.2084% | 70.4096% | 69.4733% |


Selected by PublicTest only: **R2**, gain over frozen Public NPF **+0.0000 pp**.

The registered GO decision is **False**. The selected NPF-trajectory residual did not satisfy both the registered PublicTest gain and 71.0504% frozen Test reference gates.

No backbone parameter changed. PrivateTest was opened only after both best checkpoints were selected and frozen.
