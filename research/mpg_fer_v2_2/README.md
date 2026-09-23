# MPG-FER v2.2

Implementation for [Issue #95](https://github.com/Irthn1311/FER2013_Graph/issues/95), stacked on the exact reviewed v2.1 base `4967cc5dac3495be2300210215f72422f6f97aa4`.

The only scientific delta is dynamic hard Top-K non-self support inside the five existing motif transformer blocks. The score remains `QK/sqrt(head_dim) + geometry_bias`; softmax is restricted to the selected support with the locked schedule `[8,16,16,16,24]` from epoch 1, step 1. No parameter, loss, optimizer, scheduler, data, EMA, checkpoint-selection, or early-stopping change is introduced.

Canonical package source SHA-256: `cbdeee5d5336338115895d2484ab35c3b233c25718d6c03768b7e0f5a2e93cca`.

## Local validation

```powershell
$env:PYTHONPATH = (Resolve-Path 'research\mpg_fer_v2_2\src').Path
C:\Users\ADMIN\anaconda3\envs\fer-graph\python.exe -m pytest -q research\mpg_fer_v2_2\tests
C:\Users\ADMIN\anaconda3\envs\fer-graph\python.exe research\mpg_fer_v2_2\tools\bounded_gpu_audit.py --batch-size 16
```

The full test suite passed with `68 passed`. The bounded GPU audit, micro-overfit, and frozen-checkpoint replay evidence live under `outputs/`. The full source/config/parameter review is in `../mpg_fer_audit/A3_DYNAMIC_SPARSE_ROUTING_IMPLEMENTATION.md` and its JSON companions.

The notebook `notebooks/MPG_FER_v2_2_Kaggle_T4.ipynb` is generated from the package sources. Auto-resume only discovers datasets named `mpg-fer-v2-2-resume*`; v2.1 artifacts cannot be selected accidentally.

No official v2.2 training was launched. Dense score computation is retained before Top-K, so this implementation makes no speedup claim.
