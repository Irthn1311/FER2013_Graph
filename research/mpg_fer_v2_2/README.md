# MPG-FER v2.2

Implementation tracked by [Issue #95](https://github.com/Irthn1311/FER2013_Graph/issues/95), stacked on the exact reviewed v2.1 base `4967cc5dac3495be2300210215f72422f6f97aa4`. Issue #95 is the GitHub mirror/frozen protocol record created during independent review; the experimental design had already been preregistered in the pre-implementation A3 handoff.

The only scientific delta is dynamic hard Top-K non-self support inside the five existing motif transformer blocks. The score remains `QK/sqrt(head_dim) + geometry_bias`; softmax is restricted to the selected support with the locked schedule `[8,16,16,16,24]` from epoch 1, step 1. No parameter, loss, optimizer, scheduler, data, EMA, checkpoint-selection, or early-stopping change is introduced.

Canonical package source SHA-256: `a8dc77db29e997c4c3ab69bb862704c8a948f940a4636e1c01e0d96bab40de65`.

The official execution contract is fail-closed at physical batch `16` with gradient accumulation `2`; an OOM records `BATCH16_OOM` and stops. There is no B8/acc4 fallback. Detached routing scalars are accumulated during ordinary training, and a fixed non-augmented Train batch (indices 0-15) records per-epoch support Jaccard/turnover without affecting RNG, gradients, optimizer/EMA state, or checkpoint selection. `history.json` and `routing_diagnostics.json` carry the trajectory.

## Local validation

```powershell
$env:PYTHONPATH = (Resolve-Path 'research\mpg_fer_v2_2\src').Path
C:\Users\ADMIN\anaconda3\envs\fer-graph\python.exe -m pytest -q research\mpg_fer_v2_2\tests
C:\Users\ADMIN\anaconda3\envs\fer-graph\python.exe research\mpg_fer_v2_2\tools\bounded_gpu_audit.py --batch-size 16
```

The final A3.1 test count and gate evidence are recorded in `IMPLEMENTATION_REPORT.md`. The bounded GPU audit, micro-overfit, frozen-checkpoint replay, and synthetic routing-diagnostic timing evidence live under `outputs/`. The full source/config/parameter review is in `../mpg_fer_audit/A3_DYNAMIC_SPARSE_ROUTING_IMPLEMENTATION.md` and its JSON companions.

The notebook `notebooks/MPG_FER_v2_2_Kaggle_T4.ipynb` is generated from the package sources. Auto-resume only discovers datasets named `mpg-fer-v2-2-resume*`; v2.1 artifacts cannot be selected accidentally.

No official v2.2 training was launched. Dense score computation is retained before Top-K, so this implementation makes no speedup claim.
