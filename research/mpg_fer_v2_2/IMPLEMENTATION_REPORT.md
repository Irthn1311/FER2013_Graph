# MPG-FER v2.2 implementation and hardening summary

## Provenance

- Issue: [#95](https://github.com/Irthn1311/FER2013_Graph/issues/95)
- Branch: `research/mpg-fer-v2-2-dynamic-sparse-routing`
- Exact v2.1 base: `4967cc5dac3495be2300210215f72422f6f97aa4`
- Frozen v2.1 source hash: `d86d93655c83810d36c89a632baee0745f1de0f0e701b762b719d44445a6e679`
- Final v2.2 source hash: `cbdeee5d5336338115895d2484ab35c3b233c25718d6c03768b7e0f5a2e93cca`
- Official v2.1 checkpoint SHA-256: `4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75`

## Exact scientific delta

The five existing geometry-aware motif transformer blocks retain the exact v2.1 score, self mask, projections, geometry bias, dropout, DropPath, residual, and feed-forward paths. For every sample, head, and query node, v2.2 selects exactly K non-self key indices with `torch.topk`, masks every other score, and applies softmax only over the selected support. The layer schedule is `[8,16,16,16,24]` from the first forward pass.

No score perturbation is used. Equal-score ordering follows PyTorch `topk` behavior and is not guaranteed stable. Repeated evaluation was exact in the tested environment. A real PublicTest batch produced no FP32 cutoff ties and AMP FP16 cutoff-tie query counts `[15,16,30,25,20]`; the exact selected degree remained K in every case.

The implementation still computes dense 49x49 scores before Top-K. It tests sparse routing as an inductive constraint and does not establish a computational speedup.

## Independent hardening fixes

- replaced threshold-based `score >= kth_score` masking, which could select more than K tied edges, with exact index-scatter selection;
- added fail-closed schedule length/type/range validation and removed the silent K=48 fallback;
- exposed pre-dropout attention for correct normalization/zero-support tests and detached read-only diagnostics;
- corrected v2.2-only resume discovery, notebook Issue identity, kernel identity, and source lock;
- replaced the vacuous attention test with behavioral assertions and expanded coverage for all registered contracts;
- replaced stale copied v2.1 reports and runtime evidence;
- corrected the false lower-key-index tie claim.

## Verification results

| Gate | Result |
|---|---|
| v2.2 tests | `68 passed in 193.68s` |
| frozen v2.1 regression tests | `44 passed in 212.08s` |
| parameter count | `2,304,528`; delta `0` |
| K=48 dense equivalence | max absolute logits error `0.0` |
| sparse replay Public TTA | accuracy `0.6926720535`; macro-F1 `0.6707896323`; exact A2-R match |
| sparse replay Private TTA | accuracy `0.6988018947`; macro-F1 `0.6898134831`; exact A2-R match |
| micro-overfit | 16 Train samples; `100%` at step 30; loss `0.5523311`; gate `>=87.5%` by step 80 |
| bounded AMP audit | batch 16 PASS; all five layers have finite non-zero Q/K/V/geometry gradients |
| notebook | generated from source; six code cells compile |

Black and Ruff were not available in the tested environment. Python compilation, notebook compilation, pytest, and `git diff --check` were used instead.

## Resume and EMA

Resume schema remains `3` because the bundle layout is unchanged. The scientific config hash now includes `motif_topk_schedule`, and the source hash is v2.2-specific, so a v2.1 resume fails both identity guards. Model, EMA, optimizer, scheduler, GradScaler, comparator, patience, history, Python/NumPy/Torch/DataLoader RNG, sampler, augmentation, consistency state, and scheduled tau remain covered by the unchanged resume semantics.

EMA is constructed from the same v2.2 model/config. Top-K has no parameter or mutable support state. Scheduled tau remains a copied buffer rather than an averaged parameter.

## Scientific boundary

The replay verifies implementation equivalence to the previously audited intervention. It is not a new model-selection result. PrivateTest was read only for this preregistered frozen-checkpoint verification and did not influence source, K, thresholds, or follow-up choices.

No official from-scratch v2.2 training was launched. Whether dynamic sparse routing improves, preserves, or degrades generalization remains UNKNOWN.

Final status: `A3_REVIEWED_READY_FOR_OFFICIAL_V22_RUN`.
