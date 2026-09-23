# A3 independent source review: dynamic sparse motif routing

Final verdict: `A3_REVIEWED_READY_FOR_OFFICIAL_V22_RUN`

## A. Provenance

- Repository: `Irthn1311/FER2013_Graph`
- Preregistered implementation Issue: [#95](https://github.com/Irthn1311/FER2013_Graph/issues/95)
- Frozen base branch: `research/mpg-fer-v2-1-issue93`
- Exact base SHA: `4967cc5dac3495be2300210215f72422f6f97aa4`
- Review branch: `research/mpg-fer-v2-2-dynamic-sparse-routing`
- Frozen v2.1 source hash: `d86d93655c83810d36c89a632baee0745f1de0f0e701b762b719d44445a6e679`
- Final v2.2 source hash: `cbdeee5d5336338115895d2484ab35c3b233c25718d6c03768b7e0f5a2e93cca`
- Official v2.1 EMA checkpoint: `4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75`, epoch 57

The A0/A1/A2/A2-R/A2-R2 chain justifies this controlled experiment. It does not establish that a from-scratch v2.2 model will be better. The official v2.2 result remains UNKNOWN.

## B. Recovered previous-agent work status

The prior A3 tree was recovered as entirely untracked local work on `research/mpg-fer-v2-1-issue93` at exact HEAD `4967cc5...`. There were no tracked or staged A3 changes and no local or remote v2.2 branch. The work was preserved by creating `research/mpg-fer-v2-2-dynamic-sparse-routing` directly at the exact base; no reset, clean, restore, or destructive checkout was used.

The previous report was not accepted as evidence. Its source, tests, notebook, hashes, and runtime claims were independently inspected and rerun.

## C. Direct diff review findings

Every Python source file was compared against `research/mpg_fer_v2_1/src/mpg_fer_v2_1` at the exact base. The machine-readable review is `a3_source_diff.json`; the config review is `a3_config_diff.json`.

| File | v2.1 SHA-256 | v2.2 SHA-256 | Category | Scientific behavior changed? | Reason |
|---|---|---|---|---|---|
| `__init__.py` | `1dceb96fa42f888db3734db751565de1f257b5cf88d76f8957b7514bd034f28e` | `4cb96c1c7b3a85e3d4aa8669f323dda77d743e9ca605aeb08c2841c0f203bb59` | identity | no | package and Issue identity |
| `checkpoint.py` | `9a500835fed567397fc77d00b856b28fec0f65e856088344f9df1d4ca6178dca` | `8d08596d171c83cf2d47bb362a4c9c5a89507c22fffcdbfa4444d19ab38ea143` | identity | no | docstring only; schema/layout unchanged |
| `config.py` | `5025451fbcd68639118c1eea6c457107ee10e724ff5e0183e2969647b73e2887` | `2bf041e4756921749d1492620091ad6e1a9bcd510e7ce5fa3dc114f95bb17a0e` | registered config | no | declares and validates locked schedule; operator is in `model.py` |
| `data.py` | `17d74c9b3672ad281629aeda8fb16936d61cb8daac5873058e371a000cdb3d6e` | same | byte-identical | no | unchanged |
| `ema.py` | `abfb6543100a6e628f95607351a998711b6d53062c3fb3ad91555510d9651957` | same | byte-identical | no | unchanged |
| `evaluate.py` | `3cdfca32172288cf51784840cfdac62e65f35aaee79bb9e004065652ccbe1b19` | same | byte-identical | no | unchanged |
| `features.py` | `2a39fe1aed2450b99ef65eaaae4b4f78837204e71757aaf31a73211934aa3c6c` | same | byte-identical | no | unchanged |
| `graph.py` | `4a417b2cc7aa48bc79d9cb6249776a8f8e6d8d08f48aa57db065c4aa7237a60a` | same | byte-identical | no | unchanged |
| `kaggle.py` | `6544493c6dc4c66f35aeeed7cf25139e31de65b1124cdf8092e3f5c96e80b18e` | `6ec77dc85b4777d7523d9d384cefbaf0f6d80b4522ea8e3a27b39703020f8614` | engineering identity | no | v2.2-only resume namespace |
| `losses.py` | `76be06c69c4a7b3c52854cfd6793cf3420f4b93ca7513dd93ad0a752fbd264e2` | `fc7d9ca1c1230125a4ca0cc4b6c51879e9c76dcd4c6089fe511b272b6641c66c` | identity | no | docstring only |
| `model.py` | `0d16763b02d72b3d6cb880b567013c9a8c1703fedfa017af5e208654f925d9b6` | `915aaf12790eb8a145451da2d9f3920401d061bf4110902259bcd8cc35d97a30` | scientific operator | **yes** | exact dynamic hard Top-K support plus read-only diagnostics |
| `motif.py` | `694886e7e2438c96c9bf7b8edcb83d24339eed0bae33285b2ac01d20b305a6e2` | `25bdcfaee69b59abe95a3a57c415911ec81a7e542f53a0abe9462e24d9aefc43` | identity | no | version text only |
| `train.py` | `96742599e21973b8102028bda046d18defc71087637f336eff91f6a303364ab5` | `0c2de7818d5a872602fb0c11e3aefc0fbb5baa5af116c1507a9091f224c72c83` | identity | no | version text only |
| `utils.py` | `a7825cdc1cfc2d10363b4cd3de6aa5962830991093b106f2eb10255595cde9c9` | same | byte-identical | no | unchanged |

## D. Findings by severity

### BLOCKER findings, all fixed

1. Threshold masking used `scores >= kth_score`; exact ties could select more than K edges. Fixed with the unmodified indices returned by `torch.topk` and boolean scatter.
2. Invalid schedule lengths and K values did not fail closed; missing entries silently fell back to dense K=48. Fixed with config and block validation and removal of fallback behavior.
3. Resume discovery still searched `mpg-fer-v2-1-resume*`, permitting v2.1 artifact selection before later hash rejection. Fixed to the v2.2-only namespace.
4. The notebook expected the v2.1 source hash and used a v2.1 kernel identity. Fixed and regenerated from the final v2.2 sources.
5. No A3/v2.2 GitHub Issue existed. Issue #95 now contains the frozen contract and explicitly forbids official training.

### MAJOR findings, all fixed

1. The attention normalization/zero-support test had no assertions. It now checks pre-dropout normalization and exact zeros off support.
2. The previous report falsely guaranteed lower-index tie-breaking. The implementation/report now state the actual PyTorch contract.
3. `README.md`, `IMPLEMENTATION_REPORT.md`, and bounded/resume outputs were copied v2.1 artifacts. They were replaced with measured v2.2 evidence.
4. Sparse tests did not independently cover invalid schedule, exact ties, non-zero selected-path gradients, AMP, TTA, full config equality, or per-module parameter equality. Coverage was added.

### MINOR findings, all fixed

1. Routing entropy/top-1 diagnostics were computed after attention dropout. They are now read-only, detached, and based on pre-dropout normalized attention.
2. Stale v2.1 run-ID strings remained in copied resume tests. They were corrected to v2.2 identity.

No unresolved BLOCKER, MAJOR, or MINOR finding remains.

## E. Fixes made

The fixes are limited to exact-K correctness, fail-closed validation, diagnostic correctness, test completeness, and v2.2 provenance/identity. Scores were not perturbed; no deterministic epsilon, STE, warmup, curriculum, loss, architecture redesign, or sparse kernel was added.

## F. Exact scientific diff

For score tensor shape `[B,H,49,49]`, each block keeps the existing v2.1 score and self mask. `torch.topk(..., dim=-1)` selects exactly K key indices independently for every sample, head, and query. Non-selected scores are masked with `torch.finfo(dtype).min`; softmax output is explicitly zeroed off support. The schedule maps layers 1-5 to `[8,16,16,16,24]`.

This is active from epoch 1, step 1. No parameters or mutable routing state are added. Gradients flow through selected Q/K/V/geometry score/value paths; gradients do not propagate through the discrete index choice.

## G. Config diff

`a3_config_diff.json` confirms every v2.1 field is equal. The sole additional field is:

```json
{"motif_topk_schedule": [8, 16, 16, 16, 24]}
```

Learning rate, decay horizon, tau schedule, MI, dropout, DropPath, SupCon, consistency, EMA, augmentation, batching, seed, early stop, selector, TTA, label smoothing, and all loss weights are unchanged.

## H. Parameter reconciliation

`a3_parameter_reconciliation.json` confirms exact equality for every named parameter and every top-level module. Totals are `2,304,528` for v2.1 and v2.2; delta `0`.

## I. Dense equivalence

The official v2.1 EMA checkpoint strict-loaded into v2.2 with schedule `[48,48,48,48,48]`. On fixed synthetic inputs, maximum absolute logits error versus the real v2.1 model was `0.0` (gate `<=1e-6`).

## J. A2-R sparse replay equivalence

The same frozen checkpoint strict-loaded with `[8,16,16,16,24]`:

| Split | Metric | A2-R expected | Measured | Difference |
|---|---|---:|---:|---:|
| PublicTest | TTA accuracy | `0.6926720534967957` | `0.6926720534967957` | `0.0` |
| PublicTest | TTA macro-F1 | `0.6707896323378159` | `0.6707896323378159` | `0.0` |
| PrivateTest | TTA accuracy | `0.6988018946781833` | `0.6988018946781833` | `0.0` |
| PrivateTest | TTA macro-F1 | `0.6898134831099085` | `0.6898134831099085` | `0.0` |

This was implementation verification only. PrivateTest did not influence code, K, thresholds, or model selection.

## K. Gradient and AMP tests

The bounded RTX 3050 Ti audit used batch 16, AMP, both original and flipped forwards, the full v2.1 objective, backward, clipping, optimizer, and EMA. All five motif layers had finite non-zero Q/K/V/geometry gradients. Peak allocated/reserved memory was `2819.629/3014.0 MiB`. This is a local bounded audit, not a Kaggle T4 benchmark.

## L. Micro-overfit

The independent gate used 16 real Train samples, the full objective, and sparse routing from step 1. It reached `100%` training classification at step 30 with final loss `0.5523311496`, passing the registered `>=87.5%` within 80 steps threshold.

## M. Resume and checkpoint verification

Schema stays at `3`: bundle fields and state layout are unchanged. The v2.2 config hash includes the schedule and the v2.2 source hash differs, so v2.1 resumes fail closed. Auto-discovery accepts only `mpg-fer-v2-2-resume*`. Model, optimizer, scheduler, GradScaler, comparator, patience, history, Python/NumPy/Torch CPU/Torch CUDA/DataLoader RNG, sampler, augmentation, consistency selection, and tau state are unchanged and covered by the passing resume suites.

Measured CPU resume artifact: `36.6609 MiB`; save+hash `0.2690s`; load+hash `0.3065s`.

## N. EMA verification

EMA deep-copies the configured v2.2 model, including the same schedule. Top-K has no parameter or buffer. Existing model parameters are averaged exactly as v2.1; scheduled tau remains an authoritative copied buffer. No support indices or runtime support state are serialized.

## O. Notebook and source lock

`MPG_FER_v2_2_Kaggle_T4.ipynb` imports only `mpg_fer_v2_2`, embeds all canonical package files, expects source hash `cbdeee5d...`, uses v2.2 output/kernel identities, and compiles all six code cells. Public checkpoint selection is unchanged; Private evaluation is refused unless training status is complete and the frozen checkpoint hash matches.

## P. Staging and Kaggle readiness

The staging tool uses only the v2.2 notebook, writes v2.2 kernel titles and paths, keeps Internet disabled, and preserves explicit dataset declarations. Segment-02 resume is guarded by checkpoint SHA, schema, run ID, v2.2 source hash, and config hash. No Kaggle kernel was submitted and no official training was launched.

## Q. Routing diagnostic side-effect audit

Routing diagnostics are derived from the already-computed pre-dropout attention, detached, and reduced to scalars before being returned by the full model. They add no loss, parameter, forward, DataLoader access, augmentation, consistency choice, or RNG consumption. Full masks/attention are exposed only by the block-level opt-in test interface and are not retained across batches or epochs.

## R. `torch.topk` tie semantics correction

PyTorch does not guarantee stable indices for equal values. No lower-index guarantee is made. No score perturbation was added. Exact-K selection uses returned indices directly.

On a real PublicTest batch, cutoff-tie query counts were `[0,0,0,0,0]` in FP32 and `[15,16,30,25,20]` under AMP FP16 for layers 1-5. Repeated eval forwards were bit-identical in the tested environment, but this observation is not a cross-platform stable-order guarantee.

## S. Source hash

Canonical v2.2 package source SHA-256: `cbdeee5d5336338115895d2484ab35c3b233c25718d6c03768b7e0f5a2e93cca`.

## T. Git commit SHA

Implementation commit: `PENDING_COMMIT`.

## U. Draft PR

Draft PR: `PENDING_PR`.

## Validation commands and outcomes

- v2.2 suite: `68 passed in 193.68s`.
- frozen v2.1 suite: `44 passed in 212.08s`.
- notebook JSON parsed and all six code cells compiled.
- real-data sparse replay matched all registered A2-R metrics exactly.
- `git diff --check`: passed after staging.
- Black/Ruff: unavailable in the selected environment; not claimed as run.

## Final scope confirmation

- v2.1 source changed: **NO**
- v2.1 branch changed: **NO**
- official training launched: **NO**
- Private used for model selection: **NO**

The only scientific delta is dynamic hard Top-K motif support with the Train-derived layer schedule `[8,16,16,16,24]`; all other scientific behavior remains frozen to v2.1.
