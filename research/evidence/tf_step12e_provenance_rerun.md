# TensorFlow Step 12E provenance rerun

## Disposition

**VALID CHECKPOINT-CONDITIONED SINGLE-SEED VALIDATION RESULT.**

The provenance-clean rerun authorized in Issue #37 completed normally under the reviewed Issue #35 / PR #36 checkpoint-conditioned continuation protocol.

- Wrapper status: `COMPLETE`
- Subprocess return code: `0`
- Scientific result valid: `true`
- Registered decision: `CHECKPOINT_CONTINUATION_NO_CLEAR_SINGLE_SEED_DIFFERENCE`
- Test access: `false`
- Automatic retry: `false`

This result is a checkpoint-conditioned restart from the reviewed epoch-30 state. It is not an uninterrupted original Step-12 run, and the original censored Step-12 execution remains scientifically invalid.

## Provenance

- Issue: #37
- Related: #31, #35, draft PR #36
- Reviewed adapter head used before this rerun: `248e76e29e4791fc388b2a760bb30d329d8b938f`
- Detached scientific execution checkout: `0f4fde1d4e6645096711a800509f4db2deedf38f`
- Notebook SHA-256: `525e6031cda190c608fc8bd8e6863c272ebd99ac6bada576cb982dbaba59aa4f`
- Builder SHA-256: `42ac86fec54335d1e3cf96ce447f2649c7d87fbc4e21e06423c364f9a7efc5a8`
- Continuation harness SHA-256: `dba0d749b9a8e05b3cd67dad0749ef4235fc06f2a389b552229c76f691edde40`
- Candidate model SHA-256: `0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca`
- Candidate execution adapter SHA-256: `48c0e5f8ad4676e17fb4127b3a30ad053beedca8e04e05cfb6fb24f2bb9236f9`
- Candidate execution contract SHA-256: `331570bacd3ec97474c85f25e7e3cb461ef42b0aa3f442caf3dd1f52314bcbc7`
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`
- Baseline execution contract SHA-256: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`

Source transport and reviewed source archive:

- Base64 transport SHA-256: `66bc813bd3e3dcc38a1dd4c0c36e41ddb794831895f15e099cec566d1ad51b8d`
- Decoded source archive SHA-256: `2ada6cfd1ce1c07f6d7ae36264a1f14840a0936e9448a72e6bb464ae6ab71357`
- Epoch-30 source checkpoint SHA-256: `818450d56cb480cf08637bee01061e8028a3d58c0f13346716618f0ee186d932`
- Epoch-30 source Q SHA-256: `166f6e09191f94c52c17af81c2d9ba357c765b2077aab5fc809563a9de6d6270`

Downloaded rerun continuation archive:

- `tf_step12c_checkpoint_continuation_kaggle_t4 (1).zip`
- size: `98,579,503` bytes
- SHA-256: `f436b0a7a20c751b2fd2f47738469fb409ecf9a1a40628e05d20974639927451`
- members: `30`
- ZIP integrity: PASS

## Runtime and pre-update reproduction gate

Registered runtime reproduced:

- Python `3.12.12`
- TensorFlow `2.18.1`
- Keras `3.15.0`
- CUDA `12.5.1`
- cuDNN major `9`
- exactly two Tesla T4 GPUs
- seed `42`
- train/eval batch `16/32`
- graph workers `2`
- tf-data prefetch `2`
- tf-data parallel calls `1`
- graph cache `64`
- mixed precision ON
- XLA OFF
- memory growth ON

Epoch-30 full-validation gate before any optimizer update:

- samples: `3589`
- accuracy: `0.603789356366676`
- macro-F1: `0.5634445160028113`
- loss: `1.1265364869505958`
- absolute differences from registered references: `0.0 / 0.0 / 0.0`
- optimizer updates before gate: `0`
- status: `PASS`

## Completion and history integrity

The rerun completed by natural early stopping at epoch `44`.

Final row:

- `early_stopping_wait = 15`
- `early_stopping_patience = 15`
- `stop_requested = true`

Independent audit verified:

- combined history has exactly epochs `1..44`;
- source rows `1..30` exactly equal the locked source history prefix;
- every row `31..44` carries `row_origin=CHECKPOINT_CONDITIONED_CONTINUATION` and protocol `tf-step12c-checkpoint-conditioned-continuation-v1`;
- original censored source rows 31/32 remain descriptive overlap diagnostics only;
- no test artifact is present in the archive inventory;
- completion marker schema is `1` and reports `final_test_skipped=true`, `test_access=false`, and `test_data_constructed=false`.

## Registered primary endpoint

Locked baseline best validation macro-F1:

`0.601166548701511`

Independent recomputation from all 44 combined-history rows:

- candidate best validation macro-F1: `0.596090717851928`
- earliest best-macro epoch: `42`
- delta vs locked baseline: `-0.5075830849582963` pp

By the preregistered ±1.0 pp rule:

`CHECKPOINT_CONTINUATION_NO_CLEAR_SINGLE_SEED_DIFFERENCE`

Secondary diagnostics:

- clean-train macro-F1 at epoch 42: `0.7626577994605199`
- train-validation macro gap at epoch 42: `16.65670816085919` pp
- best validation accuracy: `0.6232933964892727` at epoch `42`
- best validation loss: `1.1009891497350373` at epoch `29`
- best-validation-accuracy checkpoint metadata selects epoch `42`, the earliest global maximum validation-accuracy epoch

Best-validation-accuracy artifacts:

- `.keras` SHA-256: `e0d633cb6200e963f31a28750e28c7febdaae40344c90ba9d94b826a09e4b78c`
- `.weights.h5` SHA-256: `a18a372f70ce56868ae43257e9b7fa5e20517499c2c1e35c48dba4d65eaaaa74`
- metadata SHA-256: `a5ee759bc6fbef587e025199d0dcfe6ebd3a1764cffa567f793c53e972eb47cf`

Canonical final state:

- completed epoch: `44`
- next epoch: `45`
- combined-history SHA-256: `036dd31c07d573d8424754d56df1bc1028118cb7103b383b04010dcacded9a46`
- final model-container SHA-256: `6f0ca5a010bdac6e89e28d55961ad345a9d1529526c8fa4f1d572c163ccc90a4`
- final Q SHA-256: `e9e6484277cbaf5d50936c53322ac4bcff01c46bdeb7a285aa22e06022016b81`
- full optimizer-state SHA-256: `08a07fc9387a1e71f17fe138bdf13bcac9bd8c9654029c5d2a26e27387677d2a`

## Reproducibility against the earlier diagnostic continuation

The earlier technically complete but preregistration-invalid continuation archive had SHA-256:

`4443eed620c7f23baa42abe377d3a78401deafc5a0405db4a6006b30c851e457`

The provenance-clean rerun reproduced the entire scientific trajectory through epoch 44. Independent row-by-row comparison found all registered metric/state fields identical across all 44 epochs. Differences were limited to wall-clock timing fields such as `train_phase_sec`, `val_phase_sec`, `train_eval_phase_sec`, and `epoch_time_sec`.

Additional stable identities across the two executions include:

- final Q SHA-256: `e9e6484277cbaf5d50936c53322ac4bcff01c46bdeb7a285aa22e06022016b81`
- final full optimizer-state SHA-256: `08a07fc9387a1e71f17fe138bdf13bcac9bd8c9654029c5d2a26e27387677d2a`
- best checkpoint `.weights.h5` SHA-256: `a18a372f70ce56868ae43257e9b7fa5e20517499c2c1e35c48dba4d65eaaaa74`
- best checkpoint metadata SHA-256: `a5ee759bc6fbef587e025199d0dcfe6ebd3a1764cffa567f793c53e972eb47cf`

The `.keras` container SHA differs between runs even though the weights/metadata identities and scientific trajectory agree; that container-level serialization difference is not used as a scientific outcome.

## Bounded scientific conclusion

For seed 42 under the fixed checkpoint-conditioned continuation protocol, replacing the four fixed anatomical local residual embeddings with four learned graph-local slots yields validation macro-F1 within the preregistered ±1 pp practical-comparability band relative to the frozen baseline.

This does **not** show that overfitting/generalization gap was cured. At the candidate best-macro epoch, the train-validation macro gap is still `16.65670816085919` pp versus the baseline reference gap `15.511397995653287` pp.

The result supports a narrower representation claim: fixed anatomical local residual pooling is not required to preserve near-baseline seed42 validation performance under the retained MediaPipe-derived scaffold. The candidate remains prior-conditioned through topology, node/edge semantic channels, context, support/motif priors, and validity pathways.

No test-set access is authorized by this result.
