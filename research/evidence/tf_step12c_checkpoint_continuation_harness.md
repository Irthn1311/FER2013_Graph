# TF Step 12C checkpoint continuation harness

Status: `HARNESS_IMPLEMENTATION_ONLY`

## Scientific boundary

The original Step-12 Kaggle run was hard-censored during epoch 33 before valid
completion markers were produced. It remains classified as:

- `scientific_result_valid = false`
- `scientific_interpretation = null`

This change implements a `CHECKPOINT-CONDITIONED CONTINUATION` from the reviewed
epoch-30 checkpoint. It is explicitly a `checkpoint_conditioned_restart`, not a
bit-for-bit uninterrupted seed42 trajectory. No Kaggle continuation, full FER
continuation, test access, or scientific performance interpretation was run or
produced for this implementation PR.

## Exact locks

- implementation base: `cc54ec045f2af0dad6aca4bf4b8b1710677ab1a4`
- censored rolling archive: `2ada6cfd1ce1c07f6d7ae36264a1f14840a0936e9448a72e6bb464ae6ab71357`
- epoch-30 checkpoint: `818450d56cb480cf08637bee01061e8028a3d58c0f13346716618f0ee186d932`
- epoch-30 weights: `981b1864a5b997b092b128a0c863a9f8dee41105425fce25e63c94e1c165ed78`
- epoch-30 metadata: `adf4fc95e36f85610a280056e1e518cffe71108b8160b852860058ae6708f9ce`
- censored history: `0a2edffbc595f09660e01ccacc5338656aef06892949aad4a9e209aac280789c`
- censored resolved config: `3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32`
- candidate model: `0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca`
- candidate execution adapter: `48c0e5f8ad4676e17fb4127b3a30ad053beedca8e04e05cfb6fb24f2bb9236f9`
- candidate execution contract: `331570bacd3ec97474c85f25e7e3cb461ef42b0aa3f442caf3dd1f52314bcbc7`
- candidate validation harness: `1b0707c41f30a9a5b9b9dba3995030ac50fccc90cf439d1ac26a31a32a878f2f`
- frozen scientific payload: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`
- baseline execution contract: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`
- seed42 config: `aa3bf2d3932bbad6c5f8cdcc347f4a9866e2c027d6135a60b5002a8f6a3b6908`
- continuation harness at implementation-test close: `0618df63e6bad0239ddf6d7a1b6500da53adfd867099258ed25b0d60ba4a8cdc`

## Implemented fail-closed contract

The harness verifies the archive and every registered member before use. It
loads `best_val_accuracy.keras` with its serialized optimizer and requires the
exact candidate class, 1,061,576 parameters, 128 trainable variables, Q at
index 127 with shape `[4, 96]`, float32 dtype, and digest
`166f6e09191f94c52c17af81c2d9ba357c765b2077aab5fc809563a9de6d6270`.
It also requires `LossScaleOptimizer`, 53,822 iterations, 262 optimizer
variables, and learning rate `0.0001500000071246177` before training.

The post-epoch-30 controls are reconstructed directly:

- early stopping: best `1.1009891497350373`, wait 1, min epoch 30, patience 15;
- scheduler: best `1.1009891497350373`, one bad epoch, cooldown 0, last epoch
  30, current LR `0.0001500000071246177`; epoch 30 is not replayed;
- checkpoint policy: best macro-F1 `0.5634445160028113` at epoch 30 and best
  accuracy `0.603789356366676` at epoch 30.

Before any optimizer update, the future production execution must reproduce
the full 3,589-sample epoch-30 validation reference within the registered
`0.001 / 0.001 / 0.005` accuracy, macro-F1, and loss tolerances. The real
archive/checkpoint validation result is `UNKNOWN` in this implementation-only
PR because the registered continuation was not executed. Bounded PASS and
fail-closed fixtures cover the gate logic.

The combined history is exactly source epochs 1–30 plus resumed epochs 31 to
termination. Original source epochs 31 and 32 are retained separately as
`FIRST_RUN_OVERLAP_DIAGNOSTICS`; they cannot gate, tune, retry, stop, schedule,
select a checkpoint, or alter the endpoint. Generator epoch arguments begin at
31. The epoch-body order remains train, validation, clean train evaluation,
early update, checkpoint update, scheduler step, history persistence, summary,
then completed-state publication and stop handling.

After each fully completed resumed epoch, a temporary Keras model with optimizer
is reloaded and identity-checked before `latest_state.keras` and its metadata
are published. A partial next epoch never invokes publication, preserving the
prior fully completed state. The metadata records model/Q/optimizer identities,
post-step controls, combined-history and best-checkpoint hashes, source hashes,
`partial_epoch=false`, and `test_access=false`.

## Bounded verification

The focused suite completed with 33 passing tests. Coverage includes archive
and member SHA rejection, immutable epoch 1–30 prefix, exclusion/preservation
of first-run epochs 31–32, model/Q/optimizer identity guards, exact control
reconstruction without duplicate scheduler stepping, atomic epoch-30 checkpoint
copy, pre-train gate PASS/failure cases, actual epoch 31/32 generator arguments,
candidate G1-A Q update, optimizer iteration advance, frozen epoch-body order,
latest-state Keras roundtrip, control-state roundtrip, partial-save rejection,
hard-censor preservation, absence of a test lifecycle, absence of test
artifacts, frozen-package diff, and `git diff --check`.

The combined focused/candidate/checkpoint/control regression selection completed
with 70 passing tests. Parent-import and PyTorch-runtime isolation completed
with 2 passing tests. Frozen package checksum verification reported
`PASS checked=267 failures=0`; the frozen-package diff from the exact base was
empty.

The mixed-precision bounded Q update may consume loss-scale adjustment attempts
before the first accepted optimizer update; the test requires and observes one
accepted update and Q change without running FER training.

## Runtime artifacts for future reviewed execution

Expected continuation-only artifacts include the combined history, pre-train
validation gate, original overlap source and per-epoch overlap diagnostics,
epoch-30 or improved best-validation-accuracy checkpoint, atomic latest
completed state and metadata, telemetry, and a
`CHECKPOINT_CONTINUATION_VALIDATION_ONLY_COMPLETE.json` marker. No test
generator, test inference, prediction, confusion-matrix, final-test checkpoint
resolution, or `TRAINING_COMPLETE.json` path exists in the harness.
