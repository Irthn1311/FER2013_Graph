# LAP-GNN LS0.05 runtime-equivalent preparation evidence

Status: `LAP_LS005_RUNTIME_EQUIVALENT_PREPARATION_ONLY`

Issue: #61

Exact parent: `6d89d17b2d3c39b7bf57084f9de4b707a92eb73e`

## Scope

This additive candidate prepares the exact Issue #60 LS0.05 validation-only
experiment with one runtime-only lifecycle change: full clean-train evaluation
is not constructed or run inside the epoch loop. The frozen trainer remains the
lifecycle owner, so its training, validation, earliest-strict-max-validation-
accuracy checkpoint, validation-loss early-stop, and validation-loss scheduler
semantics remain unchanged.

After a normal or registered early-stop completion, the adapter reloads
`best_val_accuracy.keras` with `compile=False`, verifies that it is the earliest
strict maximum validation-accuracy checkpoint, evaluates the complete clean
train split once with augmentation/prior corruption off and frozen hard CE,
then evaluates the complete validation split once with frozen hard CE. It
records the Issue #60 endpoint gaps, deltas, gap improvement, and registered
decision. It never constructs or accesses the test split.

No FER2013 train/validation execution and no Kaggle execution occurred during
this implementation pass.

## Frozen identities and semantic diff

- scientific payload SHA-256:
  `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`
- execution contract SHA-256:
  `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`
- frozen trainer SHA-256:
  `4c3cb1aa311578038ff656cb7d119103ae5a651135f8ee1c76e37c2c04c1fc75`
- frozen validation-only wrapper SHA-256:
  `c94c122066fdd19210c8ba64a2a61567b249fad4f69c69cb4236b68cce6ff7b4`
- accepted LS0.05 loss adapter SHA-256:
  `72435f59cbc138f4e2184b384d0a4e2c85cdd0082655a6371e0a61ee6362e1e5`
- model parameters: `1,061,192`
- trainable variables: `127`

Against the frozen seed42 base config, the loaded candidate has exactly these
semantic differences:

```text
loss.label_smoothing: 0.0 -> 0.05
run_name: ofix7_mid_seed42 -> ofix7_mid_seed42_ls005_runtime_equivalent
training.eval_train_metrics: true -> false
```

Relative to the accepted Issue #60 LS0.05 config, the scientific loss remains
identical; only run metadata changes and the observational per-epoch
clean-train measurement is disabled. Seed 42, the exact smoothed training
target, augmentation, batch size, graph/prior/data semantics, optimizer,
learning rate, weight decay, clipping, scheduler, early stopping, maximum
epochs, checkpoint selection, hard-CE validation, and future two-T4 execution
class are unchanged.

The scientific CLI fails closed unless exactly two TensorFlow-visible GPUs are
present and both report a device name containing `T4`; the inherited CPU
override is not exposed by this candidate.

The actual frozen epoch bookkeeping order is preserved byte-for-byte:
training, validation, optional observational clean-train evaluation, early-stop
update, checkpoint update, scheduler update. In the lean configuration the
optional evaluation is absent; no bookkeeping operation is reimplemented or
reordered.

## Deterministic lifecycle equivalence

The tiny harness compared three epochs from identical initialization and seed
semantics:

```text
A: train -> clean-train inference -> validation -> frozen bookkeeping
B: train -> validation -> frozen bookkeeping
```

Result: `PASS`, exact equality, floating tolerance `0.0`, no differing epochs.
The complete lean state record has SHA-256
`50f4af20469f6e97c22f3cf3456a729d251c9e75142833239a18052eb0574305`.

Equality was checked after every epoch for trainable and non-trainable model
variables, optimizer variables/slots and iteration, learning rate, scheduler
bookkeeping, validation metrics, earliest-strict-max checkpoint state,
early-stop state, next-epoch shuffled sample order, and next-epoch augmentation
sequence. The registered validation loss is the exact input to both scheduler
and early stopping. The augmentation/RNG sequence did not change.

## Issue #60 resume audit

The read-only audit did not load or execute the checkpoint. The partial
checkpoint SHA-256 was
`5205cee9069673c1dda9ffdea38b205def3ebb1ba8e35f89ab6abff5d1420eaf`.
Its Keras container passed integrity checking and contains optimizer datasets;
the metadata reports optimizer iteration/state, scheduler/LR, early-stop state,
and checkpoint epoch 31. The partial history has 32 completed rows.

Exact continuation is not provable because the selected checkpoint is not the
latest completed epoch, its captured scheduler state precedes that epoch's
scheduler step, and data-generator state, global TensorFlow RNG state,
augmentation continuation state, and partial-epoch model/optimizer state were
not serialized. Disposition:

`ISSUE60_CHECKPOINT_NOT_AUTHORIZED_FOR_EXACT_RESUME`

The new training path contains no reference to, and never loads, the Issue #60
partial checkpoint. The audit contains no scientific interpretation.

## Technical runtime accounting

Source: the Issue #60 merged stdout/stderr log, SHA-256
`97bbaa9e4813d76f4b067060e2a428eaae8c5efce18586bcda347c2a8c56ee8a`.
Only 32 completed-epoch timing lines were parsed; partial accuracy/F1 values
were deliberately ignored.

| Phase | Mean seconds | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Training | 886.78125 | 883 | 871 | 951 |
| Clean-train evaluation | 388.59375 | 388 | 383 | 398 |
| Validation | 49.34375 | 49 | 48 | 62 |
| Other logged lifecycle overhead | 0.4375 | 0 | -1 | 2 |

The `-1` minimum is a one-second log-rounding residual, not negative physical
runtime. Using the all-epoch means for lean per-epoch training, validation and
other overhead, then adding one final mean clean-train evaluation and one final
mean validation, gives:

| Termination epoch | Projected seconds | Projected hours |
|---|---:|---:|
| 32 | 30,407.9375 | 8.4466493056 |
| 40 | 37,900.4375 | 10.5278993056 |
| 45 | 42,583.25 | 11.8286805556 |

These are technical projections, not runtime guarantees. They exclude Kaggle
startup/upload contingency; the 45-epoch projection is therefore particularly
close to the hard limit and does not authorize a run.

## Candidate source identities

| File | SHA-256 |
|---|---|
| `__init__.py` | `a64f380abc110b493db3017bbd652f6133e6a284e14198dee38dd0b2f275156e` |
| `configs/fer2013_ofix7_mid_seed42_ls005_lean.yaml` | `8c1947e86eba24cb9c783d6a3c1e11cd5c4d4dcce4db0e84e10f617ae7c8d38f` |
| `equivalence.py` | `14ab787e728266fd87421f5436f2f59fbe63ce587a0c24f9c0f1760fc3c38362` |
| `resume_audit.py` | `beae8f0138aaec6252f66e66b624e9424b90b857a84044ee653c48cbba90f74d` |
| `runtime_accounting.py` | `4f106372564ea3f387337af98707c216e9b4742e74a2b0aa94942b4298afe5f5` |
| `train_validation_only.py` | `6886315999cf4eaa18381430357b2228ada4f005d04e9855617a2768475ea871` |
| `tests/test_tf_lap_gnn_ls005_runtime_equivalent.py` | `5dd5d3228c4c1106c551385d6ed2657e6ebad7b811f13face06737f5d3e244ce` |

## Verification

- focused runtime-equivalent suite: `24 passed`;
- focused runtime-equivalent plus accepted Issue #58 suite: `58 passed`;
- frozen lifecycle/checkpoint/scheduler/early-stop/identity/isolation selection:
  `26 passed`;
- fresh absolute CLI `--help` from outside the repository with `PYTHONPATH`
  removed: exit `0`;
- parent/frozen package diff: zero files;
- PyTorch runtime imports: none;
- lexical train/prior/cache test-path guard: PASS;
- frozen checksum verifier: PASS;
- `git diff --check`: PASS.

The final verification counts above are recorded after the corresponding
commands complete. This evidence establishes implementation and state
equivalence only; it does not authorize training, validate a FER endpoint, or
revive the invalid Issue #60 attempt.
