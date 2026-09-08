# LAP-GNN LS0.05 exact epoch-boundary continuation preparation

Status: `LAP_LS005_EPOCH_BOUNDARY_EXACT_CONTINUATION_PREPARATION_ONLY`

Issue: #63

Stacked implementation parent (Issue #61 head):
`72624bf7bbf556a1d8909eb6f4cc9f00350255a9`

## Scope and permanent Issue #60 exclusion

This technical-only extension protects a future epoch-0 Issue #61 scientific
run from an external platform interruption. It does not authorize FER2013 or
Kaggle execution. Issue #60 remains permanently invalid: its partial
checkpoint is not referenced, opened, copied, loaded, or accepted by this
continuation format. Resume paths explicitly identifying Issue #60 fail before
the runtime lifecycle is entered, and every capsule records
`issue60_artifacts_used=false` in its state, manifest, pointer, and locked
scientific identity.

A future uninterrupted run must start at epoch 0 with an absent or empty
continuation root. Resume is permitted only from `LATEST.json` in a verified
capsule lineage, only after a fully completed epoch, and only when the complete
state and every hash pass. A partial epoch is never persisted. An incomplete
newer staging directory, corrupt latest capsule, missing lineage epoch, or
older `LATEST` pointer causes a hard failure instead of fallback.

## State inventory

Each completed-epoch capsule explicitly preserves and proves:

1. all trainable model variables;
2. all non-trainable model variables, including Keras layer seed generators;
3. every optimizer variable/slot;
4. optimizer iteration;
5. current learning rate;
6. scheduler state;
7. early-stop best value and wait counter;
8. checkpoint-selection best values and epochs;
9. completed epoch and full completed history;
10. training generator split/sample count/batch size/seed/shuffle/cache/worker
    contract and dataset epoch;
11. exact next-epoch sample order;
12. Python RNG state;
13. NumPy legacy RNG state;
14. Keras/TensorFlow layer RNG state through the explicit non-trainable
    variable inventory;
15. the complete next-epoch stateless augmentation seed material.

The registered graph augmentation is not backed by a mutable global generator.
Ordering is `default_rng(seed + epoch*1000003)` and per-sample prior corruption
is `default_rng(seed + epoch*1000003 + sample_index*97531)`. The capsule stores
the next order and every mixed seed as arrays and hashes both. The TensorFlow
global generator is not used by the registered training path; Keras stochastic
layer state is an explicit restored non-trainable model variable.

## Capsule format and integrity

The capsule contract SHA-256 is
`4d689f29a1866ff60a74cbc168b75727aac05670f75cec61bff165a62c1c02e1`.

```text
continuation_root/
  LATEST.json
  epoch_000001/
    manifest.json
    state.json
    explicit_state.npz
    runtime_state.index
    runtime_state.data-*
    selected_checkpoint/
      best_val_accuracy.keras
      best_val_accuracy.weights.h5
      best_val_accuracy.metadata.json
  epoch_000002/
    ...
```

`tf.train.Checkpoint.write/read` supplies TensorFlow trackable restoration, but
it is not trusted as a complete state source. `explicit_state.npz` stores every
model and optimizer variable with index, name, dtype, shape, raw bytes, and
SHA-256 inventory. Restore actively assigns this inventory and then verifies
every array exactly. This is required because the synthetic audit demonstrated
that a Keras `SeedGenerator` non-trainable variable was not restored by the
generic TensorFlow checkpoint alone.

`manifest.json` locks schema/status, scientific identity, required state
inventory, completed epoch, per-member SHA-256 values, and a canonical-JSON
aggregate SHA-256. `LATEST.json` locks the latest epoch, capsule directory,
manifest SHA-256, and aggregate hash. Capsule creation uses a unique hidden
staging directory followed by an atomic directory rename and atomic latest
pointer write. A staging directory left by interruption remains detectable and
blocks fallback.

The three selected-checkpoint artifacts are copied into every capsule so a new
process can reconstruct earliest-strict-max-validation-accuracy selection
state without depending on a platform working directory. The current-epoch
runtime state is restored from the explicit capsule, not from the selected
scientific checkpoint.

## Frozen-trainer integration

The frozen trainer file remains byte-identical. Its exact locked
`run_training()` source is compiled in memory with exactly three reversible
insertions:

1. initialize or restore state after all model/optimizer/scheduler/early-stop/
   checkpoint-policy objects exist;
2. start the epoch range at restored epoch + 1;
3. persist the capsule after scheduler/checkpoint/history/summary bookkeeping
   for the fully completed epoch and before stop/next epoch.

A source regression removes those three insertions and obtains byte-for-byte
the original function source. The temporary function binding is restored in
`finally`, including `KeyboardInterrupt`. The existing validation-only wrapper
still owns the no-test boundary, and the existing LS0.05 context still restores
the hard-CE training binding on every exit.

## Fresh-process equivalence

The deterministic representative uses the registered seed semantics,
mixed-float16 model boundary, `LossScaleOptimizer` around the frozen
Torch-compatible AdamW, LS0.05 training CE, dropout/Keras seed state, frozen
scheduler, frozen early stop, strict-max checkpoint state, and deterministic
epoch/sample augmentation.

Three separate processes ran:

```text
A: epoch 1 -> 2 -> 3 -> 4
B1: epoch 1 -> 2 -> capsule -> process exit
B2: fresh objects -> restore capsule -> epoch 3 -> 4
```

Result:

- status: `PASS`;
- post-restore epochs: `3`, `4`;
- per-epoch exact equality: `true`, `true`;
- floating tolerance: `0.0`;
- restored-state SHA-256:
  `94b2dc6e0b4c7d66696cac7c415fe9d10a069d4730a8ac4408092a7c1ea7545a`.

Equality covers trainable/non-trainable variables, all optimizer/LossScale
slots and iteration, LR, scheduler, early-stop, checkpoint selection,
validation metrics, next-epoch order, augmentation sequence, Python RNG, and
NumPy RNG. Failure of any comparison returns
`LAP_LS005_EXACT_RESUME_NOT_PROVEN`.

## Scientific diff from Issue #61

Scientific diff: none.

- candidate YAML: unchanged;
- frozen LAP package: unchanged;
- accepted LS0.05 adapter: unchanged;
- architecture and model identity: unchanged (`1,061,192` parameters, `127`
  trainable variables);
- epsilon/formula, optimizer, batch size, LR, weight decay, clipping,
  scheduler, patience/max epochs, augmentation, priors, validation cadence,
  checkpoint policy, final hard-CE evaluation, two-T4 guard, and test isolation:
  unchanged.

The only execution changes are the verified technical hooks, capsule I/O,
explicit restoration, and two new CLI paths: required writable
`--continuation-root` and optional exact `--resume-capsule-root`.

## Source SHA-256 values

| File | SHA-256 |
|---|---|
| `continuation.py` | `05e0ed233f3eeda4e2a99b7b4df696b97fb9735cdf2b9661322d8a81802e81c6` |
| `continuation_equivalence.py` | `418f6a049a44e51d6f31c628bebc8d37346f81628d73dd9851b10b5ff6c3e998` |
| `trainer_continuation_adapter.py` | `8c188846adab119fefebf661155404f34a04a4ce589dd9c6d8b3615e6b0da7c2` |
| `train_validation_only.py` | `ec0bf836a890312dbd68ed183f7cf5d31cdc99d90b2fa6f8a1ad17e613dc3905` |
| `tests/test_tf_lap_gnn_ls005_exact_continuation.py` | `83f5b050c6e697c186e490b97c11b71ed02d3d84b6f807d85e8ed2f8c852df3a` |
| `tests/test_tf_lap_gnn_ls005_runtime_equivalent.py` | `7ddd289ed0bb322850e2de53298c3ee34ff31e16e57e95ad307a2f54a78e7b27` |

These hashes are refreshed after the final implementation edit.

## Verification

- exact-continuation focused suite: `20 passed`;
- existing Issue #61 focused suite: `24 passed`;
- combined continuation + Issue #61 + accepted Issue #58 suites: `78 passed`;
- frozen lifecycle/checkpoint/scheduler/early-stop/identity/isolation selection:
  `26 passed`;
- missing member, corrupt member, missing state inventory, incomplete staging,
  stale latest pointer, missing lineage epoch, wrong scientific identity,
  partial epoch, and nonempty epoch-0 root regressions: PASS;
- fresh-process restore with `PYTHONPATH` removed: PASS;
- frozen checksum verifier: `PASS checked=267 failures=0`;
- parent-import isolation: PASS;
- PyTorch runtime isolation: PASS;
- frozen-package diff from Issue #61: zero files;
- candidate config and accepted LS0.05 source diff from Issue #61: zero files;
- `git diff --check`: PASS.

Only deterministic synthetic/tiny tensors were used. No FER2013 training or
validation, Kaggle submission, test access, scientific resume, or Issue #60
checkpoint load occurred.
