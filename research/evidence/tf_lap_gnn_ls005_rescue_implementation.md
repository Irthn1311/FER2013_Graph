# TF LAP-GNN LS005 rescue implementation evidence

Status: `LAP_GNN_LS005_RESCUE_IMPLEMENTATION_ONLY`

Issue: #58

Exact implementation base: `f8a183520ab1ef7d949e66b48ee7ce7e352f1cb6`

## Scope and frozen identity

The candidate is additive under
`research/candidates/tf_lap_gnn_ls005_rescue/`. The frozen package under
`standalone/lap_gnn_tensorflow_ofix7_mid_candidate/` has zero diff from the
implementation base.

- frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`
- frozen trainer SHA-256: `4c3cb1aa311578038ff656cb7d119103ae5a651135f8ee1c76e37c2c04c1fc75`
- frozen validation-only wrapper SHA-256: `c94c122066fdd19210c8ba64a2a61567b249fad4f69c69cb4236b68cce6ff7b4`
- model parameters: `1,061,192`
- trainable variables: `127`

No model, graph, prior, optimizer, scheduler, early-stop, checkpoint, data, or
evaluation-loss semantics were changed.

## Candidate identity

| File | SHA-256 |
|---|---|
| `__init__.py` | `922044a681dbe862253b11eaf9da8e2e67bab66fbbbdbfa35ba4acb551d444df` |
| `configs/fer2013_ofix7_mid_seed42_ls005.yaml` | `bb89c1df7d563ec26a74469c86091fc58cb9f60698d447f24ea2589cdd605535` |
| `loss_adapter.py` | `72435f59cbc138f4e2184b384d0a4e2c85cdd0082655a6371e0a61ee6362e1e5` |
| `train_validation_only.py` | `8afb489e1fd5444f50c4be48d25e12157ecb6759e7f29521eeb17d9e35697216` |
| `tests/test_tf_lap_gnn_ls005_rescue.py` | `aae0443babd8c5ba36545816e99961a2658c719cec144bc46cb4ba6d48d02ba3` |

The loaded candidate config inherits the frozen seed42 config. Its complete
semantic deep-diff is exactly:

```text
loss.label_smoothing: 0.0 -> 0.05
run_name: ofix7_mid_seed42 -> ofix7_mid_seed42_ls005_rescue
```

## Registered loss and lifecycle isolation

Training targets are computed in float32 as:

```text
(1 - 0.05) * one_hot(label, 7) + 0.05 / 7
```

Only `lap_gnn_tf.training.execution.sparse_cross_entropy` is temporarily
rebound before the registered restricted train step is constructed. The
evaluator binding remains object-identical to frozen hard CE. A `finally`
restores the execution binding after normal return, ordinary exception, and
`KeyboardInterrupt` propagation.

The independent synthetic two-sample loss golden was
`1.179726481437683`; the adapter matched its float64 log-softmax calculation
within `2e-7`. The epsilon-zero control matched frozen hard CE within `1e-7`.
Synthetic forward/backward produced finite loss and finite, non-null gradients.

The wrapper delegates lifecycle ownership to the reviewed frozen
validation-only wrapper, exposes no test path/test-batch option, rejects
test-like basenames and the lexical directory components `test`, `testing`,
`test_split`, and `test-split` for FER, prior, and clean graph-cache inputs. The
guard performs no path I/O. A test-specific clean graph-cache path fails before
the frozen wrapper is loaded. The candidate never constructs a graph generator
or calls the post-validation checkpoint resolver itself.

## Verification

Executed with Python `3.11.15`, TensorFlow `2.18.1`:

- `python -m pytest -q tests/test_tf_lap_gnn_ls005_rescue.py`
  - `34 passed`
- frozen validation-wrapper/execution regression selection:
  - `20 passed`
- `python tools/verify_checksums.py`
  - `PASS checked=267 failures=0`
- `python tools/verify_no_parent_imports.py`
  - `runtime_imports_parent=false`, no violations
- `python tools/verify_no_torch_runtime.py`
  - `runtime_imports_torch=false`, no violations
- fresh absolute candidate CLI `--help` from outside the repository with
  `PYTHONPATH` removed
  - exit code `0`
- frozen package diff versus exact base
  - zero changed files
- `git diff --check`
  - PASS

Only synthetic/golden package fixtures were used. No FER2013 training or
validation, Kaggle execution, optimizer scientific run, test split access,
epsilon sweep, DropMessage/DropEdge, or architecture change occurred.
