# TF Step 11 candidate validation-training harness

Status: `HARNESS_IMPLEMENTATION_ONLY`

Issue: `#29`, including Protocol Amendment A comment `5437568570`.
Implementation base: `572885a0bb650434f5b36bd3be2049524377067b`.

## Locked identities

- Candidate model SHA-256: `0069b2fa4a548719c1eeb464b820d22b6d12a686606dcf81ea3e498d24d4515d`.
- Candidate execution adapter SHA-256: `48c0e5f8ad4676e17fb4127b3a30ad053beedca8e04e05cfb6fb24f2bb9236f9`.
- Candidate execution contract SHA-256: `7d0c8fec08564bc405276413b70d7ad3d1f0adbcfc9c18d4d50b22d5efd4ce6f`.
- Candidate validation harness SHA-256: `9b7d4e76acd953334261a79ea82ae09cfb0b0816435dedad60420899cca0d75f`.
- Frozen validation-only wrapper SHA-256: `c94c122066fdd19210c8ba64a2a61567b249fad4f69c69cb4236b68cce6ff7b4`.
- Frozen trainer SHA-256: `4c3cb1aa311578038ff656cb7d119103ae5a651135f8ee1c76e37c2c04c1fc75`.
- Frozen execution source SHA-256: `2f0a579f51fb216d859b2a7e063614e7f76e5a74948067b7d7abd9f2d59e2f70`.
- Inherited baseline execution contract: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`.
- Frozen scientific payload: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.

## Authorized execution extension

The external adapter preserves the frozen `restricted_tf_function` / G1-A
builder structurally and delegates to the frozen gradient validator with the
only semantic extension `expected_count=128`. The inherited loss, loss-scaling,
gradient, optimizer, clipping, update, tracing, and input-signature behavior is
unchanged. `eager_exact` remains unsupported and out of scope.

The production harness temporarily replaces exactly
`trainer.LapGNN` and `trainer.build_restricted_graph_train_step`, calls the
frozen `train_validation_only.run_validation_only(...)`, and restores both
bindings in `finally`. Tests cover normal completion, training failure, wrapper
failure, and provenance failure.

The frozen config remains unchanged: its parameter lock is `1061192` and its
execution-contract lock remains the inherited baseline contract. Candidate
provenance separately records `1061576` parameters and the candidate execution
contract SHA-256.

## Bounded closure

Golden/synthetic-only evidence established:

- baseline trainable variables: `127`;
- candidate trainable variables: `128`;
- candidate variables `0..126`: exact baseline name/path/shape/dtype order;
- candidate variable `127`: `Q`, shape `[4,96]`, dtype `float32`;
- 127-variable input: rejected;
- 129-variable input: rejected;
- exact 128-variable input: accepted;
- Q gradient L2 norm: `0.015423387289047241`;
- Q gradient: present, float32, finite, and non-zero;
- maximum absolute Q change after one registered step: `0.0003000088036060333`;
- optimizer iterations: `0 -> 1`;
- optimizer variables: `258 -> 258` after the already-built optimizer step;
- model trainable-variable identities: unchanged by the step;
- bounded checkpoint reload: exact `LearnedLocalResidualSlotLapGNN`, `1061576`
  parameters, Q `[4,96]` float32, exact Q state preserved.

The bounded Q update used the selected G1-A `restricted_tf_function` path with
the global float32 policy. Structural equivalence locks the inherited
mixed-precision loss-scaling branches unchanged. A separate pre-run observation
is that the locked Step-10 model currently raises a dtype mismatch under a
global `mixed_float16` policy when its float32 learned slot embeddings are
stacked with the inherited float16 global residual. This Issue does not
authorize modifying the locked candidate-model SHA; no such change was made.

## Verification

- Focused Step-11 suite: `27 passed`.
- Step-10 candidate regression suite: `11 passed`.
- Frozen validation-only wrapper, selected execution/contract, optimizer
  mixed-precision smoke, parent-import, and PyTorch-runtime isolation selection:
  `21 passed`.
- Fresh Step-11 import: passed.
- Frozen package diff from implementation base: empty.
- `git diff --check`: passed.

No Kaggle execution, full FER training, full validation experiment, or test
split access occurred. No candidate scientific performance result or baseline
performance comparison was produced.
