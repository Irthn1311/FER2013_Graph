# TF Step 11 candidate validation-training harness

Status: `HARNESS_IMPLEMENTATION_ONLY`

Issue: `#29`, including Protocol Amendment A comment `5437568570` and
Protocol Amendment B comment `5437895559`.
Implementation base: `572885a0bb650434f5b36bd3be2049524377067b`.

## Locked identities

- Pre-Amendment-B candidate model SHA-256: `0069b2fa4a548719c1eeb464b820d22b6d12a686606dcf81ea3e498d24d4515d`.
- Relocked candidate model SHA-256: `0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca`.
- Candidate execution adapter SHA-256: `48c0e5f8ad4676e17fb4127b3a30ad053beedca8e04e05cfb6fb24f2bb9236f9`.
- Pre-Amendment-B candidate execution contract SHA-256: `7d0c8fec08564bc405276413b70d7ad3d1f0adbcfc9c18d4d50b22d5efd4ce6f`.
- Relocked candidate execution contract SHA-256: `331570bacd3ec97474c85f25e7e3cb461ef42b0aa3f442caf3dd1f52314bcbc7`.
- Relocked candidate validation harness SHA-256: `1b0707c41f30a9a5b9b9dba3995030ac50fccc90cf439d1ac26a31a32a878f2f`.
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

Protocol Amendment B adds one candidate-only precision boundary. The learned
slot pool, Q, attention math, and raw diagnostics remain float32. Only a copy
of the raw slot embeddings is cast to the exact official-global dtype for the
residual dictionary, residual stack, and readout residual input. The official
global residual is not cast. Outside this explicit boundary, inherited mixed
precision arithmetic is unchanged.

## Secondary pre-hotfix float32 evidence

The previous float32 bounded implementation evidence remains recorded:

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

This evidence used the selected G1-A path with global float32 and is secondary,
not the mixed-production readiness basis. The original Step-10 model SHA
`0069b2...` failed before an optimizer update under `mixed_float16` because it
combined float32 learned slots with the inherited float16 global residual.

## Mixed-float16 bounded production-path closure

Golden/synthetic-only evidence under the actual `mixed_float16`,
`restricted_tf_function`, G1-A path established:

- candidate forward: passed;
- raw learned-slot diagnostics: float32, shape `[8,4,96]`;
- official global residual: unchanged float16;
- residual stack: homogeneous float16, shape `[8,5,96]`;
- readout residual inputs: the four cast float16 slot copies plus the untouched
  float16 official global residual;
- parameters: `1061576`;
- trainable variables: `128`, exact baseline prefix `0..126`, Q at index `127`;
- Q: `[4,96]`, float32, finite non-zero gradient;
- Q gradient L2 norm: `0.012508180923759937`;
- inherited dynamic-loss-scale calibration calls before the accepted update:
  `1` (`32768 -> 16384`, with no iteration or Q change);
- accepted registered optimizer call: iterations `0 -> 1`;
- maximum absolute Q change on the accepted call:
  `0.0003000088036060333`;
- optimizer variables: `262 -> 262` across calibration and accepted update;
- model trainable-variable identities: unchanged;
- mixed-policy checkpoint reload: exact
  `LearnedLocalResidualSlotLapGNN`, `1061576` parameters, 128 variables, Q
  `[4,96]` float32, exact Q state preserved.

## Verification

- Focused Step-11 suite: `31 passed`.
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
