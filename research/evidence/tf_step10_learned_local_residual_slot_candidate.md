# TensorFlow Step 10 learned local residual-slot candidate

Status: `ARCHITECTURE_IMPLEMENTATION_ONLY`

Issue: #27

## Exact base and frozen boundary

- Exact implementation base: `4c7d88e6d03f4aa35f657d7ade69f2436c3b89cd`.
- Frozen baseline package: `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/`.
- Frozen scientific payload: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- Frozen execution contract: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`.
- `git diff` under the frozen baseline package relative to the exact base: empty.

## Candidate files

- `research/candidates/tf_learned_local_residual_slots/__init__.py`
- `research/candidates/tf_learned_local_residual_slots/model.py`
- `tests/test_tf_learned_local_residual_slots.py`
- this compact implementation evidence

Learned-slot model file SHA-256:
`0069b2fa4a548719c1eeb464b820d22b6d12a686606dcf81ea3e498d24d4515d`

## Registered learned-slot formula

`LearnedLocalResidualSlotPool` accepts only post-context node embeddings `h`,
`node_graph_index`, and `num_graphs`. It contains one float32 trainable matrix:

`Q` with shape `[4, 96]`, initialized by
`tf.keras.initializers.RandomNormal(stddev=0.02)`.

For graph `g`, query `k`, and node `i`:

`scores[k,i] = dot(Q[k], H_g[i]) / sqrt(96)`

`alpha[k,:] = softmax(scores[k,:])` over nodes of graph `g` only

`slot[k] = sum_i alpha[k,i] * H_g[i]`

The implementation uses graph-indexed segment maxima and sums, so attention
normalization and reductions are independent for every graph and slot. It adds
no projection, bias, MLP, slot FFN, regularizer, or auxiliary objective.

## Parameter count

- Frozen baseline trainable parameters: `1061192`.
- Candidate trainable parameters: `1061576`.
- Exact delta: `384` trainable scalars.
- New trainable variables: exactly one, `Q[4,96]`.

All `127` frozen mapped tensors load completely into the candidate. The new
query matrix is the only candidate-only trainable state.

## Residual and support integration

The candidate continues to call the official `part_pool` and preserves its
exact global mean residual and official `valid_groups`. The four official local
pooled values are discarded from the residual branch. The residual order is
fixed as learned query slots `0,1,2,3`, followed by the official global value,
giving `[B,5,96]` and exactly `480` flattened residual dimensions.

The frozen `MicroMotifSupportReadout` implementation is reused directly. Its
major/micro branches continue to receive the official `part_soft`, official
validity, group priors, node features, and gx/gy detail path. Learned-slot
attention is not an input to those support branches. The dictionary keys used
internally by the unchanged readout are positional interface aliases only and
do not assign anatomical meaning to learned queries.

Descriptive outputs expose learned slot embeddings `[B,4,96]`, node-aligned
graph-local attention weights, per-slot entropy, and per-slot peak attention.
They do not enter an optimization objective or architecture decision.

## Verification

- Focused candidate suite: `11 passed`.
- Exact attention normalization per graph/slot: PASS.
- Cross-graph isolation: PASS.
- Within-graph node permutation invariance: PASS.
- Official global residual identity on the golden fixture: exact PASS.
- Residual order and `480`-dimension shape: PASS.
- Support-branch isolation from learned queries: exact PASS.
- Test-only official-local substitution reproduces frozen logits,
  probabilities, node embeddings, image embedding, and relevant support
  intermediates within the existing TensorFlow tolerance: PASS.
- Candidate `.keras` save/reload preserves `Q` exactly and reproduces logits:
  PASS.
- Frozen parameter/readout/logit parity plus parent-import/PyTorch isolation:
  `5 passed`.
- Fresh candidate import: PASS; `0` PyTorch modules loaded.
- `git diff --check`: PASS.

## Run and interpretation boundary

No Kaggle execution, full FER training, full validation experiment, checkpoint
selection, optimizer experiment, or test-split access occurred. Only the
registered golden fixture and synthetic unit inputs were used for architecture
and equivalence verification.

This controlled candidate removes fixed semantic pooling only from the four
dominant local residual inputs. It retains the official graph topology and
mask, prior-conditioned node and edge channels, `PartGlobalContext`, readout
prior bias, support/motif semantic branches, and validity gating. No scientific
performance outcome or architecture-selection conclusion is produced by these
implementation tests.
