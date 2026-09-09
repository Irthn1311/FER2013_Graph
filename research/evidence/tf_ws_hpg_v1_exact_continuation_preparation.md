# WS-HPG v1.0 exact continuation preparation

Status: `WS_HPG_V1_EXACT_CONTINUATION_PREPARATION_ONLY`

This is technical harness evidence only. No FER2013 training/validation, Kaggle
execution, test-split access, or Issue #70 state restoration occurred.

## Provenance

- Issue: #71
- exact parent: `ab7c7a49e923764b6192d9e774352fddf8f1df8b`
- implementation commit before this evidence-only commit:
  `74c73449302a617be5ef01634f5c23f82127da1b`
- accepted model SHA-256:
  `177a782cd8d5c2178303c44d120dcdd22b0a2108a0b720c5091a19f3d7cbffe3`
- accepted support SHA-256:
  `b6ed2ddd20a4e82824208929709ff2d6bcb1c5557144d778ed0157fd4768aeee`
- identity: 707,213 parameters; 118 trainable variable objects; 138 total
  Keras variables (including 20 non-trainable Dropout SeedGenerator states)
- continuation capsule contract SHA-256:
  `03a52150e73e9cf0837bc7bd06e56546c0441ecdb51b4b6bedaefbab5ac12eb3`

Accepted training sources remain byte-identical:

- `__init__.py`: `b211b99534a7c3e9ff57ea5ac0b2a7f059307cb56d49f9f1e1214189e6402f43`
- `augmentation.py`: `cd89a727a2fb0037dad6a51da87227101f4d51f9e6a8945f5ccca223d66967cd`
- `data.py`: `c9a0310a2dcda5eab779366b7c2a9357961c273ea2279525ccfb6453f79823b7`
- `train_validation_only.py`: `5b84f29703bc53fbcf5941ec3abbd3af434c6b289156a56845541b35b9fadbe9`

## Capsule state inventory

Every atomically published completed-epoch capsule includes and hashes:

- all 118 trainable model variables;
- all 20 non-trainable model variables, including every Keras Dropout
  SeedGenerator, and an aggregate proof over all 138 Keras variables;
- every AdamW variable/slot, optimizer iteration, WarmupCosine position and
  current learning rate;
- completed epoch count and complete epoch history;
- EarlyStopping best, wait, stopped epoch and best epoch;
- earliest-strict-max-validation-accuracy best, selected epoch, selected
  weights identity and `.keras` checkpoint plus metadata;
- Python, NumPy and TensorFlow global RNG state;
- the full accepted 101-epoch TensorFlow shuffle plan;
- next-epoch original sample order, post-shuffle enumeration indices and
  sampled stateless augmentation parameter sequence.

The plan is materialized from the accepted
`tf.data.Dataset.shuffle(seed=42, reshuffle_each_iteration=True)` stream. The
test proves its permutations equal the accepted `_training_records()` output.
Resume loads the hashed persisted plan and never restarts the epoch-1 shuffle
sequence or substitutes another RNG.

Publication uses a staging directory, per-member SHA-256, canonical aggregate
SHA-256, atomic directory replace, a locked `LATEST.json`, contiguous epochs,
and parent-capsule hashes. Missing/corrupt members, incomplete staging, stale
LATEST, missing inventory, lineage gaps, parent mismatch, partial epochs, and
source/config identity drift all fail closed without older-capsule fallback.

## Exact fresh-process equivalence

Synthetic data only; the full accepted 707,213-parameter WS model was used.
Three distinct Python processes compared uninterrupted epochs 1-4 with epochs
1-2 plus capsule save, process destruction/recreation, restore, and epochs 3-4.

- status: `PASS`
- floating tolerance: `0.0`
- TensorFlow op determinism enabled: `false`
- epoch 3 aggregate SHA, uninterrupted/restored:
  `a4c4813e95092316e04b802ef358170ac83e6f4cb212bf709a2401ad5301af01`
- epoch 4 aggregate SHA, uninterrupted/restored:
  `905d5b35ddde2f30f8d08a5f14f036333afe8af7f5ca6e64acc041e04f4e7366`

Both post-restore epochs were exactly equal for model/non-trainable/Dropout
variables, all optimizer state, iteration/LR, callbacks, selected checkpoint
identity, train loss, validation loss/accuracy, next order, enumeration,
augmentation parameters, and Python/NumPy/TensorFlow RNG state. The synthetic
proof uses one CPU intra-op/inter-op/OMP thread to remove unrelated oneDNN
thread-scheduling variability; this setting is not applied to the future T4
scientific CLI and no op-determinism change was made.

## Planned segmentation

- fixed first-session pause: after the verified epoch-60 capsule;
- natural registered early stopping before epoch 60 remains terminal and takes
  precedence over pausing;
- pause status: `WS_HPG_V1_PLANNED_PAUSE_EPOCH60`;
- no selected-checkpoint scientific evaluation or interpretation at a pause;
- the second process restores the exact capsule and continues at epoch 61
  under the original max-100 schedule and callback state;
- terminal evaluation remains exactly clean-train normal support, validation
  normal support, and validation all-ones support, using one selected
  compile-false model and hard CE.

The invalid hard-censored Issue #70 run is used only for the authorized timing
fact that 79 full epochs completed. No Issue #70 model, optimizer, checkpoint,
history, RNG, metrics, or trajectory is loaded or read; a future run starts at
epoch 0.

## Verification

- focused continuation tests excluding the separately executed process proof:
  `21 passed, 1 deselected`;
- fresh-process 4-epoch proof: `PASS`, 3 distinct processes, tolerance `0.0`;
- combined CF v1.0-v1.3 / RA / WS regressions: `372 passed`;
- LAP runtime-equivalent and exact-continuation regressions: `44 passed`;
- frozen TensorFlow package checksums: `PASS checked=267 failures=0`;
- protected accepted WS/CF/RA/LAP/frozen-package diff: empty;
- fresh CLI from outside the repository with `PYTHONPATH` removed: PASS;
- PyTorch import isolation: PASS;
- `git diff --check`: PASS.

## Technical source hashes

- `__init__.py`: `ec9eb3646b209b94f97b20c7caa49f41d12c7fac1f6aa6597367806cd29e69b3`
- `continuation.py`: `d1d1da85190813ee4cf10cf9c3b22c5c8dcb5cc99128cbd98fe86d7cb0d4bf07`
- `continuation_equivalence.py`: `a47197068206c8fc68c94b2237fdd40645f2c3d74545ec0836502cc9c13cb33d`
- `data_order.py`: `a0bf6cf2ac1e95598f591046916ab89bbcdcef32667ac53472e4f7a2e429e539`
- `train_validation_only.py`: `e2c427a45200b89bcf1e585a59beb8396eeb92fb20a3a9a598498bc5a4829649`
- focused test: `815af83186f07adfb3791531cb2c0c38f4f21fb5978ca96ff4bd02ac171749ed`

Scientific interpretation remains null. This implementation does not authorize
a FER2013 or Kaggle run.
