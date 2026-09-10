# WS-HPG v1.0 exact continuation preparation

Status: `WS_HPG_V1_EXACT_CONTINUATION_PREPARATION_ONLY`

This is technical harness evidence only. No FER2013 training/validation, Kaggle
execution, test-split access, or Issue #70 state restoration occurred.

## Provenance

- Issue/PR: #71 / Draft PR #72
- exact scientific parent: `ab7c7a49e923764b6192d9e774352fddf8f1df8b`
- reviewed blocker head: `675b7387fe2a11e827e4776294d97572815964cf`
- implementation commit before this evidence-only update:
  `322bed121ba8d4e879ec59cf4336617072920ed3`
- accepted model SHA-256:
  `177a782cd8d5c2178303c44d120dcdd22b0a2108a0b720c5091a19f3d7cbffe3`
- accepted support SHA-256:
  `b6ed2ddd20a4e82824208929709ff2d6bcb1c5557144d778ed0157fd4768aeee`
- identity: 707,213 parameters; 118 trainable variable objects; 138 total
  Keras variables (including 20 non-trainable Dropout SeedGenerator states)
- continuation capsule contract SHA-256:
  `f6c96eb8b896eb4246152226ee00af886dee7e745dd5f23b11ac7b218cadaddf`

Accepted training sources remain byte-identical:

- `__init__.py`: `b211b99534a7c3e9ff57ea5ac0b2a7f059307cb56d49f9f1e1214189e6402f43`
- `augmentation.py`: `cd89a727a2fb0037dad6a51da87227101f4d51f9e6a8945f5ccca223d66967cd`
- `data.py`: `c9a0310a2dcda5eab779366b7c2a9357961c273ea2279525ccfb6453f79823b7`
- `train_validation_only.py`: `5b84f29703bc53fbcf5941ec3abbd3af434c6b289156a56845541b35b9fadbe9`

## Scientific-trajectory-preserving lifecycle

The accepted reference path is unmodified: accepted
`build_dataset(..., training=True)`, one four-epoch `model.fit`, accepted full
WS model, AdamW/WarmupCosine, LS0.05 loss, validation semantics, earliest
strict-max checkpoint callback, and EarlyStopping. A read-only observer runs
after those callbacks.

Keras 3.15's finite-dataset iterator constructs a replacement iterator at an
epoch boundary and another at the next epoch start when `steps_per_epoch` is
unset. The accepted one-fit path therefore consumes TensorFlow shuffle
iterations 1, 3, 5, 7, and so on. The continuation plan is derived directly
from successive traversals of the accepted
`tf.data.Dataset.shuffle(seed=42, reshuffle_each_iteration=True)` object and
selects those one-based odd traversals. Enumeration remains after shuffle.
Regression compares multiple raw successive traversals exactly; NumPy or a
replacement RNG is not used.

The production continuation path now preserves the accepted one-`model.fit`
segment structure. A fresh run supplies the full materialized epoch stream to
one fit. A resumed run supplies the remaining stream to one new-process fit;
the verified EarlyStopping state is restored after Keras' fit-boundary reset.
Checkpoint and EarlyStopping callbacks remain ahead of the capsule callback,
so every capsule is written only after accepted epoch bookkeeping.

## Proof 1: accepted lifecycle equivalence

`ACCEPTED_LIFECYCLE_EQUIVALENCE`: `PASS`.

Synthetic data only; eight samples, batch size four, two optimizer batches per
epoch, full 707,213-parameter model, separate processes, floating tolerance
`0.0`, TensorFlow op determinism disabled.

- epoch 1 accepted/new:
  `e0478df875b49d210bfee7bd7111116d8d2c63fa48163862ebf11b14d4a9ead5`
- epoch 2 accepted/new:
  `fda28fd5e93bfc80829ce157606c8496fd1ef7edcc0a9ca4d17258e83121abaf`
- epoch 3 accepted/new:
  `06bdf56c91af5d5ca968277c2acac6b7f07588d1446236c72b0d995b4584f56f`
- epoch 4 accepted/new:
  `4cdd838454aa90be846f5976bedcdb318645abc5e0c2773a49bb05cf3a5b3a7c`

Every pair is exactly equal for all 118 trainable variables, all 20
non-trainable variables/all 138 Keras variables and Dropout SeedGenerators,
all AdamW state, iteration/LR/schedule, callbacks, checkpoint identity,
training/validation metrics, original order, enumeration, augmentation
parameters, and Python/NumPy/TensorFlow RNG state.

## Proof 2: fresh-process continuation equivalence

`FRESH_PROCESS_CONTINUATION_EQUIVALENCE`: `PASS`, run only after Proof 1 passed.

Three distinct processes compared new uninterrupted epochs 1-4 with epochs
1-2, complete capsule save, process destruction/recreation, restore, and epochs
3-4. Floating tolerance is `0.0`; op determinism remains disabled.

- epoch 3 uninterrupted/restored:
  `06bdf56c91af5d5ca968277c2acac6b7f07588d1446236c72b0d995b4584f56f`
- epoch 4 uninterrupted/restored:
  `4cdd838454aa90be846f5976bedcdb318645abc5e0c2773a49bb05cf3a5b3a7c`

Both post-restore epochs are exactly equal across the complete Proof 1 state
inventory. Both independent proofs pass; scientific execution remains
unauthorized pending research-lead review.

## Capsule state and storage contract

Each atomically published completed-epoch capsule hashes and preserves:

- all model trainable/non-trainable/Keras variables and Dropout generators;
- all AdamW variables/slots, iteration, schedule position and current LR;
- complete epoch history, EarlyStopping state, and earliest-strict-max
  checkpoint state plus selected `.keras` and metadata;
- Python, NumPy and TensorFlow global RNG state;
- next-epoch original order and post-shuffle enumeration;
- exact immutable shuffle-plan and augmentation-sequence SHA references.

Schema v2 stores the 101-epoch accepted shuffle plan and deterministic
post-enumeration augmentation sequence once at continuation-root level. Both
assets have member and value SHA-256 manifests; every capsule and `LATEST.json`
locks the immutable manifest/values. They are not duplicated per capsule.
Normal persistence fully verifies the newest capsule and the complete parent
manifest chain without rehashing unchanged large members. A resume and explicit
full verification rehash every member in the entire lineage.

Publication uses staging, per-member SHA-256, canonical aggregate SHA-256,
atomic directory replacement, locked `LATEST.json`, contiguous epoch numbers,
and parent-capsule hashes. Missing/corrupt plan, augmentation, capsule member,
state inventory, incomplete staging, stale LATEST, gaps, parent mismatch,
partial epochs, and identity drift all fail closed without stale fallback.

## Production-dimension synthetic benchmark

Measured locally using `sample_count=28,709`, 101 plan epochs, the full
accepted WS model and built AdamW state. No optimizer/training step, FER data,
validation metric, or test artifact was used.

- immutable shuffle plan: 23,197,138 bytes; materialization: 15.1255 s
- immutable augmentation parameters: 1,033,798 bytes

| Epoch | Capsule bytes | Root bytes | explicit_state.npz | Selected checkpoint | Persist | Full-lineage verify | Staging upper bound |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26,556,905 | 50,789,408 | 9,049,220 | 8,802,005 | 1.5244 s | 0.3976 s | 26,556,905 |
| 30 | 26,561,546 | 821,010,077 | 9,049,220 | 8,802,005 | 3.3966 s | 3.0171 s | 26,561,546 |
| 60 | 26,566,286 | 1,617,929,927 | 9,049,220 | 8,802,005 | 12.1119 s | 11.7419 s | 26,566,286 |

The measured epoch-60 storage/persistence/verification envelope does not pose
a material risk of pushing a session near Kaggle's 12-hour censor. Peak
temporary staging was conservatively bounded by the completed capsule size.

## Planned segmentation

- fixed first-session pause follows verified epoch-60 capsule publication;
- natural EarlyStopping before/at epoch 60 remains terminal and has priority;
- pause status is `WS_HPG_V1_PLANNED_PAUSE_EPOCH60`;
- no terminal selected-checkpoint evaluation or scientific interpretation at
  pause;
- resume continues at epoch 61 under the original max-100 schedule;
- terminal evaluation remains the accepted clean-train normal, validation
  normal, and validation all-ones-support inventory using hard CE.

Issue #70 remains invalid. No Issue #70 checkpoint, model/optimizer state,
history, RNG, metrics, or trajectory was loaded or read. A future authorized
scientific run must start at epoch 0.

## Verification

- focused Issue #71 suite: `27 passed`;
- combined CF v1.0-v1.3 / RA / accepted WS suites: `372 passed`;
- LAP runtime-equivalent/exact-continuation suites: `44 passed`;
- frozen TensorFlow checksums: `PASS checked=267 failures=0`;
- accepted WS/CF/RA/LAP/frozen-package diff from exact parent: empty;
- fresh CLI outside repository with `PYTHONPATH` removed: PASS;
- PyTorch import isolation: PASS;
- Python source compilation: PASS;
- `git diff --check`: PASS.

## Technical source hashes

- `__init__.py`: `ec9eb3646b209b94f97b20c7caa49f41d12c7fac1f6aa6597367806cd29e69b3`
- `continuation.py`: `0d363cee78779461929fe177cb6b61443716c820254910e5056ca67e431cf4a1`
- `continuation_equivalence.py`: `287fcce3b28930e1b6d254a3163a9e8bc5ec9bd37c8691d9d085b26b66a32fd9`
- `data_order.py`: `43b3b409186c8982fef7c2b205a226ae772cb32e2ea418512ab0b763b1b72765`
- `train_validation_only.py`: `fcd6c790c510a5bd9ea3fbfc562dc5b4a46c26de8a4ee381572b3147945dba16`
- `capsule_benchmark.py`: `d2244d8a4c8d717d7151dc2853bd6f385a09e0544016b51e4c4c4c81ddc2c3aa`
- focused test: `369d028a41ed5253af2799c0881156248e20e62e2aa023658ad3866eb845a7a4`

Scientific interpretation remains null. This implementation does not
authorize a FER2013 or Kaggle run.
