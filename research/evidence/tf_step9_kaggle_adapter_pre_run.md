# TF Research Step 9: Kaggle adapter pre-run evidence

Status: `PRE_RUN_ADAPTER_ONLY`

Issue: #23

## Exact locks

- preregistered scientific Step-9 base: `753ae1a27b9e4467d11c5d68cb416df63de29ff5`
- reviewed technical execution commit: `73a5bd6fe1210b379287ca9e0048526ff682e7a9`
- Step-9 harness SHA-256: `50a310f622cdf9dccf13eff4edf6394f1d39b8ccf315dce5ede07d0a45bdd77a`
- Step-7 harness SHA-256: `c0b1df778e469665dd6437c58831d29dcc34fbde44231db75894c5469a1ade78`
- Step-6 support SHA-256: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`
- frozen scientific payload: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`
- frozen execution contract: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`

The notebook clones without trusting branch state, detaches the exact execution
commit, requires a clean worktree, runs package checksum verification, and then
independently verifies the payload, execution contract, and all three tool file
hashes.

## Preserved first authorized run

- classification: `PRE-INTERVENTION TECHNICAL HARNESS FAILURE`
- archive: `tf_step9_local_residual_slot_decomposition_kaggle_t4.zip`
- reviewed archive SHA-256:
  `ff19925fc4ad6f6d8144512979dd2f725355cacc31303a848bd77037d4a41b17`
- failure: `LocalResidualSlotProbeError: Frozen execution contract drift`
- root cause: Step-9 read the validated Step-6 execution-contract identity at
  the wrong return-map level
- scientific result valid: `false`
- scientific interpretation: `null`
- S0-S5 scientific outcome: none

This failed attempt is retained unchanged as technical evidence. The renewed
adapter consumes the reviewed Issue #25 Step-9 harness SHA while retaining the
original scientific Step-9 base separately.

## Registered Issue #7 artifacts

The adapter recursively searches `/kaggle/input` by exact basename and SHA-256,
outside all public sample mounts. Zero or multiple SHA-matched results fail
closed. The required separate read-only artifact input contains:

- `best_val_accuracy.keras` — `9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16`
- `best_val_accuracy.metadata.json` — `e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37`
- `resolved_config.json` — `3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32`

The metadata preflight requires epoch `31`, seed `42`, config hash
`a4038682bf09c03786e86119001cf5f81ac5fec25d09062e6a0866484c32a3cf`,
and the exact graph, feature, prior, and dataset-split signatures registered in
Issue #23.

## Kaggle inputs and access boundary

- FER dataset mount:
  `/kaggle/input/datasets/doduyquynii/fer13-split`; resolved sample input:
  `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/val.csv`.
- MediaPipe-prior mount:
  `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue`;
  resolved root:
  `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`.
  Only `val/*.npz` and required shared root schema/name metadata are read.
- Clean-cache mount and resolved root:
  `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`.
  Only `val/index.json`, its disclosed validation shards, and shared
  `CACHE_COMPLETE.json` metadata are read.
- Issue #7 checkpoint/config bundle: a separate read-only Kaggle Input with an
  arbitrary mount name; the resolved files are determined only by basename plus
  exact SHA-256.

Internet is required only for cloning the exact Git repository commit and, if
the Kaggle image versions differ, installing the registered dependencies. All
FER, prior, cache, checkpoint, metadata, and config assets are offline Kaggle
Inputs.

## Execution and evidence contract

The unexecuted adapter requires Kaggle T4, TensorFlow `2.18.1`, Keras `3.15.0`,
eval batch size `32`, graph workers `2`, graph cache size `64`, shuffle false,
and exactly `3589` validation samples. It invokes the Step-9 harness exactly
once and contains no `--limit-val-batches`, condition, slot, pair, training,
test, graph-rebuild, raw-prior-corruption, or operation-determinism path.

The fixed condition order is exactly S0-S5 from Issue #23. Gate A, Gate B S0,
Gate C S5/D3, reference tolerances, the `10 pp`/`5 pp` slot thresholds, and all
three registered overall-decision identities are locked. The adapter validates
the harness-provided per-slot inventory and labels but does not recompute or
reclassify slot deltas.

On success, the adapter requires the exact ten-file raw harness inventory,
`3589` paired predictions, all Gates A/B/C PASS, exact S0-S5 order, positive
integrity checks for every condition, source/graph/global/validity/readout/context
immutability, checkpoint/model-weight immutability, and training/test isolation.
Only then does it write `adapter_metadata/final_evidence.json` and the scientific
runtime report.

On subprocess non-zero or success-only verification failure, the failure-safe
wrapper records `TECHNICAL_OR_GATE_FAILURE`, return code, error text, artifact
hashes before/after, `artifacts_unchanged`, `scientific_result_valid=false`,
`scientific_interpretation=null`, `training=false`, and `test_access=false`.
It preserves the subprocess log and every available partial probe file, writes
an explicit failure report/status, creates and verifies the compact ZIP, and
then lets later notebook cells finish normally. It never fabricates
`final_evidence.json` on failure.

Outputs in both branches:

- report: `/kaggle/working/tf_step9_local_residual_slot_decomposition.md`
- ZIP: `/kaggle/working/tf_step9_local_residual_slot_decomposition_kaggle_t4.zip`
- subprocess log:
  `/kaggle/working/tf_step9_local_residual_slot_decomposition/step9_subprocess.log`

The archive rejects `.keras`, train/test CSVs or directories, test metrics, and
fabricated final scientific evidence.

## Static and synthetic verification

- Issue #23 adapter tests: `12 passed`
- synthetic subprocess exit `7`: PASS; the wrapper and all later cells returned
  normally, the failure ZIP preserved the recognizable subprocess log,
  pre-run manifest, wrapper execution, failure status, and partial probe JSON;
  `final_evidence.json` was absent and scientific interpretation remained null
- Step-9 harness regressions: `16 passed`
- reviewed Step-7 and Step-6 regressions: `43 passed`
- parent-import and PyTorch-runtime isolation: `2 passed`
- fresh Step-9 import: `0` PyTorch modules loaded
- package checksums: `PASS checked=267 failures=0`
- deterministic notebook rebuild SHA-256:
  `8a8b42605510bec5d07423a05693dfb551f626d861ed5e7ff73766bcf4cdeb8b`
- notebook cells: `19` total, `9` code; every code cell has
  `execution_count = null` and `outputs = []`

Kaggle was not run. The real Issue #7 checkpoint was not executed on full
validation, and no S1-S4 scientific outcomes were produced. Stop for
research-lead pre-run review.
