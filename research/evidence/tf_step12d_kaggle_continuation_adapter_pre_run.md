# TF Step 12D Kaggle continuation adapter pre-run evidence

Status: `ADAPTER_IMPLEMENTATION_ONLY`

## Scientific boundary

The original Step-12 Kaggle execution remains hard-censored and invalid:

- `scientific_result_valid = false`
- `scientific_interpretation = null`
- no original Step-12 performance label exists

PR #34 was reviewed and merged as exact commit
`0f4fde1d4e6645096711a800509f4db2deedf38f`. The merged Step-12C harness is
reviewed infrastructure for a checkpoint-conditioned continuation. This
adapter has not run Kaggle, has not run a full FER continuation, has not
accessed the test split, and has not produced a scientific label.

## Exact locks

- execution checkout: `0f4fde1d4e6645096711a800509f4db2deedf38f`
- continuation harness: `dba0d749b9a8e05b3cd67dad0749ef4235fc06f2a389b552229c76f691edde40`
- Step-12C evidence: `4e48652c4c75cbdcf985b596e04c5658483825ab1e6900f97d52d1cf7ee7f29f`
- candidate model: `0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca`
- candidate execution adapter: `48c0e5f8ad4676e17fb4127b3a30ad053beedca8e04e05cfb6fb24f2bb9236f9`
- candidate execution contract: `331570bacd3ec97474c85f25e7e3cb461ef42b0aa3f442caf3dd1f52314bcbc7`
- candidate validation harness: `1b0707c41f30a9a5b9b9dba3995030ac50fccc90cf439d1ac26a31a32a878f2f`
- frozen scientific payload: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`
- baseline execution contract: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`
- seed42 config: `aa3bf2d3932bbad6c5f8cdcc347f4a9866e2c027d6135a60b5002a8f6a3b6908`
- censored source archive: `2ada6cfd1ce1c07f6d7ae36264a1f14840a0936e9448a72e6bb464ae6ab71357`
- Base64 source transport: `66bc813bd3e3dcc38a1dd4c0c36e41ddb794831895f15e099cec566d1ad51b8d`
- epoch-30 checkpoint: `818450d56cb480cf08637bee01061e8028a3d58c0f13346716618f0ee186d932`
- notebook SHA-256: `525e6031cda190c608fc8bd8e6863c272ebd99ac6bada576cb982dbaba59aa4f`
- builder SHA-256: `42ac86fec54335d1e3cf96ce447f2649c7d87fbc4e21e06423c364f9a7efc5a8`

The notebook has 9 cells, including 4 code cells. Every code cell has
`execution_count = null` and `outputs = []`; every code cell compiles during
the synthetic adapter suite.

## Registered future invocation

The notebook checks out the exact execution commit detached and verifies a
clean checkout plus all source locks. It then requires exactly one source
transport below `/kaggle/input`: either the direct reviewed ZIP or the locked
`tf_step12_learned_local_residual_slots_seed42_kaggle_t4.zip.b64` transport.
The Base64 form prevents Kaggle dataset ingestion from recursively extracting
the scientific ZIP. Its encoded SHA is checked before decoding; decoding uses
a temporary file and atomic replacement in `/kaggle/working`, and the decoded
file must reproduce the exact reviewed archive SHA before use. Zero, duplicate,
wrong-transport-SHA, malformed-Base64, and wrong-decoded-SHA cases fail closed
without leaving a partial source archive.

It invokes `resume_validation_only.py` exactly once with the exact seed42
config, FER `train.csv`, registered prior/cache roots, verified source archive,
fresh continuation output, GPU, train/eval batches 16/32, graph workers 2,
tf-data prefetch 2, tf-data parallel calls 1, graph cache 64, mixed precision
on, XLA off, and memory growth on. It has no limit, retry, alternate-seed,
op-determinism, direct frozen-trainer, old Step-12 harness, or chained-resume
invocation.

The future runtime is locked to Python 3.12.12, TensorFlow 2.18.1, Keras
3.15.0, CUDA 12.5.1, cuDNN major 9, and exactly two Tesla T4 GPUs. Offline
inputs are:

- FER: `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`
- priors: `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`
- cache: `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`
- the exact censored archive or its registered Base64 transport mounted below
  `/kaggle/input`

Internet is separated from scientific inputs: it is used only for the exact
Git clone and pinned dependency installation if needed. The scientific inputs
are offline Kaggle datasets.

## Failure-safe rolling evidence

The adapter monitors only committed advances of `latest_state_manifest.json`.
For each new `completed_epoch`, it validates the referenced immutable
generation and atomically refreshes:

`/kaggle/working/tf_step12c_checkpoint_continuation_kaggle_t4.zip`

Publication is temporary ZIP, integrity/inventory/member-hash verification,
then `os.replace`. A publication fault leaves the prior verified ZIP intact.
The ZIP includes adapter provenance/log, available root continuation evidence,
the canonical manifest, its exact referenced model/metadata/history, and its
best-validation checkpoint snapshot. Older unreferenced generations are not
required.

Completion interpretation remains fail closed. It requires subprocess return
code zero, exact runtime/source/resources, PASS pretrain gate before optimizer
updates, completion-marker schema version 1, exact source rows 1-30, registered
resumed-row provenance, complete canonical generation and deep Step-12C loader
validation, earliest-global-max strict best-validation-accuracy checkpoint
provenance, and no test artifact. Every combined-history row must carry finite
registered metrics plus the reviewed integer/boolean early-stopping state.
Natural completion is cross-checked against that state: early stopping requires
`stop_requested=true` with wait at least the locked patience 15, while
`max_epochs` requires exact epoch 90 and `stop_requested=false`.

Epoch-31/32 overlap evidence remains descriptive-only. First-run rows must
exactly match the locked source, resumed rows must exactly match the registered
continuation history, and row origin/protocol establish branch identity. Metric
equality and all-zero deltas are accepted and cannot affect validity or retry.
The same fail-closed path-name helper rejects exact `test` directories,
`test.csv`, `test_*`, `test-*`, and the existing prediction/confusion/test
artifacts in both output-tree completion validation and rolling archive
validation, without opening those files. Any failure or hard censor remains
invalid/null and cannot create final scientific evidence or trigger an
automatic retry.

## Static and synthetic verification

- Issue #35 adapter suite: `77 passed`
- Step-12C focused continuation regression: `48 passed`
- parent-import/PyTorch isolation: `2 passed`
- frozen package checksum verification: `PASS checked=267 failures=0`
- frozen package diff from exact base: empty
- `git diff --check`: PASS

The adapter tests use synthetic files and bounded fake subprocesses only. No
real FER continuation, Kaggle execution, or test-split access occurred.
