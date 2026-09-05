# TF Step 13 execution preparation

Status: `STEP13_EXECUTION_PREPARATION_ONLY`

## Scope

This is the Issue #40 pre-run execution wrapper only. It does not implement or reinterpret any P0-P9 calculation. The reviewed scientific probe remains unchanged and is invoked exactly once through its existing `main` entry point.

A thin adapter is necessary because the reviewed probe intentionally owns only validation inference and scientific gates. The adapter adds Kaggle archive materialization, exact source/runtime preflight, fresh-output protection, a reconstructible invocation record, and compact evidence packaging without duplicating scientific logic.

## Locked identities

- Scientific base: `d90cce8c4d23f8f1c2958c76cda4ce9d8cae6608`
- Reviewed Step 13 probe SHA-256: `cf68c47d428d0b569828d65028024fcc0713e963419ff5511be91b1377327118`
- Execution adapter: `tools/run_issue40_step13_execution.py`
- Execution adapter SHA-256: `02c0452d4349d22daa8b2ad85a2ea2cbc8159aab313246127292f735333c5d5c`
- Candidate model SHA-256: `0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca`
- Step-12E archive SHA-256: `f436b0a7a20c751b2fd2f47738469fb409ecf9a1a40628e05d20974639927451`
- Epoch-42 checkpoint SHA-256: `e0d633cb6200e963f31a28750e28c7febdaae40344c90ba9d94b826a09e4b78c`
- Epoch-42 weights SHA-256: `a18a372f70ce56868ae43257e9b7fa5e20517499c2c1e35c48dba4d65eaaaa74`
- Epoch-42 metadata SHA-256: `a5ee759bc6fbef587e025199d0dcfe6ebd3a1764cffa567f793c53e972eb47cf`
- Learned-slot Q SHA-256: `54b368aa183c65d5843d8b8e340d3020412d1a2dfeaabbe8b2c0166684ab3ff9`
- Resolved config SHA-256: `3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32`
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`
- Frozen execution contract SHA-256: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`

The adapter requires the scientific base to be an ancestor of the reviewed execution checkout, a clean tracked worktree, exact probe/candidate/payload identities, and exact matching lock constants embedded in the reviewed probe. It accepts either the exact Step-12E ZIP or a Base64 transport that decodes to that exact ZIP. It extracts only the four registered `run/` members and validates every member before runtime configuration or probe invocation.

## Registered runtime and safety boundary

The wrapper requires Python `3.12.12`, TensorFlow `2.18.1`, Keras `3.15.0`, CUDA `12.5.1`, cuDNN major `9`, and exactly two Tesla T4 GPUs. Before loading the probe it configures mixed precision `mixed_float16`, disables XLA, enables memory growth on both GPUs, and validates the effective runtime.

The fixed resource record is evaluation batch `32`, graph workers `2`, graph cache `64`, and registered tf-data prefetch target `2`. The reviewed probe uses `GraphBatchGenerator.iter_epoch` and deliberately exposes no tf-data/prefetch selector. The adapter records prefetch `2` as `NOT_APPLICABLE_TO_REVIEWED_ITER_EPOCH` and does not monkeypatch the iterator or alter the reviewed scientific execution path.

There is no split option. The probe command remains fixed to its validation-only path, contains no `--limit-val-batches`, and requires the explicit confirmation `RUN_ISSUE40_VALIDATION_ONLY_P0_P9`. Existing output or archive paths fail closed. Scientific inputs are never placed in the compact output archive; `.keras`, `.h5`, train/test CSVs, and test-named paths are forbidden archive members.

Only a complete probe manifest with exactly `3,589` samples, exact P0-P9 inventory, Gate A/B/C PASS, no batch limit, and false training/optimizer/test flags can produce `scientific_result_valid=true`. The adapter copies no scientific calculation: it only validates the reviewed probe's completion record. A technical or gate failure remains explicit and has null scientific interpretation.

## Proposed Kaggle inputs

- Exact Step-12E Base64 transport: `/kaggle/input/datasets/irthn1311/tf-step12e-reviewed/tf_step12c_checkpoint_continuation_kaggle_t4.zip.b64`
- Official validation prior root: `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`
- Official clean graph cache: `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`
- Repository checkout: `/kaggle/working/FER2013_Graph`
- Fresh run output: `/kaggle/working/tf_step13_remaining_prior_probe_run`
- Compact evidence ZIP: `/kaggle/working/tf_step13_remaining_prior_probe_kaggle_t4.zip`

The exact locked resolved config extracted from the reviewed archive retains the registered FER validation CSV path. The adapter neither accepts nor discovers an alternate FER/test path.

## Exact proposed scientific command — not executed

```bash
python /kaggle/working/FER2013_Graph/tools/run_issue40_step13_execution.py \
  --repository-root /kaggle/working/FER2013_Graph \
  --step12e-source /kaggle/input/datasets/irthn1311/tf-step12e-reviewed/tf_step12c_checkpoint_continuation_kaggle_t4.zip.b64 \
  --prior-root /kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue \
  --clean-graph-cache-dir /kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records \
  --output-root /kaggle/working/tf_step13_remaining_prior_probe_run \
  --archive-path /kaggle/working/tf_step13_remaining_prior_probe_kaggle_t4.zip \
  --confirm RUN_ISSUE40_VALIDATION_ONLY_P0_P9
```

This command is documentation only in this PR and was not executed.

## Post-run evidence procedure

1. Download `tf_step13_remaining_prior_probe_kaggle_t4.zip` without rerunning.
2. Record the outer ZIP SHA-256, byte size, member count, and ZIP integrity.
3. Require `wrapper_execution.json` status `COMPLETE` and `scientific_result_valid=true`.
4. Independently verify `pre_run_manifest.json`, `probe_output/probe_manifest.json`, Gate A/B/C, exact P0-P9 inventory, sample/batch counts, paired diagnostics, and all source/model/topology immutability evidence.
5. Confirm the archive has no model container, train/test CSV, test prediction, test metric, or test-named path.
6. Only after independent review, record the bounded scientific result in a separate evidence-closure change.

## Static and synthetic verification

- Issue #40 adapter suite: `20 passed`.
- Reviewed Issue #38 probe suite in an LF checkout matching Kaggle/Linux: `50 passed`.
- Candidate plus reviewed Step-7/Step-6 regressions: `54 passed`.
- Fresh subprocess CLI with `PYTHONPATH` removed: PASS.
- Fresh adapter import with `PYTHONPATH` removed / PyTorch isolation: PASS.
- Frozen checksum verification: `PASS checked=267 failures=0`.
- Frozen package and reviewed probe diff from exact base: empty.
- `git diff --check`: PASS.

The combined local Windows checkout run completed `123` tests and exposed one pre-existing raw-working-tree SHA assertion failure because Git's configured checkout converted the reviewed candidate source from LF to CRLF. No repository file was changed to accommodate this platform artifact. The same complete Issue #38 suite passed `50/50` from a temporary `core.autocrlf=false` checkout, and the exact probe/candidate Git-blob SHA identities were independently verified.

No Kaggle execution, full FER validation, training, optimizer/gradient update, test access, P0-P9 metric, sensitivity label, or scientific decision was produced. Step 14 was not started.
