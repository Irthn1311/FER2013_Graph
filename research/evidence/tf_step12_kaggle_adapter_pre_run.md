# TensorFlow Step 12 Kaggle adapter pre-run evidence

## Scope and status

- Issue: #31, `[TF Research Step 12 Run] Run seed42 validation-only learned local residual-slot candidate`.
- Exact implementation/execution source: `cc54ec045f2af0dad6aca4bf4b8b1710677ab1a4`.
- Scope: deterministic pre-run Kaggle T4 adapter infrastructure only.
- Kaggle execution: **not run**.
- Full FER training: **not run**.
- Test split: **not accessed**.
- Candidate/scientific source: unchanged.

The generated notebook performs one future registered subprocess invocation only after a clean detached checkout and all source, contract, environment, input, and candidate-identity checks pass. The scientific decision is not computed unless the subprocess succeeds and both reviewed validation-only completion markers pass every Issue #31 validity gate.

## Locked source identities

- Execution commit: `cc54ec045f2af0dad6aca4bf4b8b1710677ab1a4`.
- Candidate model: `0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca`.
- Candidate execution adapter: `48c0e5f8ad4676e17fb4127b3a30ad053beedca8e04e05cfb6fb24f2bb9236f9`.
- Candidate execution contract: `331570bacd3ec97474c85f25e7e3cb461ef42b0aa3f442caf3dd1f52314bcbc7`.
- Candidate validation harness: `1b0707c41f30a9a5b9b9dba3995030ac50fccc90cf439d1ac26a31a32a878f2f`.
- Frozen scientific payload: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- Inherited execution contract: `14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22`.
- Frozen execution source: `2f0a579f51fb216d859b2a7e063614e7f76e5a74948067b7d7abd9f2d59e2f70`.
- Frozen validation-only wrapper: `c94c122066fdd19210c8ba64a2a61567b249fad4f69c69cb4236b68cce6ff7b4`.
- Frozen trainer: `4c3cb1aa311578038ff656cb7d119103ae5a651135f8ee1c76e37c2c04c1fc75`.
- Seed-42 config: `aa3bf2d3932bbad6c5f8cdcc347f4a9866e2c027d6135a60b5002a8f6a3b6908`.

The adapter also verifies the exact candidate class `LearnedLocalResidualSlotLapGNN`, `1,061,576` parameters, `128` ordered trainable variables, and Q at index `127` with shape `[4,96]` and dtype `float32`. The candidate execution contract remains `restricted_tf_function`, Grappler `G1-A`, with the reviewed mixed-precision boundary and no cast of the official global residual.

## Kaggle inputs and Internet boundary

- FER Kaggle Input root: `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`.
  - Resolved sample assets: `train.csv` and `val.csv` only.
- MediaPipe-prior Kaggle Input root: `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`.
  - Resolved sample assets: `train/*.npz` and `val/*.npz` only.
- Clean-cache Kaggle Input root: `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`.
  - Resolved sample assets: `train/index.json`, `val/index.json`, and only their referenced shards.
  - Shared metadata: root `CACHE_COMPLETE.json`, treated only as non-sample aggregate metadata required by the frozen loader.

Internet is required to clone the exact Git commit and only if the Kaggle image needs the pinned runtime dependencies. FER data, priors, and graph cache are read from offline Kaggle Inputs. No test CSV, prior record, cache index/shard, label, prediction, metric, checkpoint inference, or confusion matrix is opened or produced.

## Registered environment and command

The adapter fails closed unless it observes Python `3.12.12`, TensorFlow `2.18.1`, Keras `3.15.0`, and exactly two detected GPUs whose names contain `T4`.

The single registered subprocess is equivalent to:

```text
python -B research/candidates/tf_learned_local_residual_slots/train_validation_only.py
  --config standalone/lap_gnn_tensorflow_ofix7_mid_candidate/configs/fer2013_ofix7_mid_tensorflow_seed42.yaml
  --fer-csv /kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/train.csv
  --prior-root /kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue
  --output-root /kaggle/working/tf_step12_learned_local_residual_slots_seed42/run
  --device gpu
  --graph-workers 2
  --tf-data-prefetch 2
  --tf-data-parallel-calls 1
  --graph-cache-size 64
  --clean-graph-cache-dir /kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records
  --batch-size 16
  --eval-batch-size 32
  --mixed-precision
  --no-xla
  --memory-growth
  --no-resume
```

There are no `--limit-*` arguments, no op-determinism change, no direct frozen-trainer invocation, and no retry loop.

## Failure-safe publication

The subprocess uses merged stdout/stderr and writes continuously to `adapter/subprocess.log`. A separate read-only monitor polls only the atomically persisted `history.json`. For each new completed epoch it writes adapter-owned `runtime_progress.json` and atomically creates, verifies, then replaces:

`/kaggle/working/tf_step12_learned_local_residual_slots_seed42_kaggle_t4.zip`

The rolling ZIP includes the pre-run manifest, subprocess log, adapter progress, and the latest available whitelisted history, training log, epoch summary, telemetry, resolved config, provenance, best-validation-accuracy Keras/weights/metadata artifacts, and completion markers. It never pauses, resumes, restarts, or mutates trainer/model/optimizer/scheduler/checkpoint/history state.

On nonzero subprocess exit or verification failure, the wrapper records `TECHNICAL_OR_RUNTIME_FAILURE`, `scientific_result_valid=false`, and `scientific_interpretation=null`; it removes/refuses `final_evidence.json` and republishes all available partial diagnostic evidence. A Kaggle hard censor before valid markers therefore leaves only diagnostic rolling evidence and cannot receive a registered performance label.

On normal success, the adapter independently validates complete sequential history, both marker contracts, restored trainer bindings, source identity before/after, resolved runtime resources, checkpoint class/parameters/Q identity and hashes, metadata/history agreement, and test isolation. Only then does it derive the locked primary/secondary metrics and create `wrapper_execution.json`, `final_evidence.json`, and `/kaggle/working/tf_step12_learned_local_residual_slots_seed42.md`.

## Verification completed locally

- `python -m pytest -q tests/test_issue31_kaggle_adapter.py` — PASS, `21 passed`.
  - Covers deterministic unexecuted notebook generation, exact locks/resources, decision boundaries, completion gates, malformed/missing sidecars, checkpoint identity failures, source drift, atomic rolling archives, synthetic epoch progression, synthetic subprocess failure, valid success publication, hard censoring, test isolation, and absence of retry/direct-trainer execution.
- `C:\Users\ADMIN\anaconda3\envs\lap-gnn-tf\python.exe -m pytest -q tests/test_tf_candidate_validation_training_harness.py tests/test_tf_learned_local_residual_slots.py` — PASS, `42 passed` with 14 dependency deprecation warnings.
- `verify_checksums.py` — PASS, `checked=267 failures=0`.
- `verify_no_parent_imports.py` — PASS, no violations.
- `verify_no_torch_runtime.py` — PASS, no violations.
- Package isolation tests (`test_no_parent_imports.py`, `test_no_torch_runtime.py`, `test_scientific_payload_checksum_portable.py`) — PASS, `3 passed` with dependency deprecation warnings.
- `git diff --check` — PASS.

An initial attempt to collect the two Step-11 regression files with the system Anaconda interpreter stopped before test collection because that interpreter has no TensorFlow installation. The same command was then run in the repository's `lap-gnn-tf` environment and passed as reported above. Neither attempt accessed FER data or executed training.

## Scientific boundary

This PR creates and verifies execution infrastructure only. It produces no candidate trajectory, metric, registered decision, or scientific interpretation. Passing adapter and Step-11 regressions demonstrates implementation integrity, not candidate performance. The first full candidate run remains unauthorized until research-lead pre-run review passes.
