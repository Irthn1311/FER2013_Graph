# TensorFlow Step 6 Kaggle adapter pre-run evidence

## Scope and provenance

- Issue: #11, `[TF Research Step 6] Run fixed-checkpoint fixed-topology validation prior-sensitivity experiment`.
- Frozen scientific base: `main` at `69f4571c5069da9a7f8558ef3c01101635ee904a`.
- Technical execution source: runtime-policy harness hotfix commit `7fbe0ea306f23db9682833a7ff66ea65da7300e9`.
- Adapter notebook: `notebooks/kaggle-issue11-fixed-topology-prior-probe.ipynb`.
- Notebook SHA-256: `97e57273b14e403794175853db04d2366e62acd242936fc700be323a3eeee1b7`.
- Deterministic builder SHA-256: `dad60440574b641ec441dc8017673df877f5464fbc01587233bf253b58c9f9ff`.
- Reviewed Step 5 tool SHA-256: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- The checked-in notebook remains unexecuted. The first two approved Kaggle attempts failed before checkpoint loading/model evaluation and before any C0/C1/C2 metric was produced; both remain classified `PRE-METRIC TECHNICAL HARNESS FAILURE`. The subsequent reviewed run completed, and its compact report is committed at `research/evidence/tf_step6_fixed_topology_prior_sensitivity.md`.

## Runtime-blocker closure

The failed attempt found all three exact registered artifacts at their read-only Kaggle mount and verified their SHA-256 values:

- `/kaggle/input/datasets/irthn1311/kaggle-issue7-validation-only/best_val_accuracy.keras`;
- `/kaggle/input/datasets/irthn1311/kaggle-issue7-validation-only/best_val_accuracy.metadata.json`;
- `/kaggle/input/datasets/irthn1311/kaggle-issue7-validation-only/resolved_config.json`.

It then stopped at the config identity guard: the persisted JSON identity was `a4038682bf09c03786e86119001cf5f81ac5fec25d09062e6a0866484c32a3cf`, while loading that JSON through the YAML loader reinterpreted scientific-notation floats as strings and produced `916472d6e813b15dca6e7cd5016e87e95fad5d608bc1e341117675797cce8e54`. Validation assets were correct and no test access occurred.

The minimal harness hotfix now loads the persisted resolved config as UTF-8 JSON, checks metadata identity against that raw mapping before any mutation, and creates a deep copy for runtime overrides only after the identity gate passes. No identity value is hardcoded into the tool, and every existing fail-closed guard remains active. A regression test locks the exact `a403...` raw identity, reproduces the obsolete `9164...` YAML reinterpretation, and proves runtime mutation cannot alter the raw identity object.

The first attempt also verified FER validation rows `3589`, prior validation files `3589`, and clean-cache validation samples `3589` before the config-identity failure.

## Second runtime-blocker closure

The second attempt passed the corrected config-identity guard and created the T4 devices, then stopped before checkpoint loading when `tf.config.experimental.set_memory_growth(..., True)` raised `RuntimeError: Physical devices cannot be modified after being initialized`. The frozen trainer's `ResourceControls.apply()` treats that exact already-initialized condition as non-fatal, so the probe's stricter behavior was a runtime-policy mismatch rather than a scientific outcome.

The second minimal hotfix aligns only that runtime policy. It records `memory_growth_requested` plus aggregate and per-device status in the probe manifest, continues only for the exact already-initialized TensorFlow error, and remains fail-closed for every other memory-growth `RuntimeError`. The adapter continues to require at least one Kaggle T4 before invoking the probe. Regression tests prove inference setup continues after the exact initialized-device error and that unrelated GPU runtime failures remain fatal.

## Locked Issue #7 artifacts

The adapter accepts exactly one matching file for each expected basename outside the public FER/prior/cache inputs and verifies it by SHA-256 before use:

- `best_val_accuracy.keras`: `9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16`;
- `best_val_accuracy.metadata.json`: `e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37`;
- `resolved_config.json`: `3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32`.

It additionally locks epoch `31`, seed `42`, config hash `a4038682bf09c03786e86119001cf5f81ac5fec25d09062e6a0866484c32a3cf`, the execution/graph/feature/prior/split signatures from Issue #11, and the exact persisted C0 reference metrics. Artifact hashes are checked again after the probe. The reviewed run did not trust the Kaggle mount name; it resolved the artifacts only by their locked basenames and SHA-256 values.

## Kaggle inputs and access boundary

- FER dataset input: `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`; resolved sample artifact is `val.csv` only.
- MediaPipe prior input: `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`; resolved sample artifacts are `val/*.npz` only, plus shared root schema/name metadata required by the frozen loader.
- Clean graph cache input: `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`; resolved sample artifacts are `val/index.json` and its referenced validation shards. Root `CACHE_COMPLETE.json` is read only as required non-sample aggregate loader metadata and may summarize other splits.
- Issue #7 artifact input: a separate read-only Kaggle Input mounted anywhere beneath `/kaggle/input`; the three files above are located by expected basename plus exact SHA-256, with public sample inputs excluded from the search.
- Internet: required only to clone technical execution commit `7fbe0ea306f23db9682833a7ff66ea65da7300e9` (whose scientific base remains `69f4571c5069da9a7f8558ef3c01101635ee904a`) and install registered dependencies if the Kaggle image is not already TensorFlow `2.18.1` / Keras `3.15.0` compatible.

The registered command invokes `evaluate_fixed_checkpoint_prior_probe.py` once with the checkpoint, metadata, config, official prior root, official clean-cache root, and a fresh output root. It contains no `--limit-val-batches`, training command, optimizer path, test split path, or raw-prior corruption path.

## Compact outputs from reviewed execution

- Report: `/kaggle/working/tf_step6_fixed_topology_prior_sensitivity.md`.
- Archive: `/kaggle/working/tf_step6_fixed_topology_prior_sensitivity_kaggle_t4.zip`.

The archive is limited to the Step 5 probe outputs, adapter metadata/evidence, and compact report. It rejects `.keras`, train/test CSVs or directories, and test-metric artifacts.

## Verification performed without experiment execution

- `python -m pytest -q tests/test_issue11_kaggle_adapter.py` — PASS, `10 passed`.
- Relevant Step 5, payload, and isolation tests — PASS, `32 passed`.
- `python tools/verify_checksums.py` from the TensorFlow package — PASS, `checked=261 failures=0`.
- `python tools/verify_no_parent_imports.py` — PASS, zero violations.
- `python tools/verify_no_torch_runtime.py` — PASS, zero violations.
- `nbformat.validate(...)` — PASS, notebook format `4.5`, `19` cells.
- Every notebook code cell compiles; all execution counts are null and all outputs are empty.
- `git diff --check` — PASS.

## Reviewed runtime evidence closure and final review gate

- The first attempt established exact artifact discovery/hashes and all three validation asset counts at `3589`, then failed at the config-identity gate. The second passed that identity gate and reached T4 memory-growth setup, then failed on the already-initialized runtime-policy mismatch. Neither produced a scientific outcome.
- The research-lead-reviewed subsequent output archive is `/kaggle/working/tf_step6_fixed_topology_prior_sensitivity_kaggle_t4.zip`, SHA-256 `7a3d04451c0204f6362f74976dd9650b15d535bc5f080cf4c27ea9b8794900b6`, size `744685` bytes, with exactly `9` compact files.
- The committed compact runtime report is byte-identical to the reviewed ZIP member and has SHA-256 `29beaede54dea719127d726b89257d85a2891d72bacb065152e2e44d7446adb6`.
- The reviewed run passed every preregistered C0 reproduction check on `3589` samples and produced the registered label `HIGH_EXPLICIT_PRIOR_SENSITIVITY`; exact metrics, paired diagnostics, integrity evidence, and interpretation boundaries are preserved in the committed compact report.
- C0 failure is locked to `INVALID_REFERENCE_REPRODUCTION`; in that case the adapter preserves raw files and suppresses C1/C2 derived reporting and scientific interpretation.
- C2 remains explicit semantic-prior/direct-part sensitivity conditional on the official MediaPipe-derived scaffold. It is not MediaPipe removal and is not a prior-free graph.
- Evidence closure used only the reviewed archive; Codex did not run Kaggle again or create a new experiment. PR #12 remains draft pending final research-lead review before merge.
