# TensorFlow Step 1 Runtime Evidence Recovery

## Outcome

`EVIDENCE_UNAVAILABLE`: no `COMPATIBLE_TF_REFERENCE_RUN` was found in the
allowed project-local search scope. Four directories contained both required
files, but each belongs to an older TensorFlow scientific-payload/checkpoint
policy revision. The merged learning-history audit was therefore not run.
Running it on these candidates as though they represented the current frozen
reference would produce evidence for the wrong implementation state.

No training, inference, model-weight loading, checkpoint conversion, cache or
prior regeneration, or test-artifact content access occurred.

## Recovery provenance

- Issue: `#3`, TensorFlow Research Step 2.
- Repository base commit: `2ee8c0d02d7d64ce682b6cfe1b94698c8c34c289`.
- Search date: `2026-08-22` (`Asia/Saigon`).
- Repository root was resolved with Git; all paths below are repository-relative.
- Audit tool: `standalone/lap_gnn_tensorflow_ofix7_mid_candidate/tools/audit_learning_history.py`.
- Audit version: `1.0.0`.
- Audit tool SHA-256: `96983871a5f4e3f23e341fda1a3fe84a9578b95a4215bae46b097eea7bc02ba6`.

The current resolved TensorFlow reference used for classification has:

- framework `tensorflow`, seed `42`, run name `ofix7_mid_optimized_seed42`;
- parameter count `1061192`;
- checkpoint monitor/model-selection metric `val_accuracy`;
- scheduler and early-stopping monitor `val_loss`;
- final-test checkpoint `best_val_accuracy`;
- graph signature `1c7597b170fd8604056ab7787fd2880d6e84f3025962fc4b6c8fb3e3faf8e1e8`;
- feature signature `752538062fa2e40d9615c650c529e9f4117f33a030b74d281b5b21fa573731fc`;
- prior signature `ea888bab9c003af9b279719025da7c39f90537179411326c2c3119fc8c3f0824`;
- dataset-split signature `fer2013_train28709_val3589_test3589`;
- scientific payload checksum `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.

## Search scope and method

The search followed Issue #3's order and remained inside project-related
locations:

1. Root output-like directories: `outputs/`, `artifacts/`, `outputs_image/`,
   and `outputs_log/`.
2. Project-local `standalone/`, `notebooks/`, and `kaggle/` trees, plus other
   repository-local locations excluding the output trees already searched.
3. Seven project-local ZIP archives whose names referenced TensorFlow, OFIX7,
   seed42, results, or outputs. ZIP central-directory entry names were checked;
   none contained the required history/config pair.
4. The immediate project parent directory. Its only entry was the `fer_d5`
   checkout, so no additional project-related output tree existed to search.

Discovery required a candidate directory to contain sibling `history.json` and
`resolved_config.json` files. Test artifact filenames were allowed in directory
inventory, but their contents were never opened. Candidate classification read
only the two required files and the non-test `provenance.json`.

The seven checked archives were under:

- `outputs/d16_analysis/lap_gnn_tensorflow_adamw_arithmetic_closure/`;
- `outputs/d16_analysis/lap_gnn_tensorflow_adamw_final_repair/`;
- `outputs/d16_analysis/lap_gnn_tensorflow_execution_mode_closure/`;
- `outputs/d16_analysis/lap_gnn_tensorflow_main_integration/`;
- `outputs/d16_analysis/lap_gnn_tensorflow_port/`;
- `outputs/d16_analysis/ofix7_mid_5seed_posttraining_analysis/`;
- `outputs/d16_analysis/ofix7_mid_replication_preflight/`.

## Candidate inventory and classification

No two candidates have the same history/config SHA-256 pair, so no exact-copy
deduplication was applicable.

### `cand-eb83876ab284-c9e3a1777b3d`

- Path: `outputs/_tmp_tf_logging_artifact_smoke`.
- Last modified: `2026-07-27T13:14:34.2450234+07:00`.
- History SHA-256: `eb83876ab28470ad3206f675c0493c9ed22523f8b6a7098619e1eccc6ae0b04a`.
- Resolved-config SHA-256: `c9e3a1777b3dea0b1a72c1e034468c981d6c7a8395f3ec5e54ecec753719ddb6`.
- Run: `ofix7_mid_optimized_seed42`; framework `tensorflow`; seed `42`;
  `1` observed epoch; `eval_train_metrics=true`.
- Completion: `TRAINING_COMPLETE.json` is present, but this is a bounded
  one-epoch logging smoke rather than a real learning trajectory.
- Policy: checkpoint/model-selection `val_macro_f1`; scheduler `val_loss`;
  early stopping `val_loss`; final-test checkpoint `best`; no nested
  checkpoint-policy monitor is represented.
- Locked evidence: parameter count and all four graph/feature/prior/split
  signatures match the current reference.
- Candidate package checksum: `22ba6a3136f0d2b0482bd0e78398e5adea542c35e18574f5c9a6c09d468fb724`.
- Classification: `TENSORFLOW_BUT_DIFFERENT_REFERENCE` because both the
  scientific payload checksum and checkpoint/model-selection policy differ
  from the current frozen reference. The one-epoch smoke is also insufficient
  as real learning-history evidence.

### `cand-a959df011579-3883794fff60`

- Path: `outputs/_tmp_tf_logging_artifact_smoke_v2`.
- Last modified: `2026-07-27T13:47:55.0885335+07:00`.
- History SHA-256: `a959df011579ef4d8d6657116bb9ae8e3ad2c80f145e60668b7b391f7ff4d64f`.
- Resolved-config SHA-256: `3883794fff6017bdde96a727bd3783bd3b38795f455dab4845244dc6b9ce70d1`.
- Run: `ofix7_mid_optimized_seed42`; framework `tensorflow`; seed `42`;
  `1` observed epoch; `eval_train_metrics=true`.
- Completion: `TRAINING_COMPLETE.json` is present, but this is a bounded
  one-epoch logging smoke rather than a real learning trajectory.
- Policy: checkpoint/model-selection `val_macro_f1`; scheduler `val_loss`;
  early stopping `val_loss`; final-test checkpoint `best`; no nested
  checkpoint-policy monitor is represented.
- Locked evidence: parameter count and all four graph/feature/prior/split
  signatures match the current reference.
- Candidate package checksum: `3d42957151037f9389d789c1e88c4003e16cc0e4b95944931cfae89733bb7d11`.
- Classification: `TENSORFLOW_BUT_DIFFERENT_REFERENCE` because both the
  scientific payload checksum and checkpoint/model-selection policy differ
  from the current frozen reference. The one-epoch smoke is also insufficient
  as real learning-history evidence.

### `cand-b0257db30c8a-0b7cd744ef9c`

- Path: `outputs/d16_runs/lap_gnn_tensorflow_ofix7_mid_candidate/ofix7_mid_seed42`.
- Last modified: `2026-07-29T18:10:40.0515927+07:00`.
- History SHA-256: `b0257db30c8a702751cb88cbf31448e6c268d36f1a220b45abad0d21ab045d63`.
- Resolved-config SHA-256: `0b7cd744ef9c2470e34ea077498663b9b936af6fc5c3cbec6aeec303a360a035`.
- Run: `ofix7_mid_optimized_seed42`; framework `tensorflow`; seed `42`;
  `32` observed epochs; `eval_train_metrics=true`.
- Completion: `TRAINING_COMPLETE.json` is present; this appears to be a
  completed real run for its historical TensorFlow revision.
- Policy: checkpoint/model-selection `val_macro_f1`; scheduler `val_loss`;
  early stopping `val_loss`; final-test checkpoint `best`; no nested
  checkpoint-policy monitor is represented.
- Locked evidence: parameter count and all four graph/feature/prior/split
  signatures match the current reference.
- Candidate package checksum: `3d42957151037f9389d789c1e88c4003e16cc0e4b95944931cfae89733bb7d11`.
- Classification: `TENSORFLOW_BUT_DIFFERENT_REFERENCE` because the scientific
  payload checksum and checkpoint/final-selection contract differ from the
  current frozen reference.

### `cand-f93a2bc6b313-6eadc32357db`

- Path: `outputs/d16_runs/lan2/lap_gnn_tensorflow_ofix7_mid_candidate/ofix7_mid_seed42`.
- Last modified: `2026-07-29T18:11:20.8435757+07:00`.
- History SHA-256: `f93a2bc6b313038e75d56bfbd7b24aa83ba436b0036cfa3837b74dcc0e964e97`.
- Resolved-config SHA-256: `6eadc32357db2094d61634904a119d67da79c7eda17a0c23f79c4fea33da2f2b`.
- Run: `ofix7_mid_optimized_seed42`; framework `tensorflow`; seed `42`;
  `32` observed epochs; `eval_train_metrics=true`.
- Completion: `TRAINING_COMPLETE.json` is present; this appears to be a
  completed real run for its historical TensorFlow revision.
- Policy: checkpoint/model-selection `val_macro_f1`; scheduler `val_loss`;
  early stopping `val_loss`; final-test checkpoint `best_val_accuracy`; no
  nested checkpoint-policy monitor is represented.
- Locked evidence: parameter count and all four graph/feature/prior/split
  signatures match the current reference.
- Candidate package checksum: `727d56f1e595e7846e83c787e9c11782e2c30fae512517761fc21a0714750fe0`.
- Classification: `TENSORFLOW_BUT_DIFFERENT_REFERENCE` because the scientific
  payload checksum and checkpoint/model-selection metric differ from the
  current frozen reference, even though its final-test checkpoint name matches.

## Audit disposition

Unique compatible candidates: `0`.

No invocation of `tools/audit_learning_history.py` was made, and no directory
under `outputs/tf_root_cause_audit/` was created. This is the fail-closed
Outcome B required by Issue #3. The existing candidate files and all historical
outputs remain in place and unchanged.

## Commands executed

The principal read-only discovery and verification commands were:

```powershell
git rev-parse --show-toplevel
git fetch origin --prune
rg --files -uu outputs artifacts outputs_image outputs_log -g 'history.json' -g 'resolved_config.json'
rg --files -uu standalone notebooks kaggle -g 'history.json' -g 'resolved_config.json'
rg --files -uu . -g 'history.json' -g 'resolved_config.json' -g '!outputs/**' -g '!artifacts/**' -g '!outputs_image/**' -g '!outputs_log/**' -g '!standalone/**' -g '!notebooks/**' -g '!kaggle/**' -g '!.git/**'
conda run -n lap-gnn-tf python -c "from lap_gnn_tf.config import load_config; print(load_config('configs/fer2013_ofix7_mid_tensorflow_optimized_seed42.yaml'))"
Get-FileHash -Algorithm SHA256 -LiteralPath 'outputs/_tmp_tf_logging_artifact_smoke/history.json','outputs/_tmp_tf_logging_artifact_smoke/resolved_config.json'
Get-FileHash -Algorithm SHA256 -LiteralPath 'outputs/_tmp_tf_logging_artifact_smoke_v2/history.json','outputs/_tmp_tf_logging_artifact_smoke_v2/resolved_config.json'
Get-FileHash -Algorithm SHA256 -LiteralPath 'outputs/d16_runs/lap_gnn_tensorflow_ofix7_mid_candidate/ofix7_mid_seed42/history.json','outputs/d16_runs/lap_gnn_tensorflow_ofix7_mid_candidate/ofix7_mid_seed42/resolved_config.json'
Get-FileHash -Algorithm SHA256 -LiteralPath 'outputs/d16_runs/lan2/lap_gnn_tensorflow_ofix7_mid_candidate/ofix7_mid_seed42/history.json','outputs/d16_runs/lan2/lap_gnn_tensorflow_ofix7_mid_candidate/ofix7_mid_seed42/resolved_config.json'
```

Candidate metadata was selected from parsed `history.json`,
`resolved_config.json`, and `provenance.json` objects. Archive checks used .NET
`System.IO.Compression.ZipFile.OpenRead` to enumerate entry names only. No test
metric, prediction, label, report, checkpoint, or dataset content was read.

## Scientific boundary and next decision

This recovery result does not establish a learning-behavior label, identify a
causal root cause, or authorize any model, architecture, data, prior, loss,
optimizer, scheduler, checkpoint, regularization, or evaluation change. It
only establishes that the local project evidence does not contain a run for
the current frozen TensorFlow reference.

Whether to register and execute a new controlled TensorFlow baseline run is a
research-lead decision outside this Issue.
