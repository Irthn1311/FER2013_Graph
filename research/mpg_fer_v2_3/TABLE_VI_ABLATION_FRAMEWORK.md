# MPG-FER Table VI ablation framework

This directory contains the implementation-only deliverables for GitHub Issue
#101. It does not authorize or contain a Kaggle ablation run. `PrivateTest` is
not an input to the ablation API or notebook.

## Source boundary

The frozen v2.3 package remains in `src/mpg_fer_v2_3/`. The additive canonical
ablation package is `src/mpg_fer_table_vi/`; there are no per-variant source
forks. `AblationMPGFER(FULL)` directly executes the frozen `MPGFER.forward`, and
its state-dict key set is identical to the frozen model. The fixed-pooling mode
initializes the full/common module set first, removes the learned Composer, and
then installs only its 12x12 mean-pooling projection. This preserves shared
initialization while ensuring prototype machinery is absent from that variant.
The pixel-fusion ablation zeros only the classifier's 128-dimensional pixel
slice. Its SupCon head still receives the same full pixel-plus-motif source as
`FULL`.

The executable registry and its generated copy are:

- `src/mpg_fer_table_vi/model.py`
- `ablation_registry.json`

The registry order is the paper order and must never be performance-sorted.

## Training authorization

`run_ablation_training` has no PrivateTest argument. Before it resolves either
FER CSV, it requires:

1. an attached `FINAL_RECIPE_LOCK.json`;
2. an attached `ABLATION_DESIGN_LOCK.json` whose recipe SHA-256 matches;
3. `scientific_training_authorized=true` in that reviewed design lock;
4. exact equality between every non-runtime resolved config field and the
   complete recipe/base-config lock;
5. seed 42, `train.csv` for Train, and `val.csv` for PublicTest.

The current design lock intentionally has a null recipe SHA and
`scientific_training_authorized=false`. Therefore scientific training is
fail-closed until a later reviewed amendment. No variant-specific optimizer,
schedule, batch, augmentation, checkpoint selector, or loss reweighting is
available.

The future version-2 recipe schema separately binds the unchanged architecture
base commit, the complete frozen scientific/base config, every authorized
training-recipe field, an allowlisted optimizer family plus kwargs, an
allowlisted scheduler family plus kwargs, and the common EMA Public flip-TTA
checkpoint selector. Config construction starts from those complete sections;
missing, extra, unsupported, or subsequently mutated non-runtime values fail
closed. Runtime-safe differences are limited to the explicit resume-safe list
plus device selection.

## Canonical notebook

`notebooks/MPG_FER_Table_VI_Ablation_Kaggle_T4.ipynb` is the only ablation
notebook. It embeds and hashes both the frozen v2.3 source and the additive
Table VI package. Runtime selection is external to the immutable notebook via
Kaggle Secrets:

- `MPG_FER_ABLATION_MODE`
- `MPG_FER_ABLATION_RUN_ID`
- `MPG_FER_ABLATION_RESUME_MODE` (`fresh` or `required`)
- `MPG_FER_ABLATION_SEGMENT_NUMBER`

The notebook and design lock record two distinct commits: the frozen v2.3
architecture base and the reviewed ablation implementation commit. Model/run
source provenance uses the latter, never the architecture base.

Required attached offline inputs are the Train/Public CSV dataset containing
`train.csv` and `val.csv`, one `FINAL_RECIPE_LOCK.json`, and one
`ABLATION_DESIGN_LOCK.json`. A required resume additionally uses exactly one
dataset whose path contains `mpg-fer-table-vi-resume-` and whose
`resume_latest.pt` matches `resume_latest.json`. The notebook does not clone
source and does not require Internet for source acquisition. It writes under
`/kaggle/working/mpg-fer-table-vi/<RUN_ID>` and creates
`/kaggle/working/<RUN_ID>-artifacts.zip`.

Before resolving any mounted input, the notebook enumerates path names under
`/kaggle/input` and refuses the session if it sees a `test.csv` basename or a
`PrivateTest`, `private_test`, or `private-test` path component. The guard does
not open mounted files. Thus valid Train/Public inputs cannot coexist with a
mounted private marker.

## Eventual artifacts

Every completed future run must contain:

`config.json`, `ablation_manifest.json`, `history.json`, `history.csv`,
`best_val_acc.pt`, `execution_manifest.json`,
`canonical_public_metrics.json`, `segment_manifest.json`, `resume_latest.pt`,
and `checksums.sha256`.

`canonical_public_metrics.json` is produced by a raw single-view FP32 evaluator
with autocast and TF32 disabled. The Table VI aggregator accepts all eight
machine-readable run directories and writes JSON, CSV, and Markdown in the
fixed semantic order.

## Regeneration and verification

From this directory with the `fer-graph` interpreter and `PYTHONPATH=src`:

```powershell
python tools/freeze_ablation_framework.py --official-checkpoint <seed42-best_val_acc.pt>
python -m pytest -q
git diff --check
```

The freeze tool compiles every notebook code cell, verifies the official
seed-42 v2.3 checkpoint SHA and strict FULL load, runs the complete test suite,
and rewrites `preflight_report.json`. It never opens FER2013 and never launches
Kaggle.

T7 uses four synthetic samples as two physical microbatches of two samples,
with `gradient_accumulation_steps=2`. Each epoch therefore performs exactly one
optimizer/EMA/scaler update while exercising group-level consistency selection
for both FULL and SINGLE_SCALE_12 across continuous and exact-resume paths.
