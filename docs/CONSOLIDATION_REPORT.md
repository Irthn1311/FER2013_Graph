# Canonical repository consolidation record

This change deliberately replaces the tracked main tree with the closed MPG-FER v2.3 research state. It preserves old main and research history through verified archive refs and does not merge the stacked historical PRs. This is a reviewable integration/documentation change, not a new scientific experiment.

## Base and archives

| Ref | Verified commit |
|---|---|
| origin/main before consolidation | `48e5141fdff25ae6b67f2014407c701ec3166594` |
| archive/pre-final-main-2026-10-06 | `48e5141fdff25ae6b67f2014407c701ec3166594` |
| archive/mpg-fer-local-evidence-2026-10-06 | `e5b4ea293ad1af437204d8d58c47f4fcf099d2eb` |
| archive/mpg-fer-research-closure-2026-10-06 | `44daeefe3a1816220b2bb7149295ab85b77d7504` (unchanged) |

The local-evidence archive includes 243 payload files totaling 10,992,965 bytes, plus its manifest. Exact paths/sizes/SHA256 are in [archive_manifest.json](https://github.com/Irthn1311/FER2013_Graph/blob/archive/mpg-fer-local-evidence-2026-10-06/archive_manifest.json). It includes compact A0–A7/A2R/A6-R2 scientific evidence and producer source, not feature banks, checkpoints, raw Kaggle folders or cleanup noise. The archive has the closure commit as parent and an evidence-only current tree; it is not merged into main.

## Five reviewed files

| File | Resolution |
|---|---|
| notebooks/MPG_FER_final_one_shot_kaggle_T4.ipynb | Archived unchanged; architecture differs from final v2.3. |
| research/mpg_fer_v2_3/tools/run_v23_frozen_analysis.py | Archived unchanged; its A6/v2.1/v2.2 raw dependencies are outside canonical runtime. |
| scripts/build_handoff_package.py | Archived unchanged and retired from canonical main. |
| scripts/generate_handoff_files.py | Archived unchanged and retired from canonical main. |
| scripts/generate_master_summary_docx.py | Archived unchanged and retired from canonical main. |

## Deliberate canonical tree

```text
README.md / AGENTS.md / .gitignore / .gitattributes
requirements-canonical.txt
research/mpg_fer_v2_3/
  src/mpg_fer_v2_3/
  tests/ / tools/ / notebooks/
  v23_config.json / v23_provenance.json / compact final JSON evidence
  MPG_V23_MULTI_SEED_RESULTS/ (all 224 files)
research/mpg_fer_v2_2/src/mpg_fer_v2_2/ (14 frozen parity modules)
analysis/final_ablation/
analysis/final_research_closure/
analysis/IMPORT_MANIFEST.json
docs/FINAL_RESULTS.md / REPRODUCIBILITY.md / RESEARCH_HISTORY.md
docs/CONSOLIDATION_REPORT.md
```

The original scientific source/config and original tests/tools/notebook contents remain unchanged. Notebook container line endings match the protected original local files and are locked by `.gitattributes`. New code is limited to a standard-library compact-results verifier and two integrity tests. Final ablation and closure evidence identifies its source paths/hashes in `analysis/IMPORT_MANIFEST.json`. Composer documentation qualifies causal overstatement and removes a prospective redesign recommendation while preserving numeric evidence and the STOP verdict. Original summaries remain archived.

## Legacy removals from the cleanup branch

Old main had 1,489 tracked files. The transformation removes obsolete legacy implementation/runtime/documentation files from the current tree; existing root documentation/control paths are replaced with canonical versions. Every old file remains at the exact pre-final-main archive and through ordinary Git history. No history rewriting or filter-repo is used.

| Fully removed top-level directory | Old tracked files |
|---|---:|
| `configs/` | 434 |
| `d16/` | 90 |
| `d17/` | 16 |
| `d18/` | 38 |
| `d19/` | 16 |
| `data/` | 9 |
| `evaluation/` | 4 |
| `kaggle/` | 25 |
| `models/` | 25 |
| `notebooks/` | 16 |
| `outputs_log/` | 5 |
| `reports/` | 44 |
| `scratch/` | 23 |
| `scripts/` | 112 |
| `standalone/` | 408 |
| `tests/` | 40 |
| `tools/` | 9 |
| `training/` | 11 |
| `utils/` | 6 |
| `visualization/` | 2 |

| Removed historical research subtree | Old tracked files |
|---|---:|
| `research/candidates/` | 55 |
| `research/evidence/` | 33 |

Legacy root notebooks, D5/D16–D19 utilities, obsolete root handoff/design documents, output logs/images and ad hoc scripts are removed as part of that archived tree replacement. Historical v1/v2/v2.1/v2.2/A0–A7 and diagnostic runtime trees from research branches are not imported. The Table VI/cumulative execution wrappers are archive-only; accepted source-locked final summaries are retained. Prior main had no root LICENSE; no new license grant is invented.

## Retention and dependency decisions

- Keep all fourteen v2.3 scientific modules and original eleven test files, eight tools and two notebook templates: source/config, exact-resume and original generated-notebook contracts remain frozen.
- Keep fourteen v2.2 source modules: the v2.3 contract hashes the full package, compares nine non-target modules and imports v2.2 config/model for strict scale-1 equivalence. A synthetic replacement could remove source but would weaken that parity contract; no trivial equivalent fixture-only change was demonstrated. No v2.2 raw outputs/tools/notebooks are imported.
- Keep all 224 compact final package files, including frozen prediction/confusion CSVs and eighteen small reporting PNGs. These are explicitly retained scientific output artifacts; raw FER2013 images/label CSVs are absent.
- Keep concise accepted P0–P5 ablation metrics/checkpoint identities, FIXED_POOL forward/config forensic evidence, and five terminal MD/JSON summaries. Full runtime trees remain archived.
- Keep the original aggregate tool as optional external-archive replay, without importing its official ZIPs/checkpoints/feature cache. Clone-only verification uses the new verifier. The post-hoc frozen-analysis helper is archive-only.
- Checkpoint weights stay outside ordinary Git. Document SHA256 `23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e`; recommend separately reviewed Release/artifact storage. No Release upload was performed.

## Provenance semantics

All pre-existing JSON fields, including the meaningful uncommitted local provenance change and `official_training.performance_status=V23_NOT_SUPPORTED`, are retained exactly as parsed values. Four additive top-level fields distinguish completed execution from the unsupported mechanism:

```json
{
  "operational_status": "TRAINING_COMPLETED",
  "scientific_hypothesis_status": "V23_NOT_SUPPORTED",
  "scientific_hypothesis_scope": "Issue97 early-depth residual-preservation mechanism",
  "canonical_result_scope": "Issue99 six-seed canonical FP32 replication"
}
```

The original checkout remains on its research branch with its original provenance bytes; only the canonical worktree has the additive clarification. No architecture, loss, seed, split, schedule, selection comparator or evaluation protocol is changed.

## Validation actually performed

- Canonical source SHA256: `1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87` PASS; seed-42 scientific config SHA256 `8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2` PASS.
- Compact package: all 223 listed checksums PASS; all 224 original files preserved, and all six published registry/aggregate identities/statistics verified without checkpoint or FER2013 loading.
- `python -B -m pytest -q research/mpg_fer_v2_3/tests --basetemp .tmp/v23-final-tests -p no:cacheprovider`: **95 passed**, no failures/errors/skips, 252.39 seconds.
- Fresh-process strict identity/parity rerun of `test_v23_contract.py`, `test_multiseed_protocol.py`, `test_sparse_routing_v2_3.py`: **44 passed**, 27.25 seconds. This covers the restored original byte-exact tests and required frozen v2.2 parity.
- Both retained notebooks parse and compile all thirteen code cells; no saved cell outputs/execution counts. Their generators match the templates in the unit suite.
- Storage/secret/stale-reference scan: PASS, 49 original runtime/test/tool/template files byte-identical to protected local originals. Largest file 1,452,007 bytes; no added file over 10 MiB, no checkpoint/feature bank/raw dataset/credential file, no detected literal secret/private key, no stale current-model framing or new-experiment advice.
- Active code/docs have no absolute Windows paths. Five historical source-archive paths remain inside checksum-locked `archive_identity.json` records; preserving them is required by the original compact-package checksum contract and they are not active runtime dependencies.
- Validation environment: Python 3.11.15, Torch 2.11.0+cu126, Torchvision 0.26.0+cu126, NumPy 2.4.3, scikit-learn 1.8.0, pytest 9.1.1. This is not a new T4 training/evaluation measurement.

The preliminary over-broad closure-tree test run included archive-only Table VI wrappers and failed three missing historical lock/registry checks. Scope was corrected by retaining only the original six-seed runtime/test/tool/template set. Those wrappers and locks remain in the closure archive; no scientific source, fixture or parity assertion was weakened to obtain passing checks. Final full-suite and dedicated strict-parity results above are measured after scope correction.

## Review boundary

Open one PR from `cleanup/mpg-fer-final-canonical` to main. It supersedes the integration intent of stacked PRs #94/#96/#98/#100/#102/#104; none are merged, closed or deleted here. Refer to Issues #97/#99 for scientific identity without auto-closing them. The PR must not merge automatically. No old PR/Issue/remote branch is removed, no existing worktree is removed, and no aggressive GC is performed.

Original main and all research/archive refs remain. Scientific model selection/experimentation is closed, and implementation/storage checks do not establish a new scientific claim. All implementation/audit writes remain within the project boundary; only the explicitly supplied request attachment was read outside it.
