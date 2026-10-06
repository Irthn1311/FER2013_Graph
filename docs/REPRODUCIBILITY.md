# Reproducing the frozen implementation and evidence

## Identity and environment

Use Python 3.11 or 3.12. Install `requirements-canonical.txt` in an isolated environment, choosing the appropriate official PyTorch CPU/CUDA wheel for your platform. Official six-seed runs recorded Torch `2.10.0+cu128` on Tesla T4; this consolidation is locally validated with Python 3.11, Torch `2.11.0+cu126` and Torchvision `0.26.0+cu126`. These are distinct environments, not a claim that local tests reproduce T4 numerical/runtime measurements. Per-seed environment documents remain unchanged in the compact package.

```sh
python -m pip install -r requirements-canonical.txt
python research/mpg_fer_v2_3/tools/verify_canonical_results.py
```

The verifier reads retained evidence only. It verifies the source/config locks, all 223 compact checksums, six execution identities and published aggregate statistics. It does not train a model or evaluate FER2013.

| Lock | SHA256 |
|---|---|
| v2.3 scientific source | `1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87` |
| Seed-42 scientific config | `8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2` |
| Selected FULL seed-42 checkpoint | `23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e` |
| Frozen v2.2 parity source | `a8dc77db29e997c4c3ab69bb862704c8a948f940a4636e1c01e0d96bab40de65` |

Source hashes concatenate each sorted `*.py` basename and its original file bytes. Config hashes use sorted compact JSON of the scientific config, excluding only the registered runtime-safe fields. `.gitattributes` preserves source/evidence bytes across Windows and Linux checkouts.

## Training entrypoint and exact resume

The existing seed-42 notebook `research/mpg_fer_v2_3/notebooks/MPG_FER_v2_3_Kaggle_T4.ipynb` is the official preflight/training entrypoint. It embeds the locked package and requires an exact 40-hex Git commit identity. Stage it from a clean reviewed checkout; the following example creates local staging files only and does not submit a run:

```sh
python research/mpg_fer_v2_3/tools/stage_kaggle_kernel.py \
  --output-dir research/mpg_fer_v2_3/staged/seed42 \
  --kernel-ref YOUR_ACCOUNT/YOUR_KERNEL --git-commit EXACT_REVIEWED_40_HEX_COMMIT \
  --segment 1 --resume-mode fresh --dataset doduyquynii/fer13-split
```

The completed study is closed; actual submission/training requires separate authorization. Physical batch 16 and accumulation 2 remain frozen, with fail-closed OOM behavior. `run_training(train_csv, val_csv, output_dir, preflight_result, config)` rejects missing real-16 fresh preflight or incompatible resume state. It never opens PrivateTest. Do not call it with a fabricated gate.

For an authorized reproduction, upload the staged directory as a private Tesla T4 Kaggle notebook, attach the listed offline assets, and run all cells. The template performs the preflight, training or exact continuation, checkpoint freeze, gated evaluation and artifact export. There is no separate training CLI that silently bypasses these gates.

For infrastructure continuation, use the original run ID, source/config/seed/schedule and complete hash-verified resume bundle. Stage a later segment with `--resume-mode required`, and attach the explicit v2.3 resume dataset. The resolver accepts explicit paths or uniquely named v2.3 resume datasets; it will not silently fall back to a best checkpoint or old-version bundle.

The source-locked multi-seed template is `MPG_FER_v2_3_MultiSeed_Kaggle_T4.ipynb`. Its existing staging tool accepts an explicit `--git-branch`, exact `--git-commit`, seed, account alias and segment/resume mode; use the reviewed canonical branch/commit. It preserves the historical assignment `{A: [0,1], B: [43,123], C: [3047]}` and is intentionally not used to restage the already completed seed 42. No registered scientific input or account plan is changed here.

## Notebook inputs, Internet and outputs

Both retained templates use attached input `doduyquynii/fer13-split`. The source resolves complete `train.csv`/`val.csv`/`test.csv` triplets at these mounts, in order:

- `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`
- `/kaggle/input/fer13-split/fer13-split`
- `/kaggle/input/fer13-split`

The data gate preserves 28,709 Train, 3,589 PublicTest and 3,589 PrivateTest rows. No FER2013 CSV is distributed in Git. Resume runs additionally attach their exact registered bundle; mount/artifact/hash identity must be recorded in their execution manifest.

Staging metadata sets Internet **off**. Cloning needs no Internet because the scientific package is embedded in the notebook. Attached offline data/resume assets are a separate requirement. The optional Kaggle Secrets download fallback needs network access and authorized Secrets; it is not an offline substitute and no credentials are stored in this repository.

Seed-42 artifacts are zipped to `/kaggle/working/mpg_fer_v2_3_artifacts.zip`. Multi-seed artifacts use `/kaggle/working/mpg_v2_3_seed_{SEED}_artifacts.zip`. No Kaggle notebook was executed during consolidation; structural compilation is not runtime evidence.

## Evaluation and reporting

The notebook training path selects the frozen EMA checkpoint using registered PublicTest flip-TTA semantics. `evaluate_private_once(test_csv, output_dir, config)` refuses before `TRAINING_COMPLETED`, verifies the selected checkpoint SHA and prevents a repeated one-shot PrivateTest call. `evaluate_raw_and_tta` computes raw and horizontal-flip logit-averaged metrics in one physical loader traversal. The multi-seed notebook additionally reports canonical FP32 evaluation with autocast/TF32 disabled and keeps official operational metrics separately.

Seed-42 canonical FP32 post-hoc evidence is retained in the compact package; its historical extraction helper is archive-only. Use the recorded frozen metrics to verify published results. New inference on external weights/data is a separately authorized reproduction, not a side effect of package verification or unit testing.

Checkpoint weights are not committed. The selected checkpoint hash is above; a reviewed GitHub Release asset or other immutable artifact store is recommended separately. No Release upload is part of this consolidation. Other seed checkpoints are identified by the registry.

## Tests and the v2.2 reference

```sh
# POSIX, from repository root
PYTHONPATH=research/mpg_fer_v2_3/src python -m pytest -q research/mpg_fer_v2_3/tests
```

```powershell
# PowerShell, from repository root
$env:PYTHONPATH = (Resolve-Path 'research/mpg_fer_v2_3/src').Path
python -m pytest -q research/mpg_fer_v2_3/tests
```

`test_v23_contract.py` hashes all fourteen frozen v2.2 modules and compares nine non-target files byte-for-byte. `test_sparse_routing_v2_3.py` imports v2.2 config/model to test scale-1 block equivalence and routing behavior. A trivial synthetic replacement would weaken those scientific parity semantics. Only `research/mpg_fer_v2_2/src/mpg_fer_v2_2/` remains; its old outputs, notebooks and tools do not belong in canonical main. Existing tests generate synthetic samples/checkpoints inside temporary directories and never train on real FER2013.

`aggregate_multiseed_results.py` remains a source-stable archive replay utility. To replay it, provide the historical `official_runs/` seed/segment ZIPs, selected seed-42 checkpoint/sidecars and `analysis_cache/v23_frozen_features.npz` outside ordinary Git. A clean clone intentionally verifies the retained compact output instead of retaining runtime baggage to satisfy this tool. The closed `run_v23_frozen_analysis.py` additionally depends on A6/v2.1/v2.2 assets and is archive-only.
