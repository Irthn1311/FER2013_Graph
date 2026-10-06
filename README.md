# MPG-FER: Motif–Pixel–Graph Facial Expression Recognition

<p align="center">
  <strong>A graph-based facial expression recognition architecture for FER2013 that learns from pixels, composes spatial motif occurrences, and reasons over a geometry-aware motif graph.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.10--2.11-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-FER2013-6F42C1">
  <img alt="Model" src="https://img.shields.io/badge/Canonical-MPG--FER%20v2.3-2EA44F">
  <img alt="Status" src="https://img.shields.io/badge/Research%20cycle-closed-555555">
</p>

---

## Overview

**MPG-FER** is a single-model FER2013 system built around two graph levels:

1. a **Pixel Graph** over all 2,304 image pixels;
2. a **Motif Graph** over 49 learned spatial motif occurrences.

The canonical pipeline is:

```text
48×48 grayscale image
        │
        ▼
2,304 pixel nodes
        │
        ▼
4-layer Pixel GNN
        │
        ▼
Spatial Motif Composer
48 learned prototypes × multi-scale supports {8, 12, 16}
        │
        ▼
49 spatial occurrence nodes on a 7×7 grid
        │
        ▼
5-layer geometry-aware Motif Graph
        │
        ▼
Pixel readout + Motif readout
        │
        ▼
512 → 256 → 7 classifier
```

The final canonical implementation is **MPG-FER v2.3** with **2,304,528 trainable parameters**.

> **Important:** the **48 learned prototypes** are a prototype dictionary. They are not the same object as the **49 spatial occurrences** produced on the 7×7 support grid, and the occurrences are not assumed to be semantic facial landmarks.

---

## Key results

| Metric | Result |
|---|---:|
| Canonical seed-42 accuracy | **70.6604%** |
| Six-seed mean accuracy | **70.4607%** |
| Six-seed sample SD | **0.2634 pp** |
| Best registered seed | **70.7161%** |
| Trainable parameters | **2,304,528** |
| Registered seeds | `{0, 1, 42, 43, 123, 3047}` |

The six-seed replication uses the full registered seed set; no best-seed filtering is used.

### Six-seed results

| Seed | Accuracy (%) |
|---:|---:|
| 0 | **70.7161** |
| 1 | **70.2981** |
| 42 | **70.6604** |
| 43 | **70.6046** |
| 123 | **70.0195** |
| 3047 | **70.4653** |
| **Mean ± sample SD** | **70.4607 ± 0.2634** |

Seed 42 is retained as the canonical reference run. The complete compact six-seed evidence package is available at:

- [Multi-seed report](research/mpg_fer_v2_3/MPG_V23_MULTI_SEED_RESULTS/MULTI_SEED_REPORT.md)
- [Aggregate statistics](research/mpg_fer_v2_3/MPG_V23_MULTI_SEED_RESULTS/aggregate_statistics.json)
- [Seed registry](research/mpg_fer_v2_3/MPG_V23_MULTI_SEED_RESULTS/multi_seed_registry.csv)
- [Final results](docs/FINAL_RESULTS.md)

---

## Architecture

### 1. Pixel Graph

The input is a **48×48 grayscale image**, represented as **2,304 pixel nodes**.

| Component | Canonical setting |
|---|---:|
| Pixel nodes | 2,304 |
| Raw pixel descriptor | 32-D |
| Pixel embedding | 96-D |
| Pixel GNN layers | 4 |
| Pixel attention heads | 4 |
| Local connectivity | 8-neighbor |
| Pixel edge features | 5-D |
| Pixel readout | 128-D |

The Pixel Graph establishes local contextual representations before spatial motif composition.

### 2. Spatial Motif Composer

The Spatial Motif Composer converts contextualized pixel evidence into a compact set of spatial occurrences.

| Component | Canonical setting |
|---|---:|
| Learned prototypes | 48 |
| Spatial occurrences | 49 |
| Occurrence layout | 7×7 |
| Window scales | 8, 12, 16 |
| Stride | 6 |
| WHAT descriptor | 96-D |
| TYPE descriptor | 32-D |
| WHERE descriptor | 5-D |
| Raw occurrence descriptor | 133-D |
| Motif embedding | 192-D |

The Composer learns **what** is present, **which prototype structure** is active, and **where** the occurrence is located.

### 3. Geometry-aware Motif Graph

The 49 occurrence nodes are refined through a five-layer geometry-aware graph stack.

| Component | Canonical setting |
|---|---:|
| Motif layers | 5 |
| Motif embedding | 192-D |
| Attention heads | 6 |
| Geometry descriptor | 6-D |
| Dynamic Top-K schedule | `[8, 16, 16, 16, 24]` |
| Residual scale schedule | `[0.5, 0.5, 1.0, 1.0, 1.0]` |

Dynamic Top-K restricts which relations are used for aggregation. The implementation still computes dense relation scores before selection, so **no subquadratic-attention claim is made**.

### 4. Readout and classifier

The final prediction combines information from both graph levels.

```text
Pixel branch → 128-D
Motif branch → mean + max + learned attention → 384-D

128 + 384 = 512
        │
        ▼
      256
        │
        ▼
   7 expressions
```

The seven FER2013 classes are:

`Angry · Disgust · Fear · Happy · Sad · Surprise · Neutral`

---

## Accepted cumulative ablation

The final paper-facing ladder is a source-locked selection of completed experiments.

| Stage | Configuration | Accuracy (%) | Gain |
|---|---|---:|---:|
| P0 | Pixel Graph baseline | 62.50 | — |
| P1 | + Spatial Motif Composer | 63.92 | +1.42 |
| P2 | + Geometry-aware Motif Graph | 65.31 | +1.39 |
| P3 | + Multi-scale Composition | 65.73 | +0.42 |
| P4 | + Learnable Motif Attention Readout | 70.33 | +4.60 |
| P5 | **Full MPG-FER** + Dynamic Top-K | **70.66** | +0.33 |

See [analysis/final_ablation](analysis/final_ablation/README.md) for the accepted ladder, source mapping, checkpoint identities, and the resolution of the historical `FIXED_POOL` naming discrepancy.

---

## Repository structure

```text
FER2013_Graph/
├── README.md
├── requirements-canonical.txt
├── docs/
│   ├── FINAL_RESULTS.md
│   ├── REPRODUCIBILITY.md
│   ├── RESEARCH_HISTORY.md
│   └── CONSOLIDATION_REPORT.md
│
├── analysis/
│   ├── final_ablation/
│   └── final_research_closure/
│
└── research/
    ├── mpg_fer_v2_3/
    │   ├── src/mpg_fer_v2_3/          # canonical implementation
    │   ├── tests/                     # frozen unit / parity / resume tests
    │   ├── tools/                     # staging, verification, aggregation
    │   ├── notebooks/                 # source-locked Kaggle templates
    │   ├── MPG_V23_MULTI_SEED_RESULTS/
    │   ├── v23_config.json
    │   └── v23_provenance.json
    │
    └── mpg_fer_v2_2/
        └── src/mpg_fer_v2_2/          # minimal frozen parity reference
```

The canonical `main` branch intentionally excludes obsolete runtime outputs, historical experiment trees, checkpoints, and FER2013 CSV files. Historical research is preserved in Git archive refs rather than mixed into the active project tree.

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/Irthn1311/FER2013_Graph.git
cd FER2013_Graph
```

### 2. Create an environment

Python **3.11 or 3.12** is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the canonical dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-canonical.txt
```

Choose the appropriate official PyTorch CPU/CUDA wheel for your platform when necessary.

### 3. Verify the frozen evidence

```bash
python research/mpg_fer_v2_3/tools/verify_canonical_results.py
```

A valid canonical checkout verifies:

- the frozen v2.3 source hash;
- the scientific config hash;
- **223/223 compact evidence checksums**;
- all six registered execution identities;
- the published aggregate statistics.

This verifier does **not** train the network and does **not** evaluate FER2013.

---

## FER2013 data

The dataset itself is **not distributed in this repository**.

The registered split contains:

| Split | Samples |
|---|---:|
| Train | 28,709 |
| PublicTest / validation | 3,589 |
| PrivateTest / test | 3,589 |

For local work, provide the FER2013 CSV files through your own authorized dataset source.

The retained Kaggle templates resolve the registered `doduyquynii/fer13-split` dataset layout documented in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## Training from scratch

The canonical seed-42 training entrypoint is:

`research/mpg_fer_v2_3/notebooks/MPG_FER_v2_3_Kaggle_T4.ipynb`

The notebook is source-locked and includes the project preflight, training, checkpoint selection, evaluation gates, and artifact export.

To stage a **fresh** run:

```bash
python research/mpg_fer_v2_3/tools/stage_kaggle_kernel.py \
  --output-dir research/mpg_fer_v2_3/staged/seed42 \
  --kernel-ref YOUR_ACCOUNT/YOUR_KERNEL \
  --git-commit EXACT_REVIEWED_40_HEX_COMMIT \
  --segment 1 \
  --resume-mode fresh \
  --dataset doduyquynii/fer13-split
```

The canonical scientific training configuration includes:

<details>
<summary><strong>Show frozen training configuration</strong></summary>

| Setting | Value |
|---|---:|
| Seed | 42 |
| Batch size | 16 |
| Gradient accumulation | 2 |
| Learning rate | 3e-4 |
| Weight decay | 5e-4 |
| Max epochs | 120 |
| Minimum epochs | 50 |
| Warmup epochs | 5 |
| LR decay end | 85 |
| Minimum LR | 1e-6 |
| EMA decay | 0.999 |
| Gradient clip | 1.0 |
| Label smoothing | 0.05 |
| AMP during training | enabled |

The complete frozen configuration is stored in [v23_config.json](research/mpg_fer_v2_3/v23_config.json).

</details>

For exact operational details, resume behavior, data gates, and Kaggle staging, read [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## Testing

### POSIX

```bash
PYTHONPATH=research/mpg_fer_v2_3/src:research/mpg_fer_v2_2/src \
python -m pytest -q research/mpg_fer_v2_3/tests
```

### PowerShell

```powershell
$env:PYTHONPATH = "$(Resolve-Path 'research/mpg_fer_v2_3/src');$(Resolve-Path 'research/mpg_fer_v2_2/src')"
python -m pytest -q research/mpg_fer_v2_3/tests
```

The canonical consolidation was validated with **95 passing v2.3 tests**, including strict identity/parity coverage against the minimal frozen v2.2 reference.

---

## Checkpoint and integrity locks

Model weights are intentionally not committed to ordinary Git history.

| Artifact | SHA256 |
|---|---|
| v2.3 scientific source | `1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87` |
| Seed-42 scientific config | `8f14b91e95663833248fd8cd40bb1b63234dea58cc4bc554e96710d822fb64c2` |
| Canonical seed-42 checkpoint | `23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e` |
| Frozen v2.2 parity source | `a8dc77db29e997c4c3ab69bb862704c8a948f940a4636e1c01e0d96bab40de65` |

The selected seed-42 checkpoint contains **EMA weights from epoch 57**.

If you retain or distribute a checkpoint externally, verify its SHA256 before use.

---

## Research closure

The final incremental research cycle is closed.

The terminal diagnostic sequence was:

| Diagnostic | Final decision |
|---|---|
| Motif Readout | `READOUT_DIAGNOSTIC_NO_GO` |
| Contextual Scale Recomposition | `CSR_DIAGNOSTIC_NO_GO` |
| Pixel Reinspection | `PIXEL_REINSPECTION_NO_GO` |
| Upstream Representation / Optimization | `UPSTREAM_AUDIT_STOP_INCREMENTAL` |
| Spatial Motif Composer | `COMPOSER_AUDIT_STOP` |

These results support the decision to stop incremental local redesign of MPG-FER v2.3 under the tested protocols. They **do not** establish that 70.66% is an absolute performance ceiling, nor do they prove universal causal absence of other possible bottlenecks.

The compact terminal evidence is preserved at [analysis/final_research_closure](analysis/final_research_closure/README.md).

---

## Research history

This repository represents the end of a long iterative research cycle that included multiple graph formulations, TensorFlow and parity investigations, pixel-relational candidates, successive MPG-FER versions, ablations, and targeted diagnostics.

The canonical `main` branch now contains only the final implementation and evidence needed to understand and reproduce MPG-FER v2.3.

Historical material remains preserved through the repository's `archive/*` refs, including:

- the pre-consolidation main snapshot;
- compact local scientific evidence;
- the terminal research-closure chain;
- legacy branch-tip reachability;
- local-only branch-tip reachability.

See [docs/RESEARCH_HISTORY.md](docs/RESEARCH_HISTORY.md) for the research timeline and source-locked references.

---

## Reproducibility notes

- The canonical result is from **one MPG-FER model/checkpoint**, not an ensemble.
- The architecture is trained end-to-end from scratch for fresh runs.
- FER2013 PrivateTest is used only after the selected checkpoint is frozen in the registered workflow.
- Dynamic Top-K changes relation selection, but dense relation scores are computed before selection.
- Diagnostic associations are interpreted descriptively; they are not presented as causal proof.
- The 49 motif occurrences are spatial computational units, not manually assigned facial anatomy.
- The 48 prototypes are learned dictionary entries and are distinct from occurrence nodes.

---

## Documentation

| Document | Purpose |
|---|---|
| [Final results](docs/FINAL_RESULTS.md) | Frozen per-seed and aggregate results |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Environment, data, training, resume, evaluation, tests |
| [Research history](docs/RESEARCH_HISTORY.md) | Eight-month development and archive map |
| [Final ablation](analysis/final_ablation/README.md) | Accepted P0–P5 cumulative ladder |
| [Research closure](analysis/final_research_closure/README.md) | Final diagnostic evidence and stopping decisions |
| [Canonical package](research/mpg_fer_v2_3/README.md) | v2.3 source/evidence package overview |

---

## Limitations

- FER2013 is a low-resolution, grayscale benchmark; the results here do not by themselves establish cross-dataset generalization.
- This repository preserves the canonical FER2013 research configuration rather than presenting a production face-analysis system.
- The final diagnostics close the tested incremental redesign families; they do not rule out fundamentally different future formulations.
- Checkpoint weights and dataset files are external artifacts and are identified by hashes rather than committed to Git.

---

## Project status

**Canonical version:** MPG-FER v2.3  
**Repository state:** frozen canonical research artifact  
**Incremental architecture-search cycle:** closed

Future work, if resumed, should be treated as a **new research cycle** rather than an untracked modification of the v2.3 canonical result.

---

## Repository maintainer

This canonical MPG-FER repository and its current `main` branch are maintained by **[@Irthn1311](https://github.com/Irthn1311)**.

---

## Acknowledgements

This project is built with [PyTorch](https://pytorch.org/) and evaluates on the FER2013 facial-expression benchmark. Official experimental runs were executed on Kaggle Tesla T4 infrastructure.

If you use this repository for research, please link to the repository and pin the exact commit and artifact hashes used in your experiment.
