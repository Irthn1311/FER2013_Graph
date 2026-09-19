# Issue #86 Execution Amendment E2 Governance

## Objective & Classification

Execution Amendment E2 is a **wrapper-only amendment** addressing a pre-execution routing defect in the E1 generated `FINALIZE_TRAIN` script.

Classification: **`TECHNICAL WRAPPER AMENDMENT — SCIENTIFIC SOURCE IMMUTABLE`**.

---

## Provenance & Locks

- **Preregistration**: `6e6ba806cc9ff0bc2b8a5534807d55f7feb00d73`
- **Scientific implementation**: `16a84b2b36f0d3584afd0a547487c4373dc22128`
- **E1 wrapper commit**: `c23319fccebd4c73fa6df6d998ad04cbf75bf7d0`
- **Canonical substrate identity**: `ee5d8262f7ec439bd6e2cd8ad01a17df17cd152da10b474c49ea93872fbfc46c`
- **Accepted CRS Train model SHA256**: `77b8a41d4a7de79b2216b4b9c7ad2d19e326cac25a46ec2a7a3dcaaa1f95607a`
- **40 Replicates status**: 40/40 completed, verified, and preserved in local canonical archive.

---

## Defect Addressed

In `kaggle/motif_issue86/generate_execution_units.py` and the resulting unit scripts, `_resolve_replicates_dir` probed:
`candidate.glob("motif_stability_replicate_*.npz")`

However, the scientific loader at `16a84b2b36f0d3584afd0a547487c4373dc22128` explicitly expects the registered filename pattern:
`motif_stability_[MC]_rep_[0-9][0-9].npz`
together with matching `.manifest.json` sidecars for all 40 units (`M00..M19`, `C00..C19`).

---

## E2 Resolution

Updated `_resolve_replicates_dir` in `generate_execution_units.py` to be fail-closed:
1. Constructs the exact set of 40 required filenames: `motif_stability_{arm}_rep_{replicate_id:02d}.npz` for `arm in ("M", "C")` and `replicate_id in range(20)`.
2. Verifies that candidate directory contains exactly this set (no missing, no extra) and that every `.npz` has a corresponding `.manifest.json` sidecar.
3. If no candidate directory satisfies this exact set, searches `/kaggle/input` for `**/motif_stability_M_rep_00.npz` and checks its parent.
4. If no valid directory is found, raises an explicit `FileNotFoundError`.

Regenerated all 42 execution units and `execution_units_manifest.json` deterministically.
Added unit regression tests in `research/pixel_relational_motif_e0/tests/test_motif_train_e2_resolver.py`.
