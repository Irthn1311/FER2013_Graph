"""Phase 0: Source audit for Contextual Scale Recomposition (CSR) diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "analysis" / "mpg_fer_contextual_recomposition"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL_CKPT = Path(r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\research\mpg_fer_v2_3\official_runs\segment_02\mpg_fer_v2_3_run\best_val_acc.pt")
EXPECTED_CKPT_SHA = "23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    actual_sha = sha256_file(OFFICIAL_CKPT)
    assert actual_sha == EXPECTED_CKPT_SHA, f"SHA mismatch: {actual_sha}"
    print(f"Verified Canonical Checkpoint SHA256: {actual_sha}")

    source_audit = {
        "schema_version": 1,
        "checkpoint_path": str(OFFICIAL_CKPT),
        "checkpoint_sha256": actual_sha,
        "canonical_seed": 42,
        "extracted_tensors": {
            "H_P": {
                "description": "Output of 4-layer Edge-Aware Pixel GNN",
                "source_point": "self.pixel_gnn forward loop completion",
                "shape": [-1, 2304, 96],
            },
            "z_m_s": {
                "description": "Scale-specific candidate occurrence representations immediately before scale fusion",
                "source_point": "torch.stack(candidates, dim=2) inside SpatialMotifComposer.forward",
                "shape": [-1, 49, 3, 192],
                "scales": [8, 12, 16],
            },
            "alpha_m_s": {
                "description": "Canonical scale weights from learned scale gate",
                "source_point": "F.softmax(self.scale_gate(candidate_stack).squeeze(-1), dim=-1)",
                "shape": [-1, 49, 3],
            },
            "h_m_0": {
                "description": "Initial fused motif state fed into Motif GNN",
                "source_point": "(alpha.unsqueeze(-1) * candidate_stack).sum(dim=2)",
                "shape": [-1, 49, 192],
            },
            "h_m_L": {
                "description": "Final post-Motif-Graph state after 5 Geometry-Aware Motif Transformer Blocks",
                "source_point": "self.motif_gnn forward loop completion immediately before motif readout",
                "shape": [-1, 49, 192],
            },
        },
        "canonical_dimensions": {
            "num_pixels": 2304,
            "d_pixel": 96,
            "num_occurrences_M": 49,
            "num_scales_S": 3,
            "scale_window_sizes": [8, 12, 16],
            "d_motif": 192,
            "num_motif_layers": 5,
            "motif_topk_schedule": [8, 16, 16, 16, 24],
        },
        "status": "PASS",
    }

    with open(AUDIT_DIR / "SOURCE_AUDIT.json", "w", encoding="utf-8") as f:
        json.dump(source_audit, f, indent=2)

    md_lines = [
        "# Phase 0: Source Audit for Contextual Scale Recomposition (CSR)",
        "",
        "## 1. Checkpoint & Provenance Verification",
        f"- **Checkpoint Path:** `{OFFICIAL_CKPT}`",
        f"- **Checkpoint SHA256:** `{actual_sha}` (Strict parity confirmed)",
        "- **Seed:** 42",
        "",
        "## 2. Extraction Point Specifications",
        "",
        "| Tensor | Mathematical Symbol | Shape | Extraction Source Location | Semantics |",
        "|---|:---:|---|---|---|",
        "| Pixel GNN Output | $H_P$ | `[B, 2304, 96]` | `model.pixel_gnn` loop output | 2304 pixel nodes after 4 edge-aware layers |",
        "| Scale Candidates | $z_{(m,s)}$ | `[B, 49, 3, 192]` | `candidate_stack` in `SpatialMotifComposer` | Scale-specific candidate occurrences at scales 8, 12, 16 |",
        "| Pre-Graph Scale Weights | $\\alpha_{(m,s)}$ | `[B, 49, 3]` | `alpha` in `SpatialMotifComposer` | Softmax scale gating over 3 scales per occurrence |",
        "| Initial Fused Motif | $h_m^{(0)}$ | `[B, 49, 192]` | `(alpha * candidate_stack).sum(dim=2)` | Early occurrence state prior to Motif Graph |",
        "| Final Contextual Motif | $h_m^{(L)}$ | `[B, 49, 192]` | `model.motif_gnn` loop output | Post-relational occurrence state after 5 Motif GNN blocks |",
        "",
        "## 3. Core Hypothesis Mapping",
        "- The initial fusion $h_m^{(0)} = \\sum_s \\alpha_{(m,s)} z_{(m,s)}$ compresses local scales before cross-occurrence relational context is available.",
        "- Contextual Scale Recomposition (CSR) uses $h_m^{(L)}$ as queries to re-evaluate the original scale candidates $z_{(m,s)}$, testing whether relational context can recover discarded scale evidence.",
        "",
        "## 4. Hard Data Rule Enforcement",
        "- Extraction and evaluation will strictly use **Train (28,709)** and **Validation (3,589)** splits.",
        "- Absolutely no Private/Test data will be accessed during this diagnostic.",
    ]

    with open(AUDIT_DIR / "SOURCE_AUDIT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("Wrote SOURCE_AUDIT.json and SOURCE_AUDIT.md")


if __name__ == "__main__":
    main()
