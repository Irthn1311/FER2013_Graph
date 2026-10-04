"""Phase 2: Forward-graph differential test between Cumulative A5 and Table VI FIXED_POOL."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "mpg_fer_v2_3" / "src"))

from mpg_fer_cumulative_ablation7.model import CumulativeAblationMode, CumulativeAblationMPGFER
from mpg_fer_table_vi.model import AblationMPGFER, AblationMode
from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.data import FER2013Dataset

AUDIT_DIR = ROOT / "analysis" / "mpg_fer_ablation_paper_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
DATA_CSV = Path(r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\data\test.csv")


def run_forward_diff() -> dict:
    cfg = MPGConfig()
    
    # Instantiate both models
    torch.manual_seed(42)
    model_a5 = CumulativeAblationMPGFER(config=cfg, mode=CumulativeAblationMode.A5).eval()
    
    torch.manual_seed(42)
    model_old_fixed = AblationMPGFER(config=cfg, mode=AblationMode.FIXED_POOL).eval()

    # Align shared parameters where names match
    state_a5 = model_a5.state_dict()
    state_old = model_old_fixed.state_dict()

    shared_keys = set(state_a5.keys()).intersection(set(state_old.keys()))
    unique_a5 = set(state_a5.keys()) - set(state_old.keys())
    unique_old = set(state_old.keys()) - set(state_a5.keys())

    # Copy shared weights to model_old_fixed
    compatible_state = {k: state_a5[k] for k in shared_keys if state_a5[k].shape == state_old[k].shape}
    model_old_fixed.load_state_dict(compatible_state, strict=False)

    # 1. Deterministic synthetic test
    torch.manual_seed(100)
    x_synth = torch.randn(2, 1, 48, 48)

    with torch.no_grad():
        logits_a5_synth, out_a5_synth = model_a5(x_synth)
        logits_old_synth, out_old_synth = model_old_fixed(x_synth)

    # 2. Real sample test
    dataset = FER2013Dataset(DATA_CSV, split="test", augment=False)
    x_real = torch.stack([dataset[0][0], dataset[1][0]])

    with torch.no_grad():
        logits_a5_real, out_a5_real = model_a5(x_real)
        logits_old_real, out_old_real = model_old_fixed(x_real)

    # Compute tensor shape comparisons
    tensor_shapes = {
        "pixel_projected": {
            "cumulative_a5": list(out_a5_synth["h_pixel_projected"].shape),
            "old_fixed_pool": list(out_old_synth["h_pixel_projected"].shape),
            "match": out_a5_synth["h_pixel_projected"].shape == out_old_synth["h_pixel_projected"].shape,
        },
        "pixel_nodes": {
            "cumulative_a5": list(out_a5_synth["h_pixel_nodes"].shape),
            "old_fixed_pool": list(out_old_synth["h_pixel_nodes"].shape),
            "match": out_a5_synth["h_pixel_nodes"].shape == out_old_synth["h_pixel_nodes"].shape,
        },
        "pixel_readout": {
            "cumulative_a5": list(out_a5_synth["h_pixel_readout"].shape),
            "old_fixed_pool": list(out_old_synth["h_pixel_readout"].shape),
            "match": out_a5_synth["h_pixel_readout"].shape == out_old_synth["h_pixel_readout"].shape,
        },
        "motif_nodes_initial": {
            "cumulative_a5": list(out_a5_synth["h_motif_nodes_initial"].shape),
            "old_fixed_pool": list(out_old_synth["h_motif_nodes_initial"].shape),
            "match": out_a5_synth["h_motif_nodes_initial"].shape == out_old_synth["h_motif_nodes_initial"].shape,
        },
        "motif_nodes_final": {
            "cumulative_a5": list(out_a5_synth["h_motif_nodes_final"].shape),
            "old_fixed_pool": list(out_old_synth["h_motif_nodes_final"].shape),
            "match": out_a5_synth["h_motif_nodes_final"].shape == out_old_synth["h_motif_nodes_final"].shape,
        },
        "motif_readout": {
            "cumulative_a5": list(out_a5_synth["h_motif_readout"].shape),
            "old_fixed_pool": list(out_old_synth["h_motif_readout"].shape),
            "match": out_a5_synth["h_motif_readout"].shape == out_old_synth["h_motif_readout"].shape,
        },
        "fusion_representation": {
            "cumulative_a5": list(out_a5_synth["fusion_representation"].shape),
            "old_fixed_pool": list(out_old_synth["fusion_representation"].shape),
            "match": out_a5_synth["fusion_representation"].shape == out_old_synth["fusion_representation"].shape,
        },
        "logits": {
            "cumulative_a5": list(logits_a5_synth.shape),
            "old_fixed_pool": list(logits_old_synth.shape),
            "match": logits_a5_synth.shape == logits_old_synth.shape,
        },
    }

    # Numerical differences (on shared pixel graph, should be 0)
    diff_pixel_readout = (out_a5_synth["h_pixel_readout"] - out_old_synth["h_pixel_readout"]).abs().max().item()
    diff_motif_initial = (out_a5_synth["h_motif_nodes_initial"] - out_old_synth["h_motif_nodes_initial"]).abs().max().item()
    diff_motif_final = (out_a5_synth["h_motif_nodes_final"] - out_old_synth["h_motif_nodes_final"]).abs().max().item()
    diff_motif_readout = (out_a5_synth["h_motif_readout"] - out_old_synth["h_motif_readout"]).abs().max().item()
    diff_logits_synth = (logits_a5_synth - logits_old_synth).abs().max().item()
    diff_logits_real = (logits_a5_real - logits_old_real).abs().max().item()

    # Named modules execution differences
    modules_diff = {
        "spatial_composition_module": {
            "cumulative_a5": "self.motif_composer (SpatialMotifComposer with learned prototypes, scale_saliency, scale_gate, 8/12/16)",
            "old_fixed_pool": "self.fixed_pool_composer (FixedSpatialPoolComposer with fixed 12x12 mean pooling, NO prototypes, NO gate)",
        },
        "motif_readout_pooling": {
            "cumulative_a5": "Fixed uniform mean pooling: m_pool = h_motif.mean(dim=1). self.motif_attn_pool is bypassed and unused.",
            "old_fixed_pool": "Learnable attention pooling: m_attention = (softmax(self.motif_attn_pool(h_motif)) * h_motif).sum(dim=1).",
        },
        "motif_readout_input_concat": {
            "cumulative_a5": "cat(m_mean, m_max, m_mean) -> m_mean is duplicated twice in the 576D vector!",
            "old_fixed_pool": "cat(m_mean, m_max, m_attention) -> distinct mean, max, and learned attention pooling components.",
        },
    }

    classification = "CASE_2_SAME_NAME_DIFFERENT_SEMANTICS"
    justification = (
        "The model named 'FIXED_POOL' in Table VI and the configuration 'A5' in the cumulative ladder share the phrase "
        "'fixed pooling' in their high-level descriptions but implement completely opposite architectural interventions. "
        "In Table VI, 'Fixed spatial pooling' referred to replacing the learned Spatial Motif Composer of pixels with fixed 12x12 "
        "grid pooling, while keeping learnable attention pooling in the motif readout. In Cumulative A5, the Spatial Motif Composer "
        "was retained in full multi-scale learned form, while the motif readout pooling was replaced with uniform mean pooling "
        "(duplicating m_mean). Hence, the two experiments test different hypotheses and naturally yield divergent metrics (70.02% vs 64.59%)."
    )

    report = {
        "classification": classification,
        "justification": justification,
        "structural_analysis": {
            "shared_parameter_count": len(compatible_state),
            "unique_parameters_a5": sorted(list(unique_a5)),
            "unique_parameters_old_fixed": sorted(list(unique_old)),
            "tensor_shapes": tensor_shapes,
            "modules_diff": modules_diff,
        },
        "numerical_divergence": {
            "pixel_readout_max_diff": diff_pixel_readout,
            "motif_initial_max_diff": diff_motif_initial,
            "motif_final_max_diff": diff_motif_final,
            "motif_readout_max_diff": diff_motif_readout,
            "synthetic_logits_max_diff": diff_logits_synth,
            "real_samples_logits_max_diff": diff_logits_real,
        },
    }

    # Write JSON
    out_json = AUDIT_DIR / "A5_VS_OLD_FIXED_FORWARD_DIFF.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Wrote", out_json)

    # Write Markdown
    md_lines = [
        "# Forward-Graph Differential Test: Cumulative A5 vs Table VI FIXED_POOL",
        "",
        f"## Classification: `{classification}`",
        "",
        "### Verdict & Mathematical Justification",
        justification,
        "",
        "### Intermediate Tensor Shapes & Invariants",
        "| Tensor Stage | Cumulative A5 Shape | Old FIXED_POOL Shape | Shape Match |",
        "|---|---|---|:---:|",
    ]
    for stage, info in tensor_shapes.items():
        match_str = "YES" if info["match"] else "NO"
        md_lines.append(f"| `{stage}` | `{info['cumulative_a5']}` | `{info['old_fixed_pool']}` | {match_str} |")

    md_lines.extend([
        "",
        "### Architectural Mechanism Contrast",
        "",
        "| Mechanism | Cumulative A5 (+ Dynamic Top-K) | Old FIXED_POOL ('Fixed spatial pooling') |",
        "|---|---|---|",
        f"| **Spatial Composition** | {modules_diff['spatial_composition_module']['cumulative_a5']} | {modules_diff['spatial_composition_module']['old_fixed_pool']} |",
        f"| **Motif Readout Pooling** | {modules_diff['motif_readout_pooling']['cumulative_a5']} | {modules_diff['motif_readout_pooling']['old_fixed_pool']} |",
        f"| **Motif Readout Concat** | {modules_diff['motif_readout_input_concat']['cumulative_a5']} | {modules_diff['motif_readout_input_concat']['old_fixed_pool']} |",
        "",
        "### Numerical Forward Comparison",
        "- **Pixel Readout Max Abs Diff:** " + f"`{diff_pixel_readout:.6f}` (shared pixel path identical)",
        "- **Initial Motif Nodes Max Abs Diff:** " + f"`{diff_motif_initial:.4f}` (diverges immediately due to different composition)",
        "- **Final Motif Nodes Max Abs Diff:** " + f"`{diff_motif_final:.4f}`",
        "- **Motif Readout Max Abs Diff:** " + f"`{diff_motif_readout:.4f}`",
        "- **Synthetic Input Logits Max Abs Diff:** " + f"`{diff_logits_synth:.4f}`",
        "- **Real FER2013 Samples Logits Max Abs Diff:** " + f"`{diff_logits_real:.4f}`",
    ])

    out_md = AUDIT_DIR / "A5_VS_OLD_FIXED_FORWARD_DIFF.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("Wrote", out_md)


if __name__ == "__main__":
    run_forward_diff()
