"""A4.2 Parameter Divergence Analysis between v2.1 and v2.2."""

from __future__ import annotations

import json
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a4"

V21_CKPT = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "best_val_acc.pt"
V22_CKPT = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "best_val_acc.pt"


def assign_module(name: str) -> str:
    if name.startswith("pixel_extractor"):
        return "pixel_extractor"
    if name.startswith("pixel_proj"):
        return "pixel projection"
    if name.startswith("pixel_topology"):
        return "pixel_topology"
    if name.startswith("pixel_gnn"):
        return "pixel GNN blocks"
    if name.startswith("pixel_attn_pool") or name.startswith("pixel_readout_proj") or name.startswith("aux_pixel_head"):
        return "pixel readout"
    if "scale_gate" in name:
        return "scale gate"
    if name.startswith("motif_composer"):
        return "motif composer"
    if name.startswith("motif_gnn.0"):
        return "motif block 1"
    if name.startswith("motif_gnn.1"):
        return "motif block 2"
    if name.startswith("motif_gnn.2"):
        return "motif block 3"
    if name.startswith("motif_gnn.3"):
        return "motif block 4"
    if name.startswith("motif_gnn.4"):
        return "motif block 5"
    if name.startswith("motif_attn_pool") or name.startswith("motif_readout_proj") or name.startswith("aux_motif_head"):
        return "motif readout"
    if name.startswith("classifier"):
        return "fusion/classifier"
    if name.startswith("supcon_head"):
        return "SupCon head"
    return "other"


def main():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "src"))
    from mpg_fer_v2_1.model import MPGFER
    from mpg_fer_v2_1.config import MPGConfig
    
    ref_model = MPGFER(MPGConfig())
    named_param_names = set(dict(ref_model.named_parameters()).keys())

    s21 = torch.load(V21_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    s22 = torch.load(V22_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]

    tensor_details = []
    module_tensors = {}

    for name in s21.keys():
        if name not in named_param_names:
            continue
        w21 = s21[name].float()
        w22 = s22[name].float()

        if not w21.is_floating_point() or w21.numel() == 0:
            continue

        mod = assign_module(name)
        f21 = w21.flatten()
        f22 = w22.flatten()

        norm21 = torch.norm(f21).item()
        norm22 = torch.norm(f22).item()
        diff_norm = torch.norm(f22 - f21).item()
        rel_diff = diff_norm / (norm21 + 1e-12)

        dot = torch.dot(f21, f22).item()
        denom = norm21 * norm22
        cos_sim = (dot / denom) if denom > 1e-12 else 1.0

        detail = {
            "tensor_name": name,
            "module": mod,
            "param_count": w21.numel(),
            "relative_l2_diff": float(rel_diff),
            "cosine_similarity": float(cos_sim),
            "norm_v21": float(norm21),
            "norm_v22": float(norm22),
            "l2_diff": float(diff_norm),
        }
        tensor_details.append(detail)
        module_tensors.setdefault(mod, []).append((f21, f22))

    # Aggregated module-level analysis
    module_summary = {}
    for mod, pairs in module_tensors.items():
        all_f21 = torch.cat([p[0] for p in pairs])
        all_f22 = torch.cat([p[1] for p in pairs])

        norm21 = torch.norm(all_f21).item()
        norm22 = torch.norm(all_f22).item()
        diff_norm = torch.norm(all_f22 - all_f21).item()
        rel_diff = diff_norm / (norm21 + 1e-12)

        dot = torch.dot(all_f21, all_f22).item()
        denom = norm21 * norm22
        cos_sim = (dot / denom) if denom > 1e-12 else 1.0

        module_summary[mod] = {
            "module": mod,
            "parameter_count": int(all_f21.numel()),
            "relative_l2_diff": float(rel_diff),
            "cosine_similarity": float(cos_sim),
            "l2_diff": float(diff_norm),
            "norm_v21": float(norm21),
            "norm_v22": float(norm22),
        }

    # Full network
    full_f21 = torch.cat([t["norm_v21"] * torch.zeros(0) for t in tensor_details])
    all_21 = torch.cat([p[0] for pairs in module_tensors.values() for p in pairs])
    all_22 = torch.cat([p[1] for pairs in module_tensors.values() for p in pairs])
    full_norm21 = torch.norm(all_21).item()
    full_diff = torch.norm(all_22 - all_21).item()
    full_cos = (torch.dot(all_21, all_22) / (full_norm21 * torch.norm(all_22))).item()

    full_summary = {
        "full_network": {
            "parameter_count": int(all_21.numel()),
            "relative_l2_diff": float(full_diff / full_norm21),
            "cosine_similarity": float(full_cos),
        },
        "modules": module_summary,
        "per_tensor_details": tensor_details,
    }

    out_path = AUDIT_DIR / "a4_parameter_divergence.json"
    out_path.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")

    # Print summary table sorted by relative L2 diff descending
    print("\n--- Major Modules Parameter Divergence (v2.2 vs v2.1) ---")
    print(f"{'Module':<22} | {'Params':>10} | {'Rel L2 Diff':>12} | {'Cosine Sim':>12}")
    print("-" * 64)
    target_modules = [
        "pixel projection", "pixel GNN blocks", "motif composer", "scale gate",
        "motif block 1", "motif block 2", "motif block 3", "motif block 4", "motif block 5",
        "motif readout", "fusion/classifier", "SupCon head",
    ]
    for mod in target_modules:
        if mod in module_summary:
            info = module_summary[mod]
            print(f"{mod:<22} | {info['parameter_count']:>10,d} | {info['relative_l2_diff']:>12.4f} | {info['cosine_similarity']:>12.4f}")

    print("-" * 64)
    print(f"{'TOTAL NETWORK':<22} | {full_summary['full_network']['parameter_count']:>10,d} | {full_summary['full_network']['relative_l2_diff']:>12.4f} | {full_summary['full_network']['cosine_similarity']:>12.4f}")


if __name__ == "__main__":
    main()
