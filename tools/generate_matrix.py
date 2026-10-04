"""Generate cumulative ablation matrix and pairwise transition validation."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "mpg_fer_v2_3" / "src"))

from mpg_fer_cumulative_ablation7.protocol import (
    CUMULATIVE_ABLATION_ORDER,
    CUMULATIVE_REGISTRY,
    build_cumulative_ablation_matrix,
)

def main() -> None:
    matrix = build_cumulative_ablation_matrix()
    out_dir = ROOT / "analysis" / "mpg_fer_cumulative_ablation7"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "CUMULATIVE_ABLATION_MATRIX.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)
    print("Wrote", json_path)

    md_lines = [
        "# MPG-FER Cumulative Ablation Matrix",
        "",
        "Strict Nested Subset Ladder: **A0 ⊂ A1 ⊂ A2 ⊂ A3 ⊂ A4 ⊂ A5 ⊂ A6**",
        "",
        "| Config | Display Name | Pixel GNN | Composer | Scales | Motif GNN | Geom Bias | Relation Mode | Top-K Schedule | Motif Pooling | Direct Pixel Fusion | Classifier |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in matrix["rows"]:
        scales_str = str(r["composer_scales"]) if r["composer_scales"] else "—"
        sched_str = str(r["motif_topk_schedule"]) if r["motif_topk_schedule"] else "—"
        px_gnn = "Yes" if r["pixel_gnn"] else "No"
        m_gnn = "Yes" if r["motif_gnn"] else "No"
        geom = "Yes" if r["geometry_bias"] else "No"
        fusion = "Yes" if r["direct_pixel_fusion"] else "No"
        md_lines.append(
            f"| {r['configuration']} | {r['display_name']} | {px_gnn} | {r['composer']} | {scales_str} | {m_gnn} | {geom} | {r['relation_mode']} | {sched_str} | {r['motif_pooling']} | {fusion} | {r['classifier']} |"
        )

    md_lines.extend([
        "",
        "## Pairwise Transition Validation",
        "",
        "| Transition | From | To | Intended Addition | Verified Fields | Status |",
        "|---|---|---|---|---|:---:|",
    ])
    for t in matrix["pairwise_transitions"]:
        f_str = ", ".join(t["changed_fields"].keys())
        md_lines.append(
            f"| {t['transition']} | {t['from_config']} | {t['to_config']} | {t['intended_change']} | `{f_str}` | {t['status']} |"
        )

    md_lines.extend([
        "",
        "## Loss Term Preservation & Schedule",
        "",
        "All configurations share the identical scientific recipe:",
        "- Seed: 42",
        "- Optimizer: AdamW (lr=3e-4, weight_decay=5e-4)",
        "- Batch size: 16 (gradient_accumulation=2)",
        "- Scheduler: Linear warmup 5 epochs, cosine decay through epoch 85",
        "- Early stopping: monitoring begins at epoch 85, patience 15",
        "- EMA: 0.999",
        "- Loss coefficients: final_ce=1.0, pixel_aux=0.05, consistency=0.15, supcon=0.05 across all configurations.",
        "- Motif auxiliary losses (motif_aux=0.2, lambda_mi=0.025, lambda_div=0.01) are disabled ONLY for A0 where no motif representation exists mathematically, and enabled for all subsequent configurations A1-A6.",
    ])

    md_path = out_dir / "CUMULATIVE_ABLATION_MATRIX.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("Wrote", md_path)

    # Also write per-config JSON files A0_CONFIG.json .. A6_CONFIG.json
    for mode in CUMULATIVE_ABLATION_ORDER:
        spec = CUMULATIVE_REGISTRY[mode]
        cfg_dict = {
            "configuration": spec.internal_id,
            "display_name": spec.paper_name,
            "vietnamese_name": spec.vietnamese_name,
            "description": spec.cumulative_description,
            "pixel_gnn": spec.pixel_gnn,
            "composer": spec.composer,
            "composer_scales": spec.composer_scales,
            "motif_gnn": spec.motif_gnn,
            "geometry_bias": spec.geometry_bias,
            "relation_mode": spec.relation_mode,
            "motif_topk_schedule": spec.motif_topk_schedule,
            "motif_pooling": spec.motif_pooling,
            "direct_pixel_fusion": spec.direct_pixel_fusion,
            "classifier": spec.classifier,
            "applicable_losses": spec.applicable_losses,
            "training_recipe": {
                "seed": 42,
                "batch_size": 16,
                "gradient_accumulation_steps": 2,
                "optimizer": "AdamW",
                "learning_rate": 0.0003,
                "weight_decay": 0.0005,
                "warmup_epochs": 5,
                "lr_decay_end_epoch": 85,
                "max_epochs": 120,
                "early_stop_monitor_start_epoch": 85,
                "early_stop_patience": 15,
                "ema_decay": 0.999,
                "consistency_probability": 0.5,
                "label_smoothing": 0.05,
                "grad_clip": 1.0,
                "use_amp": True,
            },
            "dataset_hashes": {
                "train.csv": "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8",
                "val.csv": "412036d077c6ec203047b2935ab14bc858d8136ee26e8db3e23023f1fc9dee08",
                "test.csv": "be385344b93606c75cb65cac3d39b13cdedb5b91621640ee08f73b9a551fbd9d",
            },
            "notes": spec.notes,
        }
        cfg_path = out_dir / f"{spec.internal_id}_CONFIG.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)
        print("Wrote", cfg_path)

if __name__ == "__main__":
    main()
