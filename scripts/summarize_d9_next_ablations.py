from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


OUTPUTS = Path("outputs")
SUMMARY_CSV = OUTPUTS / "d9_next_ablation_summary.csv"
REPORT_MD = OUTPUTS / "d9_next_ablation_report.md"

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

RUNS = [
    {
        "run_name": "d9_rg_mr_b_diag20_cap200v100",
        "architecture": "D9-RG-MR-B original",
        "relation_encoder": "EdgeAwarePixelEncoder",
        "classifier_head": "MR KxK",
        "motif_aux_weight": "0.05",
        "epochs": "20",
        "max_train_batches": "200",
        "max_val_batches": "100",
        "notes": "Original D9 collapsed to Happy.",
    },
    {
        "run_name": "d9_rg_b_no_mr_pooled_mlp_cap200v100",
        "architecture": "D9-RG-B pooled MLP",
        "relation_encoder": "EdgeAwarePixelEncoder",
        "classifier_head": "selection pooled MLP",
        "motif_aux_weight": "0.05",
        "epochs": "10",
        "max_train_batches": "200",
        "max_val_batches": "100",
        "notes": "Best cap200 pooled baseline before this task.",
    },
    {
        "run_name": "d9_sanity_global_pool_b_cap200v100",
        "architecture": "Global node-pool sanity",
        "relation_encoder": "none",
        "classifier_head": "node MLP mean+max pool",
        "motif_aux_weight": "0.0",
        "epochs": "10",
        "max_train_batches": "200",
        "max_val_batches": "100",
        "notes": "Checks loader/loss/feature-B without motif pipeline.",
    },
    {
        "run_name": "d9_mr_b_no_relation_encoder_pooled_mlp_cap200v100",
        "architecture": "D9 no-relation pooled MLP",
        "relation_encoder": "disabled, node projection only",
        "classifier_head": "selection pooled MLP",
        "motif_aux_weight": "0.05",
        "epochs": "10",
        "max_train_batches": "200",
        "max_val_batches": "100",
        "notes": "Ablates EdgeAwarePixelEncoder while keeping motif discovery and pooled head.",
    },
    {
        "run_name": "d9_rg_b_no_mr_pooled_mlp_diag20_cap300v100",
        "architecture": "D9-RG-B pooled MLP longer diagnostic",
        "relation_encoder": "EdgeAwarePixelEncoder",
        "classifier_head": "selection pooled MLP",
        "motif_aux_weight": "0.05",
        "epochs": "20",
        "max_train_batches": "300",
        "max_val_batches": "100",
        "notes": "Fallback from requested diag30 because cap200 runs were already slow.",
    },
    {
        "run_name": "d9_rg_b_mr_v2_residual_cap200v100",
        "architecture": "D9-RG-B MR-v2 residual",
        "relation_encoder": "EdgeAwarePixelEncoder",
        "classifier_head": "pooled logits + alpha residual pair relation",
        "motif_aux_weight": "0.05",
        "epochs": "10",
        "max_train_batches": "200",
        "max_val_batches": "100",
        "notes": "Tests lightweight residual relation branch instead of replacing pooled head.",
    },
    {
        "run_name": "d9_rg_b_no_mr_pooled_mlp_aux0_cap200v100",
        "architecture": "D9-RG-B pooled MLP aux0",
        "relation_encoder": "EdgeAwarePixelEncoder",
        "classifier_head": "selection pooled MLP",
        "motif_aux_weight": "0.0",
        "epochs": "10",
        "max_train_batches": "200",
        "max_val_batches": "100",
        "notes": "Optional run not executed in this batch.",
    },
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_predictions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_run(row: dict[str, str]) -> dict[str, Any]:
    run_dir = OUTPUTS / row["run_name"]
    best = read_json(run_dir / "metrics" / "best_summary.json") or {}
    metrics = read_json(run_dir / "metrics" / "val_metrics.json") or {}
    preds = read_predictions(run_dir / "metrics" / "val_predictions.csv")
    report = metrics.get("classification_report", {})
    pred_dist = Counter(int(r["y_pred"]) for r in preds if r.get("y_pred") not in (None, ""))
    out: dict[str, Any] = dict(row)
    out["exists"] = run_dir.exists()
    out["best_epoch"] = best.get("best_epoch", "")
    out["best_val_macro_f1"] = best.get("best_val_macro_f1", metrics.get("macro_f1", ""))
    out["val_accuracy"] = metrics.get("accuracy", "")
    out["val_weighted_f1"] = metrics.get("weighted_f1", "")
    out["pred_distribution"] = dict(sorted(pred_dist.items())) if pred_dist else ""
    for name in CLASS_NAMES:
        key = f"{name.lower()}_f1"
        out[key] = report.get(name, {}).get("f1-score", "")
    cm_path = run_dir / "figures" / "val_confusion_matrix.png"
    out["confusion_matrix_path"] = str(cm_path) if cm_path.exists() else ""
    if not best and run_dir.exists():
        out["notes"] = f"{out['notes']} Missing best_summary.json."
    if not run_dir.exists():
        out["notes"] = f"{out['notes']} Not executed."
    return out


def fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    rows = [load_run(run) for run in RUNS]
    fields = [
        "run_name",
        "architecture",
        "relation_encoder",
        "classifier_head",
        "motif_aux_weight",
        "epochs",
        "max_train_batches",
        "max_val_batches",
        "best_epoch",
        "best_val_macro_f1",
        "val_accuracy",
        "val_weighted_f1",
        "pred_distribution",
        "angry_f1",
        "disgust_f1",
        "fear_f1",
        "happy_f1",
        "sad_f1",
        "surprise_f1",
        "neutral_f1",
        "confusion_matrix_path",
        "notes",
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    best_rows = [r for r in rows if isinstance(r.get("best_val_macro_f1"), (int, float))]
    best_row = max(best_rows, key=lambda r: float(r["best_val_macro_f1"]))
    no_relation = next(r for r in rows if r["run_name"] == "d9_mr_b_no_relation_encoder_pooled_mlp_cap200v100")
    pooled = next(r for r in rows if r["run_name"] == "d9_rg_b_no_mr_pooled_mlp_cap200v100")
    diag20 = next(r for r in rows if r["run_name"] == "d9_rg_b_no_mr_pooled_mlp_diag20_cap300v100")
    mr_v2 = next(r for r in rows if r["run_name"] == "d9_rg_b_mr_v2_residual_cap200v100")

    lines: list[str] = []
    lines.append("# D9 Next Ablation Report")
    lines.append("")
    lines.append("## 1. Logging/debug status")
    lines.append("- Duplicate logging: fixed in `scripts/train_d9_relation_motif.py` by resetting the D9 logger handlers and disabling propagation.")
    lines.append("- Smoke run `d9_logging_smoke_no_mr` completed without duplicate lines.")
    lines.append("- Batch logs now include total_loss, cls_loss, raw_motif_loss, weighted_motif_loss, motif_aux_weight, logits stats, pred distribution, label distribution, motif map stats, and selection weight stats when available.")
    lines.append("- Per-epoch validation writes epoch-specific val metrics/predictions/confusion matrices, plus final best-checkpoint `val_metrics.json`, `val_predictions.csv`, and `val_confusion_matrix.png`.")
    lines.append("")
    lines.append("## 2. Commands run")
    lines.append("```powershell")
    lines.append("conda run -n fer-graph python -B -m scripts.train_d9_relation_motif --config configs/experiments/d9_rg_b_no_mr_pooled_mlp.yaml --env local --graph_repo_path artifacts/graph_repo --epochs 1 --max_train_batches 3 --max_val_batches 2 --experiment_name d9_logging_smoke_no_mr --device cuda --no_wandb")
    lines.append("conda run -n fer-graph python -B -m scripts.train_d9_relation_motif --config configs/experiments/d9_mr_b_no_relation_encoder_pooled_mlp.yaml --env local --graph_repo_path artifacts/graph_repo --epochs 10 --max_train_batches 200 --max_val_batches 100 --experiment_name d9_mr_b_no_relation_encoder_pooled_mlp_cap200v100 --device cuda --no_wandb")
    lines.append("conda run -n fer-graph python -B -m scripts.train_d9_relation_motif --config configs/experiments/d9_rg_b_no_mr_pooled_mlp.yaml --env local --graph_repo_path artifacts/graph_repo --epochs 20 --max_train_batches 300 --max_val_batches 100 --experiment_name d9_rg_b_no_mr_pooled_mlp_diag20_cap300v100 --device cuda --no_wandb")
    lines.append("conda run -n fer-graph python -B -m scripts.train_d9_relation_motif --config configs/experiments/d9_rg_b_mr_v2_residual.yaml --env local --graph_repo_path artifacts/graph_repo --epochs 10 --max_train_batches 200 --max_val_batches 100 --experiment_name d9_rg_b_mr_v2_residual_cap200v100 --device cuda --no_wandb")
    lines.append("```")
    lines.append("")
    lines.append("## 3. Summary table")
    lines.append("| run | architecture | best epoch | macro F1 | acc | weighted F1 | pred distribution | class F1 Angry/Disgust/Fear/Happy/Sad/Surprise/Neutral | notes |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|---|")
    for row in rows:
        class_f1 = "/".join(fmt(row.get(f"{name.lower()}_f1", "")) for name in CLASS_NAMES)
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["run_name"],
                row["architecture"],
                row.get("best_epoch", ""),
                fmt(row.get("best_val_macro_f1", "")),
                fmt(row.get("val_accuracy", "")),
                fmt(row.get("val_weighted_f1", "")),
                row.get("pred_distribution", ""),
                class_f1,
                row.get("notes", ""),
            )
        )
    lines.append("")
    lines.append("## 4. Interpretation")
    lines.append(f"- Best run in this batch: `{best_row['run_name']}` with val_macro_f1={fmt(best_row['best_val_macro_f1'])}.")
    lines.append(f"- Global pool sanity reached val_macro_f1={fmt(next(r for r in rows if r['run_name']=='d9_sanity_global_pool_b_cap200v100')['best_val_macro_f1'])}; this is weak but not a total data/training-loop collapse.")
    lines.append(f"- No-relation pooled MLP reached {fmt(no_relation['best_val_macro_f1'])}, higher than relation pooled cap200 at {fmt(pooled['best_val_macro_f1'])}; EdgeAwarePixelEncoder is not helping in this diagnostic and may be adding noise.")
    lines.append(f"- Longer pooled MLP diag20 cap300 reached {fmt(diag20['best_val_macro_f1'])}; it improved over cap200 but did not cross 0.20.")
    lines.append(f"- MR-v2 residual reached {fmt(mr_v2['best_val_macro_f1'])}, below pooled MLP baselines; relation pair features still do not help enough.")
    lines.append("- Disgust and Fear remain near zero in the strongest runs, so the current D9 signal is still fragile and class coverage is incomplete.")
    lines.append("")
    lines.append("## 5. Final diagnosis")
    lines.append("- Decision: Option H with parts of Option C/E.")
    lines.append("- D9 has real but weak signal: pooled MLP can exceed Stage2C-B slightly and diag20 reaches 0.1674, but it remains far below a strong graph classifier and below the 0.20 threshold.")
    lines.append("- Most likely causes now: motif representation is weak, EdgeAwarePixelEncoder is not yet useful, and MR pair heads are not ready. The data/training loop is not the primary failure because global pool and pooled motif heads learn above the original collapse.")
    lines.append("")
    lines.append("## 6. Next step")
    lines.append("- Single next step: run the matching-budget `no_relation_encoder + pooled_mlp` diagnostic with epochs=20, max_train_batches=300, max_val_batches=100 before any new architecture changes.")
    lines.append("- Rationale: no-relation pooled MLP beat relation pooled MLP at cap200, but the current best run uses a larger cap with relation encoder, so the fair relation-vs-no-relation comparison is not complete yet.")
    lines.append("- Do not clone to E yet, do not run full, and do not continue MR KxK or MR-v2 as the main head until the pooled baseline exceeds 0.20 with better per-class coverage.")
    lines.append("")
    lines.append(f"CSV summary: `{SUMMARY_CSV}`")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
