from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("outputs/d9_rg_mr_b_diag20_cap200v100")
ANALYSIS_MD = Path("outputs/d9_rg_mr_b_diag20_cap200v100_analysis.md")
SUMMARY_CSV = Path("outputs/d9_rg_mr_b_diag20_cap200v100_summary.csv")
PLOT_DIR = Path("outputs/d9_rg_mr_b_diag20_cap200v100_analysis_plots")

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
BASELINES = {
    "Stage2C-B sampler": 0.1446,
    "Stage2D": 0.1238,
    "D6B": 0.4985,
    "D7 window4": 0.5837,
    "D8B border020": 0.5907,
    "Best graph-only ensemble": 0.6342,
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    converted[key] = value
                    continue
                try:
                    if key == "epoch":
                        converted[key] = int(float(value))
                    else:
                        converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            rows.append(converted)
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"graph_id", "y_true", "y_pred"}:
                    out[key] = int(value)
                elif key.startswith("logit_"):
                    out[key] = float(value)
                else:
                    out[key] = value
            rows.append(out)
    return rows


def read_config_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def config_contains(config_text: str, needle: str) -> bool:
    return needle in config_text.replace("\\", "/")


def finite_scan(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for row in rows:
        epoch = row.get("epoch", "?")
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                bad.append(f"epoch {epoch}: {key}={value}")
    return len(bad) == 0, bad


def short_float(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def trend(rows: list[dict[str, Any]], key: str, better: str = "max") -> str:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return "not available"
    delta = vals[-1] - vals[0]
    picked = min(vals) if better == "min" else max(vals)
    picked_epoch = rows[vals.index(picked)].get("epoch", "?")
    label = "min" if better == "min" else "max"
    return f"first={vals[0]:.6f}, last={vals[-1]:.6f}, {label}={picked:.6f} at epoch {picked_epoch}, delta={delta:+.6f}"


def write_summary_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    preferred = [
        "epoch",
        "train_loss",
        "train_cls_loss",
        "train_motif_loss",
        "train_accuracy",
        "train_macro_f1",
        "train_weighted_f1",
        "val_loss",
        "val_cls_loss",
        "val_motif_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "val_selected_border_mass_mean",
        "val_selected_outer_border_mass_mean",
        "val_selected_foreground_mass_mean",
        "val_selection_entropy",
        "val_selection_effective_count",
        "lr",
        "epoch_seconds",
    ]
    keys = [k for k in preferred if any(k in r for r in rows)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def make_plots(rows: list[dict[str, Any]]) -> list[Path]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plots: list[Path] = []
    epoch = [r["epoch"] for r in rows if "epoch" in r]

    def plot(keys: list[str], title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        wrote = False
        for key in keys:
            vals = [r.get(key) for r in rows]
            if all(isinstance(v, (int, float)) for v in vals):
                ax.plot(epoch, vals, marker="o", linewidth=1.5, label=key)
                wrote = True
        if not wrote:
            plt.close(fig)
            return
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = PLOT_DIR / filename
        fig.savefig(out, dpi=140)
        plt.close(fig)
        plots.append(out)

    plot(["val_macro_f1"], "D9-RG-MR-B val_macro_f1", "val_macro_f1.png")
    plot(["val_accuracy"], "D9-RG-MR-B val_accuracy", "val_accuracy.png")
    plot(["val_loss"], "D9-RG-MR-B val_loss", "val_loss.png")
    plot(["train_cls_loss", "val_cls_loss", "train_motif_loss", "val_motif_loss"], "Classification and motif losses", "cls_motif_loss.png")
    plot(["val_selected_border_mass_mean", "val_selected_foreground_mass_mean"], "Motif selected border/foreground mass", "selected_border_foreground.png")
    plot(["val_selection_entropy", "val_selection_effective_count"], "Motif entropy/effective count", "motif_entropy_effective.png")
    return plots


def inspect_checkpoints(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ckpt_dir = root / "checkpoints"
    out["exists"] = ckpt_dir.exists()
    out["files"] = {name: (ckpt_dir / name).exists() for name in ["initial.pth", "best.pth", "last.pth"]}
    try:
        import torch

        for name in ["best.pth", "last.pth"]:
            path = ckpt_dir / name
            if not path.exists():
                continue
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(path, map_location="cpu")
            if isinstance(ckpt, dict):
                out[name] = {
                    "keys": sorted(list(ckpt.keys()))[:30],
                    "epoch": ckpt.get("epoch"),
                    "best_metric": ckpt.get("best_metric"),
                    "best_epoch": ckpt.get("best_epoch"),
                    "best_metric_name": ckpt.get("best_metric_name"),
                    "best_metric_mode": ckpt.get("best_metric_mode"),
                    "monitor": ckpt.get("monitor") or ckpt.get("save_best_metric"),
                }
    except Exception as exc:
        out["checkpoint_load_error"] = str(exc)
    return out


def main() -> None:
    history = read_history(ROOT / "logs" / "d9_history.csv")
    jsonl = read_jsonl(ROOT / "logs" / "val_metrics.jsonl")
    best_summary = read_json(ROOT / "metrics" / "best_summary.json")
    val_metrics = read_json(ROOT / "metrics" / "val_metrics.json")
    preds = read_predictions(ROOT / "metrics" / "val_predictions.csv")
    config_text = read_config_text(ROOT / "resolved_config.yaml")
    checkpoint_info = inspect_checkpoints(ROOT)

    finite_ok, finite_bad = finite_scan(history)
    best_row = max(history, key=lambda r: r.get("val_macro_f1", float("-inf"))) if history else {}
    last_row = history[-1] if history else {}
    best_epoch = best_row.get("epoch")
    best_macro = best_row.get("val_macro_f1")
    best_acc = best_row.get("val_accuracy")
    best_weighted = best_row.get("val_weighted_f1")

    pred_dist = Counter(row["y_pred"] for row in preds)
    true_dist = Counter(row["y_true"] for row in preds)
    logit_ranges: dict[str, float] = {}
    if preds:
        for idx in range(7):
            vals = [row[f"logit_{idx}"] for row in preds if f"logit_{idx}" in row]
            if vals:
                logit_ranges[f"logit_{idx}"] = max(vals) - min(vals)

    motif_weight = 0.05
    weighted_motif = best_row.get("train_motif_loss", 0.0) * motif_weight if best_row else None
    cls_loss = best_row.get("train_cls_loss") if best_row else None
    motif_ratio = (weighted_motif / cls_loss) if weighted_motif is not None and cls_loss else None

    write_summary_csv(history, SUMMARY_CSV)
    plots = make_plots(history)

    figure_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "figures").rglob("*") if p.is_file()) if (ROOT / "figures").exists() else []
    motif_dirs = [p for p in [ROOT / "figures" / "d9_motifs_best", ROOT / "figures" / "d9_motifs_last"] if p.exists()]

    lines: list[str] = []
    lines.append("# D9-RG-MR-B Diagnostic Analysis")
    lines.append("")
    lines.append("## 1. Run validity")
    lines.append(f"- Run folder exists: {ROOT.exists()} (`{ROOT}`).")
    lines.append(f"- Checkpoints: initial={checkpoint_info['files'].get('initial.pth')}, best={checkpoint_info['files'].get('best.pth')}, last={checkpoint_info['files'].get('last.pth')}.")
    lines.append("- Logs: d9_history.csv={}, val_metrics.jsonl={}, metrics/val_metrics.json={}, metrics/best_summary.json={}, val_predictions.csv={}.".format(
        (ROOT / "logs" / "d9_history.csv").exists(),
        (ROOT / "logs" / "val_metrics.jsonl").exists(),
        (ROOT / "metrics" / "val_metrics.json").exists(),
        (ROOT / "metrics" / "best_summary.json").exists(),
        (ROOT / "metrics" / "val_predictions.csv").exists(),
    ))
    lines.append(f"- Figure files found: {', '.join(figure_files) if figure_files else 'none'}.")
    lines.append(f"- Motif sample dirs found: {', '.join(str(p) for p in motif_dirs) if motif_dirs else 'none; d9_motifs_best/last are missing'}")
    lines.append(f"- Resolved config exists: {(ROOT / 'resolved_config.yaml').exists()}.")
    lines.append(f"- Epoch count in history: {len(history)}; first epoch={history[0].get('epoch') if history else 'n/a'}, last epoch={last_row.get('epoch') if last_row else 'n/a'}.")
    lines.append(f"- Cap in resolved config: max_train_batches=200 -> {config_contains(config_text, 'max_train_batches: 200')}; max_val_batches=100 -> {config_contains(config_text, 'max_val_batches: 100')}.")
    lines.append(f"- CUDA requested in resolved config: {config_contains(config_text, 'device: cuda')}; AMP requested: {config_contains(config_text, 'amp: true')}.")
    lines.append(f"- NaN/Inf in history: {'none found' if finite_ok else '; '.join(finite_bad)}.")
    lines.append(f"- Checkpoint/early stopping monitor in config: checkpoint.monitor=val_macro_f1 -> {config_contains(config_text, 'monitor: val_macro_f1')}; mode=max -> {config_contains(config_text, 'mode: max')}.")
    if "best.pth" in checkpoint_info:
        lines.append(f"- best.pth metadata: {checkpoint_info['best.pth']}.")
    if "last.pth" in checkpoint_info:
        lines.append(f"- last.pth metadata: {checkpoint_info['last.pth']}.")
    if checkpoint_info.get("checkpoint_load_error"):
        lines.append(f"- Checkpoint metadata load error: {checkpoint_info['checkpoint_load_error']}.")
    lines.append(f"- Best epoch by val_macro_f1: {best_epoch}; last epoch: {last_row.get('epoch') if last_row else 'n/a'}.")
    lines.append("")

    lines.append("## 2. Metric summary")
    lines.append("| epoch | train_loss | train_cls_loss | train_motif_loss | val_loss | val_accuracy | val_macro_f1 | val_weighted_f1 | lr | sel_border | sel_foreground | sel_entropy | effective_count |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in history:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row.get("epoch"),
            short_float(row.get("train_loss")),
            short_float(row.get("train_cls_loss")),
            short_float(row.get("train_motif_loss")),
            short_float(row.get("val_loss")),
            short_float(row.get("val_accuracy")),
            short_float(row.get("val_macro_f1")),
            short_float(row.get("val_weighted_f1")),
            short_float(row.get("lr")),
            short_float(row.get("val_selected_border_mass_mean")),
            short_float(row.get("val_selected_foreground_mass_mean")),
            short_float(row.get("val_selection_entropy")),
            short_float(row.get("val_selection_effective_count")),
        ))
    lines.append("")
    lines.append(f"- Best val_macro_f1={best_macro:.6f} at epoch {best_epoch}; val_accuracy={best_acc:.6f}; val_weighted_f1={best_weighted:.6f}.")
    lines.append(f"- Final epoch val_macro_f1={last_row.get('val_macro_f1'):.6f}, val_accuracy={last_row.get('val_accuracy'):.6f}, val_weighted_f1={last_row.get('val_weighted_f1'):.6f}.")
    lines.append(f"- JSONL rows: {len(jsonl)}; best_summary agrees with history: {bool(best_summary and best_summary.get('best_epoch') == best_epoch)}.")
    lines.append("")

    lines.append("## 3. Learning curves")
    lines.append(f"- train_loss trend: {trend(history, 'train_loss', 'min')}.")
    lines.append(f"- train_cls_loss trend: {trend(history, 'train_cls_loss', 'min')}.")
    lines.append(f"- train_motif_loss trend: {trend(history, 'train_motif_loss', 'min')}.")
    lines.append(f"- val_loss trend: {trend(history, 'val_loss', 'min')}.")
    lines.append(f"- val_accuracy trend: {trend(history, 'val_accuracy')}.")
    lines.append(f"- val_macro_f1 trend: {trend(history, 'val_macro_f1')}.")
    lines.append(f"- LR schedule: warmup to {max(r.get('lr', 0.0) for r in history):.6g} at epoch {max(history, key=lambda r: r.get('lr', 0)).get('epoch')}, then cosine decay to {last_row.get('lr'):.6g}.")
    lines.append(f"- Plots written: {', '.join(str(p) for p in plots) if plots else 'not written; matplotlib unavailable or no plottable data'}.")
    lines.append("")

    lines.append("## 4. Best checkpoint analysis")
    lines.append(f"- best_summary.json reports best_epoch={best_summary.get('best_epoch') if best_summary else 'n/a'}, best_val_macro_f1={best_summary.get('best_val_macro_f1') if best_summary else 'n/a'}.")
    lines.append(f"- By history recomputation, best epoch is {best_epoch}; last epoch is {last_row.get('epoch') if last_row else 'n/a'}.")
    lines.append("- The best epoch is consistent with the policy because epoch 5 has the maximum val_macro_f1, even though later epochs keep running under patience=20.")
    lines.append("")

    lines.append("## 5. Per-class / confusion analysis")
    if val_metrics and "classification_report" in val_metrics:
        report = val_metrics["classification_report"]
        lines.append("| class | precision | recall | f1 | support | predicted_count |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for idx, name in enumerate(CLASS_NAMES):
            item = report.get(name, {})
            lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {:.0f} | {} |".format(
                name,
                item.get("precision", 0.0),
                item.get("recall", 0.0),
                item.get("f1-score", 0.0),
                item.get("support", 0.0),
                pred_dist.get(idx, 0),
            ))
        lines.append("")
        lines.append(f"- Prediction distribution from val_predictions.csv: {dict(sorted(pred_dist.items()))}.")
        lines.append(f"- True distribution from val_predictions.csv: {dict(sorted(true_dist.items()))}.")
        lines.append(f"- Logit value range by class: {logit_ranges}.")
        lines.append("- Confusion matrix shows a full one-class collapse to Happy: every validation sample is predicted as class index 3.")
    else:
        lines.append("- Per-class report not available.")
    lines.append("")

    lines.append("## 6. Motif output analysis")
    lines.append("- Expected motif visualization dirs `figures/d9_motifs_best/` and `figures/d9_motifs_last/` are not present.")
    lines.append("- Only `figures/val_confusion_matrix.png` is present, so qualitative motif placement cannot be verified from saved images.")
    lines.append("- Relation attention artifacts are not present; config has relation_encoder.use_attention=false, and no motif-pair/attention visualizations were saved.")
    lines.append("")

    lines.append("## 7. Motif metric analysis")
    lines.append(f"- val_selected_border_mass_mean trend: {trend(history, 'val_selected_border_mass_mean')}.")
    lines.append(f"- val_selected_outer_border_mass_mean trend: {trend(history, 'val_selected_outer_border_mass_mean')}.")
    lines.append(f"- val_selected_foreground_mass_mean trend: {trend(history, 'val_selected_foreground_mass_mean')}.")
    lines.append(f"- val_selection_entropy trend: {trend(history, 'val_selection_entropy')}.")
    lines.append(f"- val_selection_effective_count trend: {trend(history, 'val_selection_effective_count')}.")
    lines.append("- No map_sim, emb_sim, redundant, clean, region_clean, motif_quality, or aux_f1 columns are present in the run logs.")
    lines.append("- The available motif metrics are poor for semantic motif claims: foreground mass is exactly 0.0, selected border mass is around 0.30, entropy is near log(16), and effective count is near 16, meaning selections are diffuse and not face-foreground focused.")
    lines.append("")

    lines.append("## 8. Loss balance analysis")
    lines.append(f"- Config motif_aux_weight: {motif_weight}.")
    lines.append(f"- At best epoch: train_cls_loss={cls_loss:.6f}, train_motif_loss={best_row.get('train_motif_loss'):.6f}, weighted motif term={weighted_motif:.6f}, weighted_motif/train_cls_loss={motif_ratio:.6%}.")
    lines.append("- Motif loss is not numerically dominating CE. The collapse is unlikely to be caused by motif_aux_weight=0.05 overpowering total loss.")
    lines.append("- Because motif metrics are bad while classification collapses, the next loss ablation should include motif_aux_weight=0.0 as a diagnostic, not because 0.05 is large, but to isolate whether the auxiliary objective is distracting early representation learning.")
    lines.append("")

    lines.append("## 9. Runtime analysis")
    if history:
        seconds = [r["epoch_seconds"] for r in history if isinstance(r.get("epoch_seconds"), (int, float))]
        lines.append(f"- Total epoch_seconds sum: {sum(seconds):.1f}s ({sum(seconds) / 60:.1f} minutes).")
        lines.append(f"- Mean epoch_seconds: {mean(seconds):.1f}s; min={min(seconds):.1f}s; max={max(seconds):.1f}s.")
        lines.append(f"- With cap 200 train batches and 100 val batches, rough seconds per capped train+val batch budget is {sum(seconds) / (len(seconds) * 300):.3f}s/batch-slot.")
    lines.append("- Runtime is usable for capped diagnostics on local CUDA, but full uncapped training would be much longer. Kaggle/full runs should use copied graph_repo in working storage, more workers if stable, larger batch size if memory allows, and AMP already enabled.")
    lines.append("")

    lines.append("## 10. Comparison with previous baselines")
    lines.append("| baseline | macro F1 | D9 best - baseline |")
    lines.append("|---|---:|---:|")
    for name, score in BASELINES.items():
        lines.append(f"| {name} | {score:.4f} | {best_macro - score:+.4f} |")
    lines.append("")
    lines.append("- D9-B diagnostic is below Stage2C-B and Stage2D, far below D6/D7/D8B. Because this is capped, the comparison is directional only, but it does not clear the minimum diagnostic bar of ~0.15.")
    lines.append("")

    lines.append("## 11. Recommended next steps")
    lines.append("1. Run a no-motif-aux diagnostic (`motif_aux_weight=0.0`) with the same cap to isolate CE/classifier learning from the auxiliary motif objective.")
    lines.append("2. Run a no-relation-encoder ablation with the same cap to test whether EdgeAwarePixelEncoder/relation message passing is causing near-constant logits and Happy collapse.")
    lines.append("3. Run a no-MR classifier ablation (motif embeddings pooled MLP) if the first two still collapse, to determine whether the motif relation classifier is the failure point.")
    lines.append("- Do not clone to E yet. Do not run longer yet unless a short ablation first shows macro F1 above Stage2 or removes one-class collapse.")
    lines.append("")

    lines.append("## 12. Final decision")
    lines.append("- Decision: Option 3 - D9-B diagnostic currently does not learn classification.")
    lines.append(f"- Evidence: best val_macro_f1={best_macro:.6f} (<0.15), all validation predictions collapse to Happy, and motif outputs are not available while motif metrics indicate diffuse/border-heavy/non-foreground selection.")
    lines.append("- Action: debug with controlled ablations before spending a longer run budget or cloning to E.")
    lines.append("")

    ANALYSIS_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
