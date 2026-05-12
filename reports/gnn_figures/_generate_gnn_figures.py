import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "gnn_figures"
LEGACY = OUT / "legacy_motif_visualization"
OUT.mkdir(parents=True, exist_ok=True)
LEGACY.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
created = []
skipped = []


def find_graph_repo_root():
    candidates = [
        ROOT / "artifacts" / "graph_repo",
        ROOT / "artifacts" / "graph-repo" / "graph_repo",
        ROOT / "artifacts" / "graph_repo_intensity_xy",
    ]
    for candidate in candidates:
        if (candidate / "manifest.pt").exists() and (candidate / "test").exists():
            return candidate
    return candidates[0]


GRAPH_REPO = find_graph_repo_root()


def rel(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def mark_created(path, desc):
    created.append((rel(path), desc))


def mark_skipped(name, reason):
    skipped.append((name, reason))


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    mark_created(path, "JSON source/evidence")


def write_csv(path, rows, fieldnames):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    mark_created(path, "CSV source/evidence")


def savefig(path):
    path = Path(path)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    mark_created(path, "PNG figure")


def metric_paths(base):
    base = ROOT / base if not isinstance(base, Path) else base
    metrics = next((p for p in [
        base / "evaluation" / "metrics.json",
        base / "metrics" / "val_metrics.json",
        base / "metrics" / "test_metrics.json",
        base / "metrics.json",
    ] if p.exists()), None)
    report = next((p for p in [
        base / "evaluation" / "classification_report.json",
        base / "metrics" / "classification_report.json",
    ] if p.exists()), None)
    predictions = next((p for p in [
        base / "evaluation" / "predictions.csv",
        base / "metrics" / "val_predictions.csv",
    ] if p.exists()), None)
    return {
        "base": base,
        "metrics": metrics,
        "report": report,
        "predictions": predictions,
        "history": base / "training_history.json" if (base / "training_history.json").exists() else None,
        "best_summary": base / "metrics" / "best_summary.json" if (base / "metrics" / "best_summary.json").exists() else None,
        "resolved": base / "resolved_config.yaml" if (base / "resolved_config.yaml").exists() else None,
        "ckpt": base / "checkpoints" / "best.pth" if (base / "checkpoints" / "best.pth").exists() else None,
    }


def get_metrics(base):
    paths = metric_paths(base)
    data = load_json(paths["metrics"]) if paths["metrics"] else {}
    best = load_json(paths["best_summary"]) if paths["best_summary"] else {}
    report = load_json(paths["report"]) if paths["report"] else None
    if not report and isinstance(data, dict) and isinstance(data.get("classification_report"), dict):
        report = data["classification_report"]
    out = {
        "accuracy": data.get("accuracy", data.get("val_accuracy", data.get("test_accuracy"))),
        "macro_f1": data.get("macro_f1", data.get("val_macro_f1", data.get("test_macro_f1"))),
        "weighted_f1": data.get("weighted_f1", data.get("val_weighted_f1", data.get("test_weighted_f1"))),
        "best_epoch": "",
        "report": report,
        "paths": paths,
    }
    if best:
        bm = best.get("best_metrics", {}) or {}
        out["best_epoch"] = best.get("best_epoch", "")
        out["accuracy"] = out["accuracy"] if out["accuracy"] is not None else bm.get("val_accuracy")
        out["macro_f1"] = out["macro_f1"] if out["macro_f1"] is not None else bm.get("val_macro_f1", best.get("best_val_macro_f1"))
        out["weighted_f1"] = out["weighted_f1"] if out["weighted_f1"] is not None else bm.get("val_weighted_f1")
    if paths["history"]:
        rows = load_json(paths["history"]) or []
        rows = rows if isinstance(rows, list) else rows.get("history", [])
        valid = [r for r in rows if r.get("val_macro_f1") is not None]
        if valid:
            out["best_epoch"] = out["best_epoch"] or max(valid, key=lambda r: float(r["val_macro_f1"])).get("epoch", "")
        elif rows and rows[0].get("val_loss") is not None:
            out["best_epoch"] = out["best_epoch"] or min(rows, key=lambda r: float(r.get("val_loss", 1e9))).get("epoch", "")
    return out


def softmax(scores):
    z = scores - np.nanmax(scores)
    exp = np.exp(z)
    return exp / exp.sum()


def parse_predictions(path):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        score_cols = [c for c in reader.fieldnames if c.startswith("score_") or c.startswith("logit_")]
        score_cols = sorted(score_cols, key=lambda c: int(c.split("_")[-1]))
        for r in reader:
            y_true = int(r.get("y_true", r.get("true_label", r.get("label"))))
            y_pred = int(r.get("y_pred", r.get("pred_label", r.get("prediction"))))
            scores = np.array([float(r[c]) for c in score_cols], dtype=np.float64)
            score_sum = float(np.nansum(scores))
            probs = scores if (np.all(scores >= 0) and abs(score_sum - 1.0) < 1e-3) else softmax(scores)
            top2 = np.sort(probs)[-2:]
            row = {
                "graph_id": int(r.get("graph_id", r.get("sample_idx", len(rows)))),
                "y_true": y_true,
                "y_pred": y_pred,
                "true_label": CLASS_NAMES[y_true],
                "pred_label": CLASS_NAMES[y_pred],
                "correct": int(y_true == y_pred),
                "confidence": float(probs[y_pred]),
                "top2_margin": float(top2[-1] - top2[-2]),
            }
            for i in range(7):
                row[f"prob_{i}"] = float(probs[i])
                row[f"score_{i}"] = float(scores[i])
            rows.append(row)
    return rows


def confusion_matrix(rows):
    mat = np.zeros((7, 7), dtype=int)
    for r in rows:
        mat[int(r["y_true"]), int(r["y_pred"])] += 1
    return mat


def write_matrix_csv(mat, path, normalized=False):
    data = mat.astype(float)
    if normalized:
        denom = data.sum(axis=1, keepdims=True)
        data = np.divide(data, denom, out=np.zeros_like(data), where=denom != 0)
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        row = {"true_label": name}
        for j, pred in enumerate(CLASS_NAMES):
            row[pred] = f"{data[i, j]:.6f}" if normalized else int(data[i, j])
        rows.append(row)
    write_csv(path, rows, ["true_label"] + CLASS_NAMES)


def plot_confusion(mat, path, title, normalized=False):
    data = mat.astype(float)
    if normalized:
        denom = data.sum(axis=1, keepdims=True)
        data = np.divide(data, denom, out=np.zeros_like(data), where=denom != 0)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1.0 if normalized else None)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(7), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(7), CLASS_NAMES)
    threshold = 0.5 if normalized else max(1.0, mat.max() * 0.55)
    for i in range(7):
        for j in range(7):
            txt = f"{data[i, j]:.2f}" if normalized else str(int(data[i, j]))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(path)


class GraphImageCache:
    def __init__(self, graph_root):
        self.graph_root = Path(graph_root)
        self.cache = {}

    def get(self, graph_id):
        chunk_id = int(graph_id) // 500
        idx = int(graph_id) % 500
        if chunk_id not in self.cache:
            path = self.graph_root / "test" / f"chunk_{chunk_id:03d}.pt"
            self.cache[chunk_id] = torch.load(path, map_location="cpu", weights_only=False)
        sample = self.cache[chunk_id][idx]
        return np.clip(sample.node_features[:, 0].float().reshape(48, 48).numpy(), 0, 1)


def plot_example_grid(rows, out_path, title, max_items=20, include_status=False):
    rows = rows[:max_items]
    if not rows:
        mark_skipped(Path(out_path).name, "KHONG CO mau phu hop trong predictions.csv")
        return
    cols = 5 if len(rows) >= 10 else min(5, len(rows))
    nrows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 2.25, nrows * 2.45))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    csv_rows = []
    for ax, r in zip(axes, rows):
        try:
            ax.imshow(img_cache.get(r["graph_id"]), cmap="gray", vmin=0, vmax=1)
        except Exception:
            ax.text(0.5, 0.5, "image\nmissing", ha="center", va="center")
        txt = f"id {r['graph_id']}\nT:{r['true_label']} P:{r['pred_label']}\nconf {r['confidence']:.2f}"
        if include_status:
            txt += "\nOK" if r["correct"] else "\nWRONG"
        ax.set_title(txt, fontsize=7)
        csv_rows.append({k: r[k] for k in ["graph_id", "true_label", "pred_label", "y_true", "y_pred", "correct", "confidence", "top2_margin"]})
    fig.suptitle(title, fontsize=12, fontweight="bold")
    savefig(out_path)
    write_csv(Path(out_path).with_suffix(".csv"), csv_rows,
              ["graph_id", "true_label", "pred_label", "y_true", "y_pred", "correct", "confidence", "top2_margin"])


def copy_file(src, dst, desc):
    src = ROOT / src if not isinstance(src, Path) else src
    dst = OUT / dst if not isinstance(dst, Path) else dst
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        mark_created(dst, f"{desc} from {rel(src)}")
        return True
    mark_skipped(rel(dst), f"KHONG TIM THAY source {rel(src)}")
    return False


def combine_images(srcs, dst, title):
    existing = [(ROOT / p, label) for p, label in srcs if (ROOT / p).exists()]
    if not existing:
        mark_skipped(Path(dst).name, "KHONG TIM THAY source images")
        return
    cols = min(3, len(existing))
    nrows = math.ceil(len(existing) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 4.0, nrows * 3.0))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (path, label) in zip(axes, existing):
        ax.imshow(plt.imread(path))
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontweight="bold")
    savefig(dst)
    write_json(Path(dst).with_suffix(".sources.json"),
               [{"source": rel(path), "label": label} for path, label in existing])


runs = {
    "Baseline Motif-GNN": ("d5a", "output/d5a", "D5A motif-level graph baseline / closest Baseline Motif-GNN artifact."),
    "Hierarchical Motif-GNN": ("d6a_slot_pixel_part_graph_motif", "output/d6a_slot_pixel_part_graph_motif", "D6A slot pixel part graph motif."),
    "Refined Part-Motif GNN": ("d6b_class_part_graph_motif_border075_long150", "output/d6b_class_part_graph_motif_border075_long150", "D6B long150 class-part motif attention."),
    "D9 Relation Motif Graph": ("d9_rg_mr_b_diag20_cap200v100", "outputs/d9_rg_mr_b_diag20_cap200v100", "D9 relation motif graph with MR head; rejected due collapse."),
    "D9 Motif Pooling": ("d9_rg_b_no_mr_pooled_mlp_diag20_cap300v100", "outputs/d9_rg_b_no_mr_pooled_mlp_diag20_cap300v100", "D9 no-MR pooled MLP comparator."),
    "D9 Teacher-Guided": ("d9_tgms_b_distill_a02_t2_cap300v100_outputs", "outputs/d9_tgms_b_distill_a02_t2_cap300v100_outputs", "D9 TGMS teacher-guided alpha=0.2 comparator."),
    "D10 Slot-Attention Fast": ("d10_slot_motif_fast/20260508_092631", "outputs/d10_slot_motif_full_outputs/d10_slot_motif_fast/20260508_092631", "Fast D10 slot-attention baseline."),
    "D10 Stable Slot-Attention": ("d10_v2_2_iter5_lr3e4/20260508_112654", "outputs/d10_slot_motif_full_outputs/d10_v2_2_iter5_lr3e4/20260508_112654", "D10 V2.2 iter5 lr3e-4."),
    "D10 + Cosine Scheduler": ("d10_v3_2_cosine_only/20260508_135719", "outputs/d10_slot_motif_full_outputs/d10_v3_2_cosine_only/20260508_135719", "D10 V3.2 cosine scheduler."),
    "D10 Slot Refinement": ("d10_p3_5_iter5", "outputs/d10_slot_motif_full_outputs/d10_p3_5_iter5", "D10 P3.5 refinement + iter5."),
    "D10 SupCon Best": ("d10_p5_stage2_relation_outputs_lan2", "outputs/d10_slot_motif_full_outputs/d10_p5_standard/d10_p5_stage2_relation_outputs_lan2", "Main D10 Phase 5 mean-pooled SupCon best, test macro F1 about 0.6130."),
    "D10 SupCon Moderate-Reg": ("d10_p5_stage2_run4_moderate", "outputs/d10_slot_motif_full_outputs/d10_p5_run4/d10_p5_stage2_run4_moderate", "D10 Phase 5 Run4 moderate regularization."),
    "D10 SupCon Strong-Reg": ("d10_p5_stage2_run4_strong", "outputs/d10_slot_motif_full_outputs/d10_p5_run4/d10_p5_stage2_run4_strong", "D10 Phase 5 Run4 strong regularization."),
}

metrics_by_name = {}
mapping_rows = []
for report_name, (repo_name, output_path, desc) in runs.items():
    metrics = get_metrics(ROOT / output_path)
    metrics_by_name[report_name] = metrics
    p = metrics["paths"]
    mapping_rows.append({
        "report_name": report_name,
        "repo_run_name": repo_name,
        "output_path": output_path if (ROOT / output_path).exists() else "KHONG TIM THAY",
        "config_path": rel(p["resolved"]) if p["resolved"] else "KHONG TIM THAY",
        "checkpoint_path": rel(p["ckpt"]) if p["ckpt"] else "KHONG TIM THAY",
        "short_description": desc,
    })
write_csv(OUT / "version_name_mapping.csv", mapping_rows,
          ["report_name", "repo_run_name", "output_path", "config_path", "checkpoint_path", "short_description"])

progression = ["Baseline Motif-GNN", "Hierarchical Motif-GNN", "Refined Part-Motif GNN", "D9 Motif Pooling",
               "D10 Slot-Attention Fast", "D10 Slot Refinement", "D10 SupCon Best", "D10 SupCon Moderate-Reg"]
progression_rows = []
for name in progression:
    m = metrics_by_name[name]
    progression_rows.append({
        "report_name": name,
        "output_path": runs[name][1],
        "best_epoch": m.get("best_epoch", ""),
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "weighted_f1": m.get("weighted_f1"),
        "metrics_path": rel(m["paths"]["metrics"]) if m["paths"]["metrics"] else "KHONG TIM THAY",
    })
write_csv(OUT / "gnn_macro_f1_progression.csv", progression_rows,
          ["report_name", "output_path", "best_epoch", "accuracy", "macro_f1", "weighted_f1", "metrics_path"])
write_json(OUT / "gnn_macro_f1_progression_source.json", progression_rows)

fig, ax = plt.subplots(figsize=(11, 5.5))
vals = [float(r["macro_f1"]) for r in progression_rows]
labels = [r["report_name"].replace(" ", "\n") for r in progression_rows]
bars = ax.bar(range(len(vals)), vals, color=["#7a8fa6", "#6c9a8b", "#4b8bbe", "#9e8a73", "#5b9bd5", "#3f7fbf", "#1f5d99", "#7fb069"], edgecolor="#222", linewidth=0.7)
ax.set_ylabel("Macro F1")
ax.set_title("GNN Macro F1 Progression", fontweight="bold")
ax.set_xticks(range(len(vals)), labels, fontsize=8)
ax.set_ylim(0, max(vals) * 1.18)
ax.grid(axis="y", alpha=0.25)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
savefig(OUT / "gnn_macro_f1_progression.png")

phase_names = ["D10 SupCon Best", "D10 SupCon Moderate-Reg", "D10 SupCon Strong-Reg"]
phase_rows = []
for name in phase_names:
    m = metrics_by_name[name]
    phase_rows.append({
        "report_name": name,
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "weighted_f1": m.get("weighted_f1"),
        "metrics_path": rel(m["paths"]["metrics"]) if m["paths"]["metrics"] else "KHONG TIM THAY",
    })
write_csv(OUT / "d10_phase5_comparison.csv", phase_rows,
          ["report_name", "accuracy", "macro_f1", "weighted_f1", "metrics_path"])

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(phase_names))
width = 0.25
for idx, (key, label, color) in enumerate([("accuracy", "Accuracy", "#4c78a8"), ("macro_f1", "Macro F1", "#59a14f"), ("weighted_f1", "Weighted F1", "#f28e2b")]):
    vals = [float(r[key]) for r in phase_rows]
    bars = ax.bar(x + (idx - 1) * width, vals, width, label=label, color=color, edgecolor="#222", linewidth=0.5)
    for xi, val in zip(x + (idx - 1) * width, vals):
        ax.text(xi, val + 0.008, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x, [n.replace("D10 ", "").replace(" ", "\n") for n in phase_names])
ax.set_ylim(0, max(float(r["accuracy"]) for r in phase_rows) * 1.16)
ax.set_ylabel("Score")
ax.set_title("D10 Phase 5 Comparison", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.25)
savefig(OUT / "d10_phase5_comparison.png")

best = metrics_by_name["D10 SupCon Best"]
moderate = metrics_by_name["D10 SupCon Moderate-Reg"]
best_rows = parse_predictions(best["paths"]["predictions"]) if best["paths"]["predictions"] else []
moderate_rows = parse_predictions(moderate["paths"]["predictions"]) if moderate["paths"]["predictions"] else []
if not best_rows:
    mark_skipped("D10 SupCon Best prediction-derived figures", "KHONG TIM THAY predictions.csv")
else:
    pred_fields = ["graph_id", "true_label", "pred_label", "y_true", "y_pred", "correct", "confidence", "top2_margin"] + [f"prob_{i}" for i in range(7)] + [f"score_{i}" for i in range(7)]
    write_csv(OUT / "confidence_d10_supcon_best.csv", best_rows, pred_fields)

for rows, tag, title in [
    (best_rows, "d10_supcon_best", "D10 SupCon Best Confusion Matrix"),
    (moderate_rows, "d10_moderate_reg", "D10 SupCon Moderate-Reg Confusion Matrix"),
]:
    if rows:
        mat = confusion_matrix(rows)
        write_matrix_csv(mat, OUT / f"confusion_matrix_{tag}.csv")
        write_matrix_csv(mat, OUT / f"confusion_matrix_{tag}_normalized.csv", normalized=True)
        plot_confusion(mat, OUT / f"confusion_matrix_{tag}.png", title)
        plot_confusion(mat, OUT / f"confusion_matrix_{tag}_normalized.png", title + " (row-normalized)", normalized=True)
    else:
        mark_skipped(f"confusion_matrix_{tag}.png", "KHONG TIM THAY predictions.csv")

if best_rows:
    mat = confusion_matrix(best_rows)
    pairs = []
    for i in range(7):
        for j in range(7):
            if i != j and mat[i, j] > 0:
                pairs.append({"true_label": CLASS_NAMES[i], "pred_label": CLASS_NAMES[j], "count": int(mat[i, j])})
    pairs.sort(key=lambda r: r["count"], reverse=True)
    write_csv(OUT / "top_confusion_pairs_d10_supcon_best.csv", pairs, ["true_label", "pred_label", "count"])
    top8 = pairs[:8]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [f"{r['true_label']}->{r['pred_label']}" for r in top8]
    vals = [r["count"] for r in top8]
    bars = ax.barh(range(len(vals)), vals, color="#b65f5f", edgecolor="#222", linewidth=0.5)
    ax.set_yticks(range(len(vals)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Top Confusion Pairs - D10 SupCon Best", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for bar, val in zip(bars, vals):
        ax.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2, str(val), va="center", fontsize=8)
    savefig(OUT / "top_confusion_pairs_d10_supcon_best.png")

report = best["report"]
if report:
    report_rows = []
    for cname in CLASS_NAMES:
        item = report.get(cname, report.get(cname.lower(), report.get(str(CLASS_TO_ID[cname]), {})))
        report_rows.append({
            "class": cname,
            "precision": item.get("precision", ""),
            "recall": item.get("recall", ""),
            "f1": item.get("f1-score", item.get("f1", "")),
            "support": item.get("support", ""),
        })
    write_csv(OUT / "classification_report_d10_supcon_best.csv", report_rows,
              ["class", "precision", "recall", "f1", "support"])
    hard = ["Fear", "Sad", "Disgust", "Neutral"]
    rows = [next(r for r in report_rows if r["class"] == c) for c in hard]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(hard))
    width = 0.25
    for idx, (key, label, color) in enumerate([("precision", "Precision", "#4e79a7"), ("recall", "Recall", "#59a14f"), ("f1", "F1", "#e15759")]):
        vals = [float(r[key]) for r in rows]
        ax.bar(x + (idx - 1) * width, vals, width, label=label, color=color, edgecolor="#222", linewidth=0.5)
        for xi, val in zip(x + (idx - 1) * width, vals):
            ax.text(xi, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, hard)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Hard Classes - D10 SupCon Best", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    savefig(OUT / "hard_classes_f1_d10_supcon_best.png")
else:
    mark_skipped("hard_classes_f1_d10_supcon_best.png", "KHONG TIM THAY classification_report")

hist_path = best["paths"]["history"]
if hist_path:
    rows = load_json(hist_path) or []
    rows = rows if isinstance(rows, list) else rows.get("history", [])
    fields = sorted({k for r in rows for k in r.keys()})
    write_csv(OUT / "training_history_d10_supcon_best.csv", rows, fields)
    epochs = np.array([float(r.get("epoch", i + 1)) for i, r in enumerate(rows)])
    def arr(key):
        return np.array([float(r[key]) if r.get(key) is not None else np.nan for r in rows])
    train_f1, val_f1 = arr("train_macro_f1"), arr("val_macro_f1")
    train_loss, val_loss = arr("train_loss"), arr("val_loss")
    if not np.all(np.isnan(train_f1)) and not np.all(np.isnan(val_f1)):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(epochs, train_f1, label="Train macro F1", color="#4e79a7", linewidth=2)
        ax.plot(epochs, val_f1, label="Val macro F1", color="#e15759", linewidth=2)
        final_gap = train_f1[-1] - val_f1[-1]
        ax.annotate(f"Final gap: {final_gap:.3f}", xy=(epochs[-1], val_f1[-1]), xytext=(epochs[-1] * 0.62, np.nanmax(val_f1) * 0.92), arrowprops=dict(arrowstyle="->", color="#444"), fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro F1")
        ax.set_title("Train vs Val Macro F1 - D10 SupCon Best", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend()
        savefig(OUT / "train_val_macro_f1_d10_supcon_best.png")

        gap = train_f1 - val_f1
        gap_rows = [{"epoch": float(e), "train_macro_f1": float(a), "val_macro_f1": float(b), "gap": float(g)} for e, a, b, g in zip(epochs, train_f1, val_f1, gap)]
        write_csv(OUT / "overfitting_gap_d10_supcon_best.csv", gap_rows,
                  ["epoch", "train_macro_f1", "val_macro_f1", "gap"])
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(epochs, gap, color="#8c564b", linewidth=2)
        ax.fill_between(epochs, 0, gap, color="#8c564b", alpha=0.18)
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.annotate(f"Last gap: {gap[-1]:.3f}\ntrain={train_f1[-1]:.3f}, val={val_f1[-1]:.3f}", xy=(epochs[-1], gap[-1]), xytext=(epochs[-1] * 0.55, np.nanmax(gap) * 0.75), arrowprops=dict(arrowstyle="->", color="#444"), fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train F1 - Val F1")
        ax.set_title("Overfitting Gap - D10 SupCon Best", fontweight="bold")
        ax.grid(alpha=0.25)
        savefig(OUT / "overfitting_gap_d10_supcon_best.png")
    else:
        mark_skipped("train_val_macro_f1_d10_supcon_best.png", "KHONG TIM THAY train_macro_f1/val_macro_f1")
        mark_skipped("overfitting_gap_d10_supcon_best.png", "KHONG TIM THAY train_macro_f1/val_macro_f1")
    if not np.all(np.isnan(train_loss)) and not np.all(np.isnan(val_loss)):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(epochs, train_loss, label="Train loss", color="#4e79a7", linewidth=2)
        ax.plot(epochs, val_loss, label="Val loss", color="#e15759", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train vs Val Loss - D10 SupCon Best", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend()
        savefig(OUT / "train_val_loss_d10_supcon_best.png")
    else:
        mark_skipped("train_val_loss_d10_supcon_best.png", "KHONG TIM THAY train_loss/val_loss")
else:
    mark_skipped("training curve figures", "KHONG TIM THAY training_history.json")

if best_rows:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.linspace(0, 1, 31)
    ax.hist([r["confidence"] for r in best_rows if r["correct"]], bins=bins, alpha=0.65, label="Correct", color="#59a14f")
    ax.hist([r["confidence"] for r in best_rows if not r["correct"]], bins=bins, alpha=0.65, label="Wrong", color="#e15759")
    ax.set_xlabel("Prediction confidence (softmax of saved scores)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence: Correct vs Wrong - D10 SupCon Best", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    savefig(OUT / "confidence_correct_vs_wrong.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    data = [[r["confidence"] for r in best_rows if r["true_label"] == c] for c in CLASS_NAMES]
    ax.boxplot(data, labels=CLASS_NAMES, showfliers=False, patch_artist=True, boxprops=dict(facecolor="#8ab6d6", alpha=0.75))
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence by True Class - D10 SupCon Best", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    savefig(OUT / "confidence_by_class.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist([r["top2_margin"] for r in best_rows if not r["correct"]], bins=30, color="#b07aa1", edgecolor="#222", linewidth=0.4)
    ax.set_xlabel("Top1 - Top2 probability margin")
    ax.set_ylabel("Wrong sample count")
    ax.set_title("Top-2 Margin for Wrong Examples - D10 SupCon Best", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    savefig(OUT / "top2_margin_wrong_examples.png")

img_cache = GraphImageCache(GRAPH_REPO)
if best_rows:
    eval_dir = best["paths"]["base"] / "evaluation"
    if not copy_file(eval_dir / "correct_examples.png", OUT / "correct_examples_d10_supcon_best.png", "D10 evaluation correct examples"):
        plot_example_grid(sorted([r for r in best_rows if r["correct"]], key=lambda r: r["confidence"], reverse=True), OUT / "correct_examples_d10_supcon_best.png", "Correct Examples - D10 SupCon Best", max_items=25)
    if not copy_file(eval_dir / "wrong_examples.png", OUT / "wrong_examples_d10_supcon_best.png", "D10 evaluation wrong examples"):
        plot_example_grid(sorted([r for r in best_rows if not r["correct"]], key=lambda r: r["confidence"], reverse=True), OUT / "wrong_examples_d10_supcon_best.png", "Wrong Examples - D10 SupCon Best", max_items=25)
    wrong = [r for r in best_rows if not r["correct"]]
    write_csv(OUT / "wrong_examples_d10_supcon_best.csv", wrong,
              ["graph_id", "true_label", "pred_label", "y_true", "y_pred", "correct", "confidence", "top2_margin"] + [f"prob_{i}" for i in range(7)])
    plot_example_grid(sorted([r for r in wrong if r["y_true"] == CLASS_TO_ID["Fear"] and r["y_pred"] in [CLASS_TO_ID["Sad"], CLASS_TO_ID["Surprise"]]], key=lambda r: r["confidence"], reverse=True),
                      OUT / "wrong_fear_as_sad_or_surprise.png", "Wrong Fear as Sad/Surprise - D10 SupCon Best")
    plot_example_grid(sorted([r for r in wrong if r["y_true"] == CLASS_TO_ID["Sad"] and r["y_pred"] in [CLASS_TO_ID["Neutral"], CLASS_TO_ID["Fear"]]], key=lambda r: r["confidence"], reverse=True),
                      OUT / "wrong_sad_as_neutral_or_fear.png", "Wrong Sad as Neutral/Fear - D10 SupCon Best")
    disgust_wrong = sorted([r for r in wrong if r["y_true"] == CLASS_TO_ID["Disgust"]], key=lambda r: r["confidence"], reverse=True)
    disgust_ok = sorted([r for r in best_rows if r["correct"] and r["y_true"] == CLASS_TO_ID["Disgust"]], key=lambda r: r["confidence"], reverse=True)[:6]
    plot_example_grid(disgust_wrong[:14] + disgust_ok, OUT / "wrong_disgust_cases.png", "Disgust Cases: wrong first, then correct comparisons", include_status=True)
    plot_example_grid(sorted(wrong, key=lambda r: r["top2_margin"])[:20], OUT / "ambiguous_wrong_examples.png", "Ambiguous Wrong Examples - Lowest Top-2 Margin")
    plot_example_grid(sorted(wrong, key=lambda r: r["confidence"], reverse=True)[:20], OUT / "high_confidence_wrong_examples.png", "High-Confidence Wrong Examples")

attention_check = {
    "model_forward_returns": ["slot_attn_maps", "class_motif_attn"],
    "code_evidence": ["models/d10_slot_motif_model.py:491-495", "models/d10_slot_motif_model.py:529-543"],
    "attempted_inference": False,
    "can_create_d10_attention_now": False,
    "reason": "",
}
try:
    manifest = torch.load(GRAPH_REPO / "manifest.pt", map_location="cpu")
    local_node_dim = int(manifest.get("node_dim"))
except Exception as exc:
    local_node_dim = None
    attention_check["reason"] = f"Could not read local graph_repo manifest: {exc}"
resolved = best["paths"]["resolved"].read_text(encoding="utf-8") if best["paths"]["resolved"] else ""
ckpt_node_dim = None
for line in resolved.splitlines():
    if line.strip().startswith("node_dim:"):
        ckpt_node_dim = int(line.split(":", 1)[1].strip())
        break
if local_node_dim is not None and ckpt_node_dim is not None and local_node_dim != ckpt_node_dim:
    attention_check["reason"] = f"KHONG TAO D10 attention: local graph_repo `{rel(GRAPH_REPO)}` node_dim={local_node_dim}, D10 SupCon Best resolved_config node_dim={ckpt_node_dim}; running checkpoint on mismatched graph features would be invalid."
else:
    attention_check["reason"] = "D10 forward returns attention tensors, but no saved D10 attention maps were found and inference was not run in this artifact pass."
write_json(OUT / "d10_attention_check.json", attention_check)
mark_skipped("slot_attention_examples_d10_supcon_best.png", attention_check["reason"])
mark_skipped("class_motif_attention_examples_d10_supcon_best.png", attention_check["reason"])
mark_skipped("slot_attention_wrong_cases.png", attention_check["reason"])

legacy_rows = []
legacy_sources = [
    ("output/d6b_class_part_graph_motif/figures/d6_class_part_attention/class_part_attn_grid.png", "d6b_class_part_attn_grid.png"),
    ("output/d6b_class_part_graph_motif/figures/d6_class_part_attention/class_part_attn_avg_by_true_class.png", "d6b_class_part_attn_avg_by_true_class.png"),
    ("output/d6b_class_part_graph_motif/figures/d6_class_motif_maps/class_pixel_motif_trueclass_avg.png", "d6b_class_pixel_motif_trueclass_avg.png"),
    ("outputs/d9_f0_b_intensity_xy_full_edge/figures/motif_discovery_best/clean_motif_overlay.png", "d9_clean_motif_overlay.png"),
    ("outputs/d9_f0_b_intensity_xy_full_edge/figures/motif_discovery_best/selected_motif_overlay.png", "d9_selected_motif_overlay.png"),
    ("outputs/d9_f0_b_intensity_xy_full_edge/figures/motif_discovery_best/motif_sample_000_top_maps.png", "d9_motif_sample_000_top_maps.png"),
]
for src, dst in legacy_sources:
    sp = ROOT / src
    dp = LEGACY / dst
    if sp.exists():
        shutil.copy2(sp, dp)
        mark_created(dp, f"legacy motif visualization copied from {src}")
        legacy_rows.append({"figure": rel(dp), "source": src, "note": "Legacy motif/attention visualization, not D10 SupCon Best."})
    else:
        mark_skipped(rel(dp), f"KHONG TIM THAY source {src}")
write_csv(LEGACY / "legacy_motif_visualization_sources.csv", legacy_rows, ["figure", "source", "note"])

try:
    image = img_cache.get(0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("FER2013 48x48 image")
    axes[0].axis("off")
    n = 7
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    for i in range(n):
        for j in range(n):
            for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    axes[1].plot([j, nj], [i, ni], color="#9aa0a6", linewidth=0.8, zorder=1)
    axes[1].scatter(xs.flatten(), ys.flatten(), s=35, color="#4e79a7", edgecolor="white", linewidth=0.5, zorder=2)
    axes[1].invert_yaxis()
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    axes[1].set_title("Pixel nodes + neighbor edges")
    axes[2].text(0.05, 0.72, "Nodes: 48 x 48 = 2304", fontsize=12)
    axes[2].text(0.05, 0.52, "Edges: 8-neighbor grid", fontsize=12)
    axes[2].text(0.05, 0.32, "Input tensors:\nx [B,2304,node_dim]\nedge_attr [B,E,edge_dim]", fontsize=11)
    axes[2].axis("off")
    fig.suptitle("Pixel Graph Construction Concept", fontweight="bold")
    savefig(OUT / "pixel_graph_concept.png")
    write_json(OUT / "pixel_graph_concept_source.json", {"source_graph_repo": rel(GRAPH_REPO), "sample_split": "test", "graph_id": 0, "note": "Schematic uses a small grid for readability; real graph has 2304 nodes."})
except Exception as exc:
    mark_skipped("pixel_graph_concept.png", str(exc))

copy_file("output/graph_input_audit/figures/feature_maps_samples/sample_train_0_19950_feature_maps.png", OUT / "node_feature_maps_sample.png", "graph input feature map sample")
combine_images([
    ("output/graph_input_audit/figures/node_feature_histograms/train_intensity.png", "Intensity"),
    ("output/graph_input_audit/figures/node_feature_histograms/train_x_norm.png", "x_norm"),
    ("output/graph_input_audit/figures/node_feature_histograms/train_y_norm.png", "y_norm"),
    ("output/graph_input_audit/figures/node_feature_histograms/train_gx.png", "gx"),
    ("output/graph_input_audit/figures/node_feature_histograms/train_gy.png", "gy"),
    ("output/graph_input_audit/figures/node_feature_histograms/train_grad_mag.png", "grad_mag"),
    ("output/graph_input_audit/figures/node_feature_histograms/train_local_contrast.png", "local_contrast"),
], OUT / "node_feature_histograms.png", "Node Feature Histograms")
combine_images([
    ("output/graph_input_audit/figures/edge_feature_histograms/train_dx.png", "dx"),
    ("output/graph_input_audit/figures/edge_feature_histograms/train_dy.png", "dy"),
    ("output/graph_input_audit/figures/edge_feature_histograms/train_dist.png", "dist"),
    ("output/graph_input_audit/figures/edge_feature_histograms/train_delta_intensity.png", "delta_intensity"),
    ("output/graph_input_audit/figures/edge_feature_histograms/train_intensity_similarity.png", "intensity_similarity"),
], OUT / "edge_feature_histograms.png", "Edge Feature Histograms")

figure_use = {
    "pixel_graph_concept.png": "4.2 - Bieu dien anh FER2013 thanh pixel graph",
    "node_feature_maps_sample.png": "4.2 - Node feature maps",
    "node_feature_histograms.png": "4.2 - Phan bo node features",
    "edge_feature_histograms.png": "4.2 - Phan bo edge features",
    "gnn_macro_f1_progression.png": "4.7 - Ket qua thuc nghiem GNN",
    "d10_phase5_comparison.png": "4.6/4.7 - So sanh SupCon regularization",
    "confusion_matrix_d10_supcon_best.png": "4.8 - Phan tich loi D10 SupCon Best",
    "confusion_matrix_d10_supcon_best_normalized.png": "4.8 - Confusion normalized",
    "confusion_matrix_d10_moderate_reg.png": "4.8 - D10 Moderate-Reg confusion",
    "confusion_matrix_d10_moderate_reg_normalized.png": "4.8 - D10 Moderate-Reg normalized",
    "top_confusion_pairs_d10_supcon_best.png": "4.8 - Cap nham lan lon nhat",
    "hard_classes_f1_d10_supcon_best.png": "4.8 - Lop kho Fear/Sad/Disgust/Neutral",
    "train_val_macro_f1_d10_supcon_best.png": "4.6 - Overfitting theo F1",
    "train_val_loss_d10_supcon_best.png": "4.6 - Overfitting theo loss",
    "overfitting_gap_d10_supcon_best.png": "4.6 - Train-val gap",
    "correct_examples_d10_supcon_best.png": "4.8 - Vi du dung",
    "wrong_examples_d10_supcon_best.png": "4.8 - Vi du sai",
    "wrong_fear_as_sad_or_surprise.png": "4.8 - Loi Fear",
    "wrong_sad_as_neutral_or_fear.png": "4.8 - Loi Sad",
    "wrong_disgust_cases.png": "4.8 - Loi Disgust",
    "ambiguous_wrong_examples.png": "4.8 - Sai do khong chac chan",
    "high_confidence_wrong_examples.png": "4.8 - Sai nhung tu tin cao",
    "confidence_correct_vs_wrong.png": "4.8 - Confidence analysis",
    "confidence_by_class.png": "4.8 - Confidence theo class",
    "top2_margin_wrong_examples.png": "4.8 - Top-2 margin cho mau sai",
}
readme = [
    "# GNN Figures for FER2013 Report",
    "",
    "Generated read-only from existing output/checkpoint/metrics/evaluation artifacts. No training, no model-code edits, no graph_repo rebuild.",
    "",
    "## Figure Inventory",
]
for name, section in figure_use.items():
    readme.append(f"- `{name}`: {'OK' if (OUT / name).exists() else 'KHONG TAO DUOC'}. Suggested section: {section}.")
readme += ["", "## Main Sources"]
for name in ["D10 SupCon Best", "D10 SupCon Moderate-Reg", "D10 SupCon Strong-Reg"]:
    p = metrics_by_name[name]["paths"]
    readme.append(f"- {name}: metrics=`{rel(p['metrics']) if p['metrics'] else 'KHONG TIM THAY'}`, predictions=`{rel(p['predictions']) if p['predictions'] else 'KHONG TIM THAY'}`, history=`{rel(p['history']) if p['history'] else 'KHONG TIM THAY'}`, checkpoint=`{rel(p['ckpt']) if p['ckpt'] else 'KHONG TIM THAY'}`.")
readme += ["", "## D10 Attention Visualization Status", f"- {attention_check['reason']}", "- Legacy motif/attention figures were copied to `legacy_motif_visualization/`; these are explicitly not D10 SupCon Best figures.", "", "## Skipped / Missing"]
if skipped:
    for name, reason in skipped:
        readme.append(f"- `{name}`: {reason}")
else:
    readme.append("- None")
readme += ["", "## Generated Files"]
for path, desc in created:
    readme.append(f"- `{path}`: {desc}")
(OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
mark_created(OUT / "README.md", "README describing figures and sources")

summary = {"output_dir": rel(OUT), "created_count": len(created), "skipped_count": len(skipped), "created": created, "skipped": skipped}
write_json(OUT / "generation_summary.json", summary)

print("OUTPUT_DIR", rel(OUT))
print("CREATED_COUNT", len(created))
for path, desc in created:
    print("CREATED", path, "|", desc)
print("SKIPPED_COUNT", len(skipped))
for name, reason in skipped:
    print("SKIPPED", name, "|", reason)
