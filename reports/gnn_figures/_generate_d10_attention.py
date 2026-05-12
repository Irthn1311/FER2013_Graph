import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "gnn_figures"
GRAPH_REPO = ROOT / "artifacts" / "graph-repo" / "graph_repo"
RUN = ROOT / "outputs" / "d10_slot_motif_full_outputs" / "d10_p5_standard" / "d10_p5_stage2_relation_outputs_lan2"
CKPT = RUN / "checkpoints" / "best.pth"
CONFIG = RUN / "resolved_config.yaml"
PRED = RUN / "evaluation" / "predictions.csv"
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def rel(path):
    try:
        return str(Path(path).relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    z = scores - scores.max()
    exp = np.exp(z)
    return exp / exp.sum()


def load_predictions():
    rows = []
    with PRED.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        score_cols = sorted([c for c in reader.fieldnames if c.startswith("score_")], key=lambda c: int(c.split("_")[-1]))
        for r in reader:
            y_true = int(r["y_true"])
            y_pred = int(r["y_pred"])
            scores = np.array([float(r[c]) for c in score_cols], dtype=np.float64)
            probs = softmax(scores)
            rows.append({
                "graph_id": int(r["graph_id"]),
                "y_true": y_true,
                "y_pred": y_pred,
                "true_label": CLASS_NAMES[y_true],
                "pred_label": CLASS_NAMES[y_pred],
                "correct": y_true == y_pred,
                "confidence": float(probs[y_pred]),
            })
    return rows


class GraphCache:
    def __init__(self, graph_root):
        self.graph_root = Path(graph_root)
        self.chunk_cache = {}
        self.shared = torch.load(self.graph_root / "shared" / "shared_graph.pt", map_location="cpu", weights_only=False)

    def sample(self, graph_id):
        chunk_id = int(graph_id) // 500
        idx = int(graph_id) % 500
        if chunk_id not in self.chunk_cache:
            self.chunk_cache[chunk_id] = torch.load(self.graph_root / "test" / f"chunk_{chunk_id:03d}.pt", map_location="cpu", weights_only=False)
        return self.chunk_cache[chunk_id][idx]

    def batch(self, ids):
        samples = [self.sample(i) for i in ids]
        x = torch.stack([s.node_features.float() for s in samples], dim=0)
        dyn = [s.edge_attr_dynamic.float() for s in samples]
        static = self.shared.edge_attr_static.float()
        edge_attr = torch.stack([torch.cat([static, d], dim=1) for d in dyn], dim=0)
        edge_index = self.shared.edge_index.long().unsqueeze(0).expand(len(samples), -1, -1).contiguous()
        node_mask = torch.ones(x.shape[:2], dtype=torch.bool)
        y = torch.tensor([int(s.label) for s in samples], dtype=torch.long)
        graph_id = torch.tensor([int(s.graph_id) for s in samples], dtype=torch.long)
        return {
            "x": x,
            "node_features": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "node_mask": node_mask,
            "y": y,
            "graph_id": graph_id,
        }


def load_model():
    from models.registry import build_model

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model = build_model(cfg["model"])
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, ckpt


def select_rows(rows):
    correct = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    correct = sorted(correct, key=lambda r: r["confidence"], reverse=True)[:3]
    wrong = sorted(wrong, key=lambda r: r["confidence"], reverse=True)[:3]
    return correct + wrong


def select_wrong_focus(rows):
    focus = [r for r in rows if (not r["correct"]) and r["y_true"] in {2, 4, 6}]
    return sorted(focus, key=lambda r: r["confidence"], reverse=True)[:6]


def plot_slot_grid(samples, images, slot_maps, path, title):
    b, k, _ = slot_maps.shape
    fig, axes = plt.subplots(b, k + 1, figsize=((k + 1) * 1.45, b * 1.45))
    axes = np.asarray(axes)
    for row_idx, row in enumerate(samples):
        axes[row_idx, 0].imshow(images[row_idx], cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 0].set_title(f"id {row['graph_id']}\nT:{row['true_label']}\nP:{row['pred_label']}", fontsize=6)
        axes[row_idx, 0].axis("off")
        for slot_idx in range(k):
            ax = axes[row_idx, slot_idx + 1]
            heat = slot_maps[row_idx, slot_idx].reshape(48, 48)
            ax.imshow(images[row_idx], cmap="gray", vmin=0, vmax=1)
            ax.imshow(heat, cmap="magma", alpha=0.62)
            ax.set_title(f"S{slot_idx}", fontsize=6)
            ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_class_motif(samples, attn, path):
    fig, axes = plt.subplots(len(samples), 2, figsize=(7.2, len(samples) * 1.75))
    axes = np.asarray(axes)
    csv_rows = []
    x = np.arange(attn.shape[-1])
    for i, row in enumerate(samples):
        for col, class_key in enumerate(["true", "pred"]):
            cls_id = int(row["y_true"] if class_key == "true" else row["y_pred"])
            weights = attn[i, cls_id]
            axes[i, col].bar(x, weights, color="#4e79a7" if class_key == "true" else "#e15759")
            axes[i, col].set_ylim(0, max(0.01, float(weights.max()) * 1.25))
            axes[i, col].set_xticks(x)
            axes[i, col].set_title(f"id {row['graph_id']} {class_key}: {CLASS_NAMES[cls_id]}", fontsize=8)
            csv_row = {
                "graph_id": row["graph_id"],
                "class_view": class_key,
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
            }
            for m, val in enumerate(weights):
                csv_row[f"motif_{m}"] = float(val)
            csv_rows.append(csv_row)
    fig.suptitle("Class-Motif Attention - D10 SupCon Best", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    fields = ["graph_id", "class_view", "class_id", "class_name", "true_label", "pred_label"] + [f"motif_{i}" for i in range(attn.shape[-1])]
    write_csv(path.with_suffix(".csv"), csv_rows, fields)


def update_readme_and_summary(created_files, status_text):
    readme = OUT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        for name in [
            "slot_attention_examples_d10_supcon_best.png",
            "class_motif_attention_examples_d10_supcon_best.png",
            "slot_attention_wrong_cases.png",
        ]:
            text = text.replace(f"`{name}`: KHONG TAO DUOC.", f"`{name}`: OK.")
        text = text.replace(
            "D10 forward returns attention tensors, but no saved D10 attention maps were found and inference was not run in this artifact pass.",
            status_text,
        )
        text += "\n## D10 Attention Second Pass\n"
        for path in created_files:
            text += f"- `{rel(path)}`: generated by checkpoint inference on existing test graph_repo.\n"
        readme.write_text(text, encoding="utf-8")
    summary = OUT / "generation_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        data["created"].extend([(rel(p), "D10 attention figure/source from checkpoint inference") for p in created_files])
        data["created_count"] = len(data["created"])
        data["skipped"] = [item for item in data["skipped"] if not str(item[0]).startswith(("slot_attention", "class_motif_attention"))]
        data["skipped_count"] = len(data["skipped"])
        summary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    rows = load_predictions()
    selected = select_rows(rows)
    wrong_focus = select_wrong_focus(rows)
    all_ids = [r["graph_id"] for r in selected + wrong_focus]
    graph = GraphCache(GRAPH_REPO)
    model, cfg, ckpt = load_model()

    with torch.no_grad():
        batch = graph.batch(all_ids)
        out = model(batch)

    slot_maps = out["slot_attn_maps"].detach().cpu().float().numpy()
    class_attn = out["class_motif_attn"].detach().cpu().float().numpy()
    images = [graph.sample(i).node_features[:, 0].reshape(48, 48).numpy() for i in all_ids]

    n_selected = len(selected)
    created_files = []
    slot_path = OUT / "slot_attention_examples_d10_supcon_best.png"
    plot_slot_grid(selected, images[:n_selected], slot_maps[:n_selected], slot_path, "Slot Attention Examples - D10 SupCon Best")
    created_files.append(slot_path)
    write_csv(OUT / "slot_attention_examples_d10_supcon_best.csv", selected,
              ["graph_id", "true_label", "pred_label", "y_true", "y_pred", "correct", "confidence"])
    created_files.append(OUT / "slot_attention_examples_d10_supcon_best.csv")

    class_path = OUT / "class_motif_attention_examples_d10_supcon_best.png"
    plot_class_motif(selected, class_attn[:n_selected], class_path)
    created_files.extend([class_path, class_path.with_suffix(".csv")])

    wrong_path = OUT / "slot_attention_wrong_cases.png"
    plot_slot_grid(wrong_focus, images[n_selected:], slot_maps[n_selected:], wrong_path, "Slot Attention Wrong Cases - Fear/Sad/Neutral")
    created_files.append(wrong_path)
    write_csv(OUT / "slot_attention_wrong_cases.csv", wrong_focus,
              ["graph_id", "true_label", "pred_label", "y_true", "y_pred", "correct", "confidence"])
    created_files.append(OUT / "slot_attention_wrong_cases.csv")

    check = {
        "model_forward_returns": ["slot_attn_maps", "class_motif_attn"],
        "checkpoint": rel(CKPT),
        "config": rel(CONFIG),
        "graph_repo": rel(GRAPH_REPO),
        "selected_graph_ids": all_ids,
        "status": "created",
        "note": "Inference only on existing test graph_repo; no training.",
    }
    (OUT / "d10_attention_check.json").write_text(json.dumps(check, indent=2, ensure_ascii=False), encoding="utf-8")
    created_files.append(OUT / "d10_attention_check.json")
    update_readme_and_summary(created_files, "D10 attention visualizations were generated by inference-only checkpoint loading on existing test graph_repo.")
    for path in created_files:
        print("CREATED", rel(path))


if __name__ == "__main__":
    main()
