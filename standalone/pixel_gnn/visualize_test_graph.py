#!/usr/bin/env python3
"""Script to evaluate and visualize Hierarchical Pixel-to-Motif Graphs on FER2013 test set.

Generates:
  1. sample_XXX_original.png (Grayscale 48x48 face)
  2. sample_XXX_pixel_graph.png (Pixel graph 8-neighbor overlay)
  3. sample_XXX_motif_map.png (Motif assignment map 48x48)
  4. sample_XXX_overlay.png (Colored motif map blended on face)
  5. sample_XXX_motif_graph.png (Coarsened Motif Graph with node sizes and edge thicknesses)
  6. sample_XXX_attention_weights.png (Motif-to-motif attention heatmap)
  7. sample_XXX_summary.png (Side-by-side composite panel)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
STANDALONE_PIXEL = ROOT / "standalone/pixel_gnn"
if str(STANDALONE_PIXEL) not in sys.path:
    sys.path.insert(0, str(STANDALONE_PIXEL))

from pixel_gnn.dataset import FERPixelDataset
from pixel_gnn.grid import StaticGridTopology
from pixel_gnn.models import build_model
from pixel_gnn.utils import EMOTION_LABELS, load_config


def plot_single_sample(
    sample_idx: int,
    image_48: np.ndarray,
    label: int,
    pred: int,
    probs: np.ndarray,
    assignment: np.ndarray,          # [2304, K]
    A_motif: np.ndarray,             # [K, K]
    attn_weights: np.ndarray | None, # [K, K]
    output_dir: Path,
    prefix: str = "sample",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    num_motifs = assignment.shape[1]

    # Map each pixel to argmax motif
    pixel_motifs = np.argmax(assignment, axis=-1).reshape(48, 48)

    # Compute physical center coordinates for each motif on [0, 47] image space
    yy, xx = np.mgrid[0:48, 0:48]
    x_flat = xx.reshape(-1)
    y_flat = yy.reshape(-1)

    motif_weights = np.sum(assignment, axis=0) + 1e-6  # [K]
    motif_x = np.sum(assignment * x_flat[:, None], axis=0) / motif_weights
    motif_y = np.sum(assignment * y_flat[:, None], axis=0) / motif_weights

    # Distinct colormap for motifs
    cmap = plt.cm.get_cmap("tab20", min(num_motifs, 20))
    colors = [cmap(i % 20) for i in range(num_motifs)]

    true_name = EMOTION_LABELS.get(label, str(label))
    pred_name = EMOTION_LABELS.get(pred, str(pred))
    conf = float(probs[pred]) * 100.0

    tag = f"{prefix}_{sample_idx:03d}"

    # -------------------------------------------------------------
    # 1. Original Image
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(image_48, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Original | True: {true_name}\nPred: {pred_name} ({conf:.1f}%)", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_original.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # 2. Pixel Graph Overlay
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(image_48, cmap="gray", vmin=0, vmax=1, alpha=0.85)
    # Draw subsampled 8-neighbor grid connections (every 4 pixels for visual clarity)
    step = 3
    for r in range(0, 48, step):
        for c in range(0, 48, step):
            for dr, dc in [(0, step), (step, 0), (step, step), (step, -step)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 48 and 0 <= nc < 48:
                    ax.plot([c, nc], [r, nr], color="cyan", alpha=0.35, linewidth=0.6)
    ax.scatter(xx[::step, ::step], yy[::step, ::step], s=3, color="yellow", alpha=0.6)
    ax.set_title("Pixel Graph (8-Neighbors Grid)", fontsize=10)
    ax.set_xlim(-0.5, 47.5)
    ax.set_ylim(47.5, -0.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_pixel_graph.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # 3. Motif Assignment Map
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    im = ax.imshow(pixel_motifs, cmap="tab20", vmin=0, vmax=max(num_motifs - 1, 1))
    ax.set_title(f"Motif Clustering Map (K={num_motifs})", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_motif_map.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # 4. Colored Overlay on Face
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(image_48, cmap="gray", vmin=0, vmax=1)
    ax.imshow(pixel_motifs, cmap="tab20", vmin=0, vmax=max(num_motifs - 1, 1), alpha=0.45)
    ax.set_title("Overlay: Face + Motif Clusters", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_overlay.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # 5. Coarsened Motif Graph
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(image_48, cmap="gray", vmin=0, vmax=1, alpha=0.3)

    # Draw edges between motifs based on A_motif
    A_sym = 0.5 * (A_motif + A_motif.T)
    # Zero out self-loops for drawing edges
    np.fill_diagonal(A_sym, 0.0)
    max_edge = np.max(A_sym) + 1e-6
    edge_threshold = 0.05 * max_edge

    for i in range(num_motifs):
        for j in range(i + 1, num_motifs):
            w = A_sym[i, j]
            if w > edge_threshold:
                norm_w = float(w / max_edge)
                ax.plot(
                    [motif_x[i], motif_x[j]],
                    [motif_y[i], motif_y[j]],
                    color="lime",
                    alpha=min(max(norm_w * 1.2, 0.2), 0.9),
                    linewidth=max(norm_w * 3.5, 0.8),
                    zorder=2,
                )

    # Draw motif nodes
    max_node_weight = np.max(motif_weights) + 1e-6
    for k in range(num_motifs):
        size = 40 + 200 * (motif_weights[k] / max_node_weight)
        ax.scatter(
            motif_x[k],
            motif_y[k],
            s=size,
            color=colors[k],
            edgecolors="white",
            linewidth=1.2,
            zorder=3,
        )
        if motif_weights[k] / max_node_weight > 0.1:
            ax.text(
                motif_x[k],
                motif_y[k],
                str(k),
                color="black",
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=4,
            )

    ax.set_title(f"Motif Graph (Nodes={num_motifs}, Edges via $A_{{motif}}$)", fontsize=10)
    ax.set_xlim(-1, 48)
    ax.set_ylim(48, -1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_motif_graph.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # 6. Motif Attention Heatmap
    # -------------------------------------------------------------
    if attn_weights is not None:
        fig, ax = plt.subplots(figsize=(4.5, 4), dpi=150)
        im = ax.imshow(attn_weights, cmap="viridis")
        ax.set_title("Motif GNN Attention Weights ($K \\times K$)", fontsize=10)
        ax.set_xlabel("Key Motif", fontsize=8)
        ax.set_ylabel("Query Motif", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"{tag}_attention_weights.png")
        plt.close(fig)

    # -------------------------------------------------------------
    # 7. Summary Composite 5-Panel Figure
    # -------------------------------------------------------------
    cols = 5 if attn_weights is not None else 4
    fig, axs = plt.subplots(1, cols, figsize=(3.5 * cols, 3.5), dpi=160)

    axs[0].imshow(image_48, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title(f"Original\nTrue: {true_name} | Pred: {pred_name}", fontsize=9)
    axs[0].axis("off")

    axs[1].imshow(pixel_motifs, cmap="tab20", vmin=0, vmax=max(num_motifs - 1, 1))
    axs[1].set_title(f"Motif Map\n({num_motifs} clusters)", fontsize=9)
    axs[1].axis("off")

    axs[2].imshow(image_48, cmap="gray", vmin=0, vmax=1)
    axs[2].imshow(pixel_motifs, cmap="tab20", vmin=0, vmax=max(num_motifs - 1, 1), alpha=0.45)
    axs[2].set_title("Overlay\nFace + Motifs", fontsize=9)
    axs[2].axis("off")

    # Coarsened Motif Graph panel
    axs[3].imshow(image_48, cmap="gray", vmin=0, vmax=1, alpha=0.3)
    for i in range(num_motifs):
        for j in range(i + 1, num_motifs):
            w = A_sym[i, j]
            if w > edge_threshold:
                norm_w = float(w / max_edge)
                axs[3].plot(
                    [motif_x[i], motif_x[j]],
                    [motif_y[i], motif_y[j]],
                    color="lime",
                    alpha=min(max(norm_w * 1.2, 0.2), 0.9),
                    linewidth=max(norm_w * 2.5, 0.6),
                )
    for k in range(num_motifs):
        size = 30 + 150 * (motif_weights[k] / max_node_weight)
        axs[3].scatter(motif_x[k], motif_y[k], s=size, color=colors[k], edgecolors="white", linewidth=0.8)
    axs[3].set_title("Motif Graph\n$A_{motif} = S^T A_{pixel} S$", fontsize=9)
    axs[3].set_xlim(-1, 48)
    axs[3].set_ylim(48, -1)
    axs[3].axis("off")

    if cols == 5 and attn_weights is not None:
        im5 = axs[4].imshow(attn_weights, cmap="viridis")
        axs[4].set_title("Motif Attention\n($K \\times K$ weights)", fontsize=9)
        axs[4].set_xlabel("Key", fontsize=7)
        axs[4].set_ylabel("Query", fontsize=7)
        fig.colorbar(im5, ax=axs[4], fraction=0.046, pad=0.04)

    fig.suptitle(f"Sample #{sample_idx:03d}: Emotion: {true_name} -> Predicted: {pred_name} ({conf:.1f}%)", fontsize=11, y=0.98)
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_summary.png")
    plt.close(fig)


def run_visualization(
    config_path: str | Path,
    fer_csv: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path = "outputs/test_graph_visualization",
    num_samples: int = 7,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    model = build_model(config)

    # Look for checkpoint
    if checkpoint_path is None:
        output_root = Path(config.get("paths", {}).get("output_root", "outputs"))
        candidates_ckpt = [
            Path(output_dir) / "best_val_accuracy.weights.h5",
            output_root / "best_val_accuracy.weights.h5",
            Path("/kaggle/working/outputs/pixel_motif_graph/best_val_accuracy.weights.h5"),
            Path("outputs/pixel_motif_graph/best_val_accuracy.weights.h5"),
        ]
        for ckpt in candidates_ckpt:
            if ckpt.is_file():
                checkpoint_path = ckpt
                break

    # Load dataset
    if fer_csv is None:
        fer_csv = config.get("paths", {}).get("fer_csv")

    if not fer_csv or not Path(fer_csv).exists():
        candidates = [
            Path("/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/test.csv"),
            Path("/kaggle/input/fer13-split/test.csv"),
            Path("/kaggle/input/fer2013-split/test.csv"),
            Path("/kaggle/input/fer2013/test.csv"),
            Path("data/test.csv"),
        ]
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for p in kaggle_input.rglob("test.csv"):
                candidates.insert(0, p)
            for p in kaggle_input.rglob("train.csv"):
                candidates.insert(0, p)

        for c in candidates:
            if c.is_file():
                fer_csv = c
                print(f"[AUTO-DETECT] Found dataset at: {fer_csv}")
                break

    if not fer_csv or not Path(fer_csv).exists():
        print("[VIS] No test.csv found! Generating synthetic samples for visualization test...")
        from pixel_gnn.smoke import make_dummy_batch
        batch = make_dummy_batch(batch_size=min(num_samples, 4))
        images = batch["image_48"].numpy()
        labels = batch["labels"].numpy()
    else:
        test_dataset = FERPixelDataset(fer_csv, "test")
        # Select representative samples from distinct classes
        sample_indices = []
        seen_labels = set()
        for idx in range(len(test_dataset)):
            lbl = test_dataset[idx]["label"]
            if lbl not in seen_labels:
                seen_labels.add(lbl)
                sample_indices.append(idx)
            if len(sample_indices) >= num_samples:
                break

        # If not enough distinct classes found in head, fill remaining
        idx = 0
        while len(sample_indices) < num_samples and idx < len(test_dataset):
            if idx not in sample_indices:
                sample_indices.append(idx)
            idx += 1

        samples = [test_dataset[i] for i in sample_indices]
        node_features = np.stack([s["node_features"] for s in samples], axis=0)
        labels = np.array([s["label"] for s in samples], dtype=np.int64)
        images = np.stack([s["image_48"] for s in samples], axis=0)

        batch = {
            "node_features": tf.convert_to_tensor(node_features, dtype=tf.float32),
            "labels": tf.convert_to_tensor(labels, dtype=tf.int64),
            "image_48": tf.convert_to_tensor(images, dtype=tf.float32),
        }

    # Build model variables by running one forward pass
    out = model(batch, training=False)

    if checkpoint_path and Path(checkpoint_path).is_file():
        print(f"[VIS] Loading checkpoint weights from {checkpoint_path}")
        model.load_weights(str(checkpoint_path))
        out = model(batch, training=False)
    else:
        print("[VIS] No checkpoint weights loaded (running with initialized weights).")

    probs = out["probabilities"].numpy()
    preds = out["predictions"].numpy()
    assignments = out["motif_assignment"].numpy() if out.get("motif_assignment") is not None else None
    A_motifs = out["A_motif"].numpy() if out.get("A_motif") is not None else None
    attn_weights = out["motif_gnn_attention"].numpy() if out.get("motif_gnn_attention") is not None else None

    if assignments is None:
        raise ValueError("Model does not produce motif_assignment. Check config pooling_type: motif.")

    print(f"[VIS] Generating visualizations for {len(images)} samples in: {output_path}")
    for i in range(len(images)):
        plot_single_sample(
            sample_idx=i + 1,
            image_48=images[i],
            label=int(labels[i]),
            pred=int(preds[i]),
            probs=probs[i],
            assignment=assignments[i],
            A_motif=A_motifs[i] if A_motifs is not None else np.eye(assignments.shape[-1]),
            attn_weights=attn_weights[i] if attn_weights is not None else None,
            output_dir=output_path,
        )
        print(f"  [OK] Sample {i+1:03d} visualizations saved.")

    print(f"[VIS SUCCESS] All visualizations successfully saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(STANDALONE_PIXEL / "configs/fer2013_pixel_motif_graph_fast_seed42.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument("--fer-csv", help="Path to test.csv or dataset directory")
    parser.add_argument("--checkpoint", help="Path to best_val_accuracy.weights.h5")
    parser.add_argument("--output-dir", default="outputs/test_graph_visualization", help="Output directory for PNG images")
    parser.add_argument("--num-samples", type=int, default=7, help="Number of test samples to visualize")
    args = parser.parse_args()

    run_visualization(
        config_path=args.config,
        fer_csv=args.fer_csv,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
