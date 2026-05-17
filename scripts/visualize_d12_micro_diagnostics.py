"""Visualize D12A slot and micro diagnostics for selected validation/test samples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import build_dataloader, load_checkpoint_model, load_config
from data.labels import EMOTION_NAMES
from training.trainer import move_to_device


TARGET_CLASSES = [0, 1, 3, 4]  # Angry, Disgust, Happy, Sad


def _set_diagnostics(config: Dict[str, Any], args: argparse.Namespace) -> None:
    model_cfg = config.setdefault("model", {})
    diag_cfg = dict(model_cfg.get("diagnostics", {}) or {})
    diag_cfg.update(
        {
            "enable_micro_diagnostics": True,
            "save_attention_maps": True,
            "save_node_similarity": True,
            "diagnostic_max_samples": int(args.diagnostic_max_samples),
        }
    )
    model_cfg["diagnostics"] = diag_cfg


def _enable_model_diagnostics(model: torch.nn.Module, args: argparse.Namespace) -> None:
    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "set_micro_diagnostics"):
        target.set_micro_diagnostics(
            enabled=True,
            save_attention_maps=True,
            save_node_similarity=True,
            diagnostic_max_samples=int(args.diagnostic_max_samples),
        )


def _scalar_diagnostics(out: Dict[str, Any]) -> Dict[str, float]:
    diagnostics = out.get("diagnostics", {}) or {}
    values: Dict[str, float] = {}
    for key, value in diagnostics.items():
        if torch.is_tensor(value) and value.numel() == 1:
            values[key] = float(value.detach().cpu().item())
    return values


def _plot_sample(
    *,
    out_path: Path,
    image: np.ndarray,
    slot_maps: np.ndarray,
    virtual_attention: np.ndarray | None,
    class_attn: np.ndarray,
    y_true: int,
    y_pred: int,
    graph_id: int,
    top_k: int = 4,
) -> None:
    pred_top = np.argsort(class_attn[y_pred])[-top_k:][::-1]
    true_top = np.argsort(class_attn[y_true])[-top_k:][::-1]
    slot_ids = list(dict.fromkeys([int(x) for x in list(pred_top) + list(true_top)]))[:top_k]
    rows = 2
    cols = 2 + len(slot_ids)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("image")
    if virtual_attention is not None:
        axes[1, 0].imshow(virtual_attention.reshape(48, 48), cmap="magma")
        axes[1, 0].set_title("virtual")

    axes[0, 1].bar(np.arange(class_attn.shape[1]), class_attn[y_pred])
    axes[0, 1].set_title(f"pred {EMOTION_NAMES[y_pred]}")
    axes[1, 1].bar(np.arange(class_attn.shape[1]), class_attn[y_true])
    axes[1, 1].set_title(f"true {EMOTION_NAMES[y_true]}")

    for col, slot_idx in enumerate(slot_ids, start=2):
        heatmap = slot_maps[slot_idx].reshape(48, 48)
        axes[0, col].imshow(heatmap, cmap="viridis")
        axes[0, col].set_title(f"slot {slot_idx}")
        axes[1, col].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col].imshow(heatmap, cmap="magma", alpha=0.55)
        axes[1, col].set_title(f"overlay {slot_idx}")

    fig.suptitle(f"id={graph_id} true={EMOTION_NAMES[y_true]} pred={EMOTION_NAMES[y_pred]}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph_repo_path", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_per_class", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--diagnostic_max_samples", type=int, default=8)
    args = parser.parse_args()

    plt.switch_backend("Agg")
    config = load_config(args.config, environment=None)
    config.setdefault("paths", {})["graph_repo_path"] = args.graph_repo_path
    config.setdefault("training", {})["device"] = args.device
    config.setdefault("data", {})["batch_size"] = int(args.batch_size)
    config["data"]["num_workers"] = int(args.num_workers)
    _set_diagnostics(config, args)

    model, device, _ = load_checkpoint_model(config, args.checkpoint)
    _enable_model_diagnostics(model, args)
    loader = build_dataloader(config, split=args.split, shuffle=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_counts = defaultdict(int)
    saved: List[Dict[str, Any]] = []
    diag_values: Dict[str, List[float]] = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            out = model(batch)
            logits = out["logits"]
            pred = logits.argmax(dim=1)
            for key, value in _scalar_diagnostics(out).items():
                diag_values[key].append(value)

            x_cpu = batch["x"].detach().cpu()
            y_cpu = batch["y"].detach().cpu()
            pred_cpu = pred.detach().cpu()
            gid_cpu = batch["graph_id"].detach().cpu()
            part_masks = out["part_masks"].detach().cpu()
            class_attn = out["class_motif_attn"].detach().cpu()
            virtual = out.get("virtual_attention")
            virtual_cpu = virtual.detach().cpu() if torch.is_tensor(virtual) else None

            for i in range(x_cpu.shape[0]):
                y_true = int(y_cpu[i].item())
                y_pred = int(pred_cpu[i].item())
                if y_true not in TARGET_CLASSES:
                    continue
                key = EMOTION_NAMES[y_true]
                wrong_rare_key = f"wrong_{key}" if y_true in (0, 1) and y_pred != y_true else None
                if selected_counts[key] >= args.max_per_class and (
                    wrong_rare_key is None or selected_counts[wrong_rare_key] >= args.max_per_class
                ):
                    continue
                graph_id = int(gid_cpu[i].item())
                out_name = (
                    f"sample_{graph_id}_true_{EMOTION_NAMES[y_true]}_pred_{EMOTION_NAMES[y_pred]}.png"
                )
                _plot_sample(
                    out_path=output_dir / out_name,
                    image=x_cpu[i, :, 0].float().reshape(48, 48).numpy(),
                    slot_maps=part_masks[i].float().numpy(),
                    virtual_attention=virtual_cpu[i].float().numpy() if virtual_cpu is not None else None,
                    class_attn=class_attn[i].float().numpy(),
                    y_true=y_true,
                    y_pred=y_pred,
                    graph_id=graph_id,
                )
                selected_counts[key] += 1
                if wrong_rare_key is not None:
                    selected_counts[wrong_rare_key] += 1
                saved.append(
                    {
                        "graph_id": graph_id,
                        "y_true": y_true,
                        "y_true_name": EMOTION_NAMES[y_true],
                        "y_pred": y_pred,
                        "y_pred_name": EMOTION_NAMES[y_pred],
                        "file": out_name,
                    }
                )

            if all(selected_counts[EMOTION_NAMES[c]] >= args.max_per_class for c in TARGET_CLASSES):
                break

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "saved_samples": saved,
        "selected_counts": dict(selected_counts),
        "diagnostics_mean": {
            key: float(np.mean(values)) for key, values in sorted(diag_values.items()) if values
        },
    }
    (output_dir / "micro_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(saved)} figures to {output_dir}")
    print(f"Summary: {output_dir / 'micro_diagnostics_summary.json'}")


if __name__ == "__main__":
    main()
