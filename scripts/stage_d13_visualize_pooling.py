"""Visualize D13A local assignment pooling on a dataset split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device
from data.labels import EMOTION_NAMES
from models.d13_hierarchical_reduction_model import D13HierarchicalReductionModel
from training.trainer import move_to_device


def _load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _dominant_assignment(payload: Dict[str, torch.Tensor], num_nodes: int) -> np.ndarray:
    anchors = payload["anchor_index"].detach().cpu()
    weights = payload["weights"].detach().cpu()
    dominant = anchors.gather(1, weights.argmax(dim=1, keepdim=True)).squeeze(1).numpy()
    if dominant.shape[0] != num_nodes:
        out = np.full((num_nodes,), -1, dtype=np.int64)
        pix = payload["pixel_index"].detach().cpu().numpy()
        out[pix] = dominant
        dominant = out
    return dominant.reshape(48, 48)


def _region_area_map(payload: Dict[str, torch.Tensor], grid_size: int) -> np.ndarray:
    anchors = payload["anchor_index"].detach().cpu().numpy().reshape(-1)
    weights = payload["weights"].detach().cpu().numpy().reshape(-1)
    area = np.zeros((grid_size * grid_size,), dtype=np.float32)
    np.add.at(area, anchors, weights)
    return area.reshape(grid_size, grid_size)


@torch.no_grad()
def visualize(config: Dict[str, Any], checkpoint: str | Path, split: str, output_dir: str | Path, max_batches: int = 8) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config=config)
    loader = build_dataloader(config, split=split, shuffle=False)
    model = D13HierarchicalReductionModel.from_config(config.get("model", {})).to(device)
    ckpt = _load_checkpoint(checkpoint, device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    if hasattr(model.reduction, "set_save_visualization"):
        model.reduction.set_save_visualization(True)
    model.eval()
    saved_per_class = {idx: 0 for idx in range(7)}
    max_per_class = int(config.get("visualization", {}).get("max_per_class", 3))
    grid_size = int(config.get("model", {}).get("pooling", {}).get("grid_size", 12))

    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx > int(max_batches):
            break
        batch = move_to_device(batch, device)
        out = model(batch)
        probs = torch.softmax(out["logits"], dim=1)
        pred = probs.argmax(dim=1)
        payloads = out.get("aux", {}).get("assignment_maps", [])
        if not payloads:
            raise RuntimeError("No assignment_maps returned; visualization flag did not reach LocalAssignmentPool")
        x_cpu = batch["x"].detach().cpu()
        y_cpu = batch["y"].detach().cpu()
        pred_cpu = pred.detach().cpu()
        conf_cpu = probs.max(dim=1).values.detach().cpu()
        attention = out.get("region_attention")
        attention_cpu = attention.detach().cpu() if torch.is_tensor(attention) else None
        for i, payload in enumerate(payloads):
            y = int(y_cpu[i].item())
            if saved_per_class[y] >= max_per_class:
                continue
            image = x_cpu[i, :, 0].reshape(48, 48).numpy()
            assign_map = _dominant_assignment(payload, num_nodes=48 * 48)
            area_map = _region_area_map(payload, grid_size=grid_size)
            fig, axes = plt.subplots(1, 4 if attention_cpu is not None else 3, figsize=(12, 3.2))
            axes[0].imshow(image, cmap="gray")
            axes[0].set_title("image")
            axes[1].imshow(assign_map, cmap="tab20")
            axes[1].set_title("assignment")
            axes[2].imshow(area_map, cmap="viridis")
            axes[2].set_title("region area")
            if attention_cpu is not None:
                start = i * grid_size * grid_size
                attn_map = attention_cpu[start : start + grid_size * grid_size].reshape(grid_size, grid_size).numpy()
                axes[3].imshow(attn_map, cmap="magma")
                axes[3].set_title("attention")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(
                f"true={EMOTION_NAMES[y]} pred={EMOTION_NAMES[int(pred_cpu[i])]} conf={float(conf_cpu[i]):.3f}",
                fontsize=10,
            )
            name = f"{split}_class{y}_{EMOTION_NAMES[y].lower()}_{saved_per_class[y]:02d}.png"
            fig.tight_layout()
            fig.savefig(output_dir / name, dpi=140)
            plt.close(fig)
            saved_per_class[y] += 1
        if all(v >= max_per_class for v in saved_per_class.values()):
            break
    print(f"Saved D13 pooling figures to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=8)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    visualize(config, args.checkpoint, args.split, args.output_dir, max_batches=args.max_batches)


if __name__ == "__main__":
    main()

