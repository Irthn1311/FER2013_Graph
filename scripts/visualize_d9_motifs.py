"""Visualize D9 motif maps and motif-relation attention."""

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

from common import apply_cli_overrides, build_dataloader, load_config, resolve_device, resolve_existing_path, resolve_path  # noqa: E402
from data.labels import EMOTION_NAMES  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.trainer import move_to_device  # noqa: E402
from utils.feature_ablation import apply_feature_ablation, assert_feature_dims, log_feature_ablation  # noqa: E402


def _update_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = apply_cli_overrides(config, args)
    paths = dict(cfg.get("paths", {}) or {})
    if getattr(args, "graph_repo_path", None):
        paths["graph_repo_path"] = str(args.graph_repo_path)
    cfg["paths"] = paths
    return cfg


def _save_sample(
    *,
    image: np.ndarray,
    maps: np.ndarray,
    weights: np.ndarray,
    relation_attn: np.ndarray | None,
    y_true: int,
    y_pred: int,
    confidence: float,
    graph_id: int,
    output_path: Path,
    top_k: int,
) -> None:
    top_indices = np.argsort(weights)[-int(top_k) :][::-1]
    cols = max(3, len(top_indices) + 1)
    rows = 2 if relation_attn is not None else 1
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.8 * rows))
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title(
        f"id={graph_id}\ny={EMOTION_NAMES[y_true]}\np={EMOTION_NAMES[y_pred]} {confidence:.2f}",
        fontsize=8,
    )
    for col, motif_idx in enumerate(top_indices, start=1):
        axes[0, col].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col].imshow(maps[motif_idx], cmap="magma", alpha=0.65)
        axes[0, col].set_title(f"m{motif_idx} w={weights[motif_idx]:.3f}", fontsize=8)
    if relation_attn is not None:
        shown = relation_attn[np.ix_(top_indices, top_indices)]
        axes[1, 0].imshow(relation_attn, cmap="viridis")
        axes[1, 0].set_title("relation all", fontsize=8)
        axes[1, 1].imshow(shown, cmap="viridis")
        axes[1, 1].set_title("relation top", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def run_visualize(config: Dict[str, Any], args: argparse.Namespace) -> None:
    device = resolve_device(args.device, config=config)
    feature_ablation_cfg = dict(config.get("feature_ablation", {}) or {})
    model_cfg = dict(config.get("model", {}) or {})
    log_feature_ablation(
        feature_ablation_cfg,
        model_node_dim=int(model_cfg.get("node_dim", 3)),
        model_edge_dim=int(model_cfg.get("edge_dim", 5)),
    )
    model = build_model(model_cfg).to(device)
    checkpoint_path = resolve_existing_path(args.checkpoint)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    output_dir = resolve_path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        output_dir = (checkpoint_path.parent.parent / "figures" / "d9_motifs").resolve()
    loader = build_dataloader(config, split=str(args.split), shuffle=False)
    saved = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        batch = apply_feature_ablation(batch, feature_ablation_cfg)
        assert_feature_dims(
            batch,
            node_dim=int(model_cfg.get("node_dim", 3)),
            edge_dim=int(model_cfg.get("edge_dim", 5)),
        )
        out = model(batch)
        probs = torch.softmax(out["logits"].detach().float(), dim=1)
        pred = probs.argmax(dim=1)
        relation_attn = out.get("motif_relation_attention")
        for i in range(batch["x"].shape[0]):
            if saved >= int(args.max_samples):
                print(f"[Output] saved={saved} dir={output_dir}")
                return
            image = batch["x"][i, :, 0].detach().float().cpu().reshape(48, 48).numpy()
            maps = out["motif_maps"][i].detach().float().cpu().numpy()
            weights = out["selection_weights"][i].detach().float().cpu().numpy()
            attn_np = None
            if torch.is_tensor(relation_attn):
                attn_np = relation_attn[i].detach().float().cpu().numpy()
            gid = int(batch["graph_id"][i].detach().cpu())
            y_true = int(batch["y"][i].detach().cpu())
            y_pred = int(pred[i].detach().cpu())
            conf = float(probs[i, y_pred].detach().cpu())
            _save_sample(
                image=image,
                maps=maps,
                weights=weights,
                relation_attn=attn_np,
                y_true=y_true,
                y_pred=y_pred,
                confidence=conf,
                graph_id=gid,
                output_path=output_dir / f"{args.split}_sample_{saved:03d}_gid_{gid}.png",
                top_k=int(args.top_k),
            )
            saved += 1
    print(f"[Output] saved={saved} dir={output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _update_config(load_config(args.config, environment=args.environment), args)
    run_visualize(config, args)


if __name__ == "__main__":
    main()
