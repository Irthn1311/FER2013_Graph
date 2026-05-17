"""Export D12A embeddings and prediction metadata for offline rare-class analysis."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def _mean_pool(value: torch.Tensor | None) -> np.ndarray | None:
    if not torch.is_tensor(value):
        return None
    tensor = value.detach().float()
    if tensor.ndim == 3:
        tensor = tensor.mean(dim=1)
    return tensor.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph_repo_path", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config, environment=None)
    config.setdefault("paths", {})["graph_repo_path"] = args.graph_repo_path
    config.setdefault("training", {})["device"] = args.device
    config.setdefault("data", {})["batch_size"] = int(args.batch_size)
    config["data"]["num_workers"] = int(args.num_workers)

    model, device, _ = load_checkpoint_model(config, args.checkpoint)
    loader = build_dataloader(config, split=args.split, shuffle=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, List[np.ndarray]] = {
        "motif": [],
        "local_context": [],
        "local_refined": [],
        "global": [],
    }
    labels: List[int] = []
    preds: List[int] = []
    metadata: List[Dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            batch = move_to_device(batch, device)
            out = model(batch)
            pred = out["logits"].argmax(dim=1).detach().cpu().numpy()
            y = batch["y"].detach().cpu().numpy()
            graph_id = batch["graph_id"].detach().cpu().numpy()
            sample_idx = batch["sample_idx"].detach().cpu().numpy()

            for name, key in (
                ("motif", "motif_embeddings"),
                ("local_context", "local_context"),
                ("local_refined", "local_refined"),
                ("global", "global_context"),
            ):
                pooled = _mean_pool(out.get(key))
                if pooled is not None:
                    arrays[name].append(pooled)

            labels.extend(int(v) for v in y.tolist())
            preds.extend(int(v) for v in pred.tolist())
            for i in range(len(y)):
                metadata.append(
                    {
                        "sample_idx": int(sample_idx[i]),
                        "graph_id": int(graph_id[i]),
                        "label": int(y[i]),
                        "label_name": EMOTION_NAMES[int(y[i])],
                        "pred": int(pred[i]),
                        "pred_name": EMOTION_NAMES[int(pred[i])],
                        "correct": int(pred[i] == y[i]),
                    }
                )

    for name, chunks in arrays.items():
        if chunks:
            filename = {
                "motif": "embeddings_motif.npy",
                "local_context": "embeddings_local_context.npy",
                "local_refined": "embeddings_local_refined.npy",
                "global": "embeddings_global.npy",
            }[name]
            np.save(output_dir / filename, np.concatenate(chunks, axis=0))
    np.save(output_dir / "labels.npy", np.asarray(labels, dtype=np.int64))
    np.save(output_dir / "preds.npy", np.asarray(preds, dtype=np.int64))

    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_idx", "graph_id", "label", "label_name", "pred", "pred_name", "correct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Exported {len(metadata)} rows to {output_dir}")


if __name__ == "__main__":
    main()
