"""Evaluate a trained D19-A0 checkpoint without landmark/prior inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.data.collate import collate_d18_graphs
from d18.models.structure_gnn import StructureGNN
from d18.training.train_d18 import build_dataset, load_checkpoint, scientific_resume_signature

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LOCKED_SAMPLE_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"
EQUIVALENT_MODES = ("official", "zero_prior", "shuffle_structure", "forced_fallback", "missing_landmark")


def read_config(run_dir: Path) -> dict[str, Any]:
    yaml_path = run_dir / "resolved_config.yaml"
    json_path = run_dir / "resolved_config.json"
    if yaml_path.exists():
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing resolved config in {run_dir}")


def ece(y: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    result = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            result += float(mask.mean()) * abs(float((prediction[mask] == y[mask]).mean()) - float(confidence[mask].mean()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", choices=("best", "last"), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-manifest", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = read_config(run_dir)
    if str((cfg.get("graph") or {}).get("graph_mode")) != "evidence_only":
        raise RuntimeError("D19 A0 evaluator requires graph_mode=evidence_only")
    dataset = build_dataset(cfg, args.split)
    indices = list(range(len(dataset)))
    locked_hash = None
    if args.sample_manifest:
        manifest = pd.read_csv(args.sample_manifest)
        if "sample_index" not in manifest.columns:
            raise RuntimeError("sample manifest missing sample_index")
        indices = manifest["sample_index"].astype(int).tolist()
        ordered = np.asarray(indices, dtype=np.int64)
        locked_hash = hashlib.sha256(ordered.tobytes()).hexdigest()
        if locked_hash != LOCKED_SAMPLE_SHA256:
            raise RuntimeError(f"Locked sample SHA mismatch: {locked_hash}")
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_d18_graphs)
    first = next(iter(loader))
    if bool((first.edge_type_cat == 2).any()):
        raise RuntimeError("A0 evaluation batch contains structure edges")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = StructureGNN.from_config(cfg, input_dim=first.x_cat.size(1), edge_attr_dim=first.edge_attr_cat.size(1)).to(device)
    checkpoint = run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    payload = load_checkpoint(
        checkpoint,
        model,
        device=device,
        expected_resume_signature=scientific_resume_signature(cfg),
        strict_signature=True,
    )
    model.eval()
    ys, logits_rows, embedding_rows, sample_indices = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if bool((batch.edge_type_cat == 2).any()) or bool((batch.structure_edge_count != 0).any()):
                raise RuntimeError("A0 evaluation encountered structure edges")
            batch = batch.to(device)
            out = model(batch)
            ys.append(batch.y.cpu().numpy())
            logits_rows.append(out["logits"].cpu().numpy())
            embedding_rows.append(out["z_image"].cpu().numpy())
            sample_indices.append(batch.sample_index.cpu().numpy())
    y = np.concatenate(ys)
    logits = np.concatenate(logits_rows)
    embeddings = np.concatenate(embedding_rows)
    sample_index = np.concatenate(sample_indices)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(7), zero_division=0)
    cm = confusion_matrix(y, pred, labels=np.arange(7))
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    summary = {
        "run_name": cfg.get("run_name", run_dir.name),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "split": args.split,
        "count": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "brier_score": float(np.mean(np.sum((probs - np.eye(7)[y]) ** 2, axis=1))),
        "ece_15bin": float(ece(y, probs)),
        "mean_entropy": float(entropy.mean()),
        "mean_margin": float((sorted_probs[:, -1] - sorted_probs[:, -2]).mean()),
        "sample_index_sha256": hashlib.sha256(sample_index.astype(np.int64).tobytes()).hexdigest(),
        "locked_protocol_sha256": locked_hash,
        "equivalent_modes": list(EQUIVALENT_MODES),
        "graph_equivalence_rate": 1.0,
        "prediction_equivalence_rate": 1.0,
        "max_embedding_difference": 0.0,
        "max_logit_difference": 0.0,
        "note": "Counterfactual A0 modes are exact no-ops by construction and are not independent robustness scores.",
    }
    pd.DataFrame([summary]).to_csv(output / "official_metrics.csv", index=False)
    pd.DataFrame([
        {
            "mode": mode,
            "graph_equivalence_rate": 1.0,
            "prediction_equivalence_rate": 1.0,
            "max_embedding_difference": 0.0,
            "max_logit_difference": 0.0,
            "accuracy": summary["accuracy"],
            "macro_f1": summary["macro_f1"],
            "not_an_independent_robustness_score": True,
        }
        for mode in EQUIVALENT_MODES
    ]).to_csv(output / "counterfactual_equivalence_metrics.csv", index=False)
    pd.DataFrame({
        "class_id": np.arange(7), "class_name": CLASS_NAMES, "precision": precision,
        "recall": recall, "f1": f1, "support": support,
    }).to_csv(output / "per_class_metrics.csv", index=False)
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *CLASS_NAMES])
        for class_id, name in enumerate(CLASS_NAMES):
            writer.writerow([name, *cm[class_id].tolist()])
    prediction_frame = pd.DataFrame({
        "sample_index": sample_index, "true_class": y, "predicted_class": pred,
        "correct": (pred == y).astype(int), "entropy": entropy,
        "max_probability": probs.max(axis=1), "margin": sorted_probs[:, -1] - sorted_probs[:, -2],
    })
    for class_id in range(7):
        prediction_frame[f"logit_{class_id}"] = logits[:, class_id]
        prediction_frame[f"prob_{class_id}"] = probs[:, class_id]
    prediction_frame.to_csv(output / "predictions.csv", index=False)
    np.savez_compressed(output / "embeddings.npz", embeddings=embeddings.astype(np.float32), logits=logits.astype(np.float32))
    (output / "evaluation_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "EVALUATION_COMPLETE.json").write_text(json.dumps({"status": "COMPLETE", **summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
