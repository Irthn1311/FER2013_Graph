"""Post-training OFIX18 counterfactual and edge-family evaluation.

Reuses the existing OFIX18 predecision graph/model audit helpers. It does not
train or redefine the D18 model.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.data.structure_graph_cache import load_d18_graph_cache
from d18.scripts.audit_ofix18_predecision import (
    CLASS_NAMES,
    ablate_graph,
    ece_score,
    entropy_np,
    infer_graphs,
    load_model,
    load_prior,
    mode_prior,
    rebuild_structure_from_cache,
    softmax_np,
)

COUNTERFACTUALS = (
    "official",
    "remove_structure",
    "shuffle_structure",
    "permute_structure_destinations",
    "degree_matched_random_structure",
)
EDGE_ABLATIONS = {
    "full_official": "full",
    "remove_structure": "no_structure",
    "remove_knn": "no_knn",
    "remove_local": "no_local",
    "keep_local_only": "local_only",
    "keep_local_knn": "no_structure",
    "keep_local_structure": "no_knn",
    "permute_structure_destinations": "permute_structure",
    "degree_matched_random_structure": "degree_swap_structure",
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def donor_permutation(length: int, seed: int) -> np.ndarray:
    # Match the existing predecision audit semantics exactly.
    return np.random.default_rng(seed).permutation(length)


def summarize(
    mode: str,
    y: np.ndarray,
    logits: np.ndarray,
    official_pred: np.ndarray,
    official_macro: float | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    probs = softmax_np(logits)
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    cm = confusion_matrix(y, pred, labels=np.arange(len(CLASS_NAMES)))
    macro = float(f1.mean())
    weighted = float(np.average(f1, weights=support)) if int(support.sum()) else 0.0
    row: dict[str, Any] = {
        "mode": mode,
        "count": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": macro,
        "weighted_f1": weighted,
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "brier_score": float(np.mean(np.sum((probs - np.eye(len(CLASS_NAMES))[y]) ** 2, axis=1))),
        "ece_15bin": float(ece_score(y, probs)),
        "mean_predictive_entropy": float(entropy_np(probs).mean()),
        "prediction_agreement_with_official": float((pred == official_pred).mean()),
        "official_to_counterfactual_macro_f1_drop": (
            float(official_macro - macro) if official_macro is not None else 0.0
        ),
        "correct_to_wrong": int(np.sum((official_pred == y) & (pred != y))),
        "wrong_to_correct": int(np.sum((official_pred != y) & (pred == y))),
        "confusion_matrix_json": json.dumps(cm.tolist()),
    }
    for class_id, name in enumerate(CLASS_NAMES):
        row[f"precision_{name}"] = float(precision[class_id])
        row[f"recall_{name}"] = float(recall[class_id])
        row[f"f1_{name}"] = float(f1[class_id])
        row[f"support_{name}"] = int(support[class_id])
    return row, pred, probs


def write_confusion(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *CLASS_NAMES])
        for index, name in enumerate(CLASS_NAMES):
            writer.writerow([name, *matrix[index].tolist()])


def load_official_graphs(files: list[Path], cache_dir: Path) -> list[Any]:
    graphs = []
    for file in files:
        cache_file = cache_dir / "test" / file.name
        if not cache_file.exists():
            raise FileNotFoundError(cache_file)
        graph = load_d18_graph_cache(cache_file)
        graphs.append(graph)
    return graphs


def counterfactual_graphs(
    mode: str,
    official: list[Any],
    files: list[Path],
    donors: np.ndarray,
    graph_cfg: dict[str, Any],
    seed: int,
) -> list[Any]:
    if mode == "official":
        return official
    if mode == "remove_structure":
        return [ablate_graph(graph, "no_structure", seed + int(graph.sample_index)) for graph in official]
    if mode == "shuffle_structure":
        result = []
        for index, graph in enumerate(official):
            base = load_prior(files[index])
            donor = load_prior(files[int(donors[index])])
            shuffled = mode_prior(base, "shuffle_prior", donor)
            result.append(rebuild_structure_from_cache(graph, shuffled, graph_cfg))
        return result
    operation = {
        "permute_structure_destinations": "permute_structure",
        "degree_matched_random_structure": "degree_swap_structure",
    }[mode]
    return [
        ablate_graph(graph, operation, seed + int(graph.sample_index) * 31)
        for graph in official
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--graph_cache_dir", required=True)
    parser.add_argument("--checkpoint", choices=("best", "last"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        cfg_path = run_dir / "resolved_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing resolved config in {run_dir}")
    cfg = load_yaml(cfg_path) if cfg_path.suffix == ".yaml" else json.loads(cfg_path.read_text(encoding="utf-8"))
    checkpoint = run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    files = sorted((Path(args.prior_dir) / "test").glob("*.npz"))
    if args.max_samples is not None:
        files = files[: int(args.max_samples)]
    if not files:
        raise FileNotFoundError("no test prior files")
    official_graphs = load_official_graphs(files, Path(args.graph_cache_dir))
    y = np.asarray([int(graph.y) for graph in official_graphs], dtype=np.int64)
    sample_index = np.asarray([int(graph.sample_index) for graph in official_graphs], dtype=np.int64)
    donors = donor_permutation(len(files), int(args.seed))
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    spec = {
        "cfg": cfg,
        "model_family": "d18",
        "checkpoint_path": checkpoint,
    }
    model = load_model(spec, device)

    logits_by_mode: dict[str, np.ndarray] = {}
    graphs_by_mode: dict[str, list[Any]] = {}
    for mode in COUNTERFACTUALS:
        graphs = counterfactual_graphs(
            mode, official_graphs, files, donors, cfg.get("graph", {}) or {}, int(args.seed)
        )
        graphs_by_mode[mode] = graphs
        logits_by_mode[mode], _ = infer_graphs(model, graphs, device, int(args.batch_size))
        print(json.dumps({"event": "ofix18_eval_mode_done", "mode": mode, "count": len(graphs)}), flush=True)

    official_probs = softmax_np(logits_by_mode["official"])
    official_pred = official_probs.argmax(axis=1)
    _, _, official_f1, _ = precision_recall_fscore_support(
        y, official_pred, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    official_macro = float(official_f1.mean())
    rows, prediction_rows = [], []
    for mode in COUNTERFACTUALS:
        row, pred, probs = summarize(mode, y, logits_by_mode[mode], official_pred, official_macro)
        row.update({"run_name": run_dir.name, "checkpoint": args.checkpoint})
        rows.append(row)
        matrix = np.asarray(json.loads(row["confusion_matrix_json"]), dtype=np.int64)
        write_confusion(output / f"confusion_matrix_{mode}.csv", matrix)
        for i in range(len(y)):
            prediction_rows.append(
                {
                    "sample_index": int(sample_index[i]),
                    "true_class": int(y[i]),
                    "mode": mode,
                    "predicted_class": int(pred[i]),
                    "confidence": float(probs[i].max()),
                    "official_predicted_class": int(official_pred[i]),
                }
            )
    pd.DataFrame(rows).to_csv(output / "counterfactual_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "counterfactual_predictions.csv", index=False)

    ablation_rows = []
    logits_cache: dict[str, np.ndarray] = {
        "full": logits_by_mode["official"],
        "no_structure": logits_by_mode["remove_structure"],
        "permute_structure": logits_by_mode["permute_structure_destinations"],
        "degree_swap_structure": logits_by_mode["degree_matched_random_structure"],
    }
    for name, operation in EDGE_ABLATIONS.items():
        if operation not in logits_cache:
            graphs = [
                ablate_graph(graph, operation, int(args.seed) + int(graph.sample_index) * 31)
                for graph in official_graphs
            ]
            logits_cache[operation], _ = infer_graphs(model, graphs, device, int(args.batch_size))
        row, _, _ = summarize(name, y, logits_cache[operation], official_pred, official_macro)
        row.update({"run_name": run_dir.name, "checkpoint": args.checkpoint, "operation": operation})
        ablation_rows.append(row)
    pd.DataFrame(ablation_rows).to_csv(output / "edge_family_ablation_metrics.csv", index=False)

    payload = {
        "status": "COMPLETE",
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint),
        "prior_dir": str(args.prior_dir),
        "graph_cache_dir": str(args.graph_cache_dir),
        "sample_count": len(files),
        "device": str(device),
        "seed": int(args.seed),
        "counterfactual_modes": list(COUNTERFACTUALS),
        "edge_ablations": EDGE_ABLATIONS,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "AUDIT_COMPLETE.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
