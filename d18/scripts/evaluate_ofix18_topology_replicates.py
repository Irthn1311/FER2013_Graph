"""Evaluate OFIX18 stochastic topology replicates with one model/graph load.

This is an inference-only runtime wrapper around the established OFIX18
counterfactual evaluator. It preserves the per-topology-seed artifact layout
while avoiding repeated deserialization of the same locked graphs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.scripts.audit_ofix18_predecision import (
    CLASS_NAMES,
    entropy_np,
    infer_graphs,
    load_model,
    softmax_np,
)
from d18.scripts.evaluate_ofix18_factorial import (
    counterfactual_graphs,
    load_official_graphs,
    load_yaml,
    summarize,
    write_confusion,
)

LOCKED_SAMPLE_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"
MODES = ("permute_structure_destinations", "degree_matched_random_structure")
DETERMINISTIC_ABS_TOLERANCE = 1e-4


def write_seed_artifacts(
    destination: Path,
    run_dir: Path,
    checkpoint_type: str,
    checkpoint_epoch: int,
    topology_seed: int,
    y: np.ndarray,
    sample_index: np.ndarray,
    detected: np.ndarray,
    image_ids: np.ndarray,
    official_logits: np.ndarray,
    logits_by_mode: dict[str, np.ndarray],
    embeddings_by_mode: dict[str, np.ndarray],
    deterministic_diff: float,
    device: torch.device,
    prior_dir: Path,
    cache_dir: Path,
    manifest_path: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    official_probs = softmax_np(official_logits)
    official_pred = official_probs.argmax(axis=1)
    _, _, official_f1, _ = precision_recall_fscore_support(
        y, official_pred, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    official_macro = float(official_f1.mean())
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for mode in MODES:
        row, pred, probs = summarize(
            mode, y, logits_by_mode[mode], official_pred, official_macro
        )
        row.update(
            {
                "run_name": run_dir.name,
                "checkpoint": checkpoint_type,
                "checkpoint_epoch": checkpoint_epoch,
            }
        )
        rows.append(row)
        write_confusion(
            destination / f"confusion_matrix_{mode}.csv",
            np.asarray(json.loads(row["confusion_matrix_json"]), dtype=np.int64),
        )
        entropy = entropy_np(probs)
        for index in range(len(y)):
            ordered = np.partition(probs[index], -2)
            item: dict[str, Any] = {
                "run_name": run_dir.name,
                "checkpoint_type": checkpoint_type,
                "checkpoint_epoch": checkpoint_epoch,
                "image_id": str(image_ids[index]),
                "sample_index": int(sample_index[index]),
                "true_class": int(y[index]),
                "detected_state": bool(detected[index]),
                "mode": mode,
                "predicted_class": int(pred[index]),
                "entropy": float(entropy[index]),
                "max_probability": float(probs[index].max()),
                "margin": float(ordered[-1] - ordered[-2]),
                "correct": int(pred[index] == y[index]),
            }
            for class_id in range(len(CLASS_NAMES)):
                item[f"logit_{class_id}"] = float(logits_by_mode[mode][index, class_id])
                item[f"prob_{class_id}"] = float(probs[index, class_id])
            prediction_rows.append(item)

    pd.DataFrame(rows).to_csv(destination / "counterfactual_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        destination / "counterfactual_predictions.csv", index=False
    )
    np.savez_compressed(
        destination / "counterfactual_embeddings.npz",
        official=embeddings_by_mode["official"].astype(np.float32),
        **{mode: embeddings_by_mode[mode].astype(np.float32) for mode in MODES},
    )
    payload = {
        "status": "COMPLETE",
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint": checkpoint_type,
        "checkpoint_path": str(run_dir / "checkpoints" / f"{checkpoint_type}.pt"),
        "prior_dir": str(prior_dir),
        "graph_cache_dir": str(cache_dir),
        "sample_count": int(len(y)),
        "sample_manifest": str(manifest_path),
        "sample_index_sha256": hashlib.sha256(sample_index.tobytes()).hexdigest(),
        "checkpoint_epoch": checkpoint_epoch,
        "device": str(device),
        "seed": int(topology_seed),
        "counterfactual_modes": list(MODES),
        "edge_ablations": {},
        "deterministic_repeat_count": min(16, len(y)),
        "deterministic_max_abs_logit_diff": deterministic_diff,
        "deterministic_abs_tolerance": DETERMINISTIC_ABS_TOLERANCE,
        "deterministic_within_tolerance": deterministic_diff <= DETERMINISTIC_ABS_TOLERANCE,
        "zero_forced_regression_count": 0,
        "zero_forced_graph_equal": None,
        "zero_matches_remove_graph": None,
        "runtime_wrapper": "evaluate_ofix18_topology_replicates.py",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    if payload["sample_index_sha256"] != LOCKED_SAMPLE_SHA256:
        raise RuntimeError("locked sample hash mismatch")
    text = json.dumps(payload, indent=2) + "\n"
    (destination / "evaluation_manifest.json").write_text(text, encoding="utf-8")
    (destination / "AUDIT_COMPLETE.json").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--graph_cache_dir", required=True)
    parser.add_argument("--checkpoint", choices=("best", "last"), required=True)
    parser.add_argument("--sample_manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--topology_seeds", default="11,23,37,53,71")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seeds = tuple(int(value) for value in args.topology_seeds.split(",") if value.strip())
    run_dir = Path(args.run_dir)
    prior_dir = Path(args.prior_dir)
    cache_dir = Path(args.graph_cache_dir)
    manifest_path = Path(args.sample_manifest)
    output_root = Path(args.output_root)
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 715:
        raise RuntimeError(f"expected 715 locked images, got {len(manifest)}")
    files = [prior_dir / "test" / f"{str(value).zfill(6)}.npz" for value in manifest["image_id"]]
    if any(not path.exists() for path in files):
        raise FileNotFoundError("one or more locked prior files are missing")

    cfg_path = run_dir / "resolved_config.yaml"
    cfg = load_yaml(cfg_path)
    checkpoint_path = run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(payload.get("epoch", -1))
    del payload
    graphs = load_official_graphs(files, cache_dir)
    y = np.asarray([int(graph.y) for graph in graphs], dtype=np.int64)
    sample_index = np.asarray([int(graph.sample_index) for graph in graphs], dtype=np.int64)
    detected = np.asarray([bool(graph.detected) for graph in graphs], dtype=bool)
    image_ids = np.asarray([path.stem for path in files], dtype=object)
    if not np.array_equal(sample_index, manifest["sample_index"].to_numpy(dtype=np.int64)):
        raise RuntimeError("locked sample ordering mismatch")
    if not np.array_equal(y, manifest["true_class"].to_numpy(dtype=np.int64)):
        raise RuntimeError("locked labels mismatch")
    expected_detected = manifest["detected_state"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    if not np.array_equal(detected, expected_detected):
        raise RuntimeError("locked detected-state mismatch")
    if hashlib.sha256(sample_index.tobytes()).hexdigest() != LOCKED_SAMPLE_SHA256:
        raise RuntimeError("locked sample hash mismatch")

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    model = load_model(
        {"cfg": cfg, "model_family": "d18", "checkpoint_path": checkpoint_path}, device
    )
    official_logits, official_embeddings = infer_graphs(model, graphs, device, args.batch_size)
    repeat_count = min(16, len(graphs))
    repeat_logits, _ = infer_graphs(model, graphs[:repeat_count], device, args.batch_size)
    deterministic_diff = float(
        np.max(np.abs(repeat_logits - official_logits[:repeat_count]))
    )
    if deterministic_diff > DETERMINISTIC_ABS_TOLERANCE:
        raise RuntimeError(f"nondeterministic official inference: {deterministic_diff}")

    completed: list[int] = []
    for topology_seed in seeds:
        destination = output_root / f"locked_topology_seed{topology_seed}"
        if (destination / "AUDIT_COMPLETE.json").exists() and not args.overwrite:
            completed.append(topology_seed)
            print(json.dumps({"event": "topology_seed_skip", "seed": topology_seed}), flush=True)
            continue
        logits_by_mode: dict[str, np.ndarray] = {}
        embeddings_by_mode: dict[str, np.ndarray] = {"official": official_embeddings}
        for mode in MODES:
            mode_graphs = counterfactual_graphs(
                mode,
                graphs,
                files,
                np.arange(len(files)),
                cfg.get("graph", {}) or {},
                topology_seed,
            )
            logits_by_mode[mode], embeddings_by_mode[mode] = infer_graphs(
                model, mode_graphs, device, args.batch_size
            )
        write_seed_artifacts(
            destination,
            run_dir,
            args.checkpoint,
            checkpoint_epoch,
            topology_seed,
            y,
            sample_index,
            detected,
            image_ids,
            official_logits,
            logits_by_mode,
            embeddings_by_mode,
            deterministic_diff,
            device,
            prior_dir,
            cache_dir,
            manifest_path,
        )
        completed.append(topology_seed)
        print(json.dumps({"event": "topology_seed_done", "seed": topology_seed}), flush=True)

    marker = output_root / "locked_topology_bundle"
    marker.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "COMPLETE",
        "run_name": run_dir.name,
        "checkpoint": args.checkpoint,
        "topology_seeds": list(seeds),
        "completed_seeds": completed,
        "sample_index_sha256": LOCKED_SAMPLE_SHA256,
        "deterministic_max_abs_logit_diff": deterministic_diff,
        "deterministic_abs_tolerance": DETERMINISTIC_ABS_TOLERANCE,
        "deterministic_within_tolerance": deterministic_diff <= DETERMINISTIC_ABS_TOLERANCE,
        "full_training_run": False,
    }
    (marker / "AUDIT_COMPLETE.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
