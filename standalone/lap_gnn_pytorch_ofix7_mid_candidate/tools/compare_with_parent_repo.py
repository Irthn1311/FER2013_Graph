"""Bounded parity comparison against the exact historical source commit.

This explicit extraction tool is allowed to read the parent Git repository.
Normal package runtime never imports the parent repository.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from extract_runtime_sources import SOURCE_COMMIT, SOURCE_MAP, git_blob


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_reference_package(repo: Path, root: Path) -> None:
    package = root / "reference_d16"
    for source in SOURCE_MAP:
        relative = Path(source).relative_to("d16")
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = git_blob(repo, source).decode("utf-8")
        text = text.replace("from d16.", "from reference_d16.")
        text = text.replace("import d16.", "import reference_d16.")
        if source == "d16/training/train_d16.py":
            block = (
                "PROJECT_ROOT = Path(__file__).resolve().parents[2]\n"
                "if str(PROJECT_ROOT) not in sys.path:\n"
                "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            )
            text = text.replace(block, "")
        target.write_text(text, encoding="utf-8", newline="\n")
    for folder in ["", "data", "losses", "models", "training"]:
        marker = package / folder / "__init__.py"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("", encoding="utf-8")


def dataset_kwargs(cfg: dict[str, Any], prior_root: Path, dataset_class):
    graph = cfg["graph"]
    return dataset_class(
        prior_root,
        split="train",
        graph_mode=graph["graph_mode"],
        face_threshold=graph["face_threshold"],
        context_pixels=graph["context_pixels"],
        detail_features=graph["detail_features"],
        edge_features=graph["edge_features"],
        anchor_nodes=graph["anchor_nodes"],
        prior_corruption={"enabled": False},
    )


def select_samples(prior_root: Path, count: int = 32) -> list[dict[str, Any]]:
    rows = []
    with (prior_root / "train_coverage_rows.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append({
                "sample_index": int(row["sample_index"]),
                "label": int(row["label"]),
                "detected": row["detected"].strip().lower() == "true",
                "fallback_type": row["fallback_type"],
            })
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for label in range(7):
        item = next(row for row in rows if row["label"] == label and row["detected"])
        chosen.append(item)
        used.add(item["sample_index"])
    fallback = next(row for row in rows if not row["detected"])
    chosen.append(fallback)
    for detected in (False, True):
        for label in range(7):
            for row in rows:
                if len(chosen) >= count:
                    break
                if row["sample_index"] not in used and row["detected"] == detected and row["label"] == label:
                    chosen.append(row)
                    used.add(row["sample_index"])
                    break
    for row in rows:
        if len(chosen) >= count:
            break
        if row["sample_index"] not in used:
            chosen.append(row)
            used.add(row["sample_index"])
    if len(chosen) != count:
        raise RuntimeError(f"Could select only {len(chosen)} samples")
    return chosen


def tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_match": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    a = left.detach().cpu().to(torch.float64)
    b = right.detach().cpu().to(torch.float64)
    delta = (a - b).abs()
    denom = torch.linalg.vector_norm(a)
    relative = torch.linalg.vector_norm(a - b) / torch.clamp(denom, min=1e-30)
    return {
        "shape_match": True,
        "exact": bool(torch.equal(left.detach().cpu(), right.detach().cpu())),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "relative_l2": float(relative),
    }


def graph_arrays(graph) -> dict[str, torch.Tensor]:
    return {
        "x": graph.x,
        "edge_index": graph.edge_index,
        "edge_attr": graph.edge_attr,
        "pos": graph.pos,
        "y": graph.y,
        "sample_index": graph.sample_index,
        "part_soft": graph.part_soft,
        "face_mask": graph.face_mask,
        "valid_part_mask": graph.valid_part_mask,
        "valid_anchor_mask": graph.valid_anchor_mask,
        "detected": graph.detected,
        "landmark_missing_flag": graph.landmark_missing_flag,
        "image_48": graph.image_48,
    }


def node_types(graph, anchor_count: int = 5) -> np.ndarray:
    count = int(graph.x.shape[0])
    pixel_count = count - anchor_count
    types = np.full((count,), 2, dtype=np.int8)
    face = graph.face_mask[:pixel_count].detach().cpu().numpy()
    types[:pixel_count] = np.where(face > 0.15, 0, 1)
    return types


def capture_forward(model, batch) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def output_hook(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(value):
                captured[name] = value.detach().cpu()
        return hook

    def input_hook(name):
        def hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                captured[name] = inputs[0].detach().cpu()
        return hook

    handles.append(model.encoder.register_forward_hook(output_hook("input_projection")))
    for index, layer in enumerate(model.gnn.layers, start=1):
        handles.append(layer.register_forward_hook(output_hook(f"gnn_layer_{index}")))
    handles.append(model.classifier.register_forward_pre_hook(input_hook("classifier_input")))
    try:
        with torch.no_grad():
            output = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    captured["pre_readout_node_representation"] = output["node_embeddings"].detach().cpu()
    captured["pooled_graph_embedding"] = output["z_image"].detach().cpu()
    for key in [
        "micro_major_motif_tokens",
        "micro_major_motif_transformed_tokens",
        "micro_motif_tokens",
        "micro_motif_transformed_tokens",
        "micro_support_gate",
    ]:
        value = output.get(key)
        if torch.is_tensor(value):
            captured[key] = value.detach().cpu()
    return captured, output


def compare_optimizer_state(left, right) -> float:
    left_states = list(left.state.values())
    right_states = list(right.state.values())
    if len(left_states) != len(right_states):
        return math.inf
    maximum = 0.0
    for a_state, b_state in zip(left_states, right_states):
        if set(a_state) != set(b_state):
            return math.inf
        for key in a_state:
            a, b = a_state[key], b_state[key]
            if torch.is_tensor(a):
                maximum = max(maximum, tensor_metrics(a, b)["max_abs"])
            elif a != b:
                return math.inf
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    repo = args.repo.resolve()
    package = args.package_root.resolve()
    prior_root = args.prior_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    golden = package / "validation_assets" / "golden"
    golden.mkdir(parents=True, exist_ok=True)

    temp = Path(tempfile.mkdtemp(prefix="lap_gnn_parent_reference_"))
    try:
        write_reference_package(repo, temp)
        sys.path.insert(0, str(temp))
        sys.path.insert(0, str(package / "src"))
        ref_dataset_mod = importlib.import_module("reference_d16.data.pixel_prior_dataset")
        ref_graph_mod = importlib.import_module("reference_d16.data.graph_builder")
        ref_model_mod = importlib.import_module("reference_d16.models.d16_model")
        ref_engine = importlib.import_module("reference_d16.training.train_d16")
        standalone_dataset_mod = importlib.import_module("lap_gnn.data.pixel_prior_dataset")
        standalone_graph_mod = importlib.import_module("lap_gnn.data.graph_builder")
        standalone_model_mod = importlib.import_module("lap_gnn.model.d16_model")
        standalone_metrics = importlib.import_module("lap_gnn.training.metrics")

        cfg = yaml.safe_load(
            (repo / "outputs/d16_runs/final/ofix7_mid_seed42/resolved_config.yaml").read_text(encoding="utf-8")
        )
        selected = select_samples(prior_root, 32)
        ref_dataset = dataset_kwargs(cfg, prior_root, ref_dataset_mod.D16PixelPriorDataset)
        standalone_dataset = dataset_kwargs(cfg, prior_root, standalone_dataset_mod.D16PixelPriorDataset)
        graph_rows = []
        ref_graphs = []
        standalone_graphs = []
        for item in selected:
            index = item["sample_index"]
            ref_graph = ref_dataset[index]
            standalone_graph = standalone_dataset[index]
            ref_graphs.append(ref_graph)
            standalone_graphs.append(standalone_graph)
            fields = {}
            pass_all = True
            for name, left in graph_arrays(ref_graph).items():
                right = graph_arrays(standalone_graph)[name]
                metric = tensor_metrics(left, right)
                fields[name] = metric
                pass_all = pass_all and metric["shape_match"] and metric["max_abs"] == 0.0
            graph_rows.append({
                **item,
                "node_count": int(ref_graph.x.shape[0]),
                "edge_count": int(ref_graph.edge_index.shape[1]),
                "pass": pass_all,
                "max_node_abs": fields["x"]["max_abs"],
                "max_edge_abs": fields["edge_attr"]["max_abs"],
                "max_position_abs": fields["pos"]["max_abs"],
                "edge_index_exact": fields["edge_index"]["exact"],
                "node_type_exact": bool(np.array_equal(node_types(ref_graph), node_types(standalone_graph))),
            })

        graph_parity_pass = all(row["pass"] and row["node_type_exact"] for row in graph_rows)
        with (output / "graph_parity.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(graph_rows[0]))
            writer.writeheader()
            writer.writerows(graph_rows)

        fixture_count = 8
        ref_batch = ref_graph_mod.collate_d16_graphs(ref_graphs[:fixture_count])
        standalone_batch = standalone_graph_mod.collate_d16_graphs(standalone_graphs[:fixture_count])
        counts = np.diff(ref_batch.ptr.numpy()).astype(np.int64)
        edge_counts = np.asarray([graph.edge_index.shape[1] for graph in ref_graphs[:fixture_count]], dtype=np.int64)
        fixture_types = np.concatenate([node_types(graph) for graph in ref_graphs[:fixture_count]])
        np.savez_compressed(
            golden / "graph_batch.npz",
            node_features=ref_batch.x_cat.numpy(),
            edge_index=ref_batch.edge_index_cat.numpy(),
            edge_features=ref_batch.edge_attr_cat.numpy(),
            node_types=fixture_types,
            batch_index=ref_batch.batch_index.numpy(),
            ptr=ref_batch.ptr.numpy(),
            node_counts=counts,
            edge_counts=edge_counts,
            sample_ids=ref_batch.sample_index.numpy(),
            positions=ref_batch.pos_cat.numpy(),
            part_soft=ref_batch.part_soft_cat.numpy(),
            face_mask=ref_batch.face_mask_cat.numpy(),
            anchor_mask=(fixture_types == 2),
            valid_part_mask=ref_batch.valid_part_mask.numpy(),
            valid_anchor_mask=ref_batch.valid_anchor_mask.numpy(),
            detected=ref_batch.detected.numpy(),
            landmark_missing_flag=ref_batch.landmark_missing_flag.numpy(),
            image_48=ref_batch.image_48.numpy(),
        )
        np.save(golden / "labels.npy", ref_batch.y.numpy())

        checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint.get("model")))
        if not isinstance(state, dict):
            raise RuntimeError("Could not find model state in locked checkpoint")
        torch.manual_seed(42)
        ref_model = ref_model_mod.D16Model.from_config(cfg, input_dim=37)
        torch.manual_seed(42)
        standalone_model = standalone_model_mod.D16Model.from_config(cfg, input_dim=37)
        ref_model.load_state_dict(state, strict=True)
        standalone_model.load_state_dict(state, strict=True)
        state_key_match = list(ref_model.state_dict()) == list(standalone_model.state_dict())
        state_shape_match = all(
            ref_model.state_dict()[key].shape == standalone_model.state_dict()[key].shape
            for key in ref_model.state_dict()
        )
        ref_model.eval()
        standalone_model.eval()
        ref_layers, ref_out = capture_forward(ref_model, ref_batch)
        standalone_layers, standalone_out = capture_forward(standalone_model, standalone_batch)
        layer_metrics = {key: tensor_metrics(ref_layers[key], standalone_layers[key]) for key in ref_layers}
        logits_metric = tensor_metrics(ref_out["logits"], standalone_out["logits"])
        probabilities_ref = torch.softmax(ref_out["logits"], dim=1)
        probabilities_new = torch.softmax(standalone_out["logits"], dim=1)
        probabilities_metric = tensor_metrics(probabilities_ref, probabilities_new)
        predictions_match = bool(torch.equal(probabilities_ref.argmax(1), probabilities_new.argmax(1)))
        forward_pass = (
            logits_metric["max_abs"] <= 1e-6
            and probabilities_metric["max_abs"] <= 1e-6
            and predictions_match
            and all(metric["max_abs"] <= 1e-6 for metric in layer_metrics.values())
        )
        np.savez_compressed(golden / "layer_outputs.npz", **{key: value.numpy() for key, value in ref_layers.items()})
        np.save(golden / "pooled_embeddings.npy", ref_out["z_image"].detach().numpy())
        np.save(golden / "logits.npy", ref_out["logits"].detach().numpy())
        np.save(golden / "probabilities.npy", probabilities_ref.detach().numpy())
        reference_loss = F.cross_entropy(ref_out["logits"], ref_batch.y)
        (golden / "losses.json").write_text(
            json.dumps({"cross_entropy": float(reference_loss), "dtype": "float32"}, indent=2) + "\n",
            encoding="utf-8",
        )
        state_manifest = {
            "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "state_key_count": len(state),
            "state_keys_sha256": hashlib.sha256("\n".join(state).encode()).hexdigest(),
            "parameter_count": sum(parameter.numel() for parameter in ref_model.parameters()),
        }
        (golden / "initial_state_manifest.json").write_text(
            json.dumps(state_manifest, indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            golden / "model_state.npz",
            **{key: value.detach().cpu().numpy() for key, value in state.items()},
        )

        # Two CPU float32 optimizer steps, using two batches of two graphs.
        torch.manual_seed(20260724)
        ref_train = ref_model_mod.D16Model.from_config(cfg, input_dim=37)
        torch.manual_seed(20260724)
        standalone_train = standalone_model_mod.D16Model.from_config(cfg, input_dim=37)
        standalone_train.load_state_dict(ref_train.state_dict(), strict=True)
        ref_train.train()
        standalone_train.train()
        ref_optimizer = torch.optim.AdamW(ref_train.parameters(), lr=3e-4, weight_decay=1e-3)
        standalone_optimizer = torch.optim.AdamW(standalone_train.parameters(), lr=3e-4, weight_decay=1e-3)
        ref_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            ref_optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4, min_lr=3e-5
        )
        standalone_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            standalone_optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4, min_lr=3e-5
        )
        training_steps = []
        for step in range(2):
            left_batch = ref_graph_mod.collate_d16_graphs(ref_graphs[step * 2:(step + 1) * 2])
            right_batch = standalone_graph_mod.collate_d16_graphs(standalone_graphs[step * 2:(step + 1) * 2])
            start_rng = torch.get_rng_state()
            ref_optimizer.zero_grad(set_to_none=True)
            ref_logits = ref_train(left_batch)["logits"]
            ref_loss = F.cross_entropy(ref_logits, left_batch.y)
            ref_loss.backward()
            ref_grads = [p.grad.detach().clone() for p in ref_train.parameters() if p.grad is not None]
            ref_optimizer.step()
            ref_rng = torch.get_rng_state()
            torch.set_rng_state(start_rng)
            standalone_optimizer.zero_grad(set_to_none=True)
            standalone_logits = standalone_train(right_batch)["logits"]
            standalone_loss = F.cross_entropy(standalone_logits, right_batch.y)
            standalone_loss.backward()
            standalone_grads = [p.grad.detach().clone() for p in standalone_train.parameters() if p.grad is not None]
            standalone_optimizer.step()
            standalone_rng = torch.get_rng_state()
            grad_max = max(
                tensor_metrics(left, right)["max_abs"]
                for left, right in zip(ref_grads, standalone_grads)
            )
            parameter_max = max(
                tensor_metrics(left, right)["max_abs"]
                for left, right in zip(ref_train.parameters(), standalone_train.parameters())
            )
            training_steps.append({
                "step": step + 1,
                "logit_max_abs": tensor_metrics(ref_logits, standalone_logits)["max_abs"],
                "loss_abs": abs(float(ref_loss) - float(standalone_loss)),
                "gradient_max_abs": grad_max,
                "parameter_max_abs": parameter_max,
                "optimizer_state_max_abs": compare_optimizer_state(ref_optimizer, standalone_optimizer),
                "rng_equal": bool(torch.equal(ref_rng, standalone_rng)),
            })
            torch.set_rng_state(ref_rng)
        training_step_pass = all(
            row["logit_max_abs"] <= 1e-6
            and row["loss_abs"] <= 1e-7
            and row["gradient_max_abs"] <= 1e-6
            and row["parameter_max_abs"] <= 1e-7
            and row["optimizer_state_max_abs"] <= 1e-7
            and row["rng_equal"]
            for row in training_steps
        ) and ref_scheduler.state_dict() == standalone_scheduler.state_dict()

        # Fixed-array metrics: historical accuracy/macro/per-class plus extended definitions.
        labels = np.asarray([0, 1, 2, 3, 4, 5, 6, 0, 2, 4, 6, 6], dtype=np.int64)
        predictions = np.asarray([0, 1, 4, 3, 4, 5, 6, 2, 2, 6, 6, 0], dtype=np.int64)
        rng = np.random.default_rng(123)
        raw = rng.uniform(0.01, 1.0, size=(len(labels), 7))
        probabilities = raw / raw.sum(axis=1, keepdims=True)
        parent_basic = ref_engine._metrics(labels, predictions)
        parent_rows = ref_engine._per_class_rows(labels, predictions, "fixed", 0)
        extended = standalone_metrics.classification_metrics(labels, predictions, probabilities)
        metric_differences = {
            "accuracy": abs(parent_basic["accuracy"] - extended["accuracy"]),
            "macro_f1": abs(parent_basic["macro_f1"] - extended["macro_f1"]),
            "precision": max(abs(parent_rows[i]["precision"] - extended["per_class_precision"][i]) for i in range(7)),
            "recall": max(abs(parent_rows[i]["recall"] - extended["per_class_recall"][i]) for i in range(7)),
            "f1": max(abs(parent_rows[i]["f1"] - extended["per_class_f1"][i]) for i in range(7)),
        }
        metric_pass = max(metric_differences.values()) <= 1e-12

        # Strict checkpoint load and roundtrip.
        roundtrip_path = output / "bounded_checkpoint_roundtrip.pt"
        torch.save({"model_state": standalone_model.state_dict(), "epoch": checkpoint.get("epoch")}, roundtrip_path)
        reloaded = torch.load(roundtrip_path, map_location="cpu", weights_only=False)["model_state"]
        checkpoint_roundtrip_pass = all(torch.equal(state[key], reloaded[key]) for key in state)
        roundtrip_path.unlink()

        results = {
            "source_commit": SOURCE_COMMIT,
            "sample_count": 32,
            "fixture_sample_count": fixture_count,
            "selected_samples": selected,
            "graph_parity_pass": graph_parity_pass,
            "graph_max_node_abs": max(row["max_node_abs"] for row in graph_rows),
            "graph_max_edge_abs": max(row["max_edge_abs"] for row in graph_rows),
            "forward_parity_pass": forward_pass,
            "logit_max_abs": logits_metric["max_abs"],
            "probability_max_abs": probabilities_metric["max_abs"],
            "prediction_agreement": 1.0 if predictions_match else 0.0,
            "layer_metrics": layer_metrics,
            "state_key_match": state_key_match,
            "state_shape_match": state_shape_match,
            "training_steps": training_steps,
            "training_step_parity_pass": training_step_pass,
            "scheduler_state_match": ref_scheduler.state_dict() == standalone_scheduler.state_dict(),
            "metric_differences": metric_differences,
            "metric_parity_pass": metric_pass,
            "extended_metric_values": extended,
            "checkpoint_roundtrip_pass": checkpoint_roundtrip_pass,
            "parameter_count": state_manifest["parameter_count"],
            "full_training_launched": False,
            "completed_epoch": False,
            "optimizer_steps": 2,
        }
        (output / "parity_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        (package / "validation_assets" / "parity_results.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        fixture_files = sorted(path for path in golden.iterdir() if path.is_file())
        manifest = {
            "fixture_version": "ofix7_mid_golden_v1",
            "source": "exact historical parent implementation",
            "source_commit": SOURCE_COMMIT,
            "sample_ids": ref_batch.sample_index.tolist(),
            "labels": ref_batch.y.tolist(),
            "covers_all_seven_classes": sorted(set(ref_batch.y.tolist())) == list(range(7)),
            "contains_fallback": any(not item["detected"] for item in selected[:fixture_count]),
            "checkpoint_sha256": state_manifest["checkpoint_sha256"],
            "tolerances": {"cpu_float32_target": 1e-6, "maximum_allowed": 1e-5},
            "files": {path.name: sha256_file(path) for path in fixture_files},
        }
        (package / "validation_assets" / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "graph_parity_pass": graph_parity_pass,
            "forward_parity_pass": forward_pass,
            "training_step_parity_pass": training_step_pass,
            "metric_parity_pass": metric_pass,
            "checkpoint_roundtrip_pass": checkpoint_roundtrip_pass,
        }, indent=2))
    finally:
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()
