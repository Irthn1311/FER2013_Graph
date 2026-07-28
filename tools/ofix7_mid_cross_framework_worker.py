"""Bounded OFIX7-mid training worker for one framework.

This is a diagnostic tool. It does not modify either production trainer and it
does not consume the FER2013 train/validation/test split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTORCH_PACKAGE = REPO_ROOT / "standalone" / "lap_gnn_pytorch_ofix7_mid_candidate"
TENSORFLOW_PACKAGE = REPO_ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
GOLDEN_ROOT = TENSORFLOW_PACKAGE / "validation_assets" / "golden"
SNAPSHOT_STEPS = {0, 1, 2, 5, 10, 25, 50, 100}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("pytorch", "tensorflow"), required=True)
    parser.add_argument(
        "--mode",
        choices=("dropout_off", "native_dropout", "shared_dropout"),
        required=True,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-dropout-seed", type=int, default=20260728)
    parser.add_argument("--export-initial-state", action="store_true")
    return parser.parse_args()


def add_package_source(package: Path) -> None:
    source = str(package / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ofix7_mid_cross_framework_audit_v1":
        raise ValueError(f"Unsupported manifest schema: {payload.get('schema')!r}")
    if not payload.get("steps"):
        raise ValueError("Audit manifest has no steps")
    return payload


def load_fixture() -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(GOLDEN_ROOT / "graph_batch.npz", allow_pickle=False) as data:
        arrays = {key: np.array(data[key], copy=True) for key in data.files}
    labels = np.load(GOLDEN_ROOT / "labels.npy", allow_pickle=False).astype(np.int64)
    return arrays, labels


def select_graphs(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    graph_ids: list[int],
) -> dict[str, np.ndarray]:
    node_counts = arrays["node_counts"].astype(np.int64)
    edge_counts = arrays["edge_counts"].astype(np.int64)
    node_offsets = np.concatenate([[0], np.cumsum(node_counts)])
    edge_offsets = np.concatenate([[0], np.cumsum(edge_counts)])
    node_keys = [
        "node_features",
        "node_types",
        "positions",
        "part_soft",
        "face_mask",
        "anchor_mask",
    ]
    graph_keys = [
        "sample_ids",
        "valid_part_mask",
        "valid_anchor_mask",
        "detected",
        "landmark_missing_flag",
        "image_48",
    ]
    node_chunks: dict[str, list[np.ndarray]] = {key: [] for key in node_keys}
    edge_index_chunks: list[np.ndarray] = []
    edge_feature_chunks: list[np.ndarray] = []
    batch_index_chunks: list[np.ndarray] = []
    selected_node_counts: list[int] = []
    selected_edge_counts: list[int] = []
    new_node_offset = 0
    for new_graph_id, graph_id in enumerate(graph_ids):
        graph_id = int(graph_id)
        n0, n1 = int(node_offsets[graph_id]), int(node_offsets[graph_id + 1])
        e0, e1 = int(edge_offsets[graph_id]), int(edge_offsets[graph_id + 1])
        node_count = n1 - n0
        edge_count = e1 - e0
        selected_node_counts.append(node_count)
        selected_edge_counts.append(edge_count)
        for key in node_keys:
            node_chunks[key].append(arrays[key][n0:n1])
        local_edges = arrays["edge_index"][:, e0:e1] - n0 + new_node_offset
        edge_index_chunks.append(local_edges)
        edge_feature_chunks.append(arrays["edge_features"][e0:e1])
        batch_index_chunks.append(
            np.full((node_count,), new_graph_id, dtype=np.int64)
        )
        new_node_offset += node_count
    result = {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in node_chunks.items()
    }
    result.update(
        {
            "edge_index": np.concatenate(edge_index_chunks, axis=1).astype(np.int64),
            "edge_features": np.concatenate(edge_feature_chunks, axis=0).astype(np.float32),
            "batch_index": np.concatenate(batch_index_chunks).astype(np.int64),
            "node_counts": np.asarray(selected_node_counts, dtype=np.int64),
            "edge_counts": np.asarray(selected_edge_counts, dtype=np.int64),
            "ptr": np.concatenate(
                [[0], np.cumsum(selected_node_counts)]
            ).astype(np.int64),
            "labels": labels[np.asarray(graph_ids, dtype=np.int64)],
        }
    )
    for key in graph_keys:
        result[key] = arrays[key][np.asarray(graph_ids, dtype=np.int64)]
    return result


def shared_mask(
    shape: tuple[int, ...],
    probability: float,
    seed: int,
    step: int,
    call_index: int,
) -> np.ndarray:
    sequence = np.random.SeedSequence(
        [int(seed), int(step), int(call_index), len(shape), *map(int, shape)]
    )
    rng = np.random.default_rng(sequence)
    keep_probability = 1.0 - float(probability)
    mask = rng.random(shape) < keep_probability
    return mask.astype(np.float32) / np.float32(keep_probability)


class SharedDropoutTrace:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.step = 0
        self.call_index = 0
        self.current: list[dict[str, Any]] = []
        self.first_step: list[dict[str, Any]] = []

    def begin(self, step: int) -> None:
        self.step = int(step)
        self.call_index = 0
        self.current = []

    def next_mask(self, shape: tuple[int, ...], probability: float) -> np.ndarray:
        self.call_index += 1
        entry = {
            "call": self.call_index,
            "shape": list(map(int, shape)),
            "probability": float(probability),
        }
        self.current.append(entry)
        if self.step == 1:
            self.first_step.append(entry)
        return shared_mask(
            shape,
            probability,
            self.seed,
            self.step,
            self.call_index,
        )


def save_state_npz(path: Path, state: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **state)


def state_metrics(state: dict[str, np.ndarray]) -> dict[str, float]:
    squared = 0.0
    maximum = 0.0
    count = 0
    for value in state.values():
        flat = np.asarray(value, dtype=np.float64).ravel()
        squared += float(np.dot(flat, flat))
        if flat.size:
            maximum = max(maximum, float(np.max(np.abs(flat))))
        count += int(flat.size)
    return {
        "parameter_l2": math.sqrt(squared),
        "parameter_max_abs": maximum,
        "parameter_count": count,
    }


def run_pytorch(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    add_package_source(PYTORCH_PACKAGE)
    import torch
    import torch.nn.functional as functional

    from lap_gnn.config import load_config
    from lap_gnn.data.graph_builder import D16Batch
    from lap_gnn.model.d16_model import D16Model
    from lap_gnn.training.engine import set_seed

    config = load_config(PYTORCH_PACKAGE / "configs/fer2013_ofix7_mid_seed42.yaml")
    set_seed(args.seed)
    model = D16Model.from_config(config, input_dim=37)
    if args.export_initial_state:
        save_state_npz(
            args.initial_state,
            {
                key: value.detach().cpu().numpy()
                for key, value in model.state_dict().items()
            },
        )
    with np.load(args.initial_state, allow_pickle=False) as source:
        state = {
            key: torch.from_numpy(np.array(source[key], copy=True))
            for key in source.files
        }
    model.load_state_dict(state, strict=True)
    model.train()
    if args.mode == "dropout_off":
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            if isinstance(module, torch.nn.MultiheadAttention):
                module.dropout = 0.0
    elif args.mode == "shared_dropout":
        # PyTorch may execute attention dropout inside a fused SDPA kernel,
        # bypassing torch.nn.functional.dropout. Keep that one operation off in
        # both workers so every remaining stochastic mask is explicitly shared.
        for module in model.modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                module.dropout = 0.0

    controller = SharedDropoutTrace(args.shared_dropout_seed)
    original_dropout = functional.dropout
    if args.mode == "shared_dropout":
        def deterministic_dropout(input, p=0.5, training=True, inplace=False):
            if not training or float(p) == 0.0:
                return input
            if inplace:
                raise ValueError("In-place dropout is unsupported by this audit")
            mask = controller.next_mask(tuple(input.shape), float(p))
            return input * torch.from_numpy(mask).to(
                device=input.device, dtype=input.dtype
            )

        functional.dropout = deterministic_dropout

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    arrays, labels = load_fixture()

    def make_batch(selection: list[int]) -> D16Batch:
        raw = select_graphs(arrays, labels, selection)
        return D16Batch(
            x_cat=torch.from_numpy(raw["node_features"]).float(),
            edge_index_cat=torch.from_numpy(raw["edge_index"]).long(),
            edge_attr_cat=torch.from_numpy(raw["edge_features"]).float(),
            batch_index=torch.from_numpy(raw["batch_index"]).long(),
            ptr=torch.from_numpy(raw["ptr"]).long(),
            y=torch.from_numpy(raw["labels"]).long(),
            sample_index=torch.from_numpy(raw["sample_ids"]).long(),
            pos_cat=torch.from_numpy(raw["positions"]).float(),
            part_soft_cat=torch.from_numpy(raw["part_soft"]).float(),
            face_mask_cat=torch.from_numpy(raw["face_mask"]).float(),
            valid_part_mask=torch.from_numpy(raw["valid_part_mask"]).float(),
            valid_anchor_mask=torch.from_numpy(raw["valid_anchor_mask"]).float(),
            detected=torch.from_numpy(raw["detected"]).bool(),
            landmark_missing_flag=torch.from_numpy(
                raw["landmark_missing_flag"]
            ).long(),
            image_48=torch.from_numpy(raw["image_48"]).float(),
            node_feature_names=None,
            edge_feature_names=None,
        )

    def model_state() -> dict[str, np.ndarray]:
        return {
            key: value.detach().cpu().numpy()
            for key, value in model.state_dict().items()
        }

    records = []
    save_state_npz(args.output_dir / "state_step_000.npz", model_state())
    try:
        for item in manifest["steps"]:
            step = int(item["step"])
            controller.begin(step)
            batch = make_batch([int(value) for value in item["graph_ids"]])
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = functional.cross_entropy(output["logits"], batch.y)
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
            )
            optimizer.step()
            logits = output["logits"].detach().cpu().numpy()
            record = {
                "step": step,
                "graph_ids": item["graph_ids"],
                "loss": float(loss.detach().cpu().item()),
                "gradient_norm": gradient_norm,
                "logits": logits.tolist(),
                "predictions": logits.argmax(axis=1).tolist(),
                "dropout_calls": len(controller.current),
                **state_metrics(model_state()),
            }
            records.append(record)
            if step in SNAPSHOT_STEPS:
                save_state_npz(
                    args.output_dir / f"state_step_{step:03d}.npz",
                    model_state(),
                )
    finally:
        functional.dropout = original_dropout
    return {
        "framework": "pytorch",
        "framework_version": torch.__version__,
        "mode": args.mode,
        "seed": args.seed,
        "records": records,
        "first_step_dropout_trace": controller.first_step,
    }


def run_tensorflow(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    add_package_source(TENSORFLOW_PACKAGE)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    from lap_gnn_tf.conversion import load_pytorch_npz
    from lap_gnn_tf.model import LapGNN
    from lap_gnn_tf.seed import seed_everything
    from lap_gnn_tf.training.optimizer import TorchCompatibleAdamW

    seed_everything(args.seed)
    model = LapGNN()
    load_pytorch_npz(model, args.initial_state, strict=True)
    if args.mode == "dropout_off":
        for layer in model._flatten_layers(include_self=True, recursive=True):
            if hasattr(layer, "dropout_rate"):
                layer.dropout_rate = 0.0
    elif args.mode == "shared_dropout":
        for layer in model._flatten_layers(include_self=True, recursive=True):
            if layer.__class__.__name__ == "TorchMultiheadAttention":
                layer.dropout_rate = 0.0

    controller = SharedDropoutTrace(args.shared_dropout_seed)
    original_dropout = tf.nn.dropout
    if args.mode == "shared_dropout":
        def deterministic_dropout(
            x, rate, noise_shape=None, seed=None, name=None
        ):
            del seed, name
            if noise_shape is not None:
                raise ValueError("noise_shape is unsupported by this audit")
            probability = float(rate)
            if probability == 0.0:
                return x
            shape = tuple(int(value) for value in tf.shape(x).numpy())
            mask = controller.next_mask(shape, probability)
            return x * tf.convert_to_tensor(mask, dtype=x.dtype)

        tf.nn.dropout = deterministic_dropout

    optimizer = TorchCompatibleAdamW(
        learning_rate=3e-4,
        weight_decay=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        global_clipnorm=5.0,
    )
    optimizer.build(model.trainable_variables)
    arrays, labels = load_fixture()

    def make_batch(selection: list[int]) -> dict[str, tf.Tensor]:
        raw = select_graphs(arrays, labels, selection)
        return {
            "node_features": tf.convert_to_tensor(raw["node_features"], tf.float32),
            "edge_index": tf.convert_to_tensor(raw["edge_index"], tf.int64),
            "edge_features": tf.convert_to_tensor(raw["edge_features"], tf.float32),
            "node_types": tf.convert_to_tensor(raw["node_types"], tf.int8),
            "node_graph_index": tf.convert_to_tensor(raw["batch_index"], tf.int64),
            "edge_graph_index": tf.repeat(
                tf.range(len(selection), dtype=tf.int64),
                tf.convert_to_tensor(raw["edge_counts"], tf.int32),
            ),
            "graph_node_counts": tf.convert_to_tensor(raw["node_counts"], tf.int64),
            "graph_edge_counts": tf.convert_to_tensor(raw["edge_counts"], tf.int64),
            "labels": tf.convert_to_tensor(raw["labels"], tf.int64),
            "sample_ids": tf.convert_to_tensor(raw["sample_ids"], tf.int64),
            "coordinates": tf.convert_to_tensor(raw["positions"], tf.float32),
            "anchor_mask": tf.convert_to_tensor(raw["anchor_mask"], tf.bool),
            "part_soft": tf.convert_to_tensor(raw["part_soft"], tf.float32),
            "face_mask": tf.convert_to_tensor(raw["face_mask"], tf.float32),
            "valid_part_mask": tf.convert_to_tensor(
                raw["valid_part_mask"], tf.float32
            ),
            "valid_anchor_mask": tf.convert_to_tensor(
                raw["valid_anchor_mask"], tf.float32
            ),
            "detected": tf.convert_to_tensor(raw["detected"], tf.bool),
            "landmark_missing_flag": tf.convert_to_tensor(
                raw["landmark_missing_flag"], tf.int64
            ),
            "image_48": tf.convert_to_tensor(raw["image_48"], tf.float32),
        }

    def model_state() -> dict[str, np.ndarray]:
        result = {}
        for source_key, binding in model.mapped_trainable_variables().items():
            value = binding.variable.numpy()
            if binding.transform == "transpose":
                value = value.T
            result[source_key] = np.array(value, copy=True)
        return result

    records = []
    save_state_npz(args.output_dir / "state_step_000.npz", model_state())
    try:
        for item in manifest["steps"]:
            step = int(item["step"])
            controller.begin(step)
            batch = make_batch([int(value) for value in item["graph_ids"]])
            with tf.GradientTape() as tape:
                output = model(batch, training=True)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=batch["labels"], logits=output["logits"]
                )
                loss = tf.reduce_mean(tf.cast(losses, tf.float32))
            gradients = tape.gradient(loss, model.trainable_variables)
            gradient_norm = float(tf.linalg.global_norm(gradients).numpy())
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            logits = output["logits"].numpy()
            record = {
                "step": step,
                "graph_ids": item["graph_ids"],
                "loss": float(loss.numpy()),
                "gradient_norm": gradient_norm,
                "logits": logits.tolist(),
                "predictions": logits.argmax(axis=1).tolist(),
                "dropout_calls": len(controller.current),
                **state_metrics(model_state()),
            }
            records.append(record)
            if step in SNAPSHOT_STEPS:
                save_state_npz(
                    args.output_dir / f"state_step_{step:03d}.npz",
                    model_state(),
                )
    finally:
        tf.nn.dropout = original_dropout
    return {
        "framework": "tensorflow",
        "framework_version": tf.__version__,
        "mode": args.mode,
        "seed": args.seed,
        "records": records,
        "first_step_dropout_trace": controller.first_step,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    if args.framework == "pytorch":
        result = run_pytorch(args, manifest)
    else:
        if args.export_initial_state:
            raise ValueError("Only the PyTorch worker may export initial state")
        result = run_tensorflow(args, manifest)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "framework": result["framework"],
                "mode": result["mode"],
                "steps": len(result["records"]),
                "output_dir": str(args.output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
