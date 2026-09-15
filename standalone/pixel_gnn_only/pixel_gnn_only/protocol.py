"""Fail closed on training-protocol drift; separate ablation provenance."""

import hashlib
import json
from pathlib import Path

from lap_gnn_tf.config import canonical_config_hash, load_config
from pixel_gnn_only.graph import NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES

BASE_COMMIT = "62db5a620b61e129fa1bcabf1bd25418a4578b7e"
REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "lap_gnn_tensorflow_ofix7_mid_candidate"
REFERENCE_CONFIG = REFERENCE_ROOT / "configs/fer2013_ofix7_mid_tensorflow_seed42.yaml"
EXPECTED_PARAMETER_COUNT = 189319


def apply_batch_size_override(config, batch_size):
    if batch_size is None:
        return
    if not config.get("runtime", {}).get("allow_batch_size_override", False):
        raise ValueError("Batch override is allowed only in the separate Kaggle fast configuration")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    previous_batch = config["training"]["batch_size"]
    for section in ["data", "training", "resources"]:
        config[section]["batch_size"] = batch_size
    config["run_name"] = f"pixel_gnn_only_fast_bs{batch_size}_seed{config['seed']}"
    if config.get("paths", {}).get("output_root"):
        config["paths"]["output_root"] = config["paths"]["output_root"].replace(
            f"_bs{previous_batch}_", f"_bs{batch_size}_")


def validate_pixel_config(config):
    baseline = load_config(REFERENCE_CONFIG)
    for key in ["training", "loss", "seed", "from_scratch", "init_checkpoint"]:
        actual = config.get(key)
        expected = baseline.get(key)
        if key == "training" and config.get("runtime", {}).get("allow_batch_size_override", False):
            actual = {k: v for k, v in actual.items() if k != "batch_size"}
            expected = {k: v for k, v in expected.items() if k != "batch_size"}
        if actual != expected:
            raise ValueError(f"Pixel-GNN training protocol drift: {key}")
    batch_size = config["training"]["batch_size"]
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Training batch size must be a positive integer")
    if any(config[section]["batch_size"] != batch_size for section in ["data", "resources"]):
        raise ValueError("data/training/resources batch sizes must agree")
    graph, model, data = config["graph"], config["model"], config["data"]
    checks = {
        "no_mediapipe": data.get("use_mediapipe_priors") is False,
        "no_prior_path": data.get("prior_dir") is None,
        "no_graph_cache": data.get("graph_cache_dir") is None,
        "full_pixel_grid": graph["graph_mode"] == "full_pixel_grid",
        "five_features": graph["node_features"]["features"] == NODE_FEATURE_NAMES,
        "six_edges": graph["edge_features"]["features"] == EDGE_FEATURE_NAMES,
        "no_extra_node_features": graph["detail_features"]["enabled"] is False,
        "no_anchors": graph["anchor_nodes"]["enabled"] is False,
        "no_knn": graph["knn_edges"]["enabled"] is False,
        "no_prior_corruption": graph["prior_corruption"]["enabled"] is False,
        "pixel_model": model["name"] == "pixel_gnn_only",
        "five_channels": model["node_feature_dim"] == 5,
        "shared_mlp": model["encoder_type"] == "shared_linear_gelu",
        "encoder_dropout": model["dropout"] == baseline["model"]["dropout"] == 0.2,
        "edge_features_enabled": graph["edge_features"]["enabled"] is True,
        "mean_readout": model["readout_type"] == "global_mean",
        "no_context": model["edge_context_gnn"]["context_injection"]["enabled"] is False,
        "no_motif": model.get("micro_motif_support") is None,
        "hidden": model["hidden_dim"] == 96,
        "layers": model["gnn_layers"] == model["edge_context_gnn"]["num_layers"] == 3,
        "classes": model["num_classes"] == 7,
        "edge_dim": model["edge_context_gnn"]["edge_attr_dim"] == 6,
    }
    for key in ["edge_hidden_dim", "dropout", "residual", "layer_norm", "message_type", "aggregation", "layer_output_concat"]:
        checks[f"gnn_{key}"] = model["edge_context_gnn"][key] == baseline["model"]["edge_context_gnn"][key]
    failures = [key for key, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"Pixel-GNN architecture drift: {failures}")
    config["locked"] = {
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "graph_signature": canonical_config_hash({"nodes": 2304, "connectivity": "reference_directed_8_neighborhood", "edges_per_image": 17860}),
        "feature_signature": canonical_config_hash({"names": NODE_FEATURE_NAMES}),
        "edge_signature": canonical_config_hash({"names": EDGE_FEATURE_NAMES}),
        "prior_signature": "NONE_PIXEL_ONLY",
        "dataset_split_signature": baseline["locked"]["dataset_split_signature"],
        "package_checksum": source_checksum(),
        # The full model's scientific/parity contract does not apply to this architecture.
        "execution_contract_sha256": None,
    }
    config["ablation_provenance"] = {"base_commit": BASE_COMMIT,
        "baseline_config": str(REFERENCE_CONFIG), "baseline_config_hash": canonical_config_hash(baseline),
        "training_protocol_equal": config["training"] == baseline["training"],
        "baseline_global_batch_size": baseline["training"]["batch_size"],
        "effective_global_batch_size": batch_size,
        "authorized_training_changes": [] if batch_size == baseline["training"]["batch_size"] else ["batch_size"],
        "architecture_only_comparison": batch_size == baseline["training"]["batch_size"],
        "prior_operations": "absent; baseline corruption affects priors only",
        "image_augmentation": "none in the frozen TensorFlow loader; raw grayscale retained"}
    return checks


def source_checksum():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    paths = sorted(root.rglob("*.py")) + sorted((REFERENCE_ROOT / "src/lap_gnn_tf").rglob("*.py"))
    paths += sorted(root.rglob("*.yaml")) + [root.parent.parent / "run_pixel_gnn.py"]
    paths += [REFERENCE_CONFIG, REFERENCE_ROOT / "configs/fer2013_ofix7_mid_tensorflow_baseline.yaml"]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_pixel_provenance(output_dir, datasets):
    path = Path(output_dir) / "pixel_inputs.json"
    path.write_text(json.dumps({"artifacts_read": [d.provenance for d in datasets],
                    "prior_artifacts_read": [], "base_commit": BASE_COMMIT}, indent=2), encoding="utf-8")
    return path
