"""Build portable configs and framework-neutral contracts from locked artifacts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


SEEDS = [42, 1009, 1337, 777, 3407]
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
ANCHORS = ["mouth", "eye", "brow", "nose_cheek", "global"]
NODE_FEATURES = [
    "intensity", "gx", "gy", "x_norm", "y_norm", "face_mask",
    *[f"part_soft_{index}" for index in range(13)],
    *[f"distance_map_{index}" for index in range(12)],
    "landmark_missing_flag", "grad_mag", "local_mean_3x3",
    "local_std_3x3", "laplacian_abs", "center_surround",
]
EDGE_FEATURES = [
    "dx", "dy", "spatial_dist", "abs_intensity_diff",
    "abs_grad_mag_diff", "abs_laplacian_diff",
    "part_similarity", "same_dominant_part",
]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    package = args.package_root.resolve()

    source_cfg = yaml.safe_load(
        (repo / "outputs/d16_runs/final/ofix7_mid_seed42/resolved_config.yaml").read_text(encoding="utf-8")
    )
    baseline = copy.deepcopy(source_cfg)
    baseline["run_name"] = "ofix7_mid_baseline"
    baseline["seed"] = 42
    baseline["data"]["prior_dir"] = None
    baseline["data"]["graph_cache_dir"] = None
    baseline["data"]["graph_cache_dir_detected"] = None
    baseline["data"]["graph_cache_dir_fallback"] = None
    baseline["training"]["seed"] = 42
    baseline["graph"]["prior_corruption"]["seed"] = 42 + 7699
    baseline["logging"]["wandb"]["enabled"] = True
    baseline["standalone"] = {
        "candidate": True,
        "resume_enabled": False,
        "paths_from_cli": ["fer_csv", "prior_root", "cache_root", "output_root"],
    }
    config_dir = package / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "fer2013_ofix7_mid_baseline.yaml").write_text(
        yaml.safe_dump(baseline, sort_keys=False), encoding="utf-8"
    )
    for seed in SEEDS:
        payload = {
            "extends": "fer2013_ofix7_mid_baseline.yaml",
            "run_name": f"ofix7_mid_seed{seed}",
            "seed": seed,
            "graph": {"prior_corruption": {"seed": seed + 7699}},
            "training": {"seed": seed},
        }
        (config_dir / f"fer2013_ofix7_mid_seed{seed}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )

    feature_schema = {
        "schema_id": "ofix7_mid_node_features_v1",
        "dimension": 37,
        "dtype": "float32",
        "ordered_features": [{"index": i, "name": name} for i, name in enumerate(NODE_FEATURES)],
    }
    edge_schema = {
        "schema_id": "ofix7_mid_edge_features_v1",
        "dimension": 8,
        "dtype": "float32",
        "ordered_features": [{"index": i, "name": name} for i, name in enumerate(EDGE_FEATURES)],
        "edge_index_layout": "[2, E]",
        "convention": "edge_index[0] is source; edge_index[1] is destination",
    }
    node_schema = {
        "pixel_nodes": {
            "message_passing": True,
            "order": "np.argwhere selected mask, row-major",
            "selection": "face_mask > 0.15, binary dilation radius 2",
        },
        "context_pixel_nodes": {"message_passing": True, "distinguished_by_type_tensor": False},
        "semantic_anchor_nodes": {
            "message_passing": True,
            "count": 5,
            "order": ANCHORS,
            "appended_after_pixels": True,
        },
        "readout_cls_and_motif_tokens": {
            "message_passing": False,
            "scope": "micro_motif_support readout only",
        },
        "node_type_tensor": "not explicitly stored; pixel/context versus anchor is inferred from trailing five nodes",
    }
    graph_batch = {
        "x_cat": ["sum_nodes", 37],
        "edge_index_cat": [2, "sum_edges"],
        "edge_attr_cat": ["sum_edges", 8],
        "batch_index": ["sum_nodes"],
        "ptr": ["batch_size + 1"],
        "y": ["batch_size"],
        "sample_index": ["batch_size"],
        "pos_cat": ["sum_nodes", 2],
        "part_soft_cat": ["sum_nodes", 13],
        "face_mask_cat": ["sum_nodes"],
        "variable_node_count": True,
    }
    preprocessing = {
        "image_shape": [48, 48],
        "channels": 1,
        "grayscale": True,
        "input_pixel_text": "2304 whitespace-separated integer values in FER CSV",
        "runtime_value_range": [0.0, 1.0],
        "normalization": "divide by 255 only when stored maximum exceeds 1",
        "class_order": CLASS_NAMES,
        "split_counts": {"train": 28709, "val": 3589, "test": 3589},
        "prior_schema_id": "d16_mediapipe_pixel_priors_v1",
    }
    checkpoint_policy = {
        "lock": "VAL_MACRO_F1",
        "primary": "best_val_macro_f1.pt",
        "best_alias": "byte-identical copy of best_val_macro_f1.pt",
        "secondary": "best_val_accuracy.pt",
        "scheduler_monitor": "val_loss",
        "early_stopping_monitor": "val_loss",
        "strict_improvement_tie_break": "earliest epoch",
    }
    class_mapping = {"classes": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]}
    for name, payload in {
        "feature_schema.json": feature_schema,
        "edge_schema.json": edge_schema,
        "node_schema.json": node_schema,
        "graph_batch_schema.json": graph_batch,
        "checkpoint_policy.json": checkpoint_policy,
        "class_mapping.json": class_mapping,
        "preprocessing_contract.json": preprocessing,
    }.items():
        write_json(package / "contracts" / name, payload)

    architecture = """# Architecture Contract

This is the framework-neutral contract for the locked OFIX7-mid candidate.

- Input: variable-size graphs from 48x48 grayscale FER2013 images.
- Node features: 37 ordered float32 values from `feature_schema.json`.
- Edge features: 8 ordered float32 values from `edge_schema.json`.
- Graph nodes: selected face/context pixels followed by five semantic anchors.
- Encoder: Linear(37,96), LayerNorm(96), GELU, Dropout(0.2).
- Graph stack: three gated-edge-MLP layers, mean aggregation, residual and LayerNorm enabled, edge hidden size 32, layer dropout 0.25.
- Context injection: after the final graph layer; five pooled part-group tokens; one 4-head transformer layer; initial context scale 0.5.
- Readout: micro-motif-support with major counts 3/3/3/1/2 and micro counts 2/2/2/1/1 for mouth/eye/brow/nose_cheek/global.
- Readout-only CLS and motif tokens do not participate in graph message passing.
- Classifier output: seven logits in FER class order.
- Exact parameter count: 1,061,192.

Tensor order, reductions, normalization, residual placement and initializers are
defined by the mechanically extracted PyTorch source and golden fixtures.
"""
    (package / "contracts" / "architecture_contract.md").write_text(architecture, encoding="utf-8")


if __name__ == "__main__":
    main()
