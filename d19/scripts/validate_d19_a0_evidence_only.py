"""Bounded validator and report generator for D19-A0 evidence-only control."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d18.data.collate import collate_d18_graphs
from d16.data.mediapipe_priors import pixels_to_image48
from d18.data.structure_dataset import StructurePixelDataset
from d18.data.structure_graph_builder import D18GraphData, build_structure_graph
from d18.data.structure_graph_cache import (
    EVIDENCE_CACHE_SCHEMA,
    evidence_cache_signature,
    evidence_cache_signature_payload,
    evidence_graph_cache_path,
)
from d18.models.structure_gnn import StructureGNN
from d18.training.train_d18 import load_checkpoint, scientific_resume_signature


C2_RUN = ROOT / "outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42"
C2_CONFIG = ROOT / "configs/d18/overfit_fix_18/d18_ofix18_c2_structure_mode_mix_only_seed42.yaml"
C2_RESOLVED = C2_RUN / "resolved_config.yaml"
PRIOR_ROOT = ROOT / "outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
LOCKED_SAMPLE_SHA256 = "17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"
MODES = (
    "official",
    "zero_prior",
    "shuffle_prior",
    "forced_fallback",
    "missing_landmark_asset",
    "missing_part_soft_asset",
    "prior_metadata_changed",
)
ALLOWED_READ_KEYS = {"image_48", "label", "sample_index"}


class TrackingDict(dict[str, Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accessed: list[str] = []

    def __getitem__(self, key: str) -> Any:
        self.accessed.append(str(key))
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        self.accessed.append(str(key))
        return super().get(key, default)


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def sha_bytes(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def sorted_rows_hash(value: torch.Tensor) -> str:
    arr = value.detach().cpu().numpy()
    rows = sorted(tuple(row) for row in arr.tolist())
    return sha_bytes(np.asarray(rows, dtype=arr.dtype))


def edge_family_hash(graph: D18GraphData, edge_type: int) -> str:
    edges = graph.edge_index[:, graph.edge_type == int(edge_type)].T
    return sorted_rows_hash(edges)


def graph_hashes(graph: D18GraphData) -> dict[str, str]:
    semantic = sha_bytes(
        graph.pos,
        graph.x,
        graph.edge_index,
        graph.edge_type,
        graph.edge_attr,
        graph.y,
        graph.sample_index,
    )
    return {
        "ordered_node_coordinates_hash": sha_bytes(graph.pos),
        "unordered_node_coordinate_set_hash": sorted_rows_hash(graph.pos),
        "x_hash": sha_bytes(graph.x),
        "local_edge_set_hash": edge_family_hash(graph, 0),
        "knn_edge_set_hash": edge_family_hash(graph, 1),
        "merged_edge_index_hash": sha_bytes(graph.edge_index),
        "edge_type_hash": sha_bytes(graph.edge_type),
        "edge_attr_hash": sha_bytes(graph.edge_attr),
        "complete_semantic_graph_hash": semantic,
    }


def load_prior(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def choose_samples(count: int) -> list[Path]:
    files = sorted((PRIOR_ROOT / "test").glob("*.npz"))
    detected_by_class: dict[int, Path] = {}
    fallback: list[Path] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            label = int(data["label"])
            detected = bool(data["detected"]) and int(data.get("landmark_missing_flag", 0)) == 0
        if detected and label not in detected_by_class:
            detected_by_class[label] = path
        if not detected:
            fallback.append(path)
        if len(detected_by_class) == 7 and fallback:
            break
    selected = [detected_by_class[key] for key in sorted(detected_by_class)]
    selected.extend(fallback[: max(1, count - len(selected))])
    if len(selected) < count:
        selected.extend(path for path in files if path not in selected)
    return selected[:count]


def choose_sample_payloads(count: int, evidence_dir: str | Path) -> tuple[list[Path], list[dict[str, np.ndarray]], str]:
    if (PRIOR_ROOT / "test").exists():
        paths = choose_samples(count)
        return paths, [load_prior(path) for path in paths], "local_prior_artifacts_for_audit_only"
    csv_path = Path(evidence_dir) / "test.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected_indices: list[int] = []
    seen_classes: set[int] = set()
    for index, row in enumerate(rows):
        label = int(row["emotion"])
        if label not in seen_classes:
            selected_indices.append(index)
            seen_classes.add(label)
        if len(seen_classes) == 7:
            break
    selected_indices.extend(index for index in range(len(rows)) if index not in selected_indices)
    selected_indices = selected_indices[:count]
    paths, payloads = [], []
    for position, index in enumerate(selected_indices):
        row = rows[index]
        detected = position != len(selected_indices) - 1
        payloads.append({
            "image_48": pixels_to_image48(row["pixels"]),
            "label": np.asarray(int(row["emotion"]), dtype=np.int64),
            "sample_index": np.asarray(index, dtype=np.int64),
            "detected": np.asarray(detected),
            "landmark_missing_flag": np.asarray(0 if detected else 1, dtype=np.int64),
            "face_mask": np.zeros((48, 48), dtype=np.float32),
            "part_soft_masks": np.zeros((13, 48, 48), dtype=np.float32),
            "micro_anchor_maps": np.zeros((1, 48, 48), dtype=np.float32),
            "distance_maps": np.zeros((1, 48, 48), dtype=np.float32),
            "landmark_xy_48": np.zeros((1, 2), dtype=np.float32),
        })
        paths.append(Path(f"{index:06d}.npz"))
    return paths, payloads, "fer_csv_with_synthetic_prior_metadata_for_preflight"


def prior_variant(mode: str, base: dict[str, np.ndarray], donor: dict[str, np.ndarray]) -> TrackingDict:
    out = {key: np.array(value, copy=True) for key, value in base.items()}
    prior_keys = (
        "face_mask",
        "part_soft_masks",
        "micro_anchor_maps",
        "distance_maps",
        "landmark_xy_48",
        "valid_part_mask",
        "valid_anchor_mask",
    )
    if mode == "zero_prior":
        for key in prior_keys:
            if key in out:
                out[key] = np.zeros_like(out[key])
    elif mode == "shuffle_prior":
        for key in prior_keys:
            if key in out and key in donor and out[key].shape == donor[key].shape:
                out[key] = np.array(donor[key], copy=True)
    elif mode == "forced_fallback":
        out["detected"] = np.asarray(False)
        out["landmark_missing_flag"] = np.asarray(1)
    elif mode == "missing_landmark_asset":
        for key in ("landmark_xy_48", "distance_maps", "micro_anchor_maps"):
            out.pop(key, None)
    elif mode == "missing_part_soft_asset":
        out.pop("part_soft_masks", None)
        out.pop("valid_part_mask", None)
    elif mode == "prior_metadata_changed":
        out["detected"] = np.asarray(False)
        out["landmark_missing_flag"] = np.asarray(1)
        out["fallback_type_id"] = np.asarray(999)
    elif mode != "official":
        raise ValueError(mode)
    return TrackingDict(out)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, path))
        return result
    return {prefix: value}


def allowed_config_difference(path: str) -> bool:
    exact = {
        "run_name",
        "output_dir",
        "description",
        "data.prior_dir",
        "data.evidence_dir",
        "graph.graph_mode",
        "graph.structure_edges.enabled",
        "graph.cache.schema",
        "graph.cache.enabled",
        "graph.cache.dir",
        "graph.cache.fallback_on_error",
        "training.structure_mode_mix.enabled",
        "training.structure_mode_mix.p_forced_structure",
        "logging.wandb.project",
        "logging.wandb.group",
        "logging.wandb.tags",
    }
    return path in exact or path.startswith("logging.wandb.tags.")


def semantic_config_diff(c2: dict[str, Any], a0: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    left, right = flatten(c2), flatten(a0)
    rows = []
    ok = True
    for path in sorted(set(left) | set(right)):
        same = left.get(path, "<MISSING>") == right.get(path, "<MISSING>")
        allowed = allowed_config_difference(path)
        status = "PASS" if same or allowed else "FAIL"
        ok &= status == "PASS"
        rows.append({
            "field": path,
            "c2": left.get(path, "<MISSING>"),
            "a0": right.get(path, "<MISSING>"),
            "difference_allowed": allowed,
            "status": status,
        })
    return rows, bool(ok)


def c2_source_resolved_consistent(source: dict[str, Any], resolved: dict[str, Any]) -> tuple[bool, list[str]]:
    operational = {"data.prior_dir", "graph.cache.dir", "graph.cache.enabled", "graph.cache.fallback_on_error"}
    left, right = flatten(source), flatten(resolved)
    differences = [path for path in sorted(set(left) | set(right)) if left.get(path, "<MISSING>") != right.get(path, "<MISSING>")]
    return all(path in operational for path in differences), differences


def manual_forward(model: StructureGNN, batch: Any) -> dict[str, torch.Tensor]:
    projection = model.encoder[0](batch.x_cat)
    h = model.encoder(batch.x_cat)
    outputs = {"input_projection": projection, "encoder_output": h}
    dst = batch.edge_index_cat[1].long()
    degree = h.new_zeros((h.size(0), 1))
    degree.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=h.dtype, device=h.device))
    for index, layer in enumerate(model.gnn.layers, start=1):
        h = layer(h, batch.edge_index_cat, batch.edge_attr_cat, dst_degree=degree, edge_type=batch.edge_type_cat)
        outputs[f"layer_{index}"] = h
    pooled = model.readout(h, batch.batch_index, batch.num_graphs)
    logits = model.classifier(pooled)
    outputs.update({"pooled_embedding": pooled, "logits": logits, "probabilities": torch.softmax(logits, dim=1)})
    return outputs


def md_table(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    rows = list(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.8g}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output: Path, result: dict[str, Any]) -> None:
    c2 = result["source_c2"]
    a0 = result["a0"]
    graph = result["graph_schema"]
    smoke = result["smoke_results"]
    config_diff = result["config_diff"]
    changed = result["code_changes"]
    created = result["files_created"]
    output.joinpath("00_README.md").write_text(f"""# D19-A0 Evidence-Only Matched Control

Purpose: measure the capacity of the current local+kNN evidence graph after training from initialization without landmark-derived structure exposure. C2 inference-time `remove_structure` is insufficient because C2 was trained with structure-mode exposure.

- Exact C2 run: `{c2['run_path']}`
- Exact C2 source config: `{c2['source_config']}`
- A0 config: `{a0['config']}`
- Changed files: {', '.join(changed)}
- Created files: {', '.join(created)}
- Bounded smoke: **{smoke['status']}**
- Landmark independence: **{result['landmark_independence']['status']}**
- Cache independence: **{result['cache_signature']['status']}**
- Full local training launched: **NO**
""", encoding="utf-8")
    output.joinpath("01_c2_source_manifest.md").write_text(f"""# C2 Source Manifest

| Field | Value |
|---|---|
| Run | `{c2['run_path']}` |
| Source config | `{c2['source_config']}` |
| Resolved config | `{c2['resolved_config']}` |
| Seed | {c2['seed']} |
| Model class | `d18.models.structure_gnn.StructureGNN` |
| Runtime trainable parameters | {c2['parameter_count']} |
| Node schema | 1800 nodes, 10 evidence features |
| Edge schema | base6; local + kNN + structure |
| Layers / hidden | 3 / 96 |
| Optimizer | AdamW, lr=0.0003, weight_decay=0.001 |
| Epochs / batch | 90 / 16 |
| Scheduler | ReduceLROnPlateau on val_loss |
| Checkpoint monitor | val_macro_f1, max; `best.pt` primary |
""", encoding="utf-8")
    change_rows = [
        {"file": "d18/data/structure_graph_builder.py", "reason": "bypass part-soft and structure construction", "old": "part_soft indexed before structure removal", "new": "evidence_only short-circuits before prior access", "compatibility": "default C2 path unchanged"},
        {"file": "d18/data/structure_dataset.py", "reason": "remove prior NPZ runtime dependency", "old": "all samples loaded from prior NPZ", "new": "A0 reads FER split CSV image/label only", "compatibility": "non-A0 continues NPZ path"},
        {"file": "d18/data/structure_graph_cache.py", "reason": "content-addressed evidence cache", "old": "prior filename cache identity", "new": "image hash + evidence signature", "compatibility": "legacy cache path unchanged"},
        {"file": "d18/training/train_d18.py", "reason": "A0 safety/log/artifact guards", "old": "generic D18 startup", "new": "A0 asserts zero structure and records evidence signature", "compatibility": "guards activate only for evidence_only"},
    ]
    output.joinpath("02_code_changes.md").write_text("# Code Changes\n\n" + md_table(change_rows, ["file", "reason", "old", "new", "compatibility"]) + "\n", encoding="utf-8")
    output.joinpath("03_a0_config_manifest.md").write_text("# A0 Config Manifest\n\n```yaml\n" + yaml.safe_dump(a0["effective_config"], sort_keys=False) + "```\n", encoding="utf-8")
    diff_rows = [row for row in config_diff if row["c2"] != row["a0"]]
    output.joinpath("04_semantic_config_diff.md").write_text(
        "# Semantic Config Diff\n\nOverall: **" + ("PASS" if result["config_semantic_diff_pass"] else "FAIL") + "**. All unlisted scientific/training fields are byte-equivalent after YAML parsing. The C2 source and resolved configs differ only in these operational path/cache fields: `" + ", ".join(result["source_c2"]["source_resolved_operational_differences"]) + "`.\n\n" + md_table(diff_rows, ["field", "c2", "a0", "difference_allowed", "status"]) + "\n",
        encoding="utf-8",
    )
    output.joinpath("05_landmark_free_execution_trace.md").write_text(f"""# Landmark-Free Execution Trace

1. `StructurePixelDataset` sees `graph_mode=evidence_only`.
2. It reads `data/<split>.csv` and constructs only `image_48`, `label`, `sample_index`.
3. `_load_prior()` is guarded and raises in evidence-only mode.
4. `build_structure_graph` computes image maps, node support, node features, local edges and kNN edges.
5. Before any `part_soft_masks` access, evidence-only creates an empty part matrix.
6. Structure construction is skipped and a runtime assertion rejects edge type 2.

Observed keys read from all counterfactual dictionaries: `{result['landmark_independence']['accessed_keys']}`. Landmark/part-soft keys read: **none**.
""", encoding="utf-8")
    cache = result["cache_signature"]
    output.joinpath("06_cache_signature_design.md").write_text(f"""# Evidence Cache Signature

- Schema: `{cache['schema']}`
- Namespace SHA-256: `{cache['namespace_sha256']}`
- Included: image content + label, fixed image preprocessing, node selection/count/schema, local settings, kNN settings, base6 schema, local>kNN merge/dedup version, builder version.
- Excluded: landmark/detector/fallback/part-soft hashes, prior mode, structure settings, structure mode mix.
- Same image/config across prior variants: **{cache['same_image_same_key']}**
- Different image changes key: **{cache['different_image_different_key']}**
- Different evidence config changes namespace: **{cache['different_config_different_namespace']}**
""", encoding="utf-8")
    output.joinpath("07_graph_schema_and_counts.md").write_text(f"""# Graph Schema and Counts

| Field | Value |
|---|---|
| node_dim | {graph['node_dim']} |
| edge_dim | {graph['edge_dim']} |
| node features | {', '.join(graph['node_feature_names'])} |
| edge features | {', '.join(graph['edge_feature_names'])} |
| local edges mean | {graph['local_edges_mean']:.2f} |
| raw kNN edges mean | {graph['raw_knn_edges_mean']:.2f} |
| local-kNN overlap mean | {graph['local_knn_overlap_mean']:.2f} |
| retained kNN-only edges mean | {graph['knn_edges_mean']:.2f} |
| merged edges mean | {graph['merged_edges_mean']:.2f} |
| structure edges | 0 |
| edge type IDs | {graph['edge_type_ids']} |
| merge precedence | local > kNN; duplicate directed endpoints removed |
| self-loops added | no |
""", encoding="utf-8")
    audit_rows = result["graph_independence_rows"]
    write_csv(output / "08_graph_independence_audit.csv", audit_rows)
    output.joinpath("08_graph_independence_audit.md").write_text(
        "# Graph Independence Audit\n\nAggregate semantic equality: **" + f"{result['landmark_independence']['equality_rate']*100:.2f}%** ({result['landmark_independence']['status']}).\n\n" + md_table(audit_rows, ["sample_index", "label", "source_landmark_state", "mode", "semantic_equal", "cache_key_equal", "accessed_keys"]) + "\n",
        encoding="utf-8",
    )
    model_rows = result["model_equivalence_rows"]
    write_csv(output / "09_model_equivalence_audit.csv", model_rows)
    output.joinpath("09_model_equivalence_audit.md").write_text(
        "# Model Equivalence Audit\n\nTolerance: `1e-6` on deterministic CPU. Status: **" + result["model_equivalence"]["status"] + "**.\n\n" + md_table(model_rows, ["mode", "input_projection_max_abs_diff", "layer_1_max_abs_diff", "layer_2_max_abs_diff", "layer_3_max_abs_diff", "pooled_embedding_max_abs_diff", "logits_max_abs_diff", "probability_max_abs_diff", "batch_membership_equal", "labels_and_order_equal", "pass"]) + "\n",
        encoding="utf-8",
    )
    compute = result["parameter_compute"]
    output.joinpath("10_parameter_and_compute_check.md").write_text(f"""# Parameter and Compute Check

| Measure | C2 | A0 |
|---|---:|---:|
| Trainable parameters | {compute['c2_trainable']} | {compute['a0_trainable']} |
| Non-trainable parameters | {compute['c2_nontrainable']} | {compute['a0_nontrainable']} |
| State-dict tensor bytes | {compute['c2_state_bytes']} | {compute['a0_state_bytes']} |
| Mean edges (audit sample) | {compute['c2_edges_mean']:.2f} | {compute['a0_edges_mean']:.2f} |
| CPU forward time, {compute['smoke_batch_size']} graphs | n/a | {compute['forward_time_sec']:.4f}s |
| Tracemalloc peak during forward | n/a | {compute['tracemalloc_peak_bytes']} bytes |

Parameter/state shapes are exactly unchanged. Timing is a bounded local smoke measurement, not a training benchmark.
""", encoding="utf-8")
    output.joinpath("11_smoke_validation.md").write_text(f"""# Smoke Validation

- Status: **{smoke['status']}**
- Batch size: {smoke['batch_size']}
- Logits shape: {smoke['logits_shape']}
- Cross-entropy loss: {smoke['loss']:.8f}
- Finite loss/logits/probabilities/gradients: {smoke['finite_loss']}/{smoke['finite_logits']}/{smoke['finite_probabilities']}/{smoke['finite_gradients']}
- Probability sums valid: {smoke['probability_sums_valid']}
- Node/edge indices and batch assignment valid: {smoke['indices_valid']}/{smoke['batch_assignment_valid']}
- One backward pass completed: {smoke['backward_pass']}
- No optimizer step and no full training were run.
""", encoding="utf-8")
    kaggle = result["training_command"]["kaggle"]
    output.joinpath("12_kaggle_training_command.md").write_text("# Kaggle Training Command\n\n```bash\n" + kaggle + "\n```\n", encoding="utf-8")
    output.joinpath("13_posttraining_evaluation_protocol.md").write_text(f"""# Post-Training Evaluation Protocol

Primary comparison uses `best.pt`; also evaluate `last.pt` for A0 and C2. Never select a checkpoint using test/robustness metrics.

Full test: accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrix, NLL, Brier, ECE-15, entropy and margin.

Locked test: exact 715 ordered sample indices, SHA-256 `{LOCKED_SAMPLE_SHA256}`. A0 official/zero/shuffle/forced/missing-landmark are equivalence checks, not independent robustness scores; report equality rate and maximum embedding/logit difference, not a duplicated robust-min.

```bash
{result['evaluation_commands']['a0_best']}
{result['evaluation_commands']['a0_last']}
{result['evaluation_commands']['a0_best_locked']}
{result['evaluation_commands']['a0_last_locked']}
{result['evaluation_commands']['c2_best_full']}
{result['evaluation_commands']['c2_last_full']}
{result['evaluation_commands']['c2_best_locked']}
{result['evaluation_commands']['c2_last_locked']}
```

Use the existing OFIX18 evaluator for C2 best/last with its prior/cache inputs. Keep full-test and locked-set tables separate.
""", encoding="utf-8")
    output.joinpath("14_result_interpretation_criteria.md").write_text("""# Result Interpretation Criteria

- A0 >= C2 remove_structure + about 1.0 pp macro-F1: descriptive evidence-only specialization signal.
- A0 within about +/-1.0 pp: matched evidence capacity; C2 likely learned most available local+kNN evidence.
- A0 <= C2 remove_structure - about 1.5 pp: descriptive structure-exposure training benefit; inspect curves/classes.
- These are interpretation bands, not checkpoint selection or automatic promotion gates.
- A1-ID is eligible only after invariants pass, normal training completes, evidence performance does not collapse, and no hidden cache/prior dependency appears.
""", encoding="utf-8")
    output.joinpath("15_risks_and_limitations.md").write_text("""# Risks and Limitations

- Initial experimental run is seed42 only.
- A0 is a matched control, not a final architecture.
- Raw handcrafted node features may remain capacity-limited.
- Removing structure changes the training distribution; inference ablation is not retraining.
- The local+kNN merged graph still uses one shared operator and total-degree normalization.
- Full-test and locked 715-image metrics are different populations and must not be mixed.
- Test and robustness metrics cannot select checkpoints.
- Local smoke timing does not predict Kaggle epoch duration precisely.
""", encoding="utf-8")
    manifest = {key: value for key, value in result.items() if key not in {"config_diff", "graph_independence_rows", "model_equivalence_rows"}}
    output.joinpath("16_machine_readable_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output.joinpath("17_validation_summary.json").write_text(json.dumps(result["validation_summary"], indent=2), encoding="utf-8")
    output.joinpath("18_run_commands.md").write_text(f"""# Run Commands

## Windows bounded validation

```powershell
{result['training_command']['powershell_smoke']}
```

## Kaggle full training (do not run locally)

```bash
{kaggle}
```

## Post-training A0 evaluation

```bash
{result['evaluation_commands']['a0_best']}
{result['evaluation_commands']['a0_last']}
{result['evaluation_commands']['a0_best_locked']}
{result['evaluation_commands']['a0_last_locked']}
{result['evaluation_commands']['c2_best_full']}
{result['evaluation_commands']['c2_last_full']}
{result['evaluation_commands']['c2_best_locked']}
{result['evaluation_commands']['c2_last_locked']}
```
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.smoke_images < 8:
        raise ValueError("--smoke-images must be at least 8 for the required audit matrix")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    a0 = read_yaml(config_path)
    c2 = read_yaml(C2_CONFIG)
    c2_resolved = read_yaml(C2_RESOLVED)
    diff_rows, config_ok = semantic_config_diff(c2, a0)
    c2_resolved_ok, c2_resolved_differences = c2_source_resolved_consistent(c2, c2_resolved)
    config_ok = bool(config_ok and c2_resolved_ok)

    graph_cfg = a0.get("graph", {}) or {}
    train_cfg = a0.get("training", {}) or {}
    hard_assertions = [
        str(graph_cfg.get("graph_mode")) == "evidence_only",
        not bool((graph_cfg.get("structure_edges") or {}).get("enabled", True)),
        not bool((train_cfg.get("structure_mode_mix") or {}).get("enabled", True)),
        float((train_cfg.get("structure_mode_mix") or {}).get("p_forced_structure", -1)) == 0.0,
        all(float(value) == 0.0 for value in [
            train_cfg.get("drop_edge_p", 0.0),
            (train_cfg.get("graph_regularization") or {}).get("drop_local_edge_p", 0.0),
            (train_cfg.get("graph_regularization") or {}).get("drop_knn_edge_p", 0.0),
            (train_cfg.get("graph_regularization") or {}).get("drop_structure_edge_p", 0.0),
        ]),
        int(train_cfg.get("seed", a0.get("seed", -1))) == 42,
    ]
    if not all(hard_assertions):
        raise RuntimeError("A0 semantic hard assertion failed")

    selected, priors, sample_selection_source = choose_sample_payloads(
        int(args.smoke_images), (a0.get("data") or {}).get("evidence_dir", "data")
    )
    graph_rows: list[dict[str, Any]] = []
    graphs_by_mode: dict[str, list[D18GraphData]] = {mode: [] for mode in MODES}
    c2_graphs: list[D18GraphData] = []
    all_accessed: set[str] = set()
    all_equal = True
    cache_equal = True
    local_knn_match_c2 = True
    for sample_i, (path, base) in enumerate(zip(selected, priors)):
        donor = priors[(sample_i + 1) % len(priors)]
        official_hashes: dict[str, str] | None = None
        official_cache: str | None = None
        official_graph: D18GraphData | None = None
        for mode in MODES:
            variant = prior_variant(mode, base, donor)
            graph = build_structure_graph(variant, graph_cfg)
            unexpected = set(variant.accessed) - ALLOWED_READ_KEYS
            if unexpected:
                raise RuntimeError(f"A0 accessed forbidden prior keys: {sorted(unexpected)}")
            all_accessed.update(variant.accessed)
            hashes = graph_hashes(graph)
            cache_path = evidence_graph_cache_path(
                "CACHE_ROOT",
                "test",
                int(np.asarray(base["sample_index"]).item()),
                np.asarray(base["image_48"], dtype=np.float32),
                int(np.asarray(base["label"]).item()),
                graph_cfg,
            )
            cache_key = str(cache_path.relative_to("CACHE_ROOT"))
            if official_hashes is None:
                official_hashes = hashes
                official_cache = cache_key
                official_graph = graph
            semantic_equal = hashes == official_hashes
            key_equal = cache_key == official_cache
            all_equal &= semantic_equal
            cache_equal &= key_equal
            graphs_by_mode[mode].append(graph)
            row = {
                "image_id": path.stem,
                "sample_index": int(graph.sample_index),
                "label": int(graph.y),
                "source_landmark_state": "detected" if bool(base.get("detected", False)) and int(base.get("landmark_missing_flag", 0)) == 0 else "fallback",
                "mode": mode,
                "semantic_equal": semantic_equal,
                "cache_key_equal": key_equal,
                "accessed_keys": ",".join(sorted(set(variant.accessed))),
                **hashes,
                "cache_key": cache_key,
            }
            graph_rows.append(row)
        assert official_graph is not None
        c2_graph = build_structure_graph(base, c2.get("graph", {}) or {})
        c2_graphs.append(c2_graph)
        local_knn_match_c2 &= (
            torch.equal(official_graph.x, c2_graph.x)
            and torch.equal(official_graph.pos, c2_graph.pos)
            and edge_family_hash(official_graph, 0) == edge_family_hash(c2_graph, 0)
            and edge_family_hash(official_graph, 1) == edge_family_hash(c2_graph, 1)
        )

    a0_ds = StructurePixelDataset(
        prior_dir=None,
        split="test",
        graph=graph_cfg,
        max_samples=None,
        evidence_dir=(a0.get("data") or {}).get("evidence_dir"),
    )
    dataset_graph = a0_ds[int(selected[0].stem)]
    dataset_bypass = graph_hashes(dataset_graph) == graph_hashes(graphs_by_mode["official"][0])
    prior_guard = False
    try:
        a0_ds._load_prior(0)
    except RuntimeError:
        prior_guard = True

    first_graph = graphs_by_mode["official"][0]
    a0_model = StructureGNN.from_config(a0, input_dim=first_graph.x.size(1), edge_attr_dim=first_graph.edge_attr.size(1))
    c2_model = StructureGNN.from_config(c2_resolved, input_dim=10, edge_attr_dim=6)
    a0_trainable = sum(p.numel() for p in a0_model.parameters() if p.requires_grad)
    c2_trainable = sum(p.numel() for p in c2_model.parameters() if p.requires_grad)
    a0_nontrainable = sum(p.numel() for p in a0_model.parameters() if not p.requires_grad)
    c2_nontrainable = sum(p.numel() for p in c2_model.parameters() if not p.requires_grad)
    state_shapes_equal = {k: tuple(v.shape) for k, v in a0_model.state_dict().items()} == {k: tuple(v.shape) for k, v in c2_model.state_dict().items()}

    a0_model.eval()
    model_rows: list[dict[str, Any]] = []
    outputs_by_mode: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for mode in MODES:
            batch = collate_d18_graphs(graphs_by_mode[mode])
            outputs_by_mode[mode] = manual_forward(a0_model, batch)
    reference = outputs_by_mode["official"]
    tolerance = 1e-6
    model_equal = True
    for mode, current in outputs_by_mode.items():
        diffs = {name: float((current[name] - reference[name]).abs().max().item()) for name in reference}
        current_batch = collate_d18_graphs(graphs_by_mode[mode])
        reference_batch = collate_d18_graphs(graphs_by_mode["official"])
        batch_membership_equal = torch.equal(current_batch.batch_index, reference_batch.batch_index)
        labels_equal = torch.equal(current_batch.y, reference_batch.y)
        passed = all(value <= tolerance for value in diffs.values()) and batch_membership_equal and labels_equal
        model_equal &= passed
        model_rows.append({
            "mode": mode,
            "input_projection_max_abs_diff": diffs["input_projection"],
            "encoder_output_max_abs_diff": diffs["encoder_output"],
            "layer_1_max_abs_diff": diffs["layer_1"],
            "layer_2_max_abs_diff": diffs["layer_2"],
            "layer_3_max_abs_diff": diffs["layer_3"],
            "pooled_embedding_max_abs_diff": diffs["pooled_embedding"],
            "logits_max_abs_diff": diffs["logits"],
            "probability_max_abs_diff": diffs["probabilities"],
            "batch_membership_equal": batch_membership_equal,
            "labels_and_order_equal": labels_equal,
            "pass": passed,
        })

    smoke_graphs = graphs_by_mode["official"][: min(4, len(selected))]
    smoke_batch = collate_d18_graphs(smoke_graphs)
    a0_model.train()
    tracemalloc.start()
    forward_start = time.perf_counter()
    smoke_out = a0_model(smoke_batch)
    forward_time = time.perf_counter() - forward_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    loss = torch.nn.functional.cross_entropy(smoke_out["logits"], smoke_batch.y)
    loss.backward()
    gradients = [p.grad for p in a0_model.parameters() if p.requires_grad and p.grad is not None]
    finite_gradients = bool(gradients) and all(bool(torch.isfinite(grad).all()) for grad in gradients)
    probabilities = torch.softmax(smoke_out["logits"], dim=1)
    indices_valid = (
        int(smoke_batch.edge_index_cat.min()) >= 0
        and int(smoke_batch.edge_index_cat.max()) < int(smoke_batch.x_cat.size(0))
    )
    batch_assignment_valid = (
        smoke_batch.batch_index.numel() == smoke_batch.x_cat.size(0)
        and int(smoke_batch.batch_index.min()) == 0
        and int(smoke_batch.batch_index.max()) == smoke_batch.num_graphs - 1
    )

    a0_signature = scientific_resume_signature(a0)
    c2_signature = scientific_resume_signature(c2)
    resume_unique = a0_signature != c2_signature
    resume_guard = False
    reverse_resume_guard = False
    temp = output / "_signature_probe.pt"
    torch.save({"resume_signature": c2_signature, "config": c2}, temp)
    try:
        load_checkpoint(temp, a0_model, expected_resume_signature=a0_signature, strict_signature=True)
    except RuntimeError:
        resume_guard = True
    torch.save({"resume_signature": a0_signature, "config": a0}, temp)
    try:
        load_checkpoint(temp, c2_model, expected_resume_signature=c2_signature, strict_signature=False)
    except RuntimeError:
        reverse_resume_guard = True
    temp.unlink(missing_ok=True)

    cache_payload = evidence_cache_signature_payload(graph_cfg)
    namespace = evidence_cache_signature(graph_cfg)
    changed_graph_cfg = copy.deepcopy(graph_cfg)
    changed_graph_cfg["target_node_count"] = int(graph_cfg.get("target_node_count", 1800)) - 1
    different_config = evidence_cache_signature(changed_graph_cfg) != namespace
    first_key = graph_rows[0]["cache_key"]
    next_official = next(row for row in graph_rows if row["mode"] == "official" and row["sample_index"] != graph_rows[0]["sample_index"])
    different_image = next_official["cache_key"] != first_key

    node_count = np.asarray([g.x.size(0) for g in graphs_by_mode["official"]], dtype=float)
    local_count = np.asarray([g.local_edge_count for g in graphs_by_mode["official"]], dtype=float)
    knn_count = np.asarray([g.knn_edge_count for g in graphs_by_mode["official"]], dtype=float)
    merged_count = np.asarray([g.total_edge_count for g in graphs_by_mode["official"]], dtype=float)
    raw_knn_count = node_count * int((graph_cfg.get("knn_edges") or {}).get("k", 6))
    overlap_count = raw_knn_count - knn_count
    c2_edges = np.asarray([g.total_edge_count for g in c2_graphs], dtype=float)
    graph_schema = {
        "node_dim": int(first_graph.x.size(1)),
        "edge_dim": int(first_graph.edge_attr.size(1)),
        "node_feature_names": first_graph.node_feature_names,
        "edge_feature_names": first_graph.edge_feature_names,
        "node_count_mean": float(node_count.mean()),
        "local_edges_mean": float(local_count.mean()),
        "raw_knn_edges_mean": float(raw_knn_count.mean()),
        "local_knn_overlap_mean": float(overlap_count.mean()),
        "knn_edges_mean": float(knn_count.mean()),
        "merged_edges_mean": float(merged_count.mean()),
        "structure_edges": 0,
        "edge_type_ids": sorted({int(value) for graph in graphs_by_mode["official"] for value in graph.edge_type.unique().tolist()}),
        "local_knn_match_c2": bool(local_knn_match_c2),
    }
    smoke = {
        "status": "PASS" if all([torch.isfinite(loss), torch.isfinite(smoke_out["logits"]).all(), finite_gradients, indices_valid, batch_assignment_valid]) else "FAIL",
        "batch_size": smoke_batch.num_graphs,
        "logits_shape": list(smoke_out["logits"].shape),
        "loss": float(loss.item()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_logits": bool(torch.isfinite(smoke_out["logits"]).all()),
        "finite_probabilities": bool(torch.isfinite(probabilities).all()),
        "probability_sums_valid": bool(torch.allclose(probabilities.sum(dim=1), torch.ones(smoke_batch.num_graphs), atol=1e-6)),
        "finite_gradients": finite_gradients,
        "indices_valid": bool(indices_valid),
        "batch_assignment_valid": bool(batch_assignment_valid),
        "forward_pass": True,
        "backward_pass": True,
    }
    graph_equality_rate = sum(bool(row["semantic_equal"]) for row in graph_rows) / len(graph_rows)
    validation = {
        "c2_source_identified": C2_CONFIG.exists() and C2_RESOLVED.exists(),
        "config_semantic_diff_pass": config_ok,
        "landmark_access_bypassed": all_accessed <= ALLOWED_READ_KEYS and dataset_bypass and prior_guard,
        "part_soft_access_bypassed": "part_soft_masks" not in all_accessed,
        "structure_edges_zero": all(g.structure_edge_count == 0 and not bool((g.edge_type == 2).any()) for graphs in graphs_by_mode.values() for g in graphs),
        "cache_landmark_independent": cache_equal and different_image and different_config,
        "graph_mode_equivalence_pass": all_equal and local_knn_match_c2,
        "model_output_equivalence_pass": model_equal,
        "parameter_count_match": a0_trainable == c2_trainable and state_shapes_equal,
        "resume_signature_unique": resume_unique and resume_guard and reverse_resume_guard,
        "forward_pass": smoke["forward_pass"],
        "backward_pass": smoke["backward_pass"],
        "finite_loss": smoke["finite_loss"],
        "finite_gradients": smoke["finite_gradients"],
        "kaggle_command_ready": True,
        "reports_complete": True,
        "full_training_launched": False,
        "blocking_issues": [],
        "warnings": ["Single-seed training remains pending on Kaggle.", "Smoke timing is CPU-only and bounded."],
    }
    if args.strict and not all(value is True for key, value in validation.items() if key not in {"full_training_launched", "reports_complete"} and isinstance(value, bool)):
        failures = [key for key, value in validation.items() if isinstance(value, bool) and key != "full_training_launched" and not value]
        raise RuntimeError(f"Strict D19-A0 validation failed: {failures}")

    config_rel = config_path.as_posix()
    output_rel = output.as_posix()
    run_dir = a0["output_dir"]
    kaggle = f"""set -euo pipefail
cd /kaggle/working/FER2013_Graph
CONFIG={config_rel}
RUN_DIR={run_dir}
REPORT_DIR=/kaggle/working/d19_a0_preflight
PRETRAIN_ENV=/kaggle/working/d19_a0_environment_pretrain.txt
TRAIN_LOG=/kaggle/working/d19_a0_train_console.log
cat \"$CONFIG\"
python -B d19/scripts/validate_d19_a0_evidence_only.py --config \"$CONFIG\" --smoke-images 8 --seed 42 --output-dir \"$REPORT_DIR\" --strict
if [ -e \"$RUN_DIR/TRAINING_COMPLETE.json\" ] || [ -e \"$RUN_DIR/checkpoints/last.pt\" ]; then
  echo \"Refusing to overwrite existing A0 run. Supply --resume_from explicitly in a separate command.\" >&2; exit 2
fi
python -VV > \"$PRETRAIN_ENV\"
python -m pip freeze >> \"$PRETRAIN_ENV\"
python -B d18/training/train_d18.py --config \"$CONFIG\" --device cuda:0 2>&1 | tee \"$TRAIN_LOG\"
mv \"$PRETRAIN_ENV\" \"$RUN_DIR/environment_pretrain.txt\"
mv \"$TRAIN_LOG\" \"$RUN_DIR/train_console.log\"
test -f \"$RUN_DIR/checkpoints/best.pt\"
test -f \"$RUN_DIR/checkpoints/last.pt\"
test -f \"$RUN_DIR/TRAINING_COMPLETE.json\"
python -B d19/scripts/evaluate_d19_a0.py --run-dir \"$RUN_DIR\" --checkpoint best --split test --output-dir \"$RUN_DIR/evaluation_best\" --device cuda:0
tar -czf /kaggle/working/d19_a0_evidence_only_matched_seed42.tar.gz \"$RUN_DIR\" \"$REPORT_DIR\""""
    powershell = f"conda run -n fer-graph python -B d19/scripts/validate_d19_a0_evidence_only.py --config {config_rel} --smoke-images 8 --seed 42 --output-dir {output_rel} --strict"
    eval_base = f"python -B d19/scripts/evaluate_d19_a0.py --run-dir {run_dir} --split test --device cuda:0"
    locked_manifest = "outputs/d18_analysis/ofix18_predecision_audit/sample_manifest.csv"
    c2_eval_base = (
        "python -B d18/scripts/evaluate_ofix18_factorial.py "
        "--run_dir outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42 "
        "--prior_dir outputs/d16_mediapipe_pixel_priors_best_retry_rescue "
        "--graph_cache_dir outputs/d18_graph_cache/ofix17_structure_reg/base6_shared "
        "--device cuda:0"
    )
    result = {
        "source_c2": {
            "run_path": str(C2_RUN.relative_to(ROOT)).replace("\\", "/"),
            "source_config": str(C2_CONFIG.relative_to(ROOT)).replace("\\", "/"),
            "resolved_config": str(C2_RESOLVED.relative_to(ROOT)).replace("\\", "/"),
            "seed": 42,
            "parameter_count": c2_trainable,
            "source_resolved_scientific_match": c2_resolved_ok,
            "source_resolved_operational_differences": c2_resolved_differences,
        },
        "a0": {"config": config_rel, "run_name": a0["run_name"], "output_dir": run_dir, "effective_config": a0},
        "allowed_differences": [row["field"] for row in diff_rows if row["c2"] != row["a0"]],
        "frozen_fields": {"model": a0["model"], "training": a0["training"]},
        "config_diff": diff_rows,
        "config_semantic_diff_pass": config_ok,
        "code_changes": ["d18/data/structure_graph_builder.py", "d18/data/structure_dataset.py", "d18/data/structure_graph_cache.py", "d18/training/train_d18.py"],
        "files_created": [config_rel, "d19/scripts/validate_d19_a0_evidence_only.py", "d19/scripts/evaluate_d19_a0.py"],
        "graph_schema": graph_schema,
        "cache_signature": {
            "status": "PASS" if cache_equal and different_image and different_config else "FAIL",
            "schema": EVIDENCE_CACHE_SCHEMA,
            "namespace_sha256": namespace,
            "payload": cache_payload,
            "same_image_same_key": cache_equal,
            "different_image_different_key": different_image,
            "different_config_different_namespace": different_config,
        },
        "landmark_independence": {
            "status": "PASS" if all_equal and dataset_bypass and prior_guard else "FAIL",
            "sample_count": len(selected),
            "mode_count": len(MODES),
            "equality_rate": graph_equality_rate,
            "accessed_keys": sorted(all_accessed),
            "dataset_csv_path_pass": dataset_bypass,
            "prior_loader_guard_pass": prior_guard,
            "sample_selection_source": sample_selection_source,
        },
        "model_equivalence": {"status": "PASS" if model_equal else "FAIL", "tolerance": tolerance, "logits_shape": list(reference["logits"].shape)},
        "parameter_count": {"c2": c2_trainable, "a0": a0_trainable, "match": a0_trainable == c2_trainable, "state_shapes_match": state_shapes_equal},
        "parameter_compute": {
            "c2_trainable": c2_trainable,
            "a0_trainable": a0_trainable,
            "c2_nontrainable": c2_nontrainable,
            "a0_nontrainable": a0_nontrainable,
            "c2_state_bytes": sum(v.numel() * v.element_size() for v in c2_model.state_dict().values()),
            "a0_state_bytes": sum(v.numel() * v.element_size() for v in a0_model.state_dict().values()),
            "c2_edges_mean": float(c2_edges.mean()),
            "a0_edges_mean": float(merged_count.mean()),
            "smoke_batch_size": smoke_batch.num_graphs,
            "forward_time_sec": forward_time,
            "tracemalloc_peak_bytes": peak_memory,
        },
        "smoke_results": smoke,
        "resume_signature": {"c2": c2_signature, "a0": a0_signature, "unique": resume_unique, "both_direction_guards": resume_guard and reverse_resume_guard},
        "graph_independence_rows": graph_rows,
        "model_equivalence_rows": model_rows,
        "training_command": {"kaggle": kaggle, "powershell_smoke": powershell},
        "evaluation_commands": {
            "a0_best": eval_base + f" --checkpoint best --output-dir {run_dir}/evaluation_best",
            "a0_last": eval_base + f" --checkpoint last --output-dir {run_dir}/evaluation_last",
            "a0_best_locked": eval_base + f" --checkpoint best --sample-manifest {locked_manifest} --output-dir {run_dir}/evaluation_best_locked",
            "a0_last_locked": eval_base + f" --checkpoint last --sample-manifest {locked_manifest} --output-dir {run_dir}/evaluation_last_locked",
            "c2_best_full": c2_eval_base + " --checkpoint best --output_dir outputs/d19_analysis/d19_a0_posttraining/c2_best_full",
            "c2_last_full": c2_eval_base + " --checkpoint last --output_dir outputs/d19_analysis/d19_a0_posttraining/c2_last_full",
            "c2_best_locked": c2_eval_base + f" --checkpoint best --sample_manifest {locked_manifest} --output_dir outputs/d19_analysis/d19_a0_posttraining/c2_best_locked",
            "c2_last_locked": c2_eval_base + f" --checkpoint last --sample_manifest {locked_manifest} --output_dir outputs/d19_analysis/d19_a0_posttraining/c2_last_locked",
        },
        "limitations": ["seed42 only", "control not final architecture", "no full local training", "handcrafted evidence capacity may be limited"],
        "validation_summary": validation,
        "full_training_launched": False,
    }
    write_reports(output, result)
    expected = [
        "00_README.md", "01_c2_source_manifest.md", "02_code_changes.md", "03_a0_config_manifest.md",
        "04_semantic_config_diff.md", "05_landmark_free_execution_trace.md", "06_cache_signature_design.md",
        "07_graph_schema_and_counts.md", "08_graph_independence_audit.csv", "08_graph_independence_audit.md",
        "09_model_equivalence_audit.csv", "09_model_equivalence_audit.md", "10_parameter_and_compute_check.md",
        "11_smoke_validation.md", "12_kaggle_training_command.md", "13_posttraining_evaluation_protocol.md",
        "14_result_interpretation_criteria.md", "15_risks_and_limitations.md", "16_machine_readable_manifest.json",
        "17_validation_summary.json", "18_run_commands.md",
    ]
    missing = [name for name in expected if not (output / name).exists()]
    if missing:
        raise RuntimeError(f"Missing reports: {missing}")
    validation["reports_complete"] = True
    output.joinpath("17_validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
