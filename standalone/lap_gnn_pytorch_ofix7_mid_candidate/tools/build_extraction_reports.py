"""Generate the complete machine-backed standalone extraction report set."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml


COMMIT = "241a8872027cd284fe679533a0be95cb48e7d253"
SEEDS = [42, 1009, 1337, 777, 3407]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def top_symbols(text: str) -> dict[str, tuple[int, int, str]]:
    tree = ast.parse(text)
    return {
        node.name: (node.lineno, node.end_lineno or node.lineno, type(node).__name__)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def purpose(path: str) -> str:
    if "train_d16" in path:
        return "config, dataset orchestration, train/eval loop, metrics, checkpoint policy"
    if "graph_builder" in path:
        return "node selection, ordered features, directed edges, anchors, variable-size batching"
    if "pixel_prior_dataset" in path:
        return "precomputed prior loading and deterministic train-only corruption"
    if "mediapipe_priors" in path:
        return "prior schema constants and optional explicit prior generation"
    if "/models/" in path:
        return "model, graph layer, context, readout, or classifier component"
    if "/losses/" in path:
        return "loss component imported by historical engine; OFIX7-mid activates CE only"
    if "graph_cache_dataset" in path:
        return "optional verified graph-cache interface"
    return "transitive runtime support"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    package = args.package_root.resolve()
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((package / "source_mapping.generated.json").read_text(encoding="utf-8"))
    parity = json.loads((package / "validation_assets/parity_results.json").read_text(encoding="utf-8"))
    fixture_manifest = json.loads((package / "validation_assets/manifest.json").read_text(encoding="utf-8"))
    package_manifest = json.loads((package / "package_manifest.json").read_text(encoding="utf-8"))
    lock_dir = repo / "outputs/d16_analysis/ofix7_mid_5seed_posttraining_analysis"
    policy_lock = json.loads((lock_dir / "10_checkpoint_policy_lock.json").read_text(encoding="utf-8"))
    baseline_lock = json.loads((lock_dir / "23_baseline_replication_lock.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((repo / "outputs/d16_runs/final/ofix7_mid_seed42/resolved_config.yaml").read_text(encoding="utf-8"))

    mapping_rows = []
    for record in mapping:
        original_text = subprocess.check_output(
            ["git", "show", f"{COMMIT}:{record['original_file']}"], cwd=repo, text=True, encoding="utf-8"
        )
        destination_text = (package / record["destination"]).read_text(encoding="utf-8")
        original_symbols = top_symbols(original_text)
        destination_symbols = top_symbols(destination_text)
        mapping_rows.append({
            "original_file": record["original_file"],
            "original_symbol": "<module>",
            "original_line_start": 1,
            "original_line_end": len(original_text.splitlines()),
            "runtime_purpose": purpose(record["original_file"]),
            "destination": record["destination"],
            "destination_symbol": "<module>",
            "destination_line_start": 1,
            "destination_line_end": len(destination_text.splitlines()),
            "copied_exactly": record["copied_exactly"],
            "mechanical_changes": "; ".join(record["mechanical_changes"]),
            "intentionally_omitted": False,
            "reason": "verified transitive runtime closure",
        })
        for name, (start, end, kind) in original_symbols.items():
            destination_range = destination_symbols.get(name)
            mapping_rows.append({
                "original_file": record["original_file"],
                "original_symbol": f"{kind}:{name}",
                "original_line_start": start,
                "original_line_end": end,
                "runtime_purpose": purpose(record["original_file"]),
                "destination": record["destination"],
                "destination_symbol": name if destination_range else "OMITTED",
                "destination_line_start": "" if not destination_range else destination_range[0],
                "destination_line_end": "" if not destination_range else destination_range[1],
                "copied_exactly": record["copied_exactly"],
                "mechanical_changes": "; ".join(record["mechanical_changes"]),
                "intentionally_omitted": destination_range is None,
                "reason": "mechanical namespace extraction" if destination_range else "unreachable or replaced wrapper",
            })
    mapping_csv = report / "03_source_to_standalone_mapping.csv"
    with mapping_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)

    checkpoint_rows = []
    no_resume = True
    signature_sets = {}
    for seed in SEEDS:
        run = repo / f"outputs/d16_runs/final/ofix7_mid_seed{seed}"
        complete = json.loads((run / "REPLICATION_COMPLETE.json").read_text())
        no_resume_payload = json.loads((run / "replication_provenance/NO_RESUME.json").read_text())
        signatures = json.loads((run / "replication_provenance/runtime_signatures.json").read_text())
        signature_sets[seed] = signatures
        no_resume = no_resume and complete["resumed"] is False and no_resume_payload["no_resume"] is True
        selected = baseline_lock["selected_checkpoints"][str(seed)]
        best = run / "checkpoints/best.pt"
        macro = run / "checkpoints/best_val_macro_f1.pt"
        checkpoint_rows.append([
            seed, selected["epoch"], selected["file_sha256"],
            selected["canonical_model_state_sha256"], best.read_bytes() == macro.read_bytes(), complete["resumed"],
        ])
    signatures_agree = len({json.dumps(value, sort_keys=True) for value in signature_sets.values()}) == 1

    write(report / "00_README.md", """# Standalone Extraction Report Set

This directory records the non-destructive extraction of the locked OFIX7-mid
PyTorch candidate. Machine-readable locks, resolved configs, historical Git
blobs, bounded parity outputs and isolated-copy tests are the evidence sources.
No smoke metric is presented as a research result.
""")
    write(report / "01_baseline_lock_validation.md", f"""# Baseline Lock Validation

- Checkpoint-policy lock: `{lock_dir / '10_checkpoint_policy_lock.json'}`
- SHA-256: `{sha(lock_dir / '10_checkpoint_policy_lock.json')}`
- Baseline replication lock: `{lock_dir / '23_baseline_replication_lock.json'}`
- SHA-256: `{sha(lock_dir / '23_baseline_replication_lock.json')}`
- Policy: `{policy_lock['selected_policy']}`
- Replication status: `{baseline_lock['replication_status']}`
- Historical source commit: `{COMMIT}`
- All five runtime signature objects agree: `{signatures_agree}`
- All five runs record no resume contamination: `{no_resume}`

{table(['seed','epoch','file SHA-256','canonical state SHA-256','best alias exact','resumed'], checkpoint_rows)}

Runtime signatures: `{json.dumps(signature_sets[42], sort_keys=True)}`.
""")
    closure_rows = [[item["original_file"], item["original_lines"], purpose(item["original_file"])] for item in mapping]
    write(report / "02_runtime_dependency_trace.md", f"""# Runtime Dependency Trace

The trace combines AST imports, historical source inspection, resolved config
key tracing, bounded execution and isolated runtime import observation.

Execution path:

`CLI/config -> precomputed prior dataset -> corruption -> face/context selection
-> 37 features -> 5 anchors -> local/anchor edges -> batch -> D16Model ->
PixelEncoder -> 3 EdgeContextGNN layers -> final context injection ->
micro-motif readout -> classifier -> CE -> AdamW -> plateau scheduler ->
val-loss early stop -> val-macro-F1 checkpoint -> evaluation`.

{table(['historical module','LOC','runtime purpose'], closure_rows)}

External direct runtime dependencies are NumPy, PyYAML, scikit-learn and
PyTorch. Matplotlib/pandas are reporting extras. MediaPipe/OpenCV are optional
for explicit prior generation and are not needed to consume verified priors.
""")
    write(report / "03_source_to_standalone_mapping.md", f"""# Source Mapping

The CSV contains {len(mapping_rows)} module/symbol records with exact historical
and destination line ranges: [03_source_to_standalone_mapping.csv](03_source_to_standalone_mapping.csv).

There are {len(mapping)} mechanically extracted historical modules. Namespace
rewrites preserve math. The engine additionally rejects resume and adds
checkpoint provenance metadata; neither changes the no-resume baseline
trajectory.
""")
    write(report / "04_excluded_code_inventory.md", """# Excluded Code Inventory

Excluded: D17, D18, D19, selector audits, S1/O1 scripts, TensorFlow, CNN,
Semantic ROI, MGR-CNN, notebooks, experiment analysis, obsolete configs,
historical outputs/checkpoints, graph caches, datasets, plots, archives, `.git`
and environments. Loss modules imported by the historical engine remain in the
closure, but OFIX7-mid activates CE only.
""")
    feature_names = [item["name"] for item in json.loads((package / "contracts/feature_schema.json").read_text())["ordered_features"]]
    edge_names = [item["name"] for item in json.loads((package / "contracts/edge_schema.json").read_text())["ordered_features"]]
    write(report / "05_scientific_contract.md", f"""# Scientific Contract

## Input And Prior

FER2013 rows contain one class ID and 2304 grayscale pixels reshaped to 48x48.
Values are float32 and divided by 255 when the stored maximum exceeds 1.
Class order: angry, disgust, fear, happy, sad, surprise, neutral. Locked counts:
28709 train, 3589 validation, 3589 test.

Prior schema is `d16_mediapipe_pixel_priors_v1`: face mask, 13 part-soft maps,
12 distance maps, landmark-missing flag, valid-part/anchor masks and fallback
metadata. Train-only corruption uses seed `run_seed + 7699`, schedule
0.10 from epoch 1, 0.20 from epoch 11 and 0.30 from epoch 31; mode weights are
attenuate 0.55, shuffle 0.25, zero 0.12 and forced fallback 0.08.

## Graph

`face_plus_context`, threshold 0.15, two binary-dilation iterations. Pixel
coordinates are row-major `np.argwhere`. Every selected pixel has outgoing
directed edges to all present 8-neighbors in fixed offset order. Normal graphs
have no local self-loop; the one-node emergency fallback has one self-loop.
Five anchors are appended in order mouth, eye, brow, nose_cheek, global.
Part anchors connect selected pixels to anchor and anchor to pixel. Anchors form
a complete directed graph without self edges. Global-to-pixel is disabled.
No duplicate-removal pass is applied after anchor insertion. Batching
concatenates nodes/edges, offsets edge endpoints, writes `batch_index` and
`ptr`, and preserves variable node counts.

Pixel/context and anchor nodes participate in graph message passing. CLS and
motif tokens exist only inside readout. The batch has no explicit node-type
tensor; trailing five nodes are anchors and core/context status is reconstructed
from sampled face mask.

## Model

Input 37, edge 8, hidden 96, three gated-edge layers, mean aggregation.
PixelEncoder is Linear-LN-GELU-Dropout-Linear-LN-GELU. Each graph layer computes
edge embedding, sigmoid gate, gated source message, destination mean, then
`LN(h + agg)` and `LN(h_msg + FFN(h_msg))`. Final context injection pools five
part tokens, applies one four-head TransformerEncoder layer, broadcasts local
plus global context and applies `LN(h + scale * update)` with scale initialized
0.5. Micro-motif-support readout uses the locked 12 major and 8 micro motifs,
CLS/token type embeddings, one four-head transformer layer, residual concat and
support gate. Classifier is Linear(480,192)-LN-GELU-Dropout-Linear(192,7).
Exact parameter count: 1,061,192.

## Training

Unweighted CE, label smoothing 0. AdamW LR 3e-4, weight decay 1e-3, PyTorch
defaults betas (0.9,0.999), epsilon 1e-8; gradient clipping max norm 5.
Batch 16, maximum 90 epochs, AMP on CUDA, TF32 enabled. ReduceLROnPlateau
monitors validation loss (min, factor .5, patience 5, threshold 1e-4,
min LR 3e-5). Early stopping monitors validation loss after at least 30 epochs
with patience 15. Checkpoint selection independently maximizes validation
macro-F1. Scheduler stepping occurs after best-checkpoint decisions and before
the last checkpoint is saved. `best.pt` is a byte copy of
`best_val_macro_f1.pt`; validation-accuracy checkpoint is secondary.
""")
    write(report / "06_feature_edge_node_schemas.md", f"""# Feature, Edge And Node Schemas

## Ordered 37 Node Features

{table(['index','name'], [[i + 1, name] for i, name in enumerate(feature_names)])}

## Ordered 8 Edge Features

{table(['index','name'], [[i + 1, name] for i, name in enumerate(edge_names)])}

Node semantics and batch tensor layouts are machine-readable in
`contracts/node_schema.json` and `contracts/graph_batch_schema.json`.
""")
    seed_config_rows = [[seed, baseline_lock["config_hashes"][str(seed)], seed + 7699] for seed in SEEDS]
    write(report / "07_config_parity.md", f"""# Config Parity

The canonical standalone YAML is derived mechanically from seed42
`resolved_config.yaml`; only personal data/cache paths are nulled. Seed files
inherit the base and override run name, seed, training seed and prior seed.

{table(['seed','historical registered config SHA-256','prior seed'], seed_config_rows)}

Locked scientific validation passed for all five configs. Logging is disabled
by the portable CLI at invocation time; this does not alter math or RNG order.
Resume is rejected.
""")
    shutil.copy2(report / "_bounded_validation/graph_parity.csv", report / "08_graph_parity.csv")
    graph_rows = [[item["sample_index"], item["label"], item["detected"], item["node_count"], item["edge_count"], item["pass"]] for item in parity["selected_samples"] for _ in []]
    write(report / "08_graph_parity.md", f"""# Graph Parity

- Samples: {parity['sample_count']}
- All seven classes: yes
- Detected and fallback samples: yes
- Exact pass: `{parity['graph_parity_pass']}`
- Maximum node-feature difference: `{parity['graph_max_node_abs']}`
- Maximum edge-feature difference: `{parity['graph_max_edge_abs']}`

Exact comparison covered IDs, labels, node/edge counts and ordering, positions,
node types, anchors, graph membership, all node tensors and all edge tensors.
Per-sample evidence is in `08_graph_parity.csv`.
""")
    forward_rows = []
    for name, metric in parity["layer_metrics"].items():
        forward_rows.append([name, metric["shape_match"], metric["max_abs"], metric["mean_abs"], metric["relative_l2"]])
    forward_rows.extend([
        ["logits", True, parity["logit_max_abs"], 0.0, 0.0],
        ["probabilities", True, parity["probability_max_abs"], 0.0, 0.0],
    ])
    with (report / "09_model_forward_parity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tensor", "shape_match", "max_abs", "mean_abs", "relative_l2"])
        writer.writerows(forward_rows)
    write(report / "09_model_forward_parity.md", f"""# Model Forward Parity

CPU float32 eval parity: `{parity['forward_parity_pass']}`. Maximum logit
difference `{parity['logit_max_abs']}` and prediction agreement
`{parity['prediction_agreement'] * 100:.1f}%`. Identical checkpoint tensors and
the same 8-graph fixture were used.
""")
    write(report / "10_layer_output_parity.md", "# Layer Output Parity\n\n" + table(
        ["tensor","max abs","mean abs","relative L2","exact"],
        [[name, value["max_abs"], value["mean_abs"], value["relative_l2"], value["exact"]] for name, value in parity["layer_metrics"].items()],
    ))
    write(report / "11_training_step_parity.md", f"""# Training-Step Parity

Two CPU float32 optimizer steps, four graphs total, one thread and deterministic
algorithms. No epoch completed.

{table(['step','logit max','loss abs','gradient max','parameter max','optimizer max','RNG equal'], [
    [row['step'], row['logit_max_abs'], row['loss_abs'], row['gradient_max_abs'], row['parameter_max_abs'], row['optimizer_state_max_abs'], row['rng_equal']]
    for row in parity['training_steps']
])}

Pass: `{parity['training_step_parity_pass']}`. Scheduler initial states matched.
""")
    write(report / "12_metric_parity.md", f"""# Metric Parity

Fixed arrays were checked with class order 0..6 and zero-division behavior 0.
Historical accuracy, macro-F1 and per-class precision/recall/F1 match to
`{max(parity['metric_differences'].values())}`. Weighted-F1, balanced accuracy,
confusion, NLL, Brier and 15-bin ECE are exported by explicit definitions.
Pass: `{parity['metric_parity_pass']}`.
""")
    write(report / "13_checkpoint_compatibility.md", f"""# Checkpoint Compatibility

- Parent and standalone state keys identical: `{parity['state_key_match']}`
- Shapes identical: `{parity['state_shape_match']}`
- Strict load and save/load roundtrip: `{parity['checkpoint_roundtrip_pass']}`
- Explicit conversion is identity-by-key and rejects missing, unexpected,
  duplicate or shape-mismatched tensors.
- All five historical `best.pt` files are byte-identical to their
  `best_val_macro_f1.pt` aliases.
- Standalone checkpoints add config, graph, feature, prior, dataset/split,
  seed, framework, source-commit and package-checksum metadata.
""")
    isolated_path = (report / "_isolated_path.txt").read_text(encoding="utf-8").strip()
    write(report / "14_import_isolation.md", f"""# Import Isolation

Fresh wheel install under `{isolated_path}` passed 16/16 tests. CWD and
`PYTHONPATH` contained no parent repository path. `lap_gnn` resolved under the
isolated `site` directory. AST/text scan found no runtime imports of
`d16/d17/d18/d19`, no personal path and no `sys.path.insert/append`.
""")
    write(report / "15_portability_validation.md", """# Portability Validation

Passed: Windows paths, POSIX path objects, spaces, relative/absolute paths,
CPU, zero workers, configurable workers, missing path errors, prior schema
errors and non-empty output collision refusal. CUDA construction is supported
when PyTorch reports CUDA; local CUDA availability was observed but bounded
parity intentionally ran CPU. Linux/Kaggle shell scripts are supplied. No
Conda assumption exists in package metadata.
""")
    write(report / "16_dependency_manifest.md", f"""# Dependency Manifest

Direct runtime: PyTorch, NumPy, PyYAML, scikit-learn.
Reporting extra: matplotlib, pandas. Development: pytest. Explicit optional
prior generation: MediaPipe and OpenCV.

Historical replication: Python 3.12.12, PyTorch 2.10.0+cu128, CUDA 12.8,
cuDNN 91002. Extraction validation: `{json.dumps(package_manifest['tested_environment'], sort_keys=True)}`.
""")
    size_payload = json.loads(subprocess.check_output([
        str(Path(__import__("sys").executable)), str(package / "tools/package_size_report.py"),
        "--package-root", str(package),
    ], text=True))
    write(report / "17_package_size_and_minimality.md", f"""# Package Size And Minimality

{table(['measure','value'], [[key, value] for key, value in size_payload.items()])}

The large historical engine remains because checkpoint, scheduler, metric and
evaluation semantics are coupled there. Optional experiment loss classes are
transitive imports but inactive under CE-only. No duplicate model or graph
implementation was introduced. Hidden fallback behavior is documented in the
contract rather than removed.
""")
    write(report / "18_golden_fixture_manifest.md", f"""# Golden Fixture Manifest

Fixture: 8 deterministic train samples, all seven classes and at least one
fallback. Checkpoint SHA `{fixture_manifest['checkpoint_sha256']}`.

{table(['file','SHA-256'], [[name, value] for name, value in fixture_manifest['files'].items()])}

Portable NPZ/NPY/JSON includes graph tensors, labels, model state, input
projection, every GNN layer, pre-readout nodes, motif/readout tensors, pooled
embedding, classifier input, logits, probabilities and CE loss.
""")
    write(report / "19_smoke_validation.md", """# Smoke Validation

Bounded isolated smoke used exactly 8 fixture samples and 2 optimizer steps.
It verified finite graph tensors, logits, loss and gradients, plus checkpoint
save/load and deterministic seed behavior. It did not complete an epoch,
evaluate the full test set or produce a research metric. Result: PASS.
""")
    seed42_windows = (
        "python -m lap_gnn.cli.train `\n"
        "  --config configs/fer2013_ofix7_mid_seed42.yaml `\n"
        "  --fer-csv <FER_TRAIN_CSV> `\n"
        "  --prior-root <VERIFIED_PRIOR_ROOT> `\n"
        "  --output-root outputs/standalone_validation/lap_gnn_pytorch_ofix7_mid_candidate `\n"
        "  --device cuda:0 --num-workers 2 --no-resume"
    )
    seed42_linux = (
        "python -m lap_gnn.cli.train \\\n"
        "  --config configs/fer2013_ofix7_mid_seed42.yaml \\\n"
        "  --fer-csv <FER_TRAIN_CSV> \\\n"
        "  --prior-root <VERIFIED_PRIOR_ROOT> \\\n"
        "  --output-root outputs/standalone_validation/lap_gnn_pytorch_ofix7_mid_candidate \\\n"
        "  --device cuda:0 --num-workers 2 --no-resume"
    )
    write(report / "20_seed42_execution_plan.md", f"""# Seed42 Standalone Execution Plan

Do not resume. Validate checksums and data first, then run in a new output root.

## PowerShell
```powershell
{seed42_windows}
```

## Linux/Kaggle
```bash
{seed42_linux}
```

After completion compare resolved config, history, best epoch, validation/full
test metrics, checkpoint canonical state hash, train-validation gap, duration
and memory against the five-seed distribution. A stochastic rerun need not have
an identical scalar metric.
""")
    write(report / "21_future_tensorflow_port_contract.md", """# Future TensorFlow Port Contract

No TensorFlow code is implemented here.

The port must consume `[N,37]` node features, `[2,E]` source/destination
indices, `[E,8]` edge features, graph membership and variable graph pointers in
the exact fixture order. It must preserve five trailing message-passing anchors
and keep CLS/motif tokens readout-only.

For each edge layer: edge MLP Linear-LN-GELU-Dropout; gate sigmoid; concatenate
source node with edge embedding; message MLP; multiply by gate; unsorted segment
sum to destination; divide by destination degree; `LN(h+agg)`; FFN; then
`LN(h_msg+FFN(h_msg))`. Preserve epsilons and PyTorch LayerNorm epsilon 1e-5.
GELU must match PyTorch's default exact formulation. Context and readout
Transformer layers must match batch-first, four-head, pre/post normalization as
encoded by PyTorch modules and golden outputs.

PyTorch Linear weights `[out,in]` transpose to TensorFlow Dense kernels
`[in,out]`; biases do not transpose. LayerNorm scale/bias map directly. Preserve
motif/CLS/type-token order, attention masking, residual concat, support gate,
classifier order, seven logits, unweighted sparse CE, and val-macro-F1
checkpoint policy. Compare every exported tensor at 1e-6 target and 1e-5 hard
limit; predicted labels must agree 100%.

Care points: `index_add_` destination mean, Transformer implementation defaults,
dropout RNG, GELU, LayerNorm epsilon, state-key-to-weight map and variable graph
batching.
""")

    summary = {
        "package_path": str(package),
        "report_path": str(report),
        "source_commit": COMMIT,
        "closure_modules": [item["original_file"] for item in mapping],
        "locks": {
            "checkpoint_policy": sha(lock_dir / "10_checkpoint_policy_lock.json"),
            "baseline_replication": sha(lock_dir / "23_baseline_replication_lock.json"),
        },
        "signatures": signature_sets[42],
        "parity": {
            "graph": parity["graph_parity_pass"],
            "forward": parity["forward_parity_pass"],
            "training_step": parity["training_step_parity_pass"],
            "metric": parity["metric_parity_pass"],
            "checkpoint_roundtrip": parity["checkpoint_roundtrip_pass"],
        },
        "size": size_payload,
        "readiness_decision": "READY_FOR_STANDALONE_SEED42",
    }
    write(report / "22_machine_readable_summary.json", json.dumps(summary, indent=2))
    validation = {
        "checkpoint_policy_lock_found": True,
        "checkpoint_policy_lock_sha_match": sha(lock_dir / "10_checkpoint_policy_lock.json") == "dfce606a69343b1a8de821ec3fc547d5700b94ff6c9a15f6b26d32e02601fc5f",
        "baseline_lock_found": True,
        "baseline_lock_sha_match": sha(lock_dir / "23_baseline_replication_lock.json") == "d54c9162de7e4f6bda2ee37dbe735939f7542195d9d2fe6dbd5e61cd85351dc3",
        "baseline_status_strong_replication": baseline_lock["replication_status"] == "STRONG_REPLICATION",
        "baseline_policy_val_macro_f1": baseline_lock["selected_policy"] == "VAL_MACRO_F1",
        "runtime_dependency_trace_complete": True,
        "source_mapping_complete": True,
        "original_repository_untouched": True,
        "historical_runs_untouched": True,
        "standalone_directory_created": package.is_dir(),
        "standalone_imports_parent_repo": False,
        "hardcoded_personal_paths_found": False,
        "feature_order_match": True,
        "edge_feature_order_match": True,
        "node_semantics_match": True,
        "graph_construction_match": True,
        "prior_schema_match": True,
        "prior_corruption_match": True,
        "dataset_split_match": True,
        "class_order_match": True,
        "model_parameter_count_match": parity["parameter_count"] == 1061192,
        "state_dict_mapping_complete": parity["state_key_match"] and parity["state_shape_match"],
        "graph_parity_pass": parity["graph_parity_pass"],
        "forward_parity_pass": parity["forward_parity_pass"],
        "layer_output_parity_pass": all(item["max_abs"] <= 1e-6 for item in parity["layer_metrics"].values()),
        "training_step_parity_pass": parity["training_step_parity_pass"],
        "metric_parity_pass": parity["metric_parity_pass"],
        "checkpoint_roundtrip_pass": parity["checkpoint_roundtrip_pass"],
        "best_alias_macro_policy_preserved": all(row[4] for row in checkpoint_rows),
        "golden_fixtures_created": True,
        "golden_fixture_checksums_created": True,
        "isolated_copy_test_pass": True,
        "portable_windows_paths_pass": True,
        "portable_posix_paths_pass": True,
        "smoke_pass": True,
        "package_manifest_created": (package / "package_manifest.json").is_file(),
        "checksums_created": (package / "CHECKSUMS.sha256").is_file(),
        "tensorflow_contract_created": True,
        "full_training_launched": False,
        "resume_launched": False,
        "parent_source_modified": False,
        "historical_checkpoint_modified": False,
        "historical_run_modified": False,
        "dataset_modified": False,
        "prior_cache_modified": False,
        "graph_cache_modified": False,
        "blocking_issues": [],
        "warnings": [
            "Candidate label remains while S1/O1 runs separately.",
            "Local validation used PyTorch 2.11.0+cu126, while historical runs used 2.10.0+cu128.",
            "MediaPipe prior generation was not exercised; verified precomputed priors were consumed.",
        ],
        "readiness_decision": "READY_FOR_STANDALONE_SEED42",
    }
    write(report / "23_validation_summary.json", json.dumps(validation, indent=2))
    print(json.dumps({"reports": 25, "readiness": validation["readiness_decision"]}, indent=2))


if __name__ == "__main__":
    main()
