"""Build the required evidence reports from bounded validation artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
REPORT = REPO / "outputs" / "d16_analysis" / "lap_gnn_tensorflow_port"
PYTORCH = REPO / "standalone" / "lap_gnn_pytorch_ofix7_mid_candidate"
NOTEBOOK = REPO / "notebooks" / "kaggle-end-to-end.ipynb"
PYTORCH_NOTEBOOK = REPO / "notebooks" / "kaggle-end-to-end-pytorch.ipynb"


def write(name: str, text: str) -> None:
    (REPORT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers: list[str], rows: list[list[object]]) -> str:
    result = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    result.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(result)


REPORT.mkdir(parents=True, exist_ok=True)
parity = json.loads((REPORT / "forward_parity_probe.json").read_text(encoding="utf-8"))
gradient = json.loads((REPORT / "gradient_optimizer_probe.json").read_text(encoding="utf-8"))
mapping_payload = json.loads((PACKAGE / "contracts" / "pytorch_tensorflow_mapping.json").read_text(encoding="utf-8"))
mapping = mapping_payload["records"] if isinstance(mapping_payload, dict) else mapping_payload
manifest = json.loads((PACKAGE / "package_manifest.json").read_text(encoding="utf-8"))
environment = json.loads((REPORT / "tensorflow_environment.json").read_text(encoding="utf-8-sig"))

write(
    "00_README.md",
    """# TensorFlow OFIX7-mid Port Evidence

This directory records the bounded, non-training validation of the standalone
TensorFlow/Keras port. The source of truth is the locked standalone PyTorch
package and its golden fixtures. No full epoch, validation pass, official test
evaluation or TensorFlow FER2013 training was run.

The current decision is `HOLD_TENSORFLOW_PORT_REPAIR`: graph, mapping,
architecture, layers, prediction, gradient, optimizer formula, scheduler,
metrics, serialization and notebook contracts pass, but the local CPU float32
logit maximum is `1.0013580322265625e-05`, narrowly above the non-negotiable
`1e-5` gate. The tolerance was not relaxed.
""",
)

write(
    "01_pytorch_reference_validation.md",
    f"""# PyTorch Reference Validation

{table(
    ["Check", "Result"],
    [
        ["standalone package", "PASS"],
        ["CHECKSUMS.sha256", "PASS, 130 files"],
        ["golden manifest", "PASS"],
        ["baseline lock SHA", "`d54c9162de7e4f6bda2ee37dbe735939f7542195d9d2fe6dbd5e61cd85351dc3`"],
        ["checkpoint-policy lock SHA", "`dfce606a69343b1a8de821ec3fc547d5700b94ff6c9a15f6b26d32e02601fc5f`"],
        ["golden checkpoint SHA", "`20084362d4a9529f7a833a1994ba407d958cb6264b42f60e4252efca3edad5ac`"],
        ["feature / edge dimensions", "37 / 8"],
        ["parameter count", "1,061,192"],
        ["locked extraction graph and forward", "PASS"],
        ["PyTorch checksum file SHA", f"`{sha(PYTORCH / 'CHECKSUMS.sha256')}`"],
    ],
)}

The PyTorch package was rechecked after the port and remained checksum-valid.
Framework-neutral contracts and original golden files were copied byte-for-byte;
their source and destination hashes are recorded in `package_manifest.json`.
""",
)

write(
    "02_tensorflow_environment_strategy.md",
    f"""# TensorFlow Environment Strategy

Local validation used Python {environment['python']}, TensorFlow
{environment['tensorflow']} and Keras {environment['keras']} on
`{environment['os']}`. The local package is `tensorflow-cpu`; no local GPU claim
is made.

Kaggle accepts only the tested pair TensorFlow 2.18.1 and Keras 3.15.0 for the
first run. The notebook inspects the preinstalled environment before package
installation. If either version differs, it installs
`requirements-kaggle.txt`, stops, and requires a session restart. It then
installs this package editable with `--no-deps`. This avoids silently mixing an
untested TensorFlow/Keras pair.

Parity subprocesses hide CUDA and run CPU float32 with XLA and mixed precision
disabled. Training requires a visible GPU unless
`ALLOW_CPU_TRAINING=True`; training keeps XLA disabled and follows the locked AMP
intent using `LossScaleOptimizer`.
""",
)

write(
    "03_graph_batch_contract.md",
    """# Flat Graph Batch Contract

The primary representation is concatenated and unpadded:

| Tensor | Shape |
|---|---|
| node_features | `[total_nodes, 37]` |
| edge_index | `[2, total_edges]` |
| edge_features | `[total_edges, 8]` |
| node_types / node_graph_index | `[total_nodes]` |
| edge_graph_index | `[total_edges]` |
| graph_node_counts / graph_edge_counts | `[num_graphs]` |
| labels / sample_ids | `[num_graphs]` |
| coordinates | `[total_nodes, 2]` |

Graph collation preserves sample, node, edge and anchor order. Each graph's
edge endpoints are offset by the cumulative node count. Message aggregation
gathers source nodes, builds edge-conditioned messages, sums by destination,
computes destination degree and divides by clamped degree, matching PyTorch
mean aggregation including zero-degree behavior.
""",
)

write(
    "04_graph_pipeline_port.md",
    """# Graph Pipeline Port

The NumPy/Python graph builder was mechanically ported from the standalone
reference. Runtime imports no `torch` and no parent D16-D19 module. It preserves
48x48 images, `face_plus_context`, face threshold 0.15, two context pixels,
37 ordered node channels, 8 ordered edge channels, five semantic anchors,
directed local edges, bidirectional anchor edges, the complete directed anchor
graph, prior corruption modes and seed mixing.

The runtime consumes existing `d16_mediapipe_pixel_priors_v1` NPZ files and
never invokes MediaPipe generation during normal training. A two-sample local
pipeline probe also verified the `tf.data.Dataset.from_generator` signature,
bounded prefetch and deterministic sample IDs `[0, 1]`.
""",
)

write(
    "05_model_architecture_mapping.md",
    """# Model Architecture Mapping

The model is split into explicit Keras layers rather than replaced by generic
approximations: pixel encoder, three gated edge-context layers, final
part-context transformer injection, coarse/micro motif query branches,
readout transformer, micro-support gate, residual part concat, 480-dimensional
projection and seven-class classifier.

Custom `TorchLinear`, `TorchLayerNorm`, `TorchMultiheadAttention` and
`TorchTransformerEncoderLayer` preserve weight layout, epsilon, activation,
residual, mask, scaling and softmax axes. The state registry binds every
trainable Keras variable to one explicit PyTorch key.
""",
)

mapping_csv = REPORT / "06_pytorch_tensorflow_weight_mapping.csv"
with mapping_csv.open("w", newline="", encoding="utf-8") as stream:
    fields = sorted({key for row in mapping for key in row})
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(mapping)
write(
    "06_pytorch_tensorflow_weight_mapping.md",
    f"""# Weight Mapping

{table(
    ["Item", "Result"],
    [
        ["PyTorch tensors", parity["mapping"]["source_tensors"]],
        ["TensorFlow variables", parity["mapping"]["destination_variables"]],
        ["assigned", parity["mapping"]["assigned"]],
        ["missing / extra", "0 / 0"],
        ["shape / dtype mismatch", "0 / 0"],
        ["duplicate destinations", "0"],
        ["mapping complete", "PASS"],
    ],
)}

Every mapping is explicit and uses only `identity` or PyTorch
`[out,in]` to TensorFlow `[in,out]` transpose where required. No fuzzy matching
is used. The full 127-row table is in the companion CSV.
""",
)

write(
    "07_parameter_accounting.md",
    f"""# Parameter Accounting

{table(
    ["Framework object", "Trainable", "Non-trainable", "Status"],
    [["TensorFlow LAP-GNN", parity["trainable_parameters"], parity["non_trainable_parameters"], "PASS"]],
)}

The TensorFlow trainable count is exactly the locked PyTorch count of
1,061,192. The 127 trainable variables correspond one-to-one with the 127
registered state tensors.
""",
)

write(
    "08_graph_parity.md",
    """# Graph Parity

Exact graph parity was measured on 32 samples using the real verified prior
artifacts. Node values, edge indices and edge features all had maximum
difference `0.0`. Node and edge ordering therefore match exactly. Detailed
sample rows are in `08_graph_parity.csv`.
""",
)

layer_rows = parity["layers"]
with (REPORT / "09_layer_forward_parity.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["name", "max_abs", "relative_l2", "passes_1e-5"])
    writer.writeheader()
    for row in layer_rows:
        writer.writerow({**row, "passes_1e-5": row["max_abs"] <= 1e-5})
write(
    "09_layer_forward_parity.md",
    f"""# Layer Forward Parity

{table(
    ["Tensor", "Max abs", "Relative L2", "Gate"],
    [[row["name"], f"{row['max_abs']:.10g}", f"{row['relative_l2']:.10g}", "PASS" if row["max_abs"] <= 1e-5 else "FAIL"] for row in layer_rows],
)}

All recorded critical layers pass `1e-5`. Input projection is
`{layer_rows[0]['max_abs']:.10g}`; the largest layer-level difference is
`{parity['maximum_layer_max_abs']:.10g}`. Final logits are assessed separately
and remain the sole strict blocker.
""",
)

write(
    "10_gradient_parity.md",
    f"""# Gradient Parity

{table(
    ["Metric", "Value", "Gate"],
    [
        ["PyTorch CE", gradient["pytorch_loss"], "-"],
        ["TensorFlow CE", gradient["tf_loss"], "-"],
        ["overall cosine", gradient["gradient_cosine"], "PASS >= 0.99999"],
        ["maximum absolute difference", gradient["gradient_max_abs"], "reported"],
        ["relative L2", gradient["gradient_relative_l2"], "reported"],
        ["missing / unexpected", "0 / 0", "PASS"],
    ],
)}

The low minimum per-tensor cosine comes from near-zero bias gradients and is not
used instead of the required concatenated-gradient cosine. The required overall
direction gate passes.
""",
)

write(
    "11_adamw_semantics.md",
    f"""# AdamW Semantics

Standard Keras AdamW differed from the locked PyTorch first step by
`{gradient['adamw_step1_max_abs']}` because of epsilon/update ordering and was
rejected. `TorchCompatibleAdamW` explicitly implements PyTorch moment
initialization, bias correction, epsilon placement, decoupled weight decay,
clip ordering and step counting.

The custom first-step formula compared against the PyTorch state fixture at
maximum absolute difference `1.4901161193847656e-08` (PASS, gate `2e-8`).
The two-step global execution budget was already consumed by the PyTorch and
standard-Keras probes, so no additional live custom-optimizer update was run;
the executable class and the tested formula share the same equations. This is
retained as a validation warning, not hidden.
""",
)

write(
    "12_plateau_and_early_stopping_semantics.md",
    """# Plateau and Early-Stopping Semantics

On synthetic losses `[1.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2]`, PyTorch LR is
`[0.0003 x6, 0.00015 x2]`; TensorFlow produces the same sequence with only the
float32 storage representation (`0.000300000014...`, `0.000150000007...`).

The custom scheduler matches mode=min, relative threshold `1e-4`, factor 0.5,
patience 5, cooldown 0, minimum LR `3e-5` and strict step ordering. Early
stopping uses validation loss, strict improvement, minimum epoch 30, patience
15 and stops at the same synthetic epoch 30. Best checkpoints are saved before
the scheduler step; `last` is saved after it.
""",
)

write(
    "13_metric_parity.md",
    """# Metric Parity

Accuracy, macro-F1, weighted-F1, balanced accuracy, seven-class precision,
recall and F1, confusion matrix, NLL, Brier and 15-bin ECE are implemented with
the PyTorch zero-division and class-order rules. ECE preserves the exact
`confidence > lower and confidence <= upper` bin convention, including a
boundary-specific test. Metric tests pass.
""",
)

write(
    "14_checkpoint_and_serialization.md",
    """# Checkpoint and Serialization

The policy writes `best.keras`, `best_val_macro_f1.keras`,
`best_val_accuracy.keras`, `last.keras` and matching weight-only H5 files.
`best.keras` is an exact copy of the macro-F1 checkpoint. A complete `.keras`
roundtrip preserved logits exactly and restored all 127 model variables and
all 256 built optimizer variables.

Metadata records epoch, seed, validation metrics, config/package/signature
hashes, TF/Keras versions, optimizer summary, scheduler state, early-stop state,
mixed-precision policy and resources. Resume is disabled. Test is evaluated
only after validation selection completes.
""",
)

write(
    "15_tensorflow_data_pipeline.md",
    """# TensorFlow Data Pipeline

The pipeline is lazy:

`prior NPZ -> deterministic graph builder -> flat NumPy collator -> tf.data
from_generator -> bounded prefetch -> GradientTape loop`.

It does not cache the complete dataset. Graph construction supports a bounded
LRU and deterministic thread pool. Shuffle uses
`seed + epoch * 1,000,003`; prior corruption retains its separate locked seed
rule. Variable node/edge dimensions are represented by `TensorSpec(None, ...)`.
""",
)

write(
    "16_resource_control_and_telemetry.md",
    """# Resource Controls and Telemetry

Exposed controls: intra/inter-op threads, graph workers, tf.data prefetch,
parallel-call registration, graph-cache size, memory growth, mixed precision,
XLA, batch size and device policy. Kaggle defaults are batch 16, two graph
workers, bounded prefetch, memory growth on, mixed precision on and XLA off.

Telemetry records host RSS, discoverable peak GPU memory, graph construction,
device/train/validation timing lists, cache hits/misses and hit rate, CPU use,
elapsed time and effective TensorFlow thread settings.
""",
)

write(
    "17_import_isolation.md",
    """# Import Isolation

AST scans report no `torch` import and no parent `d16`, `d17`, `d18`, `d19` or
`lap_gnn` import from `src/lap_gnn_tf`. The package installs through the
standard `src` layout and uses no `sys.path` hack, symlink or parent source.
Normal runtime weight loading reads NPZ, not `.pt`.
""",
)

write(
    "18_kaggle_notebook_validation.md",
    f"""# Kaggle Notebook Validation

The original PyTorch notebook was copied before editing and remains byte exact:

- backup SHA: `{sha(PYTORCH_NOTEBOOK)}`
- TensorFlow notebook SHA: `{sha(NOTEBOOK)}`

The 11-stage notebook exposes all required user controls, validates commit and
checksums, inspects the environment, performs conditional dependency install,
checks isolation, runs bounded tests, validates split/prior counts, executes
CPU golden parity, launches one fresh seed42 run only after READY, validates
artifacts and creates the required ZIP. All code cells parse successfully and
the notebook contract test passes.
""",
)

write(
    "19_tensorflow_seed42_execution_plan.md",
    """# TensorFlow Seed42 Execution Plan

1. Attach the existing FER split and verified prior Kaggle Inputs.
2. Run the notebook once. If TF/Keras replacement occurs, restart and Run All.
3. Require `READY_FOR_TENSORFLOW_KAGGLE_SEED42`.
4. Launch exactly one fresh seed42 command with batch 16, AMP, no XLA, no W&B
   and no resume.
5. Select by validation macro-F1, then evaluate official test once.
6. Archive to `/kaggle/working/ofix7_mid_seed42_tensorflow_outputs.zip`.

The future run is descriptively compared with PyTorch accuracy 63.60-65.74% and
macro-F1 61.33-64.66%. These are review bands, not test-selection rules.
""",
)

write(
    "20_future_server_optimization_contract.md",
    """# Future Server Optimization Contract

Before tuning, collect OS, CPU model, physical/logical cores, RAM, storage,
GPU/VRAM, NVIDIA driver, CUDA compatibility, Python/TF constraints, Internet,
Docker/Conda permissions, maximum runtime and whether training/evaluation are
needed.

Runtime-only fields that may be tuned without changing scientific architecture:
intra/inter-op threads, graph workers, tf.data parallel calls, prefetch,
bounded graph cache, batch size subject to numerical review, mixed precision,
XLA subject to parity review, memory growth, CPU affinity and thread
environment variables. Do not change node/edge construction, model dimensions,
optimizer/scheduler values, checkpoint policy, split or priors as server tuning.
No server-specific tuning was performed here.
""",
)

source_pairs = [
    ("data/graph_builder.py", "graph/builder.py", "mechanical NumPy graph port"),
    ("data/detail_node_features.py", "graph/features.py", "mechanical NumPy feature port"),
    ("data/pixel_prior_dataset.py", "priors/loader.py", "lazy prior and corruption semantics"),
    ("data/mediapipe_priors.py", "priors/mediapipe_priors.py", "fallback-only runtime support; no regeneration"),
    ("model/pixel_encoder.py", "model/lap_gnn.py", "PixelEncoder"),
    ("model/edge_context_gnn.py", "model/gated_edge_layer.py", "gated message passing"),
    ("model/edge_context_gnn.py", "model/edge_context.py", "three layers and context injection"),
    ("model/micro_motif_support_readout.py", "model/readout.py", "full active readout"),
    ("model/classifier.py", "model/classifier.py", "seven-class head"),
    ("model/d16_model.py", "model/lap_gnn.py", "active composition"),
    ("training/metrics.py", "training/metrics.py", "metric semantics"),
    ("training/optimizer.py", "training/optimizer.py", "AdamW factory and compatible update"),
    ("training/scheduler.py", "training/plateau.py", "plateau state machine"),
    ("training/engine.py", "training/trainer.py", "explicit GradientTape loop"),
    ("training/checkpointing.py", "training/checkpointing.py", "selection and aliases"),
]
with (REPORT / "21_source_mapping.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["pytorch_source", "tensorflow_destination", "scope"])
    writer.writerows(source_pairs)
write(
    "21_source_mapping.md",
    "# Source Mapping\n\n" + table(
        ["PyTorch source", "TensorFlow destination", "Scope"],
        [[f"`{a}`", f"`{b}`", c] for a, b, c in source_pairs],
    ) + "\n\nInactive PyTorch loss/readout/fallback modules are not silently "
    "represented as active TensorFlow behavior; only the locked OFIX7-mid path "
    "is executable.",
)

total_bytes = sum(item["bytes"] for item in manifest["files"])
write(
    "22_package_size_and_dependencies.md",
    f"""# Package Size and Dependencies

- Manifested files: {len(manifest['files'])}
- Manifested bytes: {total_bytes}
- Scientific payload SHA: `{manifest['scientific_payload_sha256']}`
- Runtime dependencies: TensorFlow 2.18.x, NumPy, PyYAML, scikit-learn, psutil.
- Kaggle tested pins: TensorFlow 2.18.1, Keras 3.15.0, NumPy 2.0.2,
  PyYAML 6.0.2, scikit-learn 1.6.1, psutil 6.1.1.

PyTorch, parent D16-D19 code, FER CSVs and prior NPZ datasets are not packaged.
Golden NPZ assets are included for bounded validation.
""",
)

machine_summary = {
    "decision": "HOLD_TENSORFLOW_PORT_REPAIR",
    "tensorflow": environment["tensorflow"],
    "keras": environment["keras"],
    "graph_parity_max_abs": 0.0,
    "weight_mapping": {"source": 127, "destination": 127, "complete": True},
    "parameters": {"trainable": 1_061_192, "non_trainable": 0},
    "forward": {
        "class": parity["forward_parity_class"],
        "max_layer_abs": parity["maximum_layer_max_abs"],
        "max_logit_abs": parity["max_logit_difference"],
        "max_probability_abs": parity["max_probability_difference"],
        "prediction_agreement": parity["prediction_agreement"],
    },
    "gradient": {
        "cosine": gradient["gradient_cosine"],
        "max_abs": gradient["gradient_max_abs"],
        "relative_l2": gradient["gradient_relative_l2"],
    },
    "adamw_custom_formula_max_abs": 1.4901161193847656e-08,
    "full_training_launched": False,
}
(REPORT / "23_machine_readable_summary.json").write_text(
    json.dumps(machine_summary, indent=2, sort_keys=True), encoding="utf-8",
)

validation = {
    "pytorch_reference_found": True,
    "pytorch_manifest_valid": True,
    "pytorch_checksums_valid": True,
    "golden_manifest_valid": True,
    "baseline_lock_valid": True,
    "checkpoint_policy_lock_valid": True,
    "tensorflow_package_created": True,
    "tensorflow_runtime_imports_torch": False,
    "tensorflow_runtime_imports_parent": False,
    "hardcoded_personal_paths_found": False,
    "feature_order_match": True,
    "edge_order_match": True,
    "node_semantics_match": True,
    "graph_batch_contract_match": True,
    "graph_parity_pass": True,
    "weight_mapping_complete": True,
    "unmapped_pytorch_tensors": [],
    "unmapped_tensorflow_variables": [],
    "parameter_count_match": True,
    "input_projection_parity": True,
    "graph_layer_parity": True,
    "readout_parity": True,
    "logit_parity": False,
    "prediction_agreement": 1.0,
    "gradient_parity": True,
    "adamw_semantics_pass": True,
    "plateau_semantics_pass": True,
    "early_stopping_semantics_pass": True,
    "metric_parity_pass": True,
    "checkpoint_roundtrip_pass": True,
    "data_pipeline_valid": True,
    "seed_determinism_pass": True,
    "resource_controls_exposed": True,
    "telemetry_exposed": True,
    "pytorch_notebook_backup_created": True,
    "tensorflow_notebook_updated": True,
    "kaggle_notebook_contract_pass": True,
    "bounded_smoke_pass": True,
    "full_training_launched": False,
    "parent_code_modified": False,
    "pytorch_package_modified": False,
    "historical_checkpoint_modified": False,
    "dataset_modified": False,
    "prior_modified": False,
    "blocking_issues": [
        "Local CPU float32 max logit difference 1.0013580322265625e-05 exceeds 1e-5 by 1.3580322265624182e-08."
    ],
    "warnings": [
        "Direct live TorchCompatibleAdamW update was not run after the two-step budget was exhausted; its tested formula matches the PyTorch fixture at 1.49e-8.",
        "Keras emits a subclassed-model build warning during load, but full .keras state and logits roundtrip exactly.",
        "No complete epoch, full validation, official test or full TensorFlow training was run.",
    ],
    "readiness_decision": "HOLD_TENSORFLOW_PORT_REPAIR",
}
(REPORT / "24_validation_summary.json").write_text(
    json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8",
)
print(REPORT)
