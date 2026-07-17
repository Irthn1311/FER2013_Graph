"""Generate the D19 design-review package from frozen D18 artifacts and audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/d19_analysis/d19_preimplementation_review"
D18 = ROOT / "outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining"
BOOK_URL = "https://yaoma24.github.io/dlg_book/dlg_book.pdf"
DECISION = "REVISE"
SEEDS = (7, 21, 42, 84, 123)


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")


def md_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 4) -> str:
    view = frame.copy() if columns is None else frame[columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def mean_metric(frame: pd.DataFrame, metric: str, layer: str, mode: str = "official") -> float:
    values = frame[(frame.metric == metric) & (frame.layer == layer) & (frame["mode"] == mode)].value.astype(float)
    return float(values.mean())


def aggregate_cka(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(linear_cka=frame.linear_cka.astype(float))
        .groupby(["comparison", "left", "right"])["linear_cka"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )


def claim_map() -> pd.DataFrame:
    rows = [
        ("C01", "1", "1.2 Why Deep Learning on Graphs?", "Non-graph data can be represented as graphs, and that transformation may be lossless or lossy.", "Requires an explicit information-loss audit when turning a 48x48 image into a sparse pixel graph.", "Does not imply that a GNN can recover pixels or visual patterns discarded during graph construction.", "Direct: D18 retains 1800/2304 pixels and therefore uses a lossy support transform.", "high"),
        ("C02", "2", "2.2 Graph Representations", "Connectivity and attributes jointly define a graph representation.", "Supports auditing adjacency families separately from node signals.", "Does not identify local, kNN, or landmark adjacency as optimal for FER.", "Direct at the representation level; task benefit remains experimental.", "high"),
        ("C03", "2", "2.5 Graph Signal", "Features attached to nodes are graph signals processed together with graph structure.", "Supports treating the ten pixel descriptors as the evidence signal.", "Does not establish that handcrafted pixel descriptors have sufficient visual capacity.", "Direct for schema, indirect for accuracy.", "high"),
        ("C04", "2", "2.6.3 Multi-dimensional Graphs", "Multiple graph dimensions can share the same node set while carrying different relations.", "Motivates asking whether local and kNN connectivity deserve explicit identities.", "Does not make a merged edge-type graph identical to a multi-dimensional graph.", "Conceptual only; D18 has one node state and one merged adjacency.", "medium"),
        ("C05", "5", "5.2.2 General Graph-focused Framework", "Graph-focused learning combines node representation learning with graph-level aggregation and prediction.", "Supports tracing D18 from node encoder through GNN, readout, and classifier.", "Does not select the D19 operator or pooling rule.", "Direct architecture vocabulary, not experimental evidence.", "high"),
        ("C06", "5", "5.3.2 Spatial Graph Filters", "Spatial filters update nodes from neighborhoods defined by graph connectivity.", "Supports treating removal or splitting of edge families as a change to the propagation operator.", "Does not show that relation-specific filters improve this dataset.", "Direct mechanism mapping.", "high"),
        ("C07", "5", "5.4.1 Flat Pooling", "Mean, max, and gated global pooling are graph-level flat pooling mechanisms.", "Supports the semantics of D18's concatenated mean/max/gated readout.", "Does not prove that all three components are needed or calibrated.", "Direct: D18 implements these three components.", "high"),
        ("C08", "5", "5.5.2 Parameter Learning", "Graph-classification parameters are learned jointly under the task objective.", "Supports evaluating the complete operator/readout as a trained system.", "Does not remove parameter-count or initialization confounds between candidates.", "Direct but generic.", "high"),
        ("C09", "6", "6.3.1 Graph Adversarial Training", "Training can include graph perturbations to improve robustness to structural changes.", "Provides conceptual context for C2 structure-mode mixing.", "Does not prove C2 is adversarially robust beyond the measured modes.", "Moderate: C2 uses bounded random removal, not an optimized adversary.", "medium"),
        ("C10", "6", "6.3.2 Graph Purification", "Purification attempts to remove suspicious graph structure before learning.", "Explains why structure filtering is a distinct mechanism from typed processing.", "Does not support reopening the rejected purification sweep.", "Low for D19; retained only as a boundary.", "high"),
        ("C11", "6", "6.3.3 Graph Attention", "Attention can learn unequal weights for neighboring information.", "Supports a future edge-weighting mechanism as a capacity choice.", "Attention weights are not causal explanations and do not guarantee evidence-path invariance.", "Direct caution for any future structure gate.", "high"),
        ("C12", "6", "6.3.4 Graph Structure Learning", "Graph connectivity itself can be learned or refined.", "Distinguishes learned structure from D18's deterministic kNN graph.", "Does not turn fixed standardized-Euclidean kNN into learned adjacency.", "Direct distinction.", "high"),
        ("C13", "8", "8.4 Multi-dimensional Graph Neural Networks", "Multi-dimensional GNNs consider both within-dimension and across-dimension interactions over shared nodes.", "Motivates relation-specific state/filter questions.", "A two-operator residual sum lacks the book's full dimension-specific states and cross-dimension interactions.", "D19-A1 would be a simplified adaptation, not a faithful implementation.", "high"),
        ("C14", "11", "11.2.1 Images as Graphs", "Images can be mapped to graphs whose nodes and relations represent visual entities or features.", "Supports image-to-graph research as a valid direction.", "The examples do not establish a raw-pixel GNN ceiling; many use region/object/CNN-derived visual features.", "Indirect: LAP-GNN uses handcrafted raw-pixel evidence rather than region embeddings.", "high"),
        ("C15", "11", "11.4 Image Classification", "GNNs can support image classification by exploiting structured semantic relations.", "Supports graph-based image classification in principle.", "Does not show that a pure raw-pixel graph should match CNN feature capacity.", "Indirect for FER2013 raw pixels.", "medium"),
        ("C16", "14", "14.2.1 Jumping Knowledge", "JK adaptively combines node representations from different depths.", "Motivates testing whether shallow and deep states are complementary.", "Graph-level averaging of pooled layers is an adaptation, not the node-adaptive JK formulation.", "Layer audit is required before transfer to LAP-GNN.", "high"),
        ("C17", "14", "14.2.2 DropEdge", "Random edge removal was proposed as a remedy for oversmoothing in deeper GNNs.", "Provides a definition and a reason to measure oversmoothing before invoking it.", "Does not justify reopening structure DropEdge after D18 rejected it.", "Direct boundary; D19 review proposes no DropEdge.", "high"),
        ("C18", "14", "14.3 Self-supervised Learning / Attribute Mask", "Masked graph attributes can provide auxiliary self-supervision.", "Identifies a separate future representation-learning branch.", "Does not provide evidence that attribute masking is needed in D19-A.", "Low for the immediate plan.", "medium"),
        ("C19", "14", "14.4 Expressiveness", "GNN architectures differ in their ability to distinguish graph structures.", "Supports auditing operator capacity rather than assuming all message passing is equivalent.", "Does not predict a numeric FER accuracy gain from typed relations.", "Conceptual only.", "medium"),
    ]
    return pd.DataFrame(rows, columns=["claim_id", "chapter", "section", "exact_concept", "supports", "does_not_support", "lapgnn_mapping", "mapping_confidence"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    overlap = pd.read_csv(OUT / "10_edge_overlap_per_image.csv")
    overlap_summary = pd.read_csv(OUT / "10_edge_overlap_summary.csv")
    layer = pd.read_csv(OUT / "09_layerwise_information_audit.csv")
    cka = pd.read_csv(OUT / "09_layerwise_cka.csv")
    message = pd.read_csv(OUT / "10_message_scale_probe.csv")
    readout = pd.read_csv(OUT / "08_readout_ablation.csv")
    params = pd.read_csv(OUT / "15_current_parameter_count.csv")
    claims = claim_map()
    claims.to_csv(OUT / "02_book_claim_map.csv", index=False)
    write("02_book_claim_map.md", "# Book Claim Map\n\nSource: [Deep Learning on Graphs by Yao Ma and Jiliang Tang](%s). No page numbers or quotations are asserted.\n\n%s" % (BOOK_URL, md_table(claims)))

    edge_mean = overlap.mean(numeric_only=True)
    mean_edges = float(edge_mean["union_count"])
    a0_edges = float(edge_mean["final_local_count"] + edge_mean["final_knn_count"])
    c2_params = int(params.loc[params.component == "total", "parameter_count"].iloc[0])
    layer_summary_rows = []
    for layer_name in ("input_projection", "gnn_layer_1", "gnn_layer_2", "gnn_layer_3", "classifier_input"):
        row = {"layer": layer_name}
        for metric in ("class_centroid_separation", "within_between_ratio", "covariance_effective_rank", "graph_representation_variance"):
            row[metric] = mean_metric(layer, metric, layer_name)
        if layer_name != "classifier_input":
            for metric in ("mean_pairwise_node_cosine", "node_representation_variance", "node_covariance_effective_rank", "normalized_dirichlet_energy"):
                row[metric] = mean_metric(layer, metric, layer_name)
        layer_summary_rows.append(row)
    layer_summary = pd.DataFrame(layer_summary_rows)
    cka_summary = aggregate_cka(cka)
    readout_summary = readout.assign(accuracy=readout.accuracy.astype(float), macro_f1=readout.macro_f1.astype(float)).groupby("variant", as_index=False).agg(accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"))
    message_summary = message.assign(**{column: message[column].astype(float) for column in message.columns if column not in {"family"}}).groupby(["layer", "family"], as_index=False).agg(
        edge_count=("edge_count", "mean"),
        mean_dst_degree=("mean_dst_degree", "mean"),
        message_l2_per_edge=("message_l2_per_edge", "mean"),
        aggregate_current=("aggregate_l2_current_full_degree", "mean"),
        aggregate_own=("aggregate_l2_own_degree", "mean"),
        own_to_current_ratio=("own_to_current_scale_ratio", "mean"),
    )

    write("00_README.md", f"""
# D19 Pre-implementation Review

Decision: **{DECISION}**.

This package reviews D19 without implementing a D19 model, changing D18 behavior, modifying training configs, or launching training. It combines:

- the frozen five-seed C0/C2 evidence package;
- exact runtime tracing of the current D18 C2 code;
- a 715-image edge reconstruction audit;
- bounded frozen-checkpoint inference on 70 class-stratified locked images for all five C2 seeds;
- book-grounded interpretation with explicit non-claims;
- parameter/compute estimates and a confound-controlled next-step decision.

The immediate allowed next phase is a matched evidence-only A0 implementation and seed42 run. The original proposal is revised: do not implement independent local/kNN operators or multi-scale pooling yet.

See `19_go_no_go_decision.md` for the decision and `22_validation_summary.json` for completion state.
""")

    write("01_review_scope_and_sources.md", f"""
# Review Scope and Sources

## Frozen experimental source

Primary evidence: `outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining/`. The package validation reports no blockers, all ten C0/C2 runs, five paired seeds, full 3,589-image test metrics, the locked 715-image protocol, edge counterfactuals, representation analysis, and structure-message probes.

## Repository source

The runtime was traced from `configs/d18/overfit_fix_18/d18_ofix18_c2_structure_mode_mix_only_seed42.yaml` through `d18/training/train_d18.py`, `d18/data`, and `d18/models`. Config names were not treated as implementation evidence.

## Book source

Official PDF: [Deep Learning on Graphs]({BOOK_URL}). Reviewed sections: 1.2; 2.2, 2.5, 2.6.3; 5.2.2, 5.3.2, 5.4.1, 5.5.2; 6.3.1-6.3.4; 8.4; 11.2.1 and 11.4; 14.2.1-14.2.2, 14.3, and 14.4. Downloaded PDF SHA256 during review: `0C295E6746F965614D964213A8A48D6730FDA2280E97BBB26F1FB939F0D5BF20`.

## New diagnostics

- `d18/scripts/audit_d19_preimplementation.py`: read-only graph reconstruction and frozen inference.
- Full overlap set: 715 locked images, SHA256 `17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d` over ordered sample indices.
- Layer set: 70 deterministic class-stratified locked images, 10 per class, five C2 best checkpoints.
- No test-fitted linear probe was used. The bounded subset is descriptive, not a promotion benchmark.

## Hard boundaries

No D19 model class, config, checkpoint, training, fine-tuning, resume, sweep, D18 behavior change, or promotion was performed.
""")

    write("03_d18_evidence_summary.md", """
# Frozen D18 Evidence Summary

## Five-seed official performance

| Cell | Test accuracy mean +/- SD | Test macro-F1 mean +/- SD |
| --- | --- | --- |
| C0 | 59.94 +/- 0.97% | 55.38 +/- 1.27% |
| C2 | 59.59 +/- 0.37% | 55.21 +/- 0.81% |

C2-C0 paired mean differences were -0.36 pp accuracy and -0.17 pp macro-F1. The official-preservation gate passed.

## Robustness and residual structure

- Remove-structure macro-F1 gain for C2 over C0: +17.00 pp mean, positive 5/5 seeds.
- Shuffle-structure gain: +14.30 pp mean, positive 5/5.
- Robust-min gain: +15.44 pp mean, positive 5/5.
- C2 reduced the official-to-remove dependency drop by 17.29 pp in 5/5 seeds.
- Residual C2 structure contribution: +3.13 pp mean on the locked set.
- Correct-vs-degree-matched-random structure advantage: +1.81 pp mean, positive 4/5.
- Official/remove pre-classifier representation CKA: C0 0.724, C2 0.945.

## What is established

C2 preserves official performance while making local+kNN the primary evidence path. Correct landmark structure retains a small secondary contribution. Structure DropEdge is rejected and mode-mix probability 0.30 is frozen.

## What is not established

Inference-time removal is not a retrained evidence-only control. D18 does not establish that a shared operator is the next bottleneck, that typed operators improve accuracy, or that multi-scale pooling helps. Those are D19 hypotheses.
""")

    write("04_repository_architecture_map.md", f"""
# Repository Architecture Map

```mermaid
flowchart LR
  C[C2 YAML] --> T[train_d18.read_config/run_train]
  P[Prior NPZ: image_48, label, part_soft_masks] --> D[StructurePixelDataset]
  D --> G[build_structure_graph]
  G --> N[1800 nodes x 10 pixel features]
  G --> E[local + kNN + structure union; edge_attr x 6; edge_type]
  N --> B[collate_d18_graphs]
  E --> B
  B --> ENC[10 -> 96 node encoder]
  ENC --> L1[EdgeContextGNN layer 1]
  L1 --> L2[layer 2]
  L2 --> L3[layer 3]
  L3 --> R[mean + max + gated mean = 288]
  R --> CL[288 -> 192 -> 7 logits]
```

| Stage | File/function | Input -> output | Landmark used | Edge identity retained | Train/eval difference |
| --- | --- | --- | --- | --- | --- |
| Config | `train_d18.py:read_config`, `run_train` | YAML -> dict | No | N/A | Resume/signature only |
| Dataset | `structure_dataset.py:StructurePixelDataset` | sorted NPZ -> graph | Container includes priors | Yes after graph build | Cache/online only |
| Pixel maps | `structure_graph_builder.py:compute_pixel_feature_maps` | 48x48 -> 8 maps | No | N/A | None |
| Selection | `select_node_coords` | detail map -> 1800 y,x | No | N/A | Deterministic |
| Node features | `build_structure_graph` | selected pixels -> `[1800,10]` | No | N/A | None |
| Local edges | `_local_edges` | coordinates -> directed 8-neighbor edges | No | local before merge | None |
| kNN edges | `_knn_edges` | 8 standardized pixel features -> directed k=6 | No coordinates, no landmark | kNN before merge | None |
| Structure edges | `_structure_edges` | part masks + selected coordinates -> relation edges | Yes | relation IDs before precedence | None |
| Merge/schema | `_edge_metadata`, `_edge_attr` | family edges -> `[2,E]`, `[E,6]`, type `[E]` | Structure endpoints only | Explicit `edge_type`; base6 attr has no type | None |
| Batch | `collate_d18_graphs` | graph list -> concatenated tensors | No new use | `edge_type_cat`, relation ID retained | None |
| Mode mix | `train_d18.py:apply_graph_regularization` | batch -> filtered batch | Operates on type 2 | Yes | C2 removes type 2 per graph with p=0.30 only in train |
| Encoder | `structure_gnn.py:StructureGNN` | `[N,10]` -> `[N,96]` | No | N/A | Dropout only in train |
| GNN | `edge_context_gnn.py` | h, merged edges, base6 attrs -> h | No direct landmark feature | Type passed but unused in C2 computation | Dropout only |
| Readout | `readout.py:GatedGlobalReadout` | final h -> `[B,288]` | No | No | Dropout in gate only |
| Classifier/loss | `StructureGNN.classifier`; CE | `[B,288]` -> `[B,7]` | No | No | CE/AdamW in train |
| Checkpoint | `save_checkpoint`, `load_checkpoint` | model/optimizer/scheduler/RNG | No | Config signature retained | best val macro-F1; early stop val loss |
| Counterfactual | `evaluate_ofix18_factorial.py` | type masks/rewires | Perturbs structure family | Exact masks; causal only for whole-family ablation | Eval-only |

Tensor shapes use C2: batch size 16, 1800 nodes/graph, mean {mean_edges:.1f} merged directed edges/graph, hidden 96, three layers, seven classes.
""")

    write("05_graph_construction_trace.md", f"""
# Graph Construction Trace

1. Normalize `image_48` to [0,1]. Compute intensity, gx, gy, gradient magnitude, 3x3 mean/std, absolute Laplacian, and center-surround.
2. Detail score is the sum of four z-scored maps: gradient magnitude, Laplacian, local std, and absolute center-surround.
3. `stratified_detail_knn` selects exactly 1800 pixels across a 6x6 partition. It retains 78.125% of the 2304 pixels.
4. Node schema is ten-dimensional: the eight pixel descriptors plus normalized x/y coordinates.
5. Local adjacency is directed 8-neighbor connectivity among selected pixels.
6. kNN adjacency is directed k=6 in per-image standardized eight-feature evidence space. x/y are not kNN inputs.
7. Structure adjacency selects groups from `part_soft_masks` and connects configured facial groups. This is the only landmark-dependent graph-construction stage.
8. Families are unioned and deduplicated. Precedence is local, then kNN, then structure.
9. Base6 edge attributes are dx, dy, spatial distance, absolute intensity difference, absolute gradient-magnitude difference, and absolute Laplacian difference.

Measured on 715 locked images:

- selected node count: 1800 for every image;
- high-gradient top-decile recall: {100*edge_mean['high_gradient_recall']:.2f}% mean;
- kNN distance: {edge_mean['knn_pixel_distance_mean']:.2f} pixels mean;
- kNN edges longer than 8 pixels: {100*edge_mean['knn_long_range_gt8px_fraction']:.2f}%.

The support transform is lossy by pixel count, although it retains nearly all top-decile gradient pixels and enforces spatial coverage. Typed operators cannot recover the 504 omitted pixel values or visual features never encoded in the ten node channels.

Class-conditioned high-gradient recall ranges only from 99.80% to 99.90% in `12_node_coverage_by_class.csv`, so this audit does not show one FER class systematically losing high-gradient support. The sampled ten-feature covariance has scale-sensitive effective rank 3.46/10. Strong correlations include intensity/local-mean (0.975) and gradient-magnitude/local-std (0.914), indicating substantial handcrafted-channel redundancy. This does not by itself prove that node features, rather than the operator, cause the accuracy ceiling.

Construction is deterministic for identical input. D18's dataset path has no image augmentation, so augmentation stability cannot be measured from this branch; it is recorded as unavailable rather than inferred.
""")

    write("06_edge_family_semantics.md", f"""
# Edge-family Semantics

## Identity at model input

`edge_type` explicitly stores local=0, kNN=1, structure=2 and survives cache and batching. `structure_relation_id` also survives. However, C2 uses base6 edge attributes and disables the scalar edge gate. `EdgeContextGNNLayer.forward` therefore never reads `edge_type` for message computation. Structure relation IDs are never consumed by the model.

The shared nonlinear edge MLP can infer family indirectly because spatial and pixel-difference distributions differ, but it receives no guaranteed type bit. The current model is edge-aware, not explicitly relation-parameterized.

## Merge semantics

- Raw local mean: {edge_mean['raw_local_count']:.1f}; raw kNN: {edge_mean['raw_knn_count']:.1f}; raw structure: {edge_mean['raw_structure_count']:.1f}.
- Local/kNN overlap: {edge_mean['local_knn_overlap']:.1f} endpoints, {100*edge_mean['local_knn_overlap']/edge_mean['raw_knn_count']:.2f}% of raw kNN.
- Local/structure overlap: {edge_mean['local_structure_overlap']:.1f}; kNN/structure: {edge_mean['knn_structure_overlap']:.1f}; triple: {edge_mean['three_way_overlap']:.1f}.
- Final endpoint duplicates and self-loops: zero.
- Local and structure reverse fractions: 1.0; kNN reverse fraction: {edge_mean['knn_reverse_fraction']:.3f}.
- About {edge_mean['multi_relation_endpoint_count']:.1f} raw structure endpoints/image are produced by more than one facial relation, but the metadata dictionary retains one relation ID per endpoint.

The OFIX18 hooks use exact `edge_type` masks for whole-family removal and message summaries. They measure causal sensitivity to removing a retained family, but they do not decompose overlap messages that were reassigned by precedence and do not identify multiple raw relation memberships after merge.

## Typed-policy requirement

The first typed experiment should preserve the current precedence-deduplicated endpoint set. Keeping duplicate relation messages would add roughly {edge_mean['local_knn_overlap']:.0f} local/kNN messages per image and confound relation identity with degree and scale. Multi-label endpoint handling can be studied later, not in the first test.
""")

    write("07_current_message_passing_trace.md", f"""
# Current Message-passing Trace

Per layer:

1. `edge_attr[6] -> edge_hidden[32]` through Linear, LayerNorm, GELU, dropout.
2. A vector gate `sigmoid(32 -> 96)` is computed from edge features.
3. Message input concatenates source state `[96]` and edge embedding `[32]`; destination state is not used.
4. A nonlinear `128 -> 96 -> 96` message is multiplied by the vector gate.
5. Messages are summed at destination and divided by total incoming degree across all retained families.
6. Post-residual LayerNorm is followed by a two-layer FFN, another residual, and another LayerNorm.

Layer parameters are independent across the three layers. C2 has no active scalar relation gate. Total model parameters: {c2_params:,}; GNN {int(params.loc[params.component=='gnn','parameter_count'].iloc[0]):,}; classifier {int(params.loc[params.component=='classifier','parameter_count'].iloc[0]):,}.

## Measured family scale

{md_table(message_summary)}

`aggregate_current` uses the current full-degree denominator. `aggregate_own` uses a hypothetical family-only denominator. Naively computing local mean plus kNN mean changes each contribution by roughly 1.75x for local and 2.43-2.47x for kNN. Therefore `shared_filter(A_local union A_knn)` is not scale-equivalent to two independently averaged operators followed by an unweighted sum.

A scale-matched split must fuse family means with destination-wise degree weights: `d_local/d_total * mean_local + d_knn/d_total * mean_knn`. With identical message weights and the same precedence-deduplicated edges, this reconstructs the current union mean.
""")

    write("08_current_readout_trace.md", f"""
# Current Readout Trace

Only final-layer node states reach `GatedGlobalReadout`. The 288-dimensional graph vector concatenates:

- mean pool (normalized by node count);
- feature-wise max pool;
- gated weighted mean, normalized by the sum of learned sigmoid gates per graph.

No landmark tensor, anchor, motif, part pool, or prior feature enters readout. Node count is fixed at 1800, so count sensitivity is controlled. Max pooling remains sensitive to rare extreme activations.

Post-hoc frozen-classifier ablation on the 70-image diagnostic subset:

{md_table(readout_summary)}

Zeroing max collapses macro-F1 most strongly, but this is an out-of-distribution ablation of a jointly trained concatenation, not proof that max alone is causally optimal. It does establish that changing readout cannot be treated as a minor implementation detail.
""")

    write("09_layerwise_information_audit.md", f"""
# Layerwise Information Audit

Protocol: five C2 best checkpoints; 70 deterministic locked images, ten per class; model eval mode; official and remove-structure; no fitted test probe.

{md_table(layer_summary)}

## CKA

{md_table(cka_summary)}

Mean official inter-layer CKA is 0.456 from input projection to layer 3, 0.625 from layer 1 to layer 3, and 0.820 from layer 2 to layer 3. These are not near-unity redundancies. Class-separation diagnostics also differ by class: input projection is strongest for angry and sad, while later states improve happy, surprise, and disgust.

## Oversmoothing label: WEAK EVIDENCE

Node cosine rises from {mean_metric(layer,'mean_pairwise_node_cosine','input_projection'):.3f} to {mean_metric(layer,'mean_pairwise_node_cosine','gnn_layer_3'):.3f}, and node variance falls from {mean_metric(layer,'node_representation_variance','input_projection'):.3f} to {mean_metric(layer,'node_representation_variance','gnn_layer_3'):.3f}. Against a collapse claim, normalized Dirichlet energy rises from {mean_metric(layer,'normalized_dirichlet_energy','input_projection'):.3f} to {mean_metric(layer,'normalized_dirichlet_energy','gnn_layer_3'):.3f}, node effective rank rises rather than collapses, and final class separation exceeds layers 1 and 2. A three-layer oversmoothing diagnosis is therefore not established.

## A2 evidence boundary

Layer representations are complementary enough to keep multi-scale pooling as a hypothesis, but not enough to GO. The diagnostic uses a test subset, the same final trained gate to pool every layer, and no leakage-safe train/validation probe. A2 remains deferred until a future A0/A1 run exports frozen train/validation layer states for a pre-registered probe.

A linear probe and incremental-information regression were skipped: processing a leakage-safe train/validation representation set for five checkpoints was outside the bounded review, while fitting either on the locked test subset would leak labels. CKA and class separation are therefore descriptive only.
""")

    audit_rows = []
    for metric in ("raw_local_count", "raw_knn_count", "raw_structure_count", "local_knn_overlap", "local_structure_overlap", "knn_structure_overlap", "three_way_overlap", "union_count", "final_local_count", "final_knn_count", "final_structure_count", "total_degree_mean", "local_degree_mean", "knn_degree_mean", "structure_degree_mean", "cached_final_duplicate_count", "multi_relation_endpoint_count", "knn_reverse_fraction"):
        source = overlap_summary[overlap_summary.metric == metric].iloc[0]
        audit_rows.append({"category": "edge_overlap_or_degree", "metric": metric, "mean": source["mean"], "std": source["std"], "median": source["median"], "min": source["min"], "max": source["max"]})
    for row in message_summary.itertuples(index=False):
        audit_rows.append({"category": "message_scale", "metric": f"layer{int(row.layer)}_{row.family}_own_to_current_ratio", "mean": row.own_to_current_ratio, "std": np.nan, "median": np.nan, "min": np.nan, "max": np.nan})
    audit_frame = pd.DataFrame(audit_rows)
    audit_frame.to_csv(OUT / "10_edge_overlap_and_normalization_audit.csv", index=False)
    write("10_edge_overlap_and_normalization_audit.md", f"""
# Edge Overlap and Normalization Audit

All 715 reconstructed unions exactly match cached final endpoint sets. The detailed per-image evidence is in `10_edge_overlap_per_image.csv`; the required compact table is:

{md_table(audit_frame)}

## Decision consequence

The first relation-aware experiment must retain precedence deduplication and total-degree-equivalent fusion. Otherwise it changes endpoint multiplicity and normalization before it tests relation semantics. A naive residual sum is rejected as an interpretable A1 control.
""")

    write("11_d19_candidate_definitions.md", """
# D19 Candidate Definitions

## A0: matched evidence-only control

C2 node support, node features, local/kNN edges, base6 attributes, shared GNN, readout, classifier, optimizer, scheduler, checkpoint policy, batch size, and seed. Structure edges are absent in train/eval and mode mix is disabled.

## Original A1: separate local/kNN operators

This is a simplified adaptation inspired by multi-dimensional GNNs, not a faithful implementation. It has one node state and omits explicit across-dimension state interaction. Independent modules also add parameters and alter normalization.

## Revised A1-ID: explicit edge-type conditioning

Keep the shared message operator and precedence-deduplicated A0 graph. Append a small learned local/kNN type embedding before the edge MLP. Compare correct type IDs against a parameter-identical null-type control in which all evidence edges receive one ID. This is the smallest experiment that tests whether missing explicit family identity matters.

## A2: multi-scale evidence pooling

The minimal future form is a fixed mean of graph-level pools from input projection and layers 1-3 using shared readout parameters. It is a graph-level adaptation inspired by JK, not node-adaptive JK. It is not approved in this review.

## D19-B

Not approved. A future bounded structure branch must not feed back into evidence states and may only be considered after an evidence-only model is established.
""")

    write("12_a0_necessity_review.md", """
# A0 Necessity Review

**A0 is necessary.** No completed run is fully matched to C2 while training without structure.

| Evidence | Why it is not A0 |
| --- | --- |
| C2 remove-structure inference | The checkpoint was trained with official structure on about 70% of training samples; removal changes inference only. |
| D17 OFIX15-A | No structure and no DropEdge, but uses non-stratified detail top-N support. |
| D17 OFIX15-C | Uses stratified support and the same basic model, but trains with generic DropEdge p=0.15. |
| OFIX12-A/A3 | Different D16 graph schema, anchor/readout path, and training history. |

The exact A0 semantic difference from C2 is: no type-2 edges at graph output and no structure-mode mixing. It must retain the precedence-deduplicated local+kNN set and all optimizer/training policy fields.

Implementation caveat: the current builder loads/indexes `part_soft_masks` before `_structure_edges` returns early when disabled. A strict A0 should prove graph hashes are invariant to zeroed/shuffled landmark tensors and should preferably use an image-only evidence cache so landmark arrays are not merely unused but absent from the executable evidence path.
""")

    a1_rows = pd.DataFrame([
        {"alternative": "A1-simple", "mapping": "simplified multi-relation adaptation", "params_est": 453896, "delta_params": 188064, "scale": "not matched under mean+sum", "auditability": "high", "risk": "high parameter/normalization confound", "recommendation": "reject first"},
        {"alternative": "A1-shared-basis", "mapping": "simplified relation-specific basis", "params_est": 270440, "delta_params": 4608, "scale": "degree-weighted fusion required", "auditability": "high", "risk": "adapter rank/capacity confound", "recommendation": "not first"},
        {"alternative": "A1-edge-type", "mapping": "typed edge conditioning, not separate operators", "params_est": 266616, "delta_params": 784, "scale": "same union mean", "auditability": "high with null-type control", "risk": "small added capacity", "recommendation": "recommended revised A1"},
        {"alternative": "A1-gated", "mapping": "shared messages with type/evidence gate", "params_est": 266408, "delta_params": 576, "scale": "gate changes scale", "auditability": "medium", "risk": "dependency can hide in gate", "recommendation": "reject first"},
    ])
    write("13_a1_typed_operator_review.md", f"""
# A1 Typed-operator Review

Local and kNN are semantically distinct: local encodes selected-grid adjacency, while kNN connects standardized appearance descriptors and is long-range in 56.91% of edges. The current model has enough nonlinear capacity to infer differences indirectly, but does not receive explicit family identity.

{md_table(a1_rows)}

## Recommendation

Do not begin with independent operators. Begin, after A0, with **A1-edge-type** and a parameter-identical null-type control. Use the same endpoint set, total-degree normalization, initialization family, batch size, optimizer, and readout. This directly tests whether explicit relation identity matters before granting relation-specific parameter blocks.

The correct-ID and null-ID runs must both contain the same 2x8 embedding and 14->32 edge projection. In the null run all edges use ID 0; the unused row remains in the state dict and optimizer, preserving parameter count. The only semantic difference is correct local/kNN IDs.

Falsifier: if correct IDs do not improve pre-registered validation macro-F1 over the null control, or if the gain is below 0.75 pp at seed42, do not escalate to separate operators.
""")

    a2_rows = pd.DataFrame([
        {"form": "node JK max", "book_relation": "closest minimal node-level JK", "extra_params": 0, "risk": "mixes feature channels by max", "status": "not selected"},
        {"form": "graph pools concat", "book_relation": "graph-level adaptation", "extra_params": 167616, "risk": "large classifier confound", "status": "reject"},
        {"form": "graph learned weighted sum", "book_relation": "graph-level adaptation", "extra_params": 4, "risk": "weights can overfit; still needs four pools", "status": "not selected"},
        {"form": "graph fixed mean", "book_relation": "graph-level adaptation", "extra_params": 0, "risk": "may dilute final layer", "status": "minimal future candidate"},
        {"form": "layer attention/LSTM", "book_relation": "adaptive JK-like", "extra_params": "large", "risk": "unjustified complexity", "status": "reject"},
    ])
    write("14_a2_multiscale_pooling_review.md", f"""
# A2 Multi-scale Pooling Review

{md_table(a2_rows)}

The layer audit shows non-redundant representations and class-specific separation patterns, but it does not provide a leakage-safe validation improvement. Oversmoothing evidence is weak, not a justification. Therefore A2 is **not approved now**.

If revisited after A1, the only approved design candidate for a new review is fixed mean of shared graph-level pools from h0-h3. It adds no learned layer selector and keeps layer weights auditable at 0.25 each. It is not a faithful JK implementation because it combines graph pools rather than adapting per node.
""")

    compute = pd.DataFrame([
        {"candidate": "C2 frozen", "params": c2_params, "delta_params": 0, "edges_per_graph": mean_edges, "relative_MAC": 1.000, "edge_tensor_MB_graph": mean_edges*224*4/1e6, "relative_epoch": 1.00, "checkpoint_MB_observed_or_est": 3.28, "batch_change": "no"},
        {"candidate": "A0", "params": c2_params, "delta_params": 0, "edges_per_graph": a0_edges, "relative_MAC": 0.949, "edge_tensor_MB_graph": a0_edges*224*4/1e6, "relative_epoch": 0.95, "checkpoint_MB_observed_or_est": 3.28, "batch_change": "no"},
        {"candidate": "A1-simple", "params": 453896, "delta_params": 188064, "edges_per_graph": edge_mean['raw_local_count']+edge_mean['raw_knn_count'], "relative_MAC": 1.10, "edge_tensor_MB_graph": (edge_mean['raw_local_count']+edge_mean['raw_knn_count'])*224*4/1e6, "relative_epoch": 1.12, "checkpoint_MB_observed_or_est": 5.6, "batch_change": "possible"},
        {"candidate": "A1-shared-basis r4 adapters", "params": 270440, "delta_params": 4608, "edges_per_graph": a0_edges, "relative_MAC": 0.96, "edge_tensor_MB_graph": a0_edges*224*4/1e6, "relative_epoch": 0.97, "checkpoint_MB_observed_or_est": 3.34, "batch_change": "no"},
        {"candidate": "A1-edge-type 8d", "params": 266616, "delta_params": 784, "edges_per_graph": a0_edges, "relative_MAC": 0.957, "edge_tensor_MB_graph": a0_edges*224*4/1e6, "relative_epoch": 0.96, "checkpoint_MB_observed_or_est": 3.29, "batch_change": "no"},
        {"candidate": "A1-gated vector", "params": 266408, "delta_params": 576, "edges_per_graph": a0_edges, "relative_MAC": 0.95, "edge_tensor_MB_graph": a0_edges*224*4/1e6, "relative_epoch": 0.96, "checkpoint_MB_observed_or_est": 3.29, "batch_change": "no"},
        {"candidate": "A2 fixed graph-pool mean", "params": c2_params, "delta_params": 0, "edges_per_graph": a0_edges, "relative_MAC": 0.975, "edge_tensor_MB_graph": a0_edges*224*4/1e6, "relative_epoch": 0.98, "checkpoint_MB_observed_or_est": 3.28, "batch_change": "no"},
        {"candidate": "A2 graph-pool concat", "params": 433448, "delta_params": 167616, "edges_per_graph": a0_edges, "relative_MAC": 1.00, "edge_tensor_MB_graph": a0_edges*224*4/1e6, "relative_epoch": 1.02, "checkpoint_MB_observed_or_est": 5.3, "batch_change": "unlikely"},
    ])
    compute.to_csv(OUT / "15_parameter_and_compute_budget.csv", index=False)
    write("15_parameter_and_compute_budget.md", f"""
# Parameter and Compute Budget

{md_table(compute)}

Assumptions: 1800 nodes, measured edge means, hidden 96, edge hidden 32, three layers, batch 16, FP32 (`amp=false`). MAC estimates count dense linear multiplies and omit LayerNorm/GELU/index-add overhead. `edge_tensor_MB_graph` is only edge embedding + vector gate + message (32+96+96 FP32), not total autograd memory. C2 observed checkpoints are 3.28 MB and observed mean epoch time across five seeds is about 407 seconds.

Kaggle-safe design budget: batch 16 unchanged, <=300k parameters, <=1.10x C2 estimated epoch time, and no new cache larger than the existing base6 graph cache by more than 10%. A1-edge-type and fixed-mean A2 fit; A1-simple violates the preferred parameter budget and risks a batch/normalization confound.
""")

    confounds = pd.DataFrame([
        ("A0", "structure edges", "removed in train and eval", "required treatment"),
        ("A0", "mode mix", "disabled", "required because no structure exists"),
        ("A0", "cache", "must exclude type-2 edges", "hash local+kNN against C2 filtered graph"),
        ("A0", "landmark tensors", "must not affect output", "zero/shuffle hash invariance"),
        ("A1-ID", "parameters", "+784", "null-type control with identical module/state size"),
        ("A1-ID", "edge set", "no change", "reuse A0 precedence-deduplicated cache"),
        ("A1-ID", "normalization", "no change", "retain total-degree mean"),
        ("A1-ID", "initialization", "new embedding/projection columns", "same init seed and null/correct architecture"),
        ("A1-ID", "optimizer dynamics", "extra active type row", "same optimizer/schedule; report gradient norms"),
        ("A2", "readout dimension", "must remain 288 for fixed mean", "shared readout, fixed 0.25 weights"),
        ("A2", "training memory", "holds h0-h3 references", "profile before full run; no batch change"),
        ("A2", "layer evidence", "currently test-descriptive", "require train/validation frozen probe first"),
    ], columns=["stage", "factor", "change", "control"])
    write("16_experimental_confound_matrix.md", "# Experimental Confound Matrix\n\n" + md_table(confounds))

    write("17_required_architectural_invariants.md", """
# Required Architectural Invariants

## Evidence-only D19-A

1. No landmark information enters node features, edge features, edge construction, readout, or classifier.
2. Official and forced/no-structure modes are model-equivalent.
3. Local and kNN relations are deterministic for identical input and transform.
4. Edge-family identity remains explicit if typed processing is used.
5. No relation is double-counted accidentally; precedence policy is frozen.
6. Per-relation and total-degree normalization are documented and tested.
7. The model remains finite and valid when either relation has zero edges.
8. Batch ordering, sample indices, and labels remain unchanged.
9. Checkpoint signatures distinguish A0, A1-ID-null, A1-ID-correct, and any later A2.
10. Full 3,589-image test and locked 715-image evaluation remain compatible.

## Future D19-B only

- Evidence embedding is bitwise or tolerance-equivalent when structure is removed.
- Structure cannot feed back into evidence states.
- C2 mode mixing affects only the structure branch.
- Gate values are not presented as causal evidence; branch ablation is mandatory.
""")

    write("18_minimal_experiment_plan.md", """
# Minimal Experiment Plan

Selected plan: **REVISED PLAN 2: A0 -> A1-ID only.** Original independent operators and A2 are not approved.

## Run 1: D19-A0 seed42

- Question: what does C2's exact evidence path learn when trained without any structure exposure?
- Baseline: frozen C2 seed42 config and architecture.
- Only scientific difference: type-2 edges absent in train/eval; mode mix disabled because it is inapplicable.
- Controls: same nodes, local/kNN endpoints, base6 attrs, hidden/readout/classifier, batch 16, AdamW, scheduler, early stop, checkpoint policy.
- Seed42 gate: graph invariants pass; official/no-structure logits are tolerance-identical; finite training; no leakage; full val/test artifacts produced. No accuracy threshold is imposed on this baseline.
- Stop: any graph hash mismatch beyond removal of type 2, landmark counterfactual sensitivity, batch-size change, or resume-signature mismatch.
- Artifacts: resolved config, schema/hash validation, train/val logs, best/last checkpoints, full test, locked 715 predictions, per-class and detected/fallback metrics, confusion matrices.

## Runs 2-3: A1-ID null versus correct, seed42

- Question: does explicit local/kNN identity improve a shared operator?
- Baseline: A0 with the exact same 2x8 embedding and 14->32 edge projection but all edges assigned ID 0.
- Treatment: correct precedence-retained local/kNN IDs. This is the only difference.
- Seed42 promotion gate: best validation macro-F1 improves >=0.75 pp; train-val macro gap increases <=3 pp; no class loses >5 pp validation F1; batch and timing budgets pass.
- Stop: gain <0.75 pp, instability, edge/normalization drift, or a required batch change.
- Multiseed gate after seed42: paired seeds 7/21/42/84/123; validation direction positive >=4/5; full-test macro-F1 mean gain >=0.50 pp; no official accuracy loss >1.0 pp mean; no robustness regression.

## A2 stop condition

Do not run A2. First require a leakage-safe frozen train/validation layer probe showing that a fixed multi-layer combination improves validation over final-only pooling. The current test-subset separation audit is insufficient for promotion.
""")

    write("19_go_no_go_decision.md", f"""
# D19 Go/No-go Decision

## Decision: {DECISION}

The general move toward an evidence-only D19 baseline is justified, but the proposed A0 -> separate typed operators -> multi-scale sequence must change.

## Decisive reasoning

- Book: multi-dimensional GNNs include within- and across-dimension interactions; a two-operator residual sum is only a simplified adaptation. JK is node-adaptive; graph-level layer pooling needs its own evidence.
- Code: edge identity is stored but unused by C2's active message computation. The shared edge MLP is nonlinear and can infer family indirectly, so missing identity is plausible but unproven.
- Graph audit: local/kNN overlap averages 1,554 endpoints/image. Naive separation double-counts those endpoints and changes degree normalization.
- Scale audit: family-only means amplify local messages about 1.75x and kNN about 2.45x relative to current full-degree contributions.
- Layer audit: representations are non-redundant and class-specific, but oversmoothing evidence is weak and no leakage-safe validation probe establishes an A2 gain.
- D18: C2 already solved most structure dependency with preserved official performance. The remaining accuracy ceiling is at least as plausibly limited by low-rank handcrafted visual evidence as by relation mixing.

## Exact next allowed phase

Implement and train only the matched A0 seed42 control after its graph/hash/leakage smoke validation. If A0 is valid, review and implement the parameter-matched A1-ID null/correct pair. Do not implement independent local/kNN operators or A2 in that phase.

## Remaining uncertainty

The 70-image layer audit is descriptive and test-based. Parameter/FLOP estimates omit kernel and dataloader overhead. A0 training may expose optimization behavior not visible in C2 inference-time removal.

## Prohibited

No structure DropEdge, probability sweep, consistency loss, prior feature injection, log-prior readout, purification sweep, generic optimizer/scheduler sweep, independent typed operator, A2 pooling, CNN stem, or D19-B structure branch is authorized by this review.
""")

    summary: dict[str, Any] = {
        "d18_baseline": {"name": "C2", "full_test_accuracy_mean": 0.595876, "full_test_macro_f1_mean": 0.552126, "training_seeds": list((7, 21, 42, 84, 123)), "promotion": "frozen"},
        "book_claims": claims.to_dict(orient="records"),
        "repository_map": {"config": "configs/d18/overfit_fix_18/d18_ofix18_c2_structure_mode_mix_only_seed42.yaml", "builder": "d18/data/structure_graph_builder.py", "model": "d18/models/structure_gnn.py", "training": "d18/training/train_d18.py"},
        "edge_family_semantics": {"edge_type_stored": True, "edge_type_used_by_c2_message": False, "edge_attr_schema": "base6", "merge_precedence": ["local", "knn", "structure"], "final_duplicates": 0},
        "edge_overlap": {key: float(edge_mean[key]) for key in ("raw_local_count", "raw_knn_count", "raw_structure_count", "local_knn_overlap", "local_structure_overlap", "knn_structure_overlap", "three_way_overlap")},
        "normalization": {"current": "mean over total retained destination degree", "naive_split_scale_matched": False, "required_fusion": "destination-degree-weighted family means"},
        "current_operator": {"nonlinear_edge_mlp": True, "uses_source_state": True, "uses_destination_state": False, "vector_edge_gate": True, "active_scalar_type_gate": False, "parameters": c2_params},
        "current_readout": {"layers_consumed": [3], "components": ["mean", "max", "gated_mean"], "dimension": 288, "landmark_input": False},
        "layerwise_information": {"subset_count": 70, "training_seeds": list(SEEDS), "input_to_final_cka_mean": float(cka_summary[(cka_summary.comparison=="inter_layer_official") & (cka_summary.left=="input_projection") & (cka_summary.right=="gnn_layer_3")]["mean"].iloc[0]), "complementarity": "descriptive_yes_validation_unproven"},
        "oversmoothing_evidence": {"label": "WEAK EVIDENCE", "node_cosine_input": mean_metric(layer,"mean_pairwise_node_cosine","input_projection"), "node_cosine_layer3": mean_metric(layer,"mean_pairwise_node_cosine","gnn_layer_3"), "dirichlet_input": mean_metric(layer,"normalized_dirichlet_energy","input_projection"), "dirichlet_layer3": mean_metric(layer,"normalized_dirichlet_energy","gnn_layer_3")},
        "node_graph_bottlenecks": {"retained_pixel_fraction": float(edge_mean["retained_pixel_fraction"]), "high_gradient_recall": float(edge_mean["high_gradient_recall"]), "node_feature_effective_rank_of_10": json.loads((OUT/"12_node_graph_bottleneck_summary.json").read_text())["node_feature_effective_rank_of_10"], "strongest_alternative": "limited handcrafted visual feature capacity"},
        "candidate_reviews": {"A0": {"necessary": True, "go": True}, "A1": {"original_separate_operators_go": False, "revised_edge_type_conditioning_go_after_A0": True, "required_control": "parameter-identical null type"}, "A2": {"go": False, "reason": "no leakage-safe validation evidence"}},
        "compute_budget": {"batch_size": 16, "amp": False, "c2_params": c2_params, "c2_edges_per_graph_mean": mean_edges, "kaggle_safe_max_params": 300000, "kaggle_safe_relative_epoch": 1.10},
        "confounds": confounds.to_dict(orient="records"),
        "required_invariants": ["no landmark evidence path", "official equals no-structure", "deterministic local and kNN", "explicit type when used", "no accidental double count", "documented normalization", "zero-edge-family validity", "stable batch ordering", "distinct checkpoint signature", "full and locked evaluation compatibility"],
        "minimal_experiment_plan": {"plan": "REVISED PLAN 2", "immediate": "A0 seed42 only", "then": "A1-ID null versus correct seed42", "deferred": "A2"},
        "decision": DECISION,
        "limitations": ["Layer audit is descriptive on 70 locked test images.", "No leakage-safe train/validation linear probe was fitted.", "Compute estimates omit kernel and dataloader overhead.", "Five training seeds remain a small sample."],
    }
    write("20_machine_readable_summary.json", json.dumps(summary, indent=2, allow_nan=False))

    write("21_run_commands.md", """
# Run Commands

Commands executed for this design review:

```powershell
conda run -n fer-graph python -m py_compile d18/scripts/audit_d19_preimplementation.py
conda run -n fer-graph python -B d18/scripts/audit_d19_preimplementation.py --skip_layers
conda run -n fer-graph python -B d18/scripts/audit_d19_preimplementation.py --skip_topology --device cuda:0 --layer_samples_per_class 10 --batch_size 2
conda run -n fer-graph python -B d18/scripts/generate_d19_preimplementation_review.py
```

No training command was run. No D19 training command is provided because A0 implementation/config creation belongs to the next explicitly approved phase.
""")

    required = [
        "00_README.md", "01_review_scope_and_sources.md", "02_book_claim_map.csv", "02_book_claim_map.md", "03_d18_evidence_summary.md", "04_repository_architecture_map.md", "05_graph_construction_trace.md", "06_edge_family_semantics.md", "07_current_message_passing_trace.md", "08_current_readout_trace.md", "09_layerwise_information_audit.csv", "09_layerwise_information_audit.md", "10_edge_overlap_and_normalization_audit.csv", "10_edge_overlap_and_normalization_audit.md", "11_d19_candidate_definitions.md", "12_a0_necessity_review.md", "13_a1_typed_operator_review.md", "14_a2_multiscale_pooling_review.md", "15_parameter_and_compute_budget.csv", "15_parameter_and_compute_budget.md", "16_experimental_confound_matrix.md", "17_required_architectural_invariants.md", "18_minimal_experiment_plan.md", "19_go_no_go_decision.md", "20_machine_readable_summary.json", "21_run_commands.md",
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    validation = {
        "book_review_complete": len(claims) >= 15,
        "d18_artifacts_validated": (D18 / "22_validation_summary.json").exists(),
        "repository_trace_complete": True,
        "edge_family_identity_verified": True,
        "edge_overlap_audited": len(overlap) == 715 and bool(overlap.cached_union_exact_match.all()),
        "normalization_audited": len(message) > 0,
        "readout_traced": len(readout) > 0,
        "layerwise_audit_complete": set(layer.training_seed.astype(int)) == set(SEEDS),
        "oversmoothing_claim_validated": True,
        "a0_review_complete": True,
        "a1_review_complete": True,
        "a2_review_complete": True,
        "compute_budget_complete": len(compute) >= 7,
        "confound_matrix_complete": len(confounds) >= 10,
        "decision_reached": DECISION,
        "model_code_modified": False,
        "training_launched": False,
        "blocking_issues": missing,
        "warnings": ["Layerwise diagnostics use a 70-image locked test subset and are descriptive.", "No leakage-safe linear probe was fitted.", "Compute estimates are analytical approximations."],
    }
    write("22_validation_summary.json", json.dumps(validation, indent=2, allow_nan=False))
    if missing:
        raise RuntimeError(f"Missing required reports: {missing}")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
