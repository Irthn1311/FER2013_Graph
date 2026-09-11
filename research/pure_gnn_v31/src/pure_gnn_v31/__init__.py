"""Pure-GNN v3.1 Package Namespace."""

__version__ = "0.1.0"

from pure_gnn_v31.graph_index import (
    build_grid_8neighbor_graph,
    build_complete_coarse_graph,
)
from pure_gnn_v31.local_relation import LocalAdaptiveRelationBlock
from pure_gnn_v31.coarsening import (
    FixedAntiAliasedCoarsening,
    compute_aa_weights_and_indices,
    compute_simple_mean_indices,
)
from pure_gnn_v31.coarse_conditions import CoarseBlock
from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.data import (
    load_fer2013_train_csv,
    create_research_split_manifest,
    assert_not_test_access,
)
from pure_gnn_v31.metrics import (
    compute_accuracy,
    compute_macro_f1,
    compute_confusion_matrix,
    CrossEntropyWithLabelSmoothing,
)
from pure_gnn_v31.diagnostics import (
    compute_feature_diagnostics,
    compute_boundary_diagnostics,
    compute_shift_stability,
)
from pure_gnn_v31.contracts import (
    count_parameters,
    verify_parameter_budget,
    audit_forbidden_layers,
    verify_gate_initialization,
)

__all__ = [
    "PureGNNv31",
    "build_grid_8neighbor_graph",
    "build_complete_coarse_graph",
    "LocalAdaptiveRelationBlock",
    "FixedAntiAliasedCoarsening",
    "compute_aa_weights_and_indices",
    "compute_simple_mean_indices",
    "CoarseBlock",
    "load_fer2013_train_csv",
    "create_research_split_manifest",
    "assert_not_test_access",
    "compute_accuracy",
    "compute_macro_f1",
    "compute_confusion_matrix",
    "CrossEntropyWithLabelSmoothing",
    "compute_feature_diagnostics",
    "compute_boundary_diagnostics",
    "compute_shift_stability",
    "count_parameters",
    "verify_parameter_budget",
    "audit_forbidden_layers",
    "verify_gate_initialization",
]
