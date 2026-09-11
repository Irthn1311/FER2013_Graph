"""Pure-GNN v3.1 Scientific Training Scaffold (Historical Compatible).

SCIENTIFIC TRAINING IS CURRENTLY DISABLED PENDING INDEPENDENT SOURCE REVIEW.
"""

from pure_gnn_v31.scientific.config import ScientificConfig, load_scientific_config
from pure_gnn_v31.scientific.governance import (
    assert_not_test_access,
    validate_dataset_path,
    validate_split_row_counts,
    DataGovernanceError,
)
from pure_gnn_v31.scientific.metrics import (
    ScientificMetrics,
    compute_scientific_metrics,
    toy_reference_macro_f1,
)
from pure_gnn_v31.scientific.initialization import (
    synchronize_shared_parameters,
    verify_initialization_equivalence,
)
from pure_gnn_v31.scientific.checkpoints import (
    CheckpointSelector,
    EarliestStrictCheckpoint,
)
from pure_gnn_v31.scientific.evaluator import ScientificEvaluator
from pure_gnn_v31.scientific.trainer import ScientificTrainer

__all__ = [
    "ScientificConfig",
    "load_scientific_config",
    "assert_not_test_access",
    "validate_dataset_path",
    "validate_split_row_counts",
    "DataGovernanceError",
    "ScientificMetrics",
    "compute_scientific_metrics",
    "toy_reference_macro_f1",
    "synchronize_shared_parameters",
    "verify_initialization_equivalence",
    "CheckpointSelector",
    "EarliestStrictCheckpoint",
    "ScientificEvaluator",
    "ScientificTrainer",
]
