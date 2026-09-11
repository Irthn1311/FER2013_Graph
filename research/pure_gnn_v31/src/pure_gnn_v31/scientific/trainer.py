"""Scientific training harness for Pure-GNN v3.1.

SCIENTIFIC TRAINING IS CURRENTLY DISABLED PENDING INDEPENDENT SOURCE REVIEW.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.scientific.config import ScientificConfig, load_scientific_config
from pure_gnn_v31.scientific.governance import assert_not_test_access, DataGovernanceError


class ScientificTrainer:
    """Orchestrates scientific training under strict protocol and governance contracts."""

    def __init__(self, config: Optional[ScientificConfig] = None):
        self.config = config if config is not None else load_scientific_config()

    def train_condition(
        self,
        condition: str,
        train_dataset: Optional[tf.data.Dataset] = None,
        val_dataset: Optional[tf.data.Dataset] = None,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """Attempts to execute training for a condition (e.g. G0, G0.5, G1).
        
        FAILS CLOSED: Strictly raises PermissionError because scientific_execution_authorized is false
        or unresolved hyperparameters exist.
        """
        # Hard fail-closed execution check before ANY computation or data loading
        self.config.assert_ready_for_execution()

        raise PermissionError(
            "SCIENTIFIC EXECUTION BLOCKED: Scientific training is not authorized."
        )
