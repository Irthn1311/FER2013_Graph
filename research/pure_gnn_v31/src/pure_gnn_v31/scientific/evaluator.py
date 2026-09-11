"""Validation evaluation harness for Pure-GNN scientific runs."""

from typing import Dict, Optional, Tuple
import numpy as np
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.scientific.governance import assert_not_test_access, DataGovernanceError
from pure_gnn_v31.scientific.metrics import ScientificMetrics, compute_scientific_metrics


class ScientificEvaluator:
    """Evaluates validation performance. Strictly rejects test datasets."""

    def __init__(self, model: PureGNNv31):
        self.model = model

    def evaluate_split(
        self,
        dataset: tf.data.Dataset,
        split_role: str,
    ) -> ScientificMetrics:
        """Evaluates model on validation dataset.
        
        Strictly rejects split_role == 'test'.
        """
        role = str(split_role).strip().lower()
        if role == "test" or "test" in role:
            raise DataGovernanceError(
                f"STRICT DATA GOVERNANCE VIOLATION: Evaluator cannot be run on '{split_role}'. "
                "Test holdout evaluation is strictly prohibited during model selection!"
            )
        if role != "validation":
            raise ValueError(f"Expected split_role='validation', got '{split_role}'")

        all_preds = []
        all_labels = []

        for batch_x, batch_y in dataset:
            logits = self.model(batch_x, training=False)
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            all_preds.append(preds.numpy())
            all_labels.append(batch_y.numpy())

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0)

        metrics = compute_scientific_metrics(y_true, y_pred)
        return metrics
