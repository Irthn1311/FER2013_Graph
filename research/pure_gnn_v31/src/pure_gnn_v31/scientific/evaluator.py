"""Validation evaluation harness for Pure-GNN scientific runs."""

from typing import Dict, List, Optional, Tuple
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
        assert_full_validation_count: bool = False,
    ) -> Tuple[ScientificMetrics, Dict[str, np.ndarray]]:
        """Evaluates model on validation dataset.

        Strictly rejects split_role == 'test'.

        Records and returns:
            source_row_index: (N,) int32
            y_true: (N,) int32
            predicted_class: (N,) int32
            logits: (N, num_classes) float32

        When assert_full_validation_count is True:
            Asserts exactly 3589 unique source indices, complete sorted index set, no duplicates.
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
        all_logits = []
        all_indices = []

        for batch in dataset:
            # P0-2: Scientific validation datasets must yield exactly 3 elements: (image, label, source_row_index)
            if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                raise DataGovernanceError(
                    "Scientific validation datasets must yield exactly 3 elements: "
                    "(image, label, source_row_index). Missing source index is strictly prohibited!"
                )
            batch_x, batch_y, batch_idx = batch
            all_indices.append(batch_idx.numpy())

            logits = self.model(batch_x, training=False)
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)

            all_logits.append(logits.numpy())
            all_preds.append(preds.numpy())
            all_labels.append(batch_y.numpy())

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        logits_arr = np.concatenate(all_logits, axis=0)
        idx_arr = np.concatenate(all_indices, axis=0)

        # Detailed validation assertions if evaluating full validation split
        if assert_full_validation_count:
            if len(idx_arr) != 3589:
                raise DataGovernanceError(
                    f"Expected 3589 validation examples, observed {len(idx_arr)}."
                )
            unique_indices = np.unique(idx_arr)
            if len(unique_indices) != 3589:
                raise DataGovernanceError(
                    f"Duplicate validation indices detected: {len(unique_indices)} unique out of {len(idx_arr)}."
                )
            if not np.array_equal(np.sort(unique_indices), np.arange(3589)):
                raise DataGovernanceError(
                    "Validation index set is incomplete, contains gaps, or does not equal 0..3588 exactly."
                )

        metrics = compute_scientific_metrics(y_true, y_pred)
        detailed_records = {
            "source_row_index": idx_arr,
            "y_true": y_true,
            "predicted_class": y_pred,
            "logits": logits_arr,
        }

        return metrics, detailed_records
