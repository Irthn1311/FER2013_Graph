"""Scientific training harness for Pure-GNN v3.1.

SCIENTIFIC TRAINING IS CURRENTLY DISABLED PENDING INDEPENDENT SOURCE REVIEW.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.scientific.config import ScientificConfig, load_scientific_config
from pure_gnn_v31.scientific.governance import assert_not_test_access, DataGovernanceError
from pure_gnn_v31.scientific.randomness import set_scientific_seed
from pure_gnn_v31.scientific.evaluator import ScientificEvaluator
from pure_gnn_v31.scientific.checkpoints import CheckpointSelector


@tf.keras.utils.register_keras_serializable(package="pure_gnn_v31")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Preregistered WarmupCosine learning rate schedule matching Gen2/Gen3 semantics."""

    def __init__(
        self,
        steps_per_epoch: int,
        initial_learning_rate: float = 3e-4,
        final_learning_rate: float = 1e-6,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.steps_per_epoch = int(steps_per_epoch)
        self.initial_learning_rate = float(initial_learning_rate)
        self.final_learning_rate = float(final_learning_rate)
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = float(self.steps_per_epoch * self.warmup_epochs)
        total_steps = float(self.steps_per_epoch * self.max_epochs)
        warmup = self.initial_learning_rate * step / warmup_steps
        progress = tf.clip_by_value(
            (step - warmup_steps) / (total_steps - warmup_steps), 0.0, 1.0
        )
        cosine = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * progress))
        decayed = self.final_learning_rate + (
            self.initial_learning_rate - self.final_learning_rate
        ) * cosine
        return tf.where(step < warmup_steps, warmup, decayed)

    def get_config(self):
        return {
            "steps_per_epoch": self.steps_per_epoch,
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "warmup_epochs": self.warmup_epochs,
            "max_epochs": self.max_epochs,
        }


def compute_smoothed_cross_entropy(
    labels: tf.Tensor,
    logits: tf.Tensor,
    label_smoothing: float = 0.05,
    num_classes: int = 7,
) -> tf.Tensor:
    """Categorical cross-entropy from logits with label smoothing.

    Converts sparse integer labels explicitly to one-hot depth 7.
    """
    labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    one_hot = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
    return tf.keras.losses.categorical_crossentropy(
        one_hot,
        tf.cast(logits, tf.float32),
        from_logits=True,
        label_smoothing=label_smoothing,
    )


def build_scientific_optimizer(
    steps_per_epoch: int,
    initial_learning_rate: float = 3e-4,
    final_learning_rate: float = 1e-6,
    warmup_epochs: int = 5,
    max_epochs: int = 100,
    weight_decay: float = 5e-4,
    global_clipnorm: float = 1.0,
) -> tf.keras.optimizers.AdamW:
    """Constructs the preregistered AdamW optimizer with WarmupCosine schedule."""
    schedule = WarmupCosine(
        steps_per_epoch=steps_per_epoch,
        initial_learning_rate=initial_learning_rate,
        final_learning_rate=final_learning_rate,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
    )
    return tf.keras.optimizers.AdamW(
        learning_rate=schedule,
        weight_decay=weight_decay,
        global_clipnorm=global_clipnorm,
    )


class EarlyStoppingTracker:
    """Tracks early stopping on validation loss with patience and min_delta."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped_epoch = None

    def check_stop(self, epoch_zero_based: int, val_loss: float) -> bool:
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch_zero_based
                return True
            return False


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

        FAILS CLOSED: Strictly raises PermissionError because scientific_execution_authorized is false.
        """
        # 1. Hard fail-closed execution check before ANY computation or data loading
        self.config.assert_ready_for_execution()

        raise PermissionError(
            "SCIENTIFIC EXECUTION BLOCKED: Scientific training is not authorized."
        )

    def execute_training_lifecycle_internal(
        self,
        model: PureGNNv31,
        condition: str,
        epoch_dataset_builder: Callable[[int], tf.data.Dataset],
        val_dataset: tf.data.Dataset,
        output_dir: Union[str, Path],
        max_epochs: int,
        steps_per_epoch: int,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 0.0,
        label_smoothing: float = 0.05,
        weight_decay: float = 5e-4,
        global_clipnorm: float = 1.0,
        initial_learning_rate: float = 3e-4,
        final_learning_rate: float = 1e-6,
        warmup_epochs: int = 5,
        seed: int = 42,
        use_full_validation_assertion: bool = False,
    ) -> Dict:
        """Internal lower-level execution harness for executing the scientific lifecycle.

        Strictly separated from public train_condition() to allow full testing with synthetic fixtures.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Enforce strict op determinism before epoch 1
        determinism_report = set_scientific_seed(seed, fail_on_determinism_error=True)

        # 2. Build optimizer
        optimizer = build_scientific_optimizer(
            steps_per_epoch=steps_per_epoch,
            initial_learning_rate=initial_learning_rate,
            final_learning_rate=final_learning_rate,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            weight_decay=weight_decay,
            global_clipnorm=global_clipnorm,
        )

        checkpoint_selector = CheckpointSelector(
            monitor="val_accuracy",
            mode="max",
            output_dir=output_path / "checkpoints",
        )
        early_stopper = EarlyStoppingTracker(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        )
        evaluator = ScientificEvaluator(model)

        history: List[Dict[str, float]] = []

        @tf.function
        def _train_step(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = tf.reduce_mean(compute_smoothed_cross_entropy(y, logits, label_smoothing=label_smoothing))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            acc = tf.reduce_mean(tf.cast(tf.equal(y, preds), tf.float32))
            return loss, acc

        for epoch in range(max_epochs):
            # Epoch-paired dataset
            train_ds = epoch_dataset_builder(epoch)

            train_losses = []
            train_accs = []

            for batch in train_ds:
                batch_x, batch_y = batch[0], batch[1]
                loss_val, acc_val = _train_step(batch_x, batch_y)
                train_losses.append(float(loss_val.numpy()))
                train_accs.append(float(acc_val.numpy()))

            epoch_train_loss = float(np.mean(train_losses))
            epoch_train_acc = float(np.mean(train_accs))

            # Validation evaluation
            if use_full_validation_assertion:
                val_metrics, _ = evaluator.evaluate_split(val_dataset, split_role="validation")
            else:
                val_metrics, _ = evaluator.evaluate_synthetic_subset(val_dataset, split_role="validation")

            # Compute validation loss
            val_batch_losses = []
            for batch in val_dataset:
                bx, by = batch[0], batch[1]
                vl = model(bx, training=False)
                v_loss = tf.reduce_mean(compute_smoothed_cross_entropy(by, vl, label_smoothing=label_smoothing))
                val_batch_losses.append(float(v_loss.numpy()))
            epoch_val_loss = float(np.mean(val_batch_losses))

            epoch_logs = {
                "epoch": epoch,
                "epoch_one_based": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_accuracy": val_metrics.accuracy,
                "val_macro_f1": val_metrics.macro_f1,
            }
            history.append(epoch_logs)

            # Checkpoint selection
            checkpoint_selector.update(epoch, epoch_logs, model)

            # Early stopping check on val_loss
            if early_stopper.check_stop(epoch, epoch_val_loss):
                break

        # After training ends: restore best val_accuracy weights
        best_weights_path = output_path / "checkpoints" / "best_val_accuracy.weights.h5"
        if best_weights_path.is_file():
            model.load_weights(str(best_weights_path))

        # Final evaluation on validation set with best weights
        if use_full_validation_assertion:
            final_metrics, final_records = evaluator.evaluate_split(val_dataset, split_role="validation")
        else:
            final_metrics, final_records = evaluator.evaluate_synthetic_subset(val_dataset, split_role="validation")

        return {
            "condition": condition,
            "seed": seed,
            "determinism_report": determinism_report,
            "history": history,
            "selected_epoch_zero_based": checkpoint_selector.selected_epoch_index_zero_based,
            "selected_epoch_one_based": checkpoint_selector.selected_epoch_number_one_based,
            "best_val_accuracy": checkpoint_selector.best_value,
            "final_metrics": final_metrics,
            "final_records": final_records,
        }
