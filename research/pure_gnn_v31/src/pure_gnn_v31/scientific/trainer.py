"""Scientific training harness and production screen runner for Pure-GNN v3.1.

SCIENTIFIC TRAINING IS CURRENTLY DISABLED PENDING INDEPENDENT SOURCE REVIEW.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.contracts import count_parameters
from pure_gnn_v31.scientific.config import ScientificConfig, load_scientific_config
from pure_gnn_v31.scientific.governance import assert_not_test_access, DataGovernanceError
from pure_gnn_v31.scientific.randomness import set_scientific_seed
from pure_gnn_v31.scientific.dataset import (
    compute_file_sha256,
    load_fer_csv_split,
    create_epoch_paired_training_dataset,
    create_paired_dataset,
)
from pure_gnn_v31.scientific.evaluator import ScientificEvaluator
from pure_gnn_v31.scientific.checkpoints import CheckpointSelector
from pure_gnn_v31.scientific.initialization import (
    audit_and_synchronize_shared_parameters,
    audit_and_synchronize_non_coarse_parameters,
)
from pure_gnn_v31.scientific.evidence import get_git_commit_sha


@tf.keras.utils.register_keras_serializable(package="pure_gnn_v31")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Preregistered WarmupCosine learning rate schedule matching Gen2/Gen3 semantics."""

    def __init__(
        self,
        steps_per_epoch: int,
        initial_learning_rate: float,
        final_learning_rate: float,
        warmup_epochs: int,
        max_epochs: int,
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
    label_smoothing: float,
    num_classes: int = 7,
) -> tf.Tensor:
    """Categorical cross-entropy from logits with label smoothing.

    Converts sparse integer labels explicitly to one-hot depth 7.
    Returns per-example unreduced loss of shape [B].
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
    initial_learning_rate: float,
    final_learning_rate: float,
    warmup_epochs: int,
    max_epochs: int,
    weight_decay: float,
    global_clipnorm: float,
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

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped_epoch: Optional[int] = None

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
        train_csv_path: Union[str, Path],
        val_csv_path: Union[str, Path],
        output_dir: Union[str, Path],
        prebuilt_model: Optional[PureGNNv31] = None,
    ) -> Dict:
        """Executes the complete reviewed scientific training lifecycle for a single condition.

        FAILS CLOSED: Strictly calls assert_ready_for_execution() first.
        When authorized, executes the full reviewed pipeline. NO permanent raise after the gate!
        """
        # Hard fail-closed gate: raises PermissionError if scientific_execution_authorized is False
        self.config.assert_ready_for_execution()

        # Extract config-driven values (NO defaults)
        hp = self.config.hyperparameters
        sched_cfg = self.config.warmup_cosine_config
        if sched_cfg is None:
            raise ValueError("Missing WarmupCosineConfig in scientific config.")

        seed = int(self.config.seed)
        batch_size = int(hp["batch_size"]["value"])
        max_epochs = int(hp["max_epochs"]["value"])
        weight_decay = float(hp["weight_decay"]["value"])
        label_smoothing = float(hp["label_smoothing"]["value"])
        global_clipnorm = float(hp["global_clipnorm"]["value"])
        checkpoint_monitor = str(hp["checkpoint_monitor"]["value"])
        checkpoint_mode = str(hp["checkpoint_mode"]["value"])
        early_stopping_monitor = str(hp["early_stopping_monitor"]["value"])
        early_stopping_patience = int(hp["early_stopping_patience"]["value"])

        init_lr = float(sched_cfg.initial_learning_rate)
        final_lr = float(sched_cfg.final_learning_rate)
        warmup_epochs = int(sched_cfg.warmup_epochs)
        steps_per_epoch = math.ceil(self.config.train_rows / batch_size)

        # 1. Determinism setup BEFORE model creation
        determinism_report = set_scientific_seed(seed, fail_on_determinism_error=True)

        # 2. Build model after seed setup
        if prebuilt_model is not None:
            model = prebuilt_model
        else:
            model = PureGNNv31(condition=condition)
            dummy = tf.zeros([1, 48, 48, 1], dtype=tf.float32)
            _ = model(dummy, training=False)

        # 3. Load full data
        train_imgs, train_lbls, train_indices, train_sha = load_fer_csv_split(train_csv_path, role="train")
        val_imgs, val_lbls, val_indices, val_sha = load_fer_csv_split(val_csv_path, role="validation")

        # 4. Dataset builders
        def epoch_dataset_builder(epoch: int) -> tf.data.Dataset:
            return create_epoch_paired_training_dataset(
                train_imgs, train_lbls, train_indices,
                batch_size=batch_size, base_seed=seed, epoch=epoch, augment=True
            )

        val_dataset = create_paired_dataset(
            val_imgs, val_lbls, batch_size=batch_size, seed=seed, shuffle=False, indices=val_indices
        )

        return self.execute_training_lifecycle_internal(
            model=model,
            condition=condition,
            epoch_dataset_builder=epoch_dataset_builder,
            val_dataset=val_dataset,
            output_dir=output_dir,
            max_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=0.0,
            label_smoothing=label_smoothing,
            weight_decay=weight_decay,
            global_clipnorm=global_clipnorm,
            initial_learning_rate=init_lr,
            final_learning_rate=final_lr,
            warmup_epochs=warmup_epochs,
            seed=seed,
            checkpoint_monitor=checkpoint_monitor,
            checkpoint_mode=checkpoint_mode,
            use_full_validation_assertion=True,
            train_sha256=train_sha,
            val_sha256=val_sha,
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
        early_stopping_patience: int,
        early_stopping_min_delta: float,
        label_smoothing: float,
        weight_decay: float,
        global_clipnorm: float,
        initial_learning_rate: float,
        final_learning_rate: float,
        warmup_epochs: int,
        seed: int,
        checkpoint_monitor: str,
        checkpoint_mode: str,
        use_full_validation_assertion: bool,
        train_sha256: Optional[str] = None,
        val_sha256: Optional[str] = None,
    ) -> Dict:
        """Internal lower-level execution harness for executing the scientific lifecycle.

        Enforces:
        - Sample-weighted train loss and accuracy (per-example sums / total examples)
        - Sample-weighted val loss (per-example sum over full validation set / total examples)
        - Selected checkpoint reloading and full evaluation
        - Complete output schema generation
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        determinism_report = set_scientific_seed(seed, fail_on_determinism_error=True)

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
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
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
                sample_losses = compute_smoothed_cross_entropy(y, logits, label_smoothing=label_smoothing)
                loss = tf.reduce_mean(sample_losses)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            correct_count = tf.reduce_sum(tf.cast(tf.equal(y, preds), tf.float32))
            sample_loss_sum = tf.reduce_sum(sample_losses)
            return sample_loss_sum, correct_count

        for epoch in range(max_epochs):
            train_ds = epoch_dataset_builder(epoch)

            # Sample-weighted train metric accumulation
            total_train_loss_sum = 0.0
            total_train_correct = 0.0
            total_train_examples = 0

            for batch in train_ds:
                batch_x, batch_y = batch[0], batch[1]
                b_size = int(tf.shape(batch_x)[0].numpy())
                step_loss_sum, step_correct_count = _train_step(batch_x, batch_y)
                total_train_loss_sum += float(step_loss_sum.numpy())
                total_train_correct += float(step_correct_count.numpy())
                total_train_examples += b_size

            epoch_train_loss = total_train_loss_sum / total_train_examples
            epoch_train_acc = total_train_correct / total_train_examples

            # Validation evaluation
            if use_full_validation_assertion:
                val_metrics, _ = evaluator.evaluate_split(val_dataset, split_role="validation")
            else:
                val_metrics, _ = evaluator.evaluate_synthetic_subset(val_dataset, split_role="validation")

            # Sample-weighted validation loss accumulation
            total_val_loss_sum = 0.0
            total_val_examples = 0
            for batch in val_dataset:
                bx, by = batch[0], batch[1]
                b_size = int(tf.shape(bx)[0].numpy())
                vl = model(bx, training=False)
                sample_losses = compute_smoothed_cross_entropy(by, vl, label_smoothing=label_smoothing)
                total_val_loss_sum += float(tf.reduce_sum(sample_losses).numpy())
                total_val_examples += b_size

            epoch_val_loss = total_val_loss_sum / total_val_examples

            # Current LR from schedule
            current_step = (epoch + 1) * steps_per_epoch
            current_lr = float(optimizer.learning_rate(current_step).numpy())

            epoch_logs = {
                "epoch_index_zero_based": epoch,
                "epoch_number_one_based": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_accuracy": val_metrics.accuracy,
                "val_macro_f1": val_metrics.macro_f1,
                "learning_rate": current_lr,
            }
            history.append(epoch_logs)

            # Checkpoint selection on val_accuracy
            checkpoint_selector.update(epoch, epoch_logs, model)

            # Early stopping check on val_loss
            if early_stopper.check_stop(epoch, epoch_val_loss):
                break

        # Load best val_accuracy weights
        best_weights_path = output_path / "checkpoints" / f"best_{checkpoint_monitor}.weights.h5"
        if not best_weights_path.is_file():
            raise RuntimeError(f"Required best checkpoint weights file missing: {best_weights_path}")
        model.load_weights(str(best_weights_path))

        # Full validation evaluation on best weights
        if use_full_validation_assertion:
            final_metrics, final_records = evaluator.evaluate_split(val_dataset, split_role="validation")
        else:
            final_metrics, final_records = evaluator.evaluate_synthetic_subset(val_dataset, split_role="validation")

        # Write output artifacts
        # 1. train_history.csv
        history_csv_path = output_path / "train_history.csv"
        with history_csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "epoch_index_zero_based", "epoch_number_one_based",
                "train_loss", "train_accuracy", "val_loss",
                "val_accuracy", "val_macro_f1", "learning_rate",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)

        # 2. selected_checkpoint_metadata.json
        ckpt_meta = {
            "selected_epoch_zero_based": checkpoint_selector.selected_epoch_index_zero_based,
            "selected_epoch_one_based": checkpoint_selector.selected_epoch_number_one_based,
            "best_val_accuracy": checkpoint_selector.best_value,
            "checkpoint_weights_file": str(best_weights_path.name),
        }
        (output_path / "selected_checkpoint_metadata.json").write_text(
            json.dumps(ckpt_meta, indent=2), encoding="utf-8"
        )

        # 3. val_metrics.json
        val_metrics_data = {
            "accuracy": final_metrics.accuracy,
            "macro_f1": final_metrics.macro_f1,
            "precision_per_class": final_metrics.precision_per_class,
            "recall_per_class": final_metrics.recall_per_class,
            "f1_per_class": final_metrics.f1_per_class,
            "confusion_matrix": final_metrics.confusion_matrix,
        }
        (output_path / "val_metrics.json").write_text(
            json.dumps(val_metrics_data, indent=2), encoding="utf-8"
        )

        # 4. val_predictions.csv
        val_pred_path = output_path / "val_predictions.csv"
        with val_pred_path.open("w", newline="", encoding="utf-8") as f:
            header = ["source_row_index", "y_true", "predicted_class"] + [f"logit_{c}" for c in range(7)]
            writer = csv.writer(f)
            writer.writerow(header)
            for idx, yt, yp, lg in zip(
                final_records["source_row_index"],
                final_records["y_true"],
                final_records["predicted_class"],
                final_records["logits"],
            ):
                writer.writerow([idx, yt, yp] + [float(x) for x in lg])

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
            "output_dir": str(output_path),
        }


def run_production_scientific_screen(
    config: ScientificConfig,
    train_csv_path: Union[str, Path],
    val_csv_path: Union[str, Path],
    output_root: Union[str, Path],
    reviewed_source_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Orchestrates the canonical paired production screen for seed 42 across G0, G0.5, G1.

    Order:
    1. config.assert_ready_for_execution() (fails closed if unauthorized)
    2. Strict determinism setup
    3. Construct models G0, G0.5, G1
    4. G0.5 vs G1 and G0 vs G0.5 pairing & audit
    5. Write initialization_audit.json
    6. Execute runs into newly created non-existing output dirs (exist_ok=False)
    """
    config.assert_ready_for_execution()

    protocol_id = "pure_gnn_v31_historical_v1"
    seed = int(config.seed)
    output_base = Path(output_root) / protocol_id

    # 1. Determinism setup BEFORE model creation
    determinism_report = set_scientific_seed(seed, fail_on_determinism_error=True)

    # 2. Build models under fixed seed
    dummy_input = tf.zeros([1, 48, 48, 1], dtype=tf.float32)
    models = {}
    for cond in ["G0", "G0.5", "G1"]:
        m = PureGNNv31(condition=cond)
        _ = m(dummy_input, training=False)
        models[cond] = m

    # 3. Synchronization & Audit
    audit_non_coarse = audit_and_synchronize_non_coarse_parameters(models["G0"], models["G0.5"])
    audit_shared = audit_and_synchronize_shared_parameters(models["G0.5"], models["G1"])

    init_audit = {
        "seed": seed,
        "pairing_g0_g05": audit_non_coarse,
        "pairing_g05_g1": audit_shared,
    }

    # 4. Execute runs for G0, G0.5, G1
    trainer = ScientificTrainer(config)
    repo_root = Path(__file__).resolve().parents[4]
    commit_sha = get_git_commit_sha(repo_root)

    cfg_file = Path(__file__).resolve().parents[3] / "configs" / "scientific_screen_historical_v1.yaml"
    cfg_sha = compute_file_sha256(cfg_file) if cfg_file.is_file() else "MISSING"

    results = {}

    for cond in ["G0", "G0.5", "G1"]:
        cond_dir = output_base / cond / f"seed_{seed}"
        if cond_dir.exists():
            raise FileExistsError(f"Production output directory already exists: {cond_dir}. Overwrite forbidden.")
        cond_dir.mkdir(parents=True, exist_ok=False)

        # Write initialization_audit.json BEFORE training
        (cond_dir / "initialization_audit.json").write_text(
            json.dumps(init_audit, indent=2), encoding="utf-8"
        )
        (cond_dir / "determinism_report.json").write_text(
            json.dumps(determinism_report, indent=2), encoding="utf-8"
        )

        # Train condition
        run_res = trainer.train_condition(
            condition=cond,
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            output_dir=cond_dir,
            prebuilt_model=models[cond],
        )

        train_sha = compute_file_sha256(train_csv_path)
        val_sha = compute_file_sha256(val_csv_path)
        param_counts = count_parameters(models[cond])

        run_identity = {
            "condition": cond,
            "seed": seed,
            "source_commit": commit_sha,
            "reviewed_source_tag": reviewed_source_tag,
            "config_sha256": cfg_sha,
            "train_sha256": train_sha,
            "val_sha256": val_sha,
            "parameter_count": param_counts["total_trainable_parameters"],
        }
        (cond_dir / "run_identity.json").write_text(
            json.dumps(run_identity, indent=2), encoding="utf-8"
        )
        results[cond] = run_res

    return results
