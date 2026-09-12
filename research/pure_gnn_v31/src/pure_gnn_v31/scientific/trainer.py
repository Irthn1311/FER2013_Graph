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
from pure_gnn_v31.scientific.config import ScientificConfig, load_scientific_config, ConfigurationError
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
from pure_gnn_v31.scientific.source_lock import verify_immutable_source_lock
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
) -> Tuple[tf.keras.optimizers.AdamW, WarmupCosine]:
    """Constructs the preregistered AdamW optimizer and retains explicit WarmupCosine schedule reference."""
    schedule = WarmupCosine(
        steps_per_epoch=steps_per_epoch,
        initial_learning_rate=initial_learning_rate,
        final_learning_rate=final_learning_rate,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=schedule,
        weight_decay=weight_decay,
        global_clipnorm=global_clipnorm,
    )
    return optimizer, schedule


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

    def run_screen(
        self,
        train_csv_path: Union[str, Path],
        val_csv_path: Union[str, Path],
        output_root: Union[str, Path],
        repo_root: Path,
        reviewed_source_tag: str,
    ) -> Dict[str, Any]:
        """Runs the canonical production scientific screen across G0, G0.5, G1."""
        return run_production_scientific_screen(
            config=self.config,
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            output_root=output_root,
            repo_root=repo_root,
            reviewed_source_tag=reviewed_source_tag,
        )

    def train_condition(self, *args, **kwargs):
        """Fails closed by asserting execution readiness first."""
        self.config.assert_ready_for_execution()


def _train_prebuilt_condition(
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
) -> Dict:
    """Private internal lower-level execution harness for training a single prebuilt condition."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    determinism_report = set_scientific_seed(seed, fail_on_determinism_error=True)

    optimizer, schedule = build_scientific_optimizer(
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

        # Sample-weighted validation loss accumulation over all examples
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

        # Learning rate from schedule(optimizer.iterations)
        logged_lr = float(schedule(optimizer.iterations).numpy())

        epoch_logs = {
            "epoch_index_zero_based": epoch,
            "epoch_number_one_based": epoch + 1,
            "train_loss": epoch_train_loss,
            "train_accuracy": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_accuracy": val_metrics.accuracy,
            "val_macro_f1": val_metrics.macro_f1,
            "learning_rate": logged_lr,
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

    ckpt_meta = {
        "selected_epoch_zero_based": checkpoint_selector.selected_epoch_index_zero_based,
        "selected_epoch_one_based": checkpoint_selector.selected_epoch_number_one_based,
        "best_val_accuracy": checkpoint_selector.best_value,
        "checkpoint_weights_file": str(best_weights_path.name),
    }
    (output_path / "selected_checkpoint_metadata.json").write_text(
        json.dumps(ckpt_meta, indent=2), encoding="utf-8"
    )

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
    repo_root: Path,
    reviewed_source_tag: str,
) -> Dict[str, Any]:
    """Canonical public scientific execution API for Pure-GNN v3.1.

    Orchestrates the canonical paired production screen for seed 42 across G0, G0.5, G1.

    Order:
    1. config.assert_ready_for_execution() (fails closed if unauthorized)
    2. verify_immutable_source_lock() (enforces tag existence, peeled commit, detached HEAD, clean tree)
    3. set_scientific_seed(seed) BEFORE model creation
    4. Build models G0, G0.5, G1
    5. Synchronization & audit: G0 vs G0.5 non-coarse, G0.5 vs G1 shared non-gate
    6. Initial logits equivalence check on fixed synthetic input: max_abs(logits_g05 - logits_g1) <= 1e-5
    7. Write initialization_audit.json BEFORE training
    8. Execute runs into newly created non-existing output dirs (exist_ok=False)
    """
    # 1. Config gate (strictly fails closed if scientific_execution_authorized is false)
    config.assert_ready_for_execution()

    # 2. Immutable Source Lock enforced by production runner itself
    source_lock_report = verify_immutable_source_lock(
        repo_root=repo_root,
        reviewed_source_tag=reviewed_source_tag,
    )

    # 3. Canonical Config Verification: ensure config comes strictly from tagged repo
    canonical_config_path = (
        Path(repo_root) / "research/pure_gnn_v31/configs/scientific_screen_historical_v1.yaml"
    ).resolve()
    if not canonical_config_path.is_file():
        raise FileNotFoundError(f"Canonical config not found in tagged repo: {canonical_config_path}")

    if Path(config.source_config_path).resolve() != canonical_config_path:
        raise ConfigurationError(
            f"IMMUTABLE CONFIG GOVERNANCE FAILED: Config path '{config.source_config_path}' "
            f"does not match canonical tagged config path '{canonical_config_path}'."
        )

    canonical_raw_sha256 = hashlib.sha256(canonical_config_path.read_bytes()).hexdigest()
    if config.source_config_sha256 != canonical_raw_sha256:
        raise ConfigurationError(
            f"IMMUTABLE CONFIG GOVERNANCE FAILED: Config SHA256 '{config.source_config_sha256}' "
            f"does not match canonical file SHA256 '{canonical_raw_sha256}'."
        )

    protocol_id = "pure_gnn_v31_historical_v1"
    seed = int(config.seed)
    hp = config.hyperparameters
    sched_cfg = config.warmup_cosine_config
    if sched_cfg is None:
        raise ValueError("Missing WarmupCosineConfig in scientific config.")

    batch_size = int(hp["batch_size"]["value"])
    max_epochs = int(hp["max_epochs"]["value"])
    weight_decay = float(hp["weight_decay"]["value"])
    label_smoothing = float(hp["label_smoothing"]["value"])
    global_clipnorm = float(hp["global_clipnorm"]["value"])
    checkpoint_monitor = str(hp["checkpoint_monitor"]["value"])
    checkpoint_mode = str(hp["checkpoint_mode"]["value"])
    early_stopping_patience = int(hp["early_stopping_patience"]["value"])

    init_lr = float(sched_cfg.initial_learning_rate)
    final_lr = float(sched_cfg.final_learning_rate)
    warmup_epochs = int(sched_cfg.warmup_epochs)
    steps_per_epoch = math.ceil(config.train_rows / batch_size)

    output_base = Path(output_root) / protocol_id

    # 3. Determinism setup BEFORE model creation
    determinism_report = set_scientific_seed(seed, fail_on_determinism_error=True)

    # 4. Build models under fixed seed
    dummy_input = tf.zeros([1, 48, 48, 1], dtype=tf.float32)
    models = {}
    for cond in ["G0", "G0.5", "G1"]:
        m = PureGNNv31(condition=cond)
        _ = m(dummy_input, training=False)
        models[cond] = m

    # 5. Synchronization & Audit
    audit_non_coarse = audit_and_synchronize_non_coarse_parameters(models["G0"], models["G0.5"])
    audit_shared = audit_and_synchronize_shared_parameters(models["G0.5"], models["G1"])

    # 6. Production initialization equivalence check on fixed synthetic input
    synth_eval_input = tf.random.normal([2, 48, 48, 1], seed=12345)
    logits_g05 = models["G0.5"](synth_eval_input, training=False)
    logits_g1 = models["G1"](synth_eval_input, training=False)
    initial_logits_max_abs_error = float(tf.reduce_max(tf.abs(logits_g05 - logits_g1)).numpy())

    if initial_logits_max_abs_error > 1e-5:
        raise RuntimeError(
            f"INITIALIZATION EQUIVALENCE FAILED: max abs error {initial_logits_max_abs_error} > 1e-5. "
            "G0.5 and G1 must produce equivalent initial outputs before training."
        )

    # Inspect G1 coarse gate final projection max abs value
    gate_dense2_max = 0.0
    for block in models["G1"].stage4_blocks:
        if hasattr(block, "coarse_gate_dense2"):
            for w in block.coarse_gate_dense2.weights:
                w_max = float(tf.reduce_max(tf.abs(w)).numpy())
                if w_max > gate_dense2_max:
                    gate_dense2_max = w_max

    init_audit = {
        "seed": seed,
        "pairing_g0_g05": audit_non_coarse,
        "pairing_g05_g1": audit_shared,
        "exact_shared_tensor_count": audit_shared["exact_shared_tensor_count"],
        "shared_parameter_count": audit_shared["shared_parameter_count"],
        "max_parameter_copy_error": audit_shared["max_parameter_copy_error"],
        "unmatched_g1_gate_tensor_count": audit_shared["unmatched_g1_gate_variables"],
        "g1_coarse_gate_dense2_max_abs": gate_dense2_max,
        "initial_logits_max_abs_error": initial_logits_max_abs_error,
        "initialization_equivalence_pass": True,
    }

    # Load datasets
    train_imgs, train_lbls, train_indices, train_sha = load_fer_csv_split(train_csv_path, role="train")
    val_imgs, val_lbls, val_indices, val_sha = load_fer_csv_split(val_csv_path, role="validation")

    val_dataset = create_paired_dataset(
        val_imgs, val_lbls, batch_size=batch_size, seed=seed, shuffle=False, indices=val_indices
    )

    commit_sha = get_git_commit_sha(repo_root)
    cfg_sha = config.source_config_sha256

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

        def epoch_dataset_builder(epoch: int) -> tf.data.Dataset:
            return create_epoch_paired_training_dataset(
                train_imgs, train_lbls, train_indices,
                batch_size=batch_size, base_seed=seed, epoch=epoch, augment=True
            )

        run_res = _train_prebuilt_condition(
            model=models[cond],
            condition=cond,
            epoch_dataset_builder=epoch_dataset_builder,
            val_dataset=val_dataset,
            output_dir=cond_dir,
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
        )

        param_counts = count_parameters(models[cond])
        run_identity = {
            "condition": cond,
            "seed": seed,
            "source_commit": commit_sha,
            "reviewed_source_tag": reviewed_source_tag,
            "peeled_source_commit": source_lock_report["peeled_commit"],
            "config_sha256": cfg_sha,
            "train_sha256": train_sha,
            "val_sha256": val_sha,
            "parameter_count": param_counts["total_trainable_parameters"],
            "source_lock_report": source_lock_report,
        }
        (cond_dir / "run_identity.json").write_text(
            json.dumps(run_identity, indent=2), encoding="utf-8"
        )
        results[cond] = run_res

    return results
