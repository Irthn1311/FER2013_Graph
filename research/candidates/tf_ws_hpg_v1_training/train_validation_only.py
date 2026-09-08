"""Future registered WS-HPG seed-42 train/validation lifecycle.

This module prepares the lifecycle only. Importing it performs no data access or
training, and its CLI requires explicit train/validation/cache paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from research.candidates.tf_ws_hpg_v1_weak_support.model import (
    build_ws_hpg_v1_weak_support,
)
from research.candidates.tf_ws_hpg_v1_training.data import (
    TRAIN_SAMPLES,
    VALIDATION_SAMPLES,
    build_dataset,
    load_fer_csv,
    load_support_split,
    reject_test_path,
)


STATUS = "WS_HPG_V1_TRAINING_PREPARATION_ONLY"
SEED = 42
EXPECTED_MODEL_SHA256 = "177a782cd8d5c2178303c44d120dcdd22b0a2108a0b720c5091a19f3d7cbffe3"
EXPECTED_SUPPORT_SHA256 = "b6ed2ddd20a4e82824208929709ff2d6bcb1c5557144d778ed0157fd4768aeee"
EXPECTED_IDENTITY = {"parameters": 707_213, "trainable_variables": 118, "keras_variables": 138}
ACCEPTED_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "tf_ws_hpg_v1_weak_support"
ACCEPTED_MODEL_PATH = ACCEPTED_PACKAGE_ROOT / "model.py"
ACCEPTED_SUPPORT_PATH = ACCEPTED_PACKAGE_ROOT / "support.py"
LAP_COMPARATOR = {
    "validation_accuracy": 0.6319308999721371,
    "validation_macro_f1": 0.5938407974340496,
    "clean_train_macro_f1": 0.8207655611897143,
    "macro_gap_pp": 22.69247637556647,
}
CF_COMPARATOR = {
    "validation_accuracy": 0.5806631373641683,
    "validation_macro_f1": 0.5238975323290902,
    "clean_train_accuracy": 0.614894284022432,
    "clean_train_macro_f1": 0.5614369765915708,
}
TRAINING_CONFIG = {
    "seed": 42,
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 5e-4,
    "global_clipnorm": 1.0,
    "batch_size": 64,
    "max_epochs": 100,
    "warmup_epochs": 5,
    "cosine_final_learning_rate": 1e-6,
    "training_label_smoothing": 0.05,
    "validation_every_epochs": 1,
    "checkpoint": "earliest_strict_max_val_accuracy",
    "early_stopping_monitor": "val_loss",
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.0,
    "mixed_precision": False,
    "xla": False,
    "mirrored_strategy": False,
    "support_dropout": False,
}


class TrainingPreparationError(RuntimeError):
    pass


@tf.keras.utils.register_keras_serializable(package="fer2013_graph_research")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, steps_per_epoch, initial=3e-4, final=1e-6, warmup_epochs=5, max_epochs=100):
        super().__init__()
        self.steps_per_epoch = int(steps_per_epoch)
        self.initial = float(initial)
        self.final = float(final)
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = float(self.steps_per_epoch * self.warmup_epochs)
        total_steps = float(self.steps_per_epoch * self.max_epochs)
        warmup = self.initial * step / warmup_steps
        progress = tf.clip_by_value((step - warmup_steps) / (total_steps - warmup_steps), 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * progress))
        decayed = self.final + (self.initial - self.final) * cosine
        return tf.where(step < warmup_steps, warmup, decayed)

    def get_config(self):
        return {"steps_per_epoch": self.steps_per_epoch, "initial": self.initial, "final": self.final, "warmup_epochs": self.warmup_epochs, "max_epochs": self.max_epochs}


def training_loss(labels, logits):
    labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    targets = (1.0 - 0.05) * tf.one_hot(labels, 7, dtype=tf.float32) + 0.05 / 7.0
    return tf.keras.losses.categorical_crossentropy(targets, tf.cast(logits, tf.float32), from_logits=True)


def hard_evaluation_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, tf.cast(logits, tf.float32), from_logits=True)


def build_optimizer(steps_per_epoch):
    return tf.keras.optimizers.AdamW(
        learning_rate=WarmupCosine(steps_per_epoch),
        weight_decay=5e-4,
        global_clipnorm=1.0,
    )


def validate_model_identity(candidate):
    observed = {
        "parameters": int(candidate.count_params()),
        "trainable_variables": len(candidate.trainable_variables),
        "keras_variables": len(candidate.variables),
    }
    if observed != EXPECTED_IDENTITY:
        raise TrainingPreparationError(f"Accepted WS architecture identity drift: {observed}")
    return observed


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_accepted_source_hashes(
    model_path=ACCEPTED_MODEL_PATH, support_path=ACCEPTED_SUPPORT_PATH
):
    """Fail before FER I/O when either accepted scientific source has drifted."""

    actual = {
        "model.py": _file_sha256(model_path),
        "support.py": _file_sha256(support_path),
    }
    expected = {
        "model.py": EXPECTED_MODEL_SHA256,
        "support.py": EXPECTED_SUPPORT_SHA256,
    }
    if actual != expected:
        raise TrainingPreparationError(
            f"Accepted WS scientific source identity drift: {actual}"
        )
    return actual


def earliest_strict_max_epoch(values: Sequence[float]) -> int:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise TrainingPreparationError("Finite validation accuracy history is required")
    return max(range(len(values)), key=lambda index: (float(values[index]), -index))


class EarliestStrictMaximumCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, output_root):
        super().__init__()
        self.output_root = Path(output_root)
        self.best = -float("inf")
        self.selected_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        value = None if logs is None else logs.get("val_accuracy")
        if value is None or not math.isfinite(float(value)):
            raise TrainingPreparationError("Finite val_accuracy is required")
        if float(value) > self.best:
            self.best = float(value)
            self.selected_epoch = int(epoch)
            path = self.output_root / "checkpoints" / "best_val_accuracy.keras"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)


def classify_outcome(
    *, validation_accuracy, validation_macro_f1, clean_train_macro_f1,
    support_dependency_accuracy_pp, support_dependency_macro_pp,
    capacity_limited=False,
):
    values = [validation_accuracy, validation_macro_f1, clean_train_macro_f1, support_dependency_accuracy_pp, support_dependency_macro_pp]
    if not all(math.isfinite(float(value)) for value in values):
        return "WS_HPG_V1_INCONCLUSIVE"
    macro_gap = round(100.0 * (clean_train_macro_f1 - validation_macro_f1), 12)
    improvement = round(LAP_COMPARATOR["macro_gap_pp"] - macro_gap, 12)
    if validation_accuracy >= 0.6500 and validation_macro_f1 >= 0.6200 and macro_gap <= 10.0 and support_dependency_accuracy_pp <= 3.0 and support_dependency_macro_pp <= 3.0:
        return "WS_HPG_V1_STRETCH_REPLACEMENT_CANDIDATE"
    if validation_accuracy >= 0.6369308999721371 and validation_macro_f1 >= 0.6038407974340496 and macro_gap <= 15.0 and support_dependency_accuracy_pp <= 5.0 and support_dependency_macro_pp <= 5.0:
        return "WS_HPG_V1_REPLACEMENT_CANDIDATE"
    if validation_accuracy >= 0.6269308999721371 and validation_macro_f1 >= 0.5888407974340496 and improvement >= 7.5:
        return "WS_HPG_V1_COMPETITIVE_GENERALIZATION_SIGNAL_NOT_REPLACE"
    if validation_accuracy > CF_COMPARATOR["validation_accuracy"] and validation_macro_f1 > CF_COMPARATOR["validation_macro_f1"]:
        return "WS_HPG_V1_BEATS_CF_ONLY_NOT_REPLACE"
    if capacity_limited or validation_accuracy <= CF_COMPARATOR["validation_accuracy"] or validation_macro_f1 <= CF_COMPARATOR["validation_macro_f1"]:
        return "WS_HPG_V1_UNDERFIT_OR_REGRESSION"
    return "WS_HPG_V1_INCONCLUSIVE"


def _weights_sha256(candidate):
    digest = hashlib.sha256()
    for variable in candidate.variables:
        value = np.asarray(variable.numpy())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def evaluate(candidate, dataset, *, support_override="normal"):
    labels, predictions, losses = [], [], []
    for inputs, batch_labels in dataset:
        logits = candidate(inputs, training=False, support_override=support_override)
        losses.extend(hard_evaluation_loss(batch_labels, logits).numpy().tolist())
        labels.extend(tf.reshape(batch_labels, [-1]).numpy().tolist())
        predictions.extend(tf.argmax(logits, axis=-1).numpy().tolist())
    return {
        "sample_count": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "loss": float(np.mean(losses)),
    }


def _atomic_json(path, payload: Mapping):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def run_registered_lifecycle(
    train_dataset, validation_dataset, clean_train_dataset, output_root,
    support_diagnostics=None,
):
    """Run the future preregistered lifecycle; called only after later authorization."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    candidate = build_ws_hpg_v1_weak_support()
    validate_model_identity(candidate)
    candidate.compile(
        optimizer=build_optimizer(math.ceil(TRAIN_SAMPLES / 64)),
        loss=training_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    checkpoint = EarliestStrictMaximumCheckpoint(output_root)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, min_delta=0.0, restore_best_weights=False)
    history = candidate.fit(train_dataset, validation_data=validation_dataset, validation_freq=1, epochs=100, callbacks=[checkpoint, early_stop])
    selected_path = output_root / "checkpoints" / "best_val_accuracy.keras"
    selected = tf.keras.models.load_model(selected_path, compile=False)
    before = _weights_sha256(selected)
    clean = evaluate(selected, clean_train_dataset, support_override="normal")
    normal = evaluate(selected, validation_dataset, support_override="normal")
    no_prior = evaluate(selected, validation_dataset, support_override="ones")
    after = _weights_sha256(selected)
    if before != after:
        raise TrainingPreparationError("Selected checkpoint changed during diagnostics")
    dependency_accuracy = 100.0 * (normal["accuracy"] - no_prior["accuracy"])
    dependency_macro = 100.0 * (normal["macro_f1"] - no_prior["macro_f1"])
    decision = classify_outcome(
        validation_accuracy=normal["accuracy"], validation_macro_f1=normal["macro_f1"],
        clean_train_macro_f1=clean["macro_f1"], support_dependency_accuracy_pp=dependency_accuracy,
        support_dependency_macro_pp=dependency_macro,
    )
    result = {
        "status": "COMPLETE",
        "preparation_contract": STATUS,
        "selected_epoch_zero_based": checkpoint.selected_epoch,
        "epochs_completed": len(history.history["loss"]),
        "clean_train": clean,
        "validation_normal_support": normal,
        "validation_all_ones_support": no_prior,
        "accuracy_gap_pp": 100.0 * (clean["accuracy"] - normal["accuracy"]),
        "macro_gap_pp": 100.0 * (clean["macro_f1"] - normal["macro_f1"]),
        "support_dependency_accuracy_pp": dependency_accuracy,
        "support_dependency_macro_pp": dependency_macro,
        "selected_weights_sha256_before_after": [before, after],
        "decision": decision,
        "support_diagnostics": dict(support_diagnostics or {}),
        "test_access": False,
    }
    _atomic_json(output_root / "validation_only_result.json", result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Future WS-HPG registered train/validation lifecycle")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    verify_accepted_source_hashes()
    train_csv, val_csv = reject_test_path(args.train_csv), reject_test_path(args.val_csv)
    prior_root = reject_test_path(args.prior_root)
    train_images, train_labels = load_fer_csv(train_csv, TRAIN_SAMPLES)
    val_images, val_labels = load_fer_csv(val_csv, VALIDATION_SAMPLES)
    train_support, train_detected, train_coverage = load_support_split(prior_root, "train", train_images, train_labels)
    val_support, val_detected, val_coverage = load_support_split(prior_root, "val", val_images, val_labels)
    train = build_dataset(train_images, train_support, train_labels, training=True)
    clean_train = build_dataset(train_images, train_support, train_labels, training=False)
    validation = build_dataset(val_images, val_support, val_labels, training=False)
    support_diagnostics = {
        "train": {
            "coverage_mean": float(np.mean(train_coverage)),
            "coverage_std": float(np.std(train_coverage)),
            "detector_failure_count": int(np.count_nonzero(~train_detected)),
            "detector_failure_rate": float(np.mean(~train_detected)),
        },
        "validation": {
            "coverage_mean": float(np.mean(val_coverage)),
            "coverage_std": float(np.std(val_coverage)),
            "detector_failure_count": int(np.count_nonzero(~val_detected)),
            "detector_failure_rate": float(np.mean(~val_detected)),
        },
    }
    run_registered_lifecycle(train, validation, clean_train, args.output_root, support_diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
