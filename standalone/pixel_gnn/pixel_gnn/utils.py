"""Self-contained utility functions for Pixel GNN (Metrics, Seeds, Schedulers, Optimizer)."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
SPLIT_COUNTS = {"train": 28709, "val": 3589, "test": 3589}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def classification_metrics(labels, probabilities, num_classes: int = 7) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)

    acc = float(np.mean(predictions == labels)) if len(labels) > 0 else 0.0

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for l, p in zip(labels, predictions):
        if 0 <= l < num_classes and 0 <= p < num_classes:
            cm[l, p] += 1

    tp = np.diag(cm).astype(np.float64)
    support = np.sum(cm, axis=1).astype(np.float64)
    pred_count = np.sum(cm, axis=0).astype(np.float64)

    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        precision[c] = tp[c] / pred_count[c] if pred_count[c] > 0 else 0.0
        recall[c] = tp[c] / support[c] if support[c] > 0 else 0.0
        if precision[c] + recall[c] > 0:
            f1[c] = 2.0 * precision[c] * recall[c] / (precision[c] + recall[c])

    macro_f1 = float(np.mean(f1))
    total_support = np.sum(support)
    weighted_f1 = float(np.sum(f1 * support) / total_support) if total_support > 0 else 0.0

    clipped = np.clip(probabilities, 1e-12, 1.0)
    nll = float(-np.log(clipped[np.arange(labels.size), labels]).mean()) if len(labels) > 0 else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "nll": nll,
    }


class EarlyStopping:
    def __init__(self, min_epochs: int = 30, patience: int = 15):
        self.min_epochs = int(min_epochs)
        self.patience = int(patience)
        self.best_loss = float("inf")
        self.wait = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        val_loss = float(val_loss)
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1

        if epoch >= self.min_epochs and self.wait >= self.patience:
            return True
        return False


class ReduceLROnPlateau:
    def __init__(
        self,
        optimizer,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 5,
        threshold: float = 1e-4,
        min_lr: float = 3e-5,
    ):
        self.optimizer = optimizer
        self.factor = float(factor)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.min_lr = float(min_lr)
        self.best = float("inf")
        self.wait = 0

    def step(self, metric: float):
        val = float(metric)
        if val < self.best - self.threshold:
            self.best = val
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            current_lr = float(
                self.optimizer.learning_rate.numpy()
                if hasattr(self.optimizer.learning_rate, "numpy")
                else self.optimizer.learning_rate
            )
            new_lr = max(current_lr * self.factor, self.min_lr)
            if current_lr - new_lr > 1e-8:
                if hasattr(self.optimizer.learning_rate, "assign"):
                    self.optimizer.learning_rate.assign(new_lr)
                else:
                    self.optimizer.learning_rate = new_lr
                print(f"[SCHEDULER] Reduced learning rate: {current_lr:.6f} -> {new_lr:.6f}", flush=True)
            self.wait = 0


def build_optimizer(config: dict):
    training = config.get("training", {})
    opt_cfg = training.get("optimizer", {})
    lr = float(training.get("lr", 3e-4))
    weight_decay = float(training.get("weight_decay", 1e-3))
    beta1 = float(opt_cfg.get("beta1", 0.9))
    beta2 = float(opt_cfg.get("beta2", 0.999))
    eps = float(opt_cfg.get("epsilon", 1e-8))
    clipnorm = float(opt_cfg.get("clipnorm", 5.0))

    if hasattr(tf.keras.optimizers, "AdamW"):
        return tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=eps,
            clipnorm=clipnorm,
        )
    elif hasattr(tf.keras.optimizers.experimental, "AdamW"):
        return tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=eps,
            clipnorm=clipnorm,
        )
    else:
        return tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=eps,
            clipnorm=clipnorm,
        )
