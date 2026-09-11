"""Metric definitions and evaluation utilities for FER2013 in Pure-GNN v3.1."""

from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf


EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes overall accuracy."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> float:
    """Computes macro-averaged F1 score across all classes without external dependencies."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    f1_list = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_list.append(f1)
    return float(np.mean(f1_list))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> np.ndarray:
    """Computes confusion matrix of shape (num_classes, num_classes)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true.ravel(), y_pred.ravel()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


class CrossEntropyWithLabelSmoothing(tf.keras.losses.Loss):
    """Categorical cross entropy loss with label smoothing."""

    def __init__(self, label_smoothing: float = 0.05, num_classes: int = 7, name: str = "ce_smooth", **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def call(self, y_true: tf.Tensor, y_pred_logits: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.int32)
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[-1] == 1):
            y_one_hot = tf.one_hot(tf.squeeze(y_true), depth=self.num_classes)
        else:
            y_one_hot = tf.cast(y_true, tf.float32)

        if self.label_smoothing > 0:
            smooth_pos = 1.0 - self.label_smoothing
            smooth_neg = self.label_smoothing / float(self.num_classes)
            y_one_hot = y_one_hot * smooth_pos + smooth_neg

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=y_pred_logits)
        return tf.reduce_mean(loss)
