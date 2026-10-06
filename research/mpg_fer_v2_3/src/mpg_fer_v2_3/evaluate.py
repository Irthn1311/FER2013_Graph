"""Evaluation utilities with single-pass raw and horizontal-flip TTA metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF


def _classification_metrics(
    targets: list[int], predictions: list[int], total_loss: float
) -> dict:
    y_true = np.asarray(targets, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataloader")
    return {
        "loss": float(total_loss / len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "per_class_f1": [
            float(x)
            for x in f1_score(
                y_true, y_pred, average=None, labels=np.arange(7), zero_division=0
            ).tolist()
        ],
        "precision": [
            float(x)
            for x in precision_score(
                y_true, y_pred, average=None, labels=np.arange(7), zero_division=0
            ).tolist()
        ],
        "recall": [
            float(x)
            for x in recall_score(
                y_true, y_pred, average=None, labels=np.arange(7), zero_division=0
            ).tolist()
        ],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(7)
        ).tolist(),
        "support": [int((y_true == label).sum()) for label in range(7)],
    }


@torch.no_grad()
def evaluate_raw_and_tta(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    use_amp: bool = True,
) -> dict[str, dict]:
    """Compute raw and logit-averaged flip-TTA metrics in one loader traversal.

    The image is horizontally flipped before the model extracts relational
    features. PrivateTest callers therefore make exactly one physical pass over
    the loader while obtaining both reporting views from the frozen weights.
    """
    model.eval()
    raw_preds: list[int] = []
    tta_preds: list[int] = []
    targets_all: list[int] = []
    raw_loss_total = 0.0
    tta_loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    amp_enabled = use_amp and torch.cuda.is_available()

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            raw_logits, _ = model(images)
            flipped_logits, _ = model(TF.hflip(images))
            tta_logits = 0.5 * (raw_logits + flipped_logits)
            raw_loss = criterion(raw_logits, targets)
            tta_loss = criterion(tta_logits, targets)

        batch_size = len(targets)
        raw_loss_total += raw_loss.item() * batch_size
        tta_loss_total += tta_loss.item() * batch_size
        raw_preds.extend(torch.argmax(raw_logits, dim=-1).cpu().tolist())
        tta_preds.extend(torch.argmax(tta_logits, dim=-1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())

    return {
        "raw": _classification_metrics(targets_all, raw_preds, raw_loss_total),
        "tta": _classification_metrics(targets_all, tta_preds, tta_loss_total),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    use_tta: bool = False,
    use_amp: bool = True,
) -> dict:
    """Compatibility wrapper selecting one view from the single-pass evaluator."""
    both = evaluate_raw_and_tta(model, dataloader, device, use_amp=use_amp)
    selected = dict(both["tta" if use_tta else "raw"])
    selected["use_tta"] = use_tta
    return selected
