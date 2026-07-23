"""Locked optimizer construction."""

import torch


def build_optimizer(model, cfg):
    training = cfg["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
    )
