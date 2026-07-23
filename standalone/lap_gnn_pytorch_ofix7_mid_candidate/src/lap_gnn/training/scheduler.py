"""Locked ReduceLROnPlateau construction."""

import torch


def build_scheduler(optimizer, cfg):
    scheduler = cfg["training"]["scheduler"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler["mode"]),
        factor=float(scheduler["factor"]),
        patience=int(scheduler["patience"]),
        threshold=float(scheduler["threshold"]),
        min_lr=float(scheduler["min_lr"]),
    )
