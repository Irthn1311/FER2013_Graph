"""Small freeze/unfreeze helpers for staged training scripts."""

from __future__ import annotations

from typing import Iterable

import torch


def set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    """Set requires_grad on all parameters in a module."""

    for param in module.parameters():
        param.requires_grad = bool(flag)


def freeze_by_keywords(model: torch.nn.Module, keywords: Iterable[str]) -> list[str]:
    """Freeze parameters whose qualified name contains any keyword."""

    return _set_by_keywords(model, keywords, requires_grad=False)


def unfreeze_by_keywords(model: torch.nn.Module, keywords: Iterable[str]) -> list[str]:
    """Unfreeze parameters whose qualified name contains any keyword."""

    return _set_by_keywords(model, keywords, requires_grad=True)


def count_trainable_params(model: torch.nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}


def trainable_parameter_names(model: torch.nn.Module, limit: int = 80) -> list[str]:
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    if limit > 0 and len(names) > int(limit):
        return names[: int(limit)] + [f"... ({len(names) - int(limit)} more)"]
    return names


def _set_by_keywords(model: torch.nn.Module, keywords: Iterable[str], requires_grad: bool) -> list[str]:
    keys = [str(key) for key in (keywords or []) if str(key)]
    changed: list[str] = []
    if not keys:
        return changed
    for name, param in model.named_parameters():
        if any(key in name for key in keys):
            param.requires_grad = bool(requires_grad)
            changed.append(name)
    return changed
