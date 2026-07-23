"""Strict reversible state-dict compatibility for the mechanical extraction."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _validate(source: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    missing = sorted(set(expected) - set(source))
    unexpected = sorted(set(source) - set(expected))
    mismatched = sorted(
        key for key in set(source) & set(expected)
        if tuple(source[key].shape) != tuple(expected[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"State-dict mismatch: missing={missing}, unexpected={unexpected}, shape_mismatches={mismatched}"
        )
    return {key: source[key] for key in expected}


def convert_parent_state_dict(source, standalone_template):
    return _validate(source, standalone_template)


def convert_standalone_state_dict_to_parent(source, parent_template):
    return _validate(source, parent_template)
