"""Patch tokenization utilities for D16 fallback encoders."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=8)
def grid_edge_index(grid_size: int, connectivity: int = 4) -> torch.Tensor:
    edges = []
    for y in range(int(grid_size)):
        for x in range(int(grid_size)):
            src = y * int(grid_size) + x
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if int(connectivity) == 8:
                offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < int(grid_size) and 0 <= nx < int(grid_size):
                    edges.append((src, ny * int(grid_size) + nx))
    if not edges:
        edges = [(0, 0)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def image_to_patch_tokens(image_48: torch.Tensor, patch_size: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Return patch stats tokens and normalized patch coordinates.

    Token features are intentionally landmark-free: intensity statistics,
    gradient statistics, and patch position.
    """
    if image_48.dim() == 2:
        image_48 = image_48.unsqueeze(0)
    if image_48.dim() != 3:
        raise ValueError(f"image_48 must be [B,48,48] or [48,48], got {tuple(image_48.shape)}")
    patch_size = int(patch_size)
    if 48 % patch_size != 0:
        raise ValueError(f"patch_size must divide 48, got {patch_size}")
    image = image_48.float()
    if float(image.detach().max().cpu().item()) > 1.5:
        image = image / 255.0
    bsz = image.size(0)
    grid = 48 // patch_size
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(bsz, grid * grid, patch_size * patch_size)
    mean = patches.mean(dim=-1, keepdim=True)
    std = patches.std(dim=-1, unbiased=False, keepdim=True)
    minv = patches.min(dim=-1, keepdim=True).values
    maxv = patches.max(dim=-1, keepdim=True).values

    gx = F.pad(image[:, :, 1:] - image[:, :, :-1], (0, 1, 0, 0))
    gy = F.pad(image[:, 1:, :] - image[:, :-1, :], (0, 0, 0, 1))
    grad = torch.sqrt(gx.square() + gy.square() + 1e-8)
    grad_patches = grad.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    grad_patches = grad_patches.contiguous().view(bsz, grid * grid, patch_size * patch_size)
    grad_mean = grad_patches.mean(dim=-1, keepdim=True)
    grad_std = grad_patches.std(dim=-1, unbiased=False, keepdim=True)

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, grid, device=image.device, dtype=image.dtype),
        torch.linspace(-1.0, 1.0, grid, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    pos = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    pos_batch = pos.unsqueeze(0).expand(bsz, -1, -1)
    tokens = torch.cat([mean, std, minv, maxv, grad_mean, grad_std, pos_batch], dim=-1)
    return tokens, pos_batch
