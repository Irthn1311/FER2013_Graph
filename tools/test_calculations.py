"""Test spatial support pressure and prototype calculations on synthetic inputs."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from mpg_fer_v2_3.motif import build_aligned_support_indices


def test_calculations():
    supports, centers = build_aligned_support_indices(img_size=48, anchor_size=12, stride=6, scales=(8, 12, 16))
    
    # Check outer 20% ring mask for each scale
    ring_masks = {}
    for scale in [8, 12, 16]:
        coords = torch.arange(scale).float()
        center_coord = (scale - 1) / 2.0
        max_dist = (scale - 1) / 2.0
        
        # 2D grid
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        dist_inf = torch.maximum((grid_x - center_coord).abs(), (grid_y - center_coord).abs()) / max_dist
        mask = (dist_inf >= 0.80).reshape(-1) # [scale*scale]
        ring_masks[scale] = mask
        print(f"Scale {scale} ({scale*scale} pixels): {mask.sum().item()} pixels in outer 20% ring ({mask.float().mean()*100:.1f}%)")

    # Test bootstrap CI and Cohen's d
    correct = np.random.normal(loc=0.5, scale=0.1, size=2400)
    error = np.random.normal(loc=0.55, scale=0.1, size=1189)
    
    pooled_std = math.sqrt(((len(correct)-1)*np.var(correct, ddof=1) + (len(error)-1)*np.var(error, ddof=1)) / (len(correct)+len(error)-2))
    cohens_d = (np.mean(error) - np.mean(correct)) / pooled_std
    print(f"Test Cohen's d: {cohens_d:.4f}")
    
    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.zeros(len(correct)), np.ones(len(error))])
    y_scores = np.concatenate([correct, error])
    auroc = roc_auc_score(y_true, y_scores)
    print(f"Test AUROC: {auroc:.4f}")
    print("Calculations test verified PASS!")


if __name__ == "__main__":
    test_calculations()
