"""Continuous H1-inspired 32D relational pixel feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelFeatureExtractor(nn.Module):
    """Extracts exactly 32 raw continuous relational features per pixel on a 48x48 image.

    Feature composition per pixel:
      - 1  normalized intensity I in [0, 1]
      - 2  normalized coordinates x, y in [-1, 1]
      - 24 sigma-normalized center-relative differences from 5x5 neighborhood
      - 1  log_sigma: log(sqrt(var_local + eps^2) + eps)
      - 4  gx, gy, gradient_magnitude, laplacian
      ------------------------------------------------------------------------
      Total: exactly 32 dimensions.

    Uses reflection padding for the 5x5 spatial support to preserve all 2304 pixels.
    """

    def __init__(self, img_size: int = 48, eps: float = 1e-5) -> None:
        super().__init__()
        self.img_size = img_size
        self.num_pixels = img_size * img_size
        self.eps = eps

        # Precompute normalized spatial coordinate grids in [-1, 1]
        coords = torch.linspace(-1.0, 1.0, img_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        # [1, 1, 48, 48]
        self.register_buffer("grid_x", grid_x.unsqueeze(0).unsqueeze(0).clone())
        self.register_buffer("grid_y", grid_y.unsqueeze(0).unsqueeze(0).clone())

        # Sobel gradient filters
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / 8.0
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / 4.0

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

        # 5x5 local mean filter
        ones_5x5 = torch.ones((1, 1, 5, 5), dtype=torch.float32) / 25.0
        self.register_buffer("mean_5x5", ones_5x5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: [B, 1, 48, 48] images normalized to [0, 1].

        Returns:
            features: [B, 2304, 32] continuous relational pixel features.
        """
        B, C, H, W = x.shape
        assert C == 1 and H == self.img_size and W == self.img_size

        # 1. Normalized intensity I: [B, 1, 48, 48]
        feat_I = x

        # 2. Normalized coordinates (x, y): [B, 2, 48, 48]
        feat_coords = torch.cat(
            [self.grid_x.expand(B, -1, -1, -1), self.grid_y.expand(B, -1, -1, -1)],
            dim=1,
        )

        # 3. 5x5 neighborhood unfolding with reflection padding
        # Pad by 2 on all sides to keep exact (48, 48) grid
        x_pad = F.pad(x, (2, 2, 2, 2), mode="reflect")
        # unfold: [B, 25, 48, 48]
        patches = F.unfold(x_pad, kernel_size=5, padding=0, stride=1).reshape(
            B, 25, H, W
        )

        # Center pixel is at index 12 (row 2, col 2 in 5x5)
        center_pixel = patches[:, 12:13, :, :]  # [B, 1, H, W]

        # 24 non-center neighbor pixels
        indices_24 = [i for i in range(25) if i != 12]
        neighbors_24 = patches[:, indices_24, :, :]  # [B, 24, H, W]

        # Local variance & sigma over 5x5 window
        mean_local = F.conv2d(x_pad, self.mean_5x5)
        mean_sq_local = F.conv2d(x_pad**2, self.mean_5x5)
        var_local = F.relu(mean_sq_local - mean_local**2)
        sigma_local = torch.sqrt(var_local + (self.eps**2))  # [B, 1, H, W]

        # 24 sigma-normalized center-relative differences:
        # d_k = (I_neighbor - I_center) / sqrt(var_local + eps^2)
        diffs_24 = (neighbors_24 - center_pixel) / sigma_local  # [B, 24, H, W]

        # 4. log_sigma: log(sqrt(var_local + eps^2) + eps)
        log_sigma = torch.log(sigma_local + self.eps)  # [B, 1, H, W]

        # 5. Gradients & Laplacian with reflection padding
        x_pad3 = F.pad(x, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(x_pad3, self.sobel_x)
        gy = F.conv2d(x_pad3, self.sobel_y)
        grad_mag = torch.sqrt(gx**2 + gy**2 + (self.eps**2))
        lap = F.conv2d(x_pad3, self.laplacian)
        # [B, 4, H, W]
        derivatives = torch.cat([gx, gy, grad_mag, lap], dim=1)

        # Concatenate all 32 channels:
        # 1 (I) + 2 (coords) + 24 (diffs) + 1 (log_sigma) + 4 (derivatives) = 32
        all_feats = torch.cat(
            [feat_I, feat_coords, diffs_24, log_sigma, derivatives], dim=1
        )  # [B, 32, 48, 48]

        # Reshape to [B, 2304, 32]
        # Channels-last: [B, 48, 48, 32] -> [B, 2304, 32]
        all_feats = all_feats.permute(0, 2, 3, 1).reshape(B, self.num_pixels, 32)
        return all_feats
