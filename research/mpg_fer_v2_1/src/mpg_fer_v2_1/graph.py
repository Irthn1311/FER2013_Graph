"""Fixed 8-neighbor pixel graph topology and relative edge feature construction."""

from __future__ import annotations

import torch
import torch.nn as nn


class PixelGraphTopology(nn.Module):
    """Builds and caches the static 8-neighbor directed graph over a 48x48 pixel grid.

    Neighborhood layout for receiver i at (r_i, c_i):
      0: NW (r-1, c-1)
      1: N  (r-1, c)
      2: NE (r-1, c+1)
      3: W  (r,   c-1)
      4: E  (r,   c+1)
      5: SW (r+1, c-1)
      6: S  (r+1, c)
      7: SE (r+1, c+1)

    Edges are directed receiver-sender relations: j -> i (sender j contributing to receiver i).
    Edge features (5D):
      [dx, dy, distance, delta_I, abs_delta_I]
      where dx = x_j - x_i, dy = y_j - y_i, delta_I = I_j - I_i.
    """

    def __init__(self, img_size: int = 48) -> None:
        super().__init__()
        self.img_size = img_size
        self.num_pixels = img_size * img_size
        self.num_neighbors = 8

        # 8 neighbor relative row, col offsets: (dr, dc)
        # where r_j = r_i + dr, c_j = c_i + dc
        self.neighbor_offsets = [
            (-1, -1),  # 0: NW
            (-1, 0),  # 1: N
            (-1, 1),  # 2: NE
            (0, -1),  # 3: W
            (0, 1),  # 4: E
            (1, -1),  # 5: SW
            (1, 0),  # 6: S
            (1, 1),  # 7: SE
        ]

        neighbor_idx = torch.zeros(
            (self.num_pixels, self.num_neighbors), dtype=torch.long
        )
        neighbor_mask = torch.zeros(
            (self.num_pixels, self.num_neighbors), dtype=torch.bool
        )
        geom_edges = torch.zeros(
            (self.num_pixels, self.num_neighbors, 3), dtype=torch.float32
        )

        coords = torch.linspace(-1.0, 1.0, img_size)

        for r_i in range(img_size):
            for c_i in range(img_size):
                i = r_i * img_size + c_i
                x_i = coords[c_i].item()
                y_i = coords[r_i].item()

                for k, (dr, dc) in enumerate(self.neighbor_offsets):
                    r_j = r_i + dr
                    c_j = c_i + dc

                    if 0 <= r_j < img_size and 0 <= c_j < img_size:
                        j = r_j * img_size + c_j
                        neighbor_idx[i, k] = j
                        neighbor_mask[i, k] = True

                        x_j = coords[c_j].item()
                        y_j = coords[r_j].item()

                        dx = x_j - x_i
                        dy = y_j - y_i
                        dist = (dx**2 + dy**2) ** 0.5

                        geom_edges[i, k, 0] = dx
                        geom_edges[i, k, 1] = dy
                        geom_edges[i, k, 2] = dist
                    else:
                        neighbor_idx[i, k] = (
                            i  # self-loop pad to avoid out-of-bounds gather
                        )
                        neighbor_mask[i, k] = False
                        geom_edges[i, k] = 0.0

        self.register_buffer("neighbor_idx", neighbor_idx)  # [2304, 8]
        self.register_buffer("neighbor_mask", neighbor_mask)  # [2304, 8]
        self.register_buffer("geom_edges", geom_edges)  # [2304, 8, 3]

        # Pre-expanded indices for batched gathering [2304, 8, 1]
        I_gather_idx = neighbor_idx.unsqueeze(-1)
        self.register_buffer("I_gather_idx", I_gather_idx)

    def compute_edge_features(self, pixel_intensity: torch.Tensor) -> torch.Tensor:
        """Compute the 5D edge features for directed edges j -> i.

        Args:
            pixel_intensity: [B, 2304, 1] normalized pixel intensities.

        Returns:
            edge_features: [B, 2304, 8, 5]
              [dx, dy, distance, delta_I, abs_delta_I]
        """
        B, N, _ = pixel_intensity.shape
        assert N == self.num_pixels

        # Expand static geometry: [1, 2304, 8, 3] -> [B, 2304, 8, 3]
        geom = self.geom_edges.unsqueeze(0).expand(B, -1, -1, -1)

        # Efficient flatten-gather across batch
        # pixel_intensity: [B, 2304, 1]
        # I_j for receiver i: gather sender neighbor index
        I_j = pixel_intensity[:, self.neighbor_idx, :]  # [B, 2304, 8, 1]
        I_i = pixel_intensity.unsqueeze(2)  # [B, 2304, 1, 1]

        delta_I = I_j - I_i  # [B, 2304, 8, 1]
        abs_delta_I = torch.abs(delta_I)

        # Edge features: [dx, dy, dist, delta_I, abs_delta_I] -> [B, 2304, 8, 5]
        edge_feats = torch.cat([geom, delta_I, abs_delta_I], dim=-1)

        # Zero out invalid padded boundary edges
        mask = self.neighbor_mask.unsqueeze(0).unsqueeze(-1)  # [1, 2304, 8, 1]
        edge_feats = edge_feats * mask.float()
        return edge_feats
