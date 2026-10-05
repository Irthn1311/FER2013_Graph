# Phase 0: Source Audit for Upstream Representation & Optimization Audit

## 1. Checkpoint Verification
- **FULL Checkpoint SHA256:** `23dbe9b1453fdc7e5dca81ca2e9bd26f361f5b1fe3d7ffe803c65546b22d162e`
- **NPF Checkpoint SHA256:**  `f301895cd174f8adf998d7f622279510e45bf210e2cc52ebc25376db49e5f972`
- **Strict Loading:** PASS (0 missing keys, 0 unexpected keys for both models)

## 2. Layer Extraction Points

| Layer | Name | Source Extraction Location | Expected Shape |
|---|---|---|---|
| **P** | Pixel GNN Output | `model.pixel_gnn` forward loop completion | `[B, 2304, 96]` |
| **M0** | Composer Output | `model.motif_composer` forward output before GNN | `[B, 49, 192]` |
| **M1** | Motif Block 1 | `model.motif_gnn[0]` output | `[B, 49, 192]` |
| **M2** | Motif Block 2 | `model.motif_gnn[1]` output | `[B, 49, 192]` |
| **M3** | Motif Block 3 | `model.motif_gnn[2]` output | `[B, 49, 192]` |
| **M4** | Motif Block 4 | `model.motif_gnn[3]` output | `[B, 49, 192]` |
| **M5** | Motif Block 5 | `model.motif_gnn[4]` output (final motif tensor) | `[B, 49, 192]` |
