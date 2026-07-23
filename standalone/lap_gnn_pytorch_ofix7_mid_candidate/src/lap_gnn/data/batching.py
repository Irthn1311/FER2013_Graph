"""Variable-size graph batching exported from the locked runtime."""

from lap_gnn.data.graph_builder import D16Batch, D16GraphData, collate_d16_graphs

__all__ = ["D16Batch", "D16GraphData", "collate_d16_graphs"]
