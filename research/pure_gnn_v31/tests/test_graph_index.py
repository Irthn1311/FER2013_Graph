"""Unit tests for graph indexing, edge counts, and degree distributions in Pure-GNN v3.1."""

import pytest
import tensorflow as tf
from pure_gnn_v31.graph_index import (
    build_grid_8neighbor_graph,
    build_complete_coarse_graph,
)


def test_grid_48_graph_counts_and_degrees():
    graph = build_grid_8neighbor_graph(48, 48)
    assert int(graph["num_nodes"]) == 2304
    # Theoretical edge count:
    # 4 corners * 3 = 12
    # 4 * 46 edges * 5 = 920
    # 46 * 46 interior * 8 = 16928
    # Total = 17860
    assert int(graph["num_edges"]) == 17860
    assert graph["src_indices"].shape == (17860,)
    assert graph["dst_indices"].shape == (17860,)
    assert graph["delta_x"].shape == (17860, 1)
    assert graph["delta_y"].shape == (17860, 1)
    assert graph["degrees"].shape == (2304, 1)

    # Degree checks
    degrees = graph["degrees"].numpy().ravel()
    # Corner (0, 0)
    assert degrees[0] == 3.0
    # Top edge (0, 1)
    assert degrees[1] == 5.0
    # Interior (1, 1)
    assert degrees[1 * 48 + 1] == 8.0

    # Count of nodes with each degree
    deg_3 = (degrees == 3.0).sum()
    deg_5 = (degrees == 5.0).sum()
    deg_8 = (degrees == 8.0).sum()
    assert deg_3 == 4
    assert deg_5 == 4 * 46  # 184
    assert deg_8 == 46 * 46  # 2116


def test_grid_24_and_12_graph_counts():
    g24 = build_grid_8neighbor_graph(24, 24)
    assert int(g24["num_nodes"]) == 576
    # Corners: 4 * 3 = 12
    # Borders: 4 * 22 * 5 = 440
    # Interior: 22 * 22 * 8 = 3872
    # Total = 4324
    assert int(g24["num_edges"]) == 4324

    g12 = build_grid_8neighbor_graph(12, 12)
    assert int(g12["num_nodes"]) == 144
    # Corners: 4 * 3 = 12
    # Borders: 4 * 10 * 5 = 200
    # Interior: 10 * 10 * 8 = 800
    # Total = 1012
    assert int(g12["num_edges"]) == 1012


def test_complete_coarse_graph_contract():
    coarse = build_complete_coarse_graph(6)
    assert int(coarse["num_nodes"]) == 36
    # 36 * 35 = 1260 directed non-self edges
    assert int(coarse["num_edges"]) == 1260
    assert int(coarse["edges_per_node"]) == 35
    assert coarse["src_indices"].shape == (1260,)
    assert coarse["dst_indices"].shape == (1260,)
    assert coarse["p_ij"].shape == (1260, 3)

    # Assert no self-loops: src != dst for all edges
    src = coarse["src_indices"].numpy()
    dst = coarse["dst_indices"].numpy()
    assert not (src == dst).any(), "Self-loops found in coarse complete graph!"

    # Assert exactly 35 edges per receiver
    for i in range(36):
        count_i = (dst == i).sum()
        assert count_i == 35, f"Node {i} has {count_i} incoming edges, expected 35"
