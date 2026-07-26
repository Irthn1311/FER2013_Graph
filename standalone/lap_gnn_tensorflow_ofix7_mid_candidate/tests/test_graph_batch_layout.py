from _helpers import loaded


def test_graph_batch_layout():
    _, batch = loaded()
    assert batch["node_features"].shape[1] == 37
    assert batch["edge_index"].shape[0] == 2
    assert batch["edge_features"].shape[1] == 8
    assert batch["labels"].shape[0] == 8

