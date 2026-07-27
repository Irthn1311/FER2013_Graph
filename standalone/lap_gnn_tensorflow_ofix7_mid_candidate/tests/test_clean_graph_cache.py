from io import BytesIO
import json

import numpy as np

from lap_gnn_tf.data.clean_graph_cache import (
    CACHE_SCHEMA_VERSION,
    CleanGraphCache,
    clean_graph_config_sha256,
)


def _record_bytes() -> bytes:
    stream = BytesIO()
    np.savez_compressed(
        stream,
        x=np.arange(74, dtype=np.float32).reshape(2, 37),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        edge_attr=np.arange(16, dtype=np.float32).reshape(2, 8),
        pos=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        y=np.asarray(3, dtype=np.int64),
        sample_index=np.asarray(17, dtype=np.int64),
        part_soft=np.zeros((2, 13), dtype=np.float32),
        face_mask=np.ones((2,), dtype=np.float32),
        valid_part_mask=np.ones((13,), dtype=np.float32),
        valid_anchor_mask=np.ones((12,), dtype=np.float32),
        detected=np.asarray(True, dtype=np.bool_),
        landmark_missing_flag=np.asarray(0, dtype=np.int64),
        image_48=np.zeros((48, 48), dtype=np.float32),
        anchor_mask=np.asarray([False, True], dtype=np.bool_),
    )
    return stream.getvalue()


def test_record_cache_roundtrip_and_signature(tmp_path):
    graph_config = {
        "graph_mode": "face_plus_context",
        "prior_corruption": {"enabled": True, "probability": 0.3},
    }
    signature = clean_graph_config_sha256(graph_config)
    split = tmp_path / "train"
    split.mkdir()
    payload = _record_bytes()
    (split / "shard-00000.bin").write_bytes(payload)
    (split / "index.json").write_text(json.dumps({
        "schema_version": CACHE_SCHEMA_VERSION,
        "split": "train",
        "sample_count": 1,
        "shard_size": 1,
        "shards": [{
            "path": "shard-00000.bin",
            "start": 0,
            "end": 1,
            "samples": 1,
            "offsets": [0, len(payload)],
        }],
    }), encoding="utf-8")
    (tmp_path / "CACHE_COMPLETE.json").write_text(json.dumps({
        "schema_version": CACHE_SCHEMA_VERSION,
        "graph_config_sha256": signature,
        "node_dim": 37,
        "edge_dim": 8,
        "node_feature_names": [f"node_{index}" for index in range(37)],
        "edge_feature_names": [f"edge_{index}" for index in range(8)],
    }), encoding="utf-8")

    graph = CleanGraphCache(
        tmp_path,
        "train",
        expected_graph_config_sha256=signature,
    )[0]
    assert graph.x.shape == (2, 37)
    assert graph.edge_index.shape == (2, 2)
    assert graph.edge_attr.shape == (2, 8)
    assert int(graph.y) == 3
    assert int(graph.sample_index) == 17
    assert np.array_equal(graph.anchor_mask, np.asarray([False, True]))


def test_clean_signature_ignores_train_only_corruption():
    left = {
        "graph_mode": "face_plus_context",
        "prior_corruption": {"enabled": True, "probability": 0.1},
    }
    right = {
        "graph_mode": "face_plus_context",
        "prior_corruption": {"enabled": False, "probability": 0.9},
    }
    assert clean_graph_config_sha256(left) == clean_graph_config_sha256(right)
