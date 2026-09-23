from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from mpg_fer_v2_2.config import MPGConfig
from mpg_fer_v2_2.model import DropPath, EdgeAwarePixelGNNLayer, MPGFER, masked_neighbor_softmax


def test_v2_2_parameter_count_is_only_supcon_delta_from_v2() -> None:
    parameters = sum(p.numel() for p in MPGFER().parameters() if p.requires_grad)
    assert parameters == 2_304_528
    assert parameters - 2_238_609 == 65_919
    assert (parameters - 2_238_609) / 2_238_609 < 0.03


def test_full_model_shapes_and_required_diagnostics() -> None:
    model = MPGFER().eval()
    with torch.no_grad():
        logits, outputs = model(torch.rand(1, 1, 48, 48))
    assert logits.shape == (1, 7)
    assert outputs["motif_assignments"].shape == (1, 2304, 48)
    assert outputs["motif_geometry"].shape == (1, 49, 49, 6)
    assert outputs["scale_weights"].shape == (1, 49, 3)
    assert outputs["fusion_representation"].shape == (1, 512)
    assert outputs["supcon_embeddings"].shape == (1, 128)
    assert torch.allclose(outputs["supcon_embeddings"].norm(dim=-1), torch.ones(1))
    for name in ("tau", "H_local_raw", "H_global_raw", "L_MI", "effective_motif_count"):
        assert torch.isfinite(outputs[name]).all()


def test_drop_path_train_and_eval_behavior_is_per_sample_residual() -> None:
    module = DropPath(0.5)
    x = torch.ones(128, 7, 5)
    module.eval()
    assert torch.equal(module(x), x)
    assert torch.equal(module(x), module(x))
    torch.manual_seed(4); module.train(); result = module(x)
    per_sample = result[:, 0, 0]
    assert set(per_sample.tolist()).issubset({0.0, 2.0})
    assert torch.all(result == per_sample.view(-1, 1, 1))
    assert 0 < int((per_sample == 0).sum()) < len(per_sample)


def test_drop_path_schedules_match_locked_ranges() -> None:
    model = MPGFER()
    assert [layer.drop_path1.drop_probability for layer in model.pixel_gnn] == pytest.approx(
        [0.0, 0.01, 0.02, 0.03]
    )
    assert [layer.drop_path1.drop_probability for layer in model.motif_gnn] == pytest.approx(
        [0.0, 0.0125, 0.025, 0.0375, 0.05]
    )


def test_supcon_head_does_not_change_inference_logits() -> None:
    torch.manual_seed(17)
    model = MPGFER().eval()
    image = torch.rand(1, 1, 48, 48)
    with torch.no_grad():
        logits_before, _ = model(image)
        for parameter in model.supcon_head.parameters():
            parameter.normal_(mean=100.0, std=50.0)
        logits_after, _ = model(image)
    assert torch.equal(logits_before, logits_after)


def test_linear_before_gather_is_numerically_equivalent_to_v1_order() -> None:
    torch.manual_seed(7)
    layer = EdgeAwarePixelGNNLayer(d_pixel=12, edge_dim=5, num_heads=3, dropout=0.0).eval()
    batch, nodes, neighbors = 2, 9, 8
    h = torch.randn(batch, nodes, 12)
    neighbor_idx = torch.randint(0, nodes, (nodes, neighbors))
    neighbor_mask = torch.ones(nodes, neighbors, dtype=torch.bool)
    edge = torch.randn(batch, nodes, neighbors, 5)
    with torch.no_grad():
        actual = layer(h, neighbor_idx, neighbor_mask, edge)
        normalized = layer.norm1(h)
        q = layer.q_proj(normalized).reshape(batch, nodes, 3, 4).unsqueeze(3)
        gathered = normalized[:, neighbor_idx, :]
        k = layer.k_proj(gathered).reshape(batch, nodes, 8, 3, 4).permute(0, 1, 3, 2, 4)
        v = (layer.v_proj(gathered) + layer.edge_val(edge)).reshape(batch, nodes, 8, 3, 4).permute(0, 1, 3, 2, 4)
        k_new = layer.k_proj(normalized)[:, neighbor_idx, :].reshape(batch, nodes, 8, 3, 4).permute(0, 1, 3, 2, 4)
        v_new = (layer.v_proj(normalized)[:, neighbor_idx, :] + layer.edge_val(edge)).reshape(batch, nodes, 8, 3, 4).permute(0, 1, 3, 2, 4)
        assert torch.allclose(k, k_new, atol=1e-7, rtol=1e-7)
        assert torch.allclose(v, v_new, atol=1e-7, rtol=1e-7)
        scores = q @ k.transpose(-1, -2) / 2.0
        scores = scores + layer.edge_bias(edge).permute(0, 1, 3, 2).unsqueeze(3)
        scores_new = q @ k_new.transpose(-1, -2) / 2.0
        scores_new = scores_new + layer.edge_bias(edge).permute(0, 1, 3, 2).unsqueeze(3)
        assert torch.allclose(scores, scores_new, atol=1e-7, rtol=1e-7)
        attention = masked_neighbor_softmax(scores, neighbor_mask)
        message = (attention @ v).squeeze(3).reshape(batch, nodes, 12)
        message_new = (masked_neighbor_softmax(scores_new, neighbor_mask) @ v_new).squeeze(3).reshape(batch, nodes, 12)
        assert torch.allclose(message, message_new, atol=1e-7, rtol=1e-7)
        expected = h + layer.out_proj(message)
        expected = expected + layer.ffn(layer.norm2(expected))
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
