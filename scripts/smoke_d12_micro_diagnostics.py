"""Smoke test D12A micro diagnostics without training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.registry import build_model
from training.losses import build_loss


REQUIRED_MICRO_KEYS = (
    "encoder_input_std",
    "encoder_scale1_std",
    "encoder_scale2_std",
    "encoder_final_std",
    "encoder_scale2_delta_norm",
    "encoder_scale2_delta_ratio",
    "cos_eye_scale1",
    "cos_eye_scale2",
    "slot_area_entropy",
    "slot_attention_peak",
    "slot_attention_entropy_per_slot",
    "effective_slots",
)


def make_grid_edge_index(height: int = 48, width: int = 48) -> torch.Tensor:
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    src_nodes = []
    dst_nodes = []
    for y in range(height):
        for x in range(width):
            src = y * width + x
            for dy, dx in offsets:
                yy = y + dy
                xx = x + dx
                if 0 <= yy < height and 0 <= xx < width:
                    src_nodes.append(src)
                    dst_nodes.append(yy * width + xx)
    return torch.tensor([src_nodes, dst_nodes], dtype=torch.long)


def make_batch(bsz: int = 2) -> dict:
    num_nodes = 2304
    node_dim = 7
    edge_dim = 5
    edge_index_single = make_grid_edge_index()
    edge_count = edge_index_single.shape[1]
    x = torch.randn(bsz, num_nodes, node_dim)
    ys = torch.linspace(0.0, 1.0, 48)
    xs = torch.linspace(0.0, 1.0, 48)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x[:, :, 1] = xx.reshape(-1)
    x[:, :, 2] = yy.reshape(-1)
    return {
        "x": x,
        "edge_index": edge_index_single.unsqueeze(0).expand(bsz, -1, -1),
        "edge_attr": torch.randn(bsz, edge_count, edge_dim),
        "node_mask": torch.ones(bsz, num_nodes, dtype=torch.bool),
        "y": torch.tensor([0, 1], dtype=torch.long),
        "graph_id": torch.arange(bsz),
        "sample_idx": torch.arange(bsz),
    }


def build_d12_model(enable: bool):
    return build_model(
        {
            "name": "d12_global_local_motif",
            "num_classes": 7,
            "num_nodes": 2304,
            "node_dim": 7,
            "edge_dim": 5,
            "hidden_dim": 32,
            "dropout": 0.1,
            "encoder": {
                "num_layers": 1,
                "use_scale2": True,
                "scale2_alpha": 1.0,
                "use_gate_norm": True,
            },
            "num_slots": 8,
            "slot_iterations": 2,
            "residual_slot_connection": True,
            "motif_relation_layers": 1,
            "motif_relation_heads": 4,
            "use_global_branch": True,
            "global_dim": 64,
            "global_dropout": 0.3,
            "supcon_projection_dim": 128,
            "height": 48,
            "width": 48,
            "diagnostics": {
                "enable_micro_diagnostics": enable,
                "save_attention_maps": enable,
                "save_node_similarity": enable,
                "diagnostic_max_samples": 2,
            },
        }
    )


def main() -> None:
    torch.manual_seed(123)
    batch = make_batch()
    model = build_d12_model(enable=True)
    model.train()
    out = model(batch)
    diagnostics = out.get("diagnostics", {})
    missing = [key for key in REQUIRED_MICRO_KEYS if key not in out or key not in diagnostics]
    assert not missing, f"missing micro diagnostic keys: {missing}"
    for key in REQUIRED_MICRO_KEYS:
        value = diagnostics[key]
        assert torch.is_tensor(value) and value.numel() == 1, key
        assert torch.isfinite(value).all(), key
        assert not value.requires_grad, key

    criterion = build_loss(
        {
            "name": "d12_global_local_motif_loss",
            "use_class_weights": True,
            "class_counts": [3995, 436, 528, 879, 594, 416, 626],
            "class_weight_power": 0.25,
            "label_smoothing": 0.05,
            "lambda_local": 0.3,
            "lambda_supcon": 0.0,
            "lambda_div": 0.0,
            "lambda_spatial": 0.0,
            "ce_warmup_epochs": 0,
        }
    )
    loss = criterion(out, batch["y"], batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grad_sum = sum(p.grad.detach().abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_sum > 0.0

    disabled = build_d12_model(enable=False)
    disabled_out = disabled(batch)
    assert "logits" in disabled_out and "part_masks" in disabled_out
    disabled_diag = disabled_out.get("diagnostics", {})
    assert "encoder_input_std" not in disabled_diag
    assert "cos_eye_scale1" not in disabled_diag

    print("D12 micro diagnostics smoke OK")
    print(f"loss={float(loss.detach()):.6f} grad_sum={grad_sum:.6f}")
    print(f"scale2_delta_ratio={float(diagnostics['encoder_scale2_delta_ratio']):.6f}")
    print(f"cos_eye_delta={float(diagnostics['cos_eye_delta']):.6f}")


if __name__ == "__main__":
    main()
