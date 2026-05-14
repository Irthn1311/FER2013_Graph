"""Smoke test for D12A model, loss, backward, and runtime scale-2 edges."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.registry import build_model
from training.losses import build_loss


REQUIRED_OUTPUT_KEYS = (
    "logits",
    "logits_local",
    "motif_embeddings",
    "part_masks",
    "slot_centers",
    "class_motif_attn",
    "global_context",
    "virtual_attention",
    "film_gamma",
    "film_beta",
    "motif_supcon",
)


def make_grid_edge_index(height: int = 48, width: int = 48) -> torch.Tensor:
    offsets = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
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


def build_d12_smoke_model(
    *,
    hidden_dim: int,
    num_classes: int,
    num_nodes: int,
    node_dim: int,
    edge_dim: int,
    num_slots: int,
    use_global_branch: bool,
):
    return build_model(
        {
            "name": "d12_global_local_motif",
            "num_classes": num_classes,
            "num_nodes": num_nodes,
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "hidden_dim": hidden_dim,
            "dropout": 0.1,
            "encoder": {
                "num_layers": 1,
                "use_scale2": True,
                "scale2_alpha": 1.0,
                "use_gate_norm": True,
            },
            "num_slots": num_slots,
            "slot_iterations": 2,
            "residual_slot_connection": True,
            "motif_relation_layers": 1,
            "motif_relation_heads": 4,
            "use_global_branch": use_global_branch,
            "global_dim": 64,
            "global_dropout": 0.3,
            "supcon_projection_dim": 128,
            "height": 48,
            "width": 48,
        }
    )


def assert_required_keys(out: dict, *, label: str) -> None:
    missing = [key for key in REQUIRED_OUTPUT_KEYS if key not in out]
    assert not missing, f"{label} missing keys: {missing}"


def assert_core_shapes(
    out: dict,
    *,
    bsz: int,
    num_classes: int,
    num_slots: int,
    num_nodes: int,
    hidden_dim: int,
) -> None:
    assert out["logits"].shape == (bsz, num_classes)
    assert out["logits_local"].shape == (bsz, num_classes)
    assert out["motif_embeddings"].shape == (bsz, num_slots, hidden_dim)
    assert out["part_masks"].shape == (bsz, num_slots, num_nodes)
    assert out["slot_centers"].shape == (bsz, num_slots, 2)
    assert out["class_motif_attn"].shape == (bsz, num_classes, num_slots)
    assert out["global_context"].shape == (bsz, 64)
    assert out["virtual_attention"].shape == (bsz, num_nodes)
    assert out["film_gamma"].shape == (bsz, num_slots, hidden_dim)
    assert out["film_beta"].shape == (bsz, num_slots, hidden_dim)
    assert out["motif_supcon"].shape == (bsz, 128)


def main() -> None:
    torch.manual_seed(123)
    bsz = 2
    num_nodes = 2304
    node_dim = 7
    edge_dim = 5
    hidden_dim = 32
    num_slots = 8
    num_classes = 7

    edge_index_single = make_grid_edge_index()
    assert edge_index_single.shape == (2, 17860)
    edge_count = edge_index_single.shape[1]

    x = torch.randn(bsz, num_nodes, node_dim)
    x[:, :, 1:3] = torch.rand(bsz, num_nodes, 2)
    edge_attr = torch.randn(bsz, edge_count, edge_dim)
    node_mask = torch.ones(bsz, num_nodes, dtype=torch.bool)
    y = torch.tensor([0, 1], dtype=torch.long)
    batch = {
        "x": x,
        "edge_index": edge_index_single.unsqueeze(0).expand(bsz, -1, -1),
        "edge_attr": edge_attr,
        "node_mask": node_mask,
        "y": y,
        "graph_id": ["smoke-0", "smoke-1"],
        "sample_idx": torch.arange(bsz),
    }

    model = build_d12_smoke_model(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_nodes=num_nodes,
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_slots=num_slots,
        use_global_branch=True,
    )
    model.train()
    out = model(batch)

    assert_required_keys(out, label="full")
    assert_core_shapes(
        out,
        bsz=bsz,
        num_classes=num_classes,
        num_slots=num_slots,
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
    )

    no_global_model = build_d12_smoke_model(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_nodes=num_nodes,
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_slots=num_slots,
        use_global_branch=False,
    )
    no_global_model.eval()
    with torch.no_grad():
        out_no_global = no_global_model(batch)
    assert_required_keys(out_no_global, label="no_global")
    assert_core_shapes(
        out_no_global,
        bsz=bsz,
        num_classes=num_classes,
        num_slots=num_slots,
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
    )

    criterion = build_loss(
        {
            "name": "d12_global_local_motif_loss",
            "use_class_weights": True,
            "class_counts": [3995, 436, 528, 879, 594, 416, 626],
            "class_weight_power": 0.5,
            "label_smoothing": 0.1,
            "lambda_local": 0.3,
            "lambda_supcon": 0.1,
            "lambda_div": 0.02,
            "diversity_margin": 0.3,
            "lambda_spatial": 0.0,
            "supcon_temperature": 0.2,
            "supcon_warmup_epochs": 5,
            "ce_warmup_epochs": 30,
        }
    )
    loss_dict = criterion(out, y, batch)
    loss = loss_dict["loss"]
    assert torch.isfinite(loss), loss_dict
    assert loss.requires_grad

    loss.backward()
    grad_sum = sum(
        p.grad.detach().abs().sum().item()
        for p in model.parameters()
        if p.grad is not None
    )
    assert grad_sum > 0.0

    scale2 = model.encoder.scale2_edge_index
    assert scale2.shape[0] == 2
    assert scale2.shape[1] > 0
    assert int(scale2.max()) < num_nodes

    if torch.cuda.device_count() >= 2:
        device = torch.device("cuda:0")
        dp_model = torch.nn.DataParallel(model.to(device))
        dp_batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        dp_out = dp_model(dp_batch)
        assert dp_out["logits"].shape == (bsz, num_classes)

    print("D12 smoke OK")
    print(f"internal_edge_dim={model.encoder.internal_edge_dim}")
    print(f"full_keys={','.join(sorted(REQUIRED_OUTPUT_KEYS))}")
    print(f"no_global_keys={','.join(sorted(REQUIRED_OUTPUT_KEYS))}")
    print(f"logits={tuple(out['logits'].shape)} part_masks={tuple(out['part_masks'].shape)}")
    print(f"scale1_edges={edge_count} scale2_edges={int(scale2.shape[1])}")
    print(f"loss={float(loss.detach()):.6f} grad_sum={grad_sum:.6f}")


if __name__ == "__main__":
    main()
