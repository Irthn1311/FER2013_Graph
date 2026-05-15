"""Smoke test for D12A model, loss, backward, and runtime scale-2 edges."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.registry import build_model
from common import load_config
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

REQUIRED_DIAGNOSTIC_KEYS = (
    "h_pixel_mean",
    "h_pixel_std",
    "encoder_gate_mean",
    "encoder_gate_std",
    "encoder_gate_min",
    "encoder_gate_max",
    "slot_area_entropy",
    "logits_mean",
    "logits_std",
)

CONFIG_SMOKE_PATHS = (
    "configs/experiments/d12a_ce_balance_w05.yaml",
    "configs/experiments/d12a_ce_balance_w075.yaml",
    "configs/experiments/d12a_logit_adjust_tau05.yaml",
    "configs/experiments/d12a_focal_gamma1_w05.yaml",
    "configs/experiments/d12a_iter5_w05.yaml",
    "configs/experiments/d12a_supcon_light_w05.yaml",
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


def make_fake_batch(
    *,
    bsz: int,
    num_nodes: int,
    node_dim: int,
    edge_dim: int,
    edge_index_single: torch.Tensor,
) -> dict:
    edge_count = edge_index_single.shape[1]
    x = torch.randn(bsz, num_nodes, node_dim)
    if node_dim >= 3:
        x[:, :, 1:3] = torch.rand(bsz, num_nodes, 2)
    edge_attr = torch.randn(bsz, edge_count, edge_dim)
    y = torch.arange(bsz, dtype=torch.long) % 7
    return {
        "x": x,
        "edge_index": edge_index_single.unsqueeze(0).expand(bsz, -1, -1),
        "edge_attr": edge_attr,
        "node_mask": torch.ones(bsz, num_nodes, dtype=torch.bool),
        "y": y,
        "graph_id": [f"smoke-{idx}" for idx in range(bsz)],
        "sample_idx": torch.arange(bsz),
    }


def assert_required_keys(out: dict, *, label: str) -> None:
    missing = [key for key in REQUIRED_OUTPUT_KEYS if key not in out]
    assert not missing, f"{label} missing keys: {missing}"
    missing_diag = [key for key in REQUIRED_DIAGNOSTIC_KEYS if key not in out]
    assert not missing_diag, f"{label} missing diagnostics: {missing_diag}"
    diagnostics = out.get("diagnostics", {})
    missing_diag_dict = [key for key in REQUIRED_DIAGNOSTIC_KEYS if key not in diagnostics]
    assert not missing_diag_dict, f"{label} diagnostics dict missing: {missing_diag_dict}"


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


def assert_runtime_config(config: dict, *, label: str) -> None:
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    ddp_cfg = config.get("ddp", {})
    model_cfg = config.get("model", {})
    encoder_cfg = model_cfg.get("encoder", {})
    assert data_cfg.get("batch_size") == 64, f"{label}: data.batch_size must be 64"
    assert training_cfg.get("batch_size") == 64, f"{label}: training.batch_size must be 64"
    assert training_cfg.get("amp") is True, f"{label}: training.amp must be true"
    assert training_cfg.get("use_compile") is True, f"{label}: training.use_compile must be true"
    assert ddp_cfg.get("enabled") is True, f"{label}: ddp.enabled must be true"
    assert ddp_cfg.get("find_unused_parameters") is True, f"{label}: ddp.find_unused_parameters must be true"
    assert data_cfg.get("fixed_batch_size") is True, f"{label}: data.fixed_batch_size must be true"
    compile_order = ddp_cfg.get("compile_order", training_cfg.get("compile_order"))
    assert compile_order == "before_ddp", f"{label}: compile_order must be before_ddp"
    assert model_cfg.get("use_global_branch") is True, f"{label}: use_global_branch must be true"
    assert encoder_cfg.get("use_scale2") is True, f"{label}: encoder.use_scale2 must be true"
    assert model_cfg.get("residual_slot_connection") is False, f"{label}: residual slot must be false"
    assert training_cfg.get("epochs") == 30, f"{label}: training.epochs must be 30"
    assert training_cfg.get("early_stopping_patience") == 15, f"{label}: early stopping patience must be 15"


def smoke_config(config_path: str, edge_index_single: torch.Tensor) -> None:
    config = load_config(PROJECT_ROOT / config_path)
    label = config["run"]["config_name"]
    assert_runtime_config(config, label=label)

    model_cfg = dict(config["model"])
    loss_cfg = dict(config["loss"])
    model = build_model(model_cfg)
    criterion = build_loss(loss_cfg)
    model.train()

    bsz = 2
    num_nodes = int(model_cfg.get("num_nodes", 2304))
    node_dim = int(model_cfg.get("node_dim", config.get("data", {}).get("node_dim", 7)))
    edge_dim = int(model_cfg.get("edge_dim", config.get("data", {}).get("edge_dim", 5)))
    num_classes = int(model_cfg.get("num_classes", 7))
    num_slots = int(model_cfg.get("num_slots", 8))
    hidden_dim = int(model_cfg.get("hidden_dim", 96))
    batch = make_fake_batch(
        bsz=bsz,
        num_nodes=num_nodes,
        node_dim=node_dim,
        edge_dim=edge_dim,
        edge_index_single=edge_index_single,
    )
    y = batch["y"]

    out = model(batch)
    assert_required_keys(out, label=label)
    assert_core_shapes(
        out,
        bsz=bsz,
        num_classes=num_classes,
        num_slots=num_slots,
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
    )
    loss_dict = criterion(out, y, batch)
    required_loss_keys = {
        "loss",
        "loss_ce",
        "loss_local",
        "loss_supcon",
        "loss_div",
        "effective_ce_weight",
        "effective_ce_factor",
        "effective_lambda_supcon",
        "logit_adjust_tau",
        "focal_gamma",
    }
    missing = sorted(required_loss_keys.difference(loss_dict))
    assert not missing, f"{label} missing loss keys: {missing}"
    loss = loss_dict["loss"]
    assert torch.isfinite(loss), loss_dict
    assert loss.requires_grad
    loss.backward()
    grad_sum = sum(
        p.grad.detach().abs().sum().item()
        for p in model.parameters()
        if p.grad is not None
    )
    assert grad_sum > 0.0, f"{label}: no gradient flowed"
    print(
        f"config_smoke={label} loss={float(loss.detach()):.6f} "
        f"logit_adjust_tau={float(loss_dict['logit_adjust_tau'].detach()):.3f} "
        f"focal_gamma={float(loss_dict['focal_gamma'].detach()):.3f}"
    )


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

    batch = make_fake_batch(
        bsz=bsz,
        num_nodes=num_nodes,
        node_dim=node_dim,
        edge_dim=edge_dim,
        edge_index_single=edge_index_single,
    )
    y = batch["y"]

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
    print(f"logits={tuple(out['logits'].shape)} part_masks={tuple(out['part_masks'].shape)}")
    print(f"scale1_edges={edge_count} scale2_edges={int(scale2.shape[1])}")
    print(f"loss={float(loss.detach()):.6f} grad_sum={grad_sum:.6f}")

    for config_path in CONFIG_SMOKE_PATHS:
        smoke_config(config_path, edge_index_single)
    print(f"config_smoke_count={len(CONFIG_SMOKE_PATHS)}")


if __name__ == "__main__":
    main()
