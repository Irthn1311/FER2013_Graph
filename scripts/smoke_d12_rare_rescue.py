"""Smoke checks for the D12A rare-class rescue experiment set."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import load_config  # noqa: E402
from data.ddp_chunk_aware_sampler import DDPChunkAwareBatchSampler  # noqa: E402
from models.registry import build_model  # noqa: E402
from training.losses import build_loss  # noqa: E402


CONFIG_PATHS = (
    "configs/experiments/d12a_speed_control_ce_first.yaml",
    "configs/experiments/d12a_target_repeat_disgust.yaml",
    "configs/experiments/d12a_rare_aux_bce.yaml",
    "configs/experiments/d12a_rare_aux_logit_tau05.yaml",
    "configs/experiments/d12a_repeat_disgust_rare_aux.yaml",
    "configs/experiments/d12a_rare_hardneg_margin.yaml",
)


class FakeChunkLabelDataset:
    def __init__(self) -> None:
        self.chunks = [
            list(range(0, 8)),
            list(range(8, 16)),
            list(range(16, 24)),
            list(range(24, 32)),
        ]
        labels = [0, 1, 2, 3, 4, 5, 6, 1] * 4
        self.labels = {idx: int(label) for idx, label in enumerate(labels)}

    def chunk_index_groups(self):
        return [list(chunk) for chunk in self.chunks]

    def label_at_index(self, idx: int) -> int:
        return self.labels[int(idx)]


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


def make_fake_batch(config: dict, edge_index: torch.Tensor) -> dict:
    model_cfg = config["model"]
    bsz = 4
    num_nodes = int(model_cfg.get("num_nodes", 2304))
    node_dim = int(model_cfg.get("node_dim", 7))
    edge_dim = int(model_cfg.get("edge_dim", 5))
    x = torch.randn(bsz, num_nodes, node_dim)
    if node_dim >= 3:
        x[:, :, 1:3] = torch.rand(bsz, num_nodes, 2)
    return {
        "x": x,
        "edge_index": edge_index.unsqueeze(0).expand(bsz, -1, -1),
        "edge_attr": torch.randn(bsz, edge_index.shape[1], edge_dim),
        "node_mask": torch.ones(bsz, num_nodes, dtype=torch.bool),
        "y": torch.tensor([0, 1, 2, 4], dtype=torch.long),
        "graph_id": torch.arange(bsz),
        "sample_idx": torch.arange(bsz),
    }


def assert_runtime_config(config: dict, label: str) -> None:
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    ddp_cfg = config.get("ddp", {})
    model_cfg = config.get("model", {})
    assert data_cfg.get("batch_size") == 64, f"{label}: data.batch_size must be 64"
    assert training_cfg.get("amp") is True, f"{label}: training.amp must be true"
    assert training_cfg.get("use_compile") is True, f"{label}: training.use_compile must be true"
    assert ddp_cfg.get("enabled") is True, f"{label}: ddp.enabled must be true"
    assert ddp_cfg.get("find_unused_parameters") is True, f"{label}: find_unused_parameters must be true"
    assert data_cfg.get("fixed_batch_size") is True, f"{label}: fixed_batch_size must be true"
    assert (ddp_cfg.get("compile_order") or training_cfg.get("compile_order")) == "before_ddp"
    assert model_cfg.get("use_global_branch") is True, f"{label}: use_global_branch must be true"
    assert model_cfg.get("encoder", {}).get("use_scale2") is True, f"{label}: use_scale2 must be true"
    assert model_cfg.get("slot_iterations") == 3, f"{label}: slot_iterations must be 3"
    assert model_cfg.get("residual_slot_connection") is False, f"{label}: residual slot must be false"


def smoke_config(config_path: str, edge_index: torch.Tensor) -> None:
    config = load_config(PROJECT_ROOT / config_path)
    label = config["run"]["config_name"]
    assert_runtime_config(config, label)
    model = build_model(config["model"])
    criterion = build_loss(config["loss"])
    model.train()
    batch = make_fake_batch(config, edge_index)
    out = model(batch)
    assert out["logits"].shape == (4, 7), f"{label}: bad logits shape"
    if config["model"].get("use_rare_aux_heads"):
        assert torch.is_tensor(out.get("rare_aux_logits")), f"{label}: missing rare_aux_logits"
        assert out["rare_aux_logits"].shape == (4, 2), f"{label}: bad rare_aux_logits shape"
    loss_dict = criterion(out, batch["y"], batch)
    required = {"loss", "loss_ce", "loss_local", "loss_rare_aux", "loss_rare_margin"}
    missing = sorted(required.difference(loss_dict))
    assert not missing, f"{label}: missing loss keys {missing}"
    loss = loss_dict["loss"]
    assert torch.isfinite(loss), loss_dict
    loss.backward()
    grad_sum = sum(
        p.grad.detach().abs().sum().item()
        for p in model.parameters()
        if p.grad is not None
    )
    assert grad_sum > 0.0, f"{label}: no gradient flowed"
    print(
        f"config_smoke={label} loss={float(loss.detach()):.6f} "
        f"rare_aux={float(loss_dict['lambda_rare_aux'].detach()):.3f} "
        f"rare_margin={float(loss_dict['lambda_rare_margin'].detach()):.3f}"
    )


def smoke_target_repeat_sampler() -> None:
    dataset = FakeChunkLabelDataset()
    repeated = DDPChunkAwareBatchSampler(
        dataset,
        batch_size=4,
        num_replicas=2,
        rank=0,
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        seed=7,
        ddp_drop_last_batches=True,
        fixed_batch_size=True,
        carry_over_leftovers=True,
        target_class_repeat_factors={1: 8.0},
    )
    plain = DDPChunkAwareBatchSampler(
        dataset,
        batch_size=4,
        num_replicas=2,
        rank=0,
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        seed=7,
        ddp_drop_last_batches=True,
        fixed_batch_size=True,
        carry_over_leftovers=True,
    )
    summary_repeat = repeated.summary(epoch=0)
    summary_plain = plain.summary(epoch=0)
    assert summary_repeat["batches_after_balance"][0] == summary_repeat["batches_after_balance"][1]
    assert summary_repeat["unique_batch_sizes_per_rank"] == [[4], [4]]
    repeat_hist = summary_repeat["per_rank_label_histogram_estimate"][0]
    plain_hist = summary_plain["per_rank_label_histogram_estimate"][0]
    assert repeat_hist.get(1, 0) > plain_hist.get(1, 0)
    assert summary_repeat["repeated_num_indices_total"] > 0
    print(
        "target_repeat_smoke=OK "
        f"repeat_factors={summary_repeat['target_class_repeat_factors']} "
        f"repeated_num_indices_total={summary_repeat['repeated_num_indices_total']} "
        f"hist_rank0={repeat_hist}"
    )


def main() -> None:
    torch.manual_seed(123)
    edge_index = make_grid_edge_index()
    assert edge_index.shape == (2, 17860)
    for config_path in CONFIG_PATHS:
        smoke_config(config_path, edge_index)
    smoke_target_repeat_sampler()
    print(f"d12_rare_rescue_smoke_count={len(CONFIG_PATHS)}")


if __name__ == "__main__":
    main()
