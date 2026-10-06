"""Run one bounded MPG-FER AMP train step and report component/VRAM evidence."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torchvision.transforms.functional as TF

from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.ema import ModelEMA
from mpg_fer_v2_3.model import MPGFER
from mpg_fer_v2_3.train import compute_training_loss
from mpg_fer_v2_3.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    if args.batch_size < 8:
        raise ValueError(
            "batch-size must be at least 8 so deterministic labels include "
            "a positive pair for the SupCon gradient audit"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    config = MPGConfig(batch_size=args.batch_size, use_amp=True)
    set_seed(config.seed)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = MPGFER(config).to(device).train()
    model.set_epoch_temperature(1)
    ema = ModelEMA(model, decay=config.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda")
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    images = torch.rand(args.batch_size, 1, 48, 48, device=device)
    targets = torch.arange(args.batch_size, device=device) % config.num_classes

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    step_started = time.perf_counter()
    with torch.amp.autocast("cuda", enabled=True):
        logits, outputs = model(images)
        flipped_logits, _ = model(TF.hflip(images))
        loss, components = compute_training_loss(
            logits, outputs, targets, criterion, config, flipped_logits
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    audited_parameters = {
        "pixel_projection": model.pixel_proj[0].weight,
        "pixel_readout": model.pixel_readout_proj[0].weight,
        "assignment_query": model.motif_composer.assignment_query.weight,
        "prototypes": model.motif_composer.prototypes,
        "scale_gate": model.motif_composer.scale_gate.weight,
        "scale_saliency_8": model.motif_composer.scale_saliency["8"].weight,
        "motif_readout": model.motif_readout_proj[0].weight,
        "supcon_projection": model.supcon_head[0].weight,
        "classifier": model.classifier[-1].weight,
    }
    for layer_index, layer in enumerate(model.motif_gnn, start=1):
        audited_parameters.update(
            {
                f"motif_layer_{layer_index}_q": layer.q_proj.weight,
                f"motif_layer_{layer_index}_k": layer.k_proj.weight,
                f"motif_layer_{layer_index}_v": layer.v_proj.weight,
                f"motif_layer_{layer_index}_geometry": layer.geom_proj.weight,
            }
        )
    gradient_audits = {
        name: bool(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.any(parameter.grad != 0)
        )
        for name, parameter in audited_parameters.items()
    }
    if not all(gradient_audits.values()):
        raise RuntimeError(f"Bounded audit gradient failure: {gradient_audits}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer_step_succeeded = scaler.get_scale() >= old_scale
    if optimizer_step_succeeded:
        ema.update(model)
    torch.cuda.synchronize()
    step_time_seconds = time.perf_counter() - step_started
    result = {
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "physical_batch_size": args.batch_size,
        "amp": True,
        "worst_case_consistency_forward": True,
        "supcon_enabled": True,
        "motif_topk_schedule": list(config.motif_topk_schedule),
        "motif_residual_scale_schedule": list(
            config.motif_residual_scale_schedule
        ),
        "loss": float(loss.detach()),
        "loss_components": {
            name: float(value.detach()) for name, value in components.items()
        },
        "optimizer_step_succeeded": optimizer_step_succeeded,
        "ema_updates": ema.num_updates,
        "step_time_seconds": step_time_seconds,
        "gradient_audits": gradient_audits,
        "motif_diagnostics": {
            name: float(outputs[name].detach())
            for name in (
                "tau", "H_local_raw", "H_local_normalized",
                "H_global_raw", "H_global_normalized", "L_MI",
                "mean_entropy", "effective_motif_count",
                "min_utilization", "max_utilization", "std_utilization",
                "mean_top1_probability", "mean_top2_probability",
                "mean_top1_top2_margin", "mean_offdiag_prototype_cosine",
                "mean_alpha_8", "mean_alpha_12", "mean_alpha_16",
            )
        },
        "routing_diagnostics": {
            f"motif_l{layer_index}_{name}": float(
                outputs[f"motif_l{layer_index}_{name}"].detach()
            )
            for layer_index in range(1, 6)
            for name in ("entropy", "top1_mass", "boundary_tie_count")
        },
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
