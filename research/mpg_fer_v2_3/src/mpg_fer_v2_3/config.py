"""Configuration contract for MPG-FER v2.3 (Issue #97)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MPGConfig:
    seed: int = 42
    device: str = "cuda"

    img_size: int = 48
    num_pixels: int = 2304
    num_neighbors: int = 8
    raw_pixel_dim: int = 32
    d_pixel: int = 96
    num_pixel_gnn_layers: int = 4
    num_pixel_heads: int = 4
    pixel_dropout: float = 0.10
    pixel_drop_path_max: float = 0.03
    pixel_edge_dim: int = 5
    d_pixel_readout: int = 128

    num_motifs: int = 48
    motif_window_sizes: tuple[int, int, int] = (8, 12, 16)
    motif_window_size: int = 12
    motif_stride: int = 6
    num_occurrences: int = 49
    d_what: int = 96
    d_type: int = 32
    d_where: int = 5
    d_motif_occurrence_raw: int = 133
    d_motif: int = 192
    tau_start: float = 0.70
    tau_final: float = 0.30
    tau_anneal_end_epoch: int = 35

    num_motif_layers: int = 5
    num_motif_heads: int = 6
    motif_geom_dim: int = 6
    motif_dropout: float = 0.10
    motif_drop_path_max: float = 0.05
    d_motif_readout: int = 384
    # Dynamic sparse motif routing schedule: K edges per query node for layers 1..5
    motif_topk_schedule: tuple[int, ...] = (8, 16, 16, 16, 24)
    # Issue #97 single delta: preserve early representations with half-step
    # residual updates in L1-L2; all later motif blocks remain v2.2-equivalent.
    motif_residual_scale_schedule: tuple[float, ...] = (0.5, 0.5, 1.0, 1.0, 1.0)

    d_classifier_in: int = 512
    d_classifier_hidden: int = 256
    num_classes: int = 7
    classifier_dropout: float = 0.25
    supcon_dim: int = 128
    supcon_temperature: float = 0.10

    aux_pixel_weight: float = 0.05
    aux_motif_weight: float = 0.20
    lambda_div: float = 0.01
    lambda_mi: float = 0.025
    mi_beta: float = 1.0
    consistency_probability: float = 0.50
    lambda_consistency: float = 0.15
    lambda_supcon: float = 0.05

    ema_decay: float = 0.999
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 5e-4
    max_epochs: int = 120
    min_epochs: int = 50
    warmup_epochs: int = 5
    lr_decay_end_epoch: int = 85
    min_learning_rate: float = 1e-6
    early_stop_monitor_start_epoch: int = 85
    early_stop_patience: int = 15
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    use_amp: bool = True
    num_workers: int = 2

    resume_schema_version: int = 3
    resume_snapshot_interval: int = 10
    resume_snapshots_to_keep: int = 2
    segment_soft_limit_hours: float = 10.5
    segment_safety_margin_minutes: float = 15.0
    segment_number: int = 1
    run_id: str | None = None
    output_dir: str | None = None
    resume_path: str | None = None

    micro_overfit_samples: int = 16
    micro_overfit_target: float = 0.875
    micro_overfit_max_steps: int = 80
    micro_overfit_learning_rate: float = 1e-3

    runtime_safe_resume_fields: tuple[str, ...] = field(
        default=(
            "num_workers", "output_dir", "resume_path", "segment_number",
            "segment_soft_limit_hours", "segment_safety_margin_minutes", "run_id",
        ),
        repr=False,
    )

    def __post_init__(self) -> None:
        if len(self.motif_topk_schedule) != self.num_motif_layers:
            raise ValueError(
                "motif_topk_schedule length must equal num_motif_layers"
            )
        maximum = self.num_occurrences - 1
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
            for value in self.motif_topk_schedule
        ):
            raise ValueError(
                f"motif_topk_schedule values must be integers in [1, {maximum}]"
            )
        if len(self.motif_residual_scale_schedule) != self.num_motif_layers:
            raise ValueError(
                "motif_residual_scale_schedule length must equal num_motif_layers"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) <= 1.0
            for value in self.motif_residual_scale_schedule
        ):
            raise ValueError(
                "motif_residual_scale_schedule values must be numeric in (0, 1]"
            )
