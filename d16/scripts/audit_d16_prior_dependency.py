"""Counterfactual prior-dependency audit for D16 checkpoints.

This script is evaluation-only. It does not train, resume, or modify
checkpoints. It runs the same checkpoint under controlled prior perturbations
to measure whether predictions depend too strongly on MediaPipe priors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.graph_builder import D16GraphData, build_pixel_graph, collate_d16_graphs
from d16.data.mediapipe_priors import fallback_priors
from d16.models.d16_model import D16Model
from d16.training.train_d16 import attach_hard_proto_loss_if_needed, load_checkpoint, resolve_device


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
DEFAULT_VARIANTS = ["official", "zero_prior", "shuffle_prior", "forced_fallback"]


def _read_config(run_dir: Path, config_path: Path | None = None) -> Dict[str, Any]:
    if config_path is not None:
        if config_path.suffix.lower() == ".json":
            return json.loads(config_path.read_text(encoding="utf-8"))
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for name in ("resolved_config.yaml", "resolved_config.json"):
        path = run_dir / name
        if path.exists():
            if path.suffix.lower() == ".json":
                return json.loads(path.read_text(encoding="utf-8"))
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raise FileNotFoundError(f"Missing resolved_config.yaml/json in {run_dir}")


def _checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    name = str(checkpoint)
    if name in {"best", "last", "best_val_loss"}:
        name = f"{name}.pt"
    path = run_dir / "checkpoints" / name
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def _copy_prior(prior: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in prior.items()}


def _zero_prior(prior: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = _copy_prior(prior)
    if "face_mask" in out:
        out["face_mask"] = np.zeros_like(out["face_mask"], dtype=np.float32)
    if "part_soft_masks" in out:
        out["part_soft_masks"] = np.zeros_like(out["part_soft_masks"], dtype=np.float32)
    if "micro_anchor_maps" in out:
        out["micro_anchor_maps"] = np.zeros_like(out["micro_anchor_maps"], dtype=np.float32)
    if "distance_maps" in out:
        out["distance_maps"] = np.ones_like(out["distance_maps"], dtype=np.float32)
    if "landmark_xy_48" in out:
        out["landmark_xy_48"] = np.zeros((0, 2), dtype=np.float32)
    if "valid_part_mask" in out:
        out["valid_part_mask"] = np.zeros_like(out["valid_part_mask"], dtype=np.float32)
    if "valid_anchor_mask" in out:
        out["valid_anchor_mask"] = np.zeros_like(out["valid_anchor_mask"], dtype=np.float32)
    if "quality_score" in out:
        out["quality_score"] = np.asarray(0.0, dtype=np.float32)
    return out


def _shuffle_prior(base: Dict[str, np.ndarray], donor: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = _copy_prior(base)
    for key in (
        "face_mask",
        "part_soft_masks",
        "micro_anchor_maps",
        "distance_maps",
        "landmark_xy_48",
        "valid_part_mask",
        "valid_anchor_mask",
        "quality_score",
    ):
        if key in donor:
            out[key] = np.array(donor[key], copy=True)
    # Keep the image, label, sample_index, and original detected/missing flags.
    # The perturbation asks: what happens if the spatial prior is plausible but wrong?
    return out


def _forced_fallback(prior: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    label = int(np.asarray(prior["label"]).item())
    sample_index = int(np.asarray(prior["sample_index"]).item())
    result = fallback_priors(np.asarray(prior["image_48"], dtype=np.float32), label=label, sample_index=sample_index)
    return _copy_prior(result.arrays)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class PriorCounterfactualDataset(Dataset):
    def __init__(
        self,
        prior_dir: Path,
        split: str,
        graph_cfg: Dict[str, Any],
        variant: str,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.prior_dir = Path(prior_dir)
        self.split = str(split)
        self.split_dir = self.prior_dir / self.split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {self.split_dir}")
        self.files = sorted(self.split_dir.glob("*.npz"))
        if max_samples is not None:
            self.files = self.files[: int(max_samples)]
        if not self.files:
            raise FileNotFoundError(f"No D16 prior npz files found in {self.split_dir}")
        self.graph_cfg = dict(graph_cfg or {})
        self.variant = str(variant)
        if self.variant not in set(DEFAULT_VARIANTS):
            raise ValueError(f"Unsupported prior audit variant={self.variant!r}; choices={DEFAULT_VARIANTS}")
        rng = np.random.default_rng(int(seed))
        self.shuffle_indices = np.arange(len(self.files), dtype=np.int64)
        if len(self.files) > 1:
            while True:
                candidate = rng.permutation(len(self.files)).astype(np.int64)
                if np.all(candidate != self.shuffle_indices):
                    self.shuffle_indices = candidate
                    break

    def __len__(self) -> int:
        return len(self.files)

    def _graph_mode_for(self, prior: Dict[str, np.ndarray]) -> str:
        if self.variant == "forced_fallback":
            return "full_with_mask"
        mode = str(self.graph_cfg.get("graph_mode", "face_plus_context"))
        if mode == "hybrid_detected_face_fallback_fullmask":
            detected = bool(np.asarray(prior.get("detected", np.asarray(False))).item())
            return "face_plus_context" if detected else "full_with_mask"
        return mode

    def _mutate(self, index: int, prior: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.variant == "official":
            return prior
        if self.variant == "zero_prior":
            return _zero_prior(prior)
        if self.variant == "shuffle_prior":
            donor = _load_npz(self.files[int(self.shuffle_indices[int(index)])])
            return _shuffle_prior(prior, donor)
        if self.variant == "forced_fallback":
            return _forced_fallback(prior)
        raise AssertionError(f"unhandled variant: {self.variant}")

    def __getitem__(self, index: int) -> D16GraphData:
        prior = _load_npz(self.files[int(index)])
        prior = self._mutate(int(index), prior)
        return build_pixel_graph(
            prior,
            graph_mode=self._graph_mode_for(prior),
            face_threshold=float(self.graph_cfg.get("face_threshold", 0.15)),
            context_pixels=int(self.graph_cfg.get("context_pixels", 2)),
            detail_features=self.graph_cfg.get("detail_features", {}) or {},
            edge_features=self.graph_cfg.get("edge_features", {}) or {},
            anchor_nodes=self.graph_cfg.get("anchor_nodes", {}) or {},
            node_features=self.graph_cfg.get("node_features", {}) or {},
            prior_usage=self.graph_cfg.get("prior_usage"),
        )


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> float:
    vals = []
    for cls in range(num_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        denom = 2.0 * tp + fp + fn
        vals.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
    return float(np.mean(vals))


def _rows_per_class(y_true: np.ndarray, y_pred: np.ndarray, variant: str, split: str) -> List[Dict[str, Any]]:
    rows = []
    for cls, name in enumerate(CLASS_NAMES):
        true = y_true == cls
        pred = y_pred == cls
        tp = float(np.sum(true & pred))
        fp = float(np.sum(~true & pred))
        fn = float(np.sum(true & ~pred))
        precision = 0.0 if tp + fp <= 0.0 else tp / (tp + fp)
        recall = 0.0 if tp + fn <= 0.0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
        rows.append(
            {
                "variant": variant,
                "split": split,
                "class_id": cls,
                "class_name": name,
                "support": int(np.sum(true)),
                "pred_count": int(np.sum(pred)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def _group_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    detected: np.ndarray,
    variant: str,
    split: str,
) -> List[Dict[str, Any]]:
    rows = []
    for flag, group in ((True, "detected"), (False, "fallback")):
        mask = detected.astype(bool) == flag
        if not bool(mask.any()):
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "group": group,
                    "total": 0,
                    "accuracy": math.nan,
                    "macro_f1": math.nan,
                }
            )
            continue
        rows.append(
            {
                "variant": variant,
                "split": split,
                "group": group,
                "total": int(np.sum(mask)),
                "accuracy": float(np.mean(y_true[mask] == y_pred[mask])),
                "macro_f1": _macro_f1(y_true[mask], y_pred[mask], num_classes=len(CLASS_NAMES)),
            }
        )
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_pct(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "MISSING"
    if not math.isfinite(val):
        return "MISSING"
    return f"{val * 100:.2f}%"


def _load_model(cfg: Dict[str, Any], run_dir: Path, checkpoint: str, device: torch.device, input_dim: int) -> tuple[D16Model, Dict[str, Any], Path]:
    model = D16Model.from_config(cfg, input_dim=int(input_dim)).to(device)
    hard_proto_loss = attach_hard_proto_loss_if_needed(
        model,
        cfg.get("loss", {}) or {},
        embedding_dim=int((cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5,
    )
    if hard_proto_loss is not None:
        hard_proto_loss.to(device)
    ckpt_path = _checkpoint_path(run_dir, checkpoint)
    checkpoint_payload = load_checkpoint(ckpt_path, model, device)
    model.eval()
    return model, checkpoint_payload, ckpt_path


@torch.no_grad()
def _evaluate_variant(
    cfg: Dict[str, Any],
    prior_dir: Path,
    split: str,
    variant: str,
    model: D16Model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    seed: int,
) -> Dict[str, Any]:
    dataset = PriorCounterfactualDataset(
        prior_dir=prior_dir,
        split=split,
        graph_cfg=cfg.get("graph", {}) or {},
        variant=variant,
        max_samples=max_samples,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
        collate_fn=collate_d16_graphs,
    )
    training_cfg = cfg.get("training", {}) or {}
    amp_enabled = bool(training_cfg.get("amp", training_cfg.get("mixed_precision", False))) and device.type == "cuda"
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    indices: List[np.ndarray] = []
    detected_flags: List[np.ndarray] = []
    missing_flags: List[np.ndarray] = []
    confidences: List[np.ndarray] = []
    true_probs: List[np.ndarray] = []
    pred_rows: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            out = model(batch)
            logits = out["logits"]
            loss = torch.nn.functional.cross_entropy(logits, batch.y, reduction="sum")
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        confidence = probs.max(dim=1).values
        true_prob = probs.gather(1, batch.y.view(-1, 1)).squeeze(1)
        y_np = batch.y.detach().cpu().numpy().astype(np.int64)
        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        idx_np = batch.sample_index.detach().cpu().numpy().astype(np.int64)
        detected_np = batch.detected.detach().cpu().numpy().astype(np.int64)
        missing_np = batch.landmark_missing_flag.detach().cpu().numpy().astype(np.int64)
        conf_np = confidence.detach().float().cpu().numpy().astype(np.float32)
        true_prob_np = true_prob.detach().float().cpu().numpy().astype(np.float32)
        ys.append(y_np)
        preds.append(pred_np)
        indices.append(idx_np)
        detected_flags.append(detected_np)
        missing_flags.append(missing_np)
        confidences.append(conf_np)
        true_probs.append(true_prob_np)
        total_loss += float(loss.detach().cpu().item())
        total_count += int(y_np.size)
        for row_idx in range(int(y_np.size)):
            pred_rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "sample_index": int(idx_np[row_idx]),
                    "y_true": int(y_np[row_idx]),
                    "y_pred": int(pred_np[row_idx]),
                    "correct": int(y_np[row_idx] == pred_np[row_idx]),
                    "detected": int(detected_np[row_idx]),
                    "landmark_missing_flag": int(missing_np[row_idx]),
                    "confidence": float(conf_np[row_idx]),
                    "true_prob": float(true_prob_np[row_idx]),
                }
            )
    y_true = np.concatenate(ys) if ys else np.asarray([], dtype=np.int64)
    y_pred = np.concatenate(preds) if preds else np.asarray([], dtype=np.int64)
    detected = np.concatenate(detected_flags) if detected_flags else np.asarray([], dtype=np.int64)
    return {
        "variant": variant,
        "split": split,
        "total": int(y_true.size),
        "accuracy": float(np.mean(y_true == y_pred)) if y_true.size else math.nan,
        "macro_f1": _macro_f1(y_true, y_pred, num_classes=len(CLASS_NAMES)) if y_true.size else math.nan,
        "loss": total_loss / max(total_count, 1),
        "confidence_mean": float(np.mean(np.concatenate(confidences))) if confidences else math.nan,
        "true_prob_mean": float(np.mean(np.concatenate(true_probs))) if true_probs else math.nan,
        "y_true": y_true,
        "y_pred": y_pred,
        "detected": detected,
        "pred_rows": pred_rows,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir)
    cfg = _read_config(run_dir, Path(args.config) if args.config else None)
    prior_dir = Path(args.prior_dir or (cfg.get("data", {}) or {}).get("prior_dir", ""))
    if not prior_dir.exists():
        raise FileNotFoundError(f"Missing prior_dir: {prior_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/d16_analysis/prior_dependency_audit") / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))
    training_cfg = cfg.get("training", {}) or {}
    if device.type == "cuda":
        allow_tf32 = bool(training_cfg.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    first_dataset = PriorCounterfactualDataset(
        prior_dir=prior_dir,
        split=args.split,
        graph_cfg=cfg.get("graph", {}) or {},
        variant="official",
        max_samples=args.max_samples,
        seed=args.seed,
    )
    first_batch = next(iter(DataLoader(first_dataset, batch_size=1, shuffle=False, collate_fn=collate_d16_graphs)))
    model, checkpoint_payload, ckpt_path = _load_model(cfg, run_dir, args.checkpoint, device, input_dim=first_batch.x_cat.size(1))

    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    summary_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    variant_payloads: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        payload = _evaluate_variant(
            cfg=cfg,
            prior_dir=prior_dir,
            split=args.split,
            variant=variant,
            model=model,
            device=device,
            batch_size=int(args.batch_size or training_cfg.get("batch_size", 16) or 16),
            num_workers=int(args.num_workers),
            max_samples=args.max_samples,
            seed=int(args.seed),
        )
        variant_payloads[variant] = payload
        summary_rows.append(
            {
                "variant": variant,
                "split": args.split,
                "total": payload["total"],
                "accuracy": payload["accuracy"],
                "macro_f1": payload["macro_f1"],
                "loss": payload["loss"],
                "confidence_mean": payload["confidence_mean"],
                "true_prob_mean": payload["true_prob_mean"],
            }
        )
        per_class_rows.extend(_rows_per_class(payload["y_true"], payload["y_pred"], variant, args.split))
        group_rows.extend(_group_metric_rows(payload["y_true"], payload["y_pred"], payload["detected"], variant, args.split))
        prediction_rows.extend(payload["pred_rows"])

    official = next((row for row in summary_rows if row["variant"] == "official"), None)
    if official is not None:
        for row in summary_rows:
            row["delta_acc_vs_official"] = float(row["accuracy"]) - float(official["accuracy"])
            row["delta_macro_f1_vs_official"] = float(row["macro_f1"]) - float(official["macro_f1"])
    else:
        for row in summary_rows:
            row["delta_acc_vs_official"] = math.nan
            row["delta_macro_f1_vs_official"] = math.nan

    _write_csv(
        output_dir / "prior_counterfactual_summary.csv",
        summary_rows,
        [
            "variant",
            "split",
            "total",
            "accuracy",
            "macro_f1",
            "loss",
            "confidence_mean",
            "true_prob_mean",
            "delta_acc_vs_official",
            "delta_macro_f1_vs_official",
        ],
    )
    _write_csv(
        output_dir / "prior_counterfactual_per_class.csv",
        per_class_rows,
        ["variant", "split", "class_id", "class_name", "support", "pred_count", "precision", "recall", "f1"],
    )
    _write_csv(
        output_dir / "prior_counterfactual_group_metrics.csv",
        group_rows,
        ["variant", "split", "group", "total", "accuracy", "macro_f1"],
    )
    _write_csv(
        output_dir / "prior_counterfactual_predictions.csv",
        prediction_rows,
        [
            "variant",
            "split",
            "sample_index",
            "y_true",
            "y_pred",
            "correct",
            "detected",
            "landmark_missing_flag",
            "confidence",
            "true_prob",
        ],
    )

    lines = [
        "# D16 Prior Dependency Counterfactual Audit",
        "",
        "## Inputs",
        "",
        f"- run_dir: `{run_dir}`",
        f"- checkpoint: `{ckpt_path}`",
        f"- checkpoint_epoch: {int(checkpoint_payload.get('epoch', 0) or 0)}",
        f"- prior_dir: `{prior_dir}`",
        f"- split: `{args.split}`",
        f"- max_samples: `{args.max_samples}`",
        "",
        "## Variant Meaning",
        "",
        "- `official`: normal graph from the saved prior files.",
        "- `zero_prior`: keeps image/label/sample id but zeros face, part, anchor, distance, and valid prior masks.",
        "- `shuffle_prior`: keeps image/label/sample id but replaces spatial prior tensors with a different sample's prior.",
        "- `forced_fallback`: rebuilds every sample with the fallback prior and full-mask graph.",
        "",
        "## Summary",
        "",
        "| variant | accuracy | macro_f1 | loss | confidence | true_prob | delta_acc | delta_macro_f1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {variant} | {acc} | {macro} | {loss:.4f} | {conf:.4f} | {truep:.4f} | {dacc} | {dmacro} |".format(
                variant=row["variant"],
                acc=_fmt_pct(row["accuracy"]),
                macro=_fmt_pct(row["macro_f1"]),
                loss=float(row["loss"]),
                conf=float(row["confidence_mean"]),
                truep=float(row["true_prob_mean"]),
                dacc=_fmt_pct(row["delta_acc_vs_official"]),
                dmacro=_fmt_pct(row["delta_macro_f1_vs_official"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Large drop on `zero_prior`: model relies on explicit prior tensors rather than image/detail evidence alone.",
            "- Large drop on `shuffle_prior`: model relies on the prior being spatially correct, not merely on generic regularization.",
            "- Large drop on `forced_fallback`: model is not robust to no-landmark / fallback inference.",
            "- Small drop across all counterfactuals: prior is likely not the main bottleneck for the selected checkpoint.",
            "",
            "## Files",
            "",
            "- `prior_counterfactual_summary.csv`",
            "- `prior_counterfactual_per_class.csv`",
            "- `prior_counterfactual_group_metrics.csv`",
            "- `prior_counterfactual_predictions.csv`",
        ]
    )
    (output_dir / "PRIOR_DEPENDENCY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "summary_rows": summary_rows,
        "checkpoint": str(ckpt_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D16 prior-dependency counterfactual evaluation.")
    parser.add_argument("--run_dir", required=True, help="Completed D16 run directory containing resolved_config and checkpoints.")
    parser.add_argument("--prior_dir", default=None, help="Prior directory. Defaults to data.prior_dir from resolved config.")
    parser.add_argument("--config", default=None, help="Optional explicit config path.")
    parser.add_argument("--checkpoint", default="best", help="best, last, best_val_loss, or a checkpoint file name.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    payload = run_audit(parse_args())
    print(json.dumps({"output_dir": payload["output_dir"], "checkpoint": payload["checkpoint"]}, indent=2))
    for row in payload["summary_rows"]:
        print(
            f"{row['variant']}: acc={_fmt_pct(row['accuracy'])} "
            f"macro_f1={_fmt_pct(row['macro_f1'])} "
            f"delta_acc={_fmt_pct(row['delta_acc_vs_official'])} "
            f"delta_macro={_fmt_pct(row['delta_macro_f1_vs_official'])}"
        )


if __name__ == "__main__":
    main()
