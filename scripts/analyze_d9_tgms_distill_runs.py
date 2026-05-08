from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

RUNS = [
    {
        "name": "d9_tgms_b_distill_a05_t2_cap300v100",
        "short": "a05",
        "path": OUTPUTS / "d9_tgms_b_distill_a05_t2_cap300v100_outputs",
    },
    {
        "name": "d9_tgms_b_distill_a02_t2_cap300v100",
        "short": "a02",
        "path": OUTPUTS / "d9_tgms_b_distill_a02_t2_cap300v100_outputs",
    },
]

NO_TEACHER = {
    "name": "D9 pooled MLP no-teacher reference",
    "path": OUTPUTS / "d9_rg_b_no_mr_pooled_mlp_diag20_cap300v100",
    "provided_macro_f1": 0.1674,
    "provided_accuracy": 0.2213,
    "provided_weighted_f1": 0.1984,
}

REPORT_PATH = OUTPUTS / "d9_tgms_distill_analysis_report.md"
CSV_PATH = OUTPUTS / "d9_tgms_distill_analysis_summary.csv"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = value
                    continue
                try:
                    if key == "epoch":
                        parsed[key] = int(float(value))
                    else:
                        parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def bincount(values: list[int]) -> list[int]:
    return np.bincount(np.asarray(values, dtype=np.int64), minlength=len(CLASSES)).astype(int).tolist()


def read_predictions(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    y_true: list[int] = []
    y_pred: list[int] = []
    finite = True
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        logit_cols = [c for c in (reader.fieldnames or []) if c.startswith("logit_")]
        for row in reader:
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
            for col in logit_cols:
                val = float(row[col])
                finite = finite and math.isfinite(val)
    return {
        "n": len(y_true),
        "true_distribution": bincount(y_true),
        "pred_distribution": bincount(y_pred),
        "logits_finite": finite,
    }


def numeric_csv_finite(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def localize_kaggle_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path_text = str(path_text)
    prefix = "/kaggle/working/"
    if path_text.startswith(prefix):
        return ROOT / path_text[len(prefix) :]
    p = Path(path_text)
    if p.is_absolute():
        return p
    return ROOT / p


def teacher_probe(teacher_dir: Path | None) -> dict[str, Any]:
    if teacher_dir is None:
        return {"available": False, "reason": "teacher_probs_dir missing in config"}
    required = [
        "teacher_metrics.json",
        "teacher_manifest.json",
        "train_probs.npy",
        "train_labels.npy",
        "train_indices.npy",
        "val_probs.npy",
        "val_labels.npy",
        "val_indices.npy",
    ]
    exists = {name: (teacher_dir / name).exists() for name in required}
    if not teacher_dir.exists():
        return {
            "available": False,
            "teacher_dir": str(teacher_dir),
            "reason": "teacher_probs_dir is not present in the local extracted artifacts",
            "files": exists,
        }
    out: dict[str, Any] = {
        "available": all(exists.values()),
        "teacher_dir": str(teacher_dir),
        "files": exists,
    }
    if (teacher_dir / "teacher_metrics.json").exists():
        metrics = load_json(teacher_dir / "teacher_metrics.json")
        out["teacher_metrics"] = metrics
        val = (metrics.get("splits") or {}).get("val") or {}
        out["teacher_val_macro_f1"] = val.get("macro_f1")
    for split in ("train", "val"):
        probs_path = teacher_dir / f"{split}_probs.npy"
        labels_path = teacher_dir / f"{split}_labels.npy"
        indices_path = teacher_dir / f"{split}_indices.npy"
        if probs_path.exists() and labels_path.exists() and indices_path.exists():
            probs = np.load(probs_path)
            labels = np.load(labels_path)
            indices = np.load(indices_path)
            row_sums = probs.sum(axis=1)
            conf = probs.max(axis=1)
            entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
            out[f"{split}_stats"] = {
                "probs_shape": list(probs.shape),
                "labels_shape": list(labels.shape),
                "indices_shape": list(indices.shape),
                "row_sum_min": float(row_sums.min()),
                "row_sum_max": float(row_sums.max()),
                "row_sum_max_err": float(np.abs(row_sums - 1.0).max()),
                "non_finite": int((~np.isfinite(probs)).sum()),
                "label_distribution": bincount(labels.astype(int).tolist()),
                "pred_distribution": bincount(probs.argmax(axis=1).astype(int).tolist()),
                "max_prob_mean": float(conf.mean()),
                "max_prob_std": float(conf.std()),
                "max_prob_min": float(conf.min()),
                "max_prob_max": float(conf.max()),
                "entropy_mean": float(entropy.mean()),
                "entropy_std": float(entropy.std()),
                "index_min": int(indices.min()),
                "index_max": int(indices.max()),
                "index_unique": int(len(set(indices.astype(int).tolist()))),
            }
    return out


def checkpoint_probe(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "loaded": False,
    }
    if not path.exists():
        return out
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        out["load_error"] = f"torch unavailable: {exc}"
        return out
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - checkpoint/environment dependent
        out["load_error"] = str(exc)
        return out
    if not isinstance(ckpt, dict):
        out["load_error"] = f"checkpoint is {type(ckpt).__name__}, not dict"
        return out
    out.update(
        {
            "loaded": True,
            "keys": list(ckpt.keys()),
            "epoch": ckpt.get("epoch"),
            "best_epoch": ckpt.get("best_epoch"),
            "best_metric": ckpt.get("best_metric"),
            "best_metric_name": ckpt.get("best_metric_name"),
            "best_metric_mode": ckpt.get("best_metric_mode"),
        }
    )
    metrics = ckpt.get("metrics")
    if isinstance(metrics, dict):
        out["metrics_epoch"] = metrics.get("epoch")
        out["metrics_val_macro_f1"] = metrics.get("val_macro_f1")
        out["metrics_val_accuracy"] = metrics.get("val_accuracy")
        out["metrics_val_weighted_f1"] = metrics.get("val_weighted_f1")
        out["metrics_train_distill_loss"] = metrics.get("train_distill_loss")
    return out


def top_epochs(history: list[dict[str, Any]], n: int = 5) -> list[dict[str, Any]]:
    return sorted(history, key=lambda r: float(r.get("val_macro_f1", -1.0)), reverse=True)[:n]


def row_by_epoch(history: list[dict[str, Any]], epoch: int) -> dict[str, Any] | None:
    return next((r for r in history if int(r.get("epoch", -1)) == int(epoch)), None)


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def important_epoch_table(history: list[dict[str, Any]], best_epoch: int) -> str:
    wanted = {1, best_epoch, int(history[-1]["epoch"])}
    wanted.update(int(r["epoch"]) for r in top_epochs(history, 5))
    rows = []
    for epoch in sorted(wanted):
        row = row_by_epoch(history, epoch)
        if not row:
            continue
        rows.append(
            "| {epoch} | {val_macro_f1:.6f} | {val_accuracy:.6f} | {train_cls_loss:.6f} | "
            "{train_distill_loss:.6f} | {train_motif_loss:.6f} | {lr:.8f} |".format(**row)
        )
    return "\n".join(
        [
            "| epoch | val_macro_f1 | val_accuracy | cls_loss | distill_loss | motif_loss | lr |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )


def analyze_run(run: dict[str, Any]) -> dict[str, Any]:
    path = run["path"]
    cfg = load_yaml(path / "resolved_config.yaml")
    history = read_history(path / "logs" / "d9_history.csv")
    best_summary = load_json(path / "metrics" / "best_summary.json")
    val_metrics = load_json(path / "metrics" / "val_metrics.json")
    summary_files = sorted(path.glob("*_summary.json"))
    notebook_summary = load_json(summary_files[0]) if summary_files else {}
    pred = read_predictions(path / "metrics" / "val_predictions.csv")
    best_epoch = int(best_summary["best_epoch"])
    best_metrics = best_summary["best_metrics"]
    alpha = float((cfg.get("distillation") or {}).get("alpha", 0.0))
    temperature = float((cfg.get("distillation") or {}).get("temperature", 0.0))
    teacher_dir = localize_kaggle_path((cfg.get("distillation") or {}).get("teacher_probs_dir"))
    teacher = teacher_probe(teacher_dir)
    history_finite = numeric_csv_finite(history)
    per_class = {
        cls: (val_metrics.get("classification_report") or {}).get(cls, {}).get("f1-score")
        for cls in CLASSES
    }
    checkpoints = {
        name: (path / "checkpoints" / name).exists()
        for name in ("initial.pth", "best.pth", "last.pth")
    }
    checkpoint_meta = {
        name: checkpoint_probe(path / "checkpoints" / name)
        for name in ("initial.pth", "best.pth", "last.pth")
    }
    sizes = {
        name: (path / "checkpoints" / name).stat().st_size if (path / "checkpoints" / name).exists() else 0
        for name in ("initial.pth", "best.pth", "last.pth")
    }
    required_artifacts = [
        "checkpoints/initial.pth",
        "checkpoints/best.pth",
        "checkpoints/last.pth",
        "logs/d9_history.csv",
        "logs/val_metrics.jsonl",
        "metrics/best_summary.json",
        "metrics/val_metrics.json",
        "resolved_config.yaml",
        "metrics/val_predictions.csv",
        "figures/val_confusion_matrix.png",
    ]
    artifact_presence = {rel: (path / rel).exists() for rel in required_artifacts}
    weighted_ratio = alpha * float(best_metrics["train_distill_loss"]) / float(best_metrics["train_cls_loss"])
    raw_ratio = float(best_metrics["train_distill_loss"]) / float(best_metrics["train_cls_loss"])
    return {
        **run,
        "cfg": cfg,
        "history": history,
        "best_summary": best_summary,
        "best_metrics": best_metrics,
        "val_metrics": val_metrics,
        "notebook_summary": notebook_summary,
        "pred": pred,
        "teacher": teacher,
        "alpha": alpha,
        "temperature": temperature,
        "teacher_dir": teacher_dir,
        "best_epoch": best_epoch,
        "last_epoch": int(history[-1]["epoch"]),
        "per_class_f1": per_class,
        "checkpoints": checkpoints,
        "checkpoint_meta": checkpoint_meta,
        "checkpoint_sizes": sizes,
        "artifact_presence": artifact_presence,
        "history_finite": history_finite,
        "prediction_logits_finite": bool(pred and pred["logits_finite"]),
        "weighted_distill_to_cls_ratio_best": weighted_ratio,
        "raw_distill_to_cls_ratio_best": raw_ratio,
        "total_seconds": float(sum(float(r.get("epoch_seconds", 0.0)) for r in history)),
        "mean_epoch_seconds": float(np.mean([float(r.get("epoch_seconds", 0.0)) for r in history])),
    }


def no_teacher_snapshot() -> dict[str, Any]:
    path = NO_TEACHER["path"]
    out = {"provided": NO_TEACHER}
    if path.exists():
        out["best_summary"] = load_json(path / "metrics" / "best_summary.json")
        out["val_metrics"] = load_json(path / "metrics" / "val_metrics.json")
        out["pred"] = read_predictions(path / "metrics" / "val_predictions.csv")
        out["cfg"] = load_yaml(path / "resolved_config.yaml")
    return out


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def write_summary_csv(results: list[dict[str, Any]]) -> None:
    fields = [
        "run_name",
        "alpha",
        "temperature",
        "best_epoch",
        "best_val_macro_f1",
        "val_accuracy",
        "val_weighted_f1",
        "angry_f1",
        "disgust_f1",
        "fear_f1",
        "happy_f1",
        "sad_f1",
        "surprise_f1",
        "neutral_f1",
        "pred_distribution",
        "train_cls_loss_best",
        "train_distill_loss_best",
        "train_motif_loss_best",
        "distill_to_cls_ratio_best",
        "last_val_macro_f1",
        "teacher_metrics_available",
        "teacher_val_macro_f1_if_available",
        "decision_notes",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            bm = r["best_metrics"]
            vm = r["val_metrics"]
            pc = r["per_class_f1"]
            pred_dist = (r["pred"] or {}).get("pred_distribution")
            teacher_available = bool(r["teacher"].get("available"))
            note = (
                "alpha=0.2 is better than alpha=0.5 but still below no-teacher macro-F1 reference"
                if r["short"] == "a02"
                else "alpha=0.5 has stronger weighted distill pressure and worse macro-F1"
            )
            writer.writerow(
                {
                    "run_name": r["name"],
                    "alpha": csv_value(r["alpha"]),
                    "temperature": csv_value(r["temperature"]),
                    "best_epoch": r["best_epoch"],
                    "best_val_macro_f1": csv_value(r["best_summary"]["best_val_macro_f1"]),
                    "val_accuracy": csv_value(vm.get("accuracy")),
                    "val_weighted_f1": csv_value(vm.get("weighted_f1")),
                    "angry_f1": csv_value(pc["Angry"]),
                    "disgust_f1": csv_value(pc["Disgust"]),
                    "fear_f1": csv_value(pc["Fear"]),
                    "happy_f1": csv_value(pc["Happy"]),
                    "sad_f1": csv_value(pc["Sad"]),
                    "surprise_f1": csv_value(pc["Surprise"]),
                    "neutral_f1": csv_value(pc["Neutral"]),
                    "pred_distribution": csv_value(pred_dist),
                    "train_cls_loss_best": csv_value(bm.get("train_cls_loss")),
                    "train_distill_loss_best": csv_value(bm.get("train_distill_loss")),
                    "train_motif_loss_best": csv_value(bm.get("train_motif_loss")),
                    "distill_to_cls_ratio_best": csv_value(r["weighted_distill_to_cls_ratio_best"]),
                    "last_val_macro_f1": csv_value(r["history"][-1].get("val_macro_f1")),
                    "teacher_metrics_available": str(teacher_available).lower(),
                    "teacher_val_macro_f1_if_available": csv_value(r["teacher"].get("teacher_val_macro_f1")),
                    "decision_notes": note,
                }
            )


def write_report(results: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    a05 = next(r for r in results if r["short"] == "a05")
    a02 = next(r for r in results if r["short"] == "a02")
    lines: list[str] = []
    lines.append("# D9-TGMS Distillation Analysis")
    lines.append("")
    lines.append("## 1. Run validity")
    lines.append("")
    lines.append("| run | complete artifacts | epochs | best epoch | last epoch | monitor | alpha | teacher dir | feature B | MR KxK | finite |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |")
    for r in results:
        cfg = r["cfg"]
        all_artifacts = all(r["artifact_presence"].values())
        feature = cfg.get("feature_ablation") or {}
        node_indices = feature.get("node_indices")
        edge_indices = feature.get("edge_indices")
        model = cfg.get("model") or {}
        relation = model.get("motif_relation_classifier") or {}
        monitor = (cfg.get("checkpoint") or {}).get("monitor")
        finite = r["history_finite"] and r["prediction_logits_finite"]
        lines.append(
            f"| {r['short']} | {all_artifacts} | {len(r['history'])}/30 | {r['best_epoch']} | {r['last_epoch']} | "
            f"{monitor} | {r['alpha']} | `{(cfg.get('distillation') or {}).get('teacher_probs_dir')}` | "
            f"node={node_indices}, edge={edge_indices} | pooled_mlp, use_attention={((model.get('relation_encoder') or {}).get('use_attention'))} | {finite} |"
        )
    lines.append("")
    lines.append("- Both runs have `initial.pth`, `best.pth`, `last.pth`, `d9_history.csv`, `val_metrics.jsonl`, `best_summary.json`, `val_metrics.json`, `resolved_config.yaml`, `val_predictions.csv`, and confusion-matrix PNGs.")
    lines.append("- Both completed all 30 requested epochs; no early stop was triggered. Best checkpoint is selected by `val_macro_f1`.")
    lines.append("- Feature B is correctly active: node indices `[0, 1, 2]`, edge indices `[0, 1, 2, 3, 4]`, model dims `node_dim=3`, `edge_dim=5`.")
    lines.append("- MR KxK is not enabled in these configs; relation head is `type: pooled_mlp`, `pooling: selection_weighted`, and `relation_encoder.use_attention=false`.")
    lines.append("")
    lines.append("Checkpoint metadata:")
    lines.append("")
    lines.append("| run | checkpoint | loaded | epoch | best_metric_name | best_metric_mode | best_metric | val_macro_f1 in ckpt | size MB |")
    lines.append("| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |")
    for r in results:
        for ckpt_name in ("initial.pth", "best.pth", "last.pth"):
            meta = r["checkpoint_meta"][ckpt_name]
            lines.append(
                f"| {r['short']} | {ckpt_name} | {meta.get('loaded')} | {fmt(meta.get('epoch'), 0)} | "
                f"{fmt(meta.get('best_metric_name'))} | {fmt(meta.get('best_metric_mode'))} | "
                f"{fmt(meta.get('best_metric'), 6)} | {fmt(meta.get('metrics_val_macro_f1'), 6)} | "
                f"{meta.get('size_bytes', 0) / (1024 * 1024):.2f} |"
            )
    unloaded = [
        f"{r['short']}:{name} ({meta.get('load_error')})"
        for r in results
        for name, meta in r["checkpoint_meta"].items()
        if not meta.get("loaded")
    ]
    if unloaded:
        lines.append("")
        lines.append("- Checkpoint load limitations: " + "; ".join(unloaded))
    lines.append("")
    lines.append("## 2. Teacher probs / teacher quality")
    lines.append("")
    for r in results:
        teacher = r["teacher"]
        lines.append(f"### {r['short']}")
        lines.append("")
        lines.append(f"- Resolved teacher path in config: `{(r['cfg'].get('distillation') or {}).get('teacher_probs_dir')}`.")
        lines.append(f"- Localized path checked: `{teacher.get('teacher_dir', r['teacher_dir'])}`.")
        if teacher.get("available"):
            lines.append("- Teacher files are present locally.")
            for split in ("train", "val"):
                stats = teacher.get(f"{split}_stats")
                if stats:
                    lines.append(
                        f"- {split}: probs={stats['probs_shape']}, labels={stats['labels_shape']}, "
                        f"row_sum_err={stats['row_sum_max_err']:.2e}, non_finite={stats['non_finite']}, "
                        f"label_dist={stats['label_distribution']}, pred_dist={stats['pred_distribution']}, "
                        f"conf_mean={stats['max_prob_mean']:.4f}, entropy_mean={stats['entropy_mean']:.4f}."
                    )
        else:
            lines.append(f"- Teacher files are not present in the extracted local artifact set: {teacher.get('reason')}.")
        lines.append(f"- Notebook summary field `teacher_metrics`: `{r['notebook_summary'].get('teacher_metrics')}`.")
        lines.append("")
    lines.append("- `teacher_metrics=null` is explained by the Kaggle notebook summary logic for train modes: it sets `teacher_metrics = None` outside `RUN_MODE == \"teacher_probs\"`. The generator script itself writes `teacher_metrics.json`, but that teacher-probs folder was not included in these two run folders.")
    lines.append("- Teacher probs were still loaded during training: `train_distill_loss` is non-zero every epoch and `train_teacher_conf_mean` is about 0.72. The training code also checks teacher labels against batch labels and raises on mismatch; because both 30-epoch runs completed, train-batch label alignment passed for the sampled batches. Current extracted artifacts do not allow a fresh full teacher quality or teacher val macro-F1 check.")
    lines.append("")
    lines.append("## 3. Loss curves")
    lines.append("")
    for r in results:
        bm = r["best_metrics"]
        weighted = r["alpha"] * bm["train_distill_loss"]
        lines.append(f"### {r['short']}")
        lines.append("")
        lines.append(
            f"- Best loss scale: CE={bm['train_cls_loss']:.4f}, KL={bm['train_distill_loss']:.4f}, "
            f"alpha*KL={weighted:.4f}, motif={bm['train_motif_loss']:.5f}, "
            f"(alpha*KL)/CE={r['weighted_distill_to_cls_ratio_best']:.3f}, raw KL/CE={r['raw_distill_to_cls_ratio_best']:.3f}."
        )
        first = r["history"][0]
        last = r["history"][-1]
        lines.append(
            f"- CE changes {first['train_cls_loss']:.4f} -> {last['train_cls_loss']:.4f}; "
            f"KL changes {first['train_distill_loss']:.4f} -> {last['train_distill_loss']:.4f}; "
            f"motif loss stays near {first['train_motif_loss']:.5f} -> {last['train_motif_loss']:.5f}."
        )
        lines.append(
            f"- Val macro-F1 peaks at epoch {r['best_epoch']} ({r['best_summary']['best_val_macro_f1']:.6f}) "
            f"and ends at epoch {r['last_epoch']} ({last['val_macro_f1']:.6f})."
        )
        lines.append("")
        lines.append(important_epoch_table(r["history"], r["best_epoch"]))
        lines.append("")
    lines.append("- a05 has very strong distillation pressure: alpha*KL is slightly larger than CE at best epoch. Its best appears early at epoch 11, then never clearly recovers.")
    lines.append("- a02 has moderate pressure: alpha*KL is about 45% of CE at best epoch. It peaks later at epoch 23, but the last 7 epochs decay/oscillate around 0.15-0.16 rather than trending upward cleanly.")
    lines.append("")
    lines.append("## 4. Metrics and class collapse")
    lines.append("")
    lines.append("| run | macro F1 | accuracy | weighted F1 | true distribution | pred distribution | per-class F1 |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- | --- |")
    for r in results:
        vm = r["val_metrics"]
        pred = r["pred"] or {}
        pc_text = ", ".join(f"{cls}={r['per_class_f1'][cls]:.4f}" for cls in CLASSES)
        lines.append(
            f"| {r['short']} | {vm['macro_f1']:.6f} | {vm['accuracy']:.6f} | {vm['weighted_f1']:.6f} | "
            f"{pred.get('true_distribution')} | {pred.get('pred_distribution')} | {pc_text} |"
        )
    lines.append("")
    lines.append("- Disgust is dead in both runs: 0 predictions and F1=0.")
    lines.append("- a05 nearly kills Angry: only 4 Angry predictions, Angry F1=0.0091. It keeps Neutral higher than a02.")
    lines.append("- a02 recovers Angry substantially: 135 Angry predictions, Angry F1=0.1371. Fear and Neutral are weaker than a05.")
    lines.append("- Both over-predict Happy and Surprise relative to true support. a02 pushes Happy harder (733 predictions vs 383 true), so teacher guidance did not solve class balance.")
    lines.append("")
    lines.append("## 5. Comparison with no-teacher D9")
    lines.append("")
    b_best = baseline.get("best_summary", {})
    b_metrics = baseline.get("val_metrics", {})
    b_pred = baseline.get("pred") or {}
    lines.append("| run | macro F1 | accuracy | weighted F1 | val samples | note |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    lines.append(f"| no-teacher reference provided | {NO_TEACHER['provided_macro_f1']:.4f} | {NO_TEACHER['provided_accuracy']:.4f} | {NO_TEACHER['provided_weighted_f1']:.4f} | NA | user-provided reference |")
    if b_metrics:
        lines.append(f"| no-teacher local artifact | {b_metrics['macro_f1']:.6f} | {b_metrics['accuracy']:.6f} | {b_metrics['weighted_f1']:.6f} | {b_pred.get('n')} | `d9_rg_b_no_mr_pooled_mlp_diag20_cap300v100` |")
    lines.append(f"| TGMS alpha 0.5 | {a05['val_metrics']['macro_f1']:.6f} | {a05['val_metrics']['accuracy']:.6f} | {a05['val_metrics']['weighted_f1']:.6f} | {(a05['pred'] or {}).get('n')} | below no-teacher macro-F1 |")
    lines.append(f"| TGMS alpha 0.2 | {a02['val_metrics']['macro_f1']:.6f} | {a02['val_metrics']['accuracy']:.6f} | {a02['val_metrics']['weighted_f1']:.6f} | {(a02['pred'] or {}).get('n')} | closest TGMS run, still below no-teacher macro-F1 |")
    lines.append("")
    lines.append("- Distillation slightly improves accuracy and weighted F1 versus the provided no-teacher reference, but does not improve macro F1.")
    lines.append("- The local no-teacher artifact has the same macro-F1 value but only 800 val samples in its `val_metrics.json`, while TGMS has 1600. Treat the baseline as a reference, not a perfectly identical validation slice unless the original no-teacher 1600-sample artifact is supplied.")
    lines.append("")
    lines.append("## 6. Motif metrics / output")
    lines.append("")
    lines.append("| run | motif figures | motif loss best | selected border | foreground | entropy | effective count |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        bm = r["best_metrics"]
        motif_figures = len(list((r["path"] / "figures").glob("motif*"))) + len(list((r["path"] / "figures").glob("*motif*")))
        lines.append(
            f"| {r['short']} | {motif_figures} | {bm['train_motif_loss']:.6f} | "
            f"{bm.get('val_selected_border_mass_mean', 0.0):.6f} | {bm.get('val_selected_foreground_mass_mean', 0.0):.6f} | "
            f"{bm.get('val_selection_entropy', 0.0):.6f} | {bm.get('val_selection_effective_count', 0.0):.6f} |"
        )
    lines.append("")
    lines.append("- No motif visualization files are present in these extracted run outputs; only confusion-matrix figures are present.")
    lines.append("- Motif loss is nearly constant and tiny compared with CE/KL. Available motif stats show foreground mass is 0.0, high entropy, and effective count near 15/16. This is not enough to claim motif quality improved.")
    lines.append("")
    lines.append("## 7. Runtime")
    lines.append("")
    lines.append("| run | total seconds | hours | mean sec/epoch | best epoch seconds |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            f"| {r['short']} | {r['total_seconds']:.1f} | {r['total_seconds']/3600.0:.2f} | "
            f"{r['mean_epoch_seconds']:.1f} | {r['best_metrics']['epoch_seconds']:.1f} |"
        )
    lines.append("")
    lines.append("- Runtime is about 1.9 hours per 30-epoch capped run. A longer 40-epoch cap300 run is feasible but expensive for a signal that has not crossed the no-teacher macro-F1 reference.")
    lines.append("")
    lines.append("## 8. Final decision")
    lines.append("")
    lines.append("Decision: stop the current TGMS setting as a result candidate. alpha=0.2 is better than alpha=0.5, but neither beats the no-teacher macro-F1 reference, Disgust remains dead, and teacher quality metrics are missing from the extracted run artifacts.")
    lines.append("")
    lines.append("Use D7/D8B ensemble as the performance result path if the goal is score. Keep D9-TGMS as a documented negative/neutral experiment: it improves accuracy/weighted F1 slightly, but does not improve macro-F1 or difficult-class behavior.")
    lines.append("")
    lines.append("## 9. Recommended next action")
    lines.append("")
    lines.append("If one more D9 run is still worth spending time on, run exactly one small-alpha distillation test: `alpha=0.1`, `T=2`, same cap300v100, same Feature B, same no-MR pooled MLP, and copy/retain `teacher_metrics.json` plus teacher manifest in the output bundle. Do not run alpha=0.5-style pressure again.")
    lines.append("")
    lines.append("If time is limited or the target is best validation performance, do not run more TGMS now; switch effort back to packaging and reporting the D7/D8B ensemble.")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results = [analyze_run(run) for run in RUNS]
    baseline = no_teacher_snapshot()
    write_summary_csv(results)
    write_report(results, baseline)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
