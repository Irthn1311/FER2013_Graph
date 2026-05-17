"""Create a Markdown report from D12A run diagnostics and optional exports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest(path: Path, pattern: str) -> Optional[Path]:
    matches = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _best_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not history:
        return {}
    return max(history, key=lambda row: float(row.get("val_macro_f1", -1.0) or -1.0))


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    y_true: List[int] = []
    y_pred: List[int] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((7, 7), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < 7 and 0 <= int(p) < 7:
            cm[int(t), int(p)] += 1
    return cm


def _class_report_value(report: Dict[str, Any] | None, class_name: str, key: str) -> Any:
    if not isinstance(report, dict):
        return None
    entry = report.get(class_name) or report.get(str(EMOTION_NAMES.index(class_name)))
    if isinstance(entry, dict):
        return entry.get(key)
    return None


def _pred_count_from_history(row: Dict[str, Any], prefix: str = "val_pred_count") -> List[int]:
    return [int(row.get(f"{prefix}_{i}", 0) or 0) for i in range(7)]


def _micro_summary(run_dir: Path) -> Dict[str, Any] | None:
    direct = run_dir / "micro_diagnostics" / "micro_diagnostics_summary.json"
    if direct.exists():
        return _load_json(direct)
    found = _find_latest(run_dir, "**/micro_diagnostics_summary.json")
    return _load_json(found) if found else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    evaluation_dir = run_dir / "evaluation"
    history = _load_json(run_dir / "training_history.json") or []
    metrics = _load_json(evaluation_dir / "metrics.json") or {}
    report = _load_json(evaluation_dir / "classification_report.json")
    report_local = _load_json(evaluation_dir / "classification_report_local.json")
    d6_diag = _load_json(evaluation_dir / "d6b_diagnostics.json") or {}
    micro = _micro_summary(run_dir) or {}
    micro_mean = micro.get("diagnostics_mean", {}) if isinstance(micro, dict) else {}
    y_true, y_pred = _read_predictions(evaluation_dir / "predictions.csv")
    cm = _confusion(y_true, y_pred) if len(y_true) else None
    best = _best_history(history)
    pred_count = metrics.get("pred_count") or _pred_count_from_history(best)
    active_classes = sum(int(v) > 0 for v in pred_count)

    lines: List[str] = []
    lines.append("# D12 Micro Diagnostics Report")
    lines.append("")
    lines.append(f"Run dir: `{run_dir}`")
    lines.append("")
    lines.append("## 1. Main metrics")
    lines.append("")
    lines.append(f"- best_epoch: `{int(best.get('epoch', -1)) if best else 'missing'}`")
    lines.append(f"- best_val_macro_f1: `{best.get('val_macro_f1', 'missing')}`")
    lines.append(f"- eval_macro_f1: `{metrics.get('macro_f1', 'missing')}`")
    lines.append(f"- eval_accuracy: `{metrics.get('accuracy', 'missing')}`")
    lines.append(f"- pred_count: `{pred_count}`")
    lines.append(f"- active_classes: `{active_classes}`")
    lines.append("")
    lines.append("## 2. Rare class status")
    lines.append("")
    for class_name in ("Angry", "Disgust"):
        lines.append(
            f"- {class_name}: precision=`{_class_report_value(report, class_name, 'precision')}`, "
            f"recall=`{_class_report_value(report, class_name, 'recall')}`, "
            f"f1=`{_class_report_value(report, class_name, 'f1-score')}`"
        )
    if cm is not None:
        for idx in (0, 1):
            lines.append(f"- confusion row {EMOTION_NAMES[idx]}: `{cm[idx].tolist()}`")
    lines.append("")
    lines.append("## 3. Local vs full")
    lines.append("")
    local_pred_count = metrics.get("pred_count_local") or _pred_count_from_history(best, "val_pred_count_local")
    lines.append(f"- pred_count_main: `{pred_count}`")
    lines.append(f"- pred_count_local: `{local_pred_count}`")
    lines.append(f"- macro_f1_main: `{metrics.get('macro_f1', best.get('val_macro_f1', 'missing'))}`")
    lines.append(f"- macro_f1_local: `{metrics.get('macro_f1_local', best.get('val_macro_f1_local', 'missing'))}`")
    if report_local:
        lines.append(
            "- Disgust local F1: "
            f"`{_class_report_value(report_local, 'Disgust', 'f1-score')}`"
        )
    main_disgust = int(pred_count[1]) if len(pred_count) > 1 else 0
    local_disgust = int(local_pred_count[1]) if len(local_pred_count) > 1 else 0
    if local_disgust > main_disgust:
        local_conclusion = "local branch has more Disgust signal than full logits; inspect fusion/global/classifier."
    elif local_disgust == 0 and main_disgust == 0:
        local_conclusion = "neither local nor full logits predict Disgust; suspect encoder/slot/local representation."
    else:
        local_conclusion = "local/full split is inconclusive from available artifacts."
    lines.append(f"- conclusion: {local_conclusion}")
    lines.append("")
    lines.append("## 4. Micro diagnostics")
    lines.append("")
    diag_keys = [
        "encoder_scale1_std",
        "encoder_scale2_std",
        "encoder_scale2_delta_ratio",
        "cos_eye_delta",
        "cos_nose_mouth_delta",
        "cos_center_delta",
        "cos_border_delta",
        "slot_area_entropy",
        "effective_slots",
        "class_part_similarity_disgust_angry",
    ]
    for key in diag_keys:
        value = micro_mean.get(key, best.get(f"val_diag_{key}", best.get(key, d6_diag.get(key, "missing"))))
        lines.append(f"- {key}: `{value}`")
    if d6_diag.get("avg_class_part_attn") is not None:
        avg_attn = np.asarray(d6_diag["avg_class_part_attn"], dtype=np.float32)
        if avg_attn.shape[0] >= 2:
            sim = float(np.dot(avg_attn[1], avg_attn[0]) / (np.linalg.norm(avg_attn[1]) * np.linalg.norm(avg_attn[0]) + 1e-8))
            lines.append(f"- class_part_similarity_disgust_angry_from_eval: `{sim}`")
    lines.append("")
    lines.append("## 5. Interpretation")
    lines.append("")
    scale2_delta = micro_mean.get("encoder_scale2_delta_ratio")
    cos_eye_delta = micro_mean.get("cos_eye_delta")
    slot_entropy = micro_mean.get("slot_area_entropy", d6_diag.get("slot_area_entropy"))
    if scale2_delta == "missing" or scale2_delta is None:
        lines.append("- Scale2 smoothing: missing micro diagnostics; run visualization/export with diagnostics enabled.")
    elif float(scale2_delta) > 0.5 or (cos_eye_delta is not None and float(cos_eye_delta) > 0.05):
        lines.append("- Scale2 smoothing: suspicious; scale2/context appears to change or homogenize node embeddings strongly.")
    else:
        lines.append("- Scale2 smoothing: no strong smoothing signal from current scalar diagnostics.")
    if slot_entropy is None or slot_entropy == "missing":
        lines.append("- Slot sharpness: missing slot entropy diagnostics.")
    elif float(slot_entropy) > 1.8:
        lines.append("- Slot sharpness: slots may be diffuse/uniform; inspect attention heatmaps.")
    else:
        lines.append("- Slot sharpness: slot entropy is not obviously excessive.")
    lines.append(f"- Rare-class locality: {local_conclusion}")
    lines.append("")

    out_path = Path(args.output) if args.output else run_dir / "D12_Micro_Diagnostics_Report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
