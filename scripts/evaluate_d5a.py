"""Evaluate a trained D5A checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (
    apply_cli_overrides,
    build_dataloader,
    dump_json,
    load_checkpoint_model,
    load_config,
    output_root_from_checkpoint,
    resolve_path,
)

from evaluation.evaluator import evaluate_model, save_confusion_matrix, save_predictions_csv
from evaluation.evaluator import save_example_grid


def run_evaluate(config, checkpoint=None):
    paths = config.get("paths", {})
    if checkpoint is not None:
        output_root = output_root_from_checkpoint(checkpoint) or resolve_path(paths.get("resolved_output_root") or paths.get("output_root", "outputs"))
    else:
        output_root = resolve_path(paths.get("resolved_output_root") or paths.get("output_root", "outputs"))
    checkpoint = checkpoint or output_root / "checkpoints" / "best.pth"
    model, device, _ = load_checkpoint_model(config, checkpoint)
    loader = build_dataloader(config, split="test", shuffle=False)
    metrics = evaluate_model(
        model,
        loader,
        device=device,
        max_batches=config.get("training", {}).get("max_test_batches"),
    )
    eval_dir = output_root / "evaluation"
    cm_path = eval_dir / "confusion_matrix.png"
    correct_path = eval_dir / "correct_examples.png"
    wrong_path = eval_dir / "wrong_examples.png"
    save_confusion_matrix(metrics["confusion_matrix"], cm_path)
    save_predictions_csv(metrics, eval_dir / "predictions.csv")
    save_example_grid(metrics.get("correct_examples", []), correct_path, "10 correct predictions")
    save_example_grid(metrics.get("wrong_examples", []), wrong_path, "10 wrong predictions")
    dump_json(
        {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "pred_count": metrics["pred_count"],
            "classification_report": metrics["classification_report"],
        },
        eval_dir / "metrics.json",
    )
    if metrics.get("d6b_diagnostics"):
        dump_json(metrics["d6b_diagnostics"], eval_dir / "d6b_diagnostics.json")
    report = metrics["classification_report"]
    dump_json(report, eval_dir / "classification_report.json")
    report_lines = []
    for label, values in report.items():
        if isinstance(values, dict):
            report_lines.append(
                f"{label:<14} "
                f"precision={values.get('precision', 0.0):.4f} "
                f"recall={values.get('recall', 0.0):.4f} "
                f"f1={values.get('f1-score', 0.0):.4f} "
                f"support={values.get('support', 0.0):.0f}"
            )
        else:
            report_lines.append(f"{label:<14} {values:.4f}")
    (eval_dir / "classification_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        "\n=======================================================\n"
        "TEST SET EVALUATION\n"
        "======================================================="
    )
    print(f"Accuracy:    {metrics['accuracy'] * 100.0:.2f}%")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"pred_count:  {metrics['pred_count']}")
    print("\nClassification report:")
    for label, values in report.items():
        if isinstance(values, dict):
            print(
                f"{label:<14} "
                f"precision={values.get('precision', 0.0):.4f} "
                f"recall={values.get('recall', 0.0):.4f} "
                f"f1={values.get('f1-score', 0.0):.4f} "
                f"support={values.get('support', 0.0):.0f}"
            )
        else:
            print(f"{label:<14} {values:.4f}")
    print(f"Confusion matrix saved: {cm_path}")
    print(f"Classification report saved: {eval_dir / 'classification_report.txt'}")
    print(f"Correct examples saved: {correct_path}")
    print(f"Wrong examples saved: {wrong_path}")
    print(f"Evaluation outputs: {eval_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d5a.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    run_evaluate(config, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
