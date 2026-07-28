"""Orchestrate the bounded OFIX7-mid PyTorch/TensorFlow training audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "tools" / "ofix7_mid_cross_framework_worker.py"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "d16_analysis"
    / "lap_gnn_tensorflow_cross_framework_training_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["dropout_off", "shared_dropout"],
        choices=["dropout_off", "native_dropout", "shared_dropout"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-seed", type=int, default=20260729)
    parser.add_argument("--shared-dropout-seed", type=int, default=20260728)
    parser.add_argument("--pytorch-env", default="fer-graph")
    parser.add_argument("--tensorflow-env", default="lap-gnn-tf")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_manifest(
    steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 1 <= batch_size <= 8:
        raise ValueError("batch_size must be between 1 and 8")
    rng = np.random.default_rng(seed)
    queue: list[int] = []
    records = []
    for step in range(1, steps + 1):
        while len(queue) < batch_size:
            values = np.arange(8, dtype=np.int64)
            rng.shuffle(values)
            queue.extend(int(value) for value in values)
        graph_ids = queue[:batch_size]
        del queue[:batch_size]
        records.append({"step": step, "graph_ids": graph_ids})
    return {
        "schema": "ofix7_mid_cross_framework_audit_v1",
        "steps": records,
        "steps_count": steps,
        "batch_size": batch_size,
        "manifest_seed": seed,
        "fixture": "validation_assets/golden/graph_batch.npz",
        "fixture_graph_count": 8,
    }


def conda_command(environment: str, arguments: list[str]) -> list[str]:
    conda = shutil.which("conda")
    if conda is None:
        raise FileNotFoundError("conda executable was not found")
    return [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        environment,
        "python",
        "-B",
        *arguments,
    ]


def run_worker(
    framework: str,
    mode: str,
    environment: str,
    manifest_path: Path,
    initial_state: Path,
    output_dir: Path,
    seed: int,
    shared_dropout_seed: int,
    export_initial_state: bool,
) -> dict[str, Any]:
    command = conda_command(
        environment,
        [
            str(WORKER),
            "--framework",
            framework,
            "--mode",
            mode,
            "--manifest",
            str(manifest_path),
            "--initial-state",
            str(initial_state),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(seed),
            "--shared-dropout-seed",
            str(shared_dropout_seed),
            *(["--export-initial-state"] if export_initial_state else []),
        ],
    )
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    duration = time.perf_counter() - started
    (output_dir / "worker.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"{framework}/{mode} failed after {duration:.1f}s\n{tail}"
        )
    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    payload["duration_sec"] = duration
    return payload


def state_difference(left_path: Path, right_path: Path) -> dict[str, float]:
    with np.load(left_path, allow_pickle=False) as left, np.load(
        right_path, allow_pickle=False
    ) as right:
        if set(left.files) != set(right.files):
            missing_left = sorted(set(right.files) - set(left.files))
            missing_right = sorted(set(left.files) - set(right.files))
            raise KeyError(
                f"State keys differ: missing_left={missing_left}, "
                f"missing_right={missing_right}"
            )
        squared_difference = 0.0
        squared_reference = 0.0
        maximum = 0.0
        count = 0
        for key in left.files:
            left_value = np.asarray(left[key], dtype=np.float64)
            right_value = np.asarray(right[key], dtype=np.float64)
            difference = left_value - right_value
            squared_difference += float(np.sum(np.square(difference)))
            squared_reference += float(np.sum(np.square(left_value)))
            if difference.size:
                maximum = max(maximum, float(np.max(np.abs(difference))))
            count += int(difference.size)
    return {
        "parameter_max_abs": maximum,
        "parameter_relative_l2": (
            squared_difference**0.5 / max(squared_reference**0.5, 1e-30)
        ),
        "parameter_count": count,
    }


def compare_mode(
    mode: str,
    pytorch: dict[str, Any],
    tensorflow: dict[str, Any],
    mode_root: Path,
) -> dict[str, Any]:
    pytorch_records = {int(row["step"]): row for row in pytorch["records"]}
    tensorflow_records = {int(row["step"]): row for row in tensorflow["records"]}
    if set(pytorch_records) != set(tensorflow_records):
        raise ValueError(f"Step mismatch for {mode}")
    rows = []
    for step in sorted(pytorch_records):
        pt = pytorch_records[step]
        tf = tensorflow_records[step]
        pt_logits = np.asarray(pt["logits"], dtype=np.float64)
        tf_logits = np.asarray(tf["logits"], dtype=np.float64)
        rows.append(
            {
                "mode": mode,
                "step": step,
                "loss_pytorch": pt["loss"],
                "loss_tensorflow": tf["loss"],
                "loss_abs_diff": abs(float(pt["loss"]) - float(tf["loss"])),
                "gradient_norm_pytorch": pt["gradient_norm"],
                "gradient_norm_tensorflow": tf["gradient_norm"],
                "gradient_norm_abs_diff": abs(
                    float(pt["gradient_norm"]) - float(tf["gradient_norm"])
                ),
                "logit_max_abs": float(np.max(np.abs(pt_logits - tf_logits))),
                "logit_mean_abs": float(np.mean(np.abs(pt_logits - tf_logits))),
                "prediction_agreement": float(
                    np.mean(
                        pt_logits.argmax(axis=1)
                        == tf_logits.argmax(axis=1)
                    )
                ),
                "dropout_calls_pytorch": pt["dropout_calls"],
                "dropout_calls_tensorflow": tf["dropout_calls"],
            }
        )
    snapshot_rows = []
    for pytorch_state in sorted((mode_root / "pytorch").glob("state_step_*.npz")):
        tensorflow_state = mode_root / "tensorflow" / pytorch_state.name
        if not tensorflow_state.exists():
            continue
        step = int(pytorch_state.stem.rsplit("_", 1)[1])
        snapshot_rows.append(
            {
                "mode": mode,
                "step": step,
                **state_difference(pytorch_state, tensorflow_state),
            }
        )
    trace_equal = (
        pytorch["first_step_dropout_trace"]
        == tensorflow["first_step_dropout_trace"]
    )
    return {
        "mode": mode,
        "step_rows": rows,
        "state_rows": snapshot_rows,
        "first_step_dropout_trace_equal": trace_equal,
        "pytorch_first_step_dropout_trace": pytorch[
            "first_step_dropout_trace"
        ],
        "tensorflow_first_step_dropout_trace": tensorflow[
            "first_step_dropout_trace"
        ],
        "pytorch_duration_sec": pytorch["duration_sec"],
        "tensorflow_duration_sec": tensorflow["duration_sec"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# OFIX7-mid bounded cross-framework training audit",
        "",
        "This diagnostic uses the same exported PyTorch seed42 initial state and",
        "the same registered sequence of golden graph samples. It does not train",
        "on FER2013 or modify either production trainer.",
        "",
        "## Configuration",
        "",
        f"- steps: `{summary['steps']}`",
        f"- batch size: `{summary['batch_size']}`",
        f"- modes: `{', '.join(summary['modes'])}`",
        f"- initial state: `{summary['initial_state']}`",
        "",
    ]
    for mode in summary["modes"]:
        result = summary["results"][mode]
        lines.extend(
            [
                f"## {mode}",
                "",
                f"- shared dropout trace equal: "
                f"`{result['first_step_dropout_trace_equal']}`",
                f"- PyTorch duration: `{result['pytorch_duration_sec']:.1f}s`",
                f"- TensorFlow duration: `{result['tensorflow_duration_sec']:.1f}s`",
                "",
                "| step | loss diff | logit max abs | prediction agreement | "
                "parameter max abs | parameter relative L2 |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        states = {int(row["step"]): row for row in result["state_rows"]}
        selected_steps = sorted(states)
        records = {int(row["step"]): row for row in result["step_rows"]}
        for step in selected_steps:
            record = records.get(step)
            state = states[step]
            if record is None:
                lines.append(
                    f"| {step} | n/a | n/a | n/a | "
                    f"{state['parameter_max_abs']:.6g} | "
                    f"{state['parameter_relative_l2']:.6g} |"
                )
            else:
                lines.append(
                    f"| {step} | {record['loss_abs_diff']:.6g} | "
                    f"{record['logit_max_abs']:.6g} | "
                    f"{100.0 * record['prediction_agreement']:.2f}% | "
                    f"{state['parameter_max_abs']:.6g} | "
                    f"{state['parameter_relative_l2']:.6g} |"
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This is a bounded training-operator diagnostic over the eight golden",
            "graphs. It is not an accuracy experiment and does not estimate FER2013",
            "generalization.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --overwrite for this diagnostic directory."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.steps, args.batch_size, args.manifest_seed)
    manifest_path = output_dir / "batch_manifest.json"
    atomic_json(manifest_path, manifest)
    initial_state = output_dir / "pytorch_seed42_initial_state.npz"
    results = {}
    export_initial = True
    for mode in args.modes:
        mode_root = output_dir / mode
        pytorch_dir = mode_root / "pytorch"
        tensorflow_dir = mode_root / "tensorflow"
        pytorch_dir.mkdir(parents=True, exist_ok=True)
        tensorflow_dir.mkdir(parents=True, exist_ok=True)
        pytorch = run_worker(
            "pytorch",
            mode,
            args.pytorch_env,
            manifest_path,
            initial_state,
            pytorch_dir,
            args.seed,
            args.shared_dropout_seed,
            export_initial,
        )
        export_initial = False
        tensorflow = run_worker(
            "tensorflow",
            mode,
            args.tensorflow_env,
            manifest_path,
            initial_state,
            tensorflow_dir,
            args.seed,
            args.shared_dropout_seed,
            False,
        )
        results[mode] = compare_mode(mode, pytorch, tensorflow, mode_root)
        atomic_json(output_dir / f"comparison_{mode}.json", results[mode])
        write_csv(
            output_dir / f"step_comparison_{mode}.csv",
            results[mode]["step_rows"],
        )
        write_csv(
            output_dir / f"state_comparison_{mode}.csv",
            results[mode]["state_rows"],
        )
    summary = {
        "schema": "ofix7_mid_cross_framework_audit_summary_v1",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "modes": args.modes,
        "seed": args.seed,
        "manifest_seed": args.manifest_seed,
        "shared_dropout_seed": args.shared_dropout_seed,
        "initial_state": str(initial_state),
        "results": results,
    }
    atomic_json(output_dir / "summary.json", summary)
    write_report(output_dir / "01_cross_framework_training_audit.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "steps": args.steps,
                "modes": args.modes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
