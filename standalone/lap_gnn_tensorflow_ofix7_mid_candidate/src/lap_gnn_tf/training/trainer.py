"""Explicit GradientTape loop preserving selection and scheduler ordering."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import tensorflow as tf
import yaml

from lap_gnn_tf.config import canonical_config_hash, load_config, validate_locked_config
from lap_gnn_tf.data.graph_generator import GraphBatchGenerator
from lap_gnn_tf.model import LapGNN
from lap_gnn_tf.provenance import write_provenance
from lap_gnn_tf.resources import ResourceControls, RuntimeTelemetry
from lap_gnn_tf.seed import seed_everything
from lap_gnn_tf.training.checkpointing import CheckpointPolicy
from lap_gnn_tf.training.artifacts import (
    write_artifact_manifest,
    write_confusion_matrix,
    write_per_class_metrics,
    write_predictions,
    write_training_curves,
)
from lap_gnn_tf.training.early_stopping import ValidationLossEarlyStopping
from lap_gnn_tf.training.evaluator import (
    build_compiled_evaluation_step,
    evaluate_batches,
)
from lap_gnn_tf.training.execution import (
    MAX_REGISTERED_TRAIN_STEP_TRACES,
    apply_gradients_eager_exact,
    build_compiled_gradient_function,
    build_restricted_graph_train_step,
    validate_execution_config,
)
from lap_gnn_tf.training.optimizer import build_optimizer
from lap_gnn_tf.training.plateau import TorchCompatibleReduceLROnPlateau


PROGRESS_INTERVAL_BATCHES = 100


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_history_csv(path: Path, history: list[dict]) -> None:
    if not history:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(history[0])
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def _duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:d}m{secs:02d}s"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _print_epoch_summary(row: dict, policy: CheckpointPolicy) -> None:
    train_macro = row.get("train_macro_f1")
    gap = row.get("train_val_macro_gap_pp")
    saved = ", ".join(row.get("saved") or []) or "none"
    print("=" * 96, flush=True)
    print(
        f"[EPOCH {row['epoch']:03d}] "
        f"train_loss={row['train_loss']:.4f} | "
        f"val_loss={row['val_loss']:.4f} | "
        f"val_acc={_percent(row['val_accuracy'])} | "
        f"val_macro_f1={_percent(row['val_macro_f1'])}",
        flush=True,
    )
    print(
        f"            clean_train_acc={_percent(row.get('train_accuracy'))} | "
        f"clean_train_macro_f1={_percent(train_macro)} | "
        f"macro_gap={('n/a' if gap is None else f'{gap:.2f} pp')} | "
        f"lr={row['lr']:.8f}",
        flush=True,
    )
    print(
        f"            best_macro_f1={_percent(policy.best_macro)} "
        f"(epoch {policy.best_macro_epoch}) | "
        f"best_accuracy={_percent(policy.best_accuracy)} "
        f"(epoch {policy.best_accuracy_epoch}) | checkpoints={saved}",
        flush=True,
    )
    print(
        f"            early_stop_wait={row['early_stopping_wait']}/"
        f"{row['early_stopping_patience']} | "
        f"time train={_duration(row['train_phase_sec'])} "
        f"val={_duration(row['val_phase_sec'])} "
        f"train_eval={_duration(row['train_eval_phase_sec'])} "
        f"total={_duration(row['epoch_time_sec'])}",
        flush=True,
    )
    print("=" * 96, flush=True)


def run_training(
    config_path,
    fer_csv,
    prior_root,
    output_root,
    controls: ResourceControls,
    no_resume: bool = True,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    limit_train_eval_batches: int | None = None,
    limit_test_batches: int | None = None,
    limit_epochs: int | None = None,
):
    if not no_resume:
        raise ValueError("TensorFlow candidate resume is disabled by default and must not be enabled for seed42")
    config = load_config(config_path)
    validate_locked_config(config)
    execution_state = validate_execution_config(config["training"])
    config["data"]["prior_dir"] = str(Path(prior_root).resolve())
    config["data"]["fer_csv"] = str(Path(fer_csv).resolve())
    config["training"]["batch_size"] = int(controls.batch_size)
    config["resources"].update(controls.__dict__)
    output_dir = Path(output_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Fresh TensorFlow output must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8",
    )
    _atomic_json(output_dir / "resolved_config.json", config)
    seed = int(config["seed"])
    seed_everything(seed)
    controls.apply()
    telemetry = RuntimeTelemetry()
    train_data = GraphBatchGenerator(
        prior_root, "train", config, controls.batch_size, seed, True,
        controls.graph_cache_size, telemetry, graph_workers=controls.graph_workers,
        clean_graph_cache_dir=controls.clean_graph_cache_dir,
    )
    val_data = GraphBatchGenerator(
        prior_root, "val", config, controls.eval_batch_size, seed, False,
        controls.graph_cache_size, telemetry, graph_workers=controls.graph_workers,
        clean_graph_cache_dir=controls.clean_graph_cache_dir,
    )
    eval_step = None
    train_eval_data = None
    if bool(config["training"].get("eval_train_metrics", False)):
        eval_config = json.loads(json.dumps(config))
        eval_config["graph"]["prior_corruption"]["enabled"] = False
        train_eval_data = GraphBatchGenerator(
            prior_root, "train", eval_config, controls.eval_batch_size, seed, False,
            controls.graph_cache_size, telemetry, graph_workers=controls.graph_workers,
            clean_graph_cache_dir=controls.clean_graph_cache_dir,
        )
    first_batch = next(iter(train_data.as_dataset(1, limit_batches=1)))
    model = LapGNN()
    model(first_batch, training=False)
    eval_step = build_compiled_evaluation_step(model)
    optimizer = build_optimizer(config)
    optimizer.build(model.trainable_variables)
    model.compile(optimizer=optimizer, run_eagerly=False)
    optimizer_mode = execution_state["optimizer_execution_mode"]
    if optimizer_mode == "restricted_tf_function":
        execute_train_step = build_restricted_graph_train_step(
            model,
            optimizer,
            input_signature=GraphBatchGenerator.output_signature(),
        )
    else:
        if hasattr(optimizer, "inner_optimizer"):
            raise RuntimeError(
                "Mixed-precision eager_exact is not the registered seed42 mode"
            )
        compute_gradients = build_compiled_gradient_function(
            model, training=True
        )

        def execute_train_step(batch):
            loss, _logits, gradients, finite = compute_gradients(batch)
            if not bool(finite.numpy()):
                raise FloatingPointError("Non-finite H1 gradients")
            apply_gradients_eager_exact(
                optimizer, gradients, model.trainable_variables
            )
            return loss
    scheduler_cfg = config["training"]["scheduler"]
    scheduler = TorchCompatibleReduceLROnPlateau(
        optimizer, mode=scheduler_cfg["mode"], factor=scheduler_cfg["factor"],
        patience=scheduler_cfg["patience"], threshold=scheduler_cfg["threshold"],
        min_lr=scheduler_cfg["min_lr"],
    )
    early_cfg = config["training"]["early_stopping"]
    early = ValidationLossEarlyStopping(
        min_epochs=early_cfg["min_epochs_before_stop"], patience=early_cfg["patience"],
    )
    policy = CheckpointPolicy(output_dir)
    signatures = {
        "config": canonical_config_hash(config),
        "graph": config["locked"]["graph_signature"],
        "feature": config["locked"]["feature_signature"],
        "prior": config["locked"]["prior_signature"],
        "dataset_split": config["locked"]["dataset_split_signature"],
    }
    write_provenance(output_dir, config, signatures)
    history = []
    max_epochs = int(config["training"]["max_epochs"])
    if limit_epochs is not None:
        max_epochs = min(max_epochs, int(limit_epochs))
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        total_loss = 0.0
        batches = 0
        total_train_batches = len(train_data)
        if limit_train_batches is not None:
            total_train_batches = min(
                total_train_batches, int(limit_train_batches)
            )
        print(
            f"\n[TRAIN] epoch {epoch}/{max_epochs} | "
            f"batches={total_train_batches} | "
            f"lr={float(optimizer.learning_rate.numpy()):.8f} | "
            f"prior_corruption={train_data.dataset.current_corruption_probability():.2f}",
            flush=True,
        )
        for batch in train_data.as_dataset(
            epoch,
            limit_batches=limit_train_batches,
            prefetch=controls.tf_data_prefetch,
        ):
            step_started = time.perf_counter()
            loss = execute_train_step(batch)
            total_loss += float(loss.numpy())
            batches += 1
            telemetry.train_step_sec.append(time.perf_counter() - step_started)
            trace_count = (
                int(execute_train_step.experimental_get_tracing_count())
                if hasattr(
                    execute_train_step, "experimental_get_tracing_count"
                )
                else 0
            )
            if (
                optimizer_mode == "restricted_tf_function"
                and trace_count > MAX_REGISTERED_TRAIN_STEP_TRACES
            ):
                raise RuntimeError(
                    "Registered train step retraced unexpectedly: "
                    f"{trace_count} concrete traces"
                )
            should_report = (
                batches == 1
                or batches % PROGRESS_INTERVAL_BATCHES == 0
                or batches == total_train_batches
            )
            if should_report:
                recent_steps = telemetry.train_step_sec[
                    -PROGRESS_INTERVAL_BATCHES:
                ]
                recent_builds = telemetry.batch_construction_sec[
                    -PROGRESS_INTERVAL_BATCHES:
                ]
                progress = {
                    "event": "tensorflow_train_progress",
                    "epoch": epoch,
                    "batch": batches,
                    "total_batches": total_train_batches,
                    "elapsed_sec": time.perf_counter() - epoch_started,
                    "average_loss": total_loss / batches,
                    "recent_train_step_sec": (
                        sum(recent_steps) / len(recent_steps)
                    ),
                    "recent_graph_build_sec": (
                        sum(recent_builds) / len(recent_builds)
                        if recent_builds else None
                    ),
                    "train_step_trace_count": trace_count,
                    "resources": telemetry.snapshot(),
                }
                elapsed = float(progress["elapsed_sec"])
                eta = (
                    elapsed / max(batches, 1)
                    * max(total_train_batches - batches, 0)
                )
                gpu_peak = progress["resources"].get("peak_gpu_memory_bytes", 0)
                print(
                    f"[TRAIN] epoch {epoch:03d}/{max_epochs:03d} "
                    f"batch {batches:04d}/{total_train_batches:04d} | "
                    f"loss={progress['average_loss']:.4f} | "
                    f"step={1000.0 * progress['recent_train_step_sec']:.0f}ms | "
                    f"graph={1000.0 * (progress['recent_graph_build_sec'] or 0.0):.0f}ms | "
                    f"elapsed={_duration(elapsed)} eta={_duration(eta)} | "
                    f"gpu_peak={gpu_peak / 2**30:.2f}GiB",
                    flush=True,
                )
                _atomic_json(
                    output_dir / "runtime_progress.json", progress
                )
        train_phase_sec = time.perf_counter() - epoch_started
        print(
            f"[VAL]   epoch {epoch:03d}/{max_epochs:03d} started | "
            f"batches={min(len(val_data), int(limit_val_batches)) if limit_val_batches is not None else len(val_data)}",
            flush=True,
        )
        validation_started = time.perf_counter()
        val_metrics = evaluate_batches(
            model,
            val_data.as_dataset(
                epoch,
                limit_batches=limit_val_batches,
                prefetch=controls.tf_data_prefetch,
            ),
            evaluate_step=eval_step,
        )
        val_phase_sec = time.perf_counter() - validation_started
        telemetry.validation_sec.append(val_phase_sec)
        print(
            f"[VAL]   epoch {epoch:03d}/{max_epochs:03d} done | "
            f"loss={val_metrics['loss']:.4f} | "
            f"accuracy={_percent(val_metrics['accuracy'])} | "
            f"macro_f1={_percent(val_metrics['macro_f1'])} | "
            f"time={_duration(val_phase_sec)}",
            flush=True,
        )
        train_metrics = None
        train_eval_phase_sec = 0.0
        train_eval_every = max(
            int(config["training"].get("eval_train_every_n_epochs", 1)), 1
        )
        should_eval_train = (
            train_eval_data is not None
            and (epoch == 1 or epoch % train_eval_every == 0)
        )
        if should_eval_train:
            print(
                f"[TRAIN-EVAL] epoch {epoch:03d}/{max_epochs:03d} started "
                f"(clean full-train metrics)",
                flush=True,
            )
            train_eval_started = time.perf_counter()
            train_metrics = evaluate_batches(
                model,
                train_eval_data.as_dataset(
                    epoch,
                    limit_batches=(
                        limit_train_eval_batches
                        if limit_train_eval_batches is not None
                        else config["training"].get("eval_train_limit_batches")
                    ),
                    prefetch=controls.tf_data_prefetch,
                ),
                evaluate_step=eval_step,
            )
            train_eval_phase_sec = time.perf_counter() - train_eval_started
            print(
                f"[TRAIN-EVAL] epoch {epoch:03d}/{max_epochs:03d} done | "
                f"loss={train_metrics['loss']:.4f} | "
                f"accuracy={_percent(train_metrics['accuracy'])} | "
                f"macro_f1={_percent(train_metrics['macro_f1'])} | "
                f"time={_duration(train_eval_phase_sec)}",
                flush=True,
            )
        else:
            print(
                f"[TRAIN-EVAL] epoch {epoch:03d}/{max_epochs:03d} skipped "
                f"(scheduled every {train_eval_every} epochs)",
                flush=True,
            )
        metadata = {
            "seed": seed,
            "config_hash": signatures["config"],
            "package_checksum": config["locked"]["package_checksum"],
            "graph_signature": signatures["graph"],
            "feature_signature": signatures["feature"],
            "prior_signature": signatures["prior"],
            "dataset_split_signature": signatures["dataset_split"],
            "tensorflow_version": tf.__version__,
            "keras_version": tf.keras.__version__,
            "optimizer_state": {
                "class": optimizer.__class__.__name__,
                "iterations": int(optimizer.iterations.numpy()),
                "learning_rate": float(optimizer.learning_rate.numpy()),
                "variable_count": len(optimizer.variables),
            },
            "scheduler_state": scheduler.get_state(),
            "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
            "execution_mode": execution_state,
            "execution_contract_sha256": config["locked"].get(
                "execution_contract_sha256"
            ),
            "resource_settings": controls.__dict__,
        }
        stop = early.update(epoch, val_metrics["loss"])
        metadata["early_stopping_state"] = early.get_state()
        checkpoint = policy.update_best(model, optimizer, epoch, val_metrics, metadata)
        lr = scheduler.step(val_metrics["loss"])
        metadata["scheduler_state"] = scheduler.get_state()
        metadata["optimizer_state"]["learning_rate"] = lr
        policy.save_last(model, optimizer, epoch, val_metrics, metadata)
        checkpoint["saved"].append("last")
        train_macro = None if train_metrics is None else train_metrics["macro_f1"]
        macro_gap = (
            None if train_macro is None
            else 100.0 * (float(train_macro) - float(val_metrics["macro_f1"]))
        )
        early_state = early.get_state()
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "train_eval_loss": None if train_metrics is None else train_metrics["loss"],
            "train_accuracy": None if train_metrics is None else train_metrics["accuracy"],
            "train_macro_f1": None if train_metrics is None else train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_val_macro_gap_pp": macro_gap,
            "lr": lr,
            "train_phase_sec": train_phase_sec,
            "val_phase_sec": val_phase_sec,
            "train_eval_phase_sec": train_eval_phase_sec,
            "train_eval_performed": bool(should_eval_train),
            "early_stopping_wait": int(early_state["epochs_without_improvement"]),
            "early_stopping_patience": int(early.patience),
            "stop_requested": bool(stop),
            "epoch_time_sec": time.perf_counter() - epoch_started,
            **checkpoint,
        }
        history.append(row)
        _atomic_json(output_dir / "history.json", {"epochs": history})
        _atomic_history_csv(output_dir / "train_log.csv", history)
        _atomic_json(output_dir / "latest_epoch_summary.json", row)
        write_training_curves(output_dir, history)
        _print_epoch_summary(row, policy)
        if stop:
            print(
                f"[STOP] early stopping at epoch {epoch}; "
                f"validation loss did not improve for {early.patience} epochs.",
                flush=True,
            )
            break
    _atomic_json(output_dir / "telemetry.json", telemetry.to_dict())
    primary = output_dir / "checkpoints" / "best_val_macro_f1.keras"
    selected_model = tf.keras.models.load_model(primary, compile=False)
    test_data = GraphBatchGenerator(
        prior_root, "test", config, controls.eval_batch_size, seed, False,
        controls.graph_cache_size, telemetry, graph_workers=controls.graph_workers,
        clean_graph_cache_dir=controls.clean_graph_cache_dir,
    )
    test_eval_step = build_compiled_evaluation_step(selected_model)
    print(
        f"[TEST] evaluating selected checkpoint "
        f"best_val_macro_f1.keras from epoch {policy.best_macro_epoch}",
        flush=True,
    )
    test_metrics_with_details = evaluate_batches(
        selected_model,
        test_data.as_dataset(
            0,
            limit_batches=limit_test_batches,
            prefetch=controls.tf_data_prefetch,
        ),
        evaluate_step=test_eval_step,
        return_details=True,
    )
    test_details = test_metrics_with_details.pop("details")
    test_metrics = test_metrics_with_details
    _atomic_json(output_dir / "test_metrics_best_val_macro_f1.json", test_metrics)
    artifact_paths = [
        output_dir / "resolved_config.yaml",
        output_dir / "resolved_config.json",
        output_dir / "provenance.json",
        output_dir / "history.json",
        output_dir / "train_log.csv",
        output_dir / "training_curves.png",
        output_dir / "latest_epoch_summary.json",
        output_dir / "telemetry.json",
        output_dir / "test_metrics_best_val_macro_f1.json",
        write_per_class_metrics(output_dir, test_metrics),
        write_predictions(output_dir, test_details),
    ]
    confusion_csv, confusion_png = write_confusion_matrix(
        output_dir,
        test_metrics["confusion_matrix"],
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
    )
    artifact_paths.extend([confusion_csv, confusion_png])
    run_summary = {
        "selected_checkpoint": "best_val_macro_f1.keras",
        "best_epoch": int(policy.best_macro_epoch),
        "best_val_macro_f1": float(policy.best_macro),
        "best_accuracy_epoch": int(policy.best_accuracy_epoch),
        "best_val_accuracy": float(policy.best_accuracy),
        "epochs_completed": len(history),
        "stop_reason": "early_stopping" if history[-1]["stop_requested"] else "max_epochs",
        "test_metrics": test_metrics,
    }
    _atomic_json(output_dir / "run_summary.json", run_summary)
    artifact_paths.append(output_dir / "run_summary.json")
    write_artifact_manifest(output_dir, artifact_paths)
    _atomic_json(
        output_dir / "TRAINING_COMPLETE.json",
        {
            "completed": True,
            "resume": False,
            "epochs": len(history),
            "selected_checkpoint": "best_val_macro_f1.keras",
            "test_used_for_selection": False,
        },
    )
    print(
        f"[TEST]  accuracy={_percent(test_metrics['accuracy'])} | "
        f"macro_f1={_percent(test_metrics['macro_f1'])} | "
        f"weighted_f1={_percent(test_metrics['weighted_f1'])}",
        flush=True,
    )
    print(
        f"[DONE] output={output_dir} | epochs={len(history)} | "
        f"best_epoch={policy.best_macro_epoch}",
        flush=True,
    )
    return {"output_dir": str(output_dir), "history": history, "test_metrics": test_metrics}
