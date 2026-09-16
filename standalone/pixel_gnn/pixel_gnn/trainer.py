"""Universal Training Pipeline for Pixel GNN Models."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from pixel_gnn.utils import (
    EarlyStopping,
    ReduceLROnPlateau,
    build_optimizer,
    load_config,
    seed_everything,
)

from pixel_gnn.batching import PixelBatchGenerator
from pixel_gnn.dataset import FERPixelDataset
from pixel_gnn.evaluator import evaluate_model
from pixel_gnn.losses import compute_total_loss
from pixel_gnn.models import build_model


def run_training(
    config_path: str | Path,
    fer_csv: str | Path,
    output_root: str | Path,
    limit_epochs: int | None = None,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
):
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    batch_size = int(config.get("training", {}).get("batch_size", 32))
    eval_batch_size = int(config.get("training", {}).get("eval_batch_size", batch_size))
    max_epochs = limit_epochs if limit_epochs is not None else int(config.get("training", {}).get("max_epochs", 90))

    model_name = config.get("model", {}).get("name", "pixel_neighbor_motif")
    loss_cfg = config.get("loss", {})
    lambda_diversity = float(loss_cfg.get("lambda_motif_diversity", 0.0))

    print("=" * 80)
    print(f"[INIT] Pixel GNN Universal Trainer | Model: {model_name}")
    print(f"       batch_size={batch_size} | max_epochs={max_epochs} | seed={seed}")
    print(f"       output_dir={output_dir}")
    print("=" * 80, flush=True)

    # Initialize Datasets
    train_dataset = FERPixelDataset(fer_csv, "train")
    val_dataset = FERPixelDataset(fer_csv, "val")

    train_gen = PixelBatchGenerator(fer_csv, "train", batch_size=batch_size, seed=seed, shuffle=True, dataset=train_dataset)
    val_gen = PixelBatchGenerator(fer_csv, "val", batch_size=eval_batch_size, seed=seed, shuffle=False, dataset=val_dataset)

    # Strategy setup: auto-detect 2 GPUs on Kaggle or multi-GPU environments
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        strategy = tf.distribute.MirroredStrategy()
        num_replicas = strategy.num_replicas_in_sync
        print(f"[DEVICE] Detected {len(gpus)} GPUs: {[g.name for g in gpus]}")
        print(f"[DEVICE] Activating MirroredStrategy with {num_replicas} replicas (both GPUs will be utilized)!", flush=True)
    else:
        strategy = tf.distribute.get_strategy()
        num_replicas = 1
        device_type = "1 GPU" if len(gpus) == 1 else "CPU"
        print(f"[DEVICE] Single device ({device_type}).", flush=True)

    # Instantiate model and optimizer inside strategy scope
    with strategy.scope():
        model = build_model(config)
        # Trigger build with one batch
        sample_batch = next(iter(train_gen.as_dataset(0, limit_batches=1)))
        _ = model(sample_batch, training=False)
        optimizer = build_optimizer(config)
        optimizer.build(model.trainable_variables)

    param_count = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    print(f"[MODEL] Build complete: {model.name}. Trainable parameters: {param_count:,}", flush=True)

    sched_cfg = config.get("training", {}).get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get("mode", "min"),
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 5),
        threshold=sched_cfg.get("threshold", 0.0001),
        min_lr=sched_cfg.get("min_lr", 3e-5),
    )

    early_cfg = config.get("training", {}).get("early_stopping", {})
    early_stopping = EarlyStopping(
        min_epochs=early_cfg.get("min_epochs_before_stop", 30),
        patience=early_cfg.get("patience", 15),
    )

    def step_fn(batch):
        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            loss, metrics = compute_total_loss(batch["labels"], out, lambda_diversity=lambda_diversity)
            scaled_loss = loss / float(num_replicas)
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, metrics["ce_loss"]

    if num_replicas > 1:
        @tf.function
        def dist_train_step(dist_batch):
            per_replica_loss, per_replica_ce = strategy.run(step_fn, args=(dist_batch,))
            total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None) / float(num_replicas)
            total_ce = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_ce, axis=None) / float(num_replicas)
            return total_loss, total_ce
    else:
        @tf.function
        def dist_train_step(batch):
            return step_fn(batch)

    history = []
    best_val_acc = 0.0
    best_val_epoch = 0
    history_csv = output_dir / "training_history.csv"

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        raw_train_ds = train_gen.as_dataset(epoch, limit_batches=limit_train_batches)
        if num_replicas > 1:
            train_ds = strategy.experimental_distribute_dataset(raw_train_ds)
        else:
            train_ds = raw_train_ds

        step_losses, step_ce = [], []
        for batch in train_ds:
            loss_val, ce_val = dist_train_step(batch)
            step_losses.append(float(loss_val.numpy()))
            step_ce.append(float(ce_val.numpy()))

        train_loss = float(np.mean(step_losses))

        # Validation
        val_ds = val_gen.as_dataset(epoch, limit_batches=limit_val_batches)
        val_metrics = evaluate_model(
            model,
            val_ds,
            lambda_diversity=lambda_diversity,
            include_diagnostics=(epoch % 5 == 0 or epoch == max_epochs),
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]

        scheduler.step(val_loss)
        current_lr = float(optimizer.learning_rate.numpy()) if hasattr(optimizer.learning_rate, "numpy") else float(optimizer.learning_rate)

        saved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            model.save_weights(str(output_dir / "best_val_accuracy.weights.h5"))
            saved = True

        total_time = time.perf_counter() - t0

        print(
            f"[EPOCH {epoch:03d}/{max_epochs:03d}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc*100:.2f}% | val_macro_f1={val_macro_f1*100:.2f}% | "
            f"lr={current_lr:.6f} | time={total_time:.1f}s"
            f"{' [BEST SAVED]' if saved else ''}",
            flush=True,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "lr": current_lr,
            "epoch_time_sec": total_time,
            "best_accuracy": best_val_acc,
            "best_epoch": best_val_epoch,
        }
        history.append(epoch_record)

        with history_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_record.keys()))
            writer.writeheader()
            writer.writerows(history)

        if early_stopping.step(val_loss, epoch):
            print(f"[EARLY STOPPING] Triggered at epoch {epoch} (patience={early_stopping.patience})", flush=True)
            break

    print("=" * 80)
    print(f"[COMPLETED] Best Validation Accuracy: {best_val_acc*100:.2f}% at epoch {best_val_epoch}")
    print("=" * 80, flush=True)

    # Load best checkpoint weights for final test evaluation
    best_weights_path = output_dir / "best_val_accuracy.weights.h5"
    if best_weights_path.exists():
        model.load_weights(str(best_weights_path))
        print(f"[CHECKPOINT] Loaded best weights from {best_weights_path} for final test evaluation.", flush=True)

    # Final Test Set Evaluation (3,589 images)
    print("\n" + "=" * 80)
    print("[TEST] Running final evaluation on TEST set...")
    print("=" * 80, flush=True)
    test_dataset = FERPixelDataset(fer_csv, "test")
    test_gen = PixelBatchGenerator(fer_csv, "test", batch_size=eval_batch_size, seed=seed, shuffle=False, dataset=test_dataset)
    test_ds = test_gen.as_dataset(0)

    test_metrics = evaluate_model(model, test_ds, lambda_diversity=lambda_diversity, include_diagnostics=True)
    test_acc = test_metrics["accuracy"]
    test_macro_f1 = test_metrics["macro_f1"]
    test_loss = test_metrics["loss"]

    print("=" * 80)
    print(
        f"[TEST RESULT] Final Test Accuracy: {test_acc*100:.2f}% | "
        f"Test Macro F1: {test_macro_f1*100:.2f}% | "
        f"Test Loss: {test_loss:.4f}"
    )
    print("=" * 80, flush=True)

    cm = test_metrics.get("confusion_matrix")
    if cm is not None and hasattr(cm, "tolist"):
        cm = cm.tolist()
    per_class = test_metrics.get("per_class_f1")
    if per_class is not None and hasattr(per_class, "tolist"):
        per_class = per_class.tolist()

    test_metrics_to_save = {
        "model_name": model_name,
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_loss": float(test_loss),
        "best_val_accuracy": float(best_val_acc),
        "best_val_epoch": int(best_val_epoch),
        "epochs_trained": len(history),
        "per_class_f1": per_class,
        "confusion_matrix": cm,
    }

    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics_to_save, indent=2), encoding="utf-8")

    if test_metrics.get("motif_diagnostics", {}).get("motif_enabled"):
        final_diag = test_metrics["motif_diagnostics"]
        diag_path = output_dir / "motif_diagnostics.json"
        serializable_diag = {
            "usage": final_diag["usage"].tolist(),
            "top_nodes_per_motif": final_diag["top_nodes_per_motif"],
        }
        diag_path.write_text(json.dumps(serializable_diag, indent=2), encoding="utf-8")
        print(f"[MOTIF] Diagnostics saved to {diag_path}", flush=True)

    print(f"[SUCCESS] All artifacts and test metrics saved to {output_dir}")
