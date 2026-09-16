#!/usr/bin/env python3
"""Separate Kaggle runner for Pixel-GNN Only. The full runner is unchanged."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ABLATION_ROOT = REPO_ROOT / "standalone/pixel_gnn_only"
sys.path.insert(0, str(REPO_ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/src"))
sys.path.insert(0, str(ABLATION_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ABLATION_ROOT / "configs/fer2013_pixel_gnn_only_kaggle_fast_seed42.yaml"))
    parser.add_argument("--fer-csv", help="train.csv alongside val.csv/test.csv, or original FER2013 Usage CSV")
    parser.add_argument("--output-root")
    parser.add_argument("--gpus", choices=["auto", "1", "2"], default=None)
    parser.add_argument("--batch-size", type=int, help="Global batch override, allowed only with the Kaggle fast config")
    parser.add_argument("--graph-workers", type=int, help="CPU graph workers; default uses allocated CPU cores")
    parser.add_argument("--smoke", action="store_true", help="One training batch forward/backward; no val/test evaluation")
    parser.add_argument("--synthetic", action="store_true", help="Explicitly synthetic input, allowed only with --smoke")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved protocol without loading data or training")
    args = parser.parse_args()
    print(f"[BOOT] Pixel-GNN: importing TensorFlow; config={args.config}", flush=True)
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError("Use Python 3.10–3.12, matching the frozen TensorFlow package")

    # Resolve CPU/thread controls before importing modules that may create a
    # TensorFlow tensor. Kaggle rejects thread changes after its GPU context is
    # initialized, even when the requested values match the defaults.
    from lap_gnn_tf.config import load_config
    from pixel_gnn_only.runtime import select_cpu_count

    config = load_config(args.config)
    resources = config["resources"]
    cpu_mode = config.get("runtime", {}).get("cpu_mode", "quota")
    cpu_count = select_cpu_count(cpu_mode)
    requested_gpu_count = args.gpus or config.get("runtime", {}).get("gpus", "1")
    provisional_gpu_count = 2 if requested_gpu_count == "auto" else int(requested_gpu_count)
    intra_op_threads = resources.get("intra_op_threads", 0) or cpu_count
    inter_op_threads = resources.get("inter_op_threads", 0) or provisional_gpu_count
    # These environment controls are consumed while TensorFlow initializes and
    # also cover builds that create an eager device during the import itself.
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(intra_op_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(inter_op_threads)

    import tensorflow as tf
    # Prefer the explicit API while the runtime is still mutable. The pre-import
    # environment controls remain authoritative if this TF build initialized
    # eager devices as part of import.
    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    except RuntimeError:
        print(
            "[BOOT] TensorFlow initialized during import; using pre-import "
            "TF_NUM_INTRAOP_THREADS/TF_NUM_INTEROP_THREADS settings",
            flush=True,
        )
    physical_gpus = tf.config.list_physical_devices("GPU")
    if resources["memory_growth"]:
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Harmless when the device was already initialized by this TF build.
                pass
    tf.config.optimizer.set_jit(bool(resources["xla"]))
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if resources["mixed_precision"] else "float32"
    )

    tf_version = tuple(int(x) for x in tf.__version__.split(".")[:2])
    if not (2, 13) <= tf_version < (2, 19):
        raise RuntimeError("Use TensorFlow 2.13–2.18; install requirements-kaggle.txt and restart the Kaggle session")
    from pixel_gnn_only.protocol import validate_pixel_config, apply_batch_size_override
    apply_batch_size_override(config, args.batch_size)
    validate_pixel_config(config)
    if args.synthetic and not args.smoke:
        parser.error("--synthetic is allowed only with --smoke")
    output = args.output_root or ("/kaggle/working/outputs/pixel_gnn_only_smoke" if args.smoke
                                  else config["paths"]["output_root"])
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if args.output_root is None and Path(output).exists() and any(Path(output).iterdir()):
        output += "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not args.synthetic and not args.fer_csv:
        configured_csv = config.get("paths", {}).get("fer_csv")
        if configured_csv and Path(configured_csv).is_file():
            args.fer_csv = configured_csv
        else:
            parser.error("Pass --fer-csv explicitly; the configured FER CSV was not found")
    from lap_gnn_tf.resources import ResourceControls
    from lap_gnn_tf.seed import seed_everything
    from pixel_gnn_only.execution import validate_execution_config
    from pixel_gnn_only.runtime import select_gpu_count
    gpu_count = select_gpu_count(requested_gpu_count, len(physical_gpus))
    graph_workers = args.graph_workers if args.graph_workers is not None else resources["graph_workers"]
    if graph_workers < 0:
        parser.error("--graph-workers must be zero (auto) or positive")
    graph_workers = min(graph_workers or cpu_count, config["training"]["batch_size"])
    controls = ResourceControls(
        batch_size=config["training"]["batch_size"],
        eval_batch_size=resources.get("eval_batch_size", 32),
        intra_op_threads=resources.get("intra_op_threads", 0) or cpu_count,
        inter_op_threads=resources.get("inter_op_threads", 0) or gpu_count,
        graph_workers=graph_workers, tf_data_prefetch=resources["tf_data_prefetch"],
        graph_cache_size=resources["graph_cache_size"], memory_growth=resources["memory_growth"],
        mixed_precision=resources["mixed_precision"], xla=resources["xla"], device="gpu")
    # TensorFlow runtime controls were applied before importing graph/model code.
    seed_everything(config["seed"])
    validate_execution_config(config["training"])
    print(f"[RUN] GPUs={gpu_count}; CPU mode={cpu_mode}; CPU threads={cpu_count}; graph workers={graph_workers}; "
          f"global batch={controls.batch_size}; eval batch={controls.eval_batch_size}; "
          f"prefetch={controls.tf_data_prefetch}; output={output}", flush=True)
    if args.smoke:
        if args.synthetic:
            import numpy as np
            from pixel_gnn_only.graph import build_pixel_graph, collate_pixel_graphs
            rng = np.random.default_rng(config["seed"])
            with tf.device("/CPU:0"):
                batch = collate_pixel_graphs([build_pixel_graph(rng.random((48, 48), dtype=np.float32), i % 7, i)
                                             for i in range(controls.batch_size)])
            input_kind = "synthetic; no FER data or prior artifacts read"
        else:
            from pixel_gnn_only.batching import GraphBatchGenerator
            generator = GraphBatchGenerator(args.fer_csv, "train", config, controls.batch_size,
                                            config["seed"], True, graph_workers=controls.graph_workers)
            batch = next(generator.iter_epoch(1, limit_batches=1))
            input_kind = generator.dataset.provenance
        from pixel_gnn_only.smoke import run_smoke
        run_smoke(batch, config, output, input_kind, gpu_count=gpu_count)
    else:
        from pixel_gnn_only.trainer import run_training
        run_training(config_path=args.config, fer_csv=args.fer_csv, output_root=output,
                     controls=controls, no_resume=True, gpu_count=gpu_count,
                     batch_size_override=args.batch_size, resources_initialized=True)


if __name__ == "__main__":
    main()
