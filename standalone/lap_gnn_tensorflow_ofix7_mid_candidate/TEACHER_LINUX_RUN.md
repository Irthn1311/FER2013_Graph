# Teacher Instructions

After downloading the Google Drive folder, open a terminal inside
`ofix7_mid_seed42_teacher_bundle` and run exactly one command:

```bash
bash run.sh
```

Nothing else must be configured. The script automatically:

1. verifies the bundle layout and records CPU, RAM, GPU, driver and NUMA state;
2. uses existing Conda or installs private Miniforge inside the bundle;
3. creates the locked TensorFlow 2.13.1, CUDA 11.8 and cuDNN 8.6 environment;
4. verifies package checksums, CSV, prior and graph-cache contracts;
5. selects GPU 0 and its local NUMA CPU node;
6. runs bounded graph, forward, optimizer and checkpoint preflight;
7. starts seed42 training only after every check passes;
8. verifies `TRAINING_COMPLETE.json` and the artifact manifest;
9. creates `ofix7_mid_seed42_results.tar.gz`.

The teacher does not need to run `--dry-run`, inspect a plan, select paths or
reply during execution.

## Outputs

```text
results/ofix7_mid_seed42/            complete training artifacts
logs/                                bootstrap and launcher logs
ofix7_mid_seed42_results.tar.gz      file to return
```

If a failure occurs, the terminal prints the failed stage. Detailed evidence is
kept under `logs`, including `bootstrap_failure.txt`, `launcher.log`,
`diagnostics.json` and `failure_report.json`. Return the log directory without
changing the model configuration.

One GPU is used intentionally. Combining both GPUs would change global-batch
and optimizer semantics; CPU graph workers and NUMA-local memory feed that GPU
without changing the registered scientific execution path.
