# Frozen final results

The final model is MPG-FER v2.3, with 2,304,528 trainable parameters. Canonical FP32 evaluation disables autocast and TF32. The seed set was registered before aggregation; no best-seed filtering or result-driven rerun is used. PublicTest retains the registered EMA flip-TTA checkpoint comparator. PrivateTest is final reporting after checkpoint freeze.

| Seed | PrivateTest flip-TTA (%) |
|---:|---:|
| 0 | 70.7161 |
| 1 | 70.2981 |
| 42 | 70.6604 |
| 43 | 70.6046 |
| 123 | 70.0195 |
| 3047 | 70.4653 |

Mean **70.4607%**, sample SD **0.2634 percentage points** (`ddof=1`). The canonical seed-42 raw single-view accuracy is **68.7657%**; its flip-TTA accuracy is **70.6604%**. Raw single-view remains the primary scientific metric and TTA is reported separately.

The [registry](../research/mpg_fer_v2_3/MPG_V23_MULTI_SEED_RESULTS/multi_seed_registry.csv) records exact selected epochs, checkpoint/source/config hashes, execution run IDs, operational status and raw/TTA metrics. [Aggregate statistics](../research/mpg_fer_v2_3/MPG_V23_MULTI_SEED_RESULTS/aggregate_statistics.json) reports every metric and per-seed value. The complete 224-file compact package is retained unchanged; its 223 listed checksums are the original integrity contract. Prediction CSVs and class/confusion summaries are intentional frozen reporting artifacts, not raw FER2013 image/label datasets or inputs for tuning.

Issue97's early-depth residual-preservation mechanism is `V23_NOT_SUPPORTED`. That verdict does not mean training failed. `operational_status=TRAINING_COMPLETED` and `canonical_result_scope=Issue99 six-seed canonical FP32 replication` state the other scopes explicitly. Replication statistics do not establish that the unsupported mechanism hypothesis became supported.

See the [accepted ablation ladder](../analysis/final_ablation/README.md) and [closed diagnostic cycle](../analysis/final_research_closure/README.md). No new scientific run was performed during repository consolidation.
