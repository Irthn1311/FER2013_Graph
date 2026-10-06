# MPG-FER v2.3

MPG-FER performs facial expression recognition on FER2013 with Pixel and Motif Graphs. The final canonical model has **2,304,528 parameters**.

**Canonical seed-42 PrivateTest flip-TTA: 70.6604%.** Six-seed PrivateTest flip-TTA: **70.46 ± 0.26%**, mean ± sample SD (`ddof=1`) over seeds `{0,1,42,43,123,3047}`. These are frozen canonical FP32 evaluations with autocast and TF32 disabled; raw single-view metrics are reported separately.

```text
48×48 grayscale image → 2304 Pixel Graph nodes → Spatial Motif Composer
→ 49 spatial occurrences → geometry-aware Motif Graph → readout/classifier
```

The Composer learns **48 prototypes**. They are a prototype dictionary, distinct from the **49 spatial occurrences** on a 7×7 support grid; occurrences are not semantic landmarks. Multi-scale supports use windows 8/12/16. Five motif layers use Dynamic Top-K `[8,16,16,16,24]` and fixed residual scales `[0.5,0.5,1,1,1]`. The motif readout combines mean, max and learned attention pooling; the classifier fuses pixel and motif representations.

Top-K restricts selected relational aggregation. Dense relation scores are computed first, so this implementation does not establish subquadratic attention complexity.

- **Train and evaluate:** [source-locked reproduction instructions](docs/REPRODUCIBILITY.md). Existing T4 notebook templates enforce preflight, exact resume, EMA PublicTest checkpoint selection and PrivateTest freeze gates.
- **Results and all six seeds:** [final results](docs/FINAL_RESULTS.md), [complete compact evidence](research/mpg_fer_v2_3/MPG_V23_MULTI_SEED_RESULTS/MULTI_SEED_REPORT.md).
- **Accepted P0–P5 ablations:** [final ablation](analysis/final_ablation/README.md).
- **Terminal decisions:** [final research closure](analysis/final_research_closure/README.md).
- **Eight-month history and archive refs:** [research history](docs/RESEARCH_HISTORY.md).

The incremental research cycle is **closed**. Diagnostic associations support the recorded stopping decisions under their protocols; they do not prove causation. Issue97's early-depth mechanism verdict remains `V23_NOT_SUPPORTED`, separately from `TRAINING_COMPLETED` and Issue99's completed six-seed replication.

The runtime stays at `research/mpg_fer_v2_3/` to preserve existing paths and source locks. The small v2.2 source package is retained solely for frozen regression/parity tests. Historical TensorFlow, D5/D16–D19 and superseded candidates live in archives. Checkpoint weights and FER2013 datasets are external artifacts, not ordinary Git files.
