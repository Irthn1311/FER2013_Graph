# TensorFlow Step 6 fixed-topology prior sensitivity

## Provenance

- Issue: #11.
- Frozen scientific base commit: `69f4571c5069da9a7f8558ef3c01101635ee904a`.
- Technical execution commit: `7fbe0ea306f23db9682833a7ff66ea65da7300e9`.
- Probe tool SHA-256: `3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3`.
- Frozen scientific payload SHA-256: `286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e`.
- Checkpoint SHA-256: `9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16`.
- Checkpoint metadata SHA-256: `e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37`.
- Resolved config SHA-256: `3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32`.
- Checkpoint epoch/seed: `31` / `42`.
- Runtime: Kaggle T4, TensorFlow `2.18.1`, Keras `3.15.0`.
- GPU memory growth: requested `True`, status `already_initialized`.

## C0 reproduction gate

- Sample count: `3589`; required `3589`.
- Gate checks: `{"accuracy_within_tolerance": true, "loss_within_tolerance": true, "macro_f1_within_tolerance": true, "sample_count_exact": true}`.
- Gate result: `PASS`.

## Validation metrics

- `official`: accuracy `0.63137364168292`, macro-F1 `0.5932591901893336`, loss `1.1537981724317095`.
- `direct_part_path_zero_fixed_graph`: accuracy `0.27751462803009197`, macro-F1 `0.19745892656222366`, loss `1.757720434560185`.
- `semantic_prior_zero_fixed_graph`: accuracy `0.25745332961827805`, macro-F1 `0.12183064164647293`, loss `1.8446298953706184`.

## Preregistered diagnostics

- Diagnostic label: `HIGH_EXPLICIT_PRIOR_SENSITIVITY`.
- Delta F1 C1: `39.580026362710996` pp.
- Delta F1 C2: `47.142854854286064` pp.
- Incremental C2-minus-C1 effect: `7.562828491575073` pp.
- Accuracy change C0-to-C1: `-35.3859013652828` pp.
- Accuracy change C0-to-C2: `-37.392031206464196` pp.
- Paired outcomes: `{"direct_part_path_zero_fixed_graph": {"correctness_transitions": {"c0_correct_to_intervention_incorrect": {"count": 1447, "rate": 0.4031763722485372}, "c0_incorrect_to_intervention_correct": {"count": 177, "rate": 0.04931735859570911}, "unchanged_correct": {"count": 819, "rate": 0.22819726943438284}, "unchanged_incorrect": {"count": 1146, "rate": 0.31930899972137083}}, "prediction_disagreement_count": 2461, "prediction_disagreement_rate": 0.6857063248815826}, "semantic_prior_zero_fixed_graph": {"correctness_transitions": {"c0_correct_to_intervention_incorrect": {"count": 1505, "rate": 0.4193368626358317}, "c0_incorrect_to_intervention_correct": {"count": 163, "rate": 0.04541655057118975}, "unchanged_correct": {"count": 761, "rate": 0.21203677904708831}, "unchanged_incorrect": {"count": 1160, "rate": 0.3232098077458902}}, "prediction_disagreement_count": 2644, "prediction_disagreement_rate": 0.7366954583449429}}`.
- Per-class F1: `{"direct_part_path_zero_fixed_graph": [0.20794590025359255, 0.0, 0.023346303501945526, 0.4057840036150023, 0.15384615384615385, 0.33843674456083805, 0.2528533801580334], "official": [0.5098039215686274, 0.46808510638297873, 0.4599375650364204, 0.8373126041088285, 0.5177584846093133, 0.7591069330199764, 0.6008097165991902], "semantic_prior_zero_fixed_graph": [0.0, 0.0, 0.029906542056074768, 0.3945916114790287, 0.18274111675126903, 0.24557522123893805, 0.0]}`.
- Per-class C0-minus-C1 F1 deltas: `[30.185802131503486, 46.808510638297875, 43.65912615344749, 43.15286004938262, 36.39123307631594, 42.067018845913836, 34.79563364411569]` pp.
- Per-class C0-minus-C2 F1 deltas: `[50.98039215686274, 46.808510638297875, 43.00310229803456, 44.27209926297998, 33.501736785804425, 51.35317117810384, 60.08097165991902]` pp.

## Integrity and boundaries

- Batch count: `113`; paired same-batch evaluation: `True`.
- Checkpoint file unchanged: `True`; model weights unchanged: `True`.
- Test CSV/prior records/cache index/cache shards/labels/predictions/metrics/inference accessed: `false`.
- Shared cache-root `CACHE_COMPLETE.json` accessed: `true`, solely as required non-sample aggregate loader metadata; it may summarize other splits.
- Training, fine-tuning, optimizer steps, raw-prior corruption, graph rebuild, and topology changes: `false`.
- C2 measures explicit semantic-prior/direct-part sensitivity conditional on the official MediaPipe-derived scaffold. It is not MediaPipe removal and is not a prior-free graph.
- The diagnostic label is a preregistered sensitivity heuristic, not causal proof of the Issue #7 generalization gap and not model selection.
