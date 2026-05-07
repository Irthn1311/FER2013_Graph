# D9-TGMS Distillation Design

## Goal
D9-TGMS uses a strong D7/D8B teacher, or a D7/D8B ensemble, to provide semantic soft labels for the D9 student. The teacher does not provide motif maps, motif masks, motif relations, or pixel saliency targets.

## Student
The student is the D9 pooled motif classifier. It still follows the graph-first route:

pixel graph -> motif discovery -> pooled motif classifier -> emotion logits

The feature-B contract stays runtime-masked on the existing graph repository:

- node indices: `[0, 1, 2]`
- edge indices: `[0, 1, 2, 3, 4]`
- model node_dim: `3`
- model edge_dim: `5`

## Teacher
Teacher outputs come from one or more D7/D8B checkpoints. `scripts/generate_teacher_probs.py` supports a single checkpoint or multiple checkpoints. With multiple checkpoints, the default output is an average of teacher probabilities.

## Alignment
`sample_idx` is required in every D9 batch. It is the stable index inside each split, from `0` to `len(split)-1`. Teacher arrays are saved and loaded by this index:

- `{split}_probs.npy`
- `{split}_logits.npy`
- `{split}_labels.npy`
- `{split}_indices.npy`

During D9 training, `teacher_labels[sample_idx]` must match `batch["y"]`. A mismatch raises an error.

## Loss
Training uses:

```text
total_loss = CE(student_logits, labels)
           + alpha * KL(student_logits / T, teacher_probs_or_logits)
           + motif_aux_weight * motif_aux_loss
```

Distillation is optional and backward-compatible. When `distillation.enabled: false`, the old D9 pooled MLP path is unchanged.

## Metrics
Classification checkpoints remain selected by:

```yaml
checkpoint:
  monitor: val_macro_f1
  mode: max
```

Validation accuracy is logged but is not used as the best-checkpoint monitor.

## Motif Validation
Teacher guidance only improves semantic supervision. Motif quality still needs independent checks, such as motif visualization, deletion tests, selected-region mass, entropy, and other motif-audit diagnostics.

## Commands
Generate teacher probabilities from a D7/D8B ensemble:

```bash
python -m scripts.generate_teacher_probs \
  --graph_repo_path artifacts/graph_repo \
  --splits train val test \
  --teacher_configs configs/experiments/d7a_graph_swin_region_transformer_seed44.yaml configs/experiments/d8b_face_aware_graph_swin_border020.yaml \
  --teacher_checkpoints output/d7a_graph_swin_region_transformer_seed44/checkpoints/best.pth output/d8b_face_aware_graph_swin_border020/checkpoints/best.pth \
  --output_dir outputs/teacher_probs/d7d8b_ensemble \
  --device cuda \
  --batch_size 32 \
  --num_workers 2
```

Train D9-TGMS alpha 0.5:

```bash
python -m scripts.train_d9_relation_motif \
  --config configs/experiments/d9_tgms_b_distill_a05_t2.yaml \
  --env local \
  --graph_repo_path artifacts/graph_repo \
  --device cuda \
  --no_wandb
```

Train D9-TGMS alpha 0.2:

```bash
python -m scripts.train_d9_relation_motif \
  --config configs/experiments/d9_tgms_b_distill_a02_t2.yaml \
  --env local \
  --graph_repo_path artifacts/graph_repo \
  --device cuda \
  --no_wandb
```

Smoke-only dummy teacher probs from labels:

```bash
python -m scripts.generate_teacher_probs \
  --graph_repo_path artifacts/graph_repo \
  --splits train val \
  --output_dir outputs/teacher_probs/d9_tgms_dummy_smoke \
  --device cpu \
  --batch_size 64 \
  --num_workers 0 \
  --dummy_from_labels
```
