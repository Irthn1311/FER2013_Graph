# D12 Code Audit Report

Date: 2026-05-14  
Scope: audit codebase hien tai truoc khi trien khai D12. Khong sua model/trainer/config. Bao cao nay duoc lap bang cach doc truc tiep cac file uu tien trong repo.

## 0. Executive Summary

Pipeline hien tai co 2 nhanh chinh can ke thua co chon loc:

- D10 la motif pipeline chin hon: `SharedPixelEncoder` -> `IterativeSlotAttention` -> optional position encoding -> optional cross-attention slot refinement -> `MotifRelationTransformer` -> `ClassMotifAttentionHead`.
- D11 la global-local fusion pipeline: `SharedPixelEncoder` -> FACS slot local branch + virtual global gather -> FiLM modulation -> motif relation -> pooled classifier, kem loss rieng cho local aux, SupCon, diversity va spatial prior.
- D10 best configs (`d10_p3_5_iter5`, `d10_p3_10_no_pos_enc`) thuc te inherit tu `d10_p3_1_refinement` va `d10_p2_1_multiscale`, nen co `multi_scale_gnn: true`, `residual_slot_connection: true`, `use_slot_refinement: true`, `pixel_gnn_layers: 2`; encoder phu tu dong co 3 layer.
- D11 Exp H config khong khai bao `pixel_gnn_layers`, `node_dim`, `edge_dim`; code roi ve default `SharedPixelEncoder(pixel_gnn_layers=1, node_dim=7, edge_dim=5)`. Day la mot ly do quan trong D11 co the yeu hon D10 ve local motif formation.
- Trainer/evaluator chi can `out["logits"]` de tinh metric. Loss D10/D11 moi yeu cau them cac key khac nhau. D12 nen tra ve superset key de chay duoc ca D10-style diagnostics va D11-style loss.

## A. Data / Batch Schema

### A1. Dataset item

`FullGraphDataset.__getitem__` tra ve mot graph da resolve tu `ChunkedGraphDataset(resolve=True)`:

- `sample_idx`: scalar long.
- `global_index`: scalar long, bang `sample_idx`.
- `graph_id`: scalar long.
- `node_features`: `[N, node_dim]`, float.
- `x`: alias cua `node_features`, `[N, node_dim]`.
- `edge_index`: `[2, E]`, long.
- `edge_attr`: `[E, edge_dim]`, float.
- `node_mask`: `[N]`, bool, hien tai toan `True`.
- `y`: scalar long.
- `label`: alias cua `y`.

Source: `data/full_graph_dataset.py:56-73`.

### A2. Collated batch

`collate_fn_full_graph` stack batch thanh:

- `x`: `[B, N, node_dim]`.
- `node_features`: alias cua `x`.
- `edge_index`: `[B, 2, E]`; lay `batch[0]["edge_index"]`, `unsqueeze(0)`, `expand(B, -1, -1)`, `contiguous()`.
- `edge_attr`: `[B, E, edge_dim]`.
- `node_mask`: `[B, N]`.
- `y`: `[B]`.
- `label`: alias cua `y`.
- `graph_id`: `[B]`.
- `sample_idx`: `[B]`.
- `global_index`: `[B]`.

Source: `data/full_graph_dataset.py:123-143`.

### A3. Shape mac dinh

Theo `GraphConfig`:

- `height=48`, `width=48`, `num_nodes=2304`.
- `connectivity=8`.
- Node feature V1 gom 7 chieu: `intensity`, `x_norm`, `y_norm`, `gx`, `gy`, `grad_mag`, `local_contrast`.
- Edge feature gom 5 chieu: static `dx`, `dy`, `dist` + dynamic `delta_intensity`, `intensity_similarity`.
- Edge count 8-neighbor directed: `2*h*(w-1) + 2*(h-1)*w + 4*(h-1)*(w-1) = 17860`.

Source: `data/graph_config.py:8-28`, `data/graph_config.py:31-73`.

### A4. Graph repo / artifact format

Repo graph co dang:

- `manifest.pt`: dict gom version, chunk_size, config, node/edge feature names, node_dim, edge_dim, height, width, splits.
- `shared/shared_graph.pt`: `SharedGraphStructure`, chua topology dung chung va static edge attrs.
- `train/chunk_*.pt`, `val/chunk_*.pt`, `test/chunk_*.pt`: list cac `PixelGraphSample`.

`GraphResolver` noi `shared.edge_attr_static` voi `sample.edge_attr_dynamic` theo dim=1 de tao `edge_attr` full `[E, 5]`. No cung validate node count, edge count va finite values.

Sources:

- `data/graph_repository.py:18-21`
- `data/graph_repository.py:47-60`
- `data/graph_repository.py:82-85`
- `data/graph_repository.py:138-178`
- `data/graph_resolver.py:19-55`

### A5. DataParallel va edge_index

Do collate tao `edge_index: [B,2,E]`, khi vao model D10/D11 deu xu ly:

```python
if edge_index.ndim == 3:
    edge_index = edge_index[0]
```

Nghia la moi replica DataParallel nhan sub-batch, nhung vi graph topology dung chung, model chi lay topology cua sample dau tien trong sub-batch. `edge_attr` van giu batch dimension `[B_local,E,F]`.

Sources:

- `models/d10_slot_motif_model.py:459-461`
- `models/d11_global_local_model.py:264-265`

Rui ro D12: neu multi-scale edge_index khong con dong nhat cho moi sample, logic `edge_index[0]` se sai. Neu edge_index van generated chung theo scale, cach nay van dung.

## B. D10 Architecture

### B1. Modules

`D10SlotMotifModel` gom:

- `SharedPixelEncoder`: deep edge-aware pixel encoder.
- Optional `encoder_aux` + `combine_scale` neu `multi_scale_gnn=True`.
- `IterativeSlotAttention`.
- Optional `position_mlp` neu `use_position_encoding=True`.
- Optional cross-attention `slot_refinement` + norm/MLP neu `use_slot_refinement=True`.
- `MotifRelationTransformer`.
- `ClassMotifAttentionHead`.
- Optional `aux_classifier`.
- Diagnostic buffers: `pixel_positions`, `border_mask`.

Source: `models/d10_slot_motif_model.py:252-390`.

### B2. Forward path

Forward D10:

1. Parse batch dict: lay `x`, `edge_index`, `edge_attr`, `node_mask`.
2. Neu `edge_index` la `[B,2,E]`, lay `edge_index[0]`.
3. Optional node noise / edge dropout khi training.
4. `h_pixel = self.encoder(x, edge_index, edge_attr, node_mask)`.
5. Neu `multi_scale_gnn=True`: chay `encoder_aux` tren cung input, concat `[h_pixel,h_aux]`, dua qua `combine_scale`.
6. `motif_embeddings, slot_attn_maps = self.slot_attention(h_pixel, node_mask)`.
7. Neu `use_position_encoding=True`: tinh motif center tu `slot_attn_maps` va `pixel_positions`, add `position_mlp(motif_centers)`.
8. Neu `use_slot_refinement=True`: motif query attend lai `h_pixel` bang `nn.MultiheadAttention`.
9. `motif_context = self.motif_relation(motif_embeddings)`.
10. `logits, class_motif_attn, class_repr = self.classifier(motif_context)`.
11. Tao output dict va diagnostics.

Source: `models/d10_slot_motif_model.py:437-560`.

### B3. Output dict

D10 tra ve:

- `logits`: `[B, C]`, bat buoc cho trainer/eval.
- `part_masks`: `[B, K, N]`, alias cua `slot_attn_maps`, bat buoc cho D10/D6-style loss.
- `motif_embeddings`: `[B, K, D]`, truoc relation transformer, sau position/refinement neu bat.
- `motif_context`: `[B, K, D]`, sau relation transformer.
- `slot_attn_maps`: `[B, K, N]`.
- `class_motif_attn`: `[B, C, K]`.
- `class_repr`: `[B, C, D]`.
- `class_part_attn`: alias cua `class_motif_attn`, dung cho diagnostics/evaluator D6-compatible.
- `pixel_embeddings`, `h_pixel`: `[B, N, D]`.
- Optional `motif_centers`, `part_centers`: `[B, K, 2]`.
- Optional `aux_logits`: `[B, C]`.
- `diagnostics`: dict scalar tensors.

Source: `models/d10_slot_motif_model.py:534-558`.

### B4. Slot attention

`IterativeSlotAttention`:

- Input: `inputs [B,N,D]`.
- Query tu slots `[B,K,D]`, key/value tu pixel `[B,N,D]`.
- Attention logits: `einsum("bkd,bnd->bkn") / sqrt(D)` -> `[B,K,N]`.
- Softmax theo `dim=1`, tuc moi pixel canh tranh giua cac slot, khong softmax theo pixel.
- Sau softmax, mask invalid nodes neu co.
- Normalize theo tong pixel moi slot: `attn_weights = attn_maps / attn_maps.sum(dim=2)`.
- Aggregate: `updates = einsum("bkn,bnd->bkd", attn_weights, v)`.
- GRU update slot, optional residual slot, MLP residual.

Sources:

- `models/d10_slot_motif_model.py:84-149`
- `models/d10_slot_motif_model.py:121-135`
- `models/d10_slot_motif_model.py:137-147`

### B5. Cross-attention slot refinement va residual slot

- Residual slot nam trong `IterativeSlotAttention`, bat bang `residual_slot_connection=True`; sau GRU, `slots = slots + slots_prev`.
- Cross-attention refinement nam trong `D10SlotMotifModel.forward`, bat bang `use_slot_refinement=True`: query la motif da norm, key/value la `h_pixel`, co `key_padding_mask` tu `node_mask`, roi residual + FFN.

Sources:

- `models/d10_slot_motif_model.py:143-147`
- `models/d10_slot_motif_model.py:334-350`
- `models/d10_slot_motif_model.py:512-523`

### B6. Motif relation / class motif attention

- `MotifRelationTransformer` la `nn.TransformerEncoder` batch-first tren `[B,K,D]`.
- `ClassMotifAttentionHead` co learnable `class_queries [C,D]`. No project motif thanh key/value, tinh score `[B,C,K]`, softmax theo `dim=2` tren motifs, lay weighted sum thanh `class_repr [B,C,D]`, roi `logit_head` ra `[B,C]`.

Sources:

- `models/d10_slot_motif_model.py:152-181`
- `models/d10_slot_motif_model.py:184-238`

### B7. D10 best config status

`d10_p3_5_iter5.yaml` inherit `d10_p3_1_refinement.yaml` va chi override `slot_iterations: 5`. `d10_p3_10_no_pos_enc.yaml` inherit cung parent va override `use_position_encoding: false`.

`d10_p3_1_refinement.yaml` inherit `d10_p2_1_multiscale.yaml` va bat:

- `multi_scale_gnn: true`
- `residual_slot_connection: true`
- `use_slot_refinement: true`

`d10_p2_1_multiscale.yaml` inherit `d10_v3_2_cosine_only.yaml`, bat:

- `multi_scale_gnn: true`
- `pixel_gnn_layers: 2`
- scheduler `cosine_warmup`

`d10_slot_motif.yaml` base D10 dat:

- `hidden_dim: 96`
- `num_motifs: 8`
- `slot_iterations: 3` nhung bi override len 5 trong chain.
- `motif_relation_layers: 1`
- `motif_relation_heads: 4`
- `use_aux_classifier: true`
- loss `d10_slot_motif`, class weights, slot regularizers, aux CE.

Sources:

- `configs/experiments/d10_p3_5_iter5.yaml:1-12`
- `configs/experiments/d10_p3_10_no_pos_enc.yaml:1-12`
- `configs/experiments/d10_p3_1_refinement.yaml:1-22`
- `configs/experiments/d10_p2_1_multiscale.yaml:1-27`
- `configs/experiments/d10_slot_motif.yaml:48-77`

## C. D11 Architecture

### C1. Modules

`D11GlobalLocalModel` gom:

- `SharedPixelEncoder`.
- `IterativeSlotAttentionFACS`.
- Optional `VirtualNodeGather` + `FiLMFusion` neu `use_global_branch=True`.
- `motif_relation`: `nn.TransformerEncoder` 2 layers, 4 heads.
- `aux_classifier`: local-only auxiliary classifier.
- `supcon_proj`: projection head cho pooled `local_raw`.
- `classifier`: main classifier tren fusion pooled motif.

Source: `models/d11_global_local_model.py:166-242`.

### C2. Encoder call

D11 parse batch dict, lay:

- `x = batch.get("x", batch.get("node_features"))`
- `edge_index = batch["edge_index"]`
- `edge_attr = batch.get("edge_attr")`

Neu `edge_index.ndim == 3`, lay `edge_index[0]`. Sau do:

```python
encoded_x = self.encoder(x, edge_index, edge_attr)
```

Chu y: D11 forward hien tai khong truyen `node_mask` vao encoder. Sau encoder moi gan `mask = node_mask` cho dense branch. Neu can D12 ton trong mask trong encoder, can sua call hoac implement wrapper nhan du tham so.

Source: `models/d11_global_local_model.py:244-280`.

### C3. Local branch

- `dense_x`: `[B,N,D]`.
- `dense_raw_x`: raw node features `[B,N,node_dim]`.
- `coords = dense_raw_x[:, :, coord_indices[0]:coord_indices[1]+1]`; default `(1,2)` -> `x_norm,y_norm`.
- `local_raw, slot_attn, center_of_mass = slot_attention(dense_x, coords, mask)`.
- `slot_attn`: `[B,K,N]`, softmax theo slot dim=1.
- `center_of_mass`: `[B,K,2]`, tinh tu normalized `attn_weights`.

Sources:

- `models/d11_global_local_model.py:19-109`
- `models/d11_global_local_model.py:282-286`

### C4. Global branch

Neu `use_global_branch=True`:

- `VirtualNodeGather` tinh `attn_logits [B,N]`, mask invalid nodes, softmax theo `dim=1`.
- Gather global raw: `global_context_raw = einsum("bn,bnd->bd", attn_weights, inputs)`.
- Project sang `global_context [B, global_dim]`.

Source: `models/d11_global_local_model.py:112-139`.

### C5. FiLM fusion

`FiLMFusion` nhan:

- `local_features`: `[B,K,D]`.
- `global_context`: `[B,global_dim]`.

Tra ve:

- `local_refined`: `[B,K,D]`
- `gamma`: `[B,1,D]`
- `beta`: `[B,1,D]`

Cong thuc:

```python
local_refined = local_features * (1.0 + tanh(gamma)) + beta
```

`gamma_net` va `beta_net` duoc init zero de dau training gan local-only.

Source: `models/d11_global_local_model.py:142-163`.

### C6. use_global_branch flag

- Neu `True`: tao `virtual_node_gather`, `film_fusion`; forward co `global_context`, `virtual_attn`, `gamma`, `beta`, `local_refined`.
- Neu `False`: khong tao global modules; forward dat `local_refined = local_raw`, `gamma/beta/virtual_attn=None`.
- Classifier van di qua `motif_relation(local_refined).mean(dim=1)`.

Sources:

- `models/d11_global_local_model.py:200-208`
- `models/d11_global_local_model.py:290-306`

### C7. Output dict

D11 tra ve:

- `logits`: `[B,C]`, fusion/main logits.
- `logits_local`: `[B,C]`, aux local logits.
- `center_of_mass`: `[B,K,2]`.
- `slot_attn`: `[B,K,N]`.
- `virtual_attn`: `[B,N]` hoac `None`.
- `gamma`, `beta`: `[B,1,D]` hoac `None`.
- `local_raw`: `[B,K,D]`.
- `local_raw_proj`: `[B,D]`.

Source: `models/d11_global_local_model.py:312-323`.

Rui ro: D11 khong tra `part_masks`, `class_part_attn`, `motif_embeddings`, `diagnostics`; vi vay evaluator D6-compatible chi lay duoc logits, khong co D10/D6 diagnostics.

### C8. D11 Exp H config status

`d11_exp_h_pseudo_2stage.yaml`:

- `model.name: d11_global_local_model`
- `hidden_dim: 128`
- `num_slots: 8`
- `slot_iters: 3`
- `global_dim: 64`
- `use_global_branch: true`
- loss: `d11_global_local_loss`, `lambda_local: 0.3`, `lambda_spatial: 1.0`, `lambda_supcon: 0.1`, `lambda_div: 0.02`, `ce_warmup_epochs: 30.0`

Khong co `pixel_gnn_layers`, nen D11 dung default `SharedPixelEncoder(pixel_gnn_layers=1)` tu `dual_branch_graph_swin_motif.py`.

Source: `configs/experiments/d11_exp_h_pseudo_2stage.yaml:12-29`, `models/dual_branch_graph_swin_motif.py:18-24`.

Config D11 hien co dung section `trainer`, trong khi `scripts/train_d5a.py` va `scripts/common.py` doc section `training`. Neu chay truc tiep bang `scripts/train_d5a.py`, cac field nhu `trainer.amp`, `trainer.multi_gpu`, `trainer.use_compile`, `trainer.checkpoint_monitor` se khong duoc dung. Can xac nhan notebook/script hien tai co map `trainer` -> `training` hay khong truoc khi train D12.

Sources:

- `configs/experiments/d11_exp_h_pseudo_2stage.yaml:41-49`
- `scripts/train_d5a.py:50-88`
- `scripts/common.py:382-410`

## D. Encoder Hien Tai

### D1. SharedPixelEncoder

Nam trong `models/dual_branch_graph_swin_motif.py`.

Thanh phan:

- `input_proj`: `Linear(node_dim, hidden_dim)` -> `LayerNorm` -> `GELU` -> `Dropout`.
- `pixel_layers`: `ModuleList` gom `EdgeAwarePixelMessageLayer` lap `pixel_gnn_layers` lan.

Forward:

1. `h_pixel = input_proj(x.float())`, shape `[B,N,node_dim] -> [B,N,D]`.
2. Neu co `node_mask`, mask embedding.
3. Lap tung edge-aware layer voi `edge_index`, `edge_attr.float()`, `node_mask`.
4. Return `h_pixel [B,N,D]`.

Source: `models/dual_branch_graph_swin_motif.py:15-61`.

### D2. EdgeAwarePixelMessageLayer

Nam trong `models/slot_pixel_part_graph_motif.py`.

Forward:

1. `src = edge_index[0]`, `dst = edge_index[1]`.
2. `h_src = h.index_select(dim=1, index=src)`, shape `[B,E,D]`.
3. `gate = sigmoid(edge_gate(edge_attr.float()))`.
   - Neu `edge_attr [E,F]`, gate `[E,D]`, broadcast voi `[B,E,D]`.
   - Neu `edge_attr [B,E,F]`, gate `[B,E,D]`.
4. `msg = msg_mlp(h_src) * gate`.
5. `agg = zeros_like(h)`, `agg.index_add_(dim=1, index=dst, source=msg)`.
6. Chia theo degree: `agg / deg(dst)`.
7. Optional node mask tren `agg`.
8. Residual update: `h = norm_msg(h + agg_mlp(agg))`.
9. FFN residual: `h = norm_ffn(h + ffn(h))`.
10. Optional node mask tren output.

Source: `models/slot_pixel_part_graph_motif.py:13-72`.

### D3. Edge attr usage

Edge attr chi dung trong `edge_gate`. Gate chi la ham cua `edge_attr`, khong phu thuoc truc tiep vao `h_src`/`h_dst`. Aggregation normalize theo degree, khong normalize theo tong gate.

Source: `models/slot_pixel_part_graph_motif.py:50-61`.

### D4. Multi-scale branch hien tai

Multi-scale hien tai trong D10 khong tao edge scale moi. No la hai `SharedPixelEncoder` chay tren cung `edge_index/edge_attr`:

- encoder chinh co `pixel_gnn_layers`.
- encoder phu co `pixel_gnn_layers + 1`.
- concat hidden output va project ve hidden dim bang `combine_scale`.

Source: `models/d10_slot_motif_model.py:311-323`, `models/d10_slot_motif_model.py:484-488`.

D10 best co dung branch nay vi config chain bat `multi_scale_gnn: true`. D11 Exp H khong dung branch nay.

## E. Loss Hien Tai

### E1. D10SlotMotifLoss

`D10SlotMotifLoss` ke thua `D6HierarchicalMotifLoss`.

Thanh phan inherited tu D6:

- CE tren `model_out["logits"]`.
- Slot diversity tren `model_out["part_masks"]`, cosine off-diagonal.
- Border loss tren `part_masks`.
- Slot balance loss optional.
- Slot smoothness optional tren `batch["edge_index"]`.

Sau do D10 them:

- `loss_aux_ce`: CE tren `model_out["aux_logits"]` neu co.
- `loss_supcon`: SupCon tu `model_out["motif_embeddings"].mean(dim=1)`, normalize, pairwise same-label positives.
- Diagnostics: `diag_main_accuracy`, `diag_aux_accuracy`.

Sources:

- `training/losses.py:233-352`
- `training/losses.py:867-943`

Bat buoc cho D10 loss:

- `logits`
- `part_masks`

Neu muon aux/SupCon:

- `aux_logits`
- `motif_embeddings`

### E2. D11GlobalLocalLoss

Thanh phan:

- `loss_fusion_raw = CE(model_out["logits"], y)`.
- CE warmup: neu `ce_warmup_epochs > 0`, `loss_fusion = loss_fusion_raw * current_epoch / ce_warmup_epochs` truoc khi het warmup.
- `lambda_local = lambda_local_base * ce_factor`.
- Local aux CE tren `model_out["logits_local"]`.
- Spatial prior tren `model_out["center_of_mass"]`, hard-coded slot index groups:
  - slots 0:2 bi phat neu `cx > 0.5` hoac `cy > 0.5`
  - slots 2:4 bi phat neu `cx < 0.5` hoac `cy > 0.5`
  - slots 4:6 bi phat neu `cy < 0.5`
- SupCon tren `model_out["local_raw_proj"]` bang `SupervisedContrastiveLoss`.
- SupCon warmup va optional decay.
- Diversity loss tren `model_out["local_raw"]`: normalize theo dim hidden, cosine `[B,K,K]`, phat `relu(sim - diversity_margin)` off-diagonal.
- Total:
  `loss_fusion + lambda_local*loss_local + lambda_spatial*loss_spatial + loss_supcon + lambda_div*loss_div`.

Source: `training/losses.py:973-1082`.

Bat buoc cho D11 loss:

- `logits`

Can co de dung day du D11 Exp H:

- `logits_local`
- `center_of_mass`
- `local_raw_proj`
- `local_raw`

### E3. Epoch scheduling

`D5Trainer.fit` goi:

```python
if hasattr(self.criterion, "set_epoch"):
    self.criterion.set_epoch(epoch)
```

Moi epoch truoc `train_one_epoch`. D11 loss dung `current_epoch` de tinh CE warmup, SupCon warmup/decay. D10 loss khong co `set_epoch`.

Source: `training/trainer.py:685-687`.

### E4. local auxiliary loss

D11 local aux loss lay `logits_local` tu model. D11 model tao `logits_local` bang:

```python
local_raw_pooled = self.motif_relation(local_raw).mean(dim=1)
logits_local = self.aux_classifier(local_raw_pooled)
```

Source: `models/d11_global_local_model.py:304-306`.

## F. Trainer / Eval

### F1. Build model/loss tu config

`scripts/common.py`:

- `load_config` resolve inheritance va environment.
- `prepare_training_objects` tao `GraphConfig`, inject default `height`, `width`, `connectivity` vao `model_cfg`, goi `build_model(model_cfg)`.
- Loss tao tu `build_loss(loss_cfg)`, inject `height`, `width`.
- Optimizer/scheduler build tu config.

Sources:

- `scripts/common.py:44-89`
- `scripts/common.py:360-379`

`models/registry.py` map:

- `d10_slot_motif`, `d10_slot_motif_model` -> `D10SlotMotifModel.from_config`.
- `d11_global_local`, `d11_global_local_model` -> `D11GlobalLocalModel.from_config`.

Source: `models/registry.py:23-53`.

`training/losses.py` map:

- `d10_slot_motif`, `d10_slot_motif_loss` -> `D10SlotMotifLoss`.
- `d11_global_local`, `d11_global_local_loss` -> `D11GlobalLocalLoss`.

Source: `training/losses.py:946-971`.

### F2. Training step

`D5Trainer.train_one_epoch`:

1. Batch tu dataloader.
2. Move recursive sang device.
3. `out = self.model(batch)`.
4. `loss_dict = self.criterion(out, batch["y"], batch)`.
5. Backward/optimizer/AMP/grad clip.
6. Prediction/metrics lay `out["logits"]`.
7. Log moi key trong `loss_dict`, `out["diagnostics"]`, va output diagnostics dac biet.

Sources:

- `training/trainer.py:332-410`
- `training/trainer.py:511-546`

### F3. Validation/test metric

Validation:

- `out = model(batch)`
- `loss_dict = criterion(out, batch["y"], batch)`
- `logits = out["logits"]`
- Pred = `argmax(logits, dim=1)`
- Metrics: accuracy, macro_f1, weighted_f1, per-class precision/recall/F1, pred counts.

Source: `training/trainer.py:549-591`.

Standalone evaluation:

- `scripts/evaluate_d5a.py` load checkpoint, build test loader, goi `evaluate_model`.
- `evaluate_model` chi can `out["logits"]`, `batch["graph_id"]`, `batch["x"]`, `batch["y"]`.
- Write `evaluation/metrics.json`, `predictions.csv`, confusion matrix, example grids, classification report.

Sources:

- `scripts/evaluate_d5a.py:29-105`
- `evaluation/evaluator.py:23-93`
- `evaluation/evaluator.py:170-182`

### F4. Best checkpoint

`train_d5a.py` doc:

- `training.monitor` default `val_macro_f1`.
- `checkpoint.save_best_metric` override neu co, default bang classification monitor.
- `early_stopping.monitor` override neu co.

`D5Trainer.fit` default checkpoint monitor la `val_macro_f1`, mode `max`. Neu improved, save `checkpoints/best.pth`; moi epoch save `last.pth`.

Sources:

- `scripts/train_d5a.py:50-88`
- `training/trainer.py:632-679`
- `training/trainer.py:714-745`

Checkpoint payload gom:

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `metrics`
- `best_metric`
- `best_epoch`
- `best_metric_name`
- `best_metric_mode`
- `config`
- optional scheduler/scaler state.

Source: `training/trainer.py:780-803`.

### F5. WandB / history / metrics files

WandB:

- Neu `logging.use_wandb=True`, `D5Trainer` init wandb va log full `metrics` dict moi epoch.
- Metrics dict gom train/val loss keys, diagnostics, LR keys, checkpoint/early monitor values, per-class metrics.

Sources:

- `training/trainer.py:201-216`
- `training/trainer.py:747`
- `training/trainer.py:908-910`

`training_history.json`:

- List cac metrics dict theo epoch, ghi bang `json.dump(history, indent=2)`.

Source: `training/trainer.py:902-906`.

`evaluation/metrics.json`:

- `accuracy`, `macro_f1`, `weighted_f1`, `pred_count`, `classification_report`.

Source: `scripts/evaluate_d5a.py:52-61`.

`evaluation/predictions.csv`:

- Columns: `graph_id`, `y_true`, `y_pred`, `score_0` ... `score_6`.

Source: `evaluation/evaluator.py:170-182`.

## G. Rui Ro Khi Trien Khai D12

### G1. Output keys khong nen doi neu khong sua trainer/eval/loss

Bat buoc chung:

- `logits`: `[B,7]`; trainer/eval/d11/d10 loss deu can.

Neu dung D10-style loss/diagnostics:

- `part_masks`: `[B,K,N]`
- `motif_embeddings`: `[B,K,D]`
- `class_part_attn`: `[B,7,K]`
- `class_motif_attn`: `[B,7,K]`
- optional `aux_logits`: `[B,7]`

Neu dung D11-style loss:

- `logits_local`: `[B,7]`
- `center_of_mass`: `[B,K,2]`
- `local_raw`: `[B,K,D]`
- `local_raw_proj`: `[B,D]`

Nen them alias de giam friction:

- `slot_attn` va `slot_attn_maps`
- `part_masks`
- `local_raw` va `motif_embeddings`
- `h_pixel` va `pixel_embeddings`

### G2. Shape de loi

- `edge_index`: trainer/collate dua `[B,2,E]`; encoder layer can `[2,E]`. Model phai lay `edge_index[0]`.
- `edge_attr`: hien la `[B,E,5]`; encoder can support batch edge attr. Neu D12 sinh multi-scale trong model, phai expand edge_attr dung batch.
- `node_mask`: `[B,N]`; D10 truyen vao encoder, D11 hien khong truyen. D12 nen truyen/ton trong mask nhat quan.
- `slot_attn`: loss D10 expects `[B,K,N]`, khong phai `[B,N,K]`.
- `center_of_mass`: D11 loss expects slot order co y nghia; neu D12 dung slot/refinement cua D10 ma khong dam bao FACS slot ordering, spatial prior co the phat sai.
- `coord_indices`: default `(1,2)` phu thuoc node feature order V1. Neu D12 dung node_dim 12/V2, van dung vi x/y o index 1/2 theo `NODE_FEATURE_NAMES_V2`, nhung can validate.
- `hidden_dim`: D11 Exp H dung 128; D10 best dung 96. Neu merge D10+D11 checkpoint/config, mismatch state_dict rat de xay ra.

### G3. Module nen viet moi hoan toan

- `D12MultiScaleContextEdgeEncoder`: nen la file/class moi, khong sua `SharedPixelEncoder` de tranh pha D7/D10/D11.
- `ContextAwareEdgeMessageLayer`: gate tu `[edge_attr, h_src, h_dst, h_src-h_dst]`, gate-normalized aggregation.
- Multi-scale edge builder/cache helper neu can edge scale [1,2] runtime.
- Optional `WindowPixelAttentionBlock` cho D12B, dat rieng de ablation bat/tat ro.
- `D12GlobalLocalMotifModel` wrapper hop nhat D10 motif engine + D11 FiLM.

### G4. Module chi nen tham khao / reuse co kiem soat

- Reuse `IterativeSlotAttention` D10 neu muon residual slot va slot_iterations=5.
- Reuse `MotifRelationTransformer` va `ClassMotifAttentionHead` D10 neu muon class-motif attention explainable.
- Reuse `VirtualNodeGather` va `FiLMFusion` D11 neu muon global context co bottleneck.
- Reuse `D11GlobalLocalLoss` neu D12 tra du D11 key; nhung can can nhac spatial prior slot-index hard-coded.
- Reuse `D10SlotMotifLoss` neu D12 uu tien D10-compatible motif diagnostics; nhung no khong co CE warmup/local/global ablation loss.

### G5. Artifact: sua graph repo hay sinh multi-scale trong model?

Co 2 cach:

1. Sinh multi-scale edges trong model:
   - Uu diem: khong rebuild graph repo, it pha artifact, hop D12A smoke nhanh.
   - Nhuoc diem: phai tinh/gather dynamic edge attrs cho scale 2 tu `x` trong forward neu muon `delta_intensity/similarity`; can can than AMP/device/performance.

2. Rebuild graph repo co multi-scale edges:
   - Uu diem: batch schema ro, edge_attr multi-scale day du, fast hon khi train.
   - Nhuoc diem: thay artifact contract (`edge_dim`, `edge_count`, `edge_feature_names`), can update manifest/audit, va D10/D11 cu co the khong load duoc neu dung chung path.

Khuyen nghi: D12A prototype sinh scale-2 edges trong model hoac trong D12-specific helper, khong dung chung `artifacts/graph_repo` cu. Neu ket qua tot moi build `graph_repo_d12_multiscale` rieng.

### G6. Config/trainer section risk

Repo hien co song song hai kieu section:

- D10 configs dung `training`.
- D11 configs dung `trainer`.

`scripts/train_d5a.py` va `scripts/common.py` doc `training`, khong doc `trainer`. D12 config nen dung `training` de tuong thich runner hien tai, hoac can patch runner rieng neu muon dung `trainer`.

## H. De Xuat Interface D12 Sach

### H1. Class / file

De xuat:

- File: `models/d12_global_local_motif_model.py`
- Main class: `D12GlobalLocalMotifModel`
- Encoder file: `models/d12_pixel_encoder.py`
- Encoder class: `D12MultiScaleContextEdgeEncoder`
- Optional layers:
  - `ContextAwareEdgeMessageLayer`
  - `GateNormalizedEdgeAggregation`
  - `WindowPixelAttentionBlock`

Dang ky:

- `models/registry.py`: name `d12_global_local_motif`, `d12_global_local_motif_model`.
- `models/__init__.py`: export `D12GlobalLocalMotifModel`.

### H2. Config fields

Minimal D12A:

```yaml
model:
  name: d12_global_local_motif
  num_classes: 7
  num_nodes: 2304
  node_dim: 7
  edge_dim: 5
  hidden_dim: 96
  pixel_gnn_layers: 2
  num_motifs: 8
  slot_iterations: 5
  motif_relation_layers: 1
  motif_relation_heads: 4
  dropout: 0.2

  encoder_name: d12_multiscale_context_edge
  edge_scales: [1, 2]
  use_context_aware_gate: true
  use_gate_normalized_aggregation: true
  use_edge_type_embedding: true
  edge_type_embedding_dim: 16

  use_position_encoding: false
  use_slot_refinement: true
  residual_slot_connection: true
  use_aux_classifier: true

  use_global_branch: true
  global_dim: 64
  global_dropout: 0.3
  film_type: residual_tanh
  coord_indices: [1, 2]
```

Optional D12B:

```yaml
model:
  use_window_pixel_attention: true
  window_size: 6
  shift_size: 3
  window_heads: 4
  window_layers: 1
  relative_position_bias: true
```

Training section nen dung key runner hien tai:

```yaml
training:
  device: auto
  epochs: 80
  monitor: val_macro_f1
  amp: true
  multi_gpu: true
  use_compile: true
  val_frequency: 1
  early_stopping_patience: 80
  grad_clip_norm: 3.0
```

### H3. Output dict keys bat buoc

D12 nen tra ve superset:

Bat buoc chung:

- `logits`: `[B,7]`

D10-compatible:

- `part_masks`: `[B,K,N]`
- `slot_attn_maps`: `[B,K,N]`
- `motif_embeddings`: `[B,K,D]`
- `motif_context`: `[B,K,D]`
- `class_motif_attn`: `[B,7,K]`
- `class_part_attn`: `[B,7,K]`
- `class_repr`: `[B,7,D]`
- `pixel_embeddings`: `[B,N,D]`
- `h_pixel`: `[B,N,D]`
- optional `aux_logits`: `[B,7]`
- optional `motif_centers` / `part_centers`: `[B,K,2]`

D11-compatible:

- `logits_local`: `[B,7]`
- `center_of_mass`: `[B,K,2]`
- `slot_attn`: alias `[B,K,N]`
- `virtual_attn`: `[B,N]` hoac `None`
- `gamma`: `[B,1,D]` hoac `None`
- `beta`: `[B,1,D]` hoac `None`
- `local_raw`: `[B,K,D]`
- `local_raw_proj`: `[B,D]`

Diagnostics:

- `diagnostics`: dict scalar tensors, it nhat `slot_div`, `slot_area_entropy`, `class_motif_entropy` neu co.

### H4. Loss keys bat buoc

Neu dung `D11GlobalLocalLoss`:

- `logits`
- `logits_local`
- `center_of_mass`
- `local_raw`
- `local_raw_proj`

Neu dung `D10SlotMotifLoss`:

- `logits`
- `part_masks`
- optional `aux_logits`
- optional `motif_embeddings`

Neu viet `D12GlobalLocalMotifLoss`, nen giu output loss keys:

- `loss`
- `total_loss`
- `loss_fusion` hoac `loss_ce`
- `loss_local_aux`
- `loss_spatial_prior`
- `loss_supcon`
- `loss_supcon_raw`
- `effective_lambda_supcon`
- `effective_ce_factor`
- `loss_div`
- optional `loss_slot_div`, `loss_border`, `loss_slot_balance`
- `diag_main_accuracy`
- `diag_local_accuracy`

### H5. Smoke tests can chay truoc training dai

1. Config load smoke:
   - `load_config(configs/experiments/d12a_*.yaml)` resolve du `training`, `model`, `loss`.

2. Dataloader one-batch smoke:
   - Assert batch keys: `x`, `edge_index`, `edge_attr`, `node_mask`, `y`, `graph_id`, `sample_idx`.
   - Assert shapes: `x [B,2304,7]`, `edge_index [B,2,17860]`, `edge_attr [B,17860,5]`.

3. Model forward CPU/CUDA smoke:
   - `out = model(batch)`.
   - Assert all required D12 keys.
   - Assert `logits [B,7]`, `part_masks/slot_attn [B,8,2304]`, `center_of_mass [B,8,2]`.
   - Assert all finite.

4. Loss smoke:
   - Run D12 loss or D11 loss on one batch.
   - Assert `loss` finite.
   - Assert `set_epoch(1)` va `set_epoch(30)` thay doi `effective_ce_factor` neu co warmup.

5. DataParallel edge_index smoke:
   - Simulate `edge_index [B,2,E]` va ensure model lay topology dung, khong mismatch `edge_attr [B,E,F]`.

6. Minimal train smoke:
   - `python -m scripts.train_d5a --config configs/experiments/d12a_*.yaml --max_train_batches 1 --max_val_batches 1 --epochs 1 --no_wandb`
   - Kiem tra output: `resolved_config.yaml`, `training_history.json`, `checkpoints/best.pth`, `checkpoints/last.pth`.

7. Eval smoke:
   - `python -m scripts.evaluate_d5a --config ... --checkpoint <best.pth> --max_test_batches 1 --no_wandb`
   - Kiem tra `evaluation/metrics.json`, `predictions.csv`, `classification_report.json`.

8. D12-specific diagnostics:
   - Log mean/std gate values.
   - Log gate-normalized denominator min/max.
   - Log over-smoothing cosine tren `h_pixel`.
   - Log slot area entropy va class motif entropy.

## I. Ket Luan Trien Khai

D12 nen duoc viet nhu mot mainline moi, khong sua truc tiep `SharedPixelEncoder`, `D10SlotMotifModel`, `D11GlobalLocalModel`.

Huong sach nhat:

```text
D12 = D10 motif engine
    + D11 VirtualNodeGather/FiLM
    + D11 CE warmup/SupCon/diversity schedule
    + D12 encoder moi
```

Can uu tien D12A truoc:

- context-aware edge gate
- gate-normalized aggregation
- multi-scale edge scale `[1,2]`
- output superset compatible D10/D11

Chi them window/shifted-window attention o D12B sau khi D12A chung minh `F1_Raw_Local` va full macro F1 co tin hieu tot.
