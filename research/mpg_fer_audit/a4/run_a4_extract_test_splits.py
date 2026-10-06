"""Extract matched representations and routing metrics on Public and Private splits.

Performs:
1. Deterministic stratified 1024-sample subset selection (seed 42).
2. Verification of hooked vs unhooked logits max absolute difference = 0.0.
3. Matched extraction of pooled representations (R0, R7, R8, R9, R10) for original and flipped images.
4. Matched extraction of node representations (R1 to R6) for the 1024 subsets.
5. Routing comparison between v2.1 derived Top-K and v2.2 actual Top-K support.
6. Flip-equivariance and mirror-mapped routing support comparison.
7. Saves compressed npz and intermediate json metrics for downstream analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.stats
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "research" / "mpg_fer_audit" / "a4"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

V21_SRC = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "src"
V22_SRC = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "src"
sys.path.insert(0, str(V21_SRC))
sys.path.insert(0, str(V22_SRC))

from mpg_fer_v2_1.config import MPGConfig as MPGConfigV21
from mpg_fer_v2_1.data import FER2013Dataset, validate_split_path
from mpg_fer_v2_1.model import MPGFER as MPGFERV21, compute_motif_geometry as geom_v21

from mpg_fer_v2_2.config import MPGConfig as MPGConfigV22
from mpg_fer_v2_2.model import MPGFER as MPGFERV22, compute_motif_geometry as geom_v22

V21_CKPT = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "official_runs" / "segment_02" / "mpg_fer_v2_1_run" / "best_val_acc.pt"
V22_CKPT = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "official_runs" / "segment_02" / "mpg_fer_v2_2_run" / "best_val_acc.pt"
TOPK_SCHEDULE = [8, 16, 16, 16, 24]


def compute_chebyshev_matrix(nodes: int = 49, grid_w: int = 7) -> torch.Tensor:
    occ = torch.arange(nodes)
    r = occ // grid_w
    c = occ % grid_w
    cheb = torch.maximum((r[:, None] - r[None, :]).abs(), (c[:, None] - c[None, :]).abs())
    return cheb  # [49, 49]


def compute_mirror_indices(nodes: int = 49, grid_w: int = 7) -> torch.Tensor:
    occ = torch.arange(nodes)
    r = occ // grid_w
    c = occ % grid_w
    return r * grid_w + (grid_w - 1 - c)


def js_divergence_vectors(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute JS divergence in nats between probability vectors along last dim."""
    p = p.clamp(min=0.0)
    q = q.clamp(min=0.0)
    p_norm = p / p.sum(dim=-1, keepdim=True).clamp(min=eps)
    q_norm = q / q.sum(dim=-1, keepdim=True).clamp(min=eps)
    m = 0.5 * (p_norm + q_norm)
    
    kl_pm = (p_norm * ((p_norm + eps) / (m + eps)).log()).sum(dim=-1)
    kl_qm = (q_norm * ((q_norm + eps) / (m + eps)).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running A4 feature extraction and routing audit on {device}...", flush=True)

    # Load models
    print("Loading models...", flush=True)
    cfg21 = MPGConfigV21()
    cfg22 = MPGConfigV22()

    m21 = MPGFERV21(cfg21).to(device).eval()
    m22 = MPGFERV22(cfg22).to(device).eval()

    c21 = torch.load(V21_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    c22 = torch.load(V22_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]

    m21.load_state_dict(c21, strict=True)
    m22.load_state_dict(c22, strict=True)

    # Verify hooked vs unhooked logits = 0.0
    test_x = torch.randn(2, 1, 48, 48, device=device)
    unhooked_21, _ = m21(test_x)
    unhooked_22, _ = m22(test_x)

    captured_21 = {}
    captured_22 = {}
    hooks_21 = []
    hooks_22 = []

    def make_hook(target_dict, key):
        return lambda m, inp, out: target_dict.update({key: (out[0] if isinstance(out, tuple) else out).detach()})

    def make_pre_hook(target_dict, key):
        return lambda m, inp: target_dict.update({key: inp[0].detach()})

    for model, cap_dict, hook_list in [(m21, captured_21, hooks_21), (m22, captured_22, hooks_22)]:
        hook_list.append(model.pixel_readout_proj.register_forward_hook(make_hook(cap_dict, "R0")))
        hook_list.append(model.motif_composer.register_forward_hook(make_hook(cap_dict, "R1")))
        for i in range(5):
            hook_list.append(model.motif_gnn[i].register_forward_hook(make_hook(cap_dict, f"R{i+2}")))
        hook_list.append(model.motif_readout_proj.register_forward_hook(make_hook(cap_dict, "R7")))
        hook_list.append(model.classifier.register_forward_pre_hook(make_pre_hook(cap_dict, "R8")))
        hook_list.append(model.classifier[2].register_forward_hook(make_hook(cap_dict, "R9")))
        hook_list.append(model.classifier.register_forward_hook(make_hook(cap_dict, "R10")))

    hooked_21, _ = m21(test_x)
    hooked_22, _ = m22(test_x)
    diff21 = (hooked_21 - unhooked_21).abs().max().item()
    diff22 = (hooked_22 - unhooked_22).abs().max().item()
    print(f"Hook check: v2.1 max logit diff = {diff21}, v2.2 max logit diff = {diff22}", flush=True)
    if diff21 != 0.0 or diff22 != 0.0:
        print("STOP: A4_BLOCKED_HOOK_MISMATCH", flush=True)
        sys.exit(3)

    cheb_matrix = compute_chebyshev_matrix(49, 7).to(device)  # [49, 49]
    mirror_map = compute_mirror_indices(49, 7).to(device)      # [49]
    self_mask_49 = torch.eye(49, dtype=torch.bool, device=device).view(1, 1, 49, 49)
    non_self_mask = ~torch.eye(49, dtype=torch.bool, device=device) # [49, 49]

    # Pre-select deterministic 1024 subsets
    val_path = validate_split_path(PROJECT_ROOT / "data" / "val.csv", "val")
    test_path = validate_split_path(PROJECT_ROOT / "data" / "test.csv", "test")

    val_ds = FER2013Dataset(val_path, split="val", augment=False)
    test_ds = FER2013Dataset(test_path, split="test", augment=False)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=1024, random_state=42)
    public_sub_idx, _ = next(sss.split(range(len(val_ds)), val_ds.labels))
    public_sub_idx = np.sort(public_sub_idx)

    sss_priv = StratifiedShuffleSplit(n_splits=1, train_size=1024, random_state=42)
    private_sub_idx, _ = next(sss_priv.split(range(len(test_ds)), test_ds.labels))
    private_sub_idx = np.sort(private_sub_idx)

    subset_indices_record = {
        "public_1024_indices": public_sub_idx.tolist(),
        "private_1024_indices": private_sub_idx.tolist(),
        "seed": 42,
    }
    (AUDIT_DIR / "a4_subset_indices.json").write_text(json.dumps(subset_indices_record), encoding="utf-8")
    print(f"Saved deterministic subset indices (1024 Public, 1024 Private).", flush=True)

    public_sub_set = set(public_sub_idx.tolist())
    private_sub_set = set(private_sub_idx.tolist())

    # Processing splits
    for split_name, ds, sub_set in [("public", val_ds, public_sub_set), ("private", test_ds, private_sub_set)]:
        split_start = time.time()
        print(f"\n================ Processing {split_name.upper()} split ({len(ds)} samples) ================", flush=True)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

        # Containers for all samples
        pooled_v21_orig = {"R0": [], "R7": [], "R8": [], "R9": [], "R10": []}
        pooled_v21_flip = {"R0": [], "R7": [], "R8": [], "R9": [], "R10": []}
        pooled_v22_orig = {"R0": [], "R7": [], "R8": [], "R9": [], "R10": []}
        pooled_v22_flip = {"R0": [], "R7": [], "R8": [], "R9": [], "R10": []}

        # Containers for node representations of 1024 subset
        node_v21 = {f"R{l}": [] for l in range(1, 7)}
        node_v22 = {f"R{l}": [] for l in range(1, 7)}

        # Sample-level tracking table
        sample_rows = []

        # Layer-level routing accumulators
        # For each layer: Jaccard, Overlap, ExactMatch, JS_dense_sparse, JS_topk_sparse, Cosine, Spearman, etc.
        routing_layer_stats = {
            l_idx: {
                "jaccard": [], "overlap_k": [], "exact_match": [],
                "js_dense_sparse": [], "js_topk_sparse": [],
                "cos_dense_sparse": [], "cos_topk_sparse": [],
                "spearman_scores": [],
                "v21_local_share": [], "v21_meso_share": [], "v21_far_share": [],
                "v22_local_share": [], "v22_meso_share": [], "v22_far_share": [],
                "v21_indegree": [], "v22_indegree": [],
                "v21_edge_universe": torch.zeros((49, 49), dtype=torch.bool),
                "v22_edge_universe": torch.zeros((49, 49), dtype=torch.bool),
                "v21_edge_counts": torch.zeros((49, 49), dtype=torch.float64),
                "v22_edge_counts": torch.zeros((49, 49), dtype=torch.float64),
                "v21_flip_jaccard": [], "v22_flip_jaccard": [],
            }
            for l_idx in range(5)
        }

        # Classwise routing accumulators
        routing_class_stats = {
            c: {
                l_idx: {
                    "jaccard": [], "overlap_k": [], "exact_match": [],
                    "js_dense_sparse": [], "cos_topk_sparse": [], "spearman": [],
                }
                for l_idx in range(5)
            }
            for c in range(7)
        }

        # Flip equivariance sample metrics
        flip_metrics = {
            "v21": {"pred_agree": [], "js_div": [], "logit_l2": [], "motif_cos": [], "fusion_cos": []},
            "v22": {"pred_agree": [], "js_div": [], "logit_l2": [], "motif_cos": [], "fusion_cos": []},
        }

        global_sample_idx = 0

        with torch.no_grad():
            for batch_i, (batch_x, batch_y) in enumerate(loader):
                if (batch_i + 1) % 10 == 0 or (batch_i + 1) == len(loader):
                    print(f"  Batch {batch_i + 1}/{len(loader)} in {time.time() - split_start:.1f}s")
                batch_size = len(batch_y)
                batch_x = batch_x.to(device)
                batch_y_cpu = batch_y.tolist()
                batch_x_flip = TF.hflip(batch_x)

                # --- Model 2.1 Forward (Original & Flipped) ---
                captured_21.clear()
                with torch.amp.autocast("cuda"):
                    logits_21_orig, outputs_21_orig = m21(batch_x)
                    r0_21_orig = captured_21["R0"].cpu()
                    r7_21_orig = captured_21["R7"].cpu()
                    r8_21_orig = captured_21["R8"].cpu()
                    r9_21_orig = captured_21["R9"].cpu()
                    r10_21_orig = captured_21["R10"].cpu()
                    node_21_orig_tensors = {f"R{l}": captured_21[f"R{l}"] for l in range(1, 7)}

                captured_21.clear()
                with torch.amp.autocast("cuda"):
                    logits_21_flip, _ = m21(batch_x_flip)
                    r0_21_flip = captured_21["R0"].cpu()
                    r7_21_flip = captured_21["R7"].cpu()
                    r8_21_flip = captured_21["R8"].cpu()
                    r9_21_flip = captured_21["R9"].cpu()
                    r10_21_flip = captured_21["R10"].cpu()

                # --- Model 2.2 Forward (Original & Flipped) ---
                captured_22.clear()
                with torch.amp.autocast("cuda"):
                    logits_22_orig, outputs_22_orig = m22(batch_x, return_routing_supports=True)
                    r0_22_orig = captured_22["R0"].cpu()
                    r7_22_orig = captured_22["R7"].cpu()
                    r8_22_orig = captured_22["R8"].cpu()
                    r9_22_orig = captured_22["R9"].cpu()
                    r10_22_orig = captured_22["R10"].cpu()
                    node_22_orig_tensors = {f"R{l}": captured_22[f"R{l}"] for l in range(1, 7)}

                captured_22.clear()
                with torch.amp.autocast("cuda"):
                    logits_22_flip, outputs_22_flip = m22(batch_x_flip, return_routing_supports=True)
                    r0_22_flip = captured_22["R0"].cpu()
                    r7_22_flip = captured_22["R7"].cpu()
                    r8_22_flip = captured_22["R8"].cpu()
                    r9_22_flip = captured_22["R9"].cpu()
                    r10_22_flip = captured_22["R10"].cpu()

                # Store pooled tensors
                pooled_v21_orig["R0"].append(r0_21_orig)
                pooled_v21_orig["R7"].append(r7_21_orig)
                pooled_v21_orig["R8"].append(r8_21_orig)
                pooled_v21_orig["R9"].append(r9_21_orig)
                pooled_v21_orig["R10"].append(r10_21_orig)

                pooled_v21_flip["R0"].append(r0_21_flip)
                pooled_v21_flip["R7"].append(r7_21_flip)
                pooled_v21_flip["R8"].append(r8_21_flip)
                pooled_v21_flip["R9"].append(r9_21_flip)
                pooled_v21_flip["R10"].append(r10_21_flip)

                pooled_v22_orig["R0"].append(r0_22_orig)
                pooled_v22_orig["R7"].append(r7_22_orig)
                pooled_v22_orig["R8"].append(r8_22_orig)
                pooled_v22_orig["R9"].append(r9_22_orig)
                pooled_v22_orig["R10"].append(r10_22_orig)

                pooled_v22_flip["R0"].append(r0_22_flip)
                pooled_v22_flip["R7"].append(r7_22_flip)
                pooled_v22_flip["R8"].append(r8_22_flip)
                pooled_v22_flip["R9"].append(r9_22_flip)
                pooled_v22_flip["R10"].append(r10_22_flip)

                # Check subset membership for node representation collection
                for i_in_batch in range(batch_size):
                    cur_idx = global_sample_idx + i_in_batch
                    if cur_idx in sub_set:
                        for l in range(1, 7):
                            node_v21[f"R{l}"].append(node_21_orig_tensors[f"R{l}"][i_in_batch].cpu())
                            node_v22[f"R{l}"].append(node_22_orig_tensors[f"R{l}"][i_in_batch].cpu())

                # Predictions & probabilities for TTA
                tta_logits_21 = 0.5 * (logits_21_orig + logits_21_flip)
                tta_logits_22 = 0.5 * (logits_22_orig + logits_22_flip)

                probs_21 = F.softmax(tta_logits_21, dim=-1)
                probs_22 = F.softmax(tta_logits_22, dim=-1)

                preds_21 = probs_21.argmax(dim=-1).cpu().numpy()
                preds_22 = probs_22.argmax(dim=-1).cpu().numpy()

                conf_21 = probs_21.max(dim=-1).values.cpu().numpy()
                conf_22 = probs_22.max(dim=-1).values.cpu().numpy()

                sorted_p21 = probs_21.topk(2, dim=-1).values.cpu().numpy()
                sorted_p22 = probs_22.topk(2, dim=-1).values.cpu().numpy()
                margin_21 = sorted_p21[:, 0] - sorted_p21[:, 1]
                margin_22 = sorted_p22[:, 0] - sorted_p22[:, 1]

                entropy_21 = -(probs_21 * probs_21.clamp(min=1e-12).log()).sum(dim=-1).cpu().numpy()
                entropy_22 = -(probs_22 * probs_22.clamp(min=1e-12).log()).sum(dim=-1).cpu().numpy()

                # Build sample tracking rows
                for i_in_batch in range(batch_size):
                    cur_idx = global_sample_idx + i_in_batch
                    y = batch_y_cpu[i_in_batch]
                    p21 = int(preds_21[i_in_batch])
                    p22 = int(preds_22[i_in_batch])

                    if p21 == y and p22 == y:
                        cat = "A_both_correct"
                    elif p21 == y and p22 != y:
                        cat = "B_v21_only_correct"
                    elif p21 != y and p22 == y:
                        cat = "C_v22_only_correct"
                    elif p21 == p22:
                        cat = "D_both_wrong_same_label"
                    else:
                        cat = "E_both_wrong_different_labels"

                    sample_rows.append({
                        "split": split_name,
                        "row_index": cur_idx,
                        "true_label": y,
                        "category": cat,
                        "v2_1_pred": p21,
                        "v2_2_pred": p22,
                        "v2_1_confidence": float(conf_21[i_in_batch]),
                        "v2_2_confidence": float(conf_22[i_in_batch]),
                        "v2_1_margin": float(margin_21[i_in_batch]),
                        "v2_2_margin": float(margin_22[i_in_batch]),
                        "v2_1_entropy": float(entropy_21[i_in_batch]),
                        "v2_2_entropy": float(entropy_22[i_in_batch]),
                    })

                # --- Flip Equivariance Metrics ---
                # Model 2.1 flip metrics
                p_orig_21 = F.softmax(logits_21_orig, dim=-1)
                p_flip_21 = F.softmax(logits_21_flip, dim=-1)
                pred_agree_21 = (p_orig_21.argmax(dim=-1) == p_flip_21.argmax(dim=-1)).float().cpu().numpy()
                js_21 = js_divergence_vectors(p_orig_21, p_flip_21).cpu().numpy()
                l2_21 = torch.norm(logits_21_orig - logits_21_flip, dim=-1).cpu().numpy()
                motif_cos_21 = F.cosine_similarity(r7_21_orig.to(device), r7_21_flip.to(device), dim=-1).cpu().numpy()
                fusion_cos_21 = F.cosine_similarity(r8_21_orig.to(device), r8_21_flip.to(device), dim=-1).cpu().numpy()

                flip_metrics["v21"]["pred_agree"].extend(pred_agree_21.tolist())
                flip_metrics["v21"]["js_div"].extend(js_21.tolist())
                flip_metrics["v21"]["logit_l2"].extend(l2_21.tolist())
                flip_metrics["v21"]["motif_cos"].extend(motif_cos_21.tolist())
                flip_metrics["v21"]["fusion_cos"].extend(fusion_cos_21.tolist())

                # Model 2.2 flip metrics
                p_orig_22 = F.softmax(logits_22_orig, dim=-1)
                p_flip_22 = F.softmax(logits_22_flip, dim=-1)
                pred_agree_22 = (p_orig_22.argmax(dim=-1) == p_flip_22.argmax(dim=-1)).float().cpu().numpy()
                js_22 = js_divergence_vectors(p_orig_22, p_flip_22).cpu().numpy()
                l2_22 = torch.norm(logits_22_orig - logits_22_flip, dim=-1).cpu().numpy()
                motif_cos_22 = F.cosine_similarity(r7_22_orig.to(device), r7_22_flip.to(device), dim=-1).cpu().numpy()
                fusion_cos_22 = F.cosine_similarity(r8_22_orig.to(device), r8_22_flip.to(device), dim=-1).cpu().numpy()

                flip_metrics["v22"]["pred_agree"].extend(pred_agree_22.tolist())
                flip_metrics["v22"]["js_div"].extend(js_22.tolist())
                flip_metrics["v22"]["logit_l2"].extend(l2_22.tolist())
                flip_metrics["v22"]["motif_cos"].extend(motif_cos_22.tolist())
                flip_metrics["v22"]["fusion_cos"].extend(fusion_cos_22.tolist())

                # --- Routing Analysis (Layer by Layer) ---
                # In v2.1:
                # We replicate layer-by-layer attention computation to extract raw scores, dense attn, derived topk support
                h_m21 = node_21_orig_tensors["R1"] # [B, 49, 192]
                geom21 = geom_v21(outputs_21_orig["learned_centers_x"], outputs_21_orig["learned_centers_y"])
                
                # Flipped v2.1 for mirror comparison
                h_m21_flip = captured_21["R1"].to(device) # from flipped run
                # Note: outputs_21_orig doesn't contain flipped centers, so we run motif_composer on flipped h
                with torch.amp.autocast("cuda"):
                    h_m21_flip, _, diag_flip21 = m21.motif_composer(m21.pixel_proj(m21.pixel_extractor(batch_x_flip)))
                    geom21_flip = geom_v21(diag_flip21["learned_centers_x"], diag_flip21["learned_centers_y"])

                for l_idx, layer21 in enumerate(m21.motif_gnn):
                    k_val = TOPK_SCHEDULE[l_idx]
                    
                    # Original v2.1 attention & support
                    norm21 = layer21.norm1(h_m21)
                    reshape21 = lambda v: v.reshape(batch_size, 49, layer21.num_heads, layer21.head_dim).permute(0, 2, 1, 3)
                    q21 = reshape21(layer21.q_proj(norm21))
                    k21 = reshape21(layer21.k_proj(norm21))
                    v21_vec = reshape21(layer21.v_proj(norm21))
                    scores21 = q21 @ k21.transpose(-1, -2) / (layer21.head_dim**0.5)
                    scores21 = scores21 + layer21.geom_proj(geom21).permute(0, 3, 1, 2)
                    masked_scores21 = scores21.masked_fill(self_mask_49, torch.finfo(scores21.dtype).min)
                    
                    # Dense attention
                    attn_dense21 = F.softmax(masked_scores21, dim=-1) # [B, H, 49, 49]
                    
                    # Derived Top-K support for v2.1
                    topk_res21 = torch.topk(masked_scores21, k=k_val, dim=-1)
                    support21 = torch.zeros_like(masked_scores21, dtype=torch.bool).scatter_(-1, topk_res21.indices, True)
                    attn_topk21 = F.softmax(masked_scores21.masked_fill(~support21, torch.finfo(scores21.dtype).min), dim=-1).masked_fill(~support21, 0.0)

                    # Update h_m21 forward
                    msg21 = (layer21.attn_dropout(attn_dense21) @ v21_vec).permute(0, 2, 1, 3).reshape(batch_size, 49, 192)
                    h_m21 = h_m21 + layer21.drop_path1(layer21.out_proj(msg21))
                    h_m21 = h_m21 + layer21.drop_path2(layer21.ffn(layer21.norm2(h_m21)))

                    # Flipped v2.1 support for flip Jaccard
                    norm21_f = layer21.norm1(h_m21_flip)
                    q21_f = reshape21(layer21.q_proj(norm21_f))
                    k21_f = reshape21(layer21.k_proj(norm21_f))
                    v21_f = reshape21(layer21.v_proj(norm21_f))
                    scores21_f = q21_f @ k21_f.transpose(-1, -2) / (layer21.head_dim**0.5) + layer21.geom_proj(geom21_flip).permute(0, 3, 1, 2)
                    masked_scores21_f = scores21_f.masked_fill(self_mask_49, torch.finfo(scores21_f.dtype).min)
                    topk_res21_f = torch.topk(masked_scores21_f, k=k_val, dim=-1)
                    support21_flip = torch.zeros_like(masked_scores21_f, dtype=torch.bool).scatter_(-1, topk_res21_f.indices, True)
                    # update h_m21_flip
                    msg21_f = (layer21.attn_dropout(F.softmax(masked_scores21_f, dim=-1)) @ v21_f).permute(0, 2, 1, 3).reshape(batch_size, 49, 192)
                    h_m21_flip = h_m21_flip + layer21.drop_path1(layer21.out_proj(msg21_f))
                    h_m21_flip = h_m21_flip + layer21.drop_path2(layer21.ffn(layer21.norm2(h_m21_flip)))

                    # v2.2 layer diagnostics
                    layer22 = m22.motif_gnn[l_idx]
                    # We can obtain scores and attention directly or from diagnostic
                    # Let's compute directly for perfect matched consistency
                    h_m22 = node_22_orig_tensors[f"R{l_idx+1}"]
                    norm22 = layer22.norm1(h_m22)
                    reshape22 = lambda v: v.reshape(batch_size, 49, layer22.num_heads, layer22.head_dim).permute(0, 2, 1, 3)
                    q22 = reshape22(layer22.q_proj(norm22))
                    k22 = reshape22(layer22.k_proj(norm22))
                    scores22 = q22 @ k22.transpose(-1, -2) / (layer22.head_dim**0.5)
                    scores22 = scores22 + layer22.geom_proj(outputs_22_orig["motif_geometry"]).permute(0, 3, 1, 2)
                    masked_scores22 = scores22.masked_fill(self_mask_49, torch.finfo(scores22.dtype).min)
                    topk_res22 = torch.topk(masked_scores22, k=k_val, dim=-1)
                    support22 = torch.zeros_like(masked_scores22, dtype=torch.bool).scatter_(-1, topk_res22.indices, True)
                    attn_sparse22 = F.softmax(masked_scores22.masked_fill(~support22, torch.finfo(scores22.dtype).min), dim=-1).masked_fill(~support22, 0.0)

                    # Flipped v2.2 support
                    # From outputs_22_flip:
                    # Let's compute flipped scores for layer 22
                    h_m22_f = captured_22[f"R{l_idx+1}"].to(device) # layer input from flipped forward
                    norm22_f = layer22.norm1(h_m22_f)
                    q22_f = reshape22(layer22.q_proj(norm22_f))
                    k22_f = reshape22(layer22.k_proj(norm22_f))
                    scores22_f = q22_f @ k22_f.transpose(-1, -2) / (layer22.head_dim**0.5) + layer22.geom_proj(outputs_22_flip["motif_geometry"]).permute(0, 3, 1, 2)
                    masked_scores22_f = scores22_f.masked_fill(self_mask_49, torch.finfo(scores22_f.dtype).min)
                    topk_res22_f = torch.topk(masked_scores22_f, k=k_val, dim=-1)
                    support22_flip = torch.zeros_like(masked_scores22_f, dtype=torch.bool).scatter_(-1, topk_res22_f.indices, True)

                    # Extract 48 non-self elements: [B, H, 49, 48]
                    # non_self_mask has 48 True elements per row
                    s21_nonself = support21[:, :, non_self_mask].reshape(batch_size, 6, 49, 48)
                    s22_nonself = support22[:, :, non_self_mask].reshape(batch_size, 6, 49, 48)

                    dense21_vec = attn_dense21[:, :, non_self_mask].reshape(batch_size, 6, 49, 48)
                    topk21_vec = attn_topk21[:, :, non_self_mask].reshape(batch_size, 6, 49, 48)
                    sparse22_vec = attn_sparse22[:, :, non_self_mask].reshape(batch_size, 6, 49, 48)

                    scores21_nonself = scores21[:, :, non_self_mask].reshape(batch_size, 6, 49, 48)
                    scores22_nonself = scores22[:, :, non_self_mask].reshape(batch_size, 6, 49, 48)

                    # 1. Support Jaccard, Overlap, Exact match
                    inter = (s21_nonself & s22_nonself).sum(dim=-1).float() # [B, H, 49]
                    union = (s21_nonself | s22_nonself).sum(dim=-1).float().clamp(min=1.0)
                    jaccard = inter / union # [B, H, 49]
                    overlap_k = inter / float(k_val)
                    exact_match = (s21_nonself == s22_nonself).all(dim=-1).float()

                    # Average over queries and heads per sample: [B]
                    jacc_sample = jaccard.mean(dim=(1, 2)).cpu().numpy()
                    overlap_sample = overlap_k.mean(dim=(1, 2)).cpu().numpy()
                    exact_sample = exact_match.mean(dim=(1, 2)).cpu().numpy()

                    routing_layer_stats[l_idx]["jaccard"].extend(jacc_sample.tolist())
                    routing_layer_stats[l_idx]["overlap_k"].extend(overlap_sample.tolist())
                    routing_layer_stats[l_idx]["exact_match"].extend(exact_sample.tolist())

                    # 2. Attention JS Divergence & Cosine
                    # js between original dense vs sparse22
                    js_dense_sp = js_divergence_vectors(dense21_vec, sparse22_vec).mean(dim=(1, 2)).cpu().numpy()
                    # js between topk-renorm vs sparse22
                    js_topk_sp = js_divergence_vectors(topk21_vec, sparse22_vec).mean(dim=(1, 2)).cpu().numpy()

                    cos_dense_sp = F.cosine_similarity(dense21_vec, sparse22_vec, dim=-1).mean(dim=(1, 2)).cpu().numpy()
                    cos_topk_sp = F.cosine_similarity(topk21_vec, sparse22_vec, dim=-1).mean(dim=(1, 2)).cpu().numpy()

                    routing_layer_stats[l_idx]["js_dense_sparse"].extend(js_dense_sp.tolist())
                    routing_layer_stats[l_idx]["js_topk_sparse"].extend(js_topk_sp.tolist())
                    routing_layer_stats[l_idx]["cos_dense_sparse"].extend(cos_dense_sp.tolist())
                    routing_layer_stats[l_idx]["cos_topk_sparse"].extend(cos_topk_sp.tolist())

                    # 3. Vectorized Spearman correlation of pre-softmax scores across 48 keys
                    # Fractional ranks in PyTorch along dim=-1
                    r1 = scores21_nonself.argsort(dim=-1).argsort(dim=-1).float()
                    r2 = scores22_nonself.argsort(dim=-1).argsort(dim=-1).float()
                    r1_c = r1 - r1.mean(dim=-1, keepdim=True)
                    r2_c = r2 - r2.mean(dim=-1, keepdim=True)
                    denom = (torch.norm(r1_c, dim=-1) * torch.norm(r2_c, dim=-1)).clamp(min=1e-12)
                    sp_vec = (r1_c * r2_c).sum(dim=-1) / denom  # [B, H, 49]
                    spearman_batch = sp_vec.mean(dim=(1, 2)).cpu().tolist() # [B]

                    routing_layer_stats[l_idx]["spearman_scores"].extend(spearman_batch)

                    # Classwise recording
                    for b_i in range(batch_size):
                        y_cls = batch_y_cpu[b_i]
                        routing_class_stats[y_cls][l_idx]["jaccard"].append(float(jacc_sample[b_i]))
                        routing_class_stats[y_cls][l_idx]["overlap_k"].append(float(overlap_sample[b_i]))
                        routing_class_stats[y_cls][l_idx]["exact_match"].append(float(exact_sample[b_i]))
                        routing_class_stats[y_cls][l_idx]["js_dense_sparse"].append(float(js_dense_sp[b_i]))
                        routing_class_stats[y_cls][l_idx]["cos_topk_sparse"].append(float(cos_topk_sp[b_i]))
                        routing_class_stats[y_cls][l_idx]["spearman"].append(float(spearman_batch[b_i]))

                    # 4. Distance bin composition & in-degree
                    # cheb_matrix: [49, 49]
                    cheb_exp = cheb_matrix.unsqueeze(0).unsqueeze(0) # [1, 1, 49, 49]
                    
                    loc_mask = (cheb_exp == 1)
                    meso_mask = (cheb_exp == 2) | (cheb_exp == 3)
                    far_mask = (cheb_exp >= 4)

                    def get_shares(supp):
                        cnt = supp.sum().float().clamp(min=1.0)
                        l_sh = (supp & loc_mask).sum().float() / cnt
                        m_sh = (supp & meso_mask).sum().float() / cnt
                        f_sh = (supp & far_mask).sum().float() / cnt
                        return l_sh.item(), m_sh.item(), f_sh.item()

                    l21, m21_sh, f21 = get_shares(support21)
                    l22, m22_sh, f22 = get_shares(support22)

                    routing_layer_stats[l_idx]["v21_local_share"].append(l21)
                    routing_layer_stats[l_idx]["v21_meso_share"].append(m21_sh)
                    routing_layer_stats[l_idx]["v21_far_share"].append(f21)

                    routing_layer_stats[l_idx]["v22_local_share"].append(l22)
                    routing_layer_stats[l_idx]["v22_meso_share"].append(m22_sh)
                    routing_layer_stats[l_idx]["v22_far_share"].append(f22)

                    # In-degree of key node j = sum over query nodes i of support[..., i, j]
                    # shape [B, H, 49]
                    indeg21 = support21.sum(dim=-2).float()
                    indeg22 = support22.sum(dim=-2).float()
                    routing_layer_stats[l_idx]["v21_indegree"].extend(indeg21.mean(dim=(1, 2)).cpu().tolist())
                    routing_layer_stats[l_idx]["v22_indegree"].extend(indeg22.mean(dim=(1, 2)).cpu().tolist())

                    # Universe coverage accumulator & frequency counts
                    routing_layer_stats[l_idx]["v21_edge_universe"] |= support21.any(dim=(0, 1)).cpu()
                    routing_layer_stats[l_idx]["v22_edge_universe"] |= support22.any(dim=(0, 1)).cpu()
                    routing_layer_stats[l_idx]["v21_edge_counts"] += support21.sum(dim=(0, 1)).cpu().to(torch.float64)
                    routing_layer_stats[l_idx]["v22_edge_counts"] += support22.sum(dim=(0, 1)).cpu().to(torch.float64)

                    # 5. Flip Equivariance Support Jaccard
                    # mirror_map: for query i and key j: mirror_map[i], mirror_map[j]
                    # S_flip_mirror[b, h, i, j] = S_flip[b, h, mirror[i], mirror[j]]
                    supp21_f_mirr = support21_flip[:, :, mirror_map, :][:, :, :, mirror_map]
                    supp22_f_mirr = support22_flip[:, :, mirror_map, :][:, :, :, mirror_map]

                    inter21_f = (support21 & supp21_f_mirr).sum(dim=(-1, -2)).float()
                    union21_f = (support21 | supp21_f_mirr).sum(dim=(-1, -2)).float().clamp(min=1.0)
                    jacc21_f = (inter21_f / union21_f).mean(dim=1).cpu().numpy() # [B]

                    inter22_f = (support22 & supp22_f_mirr).sum(dim=(-1, -2)).float()
                    union22_f = (support22 | supp22_f_mirr).sum(dim=(-1, -2)).float().clamp(min=1.0)
                    jacc22_f = (inter22_f / union22_f).mean(dim=1).cpu().numpy() # [B]

                    routing_layer_stats[l_idx]["v21_flip_jaccard"].extend(jacc21_f.tolist())
                    routing_layer_stats[l_idx]["v22_flip_jaccard"].extend(jacc22_f.tolist())

                global_sample_idx += batch_size

        print(f"Finished {split_name} extraction in {time.time() - split_start:.1f}s.")

        # Save pooled representations NPZ
        npz_pooled_path = AUDIT_DIR / f"{split_name}_pooled_features.npz"
        np.savez_compressed(
            npz_pooled_path,
            v21_orig_r0=torch.cat(pooled_v21_orig["R0"]).numpy().astype(np.float32),
            v21_orig_r7=torch.cat(pooled_v21_orig["R7"]).numpy().astype(np.float32),
            v21_orig_r8=torch.cat(pooled_v21_orig["R8"]).numpy().astype(np.float32),
            v21_orig_r9=torch.cat(pooled_v21_orig["R9"]).numpy().astype(np.float32),
            v21_orig_r10=torch.cat(pooled_v21_orig["R10"]).numpy().astype(np.float32),
            v21_flip_r0=torch.cat(pooled_v21_flip["R0"]).numpy().astype(np.float32),
            v21_flip_r7=torch.cat(pooled_v21_flip["R7"]).numpy().astype(np.float32),
            v21_flip_r8=torch.cat(pooled_v21_flip["R8"]).numpy().astype(np.float32),
            v21_flip_r9=torch.cat(pooled_v21_flip["R9"]).numpy().astype(np.float32),
            v21_flip_r10=torch.cat(pooled_v21_flip["R10"]).numpy().astype(np.float32),
            v22_orig_r0=torch.cat(pooled_v22_orig["R0"]).numpy().astype(np.float32),
            v22_orig_r7=torch.cat(pooled_v22_orig["R7"]).numpy().astype(np.float32),
            v22_orig_r8=torch.cat(pooled_v22_orig["R8"]).numpy().astype(np.float32),
            v22_orig_r9=torch.cat(pooled_v22_orig["R9"]).numpy().astype(np.float32),
            v22_orig_r10=torch.cat(pooled_v22_orig["R10"]).numpy().astype(np.float32),
            v22_flip_r0=torch.cat(pooled_v22_flip["R0"]).numpy().astype(np.float32),
            v22_flip_r7=torch.cat(pooled_v22_flip["R7"]).numpy().astype(np.float32),
            v22_flip_r8=torch.cat(pooled_v22_flip["R8"]).numpy().astype(np.float32),
            v22_flip_r9=torch.cat(pooled_v22_flip["R9"]).numpy().astype(np.float32),
            v22_flip_r10=torch.cat(pooled_v22_flip["R10"]).numpy().astype(np.float32),
            targets=ds.labels.astype(np.int64),
        )
        print(f"Saved {npz_pooled_path} (dtype float32).")

        # Save node representations NPZ for 1024 subset
        npz_node_path = AUDIT_DIR / f"{split_name}_1024_node_features.npz"
        np.savez_compressed(
            npz_node_path,
            v21_R1=torch.stack(node_v21["R1"]).numpy().astype(np.float32),
            v21_R2=torch.stack(node_v21["R2"]).numpy().astype(np.float32),
            v21_R3=torch.stack(node_v21["R3"]).numpy().astype(np.float32),
            v21_R4=torch.stack(node_v21["R4"]).numpy().astype(np.float32),
            v21_R5=torch.stack(node_v21["R5"]).numpy().astype(np.float32),
            v21_R6=torch.stack(node_v21["R6"]).numpy().astype(np.float32),
            v22_R1=torch.stack(node_v22["R1"]).numpy().astype(np.float32),
            v22_R2=torch.stack(node_v22["R2"]).numpy().astype(np.float32),
            v22_R3=torch.stack(node_v22["R3"]).numpy().astype(np.float32),
            v22_R4=torch.stack(node_v22["R4"]).numpy().astype(np.float32),
            v22_R5=torch.stack(node_v22["R5"]).numpy().astype(np.float32),
            v22_R6=torch.stack(node_v22["R6"]).numpy().astype(np.float32),
        )
        print(f"Saved {npz_node_path} (1024 samples x 49 nodes x 192 dims).")

        # Save sample tracking rows JSON
        (AUDIT_DIR / f"{split_name}_sample_audit_table.json").write_text(json.dumps(sample_rows, indent=2), encoding="utf-8")

        # Compile layer routing summary for this split
        split_routing_summary = {}
        for l_idx in range(5):
            k_val = TOPK_SCHEDULE[l_idx]
            stats = routing_layer_stats[l_idx]

            # Universe coverage non-self (49*48 = 2352)
            u21 = float(stats["v21_edge_universe"][non_self_mask.cpu()].sum()) / 2352.0
            u22 = float(stats["v22_edge_universe"][non_self_mask.cpu()].sum()) / 2352.0

            # Edge frequency correlation across 2352 non-self edges
            freq21 = (stats["v21_edge_counts"][non_self_mask.cpu()]).numpy()
            freq22 = (stats["v22_edge_counts"][non_self_mask.cpu()]).numpy()
            if np.std(freq21) > 1e-9 and np.std(freq22) > 1e-9:
                p_corr, _ = scipy.stats.pearsonr(freq21, freq22)
                s_corr, _ = scipy.stats.spearmanr(freq21, freq22)
            else:
                p_corr, s_corr = 0.0, 0.0

            split_routing_summary[f"layer_{l_idx+1}"] = {
                "layer_index": l_idx,
                "K": k_val,
                "support_jaccard_mean": float(np.mean(stats["jaccard"])),
                "support_jaccard_std": float(np.std(stats["jaccard"])),
                "overlap_k_mean": float(np.mean(stats["overlap_k"])),
                "exact_match_rate": float(np.mean(stats["exact_match"])),
                "js_dense_sparse_mean": float(np.mean(stats["js_dense_sparse"])),
                "js_topk_sparse_mean": float(np.mean(stats["js_topk_sparse"])),
                "cos_dense_sparse_mean": float(np.mean(stats["cos_dense_sparse"])),
                "cos_topk_sparse_mean": float(np.mean(stats["cos_topk_sparse"])),
                "spearman_scores_mean": float(np.mean(stats["spearman_scores"])),
                "v21_local_share_mean": float(np.mean(stats["v21_local_share"])),
                "v21_meso_share_mean": float(np.mean(stats["v21_meso_share"])),
                "v21_far_share_mean": float(np.mean(stats["v21_far_share"])),
                "v22_local_share_mean": float(np.mean(stats["v22_local_share"])),
                "v22_meso_share_mean": float(np.mean(stats["v22_meso_share"])),
                "v22_far_share_mean": float(np.mean(stats["v22_far_share"])),
                "v21_indegree_mean": float(np.mean(stats["v21_indegree"])),
                "v21_indegree_std": float(np.std(stats["v21_indegree"])),
                "v22_indegree_mean": float(np.mean(stats["v22_indegree"])),
                "v22_indegree_std": float(np.std(stats["v22_indegree"])),
                "v21_edge_universe_coverage": u21,
                "v22_edge_universe_coverage": u22,
                "edge_frequency_pearson": float(p_corr),
                "edge_frequency_spearman": float(s_corr),
                "v21_flip_jaccard_mean": float(np.mean(stats["v21_flip_jaccard"])),
                "v22_flip_jaccard_mean": float(np.mean(stats["v22_flip_jaccard"])),
            }

        (AUDIT_DIR / f"{split_name}_routing_summary.json").write_text(json.dumps(split_routing_summary, indent=2), encoding="utf-8")

        # Compile classwise routing summary
        class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        classwise_summary = {}
        for c in range(7):
            c_name = class_names[c]
            classwise_summary[c_name] = {}
            for l_idx in range(5):
                st = routing_class_stats[c][l_idx]
                classwise_summary[c_name][f"layer_{l_idx+1}"] = {
                    "support_jaccard_mean": float(np.mean(st["jaccard"])),
                    "overlap_k_mean": float(np.mean(st["overlap_k"])),
                    "exact_match_rate": float(np.mean(st["exact_match"])),
                    "js_dense_sparse_mean": float(np.mean(st["js_dense_sparse"])),
                    "cos_topk_sparse_mean": float(np.mean(st["cos_topk_sparse"])),
                    "spearman_mean": float(np.mean(st["spearman"])),
                }
        (AUDIT_DIR / f"{split_name}_routing_classwise.json").write_text(json.dumps(classwise_summary, indent=2), encoding="utf-8")

        # Compile flip equivariance summary
        flip_summary = {
            "v2_1": {k: float(np.mean(v)) for k, v in flip_metrics["v21"].items()},
            "v2_2": {k: float(np.mean(v)) for k, v in flip_metrics["v22"].items()},
            "routing_support_flip_jaccard": {
                f"layer_{l_idx+1}": {
                    "v2_1": float(np.mean(routing_layer_stats[l_idx]["v21_flip_jaccard"])),
                    "v2_2": float(np.mean(routing_layer_stats[l_idx]["v22_flip_jaccard"])),
                    "delta_v22_minus_v21": float(np.mean(routing_layer_stats[l_idx]["v22_flip_jaccard"])) - float(np.mean(routing_layer_stats[l_idx]["v21_flip_jaccard"])),
                }
                for l_idx in range(5)
            }
        }
        (AUDIT_DIR / f"{split_name}_flip_equivariance.json").write_text(json.dumps(flip_summary, indent=2), encoding="utf-8")

    # Combine Public and Private routing summaries into the official required output files
    pub_routing = json.loads((AUDIT_DIR / "public_routing_summary.json").read_text())
    priv_routing = json.loads((AUDIT_DIR / "private_routing_summary.json").read_text())
    (AUDIT_DIR / "a4_routing_overlap.json").write_text(
        json.dumps({"public": pub_routing, "private": priv_routing}, indent=2), encoding="utf-8"
    )

    pub_cls_routing = json.loads((AUDIT_DIR / "public_routing_classwise.json").read_text())
    priv_cls_routing = json.loads((AUDIT_DIR / "private_routing_classwise.json").read_text())
    (AUDIT_DIR / "a4_routing_classwise.json").write_text(
        json.dumps({"public": pub_cls_routing, "private": priv_cls_routing}, indent=2), encoding="utf-8"
    )

    pub_flip = json.loads((AUDIT_DIR / "public_flip_equivariance.json").read_text())
    priv_flip = json.loads((AUDIT_DIR / "private_flip_equivariance.json").read_text())
    (AUDIT_DIR / "a4_flip_equivariance.json").write_text(
        json.dumps({"public": pub_flip, "private": priv_flip}, indent=2), encoding="utf-8"
    )

    print(f"\nAll Public and Private feature extraction and routing audits completed in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
