"""
D11: Global-Local Gated Motif Graph Network với FACS Spatial Priors
Đây là kiến trúc 1-Stage End-to-End được code độc lập 100%, không kế thừa để tránh rủi ro lệch Schema.
Hỗ trợ cả chế độ "D10 + Soft Spatial Loss" (use_global_branch=False) và "D11A Simple" (use_global_branch=True).
"""

from typing import Any, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Kế thừa duy nhất từ hệ sinh thái cũ là GNN Encoder (để đảm bảo tính tương đồng về receptive field)
# Nếu muốn độc lập hoàn toàn, có thể copy mã nguồn của SharedPixelEncoder vào đây.
from models.dual_branch_graph_swin_motif import SharedPixelEncoder


class IterativeSlotAttentionFACS(nn.Module):
    """Slot Attention có tích hợp bộ tính Trọng tâm (Center of Mass) để phục vụ FACS Priors."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_slots: int = 8,
        num_iterations: int = 3,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.eps = eps

        # Khởi tạo ngẫu nhiên
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, hidden_dim) * 0.02)
        
        self.norm_inputs = nn.LayerNorm(hidden_dim)
        self.norm_slots = nn.LayerNorm(hidden_dim)
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self, 
        inputs: torch.Tensor, 
        coords: torch.Tensor, 
        node_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        inputs: [B, N, D]
        coords: [B, N, 2] (chứa x_norm, y_norm)
        """
        bsz, num_pixels, _ = inputs.shape
        inputs_normed = self.norm_inputs(inputs)
        k = self.key_proj(inputs_normed)
        v = self.value_proj(inputs_normed)

        slots = self.slot_mu.expand(bsz, -1, -1).clone()
        
        attn_maps = None
        attn_weights = None
        
        for _ in range(self.num_iterations):
            slots_prev = slots
            q = self.query_proj(self.norm_slots(slots))
            
            scale = math.sqrt(self.hidden_dim)
            attn_logits = torch.einsum("bkd,bnd->bkn", q, k) / scale
            
            if node_mask is not None:
                attn_logits = attn_logits.masked_fill(~node_mask.bool().unsqueeze(1), -1e4)
                
            attn_maps = torch.softmax(attn_logits, dim=1) # Cạnh tranh giữa các slots
            
            if node_mask is not None:
                attn_maps = attn_maps * node_mask.to(dtype=attn_maps.dtype).unsqueeze(1)
                
            attn_sum = attn_maps.sum(dim=2, keepdim=True).clamp_min(self.eps)
            attn_weights = attn_maps / attn_sum # [B, K, N]
            
            updates = torch.einsum("bkn,bnd->bkd", attn_weights, v)
            
            slots = self.gru(
                updates.reshape(-1, self.hidden_dim), 
                slots_prev.reshape(-1, self.hidden_dim)
            )
            slots = slots.reshape(bsz, self.num_slots, self.hidden_dim)
            
            slots = slots + self.mlp(self.norm_mlp(slots))
            
        # TÍNH TRỌNG TÂM (Center of Mass) để phục vụ FACS Soft Loss
        # coords: [B, N, 2] | attn_weights: [B, K, N]
        # center_of_mass: [B, K, 2]
        center_of_mass = torch.einsum("bkn,bnc->bkc", attn_weights, coords)
        
        return slots, attn_maps, center_of_mass


class VirtualNodeGather(nn.Module):
    """Thu thập thông tin toàn cục qua Gated Attention (để tránh nhiễu nền/tóc)."""
    def __init__(self, node_dim: int, global_dim: int, dropout: float = 0.3):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.Tanh(),
            nn.Linear(node_dim // 2, 1)
        )
        self.project = nn.Sequential(
            nn.Linear(node_dim, global_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, inputs: torch.Tensor, node_mask: Optional[torch.Tensor] = None):
        # inputs: [B, N, D]
        attn_logits = self.attention_net(inputs).squeeze(-1) # [B, N]
        if node_mask is not None:
            attn_logits = attn_logits.masked_fill(~node_mask.bool(), -1e4)
        
        attn_weights = torch.softmax(attn_logits, dim=1) # [B, N]
        
        # Read-only Gather
        global_context_raw = torch.einsum("bn,bnd->bd", attn_weights, inputs) # [B, D]
        global_context = self.project(global_context_raw) # [B, Global_D]
        
        return global_context, attn_weights


class FiLMFusion(nn.Module):
    """Dung hợp điều biến (Modulation) thay vì nối chuỗi (Concat)."""
    def __init__(self, local_dim: int, global_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(global_dim, local_dim)
        self.beta_net = nn.Linear(global_dim, local_dim)
        
        # Khởi tạo an toàn (gần 0) để đầu epoch mạng giống hệt nhánh Local-only
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)
        
    def forward(self, local_features: torch.Tensor, global_context: torch.Tensor):
        # local_features: [B, K, D] | global_context: [B, Global_D]
        gamma = self.gamma_net(global_context).unsqueeze(1) # [B, 1, D]
        beta = self.beta_net(global_context).unsqueeze(1) # [B, 1, D]
        
        # Modulate
        local_refined = local_features * (1.0 + torch.tanh(gamma)) + beta
        
        return local_refined, gamma, beta


class D11GlobalLocalModel(nn.Module):
    """
    Kiến trúc mạng hoàn chỉnh.
    Nếu use_global_branch=False, mô hình hoạt động như D10 + FACS Prior.
    Nếu use_global_branch=True, mô hình là D11A Simple.
    """
    def __init__(
        self,
        num_classes: int = 7,
        hidden_dim: int = 128,
        num_slots: int = 8,
        slot_iters: int = 3,
        global_dim: int = 64,
        global_dropout: float = 0.3,
        use_global_branch: bool = True,
        coord_indices: Tuple[int, int] = (1, 2), # Default: x_norm ở index 1, y_norm ở index 2
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_global_branch = use_global_branch
        self.coord_indices = coord_indices
        
        # 1. Pixel Encoder (Graph Neural Network)
        self.encoder = SharedPixelEncoder(hidden_dim=hidden_dim, **kwargs)
        
        # 2. Local Branch (Slot Attention với FACS Priors)
        self.slot_attention = IterativeSlotAttentionFACS(
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            num_iterations=slot_iters,
        )
        
        # 3. Global Branch & Fusion
        if self.use_global_branch:
            self.virtual_node_gather = VirtualNodeGather(
                node_dim=hidden_dim, 
                global_dim=global_dim, 
                dropout=global_dropout
            )
            self.film_fusion = FiLMFusion(local_dim=hidden_dim, global_dim=global_dim)
        
        # 4. Motif Relation
        self.motif_relation = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=4, 
                dim_feedforward=hidden_dim * 2, 
                dropout=0.1, 
                batch_first=True
            ),
            num_layers=2
        )
        
        # 5. Classifiers
        # Auxiliary Classifier (Chỉ nhìn Local_raw)
        self.aux_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Main Classifier (Nhìn Fusion)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, batch_or_x, edge_index=None, edge_attr=None, node_mask=None, **kwargs):
        # 1. Bóc tách dữ liệu
        if hasattr(batch_or_x, "batch"):
            batch = batch_or_x
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            batch_vector = batch.batch
        elif isinstance(batch_or_x, dict):
            batch = batch_or_x
            x = batch.get("x", batch.get("node_features"))
            edge_index = batch["edge_index"]
            edge_attr = batch.get("edge_attr")
            batch_vector = kwargs.get('batch_obj').batch if 'batch_obj' in kwargs else None
        else:
            x = batch_or_x
            batch_vector = kwargs.get('batch_obj').batch if 'batch_obj' in kwargs else None
            
        bsz = batch_vector.max().item() + 1 if batch_vector is not None else x.shape[0]
            
        if edge_index is not None and edge_index.ndim == 3:
            edge_index = edge_index[0]
            
        # 2. GNN Encoding
        encoded_x = self.encoder(x, edge_index, edge_attr) # [B*N, D]
        
        # Dense representation for Slot Attention & Gather
        if x.dim() == 2 and batch_vector is not None:
            batch_size = int(batch_vector.max().item()) + 1
            num_nodes = x.shape[0] // batch_size
            dense_x = encoded_x.reshape(batch_size, num_nodes, -1)
            dense_raw_x = x.reshape(batch_size, num_nodes, -1)
            mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=x.device)
        else:
            dense_x = encoded_x
            dense_raw_x = x
            mask = node_mask
            
        # Lấy tọa độ không gian phục vụ FACS
        coords = dense_raw_x[:, :, self.coord_indices[0]:self.coord_indices[1]+1] # [B, N, 2]
        
        # 3. Chạy Local Branch
        local_raw, slot_attn, center_of_mass = self.slot_attention(dense_x, coords, mask) # [B, K, D]
        
        gamma, beta, virtual_attn = None, None, None
        
        # 4. Chạy Global Branch
        if self.use_global_branch:
            global_context, virtual_attn = self.virtual_node_gather(dense_x, mask) # [B, Global_D]
            local_refined, gamma, beta = self.film_fusion(local_raw, global_context) # [B, K, D]
        else:
            local_refined = local_raw
            
        # 5. Quan hệ các Motif
        local_related = self.motif_relation(local_refined) # [B, K, D]
        motif_pooled = local_related.mean(dim=1) # [B, D]
        
        # 6. Dự đoán
        logits_fusion = self.classifier(motif_pooled)
        
        # Dự đoán phụ (đảm bảo Local không chết)
        local_raw_pooled = self.motif_relation(local_raw).mean(dim=1)
        logits_local = self.aux_classifier(local_raw_pooled)
        
        # Trả về Dict để Trainer tự tính các Loss
        return {
            "logits": logits_fusion,
            "logits_local": logits_local,
            "center_of_mass": center_of_mass,
            "slot_attn": slot_attn,
            "virtual_attn": virtual_attn,
            "gamma": gamma,
            "beta": beta
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'D11GlobalLocalModel':
        cfg = dict(config)
        cfg.pop('name', None)
        for key in ("height", "width", "connectivity", "edge_hidden_dim", "gnn_layers"):
            cfg.pop(key, None)
        return cls(**cfg)
