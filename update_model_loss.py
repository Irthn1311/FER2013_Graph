import sys
import torch

# 1. Update D11GlobalLocalModel
with open('models/d11_global_local_model.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re
if 'self.supcon_proj =' not in content:
    init_hook = '''        # Main Classifier (Nhìn Fusion)
        self.classifier = nn.Sequential('''
    new_init_hook = '''        # SupCon Projection Head (for pooled local_raw)
        self.supcon_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main Classifier (Nhìn Fusion)
        self.classifier = nn.Sequential('''
    content = content.replace(init_hook, new_init_hook)

if 'local_raw_proj' not in content:
    fwd_hook = '''        # Trả về Dict để Trainer tự tính các Loss
        return {
            "logits": logits_fusion,
            "logits_local": logits_local,
            "center_of_mass": center_of_mass,
            "slot_attn": slot_attn,
            "virtual_attn": virtual_attn,
            "gamma": gamma,
            "beta": beta
        }'''
    new_fwd_hook = '''        # SupCon Projection
        local_raw_mean = local_raw.mean(dim=1) # [B, D]
        local_raw_proj = self.supcon_proj(local_raw_mean) # [B, D]

        # Trả về Dict để Trainer tự tính các Loss
        return {
            "logits": logits_fusion,
            "logits_local": logits_local,
            "center_of_mass": center_of_mass,
            "slot_attn": slot_attn,
            "virtual_attn": virtual_attn,
            "gamma": gamma,
            "beta": beta,
            "local_raw": local_raw,
            "local_raw_proj": local_raw_proj
        }'''
    content = content.replace(fwd_hook, new_fwd_hook)

with open('models/d11_global_local_model.py', 'w', encoding='utf-8') as f:
    f.write(content)

# 2. Update D11GlobalLocalLoss
with open('training/losses.py', 'r', encoding='utf-8') as f:
    losses_content = f.read()

if 'self.lambda_supcon =' not in losses_content:
    loss_init_hook = '''        self.lambda_local = float(cfg.get('lambda_local', 0.5))
        self.lambda_spatial = float(cfg.get('lambda_spatial', 1.0))
        label_smoothing = float(cfg.get('label_smoothing', 0.1))
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)'''
    new_loss_init = '''        self.lambda_local = float(cfg.get('lambda_local', 0.5))
        self.lambda_spatial = float(cfg.get('lambda_spatial', 1.0))
        self.lambda_supcon = float(cfg.get('lambda_supcon', 0.1))
        self.lambda_div = float(cfg.get('lambda_div', 0.05))
        self.warmup_epochs = float(cfg.get('warmup_epochs', 5.0))
        self.current_epoch = 0.0
        
        label_smoothing = float(cfg.get('label_smoothing', 0.1))
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        from training.supcon_loss import SupervisedContrastiveLoss
        self.supcon = SupervisedContrastiveLoss(temperature=0.1)

    def set_epoch(self, epoch: float):
        self.current_epoch = float(epoch)'''
    losses_content = losses_content.replace(loss_init_hook, new_loss_init)

    loss_fwd_hook = '''        total = loss_fusion + self.lambda_local * loss_local + self.lambda_spatial * loss_spatial

        out = {
            'loss': total,
            'loss_fusion': loss_fusion,
            'loss_local_aux': loss_local,
            'loss_spatial_prior': loss_spatial,
            'diag_main_accuracy': (logits.argmax(dim=1) == y).float().mean().detach()
        }'''
    new_loss_fwd = '''        # SupCon Loss
        local_raw_proj = model_out.get('local_raw_proj')
        if local_raw_proj is not None and self.lambda_supcon > 0.0:
            loss_supcon_val = self.supcon(local_raw_proj, y)
            warmup_factor = min(1.0, max(1.0, self.current_epoch) / max(1.0, self.warmup_epochs))
            loss_supcon = loss_supcon_val * warmup_factor
        else:
            loss_supcon = logits.new_zeros(())
            
        # Slot Diversity Loss
        local_raw = model_out.get('local_raw') # [B, K, D]
        if local_raw is not None and self.lambda_div > 0.0:
            z_norm = F.normalize(local_raw, dim=-1) # [B, K, D]
            sim_matrix = torch.bmm(z_norm, z_norm.transpose(1, 2)) # [B, K, K]
            K = local_raw.size(1)
            eye = torch.eye(K, device=local_raw.device).unsqueeze(0)
            off_diag_sim = sim_matrix * (1.0 - eye)
            loss_div = F.relu(off_diag_sim).mean()
        else:
            loss_div = logits.new_zeros(())

        total = loss_fusion + self.lambda_local * loss_local + self.lambda_spatial * loss_spatial + self.lambda_supcon * loss_supcon + self.lambda_div * loss_div

        out = {
            'loss': total,
            'loss_fusion': loss_fusion,
            'loss_local_aux': loss_local,
            'loss_spatial_prior': loss_spatial,
            'loss_supcon': loss_supcon,
            'loss_div': loss_div,
            'diag_main_accuracy': (logits.argmax(dim=1) == y).float().mean().detach()
        }'''
    losses_content = losses_content.replace(loss_fwd_hook, new_loss_fwd)
    
with open('training/losses.py', 'w', encoding='utf-8') as f:
    f.write(losses_content)

print('Updated model and loss.')
