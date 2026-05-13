with open('training/losses.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Add diversity_margin to init
if 'self.diversity_margin =' not in content:
    init_hook = '''        self.lambda_div = float(cfg.get('lambda_div', 0.05))
        self.warmup_epochs = float(cfg.get('warmup_epochs', 5.0))'''
    new_init_hook = '''        self.lambda_div = float(cfg.get('lambda_div', 0.05))
        self.diversity_margin = float(cfg.get('diversity_margin', 0.3))
        self.warmup_epochs = float(cfg.get('warmup_epochs', 5.0))'''
    content = content.replace(init_hook, new_init_hook)

# Update forward loss_div
fwd_hook = '''            eye = torch.eye(K, device=local_raw.device).unsqueeze(0)
            off_diag_sim = sim_matrix * (1.0 - eye)
            loss_div = F.relu(off_diag_sim).mean()'''
new_fwd_hook = '''            eye = torch.eye(K, device=local_raw.device).unsqueeze(0)
            # Mask out diagonal completely by making it highly negative
            masked_sim = sim_matrix - eye * 999.0
            # Apply margin to off-diagonal
            loss_div = F.relu(masked_sim - self.diversity_margin).mean()'''
if fwd_hook in content:
    content = content.replace(fwd_hook, new_fwd_hook)

with open('training/losses.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated losses.py')
