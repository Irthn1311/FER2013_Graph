"""Update all Stage 1 variant configs with slot diversity and speed settings.
These layer on top of the base d10_p5_stage1_supcon.yaml via inheritance.
We only need to ensure the variant-specific overrides (like supcon_temperature)
are preserved, plus add any missing slot diversity / speed settings.
"""
import glob
from pathlib import Path

stage1_variants = glob.glob("configs/experiments/d10_p5_stage1_supcon_*.yaml")

# Each variant already inherits from base which now has:
# - lambda_slot_div: 0.1, lambda_slot_balance: 0.01
# - epochs: 80, val_frequency: 5, t_max: 80
# - supcon_temperature: 0.07
# So we only need to update scheduler t_max and training epochs to match base
# The variant-specific overrides (like temperature) are fine

for path_str in sorted(stage1_variants):
    path = Path(path_str)
    content = path.read_text()
    
    # Make sure scheduler t_max matches epochs=80 (inherited from base)
    # and remove any stale training/monitor overrides that are now in base
    
    # Remove old redundant blocks that are now in base
    lines = content.strip().split("\n")
    new_lines = []
    skip_section = None
    for line in lines:
        # Skip redundant checkpoint/early_stopping/training blocks
        # (these are now correctly set in the base config)
        stripped = line.strip()
        if stripped in ("checkpoint:", "early_stopping:", "training:"):
            skip_section = stripped
            continue
        if skip_section and (line.startswith("  ") or line.startswith("\t")):
            continue
        skip_section = None
        new_lines.append(line)
    
    # Re-add only training monitor override (inherits epochs/patience from base)
    new_content = "\n".join(new_lines).rstrip()
    if "training:" not in new_content:
        new_content += "\n\ntraining:\n  monitor: val_loss\n"
    if "checkpoint:" not in new_content:
        new_content += "\ncheckpoint:\n  save_best_metric: val_loss\n  save_best_mode: min\n"
    if "early_stopping:" not in new_content:
        new_content += "\nearly_stopping:\n  monitor: val_loss\n  mode: min\n"
    
    path.write_text(new_content + "\n")
    print(f"Updated: {path.name}")

print(f"\nTotal updated: {len(stage1_variants)} variant configs")
