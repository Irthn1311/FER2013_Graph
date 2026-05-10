import json
from pathlib import Path

base = Path(r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_slot_motif_full_outputs\d10_p5_run2")

variants = ["d10_p5_standard", "d10_p5_sharp", "d10_p5_smooth", "d10_p5_high_lr", "d10_p5_focal", "d10_p5_motif6"]

print("=" * 100)
print(f"{'Variant':<20} {'S1 Best Loss':>12} {'S1 BestEp':>10} {'S1 LastEp':>10} {'S2 BestF1':>10} {'S2 BestEp':>10} {'S2 LastEp':>10} {'TestF1':>10}")
print("=" * 100)

for variant in variants:
    vdir = base / variant
    
    # Stage 1
    s1_hist = vdir / "d10_p5_stage1_supcon_outputs" / "training_history.json"
    s1_best_loss = s1_best_ep = s1_last_ep = "N/A"
    if s1_hist.exists():
        with open(s1_hist) as f:
            s1_data = json.load(f)
        if s1_data:
            s1_last_ep = int(s1_data[-1]["epoch"])
            best_s1 = min(s1_data, key=lambda x: x.get("val_loss", 999))
            s1_best_loss = f"{best_s1['val_loss']:.4f}"
            s1_best_ep = int(best_s1["epoch"])
    
    # Stage 2
    s2_hist = vdir / "d10_p5_stage2_relation_outputs" / "training_history.json"
    s2_best_f1 = s2_best_ep = s2_last_ep = "N/A"
    if s2_hist.exists():
        with open(s2_hist) as f:
            s2_data = json.load(f)
        if s2_data:
            s2_last_ep = int(s2_data[-1]["epoch"])
            best_s2 = max(s2_data, key=lambda x: x.get("val_macro_f1", 0))
            s2_best_f1 = f"{best_s2['val_macro_f1']:.4f}"
            s2_best_ep = int(best_s2["epoch"])
    
    # Test eval
    test_f1 = "N/A"
    metrics_path = vdir / "d10_p5_stage2_relation_outputs" / "evaluation" / "metrics.json"
    if not metrics_path.exists():
        metrics_path = vdir / "d10_p5_stage1_supcon_outputs" / "evaluation" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        test_f1 = f"{m['macro_f1']:.4f}"
    
    print(f"{variant:<20} {s1_best_loss:>12} {str(s1_best_ep):>10} {str(s1_last_ep):>10} {str(s2_best_f1):>10} {str(s2_best_ep):>10} {str(s2_last_ep):>10} {test_f1:>10}")

# Joint
print("-" * 100)
joint_path = Path(r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\outputs\d10_p5_joint_outputs")
joint_hist = joint_path / "training_history.json"
if joint_hist.exists():
    with open(joint_hist) as f:
        jdata = json.load(f)
    if jdata:
        jlast = jdata[-1]
        jbest = max(jdata, key=lambda x: x.get("val_macro_f1", 0))
        joint_test = "N/A"
        jm_path = joint_path / "evaluation" / "metrics.json"
        if jm_path.exists():
            with open(jm_path) as f:
                jm = json.load(f)
            joint_test = f"{jm['macro_f1']:.4f}"
        jf1 = f"{jbest['val_macro_f1']:.4f}"
        print(f"{'d10_p5_joint':<20} {'N/A':>12} {'N/A':>10} {int(jlast['epoch']):>10} {jf1:>10} {int(jbest['epoch']):>10} {int(jlast['epoch']):>10} {joint_test:>10}")

print()
print("=" * 100)

# Per-class details for top variants
print("\n=== PER-CLASS F1 (TEST SET) ===")
for variant in variants:
    vdir = base / variant
    report_path = vdir / "d10_p5_stage2_relation_outputs" / "evaluation" / "classification_report.txt"
    if not report_path.exists():
        report_path = vdir / "d10_p5_stage1_supcon_outputs" / "evaluation" / "classification_report.txt"
    if report_path.exists():
        print(f"\n--- {variant} ---")
        print(report_path.read_text(encoding="utf-8"))
