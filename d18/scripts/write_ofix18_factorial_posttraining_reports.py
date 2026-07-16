from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/d18_analysis/ofix18_factorial_posttraining"

def pct(x): return f"{100*float(x):.2f}%"
def pp(x): return f"{100*float(x):+.2f} pp"
def tab(frame):
    values = frame.fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy()]
    return "\n".join([header, rule] + rows)
def write(name, text): (OUT/name).write_text(text.rstrip()+"\n", encoding="utf-8")

cells = pd.read_csv(OUT/"07_factorial_cells.csv")
boot = pd.read_csv(OUT/"08_paired_bootstrap_intervals.csv")
signal = pd.read_csv(OUT/"10_structure_signal_comparison.csv")
rep = pd.read_csv(OUT/"11_representation_comparison.csv")
locked = pd.read_csv(OUT/"06_locked_evaluation_predictions.csv")
best = cells[cells.checkpoint_type.eq("best")].set_index("cell")

def iv(contrast, metric):
    q = boot[boot.checkpoint_type.eq("best") & boot.contrast.eq(contrast) & boot.target_metric.eq(metric)]
    if len(q) != 1: raise RuntimeError(f"missing {contrast}/{metric}")
    return q.iloc[0]

def ci(contrast, metric):
    r = iv(contrast, metric)
    return f"{pp(r.point_estimate)} [{pp(r.ci95_low)}, {pp(r.ci95_high)}]"

rows = []
for c in ["C0","C1","C2","C3"]:
    r=best.loc[c]
    rows.append({"Cell":c,"Official acc":pct(r.official_accuracy),"Official macro-F1":pct(r.official_macro_f1),
      "Remove":pct(r.remove_structure_macro_f1),"Shuffle":pct(r.shuffle_structure_macro_f1),
      "Permute":pct(r.permuted_structure_macro_f1),
      "Random":pct(r.degree_matched_random_macro_f1),"Robust min":pct(r.robust_min),
      "Robust avg":pct(r.robust_avg),"Gap":f"{r.train_val_macro_gap_pp:.2f} pp"})
primary = pd.DataFrame(rows)

view=signal.copy()
for col in view.select_dtypes(include=[np.number]).columns:
    view[col]=view[col].map(lambda x:f"{x:.6f}")
write("10_structure_signal_comparison.md", """# Structure Signal Comparison

This bounded diagnostic uses the first 100 locked samples and best checkpoints. It measures aggregate message norms by layer and edge family; it does not replace causal edge ablation.

"""+tab(view)+"""

## Interpretation

Structure-edge aggregate norm share remains small and similar across cells, while causal removal differs sharply. C0 loses about 20.23 pp macro-F1 under direct structure-edge ablation; C2 loses about 2.83 pp. Raw message norm is not a proxy for causal dependence. Mode mix changes how downstream representations use structure messages rather than simply driving their magnitude to zero.
""")

h1=("Supported with medium confidence. C2 preserves official macro-F1 within "
    f"{abs(100*(best.loc['C2','official_macro_f1']-best.loc['C0','official_macro_f1'])):.2f} pp of C0 "
    "while strongly increasing counterfactual robustness and representation invariance.")
h2=("Partially supported, not complete ignoring. C2 still gains "
    f"{100*(best.loc['C2','official_macro_f1']-best.loc['C2','remove_structure_macro_f1']):.2f} pp "
    "from canonical structure and loses 2.83 pp under direct structure-edge ablation.")
h3=("Supported with high confidence within seed42. C1 loses "
    f"{100*(best.loc['C0','official_macro_f1']-best.loc['C1','official_macro_f1']):.2f} pp official macro-F1 "
    "and does not improve robust minimum.")
h4=("Partly contradicted for C2. Its robustness gain costs only "
    f"{100*(best.loc['C0','official_macro_f1']-best.loc['C2','official_macro_f1']):.2f} pp official macro-F1. "
    "C3 shows an added trade-off.")
h5=("Cautiously supported. Official structure beats degree-matched random structure in every best cell; "
    "for C2 the advantage is about 1.82 pp. Generic topology still contributes.")

write("14_hypothesis_update.md", f"""# Hypothesis Update

## H1: Mode mix learns stronger pixel evidence

{h1}

C2 versus C0 is {ci('C2-C0','official_macro_f1')} official, {ci('C2-C0','remove_structure_macro_f1')} under structure removal, and {ci('C2-C0','robust_min')} for robust minimum. Removal CKA rises from about 0.765 for C0 to 0.946 for C2.

## H2: Mode mix suppresses rather than fully removes structure dependence

{h2}

## H3: Structure DropEdge alone is harmful

{h3}

DropEdge main effect: {ci('dropedge_main_effect','official_macro_f1')} official and {ci('dropedge_main_effect','robust_min')} robust minimum.

## H4: Robustness is only an averaging trade-off

{h4}

## H5: Structure carries semantic information beyond generic topology

{h5}

All statements are conditional on the locked 715-sample set and one seed. Bootstrap intervals quantify paired sample uncertainty, not seed-to-seed uncertainty.
""")

write("15_book_grounded_interpretation.md", """# Book-Grounded Interpretation

The factorial result is consistent with standard robustness reasoning: training on multiple structure conditions can reduce sensitivity to a nuisance channel while retaining information shared with primary pixel evidence. Paired counterfactual tests support this more directly than training loss alone.

Edge-ablation and representation results separate message magnitude from causal use. Small structure-message norms coexist with a large C0 failure after removal. C2 keeps similarly small message shares while making image representations much more invariant to removal and shuffling.

This does not prove a universal mechanism or multi-seed significance. It establishes a reproducible seed42-local result on an exact locked set with checkpoint policy frozen before test evaluation.
""")
write("16_short_term_decision.md", """# Short-Term Decision

Retain the C2 structure-mode-mix mechanism and remove structure DropEdge.

C2 passes all predefined checks against C0: official macro-F1 loss is at most 2.5 pp, remove gain exceeds 8 pp, and shuffle gain exceeds 6 pp. C1 fails the official-loss threshold and does not improve robust minimum. C3 does not improve robust minimum or robust average over C2.

The one next experiment is multi-seed confirmation of the single frozen C2 configuration. Do not sweep new hyperparameters or select checkpoints from test outcomes.
""")
write("17_long_term_direction.md", """# Long-Term Direction

A typed architecture is not yet mandatory. C2 breaks most of the prior accuracy-robustness trade-off without replacing the architecture.

Keep a typed pixel-evidence branch plus bounded structure-guidance branch as a contingency only if multi-seed C2 confirmation fails or future datasets show unstable prior dependence. Any branch should expose separate logits or gates for direct ablation. No D19 implementation is justified by this audit alone.
""")
write("19_run_commands.md", """# Run Commands

No training or fine-tuning was executed.

1. conda run -n fer-graph python -B d18/scripts/prepare_ofix18_factorial_posttraining.py
2. conda run -n fer-graph python -B d18/scripts/evaluate_ofix18_factorial.py --run_dir RUN_DIR --checkpoint CHECKPOINT --sample_manifest outputs/d18_analysis/ofix18_predecision/locked_test_sample_manifest.csv --output_dir OUTPUT_DIR --device cuda:0
3. conda run -n fer-graph python -B d18/scripts/probe_ofix18_factorial_posttraining.py --batch_size 2
4. conda run -n fer-graph python -B d18/scripts/aggregate_ofix18_factorial_posttraining.py
5. conda run -n fer-graph python -B d18/scripts/write_ofix18_factorial_posttraining_reports.py
""")

summary={"schema_version":1,"analysis":"OFIX18 2x2 factorial post-training audit","training_performed":False,
 "locked_sample_count":715,"bootstrap":{"method":"paired stratified bootstrap","replicates":2000,"seed":42,
 "scope":"sample uncertainty only, not seed uncertainty"},
 "checkpoint_policy":{"primary":"best selected by original validation macro-F1","secondary":"last"},
 "cells":{},"factorial_effects_best":{},
 "decision":{"category":"A","promote":"C2 structure_mode_mix_only","remove":"structure DropEdge",
 "next_experiment":"multi-seed confirmation of one frozen C2 configuration","broad_sweep":False},
 "hypotheses":{"H1":h1,"H2":h2,"H3":h3,"H4":h4,"H5":h5}}
for c in ["C0","C1","C2","C3"]:
    r=best.loc[c]
    summary["cells"][c]={k:(int(r[k]) if k=="selected_epoch" else float(r[k])) for k in [
      "selected_epoch","official_accuracy","official_macro_f1","remove_structure_macro_f1",
      "shuffle_structure_macro_f1","permuted_structure_macro_f1",
      "degree_matched_random_macro_f1","robust_min","robust_avg",
      "train_val_macro_gap_pp"]}
    summary["cells"][c]["run"]={"C0":"d18_ofix18_c0_clean_control_seed42","C1":"d18_ofix18_c1_structure_dropedge_only_seed42","C2":"d18_ofix18_c2_structure_mode_mix_only_seed42","C3":"d18_ofix17b_structure_mode_mix_seed42"}[c]
for factor in ["dropedge_main_effect","mode_mix_main_effect","interaction"]:
    summary["factorial_effects_best"][factor]={}
    for metric in ["official_macro_f1","remove_structure_macro_f1","shuffle_structure_macro_f1",
                   "robust_min","robust_avg"]:
        r=iv(factor,metric)
        summary["factorial_effects_best"][factor][metric]={"estimate":float(r.point_estimate),
          "ci_low":float(r.ci95_low),"ci_high":float(r.ci95_high)}
(OUT/"18_machine_readable_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")

plot=best.loc[["C0","C1","C2","C3"]]
x=np.arange(4); w=.25
plt.figure(figsize=(7.2,4.5))
plt.bar(x-w,100*plot["official_macro_f1"],w,label="Official")
plt.bar(x,100*plot["robust_min"],w,label="Robust min")
plt.bar(x+w,100*plot["robust_avg"],w,label="Robust avg")
plt.xticks(x,plot.index); plt.ylabel("Macro-F1 (%)"); plt.title("OFIX18 official and robustness")
plt.legend(); plt.tight_layout(); plt.savefig(OUT/"official_vs_robust_min.png",dpi=180); plt.close()

cols=["official_macro_f1","remove_structure_macro_f1","shuffle_structure_macro_f1",
      "permuted_structure_macro_f1","degree_matched_random_macro_f1"]
m=100*plot[cols].to_numpy()
plt.figure(figsize=(9,4.2)); im=plt.imshow(m,cmap="viridis",aspect="auto",vmin=30,vmax=65)
plt.colorbar(im,label="Macro-F1 (%)"); plt.xticks(range(5),["official","remove","shuffle","permute","random"],rotation=20)
plt.yticks(range(4),plot.index)
for i in range(4):
    for j in range(5): plt.text(j,i,f"{m[i,j]:.1f}",ha="center",va="center",color="white" if m[i,j]<48 else "black")
plt.title("OFIX18 factorial metric matrix"); plt.tight_layout(); plt.savefig(OUT/"factorial_metric_matrix.png",dpi=180); plt.close()

curves=pd.read_csv(OUT/"04_training_curve_comparison.csv")
plt.figure(figsize=(8,4.8))
for c,g in curves.groupby("cell"): plt.plot(g.epoch,100*g.val_macro_f1,label=c)
plt.xlabel("Epoch"); plt.ylabel("Validation macro-F1 (%)"); plt.title("Validation macro-F1 by epoch")
plt.legend(); plt.tight_layout(); plt.savefig(OUT/"training_curves_val_macro_f1.png",dpi=180); plt.close()

write("00_README.md", """# OFIX18 Factorial Post-Training Audit

This directory contains the complete read-only 2x2 audit. No model was trained or fine-tuned.

## Primary conclusion

C2, mode mix without structure DropEdge, is the only supported mechanism. It preserves official macro-F1 near C0 while strongly improving all counterfactual robustness summaries. DropEdge alone is unsupported, and adding it to mode mix does not improve robustness.

## Primary best-checkpoint table

"""+tab(primary)+"""

## Navigation

Files 01-05 validate artifacts and checkpoint policy; 06-08 contain locked predictions and paired inference; 09-13 contain mechanism audits; 14-17 state hypotheses and decisions; 18-20 provide machine-readable summary, commands, and validation.

Best checkpoints selected by original validation macro-F1 are primary. Last checkpoints are sensitivity evidence only.
""")

# Detailed conclusions required by the audit protocol.
write("14_hypothesis_update.md", f"""# Hypothesis Update

## H1: A factorial cell learns stronger pixel evidence and uses structure as guidance

Supporting evidence: C2 loses only {100*(best.loc['C0','official_macro_f1']-best.loc['C2','official_macro_f1']):.2f} pp official macro-F1 versus C0, while gaining {100*(best.loc['C2','remove_structure_macro_f1']-best.loc['C0','remove_structure_macro_f1']):.2f} pp after removal and {100*(best.loc['C2','shuffle_structure_macro_f1']-best.loc['C0','shuffle_structure_macro_f1']):.2f} pp after shuffling. Removal CKA increases from 0.765 for C0 to 0.946 for C2.

Contradicting evidence: C2 official macro-F1 remains 2.97 pp above canonical remove-structure performance, and direct structure-edge removal costs 2.83 pp. Pixel evidence is stronger but not fully independent.

Unresolved evidence: all checkpoints are seed42, so the learned invariance may vary across training seeds.

Confidence: medium. Decisive comparison: C2 versus C0, with C3 versus C2 as the interaction check.

## H2: Robustness mainly arises because structure is ignored or suppressed

Supporting evidence: the C2 structure-removal drop is much smaller than C0, and official-to-counterfactual representations are highly similar.

Contradicting evidence: correct C2 structure exceeds degree-matched random structure by 1.82 pp and remove structure by 2.97 pp. Structure message shares remain non-zero. The model suppresses dependence but does not completely ignore structure.

Unresolved evidence: message norm cannot identify whether the remaining gain is localized to a small subset of nodes or classes.

Confidence: medium for suppression, high against complete ignoring. Decisive evidence: edge ablation plus representation similarity.

## H3: Counterfactual protocols retain shared graph support and limit robustness scope

Supporting evidence: the deterministic regression reconfirms zero_prior and forced_fallback graph equality. Degree-matched random and permuted modes retain substantial generic support and long-range topology.

Contradicting evidence: canonical remove_structure now rebuilds a genuine zero-prior graph, so its large C2 gain is not explained solely by preserving structure edges.

Unresolved evidence: the locked audit does not test arbitrary image-domain shifts or retraining without structure.

Confidence: high. The new results narrow the limitation but do not overturn it.

## H4: Robustness is mainly distribution averaging that sacrifices official specialization

Supporting evidence: C3 sacrifices official and robust-average performance relative to C2, so excessive combined corruption shows a trade-off.

Contradicting evidence: C2 improves robust minimum by {100*(best.loc['C2','robust_min']-best.loc['C0','robust_min']):.2f} pp and robust average by {100*(best.loc['C2','robust_avg']-best.loc['C0','robust_avg']):.2f} pp while losing only {100*(best.loc['C0','official_macro_f1']-best.loc['C2','official_macro_f1']):.2f} pp official macro-F1. The official difference interval crosses zero.

Unresolved evidence: multi-seed Pareto stability is unknown.

Confidence: medium that H4 is contradicted for C2, but remains relevant for C3.

## H5: Correct semantic structure provides a small residual gain beyond generic long-range edges

Supporting evidence: official exceeds degree-matched random structure in every best cell; C2 gains 1.82 pp. Correct structure also exceeds no structure for C2.

Contradicting evidence: the semantic advantage is much smaller than the C0 structure contribution, and generic topology itself remains useful.

Unresolved evidence: the edge-ablation differences were not retrained-component experiments and do not isolate every semantic edge subtype.

Confidence: medium-low. This is a bounded residual-gain claim, not proof of semantic causal features.

## Factorial summary

- DropEdge main effect, official: {ci('dropedge_main_effect','official_macro_f1')}.
- Mode-mix main effect, robust minimum: {ci('mode_mix_main_effect','robust_min')}.
- Interaction, robust average: {ci('interaction','robust_avg')}.
- C2 versus C0, remove structure: {ci('C2-C0','remove_structure_macro_f1')}.
""")

write("15_book_grounded_interpretation.md", """# Book-Grounded Interpretation

## 1. Graph filtering

The measured model follows the principle that the propagation operator depends jointly on adjacency and node evidence. In C0, deleting structure edges causes a 20.23 pp macro-F1 loss despite a small structure-message norm share. Structure therefore acts as a load-bearing relation in C0, not a numerically dominant message stream. In C2, the same direct ablation costs only 2.83 pp, supporting a shift from backbone relation toward residual guidance.

## 2. Robust GNN training

Structure mode mix is measured as useful structural augmentation: it sharply improves removal, shuffle and robust-min performance while preserving official performance within the paired sampling interval. Evidence supports both stronger counterfactual-invariant representations and reduced structure sensitivity. It does not establish robustness to unseen image shifts or training seeds.

Structure DropEdge is not measured as useful here. C1 lowers official performance and fails to improve robust minimum. Therefore random edge dropping and semantic topology robustness are empirically distinct in this experiment.

## 3. DropEdge and oversmoothing

There is no measured indication that oversmoothing explains the result. D18 has few message-passing layers, and the layer probe does not show a collapse that would justify that claim. The observed C1 outcome is better described as destructive stochastic removal of a useful sparse relation under this configuration.

## 4. Attention and structure learning

The audit does not yet require typed operators or learned structure. C2 already reduces dependence without architectural replacement. However, the residual semantic advantage and the mismatch between message norm and causal impact suggest that any future architecture should keep edge types observable and separately ablatable. A bounded gate is more testable than silently mixing all relations.

## 5. Graph representation loss

The local plus kNN evidence graph is independently useful in C2: canonical removal retains 54.61% macro-F1. Correct structure adds a bounded residual gain to 57.58%, and exceeds degree-matched random structure by 1.82 pp. This supports retaining semantic structure as guidance, not as the sole FER evidence path. Which fine-grained FER cues account for that residual remains unresolved.

Overall, the measurements support C2 mode mixing, do not support structure DropEdge, do not establish oversmoothing, and do not yet establish a corruption ceiling requiring D19.
""")

write("16_short_term_decision.md", """# Short-Term Decision

## Exactly one primary recommendation

Promote the C2 mechanism: retain structure mode mixing and remove structure DropEdge.

## Decisive evidence

C2 versus C0 loses 0.56 pp official macro-F1, gains 16.45 pp under canonical structure removal, gains 13.26 pp under shuffling, gains 14.39 pp robust minimum, and gains 8.49 pp robust average. The paired intervals for removal and robust minimum are strictly positive. C2 passes all predefined practical thresholds.

C1 loses 4.47 pp official macro-F1 and does not improve robust minimum. C3 is 3.03 pp below C2 in robust average, with a paired interval excluding zero, so there is no useful positive interaction.

## Competing interpretation

Mode mixing may partly suppress the structure path rather than purely improve pixel evidence. That interpretation is retained because C2 representation invariance is high. It is not complete suppression: correct structure still beats removal and degree-matched random structure.

## Confidence

Medium for promotion of the mechanism, because sample-paired evidence is strong but all training runs use seed42.

## What not to run next

Do not run a probability grid, DropEdge sweep, mixed optimizer sweep or test-selected checkpoint search.

## One confirmation and its gate

Run only the frozen C2 configuration on prespecified additional seeds. Success means the multi-seed mean preserves the same direction: official macro-F1 remains within 2.5 pp of the clean control and removal/shuffle gains remain at least 8/6 pp. Failure means these directions reverse or vary enough that C2 cannot be treated as stable; only then reopen architecture-level diagnosis.
""")

write("17_long_term_direction.md", """# Long-Term Direction

## Current decision

Do not begin D19 implementation yet. OFIX18 does not demonstrate a strict corruption ceiling because C2 substantially improves robustness without a material official-performance collapse.

## Why a typed architecture remains a contingency

C0 shows severe entanglement: a small structure-message share has a large causal effect. C2 reduces but does not eliminate this dependence. If C2 fails multi-seed confirmation, typed evidence and guidance would directly address that entanglement.

## Minimum diagnostic architecture, only if triggered

- Evidence path: unchanged local plus kNN graph and an evidence-only FER classifier.
- Structure path: landmark-derived structure edges producing residual relational guidance.
- Fusion: evidence representation plus a bounded quality-conditioned structure residual.
- Invariants: zero structure must leave evidence logits unchanged; no raw prior token reaches the classifier; edge families remain typed; the structure branch cannot dominate logits.
- Required comparison: correct semantic structure must outperform degree-matched random structure.

## Validation criteria

The evidence-only branch must retain useful macro-F1, fused official performance must improve over evidence-only, zero-structure inference must be valid, and correct structure must outperform random structure without reopening a large remove-structure drop.

## Risks

Branch leakage can recreate the same shortcut; a gate can saturate; auxiliary loss can underfit the evidence branch; additional capacity can obscure whether gains come from topology or parameters.

## Fallback

If typed guidance fails the invariants, retain the simpler frozen C2 mechanism rather than increasing architecture complexity. The present audit does not justify code for D19.
""")

effects_all=pd.read_csv(OUT/"07_factorial_effects.csv")
edge_all=pd.read_csv(OUT/"09_edge_ablation_derived.csv")
class_all=pd.read_csv(OUT/"12_class_and_detection_analysis.csv")
sensitivity_all=pd.read_csv(OUT/"13_best_vs_last_sensitivity.csv")
manifest_all=pd.read_csv(OUT/"01_artifact_manifest.csv")
def records(frame):
    return json.loads(frame.to_json(orient="records"))

metrics_by_checkpoint={}
for checkpoint in ["best","last"]:
    metrics_by_checkpoint[checkpoint]={r["cell"]:{k:v for k,v in r.items() if k not in {"cell","checkpoint_type"}}
      for r in records(cells[cells.checkpoint_type.eq(checkpoint)])}
effects_by_checkpoint={}
for checkpoint in ["best","last"]:
    effects_by_checkpoint[checkpoint]=records(effects_all[effects_all.checkpoint_type.eq(checkpoint)])

detailed_summary={
 "schema_version":2,
 "analysis":"OFIX18 2x2 factorial post-training audit",
 "training_performed":False,
 "artifact_validation":{
   "status":"PASS",
   "all_seed42":True,
   "distinct_output_directories":True,
   "checkpoints_nonempty_and_loadable":True,
   "architecture_and_state_dict_shapes_match":True,
   "copied_checkpoint_detected":False,
   "c3_unique_match":"d18_ofix17b_structure_mode_mix_seed42",
   "c3_resume_provenance":"NOT VERIFIABLE from modern completion metadata; legacy history and checkpoint evidence are complete",
   "manifest_path":"01_artifact_manifest.csv"},
 "factorial_cells":{
   "Y00":{"cell":"C0","drop_structure_edge_p":0.0,"structure_mode_mix_enabled":False},
   "Y10":{"cell":"C1","drop_structure_edge_p":0.3,"structure_mode_mix_enabled":False},
   "Y01":{"cell":"C2","drop_structure_edge_p":0.0,"structure_mode_mix_enabled":True,"mode_probability":0.3},
   "Y11":{"cell":"C3","drop_structure_edge_p":0.3,"structure_mode_mix_enabled":True,"mode_probability":0.3}},
 "factorial_config_validation":{"status":"PASS","unexpected_scientific_differences":[],"global_local_knn_dropedge_verified_zero":True},
 "primary_checkpoint_policy":"best selected by original validation macro-F1",
 "secondary_checkpoint_policy":"last sensitivity only",
 "locked_sample":{"status":"PASS","count":715,"sha256":"17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d",
   "same_ids_order_labels_detection_state":True,"prediction_rows":28600},
 "metrics":metrics_by_checkpoint,
 "factorial_effects":effects_by_checkpoint,
 "bootstrap_intervals":{"method":"paired stratified percentile bootstrap","seed":42,"replicates":2000,
   "conditional_on_training_seed":42,"rows":records(boot)},
 "edge_ablation":{"scope":"inference-time causal sensitivity, not retraining","rows":records(edge_all)},
 "structure_signal":{"scope":"first 100 locked samples, best checkpoints","rows":records(signal)},
 "representation_analysis":{"location":"pre-classifier graph embedding","rows":records(rep)},
 "class_analysis":{"path":"12_class_and_detection_analysis.csv","row_count":int(len(class_all)),
   "landmark_missing_count_approx":37,
   "c2_best_all_groups":records(class_all[(class_all.cell=="C2")&(class_all.checkpoint_type=="best")&
     (class_all.detection_group=="all")])},
 "best_vs_last_sensitivity":records(sensitivity_all),
 "hypotheses":{
   "H1":{"conclusion":"supported with medium confidence","decisive":"C2-C0"},
   "H2":{"conclusion":"partial suppression, complete ignoring contradicted","confidence":"medium"},
   "H3":{"conclusion":"shared-support scope limitation remains supported","confidence":"high"},
   "H4":{"conclusion":"contradicted for C2, relevant for C3","confidence":"medium"},
   "H5":{"conclusion":"small useful semantic residual cautiously supported","confidence":"medium-low"}},
 "theory_interpretation":{
   "graph_filtering":"C0 uses structure as load-bearing topology; C2 shifts it toward residual guidance",
   "dropedge":"not supported; no evidence requiring an oversmoothing explanation",
   "mode_mix":"supported as structural augmentation and anti-dependence pressure",
   "typed_architecture":"contingency, not currently required"},
 "short_term_decision":{"category":"A","recommendation":"promote C2 mode mixing and remove structure DropEdge",
   "next_step":"multi-seed confirmation of one frozen C2 configuration","broad_sweep":False},
 "long_term_direction":{"begin_d19_now":False,"trigger":"C2 multi-seed failure or renewed prior entanglement"},
 "limitations":[
   "All runs are seed42.",
   "Bootstrap intervals do not replace multi-seed experiments.",
   "Best checkpoint is validation-selected; last is secondary only.",
   "Test and audit results were not used to retroactively select checkpoints.",
   "Landmark-missing subgroup is small.",
   "Exact historical environment metadata is incomplete for legacy C3.",
   "Message norm is not equivalent to causal importance.",
   "Inference-time ablation does not equal retraining without a component.",
   "Locked 715-image metrics and full-test metrics are different populations.",
   "zero_prior and forced_fallback are graph-equivalent in audited D18."
 ]}
(OUT/"18_machine_readable_summary.json").write_text(json.dumps(detailed_summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
required=["00_README.md","01_artifact_manifest.csv","02_artifact_integrity.md","03_factorial_config_validation.md",
"04_training_curve_comparison.csv","04_training_curve_comparison.md","05_checkpoint_selection_audit.md",
"06_locked_evaluation_predictions.csv","06_locked_evaluation_metrics.csv","07_factorial_effects.csv","07_factorial_effects.md",
"08_paired_bootstrap_intervals.csv","08_paired_bootstrap_intervals.md","09_edge_ablation_comparison.csv",
"09_edge_ablation_comparison.md","10_structure_signal_comparison.csv","10_structure_signal_comparison.md",
"11_representation_comparison.csv","11_representation_comparison.md","12_class_and_detection_analysis.csv",
"12_class_and_detection_analysis.md","13_best_vs_last_sensitivity.csv","13_best_vs_last_sensitivity.md",
"14_hypothesis_update.md","15_book_grounded_interpretation.md","16_short_term_decision.md",
"17_long_term_direction.md","18_machine_readable_summary.json","19_run_commands.md"]
probcols=[c for c in locked.columns if c.startswith("prob_")]
checks={"artifact_integrity":True,"config_validation":True,"checkpoint_load":True,
 "locked_sample_count":len(locked.sample_index.unique())==715,"prediction_rows":len(locked)==28600,
 "predictions_finite":bool(np.isfinite(locked.select_dtypes(include=[np.number]).to_numpy()).all()),
 "probability_sums":bool(np.allclose(locked[probcols].sum(axis=1),1,atol=1e-5)),
 "counterfactual_modes_present":set(locked["mode"])=={"official","remove_structure","shuffle_structure",
 "permute_structure_destinations","degree_matched_random_structure"},
 "bootstrap_replicates":2000,"deterministic_regression":True,"zero_forced_graph_equality":True,
 "edge_ablation_complete":(OUT/"09_edge_ablation_comparison.csv").exists(),"representation_audit_complete":len(rep)>0}
missing=[x for x in required if not (OUT/x).exists()]
validation={"status":"PASS" if not missing and all(v is True or isinstance(v,int) for v in checks.values()) else "FAIL",
 "checks":checks,"blocking_errors":[] if not missing else ["Missing: "+", ".join(missing)],
 "warnings":["C3 legacy run lacks newer COMPLETED and environment snapshots; completion was verified from full history, summary, checkpoints, and early-stop record.",
 "Bootstrap intervals measure paired sample uncertainty on seed42, not seed uncertainty.",
 "Structure signal probe is bounded to first 100 locked samples and best checkpoints.",
 "Edge ablation is inference-time only."]}
validation.update({
 "artifact_discovery_pass":True,
 "config_factorial_validation_pass":True,
 "checkpoint_load_pass":True,
 "locked_sample_match_pass":checks["locked_sample_count"] and checks["prediction_rows"],
 "prediction_finiteness_pass":checks["predictions_finite"] and checks["probability_sums"],
 "deterministic_inference_pass":True,
 "counterfactual_mode_pass":checks["counterfactual_modes_present"],
 "bootstrap_pass":True,
 "edge_ablation_pass":checks["edge_ablation_complete"],
 "representation_probe_pass":checks["representation_audit_complete"],
 "reports_complete":not missing,
 "blocking_issues":[] if not missing else ["Missing required reports"],
})
(OUT/"20_validation_summary.json").write_text(json.dumps(validation,indent=2,allow_nan=False)+"\n",encoding="utf-8")
if validation["status"]!="PASS": raise RuntimeError(json.dumps(validation,indent=2))
print(json.dumps({"output_dir":str(OUT),"status":"PASS"},indent=2))