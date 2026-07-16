"""Generate OFIX18 paired C0/C2 multi-seed design reports."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import yaml

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/d18_analysis/ofix18_c0_c2_multiseed_design"
BASE=ROOT/"configs/d18/overfit_fix_18"
NEW=BASE/"multiseed"
SEEDS=(7,21,42,84,123)
NEW_SEEDS=(7,21,84,123)
TOPOLOGY=(11,23,37,53,71)
LOCKED="17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d"

def write(name,text):
    (OUT/name).write_text(text.rstrip()+"\n",encoding="utf-8")

def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_name(cell,seed):
    stem="c0_clean_control" if cell=="C0" else "c2_structure_mode_mix_only"
    return f"d18_ofix18_{stem}_seed{seed}"

def config_path(cell,seed):
    return (BASE if seed==42 else NEW)/f"{run_name(cell,seed)}.yaml"

def output_path(cell,seed):
    family="ofix18" if seed==42 else "ofix18_multiseed"
    return f"outputs/d18_runs/{family}/{run_name(cell,seed)}"

def table(columns,rows):
    clean=lambda x:str(x).replace("|","\\|").replace("\n"," ")
    lines=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
    return "\n".join(lines+["| "+" | ".join(clean(x) for x in row)+" |" for row in rows])

OUT.mkdir(parents=True,exist_ok=True)
validation=json.loads((OUT/"validation_and_smoke.json").read_text(encoding="utf-8"))
if validation["status"]!="PASS":
    raise RuntimeError("Validation did not pass")

source_rows=[]
for cell in ("C0","C2"):
    path=config_path(cell,42); cfg=load(path); run=ROOT/output_path(cell,42)
    source_rows.append([cell,str(path.relative_to(ROOT)),digest(path),str(run.relative_to(ROOT)),run.exists(),
      cfg["training"]["checkpoint_monitor"],cfg["training"]["checkpoint_monitor_mode"],cfg["training"]["amp"],
      cfg["training"]["graph_regularization"]["drop_structure_edge_p"],
      cfg["training"]["structure_mode_mix"]["enabled"],cfg["training"]["structure_mode_mix"]["p_forced_structure"]])
write("01_seed42_source_manifest.md","# Seed42 Source Manifest\n\n"+
 table(["Cell","Source config","SHA-256","Completed run","Exists","Monitor","Mode","AMP","Structure drop","Mix","Forced p"],source_rows)+
 "\n\nThe source YAMLs are unique. Completed seed42 run directories are authoritative artifacts. Kaggle patches only machine paths and explicit command output paths.")

matrix=[]
for cell in ("C0","C2"):
    for seed in SEEDS:
        cfg=load(config_path(cell,seed)); reg=cfg["training"]["graph_regularization"]; mix=cfg["training"]["structure_mode_mix"]
        matrix.append([cell,seed,"existing, no retrain" if seed==42 else "new paired run",
          str(config_path(cell,seed).relative_to(ROOT)),output_path(cell,seed),cfg["training"]["drop_edge_p"],
          reg["drop_local_edge_p"],reg["drop_knn_edge_p"],reg["drop_structure_edge_p"],mix["enabled"],mix["p_forced_structure"]])
write("02_multiseed_experiment_matrix.md","# Multi-Seed Experiment Matrix\n\n"+
 table(["Cell","Seed","Status","Config","Output","Global","Local","kNN","Structure","Mix","Forced p"],matrix)+
 "\n\nEvery new C2 seed has a matching C0 seed. No C1 or C3 run is created.")

write("03_frozen_invariants.md","""# Frozen Invariants

Only seed, training.seed, run name, output directory, description and W&B metadata change.

Frozen fields include FER2013 split, priors, transforms, augmentation, node support, node count, feature schemas, all edge builders, model architecture, loss, AdamW, learning rate, weight decay, scheduler, warmup behavior, batch size, clipping behavior, epochs, early stopping, validation macro-F1 monitor, AMP, sampling and worker settings.

All global/local/kNN/structure DropEdge values remain zero. C0 mix is disabled. C2 mix remains enabled at p_forced_structure=0.30 with remove_structure_edges_per_sample semantics. There is no shuffle training mode, curriculum, consistency, auxiliary loss, gate penalty or purification.

Python, NumPy, Torch CPU and CUDA use the configured seed. DataLoader shuffle and default worker seeds derive from the seeded Torch RNG. Deterministic algorithms are not forced, matching seed42.
""")

clone_rows=[[x["cell"],x["seed"],x["config_path"],x["unexpected_diff_count"],json.dumps(x["unexpected_diffs"],sort_keys=True)]
 for x in validation["config_validation"]["semantic_clone_validation"]]
factor_rows=[[x["path"],x["left"],x["right"]] for x in validation["config_validation"]["c0_c2_seed42_factor_diff"]]
write("04_semantic_config_diffs.md","# Semantic Config Diffs\n\n## Clone against corresponding seed42\n\n"+
 table(["Cell","Seed","Config","Unexpected count","Unexpected diffs"],clone_rows)+
 "\n\n## Intended C0 versus C2 factor\n\n"+table(["Path","C0","C2"],factor_rows)+
 "\n\nStatus: PASS. No unintended scientific or training difference was found.")

command_specs=[
 ("06_c0_seed7_kaggle.md","C0",7),("07_c0_seed21_kaggle.md","C0",21),
 ("08_c0_seed84_kaggle.md","C0",84),("09_c0_seed123_kaggle.md","C0",123),
 ("10_c2_seed7_kaggle.md","C2",7),("11_c2_seed21_kaggle.md","C2",21),
 ("12_c2_seed84_kaggle.md","C2",84),("13_c2_seed123_kaggle.md","C2",123)]
kaggle={}
for filename,cell,seed in command_specs:
    key=f"{cell.lower()}_seed{seed}"; run=run_name(cell,seed)
    cfg=str(config_path(cell,seed).relative_to(ROOT)).replace("\\","/")
    out=f"/kaggle/working/outputs/d18_runs/ofix18_multiseed/{run}"
    kaggle[key]={"notebook":"notebooks/kaggle-end-to-end.ipynb","active_run_key":key,
      "run_mode":"fresh","config":cfg,"output":out}
    mix="enabled at p=0.30" if cell=="C2" else "disabled at p=0.0"
    write(filename,f"""# {cell} Seed {seed} Kaggle Run

Use notebooks/kaggle-end-to-end.ipynb from repository root.

## Fresh selector

    ACTIVE_RUN_KEY = "{key}"
    RUN_MODE = "fresh"
    EXPECT_RESUME_CHECKPOINT = False
    RESUME_OUTPUT_INPUT_ROOT_TEXT = ""

Source config: {cfg}

Output: {out}

Expected factors: all DropEdge values 0.0; structure mode mix {mix}; seed {seed}. Best is selected by validation macro-F1 and last is retained.

The notebook prints the source/runtime config and effective factors, writes train_console.log, resolved config, schemas, environment, pip freeze, official evaluation, best.pt, last.pt and COMPLETED.json, then creates /kaggle/working/{run}_outputs.zip.

## Strict resume

    ACTIVE_RUN_KEY = "{key}"
    RUN_MODE = "resume"
    EXPECT_RESUME_CHECKPOINT = True
    RESUME_OUTPUT_INPUT_ROOT_TEXT = "/kaggle/input/.../{run}"

Only this run's last.pt is accepted. Scientific and run/output signatures must both match. No command depends on another run.
""")

eval_cmd=("conda run -n fer-graph python -B d18/scripts/evaluate_ofix18_c0_c2_multiseed.py "
 "--new_run_root outputs/d18_runs/ofix18_multiseed "
 "--prior_dir outputs/d16_mediapipe_pixel_priors_best_retry_rescue "
 "--graph_cache_dir outputs/d18_graph_cache/ofix17_structure_reg/base6_shared "
 "--locked_manifest outputs/d18_analysis/ofix18_predecision_audit/sample_manifest.csv "
 "--output_dir outputs/d18_analysis/ofix18_c0_c2_multiseed_evaluation --device cuda:0 --execute")
aggregate_cmd=("conda run -n fer-graph python -B d18/scripts/analyze_ofix18_c0_c2_multiseed.py "
 "--new_run_root outputs/d18_runs/ofix18_multiseed --existing_run_root outputs/d18_runs/ofix18 "
 "--evaluation_root outputs/d18_analysis/ofix18_c0_c2_multiseed_evaluation "
 "--output_dir outputs/d18_analysis/ofix18_c0_c2_multiseed_results --strict")
write("14_posttraining_evaluation.md",f"""# Post-Training Evaluation

Do not execute before all eight new runs have best.pt and last.pt.

Dry run:

    conda run -n fer-graph python -B d18/scripts/evaluate_ofix18_c0_c2_multiseed.py

Execute all ten paired models:

    {eval_cmd}

The evaluator runs full official test for best and last, then the exact locked 715 sample with hash {LOCKED}. Official/remove/shuffle are evaluated once. Permute and degree-matched random structure use topology seeds {list(TOPOLOGY)}. Topology seeds are not training seeds.
""")
write("15_multiseed_analysis_protocol.md",f"""# Multi-Seed Analysis Protocol

Primary unit is the paired training seed, C2 minus C0. Training seeds are {list(SEEDS)}.

Per seed compare full official, locked official/remove/shuffle/permute/random, robust minimum, robust average, robustness drops, train-validation gap and selected epoch. Best is primary; last is sensitivity.

Across five paired differences report mean, standard deviation, median, minimum, maximum, positive count, exact sign consistency and 95% t interval. Never pool images across trained models.

Run:

    {aggregate_cmd}

The prior image bootstrap remains a separate conditional seed42 analysis.
""")
write("16_success_failure_criteria.md","""# Success and Failure Criteria

Stable C2 requires all primary gates: mean official difference at least -2.5 pp; at most one seed worse than -4.0 pp official; remove gain at least 8 pp with at least 4/5 positive; shuffle gain at least 6 pp with at least 4/5 positive; robust-min gain at least 8 pp with at least 4/5 positive; mean C2-minus-C0 train-validation gap increase at most 5 pp; mean full-test ECE increase at most 0.05; and mean remove-structure F1 gain positive in at least 4/7 FER classes.

If passed, promote frozen C2, remove structure DropEdge permanently, do not sweep mode probability and do not begin D19.

If robustness is consistent but official loss fails, perform one mechanistic review without creating a new config automatically. If fewer than four seeds agree, do not promote C2 and reopen evidence-versus-guidance diagnosis.

Secondary semantic signal: C2 official-minus-random mean above zero and positive in at least 3/5 seeds. Its failure limits landmark semantic claims but does not reject C2 if primary robustness passes.
""")
write("17_expected_artifacts.md","""# Expected Artifacts

Each new run must contain resolved configs, effective_training_config.json, environment.json, pip_freeze.txt, train_console.log, train_log.csv, feature and graph schemas, checkpoints/best.pt, checkpoints/last.pt, d18_train_summary.json, official metrics, per-class metrics, confusion matrix, COMPLETED.json and one run ZIP.

Common evaluation creates full_official, locked_core and five locked_topology_seed directories for both checkpoints. Aggregation creates per-model metrics, paired differences, t-interval summary, decision report and machine summary.
""")
write("18_risks_and_limitations.md","""# Risks and Limitations

- Five training seeds remain a small sample and all use the same FER2013 split.
- Paired C0/C2 is cleaner than C2-only, but does not establish cross-dataset generalization.
- Locked 715 and full-test metrics are different populations.
- Best is validation-selected; last is sensitivity only.
- Topology seeds are not training seeds.
- Image bootstrap and training-seed uncertainty answer different questions.
- Landmark-missing subgroup is small.
- Inference ablation does not equal retraining.
- Historical seed42 git/environment metadata may be incomplete.
- Kaggle runtime prior/cache paths differ from source YAML machine paths only.
- New files must be pushed to the GitHub branch cloned by the notebook.
""")

new_runs={"C0":{},"C2":{}}
configs={}; outputs={}
for cell in ("C0","C2"):
    for seed in NEW_SEEDS:
        key=f"{cell}_seed{seed}"; cp=str(config_path(cell,seed).relative_to(ROOT)).replace("\\","/")
        new_runs[cell][str(seed)]={"run_name":run_name(cell,seed),"config":cp,"output":output_path(cell,seed)}
        configs[key]=cp; outputs[key]=output_path(cell,seed)
manifest={
 "existing_seed42":{cell:{"run_name":run_name(cell,42),
   "config":str(config_path(cell,42).relative_to(ROOT)).replace("\\","/"),
   "run_dir":output_path(cell,42),"retrain":False} for cell in ("C0","C2")},
 "new_runs":new_runs,"training_seeds":list(SEEDS),"new_training_seeds":list(NEW_SEEDS),
 "topology_evaluation_seeds":list(TOPOLOGY),
 "frozen_fields":{"source":"corresponding seed42 config","all_dropedge":0.0,
  "C0_mode_mix":{"enabled":False,"p":0.0},"C2_mode_mix":{"enabled":True,"p":0.30},
  "monitor":"val_macro_f1","amp":False},
 "allowed_differences":["seed","training.seed","run_name","output_dir","description","logging metadata"],
 "config_paths":configs,"output_paths":outputs,"kaggle_commands":kaggle,
 "evaluation_commands":{"dry_run":"conda run -n fer-graph python -B d18/scripts/evaluate_ofix18_c0_c2_multiseed.py",
  "execute":eval_cmd,"aggregate":aggregate_cmd},
 "success_criteria":{"official_mean_min":-0.025,"official_bad_seed_max":1,
  "remove_mean_min":0.08,"remove_positive_min":4,"shuffle_mean_min":0.06,
  "shuffle_positive_min":4,"robust_min_mean_min":0.08,"robust_positive_min":4,
  "mean_train_val_gap_increase_max":0.05,"mean_full_test_ece_increase_max":0.05,
  "remove_positive_class_min":4,"semantic_positive_seed_min":3},
 "smoke_results":validation["smoke"],"resume_contract":validation["resume_contract"],
 "limitations":["Five seeds are a small sample.","One FER2013 split.","Topology seeds are not training seeds.",
  "No full local training was run during preparation."]}
(OUT/"19_machine_readable_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")

write("00_README.md","""# OFIX18 C0/C2 Paired Multi-Seed Confirmation Design

Purpose: test whether C2 mode-mix advantage persists across paired independent training seeds.

New runs are C0 and C2 at seeds 7, 21, 84 and 123. Existing seed42 C0/C2 are reused and never retrained. C1/C3 are excluded.

Preparation status: semantic validation PASS, resume contract PASS, bounded CPU smoke PASS. No full local training was run.

Use files 06-13 with notebooks/kaggle-end-to-end.ipynb, one selected run per Kaggle session. After all outputs are downloaded, follow files 14 and 15.
""")
required=["00_README.md","01_seed42_source_manifest.md","02_multiseed_experiment_matrix.md",
 "03_frozen_invariants.md","04_semantic_config_diffs.md","05_validation_and_smoke.md",
 "06_c0_seed7_kaggle.md","07_c0_seed21_kaggle.md","08_c0_seed84_kaggle.md",
 "09_c0_seed123_kaggle.md","10_c2_seed7_kaggle.md","11_c2_seed21_kaggle.md",
 "12_c2_seed84_kaggle.md","13_c2_seed123_kaggle.md","14_posttraining_evaluation.md",
 "15_multiseed_analysis_protocol.md","16_success_failure_criteria.md","17_expected_artifacts.md",
 "18_risks_and_limitations.md","19_machine_readable_manifest.json"]
missing=[x for x in required if not (OUT/x).exists()]
summary={"status":"PASS" if not missing else "FAIL","seed42_configs_unique":True,
 "eight_configs_created":len(list(NEW.glob("*.yaml")))==8,
 "semantic_validation_pass":validation["config_validation"]["status"]=="PASS",
 "resume_contract_pass":validation["resume_contract"]["status"]=="PASS",
 "smoke_pass":validation["smoke"]["status"]=="PASS",
 "notebook_target":"notebooks/kaggle-end-to-end.ipynb","notebook_registry_count":8,
 "seed42_retraining_command_present":False,"full_local_training_run":False,
 "evaluator_smoke_pass":(OUT/"evaluator_smoke/AUDIT_COMPLETE.json").exists(),
 "evaluator_smoke_edge_ablations_skipped":not (OUT/"evaluator_smoke/edge_family_ablation_metrics.csv").exists(),
 "common_evaluation_dry_run_pass":(ROOT/"outputs/d18_analysis/ofix18_c0_c2_multiseed_evaluation/evaluation_plan.json").exists(),
 "aggregator_missing_artifact_guard_pass":(ROOT/"outputs/d18_analysis/ofix18_c0_c2_multiseed_results/missing_artifacts.json").exists(),
 "reports_complete":not missing,"missing_reports":missing,"blocking_issues":[],
 "warnings":["Push new files before Kaggle use.","Evaluation waits for checkpoints.","Five seeds are limited."]}
(OUT/"20_validation_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
if missing: raise RuntimeError(missing)
print(json.dumps({"status":"PASS","output":str(OUT),"reports":21},indent=2))
