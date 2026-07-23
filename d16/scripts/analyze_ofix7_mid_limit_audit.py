"""Analyze staged OFIX7-mid limit-audit runs without test-aware selection."""
from __future__ import annotations
import argparse, csv, json, math, statistics, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from d16.scripts import prepare_ofix7_mid_limit_audit as prep
from d16.scripts.run_ofix7_mid_limit_variant import test_artifacts

DEV_LOCK_NAME="development_variant_selection_lock.json"
FINAL_LOCK_NAME="final_variant_promotion_lock.json"


def load_csv(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8-sig") as handle: return list(csv.DictReader(handle))


def write_lock(path:Path,payload:dict[str,Any])->str:
    prep.write_json(path,payload); sha=prep.sha256_file(path); path.with_suffix(".sha256").write_text(sha+"\n",encoding="utf-8")
    return sha


def verify_lock(path:Path)->tuple[dict[str,Any],str]:
    side=path.with_suffix(".sha256")
    if not path.exists() or not side.exists(): raise RuntimeError(f"Missing lock/sidecar: {path}")
    sha=prep.sha256_file(path)
    if sha!=side.read_text(encoding="utf-8-sig").strip(): raise RuntimeError(f"Lock SHA mismatch: {path}")
    return prep.load_json(path),sha


def load_registration(path:Path)->tuple[dict[str,Any],str]:
    sha=prep.sha256_file(path); side=path.with_suffix(".sha256")
    if side.exists() and sha!=side.read_text(encoding="utf-8-sig").strip(): raise RuntimeError("Registration SHA mismatch")
    return prep.load_json(path),sha


def run_dir(root:Path,variant:str,seed:int)->Path: return root/prep.run_name(variant,seed)


def selected_train_row(run:Path,epoch:int)->dict[str,str]:
    rows=load_csv(run/"train_log.csv")
    matching=[row for row in rows if int(float(row["epoch"]))==epoch]
    if len(matching)!=1: raise RuntimeError(f"Selected epoch {epoch} absent/ambiguous in {run}")
    return matching[0]


def collect_seed(root:Path,variant:str,seed:int)->tuple[dict[str,Any],list[dict[str,Any]]]:
    run=run_dir(root,variant,seed)
    required=[run/"LIMIT_AUDIT_COMPLETE.json",run/"validation_snapshots/best_val_macro_f1_metrics.json",
              run/"validation_snapshots/best_val_macro_f1_per_class.csv",run/"train_log.csv",
              run/"checkpoints/best_val_macro_f1.pt"]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise RuntimeError(f"Missing validation-only artifacts: {missing}")
    leaked=test_artifacts(run)
    if leaked: raise RuntimeError(f"Test artifacts contaminate selection: {run}: {leaked}")
    metric=prep.load_json(required[1]); epoch=int(metric["epoch"]); train=selected_train_row(run,epoch)
    baseline=prep.load_json(prep.BASELINE_LOCK)["per_seed_validation_metrics"][str(seed)]
    gap=(float(train["train_macro_f1"])-float(metric["macro_f1"]))*100.0
    row={
        "variant":variant,"seed":seed,"selected_epoch":epoch,
        "validation_accuracy":float(metric["accuracy"]),"validation_macro_f1":float(metric["macro_f1"]),
        "train_macro_f1":float(train["train_macro_f1"]),"train_validation_macro_f1_gap_pp":gap,
        "baseline_validation_accuracy":float(baseline["validation_accuracy"]),
        "baseline_validation_macro_f1":float(baseline["validation_macro_f1"]),
        "baseline_train_validation_macro_f1_gap_pp":float(baseline["macro_f1_gap_pp"]),
    }
    row["validation_accuracy_gain_pp"]=(row["validation_accuracy"]-row["baseline_validation_accuracy"])*100.0
    row["validation_macro_f1_gain_pp"]=(row["validation_macro_f1"]-row["baseline_validation_macro_f1"])*100.0
    row["gap_increase_pp"]=gap-row["baseline_train_validation_macro_f1_gap_pp"]
    variant_classes=[{**r,"f1":float(r["f1"]),"class_id":int(r["class_id"])} for r in load_csv(required[2])]
    baseline_path=prep.baseline_run(seed)/"validation_snapshots/best_val_macro_f1_per_class.csv"
    baseline_classes={int(r["class_id"]):float(r["f1"]) for r in load_csv(baseline_path)}
    class_rows=[]
    for r in variant_classes:
        class_rows.append({"variant":variant,"seed":seed,"class_id":r["class_id"],"variant_f1":r["f1"],
                           "baseline_f1":baseline_classes[r["class_id"]],
                           "f1_gain_pp":(r["f1"]-baseline_classes[r["class_id"]])*100.0})
    return row,class_rows


def aggregate(rows:list[dict[str,Any]],class_rows:list[dict[str,Any]])->dict[str,Any]:
    class_gains={}
    for cls in range(7):
        values=[r["f1_gain_pp"] for r in class_rows if r["class_id"]==cls]
        class_gains[str(cls)]=statistics.mean(values)
    return {
        "mean_validation_accuracy_gain_pp":statistics.mean(r["validation_accuracy_gain_pp"] for r in rows),
        "mean_validation_macro_f1_gain_pp":statistics.mean(r["validation_macro_f1_gain_pp"] for r in rows),
        "mean_gap_increase_pp":statistics.mean(r["gap_increase_pp"] for r in rows),
        "min_per_seed_validation_accuracy_gain_pp":min(r["validation_accuracy_gain_pp"] for r in rows),
        "positive_validation_accuracy_seed_count":sum(r["validation_accuracy_gain_pp"]>0 for r in rows),
        "validation_accuracy_sample_sd_pp":statistics.stdev(r["validation_accuracy"]*100 for r in rows) if len(rows)>1 else 0.0,
        "mean_per_class_f1_gain_pp":class_gains,
        "worst_mean_per_class_f1_gain_pp":min(class_gains.values()),
        "mean_train_validation_macro_f1_gap_pp":statistics.mean(r["train_validation_macro_f1_gap_pp"] for r in rows),
    }


def evaluate_development_gate(agg:dict[str,Any],gate:dict[str,Any])->dict[str,bool]:
    return {
        "mean_validation_accuracy_gain":agg["mean_validation_accuracy_gain_pp"]>=gate["mean_validation_accuracy_gain_pp_min"],
        "mean_validation_macro_f1_gain":agg["mean_validation_macro_f1_gain_pp"]>=gate["mean_validation_macro_f1_gain_pp_min"],
        "gap_increase":agg["mean_gap_increase_pp"]<=gate["mean_train_validation_macro_f1_gap_increase_pp_max"],
        "per_seed_accuracy_floor":agg["min_per_seed_validation_accuracy_gain_pp"]>=-gate["max_per_seed_validation_accuracy_loss_pp"],
        "class_f1_floor":agg["worst_mean_per_class_f1_gain_pp"]>=-gate["max_mean_per_class_validation_f1_loss_pp"],
        "positive_direction":agg["positive_validation_accuracy_seed_count"]>=gate["positive_validation_accuracy_direction_seed_count_min"],
    }


def choose_development_winner(aggregates:dict[str,dict[str,Any]],passes:dict[str,bool])->str|None:
    eligible=[v for v in ("S1","O1") if passes.get(v)]
    if not eligible:return None
    def key(v:str):
        a=aggregates[v]
        return (a["mean_validation_accuracy_gain_pp"],a["mean_validation_macro_f1_gain_pp"],
                -a["mean_train_validation_macro_f1_gap_pp"],-a["validation_accuracy_sample_sd_pp"],
                1 if v=="S1" else 0)
    return max(eligible,key=key)


def evaluate_heldout_gate(all_agg:dict[str,Any],heldout_rows:list[dict[str,Any]],baseline_sd_pp:float,gate:dict[str,Any])->dict[str,bool]:
    heldout_gains=[r["validation_accuracy_gain_pp"] for r in heldout_rows]
    direction=(all(v>0 for v in heldout_gains) or
               (statistics.mean(heldout_gains)>0 and min(heldout_gains)>=-0.50))
    return {
        "mean_validation_accuracy_gain":all_agg["mean_validation_accuracy_gain_pp"]>=gate["mean_validation_accuracy_gain_pp_min"],
        "mean_validation_macro_f1_gain":all_agg["mean_validation_macro_f1_gain_pp"]>=gate["mean_validation_macro_f1_gain_pp_min"],
        "heldout_direction":direction,
        "gap_increase":all_agg["mean_gap_increase_pp"]<=gate["mean_train_validation_macro_f1_gap_increase_pp_max"],
        "accuracy_sd":all_agg["validation_accuracy_sample_sd_pp"]<=baseline_sd_pp+gate["validation_accuracy_sample_sd_increase_pp_max"],
    }


def development_selection(reg:dict[str,Any],reg_sha:str,root:Path,out:Path)->dict[str,Any]:
    out.mkdir(parents=True,exist_ok=True); all_rows=[]; all_classes=[]; aggregates={}; gate_rows={}; passes={}
    for variant in ("S1","O1"):
        rows=[]; classes=[]
        for seed in prep.DEVELOPMENT_SEEDS:
            row,cls=collect_seed(root,variant,seed); rows.append(row); classes.extend(cls)
        all_rows.extend(rows); all_classes.extend(classes); aggregates[variant]=aggregate(rows,classes)
        gate_rows[variant]=evaluate_development_gate(aggregates[variant],reg["development_gate"])
        passes[variant]=all(gate_rows[variant].values())
    winner=choose_development_winner(aggregates,passes)
    decision="NO_DEVELOPMENT_VARIANT_PASSES" if winner is None else f"SELECT_{winner}_FOR_HELDOUT"
    payload={"lock_version":"ofix7-mid-development-selection-v1","created_at_utc":datetime.now(timezone.utc).isoformat(),
             "registration_sha256":reg_sha,"decision":decision,"selected_variant":winner,
             "development_seeds":prep.DEVELOPMENT_SEEDS,"aggregates":aggregates,"gate_results":gate_rows,
             "validation_evidence":all_rows,"test_metrics_used":False,
             "heldout_config_sha256":({str(seed):reg["config_file_sha256"][prep.relative(prep.config_path(winner,seed))]
                                       for seed in prep.HELDOUT_SEEDS} if winner else {})}
    prep.write_csv(out/"03_development_validation_results.csv",all_rows)
    prep.write_csv(out/"04_development_classwise_results.csv",all_classes)
    prep.report if False else None
    (out/"03_development_validation_results.md").write_text("# Development Validation Results\n\n"+prep.table(all_rows)+"\n",encoding="utf-8")
    (out/"04_development_classwise_results.md").write_text("# Development Classwise Results\n\n"+prep.table(all_classes)+"\n",encoding="utf-8")
    (out/"05_development_gate_results.md").write_text("# Development Gate Results\n\n~~~json\n"+json.dumps({"aggregates":aggregates,"gates":gate_rows,"decision":decision},indent=2)+"\n~~~\n",encoding="utf-8")
    sha=write_lock(out/DEV_LOCK_NAME,payload); numbered=out/"06_development_variant_selection_lock.json"
    numbered.write_bytes((out/DEV_LOCK_NAME).read_bytes()); numbered.with_suffix(".sha256").write_text(sha+"\n",encoding="utf-8")
    return payload


def heldout_confirmation(reg:dict[str,Any],reg_sha:str,root:Path,out:Path,dev_lock_path:Path)->dict[str,Any]:
    dev,dev_sha=verify_lock(dev_lock_path); winner=dev.get("selected_variant")
    if dev.get("registration_sha256")!=reg_sha:raise RuntimeError("Development lock registration mismatch")
    if winner not in ("S1","O1"):raise RuntimeError("No development winner; held-out runs are prohibited")
    rows=[];classes=[]
    for seed in prep.ALL_SEEDS:
        row,cls=collect_seed(root,winner,seed);rows.append(row);classes.extend(cls)
    agg=aggregate(rows,classes); held=[r for r in rows if r["seed"] in prep.HELDOUT_SEEDS]
    baseline_values=[prep.load_json(prep.BASELINE_LOCK)["per_seed_validation_metrics"][str(s)]["validation_accuracy"]*100 for s in prep.ALL_SEEDS]
    baseline_sd=statistics.stdev(baseline_values)
    gates=evaluate_heldout_gate(agg,held,baseline_sd,reg["heldout_confirmation_gate"])
    decision=f"PROMOTE_{winner}" if all(gates.values()) else "RETAIN_BASELINE_AFTER_HELDOUT"
    payload={"lock_version":"ofix7-mid-final-promotion-v1","created_at_utc":datetime.now(timezone.utc).isoformat(),
             "registration_sha256":reg_sha,"development_selection_lock_sha256":dev_sha,
             "selected_variant":winner,"decision":decision,"all_five_seed_aggregate":agg,
             "heldout_rows":held,"gate_results":gates,"test_metrics_used":False}
    prep.write_csv(out/"08_heldout_validation_results.csv",held);prep.write_csv(out/"09_five_seed_variant_comparison.csv",rows)
    (out/"08_heldout_validation_results.md").write_text("# Held-Out Validation Results\n\n"+prep.table(held)+"\n",encoding="utf-8")
    (out/"09_five_seed_variant_comparison.md").write_text("# Five-Seed Variant Comparison\n\n"+prep.table(rows)+"\n",encoding="utf-8")
    (out/"10_final_promotion_gate.md").write_text("# Final Promotion Gate\n\n~~~json\n"+json.dumps({"aggregate":agg,"gates":gates,"decision":decision},indent=2)+"\n~~~\n",encoding="utf-8")
    sha=write_lock(out/FINAL_LOCK_NAME,payload);numbered=out/"11_final_variant_promotion_lock.json"
    numbered.write_bytes((out/FINAL_LOCK_NAME).read_bytes());numbered.with_suffix(".sha256").write_text(sha+"\n",encoding="utf-8")
    return payload


def test_reveal(reg_sha:str,root:Path,out:Path,prior:Path,device:str,lock_path:Path)->dict[str,Any]:
    lock,_=verify_lock(lock_path)
    if lock.get("registration_sha256")!=reg_sha:raise RuntimeError("Final lock registration mismatch")
    decision=lock.get("decision")
    if decision=="RETAIN_BASELINE_AFTER_HELDOUT":
        result={"decision":decision,"variant_test_evaluation_launched":False,"reason":"Baseline retained by validation lock"}
        prep.write_json(out/"test_reveal_status.json",result);return result
    variant=lock.get("selected_variant")
    if decision!=f"PROMOTE_{variant}":raise RuntimeError("Final promotion lock is ambiguous")
    rows=[]
    for seed in prep.ALL_SEEDS:
        run=run_dir(root,variant,seed);cp=run/"checkpoints/best_val_macro_f1.pt"
        cmd=[sys.executable,"-B","d16/training/train_d16.py","--config",str(run/"resolved_config.yaml"),
             "--prior_dir",str(prior),"--output_dir",str(run),"--device",device,"--eval_only","--checkpoint",str(cp)]
        result=subprocess.run(cmd,cwd=ROOT)
        if result.returncode:raise RuntimeError(f"Test reveal failed for {variant}/{seed}")
        metric=load_csv(run/"test_metrics.csv")[0];rows.append({"variant":variant,"seed":seed,**metric})
    prep.write_csv(out/"12_primary_test_results.csv",rows)
    (out/"12_primary_test_results.md").write_text("# Primary Test Results\n\n"+prep.table(rows)+"\n",encoding="utf-8")
    return {"decision":decision,"variant_test_evaluation_launched":True,"seeds":prep.ALL_SEEDS}


def final_summary(reg_sha:str,out:Path,lock_path:Path)->dict[str,Any]:
    lock,lock_sha=verify_lock(lock_path);decision=lock["decision"]
    if decision.startswith("PROMOTE_"):
        variant=lock["selected_variant"];limit="PRACTICAL_TRAINING_LIMIT_NOT_SUPPORTED";model=f"FINAL_MODEL_{variant}_{'COSINE' if variant=='S1' else 'RADAM'}";release=f"READY_FOR_CLEAN_RELEASE_{variant}"
    elif decision=="RETAIN_BASELINE_AFTER_HELDOUT":
        limit="PRACTICAL_TRAINING_LIMIT_SUPPORTED";model="FINAL_MODEL_BASELINE_OFIX7_MID";release="READY_FOR_CLEAN_RELEASE_BASELINE"
    else:
        limit="PRACTICAL_TRAINING_LIMIT_INCONCLUSIVE";model="BLOCKED_FINAL_MODEL";release="BLOCKED_RELEASE"
    summary={"registration_sha256":reg_sha,"final_promotion_lock_sha256":lock_sha,"promotion_decision":decision,
             "practical_limit_conclusion":limit,"final_model_decision":model,"release_readiness":release,
             "test_metrics_changed_selection":False}
    prep.write_json(out/"22_machine_readable_summary.json",summary);prep.write_json(out/"23_validation_summary.json",summary)
    (out/"16_practical_limit_conclusion.md").write_text("# Practical Limit Conclusion\n\n"+limit+"\n",encoding="utf-8")
    (out/"17_final_model_decision.md").write_text("# Final Model Decision\n\n"+model+"\n",encoding="utf-8")
    (out/"18_release_readiness.md").write_text("# Release Readiness\n\n"+release+"\n",encoding="utf-8")
    final_lock={"lock_version":"ofix7-mid-final-model-v1",**summary};sha=write_lock(out/"19_final_model_lock.json",final_lock)
    return summary


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=["development-selection","heldout-confirmation","verify-final-lock","test-reveal","final-summary"])
    p.add_argument("--registration",type=Path,required=True);p.add_argument("--run-root",type=Path,default=prep.RUN_ROOT)
    p.add_argument("--output-dir",type=Path,default=prep.POSTTRAIN_DIR);p.add_argument("--development-selection-lock",type=Path)
    p.add_argument("--final-promotion-lock",type=Path);p.add_argument("--prior-dir",type=Path);p.add_argument("--device",default="cuda:0");args=p.parse_args()
    reg,reg_sha=load_registration(args.registration);args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.stage=="development-selection":result=development_selection(reg,reg_sha,args.run_root,args.output_dir)
    elif args.stage=="heldout-confirmation":
        if args.development_selection_lock is None:raise RuntimeError("Held-out confirmation requires development lock")
        result=heldout_confirmation(reg,reg_sha,args.run_root,args.output_dir,args.development_selection_lock)
    elif args.stage=="verify-final-lock":
        if args.final_promotion_lock is None:raise RuntimeError("Final lock path required")
        lock,sha=verify_lock(args.final_promotion_lock)
        if lock.get("registration_sha256")!=reg_sha:raise RuntimeError("Final lock registration mismatch")
        result={"valid":True,"sha256":sha,"decision":lock.get("decision")}
    elif args.stage=="test-reveal":
        if args.final_promotion_lock is None or args.prior_dir is None:raise RuntimeError("Test reveal requires final lock and prior dir")
        result=test_reveal(reg_sha,args.run_root,args.output_dir,args.prior_dir,args.device,args.final_promotion_lock)
    else:
        if args.final_promotion_lock is None:raise RuntimeError("Final summary requires final lock")
        result=final_summary(reg_sha,args.output_dir,args.final_promotion_lock)
    print(json.dumps(result,indent=2,default=str))


if __name__=="__main__":main()

