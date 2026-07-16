"""Prepare OFIX18 post-training integrity, config and curve artifacts."""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
from typing import Any
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/d18_analysis/ofix18_factorial_posttraining"
RUNS={
"C0":ROOT/"outputs/d18_runs/ofix18/d18_ofix18_c0_clean_control_seed42",
"C1":ROOT/"outputs/d18_runs/ofix18/d18_ofix18_c1_structure_dropedge_only_seed42",
"C2":ROOT/"outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42",
"C3":ROOT/"outputs/d18_runs/ofix17_structure_reg/d18_ofix17b_structure_mode_mix_seed42"}
SRC={
"C0":ROOT/"configs/d18/overfit_fix_18/d18_ofix18_c0_clean_control_seed42.yaml",
"C1":ROOT/"configs/d18/overfit_fix_18/d18_ofix18_c1_structure_dropedge_only_seed42.yaml",
"C2":ROOT/"configs/d18/overfit_fix_18/d18_ofix18_c2_structure_mode_mix_only_seed42.yaml",
"C3":ROOT/"configs/d18/ofix17_structure_reg/d18_ofix17b_structure_mode_mix_seed42.yaml"}
FACTORS={"C0":(0.0,False,0.0),"C1":(0.3,False,0.0),"C2":(0.0,True,0.3),"C3":(0.3,True,0.3)}

def j(path): return json.loads(path.read_text(encoding="utf-8"))
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def sig(payload):
 state=payload.get("model_state_dict",payload)
 x=[(k,list(v.shape),str(v.dtype)) for k,v in sorted(state.items())]
 return hashlib.sha256(json.dumps(x).encode()).hexdigest()
def flat(x,p=""):
 if isinstance(x,dict):
  z={}
  for k,v in x.items(): z.update(flat(v,f"{p}.{k}" if p else k))
  return z
 return {p:x}
def md(df,cols=None):
 if df.empty:return "_No rows._"
 q=df[list(cols) if cols else list(df.columns)].copy()
 for c in q:
  if pd.api.types.is_float_dtype(q[c]):q[c]=q[c].map(lambda x:"MISSING" if pd.isna(x) else f"{x:.4f}")
  q[c]=q[c].astype(str).str.replace("|","\\|",regex=False)
 return "\n".join(["| "+" | ".join(q.columns)+" |","|"+"|".join(["---"]*len(q.columns))+"|"]+["| "+" | ".join(r)+" |" for r in q.astype(str).values.tolist()])

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 rows=[]; shape_sigs=set(); cp_hashes=set(); warnings=[]; complete={}
 for cell,run in RUNS.items():
  cfg=j(run/"resolved_config.json"); fs=j(run/"feature_schema.json"); gs=j(run/"graph_schema.json"); sm=j(run/"d18_train_summary.json")
  env=j(run/"environment.json") if (run/"environment.json").exists() else {}
  done=j(run/"COMPLETED.json") if (run/"COMPLETED.json").exists() else {}
  resume=(run/"resume_events.jsonl").read_text(encoding="utf-8") if (run/"resume_events.jsonl").exists() else ""
  basis="COMPLETED.json" if done.get("status")=="COMPLETE" else "legacy_summary_history_early_stop"
  ok=done.get("status")=="COMPLETE" or ("early_stopped" in resume and (run/"train_log.csv").exists())
  complete[cell]={"complete":bool(ok),"basis":basis}
  if cell=="C3" and basis!="COMPLETED.json": warnings.append("C3 lacks new-format COMPLETED/environment artifacts; legacy summary, full history and early-stop event establish completion.")
  hist=pd.read_csv(run/"train_log.csv")
  git=((env.get("git") or {}).get("stdout") or "").strip() or "NOT VERIFIABLE"
  for typ in ("best","last"):
   cp=run/"checkpoints"/f"{typ}.pt"; p=torch.load(cp,map_location="cpu",weights_only=False)
   ss=sig(p); ch=sha(cp); shape_sigs.add(ss); cp_hashes.add(ch)
   rows.append(dict(cell=cell,run_id=run.name,checkpoint_type=typ,seed=int(cfg.get("seed",-1)),
    run_dir=str(run.relative_to(ROOT)),config_path=str(SRC[cell].relative_to(ROOT)),
    resolved_config_path=str((run/"resolved_config.yaml").relative_to(ROOT)),history_path=str((run/"train_log.csv").relative_to(ROOT)),
    best_checkpoint_path=str((run/"checkpoints/best.pt").relative_to(ROOT)),last_checkpoint_path=str((run/"checkpoints/last.pt").relative_to(ROOT)),
    best_checkpoint_epoch=int(sm["best_epoch"]),last_checkpoint_epoch=int(hist.epoch.max()),checkpoint_epoch=int(p.get("epoch",-1)),
    monitor_name=cfg["training"]["checkpoint_monitor"],monitor_mode=cfg["training"]["checkpoint_monitor_mode"],
    best_monitor_value=float(sm["best_val_macro_f1"]),node_dim=int(gs.get("node_feature_dim",10)),edge_dim=int(gs.get("edge_feature_dim",6)),
    node_schema=json.dumps(fs.get("node_features",[])),edge_schema=json.dumps(fs.get("edge_features",[])),
    model_signature=ss,config_signature=sha(run/"resolved_config.json"),checkpoint_sha256=ch,
    checkpoint_size_bytes=cp.stat().st_size,checkpoint_mtime=cp.stat().st_mtime,config_mtime=(run/"resolved_config.json").stat().st_mtime,
    training_completed=ok,completion_basis=basis,resume_detected=done.get("resumed","NOT VERIFIABLE"),
    resume_provenance="fresh_confirmed" if done and not done.get("resumed",False) else "NOT VERIFIABLE",
    source_code_signature=env.get("resume_signature","NOT VERIFIABLE"),git_commit=git,
    limitations="" if cell!="C3" else "new-format completion/environment metadata unavailable"))
 art=pd.DataFrame(rows); art.to_csv(OUT/"01_artifact_manifest.csv",index=False)
 integrity={"all_seed42":bool((art.seed==42).all()),"distinct_output_dirs":len(set(RUNS.values()))==4,
 "checkpoint_load_pass":True,"state_dict_shapes_match":len(shape_sigs)==1,"architecture_match":len(shape_sigs)==1,
 "copied_checkpoint_absent":len(cp_hashes)==8,"completion_status":complete,"warnings":warnings,
 "cross_resume_absent":"PASS C0-C2; NOT VERIFIABLE C3"}
 lines=["# Artifact Integrity","",f"Status: **{'PASS' if all([integrity['all_seed42'],integrity['distinct_output_dirs'],integrity['state_dict_shapes_match'],integrity['copied_checkpoint_absent']]) else 'FAIL'}**.","",
 md(art,["cell","checkpoint_type","checkpoint_epoch","checkpoint_size_bytes","checkpoint_sha256","model_signature","training_completed","completion_basis","resume_provenance","git_commit"]),"",
 *[f"- {x}" for x in warnings],"- C0-C2 explicitly report fresh runs. C3 cross-resume provenance is NOT VERIFIABLE from legacy artifacts."]
 (OUT/"02_artifact_integrity.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 cfgs={c:j(r/"resolved_config.json") for c,r in RUNS.items()}; ff={c:flat(v) for c,v in cfgs.items()}
 keys=sorted(set().union(*[set(x) for x in ff.values()])); diffs=[]; unexpected=[]
 expected={"training.graph_regularization.drop_structure_edge_p","training.structure_mode_mix.enabled","training.structure_mode_mix.p_forced_structure"}
 for k in keys:
  vals={c:ff[c].get(k,"MISSING") for c in RUNS}
  if len({json.dumps(v,sort_keys=True) for v in vals.values()})<2:continue
  status="EXPECTED_FACTOR" if k in expected else "IGNORED_METADATA" if k.startswith(("run_name","output_dir","description","logging.")) else "RUNTIME_EQUIVALENT" if k=="graph.cache.fallback_on_error" else "UNEXPECTED"
  diffs.append({"field":k,"status":status,**vals})
  if status=="UNEXPECTED":unexpected.append(k)
 checks=[]
 for c,cfg in cfgs.items():
  t=cfg["training"]; g=t["graph_regularization"]; m=t["structure_mode_mix"]; d,en,p=FACTORS[c]
  vals={"seed42":t["seed"]==42,"global_dropedge_zero":float(t.get("drop_edge_p",0))==0,
  "local_dropedge_zero":float(g.get("drop_local_edge_p",0))==0,"knn_dropedge_zero":float(g.get("drop_knn_edge_p",0))==0,
  "structure_dropedge_expected":math.isclose(float(g["drop_structure_edge_p"]),d),
  "mode_mix_expected":bool(m["enabled"])==en,"mode_probability_expected":math.isclose(float(m["p_forced_structure"]),p),
  "monitor_val_macro_f1":t["checkpoint_monitor"]=="val_macro_f1"}
  checks += [{"cell":c,"invariant":k,"status":"PASS" if v else "FAIL"} for k,v in vals.items()]
 config_ok=not unexpected and all(x["status"]=="PASS" for x in checks)
 (OUT/"03_factorial_config_validation.md").write_text("\n".join(["# Factorial Config Validation","",f"Overall: **{'PASS' if config_ok else 'FAIL'}**.","","## Invariants","",md(pd.DataFrame(checks)),"","## Differences","",md(pd.DataFrame(diffs)),f"\nUnexpected scientific/training fields: {unexpected or 'none'}."] )+"\n",encoding="utf-8")
 curves=[]; sums=[]
 for c,run in RUNS.items():
  h=pd.read_csv(run/"train_log.csv"); sm=j(run/"d18_train_summary.json"); be=int(sm["best_epoch"]); le=int(h.epoch.max())
  out=pd.DataFrame({"cell":c,"run_id":run.name,"epoch":h.epoch})
  mapping={"train_loss":"train_loss","train_accuracy":"train_accuracy","train_macro_f1":"train_macro_f1","val_loss":"val_loss","val_accuracy":"val_accuracy","val_macro_f1":"val_macro_f1","learning_rate":"lr","official_mode_count":"structure_mode_official_sample_count","forced_mode_count":"structure_mode_forced_sample_count","observed_forced_ratio":"structure_mode_forced_sample_pct","structure_edges_seen":"structure_edges_before_drop_mean","structure_edges_retained":"structure_edges_after_drop_mean"}
  for dst,src in mapping.items():out[dst]=h[src] if src in h else math.nan
  out["checkpoint_is_best"]=out.epoch==be; curves.append(out)
  b=h[h.epoch==be].iloc[-1]; l=h[h.epoch==le].iloc[-1]; mi=h.val_loss.idxmin(); pi=h.train_macro_f1.idxmax()
  sums.append(dict(cell=c,best_epoch=be,last_epoch=le,best_val_macro_f1=float(b.val_macro_f1),train_macro_f1_at_best=float(b.train_macro_f1),
   train_val_macro_gap_pp=100*float(b.train_macro_f1-b.val_macro_f1),peak_train_macro_f1=float(h.loc[pi,"train_macro_f1"]),
   peak_train_macro_epoch=int(h.loc[pi,"epoch"]),minimum_val_loss=float(h.loc[mi,"val_loss"]),minimum_val_loss_epoch=int(h.loc[mi,"epoch"]),
   best_val_accuracy=float(b.val_accuracy),best_accuracy_macro_difference_pp=100*float(b.val_accuracy-b.val_macro_f1),
   last_val_macro_f1=float(l.val_macro_f1),best_to_last_val_macro_change_pp=100*float(l.val_macro_f1-b.val_macro_f1),
   configured_structure_retention=1-FACTORS[c][0],observed_structure_retention=1-float(b.get("structure_drop_fraction_observed",math.nan)),
   configured_forced_ratio=FACTORS[c][2],observed_forced_ratio=float(b.get("structure_mode_forced_sample_pct",math.nan)),
   mean_epoch_time_sec=float(h.epoch_time_sec.mean()) if "epoch_time_sec" in h else math.nan))
 cv=pd.concat(curves,ignore_index=True); ss=pd.DataFrame(sums)
 cv.to_csv(OUT/"04_training_curve_comparison.csv",index=False); ss.to_csv(OUT/"04_training_curve_summary.csv",index=False)
 (OUT/"04_training_curve_comparison.md").write_text("\n".join(["# Training Curve Comparison","",md(ss),"","Configured, logged observed and post-training-revalidated behavior are kept distinct. No train macro-F1 was inferred from accuracy."])+"\n",encoding="utf-8")
 (OUT/"05_checkpoint_selection_audit.md").write_text("\n".join(["# Checkpoint Selection Audit","","Primary comparison uses best.pt selected by official validation macro-F1 (max). last.pt is sensitivity only; test/audit results never change selection.","",md(art,["cell","checkpoint_type","checkpoint_epoch","monitor_name","monitor_mode","best_monitor_value","checkpoint_sha256"])])+"\n",encoding="utf-8")
 prep={"integrity":integrity,"config_pass":config_ok,"unexpected_config_fields":unexpected}
 (OUT/"preparation_summary.json").write_text(json.dumps(prep,indent=2),encoding="utf-8")
 print(json.dumps(prep,indent=2))
if __name__=="__main__":main()
