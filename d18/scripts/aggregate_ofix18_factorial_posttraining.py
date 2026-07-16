"""Aggregate OFIX18 locked predictions, factorial effects and uncertainty."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import sys
ROOT_HINT=Path(__file__).resolve().parents[2]
if str(ROOT_HINT) not in sys.path: sys.path.insert(0,str(ROOT_HINT))
from d18.scripts.prepare_ofix18_factorial_posttraining import ROOT,OUT,RUNS,FACTORS,j,md

CELLS=["C0","C1","C2","C3"]; CPS=["best","last"]
MODES=["official","remove_structure","shuffle_structure","permute_structure_destinations","degree_matched_random_structure"]
NAMES=["angry","disgust","fear","happy","sad","surprise","neutral"]

def ece(y,p,bins=15):
 conf=p.max(1); pred=p.argmax(1); ok=pred==y; edges=np.linspace(0,1,bins+1); z=0.
 for i in range(bins):
  m=(conf>edges[i])&(conf<=edges[i+1])
  if m.any():z+=m.mean()*abs(ok[m].mean()-conf[m].mean())
 return float(z)
def metric(g,off):
 g=g.sort_values("sample_index");off=off.sort_values("sample_index")
 if not np.array_equal(g.sample_index.to_numpy(),off.sample_index.to_numpy()):raise RuntimeError("order mismatch")
 y=g.true_class.to_numpy(int); pred=g.predicted_class.to_numpy(int)
 lc=[f"logit_{i}" for i in range(7)];pc=[f"prob_{i}" for i in range(7)]
 l=g[lc].to_numpy(float);p=g[pc].to_numpy(float);lo=off[lc].to_numpy(float);po=off[pc].to_numpy(float);prefo=off.predicted_class.to_numpy(int)
 pr,re,f1,s=precision_recall_fscore_support(y,pred,labels=np.arange(7),zero_division=0);cm=confusion_matrix(y,pred,labels=np.arange(7))
 ent=-np.sum(p*np.log(np.clip(p,1e-12,1)),1);margin=np.partition(p,-2,1)[:,-1]-np.partition(p,-2,1)[:,-2]
 mix=.5*(p+po);js=.5*np.sum(p*(np.log(np.clip(p,1e-12,1))-np.log(np.clip(mix,1e-12,1))),1)+.5*np.sum(po*(np.log(np.clip(po,1e-12,1))-np.log(np.clip(mix,1e-12,1))),1)
 cos=np.sum(l*lo,1)/np.clip(np.linalg.norm(l,axis=1)*np.linalg.norm(lo,axis=1),1e-12,None)
 r=dict(count=len(g),accuracy=float((pred==y).mean()),macro_f1=float(f1.mean()),weighted_f1=float(np.average(f1,weights=s)),
 nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),brier_score=float(np.mean(np.sum((p-np.eye(7)[y])**2,1))),
 ece=ece(y,p),mean_entropy=float(ent.mean()),mean_margin=float(margin.mean()),prediction_agreement_with_official=float((pred==prefo).mean()),
 correct_to_wrong=int(np.sum((prefo==y)&(pred!=y))),wrong_to_correct=int(np.sum((prefo!=y)&(pred==y))),
 mean_js_divergence_vs_official=float(js.mean()),mean_logit_cosine_vs_official=float(cos.mean()),mean_logit_l2_change=float(np.linalg.norm(l-lo,axis=1).mean()),
 confusion_matrix_json=json.dumps(cm.tolist()))
 for i,n in enumerate(NAMES):r.update({f"precision_{n}":float(pr[i]),f"recall_{n}":float(re[i]),f"f1_{n}":float(f1[i]),f"support_{n}":int(s[i])})
 return r
def f1fast(y,p):
 cm=np.bincount(y*7+p,minlength=49).reshape(7,7);tp=np.diag(cm).astype(float);den=2*tp+cm.sum(0)-tp+cm.sum(1)-tp
 return float(np.divide(2*tp,den,out=np.zeros(7),where=den>0).mean())
def lincka(x,y):
 x=x-x.mean(0);y=y-y.mean(0);num=np.sum((x.T@y)**2);den=math.sqrt(np.sum((x.T@x)**2)*np.sum((y.T@y)**2));return float(num/max(den,1e-12))
def repqual(z,y):
 cent=np.vstack([z[y==i].mean(0) for i in range(7)]);within=np.linalg.norm(z-cent[y],axis=1);between=[np.linalg.norm(cent[i]-cent[k]) for i in range(7) for k in range(i+1,7)]
 assign=np.argmin(np.linalg.norm(z[:,None,:]-cent[None,:,:],axis=2),axis=1);b=float(np.mean(between));w=float(within.mean())
 return dict(class_centroid_separation=b,within_class_distance=w,between_class_distance=b,within_between_ratio=w/max(b,1e-12),nearest_centroid_accuracy_descriptive=float((assign==y).mean()))
def main():
 manifest=pd.read_csv(ROOT/"outputs/d18_analysis/ofix18_predecision_audit/sample_manifest.csv")
 frames=[]; evalm=[]
 for c in CELLS:
  for cp in CPS:
   d=OUT/"evaluations"/c/cp; m=j(d/"AUDIT_COMPLETE.json");evalm.append({"cell":c,"checkpoint":cp,**m})
   q=pd.read_csv(d/"counterfactual_predictions.csv");q.insert(0,"cell",c);frames.append(q)
 pred=pd.concat(frames,ignore_index=True); expected=4*2*5*715
 if len(pred)!=expected:raise RuntimeError(f"prediction rows {len(pred)} != {expected}")
 num=pred.select_dtypes(include=[np.number]).to_numpy()
 if not np.isfinite(num).all():raise RuntimeError("nonfinite predictions")
 probs=pred[[f"prob_{i}" for i in range(7)]].to_numpy(float)
 if np.abs(probs.sum(1)-1).max()>1e-5:raise RuntimeError("probability sums")
 hashes={x["sample_index_sha256"] for x in evalm}
 if len(hashes)!=1:raise RuntimeError("sample hash mismatch")
 pred.to_csv(OUT/"06_locked_evaluation_predictions.csv",index=False)
 rows=[]
 for (c,cp,mo),g in pred.groupby(["cell","checkpoint_type","mode"],sort=False):
  offall=pred[(pred.cell==c)&(pred.checkpoint_type==cp)&(pred["mode"]=="official")]
  det=g.detected_state.astype(str).str.lower().isin(["true","1"])
  for name,sub in [("all",g),("detected",g[det]),("missing",g[~det])]:
   off=offall[offall.sample_index.isin(sub.sample_index)]
   rows.append(dict(cell=c,checkpoint_type=cp,mode=mo,detection_group=name,**metric(sub,off)))
 mets=pd.DataFrame(rows);mets.to_csv(OUT/"06_locked_evaluation_metrics.csv",index=False)
 ts=pd.read_csv(OUT/"04_training_curve_summary.csv").set_index("cell"); cellrows=[]; base=mets[mets.detection_group=="all"]
 for cp in CPS:
  for c in CELLS:
   q=base[(base.cell==c)&(base.checkpoint_type==cp)].set_index("mode"); vals=[float(q.loc[m,"macro_f1"]) for m in MODES];o=q.loc["official"]
   ep=int(ts.loc[c,"best_epoch" if cp=="best" else "last_epoch"]);h=pd.read_csv(RUNS[c]/"train_log.csv");pt=h[h.epoch==ep].iloc[-1]
   cellrows.append(dict(cell=c,checkpoint_type=cp,selected_epoch=ep,official_accuracy=float(o.accuracy),official_macro_f1=float(o.macro_f1),
   remove_structure_macro_f1=float(q.loc["remove_structure","macro_f1"]),shuffle_structure_macro_f1=float(q.loc["shuffle_structure","macro_f1"]),
   permuted_structure_macro_f1=float(q.loc["permute_structure_destinations","macro_f1"]),degree_matched_random_macro_f1=float(q.loc["degree_matched_random_structure","macro_f1"]),
   robust_min=min(vals),robust_avg=float(np.mean(vals)),official_to_remove_drop_pp=100*float(o.macro_f1-q.loc["remove_structure","macro_f1"]),
   official_to_shuffle_drop_pp=100*float(o.macro_f1-q.loc["shuffle_structure","macro_f1"]),official_to_permute_drop_pp=100*float(o.macro_f1-q.loc["permute_structure_destinations","macro_f1"]),
   official_to_random_drop_pp=100*float(o.macro_f1-q.loc["degree_matched_random_structure","macro_f1"]),official_ece=float(o.ece),official_nll=float(o.nll),
   official_mean_entropy=float(o.mean_entropy),train_val_macro_gap_pp=100*float(pt.train_macro_f1-pt.val_macro_f1),train_macro_f1=float(pt.train_macro_f1),val_macro_f1=float(pt.val_macro_f1)))
 cells=pd.DataFrame(cellrows);cells.to_csv(OUT/"07_factorial_cells.csv",index=False)
 targets=["official_macro_f1","remove_structure_macro_f1","shuffle_structure_macro_f1","permuted_structure_macro_f1","degree_matched_random_macro_f1","robust_min","robust_avg","official_to_remove_drop_pp","official_to_shuffle_drop_pp","official_ece","official_nll","official_mean_entropy","train_val_macro_gap_pp","selected_epoch"]
 er=[]
 for cp in CPS:
  q=cells[cells.checkpoint_type==cp].set_index("cell")
  for t in targets:
   y00,y10,y01,y11=[float(q.loc[c,t]) for c in CELLS]
   for n,v in {"dropedge_main_effect":.5*((y10-y00)+(y11-y01)),"mode_mix_main_effect":.5*((y01-y00)+(y11-y10)),"interaction":y11-y10-y01+y00}.items():
    er.append(dict(checkpoint_type=cp,metric=t,effect=n,value=v,preferred_direction="lower" if t in ["official_to_remove_drop_pp","official_to_shuffle_drop_pp","official_ece","official_nll","official_mean_entropy","train_val_macro_gap_pp"] else "higher",scope="seed42_locked_sample"))
 effects=pd.DataFrame(er);effects.to_csv(OUT/"07_factorial_effects.csv",index=False)
 lines=["# Factorial Effects","","Primary policy: best; last is sensitivity only."]
 for cp in CPS:
  lines += ["",f"## {cp.title()}","",md(cells[cells.checkpoint_type==cp],["cell","official_macro_f1","remove_structure_macro_f1","shuffle_structure_macro_f1","robust_min","robust_avg"]),"",md(effects[(effects.checkpoint_type==cp)&effects.metric.isin(["official_macro_f1","remove_structure_macro_f1","shuffle_structure_macro_f1","robust_min","robust_avg"])],["metric","effect","value","preferred_direction"])]
 (OUT/"07_factorial_effects.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 # paired stratified bootstrap
 ref=pred[(pred.cell=="C0")&(pred.checkpoint_type=="best")&(pred["mode"]=="official")].sort_values("sample_index");y0=ref.true_class.to_numpy(int);pos=[np.flatnonzero(y0==i) for i in range(7)]
 rng=np.random.default_rng(42); samples=[np.concatenate([rng.choice(x,len(x),replace=True) for x in pos]) for _ in range(2000)]
 cons={"C1-C0":lambda v:v["C1"]-v["C0"],"C2-C0":lambda v:v["C2"]-v["C0"],"C3-C1":lambda v:v["C3"]-v["C1"],"C3-C2":lambda v:v["C3"]-v["C2"],
 "dropedge_main_effect":lambda v:.5*((v["C1"]-v["C0"])+(v["C3"]-v["C2"])),"mode_mix_main_effect":lambda v:.5*((v["C2"]-v["C0"])+(v["C3"]-v["C1"])),"interaction":lambda v:v["C3"]-v["C1"]-v["C2"]+v["C0"]}
 btargets=["official_macro_f1","remove_structure_macro_f1","shuffle_structure_macro_f1","robust_min","robust_avg","official_to_remove_drop","official_accuracy"];br=[]
 for cp in CPS:
  pp={(c,m):pred[(pred.cell==c)&(pred.checkpoint_type==cp)&(pred["mode"]==m)].sort_values("sample_index").predicted_class.to_numpy(int) for c in CELLS for m in MODES}
  for t in btargets:
   dist={k:[] for k in cons}
   for s in samples:
    y=y0[s];v={}
    for c in CELLS:
     fm={m:f1fast(y,pp[c,m][s]) for m in MODES}
     v[c]=fm["official"] if t=="official_macro_f1" else fm["remove_structure"] if t=="remove_structure_macro_f1" else fm["shuffle_structure"] if t=="shuffle_structure_macro_f1" else min(fm.values()) if t=="robust_min" else float(np.mean(list(fm.values()))) if t=="robust_avg" else fm["official"]-fm["remove_structure"] if t=="official_to_remove_drop" else float((pp[c,"official"][s]==y).mean())
    for k,f in cons.items():dist[k].append(f(v))
   point=cells[cells.checkpoint_type==cp].set_index("cell");col=t if t!="official_to_remove_drop" else "official_to_remove_drop_pp"
   pv={c:float(point.loc[c,col])/(100 if t=="official_to_remove_drop" else 1) for c in CELLS}
   for k,f in cons.items():
    a=np.asarray(dist[k]);br.append(dict(checkpoint_type=cp,target_metric=t,contrast=k,point_estimate=f(pv),ci95_low=float(np.percentile(a,2.5)),ci95_high=float(np.percentile(a,97.5)),bootstrap_seed=42,replicates=2000,stratified_by_true_class=True))
 boot=pd.DataFrame(br);boot.to_csv(OUT/"08_paired_bootstrap_intervals.csv",index=False)
 (OUT/"08_paired_bootstrap_intervals.md").write_text("\n".join(["# Paired Stratified Bootstrap Intervals","","Seed 42, 2000 replicates, same class-stratified resample for all cells. These intervals are conditional on fixed seed42 checkpoints and do not measure training-seed variance.","",md(boot[boot.checkpoint_type=="best"]),"","Nonlinear robust metrics were recomputed inside each replicate."])+"\n",encoding="utf-8")
 # edge ablations
 ef=[]
 for c in CELLS:
  for cp in CPS:
   q=pd.read_csv(OUT/"evaluations"/c/cp/"edge_family_ablation_metrics.csv");q.insert(0,"cell",c);q.insert(1,"checkpoint_type",cp);full=float(q[q["mode"]=="full_official"].macro_f1.iloc[0]);q["macro_f1_drop_from_full_pp"]=100*(full-q.macro_f1);ef.append(q)
 edge=pd.concat(ef,ignore_index=True);edge.to_csv(OUT/"09_edge_ablation_comparison.csv",index=False);dr=[]
 for (c,cp),q in edge.groupby(["cell","checkpoint_type"]):
  x=q.set_index("mode");f=float(x.loc["full_official","macro_f1"]);rem=float(x.loc["remove_structure","macro_f1"]);rnd=float(x.loc["degree_matched_random_structure","macro_f1"])
  dr.append(dict(cell=c,checkpoint_type=cp,structure_contribution_pp=100*(f-rem),knn_contribution_pp=100*(f-float(x.loc["remove_knn","macro_f1"])),local_contribution_pp=100*(f-float(x.loc["remove_local","macro_f1"])),semantic_structure_advantage_pp=100*(f-rnd),topological_shortcut_advantage_pp=100*(rnd-rem)))
 derived=pd.DataFrame(dr);derived.to_csv(OUT/"09_edge_ablation_derived.csv",index=False)
 (OUT/"09_edge_ablation_comparison.md").write_text("\n".join(["# Edge-Family Ablation Comparison","","Inference-time causal sensitivity, not retraining.","","## Derived",md(derived),"","## Best",md(edge[edge.checkpoint_type=="best"],["cell","mode","accuracy","macro_f1","ece_15bin","mean_predictive_entropy","prediction_agreement_with_official","macro_f1_drop_from_full_pp"])])+"\n",encoding="utf-8")
 # representations
 rr=[]
 for c in CELLS:
  for cp in CPS:
   with np.load(OUT/"evaluations"/c/cp/"counterfactual_embeddings.npz") as z: em={m:z[m].astype(float) for m in MODES}
   y=pred[(pred.cell==c)&(pred.checkpoint_type==cp)&(pred["mode"]=="official")].sort_values("sample_index").true_class.to_numpy(int);o=em["official"]
   for m in MODES:
    x=em[m];cos=np.sum(o*x,1)/np.clip(np.linalg.norm(o,axis=1)*np.linalg.norm(x,axis=1),1e-12,None);nl=np.linalg.norm(o-x,axis=1)/np.clip(np.linalg.norm(o,axis=1),1e-12,None)
    rr.append(dict(cell=c,checkpoint_type=cp,mode=m,paired_cosine_similarity_mean=float(cos.mean()),paired_cosine_similarity_std=float(cos.std()),normalized_l2_distance_mean=float(nl.mean()),linear_cka=lincka(o,x),**repqual(x,y)))
 rep=pd.DataFrame(rr);rep.to_csv(OUT/"11_representation_comparison.csv",index=False)
 (OUT/"11_representation_comparison.md").write_text("\n".join(["# Representation Comparison","","z_image immediately before classifier. Nearest-centroid accuracy is descriptive on the same locked set.","",md(rep[rep.checkpoint_type=="best"])])+"\n",encoding="utf-8")
 # class/detection long form
 cr=[]
 for _,r in mets.iterrows():
  for n in NAMES:cr.append(dict(cell=r.cell,checkpoint_type=r.checkpoint_type,mode=r["mode"],detection_group=r.detection_group,class_name=n,group_count=int(r["count"]),support=int(r[f"support_{n}"]),precision=float(r[f"precision_{n}"]),recall=float(r[f"recall_{n}"]),f1=float(r[f"f1_{n}"]),overall_accuracy=float(r.accuracy),overall_macro_f1=float(r.macro_f1),confusion_matrix_json=r.confusion_matrix_json))
 cls=pd.DataFrame(cr);cls.to_csv(OUT/"12_class_and_detection_analysis.csv",index=False)
 view=cls[(cls.checkpoint_type=="best")&(cls.detection_group=="all")&cls["mode"].isin(["official","remove_structure","degree_matched_random_structure"])]
 (OUT/"12_class_and_detection_analysis.md").write_text("\n".join(["# Class and Detection Analysis","","Locked counts: 715 total, 678 detected, 37 missing. Missing subgroup is small and not overinterpreted.","",md(view,["cell","mode","class_name","support","precision","recall","f1"])])+"\n",encoding="utf-8")
 sr=[]
 for c in CELLS:
  b=cells[(cells.cell==c)&(cells.checkpoint_type=="best")].iloc[0];l=cells[(cells.cell==c)&(cells.checkpoint_type=="last")].iloc[0];sm=j(RUNS[c]/"d18_train_summary.json")
  sr.append(dict(cell=c,best_epoch=int(b.selected_epoch),last_epoch=int(l.selected_epoch),best_locked_official_macro_f1=float(b.official_macro_f1),last_locked_official_macro_f1=float(l.official_macro_f1),last_minus_best_official_pp=100*float(l.official_macro_f1-b.official_macro_f1),best_robust_min=float(b.robust_min),last_robust_min=float(l.robust_min),last_minus_best_robust_min_pp=100*float(l.robust_min-b.robust_min),best_robust_avg=float(b.robust_avg),last_robust_avg=float(l.robust_avg),best_full_test_macro_f1=float(sm["test_macro_f1"]),last_full_test_macro_f1=float(sm["last_test_macro_f1"])))
 sens=pd.DataFrame(sr);sens.to_csv(OUT/"13_best_vs_last_sensitivity.csv",index=False)
 (OUT/"13_best_vs_last_sensitivity.md").write_text("\n".join(["# Best vs Last Sensitivity","","Best is primary; last is secondary. Full-test and locked values are separate populations.","",md(sens)])+"\n",encoding="utf-8")
 summary={"sample_hash":next(iter(hashes)),"prediction_rows":len(pred),"metrics_rows":len(mets),"cells":cells.to_dict("records"),"effects":effects.to_dict("records"),"bootstrap_rows":len(boot),"edge_rows":len(edge),"representation_rows":len(rep),"class_rows":len(cls),"evaluation_manifests":evalm}
 (OUT/"aggregate_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 print(json.dumps({"status":"COMPLETE","rows":{k:summary[k] for k in ["prediction_rows","metrics_rows","bootstrap_rows","edge_rows","representation_rows","class_rows"]}},indent=2))
if __name__=="__main__":main()
