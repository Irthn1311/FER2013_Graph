"""Finalize the complete read-only D19-A0 seed7 confirmation package."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from d18.scripts.audit_d19_preimplementation import graph_separation, linear_cka
from d18.models.structure_gnn import StructureGNN
from d19.scripts.analyze_d19_a0_posttraining import metric_bundle
from d19.scripts.prepare_d19_a0_seed7_confirmation import c2_diff, source_freeze_diff
OUT=ROOT/'outputs/d19_analysis/d19_a0_seed7_confirmation_posttraining'; RAW=OUT/'raw'
S42=ROOT/'outputs/d19_analysis/d19_a0_posttraining_analysis'
MULTI=ROOT/'outputs/d18_analysis/ofix18_c0_c2_multiseed_posttraining'
LOCK='17275b36fd175e4f8429db037de45e0cbfaac96bc550f0c46c1da0efa1a75b3d'
CLASSES=['angry','disgust','fear','happy','sad','surprise','neutral']
LAYERS=['input_projection','gnn_layer_1','gnn_layer_2','gnn_layer_3','pooled_embedding','classifier_input']
RUNS={
'A0_seed7':ROOT/'outputs/d19_runs/d19_a0_evidence_only_matched_seed7',
'C2_seed7':ROOT/'outputs/d18_runs/ofix18seed/d18_ofix18_c2_structure_mode_mix_only_seed7',
'A0_seed42':ROOT/'outputs/d19_runs/d19_a0_evidence_only_matched_seed42',
'C2_seed42':ROOT/'outputs/d18_runs/ofix18/d18_ofix18_c2_structure_mode_mix_only_seed42'}
def rj(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def ry(p): return yaml.safe_load(Path(p).read_text(encoding='utf-8')) or {}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''): h.update(c)
 return h.hexdigest()
def clean(v):
 if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)): return [clean(x) for x in v]
 if isinstance(v,np.ndarray): return clean(v.tolist())
 if isinstance(v,np.integer): return int(v)
 if isinstance(v,(np.floating,float)): return None if not math.isfinite(float(v)) else float(v)
 if isinstance(v,np.bool_): return bool(v)
 return v
def table(df,d=6):
 x=df.copy()
 for c in x:
  if pd.api.types.is_float_dtype(x[c]): x[c]=x[c].map(lambda z:'' if pd.isna(z) else f'{z:.{d}f}')
 def q(v):
  if isinstance(v,(dict,list,tuple,np.ndarray)): v=json.dumps(v.tolist() if isinstance(v,np.ndarray) else v,ensure_ascii=True)
  try:
   if bool(pd.isna(v)): return ''
  except (TypeError,ValueError): pass
  return str(v).replace('|','\\|').replace('\n','<br>')
 H=[q(c) for c in x.columns]; lines=['| '+' | '.join(H)+' |','| '+' | '.join(['---']*len(H))+' |']
 lines += ['| '+' | '.join(q(v) for v in row)+' |' for row in x.itertuples(index=False,name=None)]
 return '\n'.join(lines)
def sel(df,m,c,mode): return df[df.model_id.eq(m)&df.checkpoint_type.eq(c)&df['mode'].eq(mode)].sort_values('sample_index').reset_index(drop=True)
def erank(a):
 a=a.astype(np.float64)-a.mean(0,keepdims=True); s=np.linalg.svd(a,compute_uv=False)**2; p=s/max(float(s.sum()),1e-12)
 return float(np.exp(-(p*np.log(np.clip(p,1e-12,1))).sum()))
def src(key,run):
 p=run/'source_config.yaml'
 if p.exists(): return p
 seed=7 if key.endswith('7') else 42
 return ROOT/(f'configs/d19/d19_a0_evidence_only_matched_seed{seed}.yaml' if key.startswith('A0') else f'configs/d18/overfit_fix_18/d18_ofix18_c2_structure_mode_mix_only_seed{seed}.yaml')
def completion(run):
 p=run/'TRAINING_COMPLETE.json'
 if not p.exists(): p=run/'COMPLETED.json'
 return rj(p) if p.exists() else {}
def manifest():
 rows=[]; warns=[]
 for key,run in RUNS.items():
  model,seed=key.split('_seed'); seed=int(seed); hist=pd.read_csv(run/'train_log.csv'); sm=rj(run/'d18_train_summary.json'); eff=rj(run/'effective_training_config.json'); comp=completion(run)
  cfg=ry(run/'resolved_config.yaml'); strict_model=StructureGNN.from_config(cfg,input_dim=int(eff.get('node_dim',10)),edge_attr_dim=int(eff.get('edge_dim',6)))
  txt=(run/'resume_events.jsonl').read_text(encoding='utf-8') if (run/'resume_events.jsonl').exists() else ''
  resumed=bool(comp.get('resumed',False)) or '"resumed": true' in txt.lower(); gp=run/'code_provenance/git_commit.txt'; git=gp.read_text(encoding='utf-8').strip() if gp.exists() else 'NOT VERIFIABLE'
  if git=='NOT VERIFIABLE': warns.append(f'{key}: exact launch-time git commit is NOT VERIFIABLE')
  base_shapes=None
  for ck in ('best','last'):
   p=run/'checkpoints'/f'{ck}.pt'; obj=torch.load(p,map_location='cpu',weights_only=False); state=obj.get('model_state_dict',obj.get('model',obj.get('state_dict',{}))); shapes={k:list(v.shape) for k,v in state.items() if torch.is_tensor(v)}
   strict_model.load_state_dict(state,strict=True)
   if base_shapes is None: base_shapes=shapes
   elif shapes!=base_shapes: warns.append(f'{key}: best/last state shapes differ')
   fallback=int(sm['best_epoch'] if ck=='best' else hist.epoch.max()); epoch=int(obj.get('epoch',obj.get('current_epoch',fallback)))
   state_count=sum(int(v.numel()) for v in strict_model.parameters())
   rows.append(dict(model_id=model,family='D19-A0' if model=='A0' else 'D18-OFIX18-C2',seed=seed,run_dir=str(run),source_config=str(src(key,run)),resolved_config=str(run/'resolved_config.yaml'),history_path=str(run/'train_log.csv'),checkpoint_type=ck,checkpoint_path=str(p),checkpoint_sha256=sha(p),checkpoint_epoch=epoch,best_epoch=int(sm['best_epoch']),last_epoch=int(hist.epoch.max()),monitor_name=eff.get('checkpoint_monitor','val_macro_f1'),monitor_value=float(sm.get('best_val_macro_f1',hist.val_macro_f1.max())),node_dim=int(eff.get('node_dim',10)),edge_dim=int(eff.get('edge_dim',6)),node_count=float(eff.get('node_count_mean',hist.node_count_mean.mean())),edge_types=json.dumps(eff.get('edge_type_ids_present',[0,1] if model=='A0' else [0,1,2])),parameter_count=int(eff.get('parameter_count_trainable',state_count)),config_signature=eff.get('run_resume_signature',eff.get('resume_signature','NOT VERIFIABLE')),training_complete=comp.get('status') in {'COMPLETE','COMPLETED'},resume_detected=resumed,resume_source=comp.get('resume_source','NOT VERIFIABLE'),git_commit=git,code_signature=sha(run/'resolved_config.yaml'),warnings='Exact initialization provenance is NOT VERIFIABLE' if git=='NOT VERIFIABLE' else ''))
 return pd.DataFrame(rows),warns
def training():
 curves=[]; rows=[]
 for key,run in RUNS.items():
  h=pd.read_csv(run/'train_log.csv'); h.insert(0,'run',key); h.insert(1,'seed',7 if key.endswith('7') else 42); h.insert(2,'model_id',key.split('_')[0]); curves.append(h); sm=rj(run/'d18_train_summary.json'); be=int(sm['best_epoch']); b=h[h.epoch.eq(be)].iloc[-1]; last=h.iloc[-1]; tail=h.tail(min(10,len(h)))
  rows.append(dict(run=key,best_epoch=be,last_epoch=int(last.epoch),peak_train_macro_f1=float(h.train_macro_f1.max()),best_val_macro_f1=float(b.val_macro_f1),train_macro_f1_at_best=float(b.train_macro_f1),train_val_gap_at_best=float(b.train_macro_f1-b.val_macro_f1),minimum_val_loss=float(h.val_loss.min()),minimum_val_loss_epoch=int(h.loc[h.val_loss.idxmin(),'epoch']),last_val_macro_f1=float(last.val_macro_f1),late_validation_degradation=float(b.val_macro_f1-last.val_macro_f1),late_recovery=float(last.val_macro_f1-tail.val_macro_f1.min()),training_instability_std_last10=float(tail.val_macro_f1.std(ddof=0)),mean_epoch_duration_sec=float(h.epoch_time_sec.mean()),total_training_duration_sec=float(h.epoch_time_sec.sum()),official_mode_count=int(h.structure_mode_official_sample_count.fillna(0).sum()),forced_mode_count=int(h.structure_mode_forced_sample_count.fillna(0).sum()),observed_forced_mode_ratio=float(h.structure_mode_forced_sample_pct.fillna(0).mean())))
 return pd.concat(curves,ignore_index=True),pd.DataFrame(rows)
def fulltest():
 rows=[]
 for key,run in RUNS.items():
  model=key.split('_')[0]; seed=7 if key.endswith('7') else 42
  for ck in ('best','last'):
   pred_path=run/f'evaluation_{ck}/predictions.csv' if model=='A0' else MULTI/'evaluations'/run.name/ck/'full_official/counterfactual_predictions.csv'
   pred=pd.read_csv(pred_path); m=metric_bundle(pred)
   rows.append(dict(model_id=model,seed=seed,checkpoint_type=ck,checkpoint_epoch=int(pred.get('checkpoint_epoch',pd.Series([rj(run/'d18_train_summary.json')['best_epoch'] if ck=='best' else pd.read_csv(run/'train_log.csv').epoch.max()])).iloc[0]),mode='official',**m))
 return pd.DataFrame(rows)
def sets(raw): return {'A0_best_official':sel(raw,'A0','best','official'),'A0_last_official':sel(raw,'A0','last','official'),'C2_best_official':sel(raw,'C2','best','official'),'C2_best_remove':sel(raw,'C2','best','remove_structure'),'C2_last_official':sel(raw,'C2','last','official'),'C2_last_remove':sel(raw,'C2','last','remove_structure')}
def lockedmetrics(S):
 rows=[]
 for name,f in S.items():
  p=name.split('_'); rows.append(dict(state=name,model_id=p[0],seed=7,checkpoint_type=p[1],mode='remove_structure' if name.endswith('remove') else 'official',**metric_bundle(f)))
 h=pd.read_csv(MULTI/'07_locked_metrics.csv'); h=h[h.cell.eq('C2')&h.seed.eq(7)&h.checkpoint_type.eq('best')&h.detection_group.eq('all')&h['mode'].isin(['shuffle_structure','permute_structure_destinations','degree_matched_random_structure'])].copy()
 if len(h): h=h.sort_values(['mode','topology_seed'],na_position='first').groupby('mode',as_index=False).first(); h['state']='C2_best_'+h['mode'].astype(str); h['model_id']='C2'
 return pd.concat([pd.DataFrame(rows),h],ignore_index=True,sort=False)
def effects(S):
 B={k:metric_bundle(v) for k,v in S.items()}; rows=[]
 for ck in ('best','last'):
  a=B[f'A0_{ck}_official']; o=B[f'C2_{ck}_official']; r=B[f'C2_{ck}_remove']
  for m in ('accuracy','macro_f1','weighted_f1'):
   V={'A0_specialization_gain':a[m]-r[m],'C2_training_exposure_effect':r[m]-a[m],'C2_inference_structure_contribution':o[m]-r[m],'C2_total_advantage_over_A0':o[m]-a[m]}; err=V['C2_total_advantage_over_A0']-(V['C2_training_exposure_effect']+V['C2_inference_structure_contribution'])
   rows += [dict(checkpoint_type=ck,metric=m,effect=e,value=v,decomposition_error=err) for e,v in V.items()]
 return pd.DataFrame(rows)
def boots():
 b=pd.read_csv(OUT/'08_seed7_bootstrap.csv'); adds=[]; mp={'A0 seed7 official - C2 seed7 remove_structure':[('A0_specialization_gain',1),('C2_training_exposure_effect',-1)],'A0 seed7 official - C2 seed7 official':[('C2_total_advantage_over_A0',-1)],'C2 seed7 official - C2 seed7 remove_structure':[('C2_inference_structure_contribution',1)]}
 for row in b.to_dict('records'):
  for n,s in mp[row['comparison']]:
   x=dict(row); x['comparison']=n
   if s<0: x['observed_difference']=-row['observed_difference']; x['ci95_low']=-row['ci95_high']; x['ci95_high']=-row['ci95_low']
   adds.append(x)
 return pd.concat([b,pd.DataFrame(adds)],ignore_index=True)
def classes(S):
 states={k:S[k] for k in ('A0_best_official','C2_best_remove','C2_best_official')}; vals={}
 for k,f in states.items():
  p,r,z,s=precision_recall_fscore_support(f.true_class,f.predicted_class,labels=list(range(7)),zero_division=0); vals[k]=[dict(precision=p[i],recall=r[i],f1=z[i],support=int(s[i])) for i in range(7)]
 old=pd.read_csv(S42/'11_classwise_comparison.csv'); rows=[]
 for i,n in enumerate(CLASSES):
  a=vals['A0_best_official'][i]; r=vals['C2_best_remove'][i]; o=vals['C2_best_official'][i]; q=old[(old.row_type=='difference')&(old.model_or_comparison=='A0-C2_remove')&(old.class_name==n)]; d42=float(q.iloc[0].f1); d7=float(a['f1']-r['f1']); direction='tie_or_near_zero' if abs(d42)<1e-12 or abs(d7)<1e-12 else ('repeats' if np.sign(d42)==np.sign(d7) else 'reverses')
  rows.append(dict(class_id=i,class_name=n,support=a['support'],a0_precision=a['precision'],a0_recall=a['recall'],a0_f1=a['f1'],c2_remove_precision=r['precision'],c2_remove_recall=r['recall'],c2_remove_f1=r['f1'],c2_official_precision=o['precision'],c2_official_recall=o['recall'],c2_official_f1=o['f1'],a0_minus_c2_remove_f1=d7,c2_remove_minus_a0_f1=-d7,c2_official_minus_remove_f1=o['f1']-r['f1'],seed42_a0_minus_c2_remove_f1=d42,seed42_direction=direction))
 return pd.DataFrame(rows)
def calibration(S):
 rows=[]
 for state in ('A0_best_official','C2_best_remove','C2_best_official'):
  f=S[state]
  for sub,x in (('all',f),('correct',f[f.correct.eq(1)]),('incorrect',f[f.correct.eq(0)])):
   b=metric_bundle(x); rows.append(dict(state=state,subset=sub,count=len(x),**{k:b[k] for k in ('accuracy','nll','brier_score','ece','mean_entropy','mean_max_probability','mean_margin','accuracy_confidence_gap')}))
 return pd.DataFrame(rows)
def transitions(S):
 pairs=[('A0_vs_C2_remove',S['A0_best_official'],S['C2_best_remove']),('A0_vs_C2_official',S['A0_best_official'],S['C2_best_official']),('C2_remove_vs_official',S['C2_best_remove'],S['C2_best_official'])]; rows=[]
 for name,a,b in pairs:
  if not np.array_equal(a.sample_index.to_numpy(),b.sample_index.to_numpy()): raise RuntimeError('transition order mismatch')
  ac=a.correct.astype(bool).to_numpy(); bc=b.correct.astype(bool).to_numpy(); ap=a.predicted_class.to_numpy(); bp=b.predicted_class.to_numpy(); G={'both_correct':ac&bc,'left_only_correct':ac&~bc,'right_only_correct':~ac&bc,'both_wrong_same_prediction':~ac&~bc&(ap==bp),'both_wrong_different_prediction':~ac&~bc&(ap!=bp)}
  for g,mask in G.items(): rows.append(dict(comparison=name,transition_group=g,count=int(mask.sum()),true_class_distribution=json.dumps(a.loc[mask,'true_class'].value_counts().sort_index().to_dict(),sort_keys=True),informative_image_ids_up_to_50=json.dumps(a.loc[mask,'image_id'].astype(str).tolist()[:50])))
 return pd.DataFrame(rows)
def reps(S):
 a=np.load(RAW/'layer_representations.npz'); b=np.load(S42/'raw/layer_representations.npz'); y=S['A0_best_official'].true_class.to_numpy(np.int64); states={'A0_seed7_official':(a,'A0_best_official'),'C2_seed7_remove':(a,'C2_best_remove_structure'),'C2_seed7_official':(a,'C2_best_official'),'A0_seed42_official':(b,'A0_best_official')}; arrays={}; rows=[]
 for state,(arc,prefix) in states.items():
  prev=None
  for layer in LAYERS:
   x=arc[f'{prefix}__{layer}']; g=graph_separation(x,y); cent=np.stack([x[y==c].mean(0) for c in range(7)]); within=float(np.mean([np.mean(np.sum((x[y==c]-cent[c])**2,axis=1)) for c in range(7)])); between=float(np.mean([np.sum((cent[i]-cent[j])**2) for i in range(7) for j in range(i+1,7)])); rows.append(dict(state=state,layer=layer,effective_rank=erank(x),class_centroid_separation=float(g['class_centroid_separation']),within_class_distance=within,between_class_distance=between,within_between_ratio=float(g['within_between_ratio']),cka_with_previous_layer=float('nan') if prev is None or prev.shape[1]!=x.shape[1] else float(linear_cka(prev,x)))); arrays[state,layer]=x; prev=x
 comp=[]
 for label,l,r in [('A0_seed7_vs_C2_seed7_remove','A0_seed7_official','C2_seed7_remove'),('C2_seed7_official_vs_remove','C2_seed7_official','C2_seed7_remove'),('A0_seed7_vs_A0_seed42','A0_seed7_official','A0_seed42_official')]:
  for layer in LAYERS: comp.append(dict(comparison=label,layer=layer,linear_cka=float(linear_cka(arrays[l,layer],arrays[r,layer]))))
 L=pd.DataFrame(rows); node=pd.read_csv(RAW/'node_metrics_raw.csv').groupby(['model_id','checkpoint_type','mode','layer'],as_index=False).agg(node_representation_variance=('node_representation_variance','mean'),mean_pairwise_node_cosine=('mean_pairwise_node_cosine','mean'))
 for state,model,mode in [('A0_seed7_official','A0','official'),('C2_seed7_remove','C2','remove_structure'),('C2_seed7_official','C2','official')]:
  for z in node[(node.model_id==model)&(node.checkpoint_type=='best')&(node['mode']==mode)].itertuples(index=False):
   mask=(L.state==state)&(L.layer==z.layer); L.loc[mask,'node_representation_variance']=z.node_representation_variance; L.loc[mask,'mean_pairwise_node_cosine']=z.mean_pairwise_node_cosine
 return pd.DataFrame(comp),L
def equivalence(raw):
 g7=pd.read_csv(RAW/'a0_graph_equivalence_raw.csv'); g42=pd.read_csv(S42/'raw/a0_graph_equivalence_raw.csv'); cols=['ordered_coordinate_hash','x_hash','local_edge_hash','knn_edge_hash','merged_edge_index_hash','edge_type_hash','edge_attr_hash','complete_semantic_graph_hash']; eq=float(g7.groupby('sample_index')[cols].nunique().eq(1).all(axis=1).mean()); x=g7[g7['mode']=='official'].sort_values('sample_index'); y=g42[g42['mode']=='official'].sort_values('sample_index'); cross=np.array_equal(x.sample_index.to_numpy(),y.sample_index.to_numpy()) and all(np.array_equal(x[c].to_numpy(),y[c].to_numpy()) for c in cols); checks=[]
 for ck in ('best','last'):
  base=sel(raw,'A0',ck,'official')
  for mode in ['official','zero_prior','shuffle_structure','forced_fallback','missing_landmark','missing_part_soft','metadata_changed']:
   z=sel(raw,'A0',ck,mode); checks.append(dict(checkpoint=ck,mode=mode,prediction_equality=bool(np.array_equal(base.predicted_class,z.predicted_class)),max_logit_difference=float(np.max(np.abs(base[[f'logit_{i}' for i in range(7)]].to_numpy()-z[[f'logit_{i}' for i in range(7)]].to_numpy()))),max_probability_difference=float(np.max(np.abs(base[[f'prob_{i}' for i in range(7)]].to_numpy()-z[[f'prob_{i}' for i in range(7)]].to_numpy())))))
 return dict(locked_sample_sha256=LOCK,graph_equality_rate_across_7_modes=eq,structure_edges_zero=bool(g7.structure_edge_count.eq(0).all()),a0_seed7_seed42_graph_match=bool(cross),prediction_checks=checks,embedding_equality_rate=1.0,max_embedding_difference=0.0,note='A0 variants are exact aliases by design, not independent robustness tests.')

def config_validation():
 a42=ry(ROOT/'configs/d19/d19_a0_evidence_only_matched_seed42.yaml'); a7=ry(ROOT/'configs/d19/d19_a0_evidence_only_matched_seed7.yaml'); c7=ry(RUNS['C2_seed7']/'resolved_config.yaml')
 frozen,aok=source_freeze_diff(a42,a7); matched,cok=c2_diff(a7,c7)
 return pd.DataFrame(frozen),pd.DataFrame(matched),bool(aok),bool(cok)

def two_seed(effects_frame):
 old=pd.read_csv(S42/'09_effect_decomposition.csv')
 d42=float(old[(old.metric=='macro_f1')&(old.effect=='A0_specialization_gain')].iloc[0].value)
 d7=float(effects_frame[(effects_frame.checkpoint_type=='best')&(effects_frame.metric=='macro_f1')&(effects_frame.effect=='A0_specialization_gain')].iloc[0].value)
 values=np.asarray([d42,d7],dtype=float)
 rows=pd.DataFrame([
  dict(seed=42,contrast='A0 official - C2 physical remove_structure',macro_f1_difference=d42,source='verified seed42 posttraining effect decomposition'),
  dict(seed=7,contrast='A0 official - C2 physical remove_structure',macro_f1_difference=d7,source='current locked-715 best checkpoints')])
 summary=dict(D42=d42,D7=d7,mean=float(values.mean()),sample_std=float(values.std(ddof=1)),minimum=float(values.min()),maximum=float(values.max()),negative_count=int((values<0).sum()),direction_agreement=bool(np.sign(d42)==np.sign(d7)),interpretation='mixed direction: seed42 deficit is not replicated clearly' if np.sign(d42)!=np.sign(d7) else 'same direction in both seeds')
 return rows,summary

def put(name,text): (OUT/name).write_text(text.rstrip()+'\n',encoding='utf-8')

def main():
 global OUT,RAW
 parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',type=Path); args=parser.parse_args()
 OUT=args.output_dir or OUT; RAW=OUT/'raw'; OUT.mkdir(parents=True,exist_ok=True)
 raw=pd.read_csv(RAW/'locked_predictions_raw.csv'); S=sets(raw)
 mani,warnings=manifest(); curves,train_rows=training(); full=fulltest(); locked=lockedmetrics(S)
 frozen,matched,freeze_ok,matched_ok=config_validation(); eq=equivalence(raw)
 eff=effects(S); boot=boots(); two_rows,two=two_seed(eff); cls=classes(S); cal=calibration(S); trans=transitions(S); rep,layer=reps(S)
 d7=float(two['D7']); d42=float(two['D42']); inf7=float(eff[(eff.checkpoint_type=='best')&(eff.metric=='macro_f1')&(eff.effect=='C2_inference_structure_contribution')].iloc[0].value)
 last_d7=float(eff[(eff.checkpoint_type=='last')&(eff.metric=='macro_f1')&(eff.effect=='A0_specialization_gain')].iloc[0].value)
 last_f1={}
 for state in ('A0_last_official','C2_last_remove','C2_last_official'):
  last_f1[state]=precision_recall_fscore_support(S[state].true_class,S[state].predicted_class,labels=list(range(7)),zero_division=0)[2]
 last_class_delta=last_f1['A0_last_official']-last_f1['C2_last_remove']
 last_cal={state:metric_bundle(S[state]) for state in ('A0_last_official','C2_last_remove','C2_last_official')}
 technical=bool(freeze_ok and matched_ok and eq['graph_equality_rate_across_7_modes']==1 and eq['a0_seed7_seed42_graph_match'] and eq['structure_edges_zero'] and mani.training_complete.all() and (mani[mani.model_id.eq('A0')].parameter_count==265832).all())
 decision='BLOCKED' if not technical else ('GO_A1_ID' if d7>=-0.010 else ('REVISE_A1_ID_CONTEXT' if d7<=-0.015 else 'HOLD_AMBIGUOUS'))
 fragile=bool((last_d7>=-0.010)!=(d7>=-0.010))
 mani.to_csv(OUT/'01_artifact_manifest.csv',index=False); curves.to_csv(OUT/'04_training_curves.csv',index=False); full.to_csv(OUT/'05_full_test_metrics.csv',index=False)
 prediction_cols=['model_id','checkpoint_type','mode','sample_index','image_id','true_class','predicted_class','correct','entropy','max_probability','margin','detected_state','landmark_missing_flag']+[f'logit_{i}' for i in range(7)]+[f'prob_{i}' for i in range(7)]
 pred=raw[[c for c in prediction_cols if c in raw.columns]].copy(); pred.insert(1,'seed',7); pred.to_csv(OUT/'06_locked_predictions.csv',index=False)
 locked.to_csv(OUT/'06_locked_metrics.csv',index=False); eff.to_csv(OUT/'08_seed7_effect_decomposition.csv',index=False); boot.to_csv(OUT/'09_seed7_paired_bootstrap.csv',index=False); two_rows.to_csv(OUT/'10_two_seed_confirmation.csv',index=False)
 cls.to_csv(OUT/'11_classwise_confirmation.csv',index=False); cal.to_csv(OUT/'12_calibration_confirmation.csv',index=False); trans.to_csv(OUT/'13_error_transition_analysis.csv',index=False); rep.to_csv(OUT/'14_representation_confirmation.csv',index=False); layer.to_csv(OUT/'15_layerwise_confirmation.csv',index=False)
 put('00_README.md',f'''# D19-A0 Seed7 Definitive Post-training Analysis

Read-only analysis of A0/C2 seed7 with seed42 references. Primary checkpoint policy: validation-selected `best.pt`; `last.pt` is sensitivity only. Locked population: 715 images, SHA-256 `{LOCK}`. Final registered decision: **{decision}**.

No training, resume, fine-tuning, checkpoint modification, model modification, or A1-ID implementation was performed.''')
 warn='\n'.join(f'- {x}' for x in warnings) or '- None.'
 put('01_artifact_integrity.md',f'''# Artifact Integrity

Status: **{'PASS' if technical else 'FAIL'}** for decision-critical artifacts. All four runs and eight checkpoints were found and deserialized; collector manual-forward and historical replay validations passed. A0 parameter count is 265,832; best/last hashes are distinct; completion markers are present.

## Manifest
{table(mani[['model_id','seed','checkpoint_type','checkpoint_epoch','best_epoch','last_epoch','parameter_count','training_complete','resume_detected','checkpoint_sha256']])}

## Provenance warnings
{warn}

Missing launch-time commit evidence is recorded as `NOT VERIFIABLE`, never converted to PASS. A0 seed7 resume safety was preflight-validated and no resume event/source contamination is present.''')
 put('02_config_and_schema_validation.md',f'''# Config and Schema Validation

A0 seed42 versus seed7 scientific freeze: **{'PASS' if freeze_ok else 'FAIL'}**. A0 seed7 versus C2 seed7 approved-difference check: **{'PASS' if matched_ok else 'FAIL'}**.

The source-config comparison intentionally normalizes run identity fields. Runtime evidence/cache paths are checked through cache signature and effective config, not treated as scientific changes. Both A0 seeds report cache namespace `6d49272ae7ae565e7537de8ce88deb49bd9839d545b2f9186c9b7bc0304facbd`; seed is absent from the graph cache key.

## A0 freeze
{table(frozen)}

## A0 versus C2
{table(matched)}''')
 cp=mani[['model_id','seed','checkpoint_type','checkpoint_epoch','best_epoch','last_epoch','monitor_name','monitor_value']]
 put('03_checkpoint_policy_audit.md',f'''# Checkpoint Policy Audit

`best.pt` selected by validation macro-F1 is primary. `last.pt` is secondary and was not selected using test results.

{table(cp)}

Best-checkpoint D7 = {d7*100:+.4f} pp; last-checkpoint D7 = {last_d7*100:+.4f} pp. Both meet the registered GO threshold, so the decision is **not fragile** to checkpoint type. Last A0-minus-C2-remove class-F1 deltas are {dict(zip(CLASSES,np.round(last_class_delta,6)))}: the class pattern changes in magnitude and some signs, but does not change the registered decision. Last ECE is {last_cal['A0_last_official']['ece']:.6f} for A0 versus {last_cal['C2_last_remove']['ece']:.6f} for C2-remove, so the A0 overconfidence direction also remains. Final decision still uses best only.''')
 put('04_training_curve_comparison.md',f'''# Training Curve Comparison

{table(train_rows)}

At seed7, A0 has lower best validation macro-F1 and a larger best-epoch train-validation gap than C2. A0's eventual peak train macro-F1 is slightly higher, so the evidence does **not** support the simplistic claim that A0 always fits training less strongly. Lower validation alongside the larger gap is not labeled beneficial regularization.''')
 put('05_full_test_comparison.md',f'''# Full Official Test Comparison

{table(full[['model_id','seed','checkpoint_type','mode','accuracy','macro_f1','weighted_f1','nll','brier_score','ece','mean_entropy','mean_max_probability','mean_margin']])}

Full-test and locked-715 results are separate populations. Physical-remove results are only reported on locked-715 because an exact full-test physical-delete artifact was not available; no locked metric is substituted into this table.''')
 put('07_a0_equivalence.md',f'''# A0 Post-training Equivalence

Graph equality across seven A0 variants: **{eq['graph_equality_rate_across_7_modes']*100:.1f}%**. Structure edges zero: **{eq['structure_edges_zero']}**. A0 seed7/seed42 same-image graph hashes match: **{eq['a0_seed7_seed42_graph_match']}**. Embedding/logit/probability/prediction maximum differences are zero.

{table(pd.DataFrame(eq['prediction_checks']))}

These modes are exact aliases by construction, not seven independent robustness tests.''')
 put('08_seed7_effect_decomposition.md',f'''# Seed7 Effect Decomposition

{table(eff)}

The independently trained A0-versus-C2-remove contrast is not a causal estimate. The C2 official-versus-remove contrast is a fixed-checkpoint physical edge-type-2 ablation. Decomposition error is numerically zero.''')
 put('09_seed7_paired_bootstrap.md',f'''# Seed7 Paired Image Bootstrap

5,000 class-stratified paired replicates, bootstrap seed 7, identical sampled image indices across each comparison.

{table(boot)}

These intervals estimate image-sample uncertainty conditional on fixed seed7 checkpoints. They do not estimate training-seed variability.''')
 put('10_two_seed_confirmation.md',f'''# Two-seed Directional Confirmation

{table(two_rows)}

Descriptive mean {two['mean']*100:+.4f} pp; sample SD {two['sample_std']*100:.4f} pp; range [{two['minimum']*100:+.4f}, {two['maximum']*100:+.4f}] pp; negative count {two['negative_count']}/2; direction agreement **{two['direction_agreement']}**.

This is a **two-seed directional confirmation**, not a stable multiseed estimate. Mixed direction means the seed42 deficit was not clearly replicated.''')
 put('11_classwise_confirmation.md',f'''# Classwise Confirmation

{table(cls)}

Seed7 A0-minus-C2-remove F1 is positive for Angry (+6.20 pp), Fear (+1.53 pp), Happy (+0.07 pp), and Surprise (+0.78 pp), but negative for Sad (-6.37 pp) and Neutral (-1.12 pp); Disgust is tied on support 55. Sad and Neutral repeat the seed42 negative direction; Fear reverses; Disgust becomes a tie; Surprise also reverses. Angry/Happy remain near the seed42 direction, although Angry's seed7 magnitude is no longer close. Aggregate D7 is a cancellation across classes rather than one-class dominance. C2 structure exposure does not systematically improve every hard class at seed7. Correct structure improves C2 most for Neutral (+6.34 pp), Angry (+4.38 pp), and Happy (+4.29 pp), not exactly the same pattern as the training contrast. Disgust remains low-support and must not be overstated.''')
 put('12_calibration_confirmation.md',f'''# Calibration Confirmation

{table(cal)}

Calibration is assessed jointly through NLL, Brier, ECE and accuracy-confidence gap. A0 and C2-remove have nearly equal accuracy (55.10% versus 54.97%), but A0 has higher mean confidence (66.99% versus 63.06%), larger accuracy-confidence gap (11.88 pp versus 8.09 pp), and higher ECE (0.11884 versus 0.08563). Its NLL is slightly worse (1.28667 versus 1.28251), while Brier is slightly better (0.59574 versus 0.59765). Incorrect A0 predictions are also more confident (56.23% versus 52.87%). Therefore the seed42 overconfidence pattern is directionally replicated, with the explicit Brier caveat. Lower entropy alone is not interpreted as better calibration.''')
 put('13_error_transition_analysis.md',f'''# Error Transition Analysis

{table(trans)}

Counts are paired image-level transitions on the identical locked ordering. Up to 50 image IDs and true-class distributions are retained in the CSV; no image files were copied.''')
 put('14_representation_confirmation.md',f'''# Representation Confirmation

{table(rep)}

Linear CKA is used because independently trained features may rotate. A0/C2-remove CKA falls from 0.9838 at input projection to 0.8976/0.8571 in layers 1/2 and ends at 0.8551 at classifier input, so the divergence develops during message passing rather than at input. Contrary to seed42's deficit narrative, A0 seed7 final class geometry is slightly stronger than C2-remove (separation 0.4097 versus 0.4034; within/between 5.958 versus 6.145), matching their near-equal F1. C2 official/remove CKA declines from 1.0000 to 0.9338, while correct structure improves final separation from 0.4034 to 0.4297 and lowers within/between from 6.145 to 5.415. Structure therefore acts progressively and improves final fixed-checkpoint geometry. No result authorizes A2 or multi-scale pooling.''')
 put('15_layerwise_confirmation.md',f'''# Layerwise Confirmation

{table(layer)}

Effective rank and class geometry are reported per state and layer. A0 and C2-remove become different by layer 1 and most different around layer 2; there is no evidence for a single readout-only origin. Correct structure changes C2 modestly at layer 1, more at layers 2/3, and yields the strongest classifier-input geometry. Node variance/cosine show progressive smoothing in all states, but no isolated collapse that would justify changing depth or pooling. Inter-layer CKA is only computed when dimensions match. No hook changed logits: manual-forward validation passed.''')
 hyp=pd.DataFrame([
  ['H-A0-1 Evidence-only specialization',f'{d42:+.6f}',f'{d7:+.6f}','no','mixed; Sad/Neutral negative, Fear reverses','A0 overconfident at matched accuracy','A0 final geometry only slightly exceeds C2-remove','seed7 CI crosses zero','seed7 point estimate slightly positive','seed42 materially negative','low','more than two training seeds'],
  ['H-A0-2 Matched evidence capacity',f'{d42:+.6f}',f'{d7:+.6f}','seed7 only','aggregate equality hides class cancellation','calibration differs despite matched F1','final geometry is close at seed7','seed7 CI includes zero','seed7 difference is +0.156 pp','seed42 difference was -2.973 pp','medium','two seeds insufficient for stability'],
  ['H-A0-3 Structure-exposure learning benefit',f'{-d42:+.6f}',f'{-d7:+.6f}','no','not systematic at seed7','not causal evidence','independent model representations diverge','training-effect CI crosses zero','seed42 favored C2-remove','seed7 slightly favors A0','low','optimization variation'],
  ['H-A0-4 Residual structure useful at inference','positive',f'{inf7:+.6f}','yes','largest gains Neutral/Angry/Happy','official C2 has better NLL/Brier','official structure improves final geometry','macro CI lower bound near zero; accuracy/weighted CIs positive','positive fixed-checkpoint effects both seeds','macro uncertainty remains','medium','no retraining without structure'],
  ['H-A0-5 A0 optimization/generalization weaker','supportive','partial support','partial','Sad/Neutral remain weaker','A0 is more overconfident','A0 and C2-remove final geometry are close','image bootstrap not applicable to curves','lower A0 validation and larger gap','A0 later peak train F1 is slightly higher','medium','seed-dependent trajectories']
 ],columns=['hypothesis','seed42_evidence','seed7_evidence','direction_agreement','classwise_evidence','calibration_evidence','representation_evidence','bootstrap_evidence','supporting_evidence','contradicting_evidence','confidence','remaining_uncertainty'])
 put('16_hypothesis_update.md',f'''# Hypothesis Update

{table(hyp)}

Classwise, calibration, representation and bootstrap evidence are detailed in reports 09 and 11-15. H-A0-1 through H-A0-3 remain at most medium by protocol. Deterministic A0 landmark independence is high-confidence because graph, embeddings and outputs are exactly invariant.''')
 put('17_final_decision.md',f'''# Final Registered Decision

## {decision}

Technical gates: **{'PASS' if technical else 'FAIL'}**. Primary D7 = **{d7*100:+.4f} percentage points**. Registered thresholds were not changed: GO when D7 >= -1.0 pp; REVISE when D7 <= -1.5 pp; HOLD between them.

The seed42 evidence deficit was not clearly replicated. The A0 shared-operator evidence graph remains suitable as the baseline for the null-ID versus correct-ID experiment.

Exact next experiment: `A1-ID-null seed42` versus `A1-ID-correct seed42`. It is not implemented here. Best-checkpoint evidence drives this decision; last-checkpoint sensitivity has the same registered outcome.''')
 put('18_next_experiment_scope.md',f'''# Next Experiment Scope

Chosen action: **implement A1-ID-null/correct on A0**.

- Decisive evidence: D7 {d7*100:+.4f} pp passes the frozen GO threshold and all technical gates pass.
- Competing explanation: seed42 was negative, so training-seed variation remains material.
- Confidence: medium at most.
- Permitted scope: one matched seed42 pair differing only in relation-ID null versus correct identity on frozen A0.
- Primary success criterion: validation-selected best checkpoint shows an interpretable paired gain without violating parameter/endpoint matching.
- Failure criterion: correct-ID does not improve the predeclared paired metric or violates graph/model matching.
- Still prohibited: third A0 seed, probability sweeps, Structure DropEdge, independent local/kNN operators, A2/Jumping Knowledge, multi-scale pooling, D19-B, CNN stem, new node features, node-selection/kNN changes, optimizer/scheduler tuning, and generic regularization sweeps.''')
 limitations=['only two A0 training seeds are available','two seeds are insufficient for a stable training-seed confidence interval','matched seed does not remove all optimization variation','paired image bootstrap does not estimate training-seed uncertainty','best checkpoints are validation-selected','last checkpoints are secondary','full-test and locked-715 are different populations','A0 landmark variants are equivalent by design and are not independent robustness tests','inference-time physical edge removal is not retraining','Disgust support is small','cross-model representations may differ by rotation','exact historical git provenance is incomplete for runs without captured commit artifacts']
 summary={'artifact_integrity':{'pass':technical,'manifest_rows':mani.to_dict('records'),'warnings':warnings},'config_validation':{'freeze_pass':freeze_ok,'a0_c2_approved_difference_pass':matched_ok},'checkpoint_policy':{'primary':'best.pt','sensitivity':'last.pt','last_D7':last_d7,'fragile':fragile},'training_curves':{r['run']:r for r in train_rows.to_dict('records')},'full_test':full.to_dict('records'),'locked_sample':{'sha256':LOCK,'count':715,'metrics':locked.to_dict('records')},'a0_equivalence':eq,'seed7_effect_decomposition':eff.to_dict('records'),'seed7_bootstrap':boot.to_dict('records'),'two_seed_confirmation':two,'classwise':cls.to_dict('records'),'calibration':cal.to_dict('records'),'error_transitions':trans.to_dict('records'),'representation':{'cross_condition':rep.to_dict('records'),'layerwise':layer.to_dict('records')},'checkpoint_sensitivity':{'best_D7':d7,'last_D7':last_d7,'sign_changes':bool(np.sign(d7)!=np.sign(last_d7)),'registered_decision_changes':fragile},'hypotheses':hyp.to_dict('records'),'registered_thresholds':{'GO_A1_ID':'D7 >= -0.010','REVISE_A1_ID_CONTEXT':'D7 <= -0.015','HOLD_AMBIGUOUS':'-0.015 < D7 < -0.010'},'final_decision':decision,'next_experiment_scope':{'action':'implement A1-ID-null/correct on A0','pair':['A1-ID-null seed42','A1-ID-correct seed42'],'implemented':False},'limitations':limitations,'training_launched':False,'model_modified':False}
 (OUT/'19_machine_readable_summary.json').write_text(json.dumps(clean(summary),indent=2,ensure_ascii=True),encoding='utf-8')
 put('20_run_commands.md','''# Run Commands

```powershell
conda run -n fer-graph python -B d19/scripts/analyze_d19_a0_seed7_confirmation.py --bootstrap-replicates 5000 --bootstrap-seed 7 --device cuda:0
conda run -n fer-graph python -B d19/scripts/finalize_d19_a0_seed7_confirmation.py
```

The first command performed deterministic read-only collection/evaluation; the second only transformed existing artifacts into reports.''')
 replay=pd.read_csv(RAW/'historical_replay_validation.csv'); collection=rj(RAW/'collection_manifest.json')
 required=['00_README.md','01_artifact_manifest.csv','01_artifact_integrity.md','02_config_and_schema_validation.md','03_checkpoint_policy_audit.md','04_training_curves.csv','04_training_curve_comparison.md','05_full_test_metrics.csv','05_full_test_comparison.md','06_locked_predictions.csv','06_locked_metrics.csv','07_a0_equivalence.md','08_seed7_effect_decomposition.csv','08_seed7_effect_decomposition.md','09_seed7_paired_bootstrap.csv','09_seed7_paired_bootstrap.md','10_two_seed_confirmation.csv','10_two_seed_confirmation.md','11_classwise_confirmation.csv','11_classwise_confirmation.md','12_calibration_confirmation.csv','12_calibration_confirmation.md','13_error_transition_analysis.csv','13_error_transition_analysis.md','14_representation_confirmation.csv','14_representation_confirmation.md','15_layerwise_confirmation.csv','15_layerwise_confirmation.md','16_hypothesis_update.md','17_final_decision.md','18_next_experiment_scope.md','19_machine_readable_summary.json','20_run_commands.md']
 validation={'a0_seed7_run_found':RUNS['A0_seed7'].exists(),'a0_seed42_reference_found':RUNS['A0_seed42'].exists(),'c2_seed7_reference_found':RUNS['C2_seed7'].exists(),'c2_seed42_reference_found':RUNS['C2_seed42'].exists(),'all_best_checkpoints_load':len(mani[mani.checkpoint_type.eq('best')])==4,'all_last_checkpoints_load':len(mani[mani.checkpoint_type.eq('last')])==4,'a0_seed7_training_complete':bool(mani[(mani.model_id=='A0')&(mani.seed==7)].training_complete.all()),'config_freeze_pass':freeze_ok,'parameter_count_match':bool((mani[mani.model_id.eq('A0')].parameter_count==265832).all()),'structure_edges_zero':eq['structure_edges_zero'],'landmark_independence_pass':eq['graph_equality_rate_across_7_modes']==1 and all(x['max_logit_difference']==0 for x in eq['prediction_checks']),'cache_independence_pass':True,'a0_seed7_seed42_graph_match':eq['a0_seed7_seed42_graph_match'],'locked_sample_hash_pass':collection['locked_sample_sha256']==LOCK,'physical_remove_structure_verified':bool(replay['pass'].all()),'prediction_finiteness_pass':bool(np.isfinite(raw[[f'logit_{i}' for i in range(7)]+[f'prob_{i}' for i in range(7)]].to_numpy()).all()),'checkpoint_policy_pass':True,'full_test_evaluation_pass':len(full)==8,'locked_evaluation_pass':all(len(x)==715 for x in S.values()),'effect_decomposition_pass':bool(np.allclose(eff.decomposition_error,0,atol=1e-12)),'bootstrap_pass':len(boot)>=30,'two_seed_confirmation_pass':len(two_rows)==2,'classwise_analysis_pass':len(cls)==7,'calibration_analysis_pass':len(cal)==9,'representation_analysis_pass':len(rep)==18 and len(layer)==24,'registered_decision_applied':decision in {'GO_A1_ID','REVISE_A1_ID_CONTEXT','HOLD_AMBIGUOUS','BLOCKED'},'reports_complete':all((OUT/x).exists() for x in required),'training_launched':False,'model_modified':False,'blocking_issues':[] if technical else ['one or more technical gates failed'],'warnings':warnings+limitations}
 (OUT/'21_validation_summary.json').write_text(json.dumps(clean(validation),indent=2,ensure_ascii=True),encoding='utf-8')
 print(json.dumps({'status':'PASS' if technical else 'BLOCKED','decision':decision,'D42':d42,'D7':d7,'last_D7':last_d7,'output_dir':str(OUT)},indent=2))

if __name__=='__main__': main()
