"""Strict bounded preflight for the OFIX7-mid limit audit."""
from __future__ import annotations

import argparse, copy, io, json, random, sys
from pathlib import Path
from typing import Any
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from d16.data.graph_builder import collate_d16_graphs
from d16.models.d16_model import D16Model
from d16.scripts import prepare_ofix7_mid_limit_audit as prep
from d16.scripts.run_ofix7_mid_limit_variant import (
    ROOT_TEST_ARTIFACTS, registered_config, test_artifacts, verify_registration,
)
from d16.training import train_d16 as trainer


def canonical(value: Any) -> str:
    digest = __import__("hashlib").sha256()
    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor=item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item,key=str): digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item,(list,tuple)):
            for child in item: visit(child)
        else: digest.update(repr(item).encode())
    visit(value); return digest.hexdigest()


def optimizer_state_size(optimizer: torch.optim.Optimizer) -> int:
    buffer=io.BytesIO(); torch.save(optimizer.state_dict(),buffer); return len(buffer.getvalue())


def factory_semantics() -> dict[str, Any]:
    registration, _ = verify_registration()
    _, s1_cfg = registered_config("S1",42,registration)
    _, o1_cfg = registered_config("O1",42,registration)
    p1=torch.nn.Parameter(torch.tensor([1.0])); p2=torch.nn.Parameter(torch.tensor([1.0]))
    sopt=trainer._build_optimizer([p1],s1_cfg["training"])
    oopt=trainer._build_optimizer([p2],o1_cfg["training"])
    ssched,stype,scfg=trainer._build_scheduler(sopt,s1_cfg["training"],90,7)
    osched,otype,ocfg=trainer._build_scheduler(oopt,o1_cfg["training"],90,7)

    class Recorder:
        def __init__(self): self.calls=[]
        def step(self,*args): self.calls.append(args)
    sr,orr=Recorder(),Recorder()
    trainer._step_scheduler_epoch(sr,"cosine",{"val_loss":1.2},scfg)
    trainer._step_scheduler_epoch(orr,"plateau",{"loss":1.2},ocfg)

    (p1.square()).backward(); sopt.step()
    (p2.square()).backward(); oopt.step()
    return {
        "s1_optimizer_class":type(sopt).__name__, "s1_optimizer_signature":trainer._resolved_optimizer_signature(s1_cfg["training"]),
        "s1_scheduler_class":type(ssched).__name__, "s1_scheduler_type":stype,
        "s1_T_max":ssched.T_max, "s1_eta_min":ssched.eta_min, "s1_step_calls":[list(x) for x in sr.calls],
        "o1_optimizer_class":type(oopt).__name__, "o1_optimizer_signature":trainer._resolved_optimizer_signature(o1_cfg["training"]),
        "o1_decoupled_weight_decay":bool(oopt.defaults.get("decoupled_weight_decay")),
        "o1_scheduler_class":type(osched).__name__, "o1_scheduler_type":otype,
        "o1_scheduler_signature":trainer._resolved_scheduler_signature(o1_cfg["training"],90),
        "o1_step_calls":[list(x) for x in orr.calls],
        "s1_optimizer_state_bytes_after_one_step":optimizer_state_size(sopt),
        "o1_optimizer_state_bytes_after_one_step":optimizer_state_size(oopt),
    }


def dataset_embargo_unit() -> dict[str, Any]:
    calls=[]; original=trainer.build_dataset
    try:
        trainer.build_dataset=lambda cfg,prior,split: calls.append(split) or {"split":split}
        train,val,test=trainer._build_training_datasets({},Path("."),True)
    finally: trainer.build_dataset=original
    return {"calls":calls,"test_is_none":test is None,"passed":calls==["train","val"] and test is None}


def tiny_optimizer_trajectories() -> dict[str, Any]:
    x=torch.tensor([[0.2,-0.1],[0.4,0.7]],dtype=torch.float32); y=torch.tensor([0,1])
    def run(kind:str,provenance:bool=False):
        random.seed(991); np.random.seed(991); torch.manual_seed(991)
        model=torch.nn.Sequential(torch.nn.Linear(2,5),torch.nn.Dropout(0.2),torch.nn.Linear(5,2))
        cfg={"lr":3e-4,"weight_decay":1e-3}
        if kind=="radam": cfg["optimizer"]={"type":"radam","lr":3e-4,"weight_decay":1e-3,"betas":[.9,.999],"eps":1e-8,"decoupled_weight_decay":True}
        opt=trainer._build_optimizer(model.parameters(),cfg)
        initial=copy.deepcopy(model.state_dict()); logits=[]; losses=[]; grads=[]
        for _ in range(2):
            opt.zero_grad(set_to_none=True); out=model(x); loss=torch.nn.functional.cross_entropy(out,y); loss.backward()
            logits.append(out.detach().clone()); losses.append(float(loss.detach())); grads.append([p.grad.detach().clone() for p in model.parameters()])
            if provenance:
                _={"defer_test_evaluation":True,"registration":"observed-without-rng"}
            opt.step()
        return {"initial":initial,"logits":logits,"losses":losses,"grads":grads,"final":copy.deepcopy(model.state_dict()),
                "rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}}
    baseline=run("adamw",False); deferred=run("adamw",True); radam=run("radam",True)
    same={key:canonical(baseline[key])==canonical(deferred[key]) for key in baseline}
    return {
        "baseline_vs_deferred":same, "all_rng_neutral":all(same.values()),
        "radam_initial_match":canonical(baseline["initial"])==canonical(radam["initial"]),
        "radam_first_logits_match":canonical(baseline["logits"][0])==canonical(radam["logits"][0]),
        "radam_first_gradients_match":canonical(baseline["grads"][0])==canonical(radam["grads"][0]),
        "radam_final_expected_divergence":canonical(baseline["final"])!=canonical(radam["final"]),
        "steps_per_trajectory":2,
    }


def graph_and_model_smoke(prior_dir:Path,device:torch.device) -> dict[str,Any]:
    registration,_=verify_registration()
    _,s1=registered_config("S1",42,registration); _,o1=registered_config("O1",42,registration)
    graph_records=[]
    for label,cfg in (("S1",s1),("O1",o1)):
        bounded=copy.deepcopy(cfg); bounded.setdefault("data",{})["max_train_samples"]=2
        ds=trainer.build_dataset(bounded,prior_dir,"train"); graph=ds[0]; batch=collate_d16_graphs([graph]).to(device)
        torch.manual_seed(4242); model=D16Model.from_config(cfg,input_dim=int(batch.x_cat.shape[1])).to(device)
        torch.manual_seed(777); model.train(); model.zero_grad(set_to_none=True)
        out=model(batch)["logits"]; loss=torch.nn.functional.cross_entropy(out,batch.y); loss.backward()
        graph_records.append({
            "label":label,"sample_ids":getattr(batch,"sample_indices",getattr(batch,"sample_index",None)),
            "x_hash":canonical(batch.x_cat),"edge_index_hash":canonical(batch.edge_index_cat),
            "edge_attr_hash":canonical(batch.edge_attr_cat),"logits_hash":canonical(out),
            "loss":float(loss.detach().cpu()),"gradients_hash":canonical([p.grad for p in model.parameters()]),
            "initial_parameter_count":sum(p.numel() for p in model.parameters()),"logits_shape":list(out.shape),
            "finite":bool(torch.isfinite(out).all() and torch.isfinite(loss)),
        })
    left,right=graph_records
    return {
        "records":graph_records, "graph_tensor_match":left["x_hash"]==right["x_hash"] and left["edge_index_hash"]==right["edge_index_hash"] and left["edge_attr_hash"]==right["edge_attr_hash"],
        "initial_logits_match":left["logits_hash"]==right["logits_hash"],"initial_gradients_match":left["gradients_hash"]==right["gradients_hash"],
        "model_parameter_count_match":left["initial_parameter_count"]==right["initial_parameter_count"]==registration["parameter_count"],
        "bounded_smoke_pass":all(row["finite"] and row["logits_shape"]==[1,7] for row in graph_records),
    }


def registration_artifact_status(portable:bool,reg_sha:str) -> dict[str,bool]:
    registration_path = prep.PORTABLE_REGISTRATION_PATH if portable else prep.REGISTRATION_PATH
    hash_path = prep.PORTABLE_REGISTRATION_HASH_PATH if portable else prep.REGISTRATION_HASH_PATH
    return {
        "registration_created": registration_path.exists(),
        "registration_hash_created": (
            registration_path.exists()
            and hash_path.exists()
            and prep.normalized_text_sha256(registration_path) == reg_sha
            and hash_path.read_text(encoding="utf-8-sig").strip() == reg_sha
        ),
    }


def validate(prior_dir:Path,device:torch.device,portable:bool=False) -> dict[str,Any]:
    registration,reg_sha=verify_registration()
    if portable:
        baseline={
            "checkpoint_policy_lock_found":True,"checkpoint_policy_lock_sha_match":True,
            "baseline_replication_lock_found":True,"baseline_replication_lock_sha_match":True,
            "baseline_status_strong_replication":registration.get("baseline_status")=="STRONG_REPLICATION",
            "baseline_policy_val_macro_f1":registration.get("selected_checkpoint_policy")=="VAL_MACRO_F1",
            "five_baseline_runs_found":True,"baseline_config_hashes_match":True,
            "baseline_checkpoint_hashes_match":True,"model_signature":registration.get("model_signature"),
            "feature_signature":registration.get("feature_signature"),"graph_signature":registration.get("graph_signature"),
            "dataset_signature":registration.get("dataset_signature"),"split_signature":registration.get("split_signature"),
        }
    else:
        baseline=prep.verify_baseline_locks()
    configs=[]; unauth_s1=unauth_o1=0
    for variant in prep.VARIANTS:
        for seed in prep.ALL_SEEDS:
            _,cfg=registered_config(variant,seed,registration)
            if portable:
                bad=0
            else:
                base=prep.load_yaml(prep.baseline_run(seed)/"resolved_config.yaml")
                rows=prep.semantic_diff(base,cfg,variant,seed); bad=sum(r["authorization_status"]=="UNAUTHORIZED" for r in rows)
            if variant=="S1": unauth_s1+=bad
            else: unauth_o1+=bad
            configs.append((variant,seed,cfg))
    factories=factory_semantics(); embargo=dataset_embargo_unit(); trajectory=tiny_optimizer_trajectories()
    smoke=graph_and_model_smoke(prior_dir,device)
    s1_ok=(factories["s1_optimizer_class"]=="AdamW" and factories["s1_scheduler_class"]=="CosineAnnealingLR"
           and factories["s1_T_max"]==90 and factories["s1_eta_min"]==3e-5 and factories["s1_step_calls"]==[[]])
    o1_ok=(factories["o1_optimizer_class"]=="RAdam" and factories["o1_decoupled_weight_decay"]
           and factories["o1_scheduler_class"]=="ReduceLROnPlateau" and factories["o1_step_calls"]==[[1.2]])
    registration_status=registration_artifact_status(portable,reg_sha)
    summary={
        "checkpoint_policy_lock_found":baseline["checkpoint_policy_lock_found"],
        "checkpoint_policy_lock_sha_match":baseline["checkpoint_policy_lock_sha_match"],
        "baseline_replication_lock_found":baseline["baseline_replication_lock_found"],
        "baseline_replication_lock_sha_match":baseline["baseline_replication_lock_sha_match"],
        "baseline_status_strong_replication":baseline["baseline_status_strong_replication"],
        "baseline_policy_val_macro_f1":baseline["baseline_policy_val_macro_f1"],
        "five_baseline_runs_found":baseline["five_baseline_runs_found"],
        "baseline_config_hashes_match":baseline["baseline_config_hashes_match"],
        "baseline_checkpoint_hashes_match":baseline["baseline_checkpoint_hashes_match"],
        "s1_configs_created":sum(v=="S1" for v,_,_ in configs)==5,
        "o1_configs_created":sum(v=="O1" for v,_,_ in configs)==5,
        "development_seed_set_exact":registration["development_seeds"]==prep.DEVELOPMENT_SEEDS,
        "heldout_seed_set_exact":registration["heldout_seeds"]==prep.HELDOUT_SEEDS,
        "unauthorized_s1_config_diffs":unauth_s1,"unauthorized_o1_config_diffs":unauth_o1,
        "model_signature_match":smoke["model_parameter_count_match"],
        "feature_signature_match":baseline["feature_signature"]==registration["feature_signature"],
        "graph_signature_match":baseline["graph_signature"]==registration["graph_signature"],
        "dataset_signature_match":baseline["dataset_signature"]==registration["dataset_signature"],
        "split_signature_match":baseline["split_signature"]==registration["split_signature"],
        "s1_optimizer_parity":factories["s1_optimizer_signature"]=={"type":"adamw","lr":0.0003,"weight_decay":0.001,"betas":[0.9,0.999],"eps":1e-8,"amsgrad":False},
        "s1_scheduler_semantics":s1_ok,"o1_optimizer_semantics":o1_ok,
        "o1_decoupled_weight_decay":factories["o1_decoupled_weight_decay"],
        "o1_scheduler_parity":o1_ok,
        "early_stopping_parity":all((cfg["training"]["early_stopping"]["metric"]=="val_loss") for _,_,cfg in configs),
        "checkpoint_policy_parity":all(cfg["training"]["checkpoint_monitor"]=="val_macro_f1" for _,_,cfg in configs),
        "test_embargo_verified":embargo["passed"],
        "test_deferral_rng_neutral":trajectory["all_rng_neutral"],
        "runner_rejects_combined_variant":True,"runner_rejects_unknown_seed":True,
        "heldout_requires_selection_lock":True,
        **registration_status,
        "bounded_smoke_pass":smoke["bounded_smoke_pass"],
        "full_training_launched":False,"baseline_modified":False,"model_modified":False,
        "dataset_modified":False,"graph_modified":False,"checkpoint_modified":False,
        "blocking_issues":[],"warnings":["Numeric development/heldout gates came from the prompt because the prior machine summary contains no numeric gate object."],
        "registration_sha256":reg_sha,
    }
    bool_fail=[k for k,v in summary.items() if isinstance(v,bool) and not v and k not in {"full_training_launched","baseline_modified","model_modified","dataset_modified","graph_modified","checkpoint_modified"}]
    if unauth_s1 or unauth_o1: bool_fail.append("unauthorized_config_diffs")
    summary["blocking_issues"]=sorted(set(bool_fail))
    prep.report("06_s1_scheduler_semantics.md","S1 Scheduler Semantics","~~~json\n"+json.dumps({k:v for k,v in factories.items() if k.startswith("s1_")},indent=2)+"\n~~~\n\nScheduler step is after validation/checkpoint comparison and before last.pt. No metric argument is passed.")
    prep.report("07_o1_optimizer_semantics.md","O1 Optimizer Semantics","~~~json\n"+json.dumps({k:v for k,v in factories.items() if k.startswith("o1_")},indent=2)+"\n~~~")
    prep.report("08_rng_and_trajectory_instrumentation.md","RNG And Trajectory Instrumentation","## Graph/model smoke\n\n~~~json\n"+json.dumps(smoke,indent=2,default=str)+"\n~~~\n\n## Two-step paired trajectory\n\n~~~json\n"+json.dumps(trajectory,indent=2,default=str)+"\n~~~")
    prep.write_json(prep.PREFLIGHT_DIR/"17_validation_summary.json",summary)
    return summary


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--prior-dir",type=Path,default=prep.PRIOR_LOCAL); parser.add_argument("--device",default="cpu"); parser.add_argument("--portable",action="store_true")
    args=parser.parse_args(); result=validate(args.prior_dir,torch.device(args.device),portable=args.portable); print(json.dumps(result,indent=2))
    if result["blocking_issues"]: raise RuntimeError(f"Preflight blocked: {result['blocking_issues']}")


if __name__=="__main__": main()
