"""Prepare the preregistered OFIX7-mid bounded optimizer/scheduler audit.

This is Stage P only. It validates locked baseline evidence, derives exactly ten
configs, writes semantic diffs and an immutable registration, and never trains.
"""
from __future__ import annotations

import argparse, copy, csv, hashlib, inspect, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import torch, yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
BASE = ROOT / "outputs/d16_analysis/ofix7_mid_5seed_posttraining_analysis"
POLICY_LOCK = BASE / "10_checkpoint_policy_lock.json"
BASELINE_LOCK = BASE / "23_baseline_replication_lock.json"
POLICY_SHA = "dfce606a69343b1a8de821ec3fc547d5700b94ff6c9a15f6b26d32e02601fc5f"
BASELINE_SHA = "d54c9162de7e4f6bda2ee37dbe735939f7542195d9d2fe6dbd5e61cd85351dc3"
CONFIG_DIR = ROOT / "configs/d16/final_optimization"
PREFLIGHT_DIR = ROOT / "outputs/d16_analysis/ofix7_mid_limit_audit_preflight"
RUN_ROOT = ROOT / "outputs/d16_runs/final_optimization"
POSTTRAIN_DIR = ROOT / "outputs/d16_analysis/ofix7_mid_limit_audit"
REGISTRATION_PATH = PREFLIGHT_DIR / "14_limit_audit_registration.json"
REGISTRATION_HASH_PATH = PREFLIGHT_DIR / "14_limit_audit_registration.sha256"
PORTABLE_REGISTRATION_PATH = CONFIG_DIR / "limit_audit_registration.json"
PORTABLE_REGISTRATION_HASH_PATH = CONFIG_DIR / "limit_audit_registration.sha256"
PORTABLE_POLICY_LOCK = CONFIG_DIR / "baseline_checkpoint_policy_lock.json"
PORTABLE_BASELINE_LOCK = CONFIG_DIR / "baseline_replication_lock.json"
ALL_SEEDS = [42, 1009, 1337, 777, 3407]
DEVELOPMENT_SEEDS = [42, 1009, 1337]
HELDOUT_SEEDS = [777, 3407]
VARIANTS = ("S1", "O1")
PRIOR_LOCAL = ROOT / "outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
PRIOR_KAGGLE = "/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
DEVELOPMENT_GATE = {
    "mean_validation_accuracy_gain_pp_min": 1.0,
    "mean_validation_macro_f1_gain_pp_min": 0.5,
    "mean_train_validation_macro_f1_gap_increase_pp_max": 2.0,
    "max_per_seed_validation_accuracy_loss_pp": 1.0,
    "max_mean_per_class_validation_f1_loss_pp": 3.0,
    "positive_validation_accuracy_direction_seed_count_min": 2,
}
HELDOUT_GATE = {
    "mean_validation_accuracy_gain_pp_min": 0.75,
    "mean_validation_macro_f1_gain_pp_min": 0.25,
    "heldout_direction_rule": "both_positive_or_combined_positive_with_each_loss_le_0.50pp",
    "mean_train_validation_macro_f1_gap_increase_pp_max": 2.0,
    "validation_accuracy_sample_sd_increase_pp_max": 0.5,
}
TIE_BREAKING = [
    "larger_mean_validation_accuracy_gain",
    "larger_mean_validation_macro_f1_gain",
    "smaller_train_validation_macro_f1_gap",
    "smaller_validation_accuracy_sample_sd",
    "S1_before_O1",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def relative(path: Path) -> str:
    try: return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError: return str(path)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict): return {prefix: value}
    out: dict[str, Any] = {}
    for key in sorted(value):
        out.update(flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
    return out


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def table(rows: list[dict[str, Any]]) -> str:
    if not rows: return "_No rows._"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines)


def report(name: str, title: str, body: str) -> None:
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    (PREFLIGHT_DIR / name).write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def baseline_run(seed: int) -> Path: return ROOT / f"outputs/d16_runs/final/ofix7_mid_seed{seed}"
def source_config(seed: int) -> Path: return ROOT / f"configs/d16/final_replication/ofix7_mid_seed{seed}.yaml"
def run_name(variant: str, seed: int) -> str:
    return f"ofix7_mid_{'s1_cosine' if variant == 'S1' else 'o1_radam'}_seed{seed}"
def config_path(variant: str, seed: int) -> Path: return CONFIG_DIR / f"{run_name(variant, seed)}.yaml"


def normalized_baseline(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    for key in ("run_name", "description", "limit_audit_provenance"): out.pop(key, None)
    out["seed"] = "<RUN_SEED>"; training = out.setdefault("training", {})
    training["seed"] = "<RUN_SEED>"; training.pop("defer_test_evaluation", None)
    prior = out.setdefault("graph", {}).get("prior_corruption", {}) or {}
    if "seed" in prior: prior["seed"] = "<RUN_SEED_PLUS_7699>"
    wandb = (out.get("logging", {}) or {}).get("wandb", {}) or {}
    for key in ("name", "group", "tags", "notes"): wandb.pop(key, None)
    return out


def verify_baseline_locks() -> dict[str, Any]:
    if not POLICY_LOCK.exists() or not BASELINE_LOCK.exists(): raise RuntimeError("Missing baseline locks")
    policy_sha, baseline_sha = sha256_file(POLICY_LOCK), sha256_file(BASELINE_LOCK)
    if policy_sha != POLICY_SHA or baseline_sha != BASELINE_SHA:
        raise RuntimeError(f"Baseline lock hash mismatch: {policy_sha}, {baseline_sha}")
    policy, lock = load_json(POLICY_LOCK), load_json(BASELINE_LOCK)
    if policy.get("lock_state") != "LOCK_VAL_MACRO_F1" or lock.get("selected_policy") != "VAL_MACRO_F1":
        raise RuntimeError("Baseline checkpoint policy is not VAL_MACRO_F1")
    if lock.get("replication_status") != "STRONG_REPLICATION" or lock.get("seeds") != ALL_SEEDS:
        raise RuntimeError("Baseline status or seed set mismatch")

    from d16.training.train_d16 import canonical_model_state_hash
    config_ok, checkpoint_ok, canonical_ok = {}, {}, {}
    scientific, runtimes, commits = [], [], []
    for seed in ALL_SEEDS:
        run = baseline_run(seed)
        needed = [run / "resolved_config.yaml", run / "feature_schema.json",
                  run / "replication_provenance/source_hashes.json",
                  run / "replication_provenance/runtime_signatures.json", source_config(seed)]
        missing = [str(path) for path in needed if not path.exists()]
        if missing: raise FileNotFoundError(missing)
        config_ok[str(seed)] = normalized_text_sha256(source_config(seed)) == lock["config_hashes"][str(seed)]
        cp_meta = lock["selected_checkpoints"][str(seed)]; cp = ROOT / cp_meta["path"]
        checkpoint_ok[str(seed)] = cp.exists() and sha256_file(cp) == cp_meta["file_sha256"]
        canonical_ok[str(seed)] = checkpoint_ok[str(seed)] and canonical_model_state_hash(
            torch.load(cp, map_location="cpu", weights_only=False)
        ) == cp_meta["canonical_model_state_sha256"]
        scientific.append(json_hash(normalized_baseline(load_yaml(run / "resolved_config.yaml"))))
        source = load_json(run / "replication_provenance/source_hashes.json")
        runtimes.append(load_json(run / "replication_provenance/runtime_signatures.json"))
        commits.append(str(source.get("repository_commit")))
    if not all(config_ok.values()) or not all(checkpoint_ok.values()) or not all(canonical_ok.values()):
        raise RuntimeError("Baseline config/checkpoint hashes differ from lock")
    if len(set(scientific)) != 1 or len(set(commits)) != 1: raise RuntimeError("Five baseline scientific signatures differ")
    signatures = {}
    for key in ("model_signature", "feature_signature", "graph_signature", "dataset_signature", "split_signature"):
        values = [json.dumps(row.get(key), sort_keys=True) for row in runtimes]
        if len(set(values)) != 1: raise RuntimeError(f"Baseline {key} differs across seeds")
        signatures[key] = runtimes[0].get(key)
    return {
        "checkpoint_policy_lock_found": True, "checkpoint_policy_lock_sha_match": True,
        "checkpoint_policy_lock_sha": policy_sha, "baseline_replication_lock_found": True,
        "baseline_replication_lock_sha_match": True, "baseline_replication_lock_sha": baseline_sha,
        "baseline_status_strong_replication": True, "baseline_policy_val_macro_f1": True,
        "five_baseline_runs_found": True, "baseline_config_hashes_match": True,
        "baseline_checkpoint_hashes_match": True, "baseline_checkpoint_canonical_hashes_match": True,
        "five_baseline_scientific_signatures_match": True, "baseline_commit": commits[0],
        "per_seed_config_hash_checks": config_ok, "per_seed_checkpoint_hash_checks": checkpoint_ok,
        "per_seed_canonical_hash_checks": canonical_ok, "baseline_lock": lock, **signatures,
    }


def make_variant_config(base: dict[str, Any], variant: str, seed: int) -> dict[str, Any]:
    if variant not in VARIANTS or seed not in ALL_SEEDS: raise ValueError("Unregistered variant or seed")
    cfg = copy.deepcopy(base); cfg["run_name"] = run_name(variant, seed)
    cfg["description"] = "Bounded OFIX7-mid " + ("AdamW/Cosine S1" if variant == "S1" else "RAdam/Plateau O1")
    cfg["limit_audit_provenance"] = {
        "variant": variant, "baseline_run": relative(baseline_run(seed)), "baseline_seed": seed,
        "stage": "development" if seed in DEVELOPMENT_SEEDS else "heldout", "test_embargo": True,
    }
    training = cfg.setdefault("training", {}); training["defer_test_evaluation"] = True
    if variant == "S1":
        minimum = float((training.get("scheduler") or {})["min_lr"])
        training["scheduler"] = {"type": "cosine", "t_max": int(training["max_epochs"]), "eta_min": minimum}
        training.pop("optimizer", None)
    else:
        training["optimizer"] = {
            "type": "radam", "lr": float(training["lr"]), "weight_decay": float(training["weight_decay"]),
            "betas": [0.9, 0.999], "eps": 1e-8, "decoupled_weight_decay": True,
        }
    return cfg


def classify(field: str, variant: str) -> tuple[str, bool]:
    if field in {"run_name", "description", "training.defer_test_evaluation"} or field.startswith("limit_audit_provenance."):
        return "NON_SCIENTIFIC", True
    if variant == "S1" and field.startswith("training.scheduler."): return "AUTHORIZED_SCHEDULER_CHANGE", True
    if variant == "O1" and field.startswith("training.optimizer."): return "AUTHORIZED_OPTIMIZER_CHANGE", True
    if field.startswith("data.") and field.endswith(("prior_dir", "graph_cache_dir", "graph_cache_dir_detected", "graph_cache_dir_fallback")):
        return "PATH_ONLY", True
    return "UNAUTHORIZED_SCIENTIFIC_CHANGE", False


def semantic_diff(base: dict[str, Any], cfg: dict[str, Any], variant: str, seed: int) -> list[dict[str, Any]]:
    left, right, rows = flatten(base), flatten(cfg), []
    for field in sorted(set(left) | set(right)):
        if left.get(field) == right.get(field): continue
        kind, ok = classify(field, variant)
        rows.append({
            "variant": variant, "seed": seed, "field": field,
            "baseline_value": json.dumps(left.get(field), default=str),
            "variant_value": json.dumps(right.get(field), default=str),
            "classification": kind, "authorization_status": "AUTHORIZED" if ok else "UNAUTHORIZED",
        })
    return rows


def definitions() -> tuple[dict[str, Any], dict[str, Any]]:
    s = load_yaml(config_path("S1", 42))["training"]; o = load_yaml(config_path("O1", 42))["training"]
    s1 = {
        "variant": "S1", "optimizer": {"class": "AdamW", "lr": s["lr"], "weight_decay": s["weight_decay"],
        "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False},
        "scheduler": {"class": "CosineAnnealingLR", "T_max": s["scheduler"]["t_max"],
        "eta_min": s["scheduler"]["eta_min"], "last_epoch": -1, "step_once_per_completed_epoch": True,
        "step_argument": None, "step_position": "after validation/checkpoint comparison, before last.pt save"},
    }
    o1 = {
        "variant": "O1", "optimizer": {"class": "RAdam", **{k:v for k,v in o["optimizer"].items() if k != "type"}},
        "scheduler": {"class": "ReduceLROnPlateau", **{k:v for k,v in o["scheduler"].items() if k != "type"},
        "cooldown": int(o["scheduler"].get("cooldown", 0)),
        "step_position": "after validation/checkpoint comparison, before last.pt save"},
    }
    return s1, o1


def commands(kind: str) -> list[str]:
    windows = kind == "powershell"; py = "& 'C:\\Users\\ADMIN\\anaconda3\\envs\\fer-graph\\python.exe'" if windows else "python"
    prior = str(PRIOR_LOCAL) if windows else PRIOR_KAGGLE
    runs = str(RUN_ROOT) if windows else "/kaggle/working/outputs/d16_runs/final_optimization"
    pre = str(PREFLIGHT_DIR) if windows else "configs/d16/final_optimization"
    post = str(POSTTRAIN_DIR) if windows else "/kaggle/working/outputs/d16_analysis/ofix7_mid_limit_audit"
    out = [
        f"{py} -B d16/scripts/prepare_ofix7_mid_limit_audit.py" + ("" if windows else " --verify-portable"),
        f"{py} -B d16/scripts/validate_ofix7_mid_limit_audit.py --prior-dir '{prior}' --device cpu" + ("" if windows else " --portable"),
    ]
    for variant in VARIANTS:
        for seed in DEVELOPMENT_SEEDS:
            out.append(f"{py} -B d16/scripts/run_ofix7_mid_limit_variant.py --variant {variant} --seed {seed} --data-root '{prior}' --output-root '{runs}' --device cuda:0 --num-workers 2 --no-resume")
    out.append(f"{py} -B d16/scripts/analyze_ofix7_mid_limit_audit.py --stage development-selection --registration '{pre}/14_limit_audit_registration.json' --run-root '{runs}' --output-dir '{post}'")
    out.append("$WINNER = '<S1_OR_O1_FROM_DEVELOPMENT_LOCK>'" if windows else "WINNER='<S1_OR_O1_FROM_DEVELOPMENT_LOCK>'")
    for seed in HELDOUT_SEEDS:
        out.append(f"{py} -B d16/scripts/run_ofix7_mid_limit_variant.py --variant $WINNER --seed {seed} --data-root '{prior}' --output-root '{runs}' --device cuda:0 --num-workers 2 --no-resume --development-selection-lock '{post}/development_variant_selection_lock.json'")
    out.extend([
        f"{py} -B d16/scripts/analyze_ofix7_mid_limit_audit.py --stage heldout-confirmation --registration '{pre}/14_limit_audit_registration.json' --run-root '{runs}' --output-dir '{post}' --development-selection-lock '{post}/development_variant_selection_lock.json'",
        f"{py} -B d16/scripts/analyze_ofix7_mid_limit_audit.py --stage verify-final-lock --registration '{pre}/14_limit_audit_registration.json' --output-dir '{post}' --final-promotion-lock '{post}/final_variant_promotion_lock.json'",
        f"{py} -B d16/scripts/analyze_ofix7_mid_limit_audit.py --stage test-reveal --registration '{pre}/14_limit_audit_registration.json' --run-root '{runs}' --output-dir '{post}' --prior-dir '{prior}' --device cuda:0 --final-promotion-lock '{post}/final_variant_promotion_lock.json'",
        f"{py} -B d16/scripts/analyze_ofix7_mid_limit_audit.py --stage final-summary --registration '{pre}/14_limit_audit_registration.json' --run-root '{runs}' --output-dir '{post}' --final-promotion-lock '{post}/final_variant_promotion_lock.json'",
    ])
    return out


def prepare() -> dict[str, Any]:
    checks = verify_baseline_locks()
    if RUN_ROOT.exists() and any(path.is_dir() and path.name.startswith("ofix7_mid_") for path in RUN_ROOT.iterdir()):
        raise RuntimeError("Variant runs already exist; registration rewrite refused")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    expected = {config_path(v,s).name for v in VARIANTS for s in ALL_SEEDS}
    extra = [path.name for path in CONFIG_DIR.glob("*.yaml") if path.name not in expected]
    if extra: raise RuntimeError(f"Exact-ten registration blocked by extra YAMLs: {extra}")
    paths, diffs, manifest = [], [], []
    for variant in VARIANTS:
        for seed in ALL_SEEDS:
            base = load_yaml(baseline_run(seed) / "resolved_config.yaml")
            cfg, path = make_variant_config(base, variant, seed), config_path(variant, seed)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            paths.append(path); rows = semantic_diff(base, cfg, variant, seed); diffs.extend(rows)
            manifest.append({"variant":variant, "seed":seed, "stage":"development" if seed in DEVELOPMENT_SEEDS else "heldout",
                             "config_path":relative(path), "run_name":cfg["run_name"], "file_sha256":normalized_text_sha256(path),
                             "unauthorized_scientific_diffs":sum(r["authorization_status"]=="UNAUTHORIZED" for r in rows)})
    unauthorized = [row for row in diffs if row["authorization_status"] == "UNAUTHORIZED"]
    if len(paths) != 10 or len(list(CONFIG_DIR.glob("*.yaml"))) != 10 or unauthorized:
        raise RuntimeError(f"Config registration failed: count={len(paths)}, unauthorized={unauthorized[:3]}")
    s1, o1 = definitions(); machine = load_json(BASE / "26_machine_readable_summary.json")
    registration = {
        "registration_version":"ofix7-mid-bounded-limit-audit-v1", "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "baseline_checkpoint_policy_lock":{
            "path":relative(POLICY_LOCK),"sha256":checks["checkpoint_policy_lock_sha"],
            "canonical_json_sha256":json_hash(load_json(POLICY_LOCK)),
        },
        "baseline_replication_lock":{
            "path":relative(BASELINE_LOCK),"sha256":checks["baseline_replication_lock_sha"],
            "canonical_json_sha256":json_hash(load_json(BASELINE_LOCK)),
        },
        "baseline_commit":checks["baseline_commit"], "baseline_status":"STRONG_REPLICATION",
        "selected_checkpoint_policy":"VAL_MACRO_F1", "selected_checkpoint":"best_val_macro_f1.pt",
        "best_alias_policy":"best.pt canonical model state equals best_val_macro_f1.pt",
        "architecture_signature":json_hash({"model":load_yaml(baseline_run(42)/"resolved_config.yaml")["model"]}),
        "model_signature":checks["model_signature"], "feature_signature":checks["feature_signature"],
        "graph_signature":checks["graph_signature"], "dataset_signature":checks["dataset_signature"],
        "split_signature":checks["split_signature"], "parameter_count":1061192,
        "s1_definition":s1, "o1_definition":o1,
        "radam_runtime":{"torch_version":torch.__version__, "signature":str(inspect.signature(torch.optim.RAdam)),
                         "decoupled_weight_decay_supported":"decoupled_weight_decay" in inspect.signature(torch.optim.RAdam).parameters},
        "all_registered_seeds":ALL_SEEDS, "development_seeds":DEVELOPMENT_SEEDS, "heldout_seeds":HELDOUT_SEEDS,
        "text_hash_normalization":"UTF-8 without BOM; CRLF and CR normalized to LF",
        "config_paths":[relative(p) for p in paths],
        "config_file_sha256":{relative(p):normalized_text_sha256(p) for p in paths},
        "normalized_config_sha256":{relative(p):json_hash(load_yaml(p)) for p in paths},
        "semantic_diff_sha256":json_hash(diffs),
        "allowed_difference_policy":{"S1":["training.scheduler.*"],"O1":["training.optimizer.*"],
            "non_scientific":["run_name","description","limit_audit_provenance.*","training.defer_test_evaluation"]},
        "development_gate":DEVELOPMENT_GATE,
        "development_gate_source":"prompt_preregistered_rule; prior machine summary action="+str(machine.get("optimizer_scheduler_next_action"))+" has no numeric gate",
        "development_winner_tie_breaking":TIE_BREAKING, "heldout_confirmation_gate":HELDOUT_GATE,
        "implementation_source_sha256":{
            relative(ROOT/"d16/training/train_d16.py"):normalized_text_sha256(ROOT/"d16/training/train_d16.py"),
            relative(ROOT/"d16/scripts/prepare_ofix7_mid_limit_audit.py"):normalized_text_sha256(ROOT/"d16/scripts/prepare_ofix7_mid_limit_audit.py"),
            relative(ROOT/"d16/scripts/run_ofix7_mid_limit_variant.py"):normalized_text_sha256(ROOT/"d16/scripts/run_ofix7_mid_limit_variant.py"),
            relative(ROOT/"d16/scripts/analyze_ofix7_mid_limit_audit.py"):normalized_text_sha256(ROOT/"d16/scripts/analyze_ofix7_mid_limit_audit.py"),
        },
        "test_embargo_policy":{"flag":"training.defer_test_evaluation=true","selection_uses_test_metrics":False,
            "development_and_heldout":"no test dataset read or result artifact",
            "reveal_requires":["development selection lock","final promotion lock"]},
        "final_decision_rules":{"test_may_change_selection":False,
            "practical_limit_supported":"no development pass or winner fails heldout with integrity intact",
            "practical_limit_not_supported":"winner passes development and heldout",
            "practical_limit_inconclusive":"integrity or missing evidence blocks conclusion"},
    }
    write_json(REGISTRATION_PATH, registration); reg_sha = normalized_text_sha256(REGISTRATION_PATH)
    REGISTRATION_HASH_PATH.write_text(reg_sha+"\n", encoding="utf-8")
    PORTABLE_REGISTRATION_PATH.write_bytes(REGISTRATION_PATH.read_bytes())
    PORTABLE_REGISTRATION_HASH_PATH.write_text(reg_sha+"\n", encoding="utf-8")
    PORTABLE_POLICY_LOCK.write_bytes(POLICY_LOCK.read_bytes())
    PORTABLE_BASELINE_LOCK.write_bytes(BASELINE_LOCK.read_bytes())
    write_csv(PREFLIGHT_DIR/"04_variant_config_manifest.csv",manifest); write_csv(PREFLIGHT_DIR/"05_semantic_config_diffs.csv",diffs)
    report("00_README.md","OFIX7-Mid Bounded Limit Audit Preflight","Stage P only. No full training, resume, fine-tuning, held-out inspection, or test evaluation was launched.")
    report("01_baseline_lock_validation.md","Baseline Lock Validation",table([
        {"lock":"checkpoint_policy","path":relative(POLICY_LOCK),"sha256":checks["checkpoint_policy_lock_sha"],"verified":True},
        {"lock":"baseline_replication","path":relative(BASELINE_LOCK),"sha256":checks["baseline_replication_lock_sha"],"verified":True}])+
        "\n\nAll five config, checkpoint file, and canonical model-state hashes match.")
    base_cfg=load_yaml(baseline_run(42)/"resolved_config.yaml")
    report("02_baseline_scientific_manifest.md","Baseline Scientific Manifest","~~~json\n"+json.dumps({"model":base_cfg["model"],"graph":base_cfg["graph"],"loss":base_cfg["loss"],"training":base_cfg["training"],"parameter_count":1061192},indent=2)+"\n~~~")
    report("03_variant_definitions.md","Variant Definitions","## S1\n\n~~~json\n"+json.dumps(s1,indent=2)+"\n~~~\n\n## O1\n\n~~~json\n"+json.dumps(o1,indent=2)+"\n~~~")
    report("04_variant_config_manifest.md","Variant Config Manifest",table(manifest))
    report("05_semantic_config_diffs.md","Semantic Config Diffs",table(diffs))
    report("09_test_embargo_design.md","Test Embargo Design","The defer flag defaults false. When true, train mode does not construct/read the official test dataset and skips all test prediction, metric, confusion-matrix and cached evaluation outputs.")
    report("10_runner_and_execution_gates.md","Runner And Execution Gates","The runner checks immutable registration/config hashes, exact cell semantics, seed stage, no-resume, clean output, and test-artifact absence. Held-out requires the selected winner lock.")
    report("11_development_gate_registration.md","Development Gate Registration","~~~json\n"+json.dumps(DEVELOPMENT_GATE,indent=2)+"\n~~~\n\nThe prior machine summary has no numeric gate, so prompt thresholds are registered.")
    report("12_heldout_gate_registration.md","Held-Out Gate Registration","~~~json\n"+json.dumps(HELDOUT_GATE,indent=2)+"\n~~~")
    report("13_final_decision_registration.md","Final Decision Registration","Validation locks determine promotion; test cannot revise it. Tie break: "+" -> ".join(TIE_BREAKING))
    report("15_powershell_commands.md","PowerShell Commands","~~~powershell\n"+"\n".join(commands("powershell"))+"\n~~~")
    report("16_kaggle_linux_commands.md","Kaggle Linux Commands","~~~bash\n"+"\n".join(commands("linux"))+"\n~~~")
    return {"status":"PREPARED","registration_sha256":reg_sha,"config_count":10,"semantic_diff_count":len(diffs),"unauthorized_diff_count":0}


def verify_portable_bundle() -> dict[str, Any]:
    required=[PORTABLE_REGISTRATION_PATH,PORTABLE_REGISTRATION_HASH_PATH,PORTABLE_POLICY_LOCK,PORTABLE_BASELINE_LOCK]
    missing=[str(path) for path in required if not path.exists()]
    if missing: raise FileNotFoundError(missing)
    registration_sha=normalized_text_sha256(PORTABLE_REGISTRATION_PATH)
    if registration_sha!=PORTABLE_REGISTRATION_HASH_PATH.read_text(encoding="utf-8-sig").strip():
        raise RuntimeError("Portable registration hash mismatch")
    registration=load_json(PORTABLE_REGISTRATION_PATH)
    if registration["baseline_checkpoint_policy_lock"]["sha256"]!=POLICY_SHA:
        raise RuntimeError("Historical checkpoint-policy lock SHA drift")
    if registration["baseline_replication_lock"]["sha256"]!=BASELINE_SHA:
        raise RuntimeError("Historical baseline-replication lock SHA drift")
    if json_hash(load_json(PORTABLE_POLICY_LOCK))!=registration["baseline_checkpoint_policy_lock"]["canonical_json_sha256"]:
        raise RuntimeError("Portable checkpoint-policy lock content mismatch")
    if json_hash(load_json(PORTABLE_BASELINE_LOCK))!=registration["baseline_replication_lock"]["canonical_json_sha256"]:
        raise RuntimeError("Portable baseline-replication lock content mismatch")
    for relpath,expected in registration["config_file_sha256"].items():
        path=ROOT/relpath
        if not path.exists() or normalized_text_sha256(path)!=expected:
            raise RuntimeError(f"Portable config hash mismatch: {relpath}")
    return {"portable_bundle_valid":True,"registration_sha256":registration_sha,"config_count":len(registration["config_paths"])}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--verify-only",action="store_true"); parser.add_argument("--verify-portable",action="store_true"); args=parser.parse_args()
    if args.verify_portable: result=verify_portable_bundle()
    elif args.verify_only: result=verify_baseline_locks()
    else: result=prepare()
    print(json.dumps(result,indent=2,default=str))


if __name__ == "__main__": main()
